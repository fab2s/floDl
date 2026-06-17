//! Lifecycle methods for [`super::ClusterCoordinator`]: bind, accept,
//! shutdown, and the outbound control-frame I/O.

use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::distributed::ddp_run::ApplyPolicy;
use crate::distributed::relay::mux::{MuxRecord, RelayControlMsg};
use crate::distributed::wire::{ControlFrame, ControlMsgWire, MsgKind, SessionSalt, TimingMsgWire};
use crate::tensor::{Result, TensorError};

use super::{
    ClusterCoordinator, ClusterCoordinatorConfig, CpuAvgState,
    RunPhase, initial_callback_role, relay_reader_loop,
};

impl ClusterCoordinator {
    /// Bind a control TcpListener at `bind_addr`, accept exactly
    /// `world_size` rank connections (validating the session salt at
    /// handshake), spawn per-rank reader threads, and return the
    /// configured coordinator.
    ///
    /// Returns `Err` if any handshake fails (loud error: salt mismatch,
    /// magic mismatch, version mismatch, world_size disagreement,
    /// duplicate rank_id).
    pub fn start(
        bind_addr: SocketAddr,
        salt: SessionSalt,
        config: ClusterCoordinatorConfig,
    ) -> Result<Self> {
        let (listener, _port) = Self::bind(bind_addr)?;
        Self::start_from_listener(listener, salt, config)
    }

    /// Bind the control listener without blocking on accept. Useful for
    /// tests that need to publish the bound port before spawning rank
    /// connections (the post-bind accept loop blocks the calling
    /// thread until every rank has connected).
    pub fn bind(bind_addr: SocketAddr) -> Result<(TcpListener, u16)> {
        let listener = TcpListener::bind(bind_addr).map_err(|e| {
            TensorError::new(&format!(
                "cluster_coordinator: bind {bind_addr} failed: {e}"
            ))
        })?;
        let bound_port = listener
            .local_addr()
            .map_err(|e| {
                TensorError::new(&format!(
                    "cluster_coordinator: local_addr() failed: {e}"
                ))
            })?
            .port();
        Ok((listener, bound_port))
    }

    /// Accept connections + handshake on a pre-bound listener. Pair
    /// with [`Self::bind`] when the caller needs the port before
    /// blocking on accepts (e.g. tests that spawn rank threads after
    /// publishing the port through a channel).
    pub fn start_from_listener(
        listener: TcpListener,
        salt: SessionSalt,
        config: ClusterCoordinatorConfig,
    ) -> Result<Self> {
        let world_size = config.world_size;
        if world_size == 0 {
            return Err(TensorError::new(
                "cluster_coordinator: world_size must be > 0",
            ));
        }
        let bound_port = listener
            .local_addr()
            .map_err(|e| {
                TensorError::new(&format!(
                    "cluster_coordinator: local_addr() failed: {e}"
                ))
            })?
            .port();

        // Accept per-host relay connections, each announcing the ranks it
        // carries via a `RelayHello`. Accept until every global rank is
        // covered exactly once. `control_streams` holds the write half per
        // connection (the coord is the sole writer); `rank_to_conn` maps a
        // rank to its owning connection for `send_control`. Each
        // connection's read half goes to a per-host reader thread that
        // demuxes by rank tag.
        let mut control_streams: Vec<TcpStream> = Vec::new();
        let mut rank_to_conn: Vec<Option<usize>> = (0..world_size).map(|_| None).collect();
        let mut conn_reads: Vec<TcpStream> = Vec::new();
        let mut covered = 0usize;
        while covered < world_size {
            let (mut stream, _peer) = listener.accept().map_err(|e| {
                TensorError::new(&format!("cluster_coordinator: accept failed: {e}"))
            })?;
            let _ = stream.set_nodelay(true);
            // 10s handshake timeout protects against a wedged relay.
            stream
                .set_read_timeout(Some(Duration::from_secs(10)))
                .map_err(|e| {
                    TensorError::new(&format!("cluster_coordinator: set_read_timeout: {e}"))
                })?;
            let ranks = match MuxRecord::read_from(&mut stream, &salt)? {
                Some(MuxRecord::Control(RelayControlMsg::Hello { host, ranks })) => {
                    crate::verbose!(
                        "  cluster_coordinator: relay '{host}' carries ranks {ranks:?}"
                    );
                    ranks
                }
                Some(other) => {
                    return Err(TensorError::new(&format!(
                        "cluster_coordinator: expected relay Hello, got {other:?}"
                    )));
                }
                None => {
                    return Err(TensorError::new(
                        "cluster_coordinator: relay closed connection before Hello",
                    ));
                }
            };
            let conn_idx = control_streams.len();
            for r in &ranks {
                let r = *r as usize;
                if r >= world_size {
                    return Err(TensorError::new(&format!(
                        "cluster_coordinator: relay announced rank {r} >= world_size {world_size}"
                    )));
                }
                if rank_to_conn[r].is_some() {
                    return Err(TensorError::new(&format!(
                        "cluster_coordinator: rank {r} announced by two relays"
                    )));
                }
                rank_to_conn[r] = Some(conn_idx);
                covered += 1;
            }
            MuxRecord::control(RelayControlMsg::HelloAck).write_to(&mut stream, &salt)?;
            // Reader holds a try-cloned read half (short timeout so it can
            // observe shutdown between records); the coord keeps the write
            // half for `send_control`.
            let read_half = stream.try_clone().map_err(|e| {
                TensorError::new(&format!("cluster_coordinator: relay try_clone: {e}"))
            })?;
            read_half
                .set_read_timeout(Some(Duration::from_millis(250)))
                .map_err(|e| {
                    TensorError::new(&format!(
                        "cluster_coordinator: reader set_read_timeout: {e}"
                    ))
                })?;
            control_streams.push(stream);
            conn_reads.push(read_half);
        }

        // Spawn one reader thread per relay connection; all feed the same
        // timing / metrics channels (the rank rides in each frame's mux
        // tag + payload).
        let shutdown_flag = Arc::new(AtomicBool::new(false));
        let (timing_tx, timing_rx) = mpsc::channel::<TimingMsgWire>();
        let (metrics_tx, metrics_rx) =
            mpsc::channel::<crate::distributed::wire::MetricsMsgWire>();

        let mut reader_handles: Vec<Option<JoinHandle<()>>> =
            Vec::with_capacity(conn_reads.len());
        for (conn_idx, mut read_half) in conn_reads.into_iter().enumerate() {
            let tx = timing_tx.clone();
            let mtx = metrics_tx.clone();
            let salt_for_reader = salt;
            let shutdown_for_reader = Arc::clone(&shutdown_flag);
            let spawn_result = thread::Builder::new()
                .name(format!("flodl-coord-relay{conn_idx}"))
                .spawn(move || {
                    relay_reader_loop(
                        &mut read_half,
                        &salt_for_reader,
                        &shutdown_for_reader,
                        &tx,
                        &mtx,
                    );
                });
            match spawn_result {
                Ok(handle) => reader_handles.push(Some(handle)),
                Err(e) => {
                    // Partial-init cleanup: a mid-loop spawn failure (OS
                    // thread limit) must not leak the readers already
                    // running. Signal them to stop and join before
                    // returning — without this they'd outlive the aborted
                    // bootstrap, blocked on their sockets (the control
                    // write halves drop on return, but the readers hold
                    // their own try_cloned read halves and only observe
                    // shutdown via the flag, on their 250ms poll).
                    shutdown_flag.store(true, Ordering::SeqCst);
                    for h in reader_handles.iter_mut() {
                        if let Some(j) = h.take() {
                            let _ = j.join();
                        }
                    }
                    return Err(TensorError::new(&format!(
                        "cluster_coordinator: spawn relay reader {conn_idx}: {e}"
                    )));
                }
            }
        }
        // Drop the extra senders we cloned for the closures; loop exit
        // depends on every cloned sender being dropped, but that happens
        // automatically when reader threads exit.
        drop(timing_tx);
        drop(metrics_tx);
        let streams = control_streams;

        // Resume: layer saved trajectory state on top of the user-built
        // ElChe (which carries the user's knobs from this run's
        // DdpRunConfig). When `start_elche_state` is None, the ElChe
        // stays fresh.
        let mut el_che = config.el_che;
        if let Some(ref state) = config.start_elche_state {
            el_che.restore_from_state(state)?;
        }
        // Cap the reduce window to one epoch's batches (coverage-global).
        // The overhead auto-tune may grow the schedule to amortize an
        // expensive sync, but a window must never span more than one
        // dataset pass — otherwise syncs collapse to <1/epoch (observed:
        // CPU cadence grew the window to 1092 batches against a 781-batch
        // epoch, dropping to ~1 sync/epoch and serializing the cohort).
        // No-op for NCCL (cheap sync → window stays well under) and for
        // any run where the epoch size isn't known yet.
        if config.batch_size > 0 && config.total_samples >= config.batch_size {
            el_che.set_max_total_batches(config.total_samples / config.batch_size);
        }
        // `calibrated` mirrors the post-restore ElChe state: true when
        // any rank has a positive smoothed reading. Matches the
        // invariant the snapshot was taken under.
        let calibrated = config.start_elche_state.is_some()
            && el_che.is_calibrated();
        Ok(ClusterCoordinator {
            policy: config.policy,
            backend: config.backend,
            world_size,
            overshoot_initial: config.overshoot_initial,
            overshoot_ceiling: config.overshoot_ceiling,
            overshoot_auto: config.overshoot_auto,
            elche_relax_up: config.elche_relax_up,
            el_che,
            convergence_guard: config.convergence_guard,
            version: 0,
            avg_count: config.start_avg_count,
            global_step: config.start_global_step,
            calibrated,
            active_count: world_size,
            max_overshoot: config.overshoot_initial,
            steps_since_avg: vec![0; world_size],
            wall_ms_accum: vec![0.0; world_size],
            pb_delivered_ms_accum: vec![0.0; world_size],
            pb_delivered_batches: vec![0; world_size],
            last_batch_ms: vec![0.0; world_size],
            last_step_count: vec![0; world_size],
            nccl_sync_step: vec![0; world_size],
            nccl_ack: vec![true; world_size],
            nccl_sync_divergence: vec![None; world_size],
            nccl_sync_pre_norm: vec![None; world_size],
            nccl_sync_post_norm: None,
            throttled: vec![false; world_size],
            dispatch_hold_logged: vec![false; world_size],
            last_nccl_sync_ms: 0.0,
            nccl_sync_start: None,
            // Per-epoch d-aggregator identity values; see field docs on
            // `ClusterCoordinator` + threaded `coordinator/cpu_avg.rs`.
            epoch_d_min: f64::INFINITY,
            epoch_d_max: f64::NEG_INFINITY,
            epoch_d_sum: 0.0,
            epoch_d_count: 0,
            epoch_last_d: 0.0,
            epoch_last_k_max: 0,
            lr_event_meta: if config.meta_controller {
                Some(crate::distributed::lr_event_meta::LrEventMeta::with_default_config())
            } else {
                None
            },
            last_lr_per_rank: vec![None; world_size],
            cpu_avg_state: CpuAvgState::Idle,
            lost_broadcasts: 0,
            prof_enabled: crate::log::enabled(crate::log::Verbosity::Debug),
            stall_last_global_step: 0,
            stall_since: None,
            stall_last_dump: None,
            dead_ranks: config.dead_ranks,
            heartbeat_timeout_secs: config.heartbeat_timeout_secs,
            rendezvous_timeout_secs: config.rendezvous_timeout_secs,
            last_heartbeat: vec![Instant::now(); world_size],
            exited: vec![false; world_size],
            last_step_count_at_epoch_start: vec![0; world_size],
            nccl_rendezvous_pending: None,
            local_ranks: config.local_ranks.clone(),
            max_failure: config.max_failure,
            epoch_callback_policy: config.epoch_callback_policy,
            checkpoint_role: initial_callback_role(
                config.epoch_callback_policy,
                world_size,
            ),
            eval_role: initial_callback_role(
                config.epoch_callback_policy,
                world_size,
            ),
            epoch_callback_role: initial_callback_role(
                config.epoch_callback_policy,
                world_size,
            ),
            epoch_role_dirty: true,
            checkpoint_tried_ranks: std::collections::HashMap::new(),
            last_checkpoint_elapsed_ms_ewma: None,
            last_eval_elapsed_ms_ewma: None,
            last_epoch_fn_elapsed_ms_ewma: None,
            save_path: config.save_path.clone(),
            checkpoint_every: config.checkpoint_every,
            seed: config.seed,
            checkpoint_at_epoch: config.checkpoint_at_epoch,
            start_coverage: config.start_coverage.clone(),
            checkpoint_forge: config.checkpoint_forge.clone(),
            pending_checkpoint_coverage: None,
            shutdown_with_save_dispatched: false,
            last_observed_sync_lag_ms: vec![None; world_size],
            last_observed_upload_ms: vec![None; world_size],
            rank_epoch: vec![0; world_size],
            last_aggregated_epoch: None,
            last_dispatched_epoch: None,
            run_phase: RunPhase::Training,
            epoch_plan_cache: std::collections::HashMap::new(),
            total_samples: config.total_samples,
            batch_size: config.batch_size.max(1),
            num_epochs: config.num_epochs,
            partition_ratios: config.partition_ratios,
            timing_rx,
            metrics_rx,
            metrics_buffer: std::collections::BTreeMap::new(),
            chunk_pools: std::collections::BTreeMap::new(),
            // Resolve progressive: explicit override wins, otherwise
            // auto-on for Cadence/Async, off for Sync (matches the
            // threaded coordinator's default).
            progressive: config.progressive.unwrap_or(
                !matches!(config.policy, ApplyPolicy::Sync),
            ),
            // Floor for proportional chunk sizing after calibration —
            // matches the threaded coordinator's default.
            min_chunk_batches: 4,
            final_window_plan: None,
            metrics_fn: config.metrics_fn.clone(),
            metrics_sink_tx: config.metrics_sink_tx.clone(),
            eval_result_fn: config.eval_result_fn.clone(),
            eval_every_epochs: config.eval_every_epochs,
            metrics_device_indices: (0..world_size as u8).collect(),
            control_streams: streams,
            rank_to_conn,
            reader_handles,
            shutdown_flag,
            bound_port,
            salt,
            timeline: config.timeline.clone(),
            sync_start: None,
            cpu_avg_start: None,
            dashboard_sink: config.dashboard_sink.clone(),
        })
    }

    // -----------------------------------------------------------------
    // Outbound control frame I/O
    // -----------------------------------------------------------------

    pub(super) fn send_control(&mut self, rank: usize, msg: &ControlMsgWire) -> Result<()> {
        if rank >= self.world_size {
            return Err(TensorError::new(&format!(
                "cluster_coordinator: send_control rank {rank} >= world_size {}",
                self.world_size
            )));
        }
        // Resolve the relay connection carrying this rank. Unmapped means a
        // headless coord (test fixtures via `for_test`, no streams) or a
        // rank no relay announced. Return Err so callers that tolerate
        // transient send failures (e.g. `handle_checkpoint_result`'s
        // retry-dispatch path) log + continue rather than panic.
        let Some(conn_idx) = self.rank_to_conn.get(rank).copied().flatten() else {
            return Err(TensorError::new(&format!(
                "cluster_coordinator: send_control(rank={rank}): no relay connection \
                 (headless coord, or rank not announced by any relay)"
            )));
        };
        // Encode the control frame, then wrap it as a rank-tagged mux
        // record on the per-host connection (the relay demuxes it to the
        // local rank).
        let frame = ControlFrame::encode(&self.salt, MsgKind::Control, msg)?;
        let mut buf = Vec::new();
        frame.write_to(&mut buf)?;
        MuxRecord::data(rank as u32, buf)
            .write_to(&mut self.control_streams[conn_idx], &self.salt)
            .map_err(|e| {
                TensorError::new(&format!(
                    "cluster_coordinator: send_control(rank={rank}): {e}"
                ))
            })?;
        Ok(())
    }

    /// Broadcast `msg` to every rank.
    ///
    /// BEST-EFFORT, NOT FAIL-FAST: every rank gets a send attempt even
    /// when an earlier one fails. A fail-fast `?` here left every rank
    /// AFTER the broken connection unsignaled — a Shutdown that never
    /// reached the trailing ranks parked them forever, and a partial
    /// RequestParams sent part of the cohort into a reduce barrier its
    /// peers never entered.
    ///
    /// Declared-dead ranks are STILL attempted ("dead" often means
    /// slow/stale-heartbeat, and frames like ShutdownWithSave exist
    /// precisely to reach them) but their failures are expected and only
    /// logged at verbose. Returns `Err` listing the LIVE ranks that
    /// failed, AFTER attempting all of them; callers on non-abortable
    /// paths log it (a live rank that fails has a broken connection, so
    /// heartbeat staleness reaps it shortly).
    pub(super) fn broadcast_control(&mut self, msg: &ControlMsgWire) -> Result<()> {
        let mut failed: Vec<String> = Vec::new();
        for rank in 0..self.world_size {
            let dead = self.is_dead(rank);
            if let Err(e) = self.send_control(rank, msg) {
                if dead {
                    crate::verbose!(
                        "  ddp: broadcast to declared-dead rank {rank} failed \
                         (expected): {e}"
                    );
                } else {
                    failed.push(format!("rank {rank}: {e}"));
                }
            }
        }
        if failed.is_empty() {
            Ok(())
        } else {
            // Structured trace of a dropped best-effort broadcast: a
            // silently lost SyncNow / DeclareDead can leave the survivor
            // cohort waiting on a signal that never arrives. Shutdown is
            // exempt — a failed Shutdown send is an
            // expected teardown race (the rank already exited), not lost
            // live coordination.
            if !matches!(msg, ControlMsgWire::Shutdown) {
                self.note_lost_broadcast(control_label(msg), failed.len());
            }
            Err(TensorError::new(&format!(
                "cluster_coordinator: broadcast_control failed for {} of {} ranks [{}]",
                failed.len(),
                self.world_size,
                failed.join("; "),
            )))
        }
    }

    /// Record a dropped best-effort broadcast: bump the run-long
    /// [`lost_broadcasts`](Self::lost_broadcasts) counter and emit a
    /// [`crate::monitor::EventKind::LostBroadcast`] on the shared timeline
    /// if one is attached. The caller has already logged the per-rank
    /// detail to stderr; this is the structured, queryable twin of that
    /// log. `failures` is the number of live ranks that did not receive
    /// the message.
    pub(super) fn note_lost_broadcast(&mut self, control: &str, failures: usize) {
        self.lost_broadcasts += 1;
        if let Some(ref tl) = self.timeline {
            tl.event(crate::monitor::EventKind::LostBroadcast {
                control: control.to_string(),
                failures,
            });
        }
    }

    /// Send Shutdown to every rank. Called from [`Self::shutdown`];
    /// kept public so callers running the coordinator inline can drop
    /// it from a different point in their loop if needed.
    pub fn shutdown_workers(&mut self) -> Result<()> {
        self.broadcast_control(&ControlMsgWire::Shutdown)
    }

    /// Stop reader threads, send Shutdown to every connected rank,
    /// join the threads, drop streams. Idempotent on the shutdown flag.
    pub fn shutdown(mut self) -> Result<()> {
        // Best-effort send Shutdown before tearing readers down. Ignore
        // write errors here: a rank may already have exited.
        let _ = self.shutdown_workers();
        self.shutdown_flag.store(true, Ordering::SeqCst);
        for handle_opt in self.reader_handles.iter_mut() {
            if let Some(handle) = handle_opt.take() {
                let _ = handle.join();
            }
        }
        Ok(())
    }
}

/// Short, payload-free label for a control message, used in the
/// [`EventKind::LostBroadcast`](crate::monitor::EventKind::LostBroadcast)
/// timeline trace. Exhaustive on purpose: a new `ControlMsgWire` variant
/// forces a label here rather than silently rendering as `"other"`.
fn control_label(msg: &ControlMsgWire) -> &'static str {
    match msg {
        ControlMsgWire::RequestParams => "RequestParams",
        ControlMsgWire::Update { .. } => "Update",
        ControlMsgWire::SyncNow => "SyncNow",
        ControlMsgWire::StartEpoch(_) => "StartEpoch",
        ControlMsgWire::ExtendPartition { .. } => "ExtendPartition",
        ControlMsgWire::DeclareDead { .. } => "DeclareDead",
        ControlMsgWire::RequestNewNcclId => "RequestNewNcclId",
        ControlMsgWire::NewNcclSession { .. } => "NewNcclSession",
        ControlMsgWire::Throttle => "Throttle",
        ControlMsgWire::SetGlobalStep { .. } => "SetGlobalStep",
        ControlMsgWire::Checkpoint { .. } => "Checkpoint",
        ControlMsgWire::ExecuteEvalCallback { .. } => "ExecuteEvalCallback",
        ControlMsgWire::SetEpochCallbackRole { .. } => "SetEpochCallbackRole",
        ControlMsgWire::Shutdown => "Shutdown",
        ControlMsgWire::ShutdownWithSave { .. } => "ShutdownWithSave",
        ControlMsgWire::EpochAggregated(_) => "EpochAggregated",
        ControlMsgWire::SaveConsensusModel { .. } => "SaveConsensusModel",
    }
}
