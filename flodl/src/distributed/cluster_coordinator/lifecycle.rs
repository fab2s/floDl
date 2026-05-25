//! Lifecycle methods for [`super::ClusterCoordinator`]: bind, accept,
//! shutdown, and the outbound control-frame I/O.

use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::distributed::ddp_run::ApplyPolicy;
use crate::distributed::wire::{ControlFrame, ControlMsgWire, MsgKind, SessionSalt, TimingMsgWire};
use crate::tensor::{Result, TensorError};

use super::{
    ClusterCoordinator, ClusterCoordinatorConfig, CpuAvgState, initial_callback_role,
    read_handshake_rank, reader_loop, write_handshake_ack,
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

        // Accept world_size connections, validate handshake, place each
        // at its claimed rank slot. Order-independent.
        let mut streams: Vec<Option<TcpStream>> =
            (0..world_size).map(|_| None).collect();
        let mut connected = 0usize;
        while connected < world_size {
            let (mut stream, _peer) = listener.accept().map_err(|e| {
                TensorError::new(&format!(
                    "cluster_coordinator: accept failed: {e}"
                ))
            })?;
            // 10s handshake timeout protects against wedged ranks.
            stream
                .set_read_timeout(Some(Duration::from_secs(10)))
                .map_err(|e| {
                    TensorError::new(&format!(
                        "cluster_coordinator: set_read_timeout: {e}"
                    ))
                })?;
            let rank_id = read_handshake_rank(&mut stream, world_size as u32, &salt)?;
            let rank_idx = rank_id as usize;
            if rank_idx >= world_size {
                return Err(TensorError::new(&format!(
                    "cluster_coordinator: handshake rank_id {rank_idx} >= world_size {world_size}"
                )));
            }
            if streams[rank_idx].is_some() {
                return Err(TensorError::new(&format!(
                    "cluster_coordinator: duplicate rank_id {rank_idx} connected"
                )));
            }
            write_handshake_ack(&mut stream, &salt)?;
            // Clear the handshake timeout; ControlFrame reads can take
            // arbitrarily long under load.
            stream
                .set_read_timeout(None)
                .map_err(|e| {
                    TensorError::new(&format!(
                        "cluster_coordinator: clear read_timeout: {e}"
                    ))
                })?;
            streams[rank_idx] = Some(stream);
            connected += 1;
        }
        let mut streams: Vec<TcpStream> = streams.into_iter()
            .map(|s| s.expect("all slots filled by accept loop"))
            .collect();

        // Spawn one reader thread per rank. Each thread holds the read
        // half of a try_clone'd stream; the coordinator owns the write
        // half. ControlFrame::read_from handles HMAC validation per frame.
        let shutdown_flag = Arc::new(AtomicBool::new(false));
        let (timing_tx, timing_rx) = mpsc::channel::<TimingMsgWire>();
        let (metrics_tx, metrics_rx) =
            mpsc::channel::<crate::distributed::wire::MetricsMsgWire>();

        let mut reader_handles: Vec<Option<JoinHandle<()>>> = Vec::with_capacity(world_size);
        for (rank, stream) in streams.iter_mut().enumerate() {
            let mut read_half = stream.try_clone().map_err(|e| {
                TensorError::new(&format!(
                    "cluster_coordinator: stream try_clone for rank {rank}: {e}"
                ))
            })?;
            // Reader uses a short timeout to observe shutdown between frames.
            read_half
                .set_read_timeout(Some(Duration::from_millis(250)))
                .map_err(|e| {
                    TensorError::new(&format!(
                        "cluster_coordinator: reader set_read_timeout: {e}"
                    ))
                })?;
            let tx = timing_tx.clone();
            let mtx = metrics_tx.clone();
            let salt_for_reader = salt;
            let shutdown_for_reader = Arc::clone(&shutdown_flag);
            let handle = thread::Builder::new()
                .name(format!("flodl-coord-reader:r{rank}"))
                .spawn(move || {
                    reader_loop(rank, &mut read_half, &salt_for_reader, &shutdown_for_reader, &tx, &mtx);
                })
                .map_err(|e| {
                    TensorError::new(&format!(
                        "cluster_coordinator: spawn reader for rank {rank}: {e}"
                    ))
                })?;
            reader_handles.push(Some(handle));
        }
        // Drop the extra senders we cloned for the closures; loop exit
        // depends on every cloned sender being dropped, but that happens
        // automatically when reader threads exit.
        drop(timing_tx);
        drop(metrics_tx);

        // Resume: layer saved trajectory state on top of the user-built
        // ElChe (which carries the user's knobs from this run's
        // DdpRunConfig). When `start_elche_state` is None, the ElChe
        // stays fresh.
        let mut el_che = config.el_che;
        if let Some(ref state) = config.start_elche_state {
            el_che.restore_from_state(state)?;
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
            last_batch_ms: vec![0.0; world_size],
            last_step_count: vec![0; world_size],
            nccl_sync_step: vec![0; world_size],
            nccl_ack: vec![true; world_size],
            nccl_sync_divergence: vec![None; world_size],
            nccl_sync_pre_norm: vec![None; world_size],
            nccl_sync_post_norm: None,
            throttled: vec![false; world_size],
            last_nccl_sync_ms: 0.0,
            nccl_sync_start: None,
            lr_event_meta: if config.meta_controller {
                Some(crate::distributed::lr_event_meta::LrEventMeta::with_default_config())
            } else {
                None
            },
            last_lr_per_rank: vec![None; world_size],
            cpu_avg_state: CpuAvgState::Idle,
            dead_ranks: config.dead_ranks,
            heartbeat_timeout_secs: config.heartbeat_timeout_secs,
            rendezvous_timeout_secs: config.rendezvous_timeout_secs,
            last_heartbeat: vec![Instant::now(); world_size],
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
            shutdown_with_save_dispatched: false,
            last_observed_sync_lag_ms: vec![None; world_size],
            last_observed_upload_ms: vec![None; world_size],
            rank_epoch: vec![0; world_size],
            last_aggregated_epoch: None,
            last_dispatched_epoch: None,
            shutdown_initiated: false,
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
            metrics_fn: config.metrics_fn.clone(),
            metrics_sink_tx: config.metrics_sink_tx.clone(),
            eval_result_fn: config.eval_result_fn.clone(),
            eval_every_epochs: config.eval_every_epochs,
            metrics_device_indices: (0..world_size as u8).collect(),
            control_streams: streams,
            reader_handles,
            shutdown_flag,
            bound_port,
            salt,
            timeline: config.timeline.clone(),
            sync_start: None,
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
        if rank >= self.control_streams.len() {
            // Headless coord (test fixtures via `for_test`) has no
            // streams populated. Return Err so callers that
            // tolerate transient send failures (e.g.
            // `handle_checkpoint_result`'s retry-dispatch path) can
            // log + continue rather than panic the test process.
            return Err(TensorError::new(&format!(
                "cluster_coordinator: send_control(rank={rank}): no stream \
                 (headless coord; index {rank} out of streams len {})",
                self.control_streams.len()
            )));
        }
        let frame = ControlFrame::encode(&self.salt, MsgKind::Control, msg)?;
        frame.write_to(&mut self.control_streams[rank]).map_err(|e| {
            TensorError::new(&format!(
                "cluster_coordinator: send_control(rank={rank}): {e}"
            ))
        })?;
        Ok(())
    }

    pub(super) fn broadcast_control(&mut self, msg: &ControlMsgWire) -> Result<()> {
        for rank in 0..self.world_size {
            self.send_control(rank, msg)?;
        }
        Ok(())
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
