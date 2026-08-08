//! `ClusterWorker` bridge-thread bodies: the inbound / outbound /
//! heartbeat / NCCL-watchdog loops plus the wire-write helpers they
//! use. All are free functions (no `ClusterWorker` state — they take
//! channels / streams / handles as params), spawned by the lifecycle
//! impl in the parent module.

use super::*;

// ---------------------------------------------------------------------------
// Bridge thread bodies
// ---------------------------------------------------------------------------

/// TCP → control_tx bridge: read `ControlFrame`s, decode the
/// payload, push into the in-process control channel.
///
/// Elastic-membership frames intercepted here (NOT forwarded to the
/// inner GpuWorker):
/// - `DeclareDead { rank }` → `local_dead_ranks.declare_dead(rank)`,
///   which the NCCL watchdog observes and uses to abort the in-flight
///   collective.
/// - `NewNcclSession { uid_bytes, new_rank, new_world_size }` →
///   `mailbox.replace(Some(PendingNcclSession { … }))`. The main
///   thread reads this slot after its NCCL collective errors out
///   (post-abort) to rebuild the comm.
/// - `RequestNewNcclId` → call `NcclUniqueId::new()` to generate fresh
///   bytes locally and ship them back to the coord via the timing
///   channel as `TimingMsg::NewNcclIdGenerated`. Generation happens
///   here (not on the coord) because the coord process may not link
///   libnccl while workers always do.
///
/// All other frames fall through to `control_wire_to_msg` and the
/// inner control channel as before.
#[allow(clippy::too_many_arguments)]
pub(super) fn inbound_loop(
    rank: usize,
    stream: &mut TcpStream,
    salt: &SessionSalt,
    shutdown: &Arc<AtomicBool>,
    control_tx: &mpsc::Sender<ControlMsg>,
    local_dead_ranks: &Arc<crate::distributed::controller::DeadRanks>,
    nccl_session_mailbox: &Arc<std::sync::Mutex<Option<PendingNcclSession>>>,
    timing_tx: &mpsc::Sender<TimingMsg>,
    coord_liveness_timeout_secs: u64,
) {
    // ESCAPE HATCH: any abnormal exit of this bridge (coordinator or relay
    // death, frame corruption, EOF without a prior Shutdown frame) must wake
    // the inner GpuWorker — WHEREVER it is parked:
    //
    // - Parked in a blocking `control_rx.recv()` (`wait_for_epoch_plan` /
    //   a barrier wait): injecting Shutdown breaks the cycle — the inner
    //   exits its run loop, `run_until_shutdown` drops it, the param
    //   channel disconnects and every bridge unwinds.
    // - Parked INSIDE an NCCL collective / stream synchronize: the control
    //   channel is never read there, so Shutdown alone leaves the rank a
    //   zombie at 100% CPU (its heartbeats only ever told the dead
    //   coordinator). The controller is this rank's only window on the
    //   world, so losing the link means every peer is unreachable:
    //   declare them all dead in the local ledger — the NCCL watchdog
    //   observes the ledger within its poll tick and aborts the comm,
    //   the collective errors out, and the lone-survivor bail exits the
    //   rank with a death record instead of zombifying.
    //
    // On a CLEAN coordinator shutdown the real Shutdown frame arrives
    // BEFORE the link drops; `clean_shutdown_seen` suppresses the peer
    // poison on the subsequent EOF (the main thread may legitimately
    // still be draining the final coherent reduce), and the duplicate
    // Shutdown injection stays a harmless no-op.
    let inject_shutdown = || {
        let _ = control_tx.send(ControlMsg::Shutdown);
    };
    let poison_peers = || {
        for r in 0..local_dead_ranks.world_size() {
            if r != rank {
                local_dead_ranks.declare_dead(r);
            }
        }
    };
    let mut clean_shutdown_seen = false;
    // Coord-liveness deadline: if no inbound frame (CoordHeartbeat OR real
    // traffic) arrives within this window, the coordinator is presumed
    // wedged-open (alive TCP, frozen userspace — the SIGSTOP / deadlock case
    // that never produces EOF or error) and we bail like a hard link drop.
    // Reuses `heartbeat_timeout_secs` so both liveness directions share one
    // wall-clock timescale. Reset below on every successful frame decode; the
    // handshake ACK we just read seeds the clock at loop entry.
    let coord_liveness_deadline = Duration::from_secs(coord_liveness_timeout_secs);
    let mut last_inbound = std::time::Instant::now();
    loop {
        if shutdown.load(Ordering::SeqCst) {
            return;
        }
        // The worker reaches the coord through its host relay, which
        // forwards length-framed opaque ControlFrame blobs on the loopback
        // leg. Read the blob, parse the frame, then dispatch as before.
        match try_read_len_framed(stream) {
            Ok(LenFramedRead::Blob(blob)) => {
                let frame = match ControlFrame::read_from(&mut blob.as_slice(), salt) {
                    Ok(Some(f)) => f,
                    Ok(None) => {
                        eprintln!("cluster_worker: inbound r{rank} truncated ControlFrame");
                        if !clean_shutdown_seen {
                            poison_peers();
                        }
                        inject_shutdown();
                        return;
                    }
                    Err(e) => {
                        eprintln!(
                            "cluster_worker: inbound r{rank} ControlFrame parse: {e}"
                        );
                        if !clean_shutdown_seen {
                            poison_peers();
                        }
                        inject_shutdown();
                        return;
                    }
                };
                // Any coherent frame is proof the coord is alive: reset the
                // liveness deadline (covers CoordHeartbeat AND real traffic).
                last_inbound = std::time::Instant::now();
                match frame.kind {
                MsgKind::Control => match frame.decode::<ControlMsgWire>() {
                    Ok(wire) => match wire {
                        // Pure liveness beacon: absorbed here (the deadline
                        // was just reset above), never forwarded to the inner.
                        ControlMsgWire::CoordHeartbeat => {}
                        // Elastic-membership interception (does NOT
                        // forward to the inner GpuWorker).
                        ControlMsgWire::DeclareDead { rank: dead_r } => {
                            local_dead_ranks.declare_dead(dead_r as usize);
                        }
                        ControlMsgWire::NewNcclSession {
                            uid_bytes,
                            new_rank,
                            new_world_size,
                        } => {
                            let pending = PendingNcclSession {
                                uid_bytes,
                                new_rank: new_rank as usize,
                                new_world_size: new_world_size as usize,
                            };
                            if let Ok(mut slot) = nccl_session_mailbox.lock() {
                                *slot = Some(pending);
                            }
                        }
                        ControlMsgWire::RequestNewNcclId => {
                            match crate::distributed::nccl::NcclUniqueId::new() {
                                Ok(uid) => {
                                    let uid_bytes = uid.as_bytes().to_vec();
                                    let _ = timing_tx.send(
                                        TimingMsg::NewNcclIdGenerated {
                                            rank,
                                            uid_bytes,
                                        },
                                    );
                                }
                                Err(e) => {
                                    eprintln!(
                                        "cluster_worker: inbound r{rank} \
                                         NcclUniqueId::new failed: {e}"
                                    );
                                }
                            }
                        }
                        // atomic-dispatch: the post-reduce Update may
                        // carry the rank's next reduce-window chunk. The
                        // wire-Update itself is informational (the param
                        // bridge synthesises the real
                        // `ControlMsg::Update(AveragedParams)`); when a
                        // `next_plan` rides along, synthesise a
                        // `StartEpoch` so the inner starts the next window
                        // without a separate coord round-trip. Ordering is
                        // safe: the param bridge's `Update(avg)` was sent
                        // (same control channel) before its SyncAck, and
                        // the coord only emits this frame after that ack,
                        // so the inner dequeues `Update(avg)` before this
                        // `StartEpoch` (mpsc FIFO).
                        ControlMsgWire::Update { next_plan, .. } => {
                            if let Some(plan) = next_plan {
                                let msg = ControlMsg::StartEpoch(EpochPlan {
                                    epoch: plan.epoch as usize,
                                    partition_offset: plan.partition_offset as usize,
                                    partition_size: plan.partition_size as usize,
                                });
                                if control_tx.send(msg).is_err() {
                                    // Inner GpuWorker dropped its receiver.
                                    return;
                                }
                            }
                        }
                        // Everything else: existing path through
                        // control_wire_to_msg → inner control_tx.
                        other => match control_wire_to_msg(other) {
                            Ok(Some(msg)) => {
                                if matches!(msg, ControlMsg::Shutdown) {
                                    // Clean teardown announced: the EOF
                                    // that follows is expected — do not
                                    // poison the peer ledger for it.
                                    clean_shutdown_seen = true;
                                }
                                if control_tx.send(msg).is_err() {
                                    // Inner GpuWorker dropped its receiver.
                                    return;
                                }
                            }
                            Ok(None) => {
                                // Wire-side notification with no in-process
                                // dispatch (e.g. Update{version} —
                                // informational; the param bridge handles
                                // the real ControlMsg::Update(AveragedParams).)
                            }
                            Err(e) => {
                                eprintln!(
                                    "cluster_worker: inbound r{rank} control_wire_to_msg: {e}"
                                );
                                if !clean_shutdown_seen {
                                    poison_peers();
                                }
                                inject_shutdown();
                                return;
                            }
                        },
                    },
                    Err(e) => {
                        eprintln!(
                            "cluster_worker: inbound r{rank} decode ControlMsgWire: {e}"
                        );
                        if !clean_shutdown_seen {
                            poison_peers();
                        }
                        inject_shutdown();
                        return;
                    }
                },
                other => {
                    // The control channel only carries Control frames
                    // in the coord→rank direction. Drop everything
                    // else with a diagnostic.
                    eprintln!(
                        "cluster_worker: inbound r{rank} unexpected MsgKind {other:?} \
                         on coord→rank channel; dropping"
                    );
                }
                }
            }
            Ok(LenFramedRead::WouldBlock) => {
                // Wedged-open coordinator: alive socket, no frames. Neither
                // EOF nor error ever fires, so the read-timeout poll alone
                // would loop here forever. If the coord has been silent past
                // the liveness deadline, treat it exactly like a hard link
                // drop — poison peers (NCCL watchdog aborts the collective)
                // and inject Shutdown so the rank exits with a death record.
                if last_inbound.elapsed() >= coord_liveness_deadline {
                    eprintln!(
                        "cluster_worker: inbound r{rank} coordinator silent for \
                         {coord_liveness_timeout_secs}s (presumed wedged); \
                         declaring peers dead and shutting down"
                    );
                    if !clean_shutdown_seen {
                        poison_peers();
                    }
                    inject_shutdown();
                    return;
                }
                continue;
            }
            Ok(LenFramedRead::Eof) => {
                if !clean_shutdown_seen {
                    poison_peers();
                }
                inject_shutdown();
                return;
            }
            Err(e) => {
                // Exit-time broken-pipe / EOF is the common case here:
                // the coord closed its end during shutdown. Downgrade
                // to verbose so steady-state logs stay clean.
                crate::verbose!("cluster_worker: inbound r{rank} wire error: {e}");
                if !clean_shutdown_seen {
                    poison_peers();
                }
                inject_shutdown();
                return;
            }
        }
    }
}

/// timing_rx → TCP bridge: drain in-process timing reports, encode
/// each as a `ControlFrame` and write to the coordinator.
/// Heartbeat cadence (ms). Fast enough that the coord's default 30s
/// staleness threshold catches a wedged rank within ~30 heartbeats,
/// slow enough that the per-cycle frame overhead is negligible.
const HEARTBEAT_CADENCE_MS: u64 = 1_000;

/// Worker-side heartbeat emitter. Fires a `TimingMsg::Heartbeat`
/// every [`HEARTBEAT_CADENCE_MS`] until `shutdown` is signalled or the
/// `timing_tx` channel closes (inner GpuWorker dropped). The heartbeat
/// flows through the outbound bridge alongside Batch / SyncAck / etc.,
/// so the coord receives liveness signal even while the inner is
/// blocked at the AllReduce barrier — distinguishing "alive at
/// barrier" from "dead."
pub(super) fn heartbeat_loop(
    rank: usize,
    timing_tx: mpsc::Sender<TimingMsg>,
    shutdown: Arc<AtomicBool>,
) {
    let mut step_count: usize = 0;
    while !shutdown.load(Ordering::SeqCst) {
        step_count = step_count.saturating_add(1);
        if timing_tx
            .send(TimingMsg::Heartbeat { rank, step_count })
            .is_err()
        {
            // Inner GpuWorker dropped → channel closed → exit.
            return;
        }
        thread::sleep(Duration::from_millis(HEARTBEAT_CADENCE_MS));
    }
}

/// Poll-cadence for the NCCL watchdog. 100ms keeps detection latency
/// low (a death registered by the inbound bridge is acted on within
/// this window) without burning CPU on the polling loop.
const NCCL_WATCHDOG_POLL_MS: u64 = 100;

/// NCCL watchdog thread body.
///
/// Polls `local_dead_ranks.dead_count()` and calls
/// [`abort`](crate::distributed::nccl::NcclAbortHandle::abort) on the
/// CURRENT handle each time the count
/// increases. The abort unblocks the main thread's in-flight NCCL
/// collective with an Err so the main thread can rebuild the comm on
/// the surviving cohort.
///
/// The handle is re-read from the shared slot on every firing:
/// `GpuWorker::replace_nccl_comm` refreshes the slot at each rebuild,
/// so cascading deaths always abort the live comm (a captured handle
/// would go stale after the first rebuild — its `aborted` flag already
/// tripped — leaving a second death with no abort path: permanent
/// hang). `abort()` stays idempotent per handle via that flag.
pub(super) fn nccl_watchdog_loop(
    rank: usize,
    abort_slot: crate::distributed::ddp_run::NcclAbortSlot,
    local_dead_ranks: Arc<crate::distributed::controller::DeadRanks>,
    shutdown: Arc<AtomicBool>,
) {
    let mut last_dead_count = 0usize;
    while !shutdown.load(Ordering::SeqCst) {
        let now_dead = local_dead_ranks.dead_count();
        if now_dead > last_dead_count {
            crate::verbose!(
                "  cluster_worker: rank {} NCCL watchdog: dead_count {} -> {}, \
                 aborting NCCL comm",
                rank,
                last_dead_count,
                now_dead,
            );
            let handle = abort_slot
                .lock()
                .expect("nccl abort slot poisoned")
                .clone();
            match handle {
                Some(h) => {
                    if let Err(e) = h.abort() {
                        eprintln!(
                            "cluster_worker: rank {} NCCL watchdog abort error: {}",
                            rank, e,
                        );
                    }
                }
                None => {
                    // Slot emptied (comm torn down for exit); nothing to
                    // abort.
                }
            }
            last_dead_count = now_dead;
        }
        thread::sleep(Duration::from_millis(NCCL_WATCHDOG_POLL_MS));
    }
}

pub(super) fn outbound_loop(
    rank: usize,
    stream: &mut TcpStream,
    salt: &SessionSalt,
    shutdown: &Arc<AtomicBool>,
    timing_rx: mpsc::Receiver<TimingMsg>,
    metrics_rx: mpsc::Receiver<crate::distributed::ddp_run::MetricsMsg>,
) {
    // Drain the rank-side dashboard intent stashed by the user's
    // Monitor calls (`monitor.serve`, `.watch`, `.set_metadata`,
    // captured hardware string). When a dashboard port has been
    // requested, emit the matching `TimingMsgWire::Dashboard*` frames
    // so the launcher's `ClusterDashboardSink` binds the HTTP server
    // and seeds its header / per-rank tabs. Construct a local
    // `ResourceSampler` only when something consumes the samples
    // (dashboard opt-in or the envelope's `rank_resources` flag) —
    // sampling costs a /proc/stat parse + the NVML poller thread,
    // neither worth paying for runs with no consumer.
    let pending = crate::distributed::cluster_dashboard_emit::drain();
    // Per-rank assigned CUDA device. On hosts where
    // `CUDA_VISIBLE_DEVICES` is scoped per rank (`gpu_device_count()
    // == 1`) the sampler returns a single GPU and the filter is a
    // no-op. On hosts where multiple physical GPUs are visible to
    // every rank (Pascal-via-VFIO observed: r1 uses cuda 0, r2 uses
    // cuda 1, both processes see both devices) the sampler returns
    // two snapshots — only ONE belongs to this rank's worker. Without
    // filtering, the dashboard sink would take `.first()` and report
    // the WRONG device's allocator stats (zero, since this process
    // never allocated there). Pull the assigned device index here so
    // `write_metrics` can strip foreign-device entries before shipping,
    // and so `emit_dashboard_setup` can trim the rank's hardware
    // string to its own GPU (the launcher's sink then groups per
    // host and lists per-rank GPU labels without dupes).
    let envelope = crate::distributed::LocalCluster::from_env().ok().flatten();
    let assigned_device_idx: Option<u8> = envelope
        .as_ref()
        .and_then(|c| c.my_rank().ok())
        .and_then(|(_, dev)| match dev {
            crate::tensor::Device::CUDA(idx) => Some(idx),
            _ => None,
        });
    // Ship whatever the harness stashed, gated on there being something to
    // ship rather than on a dashboard port.
    //
    // `emit_dashboard_setup` already port-gates the one frame that needs a port
    // (`DashboardRegister`); the SVG / metadata / hardware are wanted by the
    // SAVED ARCHIVE too, which binds no HTTP server at all. Gating the whole
    // sequence on `port` made `--save-dashboard` without `--monitor` produce a
    // page with no graph and no hyperparameters — the persisted dashboard
    // silently depending on a live one.
    let has_setup_payload = pending.port.is_some()
        || pending.svg.is_some()
        || pending.metadata_json.is_some()
        || pending.hardware.is_some();
    if has_setup_payload {
        emit_dashboard_setup(stream, salt, rank, &pending, assigned_device_idx);
    }
    // Sampling turns on for either consumer of the samples: the live
    // dashboard the user requested via `monitor.serve(port)`, or the
    // controller's timeline persistence requested through the
    // envelope's `rank_resources` flag.
    let want_resources = pending.port.is_some()
        || envelope.as_ref().is_some_and(|c| c.rank_resources);
    let resource_sampler: Option<std::sync::Mutex<crate::monitor::ResourceSampler>> =
        if want_resources {
            Some(std::sync::Mutex::new(
                crate::monitor::ResourceSampler::new(),
            ))
        } else {
            None
        };
    // Sub-epoch resource pacing. Resources also ride the per-epoch metrics
    // frame, but a single-pass LLM run has ONE epoch — that path would yield a
    // single GPU/VRAM reading for a run lasting hours, the same dead end
    // `reports_per_epoch` fixes for loss. So sample on a wall-clock throttle
    // here, decoupled from the epoch boundary. `sample()` reads /proc/stat
    // plus the already-running NVML poller's accumulator, so this costs a
    // frame, not a device query.
    let mut last_resource_emit: Option<std::time::Instant> = None;
    // recv_timeout so we can periodically check the shutdown flag and
    // service the lower-frequency metrics channel between timing
    // frames. Single thread = serial writes on `stream`; no socket-
    // share race.
    loop {
        if shutdown.load(Ordering::SeqCst) {
            // Drain any final messages so a SyncAck, Exiting, or
            // final-epoch MetricsMsg doesn't get lost on exit.
            while let Ok(msg) = timing_rx.try_recv() {
                let _ = write_timing(stream, salt, msg);
            }
            while let Ok(msg) = metrics_rx.try_recv() {
                let _ = write_metrics(stream, salt, msg, resource_sampler.as_ref(), assigned_device_idx);
            }
            return;
        }
        // Metrics first: lower frequency (per-chunk / per-epoch) and
        // latency-sensitive for dashboard surfacing. Cheap when empty
        // (try_recv returns immediately).
        match metrics_rx.try_recv() {
            Ok(msg) => {
                if let Err(e) = write_metrics(stream, salt, msg, resource_sampler.as_ref(), assigned_device_idx) {
                    crate::verbose!(
                        "cluster_worker: outbound r{rank} metrics write error: {e}"
                    );
                    return;
                }
                continue;
            }
            Err(mpsc::TryRecvError::Empty) => {}
            Err(mpsc::TryRecvError::Disconnected) => {
                // Metrics sender dropped; timing channel may still be
                // alive (e.g. heartbeats during teardown). Fall through
                // to timing drain — timing's Disconnected arm exits.
            }
        }
        // Due on the writer's own 250ms tick, so it fires whether or not the
        // training loop is producing frames — a rank stuck in a long barrier
        // keeps reporting its resources.
        if let Some(sampler) = resource_sampler.as_ref() {
            let due = last_resource_emit.is_none_or(|t| {
                t.elapsed() >= Duration::from_millis(super::RESOURCE_SAMPLE_INTERVAL_MS)
            });
            if due {
                let sample = {
                    let mut s = sampler.lock().unwrap();
                    let mut sample = s.sample();
                    trim_sample_to_assigned_device(&mut sample, assigned_device_idx);
                    sample
                };
                last_resource_emit = Some(std::time::Instant::now());
                if let Err(e) = write_timing_wire(
                    stream,
                    salt,
                    &crate::distributed::wire::TimingMsgWire::ResourceSample {
                        rank: rank as u64,
                        sample: sample.into(),
                    },
                ) {
                    crate::verbose!(
                        "cluster_worker: outbound r{rank} resource write error: {e}"
                    );
                    return;
                }
            }
        }
        match timing_rx.recv_timeout(Duration::from_millis(250)) {
            Ok(msg) => {
                if let Err(e) = write_timing(stream, salt, msg) {
                    // Exit-time BrokenPipe is the common case: coord
                    // dropped its end during shutdown. Downgrade so it
                    // doesn't drown steady-state logs.
                    crate::verbose!("cluster_worker: outbound r{rank} write error: {e}");
                    return;
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                // Inner GpuWorker dropped → drain just in case (no-op
                // since Disconnected means buffer empty) and exit.
                return;
            }
        }
    }
}


/// Serialize a `ControlFrame` and write it length-delimited to the
/// worker's host relay, which forwards the opaque blob upstream to the
/// coordinator. Control-channel mirror of the data channel's framing.
pub(super) fn write_framed_control<W: std::io::Write>(stream: &mut W, frame: &ControlFrame) -> Result<()> {
    let mut buf = Vec::new();
    frame.write_to(&mut buf)?;
    write_len_framed(stream, &buf)
}

pub(super) fn write_timing(
    stream: &mut TcpStream,
    salt: &SessionSalt,
    msg: TimingMsg,
) -> Result<()> {
    write_timing_wire(stream, salt, &timing_msg_to_wire(msg))
}

pub(super) fn write_metrics(
    stream: &mut TcpStream,
    salt: &SessionSalt,
    msg: crate::distributed::ddp_run::MetricsMsg,
    resource_sampler: Option<&std::sync::Mutex<crate::monitor::ResourceSampler>>,
    assigned_device_idx: Option<u8>,
) -> Result<()> {
    let mut wire = metrics_msg_to_wire(msg);
    if let Some(sampler) = resource_sampler {
        // Mutex held briefly — sampler::sample reads /proc/stat +
        // copies the GPU poller's accumulator. No collective; cheap.
        let mut s = sampler.lock().unwrap();
        let mut sample = s.sample();
        trim_sample_to_assigned_device(&mut sample, assigned_device_idx);
        wire.resources = Some(sample.into());
    }
    let frame = ControlFrame::encode(salt, MsgKind::Metrics, &wire)?;
    write_framed_control(stream, &frame)
}

/// Strip foreign-device GPU entries from a resource sample.
///
/// When `CUDA_VISIBLE_DEVICES` isn't scoped per rank, the sampler returns one
/// snapshot per physical device — but only the rank's assigned device carries
/// this process's allocator stats, and consumers take `gpus.first()`. Only
/// filter when there is something to disambiguate: a scoped-down rank
/// (`CUDA_VISIBLE_DEVICES=<phys>`, the launcher's per-child spawn recipe)
/// already sees exactly its own device, and its snapshot carries the PHYSICAL
/// index while `assigned_device_idx` carries the remapped runtime index
/// (`my_rank` returns `CUDA(0)` for scoped children) — matching them would
/// empty the list for every rank whose physical device isn't 0 (observed:
/// pascal r2 shipped no GPU slice at all, blinding both the dashboard tab and
/// the timeline).
///
/// Shared by both emit paths (the per-epoch metrics piggy-back and the
/// sub-epoch `ResourceSample` frame) so the two cannot disagree about which
/// device a rank is reporting.
pub(super) fn trim_sample_to_assigned_device(
    sample: &mut crate::monitor::ResourceSample,
    assigned_device_idx: Option<u8>,
) {
    if sample.gpus.len() > 1
        && let Some(target) = assigned_device_idx {
            sample.gpus.retain(|g| g.device_index == target);
        }
}

/// Emit the rank-side dashboard setup sequence — `DashboardRegister`
/// gated on `port`, plus `DashboardSetSvg` / `DashboardSetMetadata` /
/// `DashboardSetHardware` whenever the stash holds a value. Called
/// once at outbound-loop startup after the user's harness has had a
/// chance to populate the stash through `monitor.serve` /
/// `monitor.watch` / `monitor.set_metadata` and `Monitor::new`'s
/// hardware capture. Errors are logged verbosely but never abort the
/// rank — the dashboard is optional UX, not a training invariant.
pub(super) fn emit_dashboard_setup(
    stream: &mut TcpStream,
    salt: &SessionSalt,
    rank: usize,
    pending: &crate::distributed::cluster_dashboard_emit::PendingDashboardConfig,
    assigned_device_idx: Option<u8>,
) {
    use crate::distributed::wire::TimingMsgWire;
    let rank_u64 = rank as u64;
    let mut emit = |msg: TimingMsgWire| {
        if let Err(e) = write_timing_wire(stream, salt, &msg) {
            crate::verbose!(
                "cluster_worker: outbound r{rank} dashboard emit failed: {e}",
            );
        }
    };
    if let Some(port) = pending.port {
        emit(TimingMsgWire::DashboardRegister { rank: rank_u64, port });
    }
    if let Some(ref svg) = pending.svg {
        emit(TimingMsgWire::DashboardSetSvg {
            rank: rank_u64,
            svg: svg.clone(),
            label: pending.label.clone(),
            hash: pending.hash.clone(),
        });
    }
    if let Some(ref json) = pending.metadata_json {
        emit(TimingMsgWire::DashboardSetMetadata {
            rank: rank_u64,
            json: json.clone(),
        });
    }
    if let Some(ref hw) = pending.hardware {
        // `tensor::hardware_summary` returns `CPU | gpu0 | gpu1 | …`
        // — every visible GPU. In cluster mode each rank only USES one
        // GPU (its assigned device); listing the others puffs the
        // launcher's header and visually repeats hardware across ranks
        // on the same host. Trim to `CPU | <my_gpu>` so the sink's
        // per-host grouping can render: `host: cpu | gr=N lr=M: gpu |
        // gr=K lr=L: gpu | other_host: …`.
        let trimmed = trim_hardware_to_assigned(hw, assigned_device_idx);
        emit(TimingMsgWire::DashboardSetHardware {
            rank: rank_u64,
            summary: trimmed,
        });
    }
}

/// Split `full` on `" | "` and keep `[0]` (CPU) + the GPU at
/// `assigned_device_idx` if present. Returns the original string
/// untouched when no assigned device is known (single-process / CPU
/// builds) or when the segment count doesn't match the expected
/// `cpu | gpu0 | gpu1 | …` shape (e.g. NVML returned no GPU names).
pub(super) fn trim_hardware_to_assigned(
    full: &str,
    assigned_device_idx: Option<u8>,
) -> String {
    let Some(target) = assigned_device_idx else {
        return full.to_string();
    };
    let parts: Vec<&str> = full.split(" | ").collect();
    if parts.len() < 2 {
        return full.to_string();
    }
    let cpu = parts[0];
    // GPUs are positionally indexed: parts[1] = device 0, parts[2] =
    // device 1, etc. Use `target + 1` to index into the GPU portion.
    let gpu_idx = target as usize + 1;
    match parts.get(gpu_idx) {
        Some(gpu) => format!("{cpu} | {gpu}"),
        None => cpu.to_string(),
    }
}

/// Write an already-built [`TimingMsgWire`](crate::distributed::wire::TimingMsgWire)
/// directly. The non-wire
/// `write_timing` takes an in-process `TimingMsg`; the dashboard emit
/// path skips that intermediate and serializes the wire form directly
/// (no in-process `TimingMsg` variant exists for these — they're a
/// pure wire concern).
pub(super) fn write_timing_wire(
    stream: &mut TcpStream,
    salt: &SessionSalt,
    msg: &crate::distributed::wire::TimingMsgWire,
) -> Result<()> {
    let frame = ControlFrame::encode(salt, MsgKind::Timing, msg)?;
    write_framed_control(stream, &frame)
}

