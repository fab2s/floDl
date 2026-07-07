//! TCP rendezvous for multi-host DDP startup.
//!
//! Process-per-rank topology. The **controller** (the orchestrator
//! process running on the launcher host) binds
//! [`controller.port`](super::cluster::ControllerBlock::port) and
//! drives the entire bootstrap. Every rank — including rank 0 — dials
//! in, sends a [`Hello`](RendezvousMsgWire::Hello) carrying its
//! dataset signature, global rank, and host name, and receives back a
//! [`Role`](RendezvousMsgWire::Role) telling it whether to generate
//! the NCCL unique ID locally or wait for the controller to broadcast
//! one.
//!
//! The controller picks the generator rank by policy: the first rank
//! of a local-host worker if any, else `workers[0].ranks[0]`. The
//! controller cannot call `ncclGetUniqueId` itself — its process is
//! orchestration-only and may not link libnccl. Same constraint and
//! same delegation pattern as the elastic-resize path
//! ([`ControlMsgWire::RequestNewNcclId`]).
//!
//! Every wire frame uses the existing [`ControlFrame`] framing tagged
//! with [`MsgKind::Rendezvous`] and HMAC-signed against the per-session
//! salt the launcher generated. Cross-session frames fail
//! authentication with probability 2^-64; dataset-signature mismatch
//! across ranks fails loudly with the offending host name.
//!
//! Workers retry the TCP `connect` for ~30 s to absorb cold-boot
//! ordering jitter between hosts. Hard error after the budget
//! exhausts — silent infinite retry would hide misconfiguration.
//!
//! [`ControlFrame`]: super::wire::ControlFrame
//! [`MsgKind::Rendezvous`]: super::wire::MsgKind::Rendezvous
//! [`ControlMsgWire::RequestNewNcclId`]: super::wire::ControlMsgWire::RequestNewNcclId

use std::env;
use std::io::Write;
use std::net::{TcpListener, TcpStream};
use std::time::{Duration, Instant};

use crate::{Device, Result, TensorError};

use super::launcher::FullCluster;
use super::wire::{
    ControlFrame, MsgKind, RendezvousMsgWire, RendezvousRole, SessionSalt,
};
use super::{LocalCluster, NCCL_UNIQUE_ID_BYTES, NcclUniqueId, WorkerBlock};

const HOSTNAME_MAX_LEN: usize = 255;
const IO_TIMEOUT: Duration = Duration::from_secs(30);
const ENV_NCCL_SOCKET_IFNAME: &str = "NCCL_SOCKET_IFNAME";

/// Wall-clock budget the controller waits with NO new rank slotting
/// before it abandons cohort formation. Generous enough to absorb
/// cold-boot ordering jitter between hosts (the rank-side connect retry
/// is ~30s), but bounded so a rank that never launches — or a port
/// scanner hammering the `0.0.0.0` listener — cannot wedge the cohort
/// indefinitely while honest ranks wait. Resets on every successful
/// accept, so a legitimate staggered start (rank N arriving long after
/// rank 0) keeps the window open as long as ranks keep trickling in.
const RENDEZVOUS_IDLE_TIMEOUT: Duration = Duration::from_secs(120);

/// Poll cadence for the non-blocking accept loop while waiting on the
/// next rank to connect.
const RENDEZVOUS_POLL_INTERVAL: Duration = Duration::from_millis(200);

/// Hard cap on pre-authentication rejected connections (bad frame,
/// non-Hello first message, timeout-setup failure) before the controller
/// bails loudly. A misconfigured peer or a scanner pointed at the
/// rendezvous port can otherwise churn this loop without ever advancing
/// `accepted`.
const MAX_REJECTED_CONNECTIONS: usize = 1024;

/// Result of the TCP rendezvous: this host's local rank/device list plus the
/// cluster-wide NCCL unique ID.
///
/// Construct via [`LocalCluster::rendezvous`](super::LocalCluster::rendezvous).
#[derive(Debug)]
pub struct TcpRendezvous {
    world_size: usize,
    local_ranks: Vec<usize>,
    local_devices: Vec<Device>,
    unique_id: NcclUniqueId,
}

impl TcpRendezvous {
    /// Total ranks across the cluster.
    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// Global ranks owned by this host, in YAML-declared order.
    pub fn local_ranks(&self) -> &[usize] {
        &self.local_ranks
    }

    /// CUDA devices backing each local rank, paired by position with
    /// [`local_ranks`](Self::local_ranks).
    pub fn local_devices(&self) -> &[Device] {
        &self.local_devices
    }

    /// Shared NCCL unique ID, identical on every host.
    pub fn unique_id(&self) -> &NcclUniqueId {
        &self.unique_id
    }

    /// Drive the rank-side rendezvous handshake. Invoked from
    /// [`LocalCluster::rendezvous`](super::LocalCluster::rendezvous).
    ///
    /// The `gen_uid` closure is called at most once — only when the
    /// controller assigns this rank the [`RendezvousRole::Generate`]
    /// role. Production passes [`NcclUniqueId::new`]; tests pass a
    /// closure that returns a fixed-byte stub to avoid linking against
    /// CUDA-bound NCCL.
    pub(crate) fn establish<F>(
        cluster: &LocalCluster,
        dataset_signature: [u8; 32],
        gen_uid: F,
    ) -> Result<Self>
    where
        F: FnOnce() -> Result<NcclUniqueId>,
    {
        let this_host = cluster.this_worker()?;
        validate_socket_ifname(cluster, this_host)?;

        let local_ranks = this_host.ranks.clone();
        let local_devices: Vec<Device> = this_host
            .local_devices
            .iter()
            .map(|&d| Device::CUDA(d))
            .collect();
        let (my_global_rank, _) = cluster.my_rank()?;
        let host_name = this_host.host.clone();

        let uid_bytes = run_rank_rendezvous(
            cluster,
            &cluster.salt,
            dataset_signature,
            u32::try_from(my_global_rank).map_err(|_| {
                TensorError::new(&format!(
                    "rendezvous: global_rank {my_global_rank} does not fit in u32"
                ))
            })?,
            &host_name,
            gen_uid,
        )?;

        crate::msg!("cluster: {}", cluster_mapping(cluster));

        Ok(TcpRendezvous {
            world_size: cluster.world_size(),
            local_ranks,
            local_devices,
            unique_id: NcclUniqueId::from_bytes(uid_bytes),
        })
    }
}

// ---------------------------------------------------------------------------
// Rank side: dial controller, send Hello, await Role, generate-or-receive UID
// ---------------------------------------------------------------------------

/// Connect to the controller, do the Hello/Role/Uid round-trip, return
/// the raw UID bytes. Pure wire-protocol work — no CUDA / NCCL link
/// needed except via the `gen_uid` closure on the designated generator.
fn run_rank_rendezvous<F>(
    cluster: &LocalCluster,
    salt: &SessionSalt,
    dataset_sig: [u8; 32],
    global_rank: u32,
    host_name: &str,
    gen_uid: F,
) -> Result<[u8; NCCL_UNIQUE_ID_BYTES]>
where
    F: FnOnce() -> Result<NcclUniqueId>,
{
    if host_name.len() > HOSTNAME_MAX_LEN {
        return Err(TensorError::new(&format!(
            "rendezvous: host name {host_name:?} exceeds {HOSTNAME_MAX_LEN} bytes"
        )));
    }

    // Bracket IPv6 literals: a bare `fe80::1` concatenated with `:port`
    // is ambiguous and fails `ToSocketAddrs`; `[fe80::1]:port` is correct.
    let addr = crate::distributed::wire::join_host_port(
        &cluster.controller.host,
        cluster.controller.port,
    );
    let mut stream =
        crate::distributed::wire::connect_with_retry(addr.as_str(), "rendezvous")?;
    stream
        .set_read_timeout(Some(IO_TIMEOUT))
        .and_then(|()| stream.set_write_timeout(Some(IO_TIMEOUT)))
        .map_err(|e| {
            TensorError::new(&format!("rendezvous: setting timeouts failed: {e}"))
        })?;

    let hello = RendezvousMsgWire::Hello {
        dataset_sig,
        global_rank,
        host_name: host_name.to_string(),
    };
    write_rendezvous_frame(&mut stream, salt, &hello)?;

    let role = match read_rendezvous_frame(&mut stream, salt)? {
        RendezvousMsgWire::Role(r) => r,
        other => {
            return Err(TensorError::new(&format!(
                "rendezvous: expected Role frame from controller, got {other:?}"
            )));
        }
    };

    let uid_vec = match role {
        RendezvousRole::Generate => {
            let uid = gen_uid()?;
            let bytes = *uid.as_bytes();
            write_rendezvous_frame(
                &mut stream,
                salt,
                &RendezvousMsgWire::Uid { uid_bytes: bytes.to_vec() },
            )?;
            bytes.to_vec()
        }
        RendezvousRole::Wait => match read_rendezvous_frame(&mut stream, salt)? {
            RendezvousMsgWire::Uid { uid_bytes } => uid_bytes,
            other => {
                return Err(TensorError::new(&format!(
                    "rendezvous: expected Uid frame from controller, got {other:?}"
                )));
            }
        },
    };

    let mut uid = [0u8; NCCL_UNIQUE_ID_BYTES];
    if uid_vec.len() != NCCL_UNIQUE_ID_BYTES {
        return Err(TensorError::new(&format!(
            "rendezvous: UID payload length {} != {NCCL_UNIQUE_ID_BYTES}",
            uid_vec.len()
        )));
    }
    uid.copy_from_slice(&uid_vec);
    Ok(uid)
}

// ---------------------------------------------------------------------------
// Controller side: bind, accept world_size connections, delegate UID gen
// ---------------------------------------------------------------------------

/// Run the controller-side bootstrap rendezvous server.
///
/// Binds `0.0.0.0:<full.controller.port>`, accepts one connection per
/// rank (`full.world_size()` total), validates that every Hello carries
/// the same dataset signature, designates one rank as UID generator,
/// and broadcasts the resulting NCCL unique ID to every Wait rank.
///
/// `local_host_name` is the controller's own resolved hostname; used to
/// detect whether one of the workers is co-located (fork/exec-style)
/// and prefer that worker's first rank as the generator to minimize
/// bootstrap latency. When no local worker exists (the common
/// orchestrator-only case), the generator is `workers[0].ranks[0]`.
///
/// Blocks until every rank has its UID or any rank fails the handshake.
/// On failure, surviving streams are dropped; the launcher detects the
/// error via this function's `Result` and tears down spawned children.
///
/// Cohort formation is bounded: the accept loop gives up with a loud
/// error if no new rank slots within [`RENDEZVOUS_IDLE_TIMEOUT`] (a rank
/// that never launched, or a stalled host) or if pre-authentication
/// rejected connections exceed [`MAX_REJECTED_CONNECTIONS`] (a scanner /
/// misconfigured peer hammering the `0.0.0.0` listener). Without these
/// ceilings a single absent rank would hang the whole cohort forever.
pub fn run_controller_rendezvous(
    full: &FullCluster,
    local_host_name: &str,
) -> Result<()> {
    run_controller_rendezvous_with(full, local_host_name, RENDEZVOUS_IDLE_TIMEOUT)
}

/// Inner body of [`run_controller_rendezvous`], parameterized by the
/// no-progress `idle_timeout` so tests can exercise the wedge-break
/// ceiling without waiting the production [`RENDEZVOUS_IDLE_TIMEOUT`].
fn run_controller_rendezvous_with(
    full: &FullCluster,
    local_host_name: &str,
    idle_timeout: Duration,
) -> Result<()> {
    let world_size = full.world_size();
    if world_size == 0 {
        return Err(TensorError::new(
            "rendezvous: empty cluster (world_size = 0)",
        ));
    }

    // Back-to-back runs on the fixed port block are safe as-is: Rust's
    // `TcpListener::bind` sets SO_REUSEADDR on Unix, so TIME_WAIT remnants
    // from a previous run's connections never block this bind (probed:
    // rebind succeeds with the port verifiably in TIME_WAIT). A genuinely
    // LIVE listener from a still-running launcher still fails loudly here
    // (that would need SO_REUSEPORT) — the desirable double-run guard.
    let bind_addr = format!("0.0.0.0:{}", full.controller.port);
    let listener = TcpListener::bind(&bind_addr).map_err(|e| {
        TensorError::new(&format!(
            "rendezvous: controller failed to bind {bind_addr}: {e}"
        ))
    })?;

    let designated_rank = pick_designated_rank(full, local_host_name);
    eprintln!(
        "cluster launcher: rendezvous server bound on {} (world_size={world_size}, generator=rank {designated_rank})",
        listener.local_addr().map(|a| a.to_string()).unwrap_or_else(|_| bind_addr.clone()),
    );

    // Non-blocking accept so the loop can enforce a wall-clock ceiling
    // instead of parking forever in a blocking accept() when a rank never
    // dials in. Accepted streams are flipped back to blocking below so
    // their per-stream read/write timeouts take effect.
    listener.set_nonblocking(true).map_err(|e| {
        TensorError::new(&format!(
            "rendezvous: controller failed to set non-blocking accept: {e}"
        ))
    })?;

    // Indexed by global_rank — each accepted stream slotted by the rank
    // it claims in its Hello. `None` until that rank arrives.
    let mut streams: Vec<Option<TcpStream>> = (0..world_size).map(|_| None).collect();
    let mut reference_sig: Option<[u8; 32]> = None;

    let mut accepted = 0usize;
    // Pre-auth rejected connections (bad frame / non-Hello / setup fail).
    let mut rejected = 0usize;
    // Resets on every successful slot fill; a window with no progress is
    // the wedge signature this ceiling exists to break.
    let mut last_progress = Instant::now();
    while accepted < world_size {
        let (mut stream, peer) = match listener.accept() {
            Ok(pair) => pair,
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                if last_progress.elapsed() > idle_timeout {
                    return Err(TensorError::new(&format!(
                        "rendezvous: timed out after {}s with no new rank \
                         connecting ({accepted}/{world_size} ranks in). Check \
                         that every rank process launched and can reach the \
                         controller at {bind_addr}.",
                        idle_timeout.as_secs(),
                    )));
                }
                std::thread::sleep(RENDEZVOUS_POLL_INTERVAL);
                continue;
            }
            Err(e) => {
                return Err(TensorError::new(&format!(
                    "rendezvous: controller accept failed: {e}"
                )));
            }
        };
        // Accepted socket may inherit the listener's non-blocking flag;
        // flip it back so the per-stream read/write timeouts below are
        // honored (a non-blocking socket ignores SO_RCVTIMEO).
        if stream.set_nonblocking(false).is_err()
            || stream
                .set_read_timeout(Some(IO_TIMEOUT))
                .and_then(|()| stream.set_write_timeout(Some(IO_TIMEOUT)))
                .is_err()
        {
            rejected += 1;
            if rejected > MAX_REJECTED_CONNECTIONS {
                return Err(rejected_cap_error(world_size, accepted));
            }
            continue;
        }

        // PER-CONNECTION REJECTION. This listener sits on 0.0.0.0: any
        // stray TCP connection (port scanner, health checker, a rank
        // from another cluster) that fails frame parse / HMAC here used
        // to abort the WHOLE cohort bootstrap while honest ranks hung
        // out their timeouts. A pre-authentication failure condemns the
        // connection, not the cohort. HMAC-valid protocol violations
        // (dataset-sig mismatch, duplicate/out-of-range rank) below
        // remain cohort-fatal — those are real members misconfigured.
        let msg = match read_rendezvous_frame(&mut stream, &full.salt) {
            Ok(m) => m,
            Err(e) => {
                eprintln!(
                    "cluster launcher: rendezvous rejected connection from \
                     {peer} (bad frame: {e}); continuing to accept"
                );
                rejected += 1;
                if rejected > MAX_REJECTED_CONNECTIONS {
                    return Err(rejected_cap_error(world_size, accepted));
                }
                continue;
            }
        };
        let (dataset_sig, global_rank, host_name) = match msg {
            RendezvousMsgWire::Hello { dataset_sig, global_rank, host_name } => {
                (dataset_sig, global_rank, host_name)
            }
            other => {
                eprintln!(
                    "cluster launcher: rendezvous rejected connection from \
                     {peer} (expected Hello, got {other:?}); continuing to accept"
                );
                rejected += 1;
                if rejected > MAX_REJECTED_CONNECTIONS {
                    return Err(rejected_cap_error(world_size, accepted));
                }
                continue;
            }
        };

        match reference_sig {
            None => reference_sig = Some(dataset_sig),
            Some(ref expected) if &dataset_sig != expected => {
                return Err(TensorError::new(&format!(
                    "rendezvous: dataset_signature mismatch from host {host_name:?} \
                     rank {global_rank} (peer {peer}). Each rank must read from the \
                     same dataset; silent fan-out across diverging shards is the \
                     worst class of bug."
                )));
            }
            _ => {}
        }

        let rank_idx = usize::try_from(global_rank).map_err(|_| {
            TensorError::new(&format!(
                "rendezvous: rank {global_rank} from {host_name:?} (peer {peer}) \
                 does not fit in usize"
            ))
        })?;
        if rank_idx >= world_size {
            return Err(TensorError::new(&format!(
                "rendezvous: rank {global_rank} from {host_name:?} (peer {peer}) \
                 out of bounds for world_size {world_size}"
            )));
        }
        if streams[rank_idx].is_some() {
            // Duplicate Hello for an already-slotted rank. This is reached
            // ONLY inside the accept loop, i.e. BEFORE any Role frame is
            // dispatched (Role/UID exchange runs after the loop completes), so
            // the earlier connection was never told its role: the rank is
            // simply re-announcing after a TCP blip dropped its first socket.
            // Replace the stale (now-dead) stream with the live one rather than
            // aborting the whole cohort over a transient reconnect. Do NOT
            // re-count — the rank already filled its slot; only the underlying
            // connection changed.
            eprintln!(
                "cluster launcher: rendezvous rank {global_rank} from \
                 {host_name:?} (peer {peer}) reconnected before role dispatch; \
                 replacing stale stream (transient TCP blip, not cohort-fatal)"
            );
            streams[rank_idx] = Some(stream);
            last_progress = Instant::now();
            continue;
        }
        streams[rank_idx] = Some(stream);
        accepted += 1;
        last_progress = Instant::now();
    }

    // Every rank has connected. Send Role to each, collect UID from
    // generator, broadcast to waiters.
    for (rank_idx, slot) in streams.iter_mut().enumerate() {
        let stream = slot.as_mut().expect("every slot filled by accept loop");
        let role = if rank_idx as u32 == designated_rank {
            RendezvousRole::Generate
        } else {
            RendezvousRole::Wait
        };
        write_rendezvous_frame(stream, &full.salt, &RendezvousMsgWire::Role(role))?;
    }

    let designated_idx = designated_rank as usize;
    let uid_bytes = {
        let stream = streams[designated_idx]
            .as_mut()
            .expect("designated stream filled");
        match read_rendezvous_frame(stream, &full.salt)? {
            RendezvousMsgWire::Uid { uid_bytes } => uid_bytes,
            other => {
                return Err(TensorError::new(&format!(
                    "rendezvous: expected Uid from generator rank {designated_rank}, got {other:?}"
                )));
            }
        }
    };
    if uid_bytes.len() != NCCL_UNIQUE_ID_BYTES {
        return Err(TensorError::new(&format!(
            "rendezvous: generator rank {designated_rank} sent UID of length {} \
             (expected {NCCL_UNIQUE_ID_BYTES})",
            uid_bytes.len()
        )));
    }

    for (rank_idx, slot) in streams.iter_mut().enumerate() {
        if rank_idx == designated_idx {
            continue;
        }
        let stream = slot.as_mut().expect("every slot filled");
        write_rendezvous_frame(
            stream,
            &full.salt,
            &RendezvousMsgWire::Uid { uid_bytes: uid_bytes.clone() },
        )?;
    }

    Ok(())
}

/// Loud error when the controller's accept loop has rejected more than
/// [`MAX_REJECTED_CONNECTIONS`] pre-authentication connections without
/// forming the cohort — the scanner / misconfigured-peer wedge.
fn rejected_cap_error(world_size: usize, accepted: usize) -> TensorError {
    TensorError::new(&format!(
        "rendezvous: aborting after {MAX_REJECTED_CONNECTIONS} rejected \
         pre-auth connections with only {accepted}/{world_size} ranks in. \
         Something is hammering the rendezvous port (scanner, health \
         checker, or a peer from another session/cluster)."
    ))
}

/// Pick which rank should generate the NCCL unique ID at bootstrap.
///
/// Preference: the first rank of a co-located worker (worker whose host
/// matches `local_host_name`) to minimize controller↔generator latency
/// and keep the round-trip on loopback when possible. When no worker is
/// co-located (orchestrator-only controller, the common multi-host
/// case), falls back to `workers[0].ranks[0]`.
pub fn pick_designated_rank(full: &FullCluster, local_host_name: &str) -> u32 {
    for worker in &full.workers {
        if worker.host == local_host_name {
            if let Some(&r) = worker.ranks.first() {
                return r as u32;
            }
        }
    }
    full.workers
        .first()
        .and_then(|w| w.ranks.first())
        .copied()
        .map(|r| r as u32)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Wire helpers (shared by rank and controller sides)
// ---------------------------------------------------------------------------

fn write_rendezvous_frame<W: Write>(
    w: &mut W,
    salt: &SessionSalt,
    msg: &RendezvousMsgWire,
) -> Result<()> {
    let frame = ControlFrame::encode(salt, MsgKind::Rendezvous, msg)?;
    frame.write_to(w)
}

fn read_rendezvous_frame(
    stream: &mut TcpStream,
    salt: &SessionSalt,
) -> Result<RendezvousMsgWire> {
    let frame = ControlFrame::read_from(stream, salt)?.ok_or_else(|| {
        TensorError::new("rendezvous: peer closed stream before sending frame")
    })?;
    if frame.kind != MsgKind::Rendezvous {
        return Err(TensorError::new(&format!(
            "rendezvous: expected MsgKind::Rendezvous, got {:?}",
            frame.kind
        )));
    }
    frame.decode()
}


fn validate_socket_ifname(cluster: &LocalCluster, this_host: &WorkerBlock) -> Result<()> {
    if !cluster.spans_multiple_workers() {
        return Ok(());
    }
    if env::var(ENV_NCCL_SOCKET_IFNAME).is_ok() {
        return Ok(());
    }
    // Auto-export from cluster.yml's hosts[].nccl_socket_ifname so the
    // user declares the interface once (in YAML) instead of also having
    // to set the env var explicitly. The yml field is non-decorative:
    // fdl-cli's config validation already requires it non-empty on
    // multi-host clusters, so we expect a value here in the normal path.
    // The loud error remains for programmatic LocalCluster constructions
    // that skip fdl-cli's validation and leave the field empty.
    if !this_host.nccl_socket_ifname.trim().is_empty() {
        // SAFETY: rank process is single-threaded at this point (pre-NCCL,
        // pre-worker-spawn); no concurrent env::var readers.
        unsafe {
            env::set_var(ENV_NCCL_SOCKET_IFNAME, &this_host.nccl_socket_ifname);
        }
        return Ok(());
    }
    Err(TensorError::new(&format!(
        "rendezvous: {ENV_NCCL_SOCKET_IFNAME} must be set when the cluster spans \
         multiple hosts (auto-detection rejected -- interface naming is \
         config-specific and silent fallthrough costs hours)"
    )))
}

fn cluster_mapping(cluster: &LocalCluster) -> String {
    let h: &WorkerBlock = &cluster.worker;
    let parts: Vec<String> = h
        .ranks
        .iter()
        .zip(h.local_devices.iter())
        .map(|(r, d)| format!("{}:{} -> r{}", h.host, d, r))
        .collect();
    parts.join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU16, Ordering};
    use std::thread;

    // Env-mutating tests serialize on this Mutex.
    static ENV_MUTEX: Mutex<()> = Mutex::new(());
    // Port allocator -- each test grabs a fresh port to avoid bind collisions
    // when the suite runs in parallel (the controller binds on its own
    // thread inside the test, sharing the address space with the ranks).
    static NEXT_PORT: AtomicU16 = AtomicU16::new(29500);

    fn next_port() -> u16 {
        NEXT_PORT.fetch_add(1, Ordering::Relaxed)
    }

    /// Build the rank-side slim envelope for one of two hosts in a
    /// 2-rank cluster. Both envelopes carry the same controller
    /// coordinates and salt; they differ only in the `worker` block.
    fn slim_envelope_for(host_name: &str, port: u16) -> LocalCluster {
        let (ranks, devices) = match host_name {
            "host-a" => (vec![0], vec![0]),
            "host-b" => (vec![1], vec![0]),
            other => panic!("unknown test host {other:?}"),
        };
        let v = json!({
            "controller": { "host": "127.0.0.1", "port": port },
            "world_size": 2,
            "num_workers": 2,
            "worker": {
                "host": host_name,
                "ranks": ranks,
                "local_devices": devices,
                "nccl_socket_ifname": "lo",
                "path": format!("/tmp/test-{host_name}"),
            }
        });
        LocalCluster::from_value(&v).expect("test slim envelope")
    }

    /// Build the controller-side `FullCluster` for the same 2-host
    /// topology. Used to drive the controller-side rendezvous server in
    /// tests that exercise both ends of the wire.
    fn full_cluster_for_test(port: u16) -> FullCluster {
        let v = json!({
            "controller": {
                "host": "127.0.0.1",
                "port": port,
                "path": "/tmp/test-controller"
            },
            "workers": [
                {
                    "host": "host-a",
                    "ranks": [0],
                    "local_devices": [0],
                    "nccl_socket_ifname": "lo",
                    "path": "/tmp/test-host-a",
                    "arch": "precompiled/cu128"
                },
                {
                    "host": "host-b",
                    "ranks": [1],
                    "local_devices": [0],
                    "nccl_socket_ifname": "lo",
                    "path": "/tmp/test-host-b",
                    "arch": "precompiled/cu128"
                }
            ]
        });
        FullCluster::from_value(&v).expect("test full envelope")
    }

    #[test]
    fn pick_designated_rank_prefers_local_worker() {
        // Controller co-located with host-b: picks host-b's first rank.
        let full = full_cluster_for_test(next_port());
        assert_eq!(pick_designated_rank(&full, "host-b"), 1);
        // Controller co-located with host-a: picks host-a's first rank.
        assert_eq!(pick_designated_rank(&full, "host-a"), 0);
        // Controller has no local worker: falls back to workers[0].ranks[0].
        assert_eq!(pick_designated_rank(&full, "192.168.122.1"), 0);
    }

    #[test]
    fn controller_rendezvous_times_out_when_a_rank_never_connects() {
        // A rank that never dials in must NOT hang the controller
        // forever. With a 2-rank cluster and zero ranks connecting, the
        // no-progress ceiling fires and returns loudly instead of parking
        // in accept() indefinitely.
        let full = full_cluster_for_test(next_port());
        let start = Instant::now();
        let result = run_controller_rendezvous_with(
            &full,
            "test-controller-host",
            Duration::from_secs(1),
        );
        let err = result.expect_err("must time out, not hang");
        assert!(
            err.to_string().contains("timed out"),
            "unexpected error: {err}"
        );
        // Fired off the idle ceiling, not after some unrelated long stall.
        assert!(
            start.elapsed() < Duration::from_secs(15),
            "took too long: {:?}",
            start.elapsed()
        );
    }

    #[test]
    fn full_rendezvous_via_controller_and_two_ranks() {
        // End-to-end through the public API: controller server in one
        // thread, two rank threads dialing in. All three converge on
        // the same NCCL unique ID.
        let port = next_port();
        let mut full = full_cluster_for_test(port);
        // Use a non-zero salt so the HMAC tag is meaningful.
        full.salt = [0x77u8; 16];
        let salt = full.salt;

        let env_a = slim_envelope_for("host-a", port);
        let env_b = slim_envelope_for("host-b", port);
        // Slim envelopes share the salt with the full cluster so HMAC
        // checks pass at both ends.
        let env_a = with_salt(env_a, salt);
        let env_b = with_salt(env_b, salt);

        let sig = [0x42u8; 32];
        let stub_uid_bytes = [0xabu8; NCCL_UNIQUE_ID_BYTES];

        let _guard = ENV_MUTEX.lock().unwrap();
        let prev_ifname = env::var(ENV_NCCL_SOCKET_IFNAME).ok();
        unsafe {
            env::set_var(ENV_NCCL_SOCKET_IFNAME, "lo");
        }

        // Controller server: orchestrator-only (no local worker).
        let ctrl_handle = thread::spawn(move || {
            run_controller_rendezvous(&full, "test-controller-host")
        });

        let rank_a_handle = thread::spawn(move || {
            crate::distributed::cluster::set_thread_hostname_override(Some("host-a"));
            crate::distributed::cluster::set_thread_local_rank_override(Some(0));
            TcpRendezvous::establish(&env_a, sig, || {
                Ok(NcclUniqueId::from_bytes(stub_uid_bytes))
            })
        });
        let rank_b_handle = thread::spawn(move || {
            crate::distributed::cluster::set_thread_hostname_override(Some("host-b"));
            crate::distributed::cluster::set_thread_local_rank_override(Some(0));
            TcpRendezvous::establish(&env_b, sig, || {
                // Designated rank is workers[0].ranks[0] = 0 (host-a) since
                // the controller is not co-located. host-b should be Wait.
                panic!("host-b rank must not be the generator (controller picked host-a)")
            })
        });

        let ctrl_res = ctrl_handle.join().expect("controller thread");
        let rdv_a = rank_a_handle.join().expect("host-a thread").expect("host-a ok");
        let rdv_b = rank_b_handle.join().expect("host-b thread").expect("host-b ok");

        if let Some(v) = prev_ifname {
            unsafe { env::set_var(ENV_NCCL_SOCKET_IFNAME, v); }
        } else {
            unsafe { env::remove_var(ENV_NCCL_SOCKET_IFNAME); }
        }

        ctrl_res.expect("controller rendezvous ok");
        assert_eq!(rdv_a.world_size(), 2);
        assert_eq!(rdv_b.world_size(), 2);
        assert_eq!(rdv_a.local_ranks(), &[0usize]);
        assert_eq!(rdv_b.local_ranks(), &[1usize]);
        assert_eq!(rdv_a.unique_id().as_bytes(), &stub_uid_bytes);
        assert_eq!(rdv_b.unique_id().as_bytes(), &stub_uid_bytes);
    }

    #[test]
    fn duplicate_hello_reconnect_replaces_stream_not_cohort_fatal() {
        // M11: a rank whose TCP connection blips mid-rendezvous reconnects and
        // re-sends Hello. Before the fix the controller aborted the ENTIRE
        // cohort on the duplicate Hello; now it replaces the stale stream
        // (Role has not been dispatched yet) and completes rendezvous on the
        // live reconnected connection. Driven at the raw-wire level so we can
        // send two Hellos for the same rank from two different sockets.
        let port = next_port();
        let mut full = full_cluster_for_test(port);
        full.salt = [0x77u8; 16];
        let salt = full.salt;
        let sig = [0x42u8; 32];
        let stub_uid = [0xabu8; NCCL_UNIQUE_ID_BYTES];

        // Controller is not co-located, so the designated (generator) rank is
        // workers[0].ranks[0] = rank 0 (host-a).
        let ctrl_handle =
            thread::spawn(move || run_controller_rendezvous(&full, "test-controller-host"));

        let connect = move || -> TcpStream {
            let addr = format!("127.0.0.1:{port}");
            for _ in 0..100 {
                if let Ok(s) = TcpStream::connect(&addr) {
                    s.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
                    s.set_write_timeout(Some(Duration::from_secs(5))).unwrap();
                    return s;
                }
                thread::sleep(Duration::from_millis(20));
            }
            panic!("could not connect to controller at {addr}");
        };
        let hello = |rank: u32, host: &str| RendezvousMsgWire::Hello {
            dataset_sig: sig,
            global_rank: rank,
            host_name: host.to_string(),
        };
        // > RENDEZVOUS_POLL_INTERVAL (200ms) so the controller has accepted and
        // read each Hello before the next connection arrives, making the
        // duplicate deterministic.
        let settle = Duration::from_millis(400);

        // Rank 0's first (soon-stale) connection.
        let mut conn0a = connect();
        write_rendezvous_frame(&mut conn0a, &salt, &hello(0, "host-a")).unwrap();
        thread::sleep(settle);

        // Rank 0 reconnects after a "blip": duplicate Hello -> controller must
        // REPLACE the stale stream, not abort.
        let mut conn0b = connect();
        write_rendezvous_frame(&mut conn0b, &salt, &hello(0, "host-a")).unwrap();
        thread::sleep(settle);

        // Rank 1 completes the cohort.
        let mut conn1 = connect();
        write_rendezvous_frame(&mut conn1, &salt, &hello(1, "host-b")).unwrap();

        // Role must land on the LIVE (reconnected) rank-0 stream, and rank 0 is
        // the generator: respond with the UID.
        match read_rendezvous_frame(&mut conn0b, &salt).unwrap() {
            RendezvousMsgWire::Role(RendezvousRole::Generate) => {}
            other => panic!("rank 0 (live stream) expected Role::Generate, got {other:?}"),
        }
        write_rendezvous_frame(
            &mut conn0b,
            &salt,
            &RendezvousMsgWire::Uid { uid_bytes: stub_uid.to_vec() },
        )
        .unwrap();

        // Rank 1 waits, then receives the broadcast UID.
        match read_rendezvous_frame(&mut conn1, &salt).unwrap() {
            RendezvousMsgWire::Role(RendezvousRole::Wait) => {}
            other => panic!("rank 1 expected Role::Wait, got {other:?}"),
        }
        match read_rendezvous_frame(&mut conn1, &salt).unwrap() {
            RendezvousMsgWire::Uid { uid_bytes } => {
                assert_eq!(uid_bytes, stub_uid.to_vec(), "rank 1 must receive the UID");
            }
            other => panic!("rank 1 expected Uid, got {other:?}"),
        }

        // The controller completed instead of aborting on the duplicate Hello.
        ctrl_handle
            .join()
            .expect("controller thread")
            .expect("rendezvous must complete despite the mid-rendezvous reconnect");

        drop(conn0a);
    }

    #[test]
    fn rendezvous_rejects_signature_mismatch() {
        // Two ranks send conflicting dataset signatures. The controller
        // collects both Hellos, detects the mismatch on the second, and
        // surfaces a loud error. Both rank threads surface their own
        // failure (the controller closes streams without sending Roles).
        let port = next_port();
        let mut full = full_cluster_for_test(port);
        full.salt = [0x55u8; 16];
        let salt = full.salt;

        let env_a = with_salt(slim_envelope_for("host-a", port), salt);
        let env_b = with_salt(slim_envelope_for("host-b", port), salt);

        let sig_a = [0x42u8; 32];
        let sig_b = [0x43u8; 32]; // diverges
        let stub_uid = [0xabu8; NCCL_UNIQUE_ID_BYTES];

        let _guard = ENV_MUTEX.lock().unwrap();
        let prev_ifname = env::var(ENV_NCCL_SOCKET_IFNAME).ok();
        unsafe {
            env::set_var(ENV_NCCL_SOCKET_IFNAME, "lo");
        }

        let ctrl_handle = thread::spawn(move || {
            run_controller_rendezvous(&full, "test-controller-host")
        });

        let rank_a_handle = thread::spawn(move || {
            crate::distributed::cluster::set_thread_hostname_override(Some("host-a"));
            crate::distributed::cluster::set_thread_local_rank_override(Some(0));
            TcpRendezvous::establish(&env_a, sig_a, || {
                Ok(NcclUniqueId::from_bytes(stub_uid))
            })
        });
        let rank_b_handle = thread::spawn(move || {
            crate::distributed::cluster::set_thread_hostname_override(Some("host-b"));
            crate::distributed::cluster::set_thread_local_rank_override(Some(0));
            TcpRendezvous::establish(&env_b, sig_b, || {
                Ok(NcclUniqueId::from_bytes(stub_uid))
            })
        });

        let ctrl_err = ctrl_handle.join().expect("controller thread");
        let rdv_a = rank_a_handle.join().expect("host-a thread");
        let rdv_b = rank_b_handle.join().expect("host-b thread");

        if let Some(v) = prev_ifname {
            unsafe { env::set_var(ENV_NCCL_SOCKET_IFNAME, v); }
        } else {
            unsafe { env::remove_var(ENV_NCCL_SOCKET_IFNAME); }
        }

        let err = ctrl_err.expect_err("controller must reject sig mismatch");
        let msg = err.to_string();
        assert!(msg.contains("dataset_signature mismatch"), "got: {msg}");
        // One of the two rank hostnames must appear in the controller's
        // error (whichever arrived second loses the comparison).
        assert!(
            msg.contains("host-a") || msg.contains("host-b"),
            "got: {msg}"
        );
        // Both ranks fail too: the controller closed their streams
        // without sending Roles, so the read of the Role frame errors.
        assert!(rdv_a.is_err() || rdv_b.is_err(), "at least one rank must fail");
    }

    /// Helper: clone a LocalCluster but overwrite its salt. Lets tests
    /// drive the controller-side server (which carries the FullCluster's
    /// salt) against rank envelopes that need to share the same key.
    fn with_salt(mut c: LocalCluster, salt: SessionSalt) -> LocalCluster {
        c.salt = salt;
        c
    }

    #[test]
    fn cluster_rendezvous_single_host_no_socket_ifname_required() {
        // num_hosts = 1: single-host envelope does not require
        // NCCL_SOCKET_IFNAME. validate_socket_ifname should let it through.
        let v = json!({
            "controller": { "host": "127.0.0.1", "port": next_port() },

            "world_size": 1,

            "num_workers": 1,

            "worker": {

                "host": "solo", "ranks": [0], "local_devices": [0],
                "nccl_socket_ifname": "lo", "path": "/tmp/test-solo"
            }
        });
        let c = LocalCluster::from_value(&v).expect("parse");
        let _guard = ENV_MUTEX.lock().unwrap();
        let prev_ifname = env::var(ENV_NCCL_SOCKET_IFNAME).ok();
        unsafe {
            env::remove_var(ENV_NCCL_SOCKET_IFNAME);
        }
        let this_host = c.worker.clone();
        assert!(
            validate_socket_ifname(&c, &this_host).is_ok(),
            "single-host must not require ifname"
        );
        if let Some(v) = prev_ifname {
            unsafe {
                env::set_var(ENV_NCCL_SOCKET_IFNAME, v);
            }
        }
    }

    #[test]
    fn multi_host_auto_exports_socket_ifname_from_cluster_config() {
        // Cluster yml declared `nccl_socket_ifname: "lo"` in slim_envelope_for;
        // validate must auto-export so the user does not also have to set
        // the env var by hand. The yml field is the single source of truth.
        let cluster = slim_envelope_for("host-a", next_port());
        let this_host = cluster.worker.clone();
        let _guard = ENV_MUTEX.lock().unwrap();
        let prev_ifname = env::var(ENV_NCCL_SOCKET_IFNAME).ok();
        unsafe {
            env::remove_var(ENV_NCCL_SOCKET_IFNAME);
        }
        let result = validate_socket_ifname(&cluster, &this_host);
        let exported = env::var(ENV_NCCL_SOCKET_IFNAME).ok();
        // Restore env before assertions so a failure does not corrupt
        // sibling tests sharing ENV_MUTEX.
        unsafe {
            env::remove_var(ENV_NCCL_SOCKET_IFNAME);
            if let Some(v) = prev_ifname {
                env::set_var(ENV_NCCL_SOCKET_IFNAME, v);
            }
        }
        assert!(result.is_ok(), "auto-export must succeed: {result:?}");
        assert_eq!(exported.as_deref(), Some("lo"));
    }

    #[test]
    fn multi_host_loud_error_when_cluster_config_ifname_empty() {
        // Programmatic LocalCluster construction may bypass fdl-cli's
        // non-empty check and end up with `nccl_socket_ifname: ""` on a
        // multi-host cluster. validate must still loud-fail in that case.
        let v = json!({
            "controller": { "host": "127.0.0.1", "port": next_port() },

            "world_size": 2,

            "num_workers": 2,

            "worker": {

                "host": "a", "ranks": [0], "local_devices": [0],
                "nccl_socket_ifname": "", "path": "/tmp/test-a"
            }
        });
        let cluster = LocalCluster::from_value(&v).expect("parse");
        let this_host = cluster.worker.clone();
        let _guard = ENV_MUTEX.lock().unwrap();
        let prev_ifname = env::var(ENV_NCCL_SOCKET_IFNAME).ok();
        unsafe {
            env::remove_var(ENV_NCCL_SOCKET_IFNAME);
        }
        let err = validate_socket_ifname(&cluster, &this_host).expect_err("empty ifname must error");
        if let Some(v) = prev_ifname {
            unsafe {
                env::set_var(ENV_NCCL_SOCKET_IFNAME, v);
            }
        }
        let msg = err.to_string();
        assert!(msg.contains("NCCL_SOCKET_IFNAME"), "got: {msg}");
        assert!(msg.contains("multiple hosts"), "got: {msg}");
    }

    /// Single-host process-per-rank (the auto-promote shape on a 2-GPU
    /// host). Both rank processes share the same host name but have
    /// different local-rank indices. With the controller-binds wire,
    /// the controller is a separate thread that binds 127.0.0.1:port;
    /// both rank threads dial in. Designated generator is workers[0]
    /// .ranks[0] = global rank 0. Guards against a host-level
    /// master-check deadlock when two ranks share a hostname.
    #[test]
    fn single_host_process_per_rank_round_trip() {
        let port = next_port();
        let salt: SessionSalt = [0x33u8; 16];

        // Controller's full view: one worker with two ranks on
        // single-host. Controller is NOT co-located with the worker
        // here (test simulates an off-host controller); generator
        // falls through to workers[0].ranks[0] = 0.
        let full = {
            let v = json!({
                "controller": {
                    "host": "127.0.0.1",
                    "port": port,
                    "path": "/tmp/test-controller"
                },
                "workers": [
                    {
                        "host": "single-host",
                        "ranks": [0, 1],
                        "local_devices": [0, 1],
                        "nccl_socket_ifname": "lo",
                        "path": "/tmp/test-single-host",
                        "arch": "precompiled/cu128"
                    }
                ]
            });
            let mut f = FullCluster::from_value(&v).expect("test full envelope");
            f.salt = salt;
            f
        };

        // Slim rank envelope: both ranks share the same one (single host
        // owning ranks [0, 1]); they differ only in the thread-local
        // override of LOCAL_RANK.
        let slim = || -> LocalCluster {
            let v = json!({
                "controller": { "host": "127.0.0.1", "port": port },
                "world_size": 2,
                "num_workers": 1,
                "worker": {
                    "host": "single-host",
                    "ranks": [0, 1],
                    "local_devices": [0, 1],
                    "nccl_socket_ifname": "lo",
                    "path": "/tmp/test-single-host",
                }
            });
            let mut c = LocalCluster::from_value(&v).expect("single-host envelope");
            c.salt = salt;
            c
        };

        let sig = [0x42u8; 32];
        let stub_uid_bytes = [0xcdu8; NCCL_UNIQUE_ID_BYTES];

        let _guard = ENV_MUTEX.lock().unwrap();

        let ctrl_handle = thread::spawn(move || {
            run_controller_rendezvous(&full, "test-controller-host")
        });

        let env_0 = slim();
        let rank_0 = thread::spawn(move || {
            crate::distributed::cluster::set_thread_hostname_override(Some("single-host"));
            crate::distributed::cluster::set_thread_local_rank_override(Some(0));
            TcpRendezvous::establish(&env_0, sig, || {
                Ok(NcclUniqueId::from_bytes(stub_uid_bytes))
            })
        });
        let env_1 = slim();
        let rank_1 = thread::spawn(move || {
            crate::distributed::cluster::set_thread_hostname_override(Some("single-host"));
            crate::distributed::cluster::set_thread_local_rank_override(Some(1));
            TcpRendezvous::establish(&env_1, sig, || {
                panic!("non-zero rank must not be the generator (controller picked rank 0)")
            })
        });

        let ctrl_res = ctrl_handle.join().expect("controller thread");
        let r0 = rank_0.join().expect("rank 0 thread").expect("rank 0 ok");
        let r1 = rank_1.join().expect("rank 1 thread").expect("rank 1 ok");

        ctrl_res.expect("controller rendezvous ok");
        assert_eq!(r0.world_size(), 2);
        assert_eq!(r1.world_size(), 2);
        // Both processes see the same host's full rank list — local_ranks
        // is the host's slice, not the process's.
        assert_eq!(r0.local_ranks(), &[0usize, 1]);
        assert_eq!(r1.local_ranks(), &[0usize, 1]);
        // Both ended up with the same UID — rank 0 generated, rank 1
        // received via the controller's broadcast.
        assert_eq!(r0.unique_id().as_bytes(), &stub_uid_bytes);
        assert_eq!(r1.unique_id().as_bytes(), &stub_uid_bytes);
    }

    #[test]
    fn cluster_mapping_format() {
        // cluster_mapping now shows only THIS host's slice -- each node
        // logs its own ranks/devices banner, not a cross-cluster summary.
        let single_rank = json!({
            "controller": { "host": "127.0.0.1", "port": 29500 },

            "world_size": 3,

            "num_workers": 2,

            "worker": {

                "host": "node-a", "ranks": [0], "local_devices": [0],
                "nccl_socket_ifname": "virbr0", "path": "/tmp/test-a"
            }
        });
        let multi_rank = json!({
            "controller": { "host": "127.0.0.1", "port": 29500 },

            "world_size": 3,

            "num_workers": 2,

            "worker": {

                "host": "node-b", "ranks": [1, 2], "local_devices": [0, 1],
                "nccl_socket_ifname": "enp1s0", "path": "/tmp/test-b"
            }
        });
        let c1 = LocalCluster::from_value(&single_rank).unwrap();
        let c2 = LocalCluster::from_value(&multi_rank).unwrap();
        assert_eq!(cluster_mapping(&c1), "node-a:0 -> r0");
        assert_eq!(cluster_mapping(&c2), "node-b:0 -> r1, node-b:1 -> r2");
    }

}
