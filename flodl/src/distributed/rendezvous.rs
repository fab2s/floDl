//! TCP rendezvous for multi-host DDP startup.
//!
//! One process per host. The host owning rank 0 is the **master**: it
//! generates the [`NcclUniqueId`] and listens on
//! [`LocalCluster::master_port`](super::LocalCluster::master_port). Every
//! other host is a **worker** that connects to the master, swaps a dataset
//! signature for verification, and receives the unique ID.
//!
//! Wire protocol (one TCP connection per worker host):
//!
//! ```text
//! worker -> master :  [32 B dataset_signature][1 B name_len][N B host_name]
//! master -> worker :  [128 B NcclUniqueId bytes]
//! ```
//!
//! Sizes are fixed, so no framing is required -- `read_exact` is enough.
//!
//! Workers retry the connection for ~30 s to absorb cold-boot ordering jitter
//! between hosts. Hard error after the budget exhausts -- silent infinite
//! retry would hide misconfiguration.

use std::env;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::thread;
use std::time::Duration;

use crate::{Device, Result, TensorError};

use super::{HostBlock, LocalCluster, NCCL_UNIQUE_ID_BYTES, NcclUniqueId};

const HOSTNAME_MAX_LEN: usize = 255;
const CONNECT_RETRIES: usize = 60;
const CONNECT_RETRY_INTERVAL: Duration = Duration::from_millis(500);
const IO_TIMEOUT: Duration = Duration::from_secs(30);
const ENV_NCCL_SOCKET_IFNAME: &str = "NCCL_SOCKET_IFNAME";

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

    /// Drive the TCP handshake. Invoked from
    /// [`LocalCluster::rendezvous`](super::LocalCluster::rendezvous).
    ///
    /// The `gen_uid` closure is called at most once -- on the master host only,
    /// after `this_host` resolution and socket-interface validation succeed.
    /// Workers drop the closure unused. Production passes
    /// [`NcclUniqueId::new`]; tests pass a closure that returns a fixed-byte
    /// stub to avoid linking against CUDA-bound NCCL.
    pub(crate) fn establish<F>(
        cluster: &LocalCluster,
        dataset_signature: [u8; 32],
        gen_uid: F,
    ) -> Result<Self>
    where
        F: FnOnce() -> Result<NcclUniqueId>,
    {
        let this_host = cluster.this_host()?;
        validate_socket_ifname(cluster)?;

        let local_ranks = this_host.ranks.clone();
        let local_devices: Vec<Device> = this_host
            .local_devices
            .iter()
            .map(|&d| Device::CUDA(d))
            .collect();
        let is_master = local_ranks.contains(&0);

        let uid_bytes = if is_master {
            let uid = gen_uid()?;
            let bytes = *uid.as_bytes();
            run_master(cluster, &bytes, &dataset_signature)?;
            bytes
        } else {
            run_worker(cluster, &this_host.name, &dataset_signature)?
        };

        crate::msg!("cluster: {}", cluster_mapping(cluster));

        Ok(TcpRendezvous {
            world_size: cluster.world_size(),
            local_ranks,
            local_devices,
            unique_id: NcclUniqueId::from_bytes(uid_bytes),
        })
    }
}

/// Master-side wire protocol: bind, accept `n_workers` connections, verify
/// dataset signatures, send the unique ID. Pure byte-level work -- callable
/// from tests that don't have CUDA-backed NCCL.
fn run_master(
    cluster: &LocalCluster,
    uid_bytes: &[u8; NCCL_UNIQUE_ID_BYTES],
    expected_sig: &[u8; 32],
) -> Result<()> {
    let n_workers = cluster.num_hosts.saturating_sub(1);
    if n_workers == 0 {
        return Ok(());
    }

    let bind_addr = format!("{}:{}", cluster.master_addr, cluster.master_port);
    let listener = TcpListener::bind(&bind_addr).map_err(|e| {
        TensorError::new(&format!(
            "rendezvous: failed to bind {bind_addr}: {e}"
        ))
    })?;

    for _ in 0..n_workers {
        let (mut stream, peer) = listener.accept().map_err(|e| {
            TensorError::new(&format!("rendezvous: accept failed: {e}"))
        })?;
        stream
            .set_read_timeout(Some(IO_TIMEOUT))
            .and_then(|()| stream.set_write_timeout(Some(IO_TIMEOUT)))
            .map_err(|e| {
                TensorError::new(&format!("rendezvous: setting timeouts failed: {e}"))
            })?;

        let (worker_sig, worker_name) = recv_signature_and_name(&mut stream, peer)?;
        if &worker_sig != expected_sig {
            return Err(TensorError::new(&format!(
                "rendezvous: dataset_signature mismatch from host {worker_name:?} \
                 (peer {peer}). Each rank must read from the same dataset; \
                 silent fan-out across diverging shards is the worst class of bug."
            )));
        }

        stream.write_all(uid_bytes).map_err(|e| {
            TensorError::new(&format!(
                "rendezvous: failed to send unique ID to {worker_name:?}: {e}"
            ))
        })?;
    }

    Ok(())
}

/// Worker-side wire protocol: connect with retry, send signature + host name,
/// receive the unique ID. Returns the raw bytes; caller wraps them in an
/// [`NcclUniqueId`].
fn run_worker(
    cluster: &LocalCluster,
    my_name: &str,
    sig: &[u8; 32],
) -> Result<[u8; NCCL_UNIQUE_ID_BYTES]> {
    let name_bytes = my_name.as_bytes();
    if name_bytes.len() > HOSTNAME_MAX_LEN {
        return Err(TensorError::new(&format!(
            "rendezvous: host name {:?} exceeds {} bytes",
            my_name, HOSTNAME_MAX_LEN
        )));
    }

    let addr = format!("{}:{}", cluster.master_addr, cluster.master_port);
    let mut stream = connect_with_retry(&addr)?;
    stream
        .set_read_timeout(Some(IO_TIMEOUT))
        .and_then(|()| stream.set_write_timeout(Some(IO_TIMEOUT)))
        .map_err(|e| {
            TensorError::new(&format!("rendezvous: setting timeouts failed: {e}"))
        })?;

    stream.write_all(sig).map_err(|e| io_err("send signature", e))?;
    stream
        .write_all(&[name_bytes.len() as u8])
        .map_err(|e| io_err("send name length", e))?;
    stream.write_all(name_bytes).map_err(|e| io_err("send name", e))?;

    let mut uid = [0u8; NCCL_UNIQUE_ID_BYTES];
    stream.read_exact(&mut uid).map_err(|e| io_err("recv unique ID", e))?;
    Ok(uid)
}

fn recv_signature_and_name(
    stream: &mut TcpStream,
    peer: std::net::SocketAddr,
) -> Result<([u8; 32], String)> {
    let mut sig = [0u8; 32];
    stream.read_exact(&mut sig).map_err(|e| {
        TensorError::new(&format!(
            "rendezvous: recv signature from {peer}: {e}"
        ))
    })?;
    let mut len_buf = [0u8; 1];
    stream.read_exact(&mut len_buf).map_err(|e| {
        TensorError::new(&format!("rendezvous: recv name length from {peer}: {e}"))
    })?;
    let mut name = vec![0u8; len_buf[0] as usize];
    stream.read_exact(&mut name).map_err(|e| {
        TensorError::new(&format!("rendezvous: recv name from {peer}: {e}"))
    })?;
    let name = String::from_utf8(name).map_err(|e| {
        TensorError::new(&format!(
            "rendezvous: host name from {peer} not UTF-8: {e}"
        ))
    })?;
    Ok((sig, name))
}

fn connect_with_retry(addr: &str) -> Result<TcpStream> {
    let mut last_err: Option<std::io::Error> = None;
    for _ in 0..CONNECT_RETRIES {
        match TcpStream::connect(addr) {
            Ok(s) => return Ok(s),
            Err(e) => {
                last_err = Some(e);
                thread::sleep(CONNECT_RETRY_INTERVAL);
            }
        }
    }
    Err(TensorError::new(&format!(
        "rendezvous: failed to connect to master at {addr} after {} retries (~{}s): {}",
        CONNECT_RETRIES,
        CONNECT_RETRIES * CONNECT_RETRY_INTERVAL.as_secs() as usize
            + CONNECT_RETRIES * CONNECT_RETRY_INTERVAL.subsec_millis() as usize / 1000,
        last_err
            .map(|e| e.to_string())
            .unwrap_or_else(|| "no error captured".into())
    )))
}

fn validate_socket_ifname(cluster: &LocalCluster) -> Result<()> {
    if cluster.spans_multiple_hosts() && env::var(ENV_NCCL_SOCKET_IFNAME).is_err() {
        return Err(TensorError::new(&format!(
            "rendezvous: {ENV_NCCL_SOCKET_IFNAME} must be set when the cluster spans \
             multiple hosts (auto-detection rejected -- interface naming is \
             config-specific and silent fallthrough costs hours)"
        )));
    }
    Ok(())
}

fn cluster_mapping(cluster: &LocalCluster) -> String {
    let h: &HostBlock = &cluster.host;
    let parts: Vec<String> = h
        .ranks
        .iter()
        .zip(h.local_devices.iter())
        .map(|(r, d)| format!("{}:{} -> r{}", h.name, d, r))
        .collect();
    parts.join(", ")
}

fn io_err(stage: &str, e: std::io::Error) -> TensorError {
    TensorError::new(&format!("rendezvous: {stage}: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU16, Ordering};

    // Env-mutating tests serialize on this Mutex.
    static ENV_MUTEX: Mutex<()> = Mutex::new(());
    // Port allocator -- each test grabs a fresh port to avoid bind collisions
    // when the suite runs in parallel (test-thread is the worker; another
    // thread inside the test is the master).
    static NEXT_PORT: AtomicU16 = AtomicU16::new(29500);

    fn next_port() -> u16 {
        NEXT_PORT.fetch_add(1, Ordering::Relaxed)
    }

    /// Build a 2-host envelope from this host's perspective. The two
    /// envelopes share `master_addr`, `master_port`, `world_size`, and
    /// `num_hosts`; they differ only in the `host` field.
    fn envelope_for(host_name: &str, port: u16) -> LocalCluster {
        let (ranks, devices) = match host_name {
            "master-host" => (vec![0], vec![0]),
            "worker-host" => (vec![1], vec![0]),
            other => panic!("unknown test host {other:?}"),
        };
        let v = json!({
            "master_addr": "127.0.0.1",
            "master_port": port,
            "world_size": 2,
            "num_hosts": 2,
            "host": {
                "name": host_name,
                "ranks": ranks,
                "local_devices": devices,
                "nccl_socket_ifname": "lo",
                "path": format!("/tmp/test-{host_name}"),
            }
        });
        LocalCluster::from_value(&v).expect("test envelope")
    }

    #[test]
    fn wire_protocol_roundtrip() {
        let port = next_port();
        let master_env = envelope_for("master-host", port);
        let worker_env = envelope_for("worker-host", port);
        let sig = [0x42u8; 32];
        let expected_uid = [0xabu8; NCCL_UNIQUE_ID_BYTES];

        let master_sig = sig;
        let master_handle = thread::spawn(move || {
            run_master(&master_env, &expected_uid, &master_sig)
        });

        let got_uid = run_worker(&worker_env, "worker-host", &sig).expect("worker");
        master_handle.join().expect("master thread").expect("master ok");

        assert_eq!(got_uid, expected_uid, "uid bytes must round-trip");
    }

    #[test]
    fn wire_protocol_rejects_signature_mismatch() {
        let port = next_port();
        let master_env = envelope_for("master-host", port);
        let worker_env = envelope_for("worker-host", port);
        let master_sig = [0x42u8; 32];
        let worker_sig = [0x43u8; 32];
        let uid = [0xabu8; NCCL_UNIQUE_ID_BYTES];

        let master_handle =
            thread::spawn(move || run_master(&master_env, &uid, &master_sig));

        // Worker happily sends its (wrong) sig and tries to read the UID;
        // master will reject before sending, so worker's read fails.
        let worker_result = run_worker(&worker_env, "worker-host", &worker_sig);
        let master_result = master_handle.join().expect("master thread");

        assert!(master_result.is_err(), "master must reject sig mismatch");
        let msg = master_result.unwrap_err().to_string();
        assert!(msg.contains("dataset_signature mismatch"), "got: {msg}");
        assert!(msg.contains("worker-host"), "got: {msg}");

        // Worker's read errors out -- master closes the stream without sending.
        assert!(worker_result.is_err(), "worker must surface failure too");
    }

    #[test]
    fn cluster_rendezvous_single_host_no_master_socket_ifname_required() {
        // num_hosts = 1: single-host envelope does not require
        // NCCL_SOCKET_IFNAME. validate_socket_ifname should let it through.
        let v = json!({
            "master_addr": "127.0.0.1",
            "master_port": next_port(),
            "world_size": 1,
            "num_hosts": 1,
            "host": {
                "name": "solo", "ranks": [0], "local_devices": [0],
                "nccl_socket_ifname": "lo", "path": "/tmp/test-solo"
            }
        });
        let c = LocalCluster::from_value(&v).expect("parse");
        let _guard = ENV_MUTEX.lock().unwrap();
        let prev_ifname = env::var(ENV_NCCL_SOCKET_IFNAME).ok();
        unsafe {
            env::remove_var(ENV_NCCL_SOCKET_IFNAME);
        }
        assert!(validate_socket_ifname(&c).is_ok(), "single-host must not require ifname");
        if let Some(v) = prev_ifname {
            unsafe {
                env::set_var(ENV_NCCL_SOCKET_IFNAME, v);
            }
        }
    }

    #[test]
    fn multi_host_loud_error_when_socket_ifname_unset() {
        let cluster = envelope_for("master-host", next_port());
        let _guard = ENV_MUTEX.lock().unwrap();
        let prev_ifname = env::var(ENV_NCCL_SOCKET_IFNAME).ok();
        unsafe {
            env::remove_var(ENV_NCCL_SOCKET_IFNAME);
        }
        let err = validate_socket_ifname(&cluster).expect_err("must require ifname");
        if let Some(v) = prev_ifname {
            unsafe {
                env::set_var(ENV_NCCL_SOCKET_IFNAME, v);
            }
        }
        let msg = err.to_string();
        assert!(msg.contains("NCCL_SOCKET_IFNAME"), "got: {msg}");
        assert!(msg.contains("multiple hosts"), "got: {msg}");
    }

    #[test]
    fn full_rendezvous_through_cluster_api() {
        // End-to-end: two threads claim different hostnames via the
        // thread-local override, each holding its own per-node envelope,
        // both go through TcpRendezvous::establish (the same entry point
        // production uses), exchange a fake UID over a real TCP socket on
        // 127.0.0.1, and end up agreeing on every observable: world_size,
        // their own local ranks, local devices, and the unique-ID bytes.
        let port = next_port();
        let master_env = envelope_for("master-host", port);
        let worker_env = envelope_for("worker-host", port);
        let sig = [0x42u8; 32];
        let stub_uid_bytes = [0xabu8; NCCL_UNIQUE_ID_BYTES];

        let _guard = ENV_MUTEX.lock().unwrap();
        let prev_ifname = env::var(ENV_NCCL_SOCKET_IFNAME).ok();
        unsafe {
            env::set_var(ENV_NCCL_SOCKET_IFNAME, "lo");
        }

        let master_handle = thread::spawn(move || {
            crate::distributed::cluster::set_thread_hostname_override(Some("master-host"));
            TcpRendezvous::establish(&master_env, sig, || {
                Ok(NcclUniqueId::from_bytes(stub_uid_bytes))
            })
        });
        let worker_handle = thread::spawn(move || {
            crate::distributed::cluster::set_thread_hostname_override(Some("worker-host"));
            TcpRendezvous::establish(&worker_env, sig, || {
                panic!("worker must never invoke the uid-generator closure")
            })
        });

        let master_rdv = master_handle.join().expect("master thread").expect("master ok");
        let worker_rdv = worker_handle.join().expect("worker thread").expect("worker ok");

        if let Some(v) = prev_ifname {
            unsafe { env::set_var(ENV_NCCL_SOCKET_IFNAME, v); }
        } else {
            unsafe { env::remove_var(ENV_NCCL_SOCKET_IFNAME); }
        }

        assert_eq!(master_rdv.world_size(), 2);
        assert_eq!(worker_rdv.world_size(), 2);
        assert_eq!(master_rdv.local_ranks(), &[0usize]);
        assert_eq!(worker_rdv.local_ranks(), &[1usize]);
        assert_eq!(master_rdv.local_devices(), &[Device::CUDA(0)]);
        assert_eq!(worker_rdv.local_devices(), &[Device::CUDA(0)]);
        assert_eq!(master_rdv.unique_id().as_bytes(), &stub_uid_bytes);
        assert_eq!(worker_rdv.unique_id().as_bytes(), &stub_uid_bytes);
    }

    #[test]
    fn cluster_mapping_format() {
        // cluster_mapping now shows only THIS host's slice -- each node
        // logs its own ranks/devices banner, not a cross-cluster summary.
        let single_rank = json!({
            "master_addr": "127.0.0.1",
            "master_port": 29500,
            "world_size": 3,
            "num_hosts": 2,
            "host": {
                "name": "node-a", "ranks": [0], "local_devices": [0],
                "nccl_socket_ifname": "virbr0", "path": "/tmp/test-a"
            }
        });
        let multi_rank = json!({
            "master_addr": "127.0.0.1",
            "master_port": 29500,
            "world_size": 3,
            "num_hosts": 2,
            "host": {
                "name": "node-b", "ranks": [1, 2], "local_devices": [0, 1],
                "nccl_socket_ifname": "enp1s0", "path": "/tmp/test-b"
            }
        });
        let c1 = LocalCluster::from_value(&single_rank).unwrap();
        let c2 = LocalCluster::from_value(&multi_rank).unwrap();
        assert_eq!(cluster_mapping(&c1), "node-a:0 -> r0");
        assert_eq!(cluster_mapping(&c2), "node-b:0 -> r1, node-b:1 -> r2");
    }

}
