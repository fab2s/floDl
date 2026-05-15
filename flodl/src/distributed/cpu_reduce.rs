//! Rank-side TCP client for the CPU-averaging star topology.
//!
//! Pairs with [`controller::CpuAverager`]. Each rank running with
//! [`AverageBackend::Cpu`] opens one TCP connection to the launcher's
//! `CpuAverager`, does the handshake, then drives one `all_reduce` call
//! per averaging round.
//!
//! Protocol mirror of [`controller`]:
//!
//! 1. Connect to the controller's `master_addr:cpu_avg_port`.
//! 2. Send handshake: `(magic, version, rank_id, world_size)`.
//! 3. Wait for handshake ack from controller.
//! 4. Per averaging round: send [`RoundFrame`] (this rank's tensors),
//!    receive the averaged [`RoundFrame`].
//! 5. On training end: drop the client → clean EOF → controller's reduce
//!    loop sees it and shuts down.
//!
//! The client works on `RoundFrame` directly (no flodl `Tensor` coupling
//! at this layer). Trainer-side integration (4b.B.4) is responsible for
//! converting between `Tensor` and `RoundFrame` — keeps this transport
//! file focused on TCP + protocol.
//!
//! [`controller::CpuAverager`]: crate::distributed::controller::CpuAverager
//! [`AverageBackend::Cpu`]: crate::distributed::AverageBackend::Cpu
//! [`controller`]: crate::distributed::controller
//! [`RoundFrame`]: crate::distributed::controller::RoundFrame

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::time::Duration;

use crate::distributed::controller::{
    self, HANDSHAKE_MAGIC_CONTROLLER_ACK, HANDSHAKE_MAGIC_RANK, PROTOCOL_VERSION, RoundFrame,
};
use crate::tensor::{Result, TensorError};

/// Rank-side client for the CPU-averaging controller.
///
/// One instance per rank process. Lives for the duration of training.
/// Drop closes the underlying TCP stream and signals shutdown to the
/// controller (which expects every rank to disconnect cleanly when
/// training ends).
#[derive(Debug)]
pub struct CpuReduceClient {
    stream: TcpStream,
    rank_id: u32,
    world_size: u32,
}

impl CpuReduceClient {
    /// Connect to the controller and complete the handshake.
    ///
    /// `controller_addr` is typically `master_addr:master_port + 2` (the
    /// CPU-averaging port reserved alongside `master_port + 1` for the
    /// future log side-channel). `rank_id` must be in `0..world_size`.
    ///
    /// Loud error on connect failure, handshake mismatch, or version
    /// disagreement. Connect retries are intentionally not built in;
    /// rendezvous-level retry policy belongs upstream (the launcher
    /// ensures the controller is bound before spawning rank children).
    pub fn connect(
        controller_addr: SocketAddr,
        rank_id: u32,
        world_size: u32,
    ) -> Result<Self> {
        if world_size == 0 {
            return Err(TensorError::new(
                "cpu_reduce: world_size must be > 0",
            ));
        }
        if rank_id >= world_size {
            return Err(TensorError::new(&format!(
                "cpu_reduce: rank_id {rank_id} must be < world_size {world_size}"
            )));
        }
        let stream = TcpStream::connect(controller_addr).map_err(|e| {
            TensorError::new(&format!(
                "cpu_reduce: connect to {controller_addr} failed: {e}"
            ))
        })?;
        // Read timeout protects the handshake from a wedged controller;
        // gets cleared after the ack so the long-running reduce loop
        // doesn't trip on a slow round.
        stream
            .set_read_timeout(Some(Duration::from_secs(10)))
            .map_err(|e| TensorError::new(&format!("cpu_reduce: set_read_timeout: {e}")))?;

        let mut client = CpuReduceClient {
            stream,
            rank_id,
            world_size,
        };
        client.send_handshake()?;
        client.read_handshake_ack()?;
        client
            .stream
            .set_read_timeout(None)
            .map_err(|e| TensorError::new(&format!("cpu_reduce: clear read_timeout: {e}")))?;
        Ok(client)
    }

    fn send_handshake(&mut self) -> Result<()> {
        let mut buf = [0u8; 16];
        buf[0..4].copy_from_slice(&HANDSHAKE_MAGIC_RANK.to_le_bytes());
        buf[4..8].copy_from_slice(&PROTOCOL_VERSION.to_le_bytes());
        buf[8..12].copy_from_slice(&self.rank_id.to_le_bytes());
        buf[12..16].copy_from_slice(&self.world_size.to_le_bytes());
        self.stream.write_all(&buf).map_err(|e| {
            TensorError::new(&format!("cpu_reduce: handshake write failed: {e}"))
        })?;
        self.stream.flush().map_err(|e| {
            TensorError::new(&format!("cpu_reduce: handshake flush failed: {e}"))
        })?;
        Ok(())
    }

    fn read_handshake_ack(&mut self) -> Result<()> {
        let mut buf = [0u8; 8];
        self.stream.read_exact(&mut buf).map_err(|e| {
            TensorError::new(&format!(
                "cpu_reduce: handshake ack read failed: {e} \
                 (controller may have rejected our handshake)"
            ))
        })?;
        let magic = u32::from_le_bytes(buf[0..4].try_into().unwrap());
        if magic != HANDSHAKE_MAGIC_CONTROLLER_ACK {
            return Err(TensorError::new(&format!(
                "cpu_reduce: handshake ack magic 0x{magic:08x} != \
                 0x{HANDSHAKE_MAGIC_CONTROLLER_ACK:08x}"
            )));
        }
        let proto_ver = u32::from_le_bytes(buf[4..8].try_into().unwrap());
        if proto_ver != PROTOCOL_VERSION {
            return Err(TensorError::new(&format!(
                "cpu_reduce: controller protocol_version {proto_ver} != \
                 our version {PROTOCOL_VERSION}"
            )));
        }
        Ok(())
    }

    /// This rank's id, as told to the controller.
    pub fn rank_id(&self) -> u32 {
        self.rank_id
    }

    /// Cluster world_size, as told to the controller.
    pub fn world_size(&self) -> u32 {
        self.world_size
    }

    /// Send this rank's frame for the current round and receive the
    /// averaged frame back.
    ///
    /// Blocks until the controller has collected frames from every
    /// rank, summed them, and scattered the average back. Loud error on
    /// any wire-level failure (truncated read, EOF before the averaged
    /// frame, magic mismatch).
    ///
    /// The returned frame has the same tensor count, dtypes, and shapes
    /// as the input frame; only the tensor bytes change.
    pub fn all_reduce(&mut self, frame: &RoundFrame) -> Result<RoundFrame> {
        controller::write_round_frame(&mut self.stream, frame)?;
        match controller::read_round_frame(&mut self.stream)? {
            Some(f) => Ok(f),
            None => Err(TensorError::new(
                "cpu_reduce: controller closed connection before sending averaged \
                 frame back (controller crashed, or another rank disconnected and \
                 triggered cluster-wide shutdown mid-round)",
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::controller::{
        CpuAverager, DTYPE_F32, RoundFrame, TensorPayload,
    };
    use std::net::Ipv4Addr;
    use std::sync::mpsc;
    use std::thread;

    fn f32_to_bytes(data: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(data.len() * 4);
        for x in data {
            out.extend_from_slice(&x.to_le_bytes());
        }
        out
    }

    fn bytes_as_f32(bytes: &[u8]) -> Vec<f32> {
        let mut out = Vec::with_capacity(bytes.len() / 4);
        for i in 0..bytes.len() / 4 {
            let mut b = [0u8; 4];
            b.copy_from_slice(&bytes[i * 4..(i + 1) * 4]);
            out.push(f32::from_le_bytes(b));
        }
        out
    }

    fn frame_with(data: &[f32]) -> RoundFrame {
        RoundFrame {
            tensors: vec![TensorPayload {
                dtype: DTYPE_F32,
                shape: vec![data.len() as u32],
                bytes: f32_to_bytes(data),
            }],
        }
    }

    /// End-to-end: spawn the controller and two rank clients via this
    /// crate's `CpuReduceClient`; verify the average comes back to each.
    #[test]
    fn two_rank_client_average() {
        let avg = CpuAverager::start(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0),
            2,
        )
        .unwrap();
        let port = avg.port();
        let addr = SocketAddr::new(Ipv4Addr::LOCALHOST.into(), port);

        let (tx0, rx0) = mpsc::channel();
        let (tx1, rx1) = mpsc::channel();
        let t0 = thread::spawn(move || {
            let mut c = CpuReduceClient::connect(addr, 0, 2).unwrap();
            let avg_frame = c.all_reduce(&frame_with(&[2.0, 4.0, 6.0])).unwrap();
            tx0.send(avg_frame).unwrap();
            drop(c);
        });
        let t1 = thread::spawn(move || {
            let mut c = CpuReduceClient::connect(addr, 1, 2).unwrap();
            let avg_frame = c.all_reduce(&frame_with(&[4.0, 8.0, 12.0])).unwrap();
            tx1.send(avg_frame).unwrap();
            drop(c);
        });

        let r0 = rx0.recv().unwrap();
        let r1 = rx1.recv().unwrap();
        t0.join().unwrap();
        t1.join().unwrap();
        avg.shutdown().unwrap();

        let avg0 = bytes_as_f32(&r0.tensors[0].bytes);
        assert_eq!(avg0, vec![3.0, 6.0, 9.0]);
        assert_eq!(r0, r1);
    }

    /// Each client survives multiple rounds, gets the per-round average
    /// back. Exercises the persistent-connection path.
    #[test]
    fn client_multi_round_persistence() {
        let avg = CpuAverager::start(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0),
            2,
        )
        .unwrap();
        let port = avg.port();
        let addr = SocketAddr::new(Ipv4Addr::LOCALHOST.into(), port);

        let t0 = thread::spawn(move || -> Vec<RoundFrame> {
            let mut c = CpuReduceClient::connect(addr, 0, 2).unwrap();
            let r1 = c.all_reduce(&frame_with(&[1.0])).unwrap();
            let r2 = c.all_reduce(&frame_with(&[5.0])).unwrap();
            let r3 = c.all_reduce(&frame_with(&[9.0])).unwrap();
            vec![r1, r2, r3]
        });
        let t1 = thread::spawn(move || -> Vec<RoundFrame> {
            let mut c = CpuReduceClient::connect(addr, 1, 2).unwrap();
            let r1 = c.all_reduce(&frame_with(&[3.0])).unwrap();
            let r2 = c.all_reduce(&frame_with(&[7.0])).unwrap();
            let r3 = c.all_reduce(&frame_with(&[11.0])).unwrap();
            vec![r1, r2, r3]
        });

        let r0_results = t0.join().unwrap();
        let r1_results = t1.join().unwrap();
        avg.shutdown().unwrap();

        // Round-by-round averages: (1,3)/2=2, (5,7)/2=6, (9,11)/2=10.
        let expected = [2.0_f32, 6.0, 10.0];
        for (i, want) in expected.iter().enumerate() {
            let got_0 = bytes_as_f32(&r0_results[i].tensors[0].bytes);
            let got_1 = bytes_as_f32(&r1_results[i].tensors[0].bytes);
            assert_eq!(got_0, vec![*want], "rank 0 round {i}");
            assert_eq!(got_1, vec![*want], "rank 1 round {i}");
        }
    }

    /// Rank disagreeing on world_size with the controller must surface
    /// loudly (controller drops the bad rank; our handshake_ack read
    /// then fails).
    #[test]
    fn rejects_world_size_disagreement() {
        let avg = CpuAverager::start(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0),
            2,
        )
        .unwrap();
        let port = avg.port();
        let addr = SocketAddr::new(Ipv4Addr::LOCALHOST.into(), port);

        // Rank claims world_size = 3 but controller is configured for 2.
        let err = CpuReduceClient::connect(addr, 0, 3).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("ack") || msg.contains("handshake") || msg.contains("read"),
            "expected wire-level error, got: {msg}"
        );
        let _ = avg.shutdown();
    }

    /// Constructing a client with rank_id >= world_size is a local-only
    /// error (caught before any TCP traffic happens).
    #[test]
    fn rejects_rank_id_out_of_bounds_locally() {
        let err = CpuReduceClient::connect(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 12345),
            5,
            3,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("rank_id 5 must be < world_size 3"),
            "got: {err}"
        );
    }

    #[test]
    fn rejects_zero_world_size_locally() {
        let err = CpuReduceClient::connect(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 12345),
            0,
            0,
        )
        .unwrap_err();
        assert!(err.to_string().contains("world_size"), "got: {err}");
    }

    /// Connect failure (no controller listening) surfaces a clear error.
    #[test]
    fn surfaces_connect_failure_clearly() {
        // 1 is a reserved port — connect should fail with refused.
        let err = CpuReduceClient::connect(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 1),
            0,
            1,
        )
        .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("connect") && msg.contains(":1 failed"),
            "expected connect-failed error mentioning port 1, got: {msg}"
        );
    }
}
