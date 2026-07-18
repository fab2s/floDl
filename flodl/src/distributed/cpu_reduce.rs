//! Rank-side TCP client for the CPU-averaging star topology.
//!
//! Pairs with [`controller::ClusterController`]. Each rank running with
//! [`AverageBackend::Cpu`] opens one TCP connection to the launcher's
//! `ClusterController`, does the handshake, then drives one `all_reduce` call
//! per averaging round.
//!
//! Protocol mirror of [`controller`]:
//!
//! 1. Connect to the controller's `controller_addr:cpu_avg_port`.
//! 2. Send handshake: `(magic, version, rank_id, world_size)`.
//! 3. Wait for handshake ack from controller.
//! 4. Per averaging round: send [`RoundFrame`] (this rank's tensors),
//!    receive the averaged [`RoundFrame`].
//! 5. On training end: drop the client → clean EOF → controller's reduce
//!    loop sees it and shuts down.
//!
//! The client works on `RoundFrame` directly (no flodl `Tensor` coupling
//! at this layer). Trainer-side integration converts between `Tensor`
//! and `RoundFrame`, keeping this transport file focused on TCP +
//! protocol.
//!
//! [`controller::ClusterController`]: crate::distributed::controller::ClusterController
//! [`AverageBackend::Cpu`]: crate::distributed::AverageBackend::Cpu
//! [`controller`]: crate::distributed::controller
//! [`RoundFrame`]: crate::distributed::controller::RoundFrame

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::time::{Duration, Instant};

use crate::distributed::controller::{
    self, DTYPE_F32, HANDSHAKE_MAGIC_CONTROLLER_ACK, HANDSHAKE_MAGIC_RANK, PROTOCOL_VERSION,
    RoundFrame, RoundKind, TensorPayload,
};
use crate::distributed::wire::SessionSalt;
use crate::tensor::{DType, Device, Result, Tensor, TensorError};

/// Per-read deadline for the long-running reduce loop (replaces the
/// previously-cleared timeout). A vanished controller or relay must not
/// park the rank forever: the coordinator-side ReduceStall ceiling cannot
/// fire if the coordinator is the process that died, so the rank needs an
/// independent backstop. `SO_RCVTIMEO` counts silence per `read()` (waiting
/// for the next byte), NOT total round time, so a slow-but-live round
/// (bytes still trickling, a slow straggler holding the barrier, a long
/// eval/checkpoint callback) never trips it; only true peer-death silence
/// does. Sized to the coordinator's production stall ceiling (120s) so rank
/// and coordinator agree on what "stalled" means. This is the in-band
/// analogue of the relay's `fill_committed` starvation deadline. Scaled by
/// `FLODL_NET_TIMEOUT_SCALE` at socket setup like the rest of the
/// deadline set (see `wire::ENV_NET_TIMEOUT_SCALE`).
const REDUCE_READ_DEADLINE_SECS: u64 = 120;

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
    /// Session salt used as the HMAC key on every [`RoundFrame`] body.
    /// Mismatched salts surface as "HMAC verification failed" on the
    /// first round-trip.
    salt: SessionSalt,
    /// Instrumentation accumulators: per-phase reduce time (serialize /
    /// wire / deserialize, in ns) + total wire byte volume, summed
    /// across every `all_reduce_tensors` call and emitted once at
    /// teardown via [`Self::log_profile_summary`]. Overhead is three
    /// `Instant` deltas per reduce — negligible. Drives the
    /// "wire-bytes-bound vs CPU-bound" decision for the hierarchical /
    /// bf16 / pinned-copy levers.
    prof_serialize_ns: u128,
    prof_wire_ns: u128,
    prof_deserialize_ns: u128,
    prof_bytes: u64,
    prof_count: u64,
    /// Instrumentation gate: cached `-vvv` (`Verbosity::Debug`) at
    /// construction. When false the per-phase timing and the teardown
    /// summary are skipped entirely.
    prof_enabled: bool,
}


impl CpuReduceClient {
    /// Connect to the controller and complete the handshake.
    ///
    /// `controller_addr` is the host-local relay's data loopback
    /// (`controller_port + 4` on `127.0.0.1`); the relay folds and
    /// forwards to the controller's single mux port. `rank_id` must be
    /// in `0..world_size`.
    ///
    /// `salt` is the 128-bit session salt the launcher generated and
    /// shipped via [`LocalCluster::salt`]; the controller side must use
    /// the same value, otherwise the first [`RoundFrame`] HMAC check
    /// fails loudly.
    ///
    /// Loud error on connect failure, handshake mismatch, or version
    /// disagreement. Connect retries are intentionally not built in;
    /// rendezvous-level retry policy belongs upstream (the launcher
    /// ensures the controller is bound before spawning rank children).
    ///
    /// [`LocalCluster::salt`]: crate::distributed::cluster::LocalCluster::salt
    pub fn connect(
        controller_addr: SocketAddr,
        rank_id: u32,
        world_size: u32,
        salt: SessionSalt,
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
        // Ranks dial their host-local relay's loopback. The relay process
        // may bind a beat after the rank starts (launcher spawns both),
        // so retry briefly rather than fail on the first refusal.
        let stream = crate::distributed::wire::connect_with_retry(
            controller_addr,
            "cpu_reduce",
        )?;
        // Disable Nagle: the reduce is a small-frame write→blocking-read
        // ping-pong, which deadlocks Nagle against delayed-ACK for ~40ms
        // per round-trip. With the cross-host reduce being 97-99% of the
        // cpu-cadence wall (measured), this is the dominant lever.
        // Best-effort — a platform without TCP_NODELAY shouldn't abort
        // training, it just keeps the latency.
        let _ = stream.set_nodelay(true);
        // Read timeout protects the handshake from a wedged controller;
        // gets cleared after the ack so the long-running reduce loop
        // doesn't trip on a slow round.
        stream
            .set_read_timeout(Some(Duration::from_secs(10)))
            .map_err(|e| TensorError::new(&format!("cpu_reduce: set_read_timeout: {e}")))?;
        stream
            .set_write_timeout(Some(crate::distributed::wire::write_stall_timeout()))
            .map_err(|e| TensorError::new(&format!("cpu_reduce: set_write_timeout: {e}")))?;

        let mut client = CpuReduceClient {
            stream,
            rank_id,
            world_size,
            salt,
            prof_serialize_ns: 0,
            prof_wire_ns: 0,
            prof_deserialize_ns: 0,
            prof_bytes: 0,
            prof_count: 0,
            prof_enabled: crate::log::enabled(crate::log::Verbosity::Debug),
        };
        client.send_handshake()?;
        client.read_handshake_ack()?;
        // Swap the tight handshake timeout for a generous per-read deadline
        // (see `REDUCE_READ_DEADLINE_SECS`): keeps the reduce loop from
        // wedging forever on a vanished controller/relay without tripping on
        // a slow-but-live round.
        client
            .stream
            .set_read_timeout(Some(Duration::from_secs(
                crate::distributed::wire::scaled_deadline_secs(REDUCE_READ_DEADLINE_SECS),
            )))
            .map_err(|e| {
                TensorError::new(&format!("cpu_reduce: set reduce read deadline: {e}"))
            })?;
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
        write_framed_round(&mut self.stream, frame, &self.salt)?;
        match read_framed_round(&mut self.stream, &self.salt)? {
            Some(f) => Ok(f),
            None => Err(TensorError::new(
                "cpu_reduce: controller closed connection before sending averaged \
                 frame back (controller crashed, or another rank disconnected and \
                 triggered cluster-wide shutdown mid-round)",
            )),
        }
    }

    /// Weighted frame-level reduce: build a [`RoundFrame`], tag it with
    /// `kind` + the sender's realized-work `weight`, round-trip it, and
    /// return the reduced tensors plus the round's summed accepted mass.
    ///
    /// [`RoundKind::Model`]: the controller returns the consensus (sum
    /// divided ONCE by the mass of exactly the frames it accepted). A
    /// returned mass of `0.0` means nothing was realized this round and
    /// the tensors are meaningless zeros — callers must keep local state.
    ///
    /// [`RoundKind::Control`]: pure element-wise sum (gathers /
    /// broadcasts build on it); the returned mass is informational.
    ///
    /// v1 supports f32 only; loud error on other dtypes. Caller is
    /// responsible for moving reduced tensors back to GPU if needed.
    pub fn all_reduce_weighted(
        &mut self,
        tensors: &[&Tensor],
        kind: RoundKind,
        weight: f64,
    ) -> Result<(Vec<Tensor>, f64)> {
        // Instrumentation (gated on `-vvv`): time the three phases
        // independently so we can attribute the cpu-cadence reduce floor
        // to serialize (incl. GPU→CPU via `to_blob`) / wire (cross-host
        // TCP round-trip) / deserialize. Summed across reduces, emitted
        // at teardown by `log_profile_summary`.
        if !self.prof_enabled {
            let mut frame = tensors_to_round_frame(tensors)?;
            frame.kind = kind;
            frame.weight = weight;
            let reduced = self.all_reduce(&frame)?;
            return Ok((round_frame_to_tensors(&reduced)?, reduced.weight));
        }
        let t0 = Instant::now();
        let mut frame = tensors_to_round_frame(tensors)?;
        frame.kind = kind;
        frame.weight = weight;
        let t1 = Instant::now();
        let reduced = self.all_reduce(&frame)?;
        let t2 = Instant::now();
        let out = round_frame_to_tensors(&reduced)?;
        let t3 = Instant::now();
        self.prof_serialize_ns += (t1 - t0).as_nanos();
        self.prof_wire_ns += (t2 - t1).as_nanos();
        self.prof_deserialize_ns += (t3 - t2).as_nanos();
        self.prof_bytes += frame
            .tensors
            .iter()
            .map(|p| p.bytes.len() as u64)
            .sum::<u64>();
        self.prof_count += 1;
        Ok((out, reduced.weight))
    }

    /// Convenience: equal-weight mean over the accepted cohort. Each
    /// rank contributes mass `1.0`, so the consensus is the plain mean
    /// of exactly the frames the controller accepted into the round.
    ///
    /// No production caller (the param bridge uses the mass-weighted
    /// [`all_reduce_weighted`](Self::all_reduce_weighted) directly); retained
    /// as the entry the `two_rank_tensor_average` relay integration test
    /// drives through the RoundFrame reduce path.
    #[allow(dead_code)]
    pub fn all_reduce_tensors(&mut self, tensors: &[&Tensor]) -> Result<Vec<Tensor>> {
        Ok(self
            .all_reduce_weighted(tensors, RoundKind::Model, 1.0)?
            .0)
    }

    /// Emit a one-line per-rank summary of the accumulated reduce
    /// profile (serialize / wire / deserialize split, per-reduce
    /// averages, and effective wire bandwidth). Called once at bridge
    /// teardown. Uses `eprintln!` (not `println!`) so it isn't lost to
    /// Docker's block-buffered stdout. No-op if no reduces ran.
    pub fn log_profile_summary(&self) {
        if self.prof_count == 0 {
            return;
        }
        let n = self.prof_count as f64;
        let ser = self.prof_serialize_ns as f64 / 1e6;
        let wire = self.prof_wire_ns as f64 / 1e6;
        let de = self.prof_deserialize_ns as f64 / 1e6;
        let total = (ser + wire + de).max(1e-9);
        let mb = self.prof_bytes as f64 / 1e6;
        // Wire carries the frame up and a same-sized averaged frame
        // down, so ~2× bytes traverse the link per reduce.
        let wire_s = self.prof_wire_ns as f64 / 1e9;
        let mbps = if wire_s > 0.0 { (mb * 2.0) / wire_s } else { 0.0 };
        eprintln!(
            "[cpu-reduce-prof] rank={} reduces={} | serialize={:.0}ms ({:.0}%) \
             wire={:.0}ms ({:.0}%) deserialize={:.0}ms ({:.0}%) | per-reduce \
             ser={:.2}ms wire={:.2}ms de={:.2}ms bytes={:.2}MB | wire~{:.1}MB/s(up+down)",
            self.rank_id,
            self.prof_count,
            ser,
            100.0 * ser / total,
            wire,
            100.0 * wire / total,
            de,
            100.0 * de / total,
            ser / n,
            wire / n,
            de / n,
            mb / n,
            mbps,
        );
    }

    /// Broadcast a root rank's tensors to every rank via a pure sum.
    ///
    /// Root sends its values; every other rank sends zeros. The
    /// [`RoundKind::Control`] sum delivers root's original values to
    /// every rank — no scaling tricks, no divisor.
    ///
    /// Used by the cluster-rank entry points to align initial parameter
    /// state across ranks (mirrors `nccl_comm.broadcast(refs, root=0)` on
    /// the NCCL path). Caller passes their factory-built params; the
    /// returned tensors carry root's values and should be loaded back
    /// into the live parameters via `copy_`.
    ///
    /// v1 supports root=0 only and f32 tensors (per
    /// [`tensors_to_round_frame`]). All ranks must call concurrently.
    pub fn broadcast_from_root(
        &mut self,
        tensors: &[&Tensor],
        root: u32,
    ) -> Result<Vec<Tensor>> {
        if root >= self.world_size {
            return Err(TensorError::new(&format!(
                "cpu_reduce: broadcast root {root} >= world_size {}",
                self.world_size,
            )));
        }
        // Build the per-rank contribution: root sends a copy of its
        // values, non-root ranks send zeros_like. Tensors are moved to
        // CPU via tensors_to_round_frame downstream; the copies are
        // short-lived (single-round scratch).
        let contribution: Vec<Tensor> = if self.rank_id == root {
            tensors
                .iter()
                .map(|t| {
                    let copy = Tensor::zeros_like(t)?;
                    copy.copy_(t, false)?;
                    Ok(copy)
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            tensors
                .iter()
                .map(|t| Tensor::zeros_like(t))
                .collect::<Result<Vec<_>>>()?
        };
        let refs: Vec<&Tensor> = contribution.iter().collect();
        Ok(self
            .all_reduce_weighted(&refs, RoundKind::Control, 0.0)?
            .0)
    }

    /// AllReduce-gather a per-rank `f64` measurement vector across the
    /// cluster via a pure sum.
    ///
    /// `local` must be length `world_size`. Each rank writes its own
    /// measurement into its own slot (other slots zero); the
    /// [`RoundKind::Control`] sum yields the gathered vector on every
    /// rank. Slots of ranks whose frames the controller did not accept
    /// (dead / lost mid-round) stay zero — the gather reports realized
    /// contributions only.
    ///
    /// Counterpart to [`Ddp::all_reduce_per_rank_f64`](crate::distributed::Ddp::all_reduce_per_rank_f64)
    /// — same semantics, CPU-routed. v1 carries f32 over the wire (the
    /// controller's only supported dtype); precision is preserved at the
    /// millisecond level for ElChe timing and at f32-mantissa precision
    /// for divergence aggregation, both within tolerance of the
    /// downstream consumers.
    ///
    /// All ranks must call concurrently.
    pub fn all_reduce_per_rank_f64(&mut self, local: &mut [f64]) -> Result<()> {
        let world_size = self.world_size as usize;
        if local.len() != world_size {
            return Err(TensorError::new(&format!(
                "cpu_reduce: all_reduce_per_rank_f64: vector len ({}) must \
                 equal world_size ({})",
                local.len(),
                world_size,
            )));
        }
        let vals: Vec<f32> = local.iter().map(|v| *v as f32).collect();
        let tensor = Tensor::from_f32(
            &vals,
            &[world_size as i64],
            Device::CPU,
        )?;
        // Bookkeeping reduce: tag it `Control` so the consensus-checkpoint
        // forge never mistakes this count vector for a slice of the model.
        let mut frame = tensors_to_round_frame(&[&tensor])?;
        frame.kind = RoundKind::Control;
        let averaged = self.all_reduce(&frame)?;
        let out = round_frame_to_tensors(&averaged)?;
        let avg = out
            .first()
            .ok_or_else(|| TensorError::new("cpu_reduce: count-gather returned empty frame"))?;
        let out = avg.to_f32_vec()?;
        for (dst, src) in local.iter_mut().zip(out) {
            *dst = src as f64;
        }
        Ok(())
    }

}
// NOTE: an `AsyncCpuReduceClient` (split read/write, background reader
// thread) used to live here. It had zero production users — cpu-async
// rides the same blocking param bridge as sync/cadence (the worker's
// non-blocking behavior comes from the coordinator's 3-phase state
// machine, not a rank-side split client) — so it was removed rather
// than shipped dead. Recover from git history if a rank-side async
// client is ever wanted.

// ---------------------------------------------------------------------------
// Length-framed RoundFrame helpers (rank ↔ relay loopback leg)
// ---------------------------------------------------------------------------

/// Serialize `frame` (with its HMAC footer) and write it length-delimited
/// to `stream`. The rank talks to its host-local relay, which forwards the
/// opaque blob upstream untouched; the length prefix lets the relay frame
/// it without parsing. See [`crate::distributed::relay::mux`].
fn write_framed_round<W: Write>(
    stream: &mut W,
    frame: &RoundFrame,
    salt: &SessionSalt,
) -> Result<()> {
    let mut buf = Vec::new();
    controller::write_round_frame(&mut buf, frame, salt)?;
    crate::distributed::relay::mux::write_len_framed(stream, &buf)
}

/// Read a length-delimited [`RoundFrame`] blob and parse it. `Ok(None)` on
/// clean EOF (relay/controller closed the connection).
fn read_framed_round<R: Read>(stream: &mut R, salt: &SessionSalt) -> Result<Option<RoundFrame>> {
    match crate::distributed::relay::mux::read_len_framed(stream)? {
        Some(buf) => controller::read_round_frame(&mut buf.as_slice(), salt),
        None => Ok(None),
    }
}

// ---------------------------------------------------------------------------
// Tensor ↔ RoundFrame conversion
// ---------------------------------------------------------------------------

/// Build a [`RoundFrame`] from a slice of tensors.
///
/// Each tensor is moved to CPU via [`Tensor::to_blob`] (transparently
/// handles GPU→CPU transfer) and serialized as raw native-byte-order
/// f32 bytes. Shape is captured as `Vec<u32>` (matches the wire
/// protocol; loud error if any dim doesn't fit in u32).
///
/// v1 dtype support: f32 only. Other dtypes produce a loud error with
/// a pointer to where to extend (mirrors controller-side reduce_average
/// restriction; both must lift together when adding f16/bf16).
pub fn tensors_to_round_frame(tensors: &[&Tensor]) -> Result<RoundFrame> {
    let mut payloads = Vec::with_capacity(tensors.len());
    for (i, t) in tensors.iter().enumerate() {
        if t.dtype() != DType::Float32 {
            return Err(TensorError::new(&format!(
                "cpu_reduce: tensor[{i}] dtype {:?} not supported in v1 \
                 (only Float32). Extend cpu_reduce.rs::tensors_to_round_frame \
                 and controller.rs::reduce_average together to add support.",
                t.dtype()
            )));
        }
        let shape_i64 = t.shape();
        let shape: Vec<u32> = shape_i64
            .iter()
            .enumerate()
            .map(|(d_idx, d)| {
                u32::try_from(*d).map_err(|_| {
                    TensorError::new(&format!(
                        "cpu_reduce: tensor[{i}] dim[{d_idx}] = {d} doesn't fit in u32 \
                         (wire protocol uses u32 shape dims)"
                    ))
                })
            })
            .collect::<Result<_>>()?;
        let bytes = t.to_blob()?;
        payloads.push(TensorPayload {
            dtype: DTYPE_F32,
            shape,
            bytes,
        });
    }
    // Default to a model-weight frame with no realized-work mass;
    // senders set `kind` / `weight` on the built frame (see
    // `CpuReduceClient::all_reduce_weighted`).
    Ok(RoundFrame {
        tensors: payloads,
        kind: RoundKind::Model,
        weight: 0.0,
    })
}

/// Build a list of new CPU `Tensor`s from a [`RoundFrame`].
///
/// Inverse of [`tensors_to_round_frame`]. Each payload's bytes are
/// interpreted as little-endian f32 (matches the wire format), reshaped
/// per the payload's shape, and packed into a fresh CPU tensor. v1
/// supports f32 only.
///
/// The returned tensors live on `Device::CPU`. Callers wanting them on
/// GPU should follow up with [`Tensor::to_device`].
pub fn round_frame_to_tensors(frame: &RoundFrame) -> Result<Vec<Tensor>> {
    let mut out = Vec::with_capacity(frame.tensors.len());
    for (i, p) in frame.tensors.iter().enumerate() {
        if p.dtype != DTYPE_F32 {
            return Err(TensorError::new(&format!(
                "cpu_reduce: payload[{i}] dtype {} not supported in v1 \
                 (only DTYPE_F32 = 0)",
                p.dtype
            )));
        }
        if p.bytes.len() % 4 != 0 {
            return Err(TensorError::new(&format!(
                "cpu_reduce: payload[{i}] byte count {} not divisible by 4 \
                 (f32 element size)",
                p.bytes.len()
            )));
        }
        let n = p.bytes.len() / 4;
        let mut data = Vec::with_capacity(n);
        for j in 0..n {
            let mut b = [0u8; 4];
            b.copy_from_slice(&p.bytes[j * 4..(j + 1) * 4]);
            data.push(f32::from_le_bytes(b));
        }
        let shape: Vec<i64> = p.shape.iter().map(|&d| d as i64).collect();
        let numel_from_shape: i64 = shape.iter().product();
        if numel_from_shape != n as i64 {
            return Err(TensorError::new(&format!(
                "cpu_reduce: payload[{i}] shape {shape:?} numel {numel_from_shape} \
                 != bytes-derived numel {n}"
            )));
        }
        out.push(Tensor::from_f32(&data, &shape, Device::CPU)?);
    }
    Ok(out)
}

#[cfg(test)]
#[path = "cpu_reduce_tests.rs"]
mod tests;
