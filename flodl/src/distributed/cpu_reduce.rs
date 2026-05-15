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
use std::sync::mpsc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::distributed::controller::{
    self, DTYPE_F32, HANDSHAKE_MAGIC_CONTROLLER_ACK, HANDSHAKE_MAGIC_RANK, PROTOCOL_VERSION,
    RoundFrame, TensorPayload,
};
use crate::tensor::{DType, Device, Result, Tensor, TensorError};

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

    /// Convenience: build a [`RoundFrame`] from a slice of tensors, call
    /// [`Self::all_reduce`], and convert the averaged frame back to a
    /// `Vec<Tensor>` on CPU.
    ///
    /// v1 supports f32 only; loud error on other dtypes. Caller is
    /// responsible for moving averaged tensors back to GPU if needed.
    pub fn all_reduce_tensors(&mut self, tensors: &[&Tensor]) -> Result<Vec<Tensor>> {
        let frame = tensors_to_round_frame(tensors)?;
        let averaged = self.all_reduce(&frame)?;
        round_frame_to_tensors(&averaged)
    }

    /// Broadcast a root rank's tensors to every rank via the avg-trick.
    ///
    /// The controller only supports AllReduce-Avg (sum then divide by
    /// `world_size`). To express "every rank receives root's values" with
    /// that primitive: root scales its tensors by `world_size`, every
    /// other rank sends zeros. After avg = sum/world_size, every rank
    /// ends up with root's original values.
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
        // Build the per-rank contribution: root multiplies by world_size,
        // non-root ranks send zeros_like. Tensors are moved to CPU via
        // tensors_to_round_frame downstream; the scaled / zeroed copies
        // are short-lived (single-round scratch).
        let n = self.world_size as f64;
        let scaled: Vec<Tensor> = if self.rank_id == root {
            tensors
                .iter()
                .map(|t| {
                    let copy = Tensor::zeros_like(t)?;
                    copy.copy_(t, false)?;
                    copy.mul_scalar_(n)?;
                    Ok(copy)
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            tensors
                .iter()
                .map(|t| Tensor::zeros_like(t))
                .collect::<Result<Vec<_>>>()?
        };
        let scaled_refs: Vec<&Tensor> = scaled.iter().collect();
        self.all_reduce_tensors(&scaled_refs)
    }

    /// AllReduce-gather a per-rank `f64` measurement vector across the
    /// cluster via the avg-trick.
    ///
    /// `local` must be length `world_size`. Each rank writes its own
    /// measurement into its own slot (other slots zero). The values are
    /// scaled by `world_size` before send so the controller's
    /// divide-by-`world_size` cancels, yielding the gathered vector on
    /// every rank.
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
        let n = self.world_size as f32;
        let scaled: Vec<f32> = local.iter().map(|v| (*v as f32) * n).collect();
        let tensor = Tensor::from_f32(
            &scaled,
            &[world_size as i64],
            Device::CPU,
        )?;
        let avg = self.all_reduce_tensors(&[&tensor])?;
        let out = avg[0].to_f32_vec()?;
        for (dst, src) in local.iter_mut().zip(out) {
            *dst = src as f64;
        }
        Ok(())
    }

    /// AllReduce-average tensors and return this rank's weight-space
    /// divergence triple `(divergence, post_norm, pre_norm)`.
    ///
    /// CPU-routed counterpart to
    /// [`Ddp::average_params_with_divergence`](crate::distributed::Ddp::average_params_with_divergence).
    /// Same divergence math (`||W_pre − W_post|| / ||W_post||`,
    /// `pre_norm = sqrt(Σᵢ ||scratchᵢ||²)`,
    /// `post_norm = sqrt(Σᵢ ||paramsᵢ||²)`) — only the reduce primitive
    /// switches from in-place NCCL AllReduce to TCP round-trip via
    /// [`Self::all_reduce_tensors`].
    ///
    /// Unlike NCCL's in-place AllReduce, `all_reduce_tensors` returns
    /// *new* averaged tensors, so the caller's `params` are mutated via
    /// a `copy_` step in this method to make the post-AllReduce values
    /// visible downstream (`foreach_norm` on `params` then reads the
    /// averaged state).
    ///
    /// `scratch` must have the same length as `params` (allocate via the
    /// cluster-rank helper's pre-loop step, e.g. `zeros_like` per param).
    ///
    /// All ranks must call concurrently.
    pub fn average_params_with_divergence(
        &mut self,
        params: &[&Tensor],
        scratch: &[Tensor],
    ) -> Result<(f64, Option<f64>, Option<f64>)> {
        if params.is_empty() {
            return Ok((0.0, None, None));
        }
        if scratch.len() != params.len() {
            return Err(TensorError::new(&format!(
                "cpu_reduce: average_params_with_divergence: scratch.len() \
                 ({}) must equal params.len() ({})",
                scratch.len(),
                params.len(),
            )));
        }

        // Snapshot pre-sync params into scratch.
        for (dst, src) in scratch.iter().zip(params.iter()) {
            dst.copy_(src, false)?;
        }

        // CPU AllReduce-Avg: returns new averaged tensors; copy into
        // live params so post-AllReduce values are visible to the
        // foreach_norm pass + downstream training.
        let averaged = self.all_reduce_tensors(params)?;
        for (dst, src) in params.iter().zip(&averaged) {
            dst.copy_(src, false)?;
        }

        // pre_norm BEFORE the next foreach mutates scratch in place.
        let pre_norm_tensors = Tensor::foreach_norm(scratch, 2.0)?;
        let mut pre_sq = 0.0f64;
        for n in &pre_norm_tensors {
            let v: f64 = n.item()?;
            pre_sq += v * v;
        }
        let pre_norm = pre_sq.sqrt();

        // scratch[i] += -1 * params[i]  →  scratch[i] = pre - post.
        let param_owned: Vec<Tensor> = params.iter().map(|t| (*t).clone()).collect();
        Tensor::foreach_add_list_(scratch, &param_owned, -1.0)?;

        let diff_norms = Tensor::foreach_norm(scratch, 2.0)?;
        let post_norms = Tensor::foreach_norm(&param_owned, 2.0)?;

        let mut diff_sq = 0.0f64;
        for n in &diff_norms {
            let v: f64 = n.item()?;
            diff_sq += v * v;
        }
        let mut post_sq = 0.0f64;
        for n in &post_norms {
            let v: f64 = n.item()?;
            post_sq += v * v;
        }
        let post_norm = post_sq.sqrt();
        let divergence = if post_norm > 1e-10 {
            diff_sq.sqrt() / post_norm
        } else {
            0.0
        };

        Ok((divergence, Some(post_norm), Some(pre_norm)))
    }

    /// Transform this blocking client into an asynchronous one that can
    /// `submit_round` (non-blocking write) and `poll_round` (non-blocking
    /// try-receive of the averaged response).
    ///
    /// Spawns a background reader thread that consumes
    /// [`RoundFrame`]s from the TCP stream and forwards them through an
    /// mpsc channel. The main thread keeps writing rounds and polling
    /// the channel independently, so the training loop never blocks on
    /// the controller's round-trip — the core of the CPU-Async
    /// semantics in the cluster-rank model.
    ///
    /// Consumes `self`. After this call, the blocking
    /// [`Self::all_reduce`] / [`Self::all_reduce_tensors`] / etc. APIs
    /// are no longer available — only the async pair on
    /// [`AsyncCpuReduceClient`].
    ///
    /// Loud error if [`TcpStream::try_clone`] fails.
    pub fn into_async(self) -> Result<AsyncCpuReduceClient> {
        let CpuReduceClient { stream, rank_id, world_size } = self;

        // Clone the stream so the background reader and the main-thread
        // writer have independent handles. TCP reads and writes can
        // proceed concurrently on Linux when split this way (each
        // handle owns its own dup'd FD).
        let mut reader_stream = stream.try_clone().map_err(|e| {
            TensorError::new(&format!(
                "cpu_reduce: stream try_clone for async mode failed: {e}"
            ))
        })?;

        let (tx, rx) = mpsc::channel::<Result<RoundFrame>>();

        let reader_handle = thread::Builder::new()
            .name(format!("cpu-async-reader-r{rank_id}"))
            .spawn(move || {
                // Read frames until EOF or error. Errors are forwarded
                // to the main thread (which surfaces them on
                // poll_round); EOF is a clean shutdown signal.
                loop {
                    match controller::read_round_frame(&mut reader_stream) {
                        Ok(Some(frame)) => {
                            if tx.send(Ok(frame)).is_err() {
                                // Main side dropped the receiver.
                                break;
                            }
                        }
                        Ok(None) => break, // clean EOF
                        Err(e) => {
                            let _ = tx.send(Err(e));
                            break;
                        }
                    }
                }
            })
            .map_err(|e| TensorError::new(&format!(
                "cpu_reduce: spawn async reader thread failed: {e}"
            )))?;

        Ok(AsyncCpuReduceClient {
            writer: stream,
            incoming: rx,
            reader_handle: Some(reader_handle),
            rank_id,
            world_size,
        })
    }
}

/// Async CPU-averaging client: split read/write for non-blocking round
/// dispatch.
///
/// Obtained via [`CpuReduceClient::into_async`]. The training loop
/// submits a round (frame written immediately on the main thread) and
/// continues training; later polls drain the averaged response from a
/// background reader thread via an mpsc channel.
///
/// **Lifetime**: holds a join handle for the reader thread; on drop,
/// the reader exits when the receiver side closes (EOF on the TCP
/// stream, which the launcher's CpuAverager closes when shutdown
/// signals).
#[derive(Debug)]
pub struct AsyncCpuReduceClient {
    writer: TcpStream,
    incoming: mpsc::Receiver<Result<RoundFrame>>,
    reader_handle: Option<JoinHandle<()>>,
    rank_id: u32,
    world_size: u32,
}

impl AsyncCpuReduceClient {
    /// This rank's id, as told to the controller.
    pub fn rank_id(&self) -> u32 {
        self.rank_id
    }

    /// The world size baked into the handshake.
    pub fn world_size(&self) -> u32 {
        self.world_size
    }

    /// Submit a round's frame to the controller. The training loop
    /// continues immediately — the controller's averaged response
    /// arrives asynchronously and is drained later via
    /// [`Self::poll_round`].
    ///
    /// Write is blocking (TCP send buffer typically accepts a round
    /// without back-pressure). Loud error on wire-level write failure.
    pub fn submit_round(&mut self, frame: &RoundFrame) -> Result<()> {
        controller::write_round_frame(&mut self.writer, frame)
    }

    /// Try to receive the next averaged round from the background
    /// reader. Non-blocking.
    ///
    /// Returns:
    /// - `Ok(Some(frame))` — a round completed; caller applies EASGD
    ///   blend / divergence math.
    /// - `Ok(None)` — no completed round yet; keep training.
    /// - `Err(...)` — reader thread surfaced a wire-level error, or
    ///   the channel disconnected (controller closed the stream).
    pub fn poll_round(&self) -> Result<Option<RoundFrame>> {
        match self.incoming.try_recv() {
            Ok(Ok(frame)) => Ok(Some(frame)),
            Ok(Err(e)) => Err(e),
            Err(mpsc::TryRecvError::Empty) => Ok(None),
            Err(mpsc::TryRecvError::Disconnected) => Err(TensorError::new(
                "cpu_reduce: async reader thread disconnected \
                 (controller closed stream or reader hit a wire error \
                 already surfaced)",
            )),
        }
    }

    /// Block until the next averaged round arrives. Loops
    /// [`Self::poll_round`] with a short sleep so the OS can schedule
    /// other ranks' training threads / the reader thread.
    ///
    /// Used at K-boundary when the previous round is still in flight
    /// (`max_overshoot = 1`): submit-and-wait semantics.
    pub fn block_poll(&self) -> Result<RoundFrame> {
        loop {
            match self.poll_round()? {
                Some(frame) => return Ok(frame),
                None => thread::sleep(Duration::from_micros(100)),
            }
        }
    }
}

impl Drop for AsyncCpuReduceClient {
    fn drop(&mut self) {
        // Closing writer half EOFs the controller's read side; reader
        // thread receives EOF on its own clone and exits naturally.
        // Wait for it to finish to avoid spurious "thread leaked"
        // diagnostics; best-effort (don't propagate join errors).
        if let Some(handle) = self.reader_handle.take() {
            let _ = handle.join();
        }
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
    Ok(RoundFrame { tensors: payloads })
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

    // --- Tensor ↔ RoundFrame conversion ---

    #[test]
    fn tensor_round_trip_through_round_frame() {
        let data: &[f32] = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor::from_f32(data, &[2, 3], Device::CPU).unwrap();
        let refs = vec![&t];
        let frame = tensors_to_round_frame(&refs).unwrap();
        assert_eq!(frame.tensors.len(), 1);
        assert_eq!(frame.tensors[0].dtype, DTYPE_F32);
        assert_eq!(frame.tensors[0].shape, vec![2u32, 3]);
        assert_eq!(frame.tensors[0].bytes.len(), 6 * 4);

        let recovered = round_frame_to_tensors(&frame).unwrap();
        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0].shape(), vec![2i64, 3]);
        assert_eq!(recovered[0].to_f32_vec().unwrap(), data);
    }

    #[test]
    fn tensors_to_round_frame_rejects_non_f32() {
        let t = Tensor::from_f64(&[1.0, 2.0], &[2], Device::CPU).unwrap();
        let refs = vec![&t];
        let err = tensors_to_round_frame(&refs).unwrap_err();
        assert!(
            err.to_string().contains("Float64") && err.to_string().contains("Float32"),
            "got: {err}"
        );
    }

    #[test]
    fn round_frame_to_tensors_rejects_shape_byte_mismatch() {
        // shape claims 4 elements but only 2 f32 worth of bytes (8 bytes)
        let bogus = RoundFrame {
            tensors: vec![TensorPayload {
                dtype: DTYPE_F32,
                shape: vec![4],
                bytes: vec![0u8; 8],
            }],
        };
        let err = round_frame_to_tensors(&bogus).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("numel") && (msg.contains("!=") || msg.contains("mismatch")),
            "got: {msg}"
        );
    }

    /// End-to-end: two ranks ship Tensor lists through CpuAverager;
    /// receive averaged Tensor lists back.
    #[test]
    fn two_rank_tensor_average() {
        let avg = crate::distributed::controller::CpuAverager::start(
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
            let t = Tensor::from_f32(&[2.0, 4.0, 6.0], &[3], Device::CPU).unwrap();
            let out = c.all_reduce_tensors(&[&t]).unwrap();
            tx0.send(out).unwrap();
            drop(c);
        });
        let t1 = thread::spawn(move || {
            let mut c = CpuReduceClient::connect(addr, 1, 2).unwrap();
            let t = Tensor::from_f32(&[4.0, 8.0, 12.0], &[3], Device::CPU).unwrap();
            let out = c.all_reduce_tensors(&[&t]).unwrap();
            tx1.send(out).unwrap();
            drop(c);
        });

        let r0 = rx0.recv().unwrap();
        let r1 = rx1.recv().unwrap();
        t0.join().unwrap();
        t1.join().unwrap();
        avg.shutdown().unwrap();

        assert_eq!(r0.len(), 1);
        assert_eq!(r0[0].to_f32_vec().unwrap(), vec![3.0, 6.0, 9.0]);
        assert_eq!(r1[0].to_f32_vec().unwrap(), vec![3.0, 6.0, 9.0]);
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
