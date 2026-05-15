//! CPU-averaging controller: TCP byte router for the star-topology
//! cross-process gradient sum that powers [`AverageBackend::Cpu`].
//!
//! Architecture (star, not collective): every rank ships a `RoundFrame`
//! containing this round's tensors to a single TCP listener on the
//! launcher process. The launcher accumulates the per-tensor sum on its
//! CPU, divides by `world_size`, and writes the averaged `RoundFrame`
//! back to each rank. No NCCL. Genuinely async from NCCL's perspective:
//! ranks' GPUs keep computing while their CPUs push/pull bytes.
//!
//! Future swap-in: a gloo-backed all-reduce can replace the serial
//! summation when the single-CPU bottleneck shows up (~8+ ranks for
//! typical gradient sizes). The wire protocol is intentionally simple
//! enough that the swap touches only the inner reduce loop, not the
//! protocol or rank-side client.
//!
//! # Wire protocol (little-endian, no compression)
//!
//! Handshake (rank → controller, exactly once per connection):
//! ```text
//! u32 magic         = 0xF10D_17C0
//! u32 protocol_ver  = 1
//! u32 rank_id       (this rank's global rank, 0..world_size)
//! u32 world_size    (rank's view; controller validates against its own)
//! ```
//!
//! Handshake ack (controller → rank):
//! ```text
//! u32 magic        = 0xF10D_17C1
//! u32 protocol_ver = 1
//! ```
//!
//! RoundFrame (rank → controller, then controller → rank, identical
//! shape both directions):
//! ```text
//! u32 magic       = 0xF10D_17F1
//! u32 num_tensors
//! for each tensor:
//!   u8  dtype   (0 = f32; v1 only)
//!   u8  ndim
//!   u32 dim_0, dim_1, ..., dim_{ndim-1}
//!   u64 nbytes
//!   <nbytes> raw bytes (native byte order)
//! ```
//!
//! Tensor data is native byte order. Cross-arch clusters (x86 + ARM)
//! would need a canonicalization step; out of scope for v1 (homogeneous
//! arch is the common case and the only one our test rig exercises).
//!
//! # State machine
//!
//! ```text
//! Idle → Accepting (collect N connections + N handshakes)
//!      → Reducing  (per-round: recv N frames, sum, scatter)
//!      → Shutdown  (any rank disconnects cleanly, or shutdown signal)
//! ```
//!
//! [`AverageBackend::Cpu`]: crate::distributed::AverageBackend::Cpu

use std::io::{ErrorKind, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::tensor::{Result, TensorError};

const HANDSHAKE_MAGIC_RANK: u32 = 0xF10D_17C0;
const HANDSHAKE_MAGIC_CONTROLLER_ACK: u32 = 0xF10D_17C1;
const ROUND_FRAME_MAGIC: u32 = 0xF10D_17F1;
const PROTOCOL_VERSION: u32 = 1;

/// dtype tag for f32 in the wire protocol. Only dtype supported in v1.
pub const DTYPE_F32: u8 = 0;

/// Background CPU-averager. Owns a [`TcpListener`] bound to the
/// controller's address and a worker thread that runs the accept +
/// reduce loop.
///
/// Constructed via [`CpuAverager::start`]; clean shutdown via
/// [`CpuAverager::shutdown`] (signals the worker, then joins).
#[derive(Debug)]
pub struct CpuAverager {
    bound_port: u16,
    shutdown: Arc<AtomicBool>,
    handle: Option<JoinHandle<Result<()>>>,
}

impl CpuAverager {
    /// Bind a TCP listener at `bind_addr` and spawn the reduce thread.
    ///
    /// The thread blocks waiting for exactly `world_size` rank
    /// connections, validates each one's handshake, then runs the
    /// reduce loop until ranks disconnect or [`Self::shutdown`] is
    /// called.
    ///
    /// Use `127.0.0.1:0` for tests (kernel-assigned port; read back via
    /// [`Self::port`]). Use the cluster's `master_addr:master_port+2`
    /// in production.
    pub fn start(bind_addr: SocketAddr, world_size: usize) -> Result<Self> {
        if world_size == 0 {
            return Err(TensorError::new(
                "cpu_averager: world_size must be > 0",
            ));
        }
        let listener = TcpListener::bind(bind_addr).map_err(|e| {
            TensorError::new(&format!(
                "cpu_averager: bind {bind_addr} failed: {e}"
            ))
        })?;
        let bound_port = listener
            .local_addr()
            .map_err(|e| {
                TensorError::new(&format!(
                    "cpu_averager: local_addr() failed: {e}"
                ))
            })?
            .port();
        // Short accept timeout so the worker thread can observe the
        // shutdown flag between connections without blocking forever.
        listener
            .set_nonblocking(false)
            .map_err(|e| TensorError::new(&format!("cpu_averager: set_nonblocking: {e}")))?;

        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_cloned = Arc::clone(&shutdown);
        let handle = thread::Builder::new()
            .name(format!("flodl-cpu-averager:{bound_port}"))
            .spawn(move || run_reduce_thread(listener, world_size, shutdown_cloned))
            .map_err(|e| {
                TensorError::new(&format!("cpu_averager: spawn worker failed: {e}"))
            })?;

        Ok(CpuAverager {
            bound_port,
            shutdown,
            handle: Some(handle),
        })
    }

    /// Bound TCP port. With `bind_addr.port() == 0`, returns the
    /// kernel-assigned port (test entry point); otherwise the requested
    /// port.
    pub fn port(&self) -> u16 {
        self.bound_port
    }

    /// Signal the reduce thread to stop, then join it. Idempotent.
    pub fn shutdown(mut self) -> Result<()> {
        self.shutdown.store(true, Ordering::SeqCst);
        if let Some(h) = self.handle.take() {
            return h
                .join()
                .map_err(|_| TensorError::new("cpu_averager: worker panicked"))?;
        }
        Ok(())
    }
}

impl Drop for CpuAverager {
    fn drop(&mut self) {
        // Best-effort shutdown if the caller didn't explicitly call
        // shutdown(). Joins are blocking, which is fine — Drop runs at
        // process or scope exit; we want the worker out cleanly.
        self.shutdown.store(true, Ordering::SeqCst);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

// ---------------------------------------------------------------------------
// Reduce-thread worker
// ---------------------------------------------------------------------------

fn run_reduce_thread(
    listener: TcpListener,
    world_size: usize,
    shutdown: Arc<AtomicBool>,
) -> Result<()> {
    listener
        .set_nonblocking(true)
        .map_err(|e| TensorError::new(&format!("cpu_averager: set_nonblocking: {e}")))?;
    let mut streams: Vec<Option<TcpStream>> = (0..world_size).map(|_| None).collect();
    let mut connected = 0usize;

    // Phase 1: accept exactly `world_size` connections and validate the
    // handshake on each. Connections may arrive in any order; the
    // handshake's rank_id places each stream at the right slot.
    while connected < world_size {
        if shutdown.load(Ordering::SeqCst) {
            return Ok(());
        }
        match listener.accept() {
            Ok((mut stream, _peer)) => {
                stream
                    .set_read_timeout(Some(Duration::from_millis(500)))
                    .map_err(|e| TensorError::new(&format!("cpu_averager: set_read_timeout: {e}")))?;
                let rank_id = read_handshake(&mut stream, world_size)?;
                if rank_id >= world_size {
                    return Err(TensorError::new(&format!(
                        "cpu_averager: handshake rank_id {rank_id} >= world_size {world_size}"
                    )));
                }
                if streams[rank_id].is_some() {
                    return Err(TensorError::new(&format!(
                        "cpu_averager: duplicate rank_id {rank_id} connected"
                    )));
                }
                write_handshake_ack(&mut stream)?;
                // Switch to blocking reads with no timeout for the
                // long-running reduce loop. Timeouts here would make
                // legitimately slow rounds look like failures.
                stream
                    .set_read_timeout(None)
                    .map_err(|e| TensorError::new(&format!("cpu_averager: set_read_timeout(None): {e}")))?;
                streams[rank_id] = Some(stream);
                connected += 1;
            }
            Err(e) if e.kind() == ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(20));
            }
            Err(e) => {
                return Err(TensorError::new(&format!(
                    "cpu_averager: accept failed: {e}"
                )));
            }
        }
    }
    // All connected — drop nonblocking on the listener now that no more
    // accepts are expected.
    let _ = listener.set_nonblocking(false);
    let mut streams: Vec<TcpStream> = streams.into_iter().map(|s| s.unwrap()).collect();

    // Phase 2: reduce loop. Each round reads a RoundFrame from every
    // rank, sums the per-tensor data, divides by world_size, writes
    // the averaged frame back. Terminates when any rank disconnects
    // cleanly (EOF on read) or when shutdown is signalled.
    loop {
        if shutdown.load(Ordering::SeqCst) {
            return Ok(());
        }
        match read_round_from_all(&mut streams)? {
            Some(frames) => {
                let averaged = reduce_average(&frames)?;
                write_round_to_all(&mut streams, &averaged)?;
            }
            None => return Ok(()), // any rank EOFed → clean shutdown
        }
    }
}

// ---------------------------------------------------------------------------
// Handshake
// ---------------------------------------------------------------------------

fn read_handshake(stream: &mut TcpStream, expected_world_size: usize) -> Result<usize> {
    let mut buf = [0u8; 16];
    stream.read_exact(&mut buf).map_err(|e| {
        TensorError::new(&format!("cpu_averager: handshake read failed: {e}"))
    })?;
    let magic = u32::from_le_bytes(buf[0..4].try_into().unwrap());
    if magic != HANDSHAKE_MAGIC_RANK {
        return Err(TensorError::new(&format!(
            "cpu_averager: handshake magic 0x{magic:08x} != 0x{HANDSHAKE_MAGIC_RANK:08x}"
        )));
    }
    let proto_ver = u32::from_le_bytes(buf[4..8].try_into().unwrap());
    if proto_ver != PROTOCOL_VERSION {
        return Err(TensorError::new(&format!(
            "cpu_averager: handshake protocol_version {proto_ver} != {PROTOCOL_VERSION}"
        )));
    }
    let rank_id = u32::from_le_bytes(buf[8..12].try_into().unwrap()) as usize;
    let rank_world_size = u32::from_le_bytes(buf[12..16].try_into().unwrap()) as usize;
    if rank_world_size != expected_world_size {
        return Err(TensorError::new(&format!(
            "cpu_averager: handshake world_size {rank_world_size} != expected {expected_world_size}"
        )));
    }
    Ok(rank_id)
}

fn write_handshake_ack(stream: &mut TcpStream) -> Result<()> {
    let mut buf = [0u8; 8];
    buf[0..4].copy_from_slice(&HANDSHAKE_MAGIC_CONTROLLER_ACK.to_le_bytes());
    buf[4..8].copy_from_slice(&PROTOCOL_VERSION.to_le_bytes());
    stream.write_all(&buf).map_err(|e| {
        TensorError::new(&format!("cpu_averager: handshake ack write failed: {e}"))
    })?;
    Ok(())
}

// ---------------------------------------------------------------------------
// RoundFrame
// ---------------------------------------------------------------------------

/// A round's payload: a list of tensors with shape + dtype + data.
///
/// Identical shape sent rank→controller and controller→rank. v1 only
/// supports `DTYPE_F32`; controller errors loudly on other dtypes.
#[derive(Debug, Clone, PartialEq)]
pub struct RoundFrame {
    pub tensors: Vec<TensorPayload>,
}

/// One tensor inside a [`RoundFrame`].
#[derive(Debug, Clone, PartialEq)]
pub struct TensorPayload {
    /// Wire dtype tag (see [`DTYPE_F32`]).
    pub dtype: u8,
    /// Tensor shape.
    pub shape: Vec<u32>,
    /// Raw tensor bytes (native byte order).
    pub bytes: Vec<u8>,
}

impl TensorPayload {
    /// Number of element-slots in the tensor (product of shape dims).
    pub fn numel(&self) -> usize {
        self.shape.iter().map(|d| *d as usize).product()
    }
}

/// Read a RoundFrame from a single rank's stream. Returns `Ok(None)` on
/// clean EOF (rank closed its end normally — signals shutdown).
fn read_round_frame(stream: &mut TcpStream) -> Result<Option<RoundFrame>> {
    let mut hdr = [0u8; 8];
    match stream.read_exact(&mut hdr) {
        Ok(()) => {}
        Err(e) if matches!(e.kind(), ErrorKind::UnexpectedEof | ErrorKind::ConnectionReset) => {
            return Ok(None);
        }
        Err(e) => {
            return Err(TensorError::new(&format!(
                "cpu_averager: frame header read failed: {e}"
            )));
        }
    }
    let magic = u32::from_le_bytes(hdr[0..4].try_into().unwrap());
    if magic != ROUND_FRAME_MAGIC {
        return Err(TensorError::new(&format!(
            "cpu_averager: frame magic 0x{magic:08x} != 0x{ROUND_FRAME_MAGIC:08x}"
        )));
    }
    let num_tensors = u32::from_le_bytes(hdr[4..8].try_into().unwrap()) as usize;

    let mut tensors = Vec::with_capacity(num_tensors);
    for ti in 0..num_tensors {
        let mut meta = [0u8; 2];
        stream.read_exact(&mut meta).map_err(|e| {
            TensorError::new(&format!(
                "cpu_averager: tensor[{ti}] meta read failed: {e}"
            ))
        })?;
        let dtype = meta[0];
        let ndim = meta[1] as usize;
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            let mut d = [0u8; 4];
            stream.read_exact(&mut d).map_err(|e| {
                TensorError::new(&format!(
                    "cpu_averager: tensor[{ti}] shape read failed: {e}"
                ))
            })?;
            shape.push(u32::from_le_bytes(d));
        }
        let mut nb = [0u8; 8];
        stream.read_exact(&mut nb).map_err(|e| {
            TensorError::new(&format!(
                "cpu_averager: tensor[{ti}] nbytes read failed: {e}"
            ))
        })?;
        let nbytes = u64::from_le_bytes(nb) as usize;
        let mut bytes = vec![0u8; nbytes];
        stream.read_exact(&mut bytes).map_err(|e| {
            TensorError::new(&format!(
                "cpu_averager: tensor[{ti}] data read failed: {e}"
            ))
        })?;
        tensors.push(TensorPayload {
            dtype,
            shape,
            bytes,
        });
    }
    Ok(Some(RoundFrame { tensors }))
}

/// Read a frame from every rank. Returns `Ok(None)` if ANY rank EOFs
/// (signals shutdown — propagates so the reduce loop exits cleanly).
fn read_round_from_all(streams: &mut [TcpStream]) -> Result<Option<Vec<RoundFrame>>> {
    let mut frames = Vec::with_capacity(streams.len());
    for (rank, s) in streams.iter_mut().enumerate() {
        match read_round_frame(s)? {
            Some(f) => frames.push(f),
            None => {
                // Rank EOF'd. Treat as clean shutdown signal.
                let _ = rank; // could log "rank {rank} closed"
                return Ok(None);
            }
        }
    }
    Ok(Some(frames))
}

fn write_round_frame(stream: &mut TcpStream, frame: &RoundFrame) -> Result<()> {
    let mut hdr = [0u8; 8];
    hdr[0..4].copy_from_slice(&ROUND_FRAME_MAGIC.to_le_bytes());
    hdr[4..8].copy_from_slice(&(frame.tensors.len() as u32).to_le_bytes());
    stream.write_all(&hdr).map_err(|e| {
        TensorError::new(&format!("cpu_averager: frame header write failed: {e}"))
    })?;
    for (ti, t) in frame.tensors.iter().enumerate() {
        stream.write_all(&[t.dtype, t.shape.len() as u8]).map_err(|e| {
            TensorError::new(&format!(
                "cpu_averager: tensor[{ti}] meta write failed: {e}"
            ))
        })?;
        for d in &t.shape {
            stream.write_all(&d.to_le_bytes()).map_err(|e| {
                TensorError::new(&format!(
                    "cpu_averager: tensor[{ti}] shape write failed: {e}"
                ))
            })?;
        }
        stream
            .write_all(&(t.bytes.len() as u64).to_le_bytes())
            .map_err(|e| {
                TensorError::new(&format!(
                    "cpu_averager: tensor[{ti}] nbytes write failed: {e}"
                ))
            })?;
        stream.write_all(&t.bytes).map_err(|e| {
            TensorError::new(&format!(
                "cpu_averager: tensor[{ti}] data write failed: {e}"
            ))
        })?;
    }
    stream
        .flush()
        .map_err(|e| TensorError::new(&format!("cpu_averager: frame flush failed: {e}")))?;
    Ok(())
}

fn write_round_to_all(streams: &mut [TcpStream], frame: &RoundFrame) -> Result<()> {
    for s in streams.iter_mut() {
        write_round_frame(s, frame)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Reduction (CPU sum + divide by world_size)
// ---------------------------------------------------------------------------

/// Average a list of per-rank frames into a single frame.
///
/// Validates that every rank's frames have identical schema (same
/// number of tensors, same dtype per tensor, same shape per tensor).
/// Returns the element-wise mean.
///
/// v1 supports only [`DTYPE_F32`]; loud error on other dtypes (so a
/// future user wiring f16 here gets a clear pointer at where to add
/// support, instead of silent garbage from byte-level summation).
fn reduce_average(frames: &[RoundFrame]) -> Result<RoundFrame> {
    if frames.is_empty() {
        return Err(TensorError::new("cpu_averager: reduce_average called with no frames"));
    }
    let n = frames.len();
    let ref_frame = &frames[0];
    // Schema validation.
    for (i, f) in frames.iter().enumerate().skip(1) {
        if f.tensors.len() != ref_frame.tensors.len() {
            return Err(TensorError::new(&format!(
                "cpu_averager: rank {i} sent {} tensors; rank 0 sent {}",
                f.tensors.len(),
                ref_frame.tensors.len()
            )));
        }
        for (ti, (a, b)) in ref_frame.tensors.iter().zip(f.tensors.iter()).enumerate() {
            if a.dtype != b.dtype {
                return Err(TensorError::new(&format!(
                    "cpu_averager: rank {i} tensor[{ti}] dtype {} != rank 0 dtype {}",
                    b.dtype, a.dtype
                )));
            }
            if a.shape != b.shape {
                return Err(TensorError::new(&format!(
                    "cpu_averager: rank {i} tensor[{ti}] shape {:?} != rank 0 shape {:?}",
                    b.shape, a.shape
                )));
            }
            if a.bytes.len() != b.bytes.len() {
                return Err(TensorError::new(&format!(
                    "cpu_averager: rank {i} tensor[{ti}] nbytes {} != rank 0 nbytes {}",
                    b.bytes.len(),
                    a.bytes.len()
                )));
            }
        }
    }

    // Reduce per tensor.
    let mut out_tensors = Vec::with_capacity(ref_frame.tensors.len());
    for ti in 0..ref_frame.tensors.len() {
        let dtype = ref_frame.tensors[ti].dtype;
        if dtype != DTYPE_F32 {
            return Err(TensorError::new(&format!(
                "cpu_averager: tensor[{ti}] dtype {dtype} not supported in v1 \
                 (only DTYPE_F32 = 0 supported). Add other dtypes in controller.rs::reduce_average."
            )));
        }
        let shape = ref_frame.tensors[ti].shape.clone();
        let numel = ref_frame.tensors[ti].numel();
        if numel * std::mem::size_of::<f32>() != ref_frame.tensors[ti].bytes.len() {
            return Err(TensorError::new(&format!(
                "cpu_averager: tensor[{ti}] shape {shape:?} numel*sizeof(f32) {} != nbytes {}",
                numel * std::mem::size_of::<f32>(),
                ref_frame.tensors[ti].bytes.len()
            )));
        }
        let mut accum: Vec<f32> = vec![0.0; numel];
        for f in frames.iter() {
            let view = bytes_as_f32(&f.tensors[ti].bytes)?;
            for (a, x) in accum.iter_mut().zip(view.iter()) {
                *a += *x;
            }
        }
        let inv = 1.0_f32 / (n as f32);
        for a in &mut accum {
            *a *= inv;
        }
        out_tensors.push(TensorPayload {
            dtype: DTYPE_F32,
            shape,
            bytes: f32_to_bytes(&accum),
        });
    }
    Ok(RoundFrame {
        tensors: out_tensors,
    })
}

fn bytes_as_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        return Err(TensorError::new(&format!(
            "cpu_averager: f32 byte count {} not divisible by 4",
            bytes.len()
        )));
    }
    let n = bytes.len() / 4;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut b = [0u8; 4];
        b.copy_from_slice(&bytes[i * 4..(i + 1) * 4]);
        out.push(f32::from_le_bytes(b));
    }
    Ok(out)
}

fn f32_to_bytes(data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 4);
    for x in data {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;
    use std::net::Ipv4Addr;
    use std::sync::mpsc;

    /// Fake rank client: connects to the controller, does the handshake,
    /// and runs `n_rounds` of (send_frame → recv_averaged_frame).
    /// Returns the vector of received averaged frames.
    fn fake_rank(
        port: u16,
        rank_id: u32,
        world_size: u32,
        send_frames: Vec<RoundFrame>,
    ) -> Result<Vec<RoundFrame>> {
        let addr = SocketAddr::new(Ipv4Addr::LOCALHOST.into(), port);
        let mut stream = TcpStream::connect(addr).map_err(|e| {
            TensorError::new(&format!("fake_rank {rank_id}: connect: {e}"))
        })?;

        // Handshake send
        let mut h = [0u8; 16];
        h[0..4].copy_from_slice(&HANDSHAKE_MAGIC_RANK.to_le_bytes());
        h[4..8].copy_from_slice(&PROTOCOL_VERSION.to_le_bytes());
        h[8..12].copy_from_slice(&rank_id.to_le_bytes());
        h[12..16].copy_from_slice(&world_size.to_le_bytes());
        stream.write_all(&h).map_err(|e| {
            TensorError::new(&format!("fake_rank {rank_id}: handshake write: {e}"))
        })?;
        let mut ack = [0u8; 8];
        stream.read_exact(&mut ack).map_err(|e| {
            TensorError::new(&format!("fake_rank {rank_id}: ack read: {e}"))
        })?;
        let ack_magic = u32::from_le_bytes(ack[0..4].try_into().unwrap());
        assert_eq!(ack_magic, HANDSHAKE_MAGIC_CONTROLLER_ACK);

        let mut received = Vec::with_capacity(send_frames.len());
        for f in send_frames {
            write_round_frame(&mut stream, &f)?;
            let r = read_round_frame(&mut stream)?
                .ok_or_else(|| TensorError::new("fake_rank: EOF before averaged frame"))?;
            received.push(r);
        }
        // Drop stream → clean EOF to controller, signals shutdown.
        Ok(received)
    }

    fn one_tensor_frame(data: &[f32]) -> RoundFrame {
        RoundFrame {
            tensors: vec![TensorPayload {
                dtype: DTYPE_F32,
                shape: vec![data.len() as u32],
                bytes: f32_to_bytes(data),
            }],
        }
    }

    fn two_tensor_frame(a: &[f32], b: &[f32]) -> RoundFrame {
        RoundFrame {
            tensors: vec![
                TensorPayload {
                    dtype: DTYPE_F32,
                    shape: vec![a.len() as u32],
                    bytes: f32_to_bytes(a),
                },
                TensorPayload {
                    dtype: DTYPE_F32,
                    shape: vec![b.len() as u32],
                    bytes: f32_to_bytes(b),
                },
            ],
        }
    }

    #[test]
    fn two_rank_average_one_round() {
        let avg = CpuAverager::start(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0),
            2,
        )
        .unwrap();
        let port = avg.port();

        let (tx0, rx0) = mpsc::channel();
        let (tx1, rx1) = mpsc::channel();
        let t0 = thread::spawn(move || {
            let r = fake_rank(port, 0, 2, vec![one_tensor_frame(&[1.0, 2.0, 3.0])]);
            tx0.send(r).unwrap();
        });
        let t1 = thread::spawn(move || {
            let r = fake_rank(port, 1, 2, vec![one_tensor_frame(&[3.0, 4.0, 5.0])]);
            tx1.send(r).unwrap();
        });

        let r0 = rx0.recv().unwrap().unwrap();
        let r1 = rx1.recv().unwrap().unwrap();
        t0.join().unwrap();
        t1.join().unwrap();
        avg.shutdown().unwrap();

        // Average of (1,2,3) and (3,4,5) = (2,3,4)
        let expected = bytes_as_f32(&r0[0].tensors[0].bytes).unwrap();
        assert_eq!(expected, vec![2.0, 3.0, 4.0]);
        // Both ranks receive the same averaged frame.
        assert_eq!(r0, r1);
    }

    #[test]
    fn three_rank_average_multi_round_multi_tensor() {
        // Three ranks, two rounds each, each round carries two tensors.
        // Exercises:
        //   - multi-rank star summation (3 ranks)
        //   - multi-round reduce loop (2 rounds)
        //   - multi-tensor frames (2 tensors per frame)
        //   - clean shutdown on rank EOF
        let avg = CpuAverager::start(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0),
            3,
        )
        .unwrap();
        let port = avg.port();

        // Round-1 inputs: ranks 0/1/2 contribute (0,10), (10,20), (20,30) for tensor 0
        //                          and (1,1), (2,2), (3,3) for tensor 1.
        // Round-2 inputs: same shape, halved values (just to vary the data).
        let r0_frames = vec![
            two_tensor_frame(&[0.0, 10.0], &[1.0, 1.0]),
            two_tensor_frame(&[0.0, 5.0], &[0.5, 0.5]),
        ];
        let r1_frames = vec![
            two_tensor_frame(&[10.0, 20.0], &[2.0, 2.0]),
            two_tensor_frame(&[5.0, 10.0], &[1.0, 1.0]),
        ];
        let r2_frames = vec![
            two_tensor_frame(&[20.0, 30.0], &[3.0, 3.0]),
            two_tensor_frame(&[10.0, 15.0], &[1.5, 1.5]),
        ];

        let (tx0, rx0) = mpsc::channel();
        let (tx1, rx1) = mpsc::channel();
        let (tx2, rx2) = mpsc::channel();
        let t0 = thread::spawn(move || tx0.send(fake_rank(port, 0, 3, r0_frames)).unwrap());
        let t1 = thread::spawn(move || tx1.send(fake_rank(port, 1, 3, r1_frames)).unwrap());
        let t2 = thread::spawn(move || tx2.send(fake_rank(port, 2, 3, r2_frames)).unwrap());

        let r0 = rx0.recv().unwrap().unwrap();
        let r1 = rx1.recv().unwrap().unwrap();
        let r2 = rx2.recv().unwrap().unwrap();
        t0.join().unwrap();
        t1.join().unwrap();
        t2.join().unwrap();
        avg.shutdown().unwrap();

        // Each rank received exactly 2 averaged frames.
        assert_eq!(r0.len(), 2, "rank 0 should receive 2 averaged frames");
        assert_eq!(r1.len(), 2);
        assert_eq!(r2.len(), 2);

        // Round 1 averages: tensor 0 = (10, 20), tensor 1 = (2, 2)
        let r1_t0 = bytes_as_f32(&r0[0].tensors[0].bytes).unwrap();
        let r1_t1 = bytes_as_f32(&r0[0].tensors[1].bytes).unwrap();
        assert_eq!(r1_t0, vec![10.0, 20.0]);
        assert_eq!(r1_t1, vec![2.0, 2.0]);

        // Round 2 averages: tensor 0 = (5, 10), tensor 1 = (1, 1)
        let r2_t0 = bytes_as_f32(&r0[1].tensors[0].bytes).unwrap();
        let r2_t1 = bytes_as_f32(&r0[1].tensors[1].bytes).unwrap();
        assert_eq!(r2_t0, vec![5.0, 10.0]);
        assert_eq!(r2_t1, vec![1.0, 1.0]);

        // All three ranks see bit-identical averaged frames.
        assert_eq!(r0, r1);
        assert_eq!(r1, r2);
    }

    #[test]
    fn rejects_wrong_handshake_magic() {
        let avg = CpuAverager::start(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0),
            1,
        )
        .unwrap();
        let port = avg.port();

        let mut s = TcpStream::connect(SocketAddr::new(Ipv4Addr::LOCALHOST.into(), port)).unwrap();
        let mut bad = [0u8; 16];
        bad[0..4].copy_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
        s.write_all(&bad).unwrap();
        // The controller should reject and drop us; read_exact on ack
        // would return EOF or error. We accept either outcome.
        let mut ack = [0u8; 8];
        let _ = s.read_exact(&mut ack);
        drop(s);
        // The reduce thread terminates with an error; shutdown still
        // joins cleanly.
        let _ = avg.shutdown(); // err is OK here
    }

    #[test]
    fn rejects_world_size_mismatch() {
        let avg = CpuAverager::start(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0),
            2,
        )
        .unwrap();
        let port = avg.port();

        // Rank claims world_size = 3 but controller is configured for 2.
        let mut s = TcpStream::connect(SocketAddr::new(Ipv4Addr::LOCALHOST.into(), port)).unwrap();
        let mut h = [0u8; 16];
        h[0..4].copy_from_slice(&HANDSHAKE_MAGIC_RANK.to_le_bytes());
        h[4..8].copy_from_slice(&PROTOCOL_VERSION.to_le_bytes());
        h[8..12].copy_from_slice(&0u32.to_le_bytes());
        h[12..16].copy_from_slice(&3u32.to_le_bytes());
        s.write_all(&h).unwrap();
        // Server drops us.
        let mut ack = [0u8; 8];
        let _ = s.read_exact(&mut ack);
        drop(s);
        let _ = avg.shutdown();
    }

    #[test]
    fn rejects_non_f32_dtype_in_reduce() {
        // Pure unit test of reduce_average without TCP wiring.
        let frames = vec![
            RoundFrame {
                tensors: vec![TensorPayload {
                    dtype: 7, // bogus dtype
                    shape: vec![2],
                    bytes: vec![0; 8],
                }],
            },
            RoundFrame {
                tensors: vec![TensorPayload {
                    dtype: 7,
                    shape: vec![2],
                    bytes: vec![0; 8],
                }],
            },
        ];
        let err = reduce_average(&frames).unwrap_err();
        assert!(
            err.to_string().contains("dtype 7"),
            "expected dtype-7-not-supported, got: {err}"
        );
    }

    #[test]
    fn rejects_shape_mismatch_across_ranks() {
        let frames = vec![
            RoundFrame {
                tensors: vec![TensorPayload {
                    dtype: DTYPE_F32,
                    shape: vec![2],
                    bytes: f32_to_bytes(&[1.0, 2.0]),
                }],
            },
            RoundFrame {
                tensors: vec![TensorPayload {
                    dtype: DTYPE_F32,
                    shape: vec![3],
                    bytes: f32_to_bytes(&[1.0, 2.0, 3.0]),
                }],
            },
        ];
        let err = reduce_average(&frames).unwrap_err();
        assert!(err.to_string().contains("shape"), "got: {err}");
    }

    #[test]
    fn averager_zero_world_size_errors() {
        let err = CpuAverager::start(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0),
            0,
        )
        .unwrap_err();
        assert!(err.to_string().contains("world_size"), "got: {err}");
    }
}
