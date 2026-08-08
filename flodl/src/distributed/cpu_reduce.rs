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
    self, DTYPE_BF16, DTYPE_F32, HANDSHAKE_MAGIC_CONTROLLER_ACK, HANDSHAKE_MAGIC_RANK,
    PROTOCOL_VERSION, RoundFrame, RoundKind, TensorPayload,
};
use crate::distributed::wire::SessionSalt;
use crate::tensor::{DType, Device, Result, Tensor, TensorError, TensorOptions};

/// Whether this host can afford a page-locked model-copy staging
/// without pushing itself into swap.
///
/// HISTORICAL for the consensus decode it was written to gate: the
/// decode now lands in the snapshot staging itself
/// ([`CpuReduceClient::set_decode_into_request`], zero marginal locked
/// memory), so nothing consults this helper on that path anymore. Kept
/// (with its test) because the reasoning outlives the call site: the
/// snapshot staging is the REMAINING locked resident, and a future
/// RAM-tight host may want the same MemAvailable-anchored affordability
/// question asked about it.
///
/// `staging_bytes` is the f32 model wire size (the staging is always
/// f32 whatever the wire dtype), `local_ranks` how many ranks share
/// this host (each locks its own staging; the per-rank MemAvailable
/// reads race each other at formation, so each rank must budget for
/// the whole host), `mem_available_bytes` the kernel's `MemAvailable`.
///
/// The headroom factor counts real residents per rank, not magic —
/// and it must do so against `MemAvailable` AS READ AT RANK START,
/// before the run's own working set has allocated (the reads race the
/// formation allocations, so the honest anchor is the pre-run value).
/// Per rank: the decode staging itself locks 1× — and because locked
/// pages are unswappable, the kernel evicts OTHER pages under
/// pressure, so the gate must also cover what the reduce path
/// re-touches EVERY window and cannot afford to have evicted: the
/// pinned snapshot staging (1×), the pre-sync divergence scratch (1×),
/// and the streamed wire transient (1×); the remaining 2× stands in
/// for the per-rank runtime overhead MemAvailable must also fund
/// (CUDA host context, allocator arenas, data staging — ~1.5GB at the
/// model sizes where this gate can trip at all; at small stagings the
/// stand-in undercounts it, but small stagings pass every gate).
///
/// Rig calibration (2026-07-29, olmo 190M ⇒ 727MiB staging, two ranks
/// on a 9.4GB VM reading ~7.3GB available at rank start): ungated
/// pinned decode pushed 2GB to swap (vs 0.8GB baseline) and the
/// per-window scratch thrash cost +6-8% of epoch wall, entirely inside
/// the reduce windows — compute, data starvation, and VRAM stayed
/// arm-identical. A first cut of this gate at factor 4 (need 5.8GB)
/// did NOT refuse that host — because it was sized against the
/// mid-formation availability (~5.8GB), which no rank ever observes —
/// and the rig re-run reproduced the regression (B5, epoch 317s vs
/// 290-293s baseline). Factor 6 (need 8.5GB > 7.3GB) refuses it at
/// the anchor the ranks actually read.
#[allow(dead_code)] // kept deliberately (see HISTORICAL above); pinned by its test
pub fn pinned_decode_affordable(
    staging_bytes: u64,
    local_ranks: usize,
    mem_available_bytes: u64,
) -> bool {
    const HEADROOM_FACTOR: u64 = 6;
    let need = staging_bytes
        .saturating_mul(local_ranks.max(1) as u64)
        .saturating_mul(HEADROOM_FACTOR);
    mem_available_bytes > need
}

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
    /// Wire dtype for [`RoundKind::Model`] frames (params / buffers) —
    /// [`DTYPE_F32`] (default) or [`DTYPE_BF16`] via
    /// [`Self::set_bf16_wire`]. [`RoundKind::Control`] frames (count
    /// gathers, formation broadcasts) ALWAYS ride f32: bookkeeping must
    /// be exact (bf16 cannot represent integers above 256), and the
    /// volume lives in the model frames anyway. Must agree across the
    /// cohort — a dtype mix surfaces as a loud schema error at the first
    /// fold.
    model_wire_dtype: u8,
    /// RAM-neutral decode mode (see [`Self::set_decode_into_request`]):
    /// armed Model replies decode into the REQUEST tensors themselves —
    /// the caller's pinned snapshot staging. Zero marginal locked
    /// memory; the reply schema mirrors the sent frame by protocol, so
    /// the destinations always fit. Pinned staging is what turns the
    /// consumer's `copy_(non_blocking)` writeback into a true
    /// `cudaMemcpyAsync` (from a pageable source it silently degrades
    /// to a synchronous bounce copy).
    decode_into_request: bool,
    /// One-shot arm for `decode_into_request`: shallow clones of the
    /// request tensors the NEXT reply decodes into. Consumed by
    /// `read_reduced_tensors`; error paths leave the client un-armed.
    armed_decode_dsts: Option<Vec<Tensor>>,
    /// Once-per-run notice latch for an armed destination the decode
    /// refused (non-CPU / non-f32 — the degenerate snapshot-fallback
    /// class); values still land via fresh allocs.
    decode_dst_fallback_logged: bool,
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
            return Err(TensorError::new("cpu_reduce: world_size must be > 0"));
        }
        if rank_id >= world_size {
            return Err(TensorError::new(&format!(
                "cpu_reduce: rank_id {rank_id} must be < world_size {world_size}"
            )));
        }
        // Ranks dial their host-local relay's loopback. The relay process
        // may bind a beat after the rank starts (launcher spawns both),
        // so retry briefly rather than fail on the first refusal.
        let stream = crate::distributed::wire::connect_with_retry(controller_addr, "cpu_reduce")?;
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
            model_wire_dtype: DTYPE_F32,
            decode_into_request: false,
            armed_decode_dsts: None,
            decode_dst_fallback_logged: false,
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
            .map_err(|e| TensorError::new(&format!("cpu_reduce: set reduce read deadline: {e}")))?;
        Ok(client)
    }

    fn send_handshake(&mut self) -> Result<()> {
        let mut buf = [0u8; 16];
        buf[0..4].copy_from_slice(&HANDSHAKE_MAGIC_RANK.to_le_bytes());
        buf[4..8].copy_from_slice(&PROTOCOL_VERSION.to_le_bytes());
        buf[8..12].copy_from_slice(&self.rank_id.to_le_bytes());
        buf[12..16].copy_from_slice(&self.world_size.to_le_bytes());
        self.stream
            .write_all(&buf)
            .map_err(|e| TensorError::new(&format!("cpu_reduce: handshake write failed: {e}")))?;
        self.stream
            .flush()
            .map_err(|e| TensorError::new(&format!("cpu_reduce: handshake flush failed: {e}")))?;
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

    /// Ship [`RoundKind::Model`] frames (params / buffers) as bf16
    /// instead of f32, halving the reduce payload both directions. See
    /// [`ElCheConfig::bf16_wire`](crate::distributed::ElCheConfig::bf16_wire)
    /// for the semantics (averaging still accumulates in f32; the cast
    /// lives at the wire boundary). [`RoundKind::Control`] traffic stays
    /// f32 regardless. Must be set identically on every rank.
    pub fn set_bf16_wire(&mut self, on: bool) {
        self.model_wire_dtype = if on { DTYPE_BF16 } else { DTYPE_F32 };
    }

    /// Enable the RAM-neutral pinned decode: armed Model replies decode
    /// into the REQUEST tensors themselves (see
    /// [`Self::arm_decode_into`]) — the caller's pinned snapshot
    /// staging. The staging's bytes are dead by decode time (the
    /// streamed encode consumed them before the blocking reply read
    /// began), so the consumer's `copy_(non_blocking)` writeback
    /// becomes a true async H2D at ZERO marginal locked memory — no
    /// RAM-affordability gate needed (the retired slot-staging
    /// predecessor locked a second model copy per rank, which thrashed
    /// RAM-tight hosts into swap).
    ///
    /// SINGLE-CONSUMER CONTRACT: the returned tensors are shallow
    /// clones of the destinations — the caller's next window OVERWRITES
    /// them in place (its own D2H snapshot, then the next decode).
    /// Callers must fully consume a round's tensors (including retiring
    /// any in-flight H2D reading them) before the next round begins.
    /// The barrier-paced policies (`Sync` / `Cadence`) satisfy this
    /// structurally: the worker cannot be asked for snapshot N+1 before
    /// applying Update N, and `snapshot_params`' entry fences host-sync
    /// the comm stream — retiring the writeback — before the bridge can
    /// start the round that would overwrite the staging. `Async` does
    /// NOT (its control channel has two producers with only
    /// per-producer FIFO, so `RequestParams(N+1)` can interleave ahead
    /// of `Update(N)`, leaving two Updates live at once); keep this off
    /// there until a double-buffered variant exists.
    pub fn set_decode_into_request(&mut self, on: bool) {
        self.decode_into_request = on;
    }

    /// Arm the NEXT reply to decode into `dsts` — the tensors the
    /// caller is about to stream up, whose reply mirrors their schema
    /// by protocol. One-shot: consumed by the next reply read; error
    /// paths leave the client un-armed. Holds shallow clones, so the
    /// decode writes through to the caller's staging storage. No-op
    /// unless [`Self::set_decode_into_request`] enabled the mode
    /// (un-armed reads — count gathers, formation broadcasts — keep
    /// the fresh-alloc decode either way).
    ///
    /// The reply's realized-work mass guards the destinations: a
    /// zero-mass round's payloads are meaningless zeros the caller
    /// keep-locals over, and decoding them into `dsts` would CLOBBER
    /// the very tensors keep-local returns — the weight rides the wire
    /// ahead of the payloads, and an unrealized round decodes fresh,
    /// leaving `dsts` untouched (see `read_reduced_tensors`).
    pub fn arm_decode_into(&mut self, dsts: &[Tensor]) {
        if self.decode_into_request {
            self.armed_decode_dsts = Some(dsts.to_vec());
        }
    }

    /// Send this rank's frame for the current round and receive the
    /// averaged frame back.
    ///
    /// Blocks until the controller has collected frames from every
    /// rank, summed them, and scattered the average back. Loud error on
    /// any wire-level failure (truncated read, EOF before the averaged
    /// frame, magic mismatch).
    ///
    /// Consumes `frame` and DROPS IT right after the write, before the
    /// blocking read: model frames are hundreds of MB, and every rank
    /// sits at this barrier simultaneously, so holding the sent payload
    /// through the reply read doubles the cohort's sync-window RAM peak
    /// for no reason (measured as part of the first-sync OOM spike on
    /// the 8GB two-rank VM).
    ///
    /// The returned frame has the same tensor count, dtypes, and shapes
    /// as the input frame; only the tensor bytes change.
    pub fn all_reduce(&mut self, frame: RoundFrame) -> Result<RoundFrame> {
        write_framed_round(&mut self.stream, &frame, &self.salt)?;
        drop(frame);
        match read_framed_round(&mut self.stream, &self.salt)? {
            Some(f) => Ok(f),
            None => Err(TensorError::new(
                "cpu_reduce: controller closed connection before sending averaged \
                 frame back (controller crashed, or another rank disconnected and \
                 triggered cluster-wide shutdown mid-round)",
            )),
        }
    }

    /// Weighted frame-level reduce: ship `tensors` tagged with `kind` +
    /// the sender's realized-work `weight`, and return the reduced
    /// tensors plus the round's summed accepted mass. Equivalent to
    /// [`Self::all_reduce_scaled`] with `scale = 1.0` (tensors ship
    /// verbatim).
    ///
    /// [`RoundKind::Model`]: the controller returns the consensus (sum
    /// divided ONCE by the mass of exactly the frames it accepted). A
    /// returned mass of `0.0` means nothing was realized this round and
    /// the tensors are meaningless zeros — callers must keep local state.
    ///
    /// [`RoundKind::Control`]: pure element-wise sum (gathers /
    /// broadcasts build on it); the returned mass is informational.
    ///
    /// `Model` frames ride [`Self::set_bf16_wire`]'s dtype; `Control`
    /// frames always ride f32. Caller is responsible for moving reduced
    /// tensors back to GPU if needed.
    pub fn all_reduce_weighted(
        &mut self,
        tensors: &[&Tensor],
        kind: RoundKind,
        weight: f64,
    ) -> Result<(Vec<Tensor>, f64)> {
        self.all_reduce_scaled(tensors, 1.0, kind, weight)
    }

    /// [`Self::all_reduce_weighted`] with the sender-side pre-scale
    /// (`scale · T` element-wise) FUSED into the wire encode, streamed
    /// straight to the socket one tensor at a time.
    ///
    /// This is the realized-work send path (`scale` = the same γ-mass
    /// the frame's `weight` carries): fusing the scale means no
    /// model-sized scratch copy (`mul_scalar` before the A4b rework),
    /// and streaming means neither the frame's payload bytes nor its
    /// serialized body ever coexist on the sender — the peak transient
    /// is ONE tensor's wire bytes. The frame length is computed exactly
    /// from shapes + dtype and committed as the length prefix before any
    /// payload is produced (see
    /// [`round_frame_wire_len`](crate::distributed::controller) usage).
    ///
    /// Reading the input tensors at stream time is the caller's single
    /// consumption of its snapshot staging for the window — the reduce
    /// completes (blocking read) before any next snapshot can overwrite
    /// the buffers, so the single-consumer pinned-staging contract
    /// holds unchanged.
    ///
    /// Precision: when a tensor's dtype already matches the wire dtype,
    /// the scale runs at byte level (decode → f32 multiply →
    /// re-encode; ONE round-to-nearest-even stage for bf16 — one fewer
    /// than the old scale-then-serialize path). On a dtype mismatch
    /// (e.g. the f32 passthrough-staging fallback under bf16 wire) the
    /// scale runs on-tensor first so the wire cast stays the single
    /// rounding stage.
    ///
    /// `scale == 0.0` — the idle rank's contribution (`γ-mass` 0 for 0
    /// optimizer steps) — ships as an ELIDED frame: the payloads are
    /// structurally zeros, so only the schema crosses the wire
    /// (`nbytes = 0`, see
    /// [`TensorPayload::bytes`](crate::distributed::controller::TensorPayload::bytes))
    /// and the tensors are never read, scaled, or MAC'd. The fold
    /// treats elided as zeros, so the round is unchanged — except that
    /// a NaN-poisoned idle rank can no longer forward `0.0 × NaN = NaN`
    /// into the consensus, which is strictly better isolation.
    pub fn all_reduce_scaled(
        &mut self,
        tensors: &[&Tensor],
        scale: f64,
        kind: RoundKind,
        weight: f64,
    ) -> Result<(Vec<Tensor>, f64)> {
        // Idle contribution (see the doc above): every payload is
        // structurally `0.0 · T` — declare it instead of encoding it.
        if scale == 0.0 {
            return self.stream_zeros_frame(tensors, kind, weight);
        }
        let t0 = Instant::now();
        let wire_dtype = self.wire_dtype_for(kind);
        let wire_tensor_dtype = wire_tensor_dtype(wire_dtype)?;
        let elem = controller::payload_element_size(wire_dtype)? as u64;

        // Wire metadata first: dtype support (loud), shapes as u32 (loud
        // on overflow) + exact payload byte counts — enough to commit
        // the frame length before producing a single payload byte.
        for (i, t) in tensors.iter().enumerate() {
            if !matches!(t.dtype(), DType::Float32 | DType::BFloat16) {
                return Err(TensorError::new(&format!(
                    "cpu_reduce: tensor[{i}] dtype {:?} not supported (only Float32 \
                     / BFloat16). Extend cpu_reduce.rs and the round_frame.rs codec \
                     helpers together to add support.",
                    t.dtype()
                )));
            }
        }
        let shapes: Vec<Vec<u32>> = tensors
            .iter()
            .enumerate()
            .map(|(i, t)| wire_shape(i, t))
            .collect::<Result<_>>()?;
        let parts: Vec<controller::PayloadPart<'_>> = tensors
            .iter()
            .zip(shapes.iter())
            .map(|(t, shape)| controller::PayloadPart {
                dtype: wire_dtype,
                shape,
                nbytes: t.numel() as u64 * elem,
            })
            .collect();
        let sent_bytes: u64 = parts.iter().map(|p| p.nbytes).sum();

        crate::distributed::relay::mux::write_len_prefix(
            &mut self.stream,
            controller::round_frame_wire_len(&parts),
        )?;
        let prof = self.prof_enabled;
        let mut produce_ns: u128 = 0;
        controller::write_round_frame_streamed(
            &mut self.stream,
            kind,
            weight,
            &parts,
            &self.salt,
            &mut |ti, tee| {
                let tp = Instant::now();
                let t = tensors[ti];
                let bytes = if t.dtype() == wire_tensor_dtype {
                    let mut b = t.to_blob()?;
                    if scale != 1.0 {
                        controller::scale_payload_bytes(&mut b, wire_dtype, scale as f32)?;
                    }
                    b
                } else {
                    // Cast on-device so the transient is wire-sized; the
                    // client-side cast is what keeps the frame schema
                    // uniform whatever staging path produced the tensor
                    // (see `tensors_to_round_frame`).
                    let src = if scale != 1.0 {
                        t.mul_scalar(scale)?
                    } else {
                        t.clone()
                    };
                    src.to_dtype(wire_tensor_dtype)?.to_blob()?
                };
                if prof {
                    produce_ns += tp.elapsed().as_nanos();
                }
                tee.write_all(&bytes)
                    .map_err(|e| TensorError::new(&e.to_string()))
            },
        )?;

        let (out, weight, decode_ns) = self.read_reduced_tensors()?;
        if prof {
            let t2 = Instant::now();
            self.prof_serialize_ns += produce_ns;
            self.prof_wire_ns += (t2 - t0)
                .as_nanos()
                .saturating_sub(produce_ns)
                .saturating_sub(decode_ns);
            self.prof_deserialize_ns += decode_ns;
            self.prof_bytes += sent_bytes;
            self.prof_count += 1;
        }
        Ok((out, weight))
    }

    /// Read the reduced reply as tensors, DRAINING: each payload is
    /// decoded into its f32 CPU tensor the moment it comes off the
    /// stream and its wire bytes are freed before the next payload is
    /// read — neither the len-framed blob (which no longer exists, see
    /// [`read_framed_round`]) nor the frame's payloads ever coexist
    /// with the decoded output. Peak reply-side transient: the decoded
    /// tensors plus ONE payload's bytes, instead of blob + payloads +
    /// tensors (three model-sized residents before A4b).
    ///
    /// Tensor construction inside the sink is inert buffering under the
    /// MAC-before-use contract of
    /// [`read_round_frame_streamed`](crate::distributed::controller::read_round_frame_streamed):
    /// bytes only get copied/upcast into tensors that are dropped
    /// unadopted if the footer fails to authenticate — nothing acts on
    /// the values until the frame verifies and this method returns.
    ///
    /// Returns `(tensors, round mass, decode-time ns)` — the decode
    /// time (accumulated inside the sink, `-vvv` gated) lets the caller
    /// split its wire/deserialize profile even though the two phases
    /// now interleave.
    ///
    /// When request destinations are armed (see
    /// [`Self::arm_decode_into`]), each payload decodes into its
    /// destination tensor — UNLESS the frame's realized-work weight is
    /// zero: a zero-mass reply's payloads are meaningless zeros the
    /// caller keep-locals over, and the destinations are exactly the
    /// tensors keep-local returns, so an unrealized round decodes fresh
    /// and leaves them untouched. The weight is read off the frame
    /// header AHEAD of the payloads and is unauthenticated until the
    /// footer verifies — steering the decode DESTINATION with it is
    /// inert buffering under the reader's MAC-before-use contract (a
    /// forged weight at worst wastes a fresh alloc, or clobbers staging
    /// on a frame that then fails MAC and tears the round down; no
    /// snapshot path ever reads stale staging content).
    fn read_reduced_tensors(&mut self) -> Result<(Vec<Tensor>, f64, u128)> {
        let Some(len) = crate::distributed::relay::mux::read_len_prefix(&mut self.stream)? else {
            return Err(TensorError::new(
                "cpu_reduce: controller closed connection before sending averaged \
                 frame back (controller crashed, or another rank disconnected and \
                 triggered cluster-wide shutdown mid-round)",
            ));
        };
        // One-shot: an error path below leaves the client un-armed, so a
        // stale arm can never bleed into an unrelated later reply.
        let armed_dsts = self.armed_decode_dsts.take();
        let prof = self.prof_enabled;
        let mut decode_ns: u128 = 0;
        let mut out: Vec<Tensor> = Vec::new();
        let mut dst_fallback: Option<String> = None;
        // Realized-work mass off the frame header, set before the first
        // payload reaches the sink (Cell: the header and payload
        // closures both capture it). NAN until the header parses, and
        // `is_realized(NAN)` is false — fail-safe to the fresh path.
        let hdr_weight = std::cell::Cell::new(f64::NAN);
        let mut on_header = |_k: controller::RoundKind, w: f64| hdr_weight.set(w);
        let mut body = (&mut self.stream).take(len as u64);
        let hdr = controller::read_round_frame_streamed(
            &mut body,
            &self.salt,
            Some(&mut on_header),
            &mut |i, payload| {
                let tp = Instant::now();
                let realized = crate::distributed::realized_work::is_realized(hdr_weight.get());
                let t = match armed_dsts.as_deref() {
                    Some(dsts) if realized => {
                        decode_into_dst(i, &payload, dsts, &mut dst_fallback)?
                    }
                    _ => payload_to_cpu_tensor(i, &payload)?,
                };
                out.push(t);
                if prof {
                    decode_ns += tp.elapsed().as_nanos();
                }
                Ok(())
                // `payload` drops here — per-payload draining.
            },
        )?;
        let leftover = body.limit();
        finish_framed_body(hdr.is_some(), leftover)?;
        if let Some(msg) = dst_fallback
            && !self.decode_dst_fallback_logged
        {
            self.decode_dst_fallback_logged = true;
            eprintln!(
                "flodl cpu_reduce: rank {} armed decode destination refused \
                 ({msg}); decoded into fresh allocs instead (staging reuse and \
                 async H2D writeback lost for those payloads)",
                self.rank_id,
            );
        }
        let (_kind, weight) = hdr.expect("finish_framed_body verified Some");
        Ok((out, weight, decode_ns))
    }

    /// Wire dtype for a round kind: `Model` rides the configured dtype,
    /// `Control` is always f32 (see [`Self::set_bf16_wire`]).
    fn wire_dtype_for(&self, kind: RoundKind) -> u8 {
        match kind {
            RoundKind::Model => self.model_wire_dtype,
            RoundKind::Control => DTYPE_F32,
        }
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
        Ok(self.all_reduce_weighted(tensors, RoundKind::Model, 1.0)?.0)
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
        let mbps = if wire_s > 0.0 {
            (mb * 2.0) / wire_s
        } else {
            0.0
        };
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
    /// v1 supports root=0 only. Rides a `Control` frame, so the
    /// broadcast is byte-exact f32 regardless of the model wire dtype
    /// (every rank must start from IDENTICAL state). All ranks must
    /// call concurrently.
    pub fn broadcast_from_root(&mut self, tensors: &[&Tensor], root: u32) -> Result<Vec<Tensor>> {
        if root >= self.world_size {
            return Err(TensorError::new(&format!(
                "cpu_reduce: broadcast root {root} >= world_size {}",
                self.world_size,
            )));
        }
        // Root ships its live tensors directly — the streamed encode reads
        // them without mutation, so the zeros_like + copy_ scratch this
        // path used to build (a full model copy at formation) bought
        // nothing. Non-root ranks contribute all-zeros as an ELIDED frame
        // (`stream_zeros_frame`): schema only, `nbytes = 0` per payload —
        // no zeros model materialized (at 190M params that scratch was
        // ~762MB per rank, landing exactly on the formation-time RAM
        // peak) AND no zero bytes on the wire (the same 762MB per rank
        // used to cross the uplink and be HMAC'd on both ends). The fold
        // reads elided as zeros, so the summed broadcast is fold-identical
        // to shipping a zeros model (pinned by
        // `an_elided_frame_folds_identically_to_a_zeros_model`).
        if self.rank_id == root {
            Ok(self
                .all_reduce_weighted(tensors, RoundKind::Control, 0.0)?
                .0)
        } else {
            Ok(self.stream_zeros_frame(tensors, RoundKind::Control, 0.0)?.0)
        }
    }

    /// Ship an ELIDED frame — schema (shapes, dtype, kind, weight) taken
    /// from `tensors`, every payload declared `nbytes = 0` (the wire
    /// zero-elision: all zeros of the declared schema, see
    /// [`TensorPayload::bytes`]) — and return the reduced reply.
    /// Fold-identical to [`Self::all_reduce_weighted`] over `zeros_like`
    /// copies of `tensors`, but the frame is schema-sized: no zero bytes
    /// cross the wire and neither end MACs or folds a model of zeros.
    ///
    /// [`TensorPayload::bytes`]: crate::distributed::controller::TensorPayload::bytes
    fn stream_zeros_frame(
        &mut self,
        tensors: &[&Tensor],
        kind: RoundKind,
        weight: f64,
    ) -> Result<(Vec<Tensor>, f64)> {
        let t0 = Instant::now();
        let wire_dtype = self.wire_dtype_for(kind);
        let shapes: Vec<Vec<u32>> = tensors
            .iter()
            .enumerate()
            .map(|(i, t)| wire_shape(i, t))
            .collect::<Result<_>>()?;
        let parts: Vec<controller::PayloadPart<'_>> = shapes
            .iter()
            .map(|shape| controller::PayloadPart {
                dtype: wire_dtype,
                shape,
                nbytes: 0,
            })
            .collect();
        let sent_bytes: u64 = 0;

        crate::distributed::relay::mux::write_len_prefix(
            &mut self.stream,
            controller::round_frame_wire_len(&parts),
        )?;
        controller::write_round_frame_streamed(
            &mut self.stream,
            kind,
            weight,
            &parts,
            &self.salt,
            // Elided payloads carry no bytes — the emitter has nothing
            // to write (the declared-nbytes check holds at 0).
            &mut |_ti, _tee| Ok(()),
        )?;

        let (out, weight, decode_ns) = self.read_reduced_tensors()?;
        if self.prof_enabled {
            self.prof_wire_ns += t0.elapsed().as_nanos().saturating_sub(decode_ns);
            self.prof_deserialize_ns += decode_ns;
            self.prof_bytes += sent_bytes;
            self.prof_count += 1;
        }
        Ok((out, weight))
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
    /// — same semantics, CPU-routed. Rides a `Control` frame, so it
    /// carries f32 regardless of the model wire dtype; precision is preserved at the
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
        let tensor = Tensor::from_f32(&vals, &[world_size as i64], Device::CPU)?;
        // Bookkeeping reduce: tag it `Control` so the consensus-checkpoint
        // forge never mistakes this count vector for a slice of the model
        // (and so it rides f32 regardless of the model wire dtype — bf16
        // cannot represent batch counts above 256 exactly).
        let mut frame = tensors_to_round_frame(&[&tensor], DTYPE_F32)?;
        frame.kind = RoundKind::Control;
        let averaged = self.all_reduce(frame)?;
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
// non-blocking behavior comes from the coordinator's asynchronous
// CPU-averaging cadence (the `CpuAvgPhase` Idle/Pending window), not a
// rank-side split client) — so it was removed rather
// than shipped dead. Recover from git history if a rank-side async
// client is ever wanted.

// ---------------------------------------------------------------------------
// Length-framed RoundFrame helpers (rank ↔ relay loopback leg)
// ---------------------------------------------------------------------------

/// Write `frame` (with its HMAC footer) length-delimited to `stream`,
/// STREAMED: the exact frame length goes out as the prefix, then the
/// body is written straight from the frame's payloads — the serialized
/// body never exists as a buffer (before A4b this path materialized the
/// body TWICE: a serialize `Vec` plus `write_len_framed`'s atomic
/// `[len‖body]` copy — two model-sized transients per model frame). The
/// rank talks to its host-local relay, which forwards the opaque blob
/// upstream untouched; the length prefix lets the relay frame it without
/// parsing, and its reader commits through read timeouts once the first
/// prefix byte lands (see [`mux::write_len_prefix`]), so the split
/// writes cannot desync it. See [`crate::distributed::relay::mux`].
///
/// [`mux::write_len_prefix`]: crate::distributed::relay::mux::write_len_prefix
fn write_framed_round<W: Write>(
    stream: &mut W,
    frame: &RoundFrame,
    salt: &SessionSalt,
) -> Result<()> {
    let parts: Vec<controller::PayloadPart<'_>> = frame
        .tensors
        .iter()
        .map(|t| controller::PayloadPart {
            dtype: t.dtype,
            shape: &t.shape,
            nbytes: t.bytes.len() as u64,
        })
        .collect();
    crate::distributed::relay::mux::write_len_prefix(
        stream,
        controller::round_frame_wire_len(&parts),
    )?;
    controller::write_round_frame(stream, frame, salt)
}

/// Read a length-delimited [`RoundFrame`] and parse it STRAIGHT OFF the
/// stream — the len-framed body never exists as a buffer (before A4b it
/// was read whole, then parsed: blob + payloads coexisting, a
/// model-sized extra on every reply). `Ok(None)` on clean EOF
/// (relay/controller closed the connection).
///
/// The parse is bounded by [`Read::take`]`(len)`: a frame that needs
/// more bytes than the prefix declared hits a loud mid-frame EOF error
/// instead of eating into the next frame, and a frame that consumed
/// fewer is caught by the leftover check — either way a prefix/body
/// disagreement is a named error, never a silently desynced stream.
fn read_framed_round<R: Read>(stream: &mut R, salt: &SessionSalt) -> Result<Option<RoundFrame>> {
    let Some(len) = crate::distributed::relay::mux::read_len_prefix(stream)? else {
        return Ok(None);
    };
    let mut body = stream.take(len as u64);
    let frame = controller::read_round_frame(&mut body, salt)?;
    finish_framed_body(frame.is_some(), body.limit())?;
    Ok(frame)
}

/// Shared tail of the streamed framed-round readers: a `None` frame
/// inside a declared body means the stream ended mid-frame (the prefix
/// promised bytes that never came), and leftover take-budget means the
/// frame was shorter than its prefix — both are loud protocol errors.
fn finish_framed_body(got_frame: bool, leftover: u64) -> Result<()> {
    if !got_frame {
        return Err(TensorError::new(
            "cpu_reduce: stream ended inside a len-framed RoundFrame body \
             (peer died mid-frame, or a zero-length prefix)",
        ));
    }
    if leftover != 0 {
        return Err(TensorError::new(&format!(
            "cpu_reduce: RoundFrame consumed {leftover} bytes fewer than its \
             length prefix declared; sender/reader wire drift — stream is \
             desynced",
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tensor ↔ RoundFrame conversion
// ---------------------------------------------------------------------------

/// Build a [`RoundFrame`] from a slice of tensors, encoding every
/// payload in `wire_dtype` ([`DTYPE_F32`] or [`DTYPE_BF16`]).
///
/// Each tensor is moved to CPU via [`Tensor::to_blob`] (transparently
/// handles GPU→CPU transfer) and serialized as raw native-byte-order
/// bytes. Shape is captured as `Vec<u32>` (matches the wire protocol;
/// loud error if any dim doesn't fit in u32).
///
/// Accepts `Float32` and `BFloat16` tensors; a tensor whose dtype
/// already matches `wire_dtype` serializes verbatim, anything else is
/// cast through `to_dtype` first. Enforcing the wire dtype HERE (rather
/// than following each tensor's dtype) is load-bearing: the pinned
/// snapshot readout falls back to an f32 passthrough on failure, and a
/// single rank silently switching frame dtype mid-run would desync the
/// round schema and tear the cohort down — the cast makes every frame
/// uniform whatever staging path produced the tensors.
pub fn tensors_to_round_frame(tensors: &[&Tensor], wire_dtype: u8) -> Result<RoundFrame> {
    let wire_tensor_dtype = wire_tensor_dtype(wire_dtype)?;
    let mut payloads = Vec::with_capacity(tensors.len());
    for (i, t) in tensors.iter().enumerate() {
        if !matches!(t.dtype(), DType::Float32 | DType::BFloat16) {
            return Err(TensorError::new(&format!(
                "cpu_reduce: tensor[{i}] dtype {:?} not supported (only Float32 \
                 / BFloat16). Extend cpu_reduce.rs::tensors_to_round_frame and \
                 the round_frame.rs codec helpers together to add support.",
                t.dtype()
            )));
        }
        let shape = wire_shape(i, t)?;
        let bytes = if t.dtype() == wire_tensor_dtype {
            t.to_blob()?
        } else {
            // Cast on-device before the blob so the transient is
            // wire-sized, not always f32-sized (libtorch's cast rounds
            // to nearest-even, same as the byte codec).
            t.to_dtype(wire_tensor_dtype)?.to_blob()?
        };
        payloads.push(TensorPayload {
            dtype: wire_dtype,
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

/// Map a wire dtype tag to the tensor dtype the payload bytes carry;
/// loud error on unknown tags.
fn wire_tensor_dtype(wire_dtype: u8) -> Result<DType> {
    match wire_dtype {
        DTYPE_F32 => Ok(DType::Float32),
        DTYPE_BF16 => Ok(DType::BFloat16),
        other => Err(TensorError::new(&format!(
            "cpu_reduce: unsupported wire dtype tag {other} (0 = f32, 1 = bf16)"
        ))),
    }
}

/// Tensor shape as the wire's `Vec<u32>` (loud error if any dim doesn't
/// fit — the protocol uses u32 shape dims).
fn wire_shape(i: usize, t: &Tensor) -> Result<Vec<u32>> {
    t.shape()
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
        .collect()
}

/// Build a list of new CPU `Tensor`s from a [`RoundFrame`].
///
/// Inverse of [`tensors_to_round_frame`]. Each payload's bytes go
/// straight into a tensor via [`Tensor::from_blob`] (which validates
/// shape-vs-byte-count loudly), then bf16 payloads upcast to f32 — the
/// returned tensors are ALWAYS f32 whatever the wire carried, since
/// every consumer (param writeback, divergence math, outer-optimizer
/// state) works in f32. The blob path deliberately skips the
/// intermediate `Vec<f32>` a per-element decode would allocate: on the
/// params frame that vector was a whole extra model copy live at the
/// sync barrier on every rank at once (part of the measured first-sync
/// RAM spike).
///
/// The returned tensors live on `Device::CPU`. Callers wanting them on
/// GPU should follow up with [`Tensor::to_device`].
pub fn round_frame_to_tensors(frame: &RoundFrame) -> Result<Vec<Tensor>> {
    let mut out = Vec::with_capacity(frame.tensors.len());
    for (i, p) in frame.tensors.iter().enumerate() {
        out.push(payload_to_cpu_tensor(i, p)?);
    }
    Ok(out)
}

/// Decode ONE payload into an f32 CPU tensor: `from_blob` at the wire
/// dtype (validates shape-vs-byte-count loudly), bf16 upcast to f32.
/// An elided payload (wire zero-elision, no bytes) decodes to zeros of
/// its declared shape — zeros carry no dtype rounding, so f32 directly.
/// Shared per-payload body of [`round_frame_to_tensors`] and the
/// draining streamed decode in `CpuReduceClient`.
fn payload_to_cpu_tensor(i: usize, p: &TensorPayload) -> Result<Tensor> {
    let dtype = payload_wire_dtype(i, p)?;
    let shape: Vec<i64> = p.shape.iter().map(|&d| d as i64).collect();
    if p.is_elided() && p.numel() > 0 {
        return Tensor::zeros(
            &shape,
            TensorOptions {
                dtype: DType::Float32,
                device: Device::CPU,
            },
        )
        .map_err(|e| TensorError::new(&format!("cpu_reduce: payload[{i}]: {e}")));
    }
    let t = Tensor::from_blob(&p.bytes, &shape, dtype, Device::CPU)
        .map_err(|e| TensorError::new(&format!("cpu_reduce: payload[{i}]: {e}")))?;
    if dtype == DType::Float32 {
        Ok(t)
    } else {
        t.to_dtype(DType::Float32)
    }
}

/// Map payload `p`'s wire dtype tag to the tensor dtype its bytes
/// carry; loud error (naming the payload index) on unknown tags.
fn payload_wire_dtype(i: usize, p: &TensorPayload) -> Result<DType> {
    match p.dtype {
        DTYPE_F32 => Ok(DType::Float32),
        DTYPE_BF16 => Ok(DType::BFloat16),
        other => Err(TensorError::new(&format!(
            "cpu_reduce: payload[{i}] unsupported wire dtype tag {other} \
             (0 = f32, 1 = bf16)"
        ))),
    }
}

/// Decode ONE payload into `dsts[i]` — a caller-owned destination (the
/// request tensor whose reply payload this is: the rank's pinned
/// snapshot staging on the production path) — and return a shallow
/// clone of it. The reply mirrors the sent frame's schema by protocol,
/// so a shape mismatch is a LOUD wire-drift error, never a realloc
/// (reallocating caller-owned staging would silently break the
/// aliasing the RAM-neutral decode rides on).
///
/// Dtype pairings: matching wire/destination dtypes decode as a plain
/// memcpy (f32 wire → f32 staging; bf16 wire → the bf16 params
/// staging, VERBATIM — the fold ran in f32 controller-side and the
/// reply is bf16 on the wire regardless, so the bf16 staging holds
/// exactly the information an upcast copy would); a bf16 payload into
/// an f32 destination upcasts via `copy_` (the buffers staging, which
/// stays f32 under a bf16 wire).
///
/// Refused destinations (fresh-alloc fallback + once-per-run notice
/// via `fallback`): non-CPU — the degenerate double-failure snapshot
/// path can ship live CUDA tensors, and decoding into those would
/// write the consensus into live params from the bridge thread on the
/// wrong stream; any pairing that would DOWNCAST (an f32 payload into
/// a narrower destination would silently quantize the consensus —
/// structurally impossible today, since the staging is only bf16 when
/// the wire is, but guarded rather than assumed); unknown destination
/// dtypes.
fn decode_into_dst(
    i: usize,
    p: &TensorPayload,
    dsts: &[Tensor],
    fallback: &mut Option<String>,
) -> Result<Tensor> {
    let dtype = payload_wire_dtype(i, p)?;
    let shape: Vec<i64> = p.shape.iter().map(|&d| d as i64).collect();
    let Some(dst) = dsts.get(i) else {
        return Err(TensorError::new(&format!(
            "cpu_reduce: reply payload[{i}] has no armed decode destination \
             ({} were armed); reply/request schema drift",
            dsts.len(),
        )));
    };
    let dst_ok = dst.device() == Device::CPU
        && (dst.dtype() == dtype || (dst.dtype() == DType::Float32 && dtype == DType::BFloat16));
    if !dst_ok {
        if fallback.is_none() {
            *fallback = Some(format!(
                "dst[{i}] is {:?} on {:?} against a {dtype:?} payload \
                 (need CPU, same dtype or an f32 dst for a bf16 payload)",
                dst.dtype(),
                dst.device(),
            ));
        }
        return payload_to_cpu_tensor(i, p);
    }
    if dst.shape() != shape {
        return Err(TensorError::new(&format!(
            "cpu_reduce: reply payload[{i}] shape {shape:?} != armed \
             destination shape {:?}; reply/request schema drift",
            dst.shape(),
        )));
    }
    if p.is_elided() && p.numel() > 0 {
        // Wire zero-elision: no bytes came — the payload is zeros of
        // its declared shape, written straight into the destination.
        dst.zero_()?;
        return Ok(dst.clone());
    }
    let wire = Tensor::from_blob(&p.bytes, &shape, dtype, Device::CPU)
        .map_err(|e| TensorError::new(&format!("cpu_reduce: payload[{i}]: {e}")))?;
    dst.copy_(&wire, false)?;
    Ok(dst.clone())
}

#[cfg(test)]
#[path = "cpu_reduce_tests.rs"]
mod tests;
