//! RoundFrame wire codec + realized-work reduction: the CPU-reduce
//! on-wire tensor frame (read/write), plus sum/divide-once reduction.

use super::*;

// ---------------------------------------------------------------------------
// RoundFrame
// ---------------------------------------------------------------------------

/// What a reduce round carries, so the controller can tell model-weight
/// traffic from bookkeeping traffic without parsing tensors.
///
/// A single sync cycle issues several reduces over the same channel: a
/// per-rank `Control` count-gather, then one or two `Model` reduces
/// (params, then buffers). The consensus-checkpoint forge accumulates only
/// `Model` frames; `Control` frames scatter normally but are never fed to
/// it. This is also the framing the relay sum-and-count layer needs to fold
/// per-host partials.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RoundKind {
    /// Model weights (params / buffers) — the consensus payload.
    #[default]
    Model,
    /// Bookkeeping (e.g. the per-rank batch-count gather) — not the model.
    Control,
}

impl RoundKind {
    /// Wire byte (MAC-covered) distinguishing the kinds.
    fn to_wire(self) -> u8 {
        match self {
            RoundKind::Model => 0,
            RoundKind::Control => 1,
        }
    }

    /// Inverse of [`Self::to_wire`]; unknown bytes are a protocol error.
    fn from_wire(b: u8) -> Result<Self> {
        match b {
            0 => Ok(RoundKind::Model),
            1 => Ok(RoundKind::Control),
            other => Err(TensorError::new(&format!(
                "cluster_controller: unknown RoundKind wire byte {other}"
            ))),
        }
    }
}

/// A round's payload: a list of tensors with shape + dtype + data.
///
/// Identical shape sent rank→controller and controller→rank. v1 only
/// supports `DTYPE_F32`; controller errors loudly on other dtypes.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RoundFrame {
    pub tensors: Vec<TensorPayload>,
    /// Whether this is model-weight or bookkeeping traffic. Defaults to
    /// [`RoundKind::Model`] so existing constructors keep building model
    /// frames; the count-gather sets [`RoundKind::Control`] explicitly.
    pub kind: RoundKind,
    /// Realized-work mass, atomic with the contribution it scales.
    /// Semantics (mass policies, idle guard, zero-mass round rule) live
    /// in [`crate::distributed::realized_work`].
    ///
    /// Rank -> controller on [`RoundKind::Model`]: the sender's weight
    /// (params: `n_i^gamma` for `n_i` optimizer steps since the last
    /// sync; buffers: a 0/1 mover indicator). The tensors are pre-scaled
    /// by this weight; the controller divides the summed tensors ONCE by
    /// the summed weight of exactly the frames it accepted into the
    /// round — work it never received never enters the divisor.
    ///
    /// Controller -> rank: the summed weight of the round. `0.0` means
    /// "nothing realized" (degenerate all-idle round): the scattered
    /// tensors are meaningless zeros and the receiver must keep its
    /// local state.
    ///
    /// Ignored on [`RoundKind::Control`] (pure-sum bookkeeping).
    pub weight: f64,
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
///
/// Reads the existing v1 frame body byte-for-byte, then reads the 8-byte
/// HMAC-SHA256 footer (`PROTOCOL_VERSION = 2`) and authenticates the
/// body against `salt`. Mismatched salts surface here on the very first
/// round-trip with a clear, loud error.
///
/// `pub(crate)` so the rank-side client in `cpu_reduce` can share the
/// wire format without duplication.
pub(crate) fn read_round_frame<R: Read>(
    stream: &mut R,
    salt: &SessionSalt,
) -> Result<Option<RoundFrame>> {
    let mut mac = HMAC::new(salt.as_slice());

    let mut hdr = [0u8; 8];
    match stream.read_exact(&mut hdr) {
        Ok(()) => {}
        Err(e) if matches!(e.kind(), ErrorKind::UnexpectedEof | ErrorKind::ConnectionReset) => {
            return Ok(None);
        }
        Err(e) => {
            return Err(TensorError::new(&format!(
                "cluster_controller: frame header read failed: {e}"
            )));
        }
    }
    mac.update(hdr);
    let magic = u32::from_le_bytes(hdr[0..4].try_into().unwrap());
    if magic != ROUND_FRAME_MAGIC {
        return Err(TensorError::new(&format!(
            "cluster_controller: frame magic 0x{magic:08x} != 0x{ROUND_FRAME_MAGIC:08x}"
        )));
    }
    let num_tensors = u32::from_le_bytes(hdr[4..8].try_into().unwrap()) as usize;
    // Unauthenticated until the trailing MAC verifies — bound before
    // trusting (same discipline as the mux envelope's frame ceiling).
    if num_tensors > MAX_ROUND_FRAME_TENSORS {
        return Err(TensorError::new(&format!(
            "cluster_controller: frame claims {num_tensors} tensors \
             (> {MAX_ROUND_FRAME_TENSORS}); corrupt or hostile peer"
        )));
    }

    // Round-kind byte (model vs bookkeeping), MAC-covered, immediately
    // after the header and before the tensor list.
    let mut kind_byte = [0u8; 1];
    stream.read_exact(&mut kind_byte).map_err(|e| {
        TensorError::new(&format!("cluster_controller: frame kind read failed: {e}"))
    })?;
    mac.update(kind_byte);
    let kind = RoundKind::from_wire(kind_byte[0])?;

    // Realized-work weight (f64 LE, MAC-covered), immediately after the
    // kind byte. See [`RoundFrame::weight`].
    let mut weight_bytes = [0u8; 8];
    stream.read_exact(&mut weight_bytes).map_err(|e| {
        TensorError::new(&format!("cluster_controller: frame weight read failed: {e}"))
    })?;
    mac.update(weight_bytes);
    let weight = f64::from_le_bytes(weight_bytes);
    if !weight.is_finite() || weight < 0.0 {
        return Err(TensorError::new(&format!(
            "cluster_controller: frame weight {weight} is not a finite non-negative \
             realized-work mass"
        )));
    }

    let mut tensors = Vec::with_capacity(num_tensors);
    let mut total_bytes: usize = 0;
    for ti in 0..num_tensors {
        let mut meta = [0u8; 2];
        stream.read_exact(&mut meta).map_err(|e| {
            TensorError::new(&format!(
                "cluster_controller: tensor[{ti}] meta read failed: {e}"
            ))
        })?;
        mac.update(meta);
        let dtype = meta[0];
        let ndim = meta[1] as usize;
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            let mut d = [0u8; 4];
            stream.read_exact(&mut d).map_err(|e| {
                TensorError::new(&format!(
                    "cluster_controller: tensor[{ti}] shape read failed: {e}"
                ))
            })?;
            mac.update(d);
            shape.push(u32::from_le_bytes(d));
        }
        let mut nb = [0u8; 8];
        stream.read_exact(&mut nb).map_err(|e| {
            TensorError::new(&format!(
                "cluster_controller: tensor[{ti}] nbytes read failed: {e}"
            ))
        })?;
        mac.update(nb);
        let nbytes = u64::from_le_bytes(nb) as usize;
        total_bytes = total_bytes.saturating_add(nbytes);
        if total_bytes > crate::distributed::wire::frame_ceiling() {
            return Err(TensorError::new(&format!(
                "cluster_controller: tensor[{ti}] pushes frame past the \
                 {} byte ceiling; corrupt or hostile peer, or a model that \
                 has outgrown the frame ceiling",
                crate::distributed::wire::frame_ceiling()
            )));
        }
        // Incremental allocation: nbytes is unauthenticated until the
        // trailing MAC verifies; never pre-allocate a hostile length.
        let bytes = crate::distributed::wire::read_exact_incremental(stream, nbytes)
            .map_err(|e| {
                TensorError::new(&format!(
                    "cluster_controller: tensor[{ti}] data read failed: {e}"
                ))
            })?;
        mac.update(&bytes);
        tensors.push(TensorPayload {
            dtype,
            shape,
            bytes,
        });
    }

    // HMAC-SHA256-64 footer: 8 bytes, little-endian, equal to the first
    // 8 bytes of HMAC-SHA256(salt, body). Backwards-incompatible vs
    // PROTOCOL_VERSION = 1 (which had no footer).
    let mut footer = [0u8; 8];
    stream.read_exact(&mut footer).map_err(|e| {
        TensorError::new(&format!(
            "cluster_controller: frame HMAC footer read failed: {e} \
             (sender at PROTOCOL_VERSION < 2, or stream truncated mid-frame)"
        ))
    })?;
    let received = u64::from_le_bytes(footer);
    let computed_full: [u8; 32] = mac.finalize();
    let computed = u64::from_le_bytes(computed_full[0..8].try_into().unwrap());
    if computed != received {
        return Err(TensorError::new(&format!(
            "cluster_controller: RoundFrame HMAC verification failed (computed \
             0x{computed:016x}, wire carried 0x{received:016x}); session salt \
             disagreement, tampered frame, or payload corruption"
        )));
    }
    Ok(Some(RoundFrame { tensors, kind, weight }))
}

/// Write a RoundFrame to a stream, appending the 8-byte HMAC-SHA256
/// footer keyed by `salt`. `pub(crate)` companion to
/// [`read_round_frame`]; shared by the rank-side client.
pub(crate) fn write_round_frame<W: Write>(
    stream: &mut W,
    frame: &RoundFrame,
    salt: &SessionSalt,
) -> Result<()> {
    let mut mac = HMAC::new(salt.as_slice());

    let mut hdr = [0u8; 8];
    hdr[0..4].copy_from_slice(&ROUND_FRAME_MAGIC.to_le_bytes());
    hdr[4..8].copy_from_slice(&(frame.tensors.len() as u32).to_le_bytes());
    stream.write_all(&hdr).map_err(|e| {
        TensorError::new(&format!("cluster_controller: frame header write failed: {e}"))
    })?;
    mac.update(hdr);
    // Round-kind byte (MAC-covered), mirroring `read_round_frame`.
    let kind_byte = [frame.kind.to_wire()];
    stream.write_all(&kind_byte).map_err(|e| {
        TensorError::new(&format!("cluster_controller: frame kind write failed: {e}"))
    })?;
    mac.update(kind_byte);
    // Realized-work weight (MAC-covered), mirroring `read_round_frame`.
    let weight_bytes = frame.weight.to_le_bytes();
    stream.write_all(&weight_bytes).map_err(|e| {
        TensorError::new(&format!("cluster_controller: frame weight write failed: {e}"))
    })?;
    mac.update(weight_bytes);
    for (ti, t) in frame.tensors.iter().enumerate() {
        let meta = [t.dtype, t.shape.len() as u8];
        stream.write_all(&meta).map_err(|e| {
            TensorError::new(&format!(
                "cluster_controller: tensor[{ti}] meta write failed: {e}"
            ))
        })?;
        mac.update(meta);
        for d in &t.shape {
            let d_bytes = d.to_le_bytes();
            stream.write_all(&d_bytes).map_err(|e| {
                TensorError::new(&format!(
                    "cluster_controller: tensor[{ti}] shape write failed: {e}"
                ))
            })?;
            mac.update(d_bytes);
        }
        let nb_bytes = (t.bytes.len() as u64).to_le_bytes();
        stream.write_all(&nb_bytes).map_err(|e| {
            TensorError::new(&format!(
                "cluster_controller: tensor[{ti}] nbytes write failed: {e}"
            ))
        })?;
        mac.update(nb_bytes);
        stream.write_all(&t.bytes).map_err(|e| {
            TensorError::new(&format!(
                "cluster_controller: tensor[{ti}] data write failed: {e}"
            ))
        })?;
        mac.update(&t.bytes);
    }

    // 8-byte HMAC-SHA256-64 footer, keyed by salt.
    let computed_full: [u8; 32] = mac.finalize();
    let mut footer = [0u8; 8];
    footer.copy_from_slice(&computed_full[0..8]);
    stream.write_all(&footer).map_err(|e| {
        TensorError::new(&format!("cluster_controller: frame HMAC footer write failed: {e}"))
    })?;
    stream
        .flush()
        .map_err(|e| TensorError::new(&format!("cluster_controller: frame flush failed: {e}")))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Reduction (realized work: sum + divide once by the accepted mass)
// ---------------------------------------------------------------------------

/// Reduce per-rank frames into the realized-work consensus, over exactly
/// the frames the controller accepted into the round.
///
/// `frames[i] = None` means rank `i` contributed nothing (dead, or
/// declared dead before its frame was accepted); its work is not
/// realized, so it enters neither the sum nor the divisor. The sum and
/// its divisor come from the same accepted frames — they cannot
/// disagree, whatever the cohort did between rounds.
///
/// [`RoundKind::Model`]: contributions arrive pre-scaled by the sender's
/// realized-work mass ([`RoundFrame::weight`]); the output is the sum
/// divided ONCE by the summed accepted mass — the true consensus, which
/// is simultaneously what every rank adopts, what the checkpoint forge
/// writes, and what the outer optimizer steps. A round whose accepted
/// mass is zero (every contributor idle, or all movers lost mid-round)
/// returns the zero sum tagged `weight = 0.0`; receivers must keep
/// their local state.
///
/// [`RoundKind::Control`]: pure element-wise sum, no divide — gathers
/// and broadcasts build on this (each rank fills its own slot / root
/// sends values and peers send zeros).
///
/// Validates that every accepted frame has identical schema (same
/// number of tensors, same dtype per tensor, same shape per tensor).
///
/// v1 supports only [`DTYPE_F32`]; loud error on other dtypes (so a
/// future user wiring f16 here gets a clear pointer at where to add
/// support, instead of silent garbage from byte-level summation).
/// Element-wise sum of `frames`, masses summed, kind preserved — the
/// associative fold monoid shared by the per-host relay fold
/// ([`crate::distributed::relay`]) and the controller's final reduce.
///
/// NEVER divides. The divide-once-by-mass normalization belongs to the
/// controller alone ([`reduce_realized_work`]); a fold layer that also
/// divided would reintroduce averaging-of-averages. Associativity is
/// what makes the recursion sound: summing host folds equals summing
/// all rank frames in exact arithmetic (f32 addition order differs, so
/// results are tolerance-identical, not byte-identical).
///
/// Loud error on any schema disagreement between frames — tensor count,
/// dtype, shape, byte length, or [`RoundKind`]: every participant of a
/// round sends the same kind by protocol, so a mismatch means desynced
/// rounds, not a tolerable variation.
pub(crate) fn sum_frames(frames: &[&RoundFrame]) -> Result<RoundFrame> {
    let Some(ref_frame) = frames.first() else {
        return Err(TensorError::new(
            "cluster_controller: sum_frames called with no frames",
        ));
    };
    let w_sum: f64 = frames.iter().map(|f| f.weight).sum();
    // Schema validation.
    for (i, f) in frames.iter().enumerate().skip(1) {
        if f.kind != ref_frame.kind {
            return Err(TensorError::new(&format!(
                "cluster_controller: frame {i} kind {:?} != frame 0 kind {:?} \
                 (desynced reduce rounds)",
                f.kind, ref_frame.kind
            )));
        }
        if f.tensors.len() != ref_frame.tensors.len() {
            return Err(TensorError::new(&format!(
                "cluster_controller: frame {i} carries {} tensors; frame 0 carries {}",
                f.tensors.len(),
                ref_frame.tensors.len()
            )));
        }
        for (ti, (a, b)) in ref_frame.tensors.iter().zip(f.tensors.iter()).enumerate() {
            if a.dtype != b.dtype {
                return Err(TensorError::new(&format!(
                    "cluster_controller: frame {i} tensor[{ti}] dtype {} != frame 0 dtype {}",
                    b.dtype, a.dtype
                )));
            }
            if a.shape != b.shape {
                return Err(TensorError::new(&format!(
                    "cluster_controller: frame {i} tensor[{ti}] shape {:?} != frame 0 shape {:?}",
                    b.shape, a.shape
                )));
            }
            if a.bytes.len() != b.bytes.len() {
                return Err(TensorError::new(&format!(
                    "cluster_controller: frame {i} tensor[{ti}] nbytes {} != frame 0 nbytes {}",
                    b.bytes.len(),
                    a.bytes.len()
                )));
            }
        }
    }

    // Sum per tensor.
    let mut out_tensors = Vec::with_capacity(ref_frame.tensors.len());
    for ti in 0..ref_frame.tensors.len() {
        let dtype = ref_frame.tensors[ti].dtype;
        if dtype != DTYPE_F32 {
            return Err(TensorError::new(&format!(
                "cluster_controller: tensor[{ti}] dtype {dtype} not supported in v1 \
                 (only DTYPE_F32 = 0 supported). Add other dtypes in controller.rs::sum_frames."
            )));
        }
        let shape = ref_frame.tensors[ti].shape.clone();
        let numel = ref_frame.tensors[ti].numel();
        if numel * std::mem::size_of::<f32>() != ref_frame.tensors[ti].bytes.len() {
            return Err(TensorError::new(&format!(
                "cluster_controller: tensor[{ti}] shape {shape:?} numel*sizeof(f32) {} != nbytes {}",
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
        out_tensors.push(TensorPayload {
            dtype: DTYPE_F32,
            shape,
            bytes: f32_to_bytes(&accum),
        });
    }
    Ok(RoundFrame {
        tensors: out_tensors,
        // The summed frame carries the same kind as its inputs so the
        // forge tap and the fold layers see model vs bookkeeping
        // correctly.
        kind: ref_frame.kind,
        weight: w_sum,
    })
}

pub(super) fn reduce_realized_work(frames: &[Option<RoundFrame>]) -> Result<RoundFrame> {
    let accepted: Vec<&RoundFrame> = frames.iter().filter_map(|f| f.as_ref()).collect();
    if accepted.is_empty() {
        return Err(TensorError::new(
            "cluster_controller: reduce_realized_work called with no accepted frames \
             (all participants dead — caller should not have reached this point)",
        ));
    }
    let mut summed = sum_frames(&accepted)?;
    // Model frames normalize by the accepted realized-work mass —
    // exactly ONCE, here, regardless of how many fold layers summed
    // below us (the divide-once law: see
    // [`crate::distributed::realized_work`]). Control frames (gathers /
    // broadcasts) stay a pure sum. A zero-mass Model round leaves the
    // (zero) sum untouched — the `weight = 0.0` on the output tells
    // receivers to keep their local state.
    if matches!(summed.kind, RoundKind::Model)
        && crate::distributed::realized_work::is_realized(summed.weight)
    {
        let inv = (1.0 / summed.weight) as f32;
        for payload in &mut summed.tensors {
            let mut vals = bytes_as_f32(&payload.bytes)?;
            for v in &mut vals {
                *v *= inv;
            }
            payload.bytes = f32_to_bytes(&vals);
        }
    }
    Ok(summed)
}

pub(super) fn bytes_as_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        return Err(TensorError::new(&format!(
            "cluster_controller: f32 byte count {} not divisible by 4",
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

pub(super) fn f32_to_bytes(data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 4);
    for x in data {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}
