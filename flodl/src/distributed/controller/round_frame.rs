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
/// Identical shape sent rank→controller and controller→rank. Payloads
/// carry [`DTYPE_F32`] or [`DTYPE_BF16`] tensor bytes; every frame of a
/// round must agree on the dtype per tensor (the fold validates the
/// schema and errors loudly on a mismatch — a cohort where some ranks
/// enabled `bf16_wire` and some did not is a config error, not a
/// tolerable variation).
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
    /// Wire dtype tag (see [`DTYPE_F32`] / [`DTYPE_BF16`]).
    pub dtype: u8,
    /// Tensor shape.
    pub shape: Vec<u32>,
    /// Raw tensor bytes (native byte order).
    ///
    /// EMPTY while the shape declares elements = **wire zero-elision**:
    /// the payload is all zeros of the declared dtype/shape and no
    /// payload bytes crossed the wire (`nbytes = 0` on the frame, with
    /// the schema intact and MAC-covered). Senders that KNOW their
    /// payload is structurally zeros — the formation broadcast's
    /// non-root ranks, an idle rank's zero-mass contribution, the
    /// zero-mass scatter — declare it instead of shipping (and MACing)
    /// a model of zero bytes; the fold reads elided as "adds nothing",
    /// which is IEEE-exact (accumulating ±0.0 changes no accumulator
    /// bit). One deliberate semantic delta: an elided contribution is
    /// TRUE zeros — the legacy scale-by-zero path would forward a
    /// poisoned sender's `0.0 × NaN = NaN` into the consensus, elision
    /// cannot.
    pub bytes: Vec<u8>,
}

impl TensorPayload {
    /// Number of element-slots in the tensor (product of shape dims).
    pub fn numel(&self) -> usize {
        self.shape.iter().map(|d| *d as usize).product()
    }

    /// Wire zero-elision (see [`Self::bytes`]): no payload bytes — the
    /// tensor reads as all zeros of the declared dtype/shape. True for
    /// zero-numel tensors too, where both readings coincide (zeros of
    /// nothing IS nothing).
    pub fn is_elided(&self) -> bool {
        self.bytes.is_empty()
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
/// Delegates to [`read_round_frame_streamed`] with a frame-building
/// sink — one wire-format reader, mirroring the writer side.
///
/// `pub(crate)` so the rank-side client in `cpu_reduce` can share the
/// wire format without duplication.
pub(crate) fn read_round_frame<R: Read>(
    stream: &mut R,
    salt: &SessionSalt,
) -> Result<Option<RoundFrame>> {
    let mut tensors = Vec::new();
    match read_round_frame_streamed(stream, salt, None, &mut |_, payload| {
        tensors.push(payload);
        Ok(())
    })? {
        Some((kind, weight)) => Ok(Some(RoundFrame {
            tensors,
            kind,
            weight,
        })),
        None => Ok(None),
    }
}

/// Streaming counterpart of [`read_round_frame`] — THE frame reader.
/// Hands each parsed [`TensorPayload`] to `sink` as it comes off the
/// stream instead of accumulating the whole frame, so a receiver can
/// consume payloads one at a time (e.g. decode into a tensor and free
/// the bytes) and never hold frame + decoded output together.
///
/// MAC-BEFORE-USE CONTRACT: the sink receives UNAUTHENTICATED bytes —
/// the HMAC footer is only verified after the last payload. A sink must
/// do nothing but buffer or inertly transform (copy into a container /
/// build a tensor); no decision and no adoption may happen on the
/// values until this function returns `Ok`. On `Err`, the caller MUST
/// discard everything the sink accumulated. This is the same exposure
/// as the materialized reader (which also buffers unauthenticated
/// payload bytes before the footer check) — the trust boundary is the
/// return, not the sink.
///
/// Returns `Ok(None)` on clean EOF before the header, `Ok(Some((kind,
/// weight)))` after the footer authenticates.
///
/// `on_header` (optional) fires once, after the frame's kind + weight
/// parse and BEFORE any payload reaches `sink`. The wire layout puts
/// the realized-work weight ahead of the tensor list precisely so a
/// receiver can pick a decode DESTINATION per round (e.g. reused
/// staging vs fresh allocs for a zero-mass reply) without buffering the
/// frame. Same MAC-before-use exposure as `sink`: the values are
/// UNAUTHENTICATED at callback time — they may steer inert buffering
/// only, never be adopted as data.
pub(crate) fn read_round_frame_streamed<R: Read>(
    stream: &mut R,
    salt: &SessionSalt,
    on_header: Option<&mut dyn FnMut(RoundKind, f64)>,
    sink: &mut dyn FnMut(usize, TensorPayload) -> Result<()>,
) -> Result<Option<(RoundKind, f64)>> {
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
    if let Some(cb) = on_header {
        cb(kind, weight);
    }

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
        sink(
            ti,
            TensorPayload {
                dtype,
                shape,
                bytes,
            },
        )?;
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
    Ok(Some((kind, weight)))
}

/// Per-tensor wire metadata for the streaming writer: everything the
/// frame body needs EXCEPT the payload bytes, which the emitter
/// produces on demand. Lets a sender compute the exact frame length
/// (and write a length prefix) without materializing a single payload.
pub(crate) struct PayloadPart<'a> {
    /// Wire dtype tag (see [`DTYPE_F32`] / [`DTYPE_BF16`]).
    pub dtype: u8,
    /// Tensor shape.
    pub shape: &'a [u32],
    /// Exact payload byte count the emitter will write for this tensor.
    pub nbytes: u64,
}

/// Exact on-wire length of a round frame with these payload parts —
/// header + kind + weight + per-tensor (meta + shape + nbytes field +
/// payload bytes) + HMAC footer. Byte-for-byte what
/// [`write_round_frame_streamed`] emits, which is what lets the sender
/// commit a length prefix BEFORE producing any payload bytes.
pub(crate) fn round_frame_wire_len(parts: &[PayloadPart<'_>]) -> u64 {
    let mut len: u64 = 8 + 1 + 8; // hdr + kind byte + weight
    for p in parts {
        len += 2 + 4 * p.shape.len() as u64 + 8 + p.nbytes;
    }
    len + 8 // HMAC footer
}

/// MAC-and-count tee for the streaming writer's payload leg: every byte
/// the emitter writes goes to the underlying stream AND into the frame
/// HMAC, with a running count so [`write_round_frame_streamed`] can
/// verify the emitter honored its declared `nbytes`.
pub(crate) struct MacTee<'a, W: Write> {
    inner: &'a mut W,
    mac: &'a mut HMAC,
    written: u64,
}

impl<W: Write> Write for MacTee<'_, W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.mac.update(&buf[..n]);
        self.written += n as u64;
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// Streaming counterpart of [`write_round_frame`] — THE frame writer
/// (the materialized variant delegates here, so the wire format has a
/// single implementation to keep in lockstep with [`read_round_frame`]).
///
/// Payload bytes never coexist as a whole frame on the sender: for each
/// tensor the writer emits the wire metadata itself, then hands the
/// emitter a [`MacTee`] to write exactly `parts[i].nbytes` payload bytes
/// (loud error otherwise — by then the stream is committed, so the
/// caller must treat it as a torn connection, which every production
/// caller already does). On model-sized frames this replaces a
/// whole-frame serialize buffer with one payload transient at a time.
pub(crate) fn write_round_frame_streamed<W: Write>(
    stream: &mut W,
    kind: RoundKind,
    weight: f64,
    parts: &[PayloadPart<'_>],
    salt: &SessionSalt,
    emit: &mut dyn FnMut(usize, &mut MacTee<'_, W>) -> Result<()>,
) -> Result<()> {
    let mut mac = HMAC::new(salt.as_slice());

    let mut hdr = [0u8; 8];
    hdr[0..4].copy_from_slice(&ROUND_FRAME_MAGIC.to_le_bytes());
    hdr[4..8].copy_from_slice(&(parts.len() as u32).to_le_bytes());
    stream.write_all(&hdr).map_err(|e| {
        TensorError::new(&format!("cluster_controller: frame header write failed: {e}"))
    })?;
    mac.update(hdr);
    // Round-kind byte (MAC-covered), mirroring `read_round_frame`.
    let kind_byte = [kind.to_wire()];
    stream.write_all(&kind_byte).map_err(|e| {
        TensorError::new(&format!("cluster_controller: frame kind write failed: {e}"))
    })?;
    mac.update(kind_byte);
    // Realized-work weight (MAC-covered), mirroring `read_round_frame`.
    let weight_bytes = weight.to_le_bytes();
    stream.write_all(&weight_bytes).map_err(|e| {
        TensorError::new(&format!("cluster_controller: frame weight write failed: {e}"))
    })?;
    mac.update(weight_bytes);
    for (ti, p) in parts.iter().enumerate() {
        let meta = [p.dtype, p.shape.len() as u8];
        stream.write_all(&meta).map_err(|e| {
            TensorError::new(&format!(
                "cluster_controller: tensor[{ti}] meta write failed: {e}"
            ))
        })?;
        mac.update(meta);
        for d in p.shape {
            let d_bytes = d.to_le_bytes();
            stream.write_all(&d_bytes).map_err(|e| {
                TensorError::new(&format!(
                    "cluster_controller: tensor[{ti}] shape write failed: {e}"
                ))
            })?;
            mac.update(d_bytes);
        }
        let nb_bytes = p.nbytes.to_le_bytes();
        stream.write_all(&nb_bytes).map_err(|e| {
            TensorError::new(&format!(
                "cluster_controller: tensor[{ti}] nbytes write failed: {e}"
            ))
        })?;
        mac.update(nb_bytes);
        let mut tee = MacTee {
            inner: stream,
            mac: &mut mac,
            written: 0,
        };
        emit(ti, &mut tee).map_err(|e| {
            TensorError::new(&format!(
                "cluster_controller: tensor[{ti}] data write failed: {e}"
            ))
        })?;
        if tee.written != p.nbytes {
            return Err(TensorError::new(&format!(
                "cluster_controller: tensor[{ti}] emitter wrote {} bytes, \
                 declared {} — frame length already committed, stream is torn",
                tee.written, p.nbytes
            )));
        }
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

/// Write a RoundFrame to a stream, appending the 8-byte HMAC-SHA256
/// footer keyed by `salt`. `pub(crate)` companion to
/// [`read_round_frame`]; shared by the rank-side client. Delegates to
/// [`write_round_frame_streamed`] with a borrowing emitter — zero extra
/// copies, one wire-format implementation.
pub(crate) fn write_round_frame<W: Write>(
    stream: &mut W,
    frame: &RoundFrame,
    salt: &SessionSalt,
) -> Result<()> {
    let parts: Vec<PayloadPart<'_>> = frame
        .tensors
        .iter()
        .map(|t| PayloadPart {
            dtype: t.dtype,
            shape: &t.shape,
            nbytes: t.bytes.len() as u64,
        })
        .collect();
    write_round_frame_streamed(
        stream,
        frame.kind,
        frame.weight,
        &parts,
        salt,
        &mut |ti, tee| {
            tee.write_all(&frame.tensors[ti].bytes)
                .map_err(|e| TensorError::new(&e.to_string()))
        },
    )
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
/// Supports [`DTYPE_F32`] and [`DTYPE_BF16`] payloads; loud error on
/// other dtypes. The sum ALWAYS ACCUMULATES IN F32 whatever the payload
/// dtype — bf16 exists only on the wire, decoded per element into the
/// f32 accumulator — and the output payload re-encodes in the input
/// dtype, so a fold tier never changes the frame schema it forwards.
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
            // Wire zero-elision: an elided payload (no bytes) is valid
            // alongside full payloads of the same dtype/shape schema;
            // two FULL payloads must still agree byte-for-byte in
            // length. (Full-payload byte count vs shape numel is
            // enforced per frame inside the accumulate below.)
            if !a.is_elided() && !b.is_elided() && a.bytes.len() != b.bytes.len() {
                return Err(TensorError::new(&format!(
                    "cluster_controller: frame {i} tensor[{ti}] nbytes {} != frame 0 nbytes {}",
                    b.bytes.len(),
                    a.bytes.len()
                )));
            }
        }
    }

    // Sum per tensor: decode each payload straight into the f32
    // accumulator (no intermediate Vec per frame), re-encode the sum in
    // the input dtype.
    let mut out_tensors = Vec::with_capacity(ref_frame.tensors.len());
    for ti in 0..ref_frame.tensors.len() {
        let dtype = ref_frame.tensors[ti].dtype;
        let elem = payload_element_size(dtype).map_err(|e| {
            TensorError::new(&format!("cluster_controller: tensor[{ti}]: {e}"))
        })?;
        let shape = ref_frame.tensors[ti].shape.clone();
        let numel = ref_frame.tensors[ti].numel();
        if !ref_frame.tensors[ti].is_elided()
            && numel * elem != ref_frame.tensors[ti].bytes.len()
        {
            return Err(TensorError::new(&format!(
                "cluster_controller: tensor[{ti}] shape {shape:?} numel*element_size {} != nbytes {}",
                numel * elem,
                ref_frame.tensors[ti].bytes.len()
            )));
        }
        // Wire zero-elision, fold-through: zeros + zeros = zeros. When
        // every accepted frame elided this tensor, the sum is
        // structurally zeros too, so it STAYS elided — no bytes
        // materialize here, and the frame keeps its elided form through
        // every fold tier and the final scatter (elision is a member of
        // the fold monoid, so it composes through hierarchical relay
        // tiers exactly like the mass algebra does).
        if frames.iter().all(|f| f.tensors[ti].is_elided()) {
            out_tensors.push(TensorPayload {
                dtype,
                shape,
                bytes: Vec::new(),
            });
            continue;
        }
        let mut accum: Vec<f32> = vec![0.0; numel];
        for f in frames.iter() {
            accumulate_payload_into(&f.tensors[ti], &mut accum)?;
        }
        out_tensors.push(TensorPayload {
            dtype,
            shape,
            bytes: f32_slice_to_payload_bytes(&accum, dtype)?,
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
            scale_payload(payload, inv)?;
        }
    }
    Ok(summed)
}

// ---------------------------------------------------------------------------
// Payload byte codec (f32 / bf16)
// ---------------------------------------------------------------------------
//
// bf16 is the top 16 bits of an f32, so the codec is pure byte math —
// no tensor ops on the wire path. Encoding rounds to nearest-even
// (matching libtorch's f32→bf16 cast), decoding is exact.

/// Round-to-nearest-even f32 → bf16 conversion (bit-level, matches the
/// IEEE default rounding libtorch uses). NaN is quietened (mantissa MSB
/// forced) so a NaN payload can never round into an infinity.
pub(crate) fn f32_to_bf16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    if x.is_nan() {
        return ((bits >> 16) as u16) | 0x0040;
    }
    let round_bit = (bits >> 16) & 1;
    ((bits + 0x7FFF + round_bit) >> 16) as u16
}

/// Exact bf16 → f32 conversion (bf16 is a truncated f32).
pub(crate) fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// Element size in bytes for a wire dtype tag; loud error on unknown
/// tags (the extension pointer for any future dtype).
pub(crate) fn payload_element_size(dtype: u8) -> Result<usize> {
    match dtype {
        DTYPE_F32 => Ok(4),
        DTYPE_BF16 => Ok(2),
        other => Err(TensorError::new(&format!(
            "unsupported wire dtype tag {other} (0 = f32, 1 = bf16); extend \
             round_frame.rs::payload_element_size and the codec helpers together"
        ))),
    }
}

/// Decode `payload`'s bytes element-wise into the f32 accumulator
/// (`accum[i] += decode(payload[i])`), without materializing an
/// intermediate f32 vector. The fold's inner loop — payloads are
/// model-sized, so the zero-alloc path matters for the per-host RAM
/// peak.
pub(crate) fn accumulate_payload_into(
    payload: &TensorPayload,
    accum: &mut [f32],
) -> Result<()> {
    // Wire zero-elision: an elided payload IS zeros of its declared
    // schema, and adding zeros is adding nothing (IEEE-exact — ±0.0
    // changes no accumulator bit). Shape agreement was the caller's
    // schema check; there are no bytes left to length-check.
    if payload.is_elided() {
        return Ok(());
    }
    let elem = payload_element_size(payload.dtype)?;
    if payload.bytes.len() != accum.len() * elem {
        return Err(TensorError::new(&format!(
            "payload byte count {} != accumulator numel {} x element size {elem}",
            payload.bytes.len(),
            accum.len(),
        )));
    }
    match payload.dtype {
        DTYPE_F32 => {
            for (a, c) in accum.iter_mut().zip(payload.bytes.chunks_exact(4)) {
                *a += f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
        }
        DTYPE_BF16 => {
            for (a, c) in accum.iter_mut().zip(payload.bytes.chunks_exact(2)) {
                *a += bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]]));
            }
        }
        _ => unreachable!("payload_element_size validated the tag"),
    }
    Ok(())
}

/// Decode a payload into an owned f32 vector (exact for both dtypes —
/// bf16 upcasts losslessly). Sized from the declared SHAPE, not the
/// byte count, so an elided payload decodes to zeros of its numel and
/// a full payload whose bytes disagree with its shape errs loudly.
pub(crate) fn payload_to_f32(payload: &TensorPayload) -> Result<Vec<f32>> {
    let elem = payload_element_size(payload.dtype)?;
    let numel = payload.numel();
    if !payload.is_elided() && payload.bytes.len() != numel * elem {
        return Err(TensorError::new(&format!(
            "payload byte count {} != shape numel {numel} x element size {elem}",
            payload.bytes.len(),
        )));
    }
    let mut out = vec![0.0f32; numel];
    accumulate_payload_into(payload, &mut out)?;
    Ok(out)
}

/// Encode an f32 slice as payload bytes in `dtype` (f32 verbatim
/// little-endian, bf16 via round-to-nearest-even).
pub(crate) fn f32_slice_to_payload_bytes(data: &[f32], dtype: u8) -> Result<Vec<u8>> {
    let elem = payload_element_size(dtype)?;
    let mut out = Vec::with_capacity(data.len() * elem);
    match dtype {
        DTYPE_F32 => {
            for x in data {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        DTYPE_BF16 => {
            for x in data {
                out.extend_from_slice(&f32_to_bf16_bits(*x).to_le_bytes());
            }
        }
        _ => unreachable!("payload_element_size validated the tag"),
    }
    Ok(out)
}

/// Multiply every element of `payload` by `factor` in place (byte-level;
/// per-element decode → f32 multiply → re-encode for bf16). Used by the
/// divide-once normalization.
pub(crate) fn scale_payload(payload: &mut TensorPayload, factor: f32) -> Result<()> {
    scale_payload_bytes(&mut payload.bytes, payload.dtype, factor)
}

/// [`scale_payload`] on raw wire bytes: decode each element, multiply in
/// f32, re-encode in place (single round-to-nearest-even stage for
/// bf16). Also the rank-side γ-scale — fused into the streaming encode,
/// it replaces the model-sized `mul_scalar` scratch copy the reduce used
/// to make before shipping.
pub(crate) fn scale_payload_bytes(bytes: &mut [u8], dtype: u8, factor: f32) -> Result<()> {
    match dtype {
        DTYPE_F32 => {
            for c in bytes.chunks_exact_mut(4) {
                let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]) * factor;
                c.copy_from_slice(&v.to_le_bytes());
            }
            Ok(())
        }
        DTYPE_BF16 => {
            for c in bytes.chunks_exact_mut(2) {
                let v = bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])) * factor;
                c.copy_from_slice(&f32_to_bf16_bits(v).to_le_bytes());
            }
            Ok(())
        }
        other => Err(TensorError::new(&format!(
            "scale_payload: unsupported wire dtype tag {other}"
        ))),
    }
}

/// Test-only f32 byte helpers, superseded in production by the
/// dtype-aware codec above ([`payload_to_f32`] /
/// [`f32_slice_to_payload_bytes`]); kept for the controller tests'
/// plain-f32 fixtures.
#[cfg(test)]
pub(super) fn bytes_as_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    if !bytes.len().is_multiple_of(4) {
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

/// Test-only companion to [`bytes_as_f32`].
#[cfg(test)]
pub(super) fn f32_to_bytes(data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 4);
    for x in data {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}
