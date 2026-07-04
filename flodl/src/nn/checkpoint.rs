use std::io::{Read, Write};

use crate::tensor::{Device, DType, Result, Tensor, TensorError};

use super::buffer::Buffer;
use super::parameter::Parameter;

/// Magic bytes for `.fdl` checkpoint files.
pub(crate) const MAGIC: [u8; 4] = *b"FDLC";
/// Current checkpoint format version.
/// v1 = flodl 0.1.x naming, v2 = flodl 0.2.0+ naming (identical binary layout).
pub(crate) const VERSION: u32 = 2;
/// Maximum checkpoint version we can read.
const MAX_VERSION: u32 = 2;
/// Size of the structural hash field in the checkpoint header.
pub(crate) const HASH_LEN: usize = 32;

/// Report from a checkpoint load: what was loaded, skipped, or missing.
#[derive(Debug, Clone)]
pub struct LoadReport {
    /// Entries matched by name and loaded successfully.
    pub loaded: Vec<String>,
    /// Checkpoint entries with no matching model parameter or buffer (ignored).
    pub skipped: Vec<String>,
    /// Model parameters/buffers with no matching checkpoint entry (kept at init values).
    pub missing: Vec<String>,
}

/// Save parameters and buffers to a binary checkpoint.
///
/// Both params and buffers are stored as named tensors in the same flat list.
/// The format is: `MAGIC(4) | VERSION(u32=1) | hash(32 bytes) | num_entries(u32) | entries...`
///
/// Pass `structural_hash` from `Graph::structural_hash()` to embed architecture
/// identity. Pass `None` to write 32 zero bytes (hash validation skipped on load).
pub fn save_checkpoint<W: Write>(
    w: &mut W,
    params: &[(String, Parameter)],
    buffers: &[(String, Buffer)],
    structural_hash: Option<&str>,
) -> Result<()> {
    let total = (params.len() + buffers.len()) as u32;
    write_checkpoint_header(w, total, structural_hash)?;

    for (name, p) in params {
        write_entry_name(w, name)?;
        write_tensor_data(w, &p.variable.data())?;
    }

    for (name, b) in buffers {
        write_entry_name(w, name)?;
        write_tensor_data(w, &b.get())?;
    }

    Ok(())
}

/// Write the checkpoint header: `MAGIC(4) | VERSION(u32) | hash(32) | count(u32)`.
///
/// Shared by [`save_checkpoint`] (the `Tensor` path) and
/// [`save_checkpoint_from_raw`] (the raw-payload path) so the on-disk
/// layout has a single definition.
pub(crate) fn write_checkpoint_header<W: Write>(
    w: &mut W,
    total: u32,
    structural_hash: Option<&str>,
) -> Result<()> {
    w.write_all(&MAGIC).map_err(io_err)?;
    w.write_all(&VERSION.to_le_bytes()).map_err(io_err)?;
    let hash_bytes = match structural_hash {
        Some(hex) => hex_to_bytes(hex)?,
        None => [0u8; HASH_LEN],
    };
    w.write_all(&hash_bytes).map_err(io_err)?;
    w.write_all(&total.to_le_bytes()).map_err(io_err)?;
    Ok(())
}

/// Write an entry's `name_len(u32) | name` prefix (the bytes that precede
/// the tensor body in every checkpoint entry).
fn write_entry_name<W: Write>(w: &mut W, name: &str) -> Result<()> {
    let name_bytes = name.as_bytes();
    w.write_all(&(name_bytes.len() as u32).to_le_bytes()).map_err(io_err)?;
    w.write_all(name_bytes).map_err(io_err)?;
    Ok(())
}

/// One entry for [`save_checkpoint_from_raw`]: a name plus an already-
/// serialized tensor body (shape, dtype, native bytes). Lets a caller that
/// already holds raw native-byte tensor data (e.g. the cluster consensus
/// reduce) write a loadable `.fdl` without reconstructing `Tensor`s — no
/// bytes→Tensor→bytes round-trip, no duplicate model in RAM.
pub(crate) struct RawCheckpointEntry<'a> {
    /// Qualified parameter / buffer name (matches load-side keys).
    pub name: &'a str,
    /// Tensor shape (i64 dims, as the on-disk format stores them).
    pub shape: &'a [i64],
    /// Checkpoint dtype tag (see `dtype_tag` — `3` = Float32).
    pub dtype_tag: u8,
    /// Raw native-byte-order tensor data.
    pub raw: &'a [u8],
}

/// Save a checkpoint directly from raw, already-serialized tensor bodies,
/// bypassing `Tensor` construction. The on-disk format is byte-identical to
/// [`save_checkpoint`] (so [`load_checkpoint`] reads it unchanged); only the
/// source differs. Entry order is the load-side name-match order; pass
/// params first then buffers to mirror [`save_checkpoint`].
pub(crate) fn save_checkpoint_from_raw<W: Write>(
    w: &mut W,
    entries: &[RawCheckpointEntry<'_>],
    structural_hash: Option<&str>,
) -> Result<()> {
    write_checkpoint_header(w, entries.len() as u32, structural_hash)?;
    for e in entries {
        write_entry_name(w, e.name)?;
        // Tensor body: ndim(u32) + shape(i64*ndim) + dtype_tag(1) +
        // byte_count(u64) + raw. Mirrors `write_tensor_data` exactly.
        w.write_all(&(e.shape.len() as u32).to_le_bytes()).map_err(io_err)?;
        for &s in e.shape {
            w.write_all(&s.to_le_bytes()).map_err(io_err)?;
        }
        w.write_all(&[e.dtype_tag]).map_err(io_err)?;
        w.write_all(&(e.raw.len() as u64).to_le_bytes()).map_err(io_err)?;
        w.write_all(e.raw).map_err(io_err)?;
    }
    Ok(())
}

/// File wrapper for [`save_checkpoint_from_raw`]: gzips when `path` ends in
/// `.gz`, matching [`save_checkpoint_file`].
pub(crate) fn save_checkpoint_from_raw_file(
    path: &str,
    entries: &[RawCheckpointEntry<'_>],
    structural_hash: Option<&str>,
) -> Result<()> {
    let f = std::fs::File::create(path).map_err(io_err)?;
    if path.ends_with(".gz") {
        let mut w = flate2::write::GzEncoder::new(f, flate2::Compression::default());
        save_checkpoint_from_raw(&mut w, entries, structural_hash)?;
        w.finish().map_err(io_err)?;
        Ok(())
    } else {
        let mut w = std::io::BufWriter::new(f);
        save_checkpoint_from_raw(&mut w, entries, structural_hash)
    }
}

/// Load a checkpoint, matching entries by qualified name against both
/// parameters and buffers.
///
/// Returns a `LoadReport` describing what was matched, skipped, and missing.
/// Shape mismatches on a matched name are errors (not silent skips).
///
/// Pass `structural_hash` from `Graph::structural_hash()` to validate that the
/// checkpoint was saved from the same architecture. Pass `None` to skip validation.
/// If both the file hash and expected hash are non-zero and they differ, returns an error.
pub fn load_checkpoint<R: Read>(
    r: &mut R,
    params: &[(String, Parameter)],
    buffers: &[(String, Buffer)],
    structural_hash: Option<&str>,
) -> Result<LoadReport> {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic).map_err(io_err)?;
    if magic != MAGIC {
        return Err(TensorError::new(
            "invalid checkpoint: bad magic (expected .fdl checkpoint)"
        ));
    }

    let version = read_u32(r)?;
    if version == 0 || version > MAX_VERSION {
        return Err(TensorError::new(&format!(
            "unsupported checkpoint version {} (this build supports 1..={})",
            version, MAX_VERSION,
        )));
    }

    // Read and validate structural hash
    let mut file_hash = [0u8; HASH_LEN];
    r.read_exact(&mut file_hash).map_err(io_err)?;

    let file_nonzero = file_hash.iter().any(|&b| b != 0);
    if let Some(expected_hex) = structural_hash {
        let expected = hex_to_bytes(expected_hex)?;
        let expected_nonzero = expected.iter().any(|&b| b != 0);
        if file_nonzero && expected_nonzero && file_hash != expected {
            return Err(TensorError::new(&format!(
                "checkpoint architecture mismatch: file={} model={}",
                bytes_to_hex(&file_hash),
                expected_hex,
            )));
        }
    }

    let count = read_u32(r)? as usize;

    // Read all checkpoint entries into a map
    let mut ckpt: std::collections::HashMap<String, (Vec<i64>, DType, Vec<u8>)> =
        std::collections::HashMap::with_capacity(count);

    for _ in 0..count {
        let name_len = read_u32(r)? as usize;
        let mut name_bytes = vec![0u8; name_len];
        r.read_exact(&mut name_bytes).map_err(io_err)?;
        let name = String::from_utf8_lossy(&name_bytes).into_owned();

        let ndim = read_u32(r)? as usize;
        let mut shape = vec![0i64; ndim];
        for s in &mut shape { *s = read_i64(r)?; }
        let mut tag = [0u8; 1];
        r.read_exact(&mut tag).map_err(io_err)?;
        let dtype = dtype_from_tag(tag[0])?;
        let byte_count = read_u64(r)? as usize;
        let mut raw = vec![0u8; byte_count];
        r.read_exact(&mut raw).map_err(io_err)?;
        ckpt.insert(name, (shape, dtype, raw));
    }

    let mut loaded = Vec::new();
    let mut missing = Vec::new();

    // Match parameters
    for (name, p) in params {
        if let Some((shape, dtype, raw)) = ckpt.remove(name) {
            let model_shape = p.variable.shape();
            if shape != model_shape {
                return Err(TensorError::new(&format!(
                    "parameter {:?}: shape mismatch: checkpoint={:?} model={:?}",
                    name, shape, model_shape
                )));
            }
            let t = tensor_from_raw_bytes(&raw, &shape, dtype)?;
            let model_dtype = p.variable.data().dtype();
            let t = if t.dtype() != model_dtype { t.to_dtype(model_dtype)? } else { t };
            let dev = p.variable.data().device();
            if dev != Device::CPU {
                p.variable.set_data(t.to_device(dev)?);
            } else {
                p.variable.set_data(t);
            }
            loaded.push(name.clone());
        } else {
            missing.push(name.clone());
        }
    }

    // Match buffers
    for (name, b) in buffers {
        if let Some((shape, dtype, raw)) = ckpt.remove(name) {
            let model_shape = b.shape();
            if shape != model_shape {
                return Err(TensorError::new(&format!(
                    "buffer {:?}: shape mismatch: checkpoint={:?} model={:?}",
                    name, shape, model_shape
                )));
            }
            let t = tensor_from_raw_bytes(&raw, &shape, dtype)?;
            let model_dtype = b.get().dtype();
            let t = if t.dtype() != model_dtype { t.to_dtype(model_dtype)? } else { t };
            let dev = b.device();
            if dev != Device::CPU {
                b.set(t.to_device(dev)?);
            } else {
                b.set(t);
            }
            loaded.push(name.clone());
        } else {
            missing.push(name.clone());
        }
    }

    let skipped: Vec<String> = ckpt.into_keys().collect();

    Ok(LoadReport { loaded, skipped, missing })
}

/// Save checkpoint to a file path. Uses gzip compression if path ends with `.gz`.
pub fn save_checkpoint_file(
    path: &str,
    params: &[(String, Parameter)],
    buffers: &[(String, Buffer)],
    structural_hash: Option<&str>,
) -> Result<()> {
    // Atomic write: stream into `<path>.tmp`, then rename over the final path.
    // A crash mid-write (SIGKILL, disk-full, power loss) then never leaves a
    // torn `<path>` that resume could mistake for valid — it leaves a stale
    // `.tmp` instead, which resume ignores. gzip is chosen from the FINAL
    // extension, not the tmp name, so the `.tmp` suffix cannot defeat `.gz`
    // detection. Rename within a single directory is atomic on POSIX.
    let is_gz = path.ends_with(".gz");
    let tmp = format!("{path}.tmp");
    let write_result = (|| -> Result<()> {
        let f = std::fs::File::create(&tmp).map_err(io_err)?;
        if is_gz {
            let mut w = flate2::write::GzEncoder::new(f, flate2::Compression::default());
            save_checkpoint(&mut w, params, buffers, structural_hash)?;
            w.finish().map_err(io_err)?;
            Ok(())
        } else {
            let mut w = std::io::BufWriter::new(f);
            save_checkpoint(&mut w, params, buffers, structural_hash)?;
            // Explicit flush so a write error surfaces here rather than being
            // swallowed by BufWriter's drop-flush after we've already renamed.
            w.flush().map_err(io_err)
        }
    })();
    match write_result {
        Ok(()) => std::fs::rename(&tmp, path).map_err(io_err),
        Err(e) => {
            // Best-effort cleanup so a failed write doesn't litter a stale
            // `.tmp`; the write error is what the caller needs to see.
            let _ = std::fs::remove_file(&tmp);
            Err(e)
        }
    }
}

/// Load checkpoint from a file path. Detects gzip from `.gz` extension.
pub fn load_checkpoint_file(
    path: &str,
    params: &[(String, Parameter)],
    buffers: &[(String, Buffer)],
    structural_hash: Option<&str>,
) -> Result<LoadReport> {
    let f = std::fs::File::open(path).map_err(io_err)?;
    if path.ends_with(".gz") {
        let mut r = flate2::read::GzDecoder::new(f);
        load_checkpoint(&mut r, params, buffers, structural_hash)
    } else {
        let mut r = std::io::BufReader::new(f);
        load_checkpoint(&mut r, params, buffers, structural_hash)
    }
}

/// Peek at the version number of a checkpoint file without reading the full contents.
///
/// Read just the parameter and buffer names from a `.fdl` checkpoint
/// without loading any tensor data.
///
/// Useful when a caller needs to introspect the checkpoint's shape — for
/// example, to detect optional sub-modules (a pooler, a task head, …)
/// before constructing the matching graph. Reading is bounded by the
/// header's entry count, so a malformed file errors at parse rather
/// than allocating unbounded memory.
///
/// Detects gzip from a `.gz` extension. The structural-hash field is
/// read but not validated — pair this with `load_checkpoint_file` once
/// the matching graph is built if you need hash validation.
pub fn checkpoint_keys(path: &str) -> Result<Vec<String>> {
    let f = std::fs::File::open(path).map_err(io_err)?;
    let mut r: Box<dyn Read> = if path.ends_with(".gz") {
        Box::new(flate2::read::GzDecoder::new(f))
    } else {
        Box::new(std::io::BufReader::new(f))
    };

    let mut magic = [0u8; 4];
    r.read_exact(&mut magic).map_err(io_err)?;
    if magic != MAGIC {
        return Err(TensorError::new(
            "invalid checkpoint: bad magic (expected .fdl checkpoint)",
        ));
    }
    let version = read_u32(&mut r)?;
    if version == 0 || version > MAX_VERSION {
        return Err(TensorError::new(&format!(
            "unsupported checkpoint version {} (this build supports 1..={})",
            version, MAX_VERSION,
        )));
    }
    // Skip the 32-byte structural hash.
    let mut _hash = [0u8; HASH_LEN];
    r.read_exact(&mut _hash).map_err(io_err)?;

    let count = read_u32(&mut r)? as usize;
    let mut keys = Vec::with_capacity(count);
    for _ in 0..count {
        let name_len = read_u32(&mut r)? as usize;
        let mut name_bytes = vec![0u8; name_len];
        r.read_exact(&mut name_bytes).map_err(io_err)?;
        keys.push(String::from_utf8_lossy(&name_bytes).into_owned());
        // Skip ndim, shape, dtype tag, byte_count + raw payload.
        let ndim = read_u32(&mut r)? as usize;
        for _ in 0..ndim {
            let _ = read_i64(&mut r)?;
        }
        let mut tag = [0u8; 1];
        r.read_exact(&mut tag).map_err(io_err)?;
        let byte_count = read_u64(&mut r)? as usize;
        // Skip payload.
        std::io::copy(&mut r.by_ref().take(byte_count as u64), &mut std::io::sink())
            .map_err(io_err)?;
    }
    Ok(keys)
}

/// Returns the version field (1 for flodl 0.1.x, 2 for flodl 0.2.0+).
/// Useful to decide whether a checkpoint needs migration before loading.
pub fn checkpoint_version(path: &str) -> Result<u32> {
    let f = std::fs::File::open(path).map_err(io_err)?;
    let mut r: Box<dyn Read> = if path.ends_with(".gz") {
        Box::new(flate2::read::GzDecoder::new(f))
    } else {
        Box::new(std::io::BufReader::new(f))
    };
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic).map_err(io_err)?;
    if magic != MAGIC {
        return Err(TensorError::new(
            "invalid checkpoint: bad magic (expected .fdl checkpoint)"
        ));
    }
    read_u32(&mut r)
}

// --- Tensor state helpers for optimizer save/load ---

/// Write an optional tensor (for optimizer buffers that may not be initialized).
/// Uses native dtype — same format as v2 parameters.
pub(crate) fn write_tensor_state<W: Write>(w: &mut W, t: Option<&Tensor>) -> Result<()> {
    match t {
        None => {
            w.write_all(&[0u8]).map_err(io_err)?;
        }
        Some(t) => {
            w.write_all(&[1u8]).map_err(io_err)?;
            write_tensor_data(w, t)?;
        }
    }
    Ok(())
}

/// Read an optional tensor (returns None if the tensor was nil when saved).
pub(crate) fn read_tensor_state<R: Read>(r: &mut R, device: Device) -> Result<Option<Tensor>> {
    let mut present = [0u8; 1];
    r.read_exact(&mut present).map_err(io_err)?;
    if present[0] == 0 {
        return Ok(None);
    }

    let t = read_tensor_data(r)?;
    if device != Device::CPU {
        Ok(Some(t.to_device(device)?))
    } else {
        Ok(Some(t))
    }
}

// --- Internal: dtype-aware tensor serialization ---

/// DType tag byte for checkpoint format. `pub(crate)` so the cluster
/// consensus writer can tag raw f32 payloads for
/// [`save_checkpoint_from_raw`] without duplicating the mapping.
pub(crate) fn dtype_tag(dtype: DType) -> u8 {
    match dtype {
        DType::Float16  => 1,
        DType::BFloat16 => 2,
        DType::Float32  => 3,
        DType::Float64  => 4,
        DType::Int32    => 5,
        DType::Int64    => 6,
    }
}

fn dtype_from_tag(tag: u8) -> Result<DType> {
    match tag {
        1 => Ok(DType::Float16),
        2 => Ok(DType::BFloat16),
        3 => Ok(DType::Float32),
        4 => Ok(DType::Float64),
        5 => Ok(DType::Int32),
        6 => Ok(DType::Int64),
        _ => Err(TensorError::new(&format!("unknown dtype tag: {}", tag))),
    }
}

/// Write tensor data in native dtype: shape + dtype tag + raw bytes.
pub(crate) fn write_tensor_data<W: Write>(w: &mut W, t: &Tensor) -> Result<()> {
    let shape = t.shape();
    w.write_all(&(shape.len() as u32).to_le_bytes()).map_err(io_err)?;
    for &s in &shape {
        w.write_all(&s.to_le_bytes()).map_err(io_err)?;
    }

    let dtype = t.dtype();
    w.write_all(&[dtype_tag(dtype)]).map_err(io_err)?;

    let numel = t.numel() as usize;
    let elem_size = dtype.element_size();
    let byte_count = numel * elem_size;

    // Copy raw bytes from tensor (handles any dtype)
    let raw = copy_raw_bytes(t, byte_count)?;
    w.write_all(&(byte_count as u64).to_le_bytes()).map_err(io_err)?;
    w.write_all(&raw).map_err(io_err)?;

    Ok(())
}

/// Read tensor data written by write_tensor_data.
fn read_tensor_data<R: Read>(r: &mut R) -> Result<Tensor> {
    let ndim = read_u32(r)? as usize;
    let mut shape = vec![0i64; ndim];
    for s in &mut shape {
        *s = read_i64(r)?;
    }

    let mut tag = [0u8; 1];
    r.read_exact(&mut tag).map_err(io_err)?;
    let dtype = dtype_from_tag(tag[0])?;

    let byte_count = read_u64(r)? as usize;
    let mut raw = vec![0u8; byte_count];
    r.read_exact(&mut raw).map_err(io_err)?;

    tensor_from_raw_bytes(&raw, &shape, dtype)
}

/// Copy raw bytes from a tensor (any dtype). Moves to CPU if needed.
fn copy_raw_bytes(t: &Tensor, byte_count: usize) -> Result<Vec<u8>> {
    let mut buf = vec![0u8; byte_count];
    let err = unsafe {
        flodl_sys::flodl_copy_data(
            t.raw(),
            buf.as_mut_ptr() as *mut std::ffi::c_void,
            byte_count as i64,
        )
    };
    check_err_raw(err)?;
    Ok(buf)
}

/// Construct a tensor from raw bytes + shape + dtype.
fn tensor_from_raw_bytes(raw: &[u8], shape: &[i64], dtype: DType) -> Result<Tensor> {
    // Route through the typed constructors to get a proper owned tensor
    match dtype {
        DType::Float32 => {
            let data: Vec<f32> = raw.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            Tensor::from_f32(&data, shape, Device::CPU)
        }
        DType::Float64 => {
            let data: Vec<f64> = raw.chunks_exact(8)
                .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect();
            Tensor::from_f64(&data, shape, Device::CPU)
        }
        DType::Int64 => {
            let data: Vec<i64> = raw.chunks_exact(8)
                .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect();
            Tensor::from_i64(&data, shape, Device::CPU)
        }
        DType::Float16 | DType::BFloat16 | DType::Int32 => {
            // For f16/bf16/i32: load raw bytes via from_blob directly.
            let mut shape_v = shape.to_vec();
            let mut handle: flodl_sys::FlodlTensor = std::ptr::null_mut();
            let (dev_type, dev_idx) = crate::tensor::Device::CPU.to_ffi();
            let err = unsafe {
                flodl_sys::flodl_from_blob(
                    raw.as_ptr() as *mut std::ffi::c_void,
                    shape_v.as_mut_ptr(),
                    shape_v.len() as i32,
                    dtype as i32,
                    dev_type, dev_idx,
                    &mut handle,
                )
            };
            check_err_raw(err)?;
            debug_assert!(!handle.is_null());
            // Safety: from_blob clones the data in the shim, so handle is independent
            Ok(unsafe { Tensor::from_raw_handle(handle) })
        }
    }
}

// --- Checkpoint migration ---

/// Report from a checkpoint migration.
#[derive(Debug, Clone)]
pub struct MigrateReport {
    /// Entries that kept their original name (exact match in old and new model).
    pub unchanged: Vec<String>,
    /// Entries remapped by shape+dtype matching: `(old_name, new_name)`.
    pub remapped: Vec<(String, String)>,
    /// Checkpoint entries with no matching model parameter/buffer (not migrated).
    pub dropped: Vec<String>,
    /// Model parameters/buffers with no matching checkpoint entry (will use init values).
    pub missing: Vec<String>,
}

impl MigrateReport {
    /// True if every checkpoint entry was matched (nothing dropped or missing).
    pub fn is_complete(&self) -> bool {
        self.dropped.is_empty() && self.missing.is_empty()
    }
}

impl std::fmt::Display for MigrateReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if !self.unchanged.is_empty() {
            writeln!(f, "unchanged ({}):", self.unchanged.len())?;
            for name in &self.unchanged { writeln!(f, "  {}", name)?; }
        }
        if !self.remapped.is_empty() {
            writeln!(f, "remapped ({}):", self.remapped.len())?;
            for (old, new) in &self.remapped { writeln!(f, "  {} -> {}", old, new)?; }
        }
        if !self.dropped.is_empty() {
            writeln!(f, "dropped ({}):", self.dropped.len())?;
            for name in &self.dropped { writeln!(f, "  {}", name)?; }
        }
        if !self.missing.is_empty() {
            writeln!(f, "missing ({}):", self.missing.len())?;
            for name in &self.missing { writeln!(f, "  {}", name)?; }
        }
        Ok(())
    }
}

/// Raw checkpoint entry for migration (not loaded into a live Tensor).
struct RawEntry {
    name: String,
    shape: Vec<i64>,
    dtype: DType,
    raw: Vec<u8>,
}

/// Read checkpoint header and all raw entries without constructing tensors.
fn read_raw_checkpoint<R: Read>(r: &mut R) -> Result<Vec<RawEntry>> {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic).map_err(io_err)?;
    if magic != MAGIC {
        return Err(TensorError::new(
            "invalid checkpoint: bad magic (expected .fdl checkpoint)"
        ));
    }
    let version = read_u32(r)?;
    if version == 0 || version > MAX_VERSION {
        return Err(TensorError::new(&format!(
            "unsupported checkpoint version {} (this build supports 1..={})",
            version, MAX_VERSION,
        )));
    }
    // Skip structural hash
    let mut _hash = [0u8; HASH_LEN];
    r.read_exact(&mut _hash).map_err(io_err)?;

    let count = read_u32(r)? as usize;
    let mut entries = Vec::with_capacity(count);

    for _ in 0..count {
        let name_len = read_u32(r)? as usize;
        let mut name_bytes = vec![0u8; name_len];
        r.read_exact(&mut name_bytes).map_err(io_err)?;
        let name = String::from_utf8_lossy(&name_bytes).into_owned();

        let ndim = read_u32(r)? as usize;
        let mut shape = vec![0i64; ndim];
        for s in &mut shape { *s = read_i64(r)?; }
        let mut tag = [0u8; 1];
        r.read_exact(&mut tag).map_err(io_err)?;
        let dtype = dtype_from_tag(tag[0])?;
        let byte_count = read_u64(r)? as usize;
        let mut raw = vec![0u8; byte_count];
        r.read_exact(&mut raw).map_err(io_err)?;

        entries.push(RawEntry { name, shape, dtype, raw });
    }

    Ok(entries)
}

/// Write a single raw entry (name + tensor data) into a checkpoint stream.
fn write_raw_entry<W: Write>(w: &mut W, name: &str, e: &RawEntry) -> Result<()> {
    let name_bytes = name.as_bytes();
    w.write_all(&(name_bytes.len() as u32).to_le_bytes()).map_err(io_err)?;
    w.write_all(name_bytes).map_err(io_err)?;
    w.write_all(&(e.shape.len() as u32).to_le_bytes()).map_err(io_err)?;
    for &s in &e.shape {
        w.write_all(&s.to_le_bytes()).map_err(io_err)?;
    }
    w.write_all(&[dtype_tag(e.dtype)]).map_err(io_err)?;
    w.write_all(&(e.raw.len() as u64).to_le_bytes()).map_err(io_err)?;
    w.write_all(&e.raw).map_err(io_err)?;
    Ok(())
}

/// Migrate a checkpoint to match a model's current parameter and buffer naming.
///
/// Reads the source checkpoint and matches each entry against the model's
/// `named_parameters` and `named_buffers`:
///
/// 1. **Exact name match** — entries whose name and shape match a model target
///    are passed through unchanged.
/// 2. **Shape+dtype match** — remaining entries are matched to remaining model
///    targets by shape and dtype, in checkpoint order. This handles the common
///    case where only tag/node prefixes changed between versions.
///
/// The migrated checkpoint is written with a zeroed structural hash so it can
/// be loaded without architecture validation.
///
/// # Example
///
/// ```ignore
/// let graph = FlowBuilder::from(input)
///     .through(encoder).tag("encoder")
///     .build()?;
///
/// let report = migrate_checkpoint(
///     &mut src_reader,
///     &mut dst_writer,
///     &graph.named_parameters(),
///     &graph.named_buffers(),
/// )?;
/// println!("{}", report);
/// ```
pub fn migrate_checkpoint<R: Read, W: Write>(
    r: &mut R,
    w: &mut W,
    params: &[(String, Parameter)],
    buffers: &[(String, Buffer)],
) -> Result<MigrateReport> {
    let entries = read_raw_checkpoint(r)?;

    // Build model expectations in order: params then buffers
    let mut targets: Vec<(String, Vec<i64>, DType)> = Vec::with_capacity(
        params.len() + buffers.len()
    );
    for (name, p) in params {
        targets.push((name.clone(), p.variable.shape(), p.variable.data().dtype()));
    }
    for (name, b) in buffers {
        targets.push((name.clone(), b.shape(), b.get().dtype()));
    }

    let mut unchanged = Vec::new();
    let mut remapped = Vec::new();
    let mut missing = Vec::new();
    let mut used = vec![false; entries.len()];

    // output: (new_name, checkpoint_index) in model order
    let mut output: Vec<(String, usize)> = Vec::new();

    // Index checkpoint entries by name for O(1) exact lookup
    let name_index: std::collections::HashMap<&str, usize> =
        entries.iter().enumerate().map(|(i, e)| (e.name.as_str(), i)).collect();

    // Indices of model targets not yet matched
    let mut unmatched: Vec<usize> = Vec::new();

    // Pass 1: exact name + shape match
    for (mi, (name, shape, _)) in targets.iter().enumerate() {
        if let Some(&ci) = name_index.get(name.as_str()) {
            if !used[ci] && entries[ci].shape == *shape {
                unchanged.push(name.clone());
                used[ci] = true;
                output.push((name.clone(), ci));
                continue;
            }
        }
        unmatched.push(mi);
    }

    // Pass 2: shape+dtype matching in checkpoint order
    for &mi in &unmatched {
        let (name, shape, dtype) = &targets[mi];

        let found = entries.iter().enumerate()
            .find(|(ci, e)| !used[*ci] && e.shape == *shape && e.dtype == *dtype)
            .map(|(ci, _)| ci);

        if let Some(ci) = found {
            remapped.push((entries[ci].name.clone(), name.clone()));
            used[ci] = true;
            output.push((name.clone(), ci));
        } else {
            missing.push(name.clone());
        }
    }

    let dropped: Vec<String> = entries.iter().enumerate()
        .filter(|(i, _)| !used[*i])
        .map(|(_, e)| e.name.clone())
        .collect();

    // Write migrated checkpoint with zeroed structural hash
    w.write_all(&MAGIC).map_err(io_err)?;
    w.write_all(&VERSION.to_le_bytes()).map_err(io_err)?;
    w.write_all(&[0u8; HASH_LEN]).map_err(io_err)?;
    w.write_all(&(output.len() as u32).to_le_bytes()).map_err(io_err)?;

    for (name, ci) in &output {
        write_raw_entry(w, name, &entries[*ci])?;
    }

    Ok(MigrateReport { unchanged, remapped, dropped, missing })
}

/// Migrate a checkpoint file. Detects gzip from `.gz` extension on both paths.
///
/// Source and destination must be different paths.
pub fn migrate_checkpoint_file(
    src: &str,
    dst: &str,
    params: &[(String, Parameter)],
    buffers: &[(String, Buffer)],
) -> Result<MigrateReport> {
    let sf = std::fs::File::open(src).map_err(io_err)?;
    let df = std::fs::File::create(dst).map_err(io_err)?;

    match (src.ends_with(".gz"), dst.ends_with(".gz")) {
        (true, true) => {
            let mut r = flate2::read::GzDecoder::new(sf);
            let mut w = flate2::write::GzEncoder::new(df, flate2::Compression::default());
            let report = migrate_checkpoint(&mut r, &mut w, params, buffers)?;
            w.finish().map_err(io_err)?;
            Ok(report)
        }
        (true, false) => {
            let mut r = flate2::read::GzDecoder::new(sf);
            let mut w = std::io::BufWriter::new(df);
            migrate_checkpoint(&mut r, &mut w, params, buffers)
        }
        (false, true) => {
            let mut r = std::io::BufReader::new(sf);
            let mut w = flate2::write::GzEncoder::new(df, flate2::Compression::default());
            let report = migrate_checkpoint(&mut r, &mut w, params, buffers)?;
            w.finish().map_err(io_err)?;
            Ok(report)
        }
        (false, false) => {
            let mut r = std::io::BufReader::new(sf);
            let mut w = std::io::BufWriter::new(df);
            migrate_checkpoint(&mut r, &mut w, params, buffers)
        }
    }
}

// --- Shared helpers ---

pub(crate) fn io_err(e: impl std::fmt::Display) -> TensorError {
    TensorError::new(&format!("io: {}", e))
}

fn check_err_raw(err: *mut i8) -> Result<()> {
    if err.is_null() {
        Ok(())
    } else {
        let msg = unsafe { std::ffi::CStr::from_ptr(err) }
            .to_string_lossy()
            .into_owned();
        unsafe { flodl_sys::flodl_free_string(err) };
        Err(TensorError::new(&msg))
    }
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(io_err)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64<R: Read>(r: &mut R) -> Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf).map_err(io_err)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i64<R: Read>(r: &mut R) -> Result<i64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf).map_err(io_err)?;
    Ok(i64::from_le_bytes(buf))
}

// Pub(crate) helpers for optimizer state serialization
pub(crate) fn read_f64_le<R: Read>(r: &mut R) -> Result<f64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf).map_err(io_err)?;
    Ok(f64::from_le_bytes(buf))
}
pub(crate) fn write_f64_le<W: Write>(w: &mut W, v: f64) -> Result<()> {
    w.write_all(&v.to_le_bytes()).map_err(io_err)?;
    Ok(())
}
pub(crate) fn write_u32_le<W: Write>(w: &mut W, v: u32) -> Result<()> {
    w.write_all(&v.to_le_bytes()).map_err(io_err)?;
    Ok(())
}
pub(crate) fn write_i64_le<W: Write>(w: &mut W, v: i64) -> Result<()> {
    w.write_all(&v.to_le_bytes()).map_err(io_err)?;
    Ok(())
}
pub(crate) fn read_u32_le<R: Read>(r: &mut R) -> Result<u32> {
    read_u32(r)
}
pub(crate) fn read_i64_le<R: Read>(r: &mut R) -> Result<i64> {
    read_i64(r)
}

/// Decode a hex string to a 32-byte array.
fn hex_to_bytes(hex: &str) -> Result<[u8; HASH_LEN]> {
    if hex.len() != HASH_LEN * 2 {
        return Err(TensorError::new(&format!(
            "expected {} hex chars, got {}",
            HASH_LEN * 2,
            hex.len()
        )));
    }
    let mut out = [0u8; HASH_LEN];
    for (i, chunk) in hex.as_bytes().chunks(2).enumerate() {
        let hi = hex_nibble(chunk[0])?;
        let lo = hex_nibble(chunk[1])?;
        out[i] = (hi << 4) | lo;
    }
    Ok(out)
}

fn hex_nibble(b: u8) -> Result<u8> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Ok(b - b'A' + 10),
        _ => Err(TensorError::new(&format!("invalid hex byte: {}", b))),
    }
}

/// Encode a byte slice as a lowercase hex string.
fn bytes_to_hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        use std::fmt::Write;
        let _ = write!(s, "{:02x}", b);
    }
    s
}

#[cfg(test)]
#[path = "checkpoint_tests.rs"]
mod tests;
