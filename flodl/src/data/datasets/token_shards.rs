//! Pre-tokenized language-modeling shards in NumPy `.npy` format.
//!
//! Reads token-id arrays the way LLM pretraining corpora ship them
//! (OLMo / olmo-mix, nanoGPT-style dumps): each shard is a 1-D `.npy`
//! array of token ids, and samples are non-overlapping `seq_len`
//! windows with the target shifted by one position.
//!
//! Shards are read lazily with positioned reads (a few KB per window),
//! so a multi-GB corpus costs no RAM up front and the OS page cache
//! does the caching. Reads are a pure function of the sample index,
//! satisfying the staging-cascade purity contract.
//!
//! # Example
//!
//! ```ignore
//! let data = TokenShards::open_dir("data/olmo-mix", 1024)?;
//! // sample = ([1024] Int64 input, [1024] Int64 target, shifted by 1)
//! let loader = DataLoader::from_dataset(data).batch_size(32).build()?;
//! ```

use std::fs::File;
use std::path::{Path, PathBuf};

use crate::data::{BatchDataSet, DataSet};
use crate::tensor::{Device, Result, Tensor, TensorError};

/// Token-id dtype of a raw (headerless) shard file.
///
/// Raw dumps carry no self-description, so [`TokenShards::open_raw`]
/// needs the encoding stated explicitly. Values are little-endian.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenDtype {
    /// Unsigned 8-bit (tiny vocabularies).
    U8,
    /// Unsigned 16-bit — what OLMo / GPT-2-class dumps use (vocab < 65,536).
    U16,
    /// Unsigned 32-bit.
    U32,
    /// Signed 32-bit.
    I32,
    /// Signed 64-bit.
    I64,
}

impl TokenDtype {
    fn npy(self) -> NpyDtype {
        match self {
            TokenDtype::U8 => NpyDtype::U8,
            TokenDtype::U16 => NpyDtype::U16,
            TokenDtype::U32 => NpyDtype::U32,
            TokenDtype::I32 => NpyDtype::I32,
            TokenDtype::I64 => NpyDtype::I64,
        }
    }
}

/// Token-id dtypes supported in shard files.
///
/// Covers the encodings pretraining dumps actually use: `uint16` for
/// vocabularies under 65,536 (OLMo, GPT-2 class), wider types for
/// larger vocabularies or generic dumps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NpyDtype {
    U8,
    U16,
    U32,
    I32,
    I64,
}

impl NpyDtype {
    fn from_descr(descr: &str) -> Option<Self> {
        match descr {
            "|u1" => Some(NpyDtype::U8),
            "<u2" => Some(NpyDtype::U16),
            "<u4" => Some(NpyDtype::U32),
            "<i4" => Some(NpyDtype::I32),
            "<i8" => Some(NpyDtype::I64),
            _ => None,
        }
    }

    fn item_size(self) -> u64 {
        match self {
            NpyDtype::U8 => 1,
            NpyDtype::U16 => 2,
            NpyDtype::U32 => 4,
            NpyDtype::I32 => 4,
            NpyDtype::I64 => 8,
        }
    }

    fn decode(self, bytes: &[u8], out: &mut Vec<i64>) {
        match self {
            NpyDtype::U8 => out.extend(bytes.iter().map(|&b| b as i64)),
            NpyDtype::U16 => out.extend(
                bytes.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]]) as i64),
            ),
            NpyDtype::U32 => out.extend(
                bytes.chunks_exact(4)
                     .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64),
            ),
            NpyDtype::I32 => out.extend(
                bytes.chunks_exact(4)
                     .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64),
            ),
            NpyDtype::I64 => out.extend(bytes.chunks_exact(8).map(|c| {
                i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
            })),
        }
    }
}

struct Shard {
    file: File,
    path: PathBuf,
    dtype: NpyDtype,
    data_offset: u64,
    tokens: u64,
}

/// Language-modeling dataset over pre-tokenized `.npy` shard files.
///
/// Samples are non-overlapping windows of `seq_len` tokens; the target
/// is the same window shifted by one (next-token prediction). Windows
/// never cross a shard boundary; each shard's trailing remainder is
/// dropped. Out-of-range indices wrap (`index % len`), matching the
/// other bundled datasets.
///
/// Implements both [`DataSet`] (per-sample, gets the staging cascade)
/// and [`BatchDataSet`] (opaque batched reads); pick the entry point
/// via `DataLoader::from_dataset` / `from_batch_dataset`.
pub struct TokenShards {
    shards: Vec<Shard>,
    /// Cumulative window counts; `cum[i]..cum[i+1]` = shard i's windows.
    cum: Vec<usize>,
    seq_len: usize,
    total_tokens: u64,
}

impl TokenShards {
    /// Open an explicit list of `.npy` shard files.
    ///
    /// Shard order defines sample order. Errors loudly on a malformed
    /// or unsupported `.npy` header, a truncated file, or when no shard
    /// yields at least one window.
    ///
    /// Note: several LM corpora ship *headerless* raw dumps under a
    /// `.npy` name (OLMo's preprocessed shards, nanoGPT's `train.bin`);
    /// those fail here with "bad magic" — read them with
    /// [`TokenShards::open_raw`] instead.
    pub fn open<P: AsRef<Path>>(paths: &[P], seq_len: usize) -> Result<Self> {
        Self::build(paths, seq_len, None)
    }

    /// Open headerless raw token dumps (little-endian, C-order), stating
    /// the dtype explicitly. The token count is the file size divided by
    /// the item size; a file whose length is not a multiple of the item
    /// size errors loudly.
    ///
    /// This is the format OLMo's preprocessed shards and nanoGPT-style
    /// `.bin` dumps actually use (despite the `.npy` name OLMo gives
    /// them). A prefix slice of such a file is itself a valid shard, so
    /// partial (range) downloads stage cleanly.
    pub fn open_raw<P: AsRef<Path>>(
        paths: &[P],
        dtype: TokenDtype,
        seq_len: usize,
    ) -> Result<Self> {
        Self::build(paths, seq_len, Some(dtype.npy()))
    }

    /// Shared assembly for [`open`](Self::open) (`raw = None`, parse the
    /// npy header) and [`open_raw`](Self::open_raw) (`raw = Some(dtype)`).
    fn build<P: AsRef<Path>>(
        paths: &[P],
        seq_len: usize,
        raw: Option<NpyDtype>,
    ) -> Result<Self> {
        if seq_len == 0 {
            return Err(TensorError::new("TokenShards: seq_len must be positive"));
        }
        if paths.is_empty() {
            return Err(TensorError::new("TokenShards: no shard paths given"));
        }

        let mut shards = Vec::with_capacity(paths.len());
        let mut cum = Vec::with_capacity(paths.len() + 1);
        cum.push(0usize);
        let mut total_tokens = 0u64;

        for p in paths {
            let path = p.as_ref().to_path_buf();
            let file = File::open(&path).map_err(|e| {
                TensorError::new(&format!("TokenShards: cannot open {}: {e}", path.display()))
            })?;
            let file_len = file
                .metadata()
                .map_err(|e| {
                    TensorError::new(&format!(
                        "TokenShards: cannot stat {}: {e}",
                        path.display()
                    ))
                })?
                .len();

            let (dtype, tokens, data_offset) = match raw {
                Some(dtype) => {
                    let item = dtype.item_size();
                    if file_len % item != 0 {
                        return Err(TensorError::new(&format!(
                            "TokenShards: {} has {file_len} bytes, not a multiple \
                             of the {item}-byte item size — wrong dtype or corrupt \
                             file",
                            path.display()
                        )));
                    }
                    (dtype, file_len / item, 0)
                }
                None => {
                    let (dtype, tokens, data_offset) = parse_npy_header(&file, &path)?;
                    let needed = data_offset + tokens * dtype.item_size();
                    if file_len < needed {
                        return Err(TensorError::new(&format!(
                            "TokenShards: {} is truncated ({file_len} bytes, header \
                             declares {needed})",
                            path.display()
                        )));
                    }
                    (dtype, tokens, data_offset)
                }
            };

            // A window needs seq_len + 1 tokens (input + shifted target).
            let windows = if tokens > seq_len as u64 {
                ((tokens - 1) / seq_len as u64) as usize
            } else {
                0
            };
            total_tokens += tokens;
            cum.push(cum.last().unwrap() + windows);
            shards.push(Shard { file, path, dtype, data_offset, tokens });
        }

        if *cum.last().unwrap() == 0 {
            return Err(TensorError::new(&format!(
                "TokenShards: no shard holds a full window ({} tokens needed, \
                 largest shard has {})",
                seq_len + 1,
                shards.iter().map(|s| s.tokens).max().unwrap_or(0)
            )));
        }

        Ok(TokenShards { shards, cum, seq_len, total_tokens })
    }

    /// Open every `.npy` file in a directory, sorted by file name for a
    /// deterministic sample order.
    pub fn open_dir<P: AsRef<Path>>(dir: P, seq_len: usize) -> Result<Self> {
        let dir = dir.as_ref();
        let entries = std::fs::read_dir(dir).map_err(|e| {
            TensorError::new(&format!("TokenShards: cannot read {}: {e}", dir.display()))
        })?;
        let mut paths: Vec<PathBuf> = entries
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().map(|x| x == "npy").unwrap_or(false))
            .collect();
        paths.sort();
        if paths.is_empty() {
            return Err(TensorError::new(&format!(
                "TokenShards: no .npy files in {}",
                dir.display()
            )));
        }
        Self::open(&paths, seq_len)
    }

    /// Number of samples (windows) across all shards.
    pub fn len(&self) -> usize {
        *self.cum.last().unwrap()
    }

    /// True if the dataset holds no windows.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The window length samples are cut to.
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Total token count across all shards (including per-shard tails
    /// that don't fill a window).
    pub fn total_tokens(&self) -> u64 {
        self.total_tokens
    }

    /// Number of shard files.
    pub fn num_shards(&self) -> usize {
        self.shards.len()
    }

    /// Map a global sample index to (shard, window-within-shard).
    fn locate(&self, index: usize) -> (usize, u64) {
        let index = index % self.len();
        // partition_point: first shard whose cumulative end exceeds index.
        let s = self.cum.partition_point(|&c| c <= index) - 1;
        (s, (index - self.cum[s]) as u64)
    }

    /// Read window `w` of shard `s`: `seq_len + 1` tokens decoded to i64.
    fn read_window(&self, s: usize, w: u64) -> Result<Vec<i64>> {
        let shard = &self.shards[s];
        let item = shard.dtype.item_size();
        let offset = shard.data_offset + w * self.seq_len as u64 * item;
        let n_bytes = (self.seq_len as u64 + 1) * item;

        let mut buf = vec![0u8; n_bytes as usize];
        read_exact_at(&shard.file, &mut buf, offset).map_err(|e| {
            TensorError::new(&format!(
                "TokenShards: read failed in {} (window {w}): {e}",
                shard.path.display()
            ))
        })?;

        let mut tokens = Vec::with_capacity(self.seq_len + 1);
        shard.dtype.decode(&buf, &mut tokens);
        Ok(tokens)
    }
}

impl DataSet for TokenShards {
    fn len(&self) -> usize {
        TokenShards::len(self)
    }

    fn get(&self, index: usize) -> Result<Vec<Tensor>> {
        let (s, w) = self.locate(index);
        let tokens = self.read_window(s, w)?;
        let l = self.seq_len as i64;
        Ok(vec![
            Tensor::from_i64(&tokens[..self.seq_len], &[l], Device::CPU)?,
            Tensor::from_i64(&tokens[1..], &[l], Device::CPU)?,
        ])
    }
}

impl BatchDataSet for TokenShards {
    fn len(&self) -> usize {
        TokenShards::len(self)
    }

    fn get_batch(&self, indices: &[usize]) -> Result<Vec<Tensor>> {
        let b = indices.len();
        let mut input = Vec::with_capacity(b * self.seq_len);
        let mut target = Vec::with_capacity(b * self.seq_len);
        for &index in indices {
            let (s, w) = self.locate(index);
            let tokens = self.read_window(s, w)?;
            input.extend_from_slice(&tokens[..self.seq_len]);
            target.extend_from_slice(&tokens[1..]);
        }
        let shape = [b as i64, self.seq_len as i64];
        Ok(vec![
            Tensor::from_i64(&input, &shape, Device::CPU)?,
            Tensor::from_i64(&target, &shape, Device::CPU)?,
        ])
    }
}

/// Positioned read that leaves the file cursor untouched (thread-safe
/// over a shared `&File`).
#[cfg(unix)]
fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    std::os::unix::fs::FileExt::read_exact_at(file, buf, offset)
}

#[cfg(windows)]
fn read_exact_at(file: &File, buf: &mut [u8], mut offset: u64) -> std::io::Result<()> {
    let mut buf = buf;
    while !buf.is_empty() {
        match std::os::windows::fs::FileExt::seek_read(file, buf, offset)? {
            0 => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "unexpected end of file",
                ))
            }
            n => {
                buf = &mut buf[n..];
                offset += n as u64;
            }
        }
    }
    Ok(())
}

/// Parse a `.npy` header: magic, version, and the Python-literal dict
/// (`descr` / `fortran_order` / `shape`). Accepts 1-D C-order arrays of
/// the supported integer dtypes; anything else errors loudly.
fn parse_npy_header(file: &File, path: &Path) -> Result<(NpyDtype, u64, u64)> {
    let fail = |msg: &str| {
        TensorError::new(&format!("TokenShards: {}: {msg}", path.display()))
    };

    let mut fixed = [0u8; 8];
    read_exact_at(file, &mut fixed, 0).map_err(|e| fail(&format!("cannot read header: {e}")))?;
    if &fixed[..6] != b"\x93NUMPY" {
        return Err(fail("not a .npy file (bad magic)"));
    }
    let major = fixed[6];

    // v1 = 2-byte little-endian header length at offset 8; v2/v3 = 4-byte.
    let (header_len, header_start) = if major == 1 {
        let mut len = [0u8; 2];
        read_exact_at(file, &mut len, 8).map_err(|e| fail(&format!("short header: {e}")))?;
        (u16::from_le_bytes(len) as u64, 10u64)
    } else {
        let mut len = [0u8; 4];
        read_exact_at(file, &mut len, 8).map_err(|e| fail(&format!("short header: {e}")))?;
        (u32::from_le_bytes(len) as u64, 12u64)
    };
    if header_len > 64 * 1024 {
        return Err(fail("header implausibly large"));
    }

    let mut header = vec![0u8; header_len as usize];
    read_exact_at(file, &mut header, header_start)
        .map_err(|e| fail(&format!("short header: {e}")))?;
    let header = String::from_utf8(header).map_err(|_| fail("header is not UTF-8"))?;

    // 'descr': '<u2'
    let descr = extract_quoted(&header, "'descr'")
        .ok_or_else(|| fail("header has no 'descr' field"))?;
    let dtype = NpyDtype::from_descr(&descr).ok_or_else(|| {
        fail(&format!(
            "unsupported dtype {descr:?} (token shards must be |u1, <u2, <u4, <i4 or <i8)"
        ))
    })?;

    // 'fortran_order': False
    let fo = header
        .find("'fortran_order'")
        .ok_or_else(|| fail("header has no 'fortran_order' field"))?;
    let after_fo = &header[fo + "'fortran_order'".len()..];
    if !after_fo.trim_start_matches([':', ' ']).starts_with("False") {
        return Err(fail("fortran_order arrays are not supported"));
    }

    // 'shape': (N,)  -- 1-D only
    let sh = header
        .find("'shape'")
        .ok_or_else(|| fail("header has no 'shape' field"))?;
    let after_sh = &header[sh..];
    let open = after_sh.find('(').ok_or_else(|| fail("malformed shape"))?;
    let close = after_sh[open..].find(')').ok_or_else(|| fail("malformed shape"))? + open;
    let dims: Vec<&str> = after_sh[open + 1..close]
        .split(',')
        .map(|d| d.trim())
        .filter(|d| !d.is_empty())
        .collect();
    if dims.len() != 1 {
        return Err(fail(&format!(
            "token shards must be 1-D, got {}-D shape",
            dims.len()
        )));
    }
    let count: u64 = dims[0]
        .parse()
        .map_err(|_| fail(&format!("malformed shape dimension {:?}", dims[0])))?;

    Ok((dtype, count, header_start + header_len))
}

/// Extract the single-quoted value following `key` in a numpy header
/// dict (e.g. `key = 'descr'` -> `<u2`).
fn extract_quoted(header: &str, key: &str) -> Option<String> {
    let at = header.find(key)? + key.len();
    let rest = &header[at..];
    let open = rest.find('\'')?;
    let close = rest[open + 1..].find('\'')? + open + 1;
    Some(rest[open + 1..close].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Write a v1.0 .npy file of `tokens` in the given descr encoding.
    fn write_npy(path: &Path, descr: &str, tokens: &[i64]) {
        let dict = format!(
            "{{'descr': '{descr}', 'fortran_order': False, 'shape': ({},), }}",
            tokens.len()
        );
        // Pad with spaces so magic(6)+ver(2)+len(2)+header is 64-aligned,
        // newline-terminated (numpy format spec).
        let unpadded = 10 + dict.len() + 1;
        let pad = (64 - unpadded % 64) % 64;
        let header = format!("{dict}{}\n", " ".repeat(pad));

        let mut f = File::create(path).unwrap();
        f.write_all(b"\x93NUMPY\x01\x00").unwrap();
        f.write_all(&(header.len() as u16).to_le_bytes()).unwrap();
        f.write_all(header.as_bytes()).unwrap();
        for &t in tokens {
            match descr {
                "|u1" => f.write_all(&[(t as u8)]).unwrap(),
                "<u2" => f.write_all(&(t as u16).to_le_bytes()).unwrap(),
                "<u4" => f.write_all(&(t as u32).to_le_bytes()).unwrap(),
                "<i4" => f.write_all(&(t as i32).to_le_bytes()).unwrap(),
                "<i8" => f.write_all(&t.to_le_bytes()).unwrap(),
                other => panic!("unsupported test descr {other}"),
            }
        }
    }

    fn test_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir()
            .join(format!("flodl-token-shards-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn windows_and_shifted_targets() {
        let dir = test_dir("basic");
        let p = dir.join("a.npy");
        // 10 tokens, seq_len 4 -> (10-1)/4 = 2 windows.
        write_npy(&p, "<u2", &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let ds = TokenShards::open(&[&p], 4).unwrap();
        assert_eq!(TokenShards::len(&ds), 2);
        assert_eq!(ds.total_tokens(), 10);

        let s = DataSet::get(&ds, 0).unwrap();
        assert_eq!(s[0].to_i64_vec().unwrap(), vec![0, 1, 2, 3]);
        assert_eq!(s[1].to_i64_vec().unwrap(), vec![1, 2, 3, 4]);
        let s = DataSet::get(&ds, 1).unwrap();
        assert_eq!(s[0].to_i64_vec().unwrap(), vec![4, 5, 6, 7]);
        assert_eq!(s[1].to_i64_vec().unwrap(), vec![5, 6, 7, 8]);
    }

    #[test]
    fn multi_shard_mapping_and_tail_drop() {
        let dir = test_dir("multi");
        let a = dir.join("a.npy");
        let b = dir.join("b.npy");
        // Shard a: 5 tokens, seq_len 2 -> 2 windows; token 4 in the tail.
        write_npy(&a, "<u2", &[10, 11, 12, 13, 14]);
        // Shard b: 7 tokens -> 3 windows.
        write_npy(&b, "<u2", &[20, 21, 22, 23, 24, 25, 26]);
        let ds = TokenShards::open(&[&a, &b], 2).unwrap();
        assert_eq!(TokenShards::len(&ds), 5);
        assert_eq!(ds.num_shards(), 2);

        // Window 2 = first window of shard b (a's tail token never appears).
        let s = DataSet::get(&ds, 2).unwrap();
        assert_eq!(s[0].to_i64_vec().unwrap(), vec![20, 21]);
        let s = DataSet::get(&ds, 4).unwrap();
        assert_eq!(s[0].to_i64_vec().unwrap(), vec![24, 25]);
    }

    #[test]
    fn all_dtypes_decode_identically() {
        let dir = test_dir("dtypes");
        let tokens = [3i64, 1, 4, 1, 5, 9, 2, 6];
        for descr in ["|u1", "<u2", "<u4", "<i4", "<i8"] {
            let p = dir.join(format!("{}.npy", descr.replace(['<', '|'], "_")));
            write_npy(&p, descr, &tokens);
            let ds = TokenShards::open(&[&p], 3).unwrap();
            let s = DataSet::get(&ds, 0).unwrap();
            assert_eq!(s[0].to_i64_vec().unwrap(), vec![3, 1, 4], "descr {descr}");
        }
    }

    #[test]
    fn batch_matches_per_sample() {
        let dir = test_dir("batch");
        let p = dir.join("a.npy");
        write_npy(&p, "<u2", &(0..30).collect::<Vec<i64>>());
        let ds = TokenShards::open(&[&p], 5).unwrap();

        let batch = ds.get_batch(&[0, 3, 1]).unwrap();
        assert_eq!(batch[0].shape(), &[3, 5]);
        let flat = batch[0].to_i64_vec().unwrap();
        for (row, &idx) in [0usize, 3, 1].iter().enumerate() {
            let single = DataSet::get(&ds, idx).unwrap()[0].to_i64_vec().unwrap();
            assert_eq!(&flat[row * 5..(row + 1) * 5], &single[..], "row {row}");
        }
    }

    #[test]
    fn out_of_range_index_wraps() {
        let dir = test_dir("wrap");
        let p = dir.join("a.npy");
        write_npy(&p, "<u2", &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let ds = TokenShards::open(&[&p], 4).unwrap(); // 2 windows
        let direct = DataSet::get(&ds, 0).unwrap()[0].to_i64_vec().unwrap();
        let wrapped = DataSet::get(&ds, 2).unwrap()[0].to_i64_vec().unwrap();
        assert_eq!(direct, wrapped);
    }

    #[test]
    fn open_dir_sorts_by_name() {
        let dir = test_dir("sorted");
        // Created out of order; sample order must follow file-name sort.
        write_npy(&dir.join("b.npy"), "<u2", &[20, 21, 22]);
        write_npy(&dir.join("a.npy"), "<u2", &[10, 11, 12]);
        let ds = TokenShards::open_dir(&dir, 2).unwrap();
        assert_eq!(TokenShards::len(&ds), 2);
        let first = DataSet::get(&ds, 0).unwrap()[0].to_i64_vec().unwrap();
        assert_eq!(first, vec![10, 11]);
    }

    #[test]
    fn raw_mode_matches_npy() {
        let dir = test_dir("raw");
        let tokens: Vec<i64> = (0..20).collect();

        // Same content, once with an npy header, once headerless raw u16.
        let npy = dir.join("a.npy");
        write_npy(&npy, "<u2", &tokens);
        let raw = dir.join("a.bin");
        let mut f = File::create(&raw).unwrap();
        for &t in &tokens {
            f.write_all(&(t as u16).to_le_bytes()).unwrap();
        }

        let from_npy = TokenShards::open(&[&npy], 4).unwrap();
        let from_raw = TokenShards::open_raw(&[&raw], TokenDtype::U16, 4).unwrap();
        assert_eq!(TokenShards::len(&from_npy), TokenShards::len(&from_raw));
        for i in 0..TokenShards::len(&from_raw) {
            let a = DataSet::get(&from_npy, i).unwrap();
            let b = DataSet::get(&from_raw, i).unwrap();
            assert_eq!(a[0].to_i64_vec().unwrap(), b[0].to_i64_vec().unwrap());
            assert_eq!(a[1].to_i64_vec().unwrap(), b[1].to_i64_vec().unwrap());
        }
    }

    #[test]
    fn raw_mode_rejects_misaligned_length() {
        let dir = test_dir("raw-odd");
        let odd = dir.join("odd.bin");
        std::fs::write(&odd, [0u8; 17]).unwrap(); // not a multiple of 2
        let err = TokenShards::open_raw(&[&odd], TokenDtype::U16, 4)
            .map(|_| ())
            .unwrap_err();
        assert!(err.to_string().contains("multiple"), "unexpected: {err}");
    }

    #[test]
    fn streaming_loader_integration() {
        use crate::data::{Batch, DataLoader};

        let dir = test_dir("loader");
        let a = dir.join("a.npy");
        let b = dir.join("b.npy");
        write_npy(&a, "<u2", &(0..41).collect::<Vec<i64>>()); // 10 windows @ seq 4
        write_npy(&b, "<u2", &(100..121).collect::<Vec<i64>>()); // 5 windows
        let ds = TokenShards::open(&[&a, &b], 4).unwrap();
        assert_eq!(TokenShards::len(&ds), 15);

        // Per-sample entry point (staging cascade path), forced streaming.
        let mut loader = DataLoader::from_dataset(ds)
            .batch_size(4)
            .streaming()
            .seed(42)
            .shuffle(true)
            .drop_last(false) // loader defaults to true; full coverage asserted below
            .build()
            .unwrap();

        for epoch in 0..2 {
            let batches: Vec<Batch> = loader.epoch(epoch).map(|r| r.unwrap()).collect();
            assert_eq!(batches.len(), 4); // ceil(15 / 4)
            let mut seen = 0;
            for batch in &batches {
                assert_eq!(batch.len(), 2);
                let rows = batch[0].shape()[0];
                assert_eq!(batch[0].shape(), &[rows, 4]);
                assert_eq!(batch[1].shape(), &[rows, 4]);
                // Target is the input shifted by one everywhere.
                let x = batch[0].to_i64_vec().unwrap();
                let y = batch[1].to_i64_vec().unwrap();
                for r in 0..rows as usize {
                    assert_eq!(x[r * 4 + 1..r * 4 + 4], y[r * 4..r * 4 + 3]);
                }
                seen += rows;
            }
            assert_eq!(seen, 15, "epoch {epoch} did not cover the dataset");
        }
    }

    #[test]
    fn loud_errors() {
        let dir = test_dir("errors");

        // Bad magic.
        let bad = dir.join("bad.npy");
        std::fs::write(&bad, b"not a npy file at all........").unwrap();
        let err = TokenShards::open(&[&bad], 4).map(|_| ()).unwrap_err();
        assert!(err.to_string().contains("bad magic"), "unexpected: {err}");

        // Unsupported dtype.
        let f4 = dir.join("f4.npy");
        let dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (8,), }";
        let pad = (64 - (10 + dict.len() + 1) % 64) % 64;
        let header = format!("{dict}{}\n", " ".repeat(pad));
        let mut f = File::create(&f4).unwrap();
        f.write_all(b"\x93NUMPY\x01\x00").unwrap();
        f.write_all(&(header.len() as u16).to_le_bytes()).unwrap();
        f.write_all(header.as_bytes()).unwrap();
        f.write_all(&[0u8; 32]).unwrap();
        let err = TokenShards::open(&[&f4], 4).map(|_| ()).unwrap_err();
        assert!(err.to_string().contains("unsupported dtype"), "unexpected: {err}");

        // Truncated data (header declares more tokens than the file holds).
        let trunc = dir.join("trunc.npy");
        write_npy(&trunc, "<u2", &(0..20).collect::<Vec<i64>>());
        let len = std::fs::metadata(&trunc).unwrap().len();
        let fh = std::fs::OpenOptions::new().write(true).open(&trunc).unwrap();
        fh.set_len(len - 10).unwrap();
        let err = TokenShards::open(&[&trunc], 4).map(|_| ()).unwrap_err();
        assert!(err.to_string().contains("truncated"), "unexpected: {err}");

        // Too short for a single window.
        let tiny = dir.join("tiny.npy");
        write_npy(&tiny, "<u2", &[1, 2, 3]);
        let err = TokenShards::open(&[&tiny], 4).map(|_| ()).unwrap_err();
        assert!(err.to_string().contains("full window"), "unexpected: {err}");

        // Zero seq_len / empty path list.
        assert!(TokenShards::open(&[&tiny], 0).is_err());
        assert!(TokenShards::open::<&Path>(&[], 4).is_err());

        // Directory with no .npy files.
        let empty = test_dir("errors-empty");
        assert!(TokenShards::open_dir(&empty, 4).is_err());
    }
}
