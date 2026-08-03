//! Dataset download and caching for ddp-bench.
//!
//! Downloads standard datasets on first use and caches raw files to disk.
//! Parsing is handled by the flodl dataset parsers in `flodl::data::datasets`.

use std::fs;
use std::io::{Read, Write};
use std::path::Path;

use flodl::data::datasets::{Cifar10, Mnist, Shakespeare};
// The two-tier source/cache invariant and its mechanism live in flodl:
// every flodl binary reading a dataset faces them, and flodl still never
// learns a URL (acquisition arrives as a closure). ddp-bench keeps only
// what is its own: the URLs, the byte ranges, and what counts as valid.
use flodl::data::{data_cache_dir, publish_atomically, resolve_cached, write_error};
use flodl::tensor::{Result, TensorError};

// ---------------------------------------------------------------------------
// MNIST
// ---------------------------------------------------------------------------

const MNIST_BASE: &str = "https://storage.googleapis.com/cvdf-datasets/mnist";
const MNIST_TRAIN_IMAGES: &str = "train-images-idx3-ubyte.gz";
const MNIST_TRAIN_LABELS: &str = "train-labels-idx1-ubyte.gz";
const MNIST_TEST_IMAGES: &str = "t10k-images-idx3-ubyte.gz";
const MNIST_TEST_LABELS: &str = "t10k-labels-idx1-ubyte.gz";

/// Download MNIST training data (if not cached) and parse it.
///
/// Files are cached in `{data_dir}/mnist/`.
/// Returns 60,000 images `[N, 1, 28, 28]` and labels `[N]`.
pub fn ensure_mnist(data_dir: &Path) -> Result<Mnist> {
    ensure_mnist_split(data_dir, MNIST_TRAIN_IMAGES, MNIST_TRAIN_LABELS, "train")
}

/// Download MNIST test data (if not cached) and parse it.
///
/// Files are cached in `{data_dir}/mnist/`.
/// Returns 10,000 images `[N, 1, 28, 28]` and labels `[N]`.
pub fn ensure_mnist_test(data_dir: &Path) -> Result<Mnist> {
    ensure_mnist_split(data_dir, MNIST_TEST_IMAGES, MNIST_TEST_LABELS, "test")
}

fn ensure_mnist_split(
    data_dir: &Path, images_file: &str, labels_file: &str, split: &str,
) -> Result<Mnist> {
    let images_path = resolve_cached(data_dir, "mnist", images_file, exists, |dst| {
        download_to_file(&format!("{MNIST_BASE}/{images_file}"), dst)
    })?;
    let labels_path = resolve_cached(data_dir, "mnist", labels_file, exists, |dst| {
        download_to_file(&format!("{MNIST_BASE}/{labels_file}"), dst)
    })?;

    eprintln!("  parsing MNIST {split}...");
    let images_gz = read_file(&images_path)?;
    let labels_gz = read_file(&labels_path)?;
    Mnist::parse(&images_gz, &labels_gz)
}

// ---------------------------------------------------------------------------
// CIFAR-10
// ---------------------------------------------------------------------------

const CIFAR10_URL: &str = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz";
const CIFAR10_TRAIN_BATCHES: [&str; 5] = [
    "data_batch_1.bin",
    "data_batch_2.bin",
    "data_batch_3.bin",
    "data_batch_4.bin",
    "data_batch_5.bin",
];
const CIFAR10_TEST_BATCH: &str = "test_batch.bin";

/// Ensure the CIFAR-10 batches are available and return the directory
/// holding them.
///
/// Directory-level rather than file-level two-tier lookup, because the
/// parsers take a set of six batch files and mixing tiers per file would
/// make the returned dir meaningless. A fully populated source root is
/// used in place; otherwise everything is extracted into the node-local
/// cache. Ranks never write the source root.
pub fn ensure_cifar10_extracted(data_dir: &Path) -> Result<std::path::PathBuf> {
    let complete = |dir: &Path| {
        CIFAR10_TRAIN_BATCHES
            .iter()
            .chain(std::iter::once(&CIFAR10_TEST_BATCH))
            .all(|name| dir.join(name).exists())
    };

    let from_source = data_dir.join("cifar10");
    if complete(&from_source) {
        return Ok(from_source);
    }
    let dir = data_cache_dir().join("cifar10");
    if complete(&dir) {
        return Ok(dir);
    }
    fs::create_dir_all(&dir).map_err(|e| write_error(&dir, &e))?;

    // Stream the ~170 MB archive straight through the decoder: it never
    // lands on disk and never sits in RAM whole. So there is no staging
    // file to name uniquely, none to clean up, and no transient
    // duplication when several ranks on one host acquire concurrently --
    // which a staged tar would have cost N times over on both disk and
    // RAM, for the largest dataset the bench uses. Their concurrent
    // extraction stays safe on per-file atomic publish alone.
    //
    // A mid-stream failure leaves a partial SET of batches, each one
    // whole; `complete` then reports false and the next attempt
    // re-acquires.
    eprintln!("    downloading {CIFAR10_URL}...");
    let resp = ureq::get(CIFAR10_URL)
        .call()
        .map_err(|e| TensorError::new(&format!("GET {CIFAR10_URL}: {e}")))?;
    extract_cifar10(resp.into_body().into_reader(), &dir)?;

    Ok(dir)
}

/// Download CIFAR-10 training data (if not cached) and parse it.
///
/// Files are cached in `{data_dir}/cifar10/`.
/// Returns 50,000 images `[N, 3, 32, 32]` and labels `[N]`.
pub fn ensure_cifar10(data_dir: &Path) -> Result<Cifar10> {
    let dir = ensure_cifar10_extracted(data_dir)?;

    eprintln!("  parsing CIFAR-10 train...");
    let mut batch_data: Vec<Vec<u8>> = Vec::with_capacity(5);
    for name in &CIFAR10_TRAIN_BATCHES {
        batch_data.push(read_file(&dir.join(name))?);
    }
    let refs: Vec<&[u8]> = batch_data.iter().map(|v| v.as_slice()).collect();
    Cifar10::parse(&refs)
}

/// Download CIFAR-10 test data (if not cached) and parse it.
///
/// Files are cached in `{data_dir}/cifar10/`.
/// Returns 10,000 images `[N, 3, 32, 32]` and labels `[N]`.
pub fn ensure_cifar10_test(data_dir: &Path) -> Result<Cifar10> {
    let dir = ensure_cifar10_extracted(data_dir)?;

    eprintln!("  parsing CIFAR-10 test...");
    let test_data = read_file(&dir.join(CIFAR10_TEST_BATCH))?;
    Cifar10::parse(&[&test_data])
}

/// Extract CIFAR-10 batch files from the tar.gz archive.
fn extract_cifar10(tar_gz: impl Read, out_dir: &Path) -> Result<()> {
    let gz = flate2::read::GzDecoder::new(tar_gz);
    let mut archive = tar::Archive::new(gz);

    for entry in archive
        .entries()
        .map_err(|e| TensorError::new(&format!("tar entries: {e}")))?
    {
        let mut entry =
            entry.map_err(|e| TensorError::new(&format!("tar entry: {e}")))?;

        let path = entry
            .path()
            .map_err(|e| TensorError::new(&format!("tar path: {e}")))?
            .to_path_buf();

        // Extract only .bin files (data_batch_*.bin and test_batch.bin)
        if let Some(name) = path.file_name().and_then(std::ffi::OsStr::to_str)
            && name.ends_with(".bin")
        {
            let dest = out_dir.join(name);
            let mut buf = Vec::new();
            entry
                .read_to_end(&mut buf)
                .map_err(|e| TensorError::new(&format!("read {name}: {e}")))?;
            // Atomic per file, so the `complete()` check above cannot see a
            // half-extracted batch, and two ranks extracting the same
            // archive concurrently just rename identical bytes into place.
            publish_atomically(&dest, |f| {
                f.write_all(&buf).map_err(|e| write_error(&dest, &e))
            })?;
            eprintln!("    extracted {name} ({} bytes)", buf.len());
        }
    }

    // Verify all training batches were extracted
    for name in &CIFAR10_TRAIN_BATCHES {
        if !out_dir.join(name).exists() {
            return Err(TensorError::new(&format!(
                "CIFAR-10 archive missing {name}"
            )));
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Shakespeare
// ---------------------------------------------------------------------------

const SHAKESPEARE_URL: &str =
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt";

/// Default train fraction for Shakespeare splits (nanoGPT convention).
const SHAKESPEARE_TRAIN_RATIO: f64 = 0.9;

/// Download Shakespeare text (if not cached) and parse into sequences.
///
/// The file is cached in `{data_dir}/shakespeare/input.txt`.
pub fn ensure_shakespeare(data_dir: &Path, seq_len: usize) -> Result<Shakespeare> {
    let path = resolve_cached(data_dir, "shakespeare", "input.txt", exists, |dst| {
        download_to_file(SHAKESPEARE_URL, dst)
    })?;

    let text = fs::read_to_string(&path)
        .map_err(|e| TensorError::new(&format!("read {}: {e}", path.display())))?;

    eprintln!(
        "  parsing Shakespeare ({} chars, seq_len={seq_len})...",
        text.len()
    );
    Shakespeare::parse(&text, seq_len)
}

/// Shakespeare train split (~90% of sequences).
///
/// Built from the same parsed dataset as [`ensure_shakespeare_test`] so
/// vocab is shared and char indices are comparable across splits.
pub fn ensure_shakespeare_train(data_dir: &Path, seq_len: usize) -> Result<Shakespeare> {
    Ok(shakespeare_split(data_dir, seq_len, SHAKESPEARE_TRAIN_RATIO)?.0)
}

/// Shakespeare test split (~10% of sequences, end of corpus).
pub fn ensure_shakespeare_test(data_dir: &Path, seq_len: usize) -> Result<Shakespeare> {
    Ok(shakespeare_split(data_dir, seq_len, SHAKESPEARE_TRAIN_RATIO)?.1)
}

/// Parse the full Shakespeare corpus, then slice the resulting
/// `(data, targets)` tensors into train/test halves that share the
/// same vocab. Splitting after parsing (rather than splitting the raw
/// text first) guarantees both halves see identical char-to-index
/// mappings even if a rare character appears in only one side.
fn shakespeare_split(
    data_dir: &Path, seq_len: usize, train_ratio: f64,
) -> Result<(Shakespeare, Shakespeare)> {
    let full = ensure_shakespeare(data_dir, seq_len)?;
    let n = full.len() as i64;
    let n_train = ((n as f64) * train_ratio) as i64;
    let n_test = n - n_train;
    if n_train <= 0 || n_test <= 0 {
        return Err(TensorError::new(&format!(
            "Shakespeare split: corpus has only {n} sequences, cannot split with ratio {train_ratio}"
        )));
    }

    let train = Shakespeare {
        data: full.data.narrow(0, 0, n_train)?,
        targets: full.targets.narrow(0, 0, n_train)?,
        vocab_size: full.vocab_size,
        char_to_idx: full.char_to_idx.clone(),
        idx_to_char: full.idx_to_char.clone(),
    };
    let test = Shakespeare {
        data: full.data.narrow(0, n_train, n_test)?,
        targets: full.targets.narrow(0, n_train, n_test)?,
        vocab_size: full.vocab_size,
        char_to_idx: full.char_to_idx,
        idx_to_char: full.idx_to_char,
    };
    Ok((train, test))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Existence-only validity, the default for fixed-content datasets.
fn exists(p: &Path) -> bool {
    p.exists()
}

fn read_file(path: &Path) -> Result<Vec<u8>> {
    fs::read(path).map_err(|e| TensorError::new(&format!("read {}: {e}", path.display())))
}

fn download_to_file(url: &str, dest: &Path) -> Result<()> {
    let bytes = download_bytes(url)?;
    publish_atomically(dest, |f| {
        f.write_all(&bytes)
            .map_err(|e| write_error(dest, &e))
    })
}

fn download_bytes(url: &str) -> Result<Vec<u8>> {
    eprintln!("    downloading {url}...");
    let buf = ureq::get(url)
        .call()
        .map_err(|e| TensorError::new(&format!("GET {url}: {e}")))?
        .into_body()
        .read_to_vec()
        .map_err(|e| TensorError::new(&format!("read {url}: {e}")))?;
    eprintln!("    {} bytes", buf.len());
    Ok(buf)
}

/// Stream a large download directly to a file (bypasses ureq body size limit).
fn download_large_to_file(url: &str, dest: &Path) -> Result<()> {
    eprintln!("    downloading {url}...");
    let resp = ureq::get(url)
        .call()
        .map_err(|e| TensorError::new(&format!("GET {url}: {e}")))?;
    let mut reader = resp.into_body().into_reader();
    let mut total = 0usize;
    publish_atomically(dest, |file| {
        let mut buf = [0u8; 65536];
        loop {
            let n = reader.read(&mut buf)
                .map_err(|e| TensorError::new(&format!("read {url}: {e}")))?;
            if n == 0 { break; }
            file.write_all(&buf[..n]).map_err(|e| write_error(dest, &e))?;
            total += n;
        }
        Ok(())
    })?;
    eprintln!("    {total} bytes");
    Ok(())
}

// ---------------------------------------------------------------------------
// OLMo token shards (olmo-mix, raw headerless u16 despite the .npy name)
// ---------------------------------------------------------------------------

/// olmo-mix v1.6 books shard, GPT-NeoX-OLMo-Dolma-v1.5 tokens (vocab
/// 50,280, raw little-endian u16). Full file is ~1.46 GB; the bench
/// stages only the leading [`OLMO_TRAIN_BYTES`] — a prefix of a raw
/// dump is itself a valid shard.
const OLMO_TRAIN_URL: &str = "https://olmo-data.org/preprocessed/olmo-mix/v1_6-decontaminated/books/gpt-neox-olmo-dolma-v1_5/part-0-00000.npy";

/// Default bytes of the train shard to stage, when `--train-tokens` is
/// not given. In real-data mode the staged slice IS one data pass:
/// 4 MiB = 2,097,152 u16 tokens = 8191 windows at seq 256 (2047 batches
/// at batch 4). `--train-tokens` overrides it, and either way the size is
/// snapped so a pass divides into whole batched events
/// (`models::olmo::resolve_train_corpus`). The full shard is 1.46 GB.
pub const OLMO_TRAIN_BYTES: u64 = 4 * 1024 * 1024;

/// Held-out C4-English validation shard from OLMo's perplexity suite
/// (same tokenizer). Like the train shard, a leading slice is staged —
/// the full file is ~2 MB and eval walls scale with it.
const OLMO_EVAL_URL: &str = "https://olmo-data.org/eval-data/perplexity/v3_small_gptneox20b/c4_en/val/part-0-00000.npy";

/// Bytes of the eval shard to stage: 512 KiB = ~262k tokens.
pub const OLMO_EVAL_BYTES: u64 = 512 * 1024;

/// Download the leading `bytes` of the olmo-mix books shard (if not
/// cached at that exact size) and return its path. Cached in
/// `{data_dir}/olmo/`.
///
/// `ensure_olmo_shard` validates the cached file's length against
/// `bytes`, so changing the staged size re-downloads rather than
/// silently training on the previous corpus.
pub fn ensure_olmo_train(data_dir: &Path, bytes: u64) -> Result<std::path::PathBuf> {
    ensure_olmo_shard(
        data_dir,
        OLMO_TRAIN_URL,
        "books-part-0-00000.head.npy",
        Some(bytes),
        bytes,
    )
}

/// Download the leading `OLMO_EVAL_BYTES` of the C4 validation shard
/// (if not cached) and return its path.
pub fn ensure_olmo_eval(data_dir: &Path) -> Result<std::path::PathBuf> {
    ensure_olmo_shard(
        data_dir,
        OLMO_EVAL_URL,
        "c4-val-part-0-00000.head.npy",
        Some(OLMO_EVAL_BYTES),
        OLMO_EVAL_BYTES,
    )
}

fn ensure_olmo_shard(
    data_dir: &Path,
    url: &str,
    file_name: &str,
    range_bytes: Option<u64>,
    expected_bytes: u64,
) -> Result<std::path::PathBuf> {
    // Exact-length validity, not mere existence: changing the staged
    // corpus size must re-fetch rather than silently train on the
    // previous one. It does NOT protect against interleaved writes of
    // the same length, which is why acquisition publishes atomically.
    let right_size =
        |p: &Path| fs::metadata(p).map(|m| m.len() == expected_bytes).unwrap_or(false);

    let path = resolve_cached(data_dir, "olmo", file_name, right_size, |dst| {
        download_range_to_file(url, dst, range_bytes)?;
        let got = fs::metadata(dst)
            .map_err(|e| TensorError::new(&format!("stat {}: {e}", dst.display())))?
            .len();
        if got != expected_bytes {
            return Err(TensorError::new(&format!(
                "olmo shard {}: got {got} bytes, expected {expected_bytes} — \
                 partial download or upstream change; delete the file and retry",
                dst.display()
            )));
        }
        Ok(())
    })?;
    Ok(path)
}

/// Stream a download to a file, optionally requesting only the leading
/// `bytes` via an HTTP Range header.
fn download_range_to_file(url: &str, dest: &Path, bytes: Option<u64>) -> Result<()> {
    match bytes {
        None => download_large_to_file(url, dest),
        Some(n) => {
            eprintln!("    downloading first {n} bytes of {url}...");
            let resp = ureq::get(url)
                .header("Range", &format!("bytes=0-{}", n - 1))
                .call()
                .map_err(|e| TensorError::new(&format!("GET {url}: {e}")))?;
            if resp.status() != 206 {
                return Err(TensorError::new(&format!(
                    "GET {url}: server ignored the Range request (status {}) — \
                     refusing to stream the full file",
                    resp.status()
                )));
            }
            let mut reader = resp.into_body().into_reader();
            let mut total = 0usize;
            publish_atomically(dest, |file| {
                let mut buf = [0u8; 65536];
                loop {
                    let n_read = reader.read(&mut buf)
                        .map_err(|e| TensorError::new(&format!("read {url}: {e}")))?;
                    if n_read == 0 { break; }
                    file.write_all(&buf[..n_read]).map_err(|e| write_error(dest, &e))?;
                    total += n_read;
                }
                Ok(())
            })?;
            eprintln!("    {total} bytes");
            Ok(())
        }
    }
}
