//! Dataset download and caching for ddp-bench.
//!
//! Downloads standard datasets on first use and caches raw files to disk.
//! Parsing is handled by the flodl dataset parsers in `flodl::data::datasets`.

use std::fs;
use std::io::{Read, Write};
use std::path::Path;

use flodl::data::datasets::{Cifar10, Mnist, Shakespeare};
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
    let dir = data_dir.join("mnist");
    ensure_dir(&dir)?;

    let images_path = dir.join(images_file);
    let labels_path = dir.join(labels_file);

    if !images_path.exists() {
        let url = format!("{MNIST_BASE}/{images_file}");
        download_to_file(&url, &images_path)?;
    }
    if !labels_path.exists() {
        let url = format!("{MNIST_BASE}/{labels_file}");
        download_to_file(&url, &labels_path)?;
    }

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

/// Ensure CIFAR-10 archive is downloaded and extracted. Returns the cache dir.
pub fn ensure_cifar10_extracted(data_dir: &Path) -> Result<std::path::PathBuf> {
    let dir = data_dir.join("cifar10");
    ensure_dir(&dir)?;

    let all_present = CIFAR10_TRAIN_BATCHES
        .iter()
        .chain(std::iter::once(&CIFAR10_TEST_BATCH))
        .all(|name| dir.join(name).exists());

    if !all_present {
        // CIFAR-10 tar.gz is ~170MB, exceeds ureq default body limit.
        // Stream to a temp file, then extract.
        let tar_path = dir.join("cifar-10-binary.tar.gz");
        download_large_to_file(CIFAR10_URL, &tar_path)?;
        let tar_gz = read_file(&tar_path)?;
        extract_cifar10(&tar_gz, &dir)?;
        let _ = fs::remove_file(&tar_path); // clean up
    }

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
fn extract_cifar10(tar_gz: &[u8], out_dir: &Path) -> Result<()> {
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
            fs::write(&dest, &buf).map_err(|e| {
                TensorError::new(&format!("write {}: {e}", dest.display()))
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
    let dir = data_dir.join("shakespeare");
    ensure_dir(&dir)?;

    let path = dir.join("input.txt");
    if !path.exists() {
        download_to_file(SHAKESPEARE_URL, &path)?;
    }

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

fn ensure_dir(dir: &Path) -> Result<()> {
    // Fast-path when the dir already exists. `fs::create_dir_all` on
    // Linux issues `mkdir(2)` first and only catches `EEXIST` — on a
    // read-only mount it fails with `EROFS` even when the dir is
    // already populated. Workers running off a ro project share (e.g.
    // a libvirt virtiofs export marked `<readonly/>`) hit this when
    // the dataset cache is already there: the controller populated it
    // long ago, no actual write is needed, but the syscall fails.
    if dir.is_dir() {
        return Ok(());
    }
    fs::create_dir_all(dir)
        .map_err(|e| TensorError::new(&format!("mkdir {}: {e}", dir.display())))
}

fn read_file(path: &Path) -> Result<Vec<u8>> {
    fs::read(path).map_err(|e| TensorError::new(&format!("read {}: {e}", path.display())))
}

fn download_to_file(url: &str, dest: &Path) -> Result<()> {
    let bytes = download_bytes(url)?;
    fs::write(dest, &bytes)
        .map_err(|e| TensorError::new(&format!("write {}: {e}", dest.display())))
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
    let mut file = fs::File::create(dest)
        .map_err(|e| TensorError::new(&format!("create {}: {e}", dest.display())))?;
    let mut buf = [0u8; 65536];
    let mut total = 0usize;
    loop {
        let n = reader.read(&mut buf)
            .map_err(|e| TensorError::new(&format!("read {url}: {e}")))?;
        if n == 0 { break; }
        file.write_all(&buf[..n])
            .map_err(|e| TensorError::new(&format!("write {}: {e}", dest.display())))?;
        total += n;
    }
    eprintln!("    {total} bytes");
    Ok(())
}
