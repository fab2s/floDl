//! CIFAR-10 dataset parser.
//!
//! Parses the binary batch format into tensors.
//! Images are normalized to [0, 1] as Float32, labels are Int64.
//!
//! Each batch file contains 10,000 images in the format:
//! `[1 byte label][1024 R pixels][1024 G pixels][1024 B pixels]` per image.
//!
//! # Example
//!
//! ```ignore
//! let batch1 = std::fs::read("data_batch_1.bin")?;
//! let batch2 = std::fs::read("data_batch_2.bin")?;
//! // ... load all 5 training batches
//! let cifar = Cifar10::parse(&[&batch1, &batch2, ...])?;
//! // cifar.images: [50000, 3, 32, 32] Float32
//! // cifar.labels: [50000] Int64
//! ```

use crate::data::BatchDataSet;
use crate::tensor::{Device, Result, Tensor, TensorError};

/// CIFAR-10 class names in label order.
pub const CLASS_NAMES: [&str; 10] = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
];

const PIXELS_PER_IMAGE: usize = 3 * 32 * 32; // 3072
const BYTES_PER_RECORD: usize = 1 + PIXELS_PER_IMAGE; // 3073
const IMAGES_PER_BATCH: usize = 10_000;

/// Parsed CIFAR-10 dataset.
pub struct Cifar10 {
    /// Images as `[N, 3, 32, 32]` Float32, normalized to [0, 1].
    pub images: Tensor,
    /// Labels as `[N]` Int64 (0-9).
    pub labels: Tensor,
}

impl Cifar10 {
    /// Parse one or more raw CIFAR-10 binary batch files.
    ///
    /// Each slice should be the raw (uncompressed) contents of a batch file
    /// (e.g. `data_batch_1.bin`). Pass all 5 training batches for the full
    /// 50,000-image training set, or the single test batch for 10,000 test images.
    pub fn parse(batches: &[&[u8]]) -> Result<Self> {
        if batches.is_empty() {
            return Err(TensorError::new("CIFAR-10: no batch data provided"));
        }

        let mut all_pixels: Vec<f32> = Vec::new();
        let mut all_labels: Vec<i64> = Vec::new();

        for (batch_idx, &batch) in batches.iter().enumerate() {
            let expected = IMAGES_PER_BATCH * BYTES_PER_RECORD;
            if batch.len() != expected {
                return Err(TensorError::new(&format!(
                    "CIFAR-10 batch {}: expected {} bytes, got {}",
                    batch_idx, expected, batch.len()
                )));
            }

            for img_idx in 0..IMAGES_PER_BATCH {
                let offset = img_idx * BYTES_PER_RECORD;
                let label = batch[offset] as i64;
                if label > 9 {
                    return Err(TensorError::new(&format!(
                        "CIFAR-10 batch {} image {}: invalid label {}",
                        batch_idx, img_idx, label
                    )));
                }
                all_labels.push(label);

                // Pixels are already in CHW order: [1024 R][1024 G][1024 B]
                let pixel_start = offset + 1;
                let pixel_end = pixel_start + PIXELS_PER_IMAGE;
                for &b in &batch[pixel_start..pixel_end] {
                    all_pixels.push(b as f32 / 255.0);
                }
            }
        }

        let n = all_labels.len() as i64;
        let images = Tensor::from_f32(&all_pixels, &[n, 3, 32, 32], Device::CPU)?;
        let labels = Tensor::from_i64(&all_labels, &[n], Device::CPU)?;

        Ok(Cifar10 { images, labels })
    }

    /// Number of samples.
    pub fn len(&self) -> usize {
        self.images.shape()[0] as usize
    }

    /// True if the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl BatchDataSet for Cifar10 {
    fn len(&self) -> usize {
        self.images.shape()[0] as usize
    }

    fn get_batch(&self, indices: &[usize]) -> Result<Vec<Tensor>> {
        let idx: Vec<i64> = indices.iter().map(|&i| (i % self.len()) as i64).collect();
        let idx_tensor = Tensor::from_i64(&idx, &[idx.len() as i64], Device::CPU)?;
        let images = self.images.index_select(0, &idx_tensor)?;
        let labels = self.labels.index_select(0, &idx_tensor)?;
        Ok(vec![images, labels])
    }
}

// ---------------------------------------------------------------------------
// Cifar10Disk: per-sample reads from the raw batch files
// ---------------------------------------------------------------------------

/// The canonical training batch file names of the CIFAR-10 binary
/// distribution.
pub const TRAIN_BATCH_FILES: [&str; 5] = [
    "data_batch_1.bin",
    "data_batch_2.bin",
    "data_batch_3.bin",
    "data_batch_4.bin",
    "data_batch_5.bin",
];

/// The canonical test batch file name of the CIFAR-10 binary
/// distribution.
pub const TEST_BATCH_FILE: &str = "test_batch.bin";

/// CIFAR-10 read directly from the raw batch files, one sample per
/// read.
///
/// Where [`Cifar10`] parses everything into RAM up front, this reads
/// each sample's 3073 bytes from storage on demand (the raw format is
/// already a fixed-stride record file). It implements
/// [`DataSet`](crate::data::DataSet), so the sample-keyed tiers above
/// (RAM sample cache, disk stage, reservation staging) apply
/// automatically — this is the path for benchmarking storage-bound
/// training and for datasets that outgrow RAM.
///
/// Samples are `[image [3, 32, 32] Float32 in [0, 1], label [] Int64]`
/// — batch-stacked, identical to [`Cifar10::get_batch`] output.
pub struct Cifar10Disk {
    files: Vec<crate::data::records::FixedStrideRecords>,
    /// Cumulative record count before each file (index routing).
    starts: Vec<usize>,
    total: usize,
}

impl Cifar10Disk {
    /// Open raw CIFAR-10 batch files (any subset, in the given order).
    pub fn open<P: AsRef<std::path::Path>>(paths: &[P]) -> Result<Self> {
        if paths.is_empty() {
            return Err(TensorError::new("Cifar10Disk: no batch files provided"));
        }
        let mut files = Vec::with_capacity(paths.len());
        let mut starts = Vec::with_capacity(paths.len());
        let mut total = 0usize;
        for path in paths {
            let recs =
                crate::data::records::FixedStrideRecords::open(path, BYTES_PER_RECORD)?;
            starts.push(total);
            total += recs.count();
            files.push(recs);
        }
        Ok(Cifar10Disk { files, starts, total })
    }

    /// Open the 5 canonical training batches under `dir`
    /// (`data_batch_1.bin` … `data_batch_5.bin`).
    pub fn open_train(dir: impl AsRef<std::path::Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let paths: Vec<_> = TRAIN_BATCH_FILES.iter().map(|f| dir.join(f)).collect();
        Self::open(&paths)
    }

    /// Open the canonical test batch under `dir` (`test_batch.bin`).
    pub fn open_test(dir: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::open(&[dir.as_ref().join(TEST_BATCH_FILE)])
    }
}

impl crate::data::DataSet for Cifar10Disk {
    fn len(&self) -> usize {
        self.total
    }

    fn get(&self, index: usize) -> Result<Vec<Tensor>> {
        if index >= self.total {
            return Err(TensorError::new(&format!(
                "Cifar10Disk: sample {index} out of bounds ({} samples)",
                self.total
            )));
        }
        // Locate the owning file (a handful of files: linear scan).
        let file_idx = self
            .starts
            .iter()
            .rposition(|&start| start <= index)
            .expect("starts[0] == 0 covers every index");
        let record = self.files[file_idx].record(index - self.starts[file_idx])?;

        let label = record[0] as i64;
        if label > 9 {
            return Err(TensorError::new(&format!(
                "Cifar10Disk: sample {index} in {} has invalid label {label}",
                self.files[file_idx].path().display()
            )));
        }

        // Pixels are already CHW: [1024 R][1024 G][1024 B].
        let mut pixels = Vec::with_capacity(PIXELS_PER_IMAGE);
        for &b in &record[1..] {
            pixels.push(b as f32 / 255.0);
        }
        let image = Tensor::from_f32(&pixels, &[3, 32, 32], Device::CPU)?;
        let label = Tensor::from_i64(&[label], &[], Device::CPU)?;
        Ok(vec![image, label])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal CIFAR-10 batch with `n` images.
    fn make_batch(n: usize) -> Vec<u8> {
        let mut buf = Vec::with_capacity(n * BYTES_PER_RECORD);
        for i in 0..n {
            buf.push((i % 10) as u8); // label
            // R channel: all (i % 256)
            for _ in 0..1024 {
                buf.push((i % 256) as u8);
            }
            // G channel: all 0
            buf.extend_from_slice(&[0u8; 1024]);
            // B channel: all 255
            buf.extend_from_slice(&[255u8; 1024]);
        }
        buf
    }

    #[test]
    fn parse_single_batch() {
        let batch = make_batch(IMAGES_PER_BATCH);
        let cifar = Cifar10::parse(&[&batch]).unwrap();

        assert_eq!(cifar.images.shape(), &[10000, 3, 32, 32]);
        assert_eq!(cifar.labels.shape(), &[10000]);

        // First image label = 0
        let l = cifar.labels.select(0, 0).unwrap().to_i64_vec().unwrap()[0];
        assert_eq!(l, 0);

        // Second image label = 1
        let l = cifar.labels.select(0, 1).unwrap().to_i64_vec().unwrap()[0];
        assert_eq!(l, 1);
    }

    #[test]
    fn parse_multiple_batches() {
        let b1 = make_batch(IMAGES_PER_BATCH);
        let b2 = make_batch(IMAGES_PER_BATCH);
        let cifar = Cifar10::parse(&[&b1, &b2]).unwrap();
        assert_eq!(cifar.images.shape(), &[20000, 3, 32, 32]);
    }

    #[test]
    fn wrong_size_rejected() {
        let batch = [0u8; 100]; // way too short
        assert!(Cifar10::parse(&[&batch[..]]).is_err());
    }

    #[test]
    fn pixel_normalization() {
        let batch = make_batch(IMAGES_PER_BATCH);
        let cifar = Cifar10::parse(&[&batch]).unwrap();

        // Image 0: R channel all 0 -> 0.0
        let img0 = cifar.images.select(0, 0).unwrap();
        let r_pixel: f64 = img0.select(0, 0).unwrap() // R channel
            .select(0, 0).unwrap() // row 0
            .select(0, 0).unwrap() // col 0
            .item().unwrap();
        assert!((r_pixel - 0.0).abs() < 1e-6);

        // B channel all 255 -> 1.0
        let b_pixel: f64 = img0.select(0, 2).unwrap() // B channel
            .select(0, 0).unwrap()
            .select(0, 0).unwrap()
            .item().unwrap();
        assert!((b_pixel - 1.0).abs() < 1e-6);
    }

    /// Write batches to a scratch dir and return their paths.
    fn write_batches(name: &str, batches: &[Vec<u8>]) -> Vec<std::path::PathBuf> {
        let dir = std::env::temp_dir().join("flodl-cifar10-disk-tests");
        std::fs::create_dir_all(&dir).unwrap();
        batches
            .iter()
            .enumerate()
            .map(|(i, b)| {
                let path = dir.join(format!("{name}-{}-{i}.bin", std::process::id()));
                std::fs::write(&path, b).unwrap();
                path
            })
            .collect()
    }

    #[test]
    fn disk_matches_parsed() {
        use crate::data::DataSet;

        let b1 = make_batch(IMAGES_PER_BATCH);
        let b2 = make_batch(IMAGES_PER_BATCH);
        let parsed = Cifar10::parse(&[&b1, &b2]).unwrap();
        let paths = write_batches("match", &[b1, b2]);
        let disk = Cifar10Disk::open(&paths).unwrap();

        assert_eq!(DataSet::len(&disk), 2 * IMAGES_PER_BATCH);

        // Cross-file index routing + byte-identical content vs the
        // bulk parser (indices past 10_000 live in the second file).
        for &i in &[0usize, 9_999, 10_000, 10_005, 19_999] {
            let sample = disk.get(i).unwrap();
            assert_eq!(sample[0].shape(), &[3, 32, 32]);
            let bulk_img = parsed.images.select(0, i as i64).unwrap();
            let diff: f64 = sample[0].sub(&bulk_img).unwrap().abs().unwrap().sum().unwrap().item().unwrap();
            assert_eq!(diff, 0.0);
            let bulk_label: f64 = parsed.labels.select(0, i as i64).unwrap().item().unwrap();
            let disk_label: f64 = sample[1].item().unwrap();
            assert_eq!(disk_label, bulk_label);
            assert_eq!(sample[1].shape(), &[] as &[i64]);
        }

        for p in paths {
            std::fs::remove_file(p).unwrap();
        }
    }

    #[test]
    fn disk_bounds_and_bad_label_error() {
        use crate::data::DataSet;

        let mut bad = make_batch(2);
        bad[0] = 11; // invalid label on sample 0
        let paths = write_batches("bad", &[bad]);
        let disk = Cifar10Disk::open(&paths).unwrap();

        assert!(disk.get(2).is_err());
        let err = disk.get(0).unwrap_err();
        assert!(err.to_string().contains("invalid label"));

        for p in paths {
            std::fs::remove_file(p).unwrap();
        }
    }
}
