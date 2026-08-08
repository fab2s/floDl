//! MNIST dataset parser.
//!
//! Parses the IDX binary format (gzip-compressed) into tensors.
//! Images are normalized to [0, 1] as Float32, labels are Int64.
//!
//! # Example
//!
//! ```ignore
//! let images_gz = std::fs::read("train-images-idx3-ubyte.gz")?;
//! let labels_gz = std::fs::read("train-labels-idx1-ubyte.gz")?;
//! let mnist = Mnist::parse(&images_gz, &labels_gz)?;
//! // mnist.images: [60000, 1, 28, 28] Float32
//! // mnist.labels: [60000] Int64
//! ```

use std::io::Read;

use crate::data::BatchDataSet;
use crate::tensor::{Device, Result, Tensor, TensorError};

/// Parsed MNIST dataset (train or test split).
pub struct Mnist {
    /// Images as `[N, 1, 28, 28]` Float32, normalized to [0, 1].
    pub images: Tensor,
    /// Labels as `[N]` Int64 (0-9).
    pub labels: Tensor,
}

impl Mnist {
    /// Parse gzip-compressed IDX files into tensors.
    ///
    /// Accepts the raw bytes of `*-images-idx3-ubyte.gz` and
    /// `*-labels-idx1-ubyte.gz` files.
    pub fn parse(images_gz: &[u8], labels_gz: &[u8]) -> Result<Self> {
        let images_raw = gunzip(images_gz)?;
        let labels_raw = gunzip(labels_gz)?;

        // Parse images: magic(4) + count(4) + rows(4) + cols(4) + pixels
        if images_raw.len() < 16 {
            return Err(TensorError::new("MNIST images: file too short for header"));
        }
        let magic = read_u32_be(&images_raw[0..4]);
        if magic != 2051 {
            return Err(TensorError::new(&format!(
                "MNIST images: bad magic {magic}, expected 2051"
            )));
        }
        let n = read_u32_be(&images_raw[4..8]) as usize;
        let rows = read_u32_be(&images_raw[8..12]) as usize;
        let cols = read_u32_be(&images_raw[12..16]) as usize;
        let pixel_count = n * rows * cols;
        if images_raw.len() < 16 + pixel_count {
            return Err(TensorError::new(&format!(
                "MNIST images: expected {} pixels, got {}",
                pixel_count,
                images_raw.len() - 16
            )));
        }

        // Convert u8 pixels to f32 normalized [0, 1]
        let pixels: Vec<f32> = images_raw[16..16 + pixel_count]
            .iter()
            .map(|&b| b as f32 / 255.0)
            .collect();
        let images = Tensor::from_f32(
            &pixels,
            &[n as i64, 1, rows as i64, cols as i64],
            Device::CPU,
        )?;

        // Parse labels: magic(4) + count(4) + labels
        if labels_raw.len() < 8 {
            return Err(TensorError::new("MNIST labels: file too short for header"));
        }
        let magic = read_u32_be(&labels_raw[0..4]);
        if magic != 2049 {
            return Err(TensorError::new(&format!(
                "MNIST labels: bad magic {magic}, expected 2049"
            )));
        }
        let n_labels = read_u32_be(&labels_raw[4..8]) as usize;
        if n_labels != n {
            return Err(TensorError::new(&format!(
                "MNIST: {n} images but {n_labels} labels"
            )));
        }
        if labels_raw.len() < 8 + n {
            return Err(TensorError::new("MNIST labels: truncated"));
        }

        let label_vals: Vec<i64> = labels_raw[8..8 + n].iter().map(|&b| b as i64).collect();
        let labels = Tensor::from_i64(&label_vals, &[n as i64], Device::CPU)?;

        Ok(Mnist { images, labels })
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

impl BatchDataSet for Mnist {
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

/// Decompress gzip data.
fn gunzip(data: &[u8]) -> Result<Vec<u8>> {
    let mut decoder = flate2::read::GzDecoder::new(data);
    let mut out = Vec::new();
    decoder
        .read_to_end(&mut out)
        .map_err(|e| TensorError::new(&format!("gzip decompression failed: {e}")))?;
    Ok(out)
}

/// Read a big-endian u32 from a 4-byte slice.
fn read_u32_be(bytes: &[u8]) -> u32 {
    u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal valid IDX3 (images) file.
    fn make_idx3(n: u32, rows: u32, cols: u32, pixels: &[u8]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&2051u32.to_be_bytes());
        buf.extend_from_slice(&n.to_be_bytes());
        buf.extend_from_slice(&rows.to_be_bytes());
        buf.extend_from_slice(&cols.to_be_bytes());
        buf.extend_from_slice(pixels);
        buf
    }

    /// Build a minimal valid IDX1 (labels) file.
    fn make_idx1(n: u32, labels: &[u8]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&2049u32.to_be_bytes());
        buf.extend_from_slice(&n.to_be_bytes());
        buf.extend_from_slice(labels);
        buf
    }

    /// Gzip-compress bytes.
    fn gzip(data: &[u8]) -> Vec<u8> {
        use flate2::Compression;
        use flate2::write::GzEncoder;
        use std::io::Write;
        let mut enc = GzEncoder::new(Vec::new(), Compression::fast());
        enc.write_all(data).unwrap();
        enc.finish().unwrap()
    }

    #[test]
    fn parse_small_mnist() {
        // 2 images of 3x3 pixels
        let pixels = vec![
            0u8, 128, 255, 0, 0, 0, 255, 255, 255, 10, 20, 30, 40, 50, 60, 70, 80, 90,
        ];
        let images_raw = make_idx3(2, 3, 3, &pixels);
        let labels_raw = make_idx1(2, &[3, 7]);

        let images_gz = gzip(&images_raw);
        let labels_gz = gzip(&labels_raw);

        let mnist = Mnist::parse(&images_gz, &labels_gz).unwrap();

        assert_eq!(mnist.images.shape(), &[2, 1, 3, 3]);
        assert_eq!(mnist.labels.shape(), &[2]);

        // Check first pixel normalized: 0/255 = 0.0
        let first = mnist.images.select(0, 0).unwrap();
        let val: f64 = first
            .select(0, 0)
            .unwrap()
            .select(0, 0)
            .unwrap()
            .select(0, 0)
            .unwrap()
            .item()
            .unwrap();
        assert!((val - 0.0).abs() < 1e-6);

        // Check second pixel: 128/255
        let val: f64 = first
            .select(0, 0)
            .unwrap()
            .select(0, 0)
            .unwrap()
            .select(0, 1)
            .unwrap()
            .item()
            .unwrap();
        assert!((val - 128.0 / 255.0).abs() < 1e-4);

        // Check labels
        let l0 = mnist.labels.select(0, 0).unwrap().to_i64_vec().unwrap()[0];
        let l1 = mnist.labels.select(0, 1).unwrap().to_i64_vec().unwrap()[0];
        assert_eq!(l0, 3);
        assert_eq!(l1, 7);
    }

    #[test]
    fn get_batch_wraps_indices() {
        let pixels = vec![0u8; 2 * 3 * 3];
        let images_raw = make_idx3(2, 3, 3, &pixels);
        let labels_raw = make_idx1(2, &[0, 1]);

        let mnist = Mnist::parse(&gzip(&images_raw), &gzip(&labels_raw)).unwrap();

        // Index 5 wraps to 5 % 2 = 1
        let batch = mnist.get_batch(&[0, 5]).unwrap();
        assert_eq!(batch[0].shape(), &[2, 1, 3, 3]);
        assert_eq!(batch[1].shape(), &[2]);
    }

    #[test]
    fn bad_magic_rejected() {
        let mut images_raw = make_idx3(1, 2, 2, &[0; 4]);
        images_raw[3] = 99; // corrupt magic
        let labels_raw = make_idx1(1, &[0]);

        let result = Mnist::parse(&gzip(&images_raw), &gzip(&labels_raw));
        assert!(result.is_err());
    }
}
