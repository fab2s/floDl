//! Fixed-stride record files: positioned per-record reads.
//!
//! Many raw dataset distributions are plain record files — a fixed
//! number of bytes per sample, optionally behind a fixed header
//! (CIFAR-10 binary batches are 3073-byte records; MNIST IDX images
//! are 784-byte records behind a 16-byte header). [`FixedStrideRecords`]
//! turns such a file into lock-free random access: `record(i)` reads
//! exactly one sample's bytes at `header + i * stride` without any
//! shared seek state, so concurrent readers (prefetch, staging) never
//! contend.
//!
//! A [`DataSet`](crate::data::DataSet) built on this reads storage per
//! sample instead of parsing the whole file into RAM — the "dataset
//! larger than RAM" path. The tiers above (`sample cache`, disk stage,
//! reservation staging) compose unchanged: implement
//! [`get()`](crate::data::DataSet::get) as `parse(records.record(i)?)`
//! and the framework does the rest.

use std::fs::File;
use std::path::{Path, PathBuf};

use crate::tensor::{Result, TensorError};

/// A read-only view of a file as `count` fixed-size records.
///
/// Reads are positioned (`read_exact_at` on Unix), so `record()` is
/// `&self` and safe to call from any number of threads concurrently.
pub struct FixedStrideRecords {
    file: File,
    /// Serializes seek+read on platforms without positioned reads.
    #[cfg(not(unix))]
    seek_lock: std::sync::Mutex<()>,
    stride: usize,
    header: u64,
    count: usize,
    path: PathBuf,
}

impl FixedStrideRecords {
    /// Open `path` as records of exactly `stride` bytes each.
    ///
    /// Errors loudly if the file cannot be opened or its size is not a
    /// whole number of records (a truncated or mismatched file must
    /// not be half-served).
    pub fn open(path: impl AsRef<Path>, stride: usize) -> Result<Self> {
        Self::with_header(path, stride, 0)
    }

    /// Like [`open`](Self::open), skipping `header` leading bytes
    /// before the first record (IDX-style formats).
    pub fn with_header(path: impl AsRef<Path>, stride: usize, header: u64) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if stride == 0 {
            return Err(TensorError::new(&format!(
                "FixedStrideRecords: stride must be > 0 ({})",
                path.display()
            )));
        }
        let file = File::open(&path).map_err(|e| {
            TensorError::new(&format!(
                "FixedStrideRecords: cannot open {}: {e}",
                path.display()
            ))
        })?;
        let len = file
            .metadata()
            .map_err(|e| {
                TensorError::new(&format!(
                    "FixedStrideRecords: cannot stat {}: {e}",
                    path.display()
                ))
            })?
            .len();
        let body = len.checked_sub(header).ok_or_else(|| {
            TensorError::new(&format!(
                "FixedStrideRecords: {} is {len} bytes, smaller than its {header}-byte header",
                path.display()
            ))
        })?;
        if body % stride as u64 != 0 {
            return Err(TensorError::new(&format!(
                "FixedStrideRecords: {} holds {body} record bytes, not a multiple of stride {stride} (truncated or wrong format?)",
                path.display()
            )));
        }

        Ok(FixedStrideRecords {
            file,
            #[cfg(not(unix))]
            seek_lock: std::sync::Mutex::new(()),
            stride,
            header,
            count: (body / stride as u64) as usize,
            path,
        })
    }

    /// Number of records in the file.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Bytes per record.
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// The file backing these records.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Read record `index` (exactly [`stride`](Self::stride) bytes).
    pub fn record(&self, index: usize) -> Result<Vec<u8>> {
        if index >= self.count {
            return Err(TensorError::new(&format!(
                "FixedStrideRecords: record {index} out of bounds ({} records in {})",
                self.count,
                self.path.display()
            )));
        }
        let offset = self.header + (index * self.stride) as u64;
        let mut buf = vec![0u8; self.stride];

        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            self.file.read_exact_at(&mut buf, offset).map_err(|e| {
                TensorError::new(&format!(
                    "FixedStrideRecords: read of record {index} failed at {}: {e}",
                    self.path.display()
                ))
            })?;
        }
        #[cfg(not(unix))]
        {
            use std::io::{Read, Seek, SeekFrom};
            let _guard = self
                .seek_lock
                .lock()
                .map_err(|_| TensorError::new("FixedStrideRecords: seek lock poisoned"))?;
            let mut f = &self.file;
            f.seek(SeekFrom::Start(offset))
                .and_then(|_| f.read_exact(&mut buf))
                .map_err(|e| {
                    TensorError::new(&format!(
                        "FixedStrideRecords: read of record {index} failed at {}: {e}",
                        self.path.display()
                    ))
                })?;
        }

        Ok(buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("flodl-records-tests");
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(format!("{name}-{}", std::process::id()))
    }

    fn write_file(name: &str, bytes: &[u8]) -> PathBuf {
        let path = scratch(name);
        let mut f = File::create(&path).unwrap();
        f.write_all(bytes).unwrap();
        path
    }

    #[test]
    fn reads_records_at_stride() {
        // 4 records of 3 bytes: [0,1,2], [3,4,5], ...
        let bytes: Vec<u8> = (0..12).collect();
        let path = write_file("stride", &bytes);
        let recs = FixedStrideRecords::open(&path, 3).unwrap();
        assert_eq!(recs.count(), 4);
        assert_eq!(recs.record(0).unwrap(), vec![0, 1, 2]);
        assert_eq!(recs.record(3).unwrap(), vec![9, 10, 11]);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn header_offsets_records() {
        // 2-byte header, then 2 records of 2 bytes.
        let path = write_file("header", &[0xFF, 0xFE, 1, 2, 3, 4]);
        let recs = FixedStrideRecords::with_header(&path, 2, 2).unwrap();
        assert_eq!(recs.count(), 2);
        assert_eq!(recs.record(0).unwrap(), vec![1, 2]);
        assert_eq!(recs.record(1).unwrap(), vec![3, 4]);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn misaligned_file_errors_loudly() {
        let path = write_file("misaligned", &[0; 10]);
        let err = match FixedStrideRecords::open(&path, 3) {
            Err(e) => e,
            Ok(_) => panic!("10 bytes at stride 3 must not open"),
        };
        assert!(err.to_string().contains("not a multiple of stride"));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn out_of_bounds_and_zero_stride_error() {
        let path = write_file("bounds", &[0; 6]);
        let recs = FixedStrideRecords::open(&path, 3).unwrap();
        assert!(recs.record(2).is_err());
        assert!(FixedStrideRecords::open(&path, 0).is_err());
        std::fs::remove_file(path).unwrap();
    }
}
