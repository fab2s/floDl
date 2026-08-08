//! Model snapshot: copy graph state to CPU for background work.
//!
//! [`Graph::snapshot_cpu()`] captures parameters, buffers, metrics, and the
//! current epoch into a [`ModelSnapshot`] with all tensors on CPU. The snapshot
//! is `Send`, so it can be moved into a [`CpuWorker`](crate::CpuWorker) closure
//! for background checkpointing, evaluation, or file I/O without blocking the
//! GPU training loop.
//!
//! **CPU-only case:** `to_device(CPU)` on an already-CPU tensor returns a tensor
//! sharing the same storage (libtorch no-op). The background thread only reads
//! snapshot data, so shared storage is safe.

use std::collections::HashMap;
use std::io::Write;

use crate::nn::checkpoint::{HASH_LEN, MAGIC, VERSION, io_err, write_tensor_data};
use crate::tensor::{Device, Result, Tensor};

use super::Graph;

/// A frozen snapshot of a graph's state, with all tensors on CPU.
///
/// Implements `Send` so it can be moved across thread boundaries.
pub struct ModelSnapshot {
    /// Named parameters, all on CPU and detached from autograd.
    pub params: HashMap<String, Tensor>,
    /// Named buffers, all on CPU.
    pub buffers: HashMap<String, Tensor>,
    /// Latest observation metrics (from `latest_metrics()`).
    pub metrics: HashMap<String, f64>,
    /// Epoch number (from `flush_count()`).
    pub epoch: usize,
}

impl ModelSnapshot {
    /// Save snapshot to a writer in the standard `.fdl` checkpoint format.
    ///
    /// Writes all parameters and buffers. The structural hash field is set to
    /// zeros (hash validation is skipped on load when the file hash is zero).
    ///
    /// Compatible with [`load_checkpoint`](crate::load_checkpoint) /
    /// [`load_checkpoint_file`](crate::load_checkpoint_file).
    pub fn save<W: Write>(&self, w: &mut W) -> Result<()> {
        w.write_all(&MAGIC).map_err(io_err)?;
        w.write_all(&VERSION.to_le_bytes()).map_err(io_err)?;
        w.write_all(&[0u8; HASH_LEN]).map_err(io_err)?;

        let total = (self.params.len() + self.buffers.len()) as u32;
        w.write_all(&total.to_le_bytes()).map_err(io_err)?;

        for (name, t) in &self.params {
            let name_bytes = name.as_bytes();
            w.write_all(&(name_bytes.len() as u32).to_le_bytes())
                .map_err(io_err)?;
            w.write_all(name_bytes).map_err(io_err)?;
            write_tensor_data(w, t)?;
        }

        for (name, t) in &self.buffers {
            let name_bytes = name.as_bytes();
            w.write_all(&(name_bytes.len() as u32).to_le_bytes())
                .map_err(io_err)?;
            w.write_all(name_bytes).map_err(io_err)?;
            write_tensor_data(w, t)?;
        }

        Ok(())
    }

    /// Save snapshot to a file. Uses gzip compression if the path ends with `.gz`.
    ///
    /// ```ignore
    /// let snap = graph.snapshot_cpu()?;
    /// worker.submit(move || {
    ///     snap.save_file("checkpoint_epoch_10.fdl.gz").unwrap();
    /// });
    /// ```
    pub fn save_file(&self, path: &str) -> Result<()> {
        let f = std::fs::File::create(path).map_err(io_err)?;
        if path.ends_with(".gz") {
            let mut w = flate2::write::GzEncoder::new(f, flate2::Compression::default());
            self.save(&mut w)?;
            w.finish().map_err(io_err)?;
            Ok(())
        } else {
            let mut w = std::io::BufWriter::new(f);
            self.save(&mut w)
        }
    }
}

impl Graph {
    /// Snapshot the graph's current state onto CPU.
    ///
    /// Parameters are detached from autograd so no grad metadata crosses
    /// the thread boundary. Buffers have no grad and are copied as-is.
    ///
    /// ```ignore
    /// let snap = graph.snapshot_cpu()?;
    /// worker.submit(move || {
    ///     snap.save_file("epoch_10.fdl.gz").unwrap();
    /// });
    /// ```
    pub fn snapshot_cpu(&self) -> Result<ModelSnapshot> {
        let mut params = HashMap::new();
        for (name, p) in self.named_parameters() {
            let t = p.variable.data().to_device(Device::CPU)?.detach()?;
            params.insert(name, t);
        }

        let mut buffers = HashMap::new();
        for (name, b) in self.named_buffers() {
            let t = b.get().to_device(Device::CPU)?;
            buffers.insert(name, t);
        }

        let metrics: HashMap<String, f64> = self.latest_metrics().into_iter().collect();
        let epoch = self.flush_count();

        Ok(ModelSnapshot {
            params,
            buffers,
            metrics,
            epoch,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::Variable;
    use crate::graph::FlowBuilder;
    use crate::nn::{Linear, Module};
    use crate::tensor::{Tensor, TensorOptions, test_device};
    use crate::worker::CpuWorker;

    fn build_test_graph() -> Result<Graph> {
        let g = FlowBuilder::from(Linear::on_device(2, 3, test_device())?)
            .tag("encoder")
            .build()?;
        Ok(g)
    }

    #[test]
    fn snapshot_captures_params_on_cpu() {
        let g = build_test_graph().unwrap();
        let snap = g.snapshot_cpu().unwrap();

        assert!(!snap.params.is_empty(), "should have params");
        for (name, t) in &snap.params {
            assert_eq!(t.device(), Device::CPU, "param {} should be on CPU", name);
        }
    }

    #[test]
    fn snapshot_captures_correct_names() {
        let g = build_test_graph().unwrap();
        let snap = g.snapshot_cpu().unwrap();

        let names: Vec<&String> = snap.params.keys().collect();
        assert!(
            names.iter().any(|n| n.contains("encoder")),
            "param names should include tag prefix, got: {:?}",
            names
        );
    }

    #[test]
    fn snapshot_captures_metrics_and_epoch() {
        let g = build_test_graph().unwrap();
        // Record and flush some metrics
        g.record_scalar("loss", 0.5);
        g.flush(&[]);
        g.record_scalar("loss", 0.3);
        g.flush(&[]);

        let snap = g.snapshot_cpu().unwrap();
        assert_eq!(snap.epoch, 2);
        assert!(snap.metrics.contains_key("loss"));
        assert!((snap.metrics["loss"] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn snapshot_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<ModelSnapshot>();
    }

    #[test]
    fn snapshot_save_roundtrip() {
        let g = build_test_graph().unwrap();
        let snap = g.snapshot_cpu().unwrap();

        // Save to buffer
        let mut buf = Vec::new();
        snap.save(&mut buf).unwrap();

        // Load back using the standard checkpoint loader
        let load_params: Vec<(String, crate::nn::Parameter)> = g.named_parameters();
        let load_buffers: Vec<(String, crate::nn::Buffer)> = g.named_buffers();
        let mut cursor = std::io::Cursor::new(&buf);
        let report =
            crate::nn::load_checkpoint(&mut cursor, &load_params, &load_buffers, None).unwrap();

        assert_eq!(report.loaded.len(), snap.params.len() + snap.buffers.len());
        assert!(report.missing.is_empty());
        assert!(report.skipped.is_empty());
    }

    #[test]
    fn snapshot_save_file_gz() {
        let g = build_test_graph().unwrap();
        let snap = g.snapshot_cpu().unwrap();

        let dir = std::env::temp_dir();
        let path = dir.join("test_snapshot.fdl.gz");
        let path_str = path.to_str().unwrap();

        snap.save_file(path_str).unwrap();

        // Verify file exists and is non-empty
        let meta = std::fs::metadata(path_str).unwrap();
        assert!(meta.len() > 0);

        // Load back
        let load_params: Vec<(String, crate::nn::Parameter)> = g.named_parameters();
        let report = crate::nn::load_checkpoint_file(path_str, &load_params, &[], None).unwrap();
        assert_eq!(report.loaded.len(), snap.params.len());

        std::fs::remove_file(path_str).ok();
    }

    #[test]
    fn snapshot_in_cpu_worker() {
        let g = build_test_graph().unwrap();

        // Do a forward pass so params have data
        let x = Variable::new(
            Tensor::randn(
                &[1, 2],
                TensorOptions {
                    dtype: crate::tensor::DType::Float32,
                    device: test_device(),
                },
            )
            .unwrap(),
            false,
        );
        let _ = g.forward(&x).unwrap();

        let snap = g.snapshot_cpu().unwrap();
        let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let done2 = done.clone();

        let mut worker = CpuWorker::new();
        worker.submit(move || {
            // Verify we can access snapshot data on the worker thread
            assert!(!snap.params.is_empty());
            for t in snap.params.values() {
                assert_eq!(t.device(), Device::CPU);
            }
            done2.store(true, std::sync::atomic::Ordering::Release);
        });
        worker.finish();

        assert!(done.load(std::sync::atomic::Ordering::Acquire));
    }
}
