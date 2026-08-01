//! CUDA Graph capture and replay.
//!
//! CUDA Graphs record a sequence of GPU kernel launches into a replayable
//! graph, eliminating per-kernel dispatch overhead. This is most impactful
//! for models with many small sequential kernels (RNNs, GRU steps).
//!
//! **Constraints:**
//! - All tensor shapes must be identical between capture and replay.
//! - No CPU-GPU sync during capture (no `.item()`, no `cuda_synchronize()`).
//! - No conditional control flow during capture.
//! - Warmup runs needed before capture (cuDNN autotuner, allocator).
//! - CUDA only — returns error on CPU.
//!
//! # Usage
//!
//! ```ignore
//! // 1. Static tensors (shapes fixed)
//! let static_input = Tensor::zeros(&[B, D], cuda_opts)?;
//! let static_target = Tensor::zeros(&[B, C], cuda_opts)?;
//!
//! // 2. Capture (with 3 warmup runs)
//! let graph = cuda_graph_capture(3, None, || {
//!     let inp = Variable::new(static_input.clone(), true);
//!     let tgt = Variable::new(static_target.clone(), false);
//!     let pred = model.forward(&inp)?;
//!     let loss = mse_loss(&pred, &tgt)?;
//!     optimizer.zero_grad();
//!     loss.backward()?;
//!     optimizer.step()?;
//!     Ok(())
//! })?;
//!
//! // 3. Training loop — copy new data, replay
//! for (x, y) in batches {
//!     static_input.copy_(&x, true)?;
//!     static_target.copy_(&y, true)?;
//!     graph.replay()?;
//! }
//! ```

use std::ffi::c_void;
use std::ptr;

use flodl_sys as ffi;

use crate::tensor::{check_err, Result};

/// Memory pool identifier for sharing allocations between CUDA graphs.
///
/// When multiple graphs share a pool, they can reuse each other's memory
/// allocations, reducing peak VRAM usage. Use `cuda_graph_pool_handle()`
/// to get a shared pool, or `None` for a graph-private pool.
#[derive(Clone, Copy, Debug, Default)]
pub struct MemPoolId {
    pub hi: u64,
    pub lo: u64,
}

/// CUDA stream capture mode.
///
/// Controls how the CUDA runtime validates captured operations.
#[derive(Clone, Copy, Debug, Default)]
#[repr(i32)]
pub enum CaptureMode {
    /// Global: strictest, no other CUDA work allowed on any stream during capture.
    Global = 0,
    /// Thread-local: only the capturing thread's stream is locked.
    ThreadLocal = 1,
    /// Relaxed: minimal validation, caller ensures safety.
    /// Default — allows other threads to use CUDA freely during capture.
    #[default]
    Relaxed = 2,
}

/// A captured CUDA graph that can be replayed with minimal dispatch overhead.
///
/// Created via [`cuda_graph_capture`] or manually with `new()` + `capture_begin()`
/// / `capture_end()`.
pub struct CudaGraph {
    ptr: *mut c_void,
}

impl CudaGraph {
    /// Create a new empty CUDA graph. Errors on CPU-only builds.
    pub fn new() -> Result<Self> {
        let mut ptr: *mut c_void = ptr::null_mut();
        let err = unsafe { ffi::flodl_cuda_graph_new(&mut ptr) };
        check_err(err)?;
        Ok(CudaGraph { ptr })
    }

    /// Begin capturing GPU operations into this graph.
    ///
    /// All CUDA kernel launches on the current stream after this call
    /// will be recorded until [`capture_end`](CudaGraph::capture_end).
    pub fn capture_begin(&mut self, pool: Option<MemPoolId>, mode: CaptureMode) -> Result<()> {
        let (hi, lo) = pool.map_or((0, 0), |p| (p.hi, p.lo));
        let err = unsafe {
            ffi::flodl_cuda_graph_capture_begin(self.ptr, hi, lo, mode as i32)
        };
        check_err(err)
    }

    /// End capture and finalize the graph.
    pub fn capture_end(&mut self) -> Result<()> {
        let err = unsafe { ffi::flodl_cuda_graph_capture_end(self.ptr) };
        check_err(err)
    }

    /// Replay the captured graph. All kernels are launched with a single
    /// CUDA API call, eliminating per-kernel dispatch overhead.
    pub fn replay(&self) -> Result<()> {
        let err = unsafe { ffi::flodl_cuda_graph_replay(self.ptr) };
        check_err(err)
    }

    /// Reset the graph, allowing recapture.
    pub fn reset(&mut self) -> Result<()> {
        let err = unsafe { ffi::flodl_cuda_graph_reset(self.ptr) };
        check_err(err)
    }

    /// Get the memory pool ID for this graph.
    pub fn pool(&self) -> MemPoolId {
        let mut hi: u64 = 0;
        let mut lo: u64 = 0;
        unsafe { ffi::flodl_cuda_graph_pool(self.ptr, &mut hi, &mut lo) };
        MemPoolId { hi, lo }
    }
}

impl Drop for CudaGraph {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::flodl_cuda_graph_delete(self.ptr) };
            self.ptr = ptr::null_mut();
        }
    }
}

/// Get a shared memory pool handle for multiple CUDA graphs.
///
/// Graphs captured with the same pool can reuse each other's allocations,
/// reducing peak VRAM usage when replaying multiple graphs.
pub fn cuda_graph_pool_handle() -> MemPoolId {
    let mut hi: u64 = 0;
    let mut lo: u64 = 0;
    unsafe { ffi::flodl_cuda_graph_pool_handle(&mut hi, &mut lo) };
    MemPoolId { hi, lo }
}

/// Capture a CUDA graph from a closure.
///
/// Runs `warmup_runs` iterations of `f` first (for cuDNN autotuner and
/// allocator warmup), synchronizes, then captures one run into a graph.
///
/// ```ignore
/// let graph = cuda_graph_capture(3, None, || {
///     optimizer.zero_grad();
///     let pred = model.forward(&input)?;
///     let loss = mse_loss(&pred, &target)?;
///     loss.backward()?;
///     optimizer.step()
/// })?;
/// ```
pub fn cuda_graph_capture<F>(
    warmup_runs: usize,
    pool: Option<MemPoolId>,
    mut f: F,
) -> Result<CudaGraph>
where
    F: FnMut() -> Result<()>,
{
    // Warmup on the default stream first — this lets cuDNN autotuner
    // settle before we switch streams for capture.
    for _ in 0..warmup_runs {
        f()?;
    }

    // Synchronize before capture — on the CURRENT device, not a hardcoded
    // device 0. Capture runs `f` on whatever device is current (the model's);
    // syncing device 0 on a multi-GPU rig would let capture start before the
    // real capture device's warmup work has drained.
    crate::tensor::cuda_synchronize(crate::tensor::current_cuda_device());

    // Capture: capture_begin switches to a side stream internally,
    // and capture_end restores the default stream.
    let mut graph = CudaGraph::new()?;
    graph.capture_begin(pool, CaptureMode::default())?;
    f()?;
    graph.capture_end()?;

    Ok(graph)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Tensor, test_device, test_opts};

    // CUDA graph capture changes the per-thread CUDA stream, which can
    // interfere with other tests running on the same thread. Serialize
    // graph tests to avoid contention.
    use std::sync::Mutex;
    static GRAPH_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn test_copy_basic() {
        let opts = test_opts();
        let src = Tensor::ones(&[3, 4], opts).unwrap();
        let dst = Tensor::zeros(&[3, 4], opts).unwrap();
        dst.copy_(&src, false).unwrap();

        let buf = dst.to_f32_vec().unwrap();
        assert!(buf.iter().all(|&v| v == 1.0), "copy_ should have filled dst with 1.0");
    }

    #[test]
    fn test_cuda_graph_fails_on_cpu() {
        if test_device().is_cuda() {
            return; // skip on GPU — it should succeed there
        }
        let result = CudaGraph::new();
        assert!(result.is_err(), "CudaGraph::new() should fail on CPU");
    }

    #[test]
    #[ignore = "CUDA graph capture blocks device-wide RNG; run with: fdl gpu-test-graph"]
    fn test_cuda_graph_capture_replay() {
        if !test_device().is_cuda() {
            return;
        }
        let _lock = GRAPH_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let opts = test_opts();

        // Static tensors
        let a = Tensor::ones(&[4, 4], opts).unwrap();
        let b = Tensor::ones(&[4, 4], opts).unwrap();
        let c = Tensor::zeros(&[4, 4], opts).unwrap();

        // Capture: c = a + b
        let graph = cuda_graph_capture(1, None, || {
            let sum = a.add(&b)?;
            c.copy_(&sum, false)?;
            Ok(())
        }).unwrap();

        // Replay
        graph.replay().unwrap();
        crate::tensor::cuda_synchronize(0);

        let buf = c.to_f32_vec().unwrap();
        assert!(buf.iter().all(|&v| (v - 2.0).abs() < 1e-5),
            "c should be 2.0 after replay, got {:?}", &buf[..4]);
    }

    #[test]
    #[ignore = "CUDA graph capture blocks device-wide RNG; run with: fdl gpu-test-graph"]
    fn test_cuda_graph_with_model() {
        if !test_device().is_cuda() {
            return;
        }
        let _lock = GRAPH_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use crate::autograd::Variable;
        use crate::nn::{Linear, Module, mse_loss, Adam, Optimizer};

        let dev = test_device();
        let model = Linear::on_device(4, 2, dev).unwrap();
        let params = model.parameters();
        let mut optimizer = Adam::new(&params, 0.01);

        // Record initial param values
        let init_data = params[0].variable.data().to_f32_vec().unwrap();

        // Static input/target
        let opts = test_opts();
        let static_input = Tensor::randn(&[8, 4], opts).unwrap();
        let static_target = Tensor::randn(&[8, 2], opts).unwrap();

        let graph = cuda_graph_capture(3, None, || {
            let inp = Variable::new(static_input.clone(), true);
            let tgt = Variable::new(static_target.clone(), false);
            optimizer.zero_grad();
            let pred = model.forward(&inp)?;
            let loss = mse_loss(&pred, &tgt)?;
            loss.backward()?;
            optimizer.step()
        }).unwrap();

        // Replay a few times
        for _ in 0..5 {
            graph.replay().unwrap();
        }
        crate::tensor::cuda_synchronize(0);

        // Params should have changed
        let final_data = params[0].variable.data().to_f32_vec().unwrap();
        assert_ne!(init_data, final_data, "params should have changed after graph replay");
    }

    #[test]
    #[ignore = "CUDA graph capture blocks device-wide RNG; run with: fdl gpu-test-graph"]
    fn test_cuda_graph_pool_handle() {
        if !test_device().is_cuda() {
            return;
        }
        let _lock = GRAPH_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let pool = cuda_graph_pool_handle();
        assert!(pool.hi != 0 || pool.lo != 0, "pool handle should be nonzero");
    }

    #[test]
    #[ignore = "CUDA graph capture blocks device-wide RNG; run with: fdl gpu-test-graph"]
    fn test_cuda_graph_reset_recapture() {
        if !test_device().is_cuda() {
            return;
        }
        let _lock = GRAPH_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let opts = test_opts();

        let a = Tensor::ones(&[4], opts).unwrap();
        let b = Tensor::ones(&[4], opts).unwrap();
        let c = Tensor::zeros(&[4], opts).unwrap();

        // First capture: c = a + b (= 2.0)
        let mut graph = cuda_graph_capture(1, None, || {
            let sum = a.add(&b)?;
            c.copy_(&sum, false)?;
            Ok(())
        }).unwrap();

        graph.replay().unwrap();
        crate::tensor::cuda_synchronize(0);
        let buf = c.to_f32_vec().unwrap();
        assert!(buf.iter().all(|&v| (v - 2.0).abs() < 1e-5));

        // Reset and recapture: c = a * 3
        graph.reset().unwrap();

        let three = Tensor::full(&[4], 3.0, opts).unwrap();
        graph.capture_begin(None, CaptureMode::default()).unwrap();
        let prod = a.mul(&three).unwrap();
        c.copy_(&prod, false).unwrap();
        graph.capture_end().unwrap();

        graph.replay().unwrap();
        crate::tensor::cuda_synchronize(0);
        let buf = c.to_f32_vec().unwrap();
        assert!(buf.iter().all(|&v| (v - 3.0).abs() < 1e-5),
            "after recapture, c should be 3.0, got {:?}", &buf[..4]);
    }
}
