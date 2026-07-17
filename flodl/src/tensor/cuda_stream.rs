//! CUDA stream for async GPU operations.
//!
//! Streams represent ordered queues of GPU work. Operations on different
//! streams can execute concurrently. The default stream serializes all work.
//!
//! Common use: run GPU-to-CPU tensor copies on a non-default stream so
//! they overlap with training on the default stream.
//!
//! CUDA only. Returns error on CPU builds.
//!
//! # Usage
//!
//! ```ignore
//! let copy_stream = CudaStream::new(Device::CUDA(0), false)?;
//! {
//!     let _guard = StreamGuard::new(&copy_stream);
//!     // All CUDA ops here run on copy_stream instead of the default stream.
//!     let cpu_copy = gpu_tensor.to_device_async(Device::CPU)?;
//! }
//! // Default stream restored automatically.
//! ```

use std::ffi::c_void;
use std::ptr;

use flodl_sys as ffi;

use crate::tensor::{check_err, Device, Result, TensorError};
use super::cuda_event::CudaEvent;

/// A CUDA stream obtained from the libtorch stream pool.
///
/// RAII: the stream is returned to the pool on drop.
pub struct CudaStream {
    ptr: *mut c_void,
    device_index: i32,
}

// cudaStream_t is a device-global handle safe to reference from any thread.
unsafe impl Send for CudaStream {}

impl CudaStream {
    /// Create a new CUDA stream from the pool on the given device.
    ///
    /// `high_priority`: if true, uses a high-priority stream that preempts
    /// normal-priority work at SM boundaries.
    pub fn new(device: Device, high_priority: bool) -> Result<Self> {
        let device_index = match device {
            Device::CUDA(idx) => idx as i32,
            Device::CPU => {
                return Err(TensorError::new(
                    "CudaStream requires a CUDA device",
                ))
            }
        };
        let mut ptr: *mut c_void = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_cuda_stream_new(device_index, high_priority as i32, &mut ptr)
        };
        check_err(err)?;
        Ok(CudaStream { ptr, device_index })
    }

    /// Block the CPU thread until all work on this stream completes.
    pub fn synchronize(&self) -> Result<()> {
        let err = unsafe { ffi::flodl_cuda_stream_synchronize(self.ptr) };
        check_err(err)
    }

    /// Make this stream wait for a recorded event before executing any
    /// further work. Does not block the CPU.
    pub fn wait_event(&self, event: &CudaEvent) -> Result<()> {
        let err = unsafe {
            ffi::flodl_cuda_stream_wait_event(self.ptr, event.as_ptr())
        };
        check_err(err)
    }

    /// Non-blocking check: has all work on this stream completed?
    pub fn is_complete(&self) -> bool {
        unsafe { ffi::flodl_cuda_stream_query(self.ptr) != 0 }
    }

    /// The device this stream belongs to.
    pub fn device(&self) -> Device {
        Device::CUDA(self.device_index as u8)
    }

    /// Raw pointer for cross-module use (e.g., event.record_on).
    pub(crate) fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::flodl_cuda_stream_delete(self.ptr) };
            self.ptr = ptr::null_mut();
        }
    }
}

/// RAII guard that sets a stream as the current CUDA stream and
/// restores the **previous** stream on drop.
///
/// Nestable: inner guards restore the outer guard's stream, not the
/// default stream. This is critical for DDP where `sync_now_nccl`
/// temporarily switches to `comm_stream` inside a `compute_stream` guard.
///
/// ```ignore
/// let compute = CudaStream::new(Device::CUDA(0), false)?;
/// let comm = CudaStream::new(Device::CUDA(0), false)?;
/// {
///     let _outer = StreamGuard::new(&compute);
///     // All CUDA ops on compute_stream.
///     {
///         let _inner = StreamGuard::new(&comm);
///         // CUDA ops on comm_stream.
///     }
///     // Restored to compute_stream (not default).
/// }
/// // Restored to whatever was current before _outer.
/// ```
pub struct StreamGuard {
    /// Previous stream pointer, restored on drop. Owned (heap-allocated by
    /// `flodl_cuda_stream_get_current`), freed via `flodl_cuda_stream_delete`.
    prev: *mut std::ffi::c_void,
    device_index: i32,
}

impl StreamGuard {
    /// Set `stream` as the current CUDA stream. The previous stream
    /// is saved and restored when this guard is dropped.
    pub fn new(stream: &CudaStream) -> Self {
        let prev = unsafe { ffi::flodl_cuda_stream_get_current(stream.device_index) };
        unsafe { ffi::flodl_cuda_stream_set_current(stream.ptr) };
        StreamGuard {
            prev,
            device_index: stream.device_index,
        }
    }
}

impl Drop for StreamGuard {
    fn drop(&mut self) {
        if !self.prev.is_null() {
            unsafe { ffi::flodl_cuda_stream_set_current(self.prev) };
            unsafe { ffi::flodl_cuda_stream_delete(self.prev) };
        } else {
            unsafe { ffi::flodl_cuda_stream_restore_default(self.device_index) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::cuda_event::CudaEventFlags;
    use crate::tensor::{Tensor, test_device, test_opts};

    use std::sync::Mutex;
    static STREAM_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn test_cuda_stream_requires_cuda_device() {
        let result = CudaStream::new(Device::CPU, false);
        assert!(result.is_err(), "CudaStream::new(CPU) should fail");
    }

    #[test]
    fn test_cuda_stream_create_synchronize() {
        if !test_device().is_cuda() {
            return;
        }
        let _lock = STREAM_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let stream = CudaStream::new(test_device(), false).unwrap();
        assert_eq!(stream.device(), test_device());
        stream.synchronize().unwrap();
        assert!(stream.is_complete(), "empty stream should be complete");
    }

    #[test]
    fn test_stream_guard_restores_default() {
        if !test_device().is_cuda() {
            return;
        }
        let _lock = STREAM_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let opts = test_opts();

        let stream = CudaStream::new(test_device(), false).unwrap();
        {
            let _guard = StreamGuard::new(&stream);
            // Ops run on the non-default stream
            let _a = Tensor::randn(&[32, 32], opts).unwrap();
        }
        // Guard dropped — default stream restored.
        // Verify we can still do GPU ops normally.
        let b = Tensor::ones(&[4], opts).unwrap();
        let c = b.add(&b).unwrap();
        let vals = c.to_f32_vec().unwrap();
        assert!(vals.iter().all(|&v| (v - 2.0).abs() < 1e-5));
    }

    #[test]
    fn test_async_copy_on_stream() {
        if !test_device().is_cuda() {
            return;
        }
        let _lock = STREAM_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let opts = test_opts();

        // Create a GPU tensor with known values
        let gpu = Tensor::full(&[128], 42.0, opts).unwrap();

        // Create a non-default copy stream
        let copy_stream = CudaStream::new(test_device(), false).unwrap();

        // Record event on default stream to capture when gpu tensor is ready
        let ready = CudaEvent::new(CudaEventFlags::DisableTiming).unwrap();
        ready.record().unwrap();

        // Copy stream waits for the event, then copies
        copy_stream.wait_event(&ready).unwrap();
        let cpu_copy = {
            let _guard = StreamGuard::new(&copy_stream);
            gpu.to_device_async(Device::CPU).unwrap()
        };

        // Record completion on copy stream
        let done = CudaEvent::new(CudaEventFlags::DisableTiming).unwrap();
        done.record_on(&copy_stream).unwrap();
        done.synchronize().unwrap();

        let vals = cpu_copy.to_f32_vec().unwrap();
        assert_eq!(vals.len(), 128);
        assert!(vals.iter().all(|&v| (v - 42.0).abs() < 1e-5),
            "async copy should preserve values");
    }

    #[test]
    fn test_cross_stream_wait_event() {
        if !test_device().is_cuda() {
            return;
        }
        let _lock = STREAM_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let opts = test_opts();

        let stream_a = CudaStream::new(test_device(), false).unwrap();
        let stream_b = CudaStream::new(test_device(), false).unwrap();

        // On stream A: create a tensor
        let result = {
            let _guard = StreamGuard::new(&stream_a);
            Tensor::full(&[64], 7.0, opts).unwrap()
        };

        // Record event on stream A
        let event = CudaEvent::new(CudaEventFlags::DisableTiming).unwrap();
        event.record_on(&stream_a).unwrap();

        // Stream B waits for stream A, then reads the tensor
        stream_b.wait_event(&event).unwrap();
        let doubled = {
            let _guard = StreamGuard::new(&stream_b);
            result.add(&result).unwrap()
        };

        // Wait for stream B to finish
        stream_b.synchronize().unwrap();

        let vals = doubled.to_f32_vec().unwrap();
        assert!(vals.iter().all(|&v| (v - 14.0).abs() < 1e-5),
            "cross-stream result should be 14.0");
    }
}
