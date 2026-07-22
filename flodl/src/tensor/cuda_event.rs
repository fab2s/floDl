//! CUDA event for GPU synchronization and timing.
//!
//! Events are markers recorded on a CUDA stream. They enable:
//! - **Cross-stream synchronization**: one stream waits for another's event.
//! - **GPU timing**: measure elapsed time between two recorded events.
//! - **Non-blocking completion checks**: poll whether GPU work has finished.
//!
//! CUDA only. Returns error on CPU builds.
//!
//! # Usage
//!
//! ```ignore
//! let start = CudaEvent::new(CudaEventFlags::Default)?;
//! let end = CudaEvent::new(CudaEventFlags::Default)?;
//! start.record()?;
//! // ... GPU work ...
//! end.record()?;
//! end.synchronize()?;
//! let ms = CudaEvent::elapsed_time(&start, &end)?;
//! ```

use std::ffi::c_void;
use std::ptr;

use flodl_sys as ffi;

use crate::tensor::{check_err, Result};

/// Flags controlling CUDA event behavior.
#[derive(Clone, Copy, Debug)]
#[repr(i32)]
pub enum CudaEventFlags {
    /// Timing enabled. Required for [`CudaEvent::elapsed_time`].
    Default = 0,
    /// Timing disabled. Lower overhead for pure synchronization.
    DisableTiming = 1,
}

/// A CUDA event for stream synchronization and GPU timing.
///
/// RAII: the underlying CUDA event is destroyed on drop.
pub struct CudaEvent {
    ptr: *mut c_void,
}

// cudaEvent_t is a device-global handle safe to record/query/wait from any thread.
unsafe impl Send for CudaEvent {}

impl CudaEvent {
    /// Create a new CUDA event with the given flags.
    pub fn new(flags: CudaEventFlags) -> Result<Self> {
        let mut ptr: *mut c_void = ptr::null_mut();
        let err = unsafe { ffi::flodl_cuda_event_new(flags as i32, &mut ptr) };
        check_err(err)?;
        Ok(CudaEvent { ptr })
    }

    /// Record this event on the current CUDA stream.
    pub fn record(&self) -> Result<()> {
        let err = unsafe { ffi::flodl_cuda_event_record(self.ptr) };
        check_err(err)
    }

    /// Record this event on a specific CUDA stream.
    pub fn record_on(&self, stream: &super::cuda_stream::CudaStream) -> Result<()> {
        let err = unsafe {
            ffi::flodl_cuda_event_record_on_stream(self.ptr, stream.as_ptr())
        };
        check_err(err)
    }

    /// Block the calling CPU thread until all GPU work before this event completes.
    pub fn synchronize(&self) -> Result<()> {
        let err = unsafe { ffi::flodl_cuda_event_synchronize(self.ptr) };
        check_err(err)
    }

    /// Non-blocking check: has all GPU work before this event completed?
    pub fn is_complete(&self) -> bool {
        unsafe { ffi::flodl_cuda_event_query(self.ptr) != 0 }
    }

    /// Elapsed time in milliseconds between two recorded events.
    ///
    /// Both events must have been created with [`CudaEventFlags::Default`]
    /// (timing enabled) and both must have been recorded and completed.
    pub fn elapsed_time(start: &CudaEvent, end: &CudaEvent) -> Result<f32> {
        let mut ms: f32 = 0.0;
        let err = unsafe {
            ffi::flodl_cuda_event_elapsed_time(start.ptr, end.ptr, &mut ms)
        };
        check_err(err)?;
        Ok(ms)
    }

    /// Raw pointer for cross-module use (e.g., stream.wait_event).
    pub(crate) fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::flodl_cuda_event_delete(self.ptr) };
            self.ptr = ptr::null_mut();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Tensor, test_device, test_opts};

    #[test]
    fn test_cuda_event_create_cpu() {
        if test_device().is_cuda() {
            return; // skip on GPU — it should succeed there
        }
        let result = CudaEvent::new(CudaEventFlags::Default);
        assert!(result.is_err(), "CudaEvent::new() should fail on CPU");
    }

    use std::sync::Mutex;
    static EVENT_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn test_cuda_event_record_synchronize() {
        if !test_device().is_cuda() {
            return;
        }
        let _lock = EVENT_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let event = CudaEvent::new(CudaEventFlags::DisableTiming).unwrap();

        // Launch some GPU work
        let opts = test_opts();
        let _a = Tensor::randn(&[64, 64], opts).unwrap();

        event.record().unwrap();
        event.synchronize().unwrap();
        assert!(event.is_complete(), "event should be complete after synchronize");
    }

    #[test]
    fn test_cuda_event_elapsed_time() {
        if !test_device().is_cuda() {
            return;
        }
        let _lock = EVENT_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let start = CudaEvent::new(CudaEventFlags::Default).unwrap();
        let end = CudaEvent::new(CudaEventFlags::Default).unwrap();

        let opts = test_opts();
        start.record().unwrap();

        // Do some GPU work
        let a = Tensor::randn(&[256, 256], opts).unwrap();
        let b = Tensor::randn(&[256, 256], opts).unwrap();
        let _c = a.matmul(&b).unwrap();

        end.record().unwrap();
        end.synchronize().unwrap();

        let ms = CudaEvent::elapsed_time(&start, &end).unwrap();
        assert!(ms >= 0.0, "elapsed time should be non-negative, got {ms}");
    }

    #[test]
    fn test_cuda_event_disable_timing_elapsed_fails() {
        if !test_device().is_cuda() {
            return;
        }
        let _lock = EVENT_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let start = CudaEvent::new(CudaEventFlags::DisableTiming).unwrap();
        let end = CudaEvent::new(CudaEventFlags::DisableTiming).unwrap();

        start.record().unwrap();
        end.record().unwrap();
        end.synchronize().unwrap();

        let result = CudaEvent::elapsed_time(&start, &end);
        assert!(result.is_err(), "elapsed_time should fail with DisableTiming events");
    }
}
