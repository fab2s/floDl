//! NCCL collective operations for multi-GPU communication.
//!
//! Provides AllReduce, Broadcast, and other collective ops across
//! multiple CUDA devices within a single process. Built directly
//! on NCCL for minimal overhead.
//!
//! CUDA only. Requires 2+ GPUs at runtime.
//!
//! # Usage
//!
//! ```ignore
//! let comms = NcclComms::new(&[Device::CUDA(0), Device::CUDA(1)])?;
//!
//! // Broadcast initial parameters from device 0 to device 1
//! comms.broadcast(&[&tensor_dev0, &tensor_dev1], 0)?;
//!
//! // AllReduce gradients (sum across devices)
//! comms.all_reduce(&[&grad_dev0, &grad_dev1], ReduceOp::Sum)?;
//! ```

use std::ffi::c_void;
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use flodl_sys::{self as ffi, FlodlTensor};

use crate::tensor::{
    check_err, current_cuda_device, set_current_cuda_device,
    Device, Result, Tensor, TensorError,
};
use super::cuda_stream::CudaStream;

/// NCCL reduction operation.
#[derive(Clone, Copy, Debug)]
#[repr(i32)]
pub enum ReduceOp {
    /// Element-wise sum across devices.
    Sum = 0,
    /// Element-wise product across devices.
    Prod = 1,
    /// Element-wise maximum across devices.
    Max = 2,
    /// Element-wise minimum across devices.
    Min = 3,
    /// Element-wise average across devices.
    Avg = 4,
}

/// NCCL communicator group for multi-GPU collective operations.
///
/// Holds one communicator per device. All collective ops operate
/// across all devices in the group simultaneously.
///
/// RAII: communicators are destroyed on drop.
pub struct NcclComms {
    handle: *mut c_void,
    devices: Vec<Device>,
}

// NcclComms can be sent between threads. The underlying ncclComm_t handles
// are used from the thread that calls the collective ops (with GroupStart/End).
unsafe impl Send for NcclComms {}

impl NcclComms {
    /// Create from a raw handle and device list. Used internally for testing.
    ///
    /// # Safety
    /// Caller must ensure `handle` is a valid NCCL communicator handle
    /// (or null for mock/test use). Drop on null handle is a no-op.
    #[cfg(test)]
    pub(crate) unsafe fn from_raw(handle: *mut c_void, devices: Vec<Device>) -> Self {
        NcclComms { handle, devices }
    }

    /// Initialize NCCL communicators for the given CUDA devices.
    ///
    /// All devices must be distinct CUDA devices. Returns error on CPU
    /// builds or if NCCL initialization fails.
    pub fn new(devices: &[Device]) -> Result<Self> {
        if devices.len() < 2 {
            return Err(TensorError::new(
                "NcclComms requires at least 2 devices",
            ));
        }
        let mut devlist: Vec<i32> = Vec::with_capacity(devices.len());
        for &dev in devices {
            match dev {
                Device::CUDA(idx) => devlist.push(idx as i32),
                Device::CPU => {
                    return Err(TensorError::new(
                        "NcclComms requires CUDA devices, got CPU",
                    ))
                }
            }
        }

        let mut handle: *mut c_void = ptr::null_mut();
        // NCCL init calls cudaSetDevice internally. Save/restore so we
        // don't corrupt the caller's device context.
        let saved = current_cuda_device();
        let err = unsafe {
            ffi::flodl_nccl_init(
                devlist.len() as i32,
                devlist.as_ptr(),
                &mut handle,
            )
        };
        set_current_cuda_device(saved);
        check_err(err)?;
        Ok(NcclComms {
            handle,
            devices: devices.to_vec(),
        })
    }

    /// In-place AllReduce across all devices using default streams.
    ///
    /// Each tensor must reside on its corresponding device and all tensors
    /// must have the same shape and dtype. After completion, every tensor
    /// holds the reduced result.
    ///
    /// # Parameters
    ///
    /// - `tensors`: one tensor per device (order matches `devices()`). Modified in-place.
    /// - `op`: reduction operation applied element-wise (e.g. `ReduceOp::Sum`).
    pub fn all_reduce(&self, tensors: &[&Tensor], op: ReduceOp) -> Result<()> {
        self.validate_tensors(tensors, "all_reduce")?;
        let mut handles: Vec<FlodlTensor> = tensors.iter().map(|t| t.handle).collect();
        let saved = current_cuda_device();
        let err = unsafe {
            ffi::flodl_nccl_all_reduce(
                self.handle,
                handles.as_mut_ptr(),
                ptr::null_mut(),
                op as i32,
            )
        };
        set_current_cuda_device(saved);
        check_err(err)
    }

    /// In-place AllReduce on explicit CUDA streams (for overlapping with compute).
    ///
    /// Same semantics as [`all_reduce`](Self::all_reduce), but each rank's
    /// NCCL work is enqueued on the provided stream instead of the default stream.
    ///
    /// # Parameters
    ///
    /// - `tensors`: one tensor per device (order matches `devices()`). Modified in-place.
    /// - `op`: reduction operation applied element-wise.
    /// - `streams`: one stream per device; each must belong to its corresponding device.
    pub fn all_reduce_on_streams(
        &self,
        tensors: &[&Tensor],
        op: ReduceOp,
        streams: &[&CudaStream],
    ) -> Result<()> {
        self.validate_tensors(tensors, "all_reduce_on_streams")?;
        if streams.len() != self.devices.len() {
            return Err(TensorError::new(&format!(
                "all_reduce_on_streams: expected {} streams, got {}",
                self.devices.len(), streams.len()
            )));
        }
        let mut handles: Vec<FlodlTensor> = tensors.iter().map(|t| t.handle).collect();
        let mut stream_ptrs: Vec<*mut c_void> = streams.iter().map(|s| s.as_ptr()).collect();
        let saved = current_cuda_device();
        let err = unsafe {
            ffi::flodl_nccl_all_reduce(
                self.handle,
                handles.as_mut_ptr(),
                stream_ptrs.as_mut_ptr(),
                op as i32,
            )
        };
        set_current_cuda_device(saved);
        check_err(err)
    }

    /// Broadcast tensor from `root` device to all others (in-place).
    ///
    /// After completion, all tensors hold the value that was on `tensors[root]`.
    ///
    /// # Parameters
    ///
    /// - `tensors`: one tensor per device (order matches `devices()`). All are
    ///   overwritten in-place with the value from `tensors[root]`.
    /// - `root`: index into `tensors`/`devices()` of the source rank.
    pub fn broadcast(&self, tensors: &[&Tensor], root: usize) -> Result<()> {
        self.validate_tensors(tensors, "broadcast")?;
        if root >= self.devices.len() {
            return Err(TensorError::new(&format!(
                "broadcast: root {} out of range (have {} devices)",
                root, self.devices.len()
            )));
        }
        let mut handles: Vec<FlodlTensor> = tensors.iter().map(|t| t.handle).collect();
        let saved = current_cuda_device();
        let err = unsafe {
            ffi::flodl_nccl_broadcast(
                self.handle,
                handles.as_mut_ptr(),
                ptr::null_mut(),
                root as i32,
            )
        };
        set_current_cuda_device(saved);
        check_err(err)
    }

    /// Broadcast on explicit CUDA streams (for overlapping with compute).
    ///
    /// Same semantics as [`broadcast`](Self::broadcast), but each rank's
    /// NCCL work is enqueued on the provided stream instead of the default stream.
    ///
    /// # Parameters
    ///
    /// - `tensors`: one tensor per device. All are overwritten in-place.
    /// - `root`: index of the source rank.
    /// - `streams`: one stream per device; each must belong to its corresponding device.
    pub fn broadcast_on_streams(
        &self,
        tensors: &[&Tensor],
        root: usize,
        streams: &[&CudaStream],
    ) -> Result<()> {
        self.validate_tensors(tensors, "broadcast_on_streams")?;
        if root >= self.devices.len() {
            return Err(TensorError::new(&format!(
                "broadcast_on_streams: root {} out of range", root
            )));
        }
        if streams.len() != self.devices.len() {
            return Err(TensorError::new(&format!(
                "broadcast_on_streams: expected {} streams, got {}",
                self.devices.len(), streams.len()
            )));
        }
        let mut handles: Vec<FlodlTensor> = tensors.iter().map(|t| t.handle).collect();
        let mut stream_ptrs: Vec<*mut c_void> = streams.iter().map(|s| s.as_ptr()).collect();
        let saved = current_cuda_device();
        let err = unsafe {
            ffi::flodl_nccl_broadcast(
                self.handle,
                handles.as_mut_ptr(),
                stream_ptrs.as_mut_ptr(),
                root as i32,
            )
        };
        set_current_cuda_device(saved);
        check_err(err)
    }

    /// Number of devices in this communicator group.
    pub fn size(&self) -> usize {
        self.devices.len()
    }

    /// Devices in this communicator group.
    pub fn devices(&self) -> &[Device] {
        &self.devices
    }

    fn validate_tensors(&self, tensors: &[&Tensor], op: &str) -> Result<()> {
        if tensors.len() != self.devices.len() {
            return Err(TensorError::new(&format!(
                "{}: expected {} tensors (one per device), got {}",
                op, self.devices.len(), tensors.len()
            )));
        }
        Ok(())
    }

    /// Split this communicator group into individual per-rank communicators.
    ///
    /// Returns one [`NcclRankComm`] per device. Ownership of each rank's
    /// internal communicator is transferred; this group becomes empty and
    /// should be dropped (its destructor is a no-op for extracted ranks).
    ///
    /// This is the **recommended way** to create per-thread communicators for
    /// multi-threaded DDP. Calling `ncclCommInitRank` from worker threads
    /// corrupts the CUDA context on heterogeneous GPU setups (e.g. mixing
    /// GPU architectures), causing `cudaErrorNoKernelImageForDevice` on
    /// subsequent kernel launches. The init-on-main + split pattern avoids this:
    ///
    /// ```ignore
    /// // Main thread: safe single-thread init
    /// let group = NcclComms::new(&[Device::CUDA(0), Device::CUDA(1)])?;
    /// let rank_comms = group.split()?;
    ///
    /// // Distribute to worker threads
    /// let comm0 = rank_comms.into_iter().nth(0).unwrap(); // -> thread 0
    /// let comm1 = rank_comms.into_iter().nth(1).unwrap(); // -> thread 1
    /// ```
    pub fn split(self) -> Result<Vec<NcclRankComm>> {
        let mut comms = Vec::with_capacity(self.devices.len());
        for i in 0..self.devices.len() {
            let mut rank_handle: *mut c_void = ptr::null_mut();
            let err = unsafe {
                ffi::flodl_nccl_split_rank(
                    self.handle,
                    i as i32,
                    &mut rank_handle,
                )
            };
            check_err(err)?;
            let abort_handle = Arc::new(NcclAbortHandle {
                ptr: rank_handle,
                aborted: AtomicBool::new(false),
            });
            comms.push(NcclRankComm {
                handle: rank_handle,
                rank: i,
                world_size: self.devices.len(),
                abort_handle,
            });
        }
        Ok(comms)
    }
}

impl Drop for NcclComms {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { ffi::flodl_nccl_destroy(self.handle) };
            self.handle = ptr::null_mut();
        }
    }
}

// ---------------------------------------------------------------------------
// Per-Rank NCCL (for multi-threaded DDP)
// ---------------------------------------------------------------------------

/// Size of an NCCL unique ID in bytes.
pub const NCCL_UNIQUE_ID_BYTES: usize = 128;

/// Opaque unique ID for NCCL communicator initialization.
///
/// Generated once on any thread, then shared (via clone) with all ranks.
/// Each rank passes its copy to [`NcclRankComm::init_rank`].
#[derive(Clone)]
pub struct NcclUniqueId {
    bytes: [u8; NCCL_UNIQUE_ID_BYTES],
}

// NcclUniqueId is just bytes, safe to send/share.
unsafe impl Send for NcclUniqueId {}
unsafe impl Sync for NcclUniqueId {}

impl NcclUniqueId {
    /// Generate a new unique ID for NCCL communicator initialization.
    ///
    /// Call once on any thread, then clone and distribute to all ranks.
    pub fn new() -> Result<Self> {
        let mut bytes = [0u8; NCCL_UNIQUE_ID_BYTES];
        let err = unsafe { ffi::flodl_nccl_get_unique_id(bytes.as_mut_ptr()) };
        check_err(err)?;
        Ok(NcclUniqueId { bytes })
    }

    /// Raw bytes of the unique ID.
    pub fn as_bytes(&self) -> &[u8; NCCL_UNIQUE_ID_BYTES] {
        &self.bytes
    }
}

impl std::fmt::Debug for NcclUniqueId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Don't dump 128 bytes; just show it exists
        f.debug_struct("NcclUniqueId").finish()
    }
}

/// Thread-safe handle for aborting an NCCL communicator from any thread.
///
/// When a worker thread is stuck in an NCCL collective (e.g. AllReduce waiting
/// for a dead rank), calling [`abort`](Self::abort) from any thread unblocks it.
/// The aborted collective returns an error, and the communicator is destroyed.
///
/// Obtained via [`NcclRankComm::abort_handle`]. Multiple clones share the same
/// underlying communicator pointer.
pub struct NcclAbortHandle {
    ptr: *mut c_void,
    aborted: AtomicBool,
}

// SAFETY: ncclCommAbort is explicitly documented as thread-safe.
// The raw pointer is only used for the abort FFI call.
unsafe impl Send for NcclAbortHandle {}
unsafe impl Sync for NcclAbortHandle {}

impl NcclAbortHandle {
    /// Abort the communicator, unblocking any in-progress collective.
    ///
    /// Thread-safe and idempotent. After abort, the communicator is destroyed;
    /// the owning [`NcclRankComm`]'s Drop becomes a no-op.
    pub fn abort(&self) -> Result<()> {
        if self.aborted.swap(true, Ordering::AcqRel) {
            return Ok(()); // already aborted
        }
        let err = unsafe { ffi::flodl_nccl_abort_rank(self.ptr) };
        check_err(err)
    }

    /// Whether this communicator has been aborted or destroyed.
    pub fn is_aborted(&self) -> bool {
        self.aborted.load(Ordering::Acquire)
    }

    /// Mark as handled (comm already destroyed by normal Drop).
    /// Future `abort()` calls become no-ops, preventing use-after-free.
    fn mark_destroyed(&self) {
        self.aborted.store(true, Ordering::Release);
    }
}

/// Single-rank NCCL communicator for multi-threaded DDP.
///
/// **Preferred creation path:** [`NcclComms::new`] + [`NcclComms::split`].
/// This initializes all communicators from a single thread via `ncclCommInitAll`,
/// then splits them for distribution to worker threads. This avoids CUDA context
/// corruption that occurs when `ncclCommInitRank` is called from multiple threads
/// on heterogeneous GPU setups.
///
/// [`init_rank`](Self::init_rank) is provided for multi-process DDP (one process
/// per GPU) where the CUDA context issue does not apply.
///
/// Collective operations (e.g. [`all_reduce`](Self::all_reduce)) must be called
/// concurrently by all ranks in the communicator for the collective to complete.
///
/// RAII: the communicator is destroyed on drop.
pub struct NcclRankComm {
    handle: *mut c_void,
    rank: usize,
    world_size: usize,
    abort_handle: Arc<NcclAbortHandle>,
}

// NcclRankComm can be sent between threads (though typically stays in its GPU thread).
unsafe impl Send for NcclRankComm {}

impl NcclRankComm {
    /// Initialize this rank's communicator for multi-process DDP.
    ///
    /// The caller must set the CUDA device for this rank before calling
    /// (via `set_current_cuda_device`). All ranks must call this concurrently.
    ///
    /// For single-process multi-GPU, prefer [`NcclComms::new`] + [`NcclComms::split`]
    /// to avoid CUDA context corruption on heterogeneous GPU setups.
    ///
    /// # Parameters
    ///
    /// - `rank`: this process's rank (0-indexed).
    /// - `world_size`: total number of ranks in the communicator group.
    /// - `uid`: shared unique ID generated by [`NcclUniqueId::new`] and distributed
    ///   to all ranks (e.g. via MPI broadcast or shared memory).
    pub fn init_rank(rank: usize, world_size: usize, uid: &NcclUniqueId) -> Result<Self> {
        if rank >= world_size {
            return Err(TensorError::new(&format!(
                "NcclRankComm: rank {} >= world_size {}", rank, world_size
            )));
        }
        if world_size < 2 {
            return Err(TensorError::new(
                "NcclRankComm requires world_size >= 2"
            ));
        }
        let mut handle: *mut c_void = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_nccl_init_rank(
                rank as i32,
                world_size as i32,
                uid.bytes.as_ptr(),
                &mut handle,
            )
        };
        check_err(err)?;
        let abort_handle = Arc::new(NcclAbortHandle {
            ptr: handle,
            aborted: AtomicBool::new(false),
        });
        Ok(NcclRankComm { handle, rank, world_size, abort_handle })
    }

    /// This rank's index.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Total number of ranks in the communicator.
    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// Get a thread-safe abort handle for this communicator.
    ///
    /// The handle can be sent to another thread and used to abort a stuck
    /// collective operation (e.g. AllReduce waiting for a dead rank).
    pub fn abort_handle(&self) -> Arc<NcclAbortHandle> {
        self.abort_handle.clone()
    }

    /// In-place AllReduce on this rank's tensors using the default stream.
    ///
    /// All tensors must be on this rank's device. All ranks must call this
    /// concurrently with the same number of tensors for the collective to complete.
    ///
    /// # Parameters
    ///
    /// - `tensors`: one or more tensors on this rank's device. Modified in-place.
    ///   When multiple tensors are provided, each is reduced independently (batched
    ///   inside a single NCCL group call for efficiency).
    /// - `op`: reduction operation applied element-wise (e.g. `ReduceOp::Avg`).
    pub fn all_reduce(&self, tensors: &[&Tensor], op: ReduceOp) -> Result<()> {
        let mut handles: Vec<ffi::FlodlTensor> = tensors.iter().map(|t| t.handle).collect();
        let err = unsafe {
            ffi::flodl_nccl_all_reduce_rank(
                self.handle,
                handles.as_mut_ptr(),
                handles.len() as i32,
                ptr::null_mut(),
                op as i32,
            )
        };
        check_err(err)
    }

    /// In-place AllReduce on an explicit CUDA stream.
    ///
    /// Same semantics as [`all_reduce`](Self::all_reduce), but NCCL work is
    /// enqueued on the provided stream for overlap with compute kernels.
    ///
    /// # Parameters
    ///
    /// - `tensors`: one or more tensors on this rank's device. Modified in-place.
    /// - `op`: reduction operation applied element-wise.
    /// - `stream`: CUDA stream on this rank's device.
    pub fn all_reduce_on_stream(
        &self,
        tensors: &[&Tensor],
        op: ReduceOp,
        stream: &CudaStream,
    ) -> Result<()> {
        let mut handles: Vec<ffi::FlodlTensor> = tensors.iter().map(|t| t.handle).collect();
        let err = unsafe {
            ffi::flodl_nccl_all_reduce_rank(
                self.handle,
                handles.as_mut_ptr(),
                handles.len() as i32,
                stream.as_ptr(),
                op as i32,
            )
        };
        check_err(err)
    }
}

impl Drop for NcclRankComm {
    fn drop(&mut self) {
        // ncclCommAbort already frees the comm; skip destroy if aborted.
        if !self.handle.is_null() && !self.abort_handle.is_aborted() {
            unsafe { ffi::flodl_nccl_destroy_rank(self.handle) };
            self.handle = ptr::null_mut();
        }
        // Invalidate the abort handle so stale Arc<NcclAbortHandle> clones
        // (held by DdpHandle) don't call ncclCommAbort on a freed pointer.
        self.abort_handle.mark_destroyed();
    }
}

impl std::fmt::Debug for NcclRankComm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NcclRankComm")
            .field("rank", &self.rank)
            .field("world_size", &self.world_size)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{test_device, cuda_device_count, cuda_synchronize, TensorOptions, DType};
    use crate::distributed::ddp::NCCL_LOCK;

    fn require_multi_gpu() -> bool {
        require_n_gpu(2)
    }

    fn require_n_gpu(n: i32) -> bool {
        if !test_device().is_cuda() || cuda_device_count() < n {
            return false;
        }
        // Verify all required devices can run compute kernels (e.g., GTX 1060
        // sm_61 is unsupported by libtorch cu128 builds).
        for i in 0..n {
            let dev = Device::CUDA(i as u8);
            let opts = TensorOptions { dtype: DType::Float32, device: dev };
            if Tensor::zeros(&[1], opts).is_err() {
                eprintln!("Device CUDA({i}) cannot run compute kernels, skipping {n}-GPU test");
                return false;
            }
        }
        true
    }

    #[test]
    fn test_nccl_requires_two_devices() {
        let result = NcclComms::new(&[Device::CUDA(0)]);
        assert!(result.is_err(), "NcclComms should require 2+ devices");
    }

    #[test]
    fn test_nccl_rejects_cpu() {
        let result = NcclComms::new(&[Device::CPU, Device::CPU]);
        assert!(result.is_err(), "NcclComms should reject CPU devices");
    }

    #[test]
    #[ignore = "NCCL init needs exclusive GPU; run with: fdl cuda-test-all"]
    fn test_nccl_init_destroy() {
        if !require_multi_gpu() { return; }
        let _lock = NCCL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let comms = NcclComms::new(&[Device::CUDA(0), Device::CUDA(1)]).unwrap();
        assert_eq!(comms.size(), 2);
        assert_eq!(comms.devices(), &[Device::CUDA(0), Device::CUDA(1)]);
        // Drop cleans up
    }

    #[test]
    #[ignore = "NCCL init needs exclusive GPU; run with: fdl cuda-test-all"]
    fn test_nccl_broadcast() {
        if !require_multi_gpu() { return; }
        let _lock = NCCL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let comms = NcclComms::new(&[Device::CUDA(0), Device::CUDA(1)]).unwrap();

        let opts0 = TensorOptions { dtype: DType::Float32, device: Device::CUDA(0) };
        let opts1 = TensorOptions { dtype: DType::Float32, device: Device::CUDA(1) };

        // Set values on device 0, zeros on device 1
        let t0 = Tensor::full(&[64], 42.0, opts0).unwrap();
        let t1 = Tensor::zeros(&[64], opts1).unwrap();

        // Broadcast from device 0
        comms.broadcast(&[&t0, &t1], 0).unwrap();
        cuda_synchronize(0);
        cuda_synchronize(1);

        let vals0 = t0.to_f32_vec().unwrap();
        let vals1 = t1.to_f32_vec().unwrap();
        assert!(vals0.iter().all(|&v| (v - 42.0).abs() < 1e-5),
            "device 0 should still have 42.0");
        assert!(vals1.iter().all(|&v| (v - 42.0).abs() < 1e-5),
            "device 1 should have 42.0 after broadcast");
    }

    #[test]
    #[ignore = "NCCL init needs exclusive GPU; run with: fdl cuda-test-all"]
    fn test_nccl_all_reduce_sum() {
        if !require_multi_gpu() { return; }
        let _lock = NCCL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let comms = NcclComms::new(&[Device::CUDA(0), Device::CUDA(1)]).unwrap();

        let opts0 = TensorOptions { dtype: DType::Float32, device: Device::CUDA(0) };
        let opts1 = TensorOptions { dtype: DType::Float32, device: Device::CUDA(1) };

        // 1.0 on device 0, 2.0 on device 1
        let t0 = Tensor::full(&[128], 1.0, opts0).unwrap();
        let t1 = Tensor::full(&[128], 2.0, opts1).unwrap();

        comms.all_reduce(&[&t0, &t1], ReduceOp::Sum).unwrap();
        cuda_synchronize(0);
        cuda_synchronize(1);

        // Sum: 1.0 + 2.0 = 3.0 on both devices
        let vals0 = t0.to_f32_vec().unwrap();
        let vals1 = t1.to_f32_vec().unwrap();
        assert!(vals0.iter().all(|&v| (v - 3.0).abs() < 1e-5),
            "device 0 should have 3.0 after AllReduce Sum");
        assert!(vals1.iter().all(|&v| (v - 3.0).abs() < 1e-5),
            "device 1 should have 3.0 after AllReduce Sum");
    }

    #[test]
    #[ignore = "NCCL init needs exclusive GPU; run with: fdl cuda-test-all"]
    fn test_nccl_all_reduce_avg() {
        if !require_multi_gpu() { return; }
        let _lock = NCCL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let comms = NcclComms::new(&[Device::CUDA(0), Device::CUDA(1)]).unwrap();

        let opts0 = TensorOptions { dtype: DType::Float32, device: Device::CUDA(0) };
        let opts1 = TensorOptions { dtype: DType::Float32, device: Device::CUDA(1) };

        // 10.0 on device 0, 20.0 on device 1
        let t0 = Tensor::full(&[64], 10.0, opts0).unwrap();
        let t1 = Tensor::full(&[64], 20.0, opts1).unwrap();

        comms.all_reduce(&[&t0, &t1], ReduceOp::Avg).unwrap();
        cuda_synchronize(0);
        cuda_synchronize(1);

        // Avg: (10.0 + 20.0) / 2 = 15.0
        let vals0 = t0.to_f32_vec().unwrap();
        let vals1 = t1.to_f32_vec().unwrap();
        assert!(vals0.iter().all(|&v| (v - 15.0).abs() < 1e-5),
            "device 0 should have 15.0 after AllReduce Avg");
        assert!(vals1.iter().all(|&v| (v - 15.0).abs() < 1e-5),
            "device 1 should have 15.0 after AllReduce Avg");
    }

    #[test]
    #[ignore = "NCCL init needs exclusive GPU; run with: fdl cuda-test-all"]
    fn test_nccl_all_reduce_on_streams() {
        if !require_multi_gpu() { return; }
        let _lock = NCCL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let comms = NcclComms::new(&[Device::CUDA(0), Device::CUDA(1)]).unwrap();

        let opts0 = TensorOptions { dtype: DType::Float32, device: Device::CUDA(0) };
        let opts1 = TensorOptions { dtype: DType::Float32, device: Device::CUDA(1) };

        let stream0 = CudaStream::new(Device::CUDA(0), false).unwrap();
        let stream1 = CudaStream::new(Device::CUDA(1), false).unwrap();

        let t0 = Tensor::full(&[32], 5.0, opts0).unwrap();
        let t1 = Tensor::full(&[32], 7.0, opts1).unwrap();

        comms.all_reduce_on_streams(
            &[&t0, &t1], ReduceOp::Sum, &[&stream0, &stream1],
        ).unwrap();

        stream0.synchronize().unwrap();
        stream1.synchronize().unwrap();

        let vals0 = t0.to_f32_vec().unwrap();
        let vals1 = t1.to_f32_vec().unwrap();
        assert!(vals0.iter().all(|&v| (v - 12.0).abs() < 1e-5),
            "device 0 should have 12.0 after AllReduce Sum on streams");
        assert!(vals1.iter().all(|&v| (v - 12.0).abs() < 1e-5),
            "device 1 should have 12.0 after AllReduce Sum on streams");
    }

    // --- NcclRankComm tests ---

    #[test]
    fn test_nccl_rank_comm_rejects_invalid_rank() {
        let result = NcclRankComm::init_rank(2, 2, &NcclUniqueId { bytes: [0; NCCL_UNIQUE_ID_BYTES] });
        assert!(result.is_err(), "rank >= world_size should fail");
    }

    #[test]
    fn test_nccl_rank_comm_rejects_world_size_one() {
        let result = NcclRankComm::init_rank(0, 1, &NcclUniqueId { bytes: [0; NCCL_UNIQUE_ID_BYTES] });
        assert!(result.is_err(), "world_size < 2 should fail");
    }

    #[test]
    fn test_nccl_unique_id_clone() {
        // NcclUniqueId must be cloneable for distribution to worker threads
        fn assert_send_sync_clone<T: Send + Sync + Clone>() {}
        assert_send_sync_clone::<NcclUniqueId>();
    }

    #[test]
    fn test_nccl_rank_comm_send() {
        fn assert_send<T: Send>() {}
        assert_send::<NcclRankComm>();
    }

    #[test]
    #[ignore = "NCCL init needs exclusive GPU; run with: fdl cuda-test-all"]
    fn test_nccl_rank_comm_init_and_reduce() {
        if !require_multi_gpu() { return; }
        let _lock = NCCL_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let uid = NcclUniqueId::new().unwrap();
        let uid0 = uid.clone();
        let uid1 = uid;

        // Each rank must call init_rank concurrently. Use two threads.
        let h0 = std::thread::spawn(move || {
            crate::tensor::set_current_cuda_device(0);
            NcclRankComm::init_rank(0, 2, &uid0).unwrap()
        });
        let h1 = std::thread::spawn(move || {
            crate::tensor::set_current_cuda_device(1);
            NcclRankComm::init_rank(1, 2, &uid1).unwrap()
        });
        let comm0 = h0.join().unwrap();
        let comm1 = h1.join().unwrap();

        assert_eq!(comm0.rank(), 0);
        assert_eq!(comm0.world_size(), 2);
        assert_eq!(comm1.rank(), 1);

        // AllReduce Avg: 10.0 on dev0, 20.0 on dev1 -> 15.0 on both
        let opts0 = TensorOptions { dtype: DType::Float32, device: Device::CUDA(0) };
        let opts1 = TensorOptions { dtype: DType::Float32, device: Device::CUDA(1) };
        let t0 = Tensor::full(&[64], 10.0, opts0).unwrap();
        let t1 = Tensor::full(&[64], 20.0, opts1).unwrap();

        // AllReduce must be called concurrently from different threads
        let t0_clone = t0.clone();
        let t1_clone = t1.clone();

        let h0 = std::thread::spawn(move || {
            crate::tensor::set_current_cuda_device(0);
            comm0.all_reduce(&[&t0_clone], ReduceOp::Avg).unwrap();
            cuda_synchronize(0);
        });
        let h1 = std::thread::spawn(move || {
            crate::tensor::set_current_cuda_device(1);
            comm1.all_reduce(&[&t1_clone], ReduceOp::Avg).unwrap();
            cuda_synchronize(1);
        });
        h0.join().unwrap();
        h1.join().unwrap();

        let vals0 = t0.to_f32_vec().unwrap();
        let vals1 = t1.to_f32_vec().unwrap();
        assert!(vals0.iter().all(|&v| (v - 15.0).abs() < 1e-5),
            "rank 0 should have 15.0 after AllReduce Avg, got {}", vals0[0]);
        assert!(vals1.iter().all(|&v| (v - 15.0).abs() < 1e-5),
            "rank 1 should have 15.0 after AllReduce Avg, got {}", vals1[0]);
    }

    #[test]
    #[ignore = "NCCL init needs exclusive GPU; run with: fdl cuda-test-all"]
    fn test_nccl_rank_comm_on_stream() {
        if !require_multi_gpu() { return; }
        let _lock = NCCL_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let uid = NcclUniqueId::new().unwrap();
        let uid0 = uid.clone();
        let uid1 = uid;

        let h0 = std::thread::spawn(move || {
            crate::tensor::set_current_cuda_device(0);
            NcclRankComm::init_rank(0, 2, &uid0).unwrap()
        });
        let h1 = std::thread::spawn(move || {
            crate::tensor::set_current_cuda_device(1);
            NcclRankComm::init_rank(1, 2, &uid1).unwrap()
        });
        let comm0 = h0.join().unwrap();
        let comm1 = h1.join().unwrap();

        let opts0 = TensorOptions { dtype: DType::Float32, device: Device::CUDA(0) };
        let opts1 = TensorOptions { dtype: DType::Float32, device: Device::CUDA(1) };
        let stream0 = CudaStream::new(Device::CUDA(0), false).unwrap();
        let stream1 = CudaStream::new(Device::CUDA(1), false).unwrap();

        let t0 = Tensor::full(&[32], 3.0, opts0).unwrap();
        let t1 = Tensor::full(&[32], 7.0, opts1).unwrap();
        let t0c = t0.clone();
        let t1c = t1.clone();

        let h0 = std::thread::spawn(move || {
            crate::tensor::set_current_cuda_device(0);
            comm0.all_reduce_on_stream(&[&t0c], ReduceOp::Sum, &stream0).unwrap();
            stream0.synchronize().unwrap();
        });
        let h1 = std::thread::spawn(move || {
            crate::tensor::set_current_cuda_device(1);
            comm1.all_reduce_on_stream(&[&t1c], ReduceOp::Sum, &stream1).unwrap();
            stream1.synchronize().unwrap();
        });
        h0.join().unwrap();
        h1.join().unwrap();

        let vals0 = t0.to_f32_vec().unwrap();
        let vals1 = t1.to_f32_vec().unwrap();
        assert!(vals0.iter().all(|&v| (v - 10.0).abs() < 1e-5),
            "rank 0 should have 10.0 after Sum, got {}", vals0[0]);
        assert!(vals1.iter().all(|&v| (v - 10.0).abs() < 1e-5),
            "rank 1 should have 10.0 after Sum, got {}", vals1[0]);
    }

    #[test]
    #[ignore = "NCCL init needs exclusive GPU; run with: fdl cuda-test-all"]
    fn test_nccl_rank_comm_multi_tensor_batch() {
        if !require_multi_gpu() { return; }
        let _lock = NCCL_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let uid = NcclUniqueId::new().unwrap();
        let uid0 = uid.clone();
        let uid1 = uid;

        let h0 = std::thread::spawn(move || {
            crate::tensor::set_current_cuda_device(0);
            NcclRankComm::init_rank(0, 2, &uid0).unwrap()
        });
        let h1 = std::thread::spawn(move || {
            crate::tensor::set_current_cuda_device(1);
            NcclRankComm::init_rank(1, 2, &uid1).unwrap()
        });
        let comm0 = h0.join().unwrap();
        let comm1 = h1.join().unwrap();

        let opts0 = TensorOptions { dtype: DType::Float32, device: Device::CUDA(0) };
        let opts1 = TensorOptions { dtype: DType::Float32, device: Device::CUDA(1) };

        // Two tensors per rank (simulates multiple params)
        let a0 = Tensor::full(&[16], 1.0, opts0).unwrap();
        let b0 = Tensor::full(&[8], 100.0, opts0).unwrap();
        let a1 = Tensor::full(&[16], 3.0, opts1).unwrap();
        let b1 = Tensor::full(&[8], 200.0, opts1).unwrap();

        let a0c = a0.clone();
        let b0c = b0.clone();
        let a1c = a1.clone();
        let b1c = b1.clone();

        let h0 = std::thread::spawn(move || {
            crate::tensor::set_current_cuda_device(0);
            comm0.all_reduce(&[&a0c, &b0c], ReduceOp::Avg).unwrap();
            cuda_synchronize(0);
        });
        let h1 = std::thread::spawn(move || {
            crate::tensor::set_current_cuda_device(1);
            comm1.all_reduce(&[&a1c, &b1c], ReduceOp::Avg).unwrap();
            cuda_synchronize(1);
        });
        h0.join().unwrap();
        h1.join().unwrap();

        // a: avg(1.0, 3.0) = 2.0, b: avg(100.0, 200.0) = 150.0
        let va0 = a0.to_f32_vec().unwrap();
        let vb0 = b0.to_f32_vec().unwrap();
        assert!(va0.iter().all(|&v| (v - 2.0).abs() < 1e-5), "a0 should be 2.0");
        assert!(vb0.iter().all(|&v| (v - 150.0).abs() < 1e-5), "b0 should be 150.0");

        let va1 = a1.to_f32_vec().unwrap();
        let vb1 = b1.to_f32_vec().unwrap();
        assert!(va1.iter().all(|&v| (v - 2.0).abs() < 1e-5), "a1 should be 2.0");
        assert!(vb1.iter().all(|&v| (v - 150.0).abs() < 1e-5), "b1 should be 150.0");
    }

    // 3-GPU smoke: heterogeneous topology (e.g. RTX 5060 Ti sm_120 +
    // 2x GTX 1060 sm_61). Uses NcclComms main-thread init since
    // ncclCommInitRank from worker threads corrupts CUDA context across
    // heterogeneous architectures.
    #[test]
    #[ignore = "NCCL init needs exclusive GPU; needs 3 GPUs; run with: fdl cuda-test-all"]
    fn test_nccl_three_gpu_all_reduce_sum() {
        if !require_n_gpu(3) { return; }
        let _lock = NCCL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let comms = NcclComms::new(&[
            Device::CUDA(0), Device::CUDA(1), Device::CUDA(2),
        ]).unwrap();
        assert_eq!(comms.size(), 3);

        let opts0 = TensorOptions { dtype: DType::Float32, device: Device::CUDA(0) };
        let opts1 = TensorOptions { dtype: DType::Float32, device: Device::CUDA(1) };
        let opts2 = TensorOptions { dtype: DType::Float32, device: Device::CUDA(2) };

        let t0 = Tensor::full(&[64], 1.0, opts0).unwrap();
        let t1 = Tensor::full(&[64], 2.0, opts1).unwrap();
        let t2 = Tensor::full(&[64], 3.0, opts2).unwrap();

        comms.all_reduce(&[&t0, &t1, &t2], ReduceOp::Sum).unwrap();
        cuda_synchronize(0);
        cuda_synchronize(1);
        cuda_synchronize(2);

        // Sum: 1 + 2 + 3 = 6 on every device.
        for (i, t) in [&t0, &t1, &t2].iter().enumerate() {
            let vals = t.to_f32_vec().unwrap();
            assert!(vals.iter().all(|&v| (v - 6.0).abs() < 1e-5),
                "rank {i} should have 6.0 after AllReduce Sum, got {}", vals[0]);
        }
    }
}
