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
    check_err, current_gpu_device, set_current_gpu_device,
    Device, Result, Tensor, TensorError,
};
use crate::tensor::cuda_stream::GpuStream;

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
        let saved = current_gpu_device();
        let err = unsafe {
            ffi::flodl_nccl_init(
                devlist.len() as i32,
                devlist.as_ptr(),
                &mut handle,
            )
        };
        set_current_gpu_device(saved);
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
        let saved = current_gpu_device();
        let err = unsafe {
            ffi::flodl_nccl_all_reduce(
                self.handle,
                handles.as_mut_ptr(),
                ptr::null_mut(),
                op as i32,
            )
        };
        set_current_gpu_device(saved);
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
        streams: &[&GpuStream],
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
        let saved = current_gpu_device();
        let err = unsafe {
            ffi::flodl_nccl_all_reduce(
                self.handle,
                handles.as_mut_ptr(),
                stream_ptrs.as_mut_ptr(),
                op as i32,
            )
        };
        set_current_gpu_device(saved);
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
        let saved = current_gpu_device();
        let err = unsafe {
            ffi::flodl_nccl_broadcast(
                self.handle,
                handles.as_mut_ptr(),
                ptr::null_mut(),
                root as i32,
            )
        };
        set_current_gpu_device(saved);
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
        streams: &[&GpuStream],
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
        let saved = current_gpu_device();
        let err = unsafe {
            ffi::flodl_nccl_broadcast(
                self.handle,
                handles.as_mut_ptr(),
                stream_ptrs.as_mut_ptr(),
                root as i32,
            )
        };
        set_current_gpu_device(saved);
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
                guard: std::sync::Mutex::new(()),
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
/// The `(major, minor, patch)` of the NCCL/RCCL library this process
/// actually loads — LD_PRELOAD-aware (it asks the loaded library, not
/// the build headers), and no CUDA context is touched, so it is safe
/// before `Trainer::run`. This is the fact the join hello carries for
/// the admission skew gate: two libtorches shipping different
/// major.minor refuse each other's NCCL handshake at formation, which
/// is exactly too late. `None` on a CPU build (no library to ask) or
/// when the read fails — an unknown version gates nothing.
pub(crate) fn runtime_version() -> Option<(u32, u32, u32)> {
    let mut v: i32 = 0;
    let err = unsafe { ffi::flodl_nccl_runtime_version(&mut v) };
    if check_err(err).is_err() || v <= 0 {
        return None;
    }
    let v = v as u32;
    // NCCL's integer encoding: major*10_000 + minor*100 + patch since
    // 2.9; major*1_000 + minor*100 + patch before. Everything we can
    // meet is post-2.9, but decode the old shape rather than misread it.
    Some(if v >= 20_000 {
        (v / 10_000, (v % 10_000) / 100, v % 100)
    } else {
        (v / 1_000, (v % 1_000) / 100, v % 100)
    })
}

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

    /// Reconstruct a unique ID from raw bytes.
    ///
    /// Used by the multi-host rendezvous to materialize the master-generated
    /// ID on worker hosts after receiving it over TCP. Single-process callers
    /// should use [`new`](Self::new) instead.
    pub fn from_bytes(bytes: [u8; NCCL_UNIQUE_ID_BYTES]) -> Self {
        NcclUniqueId { bytes }
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
    /// ISSUE GUARD: mutual exclusion between enqueuing a collective and
    /// aborting the communicator. `ncclCommAbort` FREES the comm, so an
    /// abort landing in the gap BETWEEN two collectives turns the next
    /// enqueue into a use-after-free (SIGSEGV — observed live: the
    /// watchdog fired between the count-reduce and the param-reduce of
    /// a weighted sync). Every enqueue takes this lock and re-checks
    /// `aborted` under it; `abort()` takes it too, so the freed-comm
    /// state is only ever observable as a loud pre-issue error, never
    /// as a dangling pointer.
    ///
    /// No-deadlock argument: NCCL calls here are enqueue-fast — the
    /// peer-wait happens in the CUDA stream synchronize, OUTSIDE this
    /// lock (that is also why aborting a comm whose kernels are being
    /// waited on works: the kernels die and the stream sync returns).
    /// Residual: an enqueue wedging INSIDE NCCL (internal connect
    /// stall) would hold the lock and block the watchdog — strictly
    /// rarer than the deterministic use-after-free this lock removes.
    guard: std::sync::Mutex<()>,
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
        // Take the issue guard so no collective can be mid-enqueue (or
        // start enqueuing) while the comm is being freed. A collective
        // already blocked in its stream-wait is unaffected — that wait
        // is outside the guard, and killing the comm's kernels is
        // exactly how it gets released.
        let _guard = self.guard.lock().expect("nccl issue guard poisoned");
        if !self.claim() {
            return Ok(()); // already aborted or destroyed
        }
        let err = unsafe { ffi::flodl_nccl_abort_rank(self.ptr) };
        check_err(err)
    }

    /// Take the issue guard for enqueuing one collective, failing loudly
    /// if the communicator has been aborted. Callers hold the returned
    /// guard across the FFI enqueue only — stream synchronization happens
    /// after release.
    pub(crate) fn lock_for_issue(&self) -> Result<std::sync::MutexGuard<'_, ()>> {
        let guard = self.guard.lock().expect("nccl issue guard poisoned");
        if self.is_aborted() {
            return Err(TensorError::new(
                "NCCL communicator aborted (peer death); collective refused — \
                 the caller must rebuild the comm on the surviving cohort",
            ));
        }
        Ok(guard)
    }

    /// Whether this communicator has been aborted or destroyed.
    pub fn is_aborted(&self) -> bool {
        self.aborted.load(Ordering::Acquire)
    }

    /// Atomically claim the right to tear the comm down (abort OR destroy).
    /// Returns `true` for exactly one caller; everyone else gets `false`.
    ///
    /// Abort (watchdog thread) and destroy (`NcclRankComm::drop`) can race —
    /// the cluster worker's NCCL watchdog runs until a shutdown flag that is
    /// set only AFTER the comm is dropped. A check-then-act here would let
    /// `abort()` pass the check while Drop is mid-`flodl_nccl_destroy_rank`
    /// and call into a freed comm. The single swap makes the two teardown
    /// paths mutually exclusive.
    fn claim(&self) -> bool {
        !self.aborted.swap(true, Ordering::AcqRel)
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
    /// (via `set_current_gpu_device`). All ranks must call this concurrently.
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
            guard: std::sync::Mutex::new(()),
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
        let _guard = self.abort_handle.lock_for_issue()?;
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

    /// In-place work-weighted AllReduce: each rank's contribution is
    /// premultiplied by ITS OWN `factor` inside the collective
    /// (`ncclRedOpCreatePreMulSum`), so the output is `Σ fᵢ·xᵢ` with a
    /// single collective and ZERO bookend kernels. With
    /// `fᵢ = nᵢ^γ / Σn^γ` the output IS the work-weighted consensus —
    /// no pre-scale, no post-divide, and therefore no divide kernel for
    /// downstream consumers to race (the fence contract collapses to
    /// "order after the collective").
    ///
    /// The dynamic reduction op is comm-bound and window-scoped: created
    /// here, used for this batched collective, destroyed before
    /// returning (communication-free, cheap). This composes with the
    /// abort→rebuild path naturally — a rebuilt comm gets a fresh op on
    /// its next window. Requires NCCL >= 2.11 at build and run time
    /// (checked in the shim; errors loudly naming the found version).
    ///
    /// `tensors` must all be f32 (the premul scalar is created with
    /// `ncclFloat32`; NCCL requires the scalar dtype to match).
    ///
    /// `stream`: `Some` enqueues on that CUDA stream (comm-stream
    /// overlap), `None` uses the current stream.
    pub fn all_reduce_premul_sum(
        &self,
        tensors: &[&Tensor],
        factor: f32,
        stream: Option<&GpuStream>,
    ) -> Result<()> {
        for (i, t) in tensors.iter().enumerate() {
            if t.dtype() != crate::tensor::DType::Float32 {
                return Err(crate::TensorError::new(&format!(
                    "all_reduce_premul_sum: tensor {i} is {:?}, but the \
                     PreMulSum scalar is f32 — NCCL requires matching dtypes",
                    t.dtype(),
                )));
            }
        }
        // One issue-guard across create → collective → destroy: the op is
        // comm-bound, and the abort path frees the comm — none of the
        // three calls may race it (same contract as the other
        // collectives; see NcclAbortHandle::lock_for_issue).
        let _guard = self.abort_handle.lock_for_issue()?;
        let mut op: i32 = 0;
        let err = unsafe {
            ffi::flodl_nccl_redop_premulsum_create_rank(self.handle, factor, &mut op)
        };
        check_err(err)?;
        let mut handles: Vec<ffi::FlodlTensor> = tensors.iter().map(|t| t.handle).collect();
        let stream_ptr = stream.map_or(ptr::null_mut(), |s| s.as_ptr());
        let reduce_err = unsafe {
            ffi::flodl_nccl_all_reduce_rank(
                self.handle,
                handles.as_mut_ptr(),
                handles.len() as i32,
                stream_ptr,
                op,
            )
        };
        // Destroy the op regardless of the collective's outcome (it only
        // frees comm-local state; enqueue has already happened). Surface
        // the collective error first — it is the load-bearing one.
        let destroy_err = unsafe { ffi::flodl_nccl_redop_destroy_rank(self.handle, op) };
        check_err(reduce_err)?;
        check_err(destroy_err)
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
        stream: &GpuStream,
    ) -> Result<()> {
        let _guard = self.abort_handle.lock_for_issue()?;
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

    /// In-place Broadcast from `root` rank to all other ranks, using the
    /// default stream.
    ///
    /// All ranks must call this concurrently with the same number of tensors
    /// and the same `root` for the collective to complete. The `root` rank's
    /// tensor contents are sent in-place to all other ranks.
    ///
    /// # Parameters
    ///
    /// - `tensors`: one or more tensors on this rank's device. On non-root
    ///   ranks, contents are overwritten with the root rank's values.
    /// - `root`: source rank, in `0..world_size()`.
    pub fn broadcast(&self, tensors: &[&Tensor], root: usize) -> Result<()> {
        if root >= self.world_size {
            return Err(crate::TensorError::new(&format!(
                "NcclRankComm::broadcast: root {root} out of range \
                 (world_size = {})",
                self.world_size
            )));
        }
        let _guard = self.abort_handle.lock_for_issue()?;
        let mut handles: Vec<ffi::FlodlTensor> = tensors.iter().map(|t| t.handle).collect();
        let err = unsafe {
            ffi::flodl_nccl_broadcast_rank(
                self.handle,
                handles.as_mut_ptr(),
                handles.len() as i32,
                ptr::null_mut(),
                root as i32,
            )
        };
        check_err(err)
    }

    /// In-place Broadcast on an explicit CUDA stream.
    ///
    /// Same semantics as [`broadcast`](Self::broadcast), but NCCL work is
    /// enqueued on the provided stream for overlap with compute kernels.
    pub fn broadcast_on_stream(
        &self,
        tensors: &[&Tensor],
        root: usize,
        stream: &GpuStream,
    ) -> Result<()> {
        if root >= self.world_size {
            return Err(crate::TensorError::new(&format!(
                "NcclRankComm::broadcast_on_stream: root {root} out of range \
                 (world_size = {})",
                self.world_size
            )));
        }
        let _guard = self.abort_handle.lock_for_issue()?;
        let mut handles: Vec<ffi::FlodlTensor> = tensors.iter().map(|t| t.handle).collect();
        let err = unsafe {
            ffi::flodl_nccl_broadcast_rank(
                self.handle,
                handles.as_mut_ptr(),
                handles.len() as i32,
                stream.as_ptr(),
                root as i32,
            )
        };
        check_err(err)
    }
}

impl Drop for NcclRankComm {
    fn drop(&mut self) {
        // ncclCommAbort already frees the comm; destroy only if we win the
        // teardown claim. The swap-based claim (not check-then-act) makes
        // this mutually exclusive with a concurrent watchdog `abort()` —
        // see `NcclAbortHandle::claim`. Claiming also invalidates stale
        // Arc<NcclAbortHandle> clones (held by DdpHandle / the cluster
        // watchdog) so they never call ncclCommAbort on a freed pointer.
        // Claim FIRST (even with a null handle) so stale clones are always
        // invalidated; destroy only when we won the claim AND the handle is
        // live.
        if self.abort_handle.claim() && !self.handle.is_null() {
            unsafe { ffi::flodl_nccl_destroy_rank(self.handle) };
            self.handle = ptr::null_mut();
        }
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
#[path = "nccl_tests.rs"]
mod tests;
