//! Tensor — immutable, chainable wrapper around a libtorch tensor.
//!
//! Every tensor owns its C++ handle and frees it on drop. This is the
//! entire VRAM management story — no GC, no scopes, no finalizers.
//!
//! Operations are chainable and return `Result<Tensor>`:
//!
//! ```ignore
//! let z = a.add(&b)?.relu()?.sum()?;
//! ```

mod cuda;
pub mod cuda_event;
pub mod cuda_stream;
mod ops;
mod shape;
mod nn_ops;

pub use cuda::*;
pub use cuda_event::{CudaEvent, CudaEventFlags};
pub use cuda_stream::{CudaStream, StreamGuard};

pub use nn_ops::RnnParams;

use std::ffi::{c_void, CStr};
use std::fmt;
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};

use flodl_sys::{self as ffi, FlodlTensor};

/// Global counter of live C++ Tensor handles. Incremented on creation,
/// decremented on Drop. If this grows over time during training, there
/// is a Tensor handle leak. If it stays stable but RSS grows, the leak
/// is inside libtorch internals (not a handle leak).
pub(super) static LIVE_TENSOR_COUNT: AtomicU64 = AtomicU64::new(0);

/// Element data type of a tensor. Maps to PyTorch's `torch.dtype`.
///
/// Float32 is the default. Use Float16/BFloat16 for mixed precision,
/// Int64 for indices and labels, Float64 when extra precision is needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum DType {
    Float16 = ffi::FLODL_FLOAT16,
    BFloat16 = ffi::FLODL_BFLOAT16,
    Float32 = ffi::FLODL_FLOAT32,
    Float64 = ffi::FLODL_FLOAT64,
    Int32 = ffi::FLODL_INT32,
    Int64 = ffi::FLODL_INT64,
}

impl DType {
    fn from_raw(v: i32) -> Self {
        match v {
            ffi::FLODL_FLOAT16 => DType::Float16,
            ffi::FLODL_BFLOAT16 => DType::BFloat16,
            ffi::FLODL_FLOAT32 => DType::Float32,
            ffi::FLODL_FLOAT64 => DType::Float64,
            ffi::FLODL_INT32 => DType::Int32,
            ffi::FLODL_INT64 => DType::Int64,
            // An unknown code means the Rust dtype table and the shim's
            // disagree — a build/ABI mismatch (rebuild flodl-sys), NOT a
            // runtime condition. Silently masquerading as Float32 would
            // hand back a wrong-typed tensor (the TF3/TF15 silent-wrong-
            // answer family); `dtype()` is an infallible hot accessor, so
            // promoting it to `Result` is rejected as ABI churn (TF6). A
            // defined, named panic beats a silent wrong answer.
            other => panic!(
                "flodl: unknown dtype code {other} from the C++ shim — the \
                 Rust and flodl-sys dtype tables disagree; rebuild flodl-sys"
            ),
        }
    }

    /// Size of one element in bytes.
    pub fn element_size(self) -> usize {
        match self {
            DType::Float16 | DType::BFloat16 => 2,
            DType::Float32 | DType::Int32 => 4,
            DType::Float64 | DType::Int64 => 8,
        }
    }
}

/// Device represents where a tensor's data lives.
///
/// `Device::CPU` is the host. `Device::CUDA(n)` is GPU index `n`.
/// Most single-GPU code uses `Device::CUDA(0)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    CPU,
    CUDA(u8),
}

impl Device {
    /// Convert to (device_type, device_index) for FFI calls.
    pub(crate) fn to_ffi(self) -> (i32, i32) {
        match self {
            Device::CPU => (ffi::FLODL_CPU, 0),
            Device::CUDA(idx) => (ffi::FLODL_CUDA, idx as i32),
        }
    }

    /// Reconstruct from FFI (device_type, device_index).
    pub(crate) fn from_ffi(device_type: i32, device_index: i32) -> Self {
        match device_type {
            ffi::FLODL_CUDA => Device::CUDA(device_index as u8),
            _ => Device::CPU,
        }
    }

    /// Whether this is a CUDA device.
    pub fn is_cuda(&self) -> bool {
        matches!(self, Device::CUDA(_))
    }

    /// Device index (0 for CPU, GPU index for CUDA).
    pub fn index(&self) -> u8 {
        match self {
            Device::CPU => 0,
            Device::CUDA(idx) => *idx,
        }
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::CPU => write!(f, "cpu"),
            Device::CUDA(0) => write!(f, "cuda"),
            Device::CUDA(idx) => write!(f, "cuda:{}", idx),
        }
    }
}

/// Error type for tensor operations.
#[derive(Debug, Clone)]
pub struct TensorError(String);

impl TensorError {
    pub fn new(msg: &str) -> Self {
        TensorError(msg.to_string())
    }

    /// Whether this error indicates a CUDA out-of-memory condition.
    pub fn is_cuda_oom(&self) -> bool {
        self.0.contains("out of memory")
    }
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for TensorError {}

pub type Result<T> = std::result::Result<T, TensorError>;

/// Convert a C error string to Result. Frees the C string.
///
/// `*mut c_char`, not `*mut i8`: `c_char` is `i8` on x86_64 but `u8` on
/// Linux aarch64 — typing the FFI surface with `c_char` end to end is
/// what keeps `CStr::from_ptr` compiling on both without per-site casts.
pub(crate) fn check_err(err: *mut std::ffi::c_char) -> Result<()> {
    if err.is_null() {
        Ok(())
    } else {
        let msg = unsafe { CStr::from_ptr(err) }
            .to_string_lossy()
            .into_owned();
        unsafe { ffi::flodl_free_string(err) };
        Err(TensorError(msg))
    }
}

/// Call a single-output-tensor FFI op and wrap the result.
///
/// Every wrapper for an FFI op that produces one output tensor shares the
/// same four-line body: null out-handle → `unsafe` call with `&mut handle`
/// appended as the last argument → `check_err(...)?` → `Ok(Tensor::from_raw(handle))`.
/// This macro is that body. The op's own input arguments are forwarded
/// verbatim, so it fits unary, binary, scalar, and multi-arg wrappers
/// alike. Used as a method's tail expression, leaving the signature and
/// doc as plain, checked, greppable source:
///
/// ```ignore
/// pub fn add(&self, other: &Tensor) -> Result<Tensor> {
///     ffi_call!(flodl_add, self.handle, other.handle)
/// }
/// ```
macro_rules! ffi_call {
    ($ffi:ident $(, $arg:expr)* $(,)?) => {{
        let mut handle: flodl_sys::FlodlTensor = ::std::ptr::null_mut();
        let err = unsafe { flodl_sys::$ffi($($arg,)* &mut handle) };
        $crate::tensor::check_err(err)?;
        Ok($crate::tensor::Tensor::from_raw(handle))
    }};
}
pub(crate) use ffi_call;

/// Options for tensor creation.
#[derive(Debug, Clone, Copy)]
pub struct TensorOptions {
    pub dtype: DType,
    pub device: Device,
}

impl Default for TensorOptions {
    fn default() -> Self {
        Self {
            dtype: DType::Float32,
            device: Device::CPU,
        }
    }
}

/// A tensor wrapping a libtorch C++ tensor.
///
/// Owns the underlying C++ handle. When dropped, the C++ tensor is
/// freed immediately — including any GPU memory. This is the entire
/// VRAM management story.
///
/// Operations are chainable and return `Result<Tensor>`:
///
/// ```ignore
/// let y = x.matmul(&w)?.add(&b)?.relu()?;
/// ```
///
/// # Thread safety
///
/// `Tensor` is `Send + Sync`. Concurrent reads from multiple threads
/// are safe: every non-suffixed op allocates a new output tensor. The
/// `_`-suffixed in-place ops (`add_`, `copy_`, `zero_`, `fill_`,
/// `fused_*`, `foreach_*_`, ...) mutate storage without synchronization,
/// so a tensor being mutated must not be accessed from any other thread
/// at the same time. A shallow [`Clone`] shares the same storage and
/// counts as the same tensor for this rule. Share tensors across
/// threads for reading; give each thread its own deep copy (or
/// replica) when mutation is involved.
pub struct Tensor {
    pub(crate) handle: FlodlTensor,
}

/// View a typed slice as raw host bytes for the blob constructors.
fn typed_bytes<T>(data: &[T]) -> &[u8] {
    // Safety: u8 has alignment 1 and any initialized f32/f64/i64 slice is
    // readable as `size_of_val(data)` plain bytes; the lifetime stays tied
    // to the input slice.
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data)) }
}

// Safety: libtorch tensors are internally reference-counted with atomic
// refcounts, and concurrent READS (ops that allocate new outputs) are
// thread-safe. In-place ops (`add_`, `copy_`, `zero_`, `fused_*`,
// `foreach_*_`, ...) mutate storage through `&self` WITHOUT
// synchronization: callers must guarantee that a tensor being mutated is
// not concurrently accessed from any other thread, including through
// shallow clones, which share the same storage. flodl upholds this
// internally by replication (each worker owns its tensors) and
// single-consumer snapshot buffers, never by locking.
unsafe impl Send for Tensor {}
unsafe impl Sync for Tensor {}

impl Drop for Tensor {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            LIVE_TENSOR_COUNT.fetch_sub(1, Ordering::Relaxed);
            unsafe { ffi::flodl_free_tensor(self.handle) };
        }
    }
}

impl Clone for Tensor {
    /// Shallow clone: a new C++ Tensor handle sharing the same TensorImpl
    /// (and thus the same data storage). Cheap — just bumps libtorch's
    /// internal refcount, no data copied. Safe for the common case
    /// (reads, passing tensors around) because out-of-place ops allocate
    /// fresh outputs and never mutate their inputs.
    ///
    /// **Aliasing warning.** Storage is SHARED, so an in-place op (`_`
    /// suffix: `add_`, `copy_`, `mul_scalar_`, fused optimizer kernels,
    /// ...) through either handle mutates both. Unlike PyTorch, where
    /// `.clone()` is a *deep* copy, flodl's `Clone` is shallow (Rust's
    /// `Clone` trait is the cheap-share default here). When you need an
    /// independent, owned duplicate — optimizer state seeded from a
    /// gradient, a snapshot held across later mutation — use
    /// [`Tensor::copy`](Self::copy) instead.
    ///
    /// # Panics
    ///
    /// Panics if the underlying FFI clone fails. This is deliberate: the
    /// `Clone` trait signature has no error channel, and the only way a
    /// refcount-bump clone fails is an unrecoverable condition (allocation
    /// failure / corrupt handle). Per flodl's panic policy, a defined named
    /// panic is used where there is no `Result` channel and the failure is
    /// unrecoverable; fallible tensor ops that CAN return `Result` do so
    /// instead of panicking.
    fn clone(&self) -> Self {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_shallow_clone(self.handle, &mut handle) };
        if !err.is_null() {
            let msg = unsafe { CStr::from_ptr(err) }
                .to_string_lossy()
                .into_owned();
            unsafe { ffi::flodl_free_string(err) };
            panic!("tensor clone failed: {}", msg);
        }
        Self::from_raw(handle)
    }
}

impl Tensor {
    /// Wrap a raw handle. The Tensor takes ownership.
    pub(crate) fn from_raw(handle: FlodlTensor) -> Self {
        debug_assert!(!handle.is_null());
        LIVE_TENSOR_COUNT.fetch_add(1, Ordering::Relaxed);
        Self { handle }
    }

    /// Access the raw handle (for passing to FFI in sibling modules).
    pub(crate) fn raw(&self) -> FlodlTensor {
        self.handle
    }

    // --- Creation ---

    /// Create a tensor filled with zeros.
    ///
    /// ```ignore
    /// let t = Tensor::zeros(&[2, 3], TensorOptions::default())?;
    /// assert_eq!(t.shape(), vec![2, 3]);
    /// ```
    pub fn zeros(shape: &[i64], opts: TensorOptions) -> Result<Self> {
        let mut shape = shape.to_vec();
        let mut handle: FlodlTensor = ptr::null_mut();
        let (dt, di) = opts.device.to_ffi();
        let err = unsafe {
            ffi::flodl_zeros(
                shape.as_mut_ptr(),
                shape.len() as i32,
                opts.dtype as i32,
                dt, di,
                &mut handle,
            )
        };
        check_err(err)?;
        Ok(Self::from_raw(handle))
    }

    /// Create a tensor filled with ones. Like `torch.ones()`.
    ///
    /// ```ignore
    /// let t = Tensor::ones(&[2, 3], TensorOptions::default())?;
    /// ```
    pub fn ones(shape: &[i64], opts: TensorOptions) -> Result<Self> {
        let mut shape = shape.to_vec();
        let mut handle: FlodlTensor = ptr::null_mut();
        let (dt, di) = opts.device.to_ffi();
        let err = unsafe {
            ffi::flodl_ones(
                shape.as_mut_ptr(),
                shape.len() as i32,
                opts.dtype as i32,
                dt, di,
                &mut handle,
            )
        };
        check_err(err)?;
        Ok(Self::from_raw(handle))
    }

    /// Create a tensor from f32 data. `data.len()` must equal the shape
    /// product; a mismatch is a loud error.
    ///
    /// ```ignore
    /// let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], Device::CPU)?;
    /// assert_eq!(t.shape(), vec![2, 2]);
    /// ```
    pub fn from_f32(data: &[f32], shape: &[i64], device: Device) -> Result<Self> {
        Self::from_blob_impl("Tensor::from_f32", typed_bytes(data), shape, DType::Float32, device)
    }

    /// Create a Float64 tensor from f64 data. Use when full double precision
    /// is needed (e.g. loss accumulation, high-precision metrics).
    /// `data.len()` must equal the shape product; a mismatch is a loud error.
    pub fn from_f64(data: &[f64], shape: &[i64], device: Device) -> Result<Self> {
        Self::from_blob_impl("Tensor::from_f64", typed_bytes(data), shape, DType::Float64, device)
    }

    /// Create an Int64 tensor from i64 data. Commonly used for class labels,
    /// token indices, and any integer indexing (e.g. `cross_entropy_loss` targets).
    /// `data.len()` must equal the shape product; a mismatch is a loud error.
    pub fn from_i64(data: &[i64], shape: &[i64], device: Device) -> Result<Self> {
        Self::from_blob_impl("Tensor::from_i64", typed_bytes(data), shape, DType::Int64, device)
    }

    /// Construct a tensor from raw little-endian host bytes at the
    /// given `dtype`. The `data` length must equal
    /// `shape.iter().product::<i64>() as usize * dtype.element_size()`.
    /// libtorch copies the bytes (no aliasing into the input slice).
    ///
    /// Use when shuttling tensor payloads through formats like
    /// safetensors that store dtype + raw bytes; for typed inputs use
    /// the dtype-specific helpers (`from_f32`, `from_f64`, `from_i64`).
    pub fn from_blob(data: &[u8], shape: &[i64], dtype: DType, device: Device) -> Result<Self> {
        Self::from_blob_impl("Tensor::from_blob", data, shape, dtype, device)
    }

    /// Single validation home for every blob-style constructor. The length
    /// check is load-bearing: the shim's `flodl_from_blob` receives only a
    /// pointer and reads `numel × element_size` bytes trusting the caller,
    /// so an unchecked mismatch is an out-of-bounds read. `ctx` names the
    /// public constructor so errors point at the call the user wrote.
    fn from_blob_impl(
        ctx: &str,
        data: &[u8],
        shape: &[i64],
        dtype: DType,
        device: Device,
    ) -> Result<Self> {
        let numel = shape
            .iter()
            .try_fold(1i64, |acc, &d| if d < 0 { None } else { acc.checked_mul(d) })
            .ok_or_else(|| {
                TensorError::new(&format!(
                    "{ctx}: invalid shape {shape:?} (negative or overflowing dimension)"
                ))
            })?;
        let expected = (numel as usize)
            .checked_mul(dtype.element_size())
            .ok_or_else(|| {
                TensorError::new(&format!(
                    "{ctx}: invalid shape {shape:?} (byte size overflows usize)"
                ))
            })?;
        if data.len() != expected {
            return Err(TensorError::new(&format!(
                "{ctx}: data is {} bytes, expected {expected} \
                 (numel={numel} × {} bytes/elem for {dtype:?})",
                data.len(),
                dtype.element_size(),
            )));
        }
        let mut shape = shape.to_vec();
        let mut handle: FlodlTensor = ptr::null_mut();
        let (dt, di) = device.to_ffi();
        let err = unsafe {
            ffi::flodl_from_blob(
                data.as_ptr() as *mut c_void,
                shape.as_mut_ptr(),
                shape.len() as i32,
                dtype as i32,
                dt, di,
                &mut handle,
            )
        };
        check_err(err)?;
        Ok(Self::from_raw(handle))
    }

    // --- Like constructors ---

    /// Create a tensor of zeros with the same shape, dtype, and device as `t`.
    pub fn zeros_like(t: &Tensor) -> Result<Tensor> {
        ffi_call!(flodl_zeros_like, t.handle)
    }

    /// Create a tensor of ones with the same shape, dtype, and device as `t`.
    pub fn ones_like(t: &Tensor) -> Result<Tensor> {
        ffi_call!(flodl_ones_like, t.handle)
    }

    /// Create a tensor filled with `value`, same shape/dtype/device as `t`.
    pub fn full_like(t: &Tensor, value: f64) -> Result<Tensor> {
        ffi_call!(flodl_full_like, t.handle, value)
    }

    /// Create a tensor with uniform random values in [0, 1), same shape/dtype/device as `t`.
    pub fn rand_like(t: &Tensor) -> Result<Tensor> {
        ffi_call!(flodl_rand_like, t.handle)
    }

    /// Create a tensor with standard normal random values, same shape/dtype/device as `t`.
    pub fn randn_like(t: &Tensor) -> Result<Tensor> {
        ffi_call!(flodl_randn_like, t.handle)
    }

    // --- Random ---

    /// Create a tensor with uniform random values in [0, 1).
    pub fn rand(shape: &[i64], opts: TensorOptions) -> Result<Self> {
        let mut shape = shape.to_vec();
        let mut handle: FlodlTensor = ptr::null_mut();
        let (dt, di) = opts.device.to_ffi();
        let err = unsafe {
            ffi::flodl_rand(
                shape.as_mut_ptr(), shape.len() as i32,
                opts.dtype as i32, dt, di,
                &mut handle,
            )
        };
        check_err(err)?;
        Ok(Self::from_raw(handle))
    }

    /// Create a tensor with standard normal random values (mean=0, std=1).
    pub fn randn(shape: &[i64], opts: TensorOptions) -> Result<Self> {
        let mut shape = shape.to_vec();
        let mut handle: FlodlTensor = ptr::null_mut();
        let (dt, di) = opts.device.to_ffi();
        let err = unsafe {
            ffi::flodl_randn(
                shape.as_mut_ptr(), shape.len() as i32,
                opts.dtype as i32, dt, di,
                &mut handle,
            )
        };
        check_err(err)?;
        Ok(Self::from_raw(handle))
    }

    // --- Tensor creation (additional) ---

    /// Create evenly spaced values.
    pub fn linspace(start: f64, end: f64, steps: i64, opts: TensorOptions) -> Result<Self> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let (dt, di) = opts.device.to_ffi();
        let err = unsafe {
            ffi::flodl_linspace(start, end, steps, opts.dtype as i32, dt, di, &mut handle)
        };
        check_err(err)?;
        Ok(Self::from_raw(handle))
    }

    /// Create a range of values [start, end) with given step.
    pub fn arange(start: f64, end: f64, step: f64, opts: TensorOptions) -> Result<Self> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let (dt, di) = opts.device.to_ffi();
        let err = unsafe {
            ffi::flodl_arange(start, end, step, opts.dtype as i32, dt, di, &mut handle)
        };
        check_err(err)?;
        Ok(Self::from_raw(handle))
    }

    /// Create an identity matrix of size n x n.
    pub fn eye(n: i64, opts: TensorOptions) -> Result<Self> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let (dt, di) = opts.device.to_ffi();
        let err = unsafe {
            ffi::flodl_eye(n, opts.dtype as i32, dt, di, &mut handle)
        };
        check_err(err)?;
        Ok(Self::from_raw(handle))
    }

    /// Create a tensor filled with a scalar value.
    pub fn full(shape: &[i64], value: f64, opts: TensorOptions) -> Result<Self> {
        let mut shape = shape.to_vec();
        let mut handle: FlodlTensor = ptr::null_mut();
        let (dt, di) = opts.device.to_ffi();
        let err = unsafe {
            ffi::flodl_full(
                shape.as_mut_ptr(), shape.len() as i32, value,
                opts.dtype as i32, dt, di, &mut handle,
            )
        };
        check_err(err)?;
        Ok(Self::from_raw(handle))
    }

    /// Random permutation of integers `[0, n)`.
    pub fn randperm(n: i64, opts: TensorOptions) -> Result<Self> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let (dt, di) = opts.device.to_ffi();
        let err = unsafe {
            ffi::flodl_randperm(n, opts.dtype as i32, dt, di, &mut handle)
        };
        check_err(err)?;
        Ok(Self::from_raw(handle))
    }

    /// Create a tensor with random integers in `[low, high)`.
    pub fn randint(low: i64, high: i64, shape: &[i64], opts: TensorOptions) -> Result<Self> {
        let mut shape = shape.to_vec();
        let mut handle: FlodlTensor = ptr::null_mut();
        let (dt, di) = opts.device.to_ffi();
        let err = unsafe {
            ffi::flodl_randint(
                low, high,
                shape.as_mut_ptr(), shape.len() as i32,
                opts.dtype as i32, dt, di,
                &mut handle,
            )
        };
        check_err(err)?;
        Ok(Self::from_raw(handle))
    }

    /// Create an uninitialized tensor (like `torch.empty`).
    /// Contents are undefined -- use for pre-allocation before copy_.
    pub fn empty(shape: &[i64], opts: TensorOptions) -> Result<Self> {
        let mut shape = shape.to_vec();
        let mut handle: FlodlTensor = ptr::null_mut();
        let (dt, di) = opts.device.to_ffi();
        let err = unsafe {
            ffi::flodl_empty(
                shape.as_mut_ptr(), shape.len() as i32,
                opts.dtype as i32, dt, di,
                &mut handle,
            )
        };
        check_err(err)?;
        Ok(Self::from_raw(handle))
    }

    /// One-hot encode an Int64 tensor of class indices.
    /// Returns a Float32 tensor with shape `[..., num_classes]`.
    pub fn one_hot(&self, num_classes: i64) -> Result<Tensor> {
        ffi_call!(flodl_one_hot, self.handle, num_classes)
    }

    /// Sample 0/1 from Bernoulli distribution with given probabilities.
    /// `self` contains probabilities in [0, 1].
    pub fn bernoulli(&self) -> Result<Tensor> {
        ffi_call!(flodl_bernoulli, self.handle)
    }

    // --- Metadata ---

    /// Number of dimensions (rank). Like `tensor.ndim` in PyTorch.
    pub fn ndim(&self) -> usize {
        unsafe { ffi::flodl_ndim(self.handle) as usize }
    }

    /// Shape of each dimension as a Vec. Like `tensor.shape` in PyTorch.
    pub fn shape(&self) -> Vec<i64> {
        let n = self.ndim();
        (0..n)
            .map(|i| unsafe { ffi::flodl_shape(self.handle, i as i32) })
            .collect()
    }

    /// Total number of elements (product of all dimensions). Like `tensor.numel()`.
    pub fn numel(&self) -> i64 {
        unsafe { ffi::flodl_numel(self.handle) }
    }

    /// Total size in bytes of the tensor's data. Like `tensor.nbytes` in PyTorch.
    pub fn nbytes(&self) -> usize {
        self.numel() as usize * self.dtype().element_size()
    }

    /// Size in bytes of the underlying storage buffer. Like
    /// `tensor.untyped_storage().nbytes()` in PyTorch.
    ///
    /// For a tensor that owns its data this equals [`nbytes`](Self::nbytes)
    /// (up to alignment). For a **view** (`select`/`narrow`/`slice`) it is
    /// the size of the whole backing buffer — which is what a clone of the
    /// view actually keeps alive. Retention accounting must price views by
    /// this, not by their logical size.
    pub fn storage_nbytes(&self) -> usize {
        unsafe { ffi::flodl_storage_nbytes(self.handle) as usize }
    }

    /// Element data type of this tensor. Like `tensor.dtype` in PyTorch.
    pub fn dtype(&self) -> DType {
        DType::from_raw(unsafe { ffi::flodl_dtype(self.handle) })
    }

    /// Device where this tensor's data resides (CPU or CUDA). Like `tensor.device`.
    pub fn device(&self) -> Device {
        let dt = unsafe { ffi::flodl_device_type(self.handle) };
        let di = unsafe { ffi::flodl_device_index(self.handle) };
        Device::from_ffi(dt, di)
    }

    // --- Data access ---

    /// Copy tensor data to a `Vec<f32>`. Transparently moves to CPU first
    /// if the tensor lives on CUDA. Non-f32 dtypes are cast to f32 on
    /// device via libtorch before the host copy.
    pub fn to_f32_vec(&self) -> Result<Vec<f32>> {
        if self.dtype() != DType::Float32 {
            return self.to_dtype(DType::Float32)?.to_f32_vec();
        }
        let n = self.numel() as usize;
        let mut buf = vec![0f32; n];
        let bytes = (n * 4) as i64;
        let err = unsafe {
            ffi::flodl_copy_data(self.handle, buf.as_mut_ptr() as *mut c_void, bytes)
        };
        check_err(err)?;
        Ok(buf)
    }

    /// Copy tensor data to a `Vec<u8>` of raw little-endian bytes at
    /// the tensor's native dtype. The result is `numel * element_size()`
    /// bytes, ready to write into a format like safetensors that stores
    /// dtype + raw bytes. Moves to CPU first if needed.
    ///
    /// Use [`to_f32_vec`](Self::to_f32_vec) / [`to_f64_vec`](Self::to_f64_vec)
    /// when you want a typed cast; use this when you want the bytes
    /// exactly as libtorch lays them out for the current dtype.
    pub fn to_blob(&self) -> Result<Vec<u8>> {
        let bytes = self.numel() as usize * self.dtype().element_size();
        let mut buf = vec![0u8; bytes];
        let err = unsafe {
            ffi::flodl_copy_data(self.handle, buf.as_mut_ptr() as *mut c_void, bytes as i64)
        };
        check_err(err)?;
        Ok(buf)
    }

    /// Copy tensor data to a `Vec<f64>`. Moves to CPU if needed.
    /// Non-Float64 dtypes are cast to f64 on device first: exact for
    /// f16/bf16/f32/i32, and exact up to 2^53 for Int64 (the f64
    /// mantissa limit).
    pub fn to_f64_vec(&self) -> Result<Vec<f64>> {
        if self.dtype() != DType::Float64 {
            return self.to_dtype(DType::Float64)?.to_f64_vec();
        }
        let n = self.numel() as usize;
        let mut buf = vec![0.0f64; n];
        let bytes = (n * 8) as i64;
        let err = unsafe {
            ffi::flodl_copy_data(self.handle, buf.as_mut_ptr() as *mut c_void, bytes)
        };
        check_err(err)?;
        Ok(buf)
    }

    /// Copy tensor data to a `Vec<i64>`. Moves to CPU if needed.
    /// Non-Int64 dtypes are cast on device first; floats truncate
    /// toward zero, like PyTorch's `.long()`.
    pub fn to_i64_vec(&self) -> Result<Vec<i64>> {
        if self.dtype() != DType::Int64 {
            return self.to_dtype(DType::Int64)?.to_i64_vec();
        }
        let n = self.numel() as usize;
        let mut buf = vec![0i64; n];
        let bytes = (n * 8) as i64;
        let err = unsafe {
            ffi::flodl_copy_data(self.handle, buf.as_mut_ptr() as *mut c_void, bytes)
        };
        check_err(err)?;
        Ok(buf)
    }

    /// Extract a scalar value as f64. Like PyTorch's `.item()`.
    ///
    /// The tensor must contain exactly one element (any shape is fine,
    /// e.g. `[1]`, `[1, 1]`, or `[]`). Returns an error otherwise.
    /// Works for every dtype; integer values above 2^53 lose precision
    /// (the f64 mantissa limit, inherent to the return type).
    ///
    /// ```ignore
    /// let loss_val = loss_tensor.item()?;
    /// println!("loss: {:.4}", loss_val);
    /// ```
    pub fn item(&self) -> Result<f64> {
        if self.numel() != 1 {
            return Err(TensorError::new(&format!(
                "item() requires exactly 1 element, got {} (shape {:?})",
                self.numel(), self.shape()
            )));
        }
        if self.dtype() != DType::Float64 {
            // Cast on device first: flodl_copy_data copies NATIVE bytes, so
            // reading a non-f64 tensor into an f64 buffer would reinterpret
            // bit patterns (or under-fill the buffer), not convert values.
            return self.to_dtype(DType::Float64)?.item();
        }
        let mut buf = [0.0f64; 1];
        let err = unsafe {
            ffi::flodl_copy_data(self.handle, buf.as_mut_ptr() as *mut c_void, 8)
        };
        check_err(err)?;
        Ok(buf[0])
    }

    // --- Device ---

    /// Move this tensor to a different device (CPU or CUDA).
    /// Returns a new tensor; the original is unchanged.
    ///
    /// ```ignore
    /// let gpu = t.to_device(Device::CUDA(0))?;
    /// ```
    pub fn to_device(&self, device: Device) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let (dt, di) = device.to_ffi();
        let err = unsafe { ffi::flodl_to_device(self.handle, dt, di, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Move this tensor to the same device as `other`.
    /// No-op (returns a clone) if both are already on the same device.
    ///
    /// ```ignore
    /// let x = x.to_device_of(&weights)?;  // ensure same device
    /// ```
    pub fn to_device_of(&self, other: &Tensor) -> Result<Tensor> {
        let target = other.device();
        if self.device() == target {
            return Ok(self.clone());
        }
        self.to_device(target)
    }

    /// Non-blocking device transfer. Combined with [`Tensor::pin_memory`] for CPU->GPU,
    /// this allows the transfer to overlap with host computation.
    ///
    /// ```ignore
    /// let pinned = cpu_tensor.pin_memory()?;
    /// let gpu = pinned.to_device_async(Device::CUDA(0))?;
    /// // ... do CPU work while transfer runs ...
    /// cuda_synchronize(0); // ensure transfer is done before using gpu tensor
    /// ```
    pub fn to_device_async(&self, device: Device) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let (dt, di) = device.to_ffi();
        let err = unsafe { ffi::flodl_to_device_async(self.handle, dt, di, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Mark this tensor's storage as in use by `stream`, extending its
    /// caching-allocator lifetime until that stream passes the current
    /// point. Required whenever a tensor allocated on one stream is
    /// consumed on another and may be dropped while the consumer's
    /// kernels are still in flight — without it the allocator only
    /// guards the block against the ALLOCATION stream and can hand the
    /// freed block to a new allocation that overwrites it mid-read.
    pub fn record_stream(&self, stream: &crate::tensor::cuda_stream::CudaStream) -> Result<()> {
        let err = unsafe {
            ffi::flodl_tensor_record_stream(self.handle, stream.as_ptr())
        };
        check_err(err)
    }

    // --- Autograd ---

    /// Set requires_grad on this tensor. Returns a new tensor that shares
    /// storage but has the grad flag set. This enables libtorch's native
    /// autograd tracking for all subsequent operations.
    pub fn set_requires_grad(&self, requires_grad: bool) -> Result<Tensor> {
        ffi_call!(flodl_set_requires_grad, self.handle, requires_grad as i32)
    }

    /// Check whether this tensor requires gradient computation.
    pub fn requires_grad(&self) -> bool {
        unsafe { ffi::flodl_requires_grad(self.handle) != 0 }
    }

    /// Run backward pass from this scalar tensor. Populates .grad() on
    /// all leaf tensors in the computation graph.
    pub fn backward(&self) -> Result<()> {
        let err = unsafe { ffi::flodl_backward(self.handle) };
        check_err(err)
    }

    /// Get the accumulated gradient for this tensor, if any.
    /// Returns None if no gradient has been computed.
    ///
    /// Panics if the FFI call itself fails (broken tensor handle):
    /// mapping that to `None` would be indistinguishable from "no
    /// gradient yet" and make optimizers silently skip the parameter.
    pub fn grad(&self) -> Option<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_grad(self.handle, &mut handle) };
        if !err.is_null() {
            let msg = unsafe { CStr::from_ptr(err) }.to_string_lossy().into_owned();
            unsafe { ffi::flodl_free_string(err) };
            panic!("Tensor::grad failed: {msg}");
        }
        if handle.is_null() {
            None
        } else {
            Some(Tensor::from_raw(handle))
        }
    }

    /// Replace the gradient tensor (for gradient clipping / unscaling).
    pub fn set_grad(&self, grad: &Tensor) -> Result<()> {
        let err = unsafe { ffi::flodl_set_grad(self.handle, grad.handle) };
        check_err(err)
    }

    /// Zero out the accumulated gradient.
    pub fn zero_grad(&self) -> Result<()> {
        let err = unsafe { ffi::flodl_zero_grad(self.handle) };
        check_err(err)
    }

    /// Null out the gradient pointer instead of zeroing the data.
    /// No CUDA kernel — just resets the grad tensor to undefined.
    /// This is what PyTorch does by default since 1.7.
    pub fn zero_grad_set_to_none(&self) {
        unsafe { ffi::flodl_zero_grad_set_to_none(self.handle) }
    }

    /// Fused clip_grad_norm: compute global L2 norm across all param grads
    /// and scale in-place if it exceeds max_norm. Single C++ call.
    /// Returns the original total norm before clipping.
    pub fn clip_grad_norm_fused(params: &[Tensor], max_norm: f64) -> Result<f64> {
        if params.is_empty() {
            return Ok(0.0);
        }
        let mut handles: Vec<FlodlTensor> = params.iter().map(|t| t.handle).collect();
        let mut total_norm: f64 = 0.0;
        let err = unsafe {
            ffi::flodl_clip_grad_norm(
                handles.as_mut_ptr(),
                handles.len() as i32,
                max_norm,
                &mut total_norm,
            )
        };
        check_err(err)?;
        Ok(total_norm)
    }

    /// Whether this tensor is a leaf in the autograd graph.
    /// A tensor is a leaf if it was created by the user (not by an op)
    /// or if it doesn't require grad.
    pub fn is_leaf(&self) -> bool {
        unsafe { ffi::flodl_is_leaf(self.handle) != 0 }
    }

    /// Eagerly materialize the AccumulateGrad node for a leaf tensor
    /// with `requires_grad=true`, pinning its stream to the current
    /// CUDA stream at the moment of this call. Returns a handle that
    /// keeps the node alive; drop it to free.
    ///
    /// See [`Variable::ensure_grad_accumulator`](crate::autograd::Variable::ensure_grad_accumulator)
    /// for the motivation. Returns `Ok(None)` for non-leaf or
    /// non-requires-grad tensors.
    pub fn ensure_grad_accumulator(&self) -> Result<Option<GradAccumulatorHandle>> {
        let mut handle: *mut std::ffi::c_void = std::ptr::null_mut();
        let err = unsafe { ffi::flodl_ensure_grad_accumulator(self.handle, &mut handle) };
        check_err(err)?;
        if handle.is_null() {
            Ok(None)
        } else {
            Ok(Some(GradAccumulatorHandle { handle }))
        }
    }

    /// Count unique autograd nodes reachable from this tensor's grad_fn.
    /// Returns 0 for leaf tensors or tensors without gradient tracking.
    /// This is the number of backward operations libtorch will execute.
    pub fn autograd_node_count(&self) -> i64 {
        unsafe { ffi::flodl_autograd_node_count(self.handle) }
    }

    /// Detach from the computation graph. Returns a new tensor that shares
    /// storage but has no autograd history.
    pub fn detach(&self) -> Result<Tensor> {
        ffi_call!(flodl_detach, self.handle)
    }

    /// In-place detach: sever the grad_fn chain on this tensor without
    /// allocating a new handle. After this call the tensor's autograd_meta
    /// no longer references any C++ Node objects, allowing the autograd
    /// graph to be freed immediately rather than when the tensor is dropped.
    pub fn detach_(&self) -> Result<()> {
        let err = unsafe { ffi::flodl_detach_(self.handle) };
        check_err(err)
    }

    /// Deep copy: a new tensor with its OWN storage, holding the same
    /// data. Unlike [`Clone`](Tensor#impl-Clone) (which is *shallow* — a
    /// new handle aliasing the same storage) and unlike [`detach`](Self::detach)
    /// (which also shares storage), the returned tensor is fully
    /// independent: a later in-place op (`add_`, `copy_`, `mul_scalar_`,
    /// fused optimizer kernels, ...) on either side does not affect the
    /// other.
    ///
    /// This is the flodl spelling of PyTorch's deep `.clone()`. Reach for
    /// it whenever you keep a value while the original (or a shared alias)
    /// may be mutated in place later: optimizer state seeded from a
    /// gradient, an EMA / snapshot of weights, a view you want to own.
    pub fn copy(&self) -> Result<Tensor> {
        ffi_call!(flodl_deep_clone, self.handle)
    }

    // --- In-place operations ---

    /// In-place add: self += other
    pub fn add_(&self, other: &Tensor) -> Result<()> {
        let err = unsafe { ffi::flodl_add_(self.handle, other.handle) };
        check_err(err)
    }

    /// In-place subtract: self -= other
    pub fn sub_(&self, other: &Tensor) -> Result<()> {
        let err = unsafe { ffi::flodl_sub_(self.handle, other.handle) };
        check_err(err)
    }

    /// In-place scalar multiply: self *= scalar
    pub fn mul_scalar_(&self, scalar: f64) -> Result<()> {
        let err = unsafe { ffi::flodl_mul_scalar_(self.handle, scalar) };
        check_err(err)
    }

    /// In-place scalar add: self += scalar
    pub fn add_scalar_(&self, scalar: f64) -> Result<()> {
        let err = unsafe { ffi::flodl_add_scalar_(self.handle, scalar) };
        check_err(err)
    }

    /// In-place zero: self = 0
    pub fn zero_(&self) -> Result<()> {
        let err = unsafe { ffi::flodl_zero_(self.handle) };
        check_err(err)
    }

    /// In-place multiply: self *= other (tensor-tensor)
    pub fn mul_(&self, other: &Tensor) -> Result<()> {
        let err = unsafe { ffi::flodl_mul_(self.handle, other.handle) };
        check_err(err)
    }

    /// In-place divide by scalar: self /= scalar
    pub fn div_scalar_(&self, scalar: f64) -> Result<()> {
        let err = unsafe { ffi::flodl_div_scalar_(self.handle, scalar) };
        check_err(err)
    }

    /// In-place divide: self /= other (tensor-tensor)
    pub fn div_(&self, other: &Tensor) -> Result<()> {
        let err = unsafe { ffi::flodl_div_(self.handle, other.handle) };
        check_err(err)
    }

    /// In-place fill: set all elements to `value`
    pub fn fill_(&self, value: f64) -> Result<()> {
        let err = unsafe { ffi::flodl_fill_(self.handle, value) };
        check_err(err)
    }

    /// In-place copy: `self = src`.
    ///
    /// Copies the data from `src` into `self`. Both tensors must have the
    /// same shape. When `non_blocking` is true, cross-device copies may
    /// be asynchronous (useful inside CUDA Graph capture).
    pub fn copy_(&self, src: &Tensor, non_blocking: bool) -> Result<()> {
        let err = unsafe { ffi::flodl_copy_(self.handle, src.handle, non_blocking as i32) };
        check_err(err)
    }

    // --- Optimizer operations ---

    /// Fused Adam/AdamW step: updates param, m, and v tensors in-place.
    #[allow(clippy::too_many_arguments)]
    ///
    /// Performs the full Adam update in a single FFI call (~5 kernel launches
    /// instead of ~16), eliminating temporary tensor allocations.
    ///
    /// - `self` — parameter tensor (updated in-place)
    /// - `grad` — gradient (read-only)
    /// - `m`, `v` — moment buffers (updated in-place)
    /// - `weight_decay` — 0.0 for Adam, >0 for AdamW (decoupled)
    /// - `step` — timestep for bias correction
    pub fn adam_step(
        &self, grad: &Tensor, m: &Tensor, v: &Tensor,
        lr: f64, beta1: f64, beta2: f64, eps: f64,
        weight_decay: f64, step: i64,
    ) -> Result<()> {
        let err = unsafe {
            ffi::flodl_adam_step(
                self.handle, grad.handle, m.handle, v.handle,
                lr, beta1, beta2, eps, weight_decay, step,
            )
        };
        check_err(err)
    }

    /// Perform Adam/AdamW update on all params in one C++ loop.
    /// Eliminates per-param FFI overhead. `lrs[i]` supports per-group LR.
    #[allow(clippy::too_many_arguments)]
    pub fn adam_step_batched(
        params: &[Tensor], grads: &[Tensor], ms: &[Tensor], vs: &[Tensor],
        lrs: &mut [f64], beta1: f64, beta2: f64, eps: f64,
        weight_decay: f64, step: i64,
    ) -> Result<()> {
        let count = params.len() as i32;
        let mut p_handles: Vec<FlodlTensor> = params.iter().map(|t| t.handle).collect();
        let mut g_handles: Vec<FlodlTensor> = grads.iter().map(|t| t.handle).collect();
        let mut m_handles: Vec<FlodlTensor> = ms.iter().map(|t| t.handle).collect();
        let mut v_handles: Vec<FlodlTensor> = vs.iter().map(|t| t.handle).collect();
        let err = unsafe {
            ffi::flodl_adam_step_batched(
                p_handles.as_mut_ptr(), g_handles.as_mut_ptr(),
                m_handles.as_mut_ptr(), v_handles.as_mut_ptr(),
                lrs.as_mut_ptr(), count,
                beta1, beta2, eps, weight_decay, step,
            )
        };
        check_err(err)
    }

    // --- Fused Adam/AdamW (multi-tensor kernel) ---
    // Uses libtorch's _fused_adam_ / _fused_adamw_ to perform the complete
    // Adam update across ALL params in a single kernel launch on CUDA.

    /// Fused Adam update (L2 weight decay) across all params in one kernel.
    ///
    /// On CUDA, this launches a single multi-tensor kernel instead of ~4N
    /// separate kernels for N parameters. On CPU, falls back to a fused loop.
    ///
    /// - `steps[i]` is param i's own step count (libtorch's per-param
    ///   `state_steps`): bias correction is computed per param, so a param
    ///   that starts updating late corrects from its first step, not the
    ///   global one. Must have the same length as `params`.
    /// - `grad_scale` / `found_inf`: pass `None` to skip mixed-precision integration.
    #[allow(clippy::too_many_arguments)]
    pub fn fused_adam_(
        params: &[Tensor], grads: &[Tensor], exp_avgs: &[Tensor], exp_avg_sqs: &[Tensor],
        lr: f64, beta1: f64, beta2: f64, eps: f64,
        weight_decay: f64, steps: &[i64],
        grad_scale: Option<&Tensor>, found_inf: Option<&Tensor>,
    ) -> Result<()> {
        if params.is_empty() { return Ok(()); }
        if steps.len() != params.len() {
            return Err(TensorError::new(&format!(
                "fused_adam_: steps length {} does not match params length {}",
                steps.len(), params.len()
            )));
        }
        let count = params.len() as i32;
        let mut p = Self::handles(params);
        let mut g = Self::handles(grads);
        let mut m = Self::handles(exp_avgs);
        let mut v = Self::handles(exp_avg_sqs);
        let gs = grad_scale.map_or(ptr::null_mut(), |t| t.handle);
        let fi = found_inf.map_or(ptr::null_mut(), |t| t.handle);
        let err = unsafe {
            ffi::flodl_fused_adam_(
                p.as_mut_ptr(), g.as_mut_ptr(), m.as_mut_ptr(), v.as_mut_ptr(),
                count, lr, beta1, beta2, eps, weight_decay, steps.as_ptr(), gs, fi,
            )
        };
        check_err(err)
    }

    /// Fused AdamW update (decoupled weight decay) across all params in one kernel.
    ///
    /// Same as [`Tensor::fused_adam_`] but applies decoupled weight decay:
    /// `param *= (1 - lr * weight_decay)` before the Adam step.
    /// With `weight_decay = 0.0`, identical to `fused_adam_`.
    #[allow(clippy::too_many_arguments)]
    pub fn fused_adamw_(
        params: &[Tensor], grads: &[Tensor], exp_avgs: &[Tensor], exp_avg_sqs: &[Tensor],
        lr: f64, beta1: f64, beta2: f64, eps: f64,
        weight_decay: f64, steps: &[i64],
        grad_scale: Option<&Tensor>, found_inf: Option<&Tensor>,
    ) -> Result<()> {
        if params.is_empty() { return Ok(()); }
        if steps.len() != params.len() {
            return Err(TensorError::new(&format!(
                "fused_adamw_: steps length {} does not match params length {}",
                steps.len(), params.len()
            )));
        }
        let count = params.len() as i32;
        let mut p = Self::handles(params);
        let mut g = Self::handles(grads);
        let mut m = Self::handles(exp_avgs);
        let mut v = Self::handles(exp_avg_sqs);
        let gs = grad_scale.map_or(ptr::null_mut(), |t| t.handle);
        let fi = found_inf.map_or(ptr::null_mut(), |t| t.handle);
        let err = unsafe {
            ffi::flodl_fused_adamw_(
                p.as_mut_ptr(), g.as_mut_ptr(), m.as_mut_ptr(), v.as_mut_ptr(),
                count, lr, beta1, beta2, eps, weight_decay, steps.as_ptr(), gs, fi,
            )
        };
        check_err(err)
    }

    /// Collect FlodlTensor handles from a slice.
    fn handles(tensors: &[Tensor]) -> Vec<FlodlTensor> {
        tensors.iter().map(|t| t.handle).collect()
    }

    // --- Multi-tensor foreach operations ---
    // These use libtorch's _foreach_* ops which batch the same operation
    // across all tensors into fewer kernel launches on CUDA.

    /// In-place add scalar to all tensors: `tensors[i] += scalar`.
    /// Single batched kernel on CUDA instead of N separate launches.
    pub fn foreach_add_scalar_(tensors: &[Tensor], scalar: f64) -> Result<()> {
        if tensors.is_empty() { return Ok(()); }
        let mut handles: Vec<FlodlTensor> = tensors.iter().map(|t| t.handle).collect();
        let err = unsafe {
            ffi::flodl_foreach_add_scalar_(handles.as_mut_ptr(), handles.len() as i32, scalar)
        };
        check_err(err)
    }

    /// In-place multiply all tensors by scalar: `tensors[i] *= scalar`.
    /// Single batched kernel on CUDA instead of N separate launches.
    pub fn foreach_mul_scalar_(tensors: &[Tensor], scalar: f64) -> Result<()> {
        if tensors.is_empty() { return Ok(()); }
        let mut handles: Vec<FlodlTensor> = tensors.iter().map(|t| t.handle).collect();
        let err = unsafe {
            ffi::flodl_foreach_mul_scalar_(handles.as_mut_ptr(), handles.len() as i32, scalar)
        };
        check_err(err)
    }

    /// In-place zero all tensors: `tensors[i] = 0`.
    /// Single batched kernel on CUDA instead of N separate launches.
    pub fn foreach_zero_(tensors: &[Tensor]) -> Result<()> {
        if tensors.is_empty() { return Ok(()); }
        let mut handles: Vec<FlodlTensor> = tensors.iter().map(|t| t.handle).collect();
        let err = unsafe {
            ffi::flodl_foreach_zero_(handles.as_mut_ptr(), handles.len() as i32)
        };
        check_err(err)
    }

    /// In-place add two tensor lists: `tensors1[i] += alpha * tensors2[i]`.
    /// Single batched kernel on CUDA instead of N separate launches.
    pub fn foreach_add_list_(tensors1: &[Tensor], tensors2: &[Tensor], alpha: f64) -> Result<()> {
        if tensors1.is_empty() { return Ok(()); }
        if tensors1.len() != tensors2.len() {
            return Err(TensorError::new(&format!(
                "foreach_add_list_: list length mismatch ({} vs {})",
                tensors1.len(), tensors2.len(),
            )));
        }
        let mut h1: Vec<FlodlTensor> = tensors1.iter().map(|t| t.handle).collect();
        let mut h2: Vec<FlodlTensor> = tensors2.iter().map(|t| t.handle).collect();
        let err = unsafe {
            ffi::flodl_foreach_add_list_(
                h1.as_mut_ptr(), h2.as_mut_ptr(), h1.len() as i32, alpha,
            )
        };
        check_err(err)
    }

    /// Compute per-tensor norms. Returns a Vec of scalar tensors.
    /// Single batched kernel on CUDA instead of N separate norm calls.
    pub fn foreach_norm(tensors: &[Tensor], ord: f64) -> Result<Vec<Tensor>> {
        if tensors.is_empty() { return Ok(vec![]); }
        let mut handles: Vec<FlodlTensor> = tensors.iter().map(|t| t.handle).collect();
        let mut results: Vec<FlodlTensor> = vec![ptr::null_mut(); tensors.len()];
        let err = unsafe {
            ffi::flodl_foreach_norm(
                handles.as_mut_ptr(), handles.len() as i32, ord,
                results.as_mut_ptr(),
            )
        };
        check_err(err)?;
        Ok(results.into_iter().map(Tensor::from_raw).collect())
    }

    /// In-place lerp: `tensors1[i] += weight * (tensors2[i] - tensors1[i])`.
    /// Single batched kernel on CUDA instead of N separate launches.
    pub fn foreach_lerp_scalar_(tensors1: &[Tensor], tensors2: &[Tensor], weight: f64) -> Result<()> {
        if tensors1.is_empty() { return Ok(()); }
        if tensors1.len() != tensors2.len() {
            return Err(TensorError::new(&format!(
                "foreach_lerp_scalar_: list length mismatch ({} vs {})",
                tensors1.len(), tensors2.len(),
            )));
        }
        let mut h1: Vec<FlodlTensor> = tensors1.iter().map(|t| t.handle).collect();
        let mut h2: Vec<FlodlTensor> = tensors2.iter().map(|t| t.handle).collect();
        let err = unsafe {
            ffi::flodl_foreach_lerp_scalar_(
                h1.as_mut_ptr(), h2.as_mut_ptr(), h1.len() as i32, weight,
            )
        };
        check_err(err)
    }

    /// In-place sqrt: `tensors[i] = sqrt(tensors[i])`.
    /// Single batched kernel on CUDA instead of N separate launches.
    pub fn foreach_sqrt_(tensors: &[Tensor]) -> Result<()> {
        if tensors.is_empty() { return Ok(()); }
        let mut handles: Vec<FlodlTensor> = tensors.iter().map(|t| t.handle).collect();
        let err = unsafe {
            ffi::flodl_foreach_sqrt_(handles.as_mut_ptr(), handles.len() as i32)
        };
        check_err(err)
    }

    // --- Pinned memory ---

    /// Copy this CPU tensor into page-locked (pinned) memory.
    ///
    /// Pinned memory enables async CPU->GPU transfers via `cudaMemcpyAsync`.
    /// Only valid for CPU tensors. Returns a new tensor in pinned memory.
    pub fn pin_memory(&self) -> Result<Tensor> {
        ffi_call!(flodl_pin_memory, self.handle)
    }

    /// Returns true if this tensor is stored in pinned (page-locked) memory.
    pub fn is_pinned(&self) -> bool {
        unsafe { ffi::flodl_is_pinned(self.handle) != 0 }
    }

    // --- Memory format ---

    /// Convert to channels-last (NHWC) memory format. Only meaningful for 4D tensors.
    /// This is the Rust equivalent of `tensor.to(memory_format=torch.channels_last)`.
    pub fn to_channels_last(&self) -> Result<Tensor> {
        ffi_call!(flodl_to_channels_last, self.handle)
    }

    /// Returns true if this tensor is contiguous in channels-last format.
    pub fn is_channels_last(&self) -> bool {
        unsafe { ffi::flodl_is_channels_last(self.handle) != 0 }
    }

    /// Returns true if this tensor is contiguous in memory.
    pub fn is_contiguous(&self) -> bool {
        unsafe { ffi::flodl_is_contiguous(self.handle) != 0 }
    }
}

/// Opaque strong-reference handle to a leaf tensor's AccumulateGrad
/// node. Dropping it frees the node (unless a backward pass still
/// holds its own reference).
///
/// Safe to send across threads: the underlying object is an
/// immutable `std::shared_ptr<Node>` whose refcount is atomic.
pub struct GradAccumulatorHandle {
    handle: *mut std::ffi::c_void,
}

// The wrapped shared_ptr<Node> only stores a reference; dropping it
// from any thread is safe because libtorch's shared_ptr refcount is
// atomic and the Node itself is thread-safe.
unsafe impl Send for GradAccumulatorHandle {}
unsafe impl Sync for GradAccumulatorHandle {}

impl Drop for GradAccumulatorHandle {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { ffi::flodl_grad_accumulator_delete(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tensor({:?}, {:?}, {:?})",
            self.shape(),
            self.dtype(),
            self.device()
        )
    }
}

/// Returns the device to use in tests: CUDA when compiled with `--features cuda`
/// and a GPU is available, CPU otherwise.
#[cfg(test)]
pub fn test_device() -> Device {
    use std::sync::Once;
    static PRINT: Once = Once::new();
    let dev = if cfg!(feature = "cuda") && cuda_available() { Device::CUDA(0) } else { Device::CPU };
    PRINT.call_once(|| eprintln!("\n*** flodl test device: {} ***\n", dev));
    dev
}

/// Returns `TensorOptions` for tests (Float32 on `test_device()`).
#[cfg(test)]
pub fn test_opts() -> TensorOptions {
    TensorOptions { dtype: DType::Float32, device: test_device() }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
