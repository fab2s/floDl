use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;

use crate::tensor::{DType, Device, Result, Tensor};

pub(crate) struct VariableInner {
    pub data: Tensor,
    /// Monotonic counter bumped every time `set_data` replaces the tensor.
    /// Module-side caches built from parameter tensors (e.g. the GRU/LSTM
    /// cuDNN param cache) key their validity on this.
    pub data_generation: u64,
}

/// A differentiable variable wrapping a Tensor.
///
/// Variables use libtorch's native autograd. When a tensor has
/// `requires_grad=true`, all standard operations build a C++ computation
/// graph automatically. Calling `backward()` runs libtorch's backward
/// engine — no Rust-side graph walking.
///
/// Internally uses `Rc<RefCell<>>` for shared ownership — the same
/// parameter can be referenced by both a Module and an Optimizer.
///
/// ```ignore
/// let w = Variable::new(Tensor::randn(&[3, 2], opts)?, true);
/// let x = Variable::new(Tensor::from_f32(&[1.0, 2.0, 3.0], &[1, 3], Device::CPU)?, false);
/// let loss = x.matmul(&w)?.sum()?;
/// loss.backward()?;
/// println!("{:?}", w.grad()); // gradient of loss w.r.t. w
/// ```
#[derive(Clone)]
pub struct Variable {
    pub(crate) inner: Rc<RefCell<VariableInner>>,
}

impl Variable {
    /// Create a leaf variable (parameter or input data).
    /// If `requires_grad` is true, libtorch will track operations for autodiff.
    ///
    /// # Panics
    ///
    /// Panics if gradient tracking cannot be enabled — e.g. on integer
    /// dtypes (libtorch: only floating-point tensors can require
    /// gradients). PyTorch raises the same error; silently returning a
    /// non-tracking variable would train nothing for this parameter.
    pub fn new(data: Tensor, requires_grad: bool) -> Self {
        let data = if requires_grad {
            // Set requires_grad on the C++ tensor so libtorch tracks ops
            data.set_requires_grad(true)
                .unwrap_or_else(|e| panic!("Variable::new: cannot enable gradient tracking: {e}"))
        } else {
            data
        };
        Variable {
            inner: Rc::new(RefCell::new(VariableInner {
                data,
                data_generation: 0,
            })),
        }
    }

    /// Wrap a tensor that already has the correct requires_grad flag set.
    /// Used by ops to wrap libtorch output tensors (which inherit autograd
    /// metadata from their inputs automatically).
    pub(crate) fn wrap(data: Tensor) -> Self {
        Variable {
            inner: Rc::new(RefCell::new(VariableInner {
                data,
                data_generation: 0,
            })),
        }
    }

    /// Get the underlying tensor data (shallow clone sharing storage).
    ///
    /// The returned `Tensor` shares the same memory as the variable's data —
    /// the same aliasing semantics as PyTorch's `.data`. In-place mutations
    /// on either side will be visible to both. If you need an independent
    /// copy, call `.data().copy()` instead. Because the share counts as the
    /// same tensor for concurrency purposes, the thread-safety rules on
    /// [`Tensor`](crate::tensor::Tensor#thread-safety) apply across it: never
    /// mutate in place while any other thread accesses the variable.
    pub fn data(&self) -> Tensor {
        self.inner.borrow().data.clone()
    }

    /// Stable identity of the shared autograd cell: clones of the same
    /// `Variable` return the same id, independent copies differ. This
    /// is what parameter collection dedups on (a module graph can reach
    /// the same shared parameter through several paths). Valid only
    /// while some clone is alive — ids can be recycled after the last
    /// clone drops.
    pub fn id(&self) -> usize {
        Rc::as_ptr(&self.inner) as usize
    }

    /// Get the accumulated gradient, if any.
    /// Reads from the C++ tensor's .grad() field.
    pub fn grad(&self) -> Option<Tensor> {
        self.inner.borrow().data.grad()
    }

    /// Replace the gradient tensor directly (e.g. for gradient clipping or unscaling).
    /// Equivalent to `param.grad = grad` in PyTorch.
    ///
    /// Panics if the FFI write fails (broken tensor handle): dropping the
    /// write silently would let the optimizer step with the old gradient.
    pub fn set_grad(&self, grad: Tensor) {
        self.inner
            .borrow()
            .data
            .set_grad(&grad)
            .unwrap_or_else(|e| panic!("Variable::set_grad: gradient write failed: {e}"));
    }

    /// Whether this variable tracks gradients.
    pub fn requires_grad(&self) -> bool {
        self.inner.borrow().data.requires_grad()
    }

    /// Change whether this variable tracks gradients.
    /// Replaces the inner data handle (the FFI returns a new handle sharing storage).
    /// All clones of this Variable share the same `Rc<RefCell>`, so the change
    /// is visible everywhere (module, optimizer, etc.).
    pub fn set_requires_grad(&self, requires_grad: bool) -> Result<()> {
        let data = self.inner.borrow().data.set_requires_grad(requires_grad)?;
        self.inner.borrow_mut().data = data;
        Ok(())
    }

    /// Whether this is a leaf variable (no grad_fn in libtorch).
    /// A leaf tensor is one created by the user, not by an operation.
    pub fn is_leaf(&self) -> bool {
        self.inner.borrow().data.is_leaf()
    }

    /// Force creation of the AccumulateGrad node for a leaf variable
    /// (`requires_grad=true`). The node's stream is captured from the
    /// current CUDA stream at the moment of this call. Returns a
    /// handle that keeps the node alive; store it for the lifetime of
    /// the worker.
    ///
    /// Call this under a [`StreamGuard`](crate::tensor::cuda_stream::StreamGuard)
    /// on the training compute stream during DDP worker setup so that
    /// gradient accumulation runs on the same stream as subsequent
    /// forward/backward passes. Without this, the node is created
    /// lazily on first `backward()` — inside the autograd engine's
    /// worker thread, whose current stream is the device default, and
    /// libtorch fires the "AccumulateGrad node's stream does not match"
    /// warning on every run.
    ///
    /// Returns `Ok(None)` for non-leaf or non-requires-grad variables.
    pub fn ensure_grad_accumulator(&self) -> Result<Option<crate::tensor::GradAccumulatorHandle>> {
        self.inner.borrow().data.ensure_grad_accumulator()
    }

    /// Count unique autograd nodes reachable from this variable's grad_fn.
    /// Returns 0 for leaf variables. Measures graph complexity — compare
    /// against Python's equivalent to detect decomposed-op bloat.
    pub fn autograd_node_count(&self) -> i64 {
        self.inner.borrow().data.autograd_node_count()
    }

    /// Shape of the underlying tensor (e.g. `[batch, features]`).
    pub fn shape(&self) -> Vec<i64> {
        self.inner.borrow().data.shape()
    }

    /// Data type of the underlying tensor (e.g. `Float32`, `Float16`).
    pub fn dtype(&self) -> DType {
        self.inner.borrow().data.dtype()
    }

    /// Device where the underlying tensor lives (`CPU` or `CUDA(n)`).
    pub fn device(&self) -> Device {
        self.inner.borrow().data.device()
    }

    /// Extract a scalar value as `f64`. The tensor must contain exactly one element.
    pub fn item(&self) -> Result<f64> {
        self.inner.borrow().data.item()
    }

    /// Zero out the accumulated gradient (fills `.grad()` with zeros).
    /// See also [`zero_grad_set_to_none`](Self::zero_grad_set_to_none) for the faster alternative.
    ///
    /// Panics if the FFI call fails (broken tensor handle): a silently
    /// skipped zero would leak the previous step's gradient into the next.
    pub fn zero_grad(&self) {
        self.inner
            .borrow()
            .data
            .zero_grad()
            .unwrap_or_else(|e| panic!("Variable::zero_grad failed: {e}"));
    }

    /// Null out the gradient instead of zeroing it. No CUDA kernel.
    pub fn zero_grad_set_to_none(&self) {
        self.inner.borrow().data.zero_grad_set_to_none();
    }

    /// Detach from the computation graph. Returns a new leaf variable
    /// sharing the same data tensor (detached) with no gradient tracking.
    ///
    /// Panics if the FFI call fails (broken tensor handle): falling back
    /// to the attached tensor would silently keep gradients flowing where
    /// the caller asked to cut them.
    pub fn detach(&self) -> Variable {
        let detached = self
            .inner
            .borrow()
            .data
            .detach()
            .unwrap_or_else(|e| panic!("Variable::detach failed: {e}"));
        Variable::wrap(detached)
    }

    /// Move to a different device. Returns a new leaf variable.
    pub fn to_device(&self, device: Device) -> Result<Variable> {
        if self.device() == device {
            return Ok(self.clone());
        }
        let moved = self.inner.borrow().data.to_device(device)?;
        Ok(Variable::new(moved, self.requires_grad()))
    }

    /// Replace the underlying tensor data (used by optimizers).
    /// Preserves the `requires_grad` flag from the current tensor.
    /// Bumps [`data_generation`](Self::data_generation) so caches built
    /// from the old tensor can detect the replacement.
    pub fn set_data(&self, data: Tensor) {
        let rg = self.requires_grad();
        let data = if rg {
            // Silently dropping tracking here would freeze this parameter.
            data.set_requires_grad(true).unwrap_or_else(|e| {
                panic!("Variable::set_data: cannot keep gradient tracking on the replacement tensor: {e}")
            })
        } else {
            data
        };
        let mut inner = self.inner.borrow_mut();
        inner.data = data;
        inner.data_generation += 1;
    }

    /// Monotonic generation of the underlying tensor: incremented every
    /// time [`set_data`](Self::set_data) replaces it (checkpoint load,
    /// dtype cast, device move). In-place mutation (`copy_`, optimizer
    /// steps) does not change it. Module caches built from parameter
    /// tensors key their validity on this.
    pub fn data_generation(&self) -> u64 {
        self.inner.borrow().data_generation
    }

    /// Total number of elements in the tensor (product of all dimensions).
    pub fn numel(&self) -> i64 {
        self.inner.borrow().data.numel()
    }

    /// Run backward pass from this scalar variable.
    /// Populates .grad() on all leaf variables in the computation graph.
    ///
    /// After backward completes, the tensor is detached in-place to
    /// immediately release the C++ grad_fn chain. Without this, the
    /// autograd Node objects stay alive until the Variable is dropped.
    pub fn backward(&self) -> Result<()> {
        let inner = self.inner.borrow();
        inner.data.backward()?;
        inner.data.detach_()?;
        Ok(())
    }
}

impl fmt::Debug for Variable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let inner = self.inner.borrow();
        write!(
            f,
            "Variable({:?}, {:?}, {:?}, requires_grad={})",
            inner.data.shape(),
            inner.data.dtype(),
            inner.data.device(),
            inner.data.requires_grad(),
        )
    }
}
