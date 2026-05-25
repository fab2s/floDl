//! Reverse-mode automatic differentiation backed by libtorch.
//!
//! Variables wrap tensors with gradient tracking. When `requires_grad` is
//! true, libtorch's native autograd engine records operations and computes
//! gradients. Calling `backward()` delegates to libtorch's C++ backward
//! engine — no Rust-side graph walking.
//!
//! ```ignore
//! let x = Variable::new(tensor_x, true);
//! let w = Variable::new(tensor_w, true);
//! let loss = x.matmul(&w)?.sum()?;
//! loss.backward()?;
//! println!("{:?}", w.grad()); // gradient of loss w.r.t. w
//! ```

mod variable;
mod ops;
mod context;

pub use variable::Variable;
pub use context::{no_grad, is_grad_enabled, NoGradGuard};
pub use ops::{
    linear, gru_cell, lstm_cell, layer_norm,
    conv2d, conv1d, conv_transpose2d, conv_transpose1d,
    im2col, col2im,
    conv3d, conv_transpose3d,
    group_norm, instance_norm,
    max_pool2d, avg_pool2d, max_pool1d, avg_pool1d,
    adaptive_avg_pool2d, adaptive_max_pool2d,
    pixel_shuffle, pixel_unshuffle, bilinear,
    grid_sample, scaled_dot_product_attention, embedding, embedding_bag,
};

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
