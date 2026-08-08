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

mod context;
mod ops;
mod variable;

pub use context::{NoGradGuard, is_grad_enabled, no_grad};
pub use ops::{
    adaptive_avg_pool2d, adaptive_max_pool2d, avg_pool1d, avg_pool2d, bilinear, col2im,
    conv_transpose1d, conv_transpose2d, conv_transpose3d, conv1d, conv2d, conv3d, embedding,
    embedding_bag, grid_sample, group_norm, gru_cell, im2col, instance_norm, layer_norm, linear,
    lstm_cell, max_pool1d, max_pool2d, pixel_shuffle, pixel_unshuffle,
    scaled_dot_product_attention,
};
pub use variable::Variable;

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
