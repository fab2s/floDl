use crate::autograd::{Variable, NoGradGuard};
use crate::tensor::{Result, Tensor};
use super::Module;

/// Gaussian blur as a [`Module`] for use in [`FlowBuilder`](crate::graph::FlowBuilder) graphs.
///
/// Wraps [`gaussian_blur_2d`] with a fixed sigma. Stateless (no parameters or buffers).
///
/// ```ignore
/// FlowBuilder::from(GaussianBlur::new(1.5))
///     .tag("blurred")
///     .through(next_layer)
///     .build()
/// ```
pub struct GaussianBlur {
    sigma: f64,
}

impl GaussianBlur {
    /// Create a Gaussian blur module with the given sigma.
    pub fn new(sigma: f64) -> Self {
        Self { sigma }
    }
}

impl Module for GaussianBlur {
    fn name(&self) -> &str { "gaussian_blur" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        gaussian_blur_2d(input, self.sigma)
    }
}

/// Apply Gaussian blur to a 4-D input `[B, C, H, W]`.
///
/// Uses separable depthwise convolution (two 1-D passes) for efficiency.
/// Kernel size is `2 * ceil(3 * sigma) + 1`, matching OpenCV's default.
///
/// Operates at the Tensor level under a `NoGradGuard` — no autograd graph
/// is built, keeping memory flat across training batches.
///
/// ```ignore
/// let blurred = gaussian_blur_2d(&input, 1.5)?;
/// assert_eq!(blurred.shape(), input.shape());
/// ```
pub fn gaussian_blur_2d(input: &Variable, sigma: f64) -> Result<Variable> {
    let shape = input.shape();
    assert!(shape.len() == 4, "gaussian_blur_2d expects [B, C, H, W], got {:?}", shape);
    let channels = shape[1];
    let device = input.device();

    // Disable autograd — blur is a non-differentiable loss helper.
    // Without this, each call allocates C++ grad_fn nodes and
    // SavedVariable refs that accumulate until backward().
    let _guard = NoGradGuard::new();
    let data = input.data();

    // Build 1-D Gaussian kernel
    let radius = (3.0 * sigma).ceil() as i64;
    let ksize = 2 * radius + 1;
    let mut weights = vec![0.0f32; ksize as usize];
    let mut sum = 0.0f64;
    for i in 0..ksize {
        let x = (i - radius) as f64;
        let w = (-x * x / (2.0 * sigma * sigma)).exp();
        weights[i as usize] = w as f32;
        sum += w;
    }
    for w in &mut weights {
        *w /= sum as f32;
    }

    // Horizontal kernel: [C, 1, 1, K]
    let kernel_1d = Tensor::from_f32(&weights, &[1, 1, 1, ksize], device)?;
    let h_kernel = kernel_1d.expand(&[channels, 1, 1, ksize])?;

    // Vertical kernel: [C, 1, K, 1]
    let kernel_1d_v = Tensor::from_f32(&weights, &[1, 1, ksize, 1], device)?;
    let v_kernel = kernel_1d_v.expand(&[channels, 1, ksize, 1])?;

    // Horizontal pass (depthwise: groups = C) — Tensor-level conv2d, no Variable overhead
    let padded_h = data.pad(&[radius, radius, 0, 0], 0.0)?;
    let h_out = padded_h.conv2d(&h_kernel, None, [1, 1], [0, 0], [1, 1], channels)?;

    // Vertical pass
    let padded_v = h_out.pad(&[0, 0, radius, radius], 0.0)?;
    let result = padded_v.conv2d(&v_kernel, None, [1, 1], [0, 0], [1, 1], channels)?;

    Ok(Variable::wrap(result))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_blur_preserves_shape() {
        let opts = crate::tensor::test_opts();
        let input = Variable::new(Tensor::randn(&[1, 3, 16, 16], opts).unwrap(), false);
        let output = gaussian_blur_2d(&input, 1.0).unwrap();
        assert_eq!(output.shape(), vec![1, 3, 16, 16]);
    }

    #[test]
    fn test_gaussian_blur_smooths() {
        let opts = crate::tensor::test_opts();
        let input = Variable::new(Tensor::randn(&[1, 1, 8, 8], opts).unwrap(), false);
        let output = gaussian_blur_2d(&input, 2.0).unwrap();

        // Blurred output should have lower variance than input
        let in_var = input.var().unwrap().item().unwrap();
        let out_var = output.var().unwrap().item().unwrap();
        assert!(out_var < in_var, "blur should reduce variance: in={} out={}", in_var, out_var);
    }

    #[test]
    fn test_gaussian_blur_no_autograd_graph() {
        // Verify blur doesn't build autograd graph even with requires_grad input
        let opts = crate::tensor::test_opts();
        let input = Variable::new(Tensor::randn(&[1, 3, 8, 8], opts).unwrap(), true);
        let output = gaussian_blur_2d(&input, 1.0).unwrap();
        // Output should not require grad (NoGradGuard prevents graph construction)
        assert!(!output.requires_grad(), "blur output should not require grad");
        assert!(output.is_leaf(), "blur output should be a leaf tensor");
    }

    /// Regression test for CPU RAM leak during training.
    ///
    /// Mimics the fbrl training pattern: forward → gaussian_blur_2d → loss → backward,
    /// repeated many times. RSS should stay roughly flat.
    #[test]
    fn test_gaussian_blur_no_ram_leak_leakcheck() {
        if crate::tensor::test_device() != crate::tensor::Device::CPU { return; }
        use crate::nn::{Linear, Module, Adam, Optimizer};

        fn get_rss_kb() -> usize {
            std::fs::read_to_string("/proc/self/statm")
                .ok()
                .and_then(|s| s.split_whitespace().nth(1)?.parse::<usize>().ok())
                .map(|pages| pages * 4) // pages → KB (4K pages on Linux)
                .unwrap_or(0)
        }

        let opts = crate::tensor::test_opts();
        let model = Linear::new(128, 128).unwrap();
        let params = model.parameters();
        let mut opt = Adam::new(&params, 0.001);

        // Warm up — let allocator settle
        for _ in 0..50 {
            let x = Variable::new(Tensor::randn(&[2, 1, 8, 8], opts).unwrap(), false);
            let _ = gaussian_blur_2d(&x, 1.5).unwrap();
        }

        // Trim and measure baseline
        crate::tensor::malloc_trim();
        let rss_before = get_rss_kb();

        // Main loop: mimics training with blur in loss computation
        for _ in 0..500 {
            let x = Variable::new(Tensor::randn(&[4, 128], opts).unwrap(), false);
            let y = model.forward(&x).unwrap();
            let loss = y.sum().unwrap();
            opt.zero_grad();
            loss.backward().unwrap();
            opt.step().unwrap();

            // Simulate gaussian_blur_2d calls in loss (2 per batch, like fbrl)
            let img = Variable::new(Tensor::randn(&[2, 3, 16, 16], opts).unwrap(), false);
            let _ = gaussian_blur_2d(&img, 1.5).unwrap();
            let _ = gaussian_blur_2d(&img, 2.0).unwrap();
        }

        crate::tensor::malloc_trim();
        let rss_after = get_rss_kb();
        let growth_mb = (rss_after as f64 - rss_before as f64) / 1024.0;
        assert!(
            growth_mb < 20.0,
            "RSS grew by {growth_mb:.1}MB over 500 iters — likely a leak \
             (before={rss_before}KB, after={rss_after}KB)"
        );
    }

    #[test]
    fn test_gaussian_blur_module() {
        let opts = crate::tensor::test_opts();
        let blur = GaussianBlur::new(1.5);
        assert_eq!(blur.name(), "gaussian_blur");
        assert!(blur.parameters().is_empty());

        let input = Variable::new(Tensor::randn(&[1, 3, 8, 8], opts).unwrap(), false);
        let output = blur.forward(&input).unwrap();
        assert_eq!(output.shape(), vec![1, 3, 8, 8]);
    }
}
