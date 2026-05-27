//! Synthetic dataset generation for reproducible benchmarks.
//!
//! **Structured data**: every generator produces deterministic (input, target)
//! pairs from a fixed seed so that "converged" has a known meaning.
//! Random-noise generators are kept for backwards compatibility but should
//! not be used for convergence evaluation.
//!
//! Physical pool is kept small (a few thousand samples) while `len()` reports
//! the virtual size the DataLoader / training loop expects.  `get_batch`
//! wraps indices via modulo so the pool is silently recycled.

use std::sync::Arc;

use flodl::data::BatchDataSet;
use flodl::tensor::{Device, DType, Result, Tensor, TensorOptions};

/// Default pool multiplier: physical pool = batch_size * POOL_MUL.
/// 8x is enough to prevent GPU-cache distortion without eating RAM.
pub const POOL_MUL: usize = 8;

/// A pre-generated synthetic dataset stored as bulk tensors.
///
/// `pool_size` samples live in memory.  `virtual_len` (returned by `len()`)
/// can be larger; indices wrap via modulo in `get_batch`.
pub struct SyntheticDataSet {
    /// `tensors[group_idx]` = `[pool_size, per-sample dims...]`
    tensors: Vec<Tensor>,
    pool_size: usize,
    virtual_len: usize,
}

// ── Structured data generators ─────────────────────────────────────────

#[allow(dead_code, clippy::too_many_arguments)]
impl SyntheticDataSet {
    // ── Linear: known linear mapping ───────────────────────────────────

    /// target = input @ W^T + b, where W and b are generated from `seed`.
    ///
    /// The linear model can learn this exactly; known minimum MSE = 0.
    pub fn linear_mapping(
        seed: u64,
        virtual_len: usize,
        pool_size: usize,
        input_dim: i64,
        output_dim: i64,
    ) -> Result<Arc<dyn BatchDataSet>> {
        flodl::manual_seed(seed);
        let n = pool_size as i64;
        let opts = TensorOptions::default();

        // Teacher weights (scaled down to keep gradients reasonable)
        let w = Tensor::randn(&[output_dim, input_dim], opts)?.mul_scalar(0.01)?;
        let b = Tensor::randn(&[1, output_dim], opts)?.mul_scalar(0.01)?;

        let inputs = Tensor::randn(&[n, input_dim], opts)?;
        let targets = inputs.matmul(&w.transpose(0, 1)?)?.add(&b)?;

        Ok(Arc::new(SyntheticDataSet {
            tensors: vec![inputs, targets],
            pool_size,
            virtual_len,
        }))
    }

    // ── MLP: teacher-student ───────────────────────────────────────────

    /// 2-layer teacher MLP: target = ReLU(input @ W1^T + b1) @ W2^T + b2.
    ///
    /// Student (6-layer) has excess capacity; known minimum MSE near 0.
    pub fn teacher_mlp(
        seed: u64,
        virtual_len: usize,
        pool_size: usize,
        input_dim: i64,
        hidden_dim: i64,
        output_dim: i64,
    ) -> Result<Arc<dyn BatchDataSet>> {
        flodl::manual_seed(seed);
        let n = pool_size as i64;
        let opts = TensorOptions::default();
        let scale = (2.0 / input_dim as f64).sqrt(); // kaiming-like

        let w1 = Tensor::randn(&[hidden_dim, input_dim], opts)?.mul_scalar(scale)?;
        let b1 = Tensor::randn(&[1, hidden_dim], opts)?.mul_scalar(0.01)?;
        let w2 = Tensor::randn(&[output_dim, hidden_dim], opts)?
            .mul_scalar((2.0 / hidden_dim as f64).sqrt())?;
        let b2 = Tensor::randn(&[1, output_dim], opts)?.mul_scalar(0.01)?;

        let inputs = Tensor::randn(&[n, input_dim], opts)?;
        let h = inputs.matmul(&w1.transpose(0, 1)?)?.add(&b1)?.relu()?;
        let targets = h.matmul(&w2.transpose(0, 1)?)?.add(&b2)?;

        Ok(Arc::new(SyntheticDataSet {
            tensors: vec![inputs, targets],
            pool_size,
            virtual_len,
        }))
    }

    // ── ConvNet: prototype classification ──────────────────────────────

    /// 10 fixed prototype images; each sample = `prototype[class]` + noise.
    ///
    /// The convnet learns to match noisy inputs to prototypes.
    /// Known minimum: CE near 0 (bounded by noise overlap).
    pub fn prototype_classification(
        seed: u64,
        virtual_len: usize,
        pool_size: usize,
        channels: i64,
        height: i64,
        width: i64,
        num_classes: i64,
        noise_scale: f64,
    ) -> Result<Arc<dyn BatchDataSet>> {
        flodl::manual_seed(seed);
        let n = pool_size as i64;
        let opts = TensorOptions::default();

        // Fixed prototypes: one per class
        let prototypes =
            Tensor::randn(&[num_classes, channels, height, width], opts)?;

        // Random class assignments (Int64 for index_select and cross-entropy)
        let classes = Tensor::randint(0, num_classes, &[n], opts)?
            .to_dtype(DType::Int64)?;

        // Select prototype per sample and add noise
        let selected = prototypes.index_select(0, &classes)?;
        let noise =
            Tensor::randn(&[n, channels, height, width], opts)?.mul_scalar(noise_scale)?;
        let inputs = selected.add(&noise)?;

        Ok(Arc::new(SyntheticDataSet {
            tensors: vec![inputs, classes],
            pool_size,
            virtual_len,
        }))
    }

    // ── Autoencoder: low-rank reconstruction ──────────────────────────

    /// Images = random linear combination of K basis vectors, passed through
    /// tanh to match the autoencoder's bounded output.
    ///
    /// Known minimum: MSE = 0 (data lives on a K-dim subspace).
    pub fn low_rank_reconstruction(
        seed: u64,
        virtual_len: usize,
        pool_size: usize,
        channels: i64,
        height: i64,
        width: i64,
        num_basis: i64,
    ) -> Result<Arc<dyn BatchDataSet>> {
        flodl::manual_seed(seed);
        let n = pool_size as i64;
        let opts = TensorOptions::default();
        let pixels = channels * height * width;

        // Normalized basis vectors
        let basis = Tensor::randn(&[num_basis, pixels], opts)?;
        let basis = basis.normalize(2.0, 1)?;

        // Random coefficients, scaled so tanh doesn't saturate too hard
        let coeffs = Tensor::randn(&[n, num_basis], opts)?.mul_scalar(0.5)?;

        // Images = coeffs @ basis, reshaped, bounded by tanh
        let flat_images = coeffs.matmul(&basis)?;
        let images = flat_images.reshape(&[n, channels, height, width])?.tanh()?;

        // Target = input (reconstruction)
        let targets = images.clone();

        Ok(Arc::new(SyntheticDataSet {
            tensors: vec![images, targets],
            pool_size,
            virtual_len,
        }))
    }

    // ── LSTM: cumulative sequence ─────────────────────────────────────

    /// Target = projection of the cumulative sum at each checkpoint timestep.
    ///
    /// Requires temporal accumulation (not solvable by per-timestep ops).
    /// Known minimum: MSE near 0.
    pub fn cumulative_sequence(
        seed: u64,
        virtual_len: usize,
        pool_size: usize,
        seq_len: i64,
        input_dim: i64,
        output_dim: i64,
    ) -> Result<Arc<dyn BatchDataSet>> {
        flodl::manual_seed(seed);
        let n = pool_size as i64;
        let opts = TensorOptions::default();
        let scale = (1.0 / input_dim as f64).sqrt();

        // Fixed projection from input_dim to output_dim
        let w_proj =
            Tensor::randn(&[output_dim, input_dim], opts)?.mul_scalar(scale)?;

        let inputs = Tensor::randn(&[n, seq_len, input_dim], opts)?.mul_scalar(0.1)?;

        // Cumulative sum along time axis, take final timestep
        let cumsum = inputs.cumsum(1)?;
        let last = cumsum.select(1, seq_len - 1)?; // [n, input_dim]

        // Scale by 1/seq_len to normalize, then project
        let targets = last
            .div_scalar(seq_len as f64)?
            .matmul(&w_proj.transpose(0, 1)?)?;

        Ok(Arc::new(SyntheticDataSet {
            tensors: vec![inputs, targets],
            pool_size,
            virtual_len,
        }))
    }

    // ── Transformer: shift cipher ─────────────────────────────────────

    /// `target[i] = (input[i] + shift) % vocab_size`.
    ///
    /// Deterministic per-token mapping; known minimum CE = 0.
    pub fn shift_cipher(
        seed: u64,
        virtual_len: usize,
        pool_size: usize,
        seq_len: i64,
        vocab_size: i64,
        shift: i64,
    ) -> Result<Arc<dyn BatchDataSet>> {
        flodl::manual_seed(seed);
        let n = pool_size as i64;
        let opts = TensorOptions::default();

        let inputs = Tensor::randint(0, vocab_size, &[n, seq_len], opts)?;
        // (input + shift) % vocab, using float arithmetic then cast back
        let targets = inputs
            .to_dtype(DType::Float32)?
            .add_scalar(shift as f64)?
            .remainder(vocab_size as f64)?
            .to_dtype(DType::Int64)?;

        Ok(Arc::new(SyntheticDataSet {
            tensors: vec![inputs, targets],
            pool_size,
            virtual_len,
        }))
    }

    // ── MoE: clustered regression ─────────────────────────────────────

    /// 8 clusters with per-cluster linear mappings.
    ///
    /// Each expert should specialize on one cluster. Router must learn
    /// cluster membership. Known minimum: MSE = 0 if routing is correct.
    pub fn clustered_regression(
        seed: u64,
        virtual_len: usize,
        pool_size: usize,
        dim: i64,
        output_dim: i64,
        num_clusters: i64,
    ) -> Result<Arc<dyn BatchDataSet>> {
        flodl::manual_seed(seed);
        let n = pool_size as i64;
        let opts = TensorOptions::default();
        let scale = (1.0 / dim as f64).sqrt();

        // Cluster centers, spread out
        let centers = Tensor::randn(&[num_clusters, dim], opts)?.mul_scalar(3.0)?;

        // Per-cluster weight matrices: [num_clusters, output_dim, dim]
        let all_w = Tensor::randn(&[num_clusters, output_dim, dim], opts)?
            .mul_scalar(scale)?;

        // Assign each sample to a cluster (Int64 for index_select)
        let cluster_ids = Tensor::randint(0, num_clusters, &[n], opts)?
            .to_dtype(DType::Int64)?;

        // Inputs = center[k] + noise
        let selected_centers = centers.index_select(0, &cluster_ids)?;
        let noise = Tensor::randn(&[n, dim], opts)?.mul_scalar(0.5)?;
        let inputs = selected_centers.add(&noise)?;

        // Targets: batched per-cluster linear mapping
        // selected_w: [n, output_dim, dim]
        let selected_w = all_w.index_select(0, &cluster_ids)?;
        // inputs: [n, dim] -> [n, 1, dim] @ [n, dim, output_dim] -> [n, 1, output_dim]
        let targets = inputs
            .unsqueeze(1)?
            .matmul(&selected_w.transpose(1, 2)?)?
            .squeeze(1)?;

        Ok(Arc::new(SyntheticDataSet {
            tensors: vec![inputs, targets],
            pool_size,
            virtual_len,
        }))
    }

    // ── Residual: identity + perturbation ─────────────────────────────

    /// target = input + tanh(input @ W^T) * scale.
    ///
    /// Residual architecture is designed for this; skip connections pass
    /// identity while learned layers model the perturbation.
    /// Known minimum: MSE = 0.
    pub fn identity_perturbation(
        seed: u64,
        virtual_len: usize,
        pool_size: usize,
        dim: i64,
        output_dim: i64,
        perturbation_scale: f64,
    ) -> Result<Arc<dyn BatchDataSet>> {
        flodl::manual_seed(seed);
        let n = pool_size as i64;
        let opts = TensorOptions::default();
        let scale = (1.0 / dim as f64).sqrt();

        let w_perturb = Tensor::randn(&[dim, dim], opts)?.mul_scalar(scale)?;
        let w_out = Tensor::randn(&[output_dim, dim], opts)?.mul_scalar(scale)?;

        let inputs = Tensor::randn(&[n, dim], opts)?;

        // identity + small nonlinear perturbation
        let perturbation = inputs
            .matmul(&w_perturb.transpose(0, 1)?)?
            .tanh()?
            .mul_scalar(perturbation_scale)?;
        let combined = inputs.add(&perturbation)?;

        // Project to output dim
        let targets = combined.matmul(&w_out.transpose(0, 1)?)?;

        Ok(Arc::new(SyntheticDataSet {
            tensors: vec![inputs, targets],
            pool_size,
            virtual_len,
        }))
    }

    // ── Feedback: iterative denoising ─────────────────────────────────

    /// input = clean + noise; target = clean.
    ///
    /// Clean signals are low-rank (sum of basis vectors). The feedback loop
    /// should progressively remove noise. Known minimum: MSE = noise floor.
    pub fn denoising(
        seed: u64,
        virtual_len: usize,
        pool_size: usize,
        dim: i64,
        output_dim: i64,
        num_basis: i64,
        noise_scale: f64,
    ) -> Result<Arc<dyn BatchDataSet>> {
        flodl::manual_seed(seed);
        let n = pool_size as i64;
        let opts = TensorOptions::default();
        let scale = (1.0 / dim as f64).sqrt();

        // Basis for clean signals
        let basis = Tensor::randn(&[num_basis, dim], opts)?.normalize(2.0, 1)?;
        let w_out = Tensor::randn(&[output_dim, dim], opts)?.mul_scalar(scale)?;

        // Clean signals = random mix of basis vectors
        let coeffs = Tensor::randn(&[n, num_basis], opts)?;
        let clean = coeffs.matmul(&basis)?; // [n, dim]

        // Noisy input
        let noise = Tensor::randn(&[n, dim], opts)?.mul_scalar(noise_scale)?;
        let inputs = clean.add(&noise)?;

        // Target = clean signal projected to output_dim
        let targets = clean.matmul(&w_out.transpose(0, 1)?)?;

        Ok(Arc::new(SyntheticDataSet {
            tensors: vec![inputs, targets],
            pool_size,
            virtual_len,
        }))
    }
}

// ── Legacy random generators (kept for reference/timing benchmarks) ────

#[allow(dead_code)]
impl SyntheticDataSet {
    /// Random regression: input and target are independent noise.
    pub fn regression(
        seed: u64,
        virtual_len: usize,
        pool_size: usize,
        input_dim: i64,
        output_dim: i64,
    ) -> Result<Arc<dyn BatchDataSet>> {
        flodl::manual_seed(seed);
        let n = pool_size as i64;
        let opts = TensorOptions::default();

        let inputs = Tensor::randn(&[n, input_dim], opts)?;
        let targets = Tensor::randn(&[n, output_dim], opts)?;

        Ok(Arc::new(SyntheticDataSet {
            tensors: vec![inputs, targets],
            pool_size,
            virtual_len,
        }))
    }

    /// Random classification: input is noise, labels are uniform random.
    pub fn classification(
        seed: u64,
        virtual_len: usize,
        pool_size: usize,
        input_shape: &[i64],
        num_classes: i64,
    ) -> Result<Arc<dyn BatchDataSet>> {
        flodl::manual_seed(seed);
        let n = pool_size as i64;
        let opts = TensorOptions::default();

        let mut shape = vec![n];
        shape.extend_from_slice(input_shape);
        let inputs = Tensor::randn(&shape, opts)?;
        let targets = Tensor::randint(0, num_classes, &[n], opts)?;

        Ok(Arc::new(SyntheticDataSet {
            tensors: vec![inputs, targets],
            pool_size,
            virtual_len,
        }))
    }

    /// Random reconstruction: input is noise, target = input.
    pub fn reconstruction(
        seed: u64,
        virtual_len: usize,
        pool_size: usize,
        shape: &[i64],
    ) -> Result<Arc<dyn BatchDataSet>> {
        flodl::manual_seed(seed);
        let n = pool_size as i64;
        let opts = TensorOptions::default();

        let mut full_shape = vec![n];
        full_shape.extend_from_slice(shape);
        let inputs = Tensor::randn(&full_shape, opts)?;
        let targets = inputs.clone();

        Ok(Arc::new(SyntheticDataSet {
            tensors: vec![inputs, targets],
            pool_size,
            virtual_len,
        }))
    }

    /// Random sequence: input and target are independent noise.
    pub fn sequence(
        seed: u64,
        virtual_len: usize,
        pool_size: usize,
        seq_len: i64,
        input_dim: i64,
        output_dim: i64,
    ) -> Result<Arc<dyn BatchDataSet>> {
        flodl::manual_seed(seed);
        let n = pool_size as i64;
        let opts = TensorOptions::default();

        let inputs = Tensor::randn(&[n, seq_len, input_dim], opts)?;
        let targets = Tensor::randn(&[n, output_dim], opts)?;

        Ok(Arc::new(SyntheticDataSet {
            tensors: vec![inputs, targets],
            pool_size,
            virtual_len,
        }))
    }

    /// Random token sequence: input and target are independent random tokens.
    pub fn token_sequence(
        seed: u64,
        virtual_len: usize,
        pool_size: usize,
        seq_len: i64,
        vocab_size: i64,
    ) -> Result<Arc<dyn BatchDataSet>> {
        flodl::manual_seed(seed);
        let n = pool_size as i64;
        let opts = TensorOptions::default();

        let inputs = Tensor::randint(0, vocab_size, &[n, seq_len], opts)?;
        let targets = Tensor::randint(0, vocab_size, &[n, seq_len], opts)?;

        Ok(Arc::new(SyntheticDataSet {
            tensors: vec![inputs, targets],
            pool_size,
            virtual_len,
        }))
    }
}

impl BatchDataSet for SyntheticDataSet {
    fn len(&self) -> usize {
        self.virtual_len
    }

    fn get_batch(&self, indices: &[usize]) -> Result<Vec<Tensor>> {
        // Wrap indices into the physical pool via modulo.
        let idx: Vec<i64> = indices
            .iter()
            .map(|&i| (i % self.pool_size) as i64)
            .collect();
        let idx_tensor = Tensor::from_i64(&idx, &[idx.len() as i64], Device::CPU)?;

        let mut result = Vec::with_capacity(self.tensors.len());
        for bulk in &self.tensors {
            result.push(bulk.index_select(0, &idx_tensor)?);
        }
        Ok(result)
    }
}
