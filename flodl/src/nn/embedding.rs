use std::sync::OnceLock;

use crate::autograd::{self, Variable};
use crate::tensor::{Result, Tensor, TensorOptions, DType, Device};

use super::parameter::Parameter;
use super::Module;

/// One-shot flag: first f32-indices call in the process emits a deprecation
/// warning to stderr. The f32 fallback path in [`Embedding::forward`] will
/// be removed in a future release.
static F32_INDEX_DEPRECATION_WARNED: OnceLock<()> = OnceLock::new();

/// Lookup table for token embeddings.
///
/// Weight shape: `[num_embeddings, embedding_dim]`. Input: i64 token indices.
/// Output: embedded vectors of shape `[*input.shape, embedding_dim]`.
///
/// # Input dtype
///
/// Indices must be i64. An f32 input is accepted as a deprecated fallback
/// that converts via `to_f32_vec` + `from_i64`; this path is slow, drops
/// precision for vocabularies larger than 2^24, and will be removed in a
/// future release. The first call on that path emits a one-shot warning to
/// stderr. Use [`Tensor::from_i64`] directly.
///
/// # Padding
///
/// Use [`Embedding::with_padding_idx`] to designate a row whose gradient is
/// masked to zero during backward (so the PAD-token embedding does not drift
/// during fine-tuning). The forward pass still returns that row normally;
/// downstream code is expected to mask PAD positions via attention masks.
///
/// Notes for LLaMA-style checkpoints: when `pad_token_id == eos_token_id`,
/// pass `padding_idx = None` — otherwise the EOS row would be frozen.
///
/// ```ignore
/// let emb = Embedding::new(1000, 64)?;
/// // Input: [seq_len] of token indices → Output: [seq_len, 64]
/// let indices = Variable::new(Tensor::from_i64(&[0, 5, 42], &[3], Device::CPU)?, false);
/// let vectors = emb.forward(&indices)?;
/// assert_eq!(vectors.shape(), vec![3, 64]);
/// ```
pub struct Embedding {
    pub weight: Parameter,
    padding_idx: i64,
}

impl Embedding {
    /// Sentinel value for `padding_idx` meaning "no padding index set".
    /// Matches libtorch `at::embedding` convention.
    pub const NO_PADDING: i64 = -1;

    /// Create an embedding table on CPU without a padding index.
    pub fn new(num_embeddings: i64, embedding_dim: i64) -> Result<Self> {
        Self::on_device_with_padding_idx(num_embeddings, embedding_dim, None, Device::CPU)
    }

    /// Create an embedding table on a specific device without a padding index.
    pub fn on_device(num_embeddings: i64, embedding_dim: i64, device: Device) -> Result<Self> {
        Self::on_device_with_padding_idx(num_embeddings, embedding_dim, None, device)
    }

    /// Create an embedding table on CPU with an optional `padding_idx`.
    ///
    /// When `padding_idx` is `Some(i)`, the gradient of row `i` is masked to
    /// zero during backward — matching PyTorch `nn.Embedding(..., padding_idx=i)`.
    pub fn with_padding_idx(
        num_embeddings: i64, embedding_dim: i64, padding_idx: Option<i64>,
    ) -> Result<Self> {
        Self::on_device_with_padding_idx(num_embeddings, embedding_dim, padding_idx, Device::CPU)
    }

    /// Create an embedding table on a specific device with an optional
    /// `padding_idx`.
    pub fn on_device_with_padding_idx(
        num_embeddings: i64, embedding_dim: i64, padding_idx: Option<i64>, device: Device,
    ) -> Result<Self> {
        if let Some(p) = padding_idx
            && (p < 0 || p >= num_embeddings) {
                return Err(crate::tensor::TensorError::new(&format!(
                    "padding_idx {p} out of range [0, {num_embeddings})"
                )));
            }

        let weight = Variable::new(
            Tensor::randn(
                &[num_embeddings, embedding_dim],
                TensorOptions { dtype: DType::Float32, device },
            )?,
            true,
        );

        Ok(Embedding {
            weight: Parameter {
                variable: weight,
                name: "weight".into(),
            },
            padding_idx: padding_idx.unwrap_or(Self::NO_PADDING),
        })
    }
}

impl Module for Embedding {
    fn name(&self) -> &str { "embedding" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        // at::embedding accepts any-shape i64 indices and returns
        // [*indices.shape, embedding_dim] directly — no manual reshape.
        let index_tensor = if input.data().dtype() == DType::Int64 {
            input.data()
        } else {
            // Deprecated: f32 indices accepted as a legacy fallback, will be
            // removed in a future release. Emit a one-shot stderr warning the
            // first time this path is taken in the process.
            F32_INDEX_DEPRECATION_WARNED.get_or_init(|| {
                eprintln!(
                    "[flodl] deprecated: Embedding::forward received non-i64 \
                     indices; this fallback will be removed in a future \
                     release. Pass i64 tensors via Tensor::from_i64."
                );
            });
            let input_shape = input.shape();
            let flat_data = input.data().to_f32_vec()?;
            let indices: Vec<i64> = flat_data.iter().map(|&v| v as i64).collect();
            Tensor::from_i64(&indices, &input_shape, input.device())?
        };

        autograd::embedding(&self.weight.variable, &index_tensor, self.padding_idx)
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![self.weight.clone()]
    }
}

/// Fused embedding lookup + reduction (sum / mean / max).
///
/// Each "bag" is a variable-length group of indices whose embeddings are
/// reduced to a single vector. This is significantly faster than a manual
/// embedding lookup followed by a separate reduction, because libtorch fuses
/// the two into one kernel.
///
/// Reduction modes:
/// - `EmbeddingBag::SUM`  (0) — sum embeddings in each bag
/// - `EmbeddingBag::MEAN` (1) — average embeddings in each bag
/// - `EmbeddingBag::MAX`  (2) — element-wise max across each bag
///
/// # Uniform bags via `forward()`
///
/// When all bags have the same size, pass a 2-D `[num_bags, bag_size]` index
/// tensor and let the module build offsets automatically:
///
/// ```ignore
/// let eb = EmbeddingBag::new(1000, 64, EmbeddingBag::SUM)?;
/// let indices = Variable::new(Tensor::from_i64(&[0,1,2, 3,4,5], &[2, 3], Device::CPU)?, false);
/// let out = eb.forward(&indices)?;          // [2, 64]
/// ```
///
/// # Variable-length bags via `forward_bag()`
///
/// For bags of different sizes, provide flat indices and explicit offsets:
///
/// ```ignore
/// let indices = Tensor::from_i64(&[0,1,2, 3,4], &[5], Device::CPU)?;
/// let offsets = Tensor::from_i64(&[0, 3],        &[2], Device::CPU)?;
/// let out = eb.forward_bag(&indices, &offsets)?; // [2, 64]
/// ```
pub struct EmbeddingBag {
    pub weight: Parameter,
    #[allow(dead_code)]
    num_embeddings: i64,
    #[allow(dead_code)]
    embedding_dim: i64,
    mode: i64,
}

impl EmbeddingBag {
    /// Sum reduction mode.
    pub const SUM: i64 = 0;
    /// Mean reduction mode.
    pub const MEAN: i64 = 1;
    /// Element-wise max reduction mode.
    pub const MAX: i64 = 2;

    /// Create an embedding bag on CPU.
    pub fn new(num_embeddings: i64, embedding_dim: i64, mode: i64) -> Result<Self> {
        Self::on_device(num_embeddings, embedding_dim, mode, Device::CPU)
    }

    /// Create an embedding bag on a specific device.
    pub fn on_device(
        num_embeddings: i64, embedding_dim: i64, mode: i64, device: Device,
    ) -> Result<Self> {
        let weight = Variable::new(
            Tensor::randn(
                &[num_embeddings, embedding_dim],
                TensorOptions { dtype: DType::Float32, device },
            )?,
            true,
        );

        Ok(EmbeddingBag {
            weight: Parameter {
                variable: weight,
                name: "weight".into(),
            },
            num_embeddings,
            embedding_dim,
            mode,
        })
    }

    /// Variable-length bag forward: flat `indices` + explicit `offsets`.
    ///
    /// `indices`: 1-D i64 tensor of token indices.
    /// `offsets`: 1-D i64 tensor of length `num_bags`, marking the start of
    /// each bag within `indices`.
    pub fn forward_bag(&self, indices: &Tensor, offsets: &Tensor) -> Result<Variable> {
        autograd::embedding_bag(&self.weight.variable, indices, offsets, self.mode)
    }
}

impl Module for EmbeddingBag {
    fn name(&self) -> &str { "embedding_bag" }

    /// Uniform-bag forward: input is 2-D `[num_bags, bag_size]`.
    ///
    /// Offsets are computed automatically as `[0, bag_size, 2*bag_size, ...]`.
    fn forward(&self, input: &Variable) -> Result<Variable> {
        let shape = input.shape();
        if shape.len() != 2 {
            return Err(crate::tensor::TensorError::new(&format!(
                "EmbeddingBag::forward expects 2-D input [num_bags, bag_size], got {:?}",
                shape,
            )));
        }
        let num_bags = shape[0];
        let bag_size = shape[1];
        let device = input.device();

        // Build flat i64 indices from the 2-D input
        let flat_indices = if input.data().dtype() == DType::Int64 {
            input.data().reshape(&[num_bags * bag_size])?
        } else {
            let flat_data = input.data().to_f32_vec()?;
            let idx: Vec<i64> = flat_data.iter().map(|&v| v as i64).collect();
            Tensor::from_i64(&idx, &[num_bags * bag_size], device)?
        };

        // Build offsets: [0, bag_size, 2*bag_size, ...]
        let offsets_vec: Vec<i64> = (0..num_bags).map(|i| i * bag_size).collect();
        let offsets = Tensor::from_i64(&offsets_vec, &[num_bags], device)?;

        autograd::embedding_bag(&self.weight.variable, &flat_indices, &offsets, self.mode)
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![self.weight.clone()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::test_device;

    /// Hand-computed sum: bag0 = w[0]+w[1]+w[2], bag1 = w[3]+w[4].
    #[test]
    #[allow(clippy::identity_op, clippy::erasing_op)]
    fn embedding_bag_sum_known_values() {
        let dev = test_device();
        let eb = EmbeddingBag::on_device(5, 3, EmbeddingBag::SUM, dev).unwrap();
        let w = eb.weight.variable.data().to_f32_vec().unwrap();
        // w is [5, 3] flattened — row*stride indexing kept for clarity

        let indices = Tensor::from_i64(&[0, 1, 2, 3, 4], &[5], dev).unwrap();
        let offsets = Tensor::from_i64(&[0, 3], &[2], dev).unwrap();
        let out = eb.forward_bag(&indices, &offsets).unwrap();

        assert_eq!(out.shape(), vec![2, 3]);
        let vals = out.data().to_f32_vec().unwrap();

        // bag0 = w[0]+w[1]+w[2] for each of 3 dims
        for d in 0..3 {
            let expected = w[0 * 3 + d] + w[1 * 3 + d] + w[2 * 3 + d];
            assert!((vals[0 * 3 + d] - expected).abs() < 1e-5,
                "bag0 dim {d}: got {}, expected {}", vals[0 * 3 + d], expected);
        }
        // bag1 = w[3]+w[4]
        for d in 0..3 {
            let expected = w[3 * 3 + d] + w[4 * 3 + d];
            assert!((vals[1 * 3 + d] - expected).abs() < 1e-5,
                "bag1 dim {d}: got {}, expected {}", vals[1 * 3 + d], expected);
        }
    }

    /// Mean mode: bag0 = mean(w[0], w[1]), bag1 = mean(w[2], w[3]).
    #[test]
    #[allow(clippy::identity_op, clippy::erasing_op)]
    fn embedding_bag_mean() {
        let dev = test_device();
        let eb = EmbeddingBag::on_device(4, 2, EmbeddingBag::MEAN, dev).unwrap();
        let w = eb.weight.variable.data().to_f32_vec().unwrap();

        let indices = Tensor::from_i64(&[0, 1, 2, 3], &[4], dev).unwrap();
        let offsets = Tensor::from_i64(&[0, 2], &[2], dev).unwrap();
        let out = eb.forward_bag(&indices, &offsets).unwrap();

        assert_eq!(out.shape(), vec![2, 2]);
        let vals = out.data().to_f32_vec().unwrap();

        for d in 0..2 {
            let expected = (w[0 * 2 + d] + w[1 * 2 + d]) / 2.0;
            assert!((vals[0 * 2 + d] - expected).abs() < 1e-5);
        }
        for d in 0..2 {
            let expected = (w[2 * 2 + d] + w[3 * 2 + d]) / 2.0;
            assert!((vals[1 * 2 + d] - expected).abs() < 1e-5);
        }
    }

    /// 2-D forward (uniform bags) produces correct shape and matches forward_bag.
    #[test]
    fn embedding_bag_2d_forward() {
        let dev = test_device();
        let eb = EmbeddingBag::on_device(10, 4, EmbeddingBag::SUM, dev).unwrap();

        // 3 bags of size 2
        let input = Variable::new(
            Tensor::from_i64(&[0, 1, 2, 3, 4, 5], &[3, 2], dev).unwrap(),
            false,
        );
        let out = eb.forward(&input).unwrap();
        assert_eq!(out.shape(), vec![3, 4]);

        // Compare with explicit forward_bag
        let flat_idx = Tensor::from_i64(&[0, 1, 2, 3, 4, 5], &[6], dev).unwrap();
        let offsets = Tensor::from_i64(&[0, 2, 4], &[3], dev).unwrap();
        let out_bag = eb.forward_bag(&flat_idx, &offsets).unwrap();

        let v1 = out.data().to_f32_vec().unwrap();
        let v2 = out_bag.data().to_f32_vec().unwrap();
        for (a, b) in v1.iter().zip(v2.iter()) {
            assert!((a - b).abs() < 1e-6, "forward vs forward_bag mismatch: {a} != {b}");
        }
    }

    /// Gradient flows through to the weight parameter.
    #[test]
    fn embedding_bag_gradient() {
        let dev = test_device();
        let eb = EmbeddingBag::on_device(5, 3, EmbeddingBag::SUM, dev).unwrap();

        let indices = Tensor::from_i64(&[0, 1, 2, 3], &[4], dev).unwrap();
        let offsets = Tensor::from_i64(&[0, 2], &[2], dev).unwrap();

        let out = eb.forward_bag(&indices, &offsets).unwrap();
        let loss = out.sum().unwrap();
        loss.backward().unwrap();

        let grad = eb.weight.variable.grad();
        assert!(grad.is_some(), "weight should have gradient after backward");
        let g = grad.unwrap();
        assert_eq!(g.shape(), vec![5, 3]);

        // Indices 0-3 were used, so their grad rows should be nonzero;
        // index 4 was not used, so its row should be zero.
        let gv = g.to_f32_vec().unwrap();
        let row4_sum: f32 = gv[4 * 3..5 * 3].iter().sum();
        assert_eq!(row4_sum, 0.0, "unused index should have zero gradient");
    }

    /// Max mode returns the element-wise maximum across embeddings in each bag.
    #[test]
    fn embedding_bag_max() {
        let dev = test_device();
        let eb = EmbeddingBag::on_device(4, 2, EmbeddingBag::MAX, dev).unwrap();
        let w = eb.weight.variable.data().to_f32_vec().unwrap();

        // Single bag of all 4 embeddings
        let indices = Tensor::from_i64(&[0, 1, 2, 3], &[4], dev).unwrap();
        let offsets = Tensor::from_i64(&[0], &[1], dev).unwrap();
        let out = eb.forward_bag(&indices, &offsets).unwrap();

        assert_eq!(out.shape(), vec![1, 2]);
        let vals = out.data().to_f32_vec().unwrap();

        for d in 0..2 {
            let expected = (0..4)
                .map(|i| w[i * 2 + d])
                .fold(f32::NEG_INFINITY, f32::max);
            assert!((vals[d] - expected).abs() < 1e-5,
                "max dim {d}: got {}, expected {}", vals[d], expected);
        }
    }

    /// Plain Embedding: forward shape and gradient flow (no padding_idx).
    #[test]
    fn embedding_forward_and_gradient() {
        let dev = test_device();
        let emb = Embedding::on_device(10, 4, dev).unwrap();

        let input = Variable::new(
            Tensor::from_i64(&[0, 3, 7, 2], &[2, 2], dev).unwrap(),
            false,
        );
        let out = emb.forward(&input).unwrap();
        assert_eq!(out.shape(), vec![2, 2, 4]);

        let loss = out.sum().unwrap();
        loss.backward().unwrap();
        let grad = emb.weight.variable.grad().expect("weight grad missing");
        assert_eq!(grad.shape(), vec![10, 4]);

        // Indices {0, 2, 3, 7} used → their rows should be nonzero;
        // other rows should be zero.
        let gv = grad.to_f32_vec().unwrap();
        let used: std::collections::HashSet<usize> = [0, 2, 3, 7].into_iter().collect();
        for row in 0..10 {
            let row_sum: f32 = gv[row * 4..(row + 1) * 4].iter().sum();
            if used.contains(&row) {
                assert!(row_sum.abs() > 0.0, "row {row} should have nonzero grad");
            } else {
                assert_eq!(row_sum, 0.0, "unused row {row} should have zero grad");
            }
        }
    }

    /// padding_idx masks the corresponding row's gradient to zero during
    /// backward, even when the pad index IS used in the forward input.
    /// This is the gradient-correctness guarantee that matters for fine-tuning.
    #[test]
    fn embedding_padding_idx_masks_gradient() {
        let dev = test_device();
        // padding_idx = 0 — PAD row should never be updated.
        let emb = Embedding::on_device_with_padding_idx(5, 3, Some(0), dev).unwrap();

        // Use index 0 (PAD) in the forward input — its row still appears in
        // the forward output, but its gradient row must be zero.
        let input = Variable::new(
            Tensor::from_i64(&[0, 0, 1, 2], &[4], dev).unwrap(),
            false,
        );
        let out = emb.forward(&input).unwrap();
        assert_eq!(out.shape(), vec![4, 3]);

        let loss = out.sum().unwrap();
        loss.backward().unwrap();
        let grad = emb.weight.variable.grad().unwrap();
        let gv = grad.to_f32_vec().unwrap();

        // Row 0 (PAD) must be entirely zero.
        let row0_sum: f32 = gv[0..3].iter().map(|v| v.abs()).sum();
        assert_eq!(row0_sum, 0.0, "padding_idx row should have zero gradient");
        // Rows 1 and 2 were used — nonzero gradient expected.
        let row1_sum: f32 = gv[3..6].iter().map(|v| v.abs()).sum();
        let row2_sum: f32 = gv[6..9].iter().map(|v| v.abs()).sum();
        assert!(row1_sum > 0.0, "row 1 grad should be nonzero");
        assert!(row2_sum > 0.0, "row 2 grad should be nonzero");
    }

    /// padding_idx = None matches no-padding-idx constructor behaviour.
    #[test]
    fn embedding_with_padding_idx_none_equivalent() {
        let dev = test_device();
        let emb = Embedding::on_device_with_padding_idx(8, 4, None, dev).unwrap();
        let input = Variable::new(
            Tensor::from_i64(&[0, 1, 2, 3], &[4], dev).unwrap(),
            false,
        );
        let out = emb.forward(&input).unwrap();
        assert_eq!(out.shape(), vec![4, 4]);
        let loss = out.sum().unwrap();
        loss.backward().unwrap();
        // All four used rows (incl. row 0) should have nonzero grad because
        // padding is disabled.
        let grad = emb.weight.variable.grad().unwrap();
        let gv = grad.to_f32_vec().unwrap();
        for row in 0..4 {
            let row_sum: f32 = gv[row * 4..(row + 1) * 4].iter().map(|v| v.abs()).sum();
            assert!(row_sum > 0.0, "row {row} should have nonzero grad when padding disabled");
        }
    }

    /// Default constructor (no explicit padding_idx) leaves row 0 trainable:
    /// when index 0 appears in the forward input, row 0 still gets a nonzero
    /// gradient. Anchors the "padding must be opt-in" contract so a future
    /// regression flipping the default is caught immediately.
    #[test]
    fn embedding_default_has_no_padding() {
        let dev = test_device();
        let emb = Embedding::on_device(4, 3, dev).unwrap();

        let input = Variable::new(
            Tensor::from_i64(&[0, 1], &[2], dev).unwrap(),
            false,
        );
        emb.forward(&input).unwrap().sum().unwrap().backward().unwrap();
        let grad = emb.weight.variable.grad().unwrap();
        let gv = grad.to_f32_vec().unwrap();
        let row0_sum: f32 = gv[0..3].iter().map(|v| v.abs()).sum();
        assert!(row0_sum > 0.0,
            "default constructor must NOT mask row 0 gradient, got {row0_sum}");
    }

    /// Deprecated f32-index fallback must keep working until it is removed.
    /// This test pins the runtime behavior so the fallback doesn't silently
    /// break between now and its scheduled removal. Emits a one-shot stderr
    /// warning (captured by the test harness).
    #[test]
    fn embedding_f32_indices_deprecated_fallback_works() {
        let dev = test_device();
        let emb = Embedding::on_device(5, 3, dev).unwrap();
        let input = Variable::new(
            Tensor::from_f32(&[0.0, 2.0, 4.0], &[3], dev).unwrap(),
            false,
        );
        let out = emb.forward(&input).unwrap();
        assert_eq!(out.shape(), vec![3, 3]);
    }

    /// Out-of-range padding_idx is rejected at construction.
    #[test]
    fn embedding_padding_idx_out_of_range_errors() {
        let dev = test_device();
        let r = Embedding::on_device_with_padding_idx(5, 3, Some(5), dev);
        assert!(r.is_err(), "padding_idx == num_embeddings must error");
        let r = Embedding::on_device_with_padding_idx(5, 3, Some(-1), dev);
        assert!(r.is_err(), "negative padding_idx must error");
    }

    /// EmbeddingBag exposes a single parameter.
    #[test]
    fn embedding_bag_parameters() {
        let dev = test_device();
        let eb = EmbeddingBag::on_device(10, 8, EmbeddingBag::MEAN, dev).unwrap();
        let params = eb.parameters();
        assert_eq!(params.len(), 1);
        assert_eq!(params[0].name, "weight");
        assert_eq!(params[0].variable.shape(), vec![10, 8]);
    }
}
