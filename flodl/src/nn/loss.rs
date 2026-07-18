use crate::autograd::Variable;
use crate::tensor::{Result, Tensor, TensorError};

/// Mean Squared Error loss: mean((pred - target)²)
///
/// Both `pred` and `target` must be Float (Float32 or Float64), same dtype.
///
/// Uses a single fused libtorch kernel (1 autograd node).
pub fn mse_loss(pred: &Variable, target: &Variable) -> Result<Variable> {
    let result = pred.data().mse_loss(&target.data(), 1)?; // 1 = Mean
    Ok(Variable::wrap(result))
}

/// Cross-entropy loss with numerically stable log-softmax.
///
/// `pred` shape: `[batch, classes]` (raw logits).
///
/// `target` accepts two formats (auto-detected by libtorch):
/// - **Class indices** `[batch]` (Int64) — like PyTorch's `F.cross_entropy`.
///   Each value is the correct class index (0..classes-1).
/// - **One-hot / soft labels** `[batch, classes]` (Float) — probability vectors.
///
/// Uses a single fused libtorch kernel (1 autograd node).
///
/// ```ignore
/// // With class indices (no one-hot allocation needed):
/// let labels = Variable::new(Tensor::from_i64(&[0, 2, 1], &[3], Device::CPU)?, false);
/// let loss = cross_entropy_loss(&logits, &labels)?;
///
/// // With one-hot targets (same as before):
/// let onehot = Variable::new(Tensor::from_f32(&[1.,0.,0., 0.,0.,1., 0.,1.,0.], &[3, 3], Device::CPU)?, false);
/// let loss = cross_entropy_loss(&logits, &onehot)?;
/// ```
pub fn cross_entropy_loss(pred: &Variable, target: &Variable) -> Result<Variable> {
    let pred_shape = pred.shape();
    if pred_shape.len() != 2 {
        return Err(TensorError::new("cross_entropy_loss: pred must be 2D [batch, classes]"));
    }
    let result = pred.data().cross_entropy_loss(
        &target.data(), 1, -100, 0.0, // Mean, ignore_index=-100, no smoothing
    )?;
    Ok(Variable::wrap(result))
}

/// Binary cross-entropy loss from probabilities (NOT logits).
///
/// `pred`: probabilities in \[0, 1\] (e.g. after sigmoid). Any shape.
/// `target`: binary labels (same shape, values 0 or 1).
///
/// Both must be Float (same dtype). For raw logits, prefer
/// `bce_with_logits_loss` which is numerically stable.
///
/// Uses a single fused libtorch kernel (1 autograd node).
pub fn bce_loss(pred: &Variable, target: &Variable) -> Result<Variable> {
    let result = pred.data().bce_loss(&target.data(), 1)?; // 1 = Mean
    Ok(Variable::wrap(result))
}

/// Binary cross-entropy loss from raw logits (numerically stable).
///
/// `pred`: raw logits (any shape).
/// `target`: binary labels (same shape, values 0 or 1).
///
/// Both must be Float (same dtype).
///
/// Uses a single fused libtorch kernel (1 autograd node).
pub fn bce_with_logits_loss(pred: &Variable, target: &Variable) -> Result<Variable> {
    let result = pred.data().bce_with_logits_loss(&target.data(), 1)?; // 1 = Mean
    Ok(Variable::wrap(result))
}

/// L1 (Mean Absolute Error) loss: mean(|pred - target|)
///
/// Both `pred` and `target` must be Float (Float32 or Float64), same dtype.
///
/// Uses a single fused libtorch kernel (1 autograd node).
pub fn l1_loss(pred: &Variable, target: &Variable) -> Result<Variable> {
    let result = pred.data().l1_loss(&target.data(), 1)?; // 1 = Mean
    Ok(Variable::wrap(result))
}

/// Smooth L1 (Huber) loss with configurable beta.
///
/// For |x| < beta: 0.5 * x² / beta
/// For |x| >= beta: |x| - 0.5 * beta
///
/// Both `pred` and `target` must be Float (Float32 or Float64), same dtype.
///
/// Uses a single fused libtorch kernel (1 autograd node).
pub fn smooth_l1_loss(pred: &Variable, target: &Variable, beta: f64) -> Result<Variable> {
    if beta <= 0.0 {
        return Err(TensorError::new("smooth_l1_loss: beta must be positive"));
    }
    let result = pred.data().smooth_l1_loss(&target.data(), 1, beta)?; // 1 = Mean
    Ok(Variable::wrap(result))
}

/// KL Divergence loss (batchmean reduction).
///
/// `input`: log-probabilities (output of log_softmax).
/// `target`: probabilities (true distribution).
///
/// Computes: `sum(target * (log(target) - input)) / batch`
///
/// Both `input` and `target` must be Float (Float32 or Float64), same dtype.
///
/// Uses a single fused libtorch kernel (1 autograd node).
/// Follows PyTorch convention: target * (log(target) - input).
pub fn kl_div_loss(input: &Variable, target: &Variable) -> Result<Variable> {
    let shape = input.shape();
    if shape.is_empty() {
        return Err(TensorError::new("kl_div_loss: input must be at least 1D"));
    }
    let batch = shape[0] as f64;
    // Use sum reduction then divide by batch (batchmean).
    // PyTorch's "batchmean" is a Python wrapper, not in the C++ kernel.
    let sum = input.data().kl_div_loss(&target.data(), 2, false)?; // 2 = Sum
    let result = sum.mul_scalar(1.0 / batch)?;
    Ok(Variable::wrap(result))
}

/// Negative log likelihood loss.
///
/// `input`: log-probabilities `[N, C]` (output of log_softmax).
/// `target`: class indices `[N]` (Int64).
///
/// Uses a single fused libtorch kernel (1 autograd node).
pub fn nll_loss(input: &Variable, target: &Variable) -> Result<Variable> {
    let result = input.data().nll_loss(&target.data(), 1, -100)?; // Mean, ignore=-100
    Ok(Variable::wrap(result))
}

/// CTC loss for sequence-to-sequence tasks (speech recognition, OCR).
///
/// `log_probs`: `[T, N, C]` — log-probabilities per timestep.
/// `targets`: `[N, S]` or concatenated 1D target sequences (Int64).
/// `input_lengths`: `[N]` (Int64) -- actual sequence lengths in log_probs.
/// `target_lengths`: `[N]` (Int64) -- actual target lengths.
/// `blank`: label index for the blank token (default: 0).
///
/// Dtype: `log_probs` must be Float. `targets`, `input_lengths`, and
/// `target_lengths` must all be Int64.
pub fn ctc_loss(
    log_probs: &Variable,
    targets: &Variable,
    input_lengths: &Variable,
    target_lengths: &Variable,
    blank: i64,
) -> Result<Variable> {
    let result = log_probs.data().ctc_loss(
        &targets.data(), &input_lengths.data(), &target_lengths.data(),
        blank, 1, // Mean
    )?;
    Ok(Variable::wrap(result))
}

/// Focal loss for imbalanced classification (object detection).
///
/// Focal loss: `-alpha * (1 - p)^gamma * log(p)` for the correct class.
/// Reduces to cross-entropy when gamma=0.
///
/// `pred`: raw logits `[N, C]`.
/// `target`: class indices `[N]` (Int64).
/// `alpha`: weighting factor (default: 0.25).
/// `gamma`: focusing parameter (default: 2.0).
pub fn focal_loss(
    pred: &Variable,
    target: &Variable,
    alpha: f64,
    gamma: f64,
) -> Result<Variable> {
    // Focal loss via cross-entropy: compute -log(p_t), then weight by (1-p_t)^gamma
    let log_p = pred.log_softmax(-1)?;
    let p = log_p.exp()?;
    // Gather the probabilities for the target class
    let target_expanded = target.unsqueeze(-1)?;
    let idx = target_expanded.data();
    let log_pt = log_p.gather(1, &idx)?.squeeze(-1)?;
    let pt = p.gather(1, &idx)?.squeeze(-1)?;
    // focal weight: alpha * (1 - pt)^gamma
    let one_minus_pt = pt.neg()?.add_scalar(1.0)?;
    let focal_weight = one_minus_pt.pow_scalar(gamma)?.mul_scalar(alpha)?;
    // loss = -focal_weight * log_pt
    let loss = focal_weight.mul(&log_pt)?.neg()?;
    loss.mean()
}

/// Triplet margin loss for metric learning.
///
/// `loss = max(0, ||anchor - positive|| - ||anchor - negative|| + margin)`
///
/// `anchor`, `positive`, `negative`: same shape embeddings (Float, same dtype).
pub fn triplet_margin_loss(
    anchor: &Variable,
    positive: &Variable,
    negative: &Variable,
    margin: f64,
) -> Result<Variable> {
    let d_pos = anchor.sub(positive)?.pow_scalar(2.0)?.sum_dim(1, false)?.sqrt()?;
    let d_neg = anchor.sub(negative)?.pow_scalar(2.0)?.sum_dim(1, false)?.sqrt()?;
    let diff = d_pos.sub(&d_neg)?.add_scalar(margin)?;
    // Match the input dtype (zeros_like inherits dtype+device+shape) so a
    // Float64 loss stays Float64 rather than relying on implicit promotion
    // of a hardcoded Float32 operand.
    let zero = Variable::wrap(Tensor::zeros_like(&diff.data())?);
    diff.maximum(&zero)?.mean()
}

/// Cosine embedding loss for learning embeddings.
///
/// `loss = 1 - cos(x1, x2)` when label=1; `max(0, cos(x1, x2) - margin)` when label=-1.
///
/// `x1`, `x2`: embeddings (same shape, Float). `label`: +1 or -1 per sample (Float).
pub fn cosine_embedding_loss(
    x1: &Variable,
    x2: &Variable,
    label: &Variable,
    margin: f64,
) -> Result<Variable> {
    let cos = x1.cosine_similarity(x2, 1, 1e-8)?;
    // For label=1: 1 - cos; for label=-1: max(0, cos - margin)
    let ones = Variable::wrap(Tensor::ones_like(&cos.data())?);
    let loss_pos = ones.sub(&cos)?;
    let cos_minus_margin = cos.add_scalar(-margin)?;
    let zero = Variable::wrap(Tensor::zeros_like(&cos.data())?);
    let loss_neg = cos_minus_margin.maximum(&zero)?;
    // label is +1 or -1: use (1+label)/2 as mask for positive, (1-label)/2 for negative
    let label_pos = label.add_scalar(1.0)?.mul_scalar(0.5)?;
    let label_neg = label.neg()?.add_scalar(1.0)?.mul_scalar(0.5)?;
    let loss = label_pos.mul(&loss_pos)?.add(&label_neg.mul(&loss_neg)?)?;
    loss.mean()
}

/// Hinge embedding loss.
///
/// `loss = x` when label=1; `max(0, margin - x)` when label=-1.
///
/// Both `input` and `label` must be Float (same dtype).
pub fn hinge_embedding_loss(
    input: &Variable,
    label: &Variable,
    margin: f64,
) -> Result<Variable> {
    let margin_minus_x = input.neg()?.add_scalar(margin)?;
    let zero = Variable::wrap(Tensor::zeros_like(&input.data())?);
    let loss_neg = margin_minus_x.maximum(&zero)?;
    let label_pos = label.add_scalar(1.0)?.mul_scalar(0.5)?;
    let label_neg = label.neg()?.add_scalar(1.0)?.mul_scalar(0.5)?;
    let loss = label_pos.mul(input)?.add(&label_neg.mul(&loss_neg)?)?;
    loss.mean()
}

/// Margin ranking loss.
///
/// `loss = max(0, -label * (x1 - x2) + margin)`
///
/// `x1`, `x2`, and `label` must all be Float (same dtype).
pub fn margin_ranking_loss(
    x1: &Variable,
    x2: &Variable,
    label: &Variable,
    margin: f64,
) -> Result<Variable> {
    let diff = x1.sub(x2)?;
    let neg_label_diff = label.neg()?.mul(&diff)?;
    let shifted = neg_label_diff.add_scalar(margin)?;
    let zero = Variable::wrap(Tensor::zeros_like(&shifted.data())?);
    shifted.maximum(&zero)?.mean()
}

/// Poisson negative log likelihood loss.
///
/// `loss = exp(input) - target * input`
///
/// Both `input` and `target` must be Float (Float32 or Float64), same dtype.
/// For count data modeled as Poisson distributions.
pub fn poisson_nll_loss(input: &Variable, target: &Variable, log_input: bool) -> Result<Variable> {
    let loss = if log_input {
        // loss = exp(input) - target * input
        input.exp()?.sub(&target.mul(input)?)?
    } else {
        // loss = input - target * log(input + eps)
        let log_in = input.add_scalar(1e-8)?.log()?;
        input.sub(&target.mul(&log_in)?)?
    };
    loss.mean()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Tensor, test_device, test_opts};

    #[test]
    fn test_nll_loss() {
        let log_probs = Tensor::from_f32(
            &[-0.5, -1.5, -2.5, -1.0, -0.5, -2.0],
            &[2, 3], test_device(),
        ).unwrap();
        let targets = Tensor::from_i64(&[0, 1], &[2], test_device()).unwrap();
        let input = Variable::new(log_probs, false);
        let target = Variable::new(targets, false);
        let loss = nll_loss(&input, &target).unwrap();
        assert!(loss.item().unwrap() > 0.0);
    }

    #[test]
    fn test_focal_loss() {
        let logits = Tensor::from_f32(
            &[2.0, 0.5, -1.0, -0.5, 1.5, 0.3],
            &[2, 3], test_device(),
        ).unwrap();
        let targets = Tensor::from_i64(&[0, 1], &[2], test_device()).unwrap();
        let pred = Variable::new(logits, false);
        let target = Variable::new(targets, false);
        let loss = focal_loss(&pred, &target, 0.25, 2.0).unwrap();
        assert!(loss.item().unwrap() > 0.0);
    }

    #[test]
    fn test_triplet_margin_loss() {
        // Distance to positive ~ 0.14, to negative ~ 1.41, margin = 2.0
        // loss = max(0, 0.14 - 1.41 + 2.0) = 0.73
        let a = Variable::new(Tensor::from_f32(&[1.0, 0.0], &[1, 2], test_device()).unwrap(), false);
        let p = Variable::new(Tensor::from_f32(&[0.9, 0.1], &[1, 2], test_device()).unwrap(), false);
        let n = Variable::new(Tensor::from_f32(&[0.0, 1.0], &[1, 2], test_device()).unwrap(), false);
        let loss = triplet_margin_loss(&a, &p, &n, 2.0).unwrap();
        assert!(loss.item().unwrap() > 0.0);
    }

    #[test]
    fn test_comparison_losses_preserve_float64() {
        // These four losses build a zero/ones comparison operand internally
        // (for the max(0, ...) hinge). It must inherit the input's dtype:
        // documented as "Float (Float32 or Float64)", so a Float64 input must
        // yield a Float64 loss, not silently ride on implicit promotion of a
        // hardcoded Float32 operand. Also checks the value matches the
        // Float32 path (triplet == the 0.727... the Float32 test asserts).
        use crate::tensor::DType;
        let dev = test_device();
        let f64v = |d: &[f64], s: &[i64]| {
            Variable::new(Tensor::from_f64(d, s, dev).unwrap(), false)
        };

        let a = f64v(&[1.0, 0.0], &[1, 2]);
        let p = f64v(&[0.9, 0.1], &[1, 2]);
        let n = f64v(&[0.0, 1.0], &[1, 2]);
        let triplet = triplet_margin_loss(&a, &p, &n, 2.0).unwrap();
        assert_eq!(triplet.data().dtype(), DType::Float64);
        assert!((triplet.item().unwrap() - 0.7272077938642143).abs() < 1e-9);

        let label = f64v(&[1.0], &[1]);
        let cosine = cosine_embedding_loss(&a, &p, &label, 0.5).unwrap();
        assert_eq!(cosine.data().dtype(), DType::Float64);
        assert!(cosine.item().unwrap().is_finite());

        let x = f64v(&[0.3, 0.7], &[2]);
        let l2 = f64v(&[1.0, -1.0], &[2]);
        let hinge = hinge_embedding_loss(&x, &l2, 1.0).unwrap();
        assert_eq!(hinge.data().dtype(), DType::Float64);
        assert!(hinge.item().unwrap().is_finite());

        let x1 = f64v(&[0.3, 0.7], &[2]);
        let x2 = f64v(&[0.1, 0.9], &[2]);
        let mr = margin_ranking_loss(&x1, &x2, &l2, 0.5).unwrap();
        assert_eq!(mr.data().dtype(), DType::Float64);
        assert!(mr.item().unwrap().is_finite());
    }

    #[test]
    fn test_poisson_nll_loss() {
        let input = Variable::new(Tensor::from_f32(&[1.0, 2.0, 3.0], &[3], test_device()).unwrap(), false);
        let target = Variable::new(Tensor::from_f32(&[1.0, 2.0, 3.0], &[3], test_device()).unwrap(), false);
        let loss = poisson_nll_loss(&input, &target, true).unwrap();
        assert!(loss.item().unwrap() > 0.0);
    }

    #[test]
    fn test_mse_loss() {
        let pred = Variable::new(
            Tensor::from_f32(&[1.0, 2.0, 3.0], &[3], test_device()).unwrap(), true,
        );
        let target = Variable::new(
            Tensor::from_f32(&[1.5, 2.5, 3.5], &[3], test_device()).unwrap(), false,
        );
        let loss = mse_loss(&pred, &target).unwrap();
        // mean((0.5)^2) = 0.25
        assert!((loss.item().unwrap() - 0.25).abs() < 1e-5);
    }

    #[test]
    fn test_mse_loss_zero() {
        let pred = Variable::new(
            Tensor::from_f32(&[1.0, 2.0], &[2], test_device()).unwrap(), false,
        );
        let target = Variable::new(
            Tensor::from_f32(&[1.0, 2.0], &[2], test_device()).unwrap(), false,
        );
        let loss = mse_loss(&pred, &target).unwrap();
        assert!(loss.item().unwrap().abs() < 1e-6);
    }

    #[test]
    fn test_cross_entropy_loss_class_indices() {
        // Logits where class 0 is strongly predicted for sample 0, class 2 for sample 1
        let logits = Variable::new(
            Tensor::from_f32(&[10.0, 0.0, 0.0, 0.0, 0.0, 10.0], &[2, 3], test_device()).unwrap(),
            true,
        );
        let targets = Variable::new(
            Tensor::from_i64(&[0, 2], &[2], test_device()).unwrap(), false,
        );
        let loss = cross_entropy_loss(&logits, &targets).unwrap();
        // Confident correct predictions -> loss near 0
        assert!(loss.item().unwrap() < 0.1);
    }

    #[test]
    fn test_cross_entropy_loss_wrong_predictions() {
        // Logits predicting wrong class
        let logits = Variable::new(
            Tensor::from_f32(&[0.0, 0.0, 10.0, 10.0, 0.0, 0.0], &[2, 3], test_device()).unwrap(),
            false,
        );
        let targets = Variable::new(
            Tensor::from_i64(&[0, 2], &[2], test_device()).unwrap(), false,
        );
        let loss = cross_entropy_loss(&logits, &targets).unwrap();
        // Wrong predictions -> high loss
        assert!(loss.item().unwrap() > 5.0);
    }

    #[test]
    fn test_cross_entropy_loss_gradient() {
        let logits = Variable::new(
            Tensor::from_f32(&[2.0, 1.0, 0.1, 0.5, 1.5, 0.3], &[2, 3], test_device()).unwrap(),
            true,
        );
        let targets = Variable::new(
            Tensor::from_i64(&[0, 1], &[2], test_device()).unwrap(), false,
        );
        let loss = cross_entropy_loss(&logits, &targets).unwrap();
        loss.backward().unwrap();
        assert!(logits.grad().is_some());
    }

    #[test]
    fn test_bce_loss() {
        let pred = Variable::new(
            Tensor::from_f32(&[0.9, 0.1, 0.8, 0.2], &[4], test_device()).unwrap(), false,
        );
        let target = Variable::new(
            Tensor::from_f32(&[1.0, 0.0, 1.0, 0.0], &[4], test_device()).unwrap(), false,
        );
        let loss = bce_loss(&pred, &target).unwrap();
        // High-confidence correct predictions -> low loss
        assert!(loss.item().unwrap() < 0.3);
    }

    #[test]
    fn test_bce_with_logits_loss() {
        // Positive logits for label=1, negative for label=0
        let pred = Variable::new(
            Tensor::from_f32(&[5.0, -5.0, 5.0, -5.0], &[4], test_device()).unwrap(), true,
        );
        let target = Variable::new(
            Tensor::from_f32(&[1.0, 0.0, 1.0, 0.0], &[4], test_device()).unwrap(), false,
        );
        let loss = bce_with_logits_loss(&pred, &target).unwrap();
        assert!(loss.item().unwrap() < 0.1);
    }

    #[test]
    fn test_bce_with_logits_gradient() {
        let pred = Variable::new(
            Tensor::from_f32(&[0.5, -0.5], &[2], test_device()).unwrap(), true,
        );
        let target = Variable::new(
            Tensor::from_f32(&[1.0, 0.0], &[2], test_device()).unwrap(), false,
        );
        let loss = bce_with_logits_loss(&pred, &target).unwrap();
        loss.backward().unwrap();
        assert!(pred.grad().is_some());
    }

    #[test]
    fn test_l1_loss() {
        let pred = Variable::new(
            Tensor::from_f32(&[1.0, 2.0, 3.0], &[3], test_device()).unwrap(), false,
        );
        let target = Variable::new(
            Tensor::from_f32(&[1.5, 2.5, 3.5], &[3], test_device()).unwrap(), false,
        );
        let loss = l1_loss(&pred, &target).unwrap();
        // mean(|0.5|) = 0.5
        assert!((loss.item().unwrap() - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_smooth_l1_loss() {
        let pred = Variable::new(
            Tensor::from_f32(&[1.0, 5.0], &[2], test_device()).unwrap(), false,
        );
        let target = Variable::new(
            Tensor::from_f32(&[1.5, 2.0], &[2], test_device()).unwrap(), false,
        );
        let loss = smooth_l1_loss(&pred, &target, 1.0).unwrap();
        // |0.5| < 1.0 -> 0.5*0.25/1.0 = 0.125; |3.0| >= 1.0 -> 3.0 - 0.5 = 2.5
        // mean = (0.125 + 2.5) / 2 = 1.3125
        assert!((loss.item().unwrap() - 1.3125).abs() < 1e-4);
    }

    #[test]
    fn test_smooth_l1_loss_negative_beta() {
        let pred = Variable::new(
            Tensor::from_f32(&[1.0], &[1], test_device()).unwrap(), false,
        );
        let target = Variable::new(
            Tensor::from_f32(&[2.0], &[1], test_device()).unwrap(), false,
        );
        assert!(smooth_l1_loss(&pred, &target, -1.0).is_err());
    }

    #[test]
    fn test_kl_div_loss() {
        // log_softmax output as input
        let logits = Tensor::from_f32(&[2.0, 1.0, 0.1, 0.5, 1.5, 0.3], &[2, 3], test_device()).unwrap();
        let log_probs = Variable::new(logits.log_softmax(1).unwrap(), false);
        // Uniform target distribution
        let target = Variable::new(
            Tensor::from_f32(&[1.0/3.0; 6], &[2, 3], test_device()).unwrap(), false,
        );
        let loss = kl_div_loss(&log_probs, &target).unwrap();
        // KL divergence >= 0
        assert!(loss.item().unwrap() >= -1e-5);
    }

    #[test]
    fn test_ctc_loss() {
        let dev = test_device();
        // T=5 timesteps, N=1 batch, C=4 classes (including blank=0)
        let log_probs = Variable::new(
            Tensor::randn(&[5, 1, 4], test_opts()).unwrap()
                .log_softmax(2).unwrap(),
            false,
        );
        let targets = Variable::new(
            Tensor::from_i64(&[1, 2, 3], &[1, 3], dev).unwrap(), false,
        );
        let input_lengths = Variable::new(
            Tensor::from_i64(&[5], &[1], dev).unwrap(), false,
        );
        let target_lengths = Variable::new(
            Tensor::from_i64(&[3], &[1], dev).unwrap(), false,
        );
        let loss = ctc_loss(&log_probs, &targets, &input_lengths, &target_lengths, 0).unwrap();
        assert!(loss.item().unwrap() > 0.0);
    }

    #[test]
    fn test_cosine_embedding_loss_similar() {
        let dev = test_device();
        let x1 = Variable::new(Tensor::from_f32(&[1.0, 0.0, 0.0], &[1, 3], dev).unwrap(), false);
        let x2 = Variable::new(Tensor::from_f32(&[1.0, 0.0, 0.0], &[1, 3], dev).unwrap(), false);
        let label = Variable::new(Tensor::from_f32(&[1.0], &[1], dev).unwrap(), false);
        let loss = cosine_embedding_loss(&x1, &x2, &label, 0.0).unwrap();
        // Identical vectors, label=1 -> loss = 1 - cos(0) = 0
        assert!(loss.item().unwrap().abs() < 1e-4);
    }

    #[test]
    fn test_cosine_embedding_loss_dissimilar() {
        let dev = test_device();
        let x1 = Variable::new(Tensor::from_f32(&[1.0, 0.0, 0.0], &[1, 3], dev).unwrap(), false);
        let x2 = Variable::new(Tensor::from_f32(&[-1.0, 0.0, 0.0], &[1, 3], dev).unwrap(), false);
        let label = Variable::new(Tensor::from_f32(&[-1.0], &[1], dev).unwrap(), false);
        let loss = cosine_embedding_loss(&x1, &x2, &label, 0.0).unwrap();
        // Opposite vectors, label=-1, margin=0 -> max(0, cos(pi) - 0) = max(0, -1) = 0
        assert!(loss.item().unwrap().abs() < 1e-4);
    }

    #[test]
    fn test_hinge_embedding_loss() {
        let dev = test_device();
        // Positive samples (label=1): loss = input values
        let input = Variable::new(Tensor::from_f32(&[0.5, 0.3], &[2], dev).unwrap(), false);
        let label = Variable::new(Tensor::from_f32(&[1.0, 1.0], &[2], dev).unwrap(), false);
        let loss = hinge_embedding_loss(&input, &label, 1.0).unwrap();
        // mean(0.5, 0.3) = 0.4
        assert!((loss.item().unwrap() - 0.4).abs() < 1e-4);
    }

    #[test]
    fn test_hinge_embedding_loss_negative() {
        let dev = test_device();
        // Negative samples (label=-1): loss = max(0, margin - input)
        let input = Variable::new(Tensor::from_f32(&[0.5, 2.0], &[2], dev).unwrap(), false);
        let label = Variable::new(Tensor::from_f32(&[-1.0, -1.0], &[2], dev).unwrap(), false);
        let loss = hinge_embedding_loss(&input, &label, 1.0).unwrap();
        // max(0, 1.0 - 0.5) = 0.5; max(0, 1.0 - 2.0) = 0; mean = 0.25
        assert!((loss.item().unwrap() - 0.25).abs() < 1e-4);
    }

    #[test]
    fn test_margin_ranking_loss() {
        let dev = test_device();
        let x1 = Variable::new(Tensor::from_f32(&[1.0, 3.0], &[2], dev).unwrap(), false);
        let x2 = Variable::new(Tensor::from_f32(&[2.0, 1.0], &[2], dev).unwrap(), false);
        let label = Variable::new(Tensor::from_f32(&[1.0, 1.0], &[2], dev).unwrap(), false);
        let loss = margin_ranking_loss(&x1, &x2, &label, 0.0).unwrap();
        // label=1: loss = max(0, -(x1-x2)) = max(0, -(-1))=1 and max(0, -(2))=0
        // mean = 0.5
        assert!((loss.item().unwrap() - 0.5).abs() < 1e-4);
    }

    #[test]
    fn test_margin_ranking_loss_with_margin() {
        let dev = test_device();
        let x1 = Variable::new(Tensor::from_f32(&[3.0], &[1], dev).unwrap(), false);
        let x2 = Variable::new(Tensor::from_f32(&[1.0], &[1], dev).unwrap(), false);
        let label = Variable::new(Tensor::from_f32(&[1.0], &[1], dev).unwrap(), false);
        let loss = margin_ranking_loss(&x1, &x2, &label, 3.0).unwrap();
        // max(0, -(3-1) + 3) = max(0, 1) = 1.0
        assert!((loss.item().unwrap() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_poisson_nll_loss_no_log() {
        let dev = test_device();
        let input = Variable::new(
            Tensor::from_f32(&[1.0, 2.0, 3.0], &[3], dev).unwrap(), false,
        );
        let target = Variable::new(
            Tensor::from_f32(&[1.0, 2.0, 3.0], &[3], dev).unwrap(), false,
        );
        let loss = poisson_nll_loss(&input, &target, false).unwrap();
        assert!(loss.item().unwrap().is_finite());
    }

    #[test]
    fn test_focal_loss_reduces_to_ce_at_gamma_zero() {
        let dev = test_device();
        let logits = Variable::new(
            Tensor::from_f32(&[2.0, 0.5, -1.0, -0.5, 1.5, 0.3], &[2, 3], dev).unwrap(),
            false,
        );
        let targets = Variable::new(
            Tensor::from_i64(&[0, 1], &[2], dev).unwrap(), false,
        );
        // gamma=0, alpha=1 should approximate cross-entropy
        let fl = focal_loss(&logits, &targets, 1.0, 0.0).unwrap().item().unwrap();
        let ce = cross_entropy_loss(&logits, &targets).unwrap().item().unwrap();
        assert!((fl - ce).abs() < 1e-4, "focal(gamma=0, alpha=1) = {fl} != ce = {ce}");
    }

    #[test]
    fn test_triplet_margin_loss_zero_when_far() {
        let dev = test_device();
        let a = Variable::new(Tensor::from_f32(&[1.0, 0.0], &[1, 2], dev).unwrap(), false);
        let p = Variable::new(Tensor::from_f32(&[1.0, 0.1], &[1, 2], dev).unwrap(), false);
        let n = Variable::new(Tensor::from_f32(&[0.0, 10.0], &[1, 2], dev).unwrap(), false);
        // Negative is very far, margin 0.1 -> loss should be 0
        let loss = triplet_margin_loss(&a, &p, &n, 0.1).unwrap();
        assert!(loss.item().unwrap() < 1e-4);
    }
}
