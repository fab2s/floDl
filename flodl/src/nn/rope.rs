use crate::autograd::Variable;
use crate::tensor::{Device, Result, Tensor, TensorError};

/// Rotary Position Embedding (RoPE).
///
/// Encodes absolute positions as rotations of query/key pairs so that
/// attention scores depend only on relative offsets. Used in LLaMA,
/// OLMo, Gemma, and most modern decoder architectures.
///
/// The sin/cos tables are precomputed once for `max_seq_len` positions
/// and held as constant (non-trainable) device tensors; `apply` rotates
/// query and key after they are split into heads.
///
/// Shapes: `apply` expects `[batch, heads, seq, head_dim]` and returns
/// the same shape. `head_dim` must be even.
///
/// Cloning is cheap (the tables are shared, not copied), so one
/// instance can serve every layer of a deep model.
///
/// ```ignore
/// let rope = RotaryEmbedding::on_device(64, 2048, device)?;
/// let (q, k) = rope.apply(&q, &k)?; // q, k: [batch, heads, seq, 64]
/// ```
#[derive(Clone)]
pub struct RotaryEmbedding {
    cos: Variable,
    sin: Variable,
    head_dim: i64,
    max_seq_len: i64,
}

impl RotaryEmbedding {
    /// Create a rotary embedding on CPU with the default base (10000.0).
    pub fn new(head_dim: i64, max_seq_len: i64) -> Result<Self> {
        Self::on_device(head_dim, max_seq_len, Device::CPU)
    }

    /// Create a rotary embedding on a specific device with the default
    /// base (10000.0).
    pub fn on_device(head_dim: i64, max_seq_len: i64, device: Device) -> Result<Self> {
        Self::on_device_theta(head_dim, max_seq_len, 10000.0, device)
    }

    /// Create a rotary embedding with an explicit frequency base
    /// (`theta`, also called `rope_theta`).
    pub fn on_device_theta(
        head_dim: i64,
        max_seq_len: i64,
        theta: f64,
        device: Device,
    ) -> Result<Self> {
        if head_dim <= 0 || head_dim % 2 != 0 {
            return Err(TensorError::new(&format!(
                "RotaryEmbedding: head_dim ({head_dim}) must be positive and even"
            )));
        }
        if max_seq_len <= 0 {
            return Err(TensorError::new(&format!(
                "RotaryEmbedding: max_seq_len ({max_seq_len}) must be positive"
            )));
        }
        if theta <= 0.0 {
            return Err(TensorError::new(&format!(
                "RotaryEmbedding: theta ({theta}) must be positive"
            )));
        }

        let half = (head_dim / 2) as usize;
        let seq = max_seq_len as usize;

        // inv_freq[i] = theta^(-2i / head_dim); each cache row holds the
        // angle table duplicated over both halves, matching the
        // rotate-half convention used by apply().
        let inv_freq: Vec<f64> = (0..half)
            .map(|i| theta.powf(-2.0 * i as f64 / head_dim as f64))
            .collect();

        let mut cos = vec![0f32; seq * head_dim as usize];
        let mut sin = vec![0f32; seq * head_dim as usize];
        for pos in 0..seq {
            for (i, f) in inv_freq.iter().enumerate() {
                let angle = pos as f64 * f;
                let (s, c) = angle.sin_cos();
                let row = pos * head_dim as usize;
                cos[row + i] = c as f32;
                cos[row + half + i] = c as f32;
                sin[row + i] = s as f32;
                sin[row + half + i] = s as f32;
            }
        }

        let shape = [max_seq_len, head_dim];
        Ok(RotaryEmbedding {
            cos: Variable::new(Tensor::from_f32(&cos, &shape, device)?, false),
            sin: Variable::new(Tensor::from_f32(&sin, &shape, device)?, false),
            head_dim,
            max_seq_len,
        })
    }

    /// The head dimension the tables were built for.
    pub fn head_dim(&self) -> i64 {
        self.head_dim
    }

    /// The maximum sequence length the tables cover.
    pub fn max_seq_len(&self) -> i64 {
        self.max_seq_len
    }

    /// Rotate query and key. Both must be `[batch, heads, seq, head_dim]`;
    /// query and key may have different `heads`/`seq` (cross-attention,
    /// grouped-query attention).
    pub fn apply(&self, query: &Variable, key: &Variable) -> Result<(Variable, Variable)> {
        Ok((self.apply_one(query)?, self.apply_one(key)?))
    }

    /// Rotate a single `[batch, heads, seq, head_dim]` tensor.
    pub fn apply_one(&self, x: &Variable) -> Result<Variable> {
        let shape = x.shape();
        if shape.len() != 4 {
            return Err(TensorError::new(&format!(
                "RotaryEmbedding::apply expects [batch, heads, seq, head_dim], \
                 got {shape:?}"
            )));
        }
        let seq = shape[2];
        if shape[3] != self.head_dim {
            return Err(TensorError::new(&format!(
                "RotaryEmbedding::apply: head_dim mismatch (tables built for \
                 {}, input has {})",
                self.head_dim, shape[3]
            )));
        }
        if seq > self.max_seq_len {
            return Err(TensorError::new(&format!(
                "RotaryEmbedding::apply: seq ({seq}) exceeds max_seq_len ({})",
                self.max_seq_len
            )));
        }

        // [seq, head_dim] -> [1, 1, seq, head_dim] for broadcast.
        let cos = self.cos.narrow(0, 0, seq)?.unsqueeze(0)?.unsqueeze(0)?;
        let sin = self.sin.narrow(0, 0, seq)?.unsqueeze(0)?.unsqueeze(0)?;

        // rotate_half(x) = cat(-x2, x1) over the last dim.
        let half = self.head_dim / 2;
        let x1 = x.narrow(3, 0, half)?;
        let x2 = x.narrow(3, half, half)?;
        let rotated = x2.neg()?.cat(&x1, 3)?;

        x.mul(&cos)?.add(&rotated.mul(&sin)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{test_device, test_opts};

    #[test]
    fn test_rope_ctor_validation() {
        let device = test_device();
        // Odd / non-positive head_dim rejected.
        assert!(RotaryEmbedding::on_device(3, 8, device).is_err());
        assert!(RotaryEmbedding::on_device(0, 8, device).is_err());
        // Non-positive max_seq_len / theta rejected.
        assert!(RotaryEmbedding::on_device(4, 0, device).is_err());
        assert!(RotaryEmbedding::on_device_theta(4, 8, 0.0, device).is_err());
    }

    #[test]
    fn test_rope_shape_preserved() {
        let device = test_device();
        let rope = RotaryEmbedding::on_device(8, 16, device).unwrap();
        let x = Variable::new(Tensor::randn(&[2, 4, 5, 8], test_opts()).unwrap(), false);
        let y = rope.apply_one(&x).unwrap();
        assert_eq!(y.shape(), vec![2, 4, 5, 8]);
    }

    #[test]
    fn test_rope_input_validation() {
        let device = test_device();
        let rope = RotaryEmbedding::on_device(8, 4, device).unwrap();
        let opts = test_opts();
        // Not 4-D.
        let x3 = Variable::new(Tensor::randn(&[2, 5, 8], opts).unwrap(), false);
        assert!(rope.apply_one(&x3).is_err());
        // head_dim mismatch.
        let xd = Variable::new(Tensor::randn(&[1, 1, 2, 6], opts).unwrap(), false);
        assert!(rope.apply_one(&xd).is_err());
        // seq beyond the table.
        let xs = Variable::new(Tensor::randn(&[1, 1, 5, 8], opts).unwrap(), false);
        assert!(rope.apply_one(&xs).is_err());
    }

    #[test]
    fn test_rope_position_zero_is_identity() {
        // At position 0 every angle is 0 (cos=1, sin=0), so seq=1 input
        // must come back unchanged.
        let device = test_device();
        let rope = RotaryEmbedding::on_device(8, 4, device).unwrap();
        let x = Variable::new(Tensor::randn(&[1, 2, 1, 8], test_opts()).unwrap(), false);
        let y = rope.apply_one(&x).unwrap();
        let xin = x.data().to_f32_vec().unwrap();
        let out = y.data().to_f32_vec().unwrap();
        for (a, b) in xin.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-6, "position 0 changed: {a} vs {b}");
        }
    }

    #[test]
    fn test_rope_numeric_head_dim_two() {
        // head_dim=2, half=1: out = [x0*cos - x1*sin, x1*cos + x0*sin]
        // with angle = pos (inv_freq[0] = 1). Check position 1.
        let device = test_device();
        let rope = RotaryEmbedding::on_device(2, 4, device).unwrap();
        let x = Variable::new(
            Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2], device).unwrap(),
            false,
        );
        let y = rope.apply_one(&x).unwrap();
        let out = y.data().to_f32_vec().unwrap();
        // Position 0: identity.
        assert!((out[0] - 1.0).abs() < 1e-5);
        assert!((out[1] - 2.0).abs() < 1e-5);
        // Position 1: rotation by 1 radian.
        let (s, c) = 1f32.sin_cos();
        assert!((out[2] - (3.0 * c - 4.0 * s)).abs() < 1e-5, "got {}", out[2]);
        assert!((out[3] - (4.0 * c + 3.0 * s)).abs() < 1e-5, "got {}", out[3]);
    }

    #[test]
    fn test_rope_preserves_norm() {
        // Rotations are orthogonal: per-position vector norm is unchanged.
        let device = test_device();
        let rope = RotaryEmbedding::on_device(16, 8, device).unwrap();
        let x = Variable::new(Tensor::randn(&[1, 1, 8, 16], test_opts()).unwrap(), false);
        let y = rope.apply_one(&x).unwrap();
        let xin = x.data().to_f32_vec().unwrap();
        let out = y.data().to_f32_vec().unwrap();
        for pos in 0..8 {
            let nx: f32 = xin[pos * 16..(pos + 1) * 16].iter().map(|v| v * v).sum();
            let ny: f32 = out[pos * 16..(pos + 1) * 16].iter().map(|v| v * v).sum();
            assert!(
                (nx.sqrt() - ny.sqrt()).abs() < 1e-4,
                "norm changed at pos {pos}: {} vs {}",
                nx.sqrt(),
                ny.sqrt()
            );
        }
    }

    #[test]
    fn test_rope_gradient_flows() {
        let device = test_device();
        let rope = RotaryEmbedding::on_device(8, 8, device).unwrap();
        let x = Variable::new(Tensor::randn(&[1, 2, 4, 8], test_opts()).unwrap(), true);
        let y = rope.apply_one(&x).unwrap();
        y.sum().unwrap().backward().unwrap();
        let grad = x.grad().unwrap();
        assert_eq!(grad.shape(), vec![1, 2, 4, 8]);
    }

    #[test]
    fn test_rope_apply_pair() {
        let device = test_device();
        let rope = RotaryEmbedding::on_device(8, 8, device).unwrap();
        let opts = test_opts();
        // Different heads/seq for q and k (GQA / cross-attention shape).
        let q = Variable::new(Tensor::randn(&[1, 4, 6, 8], opts).unwrap(), false);
        let k = Variable::new(Tensor::randn(&[1, 2, 3, 8], opts).unwrap(), false);
        let (rq, rk) = rope.apply(&q, &k).unwrap();
        assert_eq!(rq.shape(), vec![1, 4, 6, 8]);
        assert_eq!(rk.shape(), vec![1, 2, 3, 8]);
    }
}
