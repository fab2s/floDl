use crate::autograd::Variable;
use crate::tensor::{Device, Result, Tensor};

use super::init;
use super::parameter::Parameter;
use super::rope::RotaryEmbedding;
use super::Module;

/// Multi-head attention mechanism.
///
/// Implements `MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O`
/// where each `head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)`.
///
/// Supports optional causal masking, key-value attention masks, and
/// rotary position embeddings (applied to query/key after the head
/// split when configured via [`MultiheadAttention::rotary`]).
///
/// ```ignore
/// let mha = MultiheadAttention::on_device(512, 8, device)?;
/// // Self-attention: query = key = value
/// let y = mha.forward(&x)?;
/// // Cross-attention or masked: use forward_ext
/// let y = mha.forward_ext(&query, &key, &value, Some(&mask))?;
/// // Rotary positions (LLaMA / OLMo style):
/// let mha = MultiheadAttention::on_device(512, 8, device)?
///     .rotary(RotaryEmbedding::on_device(64, 2048, device)?)?;
/// ```
pub struct MultiheadAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: i64,
    head_dim: i64,
    scale: f64,
    rotary: Option<RotaryEmbedding>,
}

struct Linear {
    weight: Parameter,
    bias: Parameter,
}

impl Linear {
    fn on_device(in_features: i64, out_features: i64, device: Device) -> Result<Self> {
        let w = init::xavier_uniform(
            &[out_features, in_features], in_features, out_features, device,
        )?;
        let b = Tensor::zeros(
            &[out_features],
            crate::tensor::TensorOptions { dtype: crate::tensor::DType::Float32, device },
        )?;
        Ok(Linear {
            weight: Parameter::new(w, "weight"),
            bias: Parameter::new(b, "bias"),
        })
    }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        crate::autograd::linear(
            input,
            &self.weight.variable,
            Some(&self.bias.variable),
        )
    }

    fn parameters(&self, prefix: &str) -> Vec<Parameter> {
        vec![
            Parameter {
                variable: self.weight.variable.clone(),
                name: format!("{prefix}.weight"),
            },
            Parameter {
                variable: self.bias.variable.clone(),
                name: format!("{prefix}.bias"),
            },
        ]
    }
}

impl MultiheadAttention {
    /// Create a multi-head attention module on CPU.
    pub fn new(embed_dim: i64, num_heads: i64) -> Result<Self> {
        Self::on_device(embed_dim, num_heads, Device::CPU)
    }

    /// Create a multi-head attention module on a specific device.
    pub fn on_device(embed_dim: i64, num_heads: i64, device: Device) -> Result<Self> {
        if num_heads <= 0 || embed_dim % num_heads != 0 {
            return Err(crate::tensor::TensorError::new(&format!(
                "MultiheadAttention: embed_dim ({embed_dim}) must be divisible by \
                 a positive num_heads ({num_heads})"
            )));
        }
        let head_dim = embed_dim / num_heads;

        Ok(MultiheadAttention {
            q_proj: Linear::on_device(embed_dim, embed_dim, device)?,
            k_proj: Linear::on_device(embed_dim, embed_dim, device)?,
            v_proj: Linear::on_device(embed_dim, embed_dim, device)?,
            out_proj: Linear::on_device(embed_dim, embed_dim, device)?,
            num_heads,
            head_dim,
            scale: 1.0 / (head_dim as f64).sqrt(),
            rotary: None,
        })
    }

    /// Attach a rotary position embedding, applied to query and key
    /// after the head split. The embedding's `head_dim` must match this
    /// module's per-head dimension.
    pub fn rotary(mut self, rope: RotaryEmbedding) -> Result<Self> {
        if rope.head_dim() != self.head_dim {
            return Err(crate::tensor::TensorError::new(&format!(
                "MultiheadAttention::rotary: RotaryEmbedding head_dim ({}) \
                 does not match per-head dim ({})",
                rope.head_dim(),
                self.head_dim
            )));
        }
        self.rotary = Some(rope);
        Ok(self)
    }

    /// Full attention forward with separate query, key, value and optional mask.
    ///
    /// Shapes:
    /// - query: `[batch, seq_q, embed_dim]`
    /// - key:   `[batch, seq_k, embed_dim]`
    /// - value: `[batch, seq_k, embed_dim]`
    /// - mask:  `[seq_q, seq_k]` or `[batch, 1, seq_q, seq_k]` (true/non-zero = masked positions)
    ///
    /// Returns: `[batch, seq_q, embed_dim]`
    pub fn forward_ext(
        &self,
        query: &Variable,
        key: &Variable,
        value: &Variable,
        mask: Option<&Tensor>,
    ) -> Result<Variable> {
        let q_shape = query.shape();
        let k_shape = key.shape();
        if q_shape.len() < 3 || k_shape.len() < 3 {
            return Err(crate::tensor::TensorError::new(&format!(
                "MultiheadAttention::forward_ext expects 3-D [batch, seq, embed] \
                 query/key, got query {q_shape:?}, key {k_shape:?}"
            )));
        }
        let batch = q_shape[0];
        let seq_q = q_shape[1];
        let seq_k = k_shape[1];

        // Project Q, K, V
        let q = self.q_proj.forward(query)?;
        let k = self.k_proj.forward(key)?;
        let v = self.v_proj.forward(value)?;

        // Reshape to [batch, num_heads, seq, head_dim]
        let q = q.reshape(&[batch, seq_q, self.num_heads, self.head_dim])?
                 .transpose(1, 2)?;
        let k = k.reshape(&[batch, seq_k, self.num_heads, self.head_dim])?
                 .transpose(1, 2)?;
        let v = v.reshape(&[batch, seq_k, self.num_heads, self.head_dim])?
                 .transpose(1, 2)?;

        // Rotary positions rotate query/key in place of additive
        // position embeddings.
        let (q, k) = match &self.rotary {
            Some(rope) => rope.apply(&q, &k)?,
            None => (q, k),
        };

        // Attention core: libtorch's fused scaled-dot-product attention,
        // which picks flash / mem-efficient / math per device + dtype. The
        // point of the fused call over the explicit matmul→softmax→matmul
        // chain is memory, not speed: the chain holds TWO
        // [batch, heads, seq_q, seq_k] tensors (scores + probs) through
        // backward — ~300MB of the OLMo-150M seq-256 batch-4 envelope,
        // double at batch 8 — while the fused backends never materialize
        // them. The math fallback matches the old chain.
        //
        // Mask conversion: this module's contract is true/non-zero = MASKED,
        // which is the INVERSE of SDPA's boolean convention (true =
        // participate). Rather than invert (and depend on the caller's mask
        // dtype), the mask becomes an ADDITIVE float mask — 0 where
        // attended, -inf where masked — the exact math `masked_fill` +
        // softmax applied, and only [seq_q, seq_k]-sized.
        let add_mask = match mask {
            Some(m) => Some(
                Tensor::zeros(
                    &m.shape(),
                    crate::tensor::TensorOptions {
                        dtype: crate::tensor::DType::Float32,
                        device: m.device(),
                    },
                )?
                .masked_fill(m, f64::NEG_INFINITY)?,
            ),
            None => None,
        };
        // [batch, heads, seq_q, head_dim]
        let out = crate::autograd::scaled_dot_product_attention(
            &q, &k, &v,
            add_mask.as_ref(),
            /*dropout_p=*/0.0,
            /*is_causal=*/false,
            Some(self.scale),
        )?;

        // Reshape back: [batch, seq_q, embed_dim]
        let out = out.transpose(1, 2)?
                     .reshape(&[batch, seq_q, self.num_heads * self.head_dim])?;

        // Output projection
        self.out_proj.forward(&out)
    }
}

impl Module for MultiheadAttention {
    fn name(&self) -> &str { "multihead_attention" }

    /// Self-attention forward: query = key = value = input, no mask.
    fn forward(&self, input: &Variable) -> Result<Variable> {
        self.forward_ext(input, input, input, None)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.q_proj.parameters("q_proj"));
        params.extend(self.k_proj.parameters("k_proj"));
        params.extend(self.v_proj.parameters("v_proj"));
        params.extend(self.out_proj.parameters("out_proj"));
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::test_device;

    #[test]
    fn test_mha_indivisible_heads_is_err_not_panic() {
        // embed_dim not divisible by num_heads at a Result ctor → Err (was
        // an assert! panic).
        let device = test_device();
        // `.map(|_| ())` because MultiheadAttention isn't Debug (unwrap_err needs it).
        let err = MultiheadAttention::on_device(10, 3, device).map(|_| ()).unwrap_err();
        assert!(err.to_string().contains("divisible"), "unexpected: {err}");
        // Zero / negative heads also rejected rather than dividing by zero.
        assert!(MultiheadAttention::on_device(8, 0, device).is_err());
    }

    #[test]
    fn test_mha_self_attention() {
        let device = test_device();
        let mha = MultiheadAttention::on_device(8, 2, device).unwrap();
        let opts = crate::tensor::test_opts();
        let x = Variable::new(
            Tensor::randn(&[2, 4, 8], opts).unwrap(), // [batch=2, seq=4, dim=8]
            false,
        );
        let y = mha.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![2, 4, 8]);
    }

    #[test]
    fn test_mha_cross_attention() {
        let device = test_device();
        let mha = MultiheadAttention::on_device(8, 2, device).unwrap();
        let opts = crate::tensor::test_opts();
        let q = Variable::new(Tensor::randn(&[1, 3, 8], opts).unwrap(), false);
        let kv = Variable::new(Tensor::randn(&[1, 5, 8], opts).unwrap(), false);
        let y = mha.forward_ext(&q, &kv, &kv, None).unwrap();
        assert_eq!(y.shape(), vec![1, 3, 8]); // seq_q=3, not seq_k=5
    }

    #[test]
    fn test_mha_causal_mask() {
        let device = test_device();
        let mha = MultiheadAttention::on_device(8, 2, device).unwrap();
        let opts = crate::tensor::test_opts();
        let x = Variable::new(Tensor::randn(&[1, 4, 8], opts).unwrap(), false);

        // Causal mask: upper triangle = true (masked)
        let mask = Tensor::ones(&[4, 4], opts).unwrap().triu(1).unwrap();
        let y = mha.forward_ext(&x, &x, &x, Some(&mask)).unwrap();
        assert_eq!(y.shape(), vec![1, 4, 8]);
    }

    #[test]
    fn test_mha_gradient() {
        let device = test_device();
        let mha = MultiheadAttention::on_device(8, 2, device).unwrap();
        let opts = crate::tensor::test_opts();
        let x = Variable::new(Tensor::randn(&[1, 3, 8], opts).unwrap(), true);
        let y = mha.forward(&x).unwrap();
        let loss = y.sum().unwrap();
        loss.backward().unwrap();

        let grad = x.grad().unwrap();
        assert_eq!(grad.shape(), vec![1, 3, 8]);
    }

    #[test]
    fn test_mha_parameters() {
        let mha = MultiheadAttention::new(16, 4).unwrap();
        let params = mha.parameters();
        // 4 projections * 2 (weight + bias) = 8 parameters
        assert_eq!(params.len(), 8);
    }

    #[test]
    fn test_mha_rotary_head_dim_mismatch_is_err() {
        let device = test_device();
        // embed_dim 8 / 2 heads -> head_dim 4; tables built for 8.
        let rope = RotaryEmbedding::on_device(8, 16, device).unwrap();
        let err = MultiheadAttention::on_device(8, 2, device)
            .unwrap()
            .rotary(rope)
            .map(|_| ())
            .unwrap_err();
        assert!(err.to_string().contains("head_dim"), "unexpected: {err}");
    }

    #[test]
    fn test_mha_rotary_forward_and_gradient() {
        let device = test_device();
        let rope = RotaryEmbedding::on_device(4, 16, device).unwrap();
        let mha = MultiheadAttention::on_device(8, 2, device)
            .unwrap()
            .rotary(rope)
            .unwrap();
        let opts = crate::tensor::test_opts();
        let x = Variable::new(Tensor::randn(&[2, 5, 8], opts).unwrap(), true);
        let y = mha.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![2, 5, 8]);
        y.sum().unwrap().backward().unwrap();
        assert_eq!(x.grad().unwrap().shape(), vec![2, 5, 8]);
    }

    /// The fused SDPA core must match the explicit
    /// matmul→scale→mask→softmax→matmul chain it replaced — same math,
    /// without holding [batch, heads, seq_q, seq_k] through backward. The
    /// reference below IS the pre-SDPA implementation, verbatim, including
    /// the module's true=masked → additive-mask conversion.
    #[test]
    fn test_sdpa_core_matches_explicit_score_chain() {
        let opts = crate::tensor::test_opts();
        let (b, h, lq, lk, e) = (2, 3, 5, 6, 4);
        let q = Tensor::randn(&[b, h, lq, e], opts).unwrap();
        let k = Tensor::randn(&[b, h, lk, e], opts).unwrap();
        let v = Tensor::randn(&[b, h, lk, e], opts).unwrap();
        let scale = 1.0 / (e as f64).sqrt();
        // Non-square skip pattern in the module's convention (non-zero = masked).
        let mask = Tensor::ones(&[lq, lk], opts).unwrap().triu(2).unwrap();

        for m in [None, Some(&mask)] {
            let mut scores = q
                .matmul(&k.transpose(2, 3).unwrap()).unwrap()
                .mul_scalar(scale).unwrap();
            if let Some(m) = m {
                scores = scores.masked_fill(m, f64::NEG_INFINITY).unwrap();
            }
            let reference = scores.softmax(-1).unwrap().matmul(&v).unwrap();

            let add_mask = m.map(|m| {
                Tensor::zeros(&m.shape(), opts).unwrap()
                    .masked_fill(m, f64::NEG_INFINITY).unwrap()
            });
            let fused = Tensor::scaled_dot_product_attention(
                &q, &k, &v, add_mask.as_ref(), 0.0, false, Some(scale),
            ).unwrap();

            let diff = fused.sub(&reference).unwrap().abs().unwrap()
                .max().unwrap().item().unwrap();
            assert!(
                diff < 1e-5,
                "SDPA diverged from the explicit chain (masked={}): max |Δ| = {diff}",
                m.is_some(),
            );
        }
    }

    #[test]
    fn test_mha_rotary_with_causal_mask() {
        let device = test_device();
        let rope = RotaryEmbedding::on_device(4, 8, device).unwrap();
        let mha = MultiheadAttention::on_device(8, 2, device)
            .unwrap()
            .rotary(rope)
            .unwrap();
        let opts = crate::tensor::test_opts();
        let x = Variable::new(Tensor::randn(&[1, 4, 8], opts).unwrap(), false);
        let mask = Tensor::ones(&[4, 4], opts).unwrap().triu(1).unwrap();
        let y = mha.forward_ext(&x, &x, &x, Some(&mask)).unwrap();
        assert_eq!(y.shape(), vec![1, 4, 8]);
    }
}
