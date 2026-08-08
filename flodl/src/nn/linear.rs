use crate::autograd::{self, Variable};
use crate::tensor::{Device, Result};

use super::Module;
use super::init;
use super::parameter::Parameter;

/// Fully connected layer: `y = x @ W^T + b`.
///
/// Weight shape: `[out_features, in_features]`.
/// Bias shape: `[out_features]` (optional).
///
/// Input shape: `[batch, in_features]`.
/// Output shape: `[batch, out_features]`.
///
/// ```ignore
/// let layer = Linear::new(4, 2)?;
/// let x = Variable::new(Tensor::randn(&[8, 4], opts)?, false);
/// let y = layer.forward(&x)?;
/// assert_eq!(y.shape(), vec![8, 2]);
/// ```
pub struct Linear {
    pub weight: Parameter,
    pub bias: Option<Parameter>,
}

impl Linear {
    /// Create a linear layer on CPU with bias.
    pub fn new(in_features: i64, out_features: i64) -> Result<Self> {
        Self::on_device(in_features, out_features, Device::CPU)
    }

    /// Create a linear layer on a specific device with bias.
    pub fn on_device(in_features: i64, out_features: i64, device: Device) -> Result<Self> {
        let w = init::kaiming_uniform(
            &[out_features, in_features],
            in_features,
            5.0_f64.sqrt(),
            device,
        )?;
        let b = init::uniform_bias(in_features, &[out_features], device)?;
        Ok(Linear {
            weight: Parameter::new(w, "weight"),
            bias: Some(Parameter::new(b, "bias")),
        })
    }

    /// Create a linear layer without bias on CPU.
    pub fn no_bias(in_features: i64, out_features: i64) -> Result<Self> {
        Self::no_bias_on_device(in_features, out_features, Device::CPU)
    }

    /// Create a linear layer without bias on a specific device.
    pub fn no_bias_on_device(in_features: i64, out_features: i64, device: Device) -> Result<Self> {
        let w = init::kaiming_uniform(
            &[out_features, in_features],
            in_features,
            5.0_f64.sqrt(),
            device,
        )?;
        Ok(Linear {
            weight: Parameter::new(w, "weight"),
            bias: None,
        })
    }

    /// Build a `Linear` around an externally-owned weight `Parameter`,
    /// enabling weight tying between this layer and another module that
    /// already holds the same `Parameter`.
    ///
    /// `Parameter` is `Clone`; a clone shares the underlying `Variable`
    /// (and therefore the C++ tensor) by `Rc`. Gradients from every path
    /// that touches the shared weight accumulate on the same leaf tensor,
    /// exactly like PyTorch's `decoder.weight = embeddings.word_embeddings.weight`
    /// pattern. `Graph::named_parameters()` deduplicates by pointer
    /// identity, so a tied weight surfaces exactly once under whichever
    /// node is visited first.
    ///
    /// Pass `bias = Some(Parameter::new(...))` for the common
    /// MLM / LM-head case (BERT, RoBERTa, DistilBERT ship a fresh
    /// per-vocab decoder bias alongside the tied weight); pass `None` for
    /// GPT-2-style heads with no bias.
    ///
    /// Shape contract matches [`Linear::on_device`]: `weight.data()` must
    /// have shape `[out_features, in_features]`. No device transfer
    /// happens here — both `weight` and `bias` must already live on the
    /// device the graph runs on.
    ///
    /// ```ignore
    /// use flodl::{Embedding, Linear, Parameter, Tensor, TensorOptions};
    ///
    /// let embed = Embedding::new(vocab_size, hidden)?;
    /// let tied = embed.weight.clone();              // shared Rc
    /// let bias = Parameter::new(
    ///     Tensor::zeros(&[vocab_size], opts)?,
    ///     "bias",
    /// );
    /// let decoder = Linear::from_shared_weight(tied, Some(bias));
    /// ```
    pub fn from_shared_weight(weight: Parameter, bias: Option<Parameter>) -> Self {
        Linear { weight, bias }
    }
}

impl Module for Linear {
    fn name(&self) -> &str {
        "linear"
    }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        autograd::linear(
            input,
            &self.weight.variable,
            self.bias.as_ref().map(|b| &b.variable),
        )
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref b) = self.bias {
            params.push(b.clone());
        }
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Tensor, test_device, test_opts};

    #[test]
    fn test_linear_forward_shape() {
        let dev = test_device();
        let layer = Linear::on_device(4, 2, dev).unwrap();
        let x = Variable::new(Tensor::randn(&[8, 4], test_opts()).unwrap(), false);
        let y = layer.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![8, 2]);
    }

    #[test]
    fn test_linear_parameters_with_bias() {
        let layer = Linear::on_device(4, 2, test_device()).unwrap();
        let params = layer.parameters();
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].variable.shape(), vec![2, 4]); // weight
        assert_eq!(params[1].variable.shape(), vec![2]); // bias
    }

    #[test]
    fn test_linear_no_bias() {
        let layer = Linear::no_bias_on_device(4, 2, test_device()).unwrap();
        let params = layer.parameters();
        assert_eq!(params.len(), 1);
        assert!(layer.bias.is_none());

        let x = Variable::new(Tensor::randn(&[3, 4], test_opts()).unwrap(), false);
        let y = layer.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![3, 2]);
    }

    #[test]
    fn test_linear_gradient_flow() {
        let dev = test_device();
        let layer = Linear::on_device(3, 2, dev).unwrap();
        let x = Variable::new(Tensor::randn(&[4, 3], test_opts()).unwrap(), false);
        let y = layer.forward(&x).unwrap();
        let loss = y.sum().unwrap();
        loss.backward().unwrap();

        let params = layer.parameters();
        assert!(
            params[0].variable.grad().is_some(),
            "weight should have gradient"
        );
        assert!(
            params[1].variable.grad().is_some(),
            "bias should have gradient"
        );
    }

    #[test]
    fn test_linear_on_device() {
        let dev = test_device();
        let layer = Linear::on_device(4, 2, dev).unwrap();
        assert_eq!(layer.weight.variable.device(), dev);
        if let Some(ref b) = layer.bias {
            assert_eq!(b.variable.device(), dev);
        }
    }

    #[test]
    fn test_linear_name() {
        let layer = Linear::new(4, 2).unwrap();
        assert_eq!(layer.name(), "linear");
    }

    /// Two `Linear`s built from the same `Parameter` share the underlying
    /// `Variable` (pointer identity) and therefore the same C++ leaf
    /// tensor. Backward through both paths must accumulate onto that one
    /// tensor, and the accumulated gradient must be visible from either
    /// handle.
    #[test]
    fn test_from_shared_weight_shares_rc_and_gradient() {
        use std::rc::Rc;

        // Shared weight: [out=2, in=3].
        let shared = Parameter::new(Tensor::randn(&[2, 3], test_opts()).unwrap(), "weight");

        let layer_a = Linear::from_shared_weight(shared.clone(), None);
        let layer_b = Linear::from_shared_weight(shared.clone(), None);

        // Rc pointer identity across both layers and the original handle.
        assert!(Rc::ptr_eq(
            &layer_a.weight.variable.inner,
            &layer_b.weight.variable.inner
        ));
        assert!(Rc::ptr_eq(
            &shared.variable.inner,
            &layer_a.weight.variable.inner
        ));

        // Two distinct inputs, two forward paths, single scalar loss.
        let x1 = Variable::new(Tensor::randn(&[4, 3], test_opts()).unwrap(), false);
        let x2 = Variable::new(Tensor::randn(&[5, 3], test_opts()).unwrap(), false);
        let y1 = layer_a.forward(&x1).unwrap();
        let y2 = layer_b.forward(&x2).unwrap();
        let loss = y1.sum().unwrap().add(&y2.sum().unwrap()).unwrap();
        loss.backward().unwrap();

        // The one leaf accumulates gradient from both paths; either
        // handle sees it.
        let g_a = layer_a
            .weight
            .variable
            .grad()
            .expect("layer_a sees gradient");
        let g_b = layer_b
            .weight
            .variable
            .grad()
            .expect("layer_b sees gradient");
        assert_eq!(g_a.shape(), vec![2, 3]);
        assert_eq!(g_b.shape(), vec![2, 3]);

        // Gradient value should equal sum over batch rows of each input
        // (since d/dW sum(x @ W^T) = sum_rows(x) broadcast to out axis).
        let expected = {
            let s1 = x1.data().sum_dim(0, false).unwrap(); // [3]
            let s2 = x2.data().sum_dim(0, false).unwrap(); // [3]
            let row = s1.add(&s2).unwrap(); // [3]
            // Broadcast to [2, 3] by stacking the same row twice.
            row.unsqueeze(0)
                .unwrap()
                .expand(&[2, 3])
                .unwrap()
                .contiguous()
                .unwrap()
        };
        let got = g_a.to_f32_vec().unwrap();
        let want = expected.to_f32_vec().unwrap();
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-4, "grad mismatch: got {g}, want {w}");
        }
    }

    /// `Graph::named_parameters()` deduplicates shared parameters by
    /// `Rc::as_ptr` identity (flodl/src/graph/graph.rs). This test pins
    /// the weight-tying contract: a shared decoder/embedding weight
    /// surfaces exactly once, under the first-visited tag.
    #[test]
    fn test_from_shared_weight_dedups_in_graph_named_parameters() {
        use crate::graph::FlowBuilder;

        let shared = Parameter::new(Tensor::randn(&[3, 4], test_opts()).unwrap(), "weight");
        let layer_a = Linear::from_shared_weight(shared.clone(), None);
        let layer_b = Linear::from_shared_weight(shared.clone(), None);

        let graph = FlowBuilder::from(layer_a)
            .tag("embeddings")
            .through(layer_b)
            .tag("decoder")
            .build()
            .unwrap();

        let named = graph.named_parameters();
        assert_eq!(named.len(), 1, "shared weight should be listed once");
        assert_eq!(named[0].0, "embeddings/weight", "first-visited tag wins");
    }
}
