use crate::autograd::Variable;
use crate::tensor::{Device, Result, Tensor};

/// A named learnable parameter wrapping a Variable with requires_grad=true.
#[derive(Clone)]
pub struct Parameter {
    pub variable: Variable,
    pub name: String,
}

impl Parameter {
    /// Create a named parameter from a tensor (always requires_grad=true).
    pub fn new(data: Tensor, name: &str) -> Self {
        Parameter {
            variable: Variable::new(data, true),
            name: name.to_string(),
        }
    }

    /// Move this parameter to a different device. Returns a new Parameter.
    /// Preserves the current requires_grad state (frozen params stay frozen).
    pub fn to_device(&self, device: Device) -> Result<Parameter> {
        let rg = self.variable.requires_grad();
        let moved = self.variable.data().to_device(device)?;
        Ok(Parameter {
            variable: Variable::new(moved, rg),
            name: self.name.clone(),
        })
    }

    /// Freeze this parameter: disable gradient tracking.
    /// The optimizer will skip frozen parameters (they have no grad).
    pub fn freeze(&self) -> Result<()> {
        self.variable.set_requires_grad(false)
    }

    /// Unfreeze this parameter: re-enable gradient tracking.
    pub fn unfreeze(&self) -> Result<()> {
        self.variable.set_requires_grad(true)
    }

    /// Whether this parameter is frozen (not tracking gradients).
    pub fn is_frozen(&self) -> bool {
        !self.variable.requires_grad()
    }
}

impl std::fmt::Debug for Parameter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Parameter({}, {:?})", self.name, self.variable)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::TensorOptions;

    fn make_param(name: &str) -> Parameter {
        let t = Tensor::randn(
            &[3, 2],
            TensorOptions {
                dtype: crate::tensor::DType::Float32,
                device: crate::tensor::test_device(),
            },
        )
        .unwrap();
        Parameter::new(t, name)
    }

    #[test]
    fn test_freeze_no_gradient() {
        let p = make_param("w");
        assert!(!p.is_frozen());
        assert!(p.variable.requires_grad());

        p.freeze().unwrap();
        assert!(p.is_frozen());
        assert!(!p.variable.requires_grad());

        // Verify no gradient accumulates on a frozen param
        // Use a simple sum so we don't need matmul
        let y = p.variable.data().sum().unwrap();
        let _ = y.backward();
        assert!(
            p.variable.grad().is_none(),
            "frozen param should have no gradient"
        );
    }

    #[test]
    fn test_unfreeze_gradient_reappears() {
        let p = make_param("w");
        p.freeze().unwrap();
        assert!(p.is_frozen());

        p.unfreeze().unwrap();
        assert!(!p.is_frozen());
        assert!(p.variable.requires_grad());

        // Forward + backward: unfrozen param should accumulate gradient
        let x = Variable::new(
            Tensor::from_f32(&[1.0, 2.0, 3.0], &[1, 3], crate::tensor::test_device()).unwrap(),
            false,
        );
        let y = x.matmul(&p.variable).unwrap();
        let loss = y.sum().unwrap();
        loss.backward().unwrap();
        assert!(
            p.variable.grad().is_some(),
            "unfrozen param should have gradient"
        );
    }

    #[test]
    fn test_optimizer_step_with_frozen_params() {
        use crate::nn::optim::{Adam, Optimizer};

        let p1 = make_param("w1");
        let p2 = make_param("w2");
        p1.freeze().unwrap();

        let mut opt = Adam::new(&[p1.clone(), p2.clone()], 0.01);

        let x = Variable::new(
            Tensor::from_f32(&[1.0, 2.0, 3.0], &[1, 3], crate::tensor::test_device()).unwrap(),
            false,
        );
        // Only p2 participates in grad-tracked computation
        let y = x.matmul(&p2.variable).unwrap();
        let loss = y.sum().unwrap();
        loss.backward().unwrap();

        let p1_before = p1.variable.data().to_f32_vec().unwrap();
        opt.step().unwrap();
        let p1_after = p1.variable.data().to_f32_vec().unwrap();
        // Frozen param unchanged
        assert_eq!(p1_before, p1_after, "frozen param should not change");
    }
}
