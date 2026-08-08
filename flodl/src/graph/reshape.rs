use crate::autograd::Variable;
use crate::nn::Module;
use crate::tensor::Result;

/// Zero-parameter module that reshapes its input to a fixed shape.
///
/// ```ignore
/// FlowBuilder::from(encoder)
///     .through(Reshape::new(&[4, 2]))  // [1, 8] → [4, 2]
///     .map(head).each()
///     .through(Reshape::new(&[1, 8]))  // [4, 2] → [1, 8]
///     .build()
/// ```
pub struct Reshape {
    shape: Vec<i64>,
}

impl Reshape {
    /// Create a reshape module targeting the given shape.
    /// The total number of elements (numel) must match the input tensor.
    pub fn new(shape: &[i64]) -> Self {
        Reshape {
            shape: shape.to_vec(),
        }
    }
}

impl Module for Reshape {
    fn name(&self) -> &str {
        "reshape"
    }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        input.reshape(&self.shape)
    }
}
