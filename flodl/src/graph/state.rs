use std::collections::HashMap;

use crate::autograd::Variable;
use crate::nn::{Module, NamedInputModule};
use crate::tensor::Result;

/// Additive state cell for forward references (Using before Tag).
///
/// On the first forward call, the graph auto-zeros the state, so
/// stream + zeros = stream (pass-through). On subsequent calls,
/// the accumulated state is added to the current stream.
///
/// ```ignore
/// FlowBuilder::from(embed)
///     .through(StateAdd)
///     .using(&["memory"])
///     .tag("memory")
///     .build()
/// ```
pub struct StateAdd;

impl Module for StateAdd {
    fn name(&self) -> &str {
        "state_add"
    }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        Ok(input.clone())
    }

    fn as_named_input(&self) -> Option<&dyn crate::nn::NamedInputModule> {
        Some(self)
    }
}

impl NamedInputModule for StateAdd {
    fn forward_named(
        &self,
        input: &Variable,
        refs: &HashMap<String, Variable>,
    ) -> Result<Variable> {
        super::node::sum_named_refs(input, refs)
    }
}
