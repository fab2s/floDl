use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::autograd::Variable;
use crate::nn::Module;
use crate::tensor::Result;

pub(crate) const DEFAULT_INPUT: &str = "input";
pub(crate) const DEFAULT_OUTPUT: &str = "output";

pub(crate) type NodeFn = Box<dyn Fn(&[Variable]) -> Result<Vec<Variable>>>;

/// Closure type for modules that accept named refs via Using.
pub(crate) type RefForwardFn =
    Rc<dyn Fn(&Variable, &HashMap<String, Variable>) -> Result<Variable>>;

/// Sum Using refs into the stream, in ref-name order.
///
/// The order is load-bearing: `HashMap` iteration order is randomized per
/// process, and float addition is not associative, so summing refs in map
/// order makes the result differ run to run at the same seed. That is visible
/// in soft routing weights and can flip a discrete branch choice outright in
/// hard routing, so named-ref accumulation always walks a sorted key list.
pub(crate) fn sum_named_refs(
    input: &Variable,
    refs: &HashMap<String, Variable>,
) -> Result<Variable> {
    let mut names: Vec<&String> = refs.keys().collect();
    names.sort();
    let mut combined = input.clone();
    for name in names {
        combined = combined.add(&refs[name])?;
    }
    Ok(combined)
}

/// Named per-iteration trace buffers, keyed by emit name.
/// Populated by loop nodes whose body implements [`crate::nn::LoopBody`]
/// and publishes via [`crate::nn::TraceEmit::publish`]. Distinct from
/// the legacy single-stream [`Node::trace_buf`] (keyed by post-loop tag).
pub(crate) type NamedTraceStore = Rc<RefCell<HashMap<String, Vec<Variable>>>>;

pub(crate) struct Node {
    pub id: String,
    pub input_ports: Vec<String>,
    pub output_ports: Vec<String>,
    pub run: NodeFn,
    pub module: Option<Rc<dyn Module>>,
    /// If set, this module can handle Using refs via forward_named.
    pub ref_forward: Option<RefForwardFn>,
    /// Trace buffer for loop nodes whose body implements Module::trace().
    pub trace_buf: Option<Rc<RefCell<Vec<Variable>>>>,
    /// Named trace store for loop nodes whose body implements [`crate::nn::LoopBody`].
    /// One `Vec<Variable>` per published name, length equal to count of iterations
    /// where the body called [`crate::nn::TraceEmit::publish`] for that name.
    pub named_trace_buf: Option<NamedTraceStore>,
    /// Shared port list for loop nodes — the loop's run closure reads this
    /// at execution time to extract refs. Updated by wire_using when
    /// .using() is chained after a loop.
    pub loop_ports: Option<Rc<RefCell<Vec<String>>>>,
}

#[derive(Clone, Debug)]
pub(crate) struct NodeRef {
    pub node_id: String,
    pub port: String,
}

#[derive(Clone, Debug)]
pub(crate) struct Edge {
    pub from_node: String,
    pub from_port: String,
    pub to_node: String,
    pub to_port: String,
}

#[derive(Clone, Debug)]
pub(crate) struct ExposedPort {
    #[allow(dead_code)]
    pub name: String,
    pub node_id: String,
    pub port: String,
}

/// Build-time forward reference: Using before Tag.
/// Resolved when Tag is called, converted to StateEntry at build.
pub(crate) struct ForwardRefSpec {
    #[allow(dead_code)]
    pub name: String,
    pub reader_id: String,
    pub writer_id: String,
    pub writer_port: String,
}

/// Pending forward reference awaiting Tag resolution.
pub(crate) struct PendingUsing {
    pub reader_id: String,
}

/// Extract ref_* ports into a name → Variable map.
pub(crate) fn extract_refs(
    ports: &[String],
    inputs: &[Variable],
) -> HashMap<String, Variable> {
    let mut refs = HashMap::new();
    for (i, port) in ports.iter().enumerate() {
        if let Some(name) = port.strip_prefix("ref_")
            && i < inputs.len()
        {
            refs.insert(name.to_string(), inputs[i].clone());
        }
    }
    refs
}

/// Wrap a Module into a NodeFn (single input → single output).
pub(crate) fn wrap_module(module: Rc<dyn Module>) -> NodeFn {
    Box::new(move |inputs: &[Variable]| {
        let output = module.forward(&inputs[0])?;
        Ok(vec![output])
    })
}

/// Wrap a ref-capable module. Checks ports at call time to extract refs.
pub(crate) fn wrap_ref_module(
    module: Rc<dyn Module>,
    ref_forward: RefForwardFn,
    ports: Vec<String>,
) -> NodeFn {
    Box::new(move |inputs: &[Variable]| {
        let refs = extract_refs(&ports, inputs);
        let output = if refs.is_empty() {
            module.forward(&inputs[0])?
        } else {
            ref_forward(&inputs[0], &refs)?
        };
        Ok(vec![output])
    })
}
