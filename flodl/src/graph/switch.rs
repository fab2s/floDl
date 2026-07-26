use std::collections::HashMap;
use std::rc::Rc;

use crate::autograd::Variable;
use crate::nn::Module;
use crate::tensor::{Result, Tensor, TensorError};

use super::node::*;
use super::FlowBuilder;

/// Wire a Switch node into the flow.
pub(super) fn wire_switch(
    mut fb: FlowBuilder,
    router: Box<dyn Module>,
    branches: Vec<Box<dyn Module>>,
) -> FlowBuilder {
    if fb.err.is_some() {
        return fb;
    }
    if fb.current.len() != 1 {
        fb.fail("switch requires single stream");
        return fb;
    }
    if branches.len() < 2 {
        fb.fail("switch requires at least 2 branches");
        return fb;
    }

    let cur = fb.current[0].clone();
    let id = fb.next_id("switch");

    let router: Rc<dyn Module> = Rc::from(router);
    let branch_modules: Vec<Rc<dyn Module>> = branches
        .into_iter()
        .map(|b| Rc::from(b) as Rc<dyn Module>)
        .collect();

    let composite: Rc<dyn Module> = Rc::new(SwitchComposite {
        router: router.clone(),
        branches: branch_modules.clone(),
    });

    let run = make_switch_func(router.clone(), branch_modules.clone());

    // Only enable ref_forward if the router actually implements NamedInputModule
    let ref_forward = if router.as_named_input().is_some() {
        Some(make_switch_ref_forward(router, branch_modules))
    } else {
        None
    };

    fb.nodes.insert(
        id.clone(),
        Node {
            id: id.clone(),
            input_ports: vec![DEFAULT_INPUT.into()],
            output_ports: vec![DEFAULT_OUTPUT.into()],
            run,
            module: Some(composite),
            ref_forward,
            trace_buf: None,
            named_trace_buf: None,
            loop_ports: None,
        },
    );

    fb.edges.push(Edge {
        from_node: cur.node_id,
        from_port: cur.port,
        to_node: id.clone(),
        to_port: DEFAULT_INPUT.into(),
    });

    let node_ref = NodeRef {
        node_id: id,
        port: DEFAULT_OUTPUT.into(),
    };
    fb.current = vec![node_ref.clone()];
    fb.on_target = Some(node_ref);
    fb
}

/// Validate one raw router output value as a 0-based branch index.
///
/// `row` names the offending sample when the router routes per-sample, so a
/// bad index points at the row that produced it instead of the whole batch.
fn branch_index(value: f64, n_branches: usize, row: Option<usize>) -> Result<usize> {
    let at = match row {
        Some(r) => format!(" for row {r}"),
        None => String::new(),
    };
    if !value.is_finite() || value.fract() != 0.0 || value < 0.0 {
        return Err(TensorError::new(&format!(
            "switch: router produced {value}{at}, which is not a whole 0-based \
             branch index"
        )));
    }
    let idx = value as usize;
    if idx >= n_branches {
        return Err(TensorError::new(&format!(
            "switch: router selected branch {}{} but only {} branches exist",
            idx,
            at,
            n_branches
        )));
    }
    Ok(idx)
}

/// Bucket row ids by the branch each row selected, preserving row order.
fn group_rows(idx: &[f64], n_branches: usize) -> Result<Vec<Vec<i64>>> {
    let mut groups = vec![Vec::new(); n_branches];
    for (row, &value) in idx.iter().enumerate() {
        groups[branch_index(value, n_branches, Some(row))?].push(row as i64);
    }
    Ok(groups)
}

/// The single branch every row picked, if the routing is unanimous.
fn unanimous(groups: &[Vec<i64>]) -> Option<usize> {
    let mut hit = None;
    for (b, rows) in groups.iter().enumerate() {
        if rows.is_empty() {
            continue;
        }
        if hit.is_some() {
            return None;
        }
        hit = Some(b);
    }
    hit
}

/// Route a stream through the branch(es) its router selected.
///
/// A selector may emit either a scalar index — one branch for the whole
/// stream — or one index per row of dim 0, which dispatches each sample to
/// its own branch. Per-sample dispatch gathers each branch's rows, runs only
/// the branches that actually received rows, then restores the original row
/// order. Unselected branches never run: that is the compute saving.
fn switch_route(
    router: &Rc<dyn Module>,
    stream: &Variable,
    refs: &HashMap<String, Variable>,
    branches: &[Rc<dyn Module>],
) -> Result<Variable> {
    let route_out = if !refs.is_empty() {
        if let Some(named) = router.as_named_input() {
            named.forward_named(stream, refs)?
        } else {
            router.forward(stream)?
        }
    } else {
        router.forward(stream)?
    };

    let route_data = route_out.data();
    if route_data.numel() == 0 {
        return Err(TensorError::new(&format!(
            "switch: router emitted no branch index (stream shape {:?})",
            stream.shape()
        )));
    }
    // to_f64_vec casts on device, so f32 selectors and Int64 argmax outputs
    // both arrive as exact whole numbers rather than reinterpreted bytes.
    let idx = route_data.to_f64_vec()?;

    // Scalar index: the batch is the unit of decision. Cheapest path — no
    // gather, no reassembly, and the only shape a 1-D (unbatched) stream can
    // produce.
    if idx.len() == 1 {
        return branches[branch_index(idx[0], branches.len(), None)?].forward(stream);
    }

    let shape = stream.shape();
    let rows = shape.first().copied().unwrap_or(0);
    if idx.len() as i64 != rows {
        return Err(TensorError::new(&format!(
            "switch: router emitted {} branch indices but the stream has {} rows \
             (shape {:?}) — a selector must emit either one scalar index for the \
             whole batch or exactly one index per row",
            idx.len(),
            rows,
            shape
        )));
    }

    let groups = group_rows(&idx, branches.len())?;

    // Every row picked the same branch: equivalent to the scalar path, and
    // gathering would only copy the batch to hand back the same rows in the
    // same order.
    if let Some(b) = unanimous(&groups) {
        return branches[b].forward(stream);
    }

    let device = stream.device();
    let mut order: Vec<i64> = Vec::with_capacity(rows as usize);
    let mut outputs: Vec<Variable> = Vec::new();
    for (b, branch_rows) in groups.iter().enumerate() {
        if branch_rows.is_empty() {
            continue;
        }
        let index = Tensor::from_i64(branch_rows, &[branch_rows.len() as i64], device)?;
        let sub_batch = stream.index_select(0, &index)?;
        outputs.push(branches[b].forward(&sub_batch)?);
        order.extend_from_slice(branch_rows);
    }

    let branch_outputs: Vec<&Variable> = outputs.iter().collect();
    let grouped = Variable::cat_many(&branch_outputs, 0).map_err(|e| {
        TensorError::new(&format!(
            "switch: could not reassemble per-sample branch outputs ({e}) — every \
             branch must map a row to the same output shape"
        ))
    })?;
    if grouped.shape().first().copied().unwrap_or(0) != rows {
        return Err(TensorError::new(&format!(
            "switch: branches returned {} rows for a {}-row stream — per-sample \
             routing needs one output row per input row",
            grouped.shape().first().copied().unwrap_or(0),
            rows
        )));
    }

    // Branch outputs are stacked in branch order; invert the gather order to
    // put every row back where it came from. index_select carries gradients,
    // so the reorder stays inside the autograd graph.
    let mut inverse = vec![0i64; rows as usize];
    for (position, &row) in order.iter().enumerate() {
        inverse[row as usize] = position as i64;
    }
    grouped.index_select(0, &Tensor::from_i64(&inverse, &[rows], device)?)
}

fn make_switch_func(
    router: Rc<dyn Module>,
    branches: Vec<Rc<dyn Module>>,
) -> NodeFn {
    Box::new(move |inputs: &[Variable]| {
        let empty = HashMap::new();
        let output = switch_route(&router, &inputs[0], &empty, &branches)?;
        Ok(vec![output])
    })
}

fn make_switch_ref_forward(
    router: Rc<dyn Module>,
    branches: Vec<Rc<dyn Module>>,
) -> RefForwardFn {
    Rc::new(move |stream: &Variable, refs: &HashMap<String, Variable>| {
        switch_route(&router, stream, refs, &branches)
    })
}


/// Bundles router + branches for parameter collection.
struct SwitchComposite {
    router: Rc<dyn Module>,
    branches: Vec<Rc<dyn Module>>,
}

impl Module for SwitchComposite {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        self.branches[0].forward(input)
    }

    fn sub_modules(&self) -> Vec<Rc<dyn Module>> {
        let mut subs = vec![self.router.clone()];
        subs.extend(self.branches.iter().cloned());
        subs
    }

    fn move_to_device(&self, device: crate::tensor::Device) {
        self.router.move_to_device(device);
        for b in &self.branches {
            b.move_to_device(device);
        }
    }

    fn set_training(&self, training: bool) {
        self.router.set_training(training);
        for b in &self.branches {
            b.set_training(training);
        }
    }
}
