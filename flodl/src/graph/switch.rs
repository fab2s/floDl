use std::collections::HashMap;
use std::rc::Rc;

use crate::autograd::Variable;
use crate::nn::Module;
use crate::tensor::Result;

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

    let idx = route_out.data().item()? as usize;
    if idx >= branches.len() {
        return Err(crate::tensor::TensorError::new(&format!(
            "switch: router selected branch {} but only {} branches exist",
            idx,
            branches.len()
        )));
    }

    branches[idx].forward(stream)
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
