use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::autograd::Variable;
use crate::nn::{Module, TraceEmit};
use crate::tensor::Result;

use super::node::*;
use super::FlowBuilder;

/// Read a loop condition's halt decision for one iteration.
///
/// The loop advances a single state tensor for the whole batch, so the halt
/// decision is batch-level and the condition must resolve to one number. A
/// per-sample condition output is rejected loudly rather than silently letting
/// row 0 decide the iteration count for every sample.
fn halt_decision(halt: &Variable, iteration: usize) -> Result<bool> {
    let data = halt.data();
    if data.numel() != 1 {
        return Err(crate::tensor::TensorError::new(&format!(
            "loop condition at iteration {}: condition returned {} values (shape \
             {:?}) but the loop halts for the whole batch — reduce the condition to \
             a single scalar (e.g. mean over the batch)",
            iteration,
            data.numel(),
            halt.shape()
        )));
    }
    let value = data.item().map_err(|e| {
        crate::tensor::TensorError::new(&format!(
            "loop condition at iteration {iteration}: {e}"
        ))
    })?;
    Ok(value > 0.0)
}

/// Builder for loop constructs. Created by [`FlowBuilder::loop_body`].
pub struct LoopBuilder {
    fb: FlowBuilder,
    body: Box<dyn Module>,
}

impl LoopBuilder {
    pub(super) fn new(fb: FlowBuilder, body: Box<dyn Module>) -> Self {
        LoopBuilder { fb, body }
    }

    /// Fixed iteration count: repeat body N times.
    pub fn for_n(self, n: usize) -> FlowBuilder {
        let mut fb = self.fb;
        if fb.err.is_some() {
            return fb;
        }
        if fb.current.len() != 1 {
            fb.fail("loop requires single stream");
            return fb;
        }
        if n < 1 {
            fb.fail("loop requires at least 1 iteration");
            return fb;
        }

        let body: Rc<dyn Module> = Rc::from(self.body);
        let trace_buf: Rc<RefCell<Vec<Variable>>> = Rc::new(RefCell::new(Vec::new()));
        let named_buf: NamedTraceStore = Rc::new(RefCell::new(HashMap::new()));
        let ports: Rc<RefCell<Vec<String>>> = Rc::new(RefCell::new(vec![DEFAULT_INPUT.into()]));
        let run = make_for_loop_func(
            body.clone(),
            n,
            trace_buf.clone(),
            named_buf.clone(),
            ports.clone(),
        );
        let composite: Rc<dyn Module> = Rc::new(LoopComposite {
            body,
            cond: None,
        });
        wire_loop(fb, composite, run, trace_buf, named_buf, ports)
    }

    /// Repeat while condition says "continue" (positive output = halt).
    /// Condition checked before each iteration — body may never run.
    pub fn while_cond(self, cond: impl Module + 'static, max_iter: usize) -> FlowBuilder {
        let mut fb = self.fb;
        if fb.err.is_some() {
            return fb;
        }
        if fb.current.len() != 1 {
            fb.fail("loop requires single stream");
            return fb;
        }
        if max_iter < 1 {
            fb.fail("loop requires max_iter >= 1");
            return fb;
        }

        let body: Rc<dyn Module> = Rc::from(self.body);
        let cond: Rc<dyn Module> = Rc::new(cond);
        let trace_buf: Rc<RefCell<Vec<Variable>>> = Rc::new(RefCell::new(Vec::new()));
        let named_buf: NamedTraceStore = Rc::new(RefCell::new(HashMap::new()));
        let ports: Rc<RefCell<Vec<String>>> = Rc::new(RefCell::new(vec![DEFAULT_INPUT.into()]));
        let run = make_while_loop_func(
            body.clone(),
            cond.clone(),
            max_iter,
            trace_buf.clone(),
            named_buf.clone(),
            ports.clone(),
        );
        let composite: Rc<dyn Module> = Rc::new(LoopComposite {
            body,
            cond: Some(cond),
        });
        wire_loop(fb, composite, run, trace_buf, named_buf, ports)
    }

    /// Repeat until condition signals halt (positive output = halt).
    /// Body always runs at least once.
    pub fn until_cond(self, cond: impl Module + 'static, max_iter: usize) -> FlowBuilder {
        let mut fb = self.fb;
        if fb.err.is_some() {
            return fb;
        }
        if fb.current.len() != 1 {
            fb.fail("loop requires single stream");
            return fb;
        }
        if max_iter < 1 {
            fb.fail("loop requires max_iter >= 1");
            return fb;
        }

        let body: Rc<dyn Module> = Rc::from(self.body);
        let cond: Rc<dyn Module> = Rc::new(cond);
        let trace_buf: Rc<RefCell<Vec<Variable>>> = Rc::new(RefCell::new(Vec::new()));
        let named_buf: NamedTraceStore = Rc::new(RefCell::new(HashMap::new()));
        let ports: Rc<RefCell<Vec<String>>> = Rc::new(RefCell::new(vec![DEFAULT_INPUT.into()]));
        let run = make_until_loop_func(
            body.clone(),
            cond.clone(),
            max_iter,
            trace_buf.clone(),
            named_buf.clone(),
            ports.clone(),
        );
        let composite: Rc<dyn Module> = Rc::new(LoopComposite {
            body,
            cond: Some(cond),
        });
        wire_loop(fb, composite, run, trace_buf, named_buf, ports)
    }
}

/// Wire a loop node into the graph and return the updated FlowBuilder.
fn wire_loop(
    mut fb: FlowBuilder,
    composite: Rc<dyn Module>,
    run: NodeFn,
    trace_buf: Rc<RefCell<Vec<Variable>>>,
    named_buf: NamedTraceStore,
    ports: Rc<RefCell<Vec<String>>>,
) -> FlowBuilder {
    let cur = fb.current[0].clone();
    let id = fb.next_id("loop");

    // If the body supports NamedInputModule, expose ref_forward on the loop node
    // so that .using() can be chained after the loop.
    let ref_forward = if composite.sub_modules().first()
        .and_then(|body| body.as_named_input())
        .is_some()
    {
        let body_rc = composite.sub_modules().into_iter().next().unwrap();
        let rf: RefForwardFn = Rc::new(move |input, refs| {
            body_rc.as_named_input().unwrap().forward_named(input, refs)
        });
        Some(rf)
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
            trace_buf: Some(trace_buf),
            named_trace_buf: Some(named_buf),
            loop_ports: Some(ports),
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

/// Execute body step: use forward_named if refs are available, plain forward otherwise.
fn body_step(
    body: &Rc<dyn Module>,
    state: &Variable,
    refs: &HashMap<String, Variable>,
) -> Result<Variable> {
    if !refs.is_empty()
        && let Some(named) = body.as_named_input()
    {
        return named.forward_named(state, refs);
    }
    body.forward(state)
}

/// Run one iteration through either the LoopBody publisher path (if the body
/// implements [`crate::nn::LoopBody`]) or the legacy `forward + trace()` path.
/// Drains per-step named emits into the shared store and pushes the legacy
/// trace if present.
fn dispatch_iteration(
    body: &Rc<dyn Module>,
    state: &Variable,
    refs: &HashMap<String, Variable>,
    trace_buf: &Rc<RefCell<Vec<Variable>>>,
    named_buf: &NamedTraceStore,
) -> Result<Variable> {
    if let Some(lb) = body.as_loop_body() {
        let mut step_named: HashMap<String, Variable> = HashMap::new();
        let mut emit = TraceEmit::new(&mut step_named);
        let next = lb.step(state, refs, &mut emit)?;
        // Drain per-step emits into the shared loop-level store
        if !step_named.is_empty() {
            let mut store = named_buf.borrow_mut();
            for (k, v) in step_named {
                store.entry(k).or_default().push(v);
            }
        }
        Ok(next)
    } else {
        let next = if refs.is_empty() {
            body.forward(state)?
        } else {
            body_step(body, state, refs)?
        };
        if let Some(t) = body.trace() {
            trace_buf.borrow_mut().push(t);
        }
        Ok(next)
    }
}

fn make_for_loop_func(
    body: Rc<dyn Module>,
    count: usize,
    trace_buf: Rc<RefCell<Vec<Variable>>>,
    named_buf: NamedTraceStore,
    ports: Rc<RefCell<Vec<String>>>,
) -> NodeFn {
    // Ports are mutated after build (by .using()), so ref detection must
    // happen at call time, not build time.
    Box::new(move |inputs: &[Variable]| {
        let mut state = inputs[0].clone();
        let refs = extract_refs(&ports.borrow(), inputs);
        trace_buf.borrow_mut().clear();
        named_buf.borrow_mut().clear();
        body.reset();
        for i in 0..count {
            state = dispatch_iteration(&body, &state, &refs, &trace_buf, &named_buf)
                .map_err(|e| {
                    crate::tensor::TensorError::new(&format!("loop iteration {}: {}", i, e))
                })?;
        }
        Ok(vec![state])
    })
}

fn make_while_loop_func(
    body: Rc<dyn Module>,
    cond: Rc<dyn Module>,
    max_iter: usize,
    trace_buf: Rc<RefCell<Vec<Variable>>>,
    named_buf: NamedTraceStore,
    ports: Rc<RefCell<Vec<String>>>,
) -> NodeFn {
    Box::new(move |inputs: &[Variable]| {
        let mut state = inputs[0].clone();
        let refs = extract_refs(&ports.borrow(), inputs);
        trace_buf.borrow_mut().clear();
        named_buf.borrow_mut().clear();
        body.reset();
        for i in 0..max_iter {
            let halt = cond.forward(&state)?;
            if halt_decision(&halt, i)? {
                break;
            }
            state = dispatch_iteration(&body, &state, &refs, &trace_buf, &named_buf)
                .map_err(|e| {
                    crate::tensor::TensorError::new(&format!("loop iteration {}: {}", i, e))
                })?;
        }
        Ok(vec![state])
    })
}

fn make_until_loop_func(
    body: Rc<dyn Module>,
    cond: Rc<dyn Module>,
    max_iter: usize,
    trace_buf: Rc<RefCell<Vec<Variable>>>,
    named_buf: NamedTraceStore,
    ports: Rc<RefCell<Vec<String>>>,
) -> NodeFn {
    Box::new(move |inputs: &[Variable]| {
        let mut state = inputs[0].clone();
        let refs = extract_refs(&ports.borrow(), inputs);
        trace_buf.borrow_mut().clear();
        named_buf.borrow_mut().clear();
        body.reset();
        for i in 0..max_iter {
            state = dispatch_iteration(&body, &state, &refs, &trace_buf, &named_buf)
                .map_err(|e| {
                    crate::tensor::TensorError::new(&format!("loop iteration {}: {}", i, e))
                })?;
            // Skip condition check on last iteration
            if i < max_iter - 1 {
                let halt = cond.forward(&state)?;
                if halt_decision(&halt, i)? {
                    break;
                }
            }
        }
        Ok(vec![state])
    })
}

/// Bundles body + optional condition for parameter collection.
struct LoopComposite {
    body: Rc<dyn Module>,
    cond: Option<Rc<dyn Module>>,
}

impl Module for LoopComposite {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        self.body.forward(input)
    }

    fn sub_modules(&self) -> Vec<Rc<dyn Module>> {
        let mut subs = vec![self.body.clone()];
        if let Some(ref cond) = self.cond {
            subs.push(cond.clone());
        }
        subs
    }

    fn move_to_device(&self, device: crate::tensor::Device) {
        self.body.move_to_device(device);
        if let Some(ref cond) = self.cond {
            cond.move_to_device(device);
        }
    }

    fn set_training(&self, training: bool) {
        self.body.set_training(training);
        if let Some(ref cond) = self.cond {
            cond.set_training(training);
        }
    }

    fn reset(&self) {
        self.body.reset();
    }

    fn detach_state(&self) {
        self.body.detach_state();
    }
}
