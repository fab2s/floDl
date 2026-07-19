use std::rc::Rc;

use crate::autograd::Variable;
use crate::nn::Module;

use super::node::*;
use super::FlowBuilder;

/// Builder for Map constructs. Created by [`FlowBuilder::map`].
pub struct MapBuilder {
    fb: FlowBuilder,
    body: Box<dyn Module>,
    batched: bool,
}

impl MapBuilder {
    pub(super) fn new(fb: FlowBuilder, body: Box<dyn Module>) -> Self {
        MapBuilder {
            fb,
            body,
            batched: false,
        }
    }

    /// Fast path: pass full tensor to body in one call instead of element-wise.
    pub fn batched(mut self) -> Self {
        self.batched = true;
        self
    }

    /// Apply body to each element along dim 0 of the current stream.
    pub fn each(self) -> FlowBuilder {
        let mut fb = self.fb;
        if fb.err.is_some() || fb.current.len() != 1 {
            return Self::err_if_bad(fb);
        }
        let source = fb.current[0].clone();
        wire_map(&mut fb, self.body, source, self.batched);
        fb
    }

    /// Iterate over a tagged tensor (backward ref) instead of current stream.
    pub fn over(self, tag: &str) -> FlowBuilder {
        let mut fb = self.fb;
        if fb.err.is_some() || fb.current.len() != 1 {
            return Self::err_if_bad(fb);
        }
        let tap = fb.taps.get(tag).cloned();
        match tap {
            Some(source) => {
                wire_map(&mut fb, self.body, source, self.batched);
                fb
            }
            None => {
                fb.fail(format!(
                    "Map.over({:?}) references unknown tag; Map requires a backward reference",
                    tag
                ));
                fb
            }
        }
    }

    /// Decompose last dim into n slices, map body over each, recompose.
    ///
    /// For input `[B, D]` with `slices(n)`: reshape `[B, D]` → `[B*n, D/n]`,
    /// map body over `B*n` elements, reshape back to `[B, outD*n]`.
    /// `D` must be divisible by `n`.
    pub fn slices(self, n: i64) -> FlowBuilder {
        let mut fb = self.fb;
        if fb.err.is_some() || fb.current.len() != 1 {
            return Self::err_if_bad(fb);
        }
        if n < 1 {
            fb.fail(format!("Map.slices requires n >= 1 (got {})", n));
            return fb;
        }

        let cur = fb.current[0].clone();
        let body: Rc<dyn Module> = Rc::from(self.body);
        let composite: Rc<dyn Module> = body.clone();
        let batched = self.batched;
        let run = make_slices_func(body, n, batched);
        let id = fb.next_id("map");

        fb.nodes.insert(
            id.clone(),
            Node {
                id: id.clone(),
                input_ports: vec![DEFAULT_INPUT.into()],
                output_ports: vec![DEFAULT_OUTPUT.into()],
                run,
                module: Some(composite),
                ref_forward: None,
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

    fn err_if_bad(mut fb: FlowBuilder) -> FlowBuilder {
        if fb.err.is_none() {
            fb.fail("map requires single stream");
        }
        fb
    }
}

fn wire_map(fb: &mut FlowBuilder, body: Box<dyn Module>, source: NodeRef, batched: bool) {
    let body: Rc<dyn Module> = Rc::from(body);
    let composite: Rc<dyn Module> = body.clone();
    let run = make_map_each_func(body, batched);
    let id = fb.next_id("map");

    fb.nodes.insert(
        id.clone(),
        Node {
            id: id.clone(),
            input_ports: vec![DEFAULT_INPUT.into()],
            output_ports: vec![DEFAULT_OUTPUT.into()],
            run,
            module: Some(composite),
            ref_forward: None,
            trace_buf: None,
            named_trace_buf: None,
            loop_ports: None,
        },
    );

    fb.edges.push(Edge {
        from_node: source.node_id,
        from_port: source.port,
        to_node: id.clone(),
        to_port: DEFAULT_INPUT.into(),
    });

    let node_ref = NodeRef {
        node_id: id,
        port: DEFAULT_OUTPUT.into(),
    };
    fb.current = vec![node_ref.clone()];
    fb.on_target = Some(node_ref);
}

fn make_map_each_func(body: Rc<dyn Module>, batched: bool) -> NodeFn {
    Box::new(move |inputs: &[Variable]| {
        let stream = &inputs[0];

        if batched {
            let output = body.forward(stream)?;
            return Ok(vec![output]);
        }

        // Element-wise: narrow along dim 0, apply body, cat results
        let n = stream.shape()[0];
        if n == 0 {
            return Ok(vec![stream.clone()]);
        }

        let first = body.forward(&stream.narrow(0, 0, 1)?)?;
        let mut result = first;

        for i in 1..n {
            let element = stream.narrow(0, i, 1)?;
            let out = body.forward(&element)?;
            result = result.cat(&out, 0)?;
        }

        Ok(vec![result])
    })
}

fn make_slices_func(body: Rc<dyn Module>, n: i64, batched: bool) -> NodeFn {
    Box::new(move |inputs: &[Variable]| {
        let source = &inputs[0];
        let shape = source.shape();
        if shape.len() < 2 {
            return Err(crate::tensor::TensorError::new(&format!(
                "map slices: input must be at least 2D (got {}D)",
                shape.len()
            )));
        }

        let last_dim = *shape.last().unwrap();
        if last_dim % n != 0 {
            return Err(crate::tensor::TensorError::new(&format!(
                "map slices: last dim {} not divisible by {}",
                last_dim, n
            )));
        }
        let slice_dim = last_dim / n;
        let orig_dim0 = shape[0];

        // Decompose: [B, D] → [B*n, D/n]
        let decomposed = source.reshape(&[orig_dim0 * n, slice_dim])?;

        if batched {
            // Fast path: pass entire decomposed batch to body
            let result = body.forward(&decomposed)?;
            let result_shape = result.shape();
            let out_features = result_shape[1] * n;
            let recomposed = result.reshape(&[orig_dim0, out_features])?;
            return Ok(vec![recomposed]);
        }

        // Element-wise over decomposed rows
        let num_rows = orig_dim0 * n;
        let first = body.forward(&decomposed.narrow(0, 0, 1)?)?;
        let mut stacked = first;

        for i in 1..num_rows {
            let elem = decomposed.narrow(0, i, 1)?;
            let out = body.forward(&elem)?;
            stacked = stacked.cat(&out, 0)?;
        }

        // Recompose: [B*n, outD] → [B, outD*n]
        let stacked_shape = stacked.shape();
        let out_features = stacked_shape[1] * n;
        let recomposed = stacked.reshape(&[orig_dim0, out_features])?;
        Ok(vec![recomposed])
    })
}
