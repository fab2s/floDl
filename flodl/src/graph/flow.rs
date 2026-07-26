use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use indexmap::IndexMap;

use crate::autograd::Variable;
use crate::nn::Module;
use crate::tensor::TensorError;

use super::loop_node::LoopBuilder;
use super::node::*;
use super::MergeOp;

/// Fluent builder for computation graphs. Reads as data flow.
///
/// ```ignore
/// use flodl::*;
///
/// let g = FlowBuilder::from(Linear::new(4, 8)?)
///     .through(GELU)
///     .through(LayerNorm::new(8)?)
///     .also(Linear::new(8, 8)?)           // residual connection
///     .through(Linear::new(8, 2)?)
///     .build()?;
///
/// let y = g.forward(&x)?;
/// ```
///
/// Advanced patterns — split/merge, tags/using, loops:
///
/// ```ignore
/// // Parallel branches with merge
/// let g = FlowBuilder::from(encoder)
///     .split(modules![head_a, head_b])
///     .merge(MergeOp::Add)
///     .build()?;
///
/// // Cross-connections via tag/using
/// let g = FlowBuilder::from(encoder)
///     .through(layer1).tag("hidden")
///     .through(layer2)
///     .through(cross_attn).using(&["hidden"])
///     .build()?;
///
/// // Fixed-iteration loop
/// let g = FlowBuilder::from(encoder)
///     .loop_body(refine_block).for_n(3)
///     .build()?;
/// ```
///
/// ## Error handling
///
/// Builder methods accumulate errors internally — invalid operations
/// (e.g., `through` after an unmerged `split`) do not panic. The error
/// surfaces when you call [`build()`](Self::build), which validates the
/// entire graph and returns `Err` with a descriptive message.
/// This lets you chain the full graph description without interruption.
pub struct FlowBuilder {
    pub(super) nodes: IndexMap<String, Node>,
    pub(super) edges: Vec<Edge>,
    pub(super) inputs: Vec<ExposedPort>,
    pub(super) current: Vec<NodeRef>,
    pub(super) taps: HashMap<String, NodeRef>,
    pub(super) on_target: Option<NodeRef>,
    /// Side-branch output from fork(), consumed by the next tag() call.
    fork_target: Option<NodeRef>,
    pub(super) counter: usize,
    pub(super) err: Option<String>,
    /// Pending forward references: Using("x") called before Tag("x").
    pub(super) pending: HashMap<String, Vec<PendingUsing>>,
    /// Resolved forward references ready for Graph::build.
    pub(super) forward_refs: Vec<ForwardRefSpec>,
    /// Tag group name → suffixed tag names (from TagGroup).
    pub(super) tag_groups: HashMap<String, Vec<String>>,
    /// Human-readable label for dashboard display.
    label: Option<String>,
    /// Tags marked as internal (hidden from parent graph tree resolution).
    internal_tags: HashSet<String>,
    /// Print tree structure on build.
    verbose: bool,
}

impl Default for FlowBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl FlowBuilder {
    /// Start a graph flow with an implicit identity entry.
    ///
    /// Equivalent to `FlowBuilder::from(Identity)`. Useful when the first
    /// meaningful operation is a tag or fork:
    ///
    /// ```ignore
    /// let g = FlowBuilder::new()
    ///     .tag("input")
    ///     .through(encoder)
    ///     .build()?;
    /// ```
    pub fn new() -> Self {
        Self::from(crate::nn::Identity)
    }

    /// Start a new graph flow with an entry module.
    /// The module's input becomes the graph's input.
    pub fn from(module: impl Module + 'static) -> Self {
        let mut fb = FlowBuilder {
            nodes: IndexMap::new(),
            edges: Vec::new(),
            inputs: Vec::new(),
            current: Vec::new(),
            taps: HashMap::new(),
            on_target: None,
            fork_target: None,
            counter: 0,
            err: None,
            pending: HashMap::new(),
            forward_refs: Vec::new(),
            tag_groups: HashMap::new(),
            label: None,
            internal_tags: HashSet::new(),
            verbose: false,
        };

        let node_ref = fb.add_module(module);

        fb.inputs.push(ExposedPort {
            name: DEFAULT_INPUT.to_string(),
            node_id: node_ref.node_id.clone(),
            port: DEFAULT_INPUT.to_string(),
        });

        fb.on_target = Some(node_ref.clone());
        fb.current = vec![node_ref];
        fb
    }

    /// Set a human-readable label for the graph (displayed in dashboard).
    /// Does not affect the structural hash or `Module::name()`.
    pub fn label(mut self, name: &str) -> Self {
        self.label = Some(name.to_string());
        self
    }

    /// Mark a tag as internal (hidden from parent graph tree resolution).
    /// Internal tags cannot be accessed via `tagged_at()` from a parent graph.
    pub fn internal(mut self, tag: &str) -> Self {
        self.internal_tags.insert(tag.to_string());
        self
    }

    /// Enable verbose output on build: prints tree structure, tag resolution, param summary.
    pub fn verbose(mut self, enabled: bool) -> Self {
        self.verbose = enabled;
        self
    }

    /// Chain a module sequentially: `stream → module → stream`.
    /// Requires a single stream (call `merge` first if split).
    pub fn through(mut self, module: impl Module + 'static) -> Self {
        if self.err.is_some() {
            return self;
        }
        if self.current.len() != 1 {
            self.fail("through requires single stream");
            return self;
        }

        let prev = self.current[0].clone();
        let node_ref = self.add_module(module);

        self.edges.push(Edge {
            from_node: prev.node_id,
            from_port: prev.port,
            to_node: node_ref.node_id.clone(),
            to_port: DEFAULT_INPUT.into(),
        });

        self.on_target = Some(node_ref.clone());
        self.current = vec![node_ref];
        self
    }

    /// Fork: run a module on the current stream as a side branch.
    /// The module's output can be tagged, but the **main stream continues unchanged**.
    ///
    /// Use this when multiple modules consume the same tensor independently
    /// (e.g. classification heads that read the same latent).
    ///
    /// ```text
    /// .through(encoder).tag("latent")
    /// .fork(letter_head).tag("letter_logits")
    /// .fork(case_head).tag("case_logits")
    /// .through(decoder)   // decoder still sees encoder output
    /// ```
    pub fn fork(mut self, module: impl Module + 'static) -> Self {
        if self.err.is_some() {
            return self;
        }
        if self.current.len() != 1 {
            self.fail("fork requires single stream");
            return self;
        }

        let prev = self.current[0].clone();
        let node_ref = self.add_module(module);

        self.edges.push(Edge {
            from_node: prev.node_id.clone(),
            from_port: prev.port.clone(),
            to_node: node_ref.node_id.clone(),
            to_port: DEFAULT_INPUT.into(),
        });

        // on_target for .using(), fork_target for .tag()
        // current stays on the main stream
        self.on_target = Some(node_ref.clone());
        self.fork_target = Some(node_ref);
        self
    }

    /// Residual connection: `output = stream + module(stream)`.
    /// The skip connection is an element-wise add.
    pub fn also(mut self, module: impl Module + 'static) -> Self {
        if self.err.is_some() {
            return self;
        }
        if self.current.len() != 1 {
            self.fail("also requires single stream");
            return self;
        }

        let prev = self.current[0].clone();
        let module_ref = self.add_module(module);
        let add_ref = self.add_add_node(2);

        // prev → module
        self.edges.push(Edge {
            from_node: prev.node_id.clone(),
            from_port: prev.port.clone(),
            to_node: module_ref.node_id.clone(),
            to_port: DEFAULT_INPUT.into(),
        });

        // prev → add input_0 (skip connection)
        self.edges.push(Edge {
            from_node: prev.node_id,
            from_port: prev.port,
            to_node: add_ref.node_id.clone(),
            to_port: "input_0".into(),
        });

        // module → add input_1
        self.edges.push(Edge {
            from_node: module_ref.node_id.clone(),
            from_port: module_ref.port.clone(),
            to_node: add_ref.node_id.clone(),
            to_port: "input_1".into(),
        });

        self.on_target = Some(module_ref);
        self.current = vec![add_ref];
        self
    }

    /// Residual connection with a custom skip path: `output = skip(stream) + main(stream)`.
    ///
    /// Generalizes [`also`](Self::also) for cases where the skip connection
    /// needs a transform (e.g. 1x1 conv + BN for channel/stride changes in ResNet).
    ///
    /// ```text
    /// // ResNet downsample block: skip uses 1x1 conv to match dimensions
    /// .also_with(downsample_1x1, conv_bn_relu_conv_bn)
    /// .through(ReLU)
    /// ```
    pub fn also_with(
        mut self,
        skip: impl Module + 'static,
        main: impl Module + 'static,
    ) -> Self {
        if self.err.is_some() {
            return self;
        }
        if self.current.len() != 1 {
            self.fail("also_with requires single stream");
            return self;
        }

        let prev = self.current[0].clone();
        let skip_ref = self.add_module(skip);
        let main_ref = self.add_module(main);
        let add_ref = self.add_add_node(2);

        // prev -> skip
        self.edges.push(Edge {
            from_node: prev.node_id.clone(),
            from_port: prev.port.clone(),
            to_node: skip_ref.node_id.clone(),
            to_port: DEFAULT_INPUT.into(),
        });

        // prev -> main
        self.edges.push(Edge {
            from_node: prev.node_id,
            from_port: prev.port,
            to_node: main_ref.node_id.clone(),
            to_port: DEFAULT_INPUT.into(),
        });

        // skip -> add input_0
        self.edges.push(Edge {
            from_node: skip_ref.node_id,
            from_port: skip_ref.port,
            to_node: add_ref.node_id.clone(),
            to_port: "input_0".into(),
        });

        // main -> add input_1
        self.edges.push(Edge {
            from_node: main_ref.node_id.clone(),
            from_port: main_ref.port.clone(),
            to_node: add_ref.node_id.clone(),
            to_port: "input_1".into(),
        });

        self.on_target = Some(main_ref);
        self.current = vec![add_ref];
        self
    }

    /// Name the current stream position for later reference via [`using`](Self::using).
    ///
    /// Tags also serve as keys for observation (`collect`/`trend`), checkpoint
    /// serialization (`named_parameters`), and profiling (`timing`).
    pub fn tag(mut self, name: &str) -> Self {
        if self.err.is_some() {
            return self;
        }
        if self.current.len() != 1 {
            self.fail("tag requires single stream");
            return self;
        }
        if self.taps.contains_key(name) {
            self.fail(format!("duplicate tag: {}", name));
            return self;
        }

        // Fork target takes priority (side-branch output), otherwise main stream.
        let cur = self.fork_target.take().unwrap_or_else(|| self.current[0].clone());
        self.taps.insert(name.to_string(), cur.clone());

        // Resolve any pending forward references to this tag
        if let Some(pending_list) = self.pending.remove(name) {
            for p in pending_list {
                self.forward_refs.push(ForwardRefSpec {
                    name: name.to_string(),
                    reader_id: p.reader_id,
                    writer_id: cur.node_id.clone(),
                    writer_port: cur.port.clone(),
                });
            }
        }

        self
    }

    /// Tag each stream in a multi-stream flow with auto-suffixed names.
    /// Given group name "head" and 3 streams, creates "head_0", "head_1", "head_2".
    /// The group is registered for expansion in observation queries.
    pub fn tag_group(mut self, name: &str) -> Self {
        if self.err.is_some() {
            return self;
        }
        if self.current.len() < 2 {
            self.fail(format!(
                "tag_group({:?}) requires multiple streams (got {}); use tag for single-stream",
                name,
                self.current.len()
            ));
            return self;
        }
        if self.tag_groups.contains_key(name) {
            self.fail(format!("duplicate tag group: {:?}", name));
            return self;
        }
        if self.taps.contains_key(name) {
            self.fail(format!(
                "tag group {:?} conflicts with existing tag",
                name
            ));
            return self;
        }

        let mut suffixed = Vec::with_capacity(self.current.len());
        for (i, cur) in self.current.iter().enumerate() {
            let tag = format!("{}_{}", name, i);
            if self.taps.contains_key(&tag) {
                self.fail(format!(
                    "tag_group({:?}): suffixed name {:?} conflicts with existing tag",
                    name, tag
                ));
                return self;
            }
            self.taps.insert(tag.clone(), cur.clone());
            suffixed.push(tag.clone());

            // Resolve pending forward refs to this suffixed tag
            if let Some(pending_list) = self.pending.remove(&tag) {
                for p in pending_list {
                    self.forward_refs.push(ForwardRefSpec {
                        name: tag.clone(),
                        reader_id: p.reader_id,
                        writer_id: cur.node_id.clone(),
                        writer_port: cur.port.clone(),
                    });
                }
            }
        }

        self.tag_groups.insert(name.to_string(), suffixed);
        self
    }

    /// Add named auxiliary inputs to the graph.
    /// Creates passthrough nodes for each name, tagged and exposed as graph inputs.
    /// Forward receives inputs in declaration order: From entry first, then each Input.
    pub fn input(mut self, names: &[&str]) -> Self {
        if self.err.is_some() {
            return self;
        }
        for &name in names {
            if self.taps.contains_key(name) {
                self.fail(format!(
                    "input tag {:?} conflicts with existing tag",
                    name
                ));
                return self;
            }

            let node_ref = self.add_input_node(name);
            self.taps.insert(name.to_string(), node_ref.clone());
            self.inputs.push(ExposedPort {
                name: name.to_string(),
                node_id: node_ref.node_id.clone(),
                port: DEFAULT_INPUT.to_string(),
            });

            // Resolve pending forward refs to this tag
            if let Some(pending_list) = self.pending.remove(name) {
                for p in pending_list {
                    self.forward_refs.push(ForwardRefSpec {
                        name: name.to_string(),
                        reader_id: p.reader_id,
                        writer_id: node_ref.node_id.clone(),
                        writer_port: node_ref.port.clone(),
                    });
                }
            }
        }
        self
    }

    /// Wire tagged outputs as extra inputs to the preceding module.
    /// The target module must implement `NamedInputModule` (override `as_named_input`).
    /// Tags may be backward refs (already tagged) or forward refs (tagged later —
    /// resolved automatically via state buffers).
    ///
    /// ```text
    /// .through(cross_attn).using(&["memory", "context"])
    /// ```
    pub fn using(mut self, refs: &[&str]) -> Self {
        if self.err.is_some() {
            return self;
        }
        if refs.is_empty() {
            return self;
        }

        // Determine target(s)
        let targets = if let Some(ref target) = self.on_target {
            vec![target.clone()]
        } else if self.current.len() > 1 {
            self.current.clone()
        } else {
            self.fail(
                "using requires a preceding through, split, or merge",
            );
            return self;
        };

        for target in &targets {
            if let Err(e) = self.wire_using(target, refs) {
                self.fail(e);
                return self;
            }
        }
        self
    }

    /// Fork the stream into parallel branches, one module per branch.
    /// All branches receive the same input. Follow with `merge` to recombine.
    ///
    /// ```text
    /// .split(modules![Linear::new(H, H)?, Linear::new(H, H)?])
    /// .merge(MergeOp::Add)
    /// ```
    pub fn split(mut self, modules: Vec<Box<dyn Module>>) -> Self {
        if self.err.is_some() {
            return self;
        }
        if self.current.len() != 1 {
            self.fail("split requires single stream");
            return self;
        }
        if modules.len() < 2 {
            self.fail("split requires at least 2 branches");
            return self;
        }

        let prev = self.current[0].clone();
        let mut branches = Vec::new();

        for module in modules {
            let node_ref = self.add_boxed_module(module);
            self.edges.push(Edge {
                from_node: prev.node_id.clone(),
                from_port: prev.port.clone(),
                to_node: node_ref.node_id.clone(),
                to_port: DEFAULT_INPUT.into(),
            });
            branches.push(node_ref);
        }

        self.on_target = None;
        self.current = branches;
        self
    }

    /// Recombine split branches using a merge operation (`Add` or `Mean`).
    pub fn merge(mut self, op: MergeOp) -> Self {
        if self.err.is_some() {
            return self;
        }
        if self.current.len() < 2 {
            self.fail("merge requires multiple streams (after split)");
            return self;
        }

        let branches = self.current.clone();
        let n = branches.len();
        let merge_ref = self.add_merge_node(n, op);

        for (i, branch) in branches.iter().enumerate() {
            self.edges.push(Edge {
                from_node: branch.node_id.clone(),
                from_port: branch.port.clone(),
                to_node: merge_ref.node_id.clone(),
                to_port: format!("input_{}", i),
            });
        }

        self.on_target = Some(merge_ref.clone());
        self.current = vec![merge_ref];
        self
    }

    /// Start a loop construct. Chain with `.for_n(n)`, `.while_cond(cond, max)`,
    /// or `.until_cond(cond, max)` to finalize.
    pub fn loop_body(self, body: impl Module + 'static) -> LoopBuilder {
        LoopBuilder::new(self, Box::new(body))
    }

    /// Hard routing: only the selected branch(es) run, the rest are skipped.
    ///
    /// The router emits 0-based branch indices — either a single scalar index
    /// for the whole stream, or one index per row of dim 0, which dispatches
    /// each sample to its own branch and runs only the branches that received
    /// rows.
    ///
    /// Under per-sample routing a branch sees only its own rows, so any
    /// batch-dependent layer inside it (BatchNorm especially) computes
    /// statistics over that sub-batch, and a branch that received no rows does
    /// not run — no buffer update, no gradient for it that step.
    pub fn switch(
        self,
        router: impl Module + 'static,
        branches: Vec<Box<dyn Module>>,
    ) -> Self {
        super::switch::wire_switch(self, Box::new(router), branches)
    }

    /// Soft routing (mixture of experts): all experts execute, outputs
    /// combined via learned router weights. Router must output shape
    /// `[..., n_experts]`.
    pub fn gate(
        self,
        router: impl Module + 'static,
        experts: Vec<Box<dyn Module>>,
    ) -> Self {
        super::gate::wire_gate(self, Box::new(router), experts)
    }

    /// Start a Map construct. Chain with `.each()` or `.batched().each()`.
    pub fn map(self, body: impl Module + 'static) -> super::map::MapBuilder {
        super::map::MapBuilder::new(self, Box::new(body))
    }

    /// Finalize the flow into an executable [`Graph`](super::Graph).
    ///
    /// Validates that all streams are merged, all forward refs are resolved,
    /// and computes the topological execution order. Returns an error if the
    /// graph is malformed.
    pub fn build(self) -> crate::tensor::Result<super::Graph> {
        if let Some(err) = self.err {
            return Err(TensorError::new(&err));
        }
        // Check for unresolved forward refs
        if !self.pending.is_empty() {
            let names: Vec<&String> = self.pending.keys().collect();
            return Err(TensorError::new(&format!(
                "unresolved forward refs: {:?}",
                names
            )));
        }
        if self.current.len() != 1 {
            return Err(TensorError::new(
                "open streams: call merge before build",
            ));
        }

        let output = ExposedPort {
            name: DEFAULT_OUTPUT.to_string(),
            node_id: self.current[0].node_id.clone(),
            port: self.current[0].port.clone(),
        };

        super::Graph::build(
            self.nodes,
            self.edges,
            self.inputs,
            vec![output],
            self.taps,
            self.forward_refs,
            self.tag_groups,
            self.label,
            self.internal_tags,
            self.verbose,
        )
    }

    // --- Internal helpers ---

    pub(super) fn next_id(&mut self, prefix: &str) -> String {
        self.counter += 1;
        format!("{}_{}", prefix, self.counter)
    }

    /// Record the first deferred builder error, stamped with the chain
    /// position it occurred at.
    ///
    /// FlowBuilder defers the first error to [`build()`](Self::build); a
    /// bare message ("through requires single stream") gives no clue which
    /// of many chained calls failed. This appends the most recently built
    /// node, whose id encodes both the module type and its global sequence
    /// number (e.g. `Conv2d_2`), so the message points at where the chain
    /// had reached. First-error-wins: a later `fail` never overwrites an
    /// earlier one (matches the entry guards on every builder method).
    pub(super) fn fail(&mut self, msg: impl std::fmt::Display) {
        if self.err.is_some() {
            return;
        }
        let at = self
            .on_target
            .as_ref()
            .or_else(|| self.current.first())
            .map(|n| format!(" (after node '{}')", n.node_id))
            .unwrap_or_else(|| " (at chain start)".to_string());
        self.err = Some(format!("{msg}{at}"));
    }

    fn add_module(&mut self, module: impl Module + 'static) -> NodeRef {
        self.add_boxed_module(Box::new(module))
    }

    fn add_boxed_module(&mut self, module: Box<dyn Module>) -> NodeRef {
        let id = self.next_id(module.name());
        let rc: Rc<dyn Module> = Rc::from(module);
        let run = wrap_module(rc.clone());

        // Auto-detect NamedInputModule capability for using() support
        let ref_forward = if rc.as_named_input().is_some() {
            let rc_clone = rc.clone();
            let rf: RefForwardFn = Rc::new(move |input, refs| {
                rc_clone.as_named_input().unwrap().forward_named(input, refs)
            });
            Some(rf)
        } else {
            None
        };

        self.nodes.insert(
            id.clone(),
            Node {
                id: id.clone(),
                input_ports: vec![DEFAULT_INPUT.into()],
                output_ports: vec![DEFAULT_OUTPUT.into()],
                run,
                module: Some(rc),
                ref_forward,
                trace_buf: None,
                named_trace_buf: None,
                loop_ports: None,
            },
        );

        NodeRef {
            node_id: id,
            port: DEFAULT_OUTPUT.into(),
        }
    }

    fn wire_using(
        &mut self,
        target: &NodeRef,
        refs: &[&str],
    ) -> std::result::Result<(), String> {
        // Check the target node supports refs
        {
            let node = self.nodes.get(&target.node_id).ok_or_else(|| {
                format!("unknown target node: {}", target.node_id)
            })?;
            if node.module.is_some() && node.ref_forward.is_none() {
                let hint = if target.node_id.contains("gate") || target.node_id.contains("switch") {
                    "the router must implement NamedInputModule and override as_named_input"
                } else {
                    "implement NamedInputModule and override as_named_input on the module"
                };
                return Err(format!(
                    "node '{}' does not support .using() refs — {}",
                    target.node_id, hint
                ));
            }
        }

        // Add ref ports and edges
        for ref_name in refs {
            let port_name = format!("ref_{}", ref_name);

            if let Some(tap) = self.taps.get(*ref_name).cloned() {
                // Backward reference: tag already exists, wire directly
                let node = self.nodes.get_mut(&target.node_id).unwrap();
                node.input_ports.push(port_name.clone());

                self.edges.push(Edge {
                    from_node: tap.node_id.clone(),
                    from_port: tap.port.clone(),
                    to_node: target.node_id.clone(),
                    to_port: port_name,
                });
            } else {
                // Forward reference: tag not set yet, create state reader node
                let reader_ref = self.add_state_read_node(ref_name);

                let node = self.nodes.get_mut(&target.node_id).unwrap();
                node.input_ports.push(port_name.clone());

                self.edges.push(Edge {
                    from_node: reader_ref.node_id.clone(),
                    from_port: reader_ref.port.clone(),
                    to_node: target.node_id.clone(),
                    to_port: port_name,
                });

                self.pending
                    .entry(ref_name.to_string())
                    .or_default()
                    .push(PendingUsing {
                        reader_id: reader_ref.node_id,
                    });
            }
        }

        // Update run function with new ports
        let node = self.nodes.get_mut(&target.node_id).unwrap();
        if let Some(ref loop_ports) = node.loop_ports {
            // Loop nodes: update the shared port list — the loop's run closure
            // already reads this at execution time via extract_refs.
            *loop_ports.borrow_mut() = node.input_ports.clone();
        } else if let (Some(module), Some(ref_forward)) =
            (node.module.clone(), node.ref_forward.clone())
        {
            // Non-loop nodes: rebuild run with wrap_ref_module
            let ports = node.input_ports.clone();
            node.run = wrap_ref_module(module, ref_forward, ports);
        }

        Ok(())
    }

    fn add_add_node(&mut self, n: usize) -> NodeRef {
        let id = self.next_id("add");
        let input_ports: Vec<String> = (0..n).map(|i| format!("input_{}", i)).collect();

        self.nodes.insert(
            id.clone(),
            Node {
                id: id.clone(),
                input_ports,
                output_ports: vec![DEFAULT_OUTPUT.into()],
                run: Box::new(move |inputs: &[Variable]| {
                    let mut result = inputs[0].clone();
                    for inp in &inputs[1..] {
                        result = result.add(inp)?;
                    }
                    Ok(vec![result])
                }),
                module: None,
                ref_forward: None,
                trace_buf: None,
                named_trace_buf: None,
                loop_ports: None,
            },
        );

        NodeRef {
            node_id: id,
            port: DEFAULT_OUTPUT.into(),
        }
    }

    fn add_state_read_node(&mut self, ref_name: &str) -> NodeRef {
        let id = self.next_id(&format!("state_read_{}", ref_name));

        // Placeholder run — will be replaced by Graph::build with the actual state buffer
        self.nodes.insert(
            id.clone(),
            Node {
                id: id.clone(),
                input_ports: vec![],
                output_ports: vec![DEFAULT_OUTPUT.into()],
                run: Box::new(|_| {
                    Err(crate::tensor::TensorError::new(
                        "state_read not wired (build bug)",
                    ))
                }),
                module: None,
                ref_forward: None,
                trace_buf: None,
                named_trace_buf: None,
                loop_ports: None,
            },
        );

        NodeRef {
            node_id: id,
            port: DEFAULT_OUTPUT.into(),
        }
    }

    fn add_input_node(&mut self, name: &str) -> NodeRef {
        let id = self.next_id(&format!("input_{}", name));
        self.nodes.insert(
            id.clone(),
            Node {
                id: id.clone(),
                input_ports: vec![DEFAULT_INPUT.into()],
                output_ports: vec![DEFAULT_OUTPUT.into()],
                run: Box::new(|inputs: &[Variable]| Ok(inputs.to_vec())),
                module: None,
                ref_forward: None,
                trace_buf: None,
                named_trace_buf: None,
                loop_ports: None,
            },
        );
        NodeRef {
            node_id: id,
            port: DEFAULT_OUTPUT.into(),
        }
    }

    fn add_merge_node(&mut self, n: usize, op: MergeOp) -> NodeRef {
        let id = self.next_id("merge");
        let input_ports: Vec<String> = (0..n).map(|i| format!("input_{}", i)).collect();

        let run: NodeFn = match op {
            MergeOp::Add => Box::new(move |inputs: &[Variable]| {
                let mut result = inputs[0].clone();
                for inp in &inputs[1..] {
                    result = result.add(inp)?;
                }
                Ok(vec![result])
            }),
            MergeOp::Mean => Box::new(move |inputs: &[Variable]| {
                let mut result = inputs[0].clone();
                for inp in &inputs[1..] {
                    result = result.add(inp)?;
                }
                result.mul_scalar(1.0 / n as f64).map(|v| vec![v])
            }),
        };

        self.nodes.insert(
            id.clone(),
            Node {
                id: id.clone(),
                input_ports,
                output_ports: vec![DEFAULT_OUTPUT.into()],
                run,
                module: None,
                ref_forward: None,
                trace_buf: None,
                named_trace_buf: None,
                loop_ports: None,
            },
        );

        NodeRef {
            node_id: id,
            port: DEFAULT_OUTPUT.into(),
        }
    }
}
