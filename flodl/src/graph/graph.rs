use std::cell::{Cell, OnceCell, RefCell};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::rc::Rc;

use indexmap::IndexMap;

use super::node::*;
use super::profile;
use super::GraphExt;
use super::LossContext;
use crate::autograd::Variable;
use crate::nn::{Buffer, Module, Parameter};
use crate::tensor::{Result, TensorError};

/// Pre-computed route from one node's output port to another node's input port.
/// Replaces HashMap-based edge routing in forward_impl for O(1) access.
#[derive(Clone)]
pub(crate) struct Route {
    // pub(crate): built in `graph.rs`, read by `forward_impl` in `execution.rs`.
    pub(crate) from_port_idx: usize,
    pub(crate) to_node_idx: usize,
    pub(crate) to_port_idx: usize,
}

/// Pre-computed graph input → node input slot mapping.
pub(crate) struct InputRoute {
    pub(crate) node_idx: usize,
    pub(crate) port_idx: usize,
}

/// Forward-reference state buffer. Persists across `forward()` calls.
pub(crate) struct StateEntry {
    pub(crate) writer_ni: usize,
    pub(crate) value: Rc<RefCell<Option<Variable>>>,
}

/// An executable computation graph. Implements `Module` for composability.
///
/// Built via `FlowBuilder`. Supports parallel execution of independent nodes,
/// observation of tagged outputs, profiling, and DOT/SVG visualization.
///
/// ```ignore
/// let g = FlowBuilder::from(Linear::new(4, 8)?)
///     .through(GELU)
///     .through(Linear::new(8, 2)?)
///     .build()?;
///
/// // Forward pass (graph implements Module)
/// let y = g.forward(&x)?;
///
/// // Observation
/// g.end_step();
/// g.end_epoch();
/// let loss_trend = g.trend("loss");
///
/// // Visualization
/// let dot = g.dot();
/// g.svg(Some("graph.svg"))?;
/// ```
pub struct Graph {
    pub(crate) nodes: Vec<Node>,
    pub(crate) node_index: HashMap<String, usize>,
    pub(crate) levels: Vec<Vec<usize>>,
    pub(crate) edges: Vec<Edge>,
    #[allow(dead_code)] // kept for DOT/debug introspection
    pub(crate) edges_from: HashMap<usize, Vec<usize>>,
    pub(crate) inputs: Vec<ExposedPort>,
    pub(crate) outputs: Vec<ExposedPort>,
    pub(crate) order: Vec<usize>,
    pub(crate) state: Vec<StateEntry>,
    // State writer lookup: node_idx → [(state_entry_idx, output_port_idx)]
    pub(crate) state_writers: HashMap<usize, Vec<(usize, usize)>>,
    // Tag groups: group name → suffixed tag names
    pub(crate) tag_groups: HashMap<String, Vec<String>>,
    // Observation: tag mapping (immutable after build)
    pub(crate) tag_names: HashMap<String, (usize, usize)>,           // tag name → (node_idx, port_idx)
    pub(crate) tag_capture: HashMap<usize, Vec<(String, usize)>>,     // node_idx → [(tag_name, port_idx)]
    // Observation: mutable state (RefCell/Cell for &self methods)
    pub(crate) tagged_outputs: RefCell<HashMap<String, Variable>>,
    pub(crate) batch_buffer: RefCell<HashMap<String, Vec<f64>>>,
    pub(crate) epoch_history: RefCell<HashMap<String, Vec<f64>>>,
    pub(crate) metric_order: RefCell<Vec<String>>,
    pub(crate) flush_count: Cell<usize>,
    // Profiling
    pub(crate) profiling: Cell<bool>,
    pub(crate) last_profile: RefCell<Option<profile::Profile>>,
    pub(crate) timing_buffer: RefCell<HashMap<String, Vec<f64>>>,
    pub(crate) timing_history: RefCell<HashMap<String, Vec<f64>>>,
    // Flush timestamps (seconds since first forward — for ETA in write_log)
    pub(crate) flush_times: RefCell<Vec<f64>>,
    pub(crate) training_start: Cell<f64>,
    // Step/epoch counters
    pub(crate) step_count: Cell<usize>,
    pub(crate) epoch_count: Cell<usize>,
    // Identity: label + structural hash
    pub(crate) label: Option<String>,
    pub(crate) structural_hash_cache: OnceCell<String>,
    // Graph tree: hierarchical composition
    pub(crate) children: HashMap<String, usize>,
    pub(crate) composed: Cell<bool>,
    pub(crate) internal_tags: HashSet<String>,
    // Pre-computed execution plan (built once, used every forward call)
    pub(crate) routes_from: Vec<Vec<Route>>,
    pub(crate) input_routes: Vec<InputRoute>,
    pub(crate) output_node_idx: usize,
    pub(crate) output_port_idx: usize,
    pub(crate) node_input_count: Vec<usize>,
    // Cached execution buffers (reused across forward calls, avoids re-allocation)
    pub(crate) exec_slots: RefCell<Vec<Vec<Option<Variable>>>>,
    // Cluster-mode DDP state (process-per-rank). Holds the local replica and
    // a `Ddp` wrapping a single `NcclRankComm` joined to the cross-process
    // group. Set by `Trainer::setup` when `LocalCluster::from_env()` returns
    // `Some`. Exists alongside `distributed` during the multi-host DDP
    // consolidation arc; a follow-up collapses both into a single field.
    pub(crate) cluster_ddp:
        RefCell<Option<(crate::distributed::ddp::Ddp, Box<dyn Module>)>>,
    // Cluster-mode El Che cadence state (heterogeneous DDP across processes).
    // When set, `step()` defers the sync + optimizer step until the local
    // cadence target is reached; cross-process timing AllReduce keeps every
    // rank's anchor in lockstep without a broadcast.
    pub(crate) cluster_el_che:
        RefCell<Option<crate::distributed::ddp::ClusterElCheState>>,
    // Optimizer for step() (works for both single-GPU and distributed)
    pub(crate) optimizer: RefCell<Option<Box<dyn crate::nn::Optimizer>>>,
    // Optional per-batch LR scheduler. When set, `step()` updates every
    // optimizer's LR from `scheduler.lr(training_step) * lr_scale` before
    // calling `optimizer.step()`.
    pub(crate) scheduler: RefCell<Option<std::sync::Arc<dyn crate::nn::Scheduler>>>,
    // DDP linear-scaling factor applied multiplicatively to scheduler output.
    // Defaults to 1.0 (no scaling). Set by `Trainer::setup_with` when the user
    // enabled `DdpConfig::lr_scale_ratio`.
    pub(crate) lr_scale: Cell<f64>,
    // Dedicated training step counter for the LR scheduler. Incremented once
    // per `step()` call, regardless of whether the caller also invokes
    // `end_step()` (which is only used by recurrent graphs to cut gradients).
    pub(crate) training_step: Cell<usize>,
    // DataLoader binding for resident DDP (set by set_data_loader(), None by default)
    pub(crate) data_binding: RefCell<Option<DataLoaderBinding>>,
    /// The bound DataLoader, in its OWN cell so an active epoch iterator can
    /// hold it exclusively (see
    /// [`GraphEpochIterator::activate`](crate::graph::GraphEpochIterator::activate)) while
    /// `forward_batch` / `data_num_batches` keep reading the metadata in
    /// `data_binding`. Set together with `data_binding` by `set_data_loader`.
    pub(crate) data_loader: RefCell<Option<crate::data::DataLoader>>,
    // Per-batch loss closure for El Che (set by set_loss_fn(), None = legacy gather path)
    #[allow(clippy::type_complexity)]
    pub(crate) loss_fn: RefCell<Option<Box<dyn Fn(&LossContext) -> Result<Variable>>>>,
    // Cached flag: trace-namespace collision check has run successfully once.
    // Set the first time trace observation is performed (single-GPU lookup or
    // El Che gather). Validates that emit-published trace names from loop
    // bodies don't collide with each other or with legacy post-loop tag names.
    pub(crate) traces_validated: Cell<bool>,
    // Optional opaque metadata attached to the graph (typically a JSON
    // config from the source the graph was built from, e.g. an HF
    // `config.json`). When set, [`Graph::save_checkpoint`] emits it as
    // a `<stem>.config.json` sidecar so downstream tools can recover
    // the architecture without re-fetching the source.
    //
    // flodl itself never inspects the contents — the field is plain
    // text passed through verbatim. Format and meaning are owned by
    // the caller (e.g. `flodl-hf`'s `AutoModel::from_pretrained` sets
    // this to the HF `config.json` it loaded).
    pub(crate) source_config: RefCell<Option<String>>,
    // Shared slot for the most recent coord-broadcast
    // [`crate::distributed::EpochMetrics`]. Writers live on the
    // cluster-worker bridge thread (`dispatch_control` for
    // `ControlMsg::EpochAggregated`); readers live on the user's
    // main training-loop thread (`latest_metrics`,
    // `aggregated_gpu_tabs`). `Arc<Mutex<...>>` so writes from one
    // thread are visible to reads on another. Empty until the coord
    // pushes the first aggregated view (single-GPU runs that never
    // hit the coord-side aggregation keep this `None` forever and
    // fall back to local epoch_history).
    pub(crate) aggregated_metrics: std::sync::Arc<
        std::sync::Mutex<Option<crate::distributed::EpochMetrics>>,
    >,
}

/// Binding between a `DataLoader` and a [`Graph`] for integrated training.
///
/// Created by [`Graph::set_data_loader`]. Maps batch tensor names to
/// graph inputs. The loader itself lives in `Graph::data_loader` (a
/// separate cell) so an active epoch iterator can borrow it exclusively
/// while this metadata stays readable.
pub(crate) struct DataLoaderBinding {
    /// Name of the batch field used as the primary forward input (e.g., "image").
    pub forward_input: String,
    /// Mappings from batch field names to graph Input port names.
    /// Only populated for graphs with `.input()` ports that match batch names.
    #[allow(dead_code)]
    pub graph_inputs: Vec<(String, String)>, // (batch_name, graph_input_name)
    /// Names of batch fields that are targets (for loss), not consumed by forward.
    #[allow(dead_code)]
    pub target_names: Vec<String>,
    /// Maps graph input index → batch tensor position.
    /// `shard_input_map[i]` is the index into `Batch` that provides
    /// `self.inputs[i]`.
    pub shard_input_map: Vec<usize>,
    /// Cached from the loader at bind time (fixed per loader config), so
    /// `data_num_batches` / `data_batch_size` stay readable while an epoch
    /// iterator holds the loader cell exclusively.
    pub num_batches: usize,
    /// See `num_batches`.
    pub batch_size: usize,
}

impl Graph {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build(
        mut node_map: IndexMap<String, Node>,
        edges: Vec<Edge>,
        inputs: Vec<ExposedPort>,
        outputs: Vec<ExposedPort>,
        tags: HashMap<String, NodeRef>,
        forward_refs: Vec<ForwardRefSpec>,
        tag_groups: HashMap<String, Vec<String>>,
        label: Option<String>,
        mut internal_tags: HashSet<String>,
        verbose: bool,
    ) -> Result<Self> {
        // Set up forward-reference state buffers and wire state read nodes
        let mut state = Vec::with_capacity(forward_refs.len());
        for fr in &forward_refs {
            let value: Rc<RefCell<Option<Variable>>> = Rc::new(RefCell::new(None));
            let reader_value = value.clone();

            // Wire the state read node to return the buffer value
            if let Some(node) = node_map.get_mut(&fr.reader_id) {
                node.run = Box::new(move |_: &[Variable]| {
                    match reader_value.borrow().as_ref() {
                        Some(v) => Ok(vec![v.clone()]),
                        None => Ok(vec![]), // empty = no state yet
                    }
                });
            }

            state.push(StateEntry {
                writer_ni: 0, // resolved after node indexing
                value,
            });
        }

        // Port-name resolution: a name that isn't among the node's declared
        // ports is a wiring bug (silently routing to port 0 would train on
        // wrong data), so it errors like an unknown node does.
        fn port_index(ports: &[String], port: &str, node: &str, what: &str) -> Result<usize> {
            ports.iter().position(|p| p == port).ok_or_else(|| {
                TensorError::new(&format!(
                    "unknown {what} port {port:?} on node {node:?} (declared ports: {ports:?})"
                ))
            })
        }

        // Convert to indexed storage
        let mut nodes = Vec::with_capacity(node_map.len());
        let mut node_index = HashMap::with_capacity(node_map.len());

        for (_key, node) in node_map {
            let idx = nodes.len();
            node_index.insert(node.id.clone(), idx);
            nodes.push(node);
        }

        // Validate edges
        for edge in &edges {
            if !node_index.contains_key(&edge.from_node) {
                return Err(TensorError::new(&format!(
                    "unknown source node: {}",
                    edge.from_node
                )));
            }
            if !node_index.contains_key(&edge.to_node) {
                return Err(TensorError::new(&format!(
                    "unknown target node: {}",
                    edge.to_node
                )));
            }
        }

        // Build edges_from lookup
        let mut edges_from: HashMap<usize, Vec<usize>> = HashMap::new();
        for (ei, edge) in edges.iter().enumerate() {
            let from_idx = node_index[&edge.from_node];
            edges_from.entry(from_idx).or_default().push(ei);
        }

        // Topological levels (Kahn's algorithm)
        let levels = topological_levels(&nodes, &node_index, &edges)?;
        let order: Vec<usize> = levels.iter().flat_map(|l| l.iter().copied()).collect();

        // Build tag capture indices for observation
        let mut tag_names_map: HashMap<String, (usize, usize)> = HashMap::new();
        let mut tag_capture: HashMap<usize, Vec<(String, usize)>> = HashMap::new();
        for (name, node_ref) in &tags {
            if let Some(&ni) = node_index.get(&node_ref.node_id) {
                let port_idx = port_index(
                    &nodes[ni].output_ports,
                    &node_ref.port,
                    &node_ref.node_id,
                    &format!("tag {name:?} output"),
                )?;
                tag_names_map.insert(name.clone(), (ni, port_idx));
                tag_capture
                    .entry(ni)
                    .or_default()
                    .push((name.clone(), port_idx));
            }
        }

        // Detect child subgraphs: labeled Graphs become tree children
        let mut children: HashMap<String, usize> = HashMap::new();
        for (idx, node) in nodes.iter().enumerate() {
            if let Some(ref module) = node.module {
                if let Some(child_graph) = module.as_graph() {
                    if let Some(child_label) = child_graph.label() {
                        if child_label.contains('.') {
                            return Err(TensorError::new(&format!(
                                "child graph label {:?} contains a dot — \
                                 dots are reserved for path separators",
                                child_label
                            )));
                        }
                        if children.contains_key(child_label) {
                            return Err(TensorError::new(&format!(
                                "duplicate child graph label {:?} at the same tree level",
                                child_label
                            )));
                        }
                        // Validate: label doesn't shadow a tag on a different node
                        if let Some(&(tag_ni, _)) = tag_names_map.get(child_label) {
                            if tag_ni != idx {
                                return Err(TensorError::new(&format!(
                                    "child graph label {:?} collides with a tag \
                                     on a different node",
                                    child_label
                                )));
                            }
                        }
                        children.insert(child_label.to_string(), idx);
                        child_graph.composed.set(true);
                    }
                    // Unlabeled graphs: not registered, no tree features, no error
                }
            }
        }

        // Auto-internal inference: underscore-prefixed tags
        for name in tag_names_map.keys() {
            if name.starts_with('_') {
                internal_tags.insert(name.clone());
            }
        }

        // Build state writer lookup: node_idx → [(state_entry_idx, port_idx)]
        // Also resolve writer_ni on each state entry for DOT rendering.
        let mut state_writers: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
        for (si, fr) in forward_refs.iter().enumerate() {
            if let Some(&ni) = node_index.get(&fr.writer_id) {
                state[si].writer_ni = ni;
                let port_idx = port_index(
                    &nodes[ni].output_ports,
                    &fr.writer_port,
                    &fr.writer_id,
                    "state-writer output",
                )?;
                state_writers.entry(ni).or_default().push((si, port_idx));
            }
        }

        // Pre-compute routing table: flat Vec lookups replace HashMap edge routing
        let n = nodes.len();
        let mut routes_from: Vec<Vec<Route>> = vec![Vec::new(); n];
        for edge in &edges {
            let from_ni = node_index[&edge.from_node];
            let to_ni = node_index[&edge.to_node];
            let from_port_idx = port_index(
                &nodes[from_ni].output_ports,
                &edge.from_port,
                &edge.from_node,
                "edge source",
            )?;
            let to_port_idx = port_index(
                &nodes[to_ni].input_ports,
                &edge.to_port,
                &edge.to_node,
                "edge target",
            )?;
            routes_from[from_ni].push(Route {
                from_port_idx,
                to_node_idx: to_ni,
                to_port_idx,
            });
        }

        // Pre-compute graph input → slot mapping
        let input_routes: Vec<InputRoute> = inputs
            .iter()
            .map(|ep| {
                let ni = node_index[&ep.node_id];
                let port_idx = port_index(
                    &nodes[ni].input_ports,
                    &ep.port,
                    &ep.node_id,
                    "graph input",
                )?;
                Ok(InputRoute {
                    node_idx: ni,
                    port_idx,
                })
            })
            .collect::<Result<_>>()?;

        // Pre-compute output location
        let output_node_idx = node_index[&outputs[0].node_id];
        let output_port_idx = port_index(
            &nodes[output_node_idx].output_ports,
            &outputs[0].port,
            &outputs[0].node_id,
            "graph output",
        )?;

        // Pre-compute input port counts and allocate execution buffers
        let node_input_count: Vec<usize> = nodes.iter().map(|nd| nd.input_ports.len()).collect();
        let exec_slots = RefCell::new(
            node_input_count.iter().map(|&c| vec![None; c]).collect(),
        );

        let graph = Ok(Graph {
            nodes,
            node_index,
            levels,
            edges,
            edges_from,
            inputs,
            outputs,
            order,
            state,
            state_writers,
            tag_groups,
            tag_names: tag_names_map,
            tag_capture,
            tagged_outputs: RefCell::new(HashMap::new()),
            batch_buffer: RefCell::new(HashMap::new()),
            epoch_history: RefCell::new(HashMap::new()),
            metric_order: RefCell::new(Vec::new()),
            flush_count: Cell::new(0),
            profiling: Cell::new(false),
            last_profile: RefCell::new(None),
            timing_buffer: RefCell::new(HashMap::new()),
            timing_history: RefCell::new(HashMap::new()),
            flush_times: RefCell::new(Vec::new()),
            training_start: Cell::new(0.0),
            step_count: Cell::new(0),
            epoch_count: Cell::new(0),
            label,
            structural_hash_cache: OnceCell::new(),
            children,
            composed: Cell::new(false),
            internal_tags,
            routes_from,
            input_routes,
            output_node_idx,
            output_port_idx,
            node_input_count,
            exec_slots,
            cluster_ddp: RefCell::new(None),
            cluster_el_che: RefCell::new(None),
            optimizer: RefCell::new(None),
            scheduler: RefCell::new(None),
            lr_scale: Cell::new(1.0),
            training_step: Cell::new(0),
            data_binding: RefCell::new(None),
            data_loader: RefCell::new(None),
            loss_fn: RefCell::new(None),
            traces_validated: Cell::new(false),
            source_config: RefCell::new(None),
            aggregated_metrics: std::sync::Arc::new(std::sync::Mutex::new(None)),
        });

        if verbose {
            if let Ok(ref g) = graph {
                crate::verbose!("{}", g.tree_summary());
            }
        }

        graph
    }
}

impl Graph {
    /// Clear all forward-reference state buffers to None.
    /// Call when starting inference on a new sequence.
    pub fn reset_state(&self) {
        for entry in &self.state {
            *entry.value.borrow_mut() = None;
        }
    }

    /// Break gradient chain on forward-reference state buffers and module state.
    /// Call between training steps to prevent unbounded graph growth.
    pub fn detach_state(&self) {
        // Detach graph-level state buffers (forward references).
        for entry in &self.state {
            let mut val = entry.value.borrow_mut();
            if let Some(ref v) = *val {
                *val = Some(v.detach());
            }
        }
        // Detach tagged outputs — these hold Variables from the forward
        // pass whose grad_fn chains reference the C++ autograd graph.
        // Without this, the Node objects persist until the next forward
        // pass replaces tagged_outputs.
        {
            let mut tagged = self.tagged_outputs.borrow_mut();
            for var in tagged.values_mut() {
                *var = var.detach();
            }
        }
        // Propagate detach to modules that hold internal state.
        for node in &self.nodes {
            if let Some(ref module) = node.module {
                module.detach_state();
            }
        }
    }

    /// Returns true if this graph has forward-reference state.
    pub fn has_state(&self) -> bool {
        !self.state.is_empty()
    }

    /// End-of-step housekeeping: detach state (cut gradient chain but
    /// preserve values for the next forward), collect timings,
    /// increment step counter.
    ///
    /// For recurrent models this implements truncated BPTT — state carries
    /// over between steps but gradients don't flow across step boundaries.
    /// Call [`end_sequence`](Self::end_sequence) to fully wipe state
    /// when starting a new independent sequence.
    ///
    /// ```ignore
    /// for token in sequence {
    ///     let y = graph.forward(&token)?;
    ///     // ... backward, optimize ...
    ///     graph.end_step();       // keep state, cut gradients
    /// }
    /// graph.end_sequence();       // wipe state for next sequence
    /// ```
    pub fn end_step(&self) {
        self.detach_state();
        if self.profiling.get() {
            self.collect_timings(&[]);
        }
        self.step_count.set(self.step_count.get() + 1);
    }

    /// End-of-sequence housekeeping: fully reset state buffers to None.
    /// Call between independent sequences so the model starts fresh.
    ///
    /// For non-recurrent graphs (no forward refs) this is a no-op.
    pub fn end_sequence(&self) {
        self.reset_state();
    }

    /// End-of-epoch housekeeping: flush all observation and timing buffers,
    /// increment epoch counter.
    pub fn end_epoch(&self) {
        self.flush(&[]);
        if self.profiling.get() {
            self.flush_timings(&[]);
        }
        self.epoch_count.set(self.epoch_count.get() + 1);
    }

    /// Number of completed training steps.
    pub fn step_count(&self) -> usize {
        self.step_count.get()
    }

    /// Number of completed training epochs.
    pub fn epoch_count(&self) -> usize {
        self.epoch_count.get()
    }

    /// Get member tags of a tag group, or None if not registered.
    pub fn tag_group(&self, name: &str) -> Option<&[String]> {
        self.tag_groups.get(name).map(|v| v.as_slice())
    }

    /// Forward with multiple inputs (for graphs with Input ports).
    /// Inputs are in declaration order: From entry first, then each Input.
    ///
    /// In cluster mode, routes to the local replica's graph (via
    /// [`GraphExt::as_graph`]) so the
    /// replica's parameters drive the forward. Loud error if the
    /// replica does not expose a graph — `forward_multi` is
    /// graph-specific and has no meaningful fallback.
    pub fn forward_multi(&self, inputs: &[Variable]) -> Result<Variable> {
        if let Some((_, replica)) = self.cluster_ddp.borrow().as_ref() {
            return match replica.as_graph() {
                Some(g) => g.forward_multi(inputs),
                None => Err(TensorError::new(
                    "forward_multi: cluster-mode replica does not expose a Graph \
                     (GraphExt::as_graph returned None) — multi-input forward has \
                     no fallback. Wrap the model in something that presents its \
                     graph through Module::as_any (e.g. HeadReplica for HasGraph \
                     wrappers).",
                )),
            };
        }
        self.forward_impl(inputs)
    }

    /// Move all parameters, state buffers, and module buffers to a device.
    pub fn set_device(&self, device: crate::tensor::Device) {
        // Move parameters — detach first so the moved tensor is a fresh leaf,
        // not a non-leaf with CopyBackward from native autograd.
        for p in self.parameters() {
            if p.variable.data().device() != device
                && let Ok(t) = p.variable.data().detach()
                    .and_then(|d| d.to_device(device))
            {
                p.variable.set_data(t);
            }
        }
        // Move state buffers
        for entry in &self.state {
            let mut val = entry.value.borrow_mut();
            if let Some(ref v) = *val
                && v.data().device() != device
                && let Ok(t) = v.data().to_device(device)
            {
                *val = Some(Variable::new(t, false));
            }
        }
        // Walk modules for move_to_device (BatchNorm running stats, etc.)
        let mut visited = HashSet::new();
        for &ni in &self.order {
            if let Some(ref module) = self.nodes[ni].module {
                crate::nn::walk_modules_visited(
                    module.as_ref(),
                    &mut visited,
                    &mut |m: &dyn crate::nn::Module| m.move_to_device(device),
                );
            }
        }
    }

    /// Return parameters with qualified names: `"prefix/param_name"`.
    ///
    /// The prefix is the tag name if the node is tagged, otherwise the node ID
    /// (e.g. `"linear_1"`). When a node has multiple parameters with the same
    /// name, suffixes `_0`, `_1`, ... are appended to disambiguate.
    pub fn named_parameters(&self) -> Vec<(String, Parameter)> {
        self.named_items(
            |m| m.parameters(),
            |p| p.variable.id(),
            |p| p.name.clone(),
        )
    }

    /// Return buffers with qualified names, using the same prefix logic
    /// as `named_parameters()`.
    pub fn named_buffers(&self) -> Vec<(String, Buffer)> {
        self.named_items(
            |m| m.buffers(),
            |b| b.id(),
            |b| b.name.clone(),
        )
    }

    /// Shared body of [`named_parameters`](Self::named_parameters) and
    /// [`named_buffers`](Self::named_buffers): walk nodes in execution
    /// order, prefix each item by its node's tag (first tag wins) or node
    /// ID, dedup by identity (`id_of`), and disambiguate same-named items
    /// within a node with `_0`/`_1`/... suffixes. `collect` pulls the
    /// items from a module; `id_of`/`name_of` read an item's identity and
    /// name (the only points where `Parameter` and `Buffer` differ).
    fn named_items<T>(
        &self,
        collect: impl Fn(&dyn crate::nn::Module) -> Vec<T>,
        id_of: impl Fn(&T) -> usize,
        name_of: impl Fn(&T) -> String,
    ) -> Vec<(String, T)> {
        // Reverse map: node_idx → tag name (first tag wins; deterministic
        // because we only need one prefix).
        let mut idx_to_tag: HashMap<usize, String> = HashMap::new();
        for (tag, &(ni, _)) in &self.tag_names {
            idx_to_tag.entry(ni).or_insert_with(|| tag.clone());
        }

        let mut result = Vec::new();
        let mut seen = HashSet::new();

        for &ni in &self.order {
            if let Some(ref module) = self.nodes[ni].module {
                let prefix = idx_to_tag.get(&ni)
                    .cloned()
                    .unwrap_or_else(|| self.nodes[ni].id.clone());

                let items = collect(module.as_ref());
                // Count duplicate names within this node.
                let mut name_counts: HashMap<String, usize> = HashMap::new();
                for it in &items {
                    *name_counts.entry(name_of(it)).or_insert(0) += 1;
                }

                let mut name_idx: HashMap<String, usize> = HashMap::new();
                for it in items {
                    if !seen.insert(id_of(&it)) {
                        continue;
                    }
                    let name = name_of(&it);
                    let qualified = if name_counts[&name] > 1 {
                        let idx = name_idx.entry(name.clone()).or_insert(0);
                        let q = format!("{}/{}_{}", prefix, name, idx);
                        *idx += 1;
                        q
                    } else {
                        format!("{}/{}", prefix, name)
                    };

                    result.push((qualified, it));
                }
            }
        }

        result
    }

    /// Human-readable label set via `FlowBuilder::label()`.
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// Full 64-character hex structural hash (computed lazily, cached).
    pub fn structural_hash(&self) -> &str {
        self.structural_hash_cache.get_or_init(|| self.compute_structural_hash())
    }

    /// First 8 characters of the structural hash.
    pub fn short_hash(&self) -> &str {
        &self.structural_hash()[..8]
    }
}

/// Current time as seconds since epoch (monotonic approximation for ETA).
pub(crate) fn instant_secs() -> f64 {
    use std::time::SystemTime;
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

/// Kahn's algorithm with level grouping for parallel execution.
fn topological_levels(
    nodes: &[Node],
    node_index: &HashMap<String, usize>,
    edges: &[Edge],
) -> Result<Vec<Vec<usize>>> {
    let n = nodes.len();

    // Build unique dependency sets (node-level, not edge-level).
    // dependents uses BTreeSet so iteration follows node index order,
    // making the topological sort deterministic across runs.
    let mut deps: Vec<HashSet<usize>> = vec![HashSet::new(); n];
    let mut dependents: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];

    for edge in edges {
        let from_ni = node_index[&edge.from_node];
        let to_ni = node_index[&edge.to_node];
        deps[to_ni].insert(from_ni);
        dependents[from_ni].insert(to_ni);
    }

    let mut in_degree: Vec<usize> = deps.iter().map(|d| d.len()).collect();

    // Seed with zero in-degree nodes
    let mut queue: Vec<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
    let mut levels = Vec::new();
    let mut visited = 0;

    while !queue.is_empty() {
        levels.push(queue.clone());
        visited += queue.len();

        let mut next_queue = Vec::new();
        for &ni in &queue {
            for &dep in &dependents[ni] {
                in_degree[dep] -= 1;
                if in_degree[dep] == 0 {
                    next_queue.push(dep);
                }
            }
        }
        queue = next_queue;
    }

    if visited != n {
        return Err(TensorError::new("cycle detected in graph"));
    }

    Ok(levels)
}


#[cfg(test)]
#[path = "tests/mod.rs"]
mod tests;

