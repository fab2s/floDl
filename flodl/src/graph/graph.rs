use std::cell::{Cell, OnceCell, RefCell};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::PathBuf;
use std::rc::Rc;
use std::time::Instant;

use indexmap::IndexMap;
use hmac_sha256::Hash as Sha256;

use super::node::*;
use super::profile;
use super::LossContext;
use crate::autograd::Variable;
use crate::nn::{Buffer, Module, Parameter};
use crate::tensor::{Result, Tensor, TensorError};

/// Pre-computed route from one node's output port to another node's input port.
/// Replaces HashMap-based edge routing in forward_impl for O(1) access.
#[derive(Clone)]
pub(crate) struct Route {
    from_port_idx: usize,
    to_node_idx: usize,
    to_port_idx: usize,
}

/// Pre-computed graph input → node input slot mapping.
pub(crate) struct InputRoute {
    node_idx: usize,
    port_idx: usize,
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
}

/// Binding between a `DataLoader` and a [`Graph`] for integrated training.
///
/// Created by [`Graph::set_data_loader`]. Maps batch tensor names to
/// graph inputs and stores the loader reference.
pub(crate) struct DataLoaderBinding {
    /// The DataLoader.
    pub loader: crate::data::DataLoader,
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
                let port_idx = nodes[ni]
                    .output_ports
                    .iter()
                    .position(|p| p == &node_ref.port)
                    .unwrap_or(0);
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
                let port_idx = nodes[ni]
                    .output_ports
                    .iter()
                    .position(|p| p == &fr.writer_port)
                    .unwrap_or(0);
                state_writers.entry(ni).or_default().push((si, port_idx));
            }
        }

        // Pre-compute routing table: flat Vec lookups replace HashMap edge routing
        let n = nodes.len();
        let mut routes_from: Vec<Vec<Route>> = vec![Vec::new(); n];
        for edge in &edges {
            let from_ni = node_index[&edge.from_node];
            let to_ni = node_index[&edge.to_node];
            let from_port_idx = nodes[from_ni]
                .output_ports
                .iter()
                .position(|p| p == &edge.from_port)
                .unwrap_or(0);
            let to_port_idx = nodes[to_ni]
                .input_ports
                .iter()
                .position(|p| p == &edge.to_port)
                .unwrap_or(0);
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
                let port_idx = nodes[ni]
                    .input_ports
                    .iter()
                    .position(|p| p == &ep.port)
                    .unwrap_or(0);
                InputRoute {
                    node_idx: ni,
                    port_idx,
                }
            })
            .collect();

        // Pre-compute output location
        let output_node_idx = node_index[&outputs[0].node_id];
        let output_port_idx = nodes[output_node_idx]
            .output_ports
            .iter()
            .position(|p| p == &outputs[0].port)
            .unwrap_or(0);

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
            loss_fn: RefCell::new(None),
            traces_validated: Cell::new(false),
            source_config: RefCell::new(None),
        });

        if verbose {
            if let Ok(ref g) = graph {
                crate::verbose!("{}", g.tree_summary());
            }
        }

        graph
    }

    pub(crate) fn forward_impl(&self, graph_inputs: &[Variable]) -> Result<Variable> {
        if graph_inputs.len() != self.inputs.len() {
            return Err(TensorError::new(&format!(
                "expected {} inputs, got {}",
                self.inputs.len(),
                graph_inputs.len()
            )));
        }

        // Record training start on first forward (for ETA).
        if self.training_start.get() == 0.0 {
            self.training_start.set(instant_secs());
        }

        let is_profiling = self.profiling.get();
        let forward_start = if is_profiling { Some(Instant::now()) } else { None };
        let mut prof_nodes: Vec<profile::NodeTiming> = Vec::new();
        let mut prof_levels: Vec<profile::LevelTiming> = Vec::new();

        // Build reverse tag lookup for profiling: node_idx → first tag name
        let tags_by_node: HashMap<usize, String> = if is_profiling {
            let mut m = HashMap::new();
            for (name, &(ni, _)) in &self.tag_names {
                m.entry(ni).or_insert_with(|| name.clone());
            }
            m
        } else {
            HashMap::new()
        };

        let has_tags = !self.tag_capture.is_empty();

        // Reuse cached execution buffers (Vec-indexed, no HashMap overhead)
        let mut slots = self.exec_slots.borrow_mut();

        // Clear previous values (drops old Variables, reuses allocations)
        for node_slots in slots.iter_mut() {
            for slot in node_slots.iter_mut() {
                *slot = None;
            }
        }

        // Clear tagged outputs
        if has_tags {
            self.tagged_outputs.borrow_mut().clear();
        }

        // Route graph inputs via pre-computed index mapping
        for (i, route) in self.input_routes.iter().enumerate() {
            slots[route.node_idx][route.port_idx] = Some(graph_inputs[i].clone());
        }

        // Will hold the output node's results until we can extract the final value
        let mut final_output: Option<Vec<Variable>> = None;

        // Execute levels sequentially
        for (level_idx, level) in self.levels.iter().enumerate() {
            let level_start = if is_profiling { Some(Instant::now()) } else { None };
            let mut level_sum_ns: u64 = 0;

            for &ni in level {
                let node = &self.nodes[ni];
                let input_count = self.node_input_count[ni];

                // Collect inputs from pre-indexed slots (no HashMap lookups)
                let inputs: Vec<Variable> = (0..input_count)
                    .map(|i| {
                        match slots[ni][i].as_ref() {
                            Some(v) => Ok(v.clone()),
                            None if i > 0 => {
                                // Zero fill for unconnected ref ports (forward refs)
                                let first = slots[ni][0].as_ref().ok_or_else(|| {
                                    TensorError::new(&format!(
                                        "node '{}': ref port {} has no data and primary input \
                                         is also missing — check that all inputs are connected",
                                        node.id, i
                                    ))
                                })?;
                                Ok(Variable::new(
                                    Tensor::zeros_like(&first.data())?,
                                    false,
                                ))
                            }
                            _ => Err(TensorError::new(&format!(
                                "node '{}': missing primary input (port {}) — check that all \
                                 inputs to this node are connected in the graph builder",
                                node.id, i
                            ))),
                        }
                    })
                    .collect::<Result<Vec<Variable>>>()?;

                // Release input slots early (frees Rc references)
                for slot in slots[ni].iter_mut() {
                    *slot = None;
                }

                // Execute node (with optional per-node timing)
                let node_start = if is_profiling { Some(Instant::now()) } else { None };
                let node_outputs = (node.run)(&inputs)?;
                if is_profiling {
                    let elapsed = node_start.unwrap().elapsed();
                    level_sum_ns += elapsed.as_nanos() as u64;
                    prof_nodes.push(profile::NodeTiming {
                        id: node.id.clone(),
                        tag: tags_by_node.get(&ni).cloned().unwrap_or_default(),
                        duration: elapsed,
                        level: level_idx,
                    });
                }

                // Route outputs via pre-computed routing table (no HashMap, no String ops)
                for route in &self.routes_from[ni] {
                    let value = if route.from_port_idx < node_outputs.len() {
                        Some(node_outputs[route.from_port_idx].clone())
                    } else {
                        None
                    };
                    slots[route.to_node_idx][route.to_port_idx] = value;
                }

                // Capture state: if this node is a state writer, store its output
                if let Some(writers) = self.state_writers.get(&ni) {
                    for &(si, port_idx) in writers {
                        if port_idx < node_outputs.len() {
                            *self.state[si].value.borrow_mut() =
                                Some(node_outputs[port_idx].clone());
                        }
                    }
                }

                // Capture tagged outputs for observation
                if has_tags {
                    if let Some(captures) = self.tag_capture.get(&ni) {
                        let mut tagged = self.tagged_outputs.borrow_mut();
                        for (tag_name, port_idx) in captures {
                            if *port_idx < node_outputs.len() {
                                tagged.insert(
                                    tag_name.clone(),
                                    node_outputs[*port_idx].clone(),
                                );
                            }
                        }
                    }
                }

                // Keep output node's results; all others drop here (early release)
                if ni == self.output_node_idx {
                    final_output = Some(node_outputs);
                }
            }

            // Record level timing
            if is_profiling {
                prof_levels.push(profile::LevelTiming {
                    index: level_idx,
                    wall_clock: level_start.unwrap().elapsed(),
                    sum_nodes: std::time::Duration::from_nanos(level_sum_ns),
                    num_nodes: level.len(),
                });
            }
        }

        // Drop the borrow before storing profile (which also borrows RefCells)
        drop(slots);

        // Store profile
        if is_profiling {
            *self.last_profile.borrow_mut() = Some(profile::Profile {
                total: forward_start.unwrap().elapsed(),
                levels: prof_levels,
                nodes: prof_nodes,
            });
        }

        // Extract graph output
        final_output
            .and_then(|o| o.into_iter().nth(self.output_port_idx))
            .ok_or_else(|| TensorError::new("graph produced no output"))
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
    /// [`Module::as_graph`]) so the replica's parameters drive the forward.
    /// Loud error if the replica does not expose a graph — `forward_multi`
    /// is graph-specific and has no meaningful fallback.
    pub fn forward_multi(&self, inputs: &[Variable]) -> Result<Variable> {
        if let Some((_, replica)) = self.cluster_ddp.borrow().as_ref() {
            return match replica.as_graph() {
                Some(g) => g.forward_multi(inputs),
                None => Err(TensorError::new(
                    "forward_multi: cluster-mode replica does not expose a Graph \
                     (Module::as_graph returned None) — multi-input forward has \
                     no fallback. Wrap the model in something that implements \
                     as_graph (e.g. HeadReplica for HasGraph wrappers).",
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
        // Build reverse map: node_idx → tag name
        let mut idx_to_tag: HashMap<usize, String> = HashMap::new();
        for (tag, &(ni, _)) in &self.tag_names {
            // First tag wins (deterministic because we only need one prefix)
            idx_to_tag.entry(ni).or_insert_with(|| tag.clone());
        }

        let mut result = Vec::new();
        let mut seen = HashSet::new();

        for &ni in &self.order {
            if let Some(ref module) = self.nodes[ni].module {
                let prefix = idx_to_tag.get(&ni)
                    .cloned()
                    .unwrap_or_else(|| self.nodes[ni].id.clone());

                let params = module.parameters();
                // Check for duplicate param names within this node
                let mut name_counts: HashMap<String, usize> = HashMap::new();
                for p in &params {
                    *name_counts.entry(p.name.clone()).or_insert(0) += 1;
                }

                let mut name_idx: HashMap<String, usize> = HashMap::new();
                for p in params {
                    let ptr = Rc::as_ptr(&p.variable.inner) as usize;
                    if !seen.insert(ptr) {
                        continue;
                    }

                    let qualified = if name_counts[&p.name] > 1 {
                        let idx = name_idx.entry(p.name.clone()).or_insert(0);
                        let q = format!("{}/{}_{}", prefix, p.name, idx);
                        *idx += 1;
                        q
                    } else {
                        format!("{}/{}", prefix, p.name)
                    };

                    result.push((qualified, p));
                }
            }
        }

        result
    }

    /// Return buffers with qualified names, using the same prefix logic
    /// as `named_parameters()`.
    pub fn named_buffers(&self) -> Vec<(String, Buffer)> {
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

                let bufs = module.buffers();
                let mut name_counts: HashMap<String, usize> = HashMap::new();
                for b in &bufs {
                    *name_counts.entry(b.name.clone()).or_insert(0) += 1;
                }

                let mut name_idx: HashMap<String, usize> = HashMap::new();
                for b in bufs {
                    let ptr = Rc::as_ptr(&b.inner) as usize;
                    if !seen.insert(ptr) {
                        continue;
                    }

                    let qualified = if name_counts[&b.name] > 1 {
                        let idx = name_idx.entry(b.name.clone()).or_insert(0);
                        let q = format!("{}/{}_{}", prefix, b.name, idx);
                        *idx += 1;
                        q
                    } else {
                        format!("{}/{}", prefix, b.name)
                    };

                    result.push((qualified, b));
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

    /// Save all parameters and buffers to a checkpoint file.
    ///
    /// Embeds the structural hash for architecture validation on load.
    /// Supports `.gz` extension for gzip compression.
    ///
    /// When [`source_config`](Self::source_config) is set on the graph
    /// (e.g. populated by `flodl_hf::models::auto::AutoModel::from_pretrained`),
    /// a sidecar `<stem>.config.json` file is written next to the
    /// checkpoint with the source config verbatim. The stem strips both
    /// `.fdl` and an optional `.gz` so `model.fdl.gz` produces
    /// `model.config.json`. Downstream tools (e.g. `fdl flodl-hf export
    /// --checkpoint`) use this to rebuild the right family without an
    /// explicit `--config` argument.
    pub fn save_checkpoint(&self, path: &str) -> Result<()> {
        let params = self.named_parameters();
        let buffers = self.named_buffers();
        let hash = self.structural_hash();
        crate::nn::save_checkpoint_file(path, &params, &buffers, Some(hash))?;

        if let Some(content) = self.source_config.borrow().as_ref() {
            let sidecar = sidecar_config_path(path);
            std::fs::write(&sidecar, content).map_err(|e| {
                TensorError::new(&format!(
                    "save_checkpoint: cannot write sidecar {}: {e}",
                    sidecar.display()
                ))
            })?;
        }
        Ok(())
    }

    /// Attach an opaque source-config string to the graph.
    ///
    /// Typically called by loaders that build a graph from an external
    /// definition (HF `config.json`, ONNX manifest, …) so subsequent
    /// `save_checkpoint` calls drop a `<stem>.config.json` sidecar.
    /// Pass an empty string to attach a non-meaningful sentinel; pass
    /// `clear_source_config` to drop attachment entirely.
    pub fn set_source_config(&self, content: String) {
        *self.source_config.borrow_mut() = Some(content);
    }

    /// Read the currently-attached source config, if any.
    pub fn source_config(&self) -> Option<String> {
        self.source_config.borrow().clone()
    }

    /// Drop any attached source config so subsequent saves emit no
    /// sidecar.
    pub fn clear_source_config(&self) {
        *self.source_config.borrow_mut() = None;
    }

    /// Load parameters and buffers from a checkpoint file.
    ///
    /// Validates the structural hash against this graph's architecture.
    /// Returns a [`LoadReport`](crate::nn::LoadReport) describing what was
    /// loaded, skipped, or missing.
    pub fn load_checkpoint(&self, path: &str) -> Result<crate::nn::LoadReport> {
        let params = self.named_parameters();
        let buffers = self.named_buffers();
        let hash = self.structural_hash();
        crate::nn::load_checkpoint_file(path, &params, &buffers, Some(hash))
    }

    fn compute_structural_hash(&self) -> String {
        let mut hasher = Sha256::new();

        // 1. Nodes in topological order
        for &ni in &self.order {
            let node = &self.nodes[ni];
            hasher.update(node.id.as_bytes());
            hasher.update(b"\0");

            if let Some(ref module) = node.module {
                hasher.update(module.name().as_bytes());
                hasher.update(b"\0");

                // Sorted parameters
                let mut params: Vec<_> = module.parameters().into_iter()
                    .map(|p| (p.name.clone(), p.variable.shape()))
                    .collect();
                params.sort_by(|a, b| a.0.cmp(&b.0));
                for (name, shape) in &params {
                    hasher.update(b"P");
                    hasher.update(name.as_bytes());
                    hasher.update(b"\0");
                    for &dim in shape {
                        hasher.update(dim.to_le_bytes());
                    }
                }

                // Sorted buffers
                let mut bufs: Vec<_> = module.buffers().into_iter()
                    .map(|b| (b.name.clone(), b.shape()))
                    .collect();
                bufs.sort_by(|a, b| a.0.cmp(&b.0));
                for (name, shape) in &bufs {
                    hasher.update(b"B");
                    hasher.update(name.as_bytes());
                    hasher.update(b"\0");
                    for &dim in shape {
                        hasher.update(dim.to_le_bytes());
                    }
                }

                // Nested graph hash
                if let Some(nested_hash) = module.structural_hash() {
                    hasher.update(b"G");
                    hasher.update(nested_hash.as_bytes());
                }
            }
        }

        // 2. Edges
        hasher.update(b"EDGES");
        for edge in &self.edges {
            hasher.update(edge.from_node.as_bytes());
            hasher.update(b"\0");
            hasher.update(edge.from_port.as_bytes());
            hasher.update(b"\0");
            hasher.update(edge.to_node.as_bytes());
            hasher.update(b"\0");
            hasher.update(edge.to_port.as_bytes());
            hasher.update(b"\0");
        }

        // 3. Tags (sorted)
        hasher.update(b"TAGS");
        let mut tags: Vec<_> = self.tag_names.iter().collect();
        tags.sort_by(|a, b| a.0.cmp(b.0));
        for (name, (node_idx, port_idx)) in &tags {
            hasher.update(name.as_bytes());
            hasher.update(b"\0");
            hasher.update((*node_idx as u64).to_le_bytes());
            hasher.update((*port_idx as u64).to_le_bytes());
        }

        // 4. Input/output ports
        hasher.update(b"INPUTS");
        for port in &self.inputs {
            hasher.update(port.name.as_bytes());
            hasher.update(b"\0");
            hasher.update(port.node_id.as_bytes());
            hasher.update(b"\0");
            hasher.update(port.port.as_bytes());
            hasher.update(b"\0");
        }
        hasher.update(b"OUTPUTS");
        for port in &self.outputs {
            hasher.update(port.name.as_bytes());
            hasher.update(b"\0");
            hasher.update(port.node_id.as_bytes());
            hasher.update(b"\0");
            hasher.update(port.port.as_bytes());
            hasher.update(b"\0");
        }

        hasher.finalize().iter().map(|b| format!("{b:02x}")).collect()
    }
}

/// Derive the sidecar config-json path for a checkpoint path.
///
/// Strips a trailing `.gz` (compression marker) if present, then
/// replaces the extension with `config.json`. Stem is preserved
/// verbatim so paths like `/some/dir/v3.fdl` map to
/// `/some/dir/v3.config.json`.
pub(crate) fn sidecar_config_path(checkpoint: &str) -> PathBuf {
    let mut p = PathBuf::from(checkpoint);
    if p.extension().and_then(|e| e.to_str()) == Some("gz") {
        // Drop the .gz to expose the inner extension (.fdl, .ckpt, ...).
        p.set_extension("");
    }
    p.set_extension("config.json");
    p
}

impl Module for Graph {
    fn name(&self) -> &str { "graph" }

    fn as_graph(&self) -> Option<&Graph> { Some(self) }

    fn structural_hash(&self) -> Option<String> {
        Some(self.structural_hash().to_string())
    }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        // Cluster mode: route through the local replica. `self`'s own nodes
        // act as a structural template only; the replica owns this rank's
        // training parameters.
        if let Some((_, replica)) = self.cluster_ddp.borrow().as_ref() {
            return replica.forward(input);
        }
        self.forward_impl(std::slice::from_ref(input))
    }

    fn parameters(&self) -> Vec<Parameter> {
        if let Some((_, replica)) = self.cluster_ddp.borrow().as_ref() {
            return replica.parameters();
        }
        let mut params = Vec::new();
        let mut seen = HashSet::new();

        for &ni in &self.order {
            if let Some(ref module) = self.nodes[ni].module {
                for p in module.parameters() {
                    let ptr = Rc::as_ptr(&p.variable.inner) as usize;
                    if seen.insert(ptr) {
                        params.push(p);
                    }
                }
            }
        }

        params
    }

    fn buffers(&self) -> Vec<crate::nn::Buffer> {
        if let Some((_, replica)) = self.cluster_ddp.borrow().as_ref() {
            return replica.buffers();
        }
        let mut bufs = Vec::new();
        let mut seen = HashSet::new();

        for &ni in &self.order {
            if let Some(ref module) = self.nodes[ni].module {
                crate::nn::walk_modules_visited(
                    module.as_ref(),
                    &mut HashSet::new(),
                    &mut |m: &dyn crate::nn::Module| {
                        for b in m.buffers() {
                            let ptr = Rc::as_ptr(&b.inner) as usize;
                            if seen.insert(ptr) {
                                bufs.push(b);
                            }
                        }
                    },
                );
            }
        }

        bufs
    }

    fn set_training(&self, training: bool) {
        if let Some((_, replica)) = self.cluster_ddp.borrow().as_ref() {
            replica.set_training(training);
            return;
        }
        let mut visited = HashSet::new();
        for &ni in &self.order {
            if let Some(ref module) = self.nodes[ni].module {
                crate::nn::walk_modules_visited(
                    module.as_ref(),
                    &mut visited,
                    &mut |m: &dyn crate::nn::Module| m.set_training(training),
                );
            }
        }
    }

    fn move_to_device(&self, device: crate::tensor::Device) {
        self.set_device(device);
    }
}


// ---------------------------------------------------------------------------
// GraphEpochIterator
// ---------------------------------------------------------------------------

/// Iterator over training batches, returned by [`Graph::epoch`].
///
/// Delegates to the attached DataLoader's epoch iterator. Yields
/// `Result<Batch>`.
pub enum GraphEpochIterator<'a> {
    /// Delegates to DataLoader's epoch iterator.
    Single(&'a Graph, usize),
}

/// Internal state once iteration starts (lazily initialized on first next()).
enum GraphEpochState<'a> {
    SingleActive(crate::data::EpochIterator<'a>),
}

/// Active graph epoch iterator (initialized from GraphEpochIterator on first call).
pub struct ActiveGraphEpochIterator<'a> {
    state: GraphEpochState<'a>,
    #[allow(dead_code)]
    graph: &'a Graph,
}

impl<'a> GraphEpochIterator<'a> {
    /// Activate the iterator (must be called to start iteration).
    /// This resolves the borrow on the DataLoader binding.
    pub fn activate(self) -> ActiveGraphEpochIterator<'a> {
        match self {
            GraphEpochIterator::Single(graph, epoch) => {
                // Need a mutable borrow to call epoch() on the inner DataLoader.
                let mut binding = graph.data_binding.borrow_mut();
                let binding = binding.as_mut().unwrap();
                let loader_ptr = &mut binding.loader as *mut crate::data::DataLoader;

                // Safety: the DataLoader lives in the Graph's DataLoaderBinding.
                // We create an EpochIterator that borrows from it. The Graph
                // outlives the iterator. No concurrent mutation occurs during
                // iteration.
                let iter = unsafe { (*loader_ptr).epoch(epoch) };
                ActiveGraphEpochIterator {
                    state: GraphEpochState::SingleActive(iter),
                    graph,
                }
            }
        }
    }
}

impl Iterator for ActiveGraphEpochIterator<'_> {
    type Item = Result<crate::data::Batch>;

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.state {
            GraphEpochState::SingleActive(iter) => iter.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match &self.state {
            GraphEpochState::SingleActive(iter) => iter.size_hint(),
        }
    }
}

impl ExactSizeIterator for ActiveGraphEpochIterator<'_> {}

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
#[path = "graph_tests.rs"]
mod tests;

