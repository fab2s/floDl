//! Graph forward execution and epoch iteration.
//!
//! Split from `graph.rs` (the `Graph` struct + construction home): the
//! forward pass (`forward_impl`, `Module::forward`) and the training-batch
//! epoch iterator. `Graph`'s fields are `pub(crate)`, so these `impl Graph`
//! blocks live here without any visibility widening.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use super::graph::{Graph, instant_secs};
use super::profile;
use crate::autograd::Variable;
use crate::nn::{Module, Parameter};
use crate::tensor::{Result, Tensor, TensorError};

impl Graph {
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
        let forward_start = if is_profiling {
            Some(Instant::now())
        } else {
            None
        };
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
            let level_start = if is_profiling {
                Some(Instant::now())
            } else {
                None
            };
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
                                Ok(Variable::new(Tensor::zeros_like(&first.data())?, false))
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
                let node_start = if is_profiling {
                    Some(Instant::now())
                } else {
                    None
                };
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
                if has_tags && let Some(captures) = self.tag_capture.get(&ni) {
                    let mut tagged = self.tagged_outputs.borrow_mut();
                    for (tag_name, port_idx) in captures {
                        if *port_idx < node_outputs.len() {
                            tagged.insert(tag_name.clone(), node_outputs[*port_idx].clone());
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

impl Module for Graph {
    fn name(&self) -> &str {
        "graph"
    }

    // The identity hook behind `GraphExt::as_graph` — framework code
    // holding `dyn Module` downcasts through this to reach the graph.
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    /// Expose Graph's shared aggregated-metrics slot to the cluster-
    /// rank worker setup. The worker stores an `Arc` clone of THIS
    /// slot so its bridge-thread writes for `ControlMsg::EpochAggregated`
    /// become visible to the user's main-thread reads through
    /// [`Graph::latest_metrics`] / [`Graph::aggregated_gpu_tabs`].
    fn aggregated_metrics_slot(
        &self,
    ) -> Option<std::sync::Arc<std::sync::Mutex<Option<crate::metrics::EpochMetrics>>>> {
        Some(std::sync::Arc::clone(&self.aggregated_metrics))
    }

    fn structural_hash(&self) -> Option<String> {
        Some(self.structural_hash().to_string())
    }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        self.forward_impl(std::slice::from_ref(input))
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        let mut seen = HashSet::new();

        for &ni in &self.order {
            if let Some(ref module) = self.nodes[ni].module {
                for p in module.parameters() {
                    let ptr = p.variable.id();
                    if seen.insert(ptr) {
                        params.push(p);
                    }
                }
            }
        }

        params
    }

    fn buffers(&self) -> Vec<crate::nn::Buffer> {
        let mut bufs = Vec::new();
        let mut seen = HashSet::new();

        for &ni in &self.order {
            if let Some(ref module) = self.nodes[ni].module {
                crate::nn::walk_modules_visited(
                    module.as_ref(),
                    &mut HashSet::new(),
                    &mut |m: &dyn crate::nn::Module| {
                        for b in m.buffers() {
                            let ptr = b.id();
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
    /// Holds the exclusive borrow of the Graph's loader cell for the
    /// iterator's whole lifetime. Declared after `state` so the iterator
    /// (which borrows the loader through a raw pointer) drops first.
    _loader_guard: std::cell::RefMut<'a, Option<crate::data::DataLoader>>,
}

impl<'a> GraphEpochIterator<'a> {
    /// Activate the iterator (must be called to start iteration).
    /// Takes the exclusive borrow of the graph's DataLoader for the
    /// iterator's lifetime.
    ///
    /// # Panics
    ///
    /// Panics if another epoch iterator is already active on this graph,
    /// or if no data loader is bound.
    pub fn activate(self) -> ActiveGraphEpochIterator<'a> {
        match self {
            GraphEpochIterator::Single(graph, epoch) => {
                let mut guard = graph
                    .data_loader
                    .try_borrow_mut()
                    .expect("Graph::epoch: an epoch iterator is already active on this graph");
                let loader = guard
                    .as_mut()
                    .expect("Graph::epoch() requires set_data_loader() first");
                let loader_ptr = loader as *mut crate::data::DataLoader;

                // Safety: `guard` moves into the returned iterator and holds
                // the RefCell's exclusive borrow for the iterator's whole
                // lifetime. The loader is only reachable through that
                // RefCell, so nothing can alias, replace, or drop it while
                // `iter` borrows it: `set_data_loader` and a second
                // `activate()` fail loudly on the dynamic borrow instead.
                // The RefCell lives in the Graph, which `'a` keeps alive and
                // in place, and moving the RefMut guard does not move the
                // loader it points into.
                let iter = unsafe { (*loader_ptr).epoch(epoch) };
                ActiveGraphEpochIterator {
                    state: GraphEpochState::SingleActive(iter),
                    graph,
                    _loader_guard: guard,
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
