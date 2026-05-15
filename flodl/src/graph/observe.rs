use crate::autograd::Variable;
use crate::tensor::{Result, TensorError};

use super::trend::{Trend, TrendGroup};
use super::Graph;

/// Reduction strategy for non-scalar tagged outputs in collect_with().
pub enum Reduce {
    /// Arithmetic mean over all elements.
    Mean,
    /// Sum of all elements.
    Sum,
    /// Maximum element value.
    Max,
    /// Minimum element value.
    Min,
    /// L2 norm (Frobenius norm for matrices).
    Norm,
}

impl Reduce {
    fn apply(&self, var: &Variable) -> Result<f64> {
        let t = var.data();
        if t.numel() == 0 {
            return Err(TensorError::new("cannot reduce empty tensor"));
        }
        let scalar = match self {
            Reduce::Mean => t.mean()?,
            Reduce::Sum  => t.sum()?,
            Reduce::Max  => t.max()?,
            Reduce::Min  => t.min()?,
            Reduce::Norm => t.norm()?,
        };
        scalar.item()
    }
}

impl Graph {
    /// Get the output of a tagged node from the last forward pass.
    pub fn tagged(&self, tag: &str) -> Option<Variable> {
        self.tagged_outputs.borrow().get(tag).cloned()
    }

    /// Get all tag names defined in this graph.
    pub fn tag_names(&self) -> Vec<String> {
        self.tag_names.keys().cloned().collect()
    }

    /// Snapshot current scalar values of tagged nodes into the batch buffer.
    /// Returns an error if any tag has a non-scalar output — use collect_with()
    /// with an explicit reduction for non-scalar tags.
    pub fn collect(&self, tags: &[&str]) -> Result<()> {
        let tagged = self.tagged_outputs.borrow();
        let mut buffer = self.batch_buffer.borrow_mut();
        let mut order = self.metric_order.borrow_mut();
        for &tag in tags {
            if let Some(var) = tagged.get(tag) {
                match var.item() {
                    Ok(val) => {
                        if !buffer.contains_key(tag) && !order.iter().any(|n| n == tag) {
                            order.push(tag.to_string());
                        }
                        buffer.entry(tag.to_string()).or_default().push(val);
                    }
                    Err(_) => {
                        return Err(TensorError::new(&format!(
                            "tag {:?} has shape {:?} (not scalar); use collect_with() to specify a reduction",
                            tag, var.shape()
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    /// Snapshot tagged node values into the batch buffer using a reduction.
    /// Tag group names are automatically expanded to their members.
    /// Each tag's output is reduced to a scalar and recorded individually.
    pub fn collect_with(&self, tags: &[&str], reduce: Reduce) -> Result<()> {
        let expanded = self.expand_groups(tags);
        let tagged = self.tagged_outputs.borrow();
        let mut buffer = self.batch_buffer.borrow_mut();
        let mut order = self.metric_order.borrow_mut();
        for tag in &expanded {
            if let Some(var) = tagged.get(tag) {
                // Scalar tags work directly, non-scalar get reduced
                let val = match var.item() {
                    Ok(v) => v,
                    Err(_) => reduce.apply(var)?,
                };
                if !buffer.contains_key(tag.as_str()) && !order.iter().any(|n| n == tag) {
                    order.push(tag.clone());
                }
                buffer.entry(tag.clone()).or_default().push(val);
            }
        }
        Ok(())
    }

    /// Inject external scalar values into the batch buffer.
    ///
    /// Recorded values accumulate per step and are averaged on
    /// [`flush()`](Self::flush). Use [`trend()`](Self::trend) to read epoch
    /// history for training decisions (early stopping, LR scheduling).
    ///
    /// For human-facing output (terminal, live dashboard), use
    /// [`Monitor::log()`](crate::monitor::Monitor::log) instead.
    pub fn record(&self, tag: &str, values: &[f64]) {
        let mut buffer = self.batch_buffer.borrow_mut();
        if !buffer.contains_key(tag) {
            let mut order = self.metric_order.borrow_mut();
            if !order.iter().any(|n| n == tag) {
                order.push(tag.to_string());
            }
        }
        buffer.entry(tag.to_string()).or_default().extend_from_slice(values);
    }

    /// Record a single scalar value. Convenience wrapper around [`record`](Self::record).
    pub fn record_scalar(&self, tag: &str, value: f64) {
        self.record(tag, &[value]);
    }

    /// Return the latest epoch value for every tag in the epoch history.
    ///
    /// **Tree-aware**: automatically collects from labeled child subgraphs
    /// with dotted prefixes (e.g. a child labeled `"subscan"` with tag `"ce"`
    /// appears as `"subscan.ce"`). Parent metrics come first, then children
    /// in registration order.
    ///
    /// Useful for bridging graph observation into
    /// [`Monitor::log()`](crate::monitor::Monitor::log). Returns an empty
    /// vec if no epochs have been flushed yet.
    ///
    /// Use [`latest_metrics_local()`](Self::latest_metrics_local) if you
    /// only want this graph's own metrics.
    pub fn latest_metrics(&self) -> Vec<(String, f64)> {
        let mut metrics = self.latest_metrics_local();
        // Collect from labeled children with dotted prefixes
        for (label, &ni) in &self.children {
            if let Some(ref module) = self.nodes[ni].module
                && let Some(child) = module.as_graph()
            {
                for (tag, val) in child.latest_metrics() {
                    metrics.push((format!("{}.{}", label, tag), val));
                }
            }
        }
        metrics
    }

    /// Return latest epoch values for this graph only, without child metrics.
    ///
    /// Use this when you need only the local metrics (e.g. when children
    /// report on a different cadence). See [`latest_metrics()`](Self::latest_metrics)
    /// for the tree-recursive version.
    pub fn latest_metrics_local(&self) -> Vec<(String, f64)> {
        let history = self.epoch_history.borrow();
        let order = self.metric_order.borrow();
        order
            .iter()
            .filter_map(|tag| {
                history.get(tag).and_then(|vals| vals.last().map(|&v| (tag.clone(), v)))
            })
            .collect()
    }

    /// Read raw batch buffer for a tag (all values since last flush).
    pub fn collected(&self, tag: &str) -> Vec<f64> {
        self.batch_buffer.borrow().get(tag).cloned().unwrap_or_default()
    }

    /// Compute batch means, append to epoch history, clear batch buffer.
    /// Call once per epoch. If tags is empty, flushes all buffered tags.
    ///
    /// **Tree-aware**: automatically recurses into labeled child subgraphs,
    /// so a single `parent.flush(&[])` flushes the entire tree. Child buffers
    /// that are already empty (e.g. flushed separately) are skipped safely.
    ///
    /// If you need **different flush cadences** per subgraph (e.g. flushing a
    /// child every 10 parent epochs), use [`flush_local()`](Self::flush_local)
    /// on both the parent and the child to manage them independently:
    ///
    /// ```ignore
    /// // Every epoch: flush parent only
    /// parent.flush_local(&[]);
    /// // Every 10 epochs: flush the child
    /// if epoch % 10 == 0 {
    ///     parent.child_graph("slow_child").unwrap().flush_local(&[]);
    /// }
    /// ```
    pub fn flush(&self, tags: &[&str]) {
        self.flush_local(tags);
        // Recurse into labeled children
        for &ni in self.children.values() {
            if let Some(ref module) = self.nodes[ni].module
                && let Some(child) = module.as_graph()
            {
                child.flush(&[]);
            }
        }
    }

    /// Flush only this graph's own batch buffer, without recursing into children.
    ///
    /// Use this when you need independent flush cadences per subgraph.
    /// See [`flush()`](Self::flush) for the tree-recursive version.
    pub fn flush_local(&self, tags: &[&str]) {
        let mut buffer = self.batch_buffer.borrow_mut();
        let mut history = self.epoch_history.borrow_mut();

        let keys: Vec<String> = if tags.is_empty() {
            buffer.keys().cloned().collect()
        } else {
            tags.iter().map(|t| t.to_string()).collect()
        };

        let mut flushed_any = false;
        for key in &keys {
            if let Some(values) = buffer.remove(key)
                && !values.is_empty()
            {
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                history.entry(key.clone()).or_default().push(mean);
                flushed_any = true;
            }
        }

        if flushed_any {
            let count = self.flush_count.get();
            self.flush_count.set(count + 1);
            self.flush_times.borrow_mut().push(
                super::instant_secs() - self.training_start.get(),
            );
        }
    }

    /// Number of flush calls that produced data.
    pub fn flush_count(&self) -> usize {
        self.flush_count.get()
    }

    /// Get epoch-level trend for a tag.
    pub fn trend(&self, tag: &str) -> Trend {
        let history = self.epoch_history.borrow();
        Trend::new(history.get(tag).cloned().unwrap_or_default())
    }

    /// Get trends for multiple tags. Tag group names are automatically
    /// expanded to their member tags.
    pub fn trends(&self, tags: &[&str]) -> TrendGroup {
        let expanded = self.expand_groups(tags);
        let history = self.epoch_history.borrow();
        let trends = expanded
            .iter()
            .map(|tag| Trend::new(history.get(tag).cloned().unwrap_or_default()))
            .collect();
        TrendGroup(trends)
    }

    /// Clear epoch history. If tags is empty, clears all.
    /// Tag group names are automatically expanded.
    pub fn reset_trend(&self, tags: &[&str]) {
        let mut history = self.epoch_history.borrow_mut();
        if tags.is_empty() {
            history.clear();
        } else {
            let expanded = self.expand_groups(tags);
            for tag in &expanded {
                history.remove(tag);
            }
        }
    }

    /// Get per-iteration trace outputs from loop nodes.
    ///
    /// `name` may identify either:
    /// - a post-loop tag (legacy [`crate::nn::Module::trace`] path), or
    /// - an emit name published by a [`crate::nn::LoopBody`] body via
    ///   [`crate::nn::TraceEmit::publish`].
    ///
    /// On the first call, validates that emit names don't collide with
    /// post-loop tags or with each other across loops, panicking with a clear
    /// message on collision (cached after first successful validation).
    ///
    /// Returns `None` if no matching trace buffer is found.
    pub fn traces(&self, name: &str) -> Option<Vec<Variable>> {
        self.validate_trace_namespace();

        // 1) Legacy path: post-loop tag with trace_buf
        if let Some(&(ni, _)) = self.tag_names.get(name)
            && let Some(ref buf) = self.nodes[ni].trace_buf
        {
            let traces = buf.borrow().clone();
            if !traces.is_empty() {
                return Some(traces);
            }
        }

        // 2) Named-emit path: any loop's named_trace_buf carrying this name
        for node in &self.nodes {
            if let Some(ref store) = node.named_trace_buf
                && let Some(traces) = store.borrow().get(name)
                && !traces.is_empty()
            {
                return Some(traces.clone());
            }
        }

        // 3) Legacy fallback: first loop with non-empty trace_buf
        for node in &self.nodes {
            if let Some(ref buf) = node.trace_buf {
                let traces = buf.borrow().clone();
                if !traces.is_empty() && node.id.contains("loop") {
                    return Some(traces);
                }
            }
        }
        None
    }

    /// Get trace buffer directly from a loop node by node ID.
    pub fn traces_by_node(&self, node_id: &str) -> Option<Vec<Variable>> {
        if let Some(&ni) = self.node_index.get(node_id)
            && let Some(ref buf) = self.nodes[ni].trace_buf
        {
            let traces = buf.borrow().clone();
            if !traces.is_empty() {
                return Some(traces);
            }
        }
        None
    }

    /// Get a single named-emit trace stream from any loop node in this graph.
    ///
    /// Equivalent to [`Self::traces`] but restricted to emit-published names
    /// from [`crate::nn::LoopBody::step`] (no legacy tag fallback).
    pub fn traces_named(&self, name: &str) -> Option<Vec<Variable>> {
        self.validate_trace_namespace();
        for node in &self.nodes {
            if let Some(ref store) = node.named_trace_buf
                && let Some(traces) = store.borrow().get(name)
                && !traces.is_empty()
            {
                return Some(traces.clone());
            }
        }
        None
    }

    /// Validate the trace namespace exactly once per graph: panics if any
    /// emit-published name collides with a legacy post-loop tag or with
    /// another loop's emit name. Cheap on cached path (single Cell load).
    pub(crate) fn validate_trace_namespace(&self) {
        use std::collections::{HashMap as Hm, HashSet as Hs};
        if self.traces_validated.get() {
            return;
        }

        // Legacy: tags whose tagged node has a trace_buf
        let mut legacy_names: Hs<String> = Hs::new();
        for (name, &(ni, _)) in &self.tag_names {
            if self.nodes[ni].trace_buf.is_some() {
                legacy_names.insert(name.clone());
            }
        }

        // Named emits: walk every loop node's named_trace_buf
        let mut seen: Hm<String, String> = Hm::new(); // emit name -> first node id
        for node in &self.nodes {
            if let Some(ref store) = node.named_trace_buf {
                let store_b = store.borrow();
                for emit_name in store_b.keys() {
                    if legacy_names.contains(emit_name) {
                        panic!(
                            "trace namespace collision: emit name {:?} from loop {:?} \
                             conflicts with a legacy post-loop trace tag of the same name",
                            emit_name, node.id
                        );
                    }
                    if let Some(prev) = seen.get(emit_name)
                        && prev != &node.id
                    {
                        panic!(
                            "trace namespace collision: emit name {:?} published by \
                             both loop {:?} and loop {:?}",
                            emit_name, prev, node.id
                        );
                    }
                    seen.insert(emit_name.clone(), node.id.clone());
                }
            }
        }

        self.traces_validated.set(true);
    }

    /// Get the last trace output from the most recent loop iteration.
    ///
    /// Convenience wrapper around [`traces()`](Self::traces) that returns only
    /// the final iteration's trace. Useful for chaining loops where the last
    /// output of one (e.g. scan) feeds into the next (e.g. read).
    ///
    /// Returns `None` if the tag has no associated loop or the body produced
    /// no traces.
    pub fn last_trace(&self, tag: &str) -> Option<Variable> {
        self.traces(tag).and_then(|v| v.into_iter().last())
    }

    /// Estimated time remaining based on average flush duration.
    ///
    /// Returns seconds remaining. Returns 0.0 if no flushes have occurred yet.
    pub fn eta(&self, total_epochs: usize) -> f64 {
        let count = self.flush_count.get();
        if count == 0 {
            return 0.0;
        }
        let times = self.flush_times.borrow();
        let elapsed = times[count - 1]; // already relative to training_start
        let per_flush = elapsed / count as f64;
        let remaining = total_epochs.saturating_sub(count);
        per_flush * remaining as f64
    }

    /// Expand tag group names into their member tags.
    /// Non-group tags pass through unchanged.
    pub(crate) fn expand_groups(&self, tags: &[&str]) -> Vec<String> {
        let mut expanded = Vec::new();
        for &tag in tags {
            if let Some(members) = self.tag_groups.get(tag) {
                expanded.extend(members.iter().cloned());
            } else {
                expanded.push(tag.to_string());
            }
        }
        expanded
    }
}
