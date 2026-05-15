use crate::autograd::Variable;
use crate::nn::Module;
use crate::tensor::{Result, Tensor, TensorError};

use super::node::DEFAULT_INPUT;
use super::graph::{Graph, DataLoaderBinding, GraphEpochIterator};
use super::LossContext;

// ---------------------------------------------------------------------------
// Distributed Data Parallel + optimizer integration
// ---------------------------------------------------------------------------

impl Graph {
    /// Attach cluster-mode DDP state (process-per-rank).
    ///
    /// Called from [`Trainer::setup`](crate::distributed::Trainer::setup) when
    /// the launcher has shipped a per-node envelope (detected via
    /// [`LocalCluster::from_env`](crate::distributed::LocalCluster::from_env)).
    /// The caller has already built the local replica on this rank's device
    /// and joined the cross-process NCCL group via [`Ddp::wrap`].
    ///
    /// From this point on, the [`Module`] surface (`forward`, `parameters`,
    /// `buffers`, `set_training`) short-circuits to `replica`; `self` acts as
    /// a structural template only. [`Graph::step`] dispatches AllReduce
    /// through `ddp` and steps the local optimizer.
    pub fn set_cluster_ddp(
        &self,
        ddp: crate::distributed::ddp::Ddp,
        replica: Box<dyn crate::nn::Module>,
    ) {
        *self.cluster_ddp.borrow_mut() = Some((ddp, replica));
    }

    /// Attach cluster-mode El Che cadence state.
    ///
    /// Called from [`Trainer::setup`](crate::distributed::Trainer::setup) /
    /// [`Trainer::setup_with`] when heterogeneous cluster DDP is in play.
    /// From this point on, [`Graph::step`] defers the actual sync +
    /// optimizer step until the local cadence target is reached; cross-rank
    /// timing AllReduce keeps every rank's anchor in lockstep.
    ///
    /// Must be called after [`Graph::set_cluster_ddp`] — cluster El Che has
    /// no meaning outside cluster mode.
    pub(crate) fn set_cluster_el_che(
        &self,
        state: crate::distributed::ddp::ClusterElCheState,
    ) {
        *self.cluster_el_che.borrow_mut() = Some(state);
    }

    /// Whether cluster-mode El Che cadence is active.
    pub fn has_cluster_el_che(&self) -> bool {
        self.cluster_el_che.borrow().is_some()
    }

    /// Set the optimizer for training. When distributed, creates one optimizer
    /// per replica. When single-GPU, creates a single optimizer.
    ///
    /// The factory receives the parameter list and returns an optimizer.
    ///
    /// ```ignore
    /// model.set_optimizer(|p| Adam::new(p, 0.001));
    /// ```
    pub fn set_optimizer<F, O>(&self, factory: F)
    where
        F: Fn(&[crate::nn::Parameter]) -> O,
        O: crate::nn::Optimizer + 'static,
    {
        // `self.parameters()` short-circuits to the cluster-mode replica
        // when set; single-device falls through to the in-process graph's
        // params. Either way, one optimizer over the local parameters.
        let opt = factory(&self.parameters());
        *self.optimizer.borrow_mut() = Some(Box::new(opt));
    }

    /// Attach a per-batch LR scheduler.
    ///
    /// When set, `step()` updates every optimizer's learning rate to
    /// `scheduler.lr(training_step) * lr_scale` before the optimizer step.
    /// The internal `training_step` counter increments once per `step()`
    /// call and is independent of the recurrent-state `step_count`.
    ///
    /// Works for both single-GPU and distributed graphs.
    ///
    /// ```ignore
    /// use std::sync::Arc;
    /// let sched: Arc<dyn Scheduler> = Arc::new(MultiStepLR::new(0.1, &[100, 150], 0.1));
    /// graph.set_scheduler(sched);
    /// ```
    pub fn set_scheduler(&self, scheduler: std::sync::Arc<dyn crate::nn::Scheduler>) {
        *self.scheduler.borrow_mut() = Some(scheduler);
    }

    /// Set the DDP linear-scaling factor (Goyal et al., 2017) applied to the
    /// attached scheduler's output every batch. Defaults to 1.0 (no scaling).
    ///
    /// Has no effect if no scheduler is attached; bake the scaling into the
    /// optimizer's base LR instead for that case.
    pub fn set_lr_scale(&self, scale: f64) {
        self.lr_scale.set(scale);
    }

    /// Current training step (increments once per `step()` call). Used by the
    /// attached scheduler, if any.
    pub fn training_step(&self) -> usize {
        self.training_step.get()
    }

    /// Compute the scheduled LR for the current training step, if a
    /// scheduler is attached. Returns `None` when no scheduler is set so
    /// the caller can leave the optimizer LR alone.
    fn scheduled_lr(&self) -> Option<f64> {
        let sched = self.scheduler.borrow();
        sched.as_ref()
            .map(|s| s.lr(self.training_step.get()) * self.lr_scale.get())
    }

    /// Perform one training step.
    ///
    /// - **Cluster mode (process-per-rank)**: AllReduce gradients across the
    ///   cluster (weighted under El Che cadence), sync buffers, step the
    ///   local optimizer, zero grad. Under El Che, `step()` is conditional
    ///   and only fires the sync + optimizer at cadence boundaries.
    /// - **Single-GPU / CPU**: step optimizer, zero grad.
    ///
    /// When a scheduler is attached via [`Self::set_scheduler`], the
    /// optimizer's LR is updated from `scheduler.lr(training_step) *
    /// lr_scale` before the step, and `training_step` increments by one
    /// after.
    pub fn step(&self) -> Result<()> {
        // Cluster mode (process-per-rank): single local replica + Ddp,
        // single local optimizer. Two paths:
        //   * El Che enabled  → step() is conditional. Accumulates
        //     gradients locally until the cadence target is reached, then
        //     runs the cross-rank timing AllReduce, weighted gradient
        //     AllReduce, and optimizer step.
        //   * El Che disabled → AllReduce gradients every batch
        //     (traditional DDP), then optimizer step.
        if self.cluster_ddp.borrow().is_some() {
            if self.cluster_el_che.borrow().is_some() {
                return self.cluster_step_el_che();
            }
            return self.cluster_step_plain();
        }

        // Single-GPU / CPU: just step the local optimizer.
        let scheduled = self.scheduled_lr();
        let mut opt = self.optimizer.borrow_mut();
        if let Some(ref mut optimizer) = *opt {
            if let Some(lr) = scheduled {
                optimizer.set_lr(lr);
            }
            optimizer.step()?;
            optimizer.zero_grad();
        }
        self.training_step.set(self.training_step.get() + 1);
        Ok(())
    }

    /// Plain cluster-mode step (no El Che): per-batch AllReduce + optimizer.
    fn cluster_step_plain(&self) -> Result<()> {
        let cluster = self.cluster_ddp.borrow();
        let (ddp, _replica) = cluster.as_ref().unwrap();
        ddp.all_reduce_gradients()?;
        ddp.sync_buffers()?;
        drop(cluster);

        let scheduled = self.scheduled_lr();
        let mut opt = self.optimizer.borrow_mut();
        if let Some(ref mut optimizer) = *opt {
            if let Some(lr) = scheduled {
                optimizer.set_lr(lr);
            }
            optimizer.step()?;
            optimizer.zero_grad();
        }
        self.training_step.set(self.training_step.get() + 1);
        Ok(())
    }

    /// Cluster-mode step with El Che cadence (heterogeneous DDP).
    ///
    /// Accumulates gradients locally until `local_batch_idx` reaches the
    /// El Che target for this rank, then:
    /// 1. Cross-rank timing AllReduce (everyone sees per-rank wall_ms).
    /// 2. Per-rank gradient clipping (if `max_grad_norm` set).
    /// 3. Normalize accumulated gradients by local count.
    /// 4. Weighted AllReduce on gradients (scale-by-count then Sum).
    /// 5. Buffer broadcast.
    /// 6. [`ElChe::report_timing`] — anchor + ratios adapt deterministically
    ///    on every rank from the same input vector, so all ranks agree on
    ///    next-cycle counts without a separate broadcast.
    /// 7. Optimizer step + zero-grad.
    /// 8. Reset cycle counter; training_step += 1.
    fn cluster_step_el_che(&self) -> Result<()> {
        // Reserve the cluster borrow scope for the whole sync — we need the
        // Ddp handle for both timing AllReduce and weighted gradient
        // AllReduce, and the replica reference for parameter scaling/clip.
        let (my_rank, world_size, target) = {
            let cluster = self.cluster_ddp.borrow();
            let (ddp, _replica) = cluster.as_ref().unwrap();
            let my_rank = ddp.rank();
            let world_size = ddp.world_size();
            let mut state_ref = self.cluster_el_che.borrow_mut();
            let state = state_ref.as_mut().unwrap();
            if state.cycle_start.is_none() {
                state.cycle_start = Some(std::time::Instant::now());
            }
            state.local_batch_idx += 1;
            let target = state.el_che.batch_counts()[my_rank];
            if state.local_batch_idx < target {
                // Accumulate more; no sync this call.
                return Ok(());
            }
            (my_rank, world_size, target)
        };

        // Cadence boundary: compute wall time for this cycle, AllReduce
        // timings across all ranks.
        let cycle_wall_ms = {
            let state_ref = self.cluster_el_che.borrow();
            let state = state_ref.as_ref().unwrap();
            state.cycle_start.unwrap().elapsed().as_secs_f64() * 1000.0
        };

        let mut wall_ms_vec = vec![0.0_f64; world_size];
        wall_ms_vec[my_rank] = cycle_wall_ms;
        {
            let cluster = self.cluster_ddp.borrow();
            let (ddp, _replica) = cluster.as_ref().unwrap();
            ddp.all_reduce_per_rank_f64(&mut wall_ms_vec)?;
        }

        // Snapshot current counts (used by both gradient ops and reporting).
        let counts: Vec<usize> = {
            let state_ref = self.cluster_el_che.borrow();
            let state = state_ref.as_ref().unwrap();
            state.el_che.batch_counts().to_vec()
        };

        // Per-rank gradient clipping (before normalize-by-count). Operates on
        // the replica's parameters since that's what backward() populated.
        {
            let cluster = self.cluster_ddp.borrow();
            let (_, replica) = cluster.as_ref().unwrap();
            let max_grad_norm = self.cluster_el_che.borrow()
                .as_ref().unwrap().max_grad_norm;
            if let Some(max_norm) = max_grad_norm {
                let param_tensors: Vec<Tensor> = replica
                    .parameters()
                    .into_iter()
                    .filter(|p| p.variable.grad().is_some())
                    .map(|p| p.variable.data())
                    .collect();
                if !param_tensors.is_empty() {
                    Tensor::clip_grad_norm_fused(&param_tensors, max_norm)?;
                }
            }

            // Normalize accumulated gradients by local count. Each rank
            // ran `target` backward passes, so without this the optimizer
            // would see grads `target`× too large.
            if target > 1 {
                for p in replica.parameters() {
                    if let Some(g) = p.variable.grad() {
                        let _ = g.mul_scalar_(1.0 / target as f64);
                    }
                }
            }
        }

        // Weighted gradient AllReduce + buffer sync.
        let sync_start = std::time::Instant::now();
        {
            let cluster = self.cluster_ddp.borrow();
            let (ddp, _replica) = cluster.as_ref().unwrap();
            ddp.weighted_all_reduce_gradients(&counts)?;
            ddp.sync_buffers()?;
        }
        let sync_ms = sync_start.elapsed().as_secs_f64() * 1000.0;

        // Report timing → ElChe anchor + ratios adapt. All ranks call this
        // with identical inputs (post-AllReduce wall_ms_vec, same counts,
        // same sync_ms scaled to local clock); deterministic state stays
        // coherent across processes without a broadcast.
        {
            let mut state_ref = self.cluster_el_che.borrow_mut();
            let state = state_ref.as_mut().unwrap();
            state.el_che.report_timing(&wall_ms_vec, &counts, sync_ms);
            state.local_batch_idx = 0;
            state.cycle_start = None;
        }

        // Optimizer step.
        let scheduled = self.scheduled_lr();
        let mut opt = self.optimizer.borrow_mut();
        if let Some(ref mut optimizer) = *opt {
            if let Some(lr) = scheduled {
                optimizer.set_lr(lr);
            }
            optimizer.step()?;
            optimizer.zero_grad();
        }
        self.training_step.set(self.training_step.get() + 1);
        Ok(())
    }

    /// Number of devices in use (1 if not distributed).
    ///
    /// In cluster mode, returns the cross-process world size from the
    /// rendezvous; this process owns one of those slots.
    pub fn world_size(&self) -> usize {
        if let Some((ddp, _)) = self.cluster_ddp.borrow().as_ref() {
            return ddp.world_size();
        }
        1
    }

    /// Whether this graph is running in distributed mode.
    pub fn is_distributed(&self) -> bool {
        self.cluster_ddp.borrow().is_some()
    }

    /// Whether El Che cadence is active (heterogeneous DDP).
    ///
    /// Alias for [`Self::has_cluster_el_che`] — the in-process El Che path
    /// has been removed; the only El Che that exists now is cluster-mode.
    pub fn has_el_che(&self) -> bool {
        self.has_cluster_el_che()
    }

    /// Set learning rate on the local optimizer.
    pub fn set_lr(&self, lr: f64) {
        let mut opt = self.optimizer.borrow_mut();
        if let Some(ref mut optimizer) = *opt {
            optimizer.set_lr(lr);
        }
    }

    // -- DataLoader integration -----------------------------------------------

    /// Attach a DataLoader for integrated training.
    ///
    /// When distributed: upgrades the loader to per-device backends (resident
    /// or streaming per device based on VRAM). Enables `model.epoch()` for
    /// zero-transfer iteration and `model.forward(&batch)` for auto-wired
    /// forward passes.
    ///
    /// When single-GPU: stores the loader as-is. `model.epoch()` delegates
    /// to `loader.epoch()` directly.
    ///
    /// The `forward_input` parameter names the batch field used as the primary
    /// model input (e.g., "image"). Other batch fields that match graph
    /// `.input()` ports are auto-wired as auxiliary inputs. All remaining
    /// batch fields are treated as targets (available in the user-facing
    /// Batch for loss computation).
    ///
    /// ```ignore
    /// model.set_data_loader(loader, "image")?;
    /// ```
    pub fn set_data_loader(
        &self,
        loader: crate::data::DataLoader,
        forward_input: &str,
    ) -> Result<()> {
        let loader_names: Vec<String> = loader.names().to_vec();

        // Validate forward_input exists in loader names
        if !loader_names.iter().any(|n| n == forward_input) {
            return Err(TensorError::new(&format!(
                "set_data_loader: forward_input '{}' not found in loader names [{}]",
                forward_input,
                loader_names.join(", ")
            )));
        }

        // Match batch names to graph Input ports
        let graph_input_names: Vec<String> = self.inputs.iter().map(|i| i.name.clone()).collect();
        let mut graph_inputs: Vec<(String, String)> = Vec::new();
        let mut target_names: Vec<String> = Vec::new();

        for name in &loader_names {
            if name == forward_input {
                continue; // primary input, handled separately
            }
            if graph_input_names.contains(name) {
                graph_inputs.push((name.clone(), name.clone()));
            } else {
                target_names.push(name.clone());
            }
        }

        // Build shard_input_map: graph input index -> loader tensor position.
        // self.inputs[0] is the entry (forward_input), self.inputs[1..] are .input() ports.
        let mut shard_input_map: Vec<usize> = Vec::with_capacity(self.inputs.len());
        for port in &self.inputs {
            let lookup_name = if port.name == DEFAULT_INPUT {
                forward_input
            } else {
                &port.name
            };
            match loader_names.iter().position(|n| n == lookup_name) {
                Some(idx) => shard_input_map.push(idx),
                None => {
                    return Err(TensorError::new(&format!(
                        "set_data_loader: graph input '{}' not found in loader names [{}]",
                        lookup_name,
                        loader_names.join(", ")
                    )));
                }
            }
        }

        let _ = loader_names; // keep the name list for the future iterator wiring
        *self.data_binding.borrow_mut() = Some(DataLoaderBinding {
            loader,
            forward_input: forward_input.to_string(),
            graph_inputs,
            target_names,
            shard_input_map,
        });

        Ok(())
    }

    /// Register a per-batch loss function for El Che distributed training.
    ///
    /// When set, `forward_distributed_el_che` runs forward + loss + backward
    /// per batch internally, keeping only ONE forward graph in VRAM at a time.
    /// Without this, all forward graphs are held simultaneously (VRAM scales
    /// with anchor * devices), which caps the practical anchor at 1.
    ///
    /// The closure receives a [`LossContext`] with live autograd on all fields.
    /// It must return a scalar loss `Variable`.
    ///
    /// `forward_batch()` returns detached gathered outputs when a loss function
    /// is registered. Tags and traces on the graph are gathered (detached) for
    /// metrics. Calling `.backward()` on the returned Variable is a no-op.
    ///
    /// ```ignore
    /// model.set_loss_fn(|ctx: &LossContext| {
    ///     let cls  = cross_entropy_loss(&ctx.tags["head"], &ctx.batch["label"])?;
    ///     let rec  = mse_loss(&ctx.tags["recon"], &ctx.batch["image"])?;
    ///     Ok(cls + rec)
    /// });
    ///
    /// for batch in model.epoch(epoch).activate() {
    ///     let _metrics = model.forward_batch(&batch?)?;
    ///     model.step()?;
    /// }
    /// ```
    pub fn set_loss_fn<F>(&self, f: F)
    where
        F: Fn(&LossContext) -> Result<Variable> + 'static,
    {
        *self.loss_fn.borrow_mut() = Some(Box::new(f));
    }

    /// Whether a per-batch loss function is registered.
    pub fn has_loss_fn(&self) -> bool {
        self.loss_fn.borrow().is_some()
    }

    /// Get an epoch iterator for integrated training.
    ///
    /// When distributed: returns a `DistributedEpochIterator` that produces
    /// per-rank shards and a user-facing Batch with targets on the gather device.
    /// When single-GPU: delegates to the DataLoader's epoch iterator.
    ///
    /// ```ignore
    /// for batch in model.epoch(epoch) {
    ///     let b = batch?;
    ///     let out = model.forward(&b)?;
    ///     let loss = mse_loss(&out, &b["letter"])?;
    ///     loss.backward()?;
    ///     model.step()?;
    /// }
    /// ```
    pub fn epoch(&self, epoch: usize) -> GraphEpochIterator<'_> {
        let binding = self.data_binding.borrow();
        if binding.is_none() {
            panic!("Graph::epoch() requires set_data_loader() first");
        }
        GraphEpochIterator::Single(self, epoch)
    }

    /// Number of batches per epoch (delegates to the attached DataLoader).
    pub fn data_num_batches(&self) -> usize {
        self.data_binding
            .borrow()
            .as_ref()
            .expect("call set_data_loader first")
            .loader
            .num_batches()
    }

    /// Batch size (delegates to the attached DataLoader).
    pub fn data_batch_size(&self) -> usize {
        self.data_binding
            .borrow()
            .as_ref()
            .expect("call set_data_loader first")
            .loader
            .batch_size()
    }


    /// Batch-aware forward pass.
    ///
    /// Extracts the primary input and auxiliary graph inputs from the named
    /// Batch, handles DDP presharding and El Che transparently.
    ///
    /// ```ignore
    /// let out = model.forward_batch(&b)?;
    /// let loss = mse_loss(&out, &b["letter"])?;
    /// ```
    pub fn forward_batch(&self, batch: &crate::data::Batch) -> Result<Variable> {
        // Scope the borrow so it is released before calling methods that re-borrow.
        let (forward_input_name, shard_input_map) = {
            let guard = self.data_binding.borrow();
            let binding = guard.as_ref()
                .expect("call set_data_loader before forward_batch");
            (binding.forward_input.clone(), binding.shard_input_map.clone())
        };

        // Build full input vector from batch using shard_input_map.
        let batch_names = batch.names();
        let graph_inputs: Vec<Variable> = shard_input_map.iter()
            .map(|&idx| Variable::new(batch[batch_names[idx].as_str()].clone(), false))
            .collect();

        if graph_inputs.is_empty() {
            return Err(TensorError::new(&format!(
                "forward_batch: batch missing forward input '{}'",
                forward_input_name,
            )));
        }

        // In cluster mode `forward` short-circuits to the local replica; in
        // single-device mode it runs the graph in-process. Either way, route
        // through the Module trait surface to honor the cluster_ddp wiring.
        if graph_inputs.len() == 1 {
            use crate::nn::Module;
            return Module::forward(self, &graph_inputs[0]);
        }
        self.forward_multi(&graph_inputs)
    }
}
