use crate::autograd::Variable;
use crate::nn::Module;
use crate::tensor::{Result, TensorError};

use super::execution::GraphEpochIterator;
use super::graph::{DataLoaderBinding, Graph};
use super::node::DEFAULT_INPUT;

// ---------------------------------------------------------------------------
// Optimizer + training-step integration (single-device)
// ---------------------------------------------------------------------------

impl Graph {
    /// Set the optimizer for training.
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
        sched
            .as_ref()
            .map(|s| s.lr(self.training_step.get()) * self.lr_scale.get())
    }

    /// Perform one training step: step the optimizer, then zero grad.
    ///
    /// When a scheduler is attached via [`Self::set_scheduler`], the
    /// optimizer's LR is updated from `scheduler.lr(training_step) *
    /// lr_scale` before the step, and `training_step` increments by one
    /// after.
    pub fn step(&self) -> Result<()> {
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
    /// Stores the loader and enables `model.epoch()` (which delegates to the
    /// loader's epoch iterator) plus `model.forward_batch(&batch)` for
    /// auto-wired forward passes.
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
        mut loader: crate::data::DataLoader,
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

        // Cover the params-proportional share of the first training
        // step's allocations in the loader's streaming VRAM sizing:
        // gradients (~1x parameter bytes) plus lazily created optimizer
        // state (~2x for Adam-family m/v) do not exist when the loader
        // first probes VRAM, but their size is known exactly from the
        // model. Activations remain unknowable from parameter count and
        // stay covered by the loader's first-fill discount. Never
        // overrides a user-declared reserve; no-op for resident/CPU
        // loaders.
        let param_bytes: usize = crate::nn::Module::parameters(self)
            .iter()
            .map(|p| p.variable.data().nbytes())
            .sum();
        loader.set_activation_reserve_auto(param_bytes.saturating_mul(3));

        // Refuse to replace the loader while an epoch iterator holds it:
        // the iterator owns the cell's exclusive borrow, and replacing the
        // loader would drop it out from under the live iteration.
        let mut loader_cell = self.data_loader.try_borrow_mut().map_err(|_| {
            TensorError::new(
                "set_data_loader: cannot replace the data loader while an epoch iterator is active",
            )
        })?;
        let num_batches = loader.num_batches();
        let batch_size = loader.batch_size();
        *loader_cell = Some(loader);
        *self.data_binding.borrow_mut() = Some(DataLoaderBinding {
            forward_input: forward_input.to_string(),
            graph_inputs,
            target_names,
            shard_input_map,
            num_batches,
            batch_size,
        });

        Ok(())
    }

    /// Get an epoch iterator for integrated training.
    ///
    /// Delegates to the attached DataLoader's epoch iterator.
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

    /// Number of batches per epoch (cached from the DataLoader at bind
    /// time; readable while an epoch iterator is active).
    pub fn data_num_batches(&self) -> usize {
        self.data_binding
            .borrow()
            .as_ref()
            .expect("call set_data_loader first")
            .num_batches
    }

    /// Batch size (cached from the DataLoader at bind time; readable
    /// while an epoch iterator is active).
    pub fn data_batch_size(&self) -> usize {
        self.data_binding
            .borrow()
            .as_ref()
            .expect("call set_data_loader first")
            .batch_size
    }

    /// Batch-aware forward pass.
    ///
    /// Extracts the primary input and auxiliary graph inputs from the named
    /// Batch and runs the graph forward.
    ///
    /// ```ignore
    /// let out = model.forward_batch(&b)?;
    /// let loss = mse_loss(&out, &b["letter"])?;
    /// ```
    pub fn forward_batch(&self, batch: &crate::data::Batch) -> Result<Variable> {
        // Scope the borrow so it is released before calling methods that re-borrow.
        let (forward_input_name, shard_input_map) = {
            let guard = self.data_binding.borrow();
            let binding = guard.as_ref().ok_or_else(|| {
                TensorError::new("Graph::forward_batch: call set_data_loader() first")
            })?;
            (
                binding.forward_input.clone(),
                binding.shard_input_map.clone(),
            )
        };

        // Build full input vector from batch using shard_input_map.
        let batch_names = batch.names();
        let graph_inputs: Vec<Variable> = shard_input_map
            .iter()
            .map(|&idx| Variable::new(batch[batch_names[idx].as_str()].clone(), false))
            .collect();

        if graph_inputs.is_empty() {
            return Err(TensorError::new(&format!(
                "forward_batch: batch missing forward input '{}'",
                forward_input_name,
            )));
        }

        if graph_inputs.len() == 1 {
            use crate::nn::Module;
            return Module::forward(self, &graph_inputs[0]);
        }
        self.forward_multi(&graph_inputs)
    }
}
