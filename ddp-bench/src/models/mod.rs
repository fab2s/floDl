//! Benchmark model definitions.
//!
//! Each model reproduces a published architecture with known convergence
//! curves, enabling verification against literature before DDP comparison.

pub mod char_rnn;
pub mod conv_ae;
pub mod gpt_nano;
pub mod lenet;
pub mod logistic;
pub mod mlp;
pub mod olmo;
pub mod resnet;
pub mod resnet_graph;

use std::path::PathBuf;
use std::sync::Arc;

use flodl::autograd::Variable;
use flodl::data::BatchDataSet;
use flodl::nn::{Module, Optimizer, Parameter, Scheduler};
use flodl::tensor::{Device, Result, Tensor};

use crate::config::ModelDefaults;

/// Where the training data lives during the run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataSource {
    /// Parse the whole dataset into RAM tensors up front (default).
    Ram,
    /// Read per sample from the raw files through flodl's `DataSet`
    /// layer, exercising the storage-read path the staging tiers
    /// absorb. Honored by the CIFAR-10 models (`resnet`,
    /// `resnet-graph`); other models error loudly.
    Disk,
}

/// Model names whose dataset factory honors [`DataSource::Disk`].
pub const DISK_SOURCE_MODELS: [&str; 2] = ["resnet", "resnet-graph"];

/// Dataset configuration passed to each model's dataset factory.
#[allow(dead_code)]
pub struct DatasetConfig {
    pub seed: u64,
    pub data_dir: PathBuf,
    pub virtual_len: usize,
    pub pool_size: usize,
    pub data_source: DataSource,
}

/// A benchmark model definition.
#[allow(clippy::type_complexity)]
pub struct ModelDef {
    /// Short name (used in CLI and output paths).
    pub name: &'static str,
    /// What this model tests (architecture + dataset + reference).
    pub description: &'static str,
    /// Build the model on a specific device.
    pub build: fn(Device) -> Result<Box<dyn Module>>,
    /// Create the dataset for this model.
    pub dataset: fn(&DatasetConfig) -> Result<Arc<dyn BatchDataSet>>,
    /// Report `dataset.len()` without actually constructing the dataset.
    ///
    /// Launcher processes in cluster mode call this in place of `dataset`
    /// to skip the heavy load (the launcher fans out to rank children
    /// and never reads training data itself, but the framework needs
    /// `total_samples` to compute per-rank partition sizes). For real-
    /// data datasets that's typically a known constant (MNIST = 60000
    /// train, CIFAR-10 = 50000 train); for synthetic datasets the hint
    /// can return `cfg.virtual_len`.
    pub dataset_size_hint: fn(&DatasetConfig) -> Result<usize>,
    /// Training step: forward + loss. Returns the loss Variable.
    pub train_fn: fn(&dyn Module, &[Tensor]) -> Result<Variable>,
    /// Optional evaluation metric (e.g. accuracy). Called after each epoch.
    pub eval_fn: Option<fn(&dyn Module, &[Tensor]) -> Result<f64>>,
    /// Optional held-out test dataset for evaluation (e.g. CIFAR-10 test split).
    /// When present, eval_fn runs on this instead of the training data.
    pub test_dataset: Option<fn(&DatasetConfig) -> Result<Arc<dyn BatchDataSet>>>,
    /// Optional per-batch augmentation (e.g. random crop + flip for CIFAR-10).
    /// Applied to training batches only, not eval. Takes [images, labels], returns augmented.
    pub augment_fn: Option<fn(&[Tensor]) -> Result<Vec<Tensor>>>,
    /// Create the optimizer for this model's parameters.
    pub optimizer: fn(&[Parameter], f64) -> Box<dyn Optimizer>,
    /// Optional LR scheduler factory. Args: (base_lr, total_batches, world_size).
    pub scheduler: Option<fn(f64, usize, usize) -> Box<dyn Scheduler>>,
    /// Default configuration.
    pub defaults: ModelDefaults,
    /// Published reference note (shown under report tables for context).
    pub reference: &'static str,
    /// Published eval target (e.g. 0.9125 for 91.25% accuracy).
    /// Used to compute delta in report tables.
    pub published_eval: Option<f64>,
    /// True if higher eval is better (accuracy). False for loss-like metrics.
    pub eval_higher_is_better: bool,
    /// True when the published baseline is reported as a per-epoch curve
    /// (loss + accuracy at every epoch). Solo runs of such models go through
    /// the dedicated `run_baseline_solo` path so we can reproduce the curve
    /// shape; every other run (multi-GPU, non-baseline solo) flows through
    /// the unified `Trainer::builder` path with final-only eval.
    pub needs_baseline_eval: bool,
}

/// All registered benchmark models.
pub fn all_models() -> Vec<ModelDef> {
    vec![
        logistic::def(),
        mlp::def(),
        lenet::def(),
        resnet::def(),
        resnet_graph::def(),
        char_rnn::def(),
        gpt_nano::def(),
        conv_ae::def(),
        olmo::def(),
    ]
}

/// Find a model by name.
pub fn find_model(name: &str) -> Option<ModelDef> {
    all_models().into_iter().find(|m| m.name == name)
}

/// Reference notes and published eval targets by model name.
pub fn model_references() -> Vec<(&'static str, &'static str, Option<f64>, bool)> {
    all_models().into_iter()
        .map(|m| (m.name, m.reference, m.published_eval, m.eval_higher_is_better))
        .collect()
}

/// All model names.
pub fn model_names() -> Vec<&'static str> {
    vec![
        "logistic",
        "mlp",
        "lenet",
        "resnet",
        "resnet-graph",
        "char-rnn",
        "gpt-nano",
        "conv-ae",
        "olmo",
    ]
}
