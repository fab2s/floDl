//! flodl — a deep learning framework built on libtorch, from Rust.
//!
//! Stack: `flodl-sys` (C++ shim FFI) → `tensor` → `autograd` → `nn` → `graph`.
//!
//! ```ignore
//! use flodl::*;
//!
//! // Build a model as a computation graph
//! let model = FlowBuilder::from(Linear::new(4, 8)?)
//!     .through(GELU)
//!     .through(Linear::new(8, 2)?)
//!     .build()?;
//!
//! // Forward pass
//! let x = Variable::new(Tensor::randn(&[1, 4], Default::default())?, false);
//! let target = Variable::new(Tensor::randn(&[1, 2], Default::default())?, false);
//! let pred = model.forward(&x)?;
//!
//! // Backward + optimize
//! let params = model.parameters();
//! let mut optimizer = Adam::new(&params, 1e-3);
//! let loss = mse_loss(&pred, &target)?;
//! optimizer.zero_grad();
//! loss.backward()?;
//! optimizer.step()?;
//! ```

pub mod autograd;
pub mod compat;
#[cfg(feature = "rng")]
pub mod data;
pub mod distributed;
pub mod graph;
pub mod log;
pub mod metrics;
pub mod monitor;
pub mod nn;
#[cfg(feature = "rng")]
pub mod rng;
pub mod sys;
pub mod tensor;
pub mod worker;

/// Shorthand for building `Vec<Box<dyn Module>>` from a list of modules.
/// Use with `split`, `gate`, and `switch` to avoid manual `Box::new()` wrapping.
///
/// ```text
/// .split(modules![read_head(H), read_head(H)])
/// .gate(router, modules![Linear::new(H, H)?, Linear::new(H, H)?])
/// ```
#[macro_export]
macro_rules! modules {
    ($($module:expr),* $(,)?) => {
        vec![$(Box::new($module) as Box<dyn $crate::Module>),*]
    };
}

pub use log::{Verbosity, set_verbosity, verbosity};
// Deprecated `cuda_*` / `Cuda*` spellings, re-exported at the root they
// used to live at so existing code keeps compiling. See `compat`.
#[allow(deprecated)]
pub use compat::{
    CudaEvent, CudaEventFlags, CudaGraph, CudaStream, cuda_active_bytes, cuda_active_bytes_idx,
    cuda_allocated_bytes, cuda_allocated_bytes_idx, cuda_available, cuda_device_count,
    cuda_device_name, cuda_device_name_idx, cuda_devices, cuda_empty_cache, cuda_graph_capture,
    cuda_graph_pool_handle, cuda_has_primary_context, cuda_manual_seed_all, cuda_memory_info,
    cuda_memory_info_idx, cuda_nvml_memory_info_idx, cuda_peak_active_bytes,
    cuda_peak_active_bytes_idx, cuda_peak_reserved_bytes, cuda_peak_reserved_bytes_idx,
    cuda_reset_peak_stats, cuda_reset_peak_stats_idx, cuda_synchronize, cuda_utilization,
    cuda_utilization_idx, current_cuda_device, set_current_cuda_device, usable_cuda_devices,
};

pub use autograd::{
    NoGradGuard, Variable, adaptive_avg_pool2d, embedding, embedding_bag, grid_sample,
    is_grad_enabled, max_pool2d, no_grad, scaled_dot_product_attention,
};
#[cfg(feature = "rng")]
pub use data::{
    Batch, BatchDataSet, DataLoader, DataLoaderBuilder, DataSet, EpochIterator, PickKey,
    RandomSampler, Sampler, SequentialSampler, SplitSampler, TransformFn,
};
pub use distributed::{
    ApplyPolicy, AverageBackend, ClusterBuilder, Ddp, DdpBuilder, DdpHandle, DdpRunConfig, ElChe,
    ElCheConfig, ElCheMode, EpochMetrics, GpuWorker, HasGraph, HostBuilder, MaxFailureThreshold,
    MetricsFn, NcclComms, NcclRankComm, NcclUniqueId, NesterovMomentum, OuterAvg, OuterOptimizer,
    ReduceOp, SlowMomentum, StepOutcome, TrainedState, Trainer, TrainerConfig, Worker,
    drain_scalars, record_scalar,
};
pub use graph::{
    ActiveGraphEpochIterator, ArgmaxSelector, FixedSelector, FlowBuilder, Graph,
    GraphEpochIterator, GraphExt, LearnedHalt, LevelTiming, MapBuilder, MergeOp, ModelSnapshot,
    NodeTiming, PathKind, Profile, ProfileSource, Reduce, Reshape, SigmoidRouter, SoftmaxRouter,
    StateAdd, ThresholdHalt, Trend, TrendGroup, format_duration,
};
pub use monitor::Monitor;
pub use nn::{
    Adagrad, AdagradBuilder, Adam, AdamBuilder, AdamW, AdamWBuilder, AdaptiveAvgPool2d,
    AdaptiveMaxPool2d, AlphaDropout, AutocastGuard, AvgPool1d, AvgPool2d, BatchNorm, BatchNorm2d,
    Bilinear, Buffer, CaptureMode, Conv1d, Conv1dBuilder, Conv2d, Conv2dBuilder, Conv3d,
    Conv3dBuilder, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d, CosineScheduler, CyclicLR,
    Dropout, Dropout2d, ELU, Embedding, EmbeddingBag, ExponentialLR, Flatten, Fold, GELU, GRU,
    GRUCell, GaussianBlur, GeluApprox, GpuGraph, GradScaler, GroupNorm, Hardsigmoid, Hardswish,
    Identity, InstanceNorm, LSTM, LSTMCell, LayerNorm, LeakyReLU, Linear, LoadReport, LogSoftmax,
    LoopBody, MaxPool1d, MaxPool2d, MemPoolId, MigrateReport, Mish, Module, MultiStepLR,
    MultiheadAttention, NAdam, NamedInputModule, OneCycleLR, Optimizer, PReLU, Parameter,
    PixelShuffle, PixelUnshuffle, PlateauScheduler, RAdam, RMSNorm, RMSprop, RMSpropBuilder, ReLU,
    ReflectionPad2d, RotaryEmbedding, SELU, SGD, SGDBuilder, Scheduler, SiLU, Sigmoid, Softmax,
    Softplus, StateKind, Stateful, StepDecay, SwiGLU, Tanh, TraceEmit, Unfold, Upsample,
    WarmupScheduler, ZeroPad2d, autocast, bce_loss, bce_with_logits_loss, cast_parameters,
    checkpoint_keys, checkpoint_version, clip_grad_norm, clip_grad_value, cosine_embedding_loss,
    cross_entropy_loss, ctc_loss, focal_loss, forward_via_step, gaussian_blur_2d,
    gpu_graph_capture, gpu_graph_pool_handle, hinge_embedding_loss, is_autocast_enabled,
    is_autocast_enabled_for, kaiming_normal, kaiming_uniform, kl_div_loss, l1_loss,
    load_checkpoint, load_checkpoint_file, margin_ranking_loss, migrate_checkpoint,
    migrate_checkpoint_file, migrate_optim_state_file, mse_loss, nll_loss, normal, orthogonal,
    poisson_nll_loss, save_checkpoint, save_checkpoint_file, smooth_l1_loss, triplet_margin_loss,
    trunc_normal, uniform, uniform_bias, walk_modules, walk_modules_visited, xavier_normal,
    xavier_uniform,
};
#[cfg(feature = "rng")]
pub use rng::Rng;
pub use tensor::{
    DType, Device, DeviceInfo, GpuEvent, GpuEventFlags, GpuStream, Result, StreamGuard, Tensor,
    TensorError, TensorOptions, cuda_compute_capability, current_gpu_device, gpu_active_bytes,
    gpu_active_bytes_idx, gpu_allocated_bytes, gpu_allocated_bytes_idx, gpu_arch_name,
    gpu_available, gpu_device_count, gpu_device_name, gpu_device_name_idx, gpu_devices,
    gpu_empty_cache, gpu_has_primary_context, gpu_manual_seed_all, gpu_memory_info,
    gpu_memory_info_idx, gpu_peak_active_bytes, gpu_peak_active_bytes_idx, gpu_peak_reserved_bytes,
    gpu_peak_reserved_bytes_idx, gpu_reset_peak_stats, gpu_reset_peak_stats_idx,
    gpu_smi_memory_info_idx, gpu_synchronize, gpu_utilization, gpu_utilization_idx,
    hardware_summary, live_tensor_count, malloc_trim, manual_seed, probe_device, rss_kb,
    set_cudnn_benchmark, set_current_gpu_device, usable_gpu_devices,
};
pub use worker::CpuWorker;
