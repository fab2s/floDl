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

pub mod compat;
pub mod log;
pub mod sys;
pub mod metrics;
pub mod tensor;
pub mod autograd;
pub mod nn;
pub mod distributed;
pub mod graph;
pub mod monitor;
pub mod worker;
#[cfg(feature = "rng")]
pub mod rng;
#[cfg(feature = "rng")]
pub mod data;

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
    cuda_active_bytes, cuda_active_bytes_idx, cuda_allocated_bytes, cuda_allocated_bytes_idx,
    cuda_available, cuda_device_count, cuda_device_name, cuda_device_name_idx, cuda_devices,
    cuda_empty_cache, cuda_graph_capture, cuda_graph_pool_handle, cuda_has_primary_context,
    cuda_manual_seed_all, cuda_memory_info, cuda_memory_info_idx, cuda_nvml_memory_info_idx,
    cuda_peak_active_bytes, cuda_peak_active_bytes_idx, cuda_peak_reserved_bytes,
    cuda_peak_reserved_bytes_idx, cuda_reset_peak_stats, cuda_reset_peak_stats_idx,
    cuda_synchronize, cuda_utilization, cuda_utilization_idx, current_cuda_device,
    set_current_cuda_device, usable_cuda_devices, CudaEvent, CudaEventFlags, CudaGraph,
    CudaStream,
};

pub use tensor::{gpu_available, gpu_device_count, gpu_memory_info, gpu_memory_info_idx, gpu_allocated_bytes, gpu_allocated_bytes_idx, gpu_active_bytes, gpu_active_bytes_idx, gpu_peak_active_bytes, gpu_peak_active_bytes_idx, gpu_peak_reserved_bytes, gpu_peak_reserved_bytes_idx, gpu_reset_peak_stats, gpu_reset_peak_stats_idx, gpu_empty_cache, gpu_utilization, gpu_utilization_idx, gpu_device_name, gpu_device_name_idx, gpu_devices, gpu_arch_name, cuda_compute_capability, gpu_has_primary_context, gpu_smi_memory_info_idx, probe_device, usable_gpu_devices, DeviceInfo, set_current_gpu_device, current_gpu_device, gpu_synchronize, hardware_summary, set_cudnn_benchmark, manual_seed, gpu_manual_seed_all, malloc_trim, live_tensor_count, rss_kb, Device, DType, Result, Tensor, TensorError, TensorOptions, GpuEvent, GpuEventFlags, GpuStream, StreamGuard};
#[cfg(feature = "rng")]
pub use rng::Rng;
pub use autograd::{Variable, no_grad, is_grad_enabled, NoGradGuard, max_pool2d, adaptive_avg_pool2d, grid_sample, scaled_dot_product_attention, embedding, embedding_bag};
pub use nn::{
    Module, NamedInputModule, LoopBody, TraceEmit, forward_via_step,
    Parameter, Buffer, Linear, Optimizer, Stateful, StateKind,
    SGD, SGDBuilder, Adam, AdamBuilder, AdamW, AdamWBuilder,
    RMSprop, RMSpropBuilder, Adagrad, AdagradBuilder, RAdam, NAdam,
    save_checkpoint, load_checkpoint, save_checkpoint_file, load_checkpoint_file,
    migrate_checkpoint, migrate_checkpoint_file, migrate_optim_state_file,
    checkpoint_version, checkpoint_keys,
    LoadReport, MigrateReport,
    GradScaler, cast_parameters, AutocastGuard, autocast, is_autocast_enabled, is_autocast_enabled_for,
    Identity, ReLU, Sigmoid, Tanh, GELU, GeluApprox, SiLU,
    LeakyReLU, ELU, Softplus, Mish,
    SELU, Hardswish, Hardsigmoid, PReLU,
    Softmax, LogSoftmax, Flatten,
    Dropout, Dropout2d, AlphaDropout, ZeroPad2d, ReflectionPad2d,
    LayerNorm, RMSNorm, Embedding, EmbeddingBag, GRUCell, GRU, LSTMCell, LSTM,
    Conv1d, Conv1dBuilder, Conv2d, Conv2dBuilder,
    ConvTranspose1d, ConvTranspose2d,
    Conv3d, Conv3dBuilder, ConvTranspose3d,
    GroupNorm, BatchNorm, BatchNorm2d, InstanceNorm,
    MaxPool2d, AvgPool2d, MaxPool1d, AvgPool1d, AdaptiveMaxPool2d, AdaptiveAvgPool2d,
    PixelShuffle, PixelUnshuffle, Upsample, Unfold, Fold, Bilinear,
    MultiheadAttention, RotaryEmbedding, SwiGLU,
    mse_loss, cross_entropy_loss, bce_loss, bce_with_logits_loss, l1_loss, smooth_l1_loss, kl_div_loss,
    nll_loss, ctc_loss, focal_loss,
    triplet_margin_loss, cosine_embedding_loss, hinge_embedding_loss, margin_ranking_loss, poisson_nll_loss,
    clip_grad_norm, clip_grad_value,
    Scheduler, StepDecay, CosineScheduler, WarmupScheduler, PlateauScheduler, ExponentialLR, MultiStepLR, OneCycleLR, CyclicLR,
    xavier_uniform, xavier_normal, kaiming_uniform, kaiming_normal, uniform_bias, uniform, normal, orthogonal, trunc_normal,
    walk_modules, walk_modules_visited,
    GpuGraph, MemPoolId, CaptureMode, gpu_graph_capture, gpu_graph_pool_handle,
    GaussianBlur, gaussian_blur_2d,
};
pub use distributed::{
    NcclComms, NcclRankComm, NcclUniqueId, ReduceOp, Ddp, HasGraph, Trainer, ElChe,
    ApplyPolicy, DdpHandle, DdpBuilder, DdpRunConfig, AverageBackend, TrainedState, EpochMetrics, MetricsFn, record_scalar, drain_scalars, GpuWorker, Worker, StepOutcome,
    ElCheConfig, ElCheMode, TrainerConfig, MaxFailureThreshold,
    ClusterBuilder, HostBuilder,
    NesterovMomentum, OuterAvg, OuterOptimizer, SlowMomentum,
};
pub use graph::{
    FlowBuilder, MergeOp, Graph, GraphExt, MapBuilder, Trend, TrendGroup,
    Profile, NodeTiming, LevelTiming, format_duration,
    SoftmaxRouter, SigmoidRouter, FixedSelector, ArgmaxSelector,
    ThresholdHalt, LearnedHalt,
    Reshape, StateAdd, Reduce, ModelSnapshot,
    PathKind,
    GraphEpochIterator, ActiveGraphEpochIterator,
};
pub use worker::CpuWorker;
pub use monitor::Monitor;
#[cfg(feature = "rng")]
pub use data::{DataSet, BatchDataSet, Sampler, RandomSampler, SequentialSampler, SplitSampler, DataLoader, DataLoaderBuilder, EpochIterator, Batch, PickKey, TransformFn};
