//! Training entry points and distributed infrastructure.
//!
//! Primary API:
//!
//! - [`Trainer::setup()`] -- user owns the training loop (Graph-based, transparent 1 or N GPU)
//! - [`Trainer::builder()`] -- framework manages threads, data, epochs, averaging
//!
//! Explicit multi-GPU control:
//!
//! - [`Ddp::wrap()`] -- manual gradient sync for advanced patterns (GAN, RL)
//!
//! Supporting infrastructure: NCCL bindings, CUDA events/streams, El Che
//! heterogeneous cadence strategy, and the async DDP runtime.

pub mod cuda_event;
pub mod cuda_stream;
pub mod nccl;
pub mod ddp;
pub mod ddp_run;
pub mod el_che;

pub use cuda_event::{CudaEvent, CudaEventFlags};
pub use cuda_stream::{CudaStream, StreamGuard};
pub use nccl::{NcclAbortHandle, NcclComms, NcclRankComm, NcclUniqueId, ReduceOp};
pub use ddp::{Ddp, DdpConfig, HasGraph, Trainer};
pub use el_che::{ElChe, Phase};
pub use ddp_run::{ApplyPolicy, DdpHandle, DdpBuilder, DdpRunConfig, AverageBackend, TrainedState, EpochMetrics, MetricsFn, record_scalar, drain_scalars, GpuWorker};
// Deprecated aliases
#[allow(deprecated)]
pub use ddp_run::{AsyncDdp, AsyncDdpBuilder, AsyncDdpConfig};
