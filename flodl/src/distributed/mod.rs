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

pub mod cluster;
pub mod controller;
pub mod cpu_reduce;
pub mod cuda_event;
pub mod cuda_stream;
pub mod launcher;
pub mod nccl;
pub mod ddp;
pub mod ddp_run;
pub mod el_che;
pub mod lr_event_meta;
pub mod rendezvous;

pub use cluster::{HostBlock, LocalCluster};
pub use controller::{CpuAverager, RoundFrame, TensorPayload, DTYPE_F32};
pub use cpu_reduce::{
    AsyncCpuReduceClient, CpuReduceClient, round_frame_to_tensors, tensors_to_round_frame,
};
pub use launcher::{FullCluster, FullHost, Role};
pub use cuda_event::{CudaEvent, CudaEventFlags};
pub use cuda_stream::{CudaStream, StreamGuard};
pub use nccl::{NCCL_UNIQUE_ID_BYTES, NcclAbortHandle, NcclComms, NcclRankComm, NcclUniqueId, ReduceOp};
pub use rendezvous::TcpRendezvous;
pub use ddp::{Ddp, DdpConfig, HasGraph, Trainer};
pub use el_che::{ElChe, Phase};
pub use lr_event_meta::{LrEventMeta, LrEventMetaConfig, MetaAction};
pub use ddp_run::{ApplyPolicy, DdpHandle, DdpBuilder, DdpRunConfig, AverageBackend, TrainedState, EpochMetrics, MetricsFn, record_scalar, drain_scalars, GpuWorker};
// Deprecated aliases
#[allow(deprecated)]
pub use ddp_run::{AsyncDdp, AsyncDdpBuilder, AsyncDdpConfig};
