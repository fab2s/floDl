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

pub mod checkpoint_meta;
pub mod chunk_pool;
pub mod cluster;
pub mod cluster_builder;
pub mod cluster_coordinator;
pub mod cluster_worker;
pub mod config;
pub mod controller;
pub mod cpu_reduce;
pub mod cuda_event;
pub mod cuda_stream;
pub mod launcher;
pub mod max_failure;
pub mod nccl;
pub mod ddp;
pub mod ddp_run;
pub mod el_che;
pub mod lr_event_meta;
pub mod rendezvous;
pub mod testing;
pub mod wire;

pub use checkpoint_meta::{
    CHECKPOINT_META_SCHEMA_VERSION, CheckpointBundle, CheckpointMeta, ElCheState, SaveReason,
};
pub use cluster::{HostBlock, LocalCluster};
pub use controller::{ClusterController, RoundFrame, TensorPayload, DTYPE_F32};
pub use cpu_reduce::{
    AsyncCpuReduceClient, CpuReduceClient, round_frame_to_tensors, tensors_to_round_frame,
};
pub use launcher::{FullCluster, FullHost, Role};
pub use max_failure::MaxFailureThreshold;
pub use cuda_event::{CudaEvent, CudaEventFlags};
pub use cuda_stream::{CudaStream, StreamGuard};
pub use nccl::{NCCL_UNIQUE_ID_BYTES, NcclAbortHandle, NcclComms, NcclRankComm, NcclUniqueId, ReduceOp};
pub use rendezvous::TcpRendezvous;
pub use testing::{discover_test_cluster, ENV_TESTING_CLUSTER_JSON};
pub use cluster_builder::{ClusterBuilder, HostBuilder};
pub use config::{ElCheConfig, ElCheMode, TrainerConfig};
pub use ddp::{Ddp, DdpConfig, HasGraph, Trainer};
pub use el_che::{ElChe, Phase};
pub use lr_event_meta::{LrEventMeta, LrEventMetaConfig, MetaAction};
pub use ddp_run::{ApplyPolicy, DdpHandle, DdpBuilder, DdpRunConfig, AverageBackend, TrainedState, EpochMetrics, MetricsFn, record_scalar, drain_scalars, GpuWorker};
// Deprecated aliases
#[allow(deprecated)]
pub use ddp_run::{AsyncDdp, AsyncDdpBuilder, AsyncDdpConfig};
