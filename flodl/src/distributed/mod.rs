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
pub(crate) mod checkpoint_forge;
pub(crate) mod chunk_pool;
pub mod cluster;
pub mod cluster_builder;
pub mod cluster_coordinator;
pub mod cluster_worker;
pub mod config;
pub(crate) mod controller;
pub(crate) mod cpu_reduce;
pub mod cuda_event;
pub mod cuda_stream;
pub mod dashboard_sink;
pub(crate) mod cluster_dashboard_emit;
pub mod launcher;
pub mod max_failure;
pub mod nccl;
pub(crate) mod relay;
pub mod ddp;
pub mod ddp_run;
pub mod el_che;
pub mod lr_event_meta;
pub(crate) mod rendezvous;
pub mod testing;
pub(crate) mod wire;

pub use checkpoint_meta::{
    CHECKPOINT_META_SCHEMA_VERSION, CheckpointBundle, CheckpointMeta, CoverageBlock,
    ElCheState, EpochCoverage, ModelSchema, RANK_DEATH_RECORD_SCHEMA_VERSION,
    RankDeathRecord, SaveReason,
};
pub(crate) use checkpoint_forge::CheckpointForge;
/// Positional loader for cluster consensus / failure-save `.fdl` bundles —
/// the resume-side counterpart to the consensus writers (`load_consensus_checkpoint`).
pub use checkpoint_forge::load_consensus_checkpoint;
pub use cluster::{WorkerBlock, LocalCluster};
pub use launcher::{FullCluster, FullWorker, Role};
pub use max_failure::MaxFailureThreshold;
pub use cuda_event::{CudaEvent, CudaEventFlags};
pub use dashboard_sink::{ClusterDashboardSink, DashboardSink};
pub use cuda_stream::{CudaStream, StreamGuard};
pub use nccl::{NCCL_UNIQUE_ID_BYTES, NcclAbortHandle, NcclComms, NcclRankComm, NcclUniqueId, ReduceOp};
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
