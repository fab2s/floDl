//! Training entry points and distributed infrastructure.
//!
//! Primary API:
//!
//! - [`Trainer::builder()`] / [`Trainer::run()`] -- framework-managed
//!   training on the controller engine; transparent single-device,
//!   single-host multi-GPU (process-per-rank auto-promote), and
//!   multi-host cluster from one code path
//!
//! User-owned training loop, controller-authoritative:
//!
//! - [`Trainer::builder()`]`.into_worker()` -- the cooperative tier. You
//!   own the loop body; the controller stays authoritative over cadence,
//!   partition, eval election, and checkpointing (see
//!   `docs/design/trainer-execution-tiers.md`).
//!
//! Explicit multi-GPU control:
//!
//! - [`Ddp::wrap()`] -- manual per-rank gradient sync for advanced
//!   patterns (GAN, RL, custom collectives)
//!
//! Supporting infrastructure: NCCL bindings, CUDA events/streams, El Che
//! heterogeneous cadence strategy, the launcher/controller/coordinator
//! cluster runtime, and the wire protocol.

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
pub(crate) mod divergence;
pub(crate) mod nccl_session;
pub(crate) mod wire_convert;
pub mod dashboard_sink;
pub(crate) mod cluster_dashboard_emit;
pub mod launcher;
pub mod max_failure;
pub(crate) mod membership;
pub mod nccl;
pub mod outer_optimizer;
pub(crate) mod port_mux;
pub(crate) mod realized_work;
pub(crate) mod relay;
pub mod ddp;
pub mod ddp_run;
pub mod el_che;
pub mod lr_event_meta;
pub(crate) mod rendezvous;
pub(crate) mod status;
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
/// Loader for the outer-optimizer momentum sidecar (`<stem>.outer.fdl`),
/// used by the launcher on resume to re-seed the outer optimizer.
pub use checkpoint_forge::load_outer_momentum;
pub use cluster::{WorkerBlock, LocalCluster, is_reserved_cluster_env_key};
pub use launcher::{FullCluster, FullWorker, Role};
pub use max_failure::MaxFailureThreshold;
pub use outer_optimizer::{NesterovMomentum, OuterAvg, OuterOptimizer, SlowMomentum};
// CUDA stream/event primitives live in `tensor` (they are device-runtime
// tools, not DDP machinery — audit D5 moved them so `data/` no longer
// reaches into `distributed/`). Re-exported here so existing
// `flodl::distributed::cuda_stream::CudaStream`-style paths keep working.
pub use crate::tensor::cuda_event;
pub use crate::tensor::cuda_stream;
pub use crate::tensor::{CudaEvent, CudaEventFlags, CudaStream, StreamGuard};
pub use dashboard_sink::{ClusterDashboardSink, DashboardSink};
pub use nccl::{NCCL_UNIQUE_ID_BYTES, NcclAbortHandle, NcclComms, NcclRankComm, NcclUniqueId, ReduceOp};
pub use testing::{discover_test_cluster, ENV_TESTING_CLUSTER_JSON};
pub use cluster_builder::{ClusterBuilder, HostBuilder};
pub use config::{ElCheConfig, ElCheMode, TrainerConfig};
pub use ddp::{Ddp, HasGraph, Trainer};
pub use el_che::{AnchorVerdict, ElChe, Phase, WindowReport};
pub use lr_event_meta::{LrEventMeta, LrEventMetaConfig, MetaAction};
pub use ddp_run::{ApplyPolicy, DdpHandle, DdpBuilder, DdpRunConfig, AverageBackend, TrainedState, EpochMetrics, MetricsFn, record_scalar, drain_scalars, GpuWorker, Worker, StepOutcome};
