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

pub(crate) mod checkpoint_forge;
pub mod checkpoint_meta;
pub(crate) mod chunk_pool;
pub mod cluster;
pub mod cluster_builder;
pub mod cluster_coordinator;
pub(crate) mod cluster_dashboard_emit;
pub mod cluster_worker;
pub mod config;
pub(crate) mod controller;
pub(crate) mod cpu_reduce;
pub mod dashboard_sink;
pub mod ddp;
pub mod ddp_run;
pub(crate) mod divergence;
pub mod el_che;
pub mod launcher;
pub mod lr_event_meta;
pub mod max_failure;
pub(crate) mod membership;
pub(crate) mod model_sig;
pub mod nccl;
pub(crate) mod nccl_session;
pub mod outer_optimizer;
pub(crate) mod port_mux;
pub(crate) mod realized_work;
pub(crate) mod relay;
pub(crate) mod rendezvous;
pub(crate) mod status;
pub mod testing;
pub(crate) mod wire;
pub(crate) mod wire_convert;

pub(crate) use checkpoint_forge::CheckpointForge;
/// Positional loader for cluster consensus / failure-save `.fdl` bundles —
/// the resume-side counterpart to the consensus writers (`load_consensus_checkpoint`).
pub use checkpoint_forge::load_consensus_checkpoint;
/// Loader for the outer-optimizer momentum sidecar (`<stem>.outer.fdl`),
/// used by the launcher on resume to re-seed the outer optimizer.
pub use checkpoint_forge::load_outer_momentum;
pub use checkpoint_meta::{
    CHECKPOINT_META_SCHEMA_VERSION, CheckpointBundle, CheckpointMeta, CoverageBlock, ElCheState,
    EpochCoverage, ModelSchema, RANK_DEATH_RECORD_SCHEMA_VERSION, RankDeathRecord, SaveReason,
};
pub use cluster::{LocalCluster, WorkerBlock, cluster_data_path, is_reserved_cluster_env_key};
pub use launcher::{FullCluster, FullWorker, Role};
pub use max_failure::MaxFailureThreshold;
/// Join-window start-switch mode (`controller.join.start:` — auto /
/// manual / hybrid). Public because [`ClusterBuilder::controller`]'s
/// `start_mode(...)` setter and the [`launcher::JoinKnobs`] mirror take
/// it; the window semantics live on the enum's docs.
pub use membership::StartMode;
pub use outer_optimizer::{NesterovMomentum, OuterAvg, OuterOptimizer, SlowMomentum};
// CUDA stream/event primitives live in `tensor` (they are device-runtime
// tools, not DDP machinery — audit D5 moved them so `data/` no longer
// reaches into `distributed/`). Re-exported here so existing
// `flodl::distributed::cuda_stream::GpuStream`-style paths keep working.
pub use crate::tensor::cuda_event;
pub use crate::tensor::cuda_stream;
pub use crate::tensor::{GpuEvent, GpuEventFlags, GpuStream, StreamGuard};
pub use cluster_builder::{ClusterBuilder, HostBuilder};
pub use config::{ElCheConfig, ElCheMode, TrainerConfig};
pub use dashboard_sink::{ClusterDashboardSink, DashboardSink};
pub use ddp::{Ddp, HasGraph, Trainer};
pub use ddp_run::{
    ApplyPolicy, AverageBackend, DdpBuilder, DdpHandle, DdpRunConfig, EpochMetrics, GpuWorker,
    MetricsFn, StepOutcome, TrainedState, Worker, drain_scalars, record_scalar,
};
pub use el_che::{AnchorVerdict, ElChe, Phase, WindowReport};
pub use lr_event_meta::{LrEventMeta, LrEventMetaConfig, MetaAction};
pub use nccl::{
    NCCL_UNIQUE_ID_BYTES, NcclAbortHandle, NcclComms, NcclRankComm, NcclUniqueId, ReduceOp,
};
pub use testing::{ENV_TESTING_CLUSTER_JSON, discover_test_cluster};
