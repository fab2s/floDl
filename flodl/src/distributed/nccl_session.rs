//! Neutral home for the NCCL-session handoff type shared across layers.
//!
//! The coordinator-facing [`crate::distributed::cluster_worker`] produces
//! a pending session (on a `NewNcclSession` control frame) and the
//! lower-level [`GpuWorker`](crate::distributed::ddp_run::GpuWorker)
//! consumes it (post-abort comm rebuild in `sync_now_nccl`). Defining it
//! here — a leaf both layers depend on — keeps the lower layer from
//! reaching up into the higher one just for a plain data struct.
//!
//! Not feature-gated: `GpuWorker` carries the mailbox field
//! unconditionally (it is `None` outside cluster mode), so the type must
//! exist in non-CUDA builds too.

/// A coordinator-broadcast NCCL session waiting to be applied.
///
/// Written into the worker's session mailbox by the `cluster_worker`
/// inbound bridge on each `NewNcclSession` arrival; drained by the
/// worker's post-abort rebuild path. Slot semantics: latest write wins;
/// old values are silently overwritten on each new session.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct PendingNcclSession {
    pub uid_bytes: Vec<u8>,
    pub new_rank: usize,
    pub new_world_size: usize,
}
