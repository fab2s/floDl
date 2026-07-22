//! Rank-side dashboard intent stash + emit helpers.
//!
//! When the user's harness on a rank process calls
//! [`crate::monitor::Monitor::serve`] / `.watch()` / `.set_metadata()`,
//! the rewired Monitor stashes the intent here instead of binding a
//! local HTTP server (the launcher hosts the dashboard post controller-
//! active refactor). The cluster_worker drains the stash at startup
//! and emits the matching [`crate::distributed::wire::TimingMsgWire`]
//! frames on the coord socket; the launcher's
//! [`crate::distributed::ClusterDashboardSink`] then binds the server
//! lazily on first arrival.
//!
//! Single global stash — every Monitor on the rank process writes here
//! (typical user code constructs one Monitor; multi-Monitor on the
//! same rank is not a documented pattern). The cluster_worker reads
//! once at startup; subsequent stashes after that point are no-ops
//! from the launcher's POV.

use std::sync::Mutex;

/// Snapshot of the user's dashboard intent, populated by Monitor on
/// the rank's harness thread and drained by `cluster_worker` after
/// rendezvous.
#[derive(Debug, Clone, Default)]
pub(crate) struct PendingDashboardConfig {
    /// HTTP port the user requested via `monitor.serve(port)`. `None`
    /// = user has not opted into the live dashboard; the cluster_worker
    /// skips every Dashboard* emit and resource sampling stays off.
    pub port: Option<u16>,
    /// Graph SVG bytes the user pushed via `monitor.watch(&model)` (or
    /// `monitor.set_svg(&svg)`).
    pub svg: Option<String>,
    /// Graph label captured by `monitor.watch`.
    pub label: Option<String>,
    /// Graph structural hash captured by `monitor.watch`.
    pub hash: Option<String>,
    /// Pre-serialized JSON metadata blob from `monitor.set_metadata`.
    pub metadata_json: Option<String>,
    /// Hardware summary string captured at `Monitor::new`. Sent
    /// per-rank so the launcher's dashboard can render per-rank tabs
    /// labelled `host:lr=<local_rank> gr=<global_rank>`.
    pub hardware: Option<String>,
}

static PENDING: Mutex<PendingDashboardConfig> = Mutex::new(PendingDashboardConfig {
    port: None,
    svg: None,
    label: None,
    hash: None,
    metadata_json: None,
    hardware: None,
});

/// Record the dashboard port the user requested. Called from
/// `Monitor::serve` on the cluster-rank path.
pub(crate) fn stash_port(port: u16) {
    PENDING.lock().unwrap().port = Some(port);
}

/// Record the graph SVG + identity. Called from `Monitor::watch` /
/// `Monitor::set_svg` on the cluster-rank path.
pub(crate) fn stash_svg(svg: String, label: Option<String>, hash: Option<String>) {
    let mut p = PENDING.lock().unwrap();
    p.svg = Some(svg);
    if label.is_some() {
        p.label = label;
    }
    if hash.is_some() {
        p.hash = hash;
    }
}

/// Record the dashboard metadata JSON. Called from `Monitor::set_metadata`.
pub(crate) fn stash_metadata(json: String) {
    PENDING.lock().unwrap().metadata_json = Some(json);
}

/// Record the rank's hardware summary string. Called from `Monitor::new`
/// (so every rank's hardware reaches the launcher, even if the user
/// never opts into the dashboard — cheap stash, drained only when port
/// is also set).
pub(crate) fn stash_hardware(hardware: String) {
    PENDING.lock().unwrap().hardware = Some(hardware);
}

/// Drain the stash (atomic snapshot). Called from `cluster_worker` at
/// startup, after rendezvous but before the training loop begins.
/// Returns the current stash and resets it to default, so a hypothetical
/// re-call would not double-emit.
pub(crate) fn drain() -> PendingDashboardConfig {
    let mut p = PENDING.lock().unwrap();
    std::mem::take(&mut *p)
}
