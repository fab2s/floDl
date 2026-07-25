//! Coordinator seam of the B3 alert lane: turn the cohort's three
//! alerting-class conditions into `kind: "event"` records on the same
//! path-keyed stream the per-window `node` records ride
//! (`.design/monitoring-portal-b3.md`).
//!
//! | class          | condition                                        |
//! |----------------|--------------------------------------------------|
//! | `rank_lost`    | a rank was declared dead (staleness / child exit) |
//! | `drift`        | the convergence guard had to correct the anchor   |
//! | `control_drop` | a best-effort control broadcast missed live ranks |
//!
//! Nothing informational belongs here — a throttle, an anchor *growth*, a
//! window report are state, not alerts, and stay in `node` records.
//!
//! The lane itself ([`crate::monitor::event_lane::EventLane`]) owns collapse
//! and the live cap, so this seam is only: resolve the origin path, format the
//! detail, log it loudly, forward whatever records the lane hands back. All of
//! that happens whether or not a dashboard sink is attached — a headless
//! cluster run is exactly the one where an unnoticed rank death hurts most.

use crate::monitor::event_lane::EventClass;

use super::window_records::rank_record_path;
use super::ClusterCoordinator;

impl ClusterCoordinator {
    /// Per-rank host list indexed by rank, as the record tree sees it.
    /// Missing entries degrade to "host unknown" (the flat `root/rankN`
    /// path), never to a bogus segment.
    fn record_hosts(&self) -> Vec<&str> {
        (0..self.world_size)
            .map(|r| self.rank_hosts.get(r).map(|s| s.as_str()).unwrap_or(""))
            .collect()
    }

    /// Origin path of a rank-scoped alert — the very node path the window
    /// tree emits for that rank, so the portal can drill straight into it.
    pub(super) fn alert_path_for_rank(&self, rank: usize) -> String {
        rank_record_path(rank, &self.record_hosts())
    }

    /// Record one alert and forward whatever the lane decides to emit.
    ///
    /// The lane may return nothing (the occurrence was collapsed into an open
    /// window), one record, or several (a collapsed window closing, plus the
    /// overflow notice). Never fails and never blocks the caller's path — an
    /// alert is an observation of trouble, not a second failure mode.
    pub(super) fn emit_alert(
        &mut self,
        class: EventClass,
        path: String,
        detail: String,
    ) {
        let now = now_ms();
        let records = self.event_lane.record(class, &path, detail, now);
        self.push_alert_records(records);
    }

    /// Close the lane's open collapse windows at teardown so the tail of a
    /// flood reaches the persisted history, not only the live feed.
    pub(super) fn flush_alerts(&mut self) {
        let records = self.event_lane.flush(now_ms());
        self.push_alert_records(records);
    }

    fn push_alert_records(&mut self, records: Vec<serde_json::Value>) {
        if records.is_empty() {
            return;
        }
        // Loud by default, sink or no sink: an alert is exactly the thing a
        // user must not have to raise the verbosity to discover. Bounded by
        // the lane's collapse, so this cannot become a flood.
        for r in &records {
            crate::msg!(
                "  ddp: [{}] {} {} — {} (x{})",
                r["sev"].as_str().unwrap_or("?"),
                r["class"].as_str().unwrap_or("?"),
                r["path"].as_str().unwrap_or("?"),
                r["detail"].as_str().unwrap_or(""),
                r["count"],
            );
        }
        if let Some(sink) = self.dashboard_sink.as_ref() {
            sink.push_events(records);
        }
    }
}

/// Wall-clock epoch milliseconds — the record stream's `ts`, and the clock the
/// lane collapses on. Saturates to 0 if the system clock predates the epoch.
pub(super) fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}
