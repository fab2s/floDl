//! Alert lane — the bounded, collapsed `kind: "event"` side of the B3 record
//! stream (`.design/monitoring-portal-b3.md`).
//!
//! Alerts ride the *same* stream as `node` records (so they persist to the
//! record log and ship to external log sinks unchanged), but they are a
//! separate feed for the portal: rank death, divergence drift, a dropped
//! control broadcast. Alerting-class only — informational state (a throttle,
//! an anchor move) stays in `node` records and never enters this lane.
//!
//! ## Bounded by construction
//!
//! An alert lane that can flood is worse than none: the one line that matters
//! scrolls away, and on a long unhealthy run the log grows without bound. Two
//! mechanisms keep it finite, and **neither truncates silently**:
//!
//! - **Collapse**: repeats of the same `(class, path)` inside
//!   [`COLLAPSE_WINDOW_MS`] are absorbed instead of re-emitted. The first
//!   occurrence still emits *immediately* (an alert a human is waiting on must
//!   not be delayed); the repeats ride the `count` of the next record past the
//!   window, so **the counts summed over the stream equal the true number of
//!   occurrences**.
//! - **Cap**: at most [`MAX_EVENTS`] live entries, the **least recently
//!   active** evicted first — a chronic alert that is still firing outlives a
//!   one-off that stopped. Every eviction bumps a drop counter that surfaces
//!   as a `class: "overflow"` record (itself collapsed on the same window, so
//!   the notice cannot flood either).
//!
//! ## Two views, one shape
//!
//! [`EventLane::record`] returns the records to append to the **stream** —
//! one per collapse window, `count` = the occurrences that record represents.
//! [`EventLane::live`] returns the **current bounded feed** for the portal —
//! one entry per `(class, path)`, `count` = its running total. Same JSON
//! shape; the stream is the history, `live` is the current state.
//!
//! Both are pure: the caller supplies `now_ms`, so collapse behaviour is
//! testable without sleeping.

use std::collections::VecDeque;

use serde_json::{Value, json};

use crate::monitor::record::Severity;

/// Repeats of the same `(class, path)` inside this window are collapsed into
/// one stream record.
pub const COLLAPSE_WINDOW_MS: u64 = 10_000;

/// Maximum live entries retained; the oldest is evicted past this, and the
/// eviction surfaces as an `overflow` record.
pub const MAX_EVENTS: usize = 200;

/// What kind of alert this is. The severity is a property of the class, not
/// of the call site, so every producer of a class agrees.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventClass {
    /// A rank was declared dead (heartbeat staleness or a reported child exit).
    RankLost,
    /// Weight-space divergence forced the convergence guard to correct.
    Drift,
    /// A best-effort coordinator→rank control broadcast did not reach every
    /// live rank.
    ControlDrop,
    /// Alerts were dropped by the live cap — the truncation notice itself.
    Overflow,
}

impl EventClass {
    /// Wire token.
    pub fn as_str(self) -> &'static str {
        match self {
            EventClass::RankLost => "rank_lost",
            EventClass::Drift => "drift",
            EventClass::ControlDrop => "control_drop",
            EventClass::Overflow => "overflow",
        }
    }

    /// Severity carried by every record of this class.
    pub fn severity(self) -> Severity {
        match self {
            // Losing a rank or a control frame breaks cohort coordination.
            EventClass::RankLost | EventClass::ControlDrop => Severity::Critical,
            // Drift is the guard doing its job loudly; overflow is a
            // bookkeeping notice. Both warn.
            EventClass::Drift | EventClass::Overflow => Severity::Warn,
        }
    }
}

/// One live entry: the collapsed state of a `(class, path)` pair.
#[derive(Debug, Clone)]
struct Alert {
    class: EventClass,
    path: String,
    detail: String,
    /// Timestamp of the most recent occurrence.
    ts: u64,
    /// Start of the open collapse window (the last emission).
    window_ms: u64,
    /// Occurrences since the lane first saw this pair — the live view's count.
    total: u64,
    /// Occurrences absorbed since the last emitted record; rides the `count`
    /// of the next one so the stream stays exact.
    pending: u64,
}

impl Alert {
    fn to_json(&self, count: u64) -> Value {
        json!({
            "v": 1,
            "ts": self.ts,
            "sev": self.class.severity().as_str(),
            "path": self.path,
            "kind": "event",
            "class": self.class.as_str(),
            "detail": self.detail,
            "count": count,
        })
    }
}

/// Bounded, collapsing alert buffer. Cheap when idle (no allocation until the
/// first alert), so it is always on — the alert history exists whether or not
/// a dashboard is attached.
#[derive(Debug)]
pub struct EventLane {
    events: VecDeque<Alert>,
    collapse_window_ms: u64,
    max_events: usize,
    /// Total entries evicted by the cap over the run.
    dropped_total: u64,
    /// Evictions not yet represented in an emitted `overflow` record.
    dropped_pending: u64,
    /// Timestamp of the most recent eviction.
    dropped_ts: u64,
    /// When the last `overflow` record was emitted (collapse gate).
    overflow_window_ms: Option<u64>,
}

impl Default for EventLane {
    fn default() -> Self {
        Self::new()
    }
}

impl EventLane {
    /// Lane with the framework defaults ([`COLLAPSE_WINDOW_MS`], [`MAX_EVENTS`]).
    pub fn new() -> Self {
        Self::with_limits(COLLAPSE_WINDOW_MS, MAX_EVENTS)
    }

    /// Lane with explicit limits — for tests that need collapse and overflow
    /// to happen without sleeping or generating 200 alerts.
    pub fn with_limits(collapse_window_ms: u64, max_events: usize) -> Self {
        EventLane {
            events: VecDeque::new(),
            collapse_window_ms,
            max_events: max_events.max(1),
            dropped_total: 0,
            dropped_pending: 0,
            dropped_ts: 0,
            overflow_window_ms: None,
        }
    }

    /// Total alerts evicted by the live cap so far.
    pub fn dropped(&self) -> u64 {
        self.dropped_total
    }

    /// Record one occurrence, returning the stream records to append (empty
    /// when the occurrence was absorbed by an open collapse window).
    ///
    /// The first occurrence of a `(class, path)` always emits — a critical
    /// alert must reach the log the moment it happens, never a collapse window
    /// later. Repeats inside the window bump the live entry only, and are
    /// accounted for by the `count` of the next record past it.
    pub fn record(
        &mut self,
        class: EventClass,
        path: &str,
        detail: impl Into<String>,
        now_ms: u64,
    ) -> Vec<Value> {
        let detail = detail.into();
        let mut out = Vec::new();

        // Newest-first scan: the live buffer is capped at a couple hundred
        // entries and alerts are rare, so a linear probe beats a second index
        // that could desync with eviction.
        let found = self
            .events
            .iter()
            .rposition(|a| a.class == class && a.path == path);
        if let Some(i) = found {
            let mut a = self.events.remove(i).expect("index came from rposition");
            a.ts = now_ms;
            a.total += 1;
            // Latest detail wins: it describes the occurrence `ts` points at.
            a.detail = detail;
            if now_ms.saturating_sub(a.window_ms) < self.collapse_window_ms {
                a.pending += 1;
            } else {
                // Window closed — this record carries itself plus everything
                // absorbed while it was open.
                let count = a.pending + 1;
                a.pending = 0;
                a.window_ms = now_ms;
                out.push(a.to_json(count));
            }
            // Re-queue at the back so the buffer is ordered by last activity:
            // "newest wins" must mean a chronic alert that is *still firing*
            // outlives a one-off that stopped, not that it is evicted for
            // having started earlier.
            self.events.push_back(a);
        } else {
            let alert = Alert {
                class,
                path: path.to_string(),
                detail,
                ts: now_ms,
                window_ms: now_ms,
                total: 1,
                pending: 0,
            };
            // Serialize before the move; `push` may emit an evicted entry's
            // owed repeats, which belong ahead of this one chronologically.
            let json = alert.to_json(1);
            self.push(alert, &mut out);
            out.push(json);
        }

        self.maybe_emit_overflow(now_ms, &mut out);
        out
    }

    /// Close every open collapse window, emitting the records that carry the
    /// still-absorbed repeats. Called at teardown so the tail of a flood is in
    /// the persisted history, not only in the live feed.
    pub fn flush(&mut self, now_ms: u64) -> Vec<Value> {
        let mut out = Vec::new();
        for a in self.events.iter_mut() {
            if a.pending > 0 {
                out.push(a.to_json(a.pending));
                a.pending = 0;
                a.window_ms = now_ms;
            }
        }
        // Force the overflow notice out regardless of its window: this is the
        // last chance for the drop count to reach the stream.
        if self.dropped_pending > 0 {
            out.push(self.overflow_json(self.dropped_pending));
            self.dropped_pending = 0;
            self.overflow_window_ms = Some(now_ms);
        }
        out
    }

    /// The current bounded feed for the portal: one entry per `(class, path)`,
    /// least-recently-active first, `count` = its running total. The
    /// `overflow` notice, when any alert was dropped, is appended last and is
    /// never itself evicted.
    pub fn live(&self) -> Vec<Value> {
        let mut out: Vec<Value> = self.events.iter().map(|a| a.to_json(a.total)).collect();
        if self.dropped_total > 0 {
            out.push(self.overflow_json(self.dropped_total));
        }
        out
    }

    /// Push a fresh entry, evicting the least-recently-active at the cap. An
    /// evicted entry still owing repeats emits them first — the cap bounds
    /// memory, it does not erase history.
    fn push(&mut self, alert: Alert, out: &mut Vec<Value>) {
        while self.events.len() >= self.max_events {
            let Some(old) = self.events.pop_front() else {
                break;
            };
            if old.pending > 0 {
                out.push(old.to_json(old.pending));
            }
            self.dropped_total += 1;
            self.dropped_pending += 1;
            self.dropped_ts = alert.ts;
        }
        self.events.push_back(alert);
    }

    /// Emit the truncation notice if drops are owed and its own collapse
    /// window has closed.
    fn maybe_emit_overflow(&mut self, now_ms: u64, out: &mut Vec<Value>) {
        if self.dropped_pending == 0 {
            return;
        }
        let due = self
            .overflow_window_ms
            .is_none_or(|t| now_ms.saturating_sub(t) >= self.collapse_window_ms);
        if !due {
            return;
        }
        out.push(self.overflow_json(self.dropped_pending));
        self.dropped_pending = 0;
        self.overflow_window_ms = Some(now_ms);
    }

    /// The `overflow` record. Path is the root: the cap is a property of the
    /// lane, not of any one node.
    fn overflow_json(&self, count: u64) -> Value {
        json!({
            "v": 1,
            "ts": self.dropped_ts,
            "sev": EventClass::Overflow.severity().as_str(),
            "path": "root",
            "kind": "event",
            "class": EventClass::Overflow.as_str(),
            "detail": format!(
                "{} alert(s) dropped by the {}-entry live cap",
                self.dropped_total, self.max_events,
            ),
            "count": count,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counts(records: &[Value], class: &str) -> Vec<u64> {
        records
            .iter()
            .filter(|r| r["class"] == class)
            .map(|r| r["count"].as_u64().unwrap())
            .collect()
    }

    #[test]
    fn first_occurrence_emits_immediately() {
        let mut lane = EventLane::new();
        let out = lane.record(EventClass::RankLost, "root/h1/rank2", "stale 34s", 1_000);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0]["kind"], "event");
        assert_eq!(out[0]["class"], "rank_lost");
        assert_eq!(out[0]["sev"], "critical");
        assert_eq!(out[0]["path"], "root/h1/rank2");
        assert_eq!(out[0]["detail"], "stale 34s");
        assert_eq!(out[0]["count"], 1);
        assert_eq!(out[0]["ts"], 1_000);
    }

    #[test]
    fn repeats_inside_the_window_are_absorbed() {
        let mut lane = EventLane::with_limits(10_000, 200);
        assert_eq!(lane.record(EventClass::Drift, "root", "d=1", 0).len(), 1);
        for t in [100, 500, 9_999] {
            assert!(lane.record(EventClass::Drift, "root", "d=1", t).is_empty());
        }
        // Live view carries the running total meanwhile.
        let live = lane.live();
        assert_eq!(live.len(), 1);
        assert_eq!(live[0]["count"], 4);
    }

    #[test]
    fn counts_over_the_stream_are_exact() {
        // 1 + 3 absorbed, then the window closes and the next record carries
        // all four; total emitted counts == total occurrences.
        let mut lane = EventLane::with_limits(1_000, 200);
        let mut emitted = 0u64;
        let mut occurrences = 0u64;
        for t in [0, 100, 200, 300, 1_500, 1_600, 3_000] {
            let out = lane.record(EventClass::Drift, "root", "d", t);
            occurrences += 1;
            emitted += counts(&out, "drift").iter().sum::<u64>();
        }
        emitted += counts(&lane.flush(9_999), "drift").iter().sum::<u64>();
        assert_eq!(emitted, occurrences);
        assert_eq!(lane.live()[0]["count"], occurrences);
    }

    #[test]
    fn distinct_paths_do_not_collapse_into_each_other() {
        let mut lane = EventLane::new();
        assert_eq!(
            lane.record(EventClass::RankLost, "root/rank0", "a", 0)
                .len(),
            1
        );
        assert_eq!(
            lane.record(EventClass::RankLost, "root/rank1", "b", 1)
                .len(),
            1
        );
        assert_eq!(lane.live().len(), 2);
    }

    #[test]
    fn distinct_classes_on_one_path_do_not_collapse() {
        let mut lane = EventLane::new();
        assert_eq!(lane.record(EventClass::Drift, "root", "a", 0).len(), 1);
        assert_eq!(
            lane.record(EventClass::ControlDrop, "root", "b", 1).len(),
            1
        );
        assert_eq!(lane.live().len(), 2);
    }

    #[test]
    fn latest_detail_and_ts_win_on_collapse() {
        let mut lane = EventLane::with_limits(10_000, 200);
        lane.record(EventClass::Drift, "root", "first", 0);
        lane.record(EventClass::Drift, "root", "latest", 500);
        let live = lane.live();
        assert_eq!(live[0]["detail"], "latest");
        assert_eq!(live[0]["ts"], 500);
    }

    #[test]
    fn cap_evicts_least_recently_active_and_says_so() {
        let mut lane = EventLane::with_limits(10_000, 2);
        lane.record(EventClass::RankLost, "root/rank0", "a", 0);
        lane.record(EventClass::RankLost, "root/rank1", "b", 1);
        let out = lane.record(EventClass::RankLost, "root/rank2", "c", 2);
        // The new alert + the non-silent truncation notice.
        assert_eq!(out.len(), 2);
        assert_eq!(out[1]["class"], "overflow");
        assert_eq!(out[1]["sev"], "warn");
        assert_eq!(out[1]["path"], "root");
        assert_eq!(out[1]["count"], 1);
        assert_eq!(lane.dropped(), 1);
        // rank0 is gone, rank1/rank2 remain, plus the overflow notice.
        let live = lane.live();
        assert_eq!(live.len(), 3);
        assert_eq!(live[0]["path"], "root/rank1");
        assert_eq!(live[2]["class"], "overflow");
    }

    #[test]
    fn overflow_notice_is_itself_collapsed() {
        let mut lane = EventLane::with_limits(10_000, 1);
        lane.record(EventClass::RankLost, "root/rank0", "a", 0);
        let a = lane.record(EventClass::RankLost, "root/rank1", "b", 1);
        assert_eq!(counts(&a, "overflow"), vec![1]);
        // Second eviction inside the window: no second notice yet...
        let b = lane.record(EventClass::RankLost, "root/rank2", "c", 2);
        assert!(counts(&b, "overflow").is_empty());
        // ...but the drop is owed and rides the next notice past the window.
        let c = lane.record(EventClass::RankLost, "root/rank3", "d", 20_000);
        assert_eq!(counts(&c, "overflow"), vec![2]);
        assert_eq!(lane.dropped(), 3);
    }

    #[test]
    fn a_still_firing_alert_outlives_a_stopped_one() {
        // rank0 fired once and stopped; rank1 keeps firing. At the cap the
        // stale one must go, not the active one.
        let mut lane = EventLane::with_limits(10_000, 2);
        lane.record(EventClass::RankLost, "root/rank0", "once", 0);
        lane.record(EventClass::Drift, "root/rank1", "again", 1);
        lane.record(EventClass::Drift, "root/rank1", "again", 2);
        lane.record(EventClass::ControlDrop, "root", "new", 3);

        let paths: Vec<String> = lane
            .live()
            .iter()
            .filter(|e| e["class"] != "overflow")
            .map(|e| e["path"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(paths, vec!["root/rank1", "root"], "{:?}", lane.live());
        assert_eq!(lane.dropped(), 1);
    }

    #[test]
    fn eviction_does_not_swallow_owed_repeats() {
        let mut lane = EventLane::with_limits(10_000, 1);
        lane.record(EventClass::Drift, "root", "x", 0);
        lane.record(EventClass::Drift, "root", "x", 10); // absorbed
        lane.record(EventClass::Drift, "root", "x", 20); // absorbed
        // Evicting the drift entry must first emit the 2 absorbed repeats.
        let out = lane.record(EventClass::RankLost, "root/rank0", "died", 30);
        assert_eq!(counts(&out, "drift"), vec![2]);
    }

    #[test]
    fn flush_closes_open_windows_and_is_idempotent() {
        let mut lane = EventLane::with_limits(10_000, 200);
        lane.record(EventClass::Drift, "root", "x", 0);
        lane.record(EventClass::Drift, "root", "x", 10);
        let out = lane.flush(100);
        assert_eq!(counts(&out, "drift"), vec![1]);
        assert!(lane.flush(200).is_empty());
    }

    #[test]
    fn idle_lane_emits_nothing() {
        let mut lane = EventLane::new();
        assert!(lane.live().is_empty());
        assert!(lane.flush(1_000).is_empty());
        assert_eq!(lane.dropped(), 0);
    }

    #[test]
    fn severities_follow_the_class() {
        assert_eq!(EventClass::RankLost.severity(), Severity::Critical);
        assert_eq!(EventClass::ControlDrop.severity(), Severity::Critical);
        assert_eq!(EventClass::Drift.severity(), Severity::Warn);
        assert_eq!(EventClass::Overflow.severity(), Severity::Warn);
    }
}
