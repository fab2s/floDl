//! Live, path-addressable view of the record stream — what the portal reads
//! (`.design/monitoring-portal-b3.md`).
//!
//! [`crate::monitor::record_log::RecordLog`] is the stream's *history* on
//! disk; this is its *current state* in memory, on the controller, shaped for
//! the two questions a viewer asks:
//!
//! - **"what is node `p` doing right now?"** → the latest `node` record at `p`
//!   plus one per direct child. O(children), never O(cluster) — the same
//!   "aggregate over direct children only" property that makes the tree
//!   renderable at any depth.
//! - **"what has node `p` been doing?"** → the last N records a subscriber at
//!   `p` would have received.
//!
//! ## One predicate for stream and history
//!
//! [`delivers`] decides both what an SSE subscriber at `p` receives and what
//! `history(p, n)` returns. Sharing it is the point: a viewer reads history
//! then subscribes, so any disagreement between the two would show up as a
//! gap or a duplicate exactly at the handover.
//!
//! ## Metrics are depth-1, alerts are full-depth
//!
//! A subscriber at `p` gets `node` records for `p` and its **direct children**
//! only — that is everything the level renders, and it is what keeps a
//! subscription lean at cluster scale (a root viewer is not streaming every
//! rank). `log` / `event` records are delivered from **anywhere under `p`**:
//! an alert is rare, critical, and must reach a root viewer even from a
//! subtree nobody is looking at.

use std::collections::HashMap;

use serde_json::{json, Value};

/// Records retained in the live arrival ring. Sized so a long run keeps a
/// useful scrollback in memory; deeper history is the on-disk log's job.
pub const MAX_RECORDS: usize = 8192;

/// Distinct paths tracked for the "current state" index. The real path set is
/// bounded by cluster size (root + hosts + ranks); this only guards against a
/// malformed producer.
pub const MAX_PATHS: usize = 4096;

/// Whether a subscriber scoped to `scope` should receive `rec`.
///
/// - `meta` — always (it declares how to roll up, so every consumer needs it).
/// - `node` — `scope` itself, or a **direct child** of `scope`.
/// - anything else (`log`, `event`) — `scope` or **any descendant**.
///
/// An unknown `kind` is treated as full-depth: a record the portal does not
/// understand yet should still reach a viewer rather than vanish.
pub fn delivers(scope: &str, rec: &Value) -> bool {
    let kind = rec.get("kind").and_then(Value::as_str).unwrap_or("");
    if kind == "meta" {
        return true;
    }
    let Some(path) = rec.get("path").and_then(Value::as_str) else {
        // No path = not addressable; only an unscoped consumer could place it.
        return false;
    };
    if path == scope {
        return true;
    }
    let Some(rest) = path.strip_prefix(scope) else {
        return false;
    };
    let Some(rest) = rest.strip_prefix('/') else {
        // `scope` was a string prefix but not a path prefix
        // ("root/rank1" vs scope "root/rank10").
        return false;
    };
    if kind == "node" {
        // Direct child only: no further separator.
        !rest.contains('/')
    } else {
        true
    }
}

/// Parent path of `path`, or `None` for a root-level path.
fn parent_of(path: &str) -> Option<&str> {
    path.rsplit_once('/').map(|(p, _)| p)
}

/// The latest record at one path from **each** cadence.
///
/// One node is described by two feeds that carry different fields: sub-epoch
/// window reports (dense; framework metrics) and epoch-boundary records
/// (sparse; user scalars + the resource sample). Keeping a single "latest"
/// slot would make the two overwrite each other, so a viewer's gauges would
/// blink — with `reports_per_epoch(20)`, GPU / VRAM / accuracy would be absent
/// 19 ticks out of 20 despite having been measured. So both are kept, and
/// [`NodeLatest::merged`] answers "what is this node doing right now" as
/// last-known-per-field, which is what a gauge means.
#[derive(Debug, Default, Clone)]
struct NodeLatest {
    window: Option<Value>,
    epoch: Option<Value>,
}

impl NodeLatest {
    fn store(&mut self, rec: Value) {
        if rec.get("epoch_complete").and_then(Value::as_bool) == Some(true) {
            self.epoch = Some(rec);
        } else {
            self.window = Some(rec);
        }
    }

    /// Both cadences folded into one snapshot: the newer record's fields win,
    /// the older fills only what the newer never measured. A field absent from
    /// both stays absent — merging never invents a value.
    fn merged(&self) -> Option<Value> {
        match (&self.window, &self.epoch) {
            (None, None) => None,
            (Some(v), None) | (None, Some(v)) => Some(v.clone()),
            (Some(w), Some(e)) => {
                let ts = |v: &Value| v.get("ts").and_then(Value::as_u64).unwrap_or(0);
                let (older, newer) = if ts(w) <= ts(e) { (w, e) } else { (e, w) };
                let mut out = older.clone();
                overlay(&mut out, newer);
                // `epoch_complete` describes the RECORD that carried a field,
                // not the node's current state. Left in, a viewer would read
                // "we are at an epoch boundary" for the whole span between
                // epochs. It stays on the history rows, where it belongs.
                if let Some(obj) = out.as_object_mut() {
                    obj.remove("epoch_complete");
                }
                Some(out)
            }
        }
    }
}

/// Overlay `newer` onto `base`, recursing one level into nested objects so
/// `metrics` / `res` merge key-wise instead of the whole map being replaced.
fn overlay(base: &mut Value, newer: &Value) {
    let (Some(base_obj), Some(new_obj)) = (base.as_object_mut(), newer.as_object()) else {
        *base = newer.clone();
        return;
    };
    for (k, v) in new_obj {
        match (base_obj.get_mut(k), v.as_object()) {
            (Some(slot), Some(_)) if slot.is_object() => overlay(slot, v),
            _ => {
                base_obj.insert(k.clone(), v.clone());
            }
        }
    }
}

/// Live path-addressable record view. Plain (no interior locking) — the
/// server owns it behind its own lock, like every other piece of shared
/// dashboard state.
#[derive(Debug, Default)]
pub struct RecordStore {
    /// Arrival-ordered ring, newest last. Bounded by [`MAX_RECORDS`].
    ring: std::collections::VecDeque<Value>,
    /// Latest `node` record per path, **per cadence** — the "current state"
    /// index. See [`NodeLatest`] for why one slot is not enough.
    latest: HashMap<String, NodeLatest>,
    /// The `meta` record, if the producer has emitted one. Replayed into
    /// every subscriber's preamble. Absent is meaningful: it says "no
    /// non-core reduction declarations", and core reductions are implicit.
    meta: Option<Value>,
    /// Whether the path cap was hit (reported once by the caller).
    path_cap_hit: bool,
}

impl RecordStore {
    /// Empty store with the framework bounds.
    pub fn new() -> Self {
        Self::default()
    }

    /// Ingest records in arrival order.
    pub fn insert_all(&mut self, records: &[Value]) {
        for r in records {
            self.insert(r.clone());
        }
    }

    /// Ingest one record.
    pub fn insert(&mut self, rec: Value) {
        let kind = rec.get("kind").and_then(Value::as_str).unwrap_or("");
        if kind == "meta" {
            self.meta = Some(rec);
            return;
        }
        if kind == "node" {
            if let Some(path) = rec.get("path").and_then(Value::as_str) {
                // A known path always updates; a new one is admitted only
                // under the cap, so a malformed producer cannot grow the
                // index without bound.
                if let Some(slot) = self.latest.get_mut(path) {
                    slot.store(rec.clone());
                } else if self.latest.len() < MAX_PATHS {
                    self.latest.entry(path.to_string()).or_default().store(rec.clone());
                } else {
                    self.path_cap_hit = true;
                }
            }
        }
        if self.ring.len() >= MAX_RECORDS {
            self.ring.pop_front();
        }
        self.ring.push_back(rec);
    }

    /// The stored `meta` record, if any.
    pub fn meta(&self) -> Option<&Value> {
        self.meta.as_ref()
    }

    /// Take the "path cap was hit" flag — true once per breach so the caller
    /// can warn without repeating.
    pub fn take_path_cap_hit(&mut self) -> bool {
        std::mem::take(&mut self.path_cap_hit)
    }

    /// Current state at exactly `path`: the latest sub-epoch window report and
    /// the latest epoch-boundary record folded into one snapshot, newer fields
    /// winning and older ones filling only what the newer never measured. So a
    /// dense window report never blanks out the resource sample and user
    /// scalars that only the epoch record carries.
    pub fn node(&self, path: &str) -> Option<Value> {
        self.latest.get(path).and_then(NodeLatest::merged)
    }

    /// Current state of each direct child of `path`, ordered by path so a
    /// rendered level is stable across polls.
    pub fn children(&self, path: &str) -> Vec<Value> {
        let mut kids: Vec<(&str, Value)> = self
            .latest
            .iter()
            .filter(|(p, _)| parent_of(p) == Some(path))
            .filter_map(|(p, v)| v.merged().map(|m| (p.as_str(), m)))
            .collect();
        kids.sort_unstable_by_key(|(p, _)| *p);
        kids.into_iter().map(|(_, v)| v).collect()
    }

    /// One-shot snapshot of a level: the node itself plus its direct
    /// children. `node` is `null` when nothing has been reported at `path`
    /// yet (a viewer opening a path before its first window) — an empty
    /// level, not an error.
    pub fn snapshot(&self, path: &str) -> Value {
        json!({
            "path": path,
            "node": self.node(path).unwrap_or(Value::Null),
            "children": self.children(path),
        })
    }

    /// The last `n` records a subscriber at `path` would have received, in
    /// arrival order. Uses the same [`delivers`] predicate as the live
    /// stream, so read-then-subscribe has no seam.
    pub fn history(&self, path: &str, n: usize) -> Vec<&Value> {
        let mut out: Vec<&Value> = self
            .ring
            .iter()
            .rev()
            .filter(|r| delivers(path, r))
            .take(n)
            .collect();
        out.reverse();
        out
    }

    /// Every path currently known to carry a `node` record, sorted — the
    /// portal's navigation index.
    pub fn paths(&self) -> Vec<&str> {
        let mut v: Vec<&str> = self.latest.keys().map(String::as_str).collect();
        v.sort_unstable();
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(path: &str, tick: u64) -> Value {
        json!({ "v": 1, "kind": "node", "path": path, "tick": tick,
                "metrics": { "loss": 0.5 }, "work": 10.0 })
    }

    fn event(path: &str, class: &str) -> Value {
        json!({ "v": 1, "kind": "event", "path": path, "class": class,
                "sev": "critical", "detail": "x", "count": 1 })
    }

    // --- delivers: the shared scoping predicate ---

    #[test]
    fn node_records_reach_the_level_and_its_direct_children_only() {
        assert!(delivers("root", &node("root", 1)));
        assert!(delivers("root", &node("root/exa", 1)));
        // A grandchild's metrics are NOT streamed to a root viewer: the level
        // renders from direct children, and this is what keeps the
        // subscription lean at scale.
        assert!(!delivers("root", &node("root/exa/rank0", 1)));
        // ...but they are streamed to a viewer scoped at that host.
        assert!(delivers("root/exa", &node("root/exa/rank0", 1)));
        assert!(!delivers("root/exa", &node("root", 1)));
    }

    #[test]
    fn alerts_reach_a_root_viewer_from_any_depth() {
        // The deliberate asymmetry: a rank death in an unwatched subtree must
        // still surface at the root.
        assert!(delivers("root", &event("root/exa/rank0", "rank_lost")));
        assert!(delivers("root", &event("root", "control_drop")));
        assert!(delivers("root/exa", &event("root/exa/rank0", "rank_lost")));
        // But an alert from a sibling subtree is not this viewer's.
        assert!(!delivers("root/exa", &event("root/pascal/rank1", "rank_lost")));
    }

    #[test]
    fn a_string_prefix_is_not_a_path_prefix() {
        // "root/rank1" must not be treated as inside "root/rank10".
        assert!(!delivers("root/rank10", &node("root/rank1", 1)));
        assert!(!delivers("root/exa", &node("root/exabyte/rank0", 1)));
        assert!(delivers("root/exa", &node("root/exa/rank0", 1)));
    }

    #[test]
    fn meta_reaches_every_subscriber_and_a_pathless_record_reaches_none() {
        let meta = json!({ "v": 1, "kind": "meta", "reductions": {} });
        assert!(delivers("root", &meta));
        assert!(delivers("root/exa/rank0", &meta));
        assert!(!delivers("root", &json!({ "v": 1, "kind": "node" })));
    }

    #[test]
    fn an_unknown_kind_is_delivered_rather_than_dropped() {
        // Forward compatibility: a record the portal does not understand yet
        // still reaches the viewer, at full depth.
        let odd = json!({ "v": 1, "kind": "future", "path": "root/exa/rank0" });
        assert!(delivers("root", &odd));
    }

    // --- store ---

    #[test]
    fn snapshot_is_the_level_plus_direct_children() {
        let mut s = RecordStore::new();
        s.insert_all(&[
            node("root", 1),
            node("root/exa", 1),
            node("root/exa/rank0", 1),
            node("root/pascal", 1),
        ]);
        let snap = s.snapshot("root");
        assert_eq!(snap["node"]["path"], "root");
        let kids: Vec<&str> = snap["children"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c["path"].as_str().unwrap())
            .collect();
        // Sorted, direct children only — the grandchild is absent.
        assert_eq!(kids, vec!["root/exa", "root/pascal"]);
    }

    #[test]
    fn snapshot_of_an_unreported_path_is_empty_not_an_error() {
        let s = RecordStore::new();
        let snap = s.snapshot("root/exa");
        assert_eq!(snap["node"], Value::Null);
        assert!(snap["children"].as_array().unwrap().is_empty());
    }

    #[test]
    fn latest_node_wins_per_path() {
        let mut s = RecordStore::new();
        s.insert(node("root", 1));
        s.insert(node("root", 7));
        assert_eq!(s.node("root").unwrap()["tick"], 7);
        // ...while the ring keeps both for history.
        assert_eq!(s.history("root", 10).len(), 2);
    }

    #[test]
    fn history_matches_what_the_stream_would_deliver() {
        let mut s = RecordStore::new();
        s.insert_all(&[
            node("root", 1),
            node("root/exa", 1),
            node("root/exa/rank0", 1),
            event("root/exa/rank0", "rank_lost"),
        ]);
        let h: Vec<&str> = s
            .history("root", 100)
            .iter()
            .map(|r| r["path"].as_str().unwrap())
            .collect();
        // root + direct child metrics + the deep alert; NOT the grandchild's
        // node record — exactly the live subscription's shape.
        assert_eq!(h, vec!["root", "root/exa", "root/exa/rank0"]);
        assert_eq!(s.history("root", 100)[2]["kind"], "event");
    }

    #[test]
    fn history_returns_the_newest_n_in_arrival_order() {
        let mut s = RecordStore::new();
        for t in 1..=10 {
            s.insert(node("root", t));
        }
        let h = s.history("root", 3);
        let ticks: Vec<u64> = h.iter().map(|r| r["tick"].as_u64().unwrap()).collect();
        assert_eq!(ticks, vec![8, 9, 10]);
    }

    #[test]
    fn the_ring_is_bounded_and_drops_oldest() {
        let mut s = RecordStore::new();
        for t in 0..(MAX_RECORDS as u64 + 50) {
            s.insert(node("root", t));
        }
        assert_eq!(s.ring.len(), MAX_RECORDS);
        let h = s.history("root", 1);
        assert_eq!(h[0]["tick"], MAX_RECORDS as u64 + 49);
        // The current-state index is unaffected by ring eviction.
        assert_eq!(s.node("root").unwrap()["tick"], MAX_RECORDS as u64 + 49);
    }

    #[test]
    fn the_path_index_is_capped_and_says_so() {
        let mut s = RecordStore::new();
        for i in 0..MAX_PATHS {
            s.insert(node(&format!("root/h{i}"), 1));
        }
        assert!(!s.take_path_cap_hit(), "no breach yet");
        s.insert(node("root/one-too-many", 1));
        assert!(s.take_path_cap_hit(), "breach reported");
        assert!(!s.take_path_cap_hit(), "and reported only once");
        assert_eq!(s.node("root/one-too-many"), None);
        // A path already in the index still updates past the cap.
        s.insert(node("root/h0", 9));
        assert_eq!(s.node("root/h0").unwrap()["tick"], 9);
    }

    #[test]
    fn meta_is_retained_for_replay_and_absent_by_default() {
        let mut s = RecordStore::new();
        assert!(s.meta().is_none());
        s.insert(json!({ "v": 1, "kind": "meta", "reductions": { "acc": "mean" } }));
        assert_eq!(s.meta().unwrap()["reductions"]["acc"], "mean");
        // `meta` is state, not history — it does not enter the ring.
        assert!(s.history("root", 10).is_empty());
    }

    fn epoch_node(path: &str, ts: u64) -> Value {
        json!({ "v": 1, "kind": "node", "path": path, "ts": ts, "epoch": 3,
                "epoch_complete": true, "work": 1.0, "label": "RTX 5060 Ti",
                "metrics": { "loss": 0.4, "accuracy": 0.9 },
                "res": { "gpu_util": 84.0, "vram_alloc": 5000.0 } })
    }
    fn window_node(path: &str, ts: u64, tick: u64) -> Value {
        json!({ "v": 1, "kind": "node", "path": path, "ts": ts, "tick": tick,
                "work": 10.0, "metrics": { "loss": 0.31, "throughput": 21.0 } })
    }

    /// The whole point of keeping both cadences: a window report must not blank
    /// out the resource sample and user scalars that only the epoch record
    /// carries. With reports_per_epoch(20) the naive single-slot index would
    /// show them 1 tick in 20.
    #[test]
    fn a_window_report_does_not_blank_the_epoch_fields() {
        let mut s = RecordStore::new();
        s.insert(epoch_node("root/rank0", 100));
        s.insert(window_node("root/rank0", 200, 7));
        let n = s.node("root/rank0").unwrap();
        // Newer window record wins on the fields it measures...
        assert_eq!(n["metrics"]["loss"], 0.31);
        assert_eq!(n["metrics"]["throughput"], 21.0);
        assert_eq!(n["tick"], 7);
        assert_eq!(n["work"], 10.0);
        // ...and the epoch-only fields survive rather than blinking out.
        assert_eq!(n["metrics"]["accuracy"], 0.9);
        assert_eq!(n["res"]["gpu_util"], 84.0);
        assert_eq!(n["label"], "RTX 5060 Ti");
        // But record provenance does NOT survive: a merged snapshot is the
        // node's current state, not "we are at an epoch boundary".
        assert!(n.get("epoch_complete").is_none(), "{n}");
        assert!(
            s.history("root/rank0", 10)
                .iter()
                .any(|r| r.get("epoch_complete").and_then(Value::as_bool) == Some(true)),
            "the flag stays on the history row",
        );
    }

    /// Order-independence: an epoch record arriving after a window report must
    /// not lose the window's dense metrics either.
    #[test]
    fn merge_is_by_timestamp_not_arrival() {
        let mut s = RecordStore::new();
        s.insert(window_node("root/rank0", 200, 7));
        s.insert(epoch_node("root/rank0", 100)); // older, arrives later
        let n = s.node("root/rank0").unwrap();
        assert_eq!(n["metrics"]["loss"], 0.31, "newer window loss still wins");
        assert_eq!(n["metrics"]["accuracy"], 0.9);
        assert_eq!(n["res"]["gpu_util"], 84.0);
    }

    /// Merging fills gaps; it never invents a value.
    #[test]
    fn a_field_absent_from_both_stays_absent() {
        let mut s = RecordStore::new();
        s.insert(window_node("root/rank0", 100, 1));
        s.insert(epoch_node("root/rank0", 200));
        let n = s.node("root/rank0").unwrap();
        assert!(n["metrics"].get("data_starve").is_none());
        assert!(n["res"].get("vram_total").is_none());
    }

    /// History is untouched by the merge — each record stays its own row, which
    /// is what makes the two cadences interleave in a level's log.
    #[test]
    fn history_keeps_both_cadences_as_separate_rows() {
        let mut s = RecordStore::new();
        s.insert(window_node("root", 100, 1));
        s.insert(window_node("root", 200, 2));
        s.insert(epoch_node("root", 300));
        s.insert(window_node("root", 400, 3));
        let h = s.history("root", 10);
        assert_eq!(h.len(), 4);
        let marks: Vec<bool> = h
            .iter()
            .map(|r| r.get("epoch_complete").and_then(Value::as_bool) == Some(true))
            .collect();
        assert_eq!(marks, vec![false, false, true, false], "epoch row is marked");
    }

    #[test]
    fn paths_is_the_sorted_navigation_index() {
        let mut s = RecordStore::new();
        s.insert_all(&[node("root/pascal", 1), node("root", 1), node("root/exa", 1)]);
        assert_eq!(s.paths(), vec!["root", "root/exa", "root/pascal"]);
    }
}
