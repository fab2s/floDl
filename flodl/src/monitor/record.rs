//! Path-keyed monitor record tree — the aggregation core of the B3 record
//! stream (`.design/monitoring-portal-b3.md`).
//!
//! Every node the monitor reports is one path-tagged record. A **leaf**
//! (a rank) carries raw per-tick values; an **interior** node (host, root)
//! carries the *same field set*, aggregated over its **direct children only**.
//! The tree is recomposed from records by splitting `path` on `/` — the format
//! is recursive, so the same builder runs at the flat coordinator today and at
//! each tier of a future hierarchy unchanged.
//!
//! ## Exact aggregation by realized work
//!
//! `Mean` is **work-weighted**: `Σ(child_value · child_work) / Σ(child_work)`.
//! The weight is *linear* realized work (samples, or the samples-proportional
//! `per_rank_batch_share`) — never ElChe's `n^γ` gradient-averaging mass, which
//! is a convergence knob and would bias the reported statistic. Because every
//! node also carries its total `work` (a `Sum`), a hierarchical work-weighted
//! mean equals the flat one:
//!
//! ```text
//! value_h = Σ_{r∈h}(v_r·w_r) / W_h,   W_h = Σ_{r∈h} w_r
//! root    = Σ_h(value_h·W_h) / Σ_h W_h  =  Σ_all(v_r·w_r) / Σ_all w_r
//! ```
//!
//! This reuses the realized-work *law* (weight by work; a zero-work
//! contribution realized nothing and is excluded — mirroring
//! [`crate::distributed`]'s `realized_work::is_realized`), **not** the tensor
//! reduce fold. Metrics are grouped coordinator-side; no wire fold is involved.
//!
//! ## absent ≠ zero
//!
//! A key not reported by any child is **absent** in the aggregate (`None`),
//! never zero-filled; means exclude non-reporting children rather than
//! averaging in a zero.
//!
//! PR1a scope: this module is the pure builder + schema + tests. Live emission,
//! the `record_scalar` reduction override, and per-window cadence land in later
//! B3 PRs.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{json, Map, Value};

use crate::metrics::EpochMetrics;

/// Record severity — doubles as the structured-log severity so a record
/// ingests into fluentd / GCP Cloud Logging as-is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// Normal metrics / curated log line.
    Info,
    /// A warning-class alert (e.g. drift).
    Warn,
    /// A critical alert (e.g. rank loss).
    Critical,
}

impl Severity {
    /// Lowercase wire token.
    pub fn as_str(self) -> &'static str {
        match self {
            Severity::Info => "info",
            Severity::Warn => "warn",
            Severity::Critical => "critical",
        }
    }
}

/// How a metric key rolls up over a node's **direct children**.
///
/// Core framework keys have a fixed reduction ([`core_reduction`]); user
/// metrics default to [`Reduction::Mean`] and may name another. The declared
/// reductions ride the `meta` record so every consumer rolls up identically.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reduction {
    /// Work-weighted mean (the default; the honest cross-rank statistic).
    Mean,
    /// Sum over children (extensive quantities: throughput, counts).
    Sum,
    /// Worst child (e.g. a data-starvation bubble surfaces upward).
    Max,
    /// Best / smallest child.
    Min,
    /// Latest value. Across children there is no time order, so this takes the
    /// highest-work child's value — the representative for a broadcast-consistent
    /// metric (e.g. the LR, where every child agrees anyway).
    Last,
}

impl Reduction {
    /// Wire token.
    pub fn as_str(self) -> &'static str {
        match self {
            Reduction::Mean => "mean",
            Reduction::Sum => "sum",
            Reduction::Max => "max",
            Reduction::Min => "min",
            Reduction::Last => "last",
        }
    }

    /// Roll `(value, work)` contributions from a node's direct children into one
    /// aggregate. `None` when nothing contributed (no child reported this key —
    /// the absent≠zero rule) or, for [`Reduction::Mean`], when the total
    /// realized work is not positive (a zero-work aggregate realized nothing).
    pub fn reduce(self, contribs: &[(f64, f64)]) -> Option<f64> {
        if contribs.is_empty() {
            return None;
        }
        match self {
            Reduction::Sum => Some(contribs.iter().map(|(v, _)| *v).sum()),
            Reduction::Max => contribs.iter().map(|(v, _)| *v).reduce(f64::max),
            Reduction::Min => contribs.iter().map(|(v, _)| *v).reduce(f64::min),
            Reduction::Mean => {
                // Work-weighted, per the realized-work law: a zero-work
                // contributor is excluded (it realized nothing), and if the
                // summed work is not positive the mean is absent, not zero.
                let wsum: f64 = contribs.iter().map(|(_, w)| *w).sum();
                if wsum > 0.0 {
                    let num: f64 = contribs.iter().map(|(v, w)| v * w).sum();
                    Some(num / wsum)
                } else {
                    None
                }
            }
            Reduction::Last => contribs
                .iter()
                .copied()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(v, _)| v),
        }
    }
}

/// Fixed reduction for a framework core metric key. `None` for non-core keys,
/// which fall to the user-declared reduction (default [`Reduction::Mean`]).
/// Core reductions are authoritative: a user cannot override `throughput` off
/// `Sum`.
pub fn core_reduction(key: &str) -> Option<Reduction> {
    Some(match key {
        "throughput" | "batch_share" => Reduction::Sum,
        "loss" => Reduction::Mean,
        "data_starve" | "compute_only_ms" => Reduction::Max,
        _ => return None,
    })
}

/// The reduction for a metric key: core (authoritative) → user-declared → Mean.
fn reduction_for(key: &str, user: &Reductions) -> Reduction {
    core_reduction(key)
        .or_else(|| user.get(key).copied())
        .unwrap_or(Reduction::Mean)
}

/// Per-metric reduction declarations for non-core user metrics.
pub type Reductions = BTreeMap<String, Reduction>;

/// Resource fields for one node; each **absent** (`None`) when unsampled
/// (absent≠zero). `gpu_util` aggregates by `Mean`, VRAM by `Sum`, and the
/// `_max` peaks by `Max`.
///
/// The gauges are sampled far more often than they are published (every
/// ~500ms on each rank, published once per reduce window or epoch), so each
/// field summarises an *interval*, not an instant. See [`ResAcc`] for how the
/// interval is accumulated and [`crate::monitor::envelope`] for why the
/// latest-wins alternative misreports a busy GPU as an idle one.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Res {
    /// GPU utilization percent, mean over the interval (then `Mean` over
    /// reporting children).
    pub gpu_util: Option<f64>,
    /// Peak GPU utilization percent over the interval (then `Max` over
    /// children — "was anything under this node pegged").
    pub gpu_util_max: Option<f64>,
    /// Allocated VRAM bytes, mean over the interval (then `Sum` over
    /// reporting children).
    pub vram_alloc: Option<f64>,
    /// Peak allocated VRAM bytes over the interval (then `Sum` over children,
    /// matching `vram_alloc` — the cohort's summed high-water mark, which is
    /// the number that answers "did we come close to OOM").
    pub vram_alloc_max: Option<f64>,
    /// Total VRAM bytes (sum over reporting children). Constant per device,
    /// so it carries no envelope.
    pub vram_total: Option<f64>,
}

/// Accumulates one rank's resource samples into a [`Res`] for the next
/// publication.
///
/// One per (rank, consumer): [`take`](Self::take) drains, so the window-report
/// consumer and the epoch-record consumer each need their own — a shared
/// accumulator would let whichever publishes first blank the interval for the
/// other. Both are fed from the same arriving sample.
#[derive(Debug, Clone, Copy, Default)]
pub struct ResAcc {
    gpu_util: crate::monitor::envelope::EnvelopeAcc,
    vram_alloc: crate::monitor::envelope::EnvelopeAcc,
    vram_total: crate::monitor::envelope::EnvelopeAcc,
}

impl ResAcc {
    /// Fold one arriving sample in. Absent fields stay absent.
    pub fn push(
        &mut self,
        gpu_util: Option<f64>,
        vram_alloc: Option<f64>,
        vram_total: Option<f64>,
    ) {
        self.gpu_util.push_opt(gpu_util);
        self.vram_alloc.push_opt(vram_alloc);
        self.vram_total.push_opt(vram_total);
    }

    /// Did anything arrive since the last drain?
    ///
    /// Replaces the previous explicit "fresh" flag: a window that saw no
    /// sample must leave `res` absent rather than repeat a stale reading, and
    /// "the accumulator is empty" states that directly.
    pub fn is_empty(&self) -> bool {
        self.gpu_util.count() == 0
            && self.vram_alloc.count() == 0
            && self.vram_total.count() == 0
    }

    /// Drain the interval into a [`Res`], resetting for the next one.
    ///
    /// `vram_total` takes the max purely as a stable pick — it is a constant
    /// per device, so mean/min/max coincide except across a device swap.
    pub fn take(&mut self) -> Res {
        let gpu = self.gpu_util.take();
        let alloc = self.vram_alloc.take();
        Res {
            gpu_util: gpu.map(|e| e.mean),
            gpu_util_max: gpu.map(|e| e.max),
            vram_alloc: alloc.map(|e| e.mean),
            vram_alloc_max: alloc.map(|e| e.max),
            vram_total: self.vram_total.take().map(|e| e.max),
        }
    }
}

/// One rank's raw contribution to the tree at a tick — the builder input.
#[derive(Debug, Clone, Default)]
pub struct Leaf {
    /// Path segments **below root**, leaf id last: `["flodl-pascal","rank1"]`,
    /// `["rank0"]`, or `[]` for a single-node root-only run (root *is* the leaf).
    pub path: Vec<String>,
    /// Linear realized work this tick — the `Mean` weight. Proportional
    /// (batch-share) or absolute (samples); must be consistent across leaves.
    pub work: f64,
    /// Present metric values recorded for this leaf (absent key = not measured).
    pub metrics: BTreeMap<String, f64>,
    /// Resource sample (each field absent when unsampled).
    pub res: Res,
    /// Physical (host-domain) device index, if known.
    pub device: Option<u8>,
    /// Whether this rank is currently alive.
    pub alive: bool,
    /// Human label for this node, when the path segment alone is not
    /// informative — a rank's GPU model, say. The path is the *identity*; this
    /// is what a legend shows next to it, which on a heterogeneous rig is the
    /// difference between "rank0, rank1" and "rank0 · RTX 5060 Ti, rank1 · GP106".
    /// Interior nodes leave it absent: a host's path segment IS its name.
    pub label: Option<String>,
}

/// A node in the path-keyed record tree. A **leaf** (`children` empty) carries
/// one rank's raw values; an **interior** node carries the same field set
/// aggregated over its direct children. `work` on every node is what makes
/// hierarchical aggregation equal the flat one.
#[derive(Debug, Clone)]
pub struct NodeRecord {
    /// Full path from root, e.g. `["root","flodl-pascal","rank1"]`.
    pub path: Vec<String>,
    /// Σ subtree realized work (a `Sum`; the `Mean` weight for the parent).
    ///
    /// **A per-record interval quantity, not a series.** Its unit is whatever
    /// the producer weighted with — absolute steps for a sub-epoch window
    /// report, `batch_share` (summing to 1.0) for an epoch-boundary record,
    /// because [`EpochMetrics`] carries shares and not per-rank sample counts.
    /// Each record's tree is internally consistent, so every rollup is exact
    /// either way; but the two cadences are not on a common scale, so `work` is
    /// meaningful *per row* and must not be plotted as one curve across them.
    pub work: f64,
    /// Metric values — raw at a leaf, aggregated over direct children at interior.
    pub metrics: BTreeMap<String, f64>,
    /// Resource fields (same aggregation rule).
    pub res: Res,
    /// Direct children (empty for a leaf).
    pub children: Vec<NodeRecord>,
    /// Leaf-only: physical device index.
    pub device: Option<u8>,
    /// Whether this node is alive: a leaf's own liveness; an interior node is
    /// alive iff any direct child is.
    pub alive: bool,
    /// Human label for a legend at the parent's level (see [`Leaf::label`]).
    pub label: Option<String>,
    /// `true` on records emitted at an **epoch boundary** (the per-epoch feed),
    /// `false` on sub-epoch window reports. Both cadences share one per-level
    /// stream — a dense interior filled by window reports, punctuated by epoch
    /// rows — so a consumer needs to tell them apart to mark an epoch on a
    /// chart or in a log table.
    pub epoch_complete: bool,
}

impl NodeRecord {
    /// `true` if this is a leaf (a rank), `false` for an interior node.
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    /// Count of alive direct children (interior); `1`/`0` for a leaf by its own
    /// liveness.
    fn alive_children(&self) -> usize {
        if self.is_leaf() {
            usize::from(self.alive)
        } else {
            self.children.iter().filter(|c| c.alive).count()
        }
    }

    /// One flat JSONL record for THIS node (no nested children), per the v1
    /// schema. Leaves carry `device`/`alive`; interior nodes carry
    /// `children`/`alive` counts. Only present resource fields are emitted.
    pub fn to_record_json(
        &self,
        ts: u64,
        tick: Option<u64>,
        epoch: Option<usize>,
        sev: Severity,
    ) -> Value {
        let mut obj = Map::new();
        obj.insert("v".into(), json!(1));
        obj.insert("ts".into(), json!(ts));
        obj.insert("sev".into(), json!(sev.as_str()));
        obj.insert("path".into(), json!(self.path.join("/")));
        obj.insert("kind".into(), json!("node"));
        // `tick` is the sub-epoch window index; epoch-boundary records have no
        // window of their own, so it is absent there. `ts` is the axis both
        // cadences share.
        if let Some(t) = tick {
            obj.insert("tick".into(), json!(t));
        }
        if let Some(e) = epoch {
            obj.insert("epoch".into(), json!(e));
        }
        if self.epoch_complete {
            obj.insert("epoch_complete".into(), json!(true));
        }
        if let Some(ref l) = self.label {
            obj.insert("label".into(), json!(l));
        }

        let mut metrics = Map::new();
        for (k, v) in &self.metrics {
            metrics.insert(k.clone(), json!(v));
        }
        obj.insert("metrics".into(), Value::Object(metrics));
        obj.insert("work".into(), json!(self.work));

        let mut res = Map::new();
        if let Some(v) = self.res.gpu_util {
            res.insert("gpu_util".into(), json!(v));
        }
        if let Some(v) = self.res.gpu_util_max {
            res.insert("gpu_util_max".into(), json!(v));
        }
        if let Some(v) = self.res.vram_alloc {
            res.insert("vram_alloc".into(), json!(v));
        }
        if let Some(v) = self.res.vram_alloc_max {
            res.insert("vram_alloc_max".into(), json!(v));
        }
        if let Some(v) = self.res.vram_total {
            res.insert("vram_total".into(), json!(v));
        }
        if !res.is_empty() {
            obj.insert("res".into(), Value::Object(res));
        }

        if self.is_leaf() {
            if let Some(d) = self.device {
                obj.insert("device".into(), json!(d));
            }
            obj.insert("alive".into(), json!(self.alive));
        } else {
            obj.insert("children".into(), json!(self.children.len()));
            obj.insert("alive".into(), json!(self.alive_children()));
        }
        Value::Object(obj)
    }

    /// Flat records for the WHOLE subtree (this node + all descendants),
    /// root-first — the JSONL the stream / persistence layer appends.
    pub fn flat_records(&self, ts: u64, tick: Option<u64>, epoch: Option<usize>) -> Vec<Value> {
        let mut out = Vec::new();
        self.push_flat(ts, tick, epoch, &mut out);
        out
    }

    fn push_flat(&self, ts: u64, tick: Option<u64>, epoch: Option<usize>, out: &mut Vec<Value>) {
        out.push(self.to_record_json(ts, tick, epoch, Severity::Info));
        for c in &self.children {
            c.push_flat(ts, tick, epoch, out);
        }
    }

    /// Build the tree from an aggregated [`EpochMetrics`] — the **epoch-boundary**
    /// half of the record stream, carrying what only the per-epoch feed has:
    /// user scalars (per-rank *and* aggregated) and the resource sample.
    ///
    /// `hosts[rank]` names each rank's host; tiering follows [`cohort_tiering`],
    /// the *same* rule the sub-epoch window tree uses, so a rank has ONE path
    /// across both cadences and the two feeds interleave in one per-level
    /// stream instead of splitting each rank in two. `extras[rank]` carries the
    /// per-rank resource sample + legend label. Work is `per_rank_batch_share`
    /// (samples-proportional; the weighted mean is identical to using absolute
    /// samples).
    ///
    /// `avg_loss` and `scalars` are the controller's **batch-weighted** means
    /// across ranks — the same law [`Reduction::Mean`] applies with
    /// `batch_share` as the weight — so they are injected at the root for keys
    /// the per-rank rollup did not produce. `loss` is always such a key:
    /// [`EpochMetrics`] has no per-rank loss, only the aggregate, so without
    /// this the tree would carry no loss at any level. Where a key exists both
    /// per-rank and aggregated, the rollup wins (it is the measured tree), and
    /// the two agree by construction.
    pub fn from_epoch_metrics(
        m: &EpochMetrics,
        hosts: Option<&[String]>,
        user_reductions: &Reductions,
        extras: &[RankExtras],
    ) -> NodeRecord {
        let n = m.device_indices.len();
        let host_refs: Vec<&str> = (0..n)
            .map(|r| hosts.and_then(|hs| hs.get(r)).map(String::as_str).unwrap_or(""))
            .collect();
        let leaves: Vec<Leaf> = (0..n)
            .map(|r| {
                let mut metrics: BTreeMap<String, f64> = m
                    .per_rank
                    .get(r)
                    .map(|hm| hm.iter().map(|(k, v)| (k.clone(), *v)).collect())
                    .unwrap_or_default();
                if let Some(&t) = m.per_rank_throughput.get(r) {
                    metrics.insert("throughput".into(), t);
                }
                if let Some(&s) = m.per_rank_batch_share.get(r) {
                    metrics.insert("batch_share".into(), s);
                }
                if let Some(&d) = m.per_rank_data_starve_ms.get(r) {
                    metrics.insert("data_starve".into(), d);
                }
                if let Some(&c) = m.per_rank_compute_only_ms.get(r) {
                    metrics.insert("compute_only_ms".into(), c);
                }
                let extra = extras.get(r);
                Leaf {
                    path: leaf_path_segments(r, &host_refs),
                    work: m.per_rank_batch_share.get(r).copied().unwrap_or(0.0),
                    metrics,
                    res: extra.map(|e| e.res).unwrap_or_default(),
                    device: m.device_indices.get(r).copied(),
                    alive: true,
                    label: extra.and_then(|e| e.label.clone()),
                }
            })
            .collect();

        let mut root = build_tree(&leaves, user_reductions);
        root.metrics.entry("loss".to_string()).or_insert(m.avg_loss);
        for (k, v) in &m.scalars {
            root.metrics.entry(k.clone()).or_insert(*v);
        }
        root.mark_epoch_complete();
        root
    }

    /// Flag this node and every descendant as an epoch-boundary record.
    fn mark_epoch_complete(&mut self) {
        self.epoch_complete = true;
        for c in &mut self.children {
            c.mark_epoch_complete();
        }
    }
}

/// Per-rank additions the metrics feed does not carry: the resource sample and
/// the legend label. Kept as one struct so the builder signature does not grow
/// a positional slice per field.
#[derive(Debug, Clone, Default)]
pub struct RankExtras {
    /// Resource sample for this rank (each field absent when unsampled).
    pub res: Res,
    /// Human label for a legend — typically the GPU model.
    pub label: Option<String>,
}

/// The cohort's tree shape from its per-rank host list (index = rank; an empty
/// string means "host unknown"): `(multi_host, root_only)`.
///
/// A host tier appears only when the cohort spans **more than one** host, and a
/// lone rank on a single host collapses to root-only (the root *is* the leaf).
///
/// **Every producer of a record path must use this.** Two producers that tier
/// differently would give one rank two paths in the same stream — the metrics
/// would silently split in half rather than interleave.
pub fn cohort_tiering(hosts: &[&str]) -> (bool, bool) {
    let mut named: Vec<&str> = hosts.iter().copied().filter(|h| !h.is_empty()).collect();
    named.sort_unstable();
    named.dedup();
    let multi_host = named.len() > 1;
    (multi_host, hosts.len() == 1 && !multi_host)
}

/// Path segments **below root** for one rank's leaf, under [`cohort_tiering`]:
/// `[]` (root-only), `["rankN"]`, or `["<host>","rankN"]`.
pub fn leaf_path_segments(rank: usize, hosts: &[&str]) -> Vec<String> {
    let (multi_host, root_only) = cohort_tiering(hosts);
    if root_only {
        return Vec::new();
    }
    let host = hosts.get(rank).copied().unwrap_or("");
    let mut p = Vec::new();
    if multi_host && !host.is_empty() {
        p.push(host.to_string());
    }
    p.push(format!("rank{rank}"));
    p
}

/// Full record path (`root`, `root/rankN`, or `root/<host>/rankN`) for one
/// rank's leaf — [`leaf_path_segments`] joined under the root.
pub fn rank_record_path(rank: usize, hosts: &[&str]) -> String {
    let mut p = vec!["root".to_string()];
    p.extend(leaf_path_segments(rank, hosts));
    p.join("/")
}

/// Build the path-keyed record tree from per-rank leaves, aggregating each
/// interior node over its **direct children** with the work-weighted law.
/// `user_reductions` names the reduction for non-core metric keys (default
/// [`Reduction::Mean`]).
pub fn build_tree(leaves: &[Leaf], user_reductions: &Reductions) -> NodeRecord {
    let entries: Vec<(&[String], &Leaf)> =
        leaves.iter().map(|l| (l.path.as_slice(), l)).collect();
    build_node(&["root".to_string()], &entries, user_reductions)
}

/// Recursively build the node at `prefix` from `entries` (each pairs a
/// remaining path with its leaf). An entry with empty remaining path terminates
/// here (a leaf); otherwise entries group by their next segment into subtrees.
fn build_node(
    prefix: &[String],
    entries: &[(&[String], &Leaf)],
    user: &Reductions,
) -> NodeRecord {
    let nested: Vec<(&[String], &Leaf)> =
        entries.iter().filter(|(p, _)| !p.is_empty()).copied().collect();

    if nested.is_empty() {
        // Leaf node: exactly one terminal entry for well-formed input.
        let leaf = entries
            .first()
            .map(|(_, l)| *l)
            .expect("build_node: node has neither children nor a terminal leaf");
        return NodeRecord {
            path: prefix.to_vec(),
            work: leaf.work,
            metrics: leaf.metrics.clone(),
            res: leaf.res,
            children: Vec::new(),
            device: leaf.device,
            alive: leaf.alive,
            label: leaf.label.clone(),
            epoch_complete: false,
        };
    }

    debug_assert!(
        entries.iter().all(|(p, _)| !p.is_empty()),
        "build_node: a node is both a leaf and an interior (mixed paths)"
    );

    // Group by next segment (BTreeMap keeps child order deterministic).
    let mut groups: BTreeMap<String, Vec<(&[String], &Leaf)>> = BTreeMap::new();
    for (p, l) in &nested {
        let (head, tail) = p.split_first().expect("nested entry has a segment");
        groups.entry(head.clone()).or_default().push((tail, *l));
    }
    let children: Vec<NodeRecord> = groups
        .iter()
        .map(|(seg, subs)| {
            let mut child_prefix = prefix.to_vec();
            child_prefix.push(seg.clone());
            build_node(&child_prefix, subs, user)
        })
        .collect();

    aggregate(prefix.to_vec(), children, user)
}

/// Aggregate an interior node from its already-built direct children.
fn aggregate(path: Vec<String>, children: Vec<NodeRecord>, user: &Reductions) -> NodeRecord {
    let work = children.iter().map(|c| c.work).sum();
    let alive = children.iter().any(|c| c.alive);

    // Metric keys = union over children; each rolls up by its reduction over
    // the children that reported it (absent excluded).
    let mut keys: BTreeSet<&str> = BTreeSet::new();
    for c in &children {
        for k in c.metrics.keys() {
            keys.insert(k.as_str());
        }
    }
    let mut metrics = BTreeMap::new();
    for k in keys {
        let red = reduction_for(k, user);
        let contribs: Vec<(f64, f64)> = children
            .iter()
            .filter_map(|c| c.metrics.get(k).map(|v| (*v, c.work)))
            .collect();
        if let Some(v) = red.reduce(&contribs) {
            metrics.insert(k.to_string(), v);
        }
    }

    let res = Res {
        gpu_util: Reduction::Mean.reduce(&res_contribs(&children, |r| r.gpu_util)),
        // Max, not Mean: the question a peak answers is "was ANYTHING under
        // this node pegged", and averaging peaks across children destroys
        // exactly that — a busy rank beside three idle ones would read as
        // quarter-busy. Same reasoning as `data_starve`.
        gpu_util_max: Reduction::Max.reduce(&res_contribs(&children, |r| r.gpu_util_max)),
        vram_alloc: Reduction::Sum.reduce(&res_contribs(&children, |r| r.vram_alloc)),
        // Sum, matching `vram_alloc`: the cohort's summed high-water mark.
        // Strictly this is sum-of-peaks rather than peak-of-sum (they differ
        // when ranks peak at different moments), and sum-of-peaks is the
        // conservative one — it answers "how much VRAM did we need to have".
        vram_alloc_max: Reduction::Sum.reduce(&res_contribs(&children, |r| r.vram_alloc_max)),
        vram_total: Reduction::Sum.reduce(&res_contribs(&children, |r| r.vram_total)),
    };

    NodeRecord {
        path,
        work,
        metrics,
        res,
        children,
        device: None,
        // An interior node's path segment is its name (a host), so it needs no
        // separate label; `epoch_complete` is stamped by the epoch builder.
        label: None,
        alive,
        epoch_complete: false,
    }
}

/// Gather `(value, child_work)` for one resource field over reporting children.
fn res_contribs(children: &[NodeRecord], f: impl Fn(&Res) -> Option<f64>) -> Vec<(f64, f64)> {
    children
        .iter()
        .filter_map(|c| f(&c.res).map(|v| (v, c.work)))
        .collect()
}

/// The `meta` record: the non-core reduction declarations, emitted once at
/// stream open (and replayed into each SSE client's catch-up preamble) so every
/// consumer rolls up identically to the controller. Core reductions are
/// implicit ([`core_reduction`]) and authoritative.
pub fn meta_record(user_reductions: &Reductions, ts: u64) -> Value {
    let mut red = Map::new();
    for (k, r) in user_reductions {
        red.insert(k.clone(), json!(r.as_str()));
    }
    json!({ "v": 1, "ts": ts, "kind": "meta", "reductions": Value::Object(red) })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leaf(path: &[&str], work: f64, metrics: &[(&str, f64)]) -> Leaf {
        Leaf {
            path: path.iter().map(|s| s.to_string()).collect(),
            work,
            metrics: metrics.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
            res: Res::default(),
            device: None,
            alive: true,
            label: None,
        }
    }

    fn m(node: &NodeRecord, key: &str) -> Option<f64> {
        node.metrics.get(key).copied()
    }

    // --- Reduction::reduce ---------------------------------------------------

    #[test]
    fn absent_is_none_not_zero() {
        // No contributor for any reduction -> absent, never a zero.
        for red in [
            Reduction::Mean,
            Reduction::Sum,
            Reduction::Max,
            Reduction::Min,
            Reduction::Last,
        ] {
            assert_eq!(red.reduce(&[]), None, "{red:?}");
        }
    }

    #[test]
    fn mean_is_work_weighted() {
        // value 0 with work 3, value 1 with work 1 -> 0.25, not the plain 0.5.
        assert_eq!(Reduction::Mean.reduce(&[(0.0, 3.0), (1.0, 1.0)]), Some(0.25));
    }

    #[test]
    fn mean_with_zero_total_work_is_absent() {
        // Every contributor did zero work: the mean realized nothing.
        assert_eq!(Reduction::Mean.reduce(&[(1.0, 0.0), (2.0, 0.0)]), None);
    }

    #[test]
    fn sum_over_reporters_only() {
        assert_eq!(Reduction::Sum.reduce(&[(2.0, 1.0), (3.0, 9.0)]), Some(5.0));
    }

    #[test]
    fn max_min_last() {
        assert_eq!(Reduction::Max.reduce(&[(2.0, 1.0), (5.0, 1.0)]), Some(5.0));
        assert_eq!(Reduction::Min.reduce(&[(2.0, 1.0), (5.0, 1.0)]), Some(2.0));
        // Last = highest-work child's value.
        assert_eq!(Reduction::Last.reduce(&[(2.0, 1.0), (5.0, 9.0)]), Some(5.0));
    }

    // --- build_tree ----------------------------------------------------------

    #[test]
    fn root_only_single_rank() {
        let root = build_tree(&[leaf(&[], 1.0, &[("loss", 0.5)])], &Reductions::new());
        assert!(root.is_leaf());
        assert_eq!(root.path, vec!["root"]);
        assert_eq!(m(&root, "loss"), Some(0.5));
        assert_eq!(root.work, 1.0);
    }

    #[test]
    fn two_ranks_sum_and_weighted_mean() {
        // rank0: loss 0.2 work 3 ; rank1: loss 0.6 work 1
        let root = build_tree(
            &[
                leaf(&["rank0"], 3.0, &[("loss", 0.2), ("throughput", 10.0)]),
                leaf(&["rank1"], 1.0, &[("loss", 0.6), ("throughput", 4.0)]),
            ],
            &Reductions::new(),
        );
        assert!(!root.is_leaf());
        assert_eq!(root.children.len(), 2);
        assert_eq!(root.work, 4.0);
        // loss = (0.2*3 + 0.6*1) / 4 = 0.3
        assert!((m(&root, "loss").unwrap() - 0.3).abs() < 1e-12);
        // throughput sums (core Sum), ignoring work.
        assert_eq!(m(&root, "throughput"), Some(14.0));
    }

    #[test]
    fn hierarchical_equals_flat() {
        // 3 ranks across 2 hosts; the root work-weighted loss must equal the
        // flat weighted mean of all 3 ranks (associativity via carried work).
        let ranks = [
            (["h1", "rank0"], 2.0, 0.10),
            (["h1", "rank1"], 3.0, 0.40),
            (["h2", "rank2"], 5.0, 0.90),
        ];
        let leaves: Vec<Leaf> = ranks
            .iter()
            .map(|(p, w, l)| leaf(&p[..], *w, &[("loss", *l)]))
            .collect();
        let root = build_tree(&leaves, &Reductions::new());

        let flat_num: f64 = ranks.iter().map(|(_, w, l)| w * l).sum();
        let flat_den: f64 = ranks.iter().map(|(_, w, _)| *w).sum();
        let flat = flat_num / flat_den;

        assert!((m(&root, "loss").unwrap() - flat).abs() < 1e-12);
        // Structure: root -> {h1 -> rank0,rank1 ; h2 -> rank2}.
        assert_eq!(root.children.len(), 2);
        assert_eq!(root.work, 10.0);
        let h1 = root.children.iter().find(|c| c.path.last().unwrap() == "h1").unwrap();
        assert_eq!(h1.children.len(), 2);
        assert_eq!(h1.work, 5.0);
    }

    #[test]
    fn uniform_field_set_and_schema_at_every_level() {
        let root = build_tree(
            &[
                leaf(&["h1", "rank0"], 1.0, &[("loss", 0.2)]),
                leaf(&["h2", "rank1"], 1.0, &[("loss", 0.4)]),
            ],
            &Reductions::new(),
        );
        // Every interior node carries the same metric keys.
        let root_keys: BTreeSet<&String> = root.metrics.keys().collect();
        for c in &root.children {
            let child_keys: BTreeSet<&String> = c.metrics.keys().collect();
            assert_eq!(root_keys, child_keys);
        }
        // Interior records serialize with children/alive counts; leaves with
        // device/alive — the "same page at every level" shape.
        let root_json = root.to_record_json(0, Some(0), None, Severity::Info);
        assert!(root_json.get("children").is_some());
        assert!(root_json.get("metrics").is_some());
        assert!(root_json.get("work").is_some());
        let leaf_node = &root.children[0].children[0];
        let leaf_json = leaf_node.to_record_json(0, Some(0), None, Severity::Info);
        assert!(leaf_json.get("alive").is_some());
        assert!(leaf_json.get("children").is_none());
    }

    #[test]
    fn user_reduction_override_and_core_wins() {
        let mut user = Reductions::new();
        user.insert("samples_seen".into(), Reduction::Sum);
        // A user tries to force throughput to Mean; core keeps it Sum.
        user.insert("throughput".into(), Reduction::Mean);
        let root = build_tree(
            &[
                leaf(&["rank0"], 1.0, &[("samples_seen", 100.0), ("throughput", 10.0)]),
                leaf(&["rank1"], 1.0, &[("samples_seen", 50.0), ("throughput", 6.0)]),
            ],
            &user,
        );
        assert_eq!(m(&root, "samples_seen"), Some(150.0)); // user Sum
        assert_eq!(m(&root, "throughput"), Some(16.0)); // core Sum wins over user Mean
    }

    #[test]
    fn absent_metric_excluded_not_zero() {
        // Only rank0 reports grad_norm; the root value is rank0's, not halved
        // by treating rank1 as a zero.
        let root = build_tree(
            &[
                leaf(&["rank0"], 1.0, &[("grad_norm", 2.0)]),
                leaf(&["rank1"], 1.0, &[("loss", 0.5)]),
            ],
            &Reductions::new(),
        );
        assert_eq!(m(&root, "grad_norm"), Some(2.0));
    }

    #[test]
    fn res_mean_sum_and_absent() {
        let mk = |util: Option<f64>, alloc: Option<f64>, work: f64| Leaf {
            path: vec!["rank".into()],
            work,
            metrics: BTreeMap::new(),
            res: Res {
                gpu_util: util,
                gpu_util_max: util,
                vram_alloc: alloc,
                vram_alloc_max: alloc,
                vram_total: None,
            },
            device: None,
            alive: true,
            label: None,
        };
        // Distinct leaf ids so they are separate children.
        let mut a = mk(Some(80.0), Some(1000.0), 3.0);
        a.path = vec!["rank0".into()];
        let mut b = mk(Some(40.0), None, 1.0);
        b.path = vec!["rank1".into()];
        let root = build_tree(&[a, b], &Reductions::new());
        // gpu_util Mean, work-weighted: (80*3 + 40*1)/4 = 70.
        assert_eq!(root.res.gpu_util, Some(70.0));
        // vram_alloc Sum over the one reporter.
        assert_eq!(root.res.vram_alloc, Some(1000.0));
        // vram_total never reported -> absent.
        assert_eq!(root.res.vram_total, None);
    }

    // --- flat records / serialization ---------------------------------------

    #[test]
    fn flat_records_cover_every_node() {
        let root = build_tree(
            &[
                leaf(&["h1", "rank0"], 1.0, &[("loss", 0.2)]),
                leaf(&["h1", "rank1"], 1.0, &[("loss", 0.4)]),
            ],
            &Reductions::new(),
        );
        // root, h1, rank0, rank1 = 4 records.
        let recs = root.flat_records(1234, Some(7), Some(3));
        assert_eq!(recs.len(), 4);
        let paths: BTreeSet<String> = recs
            .iter()
            .map(|r| r["path"].as_str().unwrap().to_string())
            .collect();
        assert!(paths.contains("root"));
        assert!(paths.contains("root/h1"));
        assert!(paths.contains("root/h1/rank0"));
        // Every record stamps the tick + epoch label.
        for r in &recs {
            assert_eq!(r["tick"], json!(7));
            assert_eq!(r["epoch"], json!(3));
            assert_eq!(r["kind"], json!("node"));
        }
    }

    #[test]
    fn leaf_record_carries_device_and_alive() {
        let mut l = leaf(&["rank0"], 1.0, &[("loss", 0.2)]);
        l.device = Some(1);
        l.alive = false;
        let root = build_tree(&[l], &Reductions::new()); // root-only? no: path non-empty
        // path ["rank0"] -> root -> rank0
        let rec = root.children[0].to_record_json(0, Some(0), None, Severity::Info);
        assert_eq!(rec["device"], json!(1));
        assert_eq!(rec["alive"], json!(false));
        // Interior root reflects the dead child in its alive count.
        let root_rec = root.to_record_json(0, Some(0), None, Severity::Info);
        assert_eq!(root_rec["alive"], json!(0));
        assert_eq!(root_rec["children"], json!(1));
    }

    #[test]
    fn meta_record_declares_user_reductions() {
        let mut user = Reductions::new();
        user.insert("samples_seen".into(), Reduction::Sum);
        let meta = meta_record(&user, 99);
        assert_eq!(meta["kind"], json!("meta"));
        assert_eq!(meta["reductions"]["samples_seen"], json!("sum"));
    }

    // --- EpochMetrics bridge -------------------------------------------------

    fn epoch_metrics_2ranks() -> EpochMetrics {
        let mut per_rank = vec![BTreeMap::new(), BTreeMap::new()];
        // record_scalar-style user metric present on both ranks.
        per_rank[0].insert("acc".to_string(), 0.90);
        per_rank[1].insert("acc".to_string(), 0.70);
        let per_rank: Vec<std::collections::HashMap<String, f64>> = per_rank
            .into_iter()
            .map(|b| b.into_iter().collect())
            .collect();
        EpochMetrics {
            epoch: 4,
            scalars: std::collections::HashMap::new(),
            per_rank,
            avg_loss: 0.3,
            epoch_ms: 100.0,
            per_rank_throughput: vec![10.0, 4.0],
            per_rank_batch_share: vec![0.75, 0.25],
            per_rank_share_complete_ms: vec![90.0, 95.0],
            per_rank_compute_only_ms: vec![80.0, 85.0],
            per_rank_data_starve_ms: vec![5.0, 40.0],
            device_indices: vec![0, 1],
        }
    }

    #[test]
    fn from_epoch_metrics_builds_weighted_tree() {
        let em = epoch_metrics_2ranks();
        let root = NodeRecord::from_epoch_metrics(&em, None, &Reductions::new(), &[]);
        assert_eq!(root.children.len(), 2);
        // work = batch_share, sums to 1.0.
        assert!((root.work - 1.0).abs() < 1e-12);
        // throughput sums.
        assert!((m(&root, "throughput").unwrap() - 14.0).abs() < 1e-9);
        // acc work-weighted: 0.90*0.75 + 0.70*0.25 = 0.85.
        assert!((m(&root, "acc").unwrap() - 0.85).abs() < 1e-12);
        // data_starve Max surfaces the worst rank.
        assert_eq!(m(&root, "data_starve"), Some(40.0));
        // device index carried onto the leaf.
        let rank1 = root.children.iter().find(|c| c.path.last().unwrap() == "rank1").unwrap();
        assert_eq!(rank1.device, Some(1));
    }

    #[test]
    fn from_epoch_metrics_hosts_tier() {
        let em = epoch_metrics_2ranks();
        let hosts = vec!["hostA".to_string(), "hostB".to_string()];
        let root = NodeRecord::from_epoch_metrics(&em, Some(&hosts), &Reductions::new(), &[]);
        // root -> hostA -> rank0 ; root -> hostB -> rank1
        assert_eq!(root.children.len(), 2);
        let host_a = root.children.iter().find(|c| c.path.last().unwrap() == "hostA").unwrap();
        assert_eq!(host_a.children.len(), 1);
        assert_eq!(host_a.children[0].path, vec!["root", "hostA", "rank0"]);
    }

    /// Co-hosted ranks get NO host tier, even though `hosts` is `Some` — the
    /// tier is a property of the cohort spanning >1 host, not of whether the
    /// caller happened to know host names. This is what keeps the epoch feed's
    /// paths identical to the sub-epoch window feed's; before
    /// [`cohort_tiering`] was shared, this case produced `root/h1/rank0` here
    /// and `root/rank0` there, splitting one rank into two paths in one stream.
    #[test]
    fn co_hosted_ranks_get_no_host_tier() {
        let em = epoch_metrics_2ranks();
        let hosts = vec!["h1".to_string(), "h1".to_string()];
        let root = NodeRecord::from_epoch_metrics(&em, Some(&hosts), &Reductions::new(), &[]);
        let paths: Vec<String> =
            root.children.iter().map(|c| c.path.join("/")).collect();
        assert_eq!(paths, vec!["root/rank0", "root/rank1"]);
    }

    /// The shared shaping helpers must agree with the tree they describe, in
    /// every cohort shape — the sub-epoch feed, the alert lane, and the epoch
    /// feed all address nodes through them.
    #[test]
    fn path_helpers_match_the_built_tree() {
        let cases: Vec<Vec<&str>> = vec![
            vec!["h1"],                 // lone rank => root-only
            vec!["h1", "h1"],           // single host => flat
            vec!["h1", "h2"],           // multi host => host tier
            vec!["", ""],               // hosts unknown => flat
        ];
        for hosts in cases {
            let leaves: Vec<Leaf> = (0..hosts.len())
                .map(|r| Leaf {
                    path: leaf_path_segments(r, &hosts),
                    work: 1.0,
                    alive: true,
                    ..Default::default()
                })
                .collect();
            let root = build_tree(&leaves, &Reductions::new());
            let mut built = Vec::new();
            collect_leaf_paths(&root, &mut built);
            let expected: Vec<String> =
                (0..hosts.len()).map(|r| rank_record_path(r, &hosts)).collect();
            let mut expected_sorted = expected.clone();
            expected_sorted.sort();
            expected_sorted.dedup();
            built.sort();
            assert_eq!(built, expected_sorted, "hosts={hosts:?}");
        }
    }

    fn collect_leaf_paths(n: &NodeRecord, out: &mut Vec<String>) {
        if n.is_leaf() {
            out.push(n.path.join("/"));
        }
        for c in &n.children {
            collect_leaf_paths(c, out);
        }
    }

    /// `EpochMetrics` has no per-rank loss — only the aggregate — so without
    /// injection the tree would carry no loss at ANY level.
    #[test]
    fn root_aggregates_fill_keys_the_rollup_cannot_produce() {
        let mut em = epoch_metrics_2ranks();
        em.scalars.insert("eval_acc".to_string(), 0.91);
        let root = NodeRecord::from_epoch_metrics(&em, None, &Reductions::new(), &[]);
        assert_eq!(m(&root, "loss"), Some(0.3), "avg_loss injected at root");
        assert_eq!(m(&root, "eval_acc"), Some(0.91), "root-only scalar injected");
        // Injection is root-only: a rank never reported these.
        assert_eq!(m(&root.children[0], "loss"), None);
        assert_eq!(m(&root.children[0], "eval_acc"), None);
    }

    /// Where a key exists BOTH per-rank and in the aggregate, the measured
    /// rollup wins — and the two agree, because `scalars` is a batch-weighted
    /// mean and the tree's `Mean` is weighted by `batch_share`. Same law.
    #[test]
    fn rollup_wins_over_injection_and_the_two_agree() {
        let mut em = epoch_metrics_2ranks();
        // acc rolls up to 0.90*0.75 + 0.70*0.25 = 0.85; the controller's own
        // batch-weighted aggregate is the same number.
        em.scalars.insert("acc".to_string(), 0.85);
        let root = NodeRecord::from_epoch_metrics(&em, None, &Reductions::new(), &[]);
        assert!((m(&root, "acc").unwrap() - 0.85).abs() < 1e-12);
    }

    #[test]
    fn epoch_records_are_marked_complete_at_every_level() {
        let em = epoch_metrics_2ranks();
        let hosts = vec!["hostA".to_string(), "hostB".to_string()];
        let root = NodeRecord::from_epoch_metrics(&em, Some(&hosts), &Reductions::new(), &[]);
        assert!(root.epoch_complete);
        assert!(root.children.iter().all(|h| h.epoch_complete));
        assert!(root.children.iter().all(|h| h.children.iter().all(|r| r.epoch_complete)));
        // ...and it reaches the wire, while a window record omits it entirely.
        let recs = root.flat_records(1, None, Some(4));
        assert!(recs.iter().all(|r| r["epoch_complete"] == true));
        assert!(recs.iter().all(|r| r.get("tick").is_none()), "no window index");
        let window = build_tree(
            &[Leaf { path: vec!["rank0".into()], work: 1.0, alive: true, ..Default::default() }],
            &Reductions::new(),
        );
        assert!(window.flat_records(1, Some(7), Some(4))[0].get("epoch_complete").is_none());
    }

    #[test]
    fn res_and_label_ride_the_epoch_leaves_and_roll_up() {
        let em = epoch_metrics_2ranks();
        let extras = vec![
            RankExtras {
                res: Res {
                    gpu_util: Some(90.0),
                    gpu_util_max: Some(95.0),
                    vram_alloc: Some(1000.0),
                    vram_alloc_max: Some(1200.0),
                    vram_total: Some(4000.0),
                },
                label: Some("RTX 5060 Ti".to_string()),
            },
            RankExtras {
                res: Res {
                    gpu_util: Some(50.0),
                    gpu_util_max: Some(99.0),
                    vram_alloc: Some(500.0),
                    vram_alloc_max: Some(600.0),
                    vram_total: Some(6000.0),
                },
                label: Some("GP106".to_string()),
            },
        ];
        let root = NodeRecord::from_epoch_metrics(&em, None, &Reductions::new(), &extras);
        // gpu_util is a work-weighted Mean: 90*0.75 + 50*0.25 = 80.
        assert!((root.res.gpu_util.unwrap() - 80.0).abs() < 1e-9);
        // The PEAK takes Max, not Mean. rank1 deliberately has the LOWER mean
        // (50) and the HIGHER peak (99), so a Mean roll-up here would give
        // 96 — the assertion only passes for Max, which is the property that
        // makes "was anything under this node pegged" answerable.
        assert_eq!(root.res.gpu_util_max, Some(99.0));
        // VRAM sums across the cohort, peak included (sum-of-peaks = how much
        // VRAM the cohort needed to have available).
        assert_eq!(root.res.vram_alloc, Some(1500.0));
        assert_eq!(root.res.vram_alloc_max, Some(1800.0));
        assert_eq!(root.res.vram_total, Some(10000.0));
        // The label is a LEAF property — a legend at the parent's level reads
        // it there; the interior node's own path segment is its name.
        let r0 = &root.children[0];
        assert_eq!(r0.label.as_deref(), Some("RTX 5060 Ti"));
        assert_eq!(root.label, None);
        let rec = r0.to_record_json(0, None, None, Severity::Info);
        assert_eq!(rec["label"], "RTX 5060 Ti");
        assert_eq!(rec["res"]["gpu_util"], 90.0);
        // An unsampled rank contributes no res at all rather than zeros.
        let bare = NodeRecord::from_epoch_metrics(&em, None, &Reductions::new(), &[]);
        assert_eq!(bare.res.gpu_util, None);
        assert!(bare.children[0].to_record_json(0, None, None, Severity::Info).get("res").is_none());
    }

    /// The interval, not the last reading. This is the whole point of `ResAcc`:
    /// ranks sample every ~500ms but publish far less often, so a GPU busy for
    /// most of an interval used to render as whatever it read at publication
    /// time — usually mid-sync, hence idle.
    #[test]
    fn res_acc_publishes_the_interval_not_the_last_reading() {
        let mut acc = ResAcc::default();
        for util in [100.0, 100.0, 100.0, 0.0] {
            acc.push(Some(util), Some(2_000.0), Some(6_000.0));
        }
        let res = acc.take();
        assert_eq!(res.gpu_util, Some(75.0), "mean over the interval");
        assert_eq!(
            res.gpu_util_max,
            Some(100.0),
            "the busy stretch survives; latest-wins would have reported 0",
        );
        assert_eq!(res.vram_total, Some(6_000.0));
    }

    /// Replaces the old explicit `fresh` flag. A window that saw no sample must
    /// leave `res` absent rather than repeat the previous value — repeating one
    /// reading across an epoch is exactly what the sub-epoch cadence exists to
    /// avoid, so this guards the property, not the mechanism that provided it.
    #[test]
    fn res_acc_drains_so_a_sampleless_interval_stays_absent() {
        let mut acc = ResAcc::default();
        acc.push(Some(80.0), Some(1_000.0), Some(4_000.0));
        assert!(!acc.is_empty());
        assert_eq!(acc.take().gpu_util, Some(80.0));

        // Nothing arrived since. Every field must be absent, NOT 80 again.
        assert!(acc.is_empty());
        let second = acc.take();
        assert_eq!(second, Res::default());
        assert_eq!(second.gpu_util, None);
        assert_eq!(second.gpu_util_max, None);
    }

    /// Fields are independent: a rank reporting GPU but no VRAM must not have
    /// the VRAM fields invented as zeros, which would then Sum into the parent.
    #[test]
    fn res_acc_keeps_absent_fields_absent() {
        let mut acc = ResAcc::default();
        acc.push(Some(60.0), None, None);
        let res = acc.take();
        assert_eq!(res.gpu_util, Some(60.0));
        assert_eq!(res.vram_alloc, None);
        assert_eq!(res.vram_alloc_max, None);
        assert_eq!(res.vram_total, None);
    }
}
