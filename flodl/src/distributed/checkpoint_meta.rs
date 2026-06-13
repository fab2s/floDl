//! Cluster training checkpoint metadata.
//!
//! Sidecar JSON file written alongside model + optimizer state when a
//! cluster training run is terminated (gracefully, by a [`max_failure`]
//! threshold breach, or because the NCCL cohort fell below the minimum
//! comm size). Carries the trajectory state a future resume API needs
//! to restore (epoch, global step, scheduler position).
//!
//! Bundle layout written next to `save_path`:
//!
//! - `<save_path>.fdl` — model params + buffers ([`crate::nn::save_checkpoint_file`])
//! - `<save_path>.optim` — optimizer state ([`crate::nn::optim::Stateful::save_state_file`])
//! - `<save_path>.meta.json` — this file
//! - `<save_path>.config.json` — Graph architecture sidecar (existing pattern, if applicable)
//!
//! All four pieces share a stem so the sidecar pattern matches the
//! existing `Graph::save_checkpoint` convention.
//!
//! [`max_failure`]: crate::distributed::cluster_coordinator::ClusterCoordinatorConfig

use serde::{Deserialize, Serialize};

use crate::tensor::{Result, TensorError};

/// Schema version. Bump when the on-disk layout changes incompatibly.
///
/// Readers reject files whose `schema_version` exceeds this constant —
/// older binaries refuse forward-incompatible bundles loudly rather
/// than silently misinterpreting fields.
///
/// Version history:
/// - 1: initial layout (epoch, global_step, sync_round, world_size_at_save,
///   save_reason)
/// - 2: adds optional `elche_state` field for Cadence/Async resume
///   (anchor + per-rank EMA throughput). v1 files still parse — the field
///   defaults to `None` via serde's `default`.
/// - 3: adds `ElCheState.trend_history` for full-fidelity resume of
///   [`crate::distributed::ddp_run::convergence::ConvergenceGuard`] state (currently just the
///   `TrendGuard` divergence ring buffer; other guards return `None`).
///   v2 files still parse — the field defaults to `None`.
pub const CHECKPOINT_META_SCHEMA_VERSION: u32 = 3;

/// ElChe trajectory snapshot for Cadence/Async resume.
///
/// Captured on `ShutdownWithSave` so a future resume API can restore the
/// heterogeneous-cadence trajectory without re-calibrating from scratch.
/// Wired into [`CheckpointMeta`] as an optional field — None for Sync
/// (no ElChe state) and for binaries that don't populate it.
///
/// Fields mirror the controller-side [`crate::distributed::ElChe`]
/// runtime state. User-set knobs (`overhead_target`, `min_anchor`,
/// `max_anchor`, `max_batch_diff`) are NOT captured here — they come
/// from the user's `DdpRunConfig` at controller construction on
/// resume, so re-binding to a different config is supported by design.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ElCheState {
    /// Anchor batches-per-cycle count. The "K" knob ElChe tunes to keep
    /// AllReduce overhead at `overhead_target`. The slow-anchor rank
    /// runs this many batches between syncs; faster ranks run
    /// proportionally more.
    pub anchor: usize,
    /// Currently elected slow-anchor rank, `None` until ElChe has run
    /// at least one calibration (`Phase::Probe` → `Phase::Warmup`
    /// transition). Resume restores anchor selection without re-running
    /// the cold-start election cycle.
    pub anchor_rank: Option<usize>,
    /// Per-rank smoothed `ms_per_batch` — mean over ElChe's trust
    /// window (5 most recent readings). Length equals
    /// `world_size_at_save`. `0.0` for ranks that haven't produced a
    /// positive reading yet. NOT throughput — directly `ms_per_batch`
    /// (the inverse). Resume seeds the trust window so cadence ratios
    /// settle without re-measurement.
    pub smoothed_ms_per_batch: Vec<f64>,
    /// Lifecycle phase. Election + anchor-swap behavior depends on
    /// phase: Probe disables election entirely, Warmup gates swaps on
    /// the Stable threshold, Stable runs hysteresis, Mature is the
    /// long-run steady state. Resume in the wrong phase causes
    /// mis-behavior in the first few cycles.
    pub phase: crate::distributed::el_che::Phase,
    /// Number of successful `report_timing` calls. Drives the
    /// phase-transition logic (Warmup → Stable → Mature) on resume.
    pub calibration_count: u64,
    /// [`crate::distributed::ddp_run::convergence::ConvergenceGuard`] resume buffer (currently
    /// `TrendGuard`'s divergence ring). Captured from the controller's
    /// boxed guard via
    /// [`crate::distributed::ddp_run::convergence::ConvergenceGuard::trend_history`] on
    /// `ShutdownWithSave`; rebuilt via
    /// [`crate::distributed::ddp_run::convergence::TrendGuard::with_history`] on resume so the
    /// first 3 cycles after resume don't silently emit `Stable` while
    /// waiting for the ring to refill (the `history.len() < 3` warm-up
    /// window inside `TrendGuard::check_trend`).
    ///
    /// `None` for guards without persisted state (`NoGuard`, `MsfGuard`)
    /// and for empty-ring snapshots (guard never observed a divergence
    /// event). v2 files default this to `None` via serde.
    #[serde(default)]
    pub trend_history: Option<Vec<f64>>,
}

/// Why the cluster wrote this checkpoint.
///
/// Marked `#[non_exhaustive]` so adding variants is a non-breaking
/// change for downstream code that pattern-matches on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum SaveReason {
    /// Normal cluster shutdown after reaching end-of-training.
    GracefulShutdown,
    /// User-configured `max_failure` threshold was breached.
    MaxFailureExceeded,
    /// NCCL cohort dropped below 2 ranks; the lone survivor cannot
    /// form a communicator (NCCL requires world_size >= 2) and saves
    /// its state before exiting.
    SingleSurvivor,
    /// CPU cohort lost its last survivor.
    AllRanksLost,
    /// A reduce cycle stalled past its hard ceiling with the cohort
    /// still alive — a scheduler wedge, not a rank failure. The
    /// coordinator saves state and shuts down instead of hanging
    /// silently.
    ReduceStall,
}

impl SaveReason {
    /// Encode for the wire as a stable single byte.
    ///
    /// The mapping is part of the wire-format contract; existing values
    /// must never change. New variants get the next available byte.
    pub const fn to_u8(self) -> u8 {
        match self {
            SaveReason::GracefulShutdown => 0,
            SaveReason::MaxFailureExceeded => 1,
            SaveReason::SingleSurvivor => 2,
            SaveReason::AllRanksLost => 3,
            SaveReason::ReduceStall => 4,
        }
    }

    /// Decode a wire byte. Unknown bytes return `None` so peers
    /// receiving a future variant can log the unknown reason and
    /// fall back to a conservative default rather than crash.
    pub fn from_u8(byte: u8) -> Option<Self> {
        match byte {
            0 => Some(SaveReason::GracefulShutdown),
            1 => Some(SaveReason::MaxFailureExceeded),
            2 => Some(SaveReason::SingleSurvivor),
            3 => Some(SaveReason::AllRanksLost),
            4 => Some(SaveReason::ReduceStall),
            _ => None,
        }
    }
}

/// Cluster checkpoint metadata sidecar.
///
/// Written to `<save_path>.meta.json` next to the model + optimizer
/// files when training terminates. Schema-versioned for forward compat
/// (see [`CHECKPOINT_META_SCHEMA_VERSION`]).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMeta {
    /// Schema version of this metadata file. Read first to determine
    /// compat — readers reject files newer than the supported version.
    pub schema_version: u32,
    /// Current epoch index (0-based) at the moment of save.
    pub epoch: usize,
    /// Global step count — total batches the cohort has trained against
    /// the model so far. Used at resume time to feed the user-supplied
    /// `Scheduler` factory and recover the LR trajectory.
    pub global_step: usize,
    /// Sync round counter — how many averaging cycles have completed.
    /// Useful for sync-cadence telemetry on resume but not load-bearing
    /// for trajectory restoration.
    pub sync_round: u64,
    /// World size at the moment of save. May be smaller than the
    /// original configured world_size if ranks died before this
    /// checkpoint (e.g. saved by the lone survivor or by a depleted
    /// cohort under `max_failure`).
    pub world_size_at_save: usize,
    /// Why this checkpoint was written.
    pub save_reason: SaveReason,
    /// ElChe trajectory snapshot for Cadence/Async resume. `None` for
    /// Sync (no per-rank cadence state) and for v1 files (defaulted on
    /// deserialize via serde).
    #[serde(default)]
    pub elche_state: Option<ElCheState>,
}

impl CheckpointMeta {
    /// Build a fresh meta record stamped with the current schema version.
    ///
    /// `elche_state` defaults to `None`; use [`Self::with_elche_state`] to
    /// attach it. Sync runs never have ElChe state to capture; Cadence /
    /// Async runs may populate it from the coord's ElChe at save time.
    pub fn new(
        epoch: usize,
        global_step: usize,
        sync_round: u64,
        world_size_at_save: usize,
        save_reason: SaveReason,
    ) -> Self {
        Self {
            schema_version: CHECKPOINT_META_SCHEMA_VERSION,
            epoch,
            global_step,
            sync_round,
            world_size_at_save,
            save_reason,
            elche_state: None,
        }
    }

    /// Attach an [`ElCheState`] snapshot. Builder-style for chaining at
    /// construction sites that have the ElChe trajectory available.
    pub fn with_elche_state(mut self, state: ElCheState) -> Self {
        self.elche_state = Some(state);
        self
    }

    /// Derive the sidecar `.meta.json` path for a checkpoint stem.
    ///
    /// Strips any single extension (so `ckpt.fdl` → `ckpt.meta.json`) so
    /// all bundle members share the same stem. A stem without extension
    /// (`ckpt_final`) gets `.meta.json` appended directly.
    pub fn sidecar_path(stem: &str) -> std::path::PathBuf {
        let mut p = std::path::PathBuf::from(stem);
        p.set_extension("meta.json");
        p
    }

    /// Serialize to JSON and write to `path`. Pretty-printed so the
    /// file is human-inspectable during recovery debugging.
    pub fn write_to_file(&self, path: &std::path::Path) -> Result<()> {
        let content = serde_json::to_string_pretty(self).map_err(|e| {
            TensorError::new(&format!(
                "CheckpointMeta: serialize JSON for {}: {e}",
                path.display(),
            ))
        })?;
        std::fs::write(path, content).map_err(|e| {
            TensorError::new(&format!(
                "CheckpointMeta: write {}: {e}",
                path.display(),
            ))
        })
    }

    /// Read JSON from `path` and deserialize. Errors loudly if the
    /// schema version is newer than this binary supports.
    pub fn read_from_file(path: &std::path::Path) -> Result<Self> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            TensorError::new(&format!(
                "CheckpointMeta: read {}: {e}",
                path.display(),
            ))
        })?;
        let meta: Self = serde_json::from_str(&content).map_err(|e| {
            TensorError::new(&format!(
                "CheckpointMeta: parse JSON from {}: {e}",
                path.display(),
            ))
        })?;
        if meta.schema_version > CHECKPOINT_META_SCHEMA_VERSION {
            return Err(TensorError::new(&format!(
                "CheckpointMeta: schema version {} in {} is newer than \
                 the version {} this binary supports",
                meta.schema_version,
                path.display(),
                CHECKPOINT_META_SCHEMA_VERSION,
            )));
        }
        Ok(meta)
    }
}

/// Last-gasp forensic record written by a cluster rank that died on an
/// unrecoverable error (the `process::exit(1)` path in
/// `run_cluster_rank_via_coord`).
///
/// Lets a postmortem / the launcher tell "a rank crashed on its own"
/// apart from "the controller asked the cohort to save and exit" (which
/// writes a [`CheckpointMeta`] via `ShutdownWithSave`): a `.death.json`
/// next to the bundle means the rank itself failed, and the `error`
/// field says why. Best-effort — written only when `save_path` is
/// configured, and a write failure is logged, never fatal (the process
/// is already on its way out).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankDeathRecord {
    /// Schema version for forward-compatible reads.
    pub schema_version: u32,
    /// Global rank that died.
    pub rank: usize,
    /// Cohort size at launch (context for the rank index).
    pub world_size: usize,
    /// The rendered error that killed the rank.
    pub error: String,
    /// Seconds since the Unix epoch at the moment of death. Best-effort:
    /// `0` if the system clock read failed.
    pub unix_time_secs: u64,
}

/// Current [`RankDeathRecord`] schema version.
pub const RANK_DEATH_RECORD_SCHEMA_VERSION: u32 = 1;

impl RankDeathRecord {
    /// Build a record stamped with the current wall-clock time.
    pub fn new(rank: usize, world_size: usize, error: String) -> Self {
        let unix_time_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        RankDeathRecord {
            schema_version: RANK_DEATH_RECORD_SCHEMA_VERSION,
            rank,
            world_size,
            error,
            unix_time_secs,
        }
    }

    /// Serialize to pretty JSON and write to `path`.
    pub fn write_to_file(&self, path: &std::path::Path) -> Result<()> {
        let content = serde_json::to_string_pretty(self).map_err(|e| {
            TensorError::new(&format!(
                "RankDeathRecord: serialize JSON for {}: {e}",
                path.display(),
            ))
        })?;
        std::fs::write(path, content).map_err(|e| {
            TensorError::new(&format!(
                "RankDeathRecord: write {}: {e}",
                path.display(),
            ))
        })
    }
}

/// Path-derivation helpers for the cluster checkpoint bundle.
///
/// Given a `save_path` stem (e.g. `"runs/job_42/ckpt_final"`), produces
/// the four bundle-member paths that share the stem:
///
/// - [`CheckpointBundle::model_path`] — `<stem>.fdl`
/// - [`CheckpointBundle::optim_path`] — `<stem>.optim`
/// - [`CheckpointBundle::meta_path`] — `<stem>.meta.json`
/// - [`CheckpointBundle::config_sidecar_path`] — `<stem>.config.json`
///
/// All four use [`std::path::PathBuf::set_extension`] which strips any
/// single existing extension before appending, so passing
/// `"ckpt_final.fdl"` as a stem still yields a consistent bundle.
pub struct CheckpointBundle;

impl CheckpointBundle {
    /// Model params + buffers path: `<stem>.fdl`.
    ///
    /// Written by [`crate::nn::save_checkpoint_file`] /
    /// [`crate::graph::Graph::save_checkpoint`].
    pub fn model_path(stem: &str) -> std::path::PathBuf {
        let mut p = std::path::PathBuf::from(stem);
        p.set_extension("fdl");
        p
    }

    /// Optimizer state path: `<stem>.optim`.
    ///
    /// Written by [`crate::nn::optim::Stateful::save_state_file`].
    pub fn optim_path(stem: &str) -> std::path::PathBuf {
        let mut p = std::path::PathBuf::from(stem);
        p.set_extension("optim");
        p
    }

    /// Metadata JSON path: `<stem>.meta.json`. Same as
    /// [`CheckpointMeta::sidecar_path`].
    pub fn meta_path(stem: &str) -> std::path::PathBuf {
        CheckpointMeta::sidecar_path(stem)
    }

    /// Graph architecture sidecar path: `<stem>.config.json`.
    ///
    /// Written by [`crate::graph::Graph::save_checkpoint`] when the
    /// graph was loaded from an HF / ONNX config; recovers the
    /// architecture description alongside the weights.
    pub fn config_sidecar_path(stem: &str) -> std::path::PathBuf {
        let mut p = std::path::PathBuf::from(stem);
        p.set_extension("config.json");
        p
    }

    /// Per-rank death record path: `<stem>.rank<N>.death.json`.
    ///
    /// Per-rank (unlike the single shared `.meta.json`) because more than
    /// one rank can die independently, and clobbering a sibling's record
    /// would lose forensic detail. Written by [`RankDeathRecord`].
    pub fn rank_death_path(stem: &str, rank: usize) -> std::path::PathBuf {
        let mut p = std::path::PathBuf::from(stem);
        // Drop a single trailing extension so the death record shares the
        // bundle stem (matches the other helpers' set_extension behavior).
        p.set_extension("");
        let name = format!(
            "{}.rank{rank}.death.json",
            p.file_name().and_then(|s| s.to_str()).unwrap_or("checkpoint"),
        );
        p.set_file_name(name);
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(label: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "flodl_meta_{label}_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn sidecar_path_from_stem_appends_meta_json() {
        let p = CheckpointMeta::sidecar_path("/tmp/run/ckpt_final");
        assert_eq!(
            p,
            std::path::PathBuf::from("/tmp/run/ckpt_final.meta.json")
        );
    }

    #[test]
    fn rank_death_path_is_per_rank_and_shares_stem() {
        // No existing extension on the stem.
        assert_eq!(
            CheckpointBundle::rank_death_path("/tmp/run/ckpt_final", 2),
            std::path::PathBuf::from("/tmp/run/ckpt_final.rank2.death.json"),
        );
        // A stem carrying a bundle extension drops it, like the siblings.
        assert_eq!(
            CheckpointBundle::rank_death_path("/tmp/run/ckpt.fdl", 0),
            std::path::PathBuf::from("/tmp/run/ckpt.rank0.death.json"),
        );
        // Distinct ranks never collide.
        assert_ne!(
            CheckpointBundle::rank_death_path("ckpt", 0),
            CheckpointBundle::rank_death_path("ckpt", 1),
        );
    }

    #[test]
    fn rank_death_record_round_trips_json() {
        let dir = temp_dir("death");
        let path = dir.join("ckpt.rank1.death.json");
        let rec = RankDeathRecord::new(1, 4, "boom: cuda OOM".to_string());
        rec.write_to_file(&path).unwrap();

        let raw = std::fs::read_to_string(&path).unwrap();
        let back: RankDeathRecord = serde_json::from_str(&raw).unwrap();
        assert_eq!(back.rank, 1);
        assert_eq!(back.world_size, 4);
        assert_eq!(back.error, "boom: cuda OOM");
        assert_eq!(back.schema_version, RANK_DEATH_RECORD_SCHEMA_VERSION);
    }

    #[test]
    fn sidecar_path_from_fdl_strips_extension() {
        let p = CheckpointMeta::sidecar_path("/tmp/run/ckpt_final.fdl");
        assert_eq!(
            p,
            std::path::PathBuf::from("/tmp/run/ckpt_final.meta.json")
        );
    }

    #[test]
    fn roundtrip_json_preserves_all_fields() {
        let dir = temp_dir("roundtrip");
        let path = dir.join("ckpt.meta.json");

        let meta = CheckpointMeta::new(
            3,
            7500,
            12,
            4,
            SaveReason::MaxFailureExceeded,
        );
        meta.write_to_file(&path).unwrap();

        let loaded = CheckpointMeta::read_from_file(&path).unwrap();
        assert_eq!(loaded.epoch, 3);
        assert_eq!(loaded.global_step, 7500);
        assert_eq!(loaded.sync_round, 12);
        assert_eq!(loaded.world_size_at_save, 4);
        assert_eq!(loaded.save_reason, SaveReason::MaxFailureExceeded);
        assert_eq!(loaded.schema_version, CHECKPOINT_META_SCHEMA_VERSION);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn save_reason_serializes_snake_case() {
        let meta = CheckpointMeta::new(0, 0, 0, 1, SaveReason::SingleSurvivor);
        let json = serde_json::to_string(&meta).unwrap();
        assert!(
            json.contains("\"save_reason\":\"single_survivor\""),
            "expected snake_case save_reason in JSON, got: {json}",
        );
    }

    #[test]
    fn bundle_paths_share_stem() {
        let stem = "/tmp/run/ckpt_final";
        assert_eq!(
            CheckpointBundle::model_path(stem),
            std::path::PathBuf::from("/tmp/run/ckpt_final.fdl"),
        );
        assert_eq!(
            CheckpointBundle::optim_path(stem),
            std::path::PathBuf::from("/tmp/run/ckpt_final.optim"),
        );
        assert_eq!(
            CheckpointBundle::meta_path(stem),
            std::path::PathBuf::from("/tmp/run/ckpt_final.meta.json"),
        );
        assert_eq!(
            CheckpointBundle::config_sidecar_path(stem),
            std::path::PathBuf::from("/tmp/run/ckpt_final.config.json"),
        );
    }

    #[test]
    fn bundle_paths_strip_existing_extension() {
        let stem = "/tmp/run/ckpt.fdl";
        // All four derived paths share the stripped stem.
        assert_eq!(
            CheckpointBundle::model_path(stem),
            std::path::PathBuf::from("/tmp/run/ckpt.fdl"),
        );
        assert_eq!(
            CheckpointBundle::optim_path(stem),
            std::path::PathBuf::from("/tmp/run/ckpt.optim"),
        );
        assert_eq!(
            CheckpointBundle::meta_path(stem),
            std::path::PathBuf::from("/tmp/run/ckpt.meta.json"),
        );
    }

    #[test]
    fn save_reason_u8_roundtrip_for_all_variants() {
        for r in [
            SaveReason::GracefulShutdown,
            SaveReason::MaxFailureExceeded,
            SaveReason::SingleSurvivor,
            SaveReason::AllRanksLost,
            SaveReason::ReduceStall,
        ] {
            let byte = r.to_u8();
            assert_eq!(SaveReason::from_u8(byte), Some(r));
        }
    }

    #[test]
    fn save_reason_from_unknown_byte_is_none() {
        assert_eq!(SaveReason::from_u8(99), None);
        assert_eq!(SaveReason::from_u8(255), None);
    }

    #[test]
    fn v1_file_loads_with_elche_state_defaulted_to_none() {
        let dir = temp_dir("v1_forward_compat");
        let path = dir.join("ckpt.meta.json");

        // v1 layout: no `elche_state` field. Schema bump to v2 added it
        // as #[serde(default)] so v1 files still parse.
        let raw_json = r#"{
            "schema_version": 1,
            "epoch": 5,
            "global_step": 10000,
            "sync_round": 25,
            "world_size_at_save": 2,
            "save_reason": "graceful_shutdown"
        }"#;
        std::fs::write(&path, raw_json).unwrap();

        let loaded = CheckpointMeta::read_from_file(&path).unwrap();
        assert_eq!(loaded.schema_version, 1);
        assert_eq!(loaded.epoch, 5);
        assert_eq!(loaded.elche_state, None);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn roundtrip_preserves_elche_state() {
        let dir = temp_dir("elche_roundtrip");
        let path = dir.join("ckpt.meta.json");

        let state = ElCheState {
            anchor: 12,
            anchor_rank: Some(1),
            smoothed_ms_per_batch: vec![5.0, 2.5, 4.0],
            phase: crate::distributed::el_che::Phase::Stable,
            calibration_count: 42,
            trend_history: Some(vec![0.01, 0.015, 0.02, 0.025, 0.03]),
        };
        let meta = CheckpointMeta::new(
            3,
            7500,
            12,
            3,
            SaveReason::GracefulShutdown,
        )
        .with_elche_state(state.clone());
        meta.write_to_file(&path).unwrap();

        let loaded = CheckpointMeta::read_from_file(&path).unwrap();
        assert_eq!(loaded.elche_state, Some(state));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn v2_file_loads_with_trend_history_defaulted_to_none() {
        // v2 layout: `elche_state` exists but no `trend_history` inside.
        // Schema bump to v3 added it as #[serde(default)] so v2 files
        // still parse. Mirror of the v1→v2 forward-compat case above.
        let dir = temp_dir("v2_forward_compat");
        let path = dir.join("ckpt.meta.json");

        let raw_json = r#"{
            "schema_version": 2,
            "epoch": 5,
            "global_step": 10000,
            "sync_round": 25,
            "world_size_at_save": 2,
            "save_reason": "graceful_shutdown",
            "elche_state": {
                "anchor": 8,
                "anchor_rank": 0,
                "smoothed_ms_per_batch": [3.0, 5.0],
                "phase": "stable",
                "calibration_count": 17
            }
        }"#;
        std::fs::write(&path, raw_json).unwrap();

        let loaded = CheckpointMeta::read_from_file(&path).unwrap();
        assert_eq!(loaded.schema_version, 2);
        let elche = loaded.elche_state.expect("v2 file carries elche_state");
        assert_eq!(elche.anchor, 8);
        assert_eq!(elche.trend_history, None);

        std::fs::remove_dir_all(&dir).ok();
    }

    // ElChe roundtrip: capture a calibrated state, build a fresh ElChe
    // with different user knobs, restore the snapshot onto it, and
    // verify the dynamic fields match while the user knobs are
    // preserved.
    #[test]
    fn elche_restore_from_state_roundtrip() {
        use crate::distributed::ElChe;

        let mut original = ElChe::new(3, 8).with_overhead_target(0.05);
        // Calibrate with three rounds of measurements so anchor + phase
        // + ms_per_batch_window all get populated.
        for _ in 0..3 {
            original.report_timing(&[5.0, 7.0, 6.0], &[8, 8, 8], 0.5);
        }
        let snap = original.to_state();
        assert!(snap.anchor_rank.is_some(), "calibrated run elects an anchor");
        assert!(
            snap.smoothed_ms_per_batch.iter().any(|&v| v > 0.0),
            "calibrated run has positive smoothed readings"
        );

        // Fresh ElChe with DIFFERENT user knobs (different overhead
        // target). Restore must preserve those knobs while seeding the
        // saved trajectory state.
        let mut restored = ElChe::new(3, 8).with_overhead_target(0.20);
        restored.restore_from_state(&snap).unwrap();

        assert_eq!(restored.to_state().anchor, snap.anchor);
        assert_eq!(restored.to_state().anchor_rank, snap.anchor_rank);
        assert_eq!(restored.to_state().phase, snap.phase);
        assert_eq!(
            restored.to_state().calibration_count,
            snap.calibration_count
        );
        assert_eq!(
            restored.to_state().smoothed_ms_per_batch,
            snap.smoothed_ms_per_batch
        );
        assert!(restored.is_calibrated(), "restored from positive-reading snap");
    }

    // Cross-size resume is a config-coherence bug, not a soft case —
    // restore_from_state must reject it loudly.
    #[test]
    fn elche_restore_from_state_rejects_world_size_mismatch() {
        use crate::distributed::ElChe;

        let mut three_rank = ElChe::new(3, 8);
        for _ in 0..3 {
            three_rank.report_timing(&[5.0, 6.0, 7.0], &[8, 8, 8], 0.5);
        }
        let snap = three_rank.to_state();

        let mut two_rank = ElChe::new(2, 8);
        let err = two_rank.restore_from_state(&snap).unwrap_err();
        assert!(
            err.to_string().contains("world_size"),
            "expected world_size mismatch error, got: {err}"
        );
    }

    // `CheckpointMeta` consumer-side: `resume_from_meta` on the coord
    // config stamps every trajectory field across.
    #[test]
    fn coord_config_resume_from_meta_applies_all_fields() {
        use crate::distributed::cluster_coordinator::ClusterCoordinatorConfig;
        use crate::distributed::ddp::ElChe;
        use crate::distributed::ddp_run::{ApplyPolicy, AverageBackend};

        let elche_state = ElCheState {
            anchor: 16,
            anchor_rank: Some(2),
            smoothed_ms_per_batch: vec![4.0, 5.0, 6.0],
            phase: crate::distributed::el_che::Phase::Mature,
            calibration_count: 99,
            trend_history: Some(vec![0.01, 0.02, 0.03]),
        };
        let meta = CheckpointMeta::new(
            7,
            42_000,
            201,
            3,
            SaveReason::GracefulShutdown,
        )
        .with_elche_state(elche_state.clone());

        let config = ClusterCoordinatorConfig::new(
            ApplyPolicy::Cadence,
            AverageBackend::Cpu,
            3,
            ElChe::new(3, 8),
        )
        .resume_from_meta(&meta);

        assert_eq!(config.start_epoch, 7);
        assert_eq!(config.start_global_step, 42_000);
        assert_eq!(config.start_avg_count, 201);
        assert_eq!(config.start_elche_state, Some(elche_state));
    }

    #[test]
    fn future_schema_version_rejected() {
        let dir = temp_dir("future");
        let path = dir.join("ckpt.meta.json");

        let raw_json = r#"{
            "schema_version": 99999,
            "epoch": 0,
            "global_step": 0,
            "sync_round": 0,
            "world_size_at_save": 1,
            "save_reason": "graceful_shutdown"
        }"#;
        std::fs::write(&path, raw_json).unwrap();

        let err = CheckpointMeta::read_from_file(&path).unwrap_err();
        assert!(
            err.to_string().contains("newer than"),
            "expected schema-version error, got: {err}",
        );

        std::fs::remove_dir_all(&dir).ok();
    }
}
