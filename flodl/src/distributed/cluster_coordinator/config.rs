//! Builder-style configuration for [`super::ClusterCoordinator`].
//!
//! Holds the user-facing knobs (policy, backend, world size, ElChe,
//! convergence guard, dead-rank ledger, checkpoint cadence, resume
//! kickoff state, etc.). Constructed via [`ClusterCoordinatorConfig::new`]
//! and refined with chained-setter methods before being consumed by
//! [`super::ClusterCoordinator::start`] /
//! [`super::ClusterCoordinator::start_from_listener`].

use std::sync::{Arc, mpsc};

use crate::distributed::ddp::ElChe;
use crate::distributed::ddp_run::convergence::{ConvergenceGuard, LevelGuard, NoGuard};
use crate::distributed::ddp_run::{ApplyPolicy, AverageBackend};

use super::NCCL_RENDEZVOUS_TIMEOUT_SECS;

/// Configuration for [`super::ClusterCoordinator::start`]. Carries the
/// fields the controller needs to drive NCCL/CPU averaging, ElChe
/// ownership, ConvergenceGuard, dispatch, and the user-callback
/// surface ([`epoch_fn`], [`checkpoint_fn`], [`metrics_fn`],
/// [`eval_fn`], etc.).
///
/// [`epoch_fn`]: crate::distributed::ddp_run::EpochFn
/// [`checkpoint_fn`]: crate::distributed::ddp_run::CheckpointFn
/// [`metrics_fn`]: crate::distributed::ddp_run::MetricsFn
/// [`eval_fn`]: crate::distributed::ddp_run::EvalFn
pub struct ClusterCoordinatorConfig {
    pub policy: ApplyPolicy,
    pub backend: AverageBackend,
    pub world_size: usize,
    pub el_che: ElChe,
    /// Boxed convergence guard. Defaults to [`LevelGuard::default()`]
    /// when omitted in the builder; set [`NoGuard`] to disable.
    pub convergence_guard: Box<dyn ConvergenceGuard>,
    /// Allow ElChe anchor relax-up on Stable convergence verdicts.
    pub elche_relax_up: bool,
    /// Initial max-overshoot (Async-only). Auto-tuned upward on Stable
    /// verdicts; reset to `overshoot_initial` on NudgeDown.
    pub overshoot_initial: usize,
    pub overshoot_ceiling: usize,
    /// True when the user did not set `max_overshoot` explicitly; lets
    /// `trigger_averaging` adjust the bound on convergence verdicts.
    pub overshoot_auto: bool,

    /// Total samples in the dataset; basis for partition sizing in
    /// [`super::ClusterCoordinator::dispatch_epoch`]. Default 0 (caller must
    /// set via [`Self::total_samples`] before dispatching epochs).
    ///
    /// Always the whole pick space, independent of [`Self::epoch_splits`]:
    /// what a split changes is how much of it one epoch covers, not how
    /// big the space is.
    pub total_samples: usize,
    /// Slices per data pass. `1` (default) keeps an epoch a full pass.
    ///
    /// Above `1`, an epoch covers `total_samples / epoch_splits` picks and
    /// the epoch index counts events. Must match the value every rank was
    /// built with (see
    /// [`crate::distributed::ddp_run::WorkerConfig::epoch_splits`]) —
    /// a divergent value puts the coordinator's ledger and the rank's
    /// expansion on different slices of the permutation.
    pub epoch_splits: usize,
    /// Batch size; consumed by the progressive chunk-pool dispatch
    /// path to size per-chunk batches.
    pub batch_size: usize,
    /// Total number of epochs to train; informs `dispatch_epoch`'s
    /// out-of-range guard. Default 0 = unbounded (caller controls).
    pub num_epochs: usize,
    /// User-specified per-rank partition ratios. When `Some`, takes
    /// precedence over ElChe-derived throughput sizing. Length must
    /// equal `world_size` if set.
    pub partition_ratios: Option<Vec<f64>>,
    /// Enable the LR-aware meta-controller above ElChe. Default:
    /// `true` (always on — LR drops are always worth catching; opt out
    /// for unconditioned-trajectory runs). When enabled, a
    /// [`crate::distributed::lr_event_meta::LrEventMeta`] is constructed
    /// and held by the coordinator; per-rank LR updates from
    /// `LrUpdate` populate `last_lr_per_rank`, and the
    /// meta is consulted after every averaging-cycle guard verdict
    /// (see the coord's private `observe_meta` hook).
    /// `MetaAction::NudgeDown` dispatches to
    /// [`crate::distributed::ddp::ElChe::nudge_anchor_down`].
    pub meta_controller: bool,

    /// Shared dead-rank ledger (with the cluster controller). When the
    /// coord declares a rank dead (stale heartbeat), it sets the
    /// ledger flag and shuts down the rank's controller-side stream
    /// so any in-flight AllReduce releases with surviving ranks only.
    /// Pass the same Arc clone to
    /// `start_with_dead_ranks`.
    /// `None` = elastic-membership disabled.
    pub dead_ranks: Option<Arc<crate::distributed::controller::DeadRanks>>,
    /// Externally-reported death queue (launcher child supervision).
    /// Drained each tick through the same side-effect chain as
    /// heartbeat-staleness detection — a faster detector, not a second
    /// policy. `None` when no external reporter exists.
    pub reported_deaths: Option<crate::distributed::cluster_coordinator::ReportedDeaths>,

    /// Launcher-owned abort flag polled by the cohort-formation accept
    /// loop ([`super::ClusterCoordinator::start_from_listener`]): when
    /// set, a coord still waiting for relay connections bails with a
    /// loud "aborted" error instead of blocking in `accept()` forever.
    /// This is how the launcher's failure path can stop and JOIN a
    /// pre-rendezvous coordinator thread and surface the original error
    /// through `DdpHandle::join` (previously it had to
    /// `process::exit(1)`). `None` (tests, standalone use) keeps the
    /// plain blocking accept.
    pub abort: Option<Arc<std::sync::atomic::AtomicBool>>,

    /// Heartbeat staleness threshold (seconds). If a rank's last
    /// TimingMsg-frame arrival is older than this, the coord declares
    /// the rank dead. Default 30s. Ignored when `dead_ranks` is None.
    pub heartbeat_timeout_secs: u64,

    /// Wall-budget (seconds) for an in-flight NCCL re-rendezvous to
    /// complete. Default 5s. On expiry, the coord retries from the
    /// next survivor in the rendezvous's `survivors_ordered`
    /// (excluding already-tried + now-dead ranks). When the candidate
    /// pool is exhausted the fallback is `dispatch_shutdown_with_save`.
    /// Tunable so tests can trigger the retry in well under 1s instead
    /// of waiting the production default.
    pub rendezvous_timeout_secs: u64,

    /// Wall-budget (seconds) for the post-join formation phase: every
    /// admitted host's relay must dial the coordinator's control plane
    /// (covering all `world_size` ranks) within this window of the
    /// accept loop starting. The join window bounds worker *admission*;
    /// this bounds the dial-in that follows it — without it, a relay
    /// that registered but never dials (crashed between spawn and
    /// connect, one dead leg of a split relay) left the accept loop
    /// spinning until the whole cohort died of its own deadlines. On
    /// expiry the coordinator errors loudly, naming the coverage
    /// reached. Default `scaled_deadline_secs(60)` (see
    /// `FLODL_NET_TIMEOUT_SCALE`).
    pub formation_timeout_secs: u64,

    /// Global ranks running on the same host as the coordinator
    /// process (the launcher's host in production, the test
    /// process in tests). NCCL re-rendezvous prefers picking the
    /// UID generator from this set when any are alive — same-host
    /// rank is same-process latency for the UID hand-off and has
    /// lower correlated-failure risk than a network peer. Defaults
    /// to empty (no preference; picker falls back to fastest
    /// surviving network rank by its window-ledger per-batch wall,
    /// breaking ties by lowest global rank).
    pub local_ranks: Vec<usize>,

    /// Host name per global rank (index = rank), from the launcher's
    /// world map. Used to host-qualify the rank-reported resource
    /// samples deposited into the timeline — device indices collide
    /// across hosts, so a sample without its host is ambiguous.
    /// Defaults to empty (samples deposit with an empty host).
    pub rank_hosts: Vec<String>,

    /// Threshold for declaring a cluster run unrecoverable.
    ///
    /// When the count of dead ranks reaches this limit, the coord
    /// broadcasts a save-and-shutdown signal to all survivors so they
    /// can persist model + optimizer + meta state to disk before
    /// exiting (rather than hanging indefinitely waiting for a
    /// re-rendezvous that cannot complete).
    ///
    /// `None` = no user-configured threshold; backend hard limits
    /// still apply (NCCL: lone survivor triggers, since NCCL requires
    /// `world_size >= 2`; CPU: all-dead triggers). Defaults to `None`.
    pub max_failure: Option<crate::distributed::max_failure::MaxFailureThreshold>,

    /// Checkpoint bundle stem for unrecoverable-failure persistence.
    ///
    /// When the coord broadcasts `ShutdownWithSave`, workers write a
    /// bundle (`.fdl` model, `.optim` optimizer state, `.meta.json`
    /// trajectory) with this stem. See
    /// [`crate::distributed::CheckpointBundle`] for path derivation.
    /// `None` in standalone-coord tests; production cluster builders
    /// require this to be set (loud error at `builder.run()` time).
    pub save_path: Option<String>,

    /// Cadence for `epoch_fn`-equivalent user checkpoint callback. When
    /// set, the coord emits
    /// `Checkpoint` every
    /// `checkpoint_every` epochs (right before dispatching the next
    /// epoch plan). Workers handle this on the rank chosen by
    /// [`crate::distributed::ddp_run::EpochCallbackPolicy`]; others see
    /// no-op. `None` or `0` = disabled.
    pub checkpoint_every: Option<usize>,

    /// User-supplied per-epoch metrics callback. Fires on the
    /// coordinator after `MetricsMsgWire`
    /// frames from every alive rank have been aggregated into
    /// [`crate::distributed::ddp_run::EpochMetrics`]. `None` = no
    /// callback wired; aggregation still happens (used internally by
    /// the elastic-balancer + future polling surfaces).
    pub metrics_fn: Option<crate::distributed::ddp_run::MetricsFn>,

    /// Optional sink for aggregated
    /// [`crate::distributed::ddp_run::EpochMetrics`]. Populated by the
    /// launcher trampoline so the user's
    /// [`crate::distributed::DdpHandle::next_metrics`] polling loop
    /// drains aggregates as they're produced. Fed in parallel with
    /// `metrics_fn` — both callbacks/sinks receive the same value.
    /// `None` skips the sink emit.
    pub metrics_sink_tx: Option<mpsc::Sender<crate::distributed::ddp_run::EpochMetrics>>,

    /// User-supplied eval-result callback (controller-side). Fires
    /// when the chosen rank's eval pass returns a scalar metric over
    /// the wire. `None` = no callback wired; result is logged to
    /// stderr instead.
    pub eval_result_fn: Option<crate::distributed::ddp_run::EvalResultFn>,

    /// Eval cadence (in epochs). `Some(n)` triggers an eval dispatch
    /// every `n` epochs from `dispatch_epoch`. `None` or `0` disables.
    pub eval_every_epochs: Option<usize>,

    /// Sub-epoch monitor-report cadence: how many per-window records to
    /// emit per epoch. `Some(x)` reports at the first reduce boundary
    /// past each `epoch_work / x` slice of realized work; `None` or `0`
    /// disables sub-epoch reporting (per-epoch metrics are unaffected
    /// either way).
    ///
    /// Expressed per-epoch rather than in steps because the step count
    /// per epoch is a derived quantity the user should not have to
    /// compute; a single-epoch run degenerates to "x reports over the
    /// whole run", which is the point for one-pass LLM training.
    pub reports_per_epoch: Option<usize>,

    /// Directory for the persisted monitor record stream, or `None` for
    /// live-only. Read by the **launcher** (which owns the dashboard sink
    /// the records flow through), not by the coordinator itself — this is
    /// the controller-scope config bag that carries it there.
    pub record_log_dir: Option<String>,

    /// Per-node byte cap for the persisted record stream (drop-oldest
    /// ring). `None` = the library default.
    pub max_log_size: Option<u64>,

    /// Where the launcher writes the self-contained dashboard archive at
    /// teardown. Like `record_log_dir`, carried here for the launcher rather
    /// than used by the coordinator itself.
    pub dashboard_html: Option<String>,

    /// Theme pinned into the saved archive, carried to the launcher alongside
    /// `dashboard_html`. `None` = leave it to the reader's OS preference.
    pub dashboard_theme: Option<String>,

    /// User-scalar roll-up declarations. Like `record_log_dir`, read by the
    /// **launcher** (which owns the dashboard sink the records flow through)
    /// rather than by the coordinator itself; this is the controller-scope bag
    /// that carries them there. Empty = every non-core key rolls up as `Mean`.
    pub scalar_reductions: crate::monitor::record::Reductions,

    /// Which rank should fire user-supplied per-epoch callbacks
    /// (`epoch_fn` / `checkpoint_fn` / `eval_fn`). Default
    /// [`EpochCallbackPolicy::Fastest`] — coord runtime-resolves the
    /// role from ElChe's per-rank smoothed throughput, with sticky
    /// retention across cadences and re-resolution only on rank death.
    /// Pin to a specific rank with [`EpochCallbackPolicy::Rank`] when
    /// the research convention demands it.
    ///
    /// [`EpochCallbackPolicy::Fastest`]:
    ///     crate::distributed::ddp_run::EpochCallbackPolicy::Fastest
    /// [`EpochCallbackPolicy::Rank`]:
    ///     crate::distributed::ddp_run::EpochCallbackPolicy::Rank
    pub epoch_callback_policy: crate::distributed::ddp_run::EpochCallbackPolicy,

    /// Progressive dispatch toggle. `None` = auto (true for
    /// `Cadence` / `Async`, false for `Sync`). `Some(true)` or
    /// `Some(false)` is the explicit user override from
    /// [`crate::distributed::ddp_run::DdpRunConfig::progressive_dispatch`]. When
    /// enabled, the coordinator streams work in small chunks
    /// adapting to measured throughput instead of dispatching one
    /// full per-rank partition per epoch.
    pub progressive: Option<bool>,

    /// Resume kickoff: starting epoch for the launcher's initial
    /// `dispatch_epoch(start_epoch)` call. Default `0` (fresh run).
    /// Set to `meta.epoch` from a loaded
    /// [`crate::distributed::CheckpointMeta`] sidecar to resume at the
    /// epoch a prior run was saved at. Carried separately from
    /// `start_elche_state` because the launcher reads it before
    /// `ClusterCoordinator::start` consumes the config.
    pub start_epoch: usize,

    /// Resume kickoff: initial `global_step` value. Default `0`. The
    /// scheduler's batch-position offset; preserved across resume so
    /// LR-warmup curves and decay schedules pick up where they left off.
    pub start_global_step: usize,

    /// Resume kickoff: initial `avg_count` value. Default `0`. The sync
    /// round counter; preserved across resume for telemetry continuity.
    /// Not load-bearing for trajectory math.
    pub start_avg_count: u64,

    /// Resume kickoff: optional [`crate::distributed::ElCheState`]
    /// snapshot from a prior run's `.meta.json`. When `Some`, the coord
    /// constructor calls `ElChe::restore_from_state(...)` on the
    /// user-built `el_che` after applying user knobs, layering saved
    /// trajectory state (anchor, anchor_rank, phase, calibration_count,
    /// trust-window seed) on top of the user's config. `None` =
    /// fresh-start (no ElChe state to restore).
    pub start_elche_state: Option<crate::distributed::ElCheState>,

    /// Shuffle base seed recorded in the coverage block at checkpoint time
    /// (the epoch `e` permutation is `Rng::seed(seed + e)`). Defaults to
    /// [`crate::distributed::ddp_run::SHUFFLE_BASE_SEED`]; resume uses it to
    /// verify the resumed run re-shuffles over the same index space.
    pub seed: u64,

    /// One-shot checkpoint trigger: when `Some(e)`, the coordinator takes a
    /// coverage-granular checkpoint at the first reduce after the cohort
    /// crosses epoch `e` (the consensus is coherent there). `None` disables.
    /// This is the explicit-checkpoint path the resume contract is validated
    /// against; the recurring cadence (`checkpoint_every`) is layered on top
    /// of the same mechanism later.
    pub checkpoint_at_epoch: Option<usize>,

    /// Resume kickoff: optional [`crate::distributed::CoverageBlock`] from a
    /// prior run's `.meta.json`. When `Some`, the coordinator reconstructs the
    /// recorded in-progress epoch pools (via
    /// `ChunkPool::from_coverage`) and
    /// dispatches only the uncovered remainder instead of a fresh epoch.
    /// `None` = fresh-start (epoch-granular dispatch from `start_epoch`).
    pub start_coverage: Option<crate::distributed::CoverageBlock>,

    /// Static model schema (param/buffer names) captured at launch. Carried
    /// here as the conduit from the typed `DdpHandle::launch` (which has the
    /// model factory) to `run_launcher_with_config` (which builds the
    /// controller-side consensus-checkpoint writer). Lets the CPU forge write
    /// a named, loadable `.fdl` from the name-less averaged frame without
    /// routing the model through a rank. `None` for the coordinator itself
    /// (it never writes the model) and for entry paths without a factory.
    pub model_schema: Option<crate::distributed::ModelSchema>,

    /// Shared CPU-forge handle, set by the launcher so the coordinator can arm
    /// a consensus model save the controller reduce thread fulfills. Not serde
    /// (an `Arc`); `None` on NCCL / non-launcher paths.
    pub checkpoint_forge: Option<std::sync::Arc<crate::distributed::CheckpointForge>>,

    /// Optional [`crate::monitor::Timeline`] shared with the user-side
    /// harness. When set, `trigger_averaging` and `finish_averaging_*`
    /// emit `SyncStart` / `SyncEnd` events so the launcher's
    /// `summary.sync_count` reflects real averaging activity.
    /// Without it the cluster path leaves `sync_count` at 0 even when
    /// NCCL / CPU allreduces are firing — cosmetic in `done:`, but
    /// also breaks any analyzer that derives "did this run sync?" from
    /// the timeline summary.
    pub timeline: Option<std::sync::Arc<crate::monitor::Timeline>>,

    /// Optional controller-side dashboard sink. Populated by the
    /// launcher when it hosts a live dashboard. The coord forwards every
    /// rank-emitted `DashboardRegister`
    /// / `DashboardSetSvg` / `DashboardSetMetadata` / `DashboardSetHardware`
    /// frame and the per-epoch resource sample piggy-backed on
    /// `resources` to this
    /// sink. `None` ⇒ no dashboard (headless cluster runs).
    pub dashboard_sink: Option<std::sync::Arc<dyn crate::distributed::DashboardSink>>,
}

impl ClusterCoordinatorConfig {
    /// Construct with sensible defaults: LevelGuard with default
    /// threshold, no anchor relax-up, overshoot_initial=3, ceiling=15.
    pub fn new(
        policy: ApplyPolicy,
        backend: AverageBackend,
        world_size: usize,
        el_che: ElChe,
    ) -> Self {
        ClusterCoordinatorConfig {
            policy,
            backend,
            world_size,
            el_che,
            convergence_guard: Box::new(LevelGuard::default()),
            elche_relax_up: false,
            overshoot_initial: 3,
            overshoot_ceiling: 15,
            overshoot_auto: true,
            total_samples: 0,
            epoch_splits: 1,
            batch_size: 1,
            num_epochs: 0,
            partition_ratios: None,
            meta_controller: true,
            dead_ranks: None,
            reported_deaths: None,
            // 30s LAN default, scaled by FLODL_NET_TIMEOUT_SCALE so the
            // coord-side staleness scan stretches with the rest of the
            // deadline set (rank-side coord-liveness mirrors this).
            abort: None,
            heartbeat_timeout_secs: crate::distributed::wire::scaled_deadline_secs(30),
            rendezvous_timeout_secs: NCCL_RENDEZVOUS_TIMEOUT_SECS,
            formation_timeout_secs: crate::distributed::wire::scaled_deadline_secs(60),
            local_ranks: Vec::new(),
            rank_hosts: Vec::new(),
            max_failure: None,
            save_path: None,
            checkpoint_every: None,
            metrics_fn: None,
            metrics_sink_tx: None,
            eval_result_fn: None,
            eval_every_epochs: None,
            reports_per_epoch: None,
            record_log_dir: None,
            dashboard_html: None,
            dashboard_theme: None,
            scalar_reductions: crate::monitor::record::Reductions::new(),
            max_log_size: None,
            epoch_callback_policy: crate::distributed::ddp_run::EpochCallbackPolicy::default(),
            progressive: None,
            start_epoch: 0,
            start_global_step: 0,
            start_avg_count: 0,
            start_elche_state: None,
            seed: crate::distributed::ddp_run::SHUFFLE_BASE_SEED,
            checkpoint_at_epoch: None,
            start_coverage: None,
            model_schema: None,
            checkpoint_forge: None,
            timeline: None,
            dashboard_sink: None,
        }
    }

    /// Attach a [`crate::distributed::DashboardSink`] for forwarding
    /// rank-emitted dashboard frames to the controller-side dashboard
    /// server. Set by the launcher when it hosts a live dashboard;
    /// leave `None` for headless runs.
    pub fn dashboard_sink(
        mut self,
        sink: std::sync::Arc<dyn crate::distributed::DashboardSink>,
    ) -> Self {
        self.dashboard_sink = Some(sink);
        self
    }

    /// Attach a [`crate::monitor::Timeline`] for `SyncStart` /
    /// `SyncEnd` event emission. The launcher / user harness
    /// constructs the timeline; threading it here makes the cluster
    /// coord's averaging activity visible in `summary.sync_count`.
    pub fn timeline(mut self, tl: std::sync::Arc<crate::monitor::Timeline>) -> Self {
        self.timeline = Some(tl);
        self
    }

    /// Resume builder: stamp the loaded
    /// [`crate::distributed::CheckpointMeta`] trajectory (epoch +
    /// global_step + sync_round + optional ElChe state) onto this
    /// config. The launcher reads `start_epoch` to drive its kickoff
    /// `dispatch_epoch(start_epoch)`; the coord constructor consumes
    /// the rest. `trend_history` inside the ElCheState is consumed by
    /// the convergence-guard build path in the orchestrator (the coord
    /// itself doesn't carry guard state — the boxed guard is already
    /// rebuilt with restored history before reaching this config).
    pub fn resume_from_meta(mut self, meta: &crate::distributed::CheckpointMeta) -> Self {
        self.start_epoch = meta.epoch;
        self.start_global_step = meta.global_step;
        self.start_avg_count = meta.sync_round;
        self.start_elche_state = meta.elche_state.clone();
        self.start_coverage = meta.coverage.clone();
        self
    }

    /// Set the shuffle base seed recorded in checkpoint coverage blocks.
    /// Defaults to [`crate::distributed::ddp_run::SHUFFLE_BASE_SEED`].
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// One-shot checkpoint trigger at the first reduce after epoch `e`.
    /// `None`-equivalent disable is just not calling this.
    pub fn checkpoint_at_epoch(mut self, epoch: usize) -> Self {
        self.checkpoint_at_epoch = Some(epoch);
        self
    }

    /// Attach the static model schema captured at launch (param/buffer names),
    /// used by the controller-side consensus-checkpoint writer to name the
    /// averaged frame's tensors.
    pub fn model_schema(mut self, schema: crate::distributed::ModelSchema) -> Self {
        self.model_schema = Some(schema);
        self
    }

    /// Attach the user-supplied eval-result callback (controller-side).
    pub fn eval_result_fn(mut self, f: crate::distributed::ddp_run::EvalResultFn) -> Self {
        self.eval_result_fn = Some(f);
        self
    }

    /// Eval cadence in epochs. `0` disables.
    pub fn eval_every_epochs(mut self, n: usize) -> Self {
        self.eval_every_epochs = if n == 0 { None } else { Some(n) };
        self
    }

    /// Sub-epoch monitor reports per epoch. `0` disables.
    /// See [`Self::reports_per_epoch`].
    pub fn reports_per_epoch(mut self, n: usize) -> Self {
        self.reports_per_epoch = if n == 0 { None } else { Some(n) };
        self
    }

    /// Persist the monitor record stream as JSONL under `dir`, each node
    /// capped at `max_bytes` (`0` = library default).
    pub fn record_log(mut self, dir: impl Into<String>, max_bytes: u64) -> Self {
        self.record_log_dir = Some(dir.into());
        self.max_log_size = if max_bytes == 0 {
            None
        } else {
            Some(max_bytes)
        };
        self
    }

    /// Carry the dashboard-archive path through to the launcher.
    pub fn dashboard_html(mut self, path: impl Into<String>) -> Self {
        self.dashboard_html = Some(path.into());
        self
    }

    /// Carry the archive theme through to the launcher.
    pub fn dashboard_theme(mut self, theme: impl Into<String>) -> Self {
        self.dashboard_theme = Some(theme.into());
        self
    }

    /// Carry the user-scalar roll-up declarations through to the launcher.
    pub fn scalar_reductions(mut self, reductions: crate::monitor::record::Reductions) -> Self {
        self.scalar_reductions = reductions;
        self
    }

    /// Set the [`EpochCallbackPolicy`] that decides which rank fires
    /// per-epoch user callbacks (`epoch_fn`, `checkpoint_fn`,
    /// `eval_fn`). Default [`EpochCallbackPolicy::Fastest`].
    ///
    /// [`EpochCallbackPolicy`]:
    ///     crate::distributed::ddp_run::EpochCallbackPolicy
    /// [`EpochCallbackPolicy::Fastest`]:
    ///     crate::distributed::ddp_run::EpochCallbackPolicy::Fastest
    pub fn epoch_callback_policy(
        mut self,
        policy: crate::distributed::ddp_run::EpochCallbackPolicy,
    ) -> Self {
        self.epoch_callback_policy = policy;
        self
    }

    /// Override progressive-dispatch mode. `None` (default) follows
    /// the auto rule: true for `Cadence` / `Async`, false for `Sync`.
    /// `Some(true)` / `Some(false)` is an explicit override. See
    /// [`Self::progressive`].
    pub fn progressive(mut self, enabled: bool) -> Self {
        self.progressive = Some(enabled);
        self
    }

    /// Attach the user-supplied per-epoch metrics callback.
    pub fn metrics_fn(mut self, f: crate::distributed::ddp_run::MetricsFn) -> Self {
        self.metrics_fn = Some(f);
        self
    }

    /// Attach an `EpochMetrics` sink. The coord clones each aggregated
    /// metric into both this sink and the optional `metrics_fn`
    /// callback. Used by the launcher trampoline to wire the user's
    /// `DdpHandle::next_metrics()` polling loop.
    pub fn metrics_sink_tx(
        mut self,
        tx: mpsc::Sender<crate::distributed::ddp_run::EpochMetrics>,
    ) -> Self {
        self.metrics_sink_tx = Some(tx);
        self
    }

    /// Cadence (in epochs) for user-supplied `checkpoint_fn` invocation.
    /// `None` or `0` disables. See [`Self::checkpoint_every`].
    pub fn checkpoint_every(mut self, every: usize) -> Self {
        self.checkpoint_every = if every == 0 { None } else { Some(every) };
        self
    }

    pub fn total_samples(mut self, n: usize) -> Self {
        self.total_samples = n;
        self
    }

    pub fn epoch_splits(mut self, n: usize) -> Self {
        self.epoch_splits = n.max(1);
        self
    }

    pub fn batch_size(mut self, n: usize) -> Self {
        self.batch_size = n.max(1);
        self
    }

    pub fn num_epochs(mut self, n: usize) -> Self {
        self.num_epochs = n;
        self
    }

    pub fn partition_ratios(mut self, ratios: Option<Vec<f64>>) -> Self {
        self.partition_ratios = ratios;
        self
    }

    pub fn with_convergence_guard(mut self, guard: Box<dyn ConvergenceGuard>) -> Self {
        self.convergence_guard = guard;
        self
    }

    pub fn no_divergence_guard(mut self) -> Self {
        self.convergence_guard = Box::new(NoGuard);
        self
    }

    pub fn elche_relax_up(mut self, enabled: bool) -> Self {
        self.elche_relax_up = enabled;
        self
    }

    pub fn overshoot(mut self, initial: usize, ceiling: usize, auto: bool) -> Self {
        self.overshoot_initial = initial;
        self.overshoot_ceiling = ceiling;
        self.overshoot_auto = auto;
        self
    }

    /// Enable the LR-aware meta-controller above ElChe. Default: true.
    ///
    /// When enabled, a [`crate::distributed::lr_event_meta::LrEventMeta`]
    /// is constructed by [`super::ClusterCoordinator::start_from_listener`]
    /// and observed after every averaging-cycle guard verdict. See
    /// [`crate::distributed::lr_event_meta`] for the design.
    pub fn meta_controller(mut self, enabled: bool) -> Self {
        self.meta_controller = enabled;
        self
    }

    /// Share a dead-rank ledger with the cluster controller. Required
    /// for elastic-membership (rank-death-survives-the-run) semantics.
    /// Pass the same `Arc<DeadRanks>` to
    /// `start_with_dead_ranks`.
    pub fn dead_ranks(mut self, ledger: Arc<crate::distributed::controller::DeadRanks>) -> Self {
        self.dead_ranks = Some(ledger);
        self
    }

    /// Attach the externally-reported death queue (see
    /// [`ClusterCoordinatorConfig::reported_deaths`]).
    pub fn reported_deaths(
        mut self,
        queue: crate::distributed::cluster_coordinator::ReportedDeaths,
    ) -> Self {
        self.reported_deaths = Some(queue);
        self
    }

    /// Override the heartbeat staleness threshold. Default 30s.
    /// Attach the launcher's abort flag (see the field doc on
    /// [`Self::abort`]).
    pub fn abort_flag(mut self, flag: Arc<std::sync::atomic::AtomicBool>) -> Self {
        self.abort = Some(flag);
        self
    }

    pub fn heartbeat_timeout_secs(mut self, secs: u64) -> Self {
        self.heartbeat_timeout_secs = secs;
        self
    }

    /// Override the NCCL re-rendezvous wall-budget. See
    /// [`Self::rendezvous_timeout_secs`].
    pub fn rendezvous_timeout_secs(mut self, secs: u64) -> Self {
        self.rendezvous_timeout_secs = secs;
        self
    }

    /// Override the formation dial-in wall-budget (see the field doc on
    /// [`Self::formation_timeout_secs`]).
    pub fn formation_timeout_secs(mut self, secs: u64) -> Self {
        self.formation_timeout_secs = secs;
        self
    }

    /// Mark ranks running on the same host as the coord process.
    /// The NCCL re-rendezvous picker prefers these when picking a
    /// UID generator (local same-process latency + lower correlated
    /// failure risk). Used by the launcher to pin its local GPU
    /// rank(s); standalone-coord tests typically leave this empty.
    pub fn local_ranks(mut self, ranks: Vec<usize>) -> Self {
        self.local_ranks = ranks;
        self
    }

    /// Set the global-rank → host map (index = rank) used to
    /// host-qualify rank-reported resource samples deposited into the
    /// timeline. The launcher fills it from its world map.
    pub fn rank_hosts(mut self, hosts: Vec<String>) -> Self {
        self.rank_hosts = hosts;
        self
    }

    /// Set the unrecoverable-failure threshold. When the dead-rank
    /// count reaches this limit, the coord broadcasts a save-and-
    /// shutdown signal to survivors so they persist state before exit.
    ///
    /// Backend hard limits apply regardless (NCCL needs `world_size >= 2`
    /// for a comm; CPU needs at least 1 survivor).
    pub fn max_failure(
        mut self,
        threshold: crate::distributed::max_failure::MaxFailureThreshold,
    ) -> Self {
        self.max_failure = Some(threshold);
        self
    }

    /// Set the checkpoint bundle stem for unrecoverable-failure
    /// persistence. Workers consult [`crate::distributed::CheckpointBundle`]
    /// to derive `<save_path>.fdl`, `<save_path>.optim`,
    /// `<save_path>.meta.json` on `ShutdownWithSave`.
    pub fn save_path(mut self, path: impl Into<String>) -> Self {
        self.save_path = Some(path.into());
        self
    }
}
