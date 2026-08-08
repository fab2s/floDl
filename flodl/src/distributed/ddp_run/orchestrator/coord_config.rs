//! Builder → [`ClusterCoordinatorConfig`] translation for cluster mode.
//!
//! The launcher trampoline (`DdpHandle::launch` on `Role::Launcher`)
//! runs the user's `main()` up to `Trainer::builder(...).run()`. Inside
//! `.run()`, the live builder state holds the controller-scope fields
//! (policy / backend / convergence guard / resume_from / etc.) that the
//! cluster coord needs at construction time. This helper threads those
//! fields into a [`ClusterCoordinatorConfig`] that the launcher hands to
//! [`crate::distributed::launcher::run_launcher_with_config`].
//!
//! [`ClusterCoordinatorConfig`]:
//!     crate::distributed::cluster_coordinator::ClusterCoordinatorConfig

use crate::distributed::ddp_run::{
    ApplyPolicy, AverageBackend, DdpRunConfig, EvalResultFn, MetricsFn, convergence,
};
use crate::tensor::Result;

/// Build a [`ClusterCoordinatorConfig`] from the user's
/// builder-side controller-scope fields.
///
/// Mirrors the guard-construction precedence used by the legacy
/// `run_cluster_rank_cadence_nccl` worker-side path: user-supplied
/// [`ConvergenceGuard`] wins, otherwise [`NoGuard`] when
/// `no_divergence_guard` is set, otherwise a [`TrendGuard`] at the
/// user threshold or the EASGD-aware default
/// ([`default_trend_threshold`]).
///
/// [`ClusterCoordinatorConfig`]:
///     crate::distributed::cluster_coordinator::ClusterCoordinatorConfig
/// [`ConvergenceGuard`]: convergence::ConvergenceGuard
/// [`NoGuard`]: convergence::NoGuard
/// [`TrendGuard`]: convergence::TrendGuard
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_coord_config_from_builder(
    policy: ApplyPolicy,
    backend: AverageBackend,
    config: &DdpRunConfig,
    convergence_guard: Option<Box<dyn convergence::ConvergenceGuard>>,
    metrics_fn: Option<MetricsFn>,
    eval_result_fn: Option<EvalResultFn>,
    world_size: usize,
    total_samples: usize,
    batch_size: usize,
    num_epochs: usize,
) -> Result<crate::distributed::cluster_coordinator::ClusterCoordinatorConfig> {
    use crate::distributed::cluster_coordinator::ClusterCoordinatorConfig;
    use crate::distributed::ddp::ElChe;

    // Geometry (including a starved-epoch refusal) is validated once per
    // run at the tier entry points, so it is not re-checked here.
    let epoch_splits = config.epoch_splits.max(1);

    // Resume: read the meta sidecar before anything else so the saved
    // ElChe / TrendGuard / trajectory state can feed the constructors
    // below. Missing file or schema mismatch surfaces loudly here
    // rather than partially seeding the controller.
    let resume_meta: Option<crate::distributed::CheckpointMeta> = match config.resume_from {
        Some(ref stem) => {
            let path = crate::distributed::CheckpointBundle::meta_path(stem);
            Some(crate::distributed::CheckpointMeta::read_from_file(&path)?)
        }
        None => None,
    };

    // ElChe construction: anchor (default 10 matches DdpRunConfig docs)
    // plus optional max/min/overhead_target/max_batch_diff knobs.
    let anchor = config.elche.anchor;
    let mut el_che = ElChe::new(world_size, anchor);
    if let Some(target) = config.elche.overhead_target {
        el_che = el_che.with_overhead_target(target);
    }
    if let Some(max) = config.elche.max_anchor {
        el_che = el_che.with_max_anchor(max);
    }
    if let Some(min) = config.elche.min_anchor {
        el_che = el_che.with_min_anchor(min);
    }
    if let Some(diff) = config.elche.max_batch_diff {
        el_che = el_che.with_max_batch_diff(diff);
    }
    // Window-pressure anchor growth is a windowed-reduce optimization; Sync
    // reduces every step (no window to amortize), so disable growth there.
    // Leaving it on under Sync would inflate the telemetry anchor and the
    // checkpointed ElCheState.anchor, mis-seeding a later Cadence resume.
    el_che = el_che.with_window_growth_applicable(policy != ApplyPolicy::Sync);

    let mut coord_config = ClusterCoordinatorConfig::new(policy, backend, world_size, el_che)
        .total_samples(total_samples)
        // Must be the same value every rank's WorkerConfig carries, or the
        // ledger and the ranks' expansions land on different slices.
        .epoch_splits(epoch_splits)
        .batch_size(batch_size)
        .num_epochs(num_epochs)
        .elche_relax_up(config.elche.relax_up)
        .meta_controller(config.elche.meta_controller)
        .partition_ratios(config.elche.partition_ratios.clone());

    // Guard precedence: user override > NoGuard (if flagged) >
    // TrendGuard with user threshold or 0.05 default. On resume, the
    // default-built TrendGuard absorbs the saved divergence ring
    // buffer so the first 3 cycles after resume don't silently emit
    // `Stable` regardless of live trajectory. User-supplied guards are
    // passed through unchanged — the caller owns their guard's resume
    // story.
    let resume_trend_history: Option<Vec<f64>> = resume_meta
        .as_ref()
        .and_then(|m| m.elche_state.as_ref())
        .and_then(|s| s.trend_history.clone());
    let guard: Box<dyn convergence::ConvergenceGuard> = match convergence_guard {
        Some(g) => g,
        None => {
            if config.elche.no_divergence_guard {
                Box::new(convergence::NoGuard)
            } else {
                let mut tg = convergence::TrendGuard::new(
                    config
                        .elche
                        .divergence_threshold
                        .unwrap_or_else(|| default_trend_threshold(config.elche.easgd_alpha)),
                );
                if let Some(history) = resume_trend_history {
                    tg = tg.with_history(history);
                }
                Box::new(tg)
            }
        }
    };
    coord_config = coord_config.with_convergence_guard(guard);

    // max_overshoot: the CpuAsync streaming lookahead bound. A user-set
    // value pins the bound and disables auto-tune (matches the
    // `overshoot_auto` contract: "true when the user did not set
    // max_overshoot explicitly"). Was previously written into ElCheConfig
    // but never read here, so the coordinator always ran the auto default
    // (3→15) and the knob was silently inert. It only has effect on the
    // Async path, so warn loudly if set on a non-Async policy rather than
    // ignore it.
    if let Some(n) = config.elche.max_overshoot {
        if policy != ApplyPolicy::Async {
            eprintln!(
                "fdl: max_overshoot={n} is ignored outside CpuAsync \
                 (mode resolves to policy {policy:?}); the async streaming \
                 lookahead bound has no effect here"
            );
        }
        coord_config = coord_config.overshoot(n, n, false);
    }
    if let Some(threshold) = config.max_failure {
        coord_config = coord_config.max_failure(threshold);
    }
    if let Some(ref stem) = config.save_path {
        coord_config = coord_config.save_path(stem.clone());
    }
    if let Some(secs) = config.heartbeat_timeout_secs {
        coord_config = coord_config.heartbeat_timeout_secs(secs);
    }
    if let Some(every) = config.checkpoint_every {
        coord_config = coord_config.checkpoint_every(every);
    }
    if let Some(epoch) = config.checkpoint_at_epoch {
        coord_config = coord_config.checkpoint_at_epoch(epoch);
    }
    if let Some(f) = metrics_fn {
        coord_config = coord_config.metrics_fn(f);
    }
    if let Some(every) = config.eval_every_epochs {
        coord_config = coord_config.eval_every_epochs(every);
    }
    if let Some(n) = config.reports_per_epoch {
        coord_config = coord_config.reports_per_epoch(n);
    }
    if let Some(dir) = config.record_log_dir.clone() {
        coord_config = coord_config.record_log(dir, config.max_log_size.unwrap_or(0));
    }
    if let Some(path) = config.dashboard_html.clone() {
        coord_config = coord_config.dashboard_html(path);
    }
    if let Some(theme) = config.dashboard_theme.clone() {
        coord_config = coord_config.dashboard_theme(theme);
    }
    if !config.scalar_reductions.is_empty() {
        coord_config = coord_config.scalar_reductions(config.scalar_reductions.clone());
    }
    if let Some(f) = eval_result_fn {
        coord_config = coord_config.eval_result_fn(f);
    }
    if let Some(enabled) = config.progressive_dispatch {
        coord_config = coord_config.progressive(enabled);
    }
    // Thread the user's epoch_callback_policy through to the coord so
    // the controller can resolve Fastest at runtime + push
    // SetEpochCallbackRole to workers.
    coord_config = coord_config.epoch_callback_policy(config.epoch_callback_policy);

    // Share the harness-side `Timeline` (if any) so the coord can emit
    // `SyncStart` / `SyncEnd` events around its averaging cycles —
    // without this, `summary.sync_count` reads 0 even when NCCL / CPU
    // allreduces are firing on every cadence.
    if let Some(ref tl) = config.timeline {
        coord_config = coord_config.timeline(std::sync::Arc::clone(tl));
    }

    // Resume: source the shuffle seed from the saved coverage block so the
    // coverage guard + the workers (which read the same meta via
    // `resolve_shuffle_seed`) reproduce the recorded epoch permutation by
    // reading the value, not by assuming the build's `SHUFFLE_BASE_SEED` still
    // matches it. No coverage block (clean-boundary save) leaves the default.
    if let Some(seed) = resume_meta
        .as_ref()
        .and_then(|m| m.coverage.as_ref())
        .map(|c| c.seed)
    {
        coord_config = coord_config.seed(seed);
    }

    // Resume trajectory: applies after every other field so the loaded
    // meta cleanly overrides the fresh defaults
    // (start_epoch/start_global_step/start_avg_count/start_elche_state).
    // The `trend_history` inside elche_state has already been consumed
    // above for the guard; we still hand the whole state to the coord
    // so `ElChe::restore_from_state` can seed the ms_per_batch trust
    // window.
    if let Some(meta) = resume_meta {
        coord_config = coord_config.resume_from_meta(&meta);
    }

    Ok(coord_config)
}

/// Default [`TrendGuard`](convergence::TrendGuard) threshold when the user
/// set none, keyed on the param-adoption semantics.
///
/// Full-overwrite modes (Sync / Cadence / un-blended Async) snap every rank
/// onto the consensus at each reduce, so the measured weight-space
/// divergence is pure per-window drift and a low floor (0.05) discriminates
/// well. EASGD blending (`easgd_alpha` set) deliberately keeps replicas on
/// an elastic spread around the consensus — the measured divergence carries
/// a standing baseline (~0.1 at α=0.5) that IS the operating point, not
/// drift toward failure. An overwrite-calibrated floor sits inside that
/// band and keeps the trend rule permanently armed on a healthy run
/// (measured 2026-07-22, resnet-graph 3-seed probe: suppression at the low
/// floor bought no convergence and cost ~35% extra reduces). Blended modes
/// therefore calibrate the floor above the elastic band; a genuine
/// divergence spiral still crosses it with sustained rises and is caught.
pub(crate) fn default_trend_threshold(easgd_alpha: Option<f64>) -> f64 {
    if easgd_alpha.is_some() { 0.3 } else { 0.05 }
}

#[cfg(test)]
mod tests {
    use super::default_trend_threshold;

    #[test]
    fn trend_threshold_default_is_easgd_aware() {
        // Overwrite semantics keep the historical floor.
        assert_eq!(default_trend_threshold(None), 0.05);
        // Blended semantics calibrate above the elastic standing spread.
        assert_eq!(default_trend_threshold(Some(0.5)), 0.3);
    }
}
