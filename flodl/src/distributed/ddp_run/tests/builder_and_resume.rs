//! DdpHandle / DdpBuilder builder tests, `Trainer::builder(...).resume_from`
//! + missing meta surfacing, epoch_fn integration tests.

use super::*;

// DdpHandle / DdpBuilder tests
// -----------------------------------------------------------------------

#[test]
fn test_builder_single_gpu_fallback() {
    // The builder entry's multi-GPU path auto-promotes to process-per-rank
    // in production and is gated off under cfg(test); the only cfg(test)
    // reachable behavior is the single-device fallback (<2 visible devices).
    // Skip on a multi-GPU box so this stays a deterministic fallback test.
    if crate::tensor::usable_gpu_devices().len() >= 2 {
        return;
    }
    let ddp = crate::distributed::Trainer::builder(
        |dev| Linear::on_device(4, 2, dev),
        |params| crate::nn::SGD::new(params, 0.01, 0.0),
        mse_train,
    )
    .dataset(Arc::new(TestDataset { n: 100 }))
    .batch_size(4)
    .num_epochs(2)
    .policy(ApplyPolicy::Sync)
    .backend(AverageBackend::Cpu) // CPU backend: no NCCL needed for this test
    .run()
    .unwrap();

    assert!(ddp.world_size() >= 1);
    let state = ddp.join().unwrap();
    // Linear(4,2): weight [2,4] + bias [2] = 2 params, 0 buffers
    assert_eq!(state.params.len(), 2);
    assert_eq!(state.buffers.len(), 0);
}

// `test_async_ddp_multi_gpu_nccl` drove the in-process multi-GPU engine via
// the builder entry; that engine was removed (production auto-promotes to
// process-per-rank). End-to-end multi-GPU training is validated by the
// `ddp-bench` binary under `fdl gpu-test-nccl`, not a cfg(test) unit test.

#[test]
fn test_ddp_handle_send_sync() {
    fn assert_send<T: Send>() {}
    assert_send::<DdpHandle>();
    assert_send::<TrainedState>();
}

// -----------------------------------------------------------------------
// DdpBuilder builder tests
// -----------------------------------------------------------------------

#[test]
fn test_builder_with_defaults() {
    let ddp = crate::distributed::Trainer::builder(
        |dev| Linear::on_device(4, 2, dev),
        |params| crate::nn::SGD::new(params, 0.01, 0.0),
        mse_train,
    )
    .dataset(Arc::new(TestDataset { n: 100 }))
    .batch_size(4)
    .num_epochs(2)
    .backend(AverageBackend::Cpu)
    .run()
    .unwrap();

    assert!(ddp.world_size() >= 1);
    let state = ddp.join().unwrap();
    assert_eq!(state.params.len(), 2);
}

#[test]
fn test_builder_with_all_options() {
    // Workload is intentionally tiny (8 samples / batch 4 / 1 epoch =
    // 2 steps): this test verifies that every builder setter is wired,
    // not that policy survives load. ElChe is sized for production
    // pools and can stall on heterogeneous hardware when the per-rank
    // share is small enough that the fast rank laps the slow one;
    // Sync alone doesn't fully isolate the test from that pathology
    // here, so we also keep the dataset small enough that any
    // remaining lapping cannot accumulate before completion.
    let ddp = crate::distributed::Trainer::builder(
        |dev| Linear::on_device(4, 2, dev),
        |params| crate::nn::SGD::new(params, 0.01, 0.0),
        mse_train,
    )
    .dataset(Arc::new(TestDataset { n: 8 }))
    .batch_size(4)
    .num_epochs(1)
    .policy(ApplyPolicy::Sync)
    .backend(AverageBackend::Cpu)
    .overhead_target(0.15)
    .max_anchor(100)
    .anchor(5)
    .divergence_threshold(0.1)
    .max_batch_diff(10)
    .run()
    .unwrap();

    let state = ddp.join().unwrap();
    assert_eq!(state.params.len(), 2);
}

#[test]
#[should_panic(expected = "dataset is required")]
fn test_builder_missing_dataset_panics() {
    let _ = crate::distributed::Trainer::builder(
        |dev| Linear::on_device(4, 2, dev),
        |params| crate::nn::SGD::new(params, 0.01, 0.0),
        mse_train,
    )
    .batch_size(4)
    .num_epochs(2)
    .run();
}

#[test]
#[should_panic(expected = "batch_size is required")]
fn test_builder_missing_batch_size_panics() {
    let _ = crate::distributed::Trainer::builder(
        |dev| Linear::on_device(4, 2, dev),
        |params| crate::nn::SGD::new(params, 0.01, 0.0),
        mse_train,
    )
    .dataset(Arc::new(TestDataset { n: 100 }))
    .num_epochs(2)
    .run();
}

#[test]
#[should_panic(expected = "num_epochs is required")]
fn test_builder_missing_num_epochs_panics() {
    let _ = crate::distributed::Trainer::builder(
        |dev| Linear::on_device(4, 2, dev),
        |params| crate::nn::SGD::new(params, 0.01, 0.0),
        mse_train,
    )
    .dataset(Arc::new(TestDataset { n: 100 }))
    .batch_size(4)
    .run();
}

// -----------------------------------------------------------------------
// Trainer::resume_from end-to-end: write a meta sidecar to disk, build
// the orchestrator's coord config via the resume_from path, and
// confirm the trajectory + ElChe state + LevelGuard history all
// transit cleanly.
// -----------------------------------------------------------------------

#[test]
fn resume_from_loads_meta_and_seeds_coord_config() {
    use super::orchestrator::build_coord_config_from_builder;
    use crate::distributed::el_che::Phase;
    use crate::distributed::{CheckpointBundle, CheckpointMeta, ElCheState, SaveReason};

    let dir = std::env::temp_dir().join(format!("flodl_resume_e2e_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let stem = dir.join("ckpt").to_string_lossy().into_owned();

    // Write a meta sidecar that the orchestrator must reload.
    let elche_state = ElCheState {
        anchor: 14,
        anchor_rank: Some(1),
        smoothed_ms_per_batch: vec![3.5, 5.5],
        phase: Phase::Stable,
        calibration_count: 17,
        trend_history: Some(vec![0.005, 0.01, 0.02, 0.025]),
    };
    let meta = CheckpointMeta::new(4, 9_876, 33, 2, SaveReason::GracefulShutdown)
        .with_elche_state(elche_state.clone());
    let meta_path = CheckpointBundle::meta_path(&stem);
    meta.write_to_file(&meta_path).unwrap();

    let user_config = DdpRunConfig::new().with_resume_from(stem.clone());
    let coord_config = build_coord_config_from_builder(
        ApplyPolicy::Cadence,
        AverageBackend::Cpu,
        &user_config,
        None,
        None,
        None,
        2,
        // Trivial test values for the new total_samples / batch_size /
        // num_epochs args; the test asserts resume-meta plumbing, not
        // dataset arithmetic.
        100,
        4,
        1,
    )
    .expect("resume meta loads cleanly");

    // Trajectory plumbed through.
    assert_eq!(coord_config.start_epoch, 4);
    assert_eq!(coord_config.start_global_step, 9_876);
    assert_eq!(coord_config.start_avg_count, 33);
    assert_eq!(
        coord_config.start_elche_state.as_ref(),
        Some(&elche_state),
        "ElCheState carries through resume_from"
    );

    // Guard rebuilt with restored trend history (default LevelGuard
    // path: user did NOT supply an explicit guard).
    let history = coord_config
        .convergence_guard
        .trend_history()
        .expect("LevelGuard surfaces a non-empty history after resume");
    assert_eq!(history, vec![0.005, 0.01, 0.02, 0.025]);

    std::fs::remove_dir_all(&dir).ok();
}

// Missing meta file must surface loudly — silent fallback to fresh
// state would mask a misconfigured resume.
#[test]
fn resume_from_missing_meta_errors() {
    use super::orchestrator::build_coord_config_from_builder;

    let user_config =
        DdpRunConfig::new().with_resume_from("/nonexistent/path/that/cannot/exist/ckpt");
    let result = build_coord_config_from_builder(
        ApplyPolicy::Cadence,
        AverageBackend::Cpu,
        &user_config,
        None,
        None,
        None,
        2,
        100,
        4,
        1,
    );
    let err = match result {
        Ok(_) => panic!("missing meta file must error, got Ok"),
        Err(e) => e,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("read") || msg.contains("CheckpointMeta"),
        "expected read-error message, got: {msg}"
    );
}

// -----------------------------------------------------------------------
// epoch_fn tests
// -----------------------------------------------------------------------

#[test]
fn test_worker_current_epoch_accessor() {
    let (mut worker, _ch) = make_test_worker();
    assert_eq!(worker.current_epoch(), 0);
    worker.current_epoch = 1;
    assert_eq!(worker.current_epoch(), 1);
}

#[test]
fn test_worker_set_lr() {
    let (mut worker, _ch) = make_test_worker();
    // set_lr should not panic; we verify it works by running a train step after
    worker.set_lr(0.1);
    let opts = test_opts();
    let batch = vec![
        Tensor::randn(&[4, 4], opts).unwrap(),
        Tensor::randn(&[4, 2], opts).unwrap(),
    ];
    let (loss, _) = worker.train_step(&batch, &mse_train).unwrap();
    assert!(loss > 0.0);
}

#[test]
fn test_epoch_fn_called_per_epoch() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let counter = Arc::new(AtomicUsize::new(0));
    let epochs_seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let counter_c = counter.clone();
    let epochs_c = epochs_seen.clone();

    let num_epochs = 3;
    // ApplyPolicy::Sync is required for this test's assertion shape.
    // The contract flodl's DDP actually guarantees is batch-based, not
    // epoch-based: `max_batch_diff` / `max_overshoot` bounds how far
    // ranks can diverge in batches per sync cycle. Epoch boundaries are
    // bookkeeping on top of that. At production scale this is invisible
    // — `max_overshoot` (default ceiling 15) is a tiny fraction of the
    // pool, so every rank receives every epoch's plan in practice.
    //
    // This test uses a tiny dataset (100 samples / batch 4 → 25
    // batches, planned share ~12 per rank) where the overshoot cap
    // exceeds the per-rank share. Under progressive mode (Cadence /
    // Async) a fast rank can legally drain
    // the whole pool, leaving the slow rank with no `StartEpoch` plan
    // for that epoch — which means fewer than `num_epochs * world`
    // epoch_fn firings. Expected behaviour at this scale, not a bug.
    //
    // Sync is the only policy where `count == num_epochs * world`
    // holds at any dataset size; that's what this test is anchoring.
    // For an epoch_fn test under progressive semantics, see a future
    // test that asserts the weaker invariant (every epoch fires at
    // least once across the cluster, no rank fires the same epoch
    // twice).
    let ddp = crate::distributed::Trainer::builder(
        |dev| Linear::on_device(4, 2, dev),
        |params| crate::nn::SGD::new(params, 0.01, 0.0),
        mse_train,
    )
    .dataset(Arc::new(TestDataset { n: 100 }))
    .batch_size(4)
    .num_epochs(num_epochs)
    .backend(AverageBackend::Cpu)
    .policy(ApplyPolicy::Sync)
    .epoch_fn(move |epoch, worker| {
        counter_c.fetch_add(1, Ordering::Relaxed);
        epochs_c.lock().unwrap().push(epoch);
        // Verify current_epoch matches the callback argument
        assert_eq!(worker.current_epoch(), epoch);
    })
    .run()
    .unwrap();

    let world = ddp.world_size();
    let _state = ddp.join().unwrap();

    // Instrumented assertions: on regression, dump the full observed
    // state. In Sync mode (pinned above) each rank must see each epoch
    // exactly once; any drift is a real bug in the Sync dispatcher, not
    // the progressive streaming path.
    let got_counter = counter.load(Ordering::Relaxed);
    let expected_counter = num_epochs * world;

    let mut seen = epochs_seen.lock().unwrap().clone();
    seen.sort();
    let mut expected_epochs: Vec<usize> =
        (0..num_epochs).cycle().take(num_epochs * world).collect();
    expected_epochs.sort();

    assert_eq!(
        got_counter, expected_counter,
        "epoch_fn fire count mismatch — got {got_counter}, expected {expected_counter}. \
         world_size={world}, num_epochs={num_epochs}, epochs_seen={seen:?}.",
    );
    assert_eq!(
        seen, expected_epochs,
        "epoch_fn epoch-index set mismatch — got {seen:?}, expected {expected_epochs:?}. \
         world_size={world}, num_epochs={num_epochs}, counter={got_counter}.",
    );
}

#[test]
fn test_epoch_fn_set_lr() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let call_count = Arc::new(AtomicUsize::new(0));
    let call_count_c = call_count.clone();

    // Sync policy required (see `test_epoch_fn_called_per_epoch` for
    // the full rationale). In Sync every rank fires `epoch_fn` for
    // every epoch, so `lr` stays consistent across ranks during each
    // gradient average — which is what this test actually verifies.
    let ddp = crate::distributed::Trainer::builder(
        |dev| Linear::on_device(4, 2, dev),
        |params| crate::nn::SGD::new(params, 0.01, 0.0),
        mse_train,
    )
    .dataset(Arc::new(TestDataset { n: 100 }))
    .batch_size(4)
    .num_epochs(3)
    .backend(AverageBackend::Cpu)
    .policy(ApplyPolicy::Sync)
    .epoch_fn(move |epoch, worker| {
        // Simulate a LR schedule: decrease LR each epoch
        let lr = 0.01 * (1.0 - epoch as f64 * 0.3);
        worker.set_lr(lr);
        call_count_c.fetch_add(1, Ordering::Relaxed);
    })
    .run()
    .unwrap();

    let world = ddp.world_size();
    let _state = ddp.join().unwrap();
    assert_eq!(call_count.load(Ordering::Relaxed), 3 * world);
}

#[test]
fn test_worker_send_final_snapshot() {
    let (mut worker, ch) = make_test_worker();
    worker.send_final_snapshot();
    let snap = ch.final_param_rx.recv().unwrap();
    assert_eq!(snap.params.len(), 2); // Linear(4,2): weight + bias
    assert_eq!(snap.rank, 0);
}

// `collect_final_state` tests lived here; they exercised the in-process
// Coordinator and were removed with it. Final-state collection on the
// process path is covered under `cluster_coordinator/tests/`.

// H13: ElCheConfig.max_overshoot must reach the coordinator config. It
// was written into the config but never read by
// build_coord_config_from_builder, so the coordinator always ran the
// auto default (initial=3, ceiling=15, auto=true) and the knob was
// silently inert. A user-set value pins the bound and disables
// auto-tune (the `overshoot_auto` contract).
#[test]
fn max_overshoot_plumbs_into_coord_config_and_pins_it() {
    use super::orchestrator::build_coord_config_from_builder;

    let user_config = DdpRunConfig::new().with_max_overshoot(7);
    let coord_config = build_coord_config_from_builder(
        ApplyPolicy::Async,
        AverageBackend::Cpu,
        &user_config,
        None,
        None,
        None,
        2,
        100,
        4,
        1,
    )
    .expect("build");
    assert_eq!(coord_config.overshoot_initial, 7);
    assert_eq!(coord_config.overshoot_ceiling, 7);
    assert!(
        !coord_config.overshoot_auto,
        "a user-set max_overshoot pins the bound and disables auto-tune"
    );
}

// Default (unset) leaves the coordinator's auto-tune defaults intact.
#[test]
fn unset_max_overshoot_leaves_auto_tune_defaults() {
    use super::orchestrator::build_coord_config_from_builder;

    let user_config = DdpRunConfig::new();
    let coord_config = build_coord_config_from_builder(
        ApplyPolicy::Async,
        AverageBackend::Cpu,
        &user_config,
        None,
        None,
        None,
        2,
        100,
        4,
        1,
    )
    .expect("build");
    assert!(
        coord_config.overshoot_auto,
        "unset max_overshoot keeps auto-tune on"
    );
    // The ceiling is non-binding by default. Auto no longer hill-climbs
    // toward a cap: it DERIVES the per-rank budget from the allocation and
    // the measured reduce, bounded structurally at one window's allocation.
    // A small absolute default (this asserted 15) silently held the fast
    // rank to a fraction of the cover its hardware asked for, which is the
    // bug the derivation replaced. The knob survives for an operator who
    // wants a deliberate hard limit.
    assert_eq!(
        coord_config.overshoot_ceiling,
        usize::MAX,
        "unset max_overshoot leaves the derived budget uncapped"
    );
}

// Same failure class as H13 above: `epoch_splits` reaches every rank
// through WorkerConfig, so a coordinator that kept the default would not
// error — it would size its ledger over the whole pass while the ranks
// expanded a slice of it, and the cohort would train on mismatched
// permutations in silence. The knob only means anything if BOTH sides
// read the same value.
#[test]
fn epoch_splits_plumbs_into_coord_config() {
    use super::orchestrator::build_coord_config_from_builder;

    let user_config = DdpRunConfig::new().with_epoch_splits(20);
    let coord_config = build_coord_config_from_builder(
        ApplyPolicy::Async,
        AverageBackend::Cpu,
        &user_config,
        None,
        None,
        None,
        2,
        100,
        4,
        1,
    )
    .expect("build");
    assert_eq!(coord_config.epoch_splits, 20);
}

#[test]
fn unset_epoch_splits_leaves_the_epoch_a_full_pass() {
    use super::orchestrator::build_coord_config_from_builder;

    let coord_config = build_coord_config_from_builder(
        ApplyPolicy::Async,
        AverageBackend::Cpu,
        &DdpRunConfig::new(),
        None,
        None,
        None,
        2,
        100,
        4,
        1,
    )
    .expect("build");
    assert_eq!(coord_config.epoch_splits, 1);
}

// The bug this pins: `num_epochs` is the user's count of DATA PASSES, so a
// split run must execute `num_epochs * epoch_splits` epochs. Without the
// multiplication it ran `num_epochs` slices and silently trained
// `1/epoch_splits` of the data — geometry all correct, run a quarter as
// long. Every geometry unit test passed while this was broken, because
// none of them observed how many epochs actually run.
#[test]
fn epoch_splits_multiply_the_epochs_actually_run() {
    if crate::tensor::usable_gpu_devices().len() >= 2 {
        return; // auto-promote path; this pins the in-process fallback
    }
    let seen = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counter = Arc::clone(&seen);

    let ddp = crate::distributed::Trainer::builder(
        |dev| Linear::on_device(4, 2, dev),
        |params| crate::nn::SGD::new(params, 0.01, 0.0),
        mse_train,
    )
    .dataset(Arc::new(TestDataset { n: 64 }))
    .batch_size(4)
    .num_epochs(2) // two passes over the data ...
    .epoch_splits(4) // ... delivered as four epochs each
    .backend(AverageBackend::Cpu)
    .epoch_fn(move |_epoch, _worker| {
        counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    })
    .run()
    .unwrap();
    ddp.join().unwrap();

    assert_eq!(
        seen.load(std::sync::atomic::Ordering::Relaxed),
        8,
        "2 passes x 4 splits must run 8 epochs, not 2",
    );
}
