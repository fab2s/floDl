//! Training harness: ties Timeline + Monitor + DDP together for each (model, mode) combo.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use flodl::autograd::Variable;
use flodl::distributed::{ApplyPolicy, AverageBackend, Trainer};
use flodl::monitor::{Monitor, Timeline};
use flodl::nn::{Module, Optimizer, Parameter};
use flodl::tensor::{Device, Result, Tensor, TensorError};

use crate::config::{DdpMode, GuardChoice, RunConfig};
use crate::models::ModelDef;

/// Whether this process is the cluster-mode launcher (full topology
/// envelope set, no slim envelope) and so should skip dataset
/// construction. The launcher fans out to rank children and never
/// reads training data itself; the framework only needs
/// `total_samples` to compute per-rank partition sizes, served via
/// [`StubDataset`] backed by each model's `dataset_size_hint`.
///
/// On rank processes (slim envelope set) and standalone single-host
/// runs (neither set), this returns false and the real datasets are
/// constructed as usual.
fn is_cluster_launcher() -> bool {
    std::env::var_os("FLODL_FULL_CLUSTER_JSON").is_some()
        && std::env::var_os("FLODL_CLUSTER_JSON").is_none()
}

/// Whether this process is a cluster rank child (slim envelope + rank
/// slot set). Means the production via_coord path is engaged:
/// `flodl::Trainer::builder().run()` resolves `Role::Rank`, training
/// flows through `cluster_worker` → controller → `cluster_coordinator`
/// (where Fastest dispatcher + checkpoint retry + role failover live).
///
/// Returns false on the launcher process (full envelope only) and on
/// standalone single-host runs (neither envelope).
fn is_cluster_rank() -> bool {
    std::env::var_os("FLODL_CLUSTER_JSON").is_some()
        && std::env::var_os("FLODL_LOCAL_RANK").is_some()
}

/// One-line role banner for operator visibility. Printed once per run
/// at the top of [`run_combo`]; tells the operator at a glance which
/// dispatch path is exercising the run, so a "this should have gone
/// through cluster_coordinator" question has an immediate answer in
/// the captured stderr.
fn role_banner() -> String {
    if is_cluster_launcher() {
        "role=launcher (fan-out, no training body)".to_string()
    } else if is_cluster_rank() {
        let slot = std::env::var("FLODL_LOCAL_RANK").unwrap_or_else(|_| "?".to_string());
        format!(
            "role=rank slot={slot} (via_coord → cluster_coordinator)",
        )
    } else {
        "role=single-device (no cluster envelope)".to_string()
    }
}

/// Reports a fixed `len()` and refuses `get_batch`. Substituted for the
/// real dataset on the cluster-mode launcher process, where the
/// framework needs `total_samples` to build the coord config but never
/// reads training data (the launcher fans out, ranks train). A
/// `get_batch` call here is a programming error: it means the launcher
/// reached the training body, which it never should.
struct StubDataset {
    len: usize,
}

impl StubDataset {
    fn new(len: usize) -> Self {
        Self { len }
    }
}

impl flodl::data::BatchDataSet for StubDataset {
    fn len(&self) -> usize {
        self.len
    }
    fn get_batch(&self, _indices: &[usize]) -> Result<Vec<Tensor>> {
        Err(TensorError::new(
            "ddp-bench StubDataset::get_batch called on cluster-mode \
             launcher process. The launcher should not read training \
             data; it fans out to rank children. This is a bug in the \
             harness's launcher/rank role detection.",
        ))
    }
}

/// Wrapper so `Box<dyn Optimizer>` satisfies the `O: Optimizer` bound
/// in DDP generic closures.
struct DynOptimizer(Box<dyn Optimizer>);


impl Optimizer for DynOptimizer {
    fn step(&mut self) -> Result<()> { self.0.step() }
    fn zero_grad(&self) { self.0.zero_grad() }
    fn lr(&self) -> f64 { self.0.lr() }
    fn set_lr(&mut self, lr: f64) { self.0.set_lr(lr) }
}

/// Result of a single (model, mode) benchmark run.
#[derive(Clone)]
pub struct RunResult {
    pub model_name: String,
    pub mode: String,
    pub final_loss: f64,
    /// Training config for baseline generation.
    pub epochs: usize,
    pub batches_per_epoch: usize,
    pub batch_size: usize,
}

/// Run a single (model, mode) combination.
pub fn run_combo(model_def: &ModelDef, mode: &DdpMode, config: &RunConfig) -> Result<RunResult> {
    let mode_str = mode.to_string();

    // Operator-visible dispatch banner: confirms which path is engaged
    // (launcher / rank / single-device) so anyone glancing at captured
    // stderr can verify the production cluster_coordinator path is
    // actually being exercised on multi-GPU runs.
    eprintln!("ddp-bench: {}", role_banner());

    let run_dir = format!("{}/{}/{}", config.output_dir, model_def.name, mode_str);
    // Shared-storage directory setup is the controller's job. The
    // launcher / single-host process creates `run_dir` once; worker
    // ranks (FLODL_CLUSTER_JSON set by the launcher on each spawned
    // rank child) find it ready and skip the create. Workers that
    // later write into the same tree (e.g. checkpoint files) operate
    // on dirs the controller already provisioned, so a worker's
    // shared mount can stay read-only-friendly without breaking
    // setup.
    let is_worker_rank = std::env::var_os("FLODL_CLUSTER_JSON").is_some();
    if !is_worker_rank {
        std::fs::create_dir_all(&run_dir)
            .map_err(|e| TensorError::new(&format!("failed to create {run_dir}: {e}")))?;
    }

    let lr_note = if (config.lr - model_def.defaults.lr).abs() > 1e-10 {
        format!(", lr={:.1e} ({:.2}x)", config.lr, config.lr / model_def.defaults.lr)
    } else {
        String::new()
    };

    // Create dataset.
    let load_start = Instant::now();
    let virtual_len = config.batches_per_epoch * config.batch_size;
    let pool_size = (config.batch_size * crate::data::POOL_MUL).min(virtual_len);
    let dataset_cfg = crate::models::DatasetConfig {
        seed: config.seed,
        data_dir: config.data_dir.clone(),
        virtual_len,
        pool_size,
    };
    // Cluster-mode launcher: skip the real dataset load. The launcher
    // fans out to rank children and never reads training data; only
    // `total_samples` is needed (for coord-side partition sizing).
    // `dataset_size_hint` reports that number without constructing the
    // real dataset. Test dataset is set to None on the launcher (the
    // launcher doesn't run per-epoch eval or the final-eval block).
    let is_launcher = is_cluster_launcher();
    let dataset: Arc<dyn flodl::data::BatchDataSet> = if is_launcher {
        let n = (model_def.dataset_size_hint)(&dataset_cfg)?;
        Arc::new(StubDataset::new(n))
    } else {
        (model_def.dataset)(&dataset_cfg)?
    };
    let test_dataset: Option<Arc<dyn flodl::data::BatchDataSet>> = if is_launcher {
        None
    } else if let Some(test_fn) = model_def.test_dataset {
        Some(test_fn(&dataset_cfg)?)
    } else {
        None
    };
    let load_ms = load_start.elapsed().as_millis();

    // Real-data mode: batches_per_epoch == 0 means "use full dataset".
    // Compute actual batches from dataset size.
    let real_data = config.batches_per_epoch == 0;
    let actual_batches = if real_data {
        dataset.len() / config.batch_size
    } else {
        config.batches_per_epoch
    };

    let preload_tag = if mode.requires_multi_gpu() { "cpu" } else { "gpu-preload" };
    let baseline_tag = if model_def.needs_baseline_eval { " [paper-baseline]" } else { "" };
    if real_data {
        eprintln!(
            "\n=== {} / {} ({} epochs, {} samples, {} batches x {}{}){} ===",
            model_def.name, mode_str, config.epochs, dataset.len(), actual_batches,
            config.batch_size, lr_note, baseline_tag,
        );
        eprintln!("  data: {} samples, mode={preload_tag} ({load_ms}ms)", dataset.len());
    } else {
        eprintln!(
            "\n=== {} / {} ({} epochs, {} batches x {}{}){} ===",
            model_def.name, mode_str, config.epochs, actual_batches, config.batch_size, lr_note,
            baseline_tag,
        );
        eprintln!("  data: pool={pool_size}, virtual={virtual_len}, mode={preload_tag} ({load_ms}ms)");
    }

    // Start timeline AFTER data loading so measurements reflect training only.
    let timeline = Timeline::new(100);
    timeline.start();

    // Create monitor. Suppress its "training complete in …" terminal
    // line: the harness owns a richer `done: loss=…, syncs=…,
    // idle=…` summary below, so emitting both is just duplication.
    // HTML archive + dashboard pushes are unaffected.
    let mut monitor = Monitor::new(config.epochs);
    monitor.silent_summary();
    if let Some(port) = config.monitor_port {
        monitor
            .serve(port)
            .map_err(|e| TensorError::new(&format!("monitor serve: {e}")))?;
    }

    let start = Instant::now();
    let result = match mode {
        DdpMode::Solo(gpu_idx) => run_baseline_solo(
            model_def,
            *gpu_idx,
            dataset,
            test_dataset,
            config,
            actual_batches,
            real_data,
            &timeline,
            &mut monitor,
        ),
        DdpMode::Builder { policy, backend } => run_unified(
            model_def,
            *policy,
            *backend,
            dataset,
            test_dataset,
            config,
            &timeline,
            &mut monitor,
        ),
    };
    let total_ms = start.elapsed().as_secs_f64() * 1000.0;

    timeline.stop();

    // Rotate existing artifacts before overwriting.
    rotate_artifact(&run_dir, "training.log");
    rotate_artifact(&run_dir, "timeline.json");
    rotate_artifact(&run_dir, "timeline.csv");
    rotate_artifact(&run_dir, "timeline.html");

    // Save artifacts
    let _ = timeline.save_json(&format!("{run_dir}/timeline.json"));
    let _ = timeline.save_csv(&format!("{run_dir}/timeline.csv"));
    let _ = timeline.save_html(&format!("{run_dir}/timeline.html"));
    monitor.finish();

    let (final_loss, _epoch_times, log_lines) = result?;

    // Save training log with GPU header and total time
    {
        let log_path = format!("{run_dir}/training.log");
        #[cfg(feature = "cuda")]
        let header = {
            let mut h = String::new();
            for dev in flodl::tensor::cuda_devices() {
                h.push_str(&format!(
                    "# gpu{}: {} ({}GB, sm_{}{})\n",
                    dev.index, dev.name, dev.total_memory / (1024 * 1024 * 1024),
                    dev.sm_major, dev.sm_minor,
                ));
            }
            h
        };
        #[cfg(not(feature = "cuda"))]
        let header = String::new();
        let total_secs = total_ms / 1000.0;
        let footer = format!(
            "# total: {:.1}s ({:.0}m {:.0}s)",
            total_secs, (total_secs / 60.0).floor(), total_secs % 60.0,
        );
        let content = header + &log_lines.join("\n") + "\n" + &footer + "\n";
        let _ = std::fs::write(&log_path, content);
    }

    // Clean up CUDA state between runs. NCCL communicators and cached
    // allocator blocks from the previous run can fragment VRAM or leave
    // stale stream state that interferes with the next NCCL init.
    #[cfg(feature = "cuda")]
    {
        let gpu_count = flodl::tensor::cuda_device_count();
        for i in 0..gpu_count {
            flodl::tensor::cuda_synchronize(i as u8);
            flodl::tensor::cuda_empty_cache();
        }
    }

    let summary = timeline.summary();
    // Cluster-mode workers (rank child processes) don't see aggregated
    // metrics — those flow to the controller. Their local `final_loss`
    // stays at its init value and the line would always read 0.0. The
    // controller-active principle says the controller is the canonical
    // narrator; let it own the end-of-run summary.
    if !is_worker_rank {
        eprintln!(
            "  done: loss={:.6}, total={:.1}s, syncs={}, idle=[{}]",
            final_loss,
            total_ms / 1000.0,
            summary.sync_count,
            summary
                .gpu_idle_pct
                .iter()
                .enumerate()
                .map(|(i, p)| format!("gpu{i}:{p:.1}%"))
                .collect::<Vec<_>>()
                .join(", "),
        );
    }

    Ok(RunResult {
        model_name: model_def.name.to_string(),
        mode: mode_str,
        final_loss,
        epochs: config.epochs,
        batches_per_epoch: actual_batches,
        batch_size: config.batch_size,
    })
}

// ---------------------------------------------------------------------------
// Preloading
// ---------------------------------------------------------------------------

/// Preload entire dataset to GPU as bulk tensors.
/// Returns one tensor per output group (e.g. [images, labels]).
fn preload_full_dataset(
    dataset: &dyn flodl::data::BatchDataSet,
    device: Device,
) -> Result<Vec<Tensor>> {
    let n = dataset.len();
    let indices: Vec<usize> = (0..n).collect();
    let tensors = dataset.get_batch(&indices)?;
    tensors
        .into_iter()
        .map(|t| t.to_device(device))
        .collect::<Result<Vec<_>>>()
}

/// Preload a small pool of batches to GPU. Returns `POOL_MUL` batches;
/// the training loop cycles through them via `batch_idx % pool.len()`.
fn preload_gpu_batches(
    dataset: &dyn flodl::data::BatchDataSet,
    device: Device,
    batch_size: usize,
) -> Result<Vec<Vec<Tensor>>> {
    let n = crate::data::POOL_MUL;
    let mut pool = Vec::with_capacity(n);
    for i in 0..n {
        let start = i * batch_size;
        let indices: Vec<usize> = (start..start + batch_size).collect();
        let batch = dataset.get_batch(&indices)?;
        let gpu_batch: Vec<Tensor> = batch
            .into_iter()
            .map(|t| t.to_device(device))
            .collect::<Result<Vec<_>>>()?;
        pool.push(gpu_batch);
    }
    Ok(pool)
}

/// Form a batch from bulk GPU tensors via index_select.
fn slice_batch(gpu_data: &[Tensor], start: usize, end: usize, device: Device) -> Result<Vec<Tensor>> {
    let idx: Vec<i64> = (start as i64..end as i64).collect();
    let idx_tensor = Tensor::from_i64(&idx, &[idx.len() as i64], device)?;
    gpu_data
        .iter()
        .map(|t| t.index_select(0, &idx_tensor))
        .collect::<Result<Vec<_>>>()
}

// ---------------------------------------------------------------------------
// Solo GPU
// ---------------------------------------------------------------------------

/// Solo GPU run for paper-baseline reproduction.
///
/// Used for models whose published baseline reports per-epoch loss + accuracy
/// curves (set `ModelDef::needs_baseline_eval = true`). Runs eval on the test
/// set after every training epoch and emits `train=` / `eval_time=` columns
/// in the log so the report can subtract eval cost from solo's wall time
/// when computing DDP speedup.
///
/// All Solo runs route here.
#[allow(clippy::too_many_arguments)]
fn run_baseline_solo(
    model_def: &ModelDef,
    gpu_idx: usize,
    dataset: Arc<dyn flodl::data::BatchDataSet>,
    test_dataset: Option<Arc<dyn flodl::data::BatchDataSet>>,
    config: &RunConfig,
    batches_per_epoch: usize,
    real_data: bool,
    timeline: &Arc<Timeline>,
    monitor: &mut Monitor,
) -> Result<(f64, Vec<f64>, Vec<String>)> {
    let device = Device::CUDA(gpu_idx as u8);
    let model = (model_def.build)(device)?;
    let params = model.parameters();
    let mut optimizer = (model_def.optimizer)(&params, config.lr);
    // Per-batch scheduling: total_steps = batches * epochs (matches nanoGPT etc.).
    // Solo: world_size=1.
    let solo_batches = if real_data {
        dataset.len() / config.batch_size
    } else {
        batches_per_epoch
    };
    let scheduler = model_def.scheduler.map(|f| f(config.lr, solo_batches * config.epochs, 1));
    let mut global_batch: usize = 0;
    let mut log_lines: Vec<String> = Vec::new();
    model.train();

    let mut epoch_times = Vec::with_capacity(config.epochs);
    let mut final_loss = 0.0;

    if real_data {
        // Full-dataset mode: preload everything, iterate through all data.
        let gpu_data = preload_full_dataset(dataset.as_ref(), device)?;
        let n = gpu_data[0].shape()[0] as usize;
        let bs = config.batch_size;

        // Preload test data for evaluation (if available).
        let test_gpu_data = test_dataset
            .as_ref()
            .map(|ds| preload_full_dataset(ds.as_ref(), device))
            .transpose()?;
        if let Some(ref tgd) = test_gpu_data {
            let tn = tgd[0].shape()[0] as usize;
            eprintln!("  eval: {tn} test samples");
        }

        // Track training-only and eval-only wall time so the report can
        // strip eval cost from the speedup denominator. DDP modes pay only
        // a single final eval; folding solo's per-epoch eval into the
        // comparison would silently inflate DDP's speedup.
        let mut total_train_ms = 0.0_f64;
        let mut total_eval_ms = 0.0_f64;

        for epoch in 0..config.epochs {
            timeline.event(flodl::monitor::EventKind::EpochStart { epoch });
            let epoch_start = Instant::now();
            let mut total_loss = 0.0;
            let mut batch_count = 0;

            for batch_start in (0..n).step_by(bs) {
                let end = (batch_start + bs).min(n);
                if end - batch_start < bs { break; } // drop incomplete last batch
                let batch = slice_batch(&gpu_data, batch_start, end, device)?;
                let batch = if let Some(aug) = model_def.augment_fn {
                    aug(&batch)?
                } else {
                    batch
                };

                let loss = (model_def.train_fn)(model.as_ref(), &batch)?;
                let loss_val = loss.item()?;

                if let Some(ref sched) = scheduler {
                    optimizer.set_lr(sched.lr(global_batch));
                }
                optimizer.zero_grad();
                loss.backward()?;
                optimizer.step()?;

                total_loss += loss_val;
                batch_count += 1;
                global_batch += 1;
            }

            let train_ms = epoch_start.elapsed().as_secs_f64() * 1000.0;
            total_train_ms += train_ms;
            final_loss = if batch_count > 0 { total_loss / batch_count as f64 } else { 0.0 };
            timeline.event(flodl::monitor::EventKind::EpochEnd {
                epoch,
                loss: final_loss,
                lr: optimizer.lr(),
            });

            // Drain per-batch scalars from record_scalar (training accuracy etc.).
            let scalars = drain_epoch_scalars();

            // Eval metric (accuracy, etc.) if available.
            // Use held-out test data when present, otherwise fall back to training data.
            if let Some(eval_fn) = model_def.eval_fn {
                model.eval();
                let eval_start = Instant::now();
                let eval_data = test_gpu_data.as_deref().unwrap_or(&gpu_data);
                let eval_n = eval_data[0].shape()[0] as usize;
                let avg = flodl::autograd::no_grad(|| -> Result<f64> {
                    let mut total_metric = 0.0;
                    let mut eval_samples = 0usize;
                    for batch_start in (0..eval_n).step_by(bs) {
                        let end = (batch_start + bs).min(eval_n);
                        if end - batch_start < bs { break; }
                        let batch = slice_batch(eval_data, batch_start, end, device)?;
                        let metric = eval_fn(model.as_ref(), &batch)?;
                        total_metric += metric * (end - batch_start) as f64;
                        eval_samples += end - batch_start;
                    }
                    Ok(if eval_samples > 0 { total_metric / eval_samples as f64 } else { 0.0 })
                })?;
                let eval_ms = eval_start.elapsed().as_secs_f64() * 1000.0;
                total_eval_ms += eval_ms;
                model.train();
                let mut line = format!("epoch {epoch}: loss={final_loss:.6}, eval={avg:.4}");
                line.push_str(&format_scalars(&scalars));
                line.push_str(&format!(
                    ", train={:.1}s, eval_time={:.1}s",
                    train_ms / 1000.0, eval_ms / 1000.0,
                ));
                eprintln!("    {line}");
                log_lines.push(line);
            } else {
                let mut line = format!("epoch {epoch}: loss={final_loss:.6}");
                line.push_str(&format_scalars(&scalars));
                line.push_str(&format!(", train={:.1}s", train_ms / 1000.0));
                eprintln!("    {line}");
                log_lines.push(line);
            }

            monitor.log(epoch, epoch_start.elapsed(), &[("loss", final_loss)]);
            epoch_times.push(train_ms);
        }

        // Summary line consumed by `analyze::parse_training_log`: lets the
        // speedup table compare DDP wall-time against solo's training-only
        // wall-time rather than total (which would include eval cost solo
        // pays per-epoch but DDP only pays once at the end).
        log_lines.push(format!(
            "# train_only: {:.1}s (eval: {:.1}s)",
            total_train_ms / 1000.0, total_eval_ms / 1000.0,
        ));
    } else {
        // Synthetic pool mode: preload small pool, recycle batches.
        let gpu_pool = preload_gpu_batches(dataset.as_ref(), device, config.batch_size)?;
        let pool_len = gpu_pool.len();

        for epoch in 0..config.epochs {
            timeline.event(flodl::monitor::EventKind::EpochStart { epoch });
            let epoch_start = Instant::now();
            let mut total_loss = 0.0;

            for batch_idx in 0..batches_per_epoch {
                let batch = &gpu_pool[batch_idx % pool_len];

                let loss = (model_def.train_fn)(model.as_ref(), batch)?;
                let loss_val = loss.item()?;

                if let Some(ref sched) = scheduler {
                    optimizer.set_lr(sched.lr(global_batch));
                }
                optimizer.zero_grad();
                loss.backward()?;
                optimizer.step()?;

                total_loss += loss_val;
                global_batch += 1;
            }

            let epoch_ms = epoch_start.elapsed().as_secs_f64() * 1000.0;
            final_loss = total_loss / batches_per_epoch as f64;
            timeline.event(flodl::monitor::EventKind::EpochEnd {
                epoch,
                loss: final_loss,
                lr: optimizer.lr(),
            });

            monitor.log(epoch, epoch_start.elapsed(), &[("loss", final_loss)]);
            epoch_times.push(epoch_ms);
        }
    }

    Ok((final_loss, epoch_times, log_lines))
}

// ---------------------------------------------------------------------------
// Unified DDP path (sync, cadence, async; nccl + cpu backends)
// ---------------------------------------------------------------------------

/// Run any DDP mode through `Trainer::builder` with thread-per-GPU.
///
/// Handles every `DdpMode::Builder { policy, backend }` combination
/// (nccl-sync, nccl-cadence, cpu-sync, cpu-cadence, cpu-async).
/// Solo runs of paper-baseline models go through [`run_baseline_solo`] instead.
#[allow(clippy::borrowed_box, clippy::type_complexity, clippy::too_many_arguments)]
fn run_unified(
    model_def: &ModelDef,
    policy: ApplyPolicy,
    backend: AverageBackend,
    dataset: Arc<dyn flodl::data::BatchDataSet>,
    test_dataset: Option<Arc<dyn flodl::data::BatchDataSet>>,
    config: &RunConfig,
    timeline: &Arc<Timeline>,
    monitor: &mut Monitor,
) -> Result<(f64, Vec<f64>, Vec<String>)> {
    let build_fn = model_def.build;
    let train_fn_ptr = model_def.train_fn;
    let augment_fn = model_def.augment_fn;
    let opt_fn = model_def.optimizer;
    let lr = config.lr;

    // Per-batch scheduler factory: receives world_size from the framework
    // so user-defined schedulers can account for multi-GPU training.
    let batches_per_epoch = dataset.len() / config.batch_size;
    let sched_factory = model_def.scheduler;
    let sched_epochs = config.epochs;

    let mut builder = Trainer::builder(
        build_fn,
        move |params: &[Parameter]| DynOptimizer(opt_fn(params, lr)),
        move |model: &Box<dyn Module>, batch: &[Tensor]| -> Result<Variable> {
            let batch = if let Some(aug) = augment_fn {
                aug(batch)?
            } else {
                batch.to_vec()
            };
            train_fn_ptr(model.as_ref(), &batch)
        },
    )
    .dataset(dataset.clone())
    .batch_size(config.batch_size)
    .num_epochs(config.epochs)
    .policy(policy)
    .backend(backend)
    .timeline(Arc::clone(timeline));

    // Heterogeneous topology: explicit per-rank shares disable the uniform
    // default. Without this, the fast GPU idles waiting for the slow ones at
    // every sync barrier (the publication-arc anti-pattern).
    if let Some(ratios) = &config.partition_ratios {
        builder = builder.partition_ratios(ratios);
    }

    // Epoch-callback policy: user-selected via `--epoch-callback-policy`.
    // `None` leaves the framework default (`Rank(0)`); `Some(Fastest)` lets
    // ElChe pick the lowest-ms-per-batch rank, sticky thereafter.
    if let Some(policy) = config.epoch_callback_policy {
        builder = builder.epoch_callback_policy(policy);
    }

    if config.elche_relax_up {
        builder = builder.elche_relax_up(true);
    }

    if config.meta_controller {
        builder = builder.meta_controller(true);
    }

    if let Some(max) = config.max_anchor {
        builder = builder.max_anchor(max);
    }

    if let Some(min) = config.min_anchor {
        builder = builder.min_anchor(min);
    }

    // cpu-async EASGD blend defaults to α=0.5 inside `DdpBuilder::run()`
    // (the (Async, Cpu) framework default); only override here when the
    // user passed `--easgd-alpha` explicitly.
    if let Some(alpha) = config.easgd_alpha {
        builder = builder.easgd_alpha(alpha);
    }

    // Materialize the configured convergence guard. NoGuard / TrendGuard /
    // MsfGuard each implement the trait; we pass through the generic
    // `convergence_guard` builder method which boxes internally.
    builder = match &config.guard {
        GuardChoice::None => builder
            .convergence_guard(flodl::distributed::ddp_run::NoGuard),
        GuardChoice::Trend { threshold } => builder
            .convergence_guard(flodl::distributed::ddp_run::TrendGuard::new(*threshold)),
        GuardChoice::Msf {
            suppress_threshold,
            suppress_sustain,
            nudge_threshold,
            nudge_sustain,
            nudge_factor,
            alpha,
        } => {
            let g = flodl::distributed::ddp_run::MsfGuard::default()
                .with_alpha(*alpha)
                .with_suppress(*suppress_threshold, *suppress_sustain)
                .with_nudge(*nudge_threshold, *nudge_sustain, *nudge_factor);
            builder.convergence_guard(g)
        }
    };

    if let Some(sf) = sched_factory {
        let bpe = batches_per_epoch;
        builder = builder.scheduler(move |world_size| {
            let total_steps = bpe * sched_epochs;
            Arc::from(sf(lr, total_steps, world_size))
        });
    }

    // Per-epoch eval hook: when --per-epoch-eval is set and the model
    // exposes eval_fn, install an EpochFn that fires on the worker's
    // transition into epoch N+1 (so the model state is post-epoch-N).
    // Only rank 0 evaluates (test data is identical across ranks; in
    // Sync mode all ranks have consensus params, in Cadence/Async rank 0
    // sees its own near-consensus state). Eval values stream back to the
    // host over an mpsc channel and are merged into the per-epoch log
    // line.
    let (eval_tx, eval_rx) = std::sync::mpsc::channel::<(usize, f64)>();
    if config.per_epoch_eval
        && let Some(eval_fn) = model_def.eval_fn
        && let Some(test_ds) = test_dataset.as_ref()
    {
        // Pre-load test data on rank 0's device. Workers run on
        // `Device::CUDA(rank)`; rank 0 is `Device::CUDA(0)`.
        let device = Device::CUDA(0);
        let test_data = Arc::new(preload_full_dataset(test_ds.as_ref(), device)?);
        let bs = config.batch_size;
        let eval_tx_efn = eval_tx.clone();
        builder = builder.epoch_fn(move |epoch: usize, worker: &mut flodl::distributed::ddp_run::GpuWorker<Box<dyn Module>>| {
            // Skip rank > 0: eval is identical across ranks (Sync) or
            // approximately so (Cadence/Async). Single eval per epoch.
            if worker.rank() != 0 {
                return;
            }
            // Skip epoch 0: this fires on transition INTO epoch 0, before
            // any training has happened. There is no "previous epoch" to
            // evaluate. The first useful eval fires on transition into
            // epoch 1 and tags the result as epoch 0.
            if epoch == 0 {
                return;
            }
            let prev_epoch = epoch - 1;
            let model: &Box<dyn Module> = worker.model();
            model.eval();
            let result = flodl::autograd::no_grad(|| -> Result<f64> {
                let n = test_data[0].shape()[0] as usize;
                let mut total_metric = 0.0;
                let mut samples = 0usize;
                for batch_start in (0..n).step_by(bs) {
                    let end = (batch_start + bs).min(n);
                    if end - batch_start < bs { break; }
                    let batch = slice_batch(&test_data, batch_start, end, device)?;
                    let metric = eval_fn(model.as_ref(), &batch)?;
                    total_metric += metric * (end - batch_start) as f64;
                    samples += end - batch_start;
                }
                Ok(if samples > 0 { total_metric / samples as f64 } else { 0.0 })
            });
            model.train();
            if let Ok(metric) = result {
                let _ = eval_tx_efn.send((prev_epoch, metric));
            }
        });
    }
    drop(eval_tx); // worker keeps its own clone via the EpochFn closure

    // Single canonical eval (cluster mode): the controller dispatches ONE
    // eval to the chosen rank (Fastest by default) on the coherent consensus
    // model after the final reduce, and the scalar flows back to
    // `eval_result_fn` here on the launcher. This replaces the redundant
    // per-rank final eval below for cluster runs. The eval closure bridges
    // the model_def's per-batch `eval_fn` to the framework's dataset-based
    // `EvalFn`, running on the worker's own device.
    let final_eval_cell: Arc<Mutex<Option<f64>>> = Arc::new(Mutex::new(None));
    if let (Some(eval_fn), Some(test_ds)) = (model_def.eval_fn, test_dataset.clone()) {
        let bs = config.batch_size;
        builder = builder
            .eval_dataset(test_ds)
            .eval_fn(move |model: &Box<dyn Module>,
                           ds: &dyn flodl::data::BatchDataSet|
                  -> Result<f64> {
                // Eval on the worker's own device (derived from the model so
                // it is correct on both CUDA_VISIBLE_DEVICES-scoped ranks and
                // unscoped ones).
                let device = model
                    .parameters()
                    .first()
                    .map(|p| p.variable.data().device())
                    .unwrap_or(Device::CPU);
                let data = preload_full_dataset(ds, device)?;
                let n = data[0].shape()[0] as usize;
                flodl::autograd::no_grad(|| -> Result<f64> {
                    let mut total = 0.0;
                    let mut samples = 0usize;
                    for start in (0..n).step_by(bs) {
                        let end = (start + bs).min(n);
                        if end - start < bs {
                            break;
                        }
                        let batch = slice_batch(&data, start, end, device)?;
                        total += eval_fn(model.as_ref(), &batch)? * (end - start) as f64;
                        samples += end - start;
                    }
                    Ok(if samples > 0 { total / samples as f64 } else { 0.0 })
                })
            });
        let cell = Arc::clone(&final_eval_cell);
        builder = builder.eval_result_fn(move |_rank: usize, metric: f64| -> Result<()> {
            *cell.lock().unwrap() = Some(metric);
            Ok(())
        });
    }

    let handle = builder.run()?;

    let mut epoch_times = Vec::new();
    let mut final_loss = 0.0;
    let mut log_lines: Vec<String> = Vec::new();
    // Eval values arrive on `eval_rx` from the worker EpochFn. The hook
    // for epoch N's eval fires on transition into epoch N+1 — which can
    // race with the host receiving metrics for epoch N. Buffer pending
    // evals here and fold them into the log line on the next metrics
    // tick if not yet present.
    let mut pending_eval: std::collections::HashMap<usize, f64> =
        std::collections::HashMap::new();

    while let Some(metrics) = handle.next_metrics() {
        final_loss = metrics.avg_loss;
        epoch_times.push(metrics.epoch_ms);
        // Drain any eval values that arrived since the last tick.
        while let Ok((ep, val)) = eval_rx.try_recv() {
            pending_eval.insert(ep, val);
        }
        // Emit eval values for past epochs (race losers) as supplemental
        // lines BEFORE printing the current metrics. Eval(N) typically
        // arrives after metrics(N) but before metrics(N+1) — surfacing it
        // here means streaming output shows `epoch N: eval=X.XXXX` right
        // before `epoch N+1: loss=...`, instead of all evals batched at
        // end-of-run. The current-epoch eval (if it somehow already
        // arrived) is still folded inline below; this drain only emits
        // strictly-past epochs.
        let mut past_keys: Vec<usize> = pending_eval
            .keys()
            .copied()
            .filter(|ep| *ep < metrics.epoch)
            .collect();
        past_keys.sort_unstable();
        for ep in past_keys {
            if let Some(val) = pending_eval.remove(&ep) {
                let line = format!("epoch {ep}: eval={val:.4}");
                eprintln!("    {line}");
                log_lines.push(line);
            }
        }
        // Build log line: loss + (eval if available) + model-defined scalars + time.
        let scalars: std::collections::BTreeMap<String, f64> =
            metrics.scalars.iter().map(|(k, v)| (k.clone(), *v)).collect();
        let mut line = format!("epoch {}: loss={:.6}", metrics.epoch, metrics.avg_loss);
        if let Some(eval_val) = pending_eval.remove(&metrics.epoch) {
            line.push_str(&format!(", eval={eval_val:.4}"));
        }
        line.push_str(&format_scalars(&scalars));
        line.push_str(&format!(", time={:.1}s", metrics.epoch_ms / 1000.0));
        eprintln!("    {line}");
        log_lines.push(line);

        // Per-rank breakdown for multi-rank runs. Layout is
        //   [highest-share]  [random middle]  [lowest-share]
        // where highest-share is the fastest rank in the cadence (gets
        // the most batches) and lowest-share is the slow-anchor rank.
        // Random middle is sampled uniformly from the in-between ranks
        // each epoch — O(1) per epoch and ergodic over the run, which
        // scales to large worlds without flooding the log line. For
        // world_size == 2 the middle is omitted; for == 3 the middle is
        // the unique non-extreme rank (no randomness needed).
        if metrics.device_indices.len() >= 2 {
            let n = metrics.device_indices.len();
            let mut sorted: Vec<usize> = (0..n).collect();
            sorted.sort_by(|&a, &b| {
                let sa = metrics.per_rank_batch_share.get(a).copied().unwrap_or(0.0);
                let sb = metrics.per_rank_batch_share.get(b).copied().unwrap_or(0.0);
                sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
            });
            let highest = sorted[0];
            let lowest = sorted[n - 1];
            let middle: Option<usize> = match n {
                2 => None,
                3 => Some(sorted[1]),
                _ => {
                    let pick = flodl::Rng::from_entropy().usize(n - 2);
                    Some(sorted[1 + pick])
                }
            };

            let render = |r: usize| -> String {
                let dev = metrics.device_indices[r];
                let share = metrics.per_rank_batch_share.get(r).copied().unwrap_or(0.0);
                let tput = metrics.per_rank_throughput.get(r).copied().unwrap_or(0.0);
                format!(" rank{r}[cuda{dev},share={share:.4},tput={tput:.2}]")
            };

            let mut rank_line = String::from("per-rank:");
            rank_line.push_str(&render(highest));
            if let Some(mid) = middle {
                rank_line.push_str(&render(mid));
            }
            rank_line.push_str(&render(lowest));
            eprintln!("    {rank_line}");
            log_lines.push(rank_line);
        }

        monitor.log(
            metrics.epoch,
            Duration::from_millis(metrics.epoch_ms as u64),
            &metrics,
        );
    }

    // Final drain: catches the last epoch's eval (which has no subsequent
    // metrics tick to surface it via the per-tick past-epoch drain above)
    // and any straggler values that arrived after the last metrics tick.
    while let Ok((ep, val)) = eval_rx.try_recv() {
        pending_eval.insert(ep, val);
    }
    let mut leftovers: Vec<(usize, f64)> = pending_eval.into_iter().collect();
    leftovers.sort_unstable_by_key(|x| x.0);
    for (ep, val) in leftovers {
        let line = format!("epoch {ep}: eval={val:.4}");
        eprintln!("    {line}");
        log_lines.push(line);
    }

    let state = handle.join()?;

    // Final evaluation.
    //
    // Cluster mode: the SINGLE canonical eval already ran on the chosen rank
    // (Fastest) via the framework's `eval_fn` + the controller's post-
    // consensus-reduce dispatch; the metric came back to `eval_result_fn`
    // and lives in `final_eval_cell` on the launcher. Print it there. Rank
    // children do NOT eval their own copy (that produced the redundant
    // per-rank numbers). Single-host (threaded / single-GPU) keeps the
    // in-process eval below — one process, one eval.
    if is_cluster_launcher() {
        if let Some(metric) = *final_eval_cell.lock().unwrap() {
            let line = format!("final eval={metric:.4}");
            eprintln!("    {line}");
            log_lines.push(line);
        }
    } else if is_cluster_rank() {
        // No-op: the launcher owns the single canonical eval.
    } else if let Some(eval_fn) = model_def.eval_fn {
        let device = Device::CUDA(0);
        let model = (model_def.build)(device)?;
        let model_params = model.parameters();
        let model_bufs = model.buffers();
        eprintln!("  final eval: loading state ({} params, {} buffers -> model has {} params, {} buffers)",
            state.params.len(), state.buffers.len(), model_params.len(), model_bufs.len());
        // Load trained state into model.
        {
            let _no_grad = flodl::autograd::NoGradGuard::new();
            for (param, src) in model_params.iter().zip(&state.params) {
                param.variable.data().copy_(&src.to_device(device)?, false)?;
            }
        }
        for (buf, src) in model_bufs.iter().zip(&state.buffers) {
            buf.get().copy_(&src.to_device(device)?, false)?;
        }
        model.eval();

        // Load test data (or fall back to training data).
        let eval_dataset = test_dataset.as_ref().unwrap_or(&dataset);
        let eval_data = preload_full_dataset(eval_dataset.as_ref(), device)?;
        let n = eval_data[0].shape()[0] as usize;
        let bs = config.batch_size;

        let avg = flodl::autograd::no_grad(|| -> Result<f64> {
            let mut total_metric = 0.0;
            let mut eval_samples = 0usize;
            for batch_start in (0..n).step_by(bs) {
                let end = (batch_start + bs).min(n);
                if end - batch_start < bs { break; }
                let batch = slice_batch(&eval_data, batch_start, end, device)?;
                let metric = eval_fn(model.as_ref(), &batch)?;
                total_metric += metric * (end - batch_start) as f64;
                eval_samples += end - batch_start;
            }
            Ok(if eval_samples > 0 { total_metric / eval_samples as f64 } else { 0.0 })
        })?;

        let line = format!("final eval={avg:.4}");
        eprintln!("    {line}");
        log_lines.push(line);
    }

    Ok((final_loss, epoch_times, log_lines))
}

// ---------------------------------------------------------------------------
// Scalar helpers (record_scalar / drain_scalars integration)
// ---------------------------------------------------------------------------

/// Drain thread-local scalars accumulated by `flodl::record_scalar()` during
/// this epoch and return the per-key mean values.
fn drain_epoch_scalars() -> std::collections::BTreeMap<String, f64> {
    flodl::drain_scalars()
        .into_iter()
        .map(|(k, (sum, count))| {
            let mean = if count > 0 { sum / count as f64 } else { 0.0 };
            (k, mean)
        })
        .collect()
}

/// Format scalars as `, key=value` pairs (sorted, for appending to a log line).
fn format_scalars(scalars: &std::collections::BTreeMap<String, f64>) -> String {
    let mut s = String::new();
    for (k, v) in scalars {
        s.push_str(&format!(", {k}={v:.4}"));
    }
    s
}

/// Rotate an existing artifact file by appending a timestamp before the extension.
/// e.g. `training.log` -> `training_YYYY-MM-DD_HH-MM-SS.log`
fn rotate_artifact(dir: &str, filename: &str) {
    let path = format!("{dir}/{filename}");
    if !std::path::Path::new(&path).exists() {
        return;
    }
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    // Format as YYYY-MM-DD-HH-MM-SS (UTC) without chrono.
    let s = secs;
    let days = s / 86400;
    let time = s % 86400;
    let hh = time / 3600;
    let mm = (time % 3600) / 60;
    let ss = time % 60;
    // Days since 1970-01-01 to (y, m, d) -- civil calendar algorithm.
    let (y, m, d) = {
        let z = days as i64 + 719468;
        let era = z.div_euclid(146097);
        let doe = z.rem_euclid(146097) as u64;
        let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
        let y = yoe as i64 + era * 400;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let d = doy - (153 * mp + 2) / 5 + 1;
        let m = if mp < 10 { mp + 3 } else { mp - 9 };
        let y = if m <= 2 { y + 1 } else { y };
        (y, m, d)
    };
    let ts = format!("{y:04}-{m:02}-{d:02}-{hh:02}-{mm:02}-{ss:02}");
    let (stem, ext) = filename.rsplit_once('.').unwrap_or((filename, ""));
    let rotated = if ext.is_empty() {
        format!("{dir}/{stem}_{ts}")
    } else {
        format!("{dir}/{stem}_{ts}.{ext}")
    };
    let _ = std::fs::rename(&path, &rotated);
}
