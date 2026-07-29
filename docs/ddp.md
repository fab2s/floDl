# Distributed Training Reference

Canonical reference for flodl's multi-GPU and multi-host training surface.
For progressive introductions, see
[Tutorial 11: Multi-GPU Training](tutorials/11-multi-gpu.md) and
[Tutorial 12: Heterogeneous & Multi-Host DDP](tutorials/12-async-ddp.md).

flodl has one training entry - `Trainer::builder(...).run()` (chained
form) or `Trainer::run(model_factory, optim_factory, train_fn, cfg)`
(config-bag form). The same call runs identically on:

- a single CPU,
- a single GPU,
- N GPUs on one host (auto-promoted to process-per-rank when
  `detect_gpus() >= 2`),
- N GPUs across many hosts (driven by `fdl.cluster.yml` or
  `ClusterBuilder`).

No code changes between tiers. Scaling is a configuration decision, not
a code rewrite. (For the API-tier rationale - universal builder, manual
`Ddp::wrap` bypass, the deprecated self-driven setup tier - see the
[trainer execution tiers design note](design/trainer-execution-tiers.md).)

---

## Quick start

### Universal form - `Trainer::builder` + step closure

The recommended shape. Works on every tier without modification.

```rust
use flodl::*;
use std::sync::Arc;

let dataset: Arc<dyn BatchDataSet> = Arc::new(MyDataset::new());

// One training step: forward + loss, returns the loss Variable.
// The framework owns backward, optimizer step, gradient sync.
fn train_step(model: &impl Module, batch: &[Tensor]) -> Result<Variable> {
    let input  = Variable::new(batch[0].clone(), false);
    let target = Variable::new(batch[1].to_dtype(DType::Int64)?, false);
    cross_entropy_loss(&model.forward(&input)?, &target)
}

let handle = Trainer::builder(
        |dev| build_model_on(dev),         // model factory
        |params| Adam::new(params, 1e-3),  // optimizer factory
        train_step,                        // one-step closure
    )
    .dataset(dataset)
    .batch_size(64)
    .num_epochs(50)
    .run()?;

let state: TrainedState = handle.join()?;  // params + buffers (CPU)
```

`state.params` / `state.buffers` are CPU tensors aligned with
`build_model_on(Device::CPU)?.parameters()` and `.buffers()`. Drop them
into a fresh CPU model for inference, or continue training via
`Trainer::builder(...).resume_from(stem)` (or `TrainerConfig::resume_from`).

Per-sample datasets plug in the same way: implement
`DataSet::get(index)` (or use a shipped disk-backed reader like
`Cifar10Disk`) and hand it to `.sample_dataset(ds)` instead of
`.dataset(ds)` - or `TrainerConfig::from_dataset(ds)` in the
config-bag form. Batching, RAM caching, and reservation staging are
the framework's job; rank workers read samples ahead of the training
frontier through the shared staging tier, so storage-backed data
(local files, network mounts) trains through the same entry as
RAM-resident tensors.

### Graph models: the same entry

A flodl `Graph` (any `FlowBuilder`) is a `Module`, so it trains through the
exact entry above - just return the built graph from the `model_factory`
closure:

```rust
fn build_model(device: Device) -> Result<Box<dyn Module>> {
    let g = FlowBuilder::from(Linear::on_device(784, 10, device)?)
        .through(GELU)
        .build()?;
    Ok(Box::new(g))
}
// then: Trainer::builder(build_model, |p| Adam::new(p, 1e-3), train_step)
//           .dataset(dataset).batch_size(64).num_epochs(5).run()?;
```

`flodl-hf` task-head wrappers train the same way (task heads `impl Module`
directly, so return the wrapper from `model_factory`). To keep the loop
yourself, use the cooperative tier - `Trainer::builder(...).into_worker()?`
returns a `Worker` whose loop body you own while the controller keeps
cadence, partition, eval-election, and checkpointing. For an explicit
per-rank loop use `Ddp::wrap` (the bypass tier).

### Config-bag form - `Trainer::run`

For config-driven launchers, the umbrella `TrainerConfig<M>` gathers
every knob into one struct:

```rust
let cfg = TrainerConfig::new(dataset)
    .batch_size(64)
    .num_epochs(50)
    .elche(ElCheConfig::nccl_cadence())   // recommended NCCL default; see "ElCheMode" below
    .resume_from("ckpts/run42.fdl")       // optional
    .checkpoint_every(5)
    .save_path("ckpts/run43");

Trainer::run(
    |dev|    build_model_on(dev),
    |params| Adam::new(params, 1e-3),
    train_step,
    cfg,
)?
.join()?;
```

`Trainer::run` and `Trainer::builder().run()` reach the same launcher
trampoline; pick whichever shape your call site prefers.

> **Invariant - "no CUDA before `Trainer::run`"**: user binaries must
> not touch libtorch's CUDA context before reaching `Trainer::run`. That
> means no `flodl::tensor::cuda_device_count()`, no
> `Module::on_device(CUDA(_))`, no CUDA-Tensor construction in `main()`.
> Cluster fan-out exits the launcher process without running training;
> touching CUDA there corrupts spawned children's context on
> heterogeneous-GPU rigs. Use `flodl::sys::detect_gpus()` (CUDA-free)
> for any pre-run GPU query.

---

## ElCheMode - cadence × backend in one name

The five ways to do parameter averaging are named directly. Each name
is a `(when to average) × (how to average)` pair. `ElCheConfig::default()`
returns `NcclCadence` (the recommended NCCL mode).

| Mode | When | How | Best for |
|---|---|---|---|
| `NcclSync` | Every slow-rank step (see note below) | NCCL AllReduce | Homogeneous GPUs, correctness-first baseline |
| `NcclCadence` | Anchor-based (ElChe) | NCCL AllReduce | **Recommended NCCL default** - heterogeneous rigs; ElChe tunes the anchor so the slow device sets the pace, fast devices process proportionally more batches per averaging window |
| `CpuSync` | Every slow-rank step (see note below) | CPU averaging | Sync without NCCL (peer-access unavailable, A/B against NCCL) |
| `CpuCadence` | Anchor-based | CPU averaging | Heterogeneous rigs without fast peer links |
| `CpuAsync` | Anchor + overshoot | CPU averaging + EASGD blending (α=0.5 default) | Genuine async - averaging decoupled from the GPU pipeline via a separate channel, barrier-free application, fault-tolerant. Trades a small early-run wall surplus for it (the divergence guard grows async's window more cautiously; the surplus amortizes on long runs). Pair with the DiLoCo outer optimizer for the best eval quality on the reference rig. |

> **What "sync" means in flodl.** The `*-sync` modes are the tightest
> cadence of the same ElChe-scheduled engine, not per-batch lockstep
> DDP. Data is dispatched as an **equal split** (standard-DDP-like
> sharding), but the reduce fires as soon as **every alive rank has
> made at least one step since the last reduce**, with each rank's
> contribution work-weighted (sum-and-count). On a homogeneous rig
> this degenerates to classic synchronous DDP - one step per rank per
> reduce. On a heterogeneous rig the fast GPU runs several steps per
> reduce within its equal share instead of stalling at a per-batch
> barrier, and idles once that share is exhausted (which is why sync
> rows show high fast-GPU idle in the benchmark tables). The
> difference vs `*-cadence`: cadence waits for each rank to complete
> its **planned proportional window** (ElChe's `batch_counts`) before
> reducing, and dispatches data proportionally to measured throughput.

Every mode routes through the same machinery, so switching between
them is one line.

> **Note**: `NcclAsync` used to exist as a sixth mode (NCCL + per-rank
> cross-epoch dispatch). It was dropped - measured benefit over
> `NcclCadence` was within noise on every tested rig, and the
> in-place AllReduce writeback raced with autograd on heterogeneous
> Pascal+Blackwell setups. CPU Async (`CpuAsync`) is the real
> asynchronous mode: averaging is decoupled from the GPU pipeline
> through a separate channel.

### `ElCheConfig` - presets + overrides

```rust
let elche = ElCheConfig::nccl_cadence()  // also the value of ElCheConfig::default()
    .max_anchor(20)
    .overhead_target(0.05);
```

| Preset constructor | Mode |
|---|---|
| `ElCheConfig::nccl_sync()` | `NcclSync` |
| `ElCheConfig::nccl_cadence()` | `NcclCadence` (**default**) |
| `ElCheConfig::cpu_sync()` | `CpuSync` |
| `ElCheConfig::cpu_cadence()` | `CpuCadence` |
| `ElCheConfig::cpu_async()` | `CpuAsync` (see [A/B testing modes](#ab-testing-modes)) |

Or build the value directly with a struct literal:

```rust
let elche = ElCheConfig {
    mode: ElCheMode::NcclCadence,
    max_anchor: Some(20),
    overhead_target: Some(0.05),
    ..Default::default()
};
```

### `ElCheConfig` knobs

| Field / setter | Default | Description |
|---|---|---|
| `.mode(ElCheMode)` | `NcclCadence` | The (when × how) pair. `ElCheConfig::default()` returns `nccl_cadence()`. |
| `.anchor(n)` | 10 (Cadence/Async); 1 (Sync) | Initial anchor count. |
| `.min_anchor(n)` / `.max_anchor(n)` | `None` (auto) | Anchor bounds. |
| `.overhead_target(f)` | `0.05` | Upper bound on `sync_ms / max(compute_ms)` per anchor window. ElChe grows the anchor when overhead exceeds the target, shrinks it when overhead drops below half. **Cadence + Async modes only** - Sync modes fire the reduce per slow-rank step (every alive rank ≥1 step; see "What sync means" above) and ignore the anchor knob. See [the overhead auto-tune section](#overhead_target-anchor-auto-tune) below. |
| `.max_batch_diff(n)` | `None` | Cap on how far the fastest rank may lead the slowest. `Some(0)` = strict lockstep regardless of mode. |
| `.relax_up(bool)` | `false` | Allow ElChe to grow the anchor in `Phase::Stable` when convergence stays clean. |
| `.partition_ratios(Vec<f64>)` | auto | Static per-rank data split (e.g. `[0.7, 0.3]`). **Honored on `Sync` policy only**; Cadence/Async use progressive dispatch driven by ElChe and ignore the static ratios. For dynamic heterogeneous scheduling under those policies, ElChe's throughput-based auto-rebalancing is the intended path. |
| `.meta_controller(bool)` | `true` | LR-aware meta-controller - watches LR + anchor + divergence; nudges anchor down on sharp LR drops or sustained divergence. On by default (LR drops are always worth catching); opt out for unconditioned-trajectory instrumentation. |
| `.convergence_guard(g)` | `TrendGuard` at the EASGD-aware threshold | Divergence guardrail. `NoGuard`, `TrendGuard`, or `MsfGuard` (rate-based). The default threshold is keyed on param-adoption semantics: `0.05` for overwrite modes, `0.3` when `easgd_alpha` is set (elastic blending keeps a deliberate standing spread that a lower floor would read as permanent divergence). |
| `.easgd_alpha(α)` | `Some(0.5)` on `CpuAsync`; `None` elsewhere | EASGD elastic blend on the `CpuAsync` path (`0 < α ≤ 1.0`) - on by default there (full overwrite is the degenerate α=1.0 case). Ignored outside `CpuAsync`. |
| `.gamma(γ)` | `1.0` | Consensus allocation-weighting exponent applied when the outer optimizer / averaging weights ranks by work. `1.0` = pre-gamma (plain work-weighting). |
| `.bf16_wire(bool)` | `false` | Ship the CPU-averaging plane's model traffic as bfloat16: halves pinned snapshots, relay fold traffic, and wire payloads both directions. Averaging still accumulates in f32 (bf16 exists only at the wire/buffer boundary); control traffic, checkpoints, and the final trained weights stay exact f32. CPU averaging modes only - `.run()` errors loudly on NCCL modes. Must match across the cohort. |
| `.divergence_threshold(f)` | `None` | Legacy primitive feeding the default `TrendGuard` threshold when no explicit `convergence_guard` is set. Prefer `.convergence_guard(...)`. |
| `.no_divergence_guard()` | `false` | Disable the divergence guardrail entirely (overhead auto-tune drives cadence alone). Use only when the workload is known stable. |
| `.max_overshoot(n)` | `None` (auto) | Max batches a rank may run past its planned sync point before being held. **`CpuAsync` only**; ignored by Sync/Cadence. |

### Convergence guards

`ElCheConfig::convergence_guard(g)` plugs an implementation of the
`ConvergenceGuard` trait into the controller. After each averaging
round it returns one of `ConvergenceAction::{Stable, SuppressGrowth,
NudgeDown { factor }}`, which the coordinator uses to drive ElChe's
anchor.

| Guard | Behavior |
|---|---|
| `NoGuard` | Passive baseline - always `Stable`. Use for instrumented runs that want an unconditioned trajectory. |
| `TrendGuard::new(thresh)` | **Production default.** Three-rises-above-threshold rule on the per-rank `\|\|pre - post\|\| / \|\|post\|\|` ring buffer (last 5 events). Returns `SuppressGrowth` on persistent rising drift. |
| `MsfGuard::default().with_suppress(s, n).with_nudge(t, n, factor)` | Rate-based detector built on the across-event MSF proxy `λ_ema = EMA((1/k_max) * log(D_t / D_{t-1}))`. Soft + hard thresholds: sustained `λ_ema > suppress_threshold` → `SuppressGrowth`; sustained `λ_ema > nudge_threshold` → `NudgeDown` with `factor` (`0.5` halves the anchor). Opt-in. |

`TrendGuard` state (the divergence ring buffer) is part of
`ElCheState` and round-trips through `resume_from` - a resumed run
inherits the calibration trajectory. `MsfGuard`'s EMA + streak
counters re-warm from scratch across resume (by design, since the
across-event proxy is a derivative signal that recovers quickly).

### Guard authority over `overhead_target`

`overhead_target`'s anchor auto-tune is **proposed**, not committed,
inside `report_timing`. The convergence guard's verdict drives the
commit:

| Verdict | Effect |
|---|---|
| `Stable` | Commit the proposal - grow or shrink the anchor by the proposed amount. |
| `SuppressGrowth` | Drop a proposed grow; **apply** a proposed shrink (shrink is the safe direction when divergence is rising). |
| `NudgeDown { factor }` | Drop the proposal entirely; nudge supersedes by shrinking the current anchor by `factor`. |

This makes the convergence guard authoritative: rising weight-space
divergence vetoes anchor growth *before* it lands, rather than
catching up after `overhead_target` has already moved the anchor. The
two-sided trade-off - `overhead_target` proposes growth on throughput
pressure, guard vetoes when convergence pressure rises - runs through
one explicit commit/veto pipeline.

---

## `TrainerConfig<M>` - the umbrella

Every knob `Trainer::run` needs sits on `TrainerConfig`. The chained
`Trainer::builder()` API exposes the same setters; pick whichever
matches your call site.

```rust
let cfg = TrainerConfig::new(dataset)
    .batch_size(64)
    .num_epochs(50)
    .elche(ElCheConfig::nccl_cadence().relax_up(true))
    .max_grad_norm(5.0)
    .checkpoint_every(5)
    .save_path("ckpts/run43")
    .resume_from("ckpts/run42.fdl")
    // .epoch_callback_policy(EpochCallbackPolicy::Fastest)  // default - pin a specific rank with EpochCallbackPolicy::Rank(n)
    .checkpoint_fn(Arc::new(|epoch, model| {
        model.save_checkpoint(&format!("ckpts/run43-ep{epoch}.fdl"))
    }))
    .eval_dataset(test_set)
    .eval_fn(Arc::new(|model, input, target| {
        let pred = model.forward(&Variable::new(input.clone(), false))?;
        // ... return f64
        Ok(0.0)
    }))
    .eval_result_fn(Arc::new(|epoch, val| {
        eprintln!("eval epoch={epoch} value={val}");
    }))
    .metrics_fn(Arc::new(|m| {
        eprintln!("epoch={} loss={:.4}", m.epoch, m.avg_loss);
        Ok(())
    }));
```

| Setter | Type | Notes |
|---|---|---|
| `.batch_size(n)` | `usize` | Per-rank batch size. |
| `.num_epochs(n)` | `usize` | Total epochs. |
| `.elche(cfg)` | `ElCheConfig` | DDP cadence + backend + tuning. |
| `.max_grad_norm(f)` | `f64` | Per-rank gradient clip applied before AllReduce. Fused kernel. |
| `.checkpoint_every(n)` | `usize` | Save a checkpoint every `n` epochs/aggregations. |
| `.save_path(p)` | `String` | Stem for checkpoint bundles (writes `<stem>.fdl` + `<stem>.meta.json`). |
| `.resume_from(p)` | `String` | Load bundle at start; restores params, buffers, optimizer, ElCheState. |
| `.checkpoint_fn(f)` | `CheckpointFn<M>` | Called on the elected callback rank with `(epoch, &M)`. |
| `.epoch_fn(f)` | `EpochFn<M>` | Per-epoch worker callback (`(epoch, &mut GpuWorker<M>)`). |
| `.metrics_fn(f)` | `MetricsFn` | Host-side per-epoch callback (`&EpochMetrics`). |
| `.scheduler_fn(f)` | `SchedulerFn` | Per-worker LR scheduler factory. |
| `.sample_dataset(ds)` | `impl DataSet` | Per-sample alternative to `.dataset()`: implement `get(index)`, the framework batches, caches, and stages. `TrainerConfig::from_dataset(ds)` is the config-bag twin. |
| `.eval_dataset(ds)` | `Arc<dyn BatchDataSet>` | Held-out data for evaluation. |
| `.eval_fn(f)` | `EvalFn<M>` | Receives `(&M, &Tensor, &Tensor)`, returns `Result<f64>`. |
| `.eval_result_fn(f)` | `EvalResultFn` | Controller-side `(epoch, scalar)` sink. |
| `.epoch_callback_policy(p)` | `EpochCallbackPolicy` | `Fastest` (default) or `Rank(n)`. |
| `.outer_optimizer(factory)` | `Fn() -> Box<dyn OuterOptimizer>` | Outer-loop optimizer on the consensus (SlowMo / DiLoCo). Default = plain work-weighted averaging (`OuterAvg`). See [Outer optimizer](#outer-optimizer---slowmo--diloco). |
| `.checkpoint_at_epoch(n)` | `usize` | One-shot coverage-granular checkpoint at the epoch any rank first reaches (progressive modes). Pairs with `.save_path`. |
| `.eval_every(n)` | `usize` | Fire `eval_fn` every `n` epochs (`0` disables). The chained `DdpBuilder::eval_every` takes an `EvalCadence` instead. |
| `.reports_per_epoch(n)` | `usize` | Emit up to `n` sub-epoch monitor reports per epoch, at reduce boundaries (`0` = off, the default). Fills the curve *between* epoch points — see [Sub-epoch reports](#sub-epoch-reports---reports_per_epoch). |
| `.record_log(dir, max_bytes)` | `(String, u64)` | Persist the monitor record stream as append-only JSONL under `dir`, one drop-oldest ring per node capped at `max_bytes` (`0` = default). Off by default — see [Persisting the record stream](#persisting-the-record-stream---record_log). |
| `.save_dashboard(path)` | `String` | Write the run's dashboard as one self-contained HTML file at teardown (the full portal, no server, no sibling files). Off by default — see [Saving the dashboard](#saving-the-dashboard---save_dashboard). |
| `.dashboard_theme(theme)` | `String` | Pin the saved dashboard's theme (`"light"` / `"dark"` / `"auto"`). Unset, a saved page follows the reader's `prefers-color-scheme`. `"light"` is the publication setting. |
| `.scalar_reduction(key, r)` | `(String, Reduction)` | Declare how a **user scalar** rolls up across ranks (`Sum` / `Max` / `Min` / `Mean` / `Last`). Repeatable. Non-core keys default to `Mean`, which is wrong for a count — see [Declaring how a scalar rolls up](#declaring-how-a-scalar-rolls-up---scalar_reduction). |
| `.timeline(t)` | `Arc<Timeline>` | Inject DDP events into a profiler stream. |
| `.with_vram_pool(b)` | `bool` | Device-resident sample pool on each rank (default `true`; `FLODL_VRAM_POOL=off` is the runtime kill-switch). |
| `.with_vram_max_usage(f)` | `f64` | Fraction of total VRAM each rank's data plane (prefetch channel + sample pool) may use. Default `0.90`, clamped to `[0.50, 0.99]` - same knob as the solo loader's `vram_max_usage`. |
| `.with_ram_max_usage(f)` | `f64` | Fraction of available host RAM each rank's staging tiers may retain; co-hosted ranks split it in proportion to their schedule share. Default `0.50`, clamped to `[0.0, 0.90]`; `0.0` disables staging retention. Same knob as the solo loader's `ram_max_usage`. |
| `.with_sample_cache(b)` | `bool` | Pinned RAM sample retention in each rank's staging tier. `false` pins the retained cache at zero - the flow window keeps the whole staging share, nothing persists across epochs. Default `true` - same knob as the solo loader's `sample_cache`. |
| `.with_disk_stage(gb)` | `u64` | Local-disk overflow tier under each rank's sample cache, in GB: samples the RAM budget declines spill to an ephemeral per-rank pack file and re-read at local-disk speed instead of source speed. Default `0` (off) - same knob as the solo loader's `disk_stage`. Pair with `.with_disk_stage_dir(path)` to point at a fast local drive. |
| `.with_augment(k)` | `usize` | Views per sample per epoch: the schedule becomes `len()*k` picks, sharded and balanced exactly like samples. Data variation comes from the transform. |
| `.with_transform(f)` | closure | Deterministic delivery transform, keyed by `PickKey { sample, repeat, epoch, seed }` per row; runs on each rank after device transfer. The chained `DdpBuilder` twins are `.augment(k)` / `.transform(f)`. See the [data-loading tutorial](tutorials/13-data-loading.md#augmentation-repeated-picks--a-keyed-transform). |
| `.cluster(c)` | `FullCluster` | Programmatic cluster topology (overrides any active overlay). |

`TrainerConfig::cluster(full)` is the seam for programmatic
multi-host launches (see [Programmatic clusters](#programmatic-clusters---clusterbuilder)
below).

---

## Host-side callbacks: `metrics_fn` / `eval_fn`

`Trainer::builder().run()?.join()?` is the canonical "just train" shape,
but per-epoch logging, monitor wiring, and held-out evaluation all want
a host-side callback that fires once per epoch with the aggregated
metrics.

### `metrics_fn`

```rust
let handle = Trainer::builder(model_factory, optim_factory, train_step)
    .dataset(dataset)
    .batch_size(64)
    .num_epochs(100)
    .metrics_fn(Arc::new(|m: &EpochMetrics| {
        eprintln!(
            "epoch={} loss={:.4} acc={:.3} {:.0}ms",
            m.epoch, m.avg_loss,
            m.scalars.get("accuracy").copied().unwrap_or(0.0),
            m.epoch_ms,
        );
        Ok(())
    }))
    .run()?;

handle.join()?;
```

Fires once per epoch on the host thread, after all ranks have reported.
Composes with the polling API (`handle.next_metrics()` /
`handle.poll_metrics()`) - the same `EpochMetrics` reaches both. Callback
errors are logged to stderr; training continues.

Transparent across tiers: fires identically on single-GPU, single-host
multi-GPU, and multi-host clusters.

### `eval_fn` + `eval_dataset` + `eval_result_fn`

```rust
.eval_dataset(test_set)
.eval_fn(Arc::new(|model, input, target| {
    let pred = model.forward(&Variable::new(input.clone(), false))?;
    let acc  = pred.argmax(-1, false)?.eq_tensor(target)?.sum()?.item::<f64>()?
             / target.shape()[0] as f64;
    Ok(acc)
}))
.eval_result_fn(Arc::new(|epoch, value| {
    eprintln!("[eval] epoch={epoch} acc={value:.4}");
}))
```

The coordinator dispatches `eval_fn` per-epoch on the elected callback
rank, and forwards the returned scalar to `eval_result_fn` on the host.

### `EpochCallbackPolicy`

Controls which rank executes per-epoch callbacks (`checkpoint_fn`,
`epoch_fn`, `eval_fn`).

| Variant | Behavior |
|---|---|
| `Rank(n)` | Pin to a fixed **global rank** in `[0, world_size)`. Ranks are assigned sequentially by worker order in the cluster topology (worker 0 owns ranks `[0..N0)`, worker 1 owns `[N0..N0+N1)`, etc.). On a 4-rank cluster across two 2-GPU hosts, `Rank(0)` fires on the first rank of the first worker host, `Rank(3)` on the last rank of the last host. Loud-errors if `n >= world_size`. |
| `Fastest` (**default**) | Cost-aware: pick the global rank with the lowest `smoothed_ms_per_batch`. On heterogeneous rigs the fastest rank has the most idle time at the sync barrier, so eval / save runs as free compute. Sticky within a run; re-resolves only on rank death. On a single-GPU run the only rank trivially satisfies "fastest". |

### `EpochMetrics` fields

| Field | Type | Description |
|---|---|---|
| `epoch` | `usize` | 0-based. |
| `avg_loss` | `f64` | Loss averaged across all ranks. |
| `epoch_ms` | `f64` | Wall time for the epoch (slowest rank). |
| `scalars` | `HashMap<String, f64>` | Aggregated custom scalars (`record_scalar(...)` inside `train_fn`). |
| `per_rank` | `Vec<HashMap<String, f64>>` | Per-rank custom scalars. |
| `per_rank_throughput` | `Vec<f64>` | Batches per second per rank. |
| `per_rank_batch_share` | `Vec<f64>` | Fraction of total batches handled per rank. |
| `device_indices` | `Vec<u8>` | CUDA device index for each rank. |

---

## Sub-epoch reports - `reports_per_epoch`

`metrics_fn` and the dashboard are driven by the **per-epoch** feed, which
is one data point per pass over the dataset. For a long epoch — and
decisively for **single-epoch (one-pass LLM) training** — that is a curve
with one point. `reports_per_epoch(n)` adds a *second*, finer feed:

```rust
Trainer::builder(model_factory, optim_factory, train_step)
    .dataset(dataset)
    .num_epochs(1)              // one pass over a token corpus
    .reports_per_epoch(20)      // ~20 loss points across the run
    .run()?;
```

- **Cadence.** A report fires at the first **reduce boundary** past each
  `epoch_work / n` slice of realized work (`epoch_work` = steps per epoch,
  known ahead from the dataset). Early small windows accumulate across
  several reduces; the rate settles toward one report per window as the
  cadence window grows. Reports always land *at* a sync boundary, never
  mid-window, and they ride the existing step clock — reporting observes
  the cadence, it never gates or delays it.
- **At most `n` per epoch.** An epoch that ends with residual work below
  the final threshold simply reports fewer times. That is not a gap: the
  epoch-boundary point comes from the per-epoch feed, so the two feeds
  compose — sub-epoch fills the interior, per-epoch marks the boundary.
- **Content.** Each report is a path-keyed node tree (`root` → host → rank,
  collapsing to `root` → rank on a single host) carrying per-rank `loss`,
  `throughput`, `compute_only_ms` and `batch_share`, with each interior
  node aggregating **its direct children only**. Cross-rank means are
  weighted by realized work, so a hierarchical rollup equals the flat one.
  A metric a rank did not measure this window is **absent**, never zeroed.
- **Cost.** Zero extra wire traffic — the per-batch loss already reaches
  the controller on the existing timing frames. Disabled (the default) the
  whole path is one `Option` check per reduce.
- **Scope.** Cluster / auto-promoted multi-GPU runs (the controller path).
  The per-epoch `metrics_fn` contract is untouched. In `ddp-bench`:
  `--reports-per-epoch N`.

Aggregation vocabulary (`Reduction`, the node record schema) lives in
`flodl::monitor::record`; the cadence scheduler in
`flodl::monitor::cadence`.

---

## Persisting the record stream - `record_log`

Reports are live-only unless you ask for them on disk:

```rust
Trainer::builder(model_factory, optim_factory, train_step)
    .dataset(dataset)
    .reports_per_epoch(20)
    .record_log("runs/exp1/records", 0)   // 0 = default per-node cap
    .run()?;
```

A record's `path` **is** its filesystem path, so the run leaves a tree
that mirrors the cluster:

```text
runs/exp1/records/
  root.log                      # cohort aggregate, one JSON per line
  root/exa-cuda.log             # host aggregate
  root/exa-cuda/rank0.log       # rank leaf, raw values
  root/flodl-pascal.log
  root/flodl-pascal/rank1.log
  root/flodl-pascal/rank2.log
```

- **JSONL, one record per line.** Each line is a standalone JSON object
  carrying `ts` / `sev` / `path`, so the files ingest into fluentd / GCP
  Cloud Logging as-is, and `jq` works on them directly.
- **Bounded, drop-oldest.** Each node's log is a ring capped at
  `max_bytes` (default 32 MiB): when the active segment fills it rotates
  and the oldest is dropped. A long run **cannot fill the disk** — the
  whole tree bounds to `nodes × max_bytes`.
- **Never fails training.** Every I/O error here is swallowed and warned
  once. A full or read-only disk costs you observability, nothing else.
- **Tail-read resume.** Every `node` record is an absolute snapshot, so
  catching up is reading the last N lines
  (`RecordLog::tail(path, n)`) — no index, no checkpoint replay.

Files stay plain JSONL while live (gzip cannot be appended to or cheaply
tailed). In a containerized run, point `dir` at a **mounted** path — a
container-local path is written inside the container and vanishes with it.

### Alerts in the same stream

Metrics records are `kind: "node"`. The stream also carries
`kind: "event"` records — the alert lane — written to the **same** files,
because an alert's `path` is the node it happened to:

```json
{"v":1,"ts":1737766000000,"sev":"critical","path":"root/flodl-pascal/rank2",
 "kind":"event","class":"rank_lost","detail":"rank 2 declared dead — heartbeat stale (>30s)","count":1}
```

| `class`        | `sev`    | raised when                                                 |
|----------------|----------|-------------------------------------------------------------|
| `rank_lost`    | critical | a rank was declared dead (heartbeat staleness or child exit) |
| `control_drop` | critical | a control broadcast did not reach every live rank            |
| `drift`        | warn     | the convergence guard had to correct the anchor              |
| `overflow`     | warn     | alerts were dropped by the live cap (never silent)           |

So `jq 'select(.kind=="event")' runs/exp1/records/root/*/*.log` is the
run's incident list, and a rank's own log holds its metrics *and* the
alert that ended it, in order. Alerts are always printed to stdout too —
you do not need `-v`, and you do not need `record_log`, to see them.

The lane is bounded: repeats of the same `(class, path)` within 10s
collapse into one record (the first occurrence is never delayed; the
absorbed repeats ride the `count` of the next one, so counts sum to the
true total), and at most 200 live entries are retained. There is no knob
— an alert stream you have to tune is one you cannot trust.

### Reading the stream live, by path

`record_log` is the stream's history on disk. Its live twin is served on
the dashboard port, addressed by the same paths:

```
GET /paths                       GET /history?path=root&n=200
GET /node?path=root/exa          GET /stream?path=root/exa    (SSE)
```

`/node` answers with one level — that node plus its **direct children**,
so a query costs `O(children)` and not `O(cluster)` at any depth — and
`/history` returns exactly what `/stream` would have sent, so
read-then-subscribe has no seam. Details:
[monitor tutorial → Querying the run by path](tutorials/09-monitor.md#querying-the-run-by-path).

## Saving the dashboard - `save_dashboard`

`record_log` gives you the stream as JSONL. `save_dashboard` gives you the
**page**: one self-contained HTML file carrying the whole portal — every level
browsable, both metric cadences interleaved, the model graph SVG and the
hyperparameters inline. Open it in a browser with no server and no sibling
files.

```rust
Trainer::builder(model, opt, step)
    .save_dashboard("runs/exp1/dashboard.html")
```

`ddp-bench` exposes it as `--save-dashboard`, which writes
`<run_dir>/dashboard.html` beside `timeline.html`.

Three properties make it safe to leave on:

- **It needs no dashboard port.** Persisting a dashboard does not require
  serving one, so a headless cluster run produces the same artifact.
- **It cannot grow without bound.** The record plane is a ring, so a longer run
  shortens the archive's *horizon* rather than enlarging its *file* — which is
  what keeps it attachable to a ticket.
- **It is written at teardown**, after the record stream has drained, so it
  captures the end of the run rather than whatever had been flushed early.

Ask for it through the **builder**, not through your own `Monitor`: on a cluster
run `Monitor::serve` returns early and the launcher's sink owns the server and
the records, so `monitor.save_html(...)` would write a page with no levels.

### Theme

The saved page follows the reader's OS by default, exactly as the live
dashboard does, and carries a toggle. Pin it when the artifact is headed
somewhere with a fixed look — a figure in a paper should not change appearance
with the reviewer's desktop:

```rust
    .save_dashboard("runs/exp1/dashboard.html")
    .dashboard_theme("light")
```

Nothing is locked in at training time: every saved page exposes the choice as a
single line near the top (`const ARCHIVE_THEME=null;`), so re-theming an
artifact you already have is one edit, and the reader's own toggle still
overrides whatever is pinned.

## Declaring how a scalar rolls up - `scalar_reduction`

Framework metrics have authoritative reductions (`loss` is a work-weighted
mean, `throughput` sums, `data_starve` takes the worst rank). **Your** scalars
cannot be guessed, so they default to `Mean` — right for a rate or an accuracy,
wrong for a count or an extremum:

```rust
Trainer::builder(model, opt, step)
    .scalar_reduction("tokens_seen", Reduction::Sum)
    .scalar_reduction("peak_mem_gb", Reduction::Max)
```

This matters more than a wrong number usually would, because the portal
**states** the reduction in its legend: an undeclared count renders as
`tokens_seen (mean)`, asserting something false rather than merely being off.

Declarations reach every consumer automatically — they ride the record stream's
`meta` record, which each subscriber receives ahead of any data record, so a
viewer can never interpret a metric without knowing how it was rolled up. Core
keys ignore any declaration here.

The dashboard at that same port **is** these paths, rendered: one view
repeated per level, a breadcrumb for navigation, and a drill-down that
re-subscribes to the child path. It watches the level you are on rather
than the whole cluster, so cost does not grow with rank count. See
[monitor tutorial → One view, repeated per level](tutorials/09-monitor.md#one-view-repeated-per-level).

---

## CUDA-free GPU detection - `flodl::sys::detect_gpus`

`detect_gpus() -> Vec<GpuInfo>` shells out to `nvidia-smi` and returns
per-device `(index, name, sm_version, vram_bytes)` without loading
libtorch. Honors `CUDA_VISIBLE_DEVICES`, so the result matches the view
the auto-promote path and child processes will see.

```rust
use flodl::sys::detect_gpus;

let gpus = detect_gpus();
for g in &gpus {
    eprintln!("GPU {}: {} (sm_{}, {} MB)",
        g.index, g.name, g.sm_version, g.vram_bytes / 1_000_000);
}

// Use the count for partition planning, but do NOT instantiate
// CUDA tensors here.
let world_size = gpus.len();
```

This is the canonical pre-`Trainer::run` GPU query. The previous habit
of calling `flodl::tensor::cuda_device_count()` from `main()`
initializes libtorch's CUDA context in the launcher process; that
context then poisons spawned children on heterogeneous-GPU rigs.
`detect_gpus` does not touch CUDA.

---

## Auto-promote: single host, N GPUs

When `Trainer::builder().run()` (or `Trainer::run`) fires on a host
where `detect_gpus() >= 2` and no cluster overlay is set, the framework
synthesizes a single-host cluster covering every visible CUDA device
and fans out one process per rank. The user's binary process becomes
the launcher; rank children run training.

```rust
// On a 2× GPU host: this auto-promotes to a 2-rank process-per-rank
// run. No code change vs the single-GPU shape above.
let handle = Trainer::builder(model_factory, optim_factory, train_step)
    .dataset(dataset)
    .batch_size(64)
    .num_epochs(50)
    .run()?;
handle.join()?;
```

Auto-promote is `cfg(not(test))`-gated for flodl's own test suite (so
`Ddp::wrap` keeps driving the thread-based multi-GPU tests in-process).
External crates that want to scope down to a single rank in tests can
set `CUDA_VISIBLE_DEVICES=0`.

---

## Programmatic clusters - `ClusterBuilder`

For tests and binaries that want to launch a multi-host cluster from
inside `main()` without depending on a yml on disk:

```rust
use flodl::ClusterBuilder;

let cluster = ClusterBuilder::new()
    .controller("controller.example.com")
        .port(1337)
        .path("/opt/flodl")
    .done()
    .host("worker-a")
        .ranks([0, 1])
        .devices([0, 1])
        .nccl_socket_ifname("enp1s0")
        .path("/opt/flodl")
        .ssh("worker-a.example.com")
        .ssh_port(22)
        .ssh_user("ubuntu")
        .ssh_identity_file("/home/me/.ssh/cluster_key")
    .done()
    .host("worker-b")
        .ranks([2])
        .devices([0])
        .nccl_socket_ifname("enp1s0")
        .path("/srv/flodl")
    .done()
    .build()?;

let cfg = TrainerConfig::new(dataset)
    .batch_size(64)
    .num_epochs(50)
    .elche(ElCheConfig::nccl_cadence())
    .cluster(cluster);

Trainer::run(model_factory, optim_factory, train_step, cfg)?.join()?;
```

`ClusterBuilder` mirrors `fdl.cluster.yml` 1:1 - same fields, same
validation, same launcher contract. `controller(...)` and `host(...)`
are sibling sub-builders, matching the YAML's `controller:` /
`workers[]:` shape. A `FullCluster` reaches the launcher the same way
an overlay-driven cluster does (via `FLODL_INTERNAL_FULL_CLUSTER_JSON`), so
`Trainer::run` accepts both.

### `ClusterBuilder::all_local_gpus()`

Single-host convenience:

```rust
let cluster = ClusterBuilder::all_local_gpus()?;
// One worker, every visible CUDA device, loopback controller.
```

This is what auto-promote synthesizes internally; expose it explicitly
when you want to drive the same shape from a test.

---

## Resume + checkpoints

`Trainer::builder(...).resume_from(stem)` (or `TrainerConfig::resume_from`
on the config-bag entry) loads a checkpoint bundle and continues training. The bundle is three files:

| File | Contents |
|---|---|
| `<stem>.fdl` | Model parameters + buffers + optimizer state. |
| `<stem>.meta.json` | `CheckpointMeta`: `ElCheState` (phase, calibration_count, anchor, partition_ratios, ring buffer) + `SaveReason`. |
| `<stem>.config.json` | (optional) Source config sidecar - `flodl-hf` writes this on export. |

```rust
TrainerConfig::new(dataset)
    .batch_size(64)
    .num_epochs(100)
    .save_path("ckpts/run43")
    .resume_from("ckpts/run42")     // loads ckpts/run42.{fdl,meta.json}
    .checkpoint_every(5);
```

The controller writes meta atomically alongside the model + optimizer
files. Compatible with `.meta.json` from any prior flodl run.

### `SaveReason`

| Variant | Trigger |
|---|---|
| `Checkpoint` | A mid-run checkpoint taken atomically at a reduce (`checkpoint_every` / `checkpoint_at_epoch`); training continues. |
| `GracefulShutdown` | Normal cluster shutdown after reaching end-of-training. |
| `MaxFailureExceeded` | User-configured `max_failure` threshold was breached. |
| `SingleSurvivor` | NCCL cohort dropped below 2 ranks; the lone survivor saves before exiting (NCCL needs world_size >= 2). |
| `AllRanksLost` | CPU cohort lost its last survivor. |
| `ReduceStall` | A reduce cycle stalled past its hard ceiling with the cohort still alive (scheduler wedge); save + shut down rather than hang. |

---

## Elastic membership

Ranks can die without aborting the run. The controller owns the
lifecycle; workers just report and follow: a dead rank is evicted for
the remainder of the run and its unprocessed work is redistributed
across survivors.

Membership only ever **shrinks**. The world is formed once, at the join
window (see [Dial-in membership](#dial-in-membership-the-join-window));
after that, neither a new rank nor a previously dead one can join the
cohort. Elastic scale-up (mid-training join) is designed but not yet
implemented - see `.design/hierarchical-elastic-ddp.md` for the
direction and its consistency invariant.

### What happens when a rank dies

1. **Death detection** - the launcher's child-exit report reaches the
   controller within milliseconds of the process dying; heartbeat
   staleness (30s) is the backstop for silent hangs. Either path
   transitions the rank to `Dead` in per-rank state and elastically
   renormalizes `partition_ratios` across survivors. (A rank that
   *completes* cleanly announces `Exiting` instead - the
   clean-completion latch - so a finished rank is never mistaken for a
   death; an error exit never sends it, so a death is never masked as
   completion.)
2. **Lone NCCL survivor** - short-circuits the wait and exits
   immediately rather than blocking on a dead-quorum AllReduce.
3. **`max_failure` threshold** - when survivor count drops below this,
   the cluster aborts cleanly. Coordinator drives a final
   `ShutdownWithSave` checkpoint through whichever rank still has the
   freshest state, then signals every survivor to exit.
4. **NCCL rendezvous-timeout retry** - if `ncclCommInitRank` doesn't
   quorum within the timeout, the coordinator picks the largest
   contiguous survivor subset, rebuilds the comm, and retries. Used at
   run start and after mid-run rank death.

### Controller-driven checkpoint retry / role failover

A save failure on the elected callback rank does not poison the run.
The coordinator picks a new callback rank from survivors (cost-aware:
lowest `smoothed_ms_per_batch` first, sticky within a run), re-issues
the save, and resumes. Failed callbacks are time-excluded from
rank-cost accounting so retry latency doesn't bias the next dispatch
decision.

---

## Multi-host clusters

A cluster spans hosts via `fdl.cluster.yml` (deployment) or
`ClusterBuilder` (programmatic). The orchestrator host fdl-cli runs on
is the **controller** and is never a NCCL rank itself; every
rank-carrying host lives under **workers**.

### `fdl.cluster.yml` schema

```yaml
cluster:
  controller:
    host: 192.168.122.1           # controller bind address
    port: 1337                    # the single controller port (default 1337)
    path: /opt/flodl              # controller's view of the shared project root
    # docker: cuda                # optional pre-flight build service
    # arch: precompiled/cu128     # optional libtorch variant for pre-flight build
    # join:                       # membership-window overrides (see below)
    #   min_rank_start: 2
    #   join_timeout: 300
    #   target_ranks: 4
    #   max_join_timeout: 600
    #   open_admission: false

  workers:
    - host: node-a                # worker identifier; default SSH target
      local_devices: [0]          # 1 device -> 1 rank
      nccl_socket_ifname: virbr0
      path: /opt/flodl
      arch: precompiled/cu128     # libtorch variant under <path>/libtorch/
      docker: cuda                # optional: training runs in this compose service

    - host: node-b
      ssh:                        # optional SSH sub-block
        target: node-b
        port: 2222
        user: ubuntu
        identity_file: /home/me/.ssh/cluster_key
        options:
          - ProxyJump=bastion
          - StrictHostKeyChecking=no
      # tunnel: true              # route training traffic through the SSH
      #                           # session (CPU ElChe modes only; see below)
      local_devices: all          # probed at dispatch via SSH+nvidia-smi
      nccl_socket_ifname: enp1s0
      path: /srv/flodl
      arch: builds/sm61-sm120     # different variant per worker is fine

  # Cluster-scope env vars (apply to every rank child)
  # env:
  #   NCCL_DEBUG: INFO
```

Conventions:

- One process per rank; each worker owns one rank per visible CUDA
  device. Global ranks are assigned sequentially by worker order:
  worker 0 owns `[0..N0)`, worker 1 owns `[N0..N0+N1)`, etc.
- `local_devices: all` probes the host at dispatch time via SSH +
  `nvidia-smi`. Explicit lists carry their own count.
- `nccl_socket_ifname:` is required on every worker when the cluster
  spans multiple hosts.
- `path:` is the project checkout dir on this host (heterogeneous
  mounts are fine - `/opt/flodl` on one host, `/srv/flodl` on another).
- `arch:` is the libtorch variant subpath under `<path>/libtorch/` on
  this host. For heterogeneous rigs, each worker can select a different
  variant (e.g. one host on `precompiled/cu128`, another on
  `builds/sm61-sm120`); the convention path stays stable while the
  variant differs per host.
- `docker:` (optional) names the compose service for training on this
  host. Per-host: mixed deployments (controller in Docker, worker
  bare-metal) are common.
- `tunnel:` (optional) routes this worker's training traffic through
  its fan-out SSH session instead of a direct TCP connection - see
  below.

### Dial-in membership: the join window

Workers **join** a run; the controller admits them. At launch the
controller opens a join window on its port; every worker - fan-out-
managed and self-deployed alike - dials in with a hello (host name,
GPU inventory, libtorch variant, dataset signature) and is assigned
its global ranks **in admission order** (contiguous by construction).
When the window closes, the world freezes: `world_size` is whatever
actually joined, and all coordination infrastructure (ElChe schedule,
heartbeats, rendezvous) is sized to that world.

`fdl @cluster <cmd>` fan-out is sugar over this protocol: it starts
one worker agent per host over SSH, and those agents dial back in like
any worker would. The defaults make fan-out behave exactly like a
fixed topology - quorum and early-close target both default to the
configured capacity, so the window closes the instant every configured
rank is in (zero added latency) and the run cannot start below full
strength.

Override via `controller.join:` to allow degraded starts or to hold
the window open for extra dial-in workers:

| Knob | Meaning | Default |
|---|---|---|
| `min_rank_start` | Quorum in ranks; the run cannot start below it. | configured capacity |
| `join_timeout` | Window in seconds. Quorum reached early does NOT close it - late workers within the window still join. | 300 |
| `target_ranks` | The window closes the moment this many ranks are in. Raise it above capacity to wait for self-deployed workers. | configured capacity |
| `max_join_timeout` | Hard cap in seconds; quorum still unmet when it expires fails the run loudly. | 600 (or the window length when set higher) |
| `open_admission` | Accept joins without the pre-shared session salt on a non-loopback bind (loudly warned). | false |
| `discovery` | Roster-free formation: the window alone defines the world, so `workers:` may be empty (walk-ins self-register). Requires an explicit `min_rank_start`; the window closes only on `target_ranks` or expiry. | false |
| `token` | Pre-shared session salt, hex (32 chars / 16 bytes), replacing the per-run generated salt so a fleet-create-injected credential can be presented by walk-ins. Forces credential-checked admission even behind sshd; contradicts `open_admission: true` (loud error). | generated per run |
| `tunnel_only` | Discovery-only: bind the controller loopback-only so walk-ins must arrive through sshd forwards (reachability = authentication). Requires a CPU averaging mode. | false |

Admission is authenticated by the join frames' HMAC key: fan-out
agents receive the per-run session salt through their SSH session, so
a peer without it cannot join. A **loopback** bind (every remote
worker tunneled) is open by construction - the only path to the port
is through sshd, so reachability itself is the authentication, and the
salt is handed out in the accept reply. `open_admission: true` extends
that hand-out to a network bind: any peer that can reach the port can
then join (and therefore influence) the run, which is why flodl warns
loudly - sound only on a fully trusted segment.

A **self-deployed worker** needs nothing but the controller address: a
process started on any GPU host with `FLODL_INTERNAL_AGENT_JSON` set
to the hex-encoded spec `{"host": "...", "controller_host": "...",
"controller_port": 1337}` (see `AgentSpec` in the API docs) resolves
its own GPUs, joins, receives the formed-world artifacts, and spawns
its relay and rank children - the training code is byte-identical to
the fan-out path. Pair it with `target_ranks` above the configured
capacity (or a bare-bones one-host config) so the window waits for it.

`discovery: true` takes that shape to its limit: no roster at all. The
controller opens the window from its bind address plus the join
credential (`token`, or sshd reachability under `tunnel_only`), holds
it for `join_timeout` seconds, and the world is whoever walked in -
the cloud shape, where worker addresses do not exist before the VMs
boot. Fan-out and discovery compose: an enumerated rig fans out as
usual while cloud legs self-register into the same window.

One contract for user binaries: `Trainer::run` dispatches the cluster
roles (agent, relay, rank) internally, so a binary that goes straight
to `Trainer::run` needs nothing. But a binary that **gates before**
`Trainer::run` (checks GPU counts, parses modes, validates datasets,
and possibly exits) must short-circuit the internal worker roles first
- otherwise the worker agent falls into the gate on the remote host
(seeing ONE host of a multi-host world) and exits without ever
joining, and the window idles to its hard cap:

```rust
fn main() {
    // Worker-role short-circuit BEFORE any gating/exit logic. Runs the
    // relay/agent role and exits when this process is one; returns
    // immediately otherwise.
    flodl::distributed::launcher::exit_if_worker_role();
    // ... your pre-run gating, then Trainer::run(...)
}
```

### One port, and tunneled workers

All cross-host traffic (membership join, NCCL bootstrap rendezvous,
CPU-reduce data, coordinator control) accepts on the single
`controller.port`; connections identify their channel with a 4-byte
magic. The same port answers plain HTTP GETs with the run's membership
state (see `fdl status` below). The traffic is HMAC-authenticated but
NOT encrypted, and flodl warns loudly whenever a cleartext channel
touches a peer outside private address space (loopback / RFC1918 /
link-local / CGNAT-shared).

`tunnel: true` on a worker is the supported way to leave the private
network: the launcher adds a remote forward
(`-R 127.0.0.1:<port>:127.0.0.1:<port>`) to that host's relay SSH
session and points the host at `127.0.0.1:<port>` - its loopback end
of the tunnel. Everything the host sends then rides the (encrypted)
SSH session; the fan-out credential is the only credential involved.
Two constraints, both validated loudly at launch:

- **CPU ElChe modes only** (`cpu_sync` / `cpu_cadence` / `cpu_async`).
  NCCL's data plane is peer-to-peer between GPU hosts and cannot ride
  a controller tunnel; CPU-mode traffic all flows through the per-host
  relay's single upstream connection, which is exactly what the
  forward carries.
- **Remote hosts only** - the launcher host already reaches the
  controller over loopback.

When every remote worker sets `tunnel: true`, the controller binds
loopback only: the training port is then unreachable except through
sshd on the controller host.

### Activating the overlay

Three equivalent forms (a command-line selector overrides `FDL_ENV`):

```bash
fdl @cluster <cmd>            # @ sigil (pre-command position only)
fdl --env cluster <cmd>       # explicit flag (position-independent)
FDL_ENV=cluster fdl <cmd>     # environment variable
```

`fdl @cluster <cmd>` fans out to every worker via SSH, pre-builds the
target binary per-host with the right libtorch variant, dispatches the
remote rank children, and tears them down on parent exit.

See [CLI reference](cli.md#fdl-cluster) for the full command surface.

### Per-case libtorch (heterogeneous rigs)

One libtorch checkout can support multiple per-host variants via
`libtorch/.active.<case>` pointer files. The `FDL_LIBTORCH_CASE=<case>`
env var selects which pointer to read; cluster.yml's per-host `arch:`
can point directly at a case file (`…/libtorch/.active.<case>`) so
cluster fan-out resolves each host's variant correctly.

Single-host setups keep using bare `.active`.

### NCCL version skew

When one host's libtorch ships NCCL 2.27.x and another's ships 2.26.x,
NCCL refuses handshake across the major.minor skew. Build a matching
libnccl on the easier side:

```bash
fdl nccl build                  # auto-detects target NCCL tag + local archs
```

Wire it in via the worker's `env: LD_PRELOAD:` block in cluster.yml.
See [CLI reference](cli.md#fdl-nccl-build) for full options.

### Readiness gate - `fdl probe`

Before launching, audit the cluster:

```bash
fdl probe                       # single-host: GPU + libtorch + NCCL + shared-data path
fdl @cluster probe               # cluster: SSHes each worker, aggregates
fdl @cluster probe --json        # machine-readable for CI gating
```

Errors loudly on misconfig; the green path is silent enough to use as
a CI smoke test. Returns non-zero on errors; zero on green or
warnings-only. See [CLI reference](cli.md#fdl-probe) for the full
field listing.

### Live run status - `fdl status`

While a run is up, the controller port answers plain HTTP GETs with
the run's membership state as `state.json` - lifecycle phase
(`waiting` / `forming` / `training` / `done` / `failed`), who has
joined with what hardware, and the join-window countdowns while it is
still open:

```bash
fdl @cluster status              # pretty summary from the overlay's controller
fdl status --addr host[:port]    # explicit target (all a self-deployed
                                 # worker's operator needs)
fdl @cluster status --json       # raw state.json for scripts
curl http://<controller>:1337/state.json   # no fdl required
```

```text
cluster run @ 192.168.122.1:1337 - training
  ranks: 3 joined across 2 host(s)   (quorum 3, target 3)
  hosts:
    node-a  ranks [0]     1x RTX 5060 Ti   libtorch precompiled/cu128  joined +0s
    node-b  ranks [1, 2]  2x GTX 1060 6GB  libtorch builds/sm61-sm120  joined +1s
```

The endpoint is read-only and lives exactly as long as the launcher
process: it is up from before the join window opens (so `waiting` and
`forming` are observable), and connection-refused afterwards is the
honest "no run listening" signal (`fdl status` exits 1 with a note).
Reachability follows the port's bind scope - an all-tunneled run
exposes it through sshd only. See
[CLI reference](cli.md#fdl-status) for address resolution details.

---

## ElChe: phase machine + meta-controller

The cadence balancer has two control layers.

### Phase lifecycle

`Probe → Warmup → Stable → Mature`. Monotonic and `>=`-comparable.
Gates the more aggressive controllers (anchor swaps, `relax_up`) to
`>= Stable`.

| Phase | When | Behavior |
|---|---|---|
| `Probe` | No calibrations yet | Equal split, gather first timings. |
| `Warmup` | First few calibrations | Sticky anchor, conservative adjustments. |
| `Stable` | Steady state | Normal overhead auto-tune with hysteresis. `relax_up` and meta-controller swaps activate here. |
| `Mature` | Long-running steady state | Same as Stable; signal for telemetry. |

### Anchor auto-tune

After each averaging round, `(overhead = sync_ms / (wall_ms - sync_ms))`
is the fraction of compute time spent in AllReduce.

- `overhead > target`: increase anchor by `ceil(anchor * overhead /
  target)` (proportional to excess - overhead is wasted GPU time).
- `overhead < target/2`: decrease anchor by 1 (gradual - lower anchor
  means fresher gradients).
- 5% dead-zone: anchor changes smaller than 5% of current are no-ops.

Anchor is clamped to `[min_anchor, max_anchor]`.

### Weighted gradient averaging

When batch counts are unequal, each replica's gradient is scaled by its
batch contribution before AllReduce Sum:

```
weight[rank]  = count[rank] / sum(counts)
grad_avg      = sum(weight[rank] * grad[rank])
```

Mathematically correct mean gradient regardless of per-device batch
counts.

Weight consensus follows the same principle at every sync (shaped by
`gamma`), on both backends. Non-learnable f32 buffers - BatchNorm
running stats and the like - ride the same sync but are averaged with
*equal* weight among the ranks that stepped in the window, never
`gamma`-weighted: running statistics must not inherit a fast rank's
dominance. Non-f32 buffers (deterministic integer counters, updated
identically on every rank) keep their local value.

### LR-aware meta-controller

`ElCheConfig::meta_controller(true)` enables an observer above ElChe
that watches LR trajectory + anchor trend + convergence-guard verdicts
in a rolling window. Reactively nudges the anchor down on sharp LR
drops or sustained divergence, and reports `is_settled()` once the
metric stops moving. On by default (opt out with
`.meta_controller(false)` for unconditioned-trajectory
instrumentation).

### EASGD elastic averaging

`ElCheConfig::easgd_alpha(α)` tunes the EASGD-style blending on the
`CpuAsync` path (on by default there at α=0.5):

```
local_t1   = (1 - α) * local_t0  +  α * center_t0
center_t1  = (1 - α) * center_t0 +  α * mean(local_t0)
```

Smooths divergence in long async runs. Honored on `CpuAsync` only;
ignored elsewhere. Note that blending keeps replicas on a deliberate
elastic spread around the consensus - the divergence guard's default
threshold accounts for it (see the `convergence_guard` knob above).

---

## Outer optimizer - SlowMo / DiLoCo

By default the cluster averages replicas with a plain work-weighted mean.
An **outer optimizer** adds a second optimization loop applied to the
work-weighted consensus *between* the reduce and the broadcast, on top of
the inner per-rank optimizers - the hook for communication-efficient
methods like SlowMo and DiLoCo. Configure it on the builder or
`TrainerConfig`:

```rust
use flodl::{NesterovMomentum, SlowMomentum, OuterAvg};

Trainer::builder(model_factory, optim_factory, train_fn)
    .dataset(dataset).batch_size(32).num_epochs(20)
    .elche(ElCheConfig::nccl_cadence())
    .outer_optimizer(|| Box::new(NesterovMomentum::new(0.7, 0.9)))  // DiLoCo
    .run()?;
```

| Variant | Behavior |
|---|---|
| `OuterAvg` (default) | Stateless identity passthrough - reproduces plain work-weighted averaging. No momentum, no artifact. |
| `SlowMomentum::new(lr, mu)` | SlowMo heavy-ball momentum on the pseudo-gradient; continuous inner loop. |
| `NesterovMomentum::new(lr, mu)` | DiLoCo-style Nesterov outer step; `resets_inner()` makes each worker reset its inner optimizer per outer round. |

- Built once per site: controller-side on the CPU backend, per-rank
  replicated lock-step on NCCL.
- `ElCheConfig::gamma(γ)` sets the consensus allocation-weighting exponent
  (default `1.0` = pre-gamma behavior).
- **Checkpointing**: momentum-bearing variants persist their slow momentum
  to a `<stem>.outer.fdl` sidecar alongside the model / optim / meta
  bundle, and reload it on resume so the outer trajectory is faithful.
  `OuterAvg` writes no sidecar.
- `ddp-bench`: `--outer-optimizer none|slowmo|diloco` plus `--outer-lr` /
  `--outer-mu` / `--gamma`.

## A/B testing modes

Five modes via `ElCheMode`. One line per mode:

```rust
// Build the base
let base = || Trainer::builder(model_factory.clone(), optim_factory.clone(), train_step)
    .dataset(dataset.clone())
    .batch_size(64)
    .num_epochs(5)            // just enough to see the trend
    .max_grad_norm(5.0);

let a = base().elche(ElCheConfig::cpu_async()).run()?.join()?;
let b = base().elche(ElCheConfig::nccl_cadence()).run()?.join()?;   // also ElCheConfig::default()
let c = base().elche(ElCheConfig::nccl_sync()).run()?.join()?;
```

Same model, same data, same seed; change one line.

| Suggested order | Rationale |
|---|---|
| 1. **`NcclCadence`** (default) | Recommended NCCL default. ElChe tunes the anchor so the slow device sets the pace, fast devices process proportionally more batches per averaging window. Anchor-based cadence with AllReduce at every boundary. |
| 2. **`CpuCadence`** | Fastest wall time in the published benchmark (512s vs 548s nccl-cadence on the 200-epoch flagship). Same cadence semantics without NCCL - the natural pick when peer access is unavailable or the rig spans hosts without fast links. Cost: a decent CPU on the controller host. |
| 3. **`CpuAsync` (+ DiLoCo)** | Genuine async: barrier-free application, averaging decoupled from the GPU pipeline, EASGD blending on by default. A few percent of wall time behind `CpuCadence` on fixed-epoch runs (the divergence guard grows its window more cautiously early on; amortizes at length) - in exchange for jitter tolerance and the strongest convergence behavior: with the DiLoCo outer optimizer it posted the best eval of all modes in the published benchmark (0.9236 vs 0.9210 solo), holding the generalization peak that solo training overfits past. |
| 4. `NcclSync` | Tightest-cadence baseline. Tells you whether near-per-step synchronization helps for your specific model. Equal data split like vanilla DDP; the reduce fires per slow-rank step, not per batch (see the "What sync means" note above) - degenerates to vanilla DDP on homogeneous rigs. |

Compare on: `loss at epoch N`, `wall time per epoch`, and `loss per
wall-second` - that last metric is usually the decider. The `ddp-bench`
suite drives every mode through the same harness; see
[the benchmark report](ddp-benchmark.md) for the published numbers and
[`ddp-bench`](https://github.com/flodl-labs/flodl/tree/main/ddp-bench)
for the canonical worked example.

---

## Manual control - `Ddp::wrap`

For complex training patterns (GAN, RL, progressive growing) where you
need explicit per-step replica control, `Ddp::wrap` is the low-level
per-rank gradient-sync primitive. It is **not** the production multi-GPU
entry (that auto-promotes to process-per-rank); it wraps **one** replica
per rank against a shared rendezvous, and it is also what each cluster
rank uses internally. One rank per thread (single-process testing) or per
process; the world size comes from the rendezvous.

```rust
// Per rank: wrap this rank's replica. `global_rank` in [0, world_size),
// `rdv` a TcpRendezvous all ranks share.
let ddp = Ddp::wrap(&model, device, global_rank, &rdv)?;

ddp.sync_params()?;
// ... forward + backward ...
ddp.all_reduce_gradients()?;                            // unweighted
ddp.weighted_all_reduce_gradients(&batch_counts)?;      // ElChe-style
ddp.sync_buffers()?;
```

For all other use cases reach for `Trainer::run` or
`Trainer::builder().run()` - the process-based path is the production
one, with per-rank logs, rank death survival, and cluster fan-out.

---

## Data pipeline

Each rank constructs its own `DataLoader` against its own dataset
shard. The coordinator computes proportional sharding from
`ElCheConfig::partition_ratios` (or auto-balances by throughput) and
pushes the epoch plan to each worker. The DataLoader is otherwise
unaware that a cluster exists.

### Modes

| Mode | Description | When |
|---|---|---|
| **Resident** | Dataset loaded into GPU VRAM once. Per-epoch reshuffling via GPU-side `index_select`. | Dataset fits in ~75% of free VRAM. |
| **Streaming** | Persistent background worker thread, async H2D on a dedicated CUDA stream. Prefetch depth auto-adapts. | Dataset too large for VRAM. |

A 16 GB rank can go resident while a 6 GB rank on the same training
run uses streaming - each rank picks its own mode independently. No
lowest-common-denominator constraint.

### VRAM-aware prefetch

In streaming mode the prefetch depth is computed automatically:

```
depth = clamp(free_vram * headroom / batch_bytes, 2, max_depth)
```

- **Bootstrap**: 4 batches at construction time (model not yet loaded).
- **epoch(0)**: re-probes VRAM after model allocation; fills to cap.
- **epoch(N)**: re-probes each epoch, adapts to fragmentation.
- **`vram_max_usage(0.90)`**: use up to 90% of total VRAM (default).
- **`.prefetch(n)`**: manual override, disables automatic adaptation.
- **OOM fallback**: if resident mode fails with CUDA OOM, automatically
  retries with streaming mode.

See [Tutorial 13: Data Loading](tutorials/13-data-loading.md) for the
full `DataLoader` reference.

---

## NCCL primitives

For the rare cases where you need to drop below `Trainer`. The
init-on-main + `split()` pattern is enforced everywhere
(`ncclCommInitRank` from worker threads corrupts CUDA context on
heterogeneous GPUs).

### `NcclComms`

Group communicator for multi-GPU collectives. RAII: destroyed on drop.

```rust
let comms = NcclComms::new(&[Device::CUDA(0), Device::CUDA(1)])?;
comms.all_reduce(&[&tensor_a, &tensor_b], ReduceOp::Avg)?;
comms.broadcast(&[&params_0, &params_1], 0)?;

// Overlapped variants
comms.all_reduce_on_streams(&tensors, ReduceOp::Avg, &streams)?;
comms.broadcast_on_streams(&tensors, 0, &streams)?;
```

### `NcclComms::split()` → `Vec<NcclRankComm>`

```rust
let group = NcclComms::new(&devices)?;        // main thread
let rank_comms: Vec<NcclRankComm> = group.split()?;
// Move rank_comms[i] into thread i; NcclRankComm is Send.
```

Never call `NcclRankComm::init_rank()` from worker threads on
heterogeneous hardware - use `split()`.

### `NcclAbortHandle`

```rust
let handle = comm.abort_handle();
handle.abort()?;                              // unblocks stuck collectives
```

After abort, the communicator's `Drop` is a no-op.

### `ReduceOp`

| Variant | Op |
|---|---|
| `Sum` | Element-wise sum |
| `Prod` | Element-wise product |
| `Max` / `Min` | Element-wise max/min |
| `Avg` | Element-wise average |

---

## CUDA synchronization primitives

### `CudaEvent`

```rust
let event = CudaEvent::new(CudaEventFlags::Default)?;
event.record()?;                  // on current stream
event.record_on(&stream)?;        // on specific stream
event.synchronize()?;             // CPU blocks until complete
let done = event.is_complete()?;  // non-blocking poll

let ms = CudaEvent::elapsed_time(&start, &end)?;
```

Use `CudaEventFlags::DisableTiming` for pure synchronization (lower
overhead; `elapsed_time` will error).

### `CudaStream`

```rust
let stream = CudaStream::new(Device::CUDA(0), false)?;  // normal priority
stream.synchronize()?;
stream.wait_event(&event)?;        // stream waits for event
```

### `StreamGuard`

```rust
{
    let _guard = StreamGuard::new(&stream);
    tensor.copy_(&source, true)?;  // non-blocking copy on `stream`
}
// Default stream restored on drop.
```

---

## Troubleshooting

### Start with `fdl probe`

`fdl probe` (single host) or `fdl @cluster probe` (cluster) is the first
stop for any "it should work, why doesn't it" question. It surfaces:

- Missing libtorch variant / wrong arch for the local GPUs.
- Missing or mismatched libnccl across hosts.
- Missing `nccl_socket_ifname` on multi-host workers.
- Stale legacy schema keys in `fdl.cluster.yml`.
- Shared-data path resolution failures.
- Dashboard port already in use.

### NCCL init failure

`ncclCommInitAll failed` typically means NCCL can't establish
peer-to-peer between devices.

```bash
nvidia-smi topo -m            # check device connectivity
fdl probe                     # check NCCL availability + libtorch wiring
```

Falls back to shared memory transport if peer-to-peer is unavailable.
Or switch to a `Cpu*` mode in `ElCheConfig` to bypass NCCL entirely.

### NCCL version skew across hosts

If one host has libtorch shipping NCCL 2.27 and another has 2.26, the
handshake fails. Build a matching libnccl on the easier side and
`LD_PRELOAD` it via the worker's `env:` block:

```bash
fdl nccl build              # auto-detects target version + archs
```

### Parameter count mismatch

`GpuWorker rank N: model has M params but config has K`. The model
factory produced a model with a different parameter count than the
initial model used to extract starting parameters. Make sure
`model_factory(dev)` produces an identical architecture for every
device.

### CUDA context corruption

`CUBLAS_STATUS_EXECUTION_FAILED` or SIGABRT after NCCL init usually
means `ncclCommInitRank` was called from multiple threads on
heterogeneous GPUs. The framework uses the init-on-main + `split()`
pattern everywhere, but if you're driving `NcclComms` manually, make
sure you follow the same pattern.

Also covered by the "no CUDA before `Trainer::run`" invariant - any
CUDA tensor created in `main()` before the launcher trampoline poisons
spawned children's contexts.

### OOM on smaller GPU

Any anchor-based mode (`NcclCadence`, `CpuAsync`, `CpuCadence`)
routes through ElChe, which assigns proportionally fewer batches to
the slower/smaller GPU. The DataLoader's per-device backend selection
also helps: the large GPU goes resident while the small GPU streams.

```rust
.elche(ElCheConfig::nccl_cadence().max_anchor(50))   // or any anchor-based preset
```

### CPU averaging timeout

The CPU averaging path now waits indefinitely for survivors and lets
the elastic-membership machinery handle the dead-rank decision. If you
need a hard time bound (e.g. CI gating), `max_failure` + `ShutdownWithSave`
is the right knob - it triggers a clean checkpoint exit rather than
hanging.

### Cluster progressive hangs

If `fdl @cluster` runs hang several epochs in, the cause is usually:

1. **Stale child processes** from a previous aborted run holding GPU
   memory or rendezvous ports. `fdl @cluster` cleans these up
   pre-spawn, but a kill -9 on the launcher bypasses cleanup.
2. **Shared-mount staleness** when the project mount is NFS or virtiofs
   and the controller and a worker see different file states. `fdl
   probe` flags mount-state divergence.

---

Previous: [Tutorial 14: HuggingFace Integration](tutorials/14-flodl-hf.md) |
Next: [PyTorch Migration Guide](pytorch_migration.md)
