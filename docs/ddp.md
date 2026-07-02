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
a code rewrite.

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
fn train_step(model: &dyn Module, batch: &[Tensor]) -> Result<Variable> {
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
into a fresh CPU model for inference, or feed them into
`Trainer::resume_from` to continue training.

### Graph-shape one-liner - `Trainer::setup`

When the model is a flodl `Graph` and the training loop is yours:

```rust
let model: Graph = build_model()?;   // any FlowBuilder graph

Trainer::setup(
    &model,
    |dev| build_model_on(dev),       // per-replica factory
    |p|   Adam::new(p, 1e-3),        // per-replica optimizer
)?;

// Same loop on 1 or N GPUs.
for (input_t, target_t) in &batches {
    let input  = Variable::new(input_t.clone(),  false);
    let target = Variable::new(target_t.clone(), false);
    let loss   = cross_entropy_loss(&model.forward(&input)?, &target)?;
    loss.backward()?;
    model.step()?;                  // AllReduce + buffers + optimizer + zero_grad
}
```

`Trainer::setup_head` is the analogue for `flodl-hf` task-head wrappers
(implements `HasGraph`); the loop stays byte-identical between
`setup` / `setup_head`.

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
| `NcclSync` | Every batch | NCCL AllReduce | Homogeneous GPUs, correctness-first baseline |
| `NcclCadence` | Anchor-based (ElChe) | NCCL AllReduce | **Recommended NCCL default** - heterogeneous rigs; ElChe tunes the anchor so the slow device sets the pace, fast devices process proportionally more batches per averaging window |
| `CpuSync` | Every batch | CPU averaging | Sync without NCCL (peer-access unavailable, A/B against NCCL) |
| `CpuCadence` | Anchor-based | CPU averaging | Heterogeneous rigs without fast peer links |
| `CpuAsync` | Anchor + overshoot | CPU averaging + optional EASGD | **Best-in-class on the reference rig** - genuine async (decoupled averaging via separate channel), fastest convergence, fault-tolerant. CPU averaging is the only cost; a future dedicated averaging tier will lift it. |

`NcclSync` is the degenerate ElChe case (anchor=1). Every mode routes
through the same machinery, so switching between them is one line.

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
| `ElCheConfig::cpu_async()` | `CpuAsync` (best convergence in practice; see [A/B testing modes](#ab-testing-modes)) |

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
| `.overhead_target(f)` | `0.05` | Upper bound on `sync_ms / max(compute_ms)` per anchor window. ElChe grows the anchor when overhead exceeds the target, shrinks it when overhead drops below half. **Cadence + Async modes only** - Sync modes hardcode per-batch AllReduce and ignore the anchor knob. See [the overhead auto-tune section](#overhead_target-anchor-auto-tune) below. |
| `.max_batch_diff(n)` | `None` | Cap on how far the fastest rank may lead the slowest. `Some(0)` = strict lockstep regardless of mode. |
| `.relax_up(bool)` | `false` | Allow ElChe to grow the anchor in `Phase::Stable` when convergence stays clean. |
| `.partition_ratios(Vec<f64>)` | auto | Static per-rank data split (e.g. `[0.7, 0.3]`). **Honored on `Sync` policy only**; Cadence/Async use progressive dispatch driven by ElChe and ignore the static ratios. For dynamic heterogeneous scheduling under those policies, ElChe's throughput-based auto-rebalancing is the intended path. |
| `.meta_controller(bool)` | `true` | LR-aware meta-controller - watches LR + anchor + divergence; nudges anchor down on sharp LR drops or sustained divergence. On by default (LR drops are always worth catching); opt out for unconditioned-trajectory instrumentation. |
| `.convergence_guard(g)` | `TrendGuard::new(0.05)` | Divergence guardrail. `NoGuard`, `TrendGuard`, or `MsfGuard` (rate-based). |
| `.easgd_alpha(α)` | `None` | EASGD elastic blend on the `CpuAsync` path (`0 < α ≤ 1.0`). Ignored elsewhere. |
| `.gamma(γ)` | `1.0` | Consensus allocation-weighting exponent applied when the outer optimizer / averaging weights ranks by work. `1.0` = pre-gamma (plain work-weighting). |
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
| `.eval_dataset(ds)` | `Arc<dyn BatchDataSet>` | Held-out data for evaluation. |
| `.eval_fn(f)` | `EvalFn<M>` | Receives `(&M, &Tensor, &Tensor)`, returns `Result<f64>`. |
| `.eval_result_fn(f)` | `EvalResultFn` | Controller-side `(epoch, scalar)` sink. |
| `.epoch_callback_policy(p)` | `EpochCallbackPolicy` | `Fastest` (default) or `Rank(n)`. |
| `.outer_optimizer(factory)` | `Fn() -> Box<dyn OuterOptimizer>` | Outer-loop optimizer on the consensus (SlowMo / DiLoCo). Default = plain work-weighted averaging (`OuterAvg`). See [Outer optimizer](#outer-optimizer---slowmo--diloco). |
| `.checkpoint_at_epoch(n)` | `usize` | One-shot coverage-granular checkpoint at the epoch any rank first reaches (progressive modes). Pairs with `.save_path`. |
| `.eval_every(n)` | `usize` | Fire `eval_fn` every `n` epochs (`0` disables). The chained `DdpBuilder::eval_every` takes an `EvalCadence` instead. |
| `.timeline(t)` | `Arc<Timeline>` | Inject DDP events into a profiler stream. |
| `.cluster(c)` | `FullCluster` | Programmatic cluster topology (overrides any active overlay). |

`TrainerConfig::cluster(full)` is the seam for programmatic
multi-host launches (see [Programmatic clusters](#programmatic-clusters-clusterbuilder)
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
an overlay-driven cluster does (via `FLODL_FULL_CLUSTER_JSON`), so
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

`Trainer::resume_from(stem)` (and `TrainerConfig::resume_from`) loads a
checkpoint bundle and continues training. The bundle is three files:

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

Ranks can die and rejoin without aborting the run. The controller owns
the lifecycle; workers just report and follow.

### What happens when a rank dies

1. **Heartbeat miss** - controller transitions the rank to `Dead` in
   per-rank state, elastically renormalizes `partition_ratios` across
   survivors.
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
    host: 192.168.122.1           # rendezvous bind address
    port: 1337                    # rendezvous port (default 1337)
    path: /opt/flodl              # controller's view of the shared project root
    # docker: cuda                # optional pre-flight build service
    # arch: precompiled/cu128     # optional libtorch variant for pre-flight build

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

### Activating the overlay

Three equivalent forms (a command-line selector overrides `FDL_ENV`):

```bash
fdl @cluster <cmd>            # @ sigil (scan-anywhere before --)
fdl --env cluster <cmd>       # explicit flag
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

### LR-aware meta-controller

`ElCheConfig::meta_controller(true)` enables an observer above ElChe
that watches LR trajectory + anchor trend + convergence-guard verdicts
in a rolling window. Reactively nudges the anchor down on sharp LR
drops or sustained divergence, and reports `is_settled()` once the
metric stops moving. Off by default.

### EASGD elastic averaging

`ElCheConfig::easgd_alpha(α)` enables EASGD-style blending on the
`CpuAsync` path:

```
local_t1   = (1 - α) * local_t0  +  α * center_t0
center_t1  = (1 - α) * center_t0 +  α * mean(local_t0)
```

Smooths divergence in long async runs. Honored on `CpuAsync` only;
ignored elsewhere.

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
| 1. **`CpuAsync`** | **Best in class** on the reference rig - fastest wall-time *and* best convergence in the published `ddp-bench` runs. The CPU averaging path decouples from the GPU forward pass (genuine async - averaging on a separate channel) and benefits most from EASGD elastic blending. Cost: a decent CPU. A future dedicated averaging tier (extra GPU or peer) will lift the cost; the convergence quality is intrinsic to the algorithm. |
| 2. **`NcclCadence`** (default) | Recommended NCCL default. ElChe tunes the anchor so the slow device sets the pace, fast devices process proportionally more batches per averaging window. Anchor-based cadence with AllReduce at every boundary. |
| 3. `NcclSync` | Strict-sync baseline. Tells you whether per-batch synchronization helps for your specific model. Identical to vanilla DDP. |

Compare on: `loss at epoch N`, `wall time per epoch`, and `loss per
wall-second` - that last metric is usually the decider. The `ddp-bench`
suite drives every mode through the same harness; see
[`ddp-bench`](https://github.com/flodl-labs/flodl/tree/main/ddp-bench)
for the canonical worked example and the published numbers.

`CpuSync` and `CpuCadence` exist for completeness - A/B against the
NCCL variants when peer-access is unavailable. They're not usually
faster or more accurate than the NCCL variants for typical workloads;
`CpuAsync` is where the CPU backend shines.

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
