# Multi-GPU Training

Scale your model from one GPU to many with **zero training-loop
changes**. flodl auto-promotes single-host multi-GPU runs to
process-per-rank fan-out when 2+ CUDA devices are visible, and the
same entry point (`Trainer::builder(...).run()` or `Trainer::run`)
scales out to multi-host clusters via `fdl.cluster.yml` or
`ClusterBuilder`.

> **Prerequisites**: [Training](04-training.md) covers single-device
> training. This tutorial assumes the universal `Trainer::builder`
> shape - extending it to N GPUs is configuration, not code.

> **Time**: ~20 minutes.

> **Canonical reference**: [DDP Reference](../ddp/01-reference.md) for the full
> knob surface.

> **Runnable example**: [`auto_promote`](../../flodl/examples/auto_promote/) is the
> same plain training code on 1 CPU, 1 GPU, or N GPUs - with zero distributed code in it.

## The one-liner

```rust
use flodl::*;
use std::sync::Arc;

fn train_step(model: &impl Module, batch: &[Tensor]) -> Result<Variable> {
    let input  = Variable::new(batch[0].clone(), false);
    let target = Variable::new(batch[1].to_dtype(DType::Int64)?, false);
    cross_entropy_loss(&model.forward(&input)?, &target)
}

let handle = Trainer::builder(
        |dev| build_model_on(dev),         // model factory (per-replica)
        |params| Adam::new(params, 1e-3),  // optimizer factory (per-replica)
        train_step,                        // one-step closure
    )
    .dataset(dataset)
    .batch_size(64)
    .num_epochs(50)
    .run()?;

let state: TrainedState = handle.join()?;
```

On a single GPU or CPU, this runs in the calling process. On a host
with 2+ visible CUDA devices, it **auto-promotes** to one process per
rank, with NCCL cadence (the default `ElCheConfig::default() =
nccl_cadence()`) and the meta-controller on by default. Zero code
change.

```
  ddp: 2 GPUs (heterogeneous) | RTX 5060 Ti (16.0 GB) | GTX 1060 (6.0 GB)
  ddp: role=launcher → spawning 2 rank children
  rank 0: device=CUDA(0) | RTX 5060 Ti | epoch 0/50 started
  rank 1: device=CUDA(1) | GTX 1060    | epoch 0/50 started
```

## What just happened - auto-promote

When `Trainer::builder(...).run()` (or `Trainer::run`) fires on a host
where `flodl::sys::detect_gpus() >= 2` and **no cluster overlay is
active**, the framework:

1. Probes visible CUDA devices via `nvidia-smi` (no libtorch context
   touched).
2. Synthesizes a single-host cluster topology covering every visible
   device.
3. Turns the calling binary process into the **launcher** - it forks
   one child per rank, each running the same binary with
   `FLODL_RANK=<n>` set.
4. Each rank child connects to the controller (also hosted on the
   launcher) over TCP, joins the NCCL rendezvous, and runs the same
   `train_step` closure on its assigned dataset shard.
5. The launcher supervises children, forwards their stdout, and tears
   them down on exit (including SIGINT / SIGTERM).

Auto-promote is **`cfg(not(test))`-gated** for flodl's own test suite
(so `Ddp::wrap` keeps driving the thread-based multi-GPU tests
in-process). External crates that want a single-rank run on a
multi-GPU host scope down via `CUDA_VISIBLE_DEVICES=0`.

## The critical invariant - no CUDA before `Trainer::run`

> **User binaries must not touch libtorch's CUDA context before
> reaching `Trainer::run` / `Trainer::builder().run()`.**

That means **no** `flodl::tensor::gpu_device_count()`, **no**
`Module::on_device(CUDA(_))`, **no** CUDA-Tensor construction in
`main()`. The launcher process exits without running training; any
CUDA tensors it instantiated would corrupt the spawned children's
contexts on heterogeneous-GPU rigs.

For any pre-`Trainer::run` GPU query, use `flodl::sys::detect_gpus()`,
which shells out to `nvidia-smi` and does **not** load libtorch:

```rust
use flodl::sys::detect_gpus;

let gpus = detect_gpus();
for g in &gpus {
    eprintln!("GPU {}: {} ({}, {} MB)",
        g.index, g.name, g.arch_label(), g.total_memory_mb);
}

if gpus.is_empty() {
    eprintln!("no CUDA GPUs visible - single-device fallback");
}
```

`detect_gpus()` honors `CUDA_VISIBLE_DEVICES`, so the result matches
the view that the auto-promote path and child processes will see.

## Graph models: the same entry

A flodl `Graph` (any `FlowBuilder`) is a `Module`, so it trains through
the same `Trainer::builder(...).run()` / `Trainer::run(...)` entry shown
above - just return the built graph from the `model_factory` closure:

```rust
fn build_model(device: Device) -> Result<Box<dyn Module>> {
    let g = FlowBuilder::from(/* ... */).build()?;
    Ok(Box::new(g))
}
// then: Trainer::builder(build_model, |p| Adam::new(p, 1e-3), train_step)
//           .dataset(dataset).batch_size(64).num_epochs(5).run()?;
```

`flodl-hf` task-head wrappers `impl Module` directly, so they ride the
same `Trainer::builder(...)` / `Trainer::run(...)` entry - see
[HuggingFace Integration](14-flodl-hf.md). To keep the loop body
yourself, the cooperative tier - `Trainer::builder(...).into_worker()?` -
returns a `Worker` while the controller owns cadence, partition,
eval-election, and checkpointing; for an explicit per-rank loop use
`Ddp::wrap` (the bypass tier).

## PyTorch comparison

PyTorch multi-GPU requires process groups, environment variables,
`torchrun`, and a `DistributedSampler`:

```python
# PyTorch: 8+ lines of setup + torchrun launcher
dist.init_process_group("nccl")
model = DDP(model.to(rank))
sampler = DistributedSampler(dataset)
loader = DataLoader(dataset, sampler=sampler)
```

flodl auto-detects and process-fans-out from one call:

```rust
// One closure-based call, no torchrun, no environment dance.
Trainer::builder(model_factory, optim_factory, train_step)
    .dataset(dataset)
    .batch_size(64)
    .num_epochs(50)
    .run()?;
```

## ElChe - heterogeneous GPUs train at their own pace

Named after the marching principle: *"the column marches at the
slowest one's pace."* ElChe is the heterogeneous-rig balancer that
runs under every cadence mode except strict Sync.

### The problem with strict sync on mixed hardware

Traditional DDP forces all GPUs to AllReduce after every batch. If
your RTX 5060 Ti processes a batch in 10 ms and your GTX 1060 takes
25 ms, the fast GPU idles 60% of the time at every barrier.

### The solution

The slow GPU anchors the cadence. The fast GPU processes more batches
per averaging window - the same AllReduce sync time amortizes over
more local compute. ElChe auto-tunes the anchor count based on
observed throughput so the AllReduce overhead stays at a small
fraction of compute time (`overhead_target`, default 5%).

`Trainer::builder().run()` activates ElChe by default (the default
mode is `NcclCadence`). No configuration needed for the common
heterogeneous-rig case.

### How it adapts

After each averaging cycle, ElChe reports:

- **Per-rank `compute_ms`**: wall time the rank spent doing forward +
  backward + optimizer step for its assigned batch count.
- **`sync_ms`**: wall time the AllReduce + divergence measurement
  took.
- **`overhead = sync_ms / max(compute_ms across ranks)`**: how big a
  tax did sync charge as a fraction of the slowest rank's compute?

ElChe then proposes an anchor adjustment:

- `overhead > overhead_target`: grow the anchor (sync less often,
  amortize the sync cost over more local work).
- `overhead < overhead_target / 2`: shrink the anchor (sync is cheap,
  can afford fresher gradients).

The convergence guard has the final say: rising weight-space
divergence vetoes growth even when `overhead_target` would propose
it. See [DDP Reference: Guard authority over
`overhead_target`](../ddp/01-reference.md#guard-authority-over-overhead_target).

## Tuning ElChe

Most rigs don't need any tuning - the defaults adapt to whatever
hardware is visible. When you do want to tune, every knob lives on
`ElCheConfig`:

```rust
use flodl::*;

let elche = ElCheConfig::nccl_cadence()  // also the default
    .overhead_target(0.05)               // tighter: aim for sync < 5% of compute
    .max_anchor(50)                      // ceiling on anchor growth
    .max_batch_diff(20)                  // cap how far fast rank may lead slow rank
    .relax_up(true);                     // grow the anchor on stable convergence

Trainer::builder(model_factory, optim_factory, train_step)
    .dataset(dataset)
    .batch_size(64)
    .num_epochs(50)
    .elche(elche)
    .run()?;
```

Common knobs (full surface in [DDP
Reference](../ddp/01-reference.md#elcheconfig-knobs)):

| Knob | Default | When to touch |
|---|---|---|
| `.overhead_target(f)` | `0.05` | Lower on fast inter-GPU links where you can afford more sync; higher (e.g. `0.10`) when AllReduce is expensive. |
| `.max_anchor(n)` | `None` (auto) | Set when you want a hard ceiling on staleness. |
| `.max_batch_diff(n)` | `None` | `Some(0)` = strict lockstep regardless of mode; useful for reproducibility. |
| `.partition_ratios([...])` | auto | Static split when ElChe's auto-balancing isn't what you want (Sync mode only). |
| `.meta_controller(false)` | `true` | Opt out for unconditioned-trajectory instrumentation runs. |

## Mode selection - `ElCheMode`

The five DDP modes:

| Mode | Best for |
|---|---|
| `CpuAsync` | **Best in class** for convergence + wall-time on the reference rig; needs a decent CPU. Genuine async - averaging decoupled from the GPU pipeline. |
| `NcclCadence` (default) | Strong NCCL default. Anchor-based scheduling; fast devices process proportionally more batches per averaging window. |
| `NcclSync` | Tightest cadence (reduce per slow-rank step, equal data split — not per-batch lockstep; see [What "sync" means](../ddp/01-reference.md#elchemode---cadence--backend-in-one-name)) - homogeneous rigs, correctness-first baseline |
| `CpuSync`, `CpuCadence` | A/B against NCCL when peer-access is unavailable |

See [A/B testing modes](../ddp/03-internals.md#ab-testing-modes) for the suggested
order and rationale.

## DataLoader integration

Each rank constructs its own `DataLoader` against its own dataset
shard. The coordinator computes proportional sharding from
`partition_ratios` (or auto-balances by throughput) and pushes the
epoch plan to each worker. The DataLoader is otherwise unaware that a
cluster exists.

Per-device backend selection is independent on every rank - a 16 GB
GPU can go resident (dataset loaded into VRAM once) while a 6 GB GPU
on the same training run uses streaming (prefetch worker with async
H2D). No lowest-common-denominator constraint.

```rust
let loader = DataLoader::from_batch_dataset(dataset)
    .batch_size(64)
    .names(&["image", "label"])
    .build()?;
```

See [Data Loading](13-data-loading.md) for the full
DataLoader surface.

## Host-side callbacks - `metrics_fn` / `eval_fn`

The shape `Trainer::builder().run()?.join()?` is the canonical "just
train" form. For per-epoch logging, monitor wiring, or held-out
evaluation, register host-side callbacks:

```rust
use std::sync::Arc;

Trainer::builder(model_factory, optim_factory, train_step)
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
    .run()?;
```

`metrics_fn` fires once per epoch on the host thread after all ranks
aggregate. `eval_fn` runs on the rank elected by
`EpochCallbackPolicy::Fastest` (the default - the rank with the lowest
`smoothed_ms_per_batch`, so eval is free compute on heterogeneous
rigs). `eval_result_fn` receives the scalar result on the host.

Pin a specific rank with `EpochCallbackPolicy::Rank(n)` - `n` is the
**global rank index** (0..world_size), assigned sequentially by
worker order in the cluster topology. See [DDP Reference:
`EpochCallbackPolicy`](../ddp/01-reference.md#epochcallbackpolicy).

## Live dashboard

`monitor.serve(port)` works transparently across single-host
multi-GPU and multi-host clusters - the launcher hosts a single
dashboard URL that aggregates every rank's metrics. Open it once;
follow the whole cluster.

```rust
use flodl::Monitor;

let mut monitor = Monitor::new(num_epochs);
monitor.serve(3000)?;             // http://launcher-host:3000

// Wire the monitor through metrics_fn:
let mon_handle = Arc::new(monitor);
let mon_for_cb = Arc::clone(&mon_handle);

Trainer::builder(model_factory, optim_factory, train_step)
    .dataset(dataset)
    .num_epochs(num_epochs)
    .metrics_fn(Arc::new(move |m| {
        let elapsed = std::time::Duration::from_millis(m.epoch_ms as u64);
        mon_for_cb.log(m.epoch, elapsed, m);
        Ok(())
    }))
    .run()?
    .join()?;
```

The dashboard shows per-rank tabs (one per rank, per host), throughput
curves, batch-share distribution, VRAM, and ElChe anchor evolution. See
[Training Monitor](09-monitor.md) for the full surface.

## Scaling out - multi-host clusters

Add an `fdl.cluster.yml` next to your `fdl.yml`:

```yaml
cluster:
  controller:
    host: 192.168.122.1
    port: 1337
    path: /opt/flodl

  workers:
    - host: node-a
      local_devices: [0, 1]
      nccl_socket_ifname: enp1s0
      path: /opt/flodl
      arch: precompiled/cu128

    - host: node-b
      local_devices: all
      nccl_socket_ifname: enp1s0
      path: /srv/flodl
      arch: builds/sm61-sm120     # different libtorch variant - fine
```

Launch with the env overlay:

```bash
fdl probe                   # readiness gate - verify before launching
fdl @cluster train           # SSHes each worker, pre-builds, fans out
```

`Trainer::run` is the **same call** on the worker - the multi-host
launcher trampoline takes care of fan-out, NCCL rendezvous, and
controller binding. See [DDP Reference: Multi-host
clusters](../ddp/02-cluster-guide.md) and [CLI Reference: cluster
commands](../cli/01-install.md) for the full surface.

For programmatic clusters (tests, embedded launchers without a yml on
disk), use `ClusterBuilder`:

```rust
use flodl::ClusterBuilder;

let cluster = ClusterBuilder::new()
    .controller("controller.example.com").port(1337).path("/opt/flodl")
    .done()
    .host("worker-a").ranks([0, 1]).devices([0, 1]).nccl_socket_ifname("enp1s0").path("/opt/flodl")
    .done()
    .host("worker-b").ranks([2]).devices([0]).nccl_socket_ifname("enp1s0").path("/srv/flodl")
    .done()
    .build()?;

let cfg = TrainerConfig::new(dataset)
    .batch_size(64)
    .num_epochs(50)
    .cluster(cluster);

Trainer::run(model_factory, optim_factory, train_step, cfg)?.join()?;
```

For the single-host "every visible GPU" shortcut, `ClusterBuilder::all_local_gpus()` synthesizes the topology auto-promote uses internally.

## Resume + checkpoints

Training survives rank death (elastic membership) and saves
periodically. Resume from any checkpoint bundle:

```rust
Trainer::builder(model_factory, optim_factory, train_step)
    .dataset(dataset)
    .batch_size(64)
    .num_epochs(100)
    .save_path("ckpts/run43")
    .resume_from("ckpts/run42")     // loads ckpts/run42.{fdl,meta.json}
    .checkpoint_every(5)
    .run()?
    .join()?;
```

`<stem>.meta.json` carries ElCheState (phase, calibration trajectory,
ring buffer) so a resumed run inherits ElChe's calibration. See
[DDP Reference: Resume + checkpoints](../ddp/01-reference.md#resume--checkpoints).

## Manual control - `Ddp::wrap` (expert bypass)

For training patterns that need explicit per-step replica control - GAN
discriminator vs generator, RL actor vs critic, progressive growing -
`Ddp::wrap` is the low-level per-rank gradient-sync primitive (it wraps
one replica per rank against a shared rendezvous, and is what each
cluster rank uses internally). It is not the production multi-GPU entry,
which auto-promotes to process-per-rank.

```rust
// Per rank: global_rank in [0, world_size), rdv a shared TcpRendezvous.
let ddp = Ddp::wrap(&model, device, global_rank, &rdv)?;

ddp.sync_params()?;
for batch in &dataset {
    let loss = model.forward(&batch)?;
    loss.backward()?;
    ddp.weighted_all_reduce_gradients(&batch_counts)?;
    ddp.sync_buffers()?;
    optimizer.step()?;
    optimizer.zero_grad();
}
```

For all standard use cases, `Trainer::run` / `Trainer::builder` is the
production path - per-rank logs, rank death survival, cluster fan-out,
elastic membership, controller-driven checkpoint retry.

## Quick reference

| Entry | When |
|---|---|
| `Trainer::builder(model_fn, opt_fn, step).run()` | Universal - any Module, any tier (CPU / 1 GPU / N GPUs / cluster). |
| `Trainer::run(model_fn, opt_fn, step, cfg)` | Same as above but takes a `TrainerConfig` data-bag - useful for config-driven launchers. |
| `Trainer::builder(model_fn, opt_fn, step).into_worker()?` | Cooperative tier - you own the loop body (`next_plan` / `next_batch` / `step` / `finish`) while the controller owns cadence, partition, eval-election, checkpointing. `flodl-hf` heads `impl Module`, so they ride this too. |
| `Ddp::wrap(&model, device, global_rank, &rdv)` | Low-level per-rank gradient-sync primitive for manual control (GAN/RL); production multi-GPU auto-promotes to processes. |

| Knob | Lives on | Common values |
|---|---|---|
| `.elche(ElCheConfig)` | TrainerConfig / DdpBuilder | `nccl_cadence()` (default), `cpu_async()`, etc. |
| `.epoch_callback_policy(p)` | TrainerConfig / DdpBuilder | `Fastest` (default), `Rank(global_rank)` |
| `.checkpoint_every(n)` | TrainerConfig / DdpBuilder | usize |
| `.save_path(p)`, `.resume_from(p)` | TrainerConfig / DdpBuilder | `&str` |
| `.metrics_fn(f)`, `.eval_fn(f)`, `.eval_result_fn(f)` | TrainerConfig / DdpBuilder | `Arc<dyn Fn(...)>` |
| `.cluster(FullCluster)` | TrainerConfig | from `ClusterBuilder` or yml |
| `.timeline(Arc<Timeline>)` | TrainerConfig / DdpBuilder | profiler events |

<!-- nav: generated by site/build_guide.py — do not edit below -->

---

Previous: [Graph Tree](10-graph-tree.md) | Next: [Heterogeneous & Multi-Host DDP](12-async-ddp.md)
