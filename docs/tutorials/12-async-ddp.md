# Heterogeneous & Multi-Host DDP

When the rigs get interesting - mixed GPU generations, mixed hosts,
mixed libtorch variants - the same `Trainer::builder` shape from
[Tutorial 11](11-multi-gpu.md) keeps working. This tutorial covers the
knobs that earn their keep on real heterogeneous deployments.

> **Prerequisites**: [Multi-GPU Training](11-multi-gpu.md). The
> universal `Trainer::builder` shape is assumed.

> **Time**: ~25 minutes.

> **Canonical reference**: [DDP Reference](../ddp.md).

## ElChe - the cadence balancer

Every cadence mode (`*Cadence`, `*Async`) routes through ElChe.
`NcclSync` and `CpuSync` are the tightest-cadence case: an equal data
split with the reduce firing as soon as every alive rank has made at
least one step since the last one — classic per-batch DDP on a
homogeneous rig, but NOT per-batch lockstep on a heterogeneous one
(the fast GPU runs several work-weighted steps per reduce; see
[What "sync" means](../ddp.md#elchemode---cadence--backend-in-one-name)).
ElChe's job is to keep AllReduce overhead bounded while respecting
weight-space divergence.

### Phase machine

ElChe progresses through four phases as calibrations accumulate:

| Phase | When | Behavior |
|---|---|---|
| `Probe` | No calibrations yet | Equal split across ranks; gather first timings. |
| `Warmup` | First few calibrations | Sticky anchor - cold-start noise on the larger/newer GPU can't flip the slow-anchor election. |
| `Stable` | Steady state | Normal overhead auto-tune with hysteresis. `relax_up` and the meta-controller anchor swaps activate here. |
| `Mature` | Long-running steady state | Same as Stable; signal for telemetry. |

The phase gate prevents the multiplicative overhead-tune scale from
compounding sparse early-reading noise (the historical "anchor jumps
from 10 to 22 on the first measurement" bug).

### Weighted gradient averaging

When ranks process different batch counts in a window, each replica's
gradient contributes proportionally to its batch count:

```
weight[rank]  = count[rank] / sum(counts)
grad_avg      = sum(weight[rank] * grad[rank])
```

This produces the mathematically correct mean gradient regardless of
per-device batch counts. No accuracy degradation from uneven splits.

### Anchor auto-tune

After each averaging cycle (gated to `Phase::Stable+`):

```
overhead = sync_ms / max(compute_ms across ranks)
```

| Observation | ElChe's proposal |
|---|---|
| `overhead > overhead_target` | Grow anchor by `ceil(anchor * overhead / target)` - sync less often, amortize the cost over more local batches. |
| `overhead < overhead_target * 0.5` | Shrink anchor by 1 - sync is cheap, can afford fresher gradients. |
| in the band, or change < 5% of current | Hold (5% dead-zone). |

The proposal is **proposed**, not committed - the convergence guard's
verdict decides whether it lands (see below). The anchor is then
clamped to `[min_anchor, max_anchor]`.

## Convergence guards - the safety belt

`overhead_target` alone would happily grow the anchor on any stable
model, eventually starving the gradients. The convergence guard
monitors weight-space divergence between replicas and pulls back.

| Guard | Behavior | When to pick |
|---|---|---|
| `TrendGuard::new(thresh)` | **Production default.** Three-rises-above-threshold rule on the `\|\|pre - post\|\| / \|\|post\|\|` ring buffer. Returns `SuppressGrowth` on persistent rising drift. | Almost everyone. |
| `MsfGuard::default().with_suppress(s, n).with_nudge(t, n, f)` | Rate-based detector built on the across-event MSF proxy `λ_ema = EMA((1/k_max) * log(D_t / D_{t-1}))`. Soft + hard thresholds escalate from `SuppressGrowth` to `NudgeDown`. | When you have time to tune thresholds for your specific architecture; can react faster than TrendGuard on sharp divergence spikes. |
| `NoGuard` | Always `Stable`. | Instrumentation runs that want an unconditioned trajectory (every overhead-tune proposal commits unconditionally). |

### Guard authority over `overhead_target`

ElChe's `report_timing` **proposes** an overhead-tune anchor change
but doesn't apply it. The guard's verdict commits:

| Verdict | Effect on the proposal |
|---|---|
| `Stable` | Commit (grow or shrink). |
| `SuppressGrowth` | Drop a proposed grow; **apply** a proposed shrink (shrink is the safe direction when divergence is rising). |
| `NudgeDown { factor }` | Drop the proposal entirely; nudge supersedes by shrinking the current anchor by `factor`. |

So `SuppressGrowth` doesn't just *not relax up* - it actively vetoes
the overhead-tune growth before it lands. The convergence guard is
authoritative over `overhead_target` by construction. See [DDP
Reference: Guard authority over
`overhead_target`](../ddp.md#guard-authority-over-overhead_target).

### LR-aware meta-controller

Above ElChe sits the LR-aware meta-controller (on by default -
`.meta_controller(true)`). It watches LR trajectory + anchor trend +
guard verdicts in a rolling window and reactively nudges the anchor
down on sharp LR drops or sustained divergence patterns. Reports
`is_settled()` once the metric stops moving - useful as an
early-stop signal.

Opt out for instrumentation:

```rust
let elche = ElCheConfig::nccl_cadence()
    .meta_controller(false);     // unconditioned trajectory
```

## EASGD elastic averaging (CpuAsync)

On the `CpuAsync` path, EASGD-style elastic blending smooths
divergence in long async runs:

```
local_t1   = (1 - α) * local_t0  +  α * center_t0
center_t1  = (1 - α) * center_t0 +  α * mean(local_t0)
```

Each rank's local params blend toward the center; the center blends
toward the mean of locals. `α` controls the blend rate (`0 < α ≤ 1.0`,
typical `0.4`-`0.8`). Honored on `CpuAsync` only; ignored elsewhere.

```rust
.elche(ElCheConfig::cpu_async().easgd_alpha(0.6))
```

## A/B testing modes - the recipe

Five modes via `ElCheMode`. Each switch is one line:

```rust
let base = || Trainer::builder(model_factory.clone(), optim_factory.clone(), train_step)
    .dataset(dataset.clone())
    .batch_size(64)
    .num_epochs(5)              // just enough to see the trend
    .max_grad_norm(5.0);

let a = base().elche(ElCheConfig::cpu_async()).run()?.join()?;     // best-in-class candidate
let b = base().elche(ElCheConfig::nccl_cadence()).run()?.join()?;  // default; recommended NCCL pick
let c = base().elche(ElCheConfig::nccl_sync()).run()?.join()?;     // tightest-cadence baseline
```

Suggested order (refined from the `ddp-bench` published numbers):

| Position | Mode | Rationale |
|---|---|---|
| 1 | **`CpuAsync`** | Best convergence + wall-time on the reference rig. CPU averaging decouples from the GPU forward path (genuine async - averaging on a separate channel) and benefits most from EASGD. Cost: a decent CPU. |
| 2 | **`NcclCadence`** (default) | Recommended NCCL default. ElChe-driven anchor; fast devices process proportionally more batches per averaging window. |
| 3 | `NcclSync` | Tightest cadence (per slow-rank step, equal split). Tells you whether tighter synchronization helps your specific model. |

Compare on: `loss at epoch N`, `wall time per epoch`, and **`loss per
wall-second`** - the last is usually the decider. A slightly higher
loss in half the time often wins.

`CpuSync` and `CpuCadence` exist for A/B against the NCCL variants
when peer-access is unavailable; they're rarely faster or more
accurate than NCCL for typical workloads. `CpuAsync` is where the
CPU backend shines.

The `ddp-bench` suite drives every mode through the same harness with
the same `train_step` closure. See
[`ddp-bench`](https://github.com/flodl-labs/flodl/tree/main/ddp-bench)
for the canonical worked example and the published convergence
numbers.

> `NcclAsync` used to exist as a sixth mode (NCCL + per-rank
> cross-epoch dispatch). It was dropped - measured benefit over
> `NcclCadence` was within noise on every tested rig, and the
> in-place AllReduce writeback raced with autograd on heterogeneous
> Pascal+Blackwell setups. `CpuAsync` is the real async mode:
> averaging is decoupled from the GPU pipeline through a separate
> channel.

## Heterogeneous-rig real-world example

A two-host cluster with mixed-generation GPUs:

- **node-a** (Blackwell): 2× RTX 5060 Ti (sm_120, 16 GB each), libtorch `precompiled/cu128`
- **node-b** (Pascal VM via virtiofs): 2× GTX 1060 (sm_61, 6 GB each), libtorch `builds/sm61-sm120` (source-built, multi-arch)

NCCL handshake across mixed major.minor versions sometimes fails
(node-a may ship NCCL 2.27, node-b may have 2.26 from libtorch). The
fix: build a matching libnccl on the easier side and `LD_PRELOAD` it
into libtorch:

```bash
fdl nccl build              # auto-detects target version + local archs
```

Then wire it into the worker's `env:` block in `fdl.cluster.yml`:

```yaml
workers:
  - host: node-b
    local_devices: all
    nccl_socket_ifname: enp1s0
    path: /srv/flodl
    arch: builds/sm61-sm120
    env:
      LD_PRELOAD: /srv/flodl/libtorch/nccl/builds/v2.27.5-sm61/lib/libnccl.so.2
```

`fdl probe` flags the version skew before launch:

```bash
fdl @cluster probe           # SSHes each worker; aggregates GPU + libtorch + NCCL inventory
```

## Cluster topology - `fdl.cluster.yml`

The structured schema with `controller:` and `workers[]:` blocks. See
[DDP Reference: Multi-host
clusters](../ddp.md#multi-host-clusters) for the full field listing.

Key conventions:

- One process per rank; each worker owns one rank per visible CUDA
  device.
- Global ranks are assigned sequentially by worker order: worker 0
  owns ranks `[0..N0)`, worker 1 owns `[N0..N0+N1)`, etc.
- `local_devices: all` probes the host at dispatch time via SSH +
  `nvidia-smi`. Explicit lists carry their own count.
- `nccl_socket_ifname:` is required on every worker when the cluster
  spans multiple hosts.
- `arch:` selects the libtorch variant *per host* - heterogeneous
  rigs run different variants without changing the convention path
  (`<path>/libtorch/<arch>/`).
- `docker:` (optional) names the compose service for training on this
  host. Mixed deployments (controller in Docker, worker bare-metal)
  are common.

Launch via the env overlay:

```bash
fdl @cluster train           # = fdl --env cluster train ; SSHes each worker
fdl @cluster probe           # readiness gate before launch
```

## Programmatic clusters - `ClusterBuilder`

For tests, embedded launchers, or any binary that wants to launch
without a yml on disk:

```rust
use flodl::ClusterBuilder;

let cluster = ClusterBuilder::new()
    .controller("192.168.122.1")
        .port(1337)
        .path("/opt/flodl")
    .done()
    .host("node-a")
        .ranks([0, 1])
        .devices([0, 1])
        .nccl_socket_ifname("enp1s0")
        .path("/opt/flodl")
        .arch("precompiled/cu128")
    .done()
    .host("node-b")
        .ranks([2, 3])
        .all_devices()
        .nccl_socket_ifname("enp1s0")
        .path("/srv/flodl")
        .arch("builds/sm61-sm120")
        .ssh_port(2222)
        .ssh_identity_file("/keys/cluster")
    .done()
    .build()?;

let cfg = TrainerConfig::new(dataset)
    .batch_size(64)
    .num_epochs(50)
    .elche(ElCheConfig::cpu_async().easgd_alpha(0.6))
    .cluster(cluster);

Trainer::run(model_factory, optim_factory, train_step, cfg)?.join()?;
```

`ClusterBuilder::all_local_gpus()` is the single-host shortcut -
synthesizes the same topology auto-promote uses on a multi-GPU host.

## Elastic membership

Ranks can die without aborting the run — the survivors carry on with
the dead rank's work redistributed (membership only shrinks; a dead or
new rank cannot join a formed world, scale-up is not yet implemented):

- **Heartbeat miss** → coordinator transitions the rank to `Dead` and
  renormalizes `partition_ratios` across survivors.
- **Lone NCCL survivor** short-circuits and exits (no dead-quorum
  AllReduce wait).
- **`max_failure` threshold** triggers a clean
  `ShutdownWithSave` - coordinator drives a final checkpoint
  through whichever survivor has the freshest state.
- **NCCL rendezvous-timeout retry** rebuilds the comm on the largest
  contiguous survivor subset.

Tune via:

```rust
TrainerConfig::new(dataset)
    .batch_size(64)
    .num_epochs(100)
    .save_path("ckpts/run")
    .checkpoint_every(5);
// max_failure comes from cluster-coordinator config (cluster.yml or
// programmatic ClusterCoordinatorConfig).
```

The save bundle is `<stem>.fdl` + `<stem>.meta.json` (carries
ElCheState - phase, calibration trajectory, ring buffer, guard
history). Resume:

```rust
TrainerConfig::new(dataset)
    .batch_size(64)
    .num_epochs(100)
    .save_path("ckpts/run43")
    .resume_from("ckpts/run42")     // loads ckpts/run42.{fdl,meta.json}
    .checkpoint_every(5);
```

A resumed run inherits ElChe's calibration trajectory - no warmup
re-run.

## `EpochCallbackPolicy::Fastest` - free-compute callbacks

By default, per-epoch callbacks (`checkpoint_fn`, `epoch_fn`,
`eval_fn`) fire on the **fastest rank** (lowest
`smoothed_ms_per_batch`). The reasoning: on heterogeneous rigs the
fastest rank has the most idle time at the sync barrier, so eval /
save runs as free compute. Sticky within a run; re-resolves only on
rank death.

Pin to a specific global rank with `EpochCallbackPolicy::Rank(n)`
when the research convention demands it. `n` is the **global rank**
(0..world_size cluster-wide), assigned sequentially by worker order
in the cluster topology.

```rust
TrainerConfig::new(dataset)
    .epoch_callback_policy(EpochCallbackPolicy::Rank(0));
```

## Quick reference

### `ElCheConfig` presets

| Constructor | Mode |
|---|---|
| `ElCheConfig::nccl_sync()` | `NcclSync` |
| `ElCheConfig::nccl_cadence()` | `NcclCadence` (**default**) |
| `ElCheConfig::cpu_sync()` | `CpuSync` |
| `ElCheConfig::cpu_cadence()` | `CpuCadence` |
| `ElCheConfig::cpu_async()` | `CpuAsync` (best-in-class on reference) |

### Knob cheat-sheet

| Setter | Default | Effect |
|---|---|---|
| `.overhead_target(f)` | `0.05` | Target ratio `sync_ms / max(compute_ms)`. Lower → grow anchor sooner; higher → tolerate more sync overhead. Cadence/Async only. |
| `.max_anchor(n)` | auto | Anchor growth ceiling. |
| `.max_batch_diff(n)` | `None` | Cap on fastest-vs-slowest lead. `Some(0)` = strict lockstep. |
| `.relax_up(true)` | `false` | Grow anchor by 1 on Stable verdict (in addition to overhead-tune proposals). |
| `.partition_ratios([...])` | auto | Static split - Sync mode only. |
| `.meta_controller(false)` | `true` | Opt out of LR-aware meta-controller. |
| `.convergence_guard(g)` | `TrendGuard::new(0.05)` | `NoGuard`, `TrendGuard`, or `MsfGuard`. |
| `.easgd_alpha(α)` | `None` | EASGD elastic blend - `CpuAsync` only. |

### Cluster launch

```bash
fdl probe                  # single-host readiness
fdl @cluster probe          # multi-host readiness (SSHes each worker)
fdl @cluster <cmd>          # fan out
FDL_ENV=cluster fdl <cmd>  # equivalent
fdl nccl build             # build libnccl for LD_PRELOAD bridge (heterogeneous NCCL versions)
```

<!-- nav: generated by site/build_guide.py — do not edit below -->

---

Previous: [Multi-GPU Training](11-multi-gpu.md) | Next: [Data Loading](13-data-loading.md)
