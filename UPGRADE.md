# Upgrading floDl

Five upgrades are documented here, newest first: the 0.9.0
consensus-artifact + observability pass, the 0.8.0 multi-vendor GPU
pass, the 0.7.0 monitor record surface, the 0.6.0 process-model
distributed rewrite, then the 0.5.0 CLI maturity pass.

---

## Upgrading to floDl 0.9.0 (the run's artifact is the consensus)

0.9.0 makes the checkpoint bundle the run's canonical artifact — the
model your callbacks receive, your evals score, and a clean end writes
is now provably the same consensus — and grows the observability plane
(graph timing heat maps, the rebuilt forensic timeline, live AMD
metrics, the `fdl ui` operations page). Most of the surface is
additive; the breaks concentrate in the distributed layer's config
structs and one worker API.

### TL;DR

| Change | What it costs you |
|---|---|
| `ClusterWorker::run_until_shutdown` returns `Result<()>` | Code binding its old `Option<ParamSnapshot>` stops compiling. The run's final model is the consensus bundle on disk. |
| New pub fields on config / data structs | Struct **literals** need the new field (or `..Default::default()`); builder and setter users are untouched. |
| `TrendGuard` → `LevelGuard`, `MsfGuard` → `GrowthGuard` | Nothing now: deprecated aliases keep the old names compiling. The deep `ddp_run::convergence::` paths are gone; ddp-bench's `--guard trend\|msf` spellings fail fast — use `level` / `growth`. |
| Managed-tier `TrainedState` comes back empty | Set `.save_path(stem)` and read the bundle; the cooperative tier's `finish()` is unchanged. |
| Cluster wire version bumped (4 → 5) | Every box in one cohort must run the same flodl. |

### The consensus bundle is the run's canonical persist form

Three guarantees replaced three holes on the CPU averaging path: a
natural clean end with `save_path` set always writes the FINAL
consensus bundle (a single-epoch run no longer trains to completion
and leaves nothing on disk), `checkpoint_fn` always receives the
consensus (never a rank's EASGD blend), and `eval_fn` always scores it
— served at the reduce, with the final canonical eval adopting the
retained consensus, so the final metric describes exactly the model
the bundle persists.

The structural consequence is the one compile break: managed-tier rank
children no longer materialize a per-rank final snapshot at teardown
(nothing consumed it, and the run's model is on disk in better form),
so `ClusterWorker::run_until_shutdown` returns `Result<()>`:

```rust
// before
let snap = worker.run_until_shutdown(train_fn)?;   // Option<ParamSnapshot>

// now
worker.run_until_shutdown(train_fn)?;
// the run's final model: .save_path("ckpts/run43") → ckpts/run43.fdl
```

Riders worth knowing: checkpoint writers create the stem's parent
directories, the run prints where the stem resolves in its first
lines, `request_checkpoint` / `request_eval` are served at the next
reduce rather than an epoch boundary, and the crash-save bundle is
documented for what it is (rank-0's own model, best-effort).

### Struct literals: the new public fields

None of these carry `#[non_exhaustive]`, so exhaustive struct literals
and destructures stop compiling until they name the field (or use
`..Default::default()` where a `Default` exists). Builder-style code is
unaffected.

| Struct | New field | Why |
|---|---|---|
| `TrainerConfig` / `DdpRunConfig` / `WorkerConfig` | `profile_graph: bool` | The timing heat map's data plane (see below). |
| `ClusterCoordinatorConfig` | `consensus_checkpoint_fn` | The controller-side consensus checkpoint callback. |
| `AveragedParams` | `realized: bool` | Distinguishes a realized consensus from a keep-local round. |
| `ModelSchema` | `f32_buffer_idx: Vec<usize>` | Bundles carry the f32 buffer subset (a non-f32 buffer no longer silently disables consensus checkpoints). Old bundles still load. **`tensor_count()` changed meaning**: params + the f32 buffer subset, no longer params + all buffers. |
| `graph::Profile` | `source: ProfileSource` | Says which clock produced the numbers (`GpuEvents` / `HostWallClock`). |
| `monitor::TimelineBroadcast` | `rank_samples` | Remote hosts' samples now ride the broadcast. |

### The convergence guards are named for what they measure

`TrendGuard` → `LevelGuard` (watches the divergence **level**) and
`MsfGuard` → `GrowthGuard` (watches the **rate** at which divergence
compounds — the axis that may cut the anchor). Deprecated aliases in
`flodl::compat` keep the old Rust names compiling at their documented
paths; the deep `flodl::distributed::ddp_run::convergence::TrendGuard`
/ `::MsfGuard` paths no longer resolve. `ddp-bench` renames hard:
`--guard none|level|growth`, validated before a container starts.

### No action needed, but worth knowing

- **The async overshoot budget is derived per rank** from the measured
  reduce and the rank's own pace; the flat default ceiling is now
  non-binding. `--max-overshoot N` still pins a flat one.
- **`record_log` / `--record-log` writes a shipped mirror**: the writer
  stages on node-local disk and a shipper mirrors to your directory —
  mid-run the destination holds `*.log.partial`, published by rename at
  teardown. A hung shared mount can no longer stall a record append.
- **AMD GPUs report live utilization** (amdgpu sysfs, no ROCm SMI, no
  extra install). In a container the probe needs `--device=/dev/kfd` —
  the node ROCm itself requires.
- **The dashboard grew tabs** (Overview / Architecture / Timeline), and
  `TrainerConfig::profile_graph` bakes per-GPU-model timing heat maps
  from device-event profiles shipped at teardown.
- **Graph profiling on CUDA now times execution, not the launch**
  (device-side events; readings resolve one pass behind). If you pinned
  profile numbers anywhere, expect them to grow to the true cost.

### Cluster cohorts must be same-version

`CONTROL_PROTOCOL_VERSION` went 4 → 5 (the graph-timings frame and the
consensus-eval arm ride the control wire). As always, a mixed cohort
refuses at the handshake, named. **Upgrade every box in a cohort
together.**

---

## Upgrading to floDl 0.8.0 (multi-vendor GPU)

0.8.0 adds AMD (ROCm) as a second GPU backend, and the API stopped
saying CUDA where it meant "the GPU". Most of the surface moved by
rename with a deprecated alias left behind, so **existing code keeps
compiling and starts warning**. Two changes do not warn, and they are
the two to read: the Rust floor, and Hub repo ids.

### TL;DR

| Change | What it costs you |
|---|---|
| `cuda_*` → `gpu_*`, `Cuda*` → `Gpu*` | Nothing now: 33 deprecated aliases ship. Deprecation warnings until you rename. |
| `fdl cuda-test` → `fdl gpu-test` (and friends) | CI configs and scripts need the new spelling. The old names are gone. |
| MSRV is **1.91** | `rustup update`. Nothing else. |
| flodl-hf: bare Hub repo ids stop resolving | **Silent breakage of working code.** Owner-qualify every repo id. |
| `bincode` → `bincode-next` | Nothing. Byte-identical wire, same API. |
| Cluster wire versions bumped | Every box in one cohort must run the same flodl. |

### The GPU API is vendor-neutral

libtorch ships one GPU backend per build, and ROCm masquerades as CUDA
all the way down to its dispatcher, so the vendor is a build-time
property rather than something the API should name. `cuda_*` became
`gpu_*`, `Cuda*` became `Gpu*`, and the C ABI `flodl_cuda_*` became
`flodl_gpu_*`.

```rust
// before
use flodl::tensor::{cuda_available, cuda_device_count, CudaStream};

// now
use flodl::tensor::{gpu_available, gpu_device_count, GpuStream};
```

Deprecated aliases for all 33 renamed items live in `flodl::compat` and
are re-exported from the crate root and from `tensor` / `nn`, so every
old path still resolves. They are one unit of removal: expect them to go
in a later major, and let the deprecation warnings drive the rename.

Three names stayed CUDA-spelled **on purpose**, so do not "finish" the
rename in your own code:

- `flodl_nccl_*` — RCCL exports the `ncclXxx` names verbatim, and
  PyTorch keeps `backend="nccl"` on ROCm. Renaming costs parity and buys
  nothing.
- `set_cudnn_benchmark` — upstream routes it to MIOpen on ROCm.
- `cuda_compute_capability` — compute capability is an NVIDIA concept. A
  neutral name would compile on ROCm and hand back a gfx arch dressed as
  an `sm_` pair, so it errors there instead.

### Feature flags name the axis they select

`gpu` is the vendor-neutral gate; `cuda` and `rocm` select a backend and
imply it. **`--features cuda` behaves exactly as before.** `build.rs`
refuses both vendors at once, and refuses `gpu` without one.

```toml
flodl = { version = "0.8", features = ["cuda"] }   # unchanged
flodl = { version = "0.8", features = ["rocm"] }   # new
```

### The `fdl cuda-*` commands are `gpu-*`

`fdl gpu-test`, `gpu-test-all`, `gpu-test-nccl`, `gpu-test-serial`,
`gpu-build`, `gpu-clippy`, `gpu-shell`. The cargo feature they pass is
derived from the **active libtorch variant** (`cu…` / `sm…` builds
`--features cuda`; `rocm…` / `gfx…` builds `--features rocm`) and
exported as `$FDL_GPU_FEATURE`, so one command line covers either
vendor. Anything calling the old names — CI, scripts, muscle memory —
needs updating; they were renamed, not aliased.

If your own `fdl.yml` hardcodes `--features cuda`, `--features
"$FDL_GPU_FEATURE"` is the portable form. It expands empty on a CPU
variant, so one line serves a CPU build and both vendors.

### Minimum Rust is 1.91

Three floors stack: let-chains in our own source (1.88), `bincode-next`
(1.90), and `xet-core-structures` (1.91), the last reached through
`hf-hub` 1.0's mandatory `hf-xet` dependency. With no committed
lockfile, consumers resolve fresh, so this floor tracks the ecosystem
and will move again.

### flodl-hf: owner-qualify your Hub repo ids

This is the one that breaks working code with no compiler warning. The
HuggingFace stack moved to `hf-hub` 1.0, `safetensors` 0.8 and
`tokenizers` 0.23. The Hub answers a bare pre-2024 repo name with a 307
redirect to its owner-qualified home; `hf-hub` 0.4's transport followed
that redirect and 1.0 does not, so a bare name now fails with
`Repository not found`.

```rust
// before -- worked through the redirect
let model = BertModel::from_pretrained("bert-base-uncased")?;

// now -- owner-qualified, which is also what the Hub documents
let model = BertModel::from_pretrained("google-bert/bert-base-uncased")?;
```

The five common ones: `google-bert/bert-base-uncased`,
`FacebookAI/roberta-base`, `FacebookAI/xlm-roberta-base`,
`distilbert/distilbert-base-uncased`, `albert/albert-base-v2`. Repo ids
that already carry an owner (every fine-tune, every `org/model`) are
unaffected.

### No action needed, but worth knowing

- **`bincode` → `bincode-next`.** bincode 2 is unmaintained
  (RUSTSEC-2025-0141) and its `3.0.0` on crates.io is a tombstone. The
  successor has the same API and a byte-identical wire encoding, each
  decoding the other's output, so nothing on the wire or on disk
  changed.
- **`flodl-hw` is a new workspace crate**, dependency-free hardware
  detection shared by `flodl` and `flodl-cli`. `flodl::sys` is a facade
  over it, so existing `flodl::sys::detect_gpus()` callers are
  untouched. Depend on `flodl-hw` directly only if you want GPU/RAM
  detection without libtorch.
- **`detect_gpus()` now returns only devices the running build can
  address**, filtered by the build's vendor. That is what keeps a CUDA
  build from spawning a rank on an AMD card it cannot reach. Its sibling
  `detect_gpus_physical()` ignores the filter and answers the
  provisioning question instead ("which libtorch variant covers this
  box"). If you were using `detect_gpus()` for the second question, move
  to the sibling.
- **Integrated GPUs no longer double-count memory.** An APU's aperture
  is carved out of host RAM, so the data plane reserves it once.
  `gpu_ram_share` on `DataLoaderBuilder` / `TrainerConfig` /
  `DdpRunConfig` overrides the fraction, and is **required** on a
  multi-socket APU: each package carries its own aperture while
  `/proc/meminfo` is system-wide, so that case is refused as an ordinary
  error from `DataLoaderBuilder::build` rather than guessed at.

### Cluster cohorts must be same-version

The control wire went `CONTROL_PROTOCOL_VERSION` 2 → 4 (the join hello
grew the vendor label, run identity, NCCL version and model signature)
and the CPU averaging plane 2 → 3 (structural-zero elision). Both bumps
are deliberate: bincode is not self-describing, so without them a mixed
cohort would fail somewhere deep and opaque. With them it fails at the
handshake, named.

**Upgrade every box in a cohort together.** A rolling upgrade of a
standing fleet is not supported.

Checkpoint metadata went schema 4 → 5 (`CoverageBlock.epoch_splits`).
Old checkpoints still load; a resume whose epoch slicing changed is now
refused up front rather than silently re-slicing coverage.

### Doc links moved

`docs/cli.md` split into `docs/cli/01-install.md` … `06-source-checkout.md`,
and `docs/ddp.md` into `docs/ddp/01-reference.md` plus a cluster guide,
an internals chapter and a troubleshooting chapter. `docs/README.md` is
the index. Site permalinks on flodl.dev are unchanged.

---

## Upgrading to floDl 0.7.0 (the monitor record surface)

0.7.0 is additive almost everywhere: the recursive dashboard portal, the
path-addressed record plane, `record_log`, `save_dashboard`,
`reports_per_epoch`, RoPE / SwiGLU / `TokenShards`, and the CPU-plane
memory work all arrive without touching existing signatures. Two public
items in `flodl::monitor::record` did change shape, and both fail at
compile time rather than silently.

If you do not name `flodl::monitor::record` directly, there is nothing
to do.

### `Res` gained two public fields

`Res` grew `gpu_util_max` and `vram_alloc_max: Option<f64>` so a resource
point summarises an *interval* rather than an instant (the previous
latest-wins reading was usually sampled at a reduce boundary, which is
exactly when a GPU sits idle, so a busy card reported as quiet). It is
not `#[non_exhaustive]`, so reading fields is unaffected but struct-literal
construction breaks:

```rust
// before
let res = Res { gpu_util, vram_alloc, vram_total };

// now: fill the envelope fields, or let Default cover them
let res = Res { gpu_util, vram_alloc, vram_total, ..Default::default() };
```

The `..Default::default()` form is also future-proof against further
envelope fields.

### `NodeRecord::to_record_json` / `flat_records` take `Option<u64>` for `tick`

An epoch record closes an epoch rather than belonging to a sub-epoch
window, so it genuinely carries no window index, and `0` would have been
indistinguishable from the first window:

```rust
// before
node.flat_records(ts, tick, epoch);

// now
node.flat_records(ts, Some(tick), epoch);   // a sub-epoch window record
node.flat_records(ts, None, epoch);         // an epoch-boundary record
```

### Two behavior changes worth knowing (no API change)

- **`MultiheadAttention` routes through fused scaled-dot-product
  attention.** The mask contract is unchanged (`true` / non-zero =
  masked). The fused backends reorder reductions, so loss curves can
  shift within kernel-level noise, the same class of drift GPU training
  already has between runs. If you pin exact loss values in a test,
  expect to re-baseline.
- **Graph `switch` dispatches per sample.** `ArgmaxSelector` now emits
  one index per row and only the branches that received rows run
  ([#32](https://github.com/flodl-labs/flodl/issues/32)). Whole-batch
  switching is still supported and still the cheaper path (a scalar
  index, `FixedSelector`, an unbatched stream, or unanimous per-sample
  routing all skip gather and reassembly). A custom selector that
  returns neither a single index nor exactly one per row is now a loud
  error instead of routing the batch somewhere arbitrary.

---

## Upgrading to floDl 0.6.0: the process-model distributed rewrite

This is the largest pre-1.0 change to the `flodl` crate's public
surface. Single-device and single-host code is unaffected; the breaks
are all in the multi-GPU / cluster distributed layer.

### The in-process multi-GPU engine is gone; multi-GPU is process-per-rank

`Trainer::builder(...).run()` and `Trainer::run(...)` now transparently
auto-promote to process-per-rank fan-out when 2+ GPUs are visible (or a
cluster overlay is active). The training entry point is unchanged — the
same code runs single-device, single-host multi-GPU, and multi-host
cluster. You do not launch anything differently.

`Ddp::wrap` remains as the explicit per-rank bypass primitive, but its
signature changed from the old multi-model/multi-device thread form to
one rank per call:

```rust
// before: one call wrapped every local replica
// Ddp::wrap(&[&model0, &model1], &[dev0, dev1])?

// now: one call per rank (one model, one device, this rank's global id,
// the shared rendezvous)
let ddp = Ddp::wrap(&model, device, global_rank, &rendezvous)?;
```

### Removed: the 0.3.0-deprecated compatibility surface

These aliased `Trainer::builder()` for two minor releases and are now
gone:

| Removed | Replacement |
|---|---|
| `AsyncDdp` / `AsyncDdpBuilder` / `AsyncDdpConfig` | `DdpHandle` / `DdpBuilder` / `DdpRunConfig` (create via `Trainer::builder()`) |
| `DdpHandle::auto()` / `auto_with()` / `builder()` | `Trainer::builder(model_factory, optim_factory, train_fn)` |

### Removed: the self-driven `Trainer::setup` tier

`Trainer::setup` / `setup_with` / `setup_head` / `setup_head_with` and
their `DdpConfig` config bag are gone. They were the only path that
scheduled without the controller, so they missed the convergence guard,
LR-aware meta-controller, outer optimizer, elastic membership, and
checkpoint orchestration. The user-owned-loop ergonomics they offered
now return as the **cooperative tier** on the controller engine.

| Removed | Replacement |
|---|---|
| `Trainer::setup(...)` / `setup_with(...)` (you own the loop) | `Trainer::builder(model_factory, optim_factory, train_fn).into_worker()?` - returns a `Worker`; you own the loop body while the controller owns cadence, partition, eval-election, and checkpointing |
| `Trainer::setup_head(&head, factory, opt)` / `setup_head_with(...)` | task heads `impl Module` directly now - drive the head through `Trainer::builder(head_factory, optim_factory, \|head, batch\| head.compute_loss(...)).into_worker()?` (cooperative) or `.run()` (managed) |
| `DdpConfig` config bag | `TrainerConfig<M>` (with `Trainer::run(...)`) or the `Trainer::builder(...)` chained setters |

The cooperative loop body reads:

```rust
let mut worker = Trainer::builder(model_factory, optim_factory, train_fn).into_worker()?;
while let Some(plan) = worker.next_plan()? {
    while let Some(batch) = worker.next_batch()? {
        let loss = /* forward + loss */;
        worker.step(&loss)?;
    }
}
let state = worker.finish()?;
```

If you don't need to own the loop, `Trainer::run(...)` / `Trainer::builder(...).run()`
(the managed tier) drives it for you. See `docs/design/trainer-execution-tiers.md`.

### Removed: `DataLoader::distributed`

Data sharding is now owned by the cluster coordinator (it dispatches
each rank's per-epoch partition from the shared dataset). Build a plain
`DataLoader` / pass the dataset to `Trainer::builder(...).dataset(...)`;
the framework shards it.

### Removed: the NCCL async mode

`ddp-bench`'s `nccl-async` and the `Async` policy on the NCCL backend are
gone. Cross-epoch lookahead on NCCL delivered near-zero real-world
speedup over `nccl-cadence` while complicating the dispatch path.
`ElCheMode` never carries an `NcclAsync` variant. The genuine async mode
is `cpu_async` (decoupled averaging on a separate channel):

| Removed | Replacement |
|---|---|
| `nccl-async` / `Async` on the NCCL backend | `ElCheConfig::nccl_cadence()` (same backend, bounded overhead) or `ElCheConfig::cpu_async()` (real decoupled averaging) |

### Removed: the graph-embedded loss hook and cluster state

`Graph::set_loss_fn` / `Graph::has_loss_fn` and the `LossContext` type
are gone; they were the vestigial distributed-gather loss hook and had no
remaining driver once the setup tier went. The graph-embedded
`cluster_ddp` / `cluster_el_che` state and the cluster branches of
`Graph::step` went with them, so **`Graph::step` is single-device only**.
`HasGraph` is unaffected. Distributed loss now lives in the `train_step`
closure you hand to `Trainer::builder(...)`, which is where the
controller can see it.

### cluster.yml: structured `controller:` / `workers:` schema

The flat top-level keys are replaced by nested blocks:

```yaml
# before
master_addr: 192.168.1.10
master_port: 29500
workers:
  - host: gpu-1
    ssh_user: ubuntu
    ssh_port: 22

# now
cluster:
  controller:
    host: 192.168.1.10          # was master_addr
    port: 1337                   # was master_port
    path: /home/me/project       # controller's view of the shared root
  workers:
    - host: gpu-1
      ssh:                       # ssh_* knobs move into an ssh: sub-block
        user: ubuntu
        port: 22
```

`fdl probe` flags the legacy flat keys with migration hints.

---

## Upgrading to floDl 0.5.0 (the fdl CLI maturity pass)

floDl 0.5.0 is the **fdl CLI maturity pass**. The framework API stays
compatible with 0.4.0; the only breaking change lives in the `fdl.yml`
manifest and the `#[derive(FdlArgs)]` attribute contract.

### TL;DR

1. Rename `scripts:` → `commands:` in your `fdl.yml`, wrapping each
   value in a `run:` field.
2. If any `#[derive(FdlArgs)]` struct has a field named `help`,
   `version`, `quiet`, or `env` (or short-flagged `h`, `V`, `q`, `v`,
   `e`), rename it.
3. Optional: rename `fdl.dev.yml` / `fdl.ci.yml` style files you had
   been selecting manually - `fdl --env <name>` now loads them
   automatically.

That's it. Everything else is additive.

---

### 1. `scripts:` → `commands:` in `fdl.yml`

In 0.4.0, `fdl.yml` had two top-level maps:

- `scripts:` - shell-string commands, no docker wrapping.
- `commands:` - docker-wrapped entries with structured config.

In 0.5.0 these are merged into one **`commands:` map** with three
kinds, chosen by which fields the entry sets: `run:` (shell), `path:`
(nested project), or preset (`ddp:` / `training:` / `output:` /
`options:` merging over an enclosing `entry:`).

#### Minimal migration

```yaml
# 0.4.0 ---------------------------------------------------
scripts:
  fmt: cargo fmt --all
  lint: cargo clippy -- -D warnings

commands:
  test:
    docker: dev
    run: cargo test --features cuda

# 0.5.0 ---------------------------------------------------
commands:
  fmt:
    run: cargo fmt --all
  lint:
    run: cargo clippy -- -D warnings
  test:
    docker: dev
    run: cargo test --features cuda
```

#### Rules of the three kinds

| Kind    | Set                              | Argv forwarded? | Notes                                                           |
|---------|----------------------------------|-----------------|-----------------------------------------------------------------|
| `run:`  | `run:` (optionally `docker:`)    | **no**          | Closed script; use shell `$VAR` inside. `docker:` is allowed here only. |
| `path:` | `path:` (or empty + sibling dir) | yes             | Nested project with its own `fdl.yml`; forwarded argv validated against its `entry:` schema. |
| preset  | neither `run:` nor `path:`       | yes             | Only legal inside a `path:`-kind sub-command's own `fdl.yml`. Deep-merges `ddp:` / `training:` / `output:` / `options:` over the enclosing defaults. |

Common gotchas:

- **`docker:` on a non-`run:` entry** now errors at load time. Move the
  `docker:` field onto the `run:` entry it belongs to, or onto the
  sub-command's own `fdl.yml` at the top level.
- **Extra argv after `fdl <cmd> ...`** is **not** forwarded to a `run:`
  entry. If you relied on `fdl my-script foo bar` passing `foo bar` to
  the shell, switch to either `$FDL_EXTRA_ARGS` inside the script, or
  migrate the entry to a `path:` kind with a typed `entry:` binary.
- **Auto-bootstrap**: if only `fdl.yml.example` is checked in, `fdl`
  now offers to copy it to a real (gitignored) `fdl.yml` on first run.

Load-time errors tell you exactly which file, which key, and which
rule failed.

---

### 2. Reserved CLI flags in `#[derive(FdlArgs)]`

In 0.4.0, a struct field named `help` silently overrode `--help`. In
0.5.0 the following longs and shorts are **reserved** and cannot be
shadowed; collisions error at derive time:

- Longs: `--help`, `--version`, `--quiet`, `--env`
- Shorts: `-h`, `-V`, `-q`, `-v`, `-e`

If you have a struct like this:

```rust
// 0.4.0
#[derive(FdlArgs)]
struct Args {
    #[option]
    help: Option<String>,   // will fail to compile in 0.5.0
}
```

rename the field:

```rust
// 0.5.0
#[derive(FdlArgs)]
struct Args {
    #[option(short = 'H')]   // a non-reserved short, if you need one
    help_text: Option<String>,
}
```

The short-flag derivation is automatic from the long name's first
letter; if that first letter is reserved, pass `short = '...'`
explicitly or let the derive skip the short.

---

### 3. Environment overlays (optional, new)

If you already maintained per-environment `fdl.yml` files manually
(e.g. `fdl.local.yml`, `fdl.ci.yml`), 0.5.0 now loads them on top of
the base via:

```bash
fdl @ci test              # @ sigil (scan-anywhere before --)
fdl --env ci test         # explicit flag
FDL_ENV=ci fdl test       # env var
```

Nothing breaks if you don't use this - overlays are purely additive.
`fdl config show [env]` prints the resolved merged config with
per-layer origin annotations, which is the fastest way to verify a
new overlay before running a long job.

---

### 4. New top-level commands (informational)

None of these replace existing commands; they are new conveniences
that existed as no-ops or were simply absent in 0.4.0:

- `fdl config show [env]` - resolved YAML with origin annotations.
- `fdl schema list | clear [<cmd>] | refresh [<cmd>]` - manage the
  per-command schema cache.
- `fdl autocomplete` - one-shot installer for shell completions.
- `--refresh-schema` per-invocation flag to refresh one entry's cache
  without a manual `fdl schema refresh`.

---

### 5. `flodl-cli-macros` on crates.io

0.5.0 adds one new published crate:

- [`flodl-cli-macros`](https://crates.io/crates/flodl-cli-macros) -
  the proc-macro derive for `FdlArgs`, re-exported by
  [`flodl-cli`](https://crates.io/crates/flodl-cli) as
  `flodl_cli::FdlArgs`. Downstream binaries depend on `flodl-cli`,
  not on this crate directly.

`flodl-cli` itself was already published on crates.io in earlier
versions; 0.5.0 bumps it along with the rest of the workspace.

You can install the CLI with `cargo install flodl-cli` or via the
pre-compiled bootstrap: `curl -sL https://flodl.dev/fdl -o fdl`.

---

### 6. Framework changes

No breaking changes to the `flodl` crate in 0.5.0. The CHANGELOG has
no `### Removed` or `### Changed (breaking)` entries outside of the
CLI / manifest scope above.

If you're upgrading from 0.3.0 or earlier, read through CHANGELOG.md
from your version forward - the 0.4.0 entry is the larger one on the
framework side.

---

## Reporting issues

Please file [GitHub issues](https://github.com/flodl-labs/flodl/issues)
with a minimal reproducing `fdl.yml` and the exact error message if
anything in this guide leaves you stuck.
