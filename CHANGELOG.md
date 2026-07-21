# Changelog

All notable changes to floDl will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

The headline of this release is a full re-architecture of the distributed layer from a thread-per-GPU in-process model to a process-per-rank cluster model with an authenticated control plane, dial-in membership, elastic failure handling, controller-driven checkpoint orchestration, and a transparent launcher trampoline. The same single training entry (`Trainer::builder()` / `Trainer::run`) now drives single-device, single-host multi-GPU, and multi-host clusters from one code path. Every cross-host channel shares ONE controller port (workers join through it, training rides it, `fdl status` reads it, and `tunnel: true` routes it through SSH). ElChe (the heterogeneous cadence balancer that landed in 0.5.x) gained a phase machine, a divergence guardrail, an LR-aware meta-controller, delivered-cost scheduling, EASGD elastic averaging on the CPU async path, a pluggable outer optimizer (SlowMo / DiLoCo), and a `Fastest` epoch-callback dispatcher for free-compute eval. `fdl` gained a cluster readiness gate (`fdl probe`), a live run-status command (`fdl status`), a libnccl bridge builder (`fdl nccl build`), a global `--gpus` flag, and a strict cluster.yml schema with controller/worker separation.

### Added

#### Multi-process cluster architecture: launcher / controller / coordinator / worker

The distributed layer is now process-based end-to-end. A single `Trainer::run` invocation transparently auto-promotes to process-per-rank fan-out when 2+ GPUs are visible (homogeneous local rig) or when a cluster overlay is active (`fdl @cluster <cmd>` / `FDL_ENV=cluster`). Thread-based multi-GPU is retained only as the lower-level `Ddp::wrap` primitive (manual / test use); the in-process multi-GPU orchestration engine is gone (see Changed).

- **`flodl::distributed::launcher`**: role detector (`Role::SingleDevice | Rank | Launcher | Relay | Agent`) plus the launcher trampoline that opens the membership window, supervises children concurrently, and tears them down on parent exit. The trampoline runs from inside the user's `main()` at the `Trainer::builder(...).run()` boundary so guard closures and other non-`Serialize` `DdpRunConfig` knobs reach the controller without crossing a process boundary as JSON. Binaries that gate before `Trainer::run` (GPU checks, mode parsing) call `launcher::exit_if_worker_role()` first so the internal per-host roles never fall into user gating.
- **`flodl::distributed::controller`**: `ClusterController` runs on the launcher host as a TCP byte router for CPU averaging round-frames and rank-side log fan-in. The controller, not rank 0, binds rendezvous (`distributed: controller binds rendezvous; drop rank-0 master pattern`) so the orchestrator host is no longer a NCCL rank itself.
- **`flodl::distributed::cluster_coordinator::ClusterCoordinator`**: state-machine port of the old in-process `Coordinator` adapted for cross-process scheduling. Drives epoch dispatch, callback role assignment, dead-rank handling, CPU finalize, NCCL via-coord routing, heartbeat, checkpoint orchestration, cost-aware dispatch, and chunk-pool replay. Split across 9 files under `cluster_coordinator/` (averaging, callback_roles, config, dead_ranks, epoch_dispatch, event_loop, lifecycle, mod, test_helpers).
- **`flodl::distributed::cluster_worker::ClusterWorker<M>`**: TCP-driven wrapper around `GpuWorker` running on each rank child. Mirrors the in-process worker loop but with control / timing / metrics / param flows traveling over the wire.
- **`flodl::distributed::cpu_reduce`**: rank-side TCP client for CPU averaging + a `CpuAverager` on the launcher side. Replaces the in-process `cpu_avg` averaging path on cluster runs; `Tensor` ↔ `RoundFrame` conversion is the boundary.
- **HMAC-authenticated wire protocol (`flodl::distributed::wire`)**: every control frame is HMAC-SHA256-keyed by a per-session 128-bit salt, truncated to 64 bits. Stale / mis-routed / forged frames fail authentication with 2^-64 probability and surface loudly. Payloads are not confidential (no TLS); the guarantee is authentication and tamper detection. Magic + version constants are independent of the data-channel protocol so the two evolve separately.
- **Dashboard relocated from rank to launcher**: ranks emit `TimingMsgWire::Dashboard*` registration frames (graph SVG, metadata, hardware summary) and piggy-back resource samples on `MetricsMsgWire::resources`; the coordinator forwards everything to a `DashboardSink` trait whose concrete implementation in `launcher` wraps the HTTP `DashboardServer`. One dashboard URL for the whole cluster, hosted off the operator-facing host, with per-rank tabs derived from the cluster topology.

#### Dial-in membership: join window, quorum knobs, worker agents

Workers **join** a run; the controller admits them. The launcher opens a join window on the controller port; every worker — fan-out-managed and self-deployed alike — dials in with a hello (host, GPU inventory, libtorch variant, dataset signature) and receives its global ranks in **admission order** (contiguous by construction). `world_size` freezes at window close, and all coordination infrastructure is sized to the world that actually formed.

- **One worker agent per host** (`Role::Agent`): fan-out starts one SSH session per remote host; the agent dials back in, joins, and — once the world forms — spawns its host's relay and rank children locally. The join connection stays open as the host control link (per-rank exit reports up, abort down, EOF = host death). A **self-deployed worker** needs nothing but the controller address: a three-field `AgentSpec` (`host`, `controller_host`, `controller_port`) in `FLODL_INTERNAL_AGENT_JSON` resolves its own GPUs and joins like any fan-out agent.
- **`controller.join:` quorum knobs** (cluster.yml, `ClusterBuilder`, or defaults): `min_rank_start` (quorum), `join_timeout` (window; quorum-early does NOT close it), `target_ranks` (early close), `max_join_timeout` (hard cap, loud fail), `open_admission`. Fan-out defaults make the window close the instant every configured rank is in — zero added latency, all-or-nothing preserved.
- **Trust follows bind scope**: pre-shared session salt keys the join hello on managed rigs (wrong key = dropped without reply); a loopback bind (all workers tunneled) is open by construction — reachability through sshd IS the authentication — and hands the salt out in the accept reply. `open_admission: true` extends that to a network bind, loudly warned.
- **Run status on the training port**: the same controller port answers plain HTTP GETs with the run's membership state as `state.json` — lifecycle phase (`waiting` / `forming` / `training` / `done` / `failed`), per-host membership, join-window countdowns. **`fdl status`** pretty-prints it (`--json` for scripts, `--addr host[:port]` for self-deploy operators); curl works without fdl.

#### One controller port: mux, SSH tunnels, cleartext guard

Every cross-host channel (membership join, NCCL bootstrap rendezvous, CPU-reduce data, coordinator control, HTTP status) accepts on the single `controller.port` (default 1337); connections route themselves with a 4-byte channel-select magic. Config is one `host:port`; one SSH forward covers all traffic.

- **`tunnel: true` per worker**: routes the host's training traffic through its fan-out SSH session (a remote forward on the agent session) instead of a direct TCP connection. CPU ElChe modes only (NCCL's data plane is peer-to-peer and cannot ride a controller tunnel); when every remote worker is tunneled the controller binds loopback-only, making the training port unreachable except through sshd.
- **Cleartext guard**: flodl channels are HMAC-authenticated but not encrypted; flodl now warns loudly whenever a cleartext channel touches a peer outside private address space (loopback / RFC1918 / link-local / CGNAT-shared), on both the accept and dial sides.
- **Model-derived frame ceiling**: the maximum accepted wire-frame size is derived from the actual model (bytes × 2, floored at 64 MiB) instead of a fixed 1 GiB constant, installed in every process from the launcher's CPU-side model probe.

#### Per-host relay: fold tier + single uplink

One transport relay per remote host (`Role::Relay`, no CUDA) multiplexes its local ranks onto a single controller connection — and is the first fold tier of the CPU reduce: it sums its local ranks' contributions element-wise (masses included) into one host frame per round and fans the controller's single consensus broadcast back out. K local ranks cost 1x the model bytes on the host uplink per direction instead of Kx. A rank death mid-window simply produces a lighter host frame; the controller accounts per-connection and forwards dead-rank declarations down to the owning relay.

#### `TrainerConfig` + `ElCheMode`: single-config training entry

Collapses the policy × backend matrix into one user-facing enum and gathers everything `Trainer::run` needs into one struct. The chained `Trainer::builder(...)` form still exists and remains the right tool for callback-heavy setups; `TrainerConfig` is the data-bag form for config-driven launchers.

- **`ElCheMode`** enum (in `flodl::distributed::config`): `NcclSync | NcclCadence | CpuSync | CpuCadence | CpuAsync`. Names match the ddp-bench / commit / design-doc vocabulary; internally splits into the legacy `(ApplyPolicy, AverageBackend)` pair. `NcclSync` is the degenerate ElChe case (anchor=1) so all five modes route through the same code path.
- **`ElCheConfig`**: controller-scope tuning (mode, anchor, max_anchor, min_anchor, overhead_target, max_batch_diff, max_overshoot, gamma, relax_up, partition_ratios, meta_controller, convergence_guard, divergence_threshold, no_divergence_guard, easgd_alpha). Five preset constructors (`nccl_sync()`, `nccl_cadence()`, `cpu_sync()`, `cpu_cadence()`, `cpu_async()`) plus a builder chain.
- **`TrainerConfig<M>`**: umbrella struct gathering dataset / batch_size / num_epochs / elche / max_grad_norm / checkpoint_every / save_path / resume_from / callbacks (`checkpoint_fn`, `epoch_fn`, `metrics_fn`, `scheduler_fn`, `eval_fn`, `eval_result_fn`) / epoch_callback_policy / timeline / optional programmatic `cluster`. Both `Trainer::run` and `Trainer::builder().run()` accept it.
- **`Trainer::run(model_factory, optim_factory, train_fn, cfg)`**: config-bag entry. Internally builds a `DdpBuilder`, sets the launcher env from `cfg.cluster` if present, and dispatches through the same launcher trampoline as fdl-cli-launched runs (one launcher contract; two construction paths).

#### `ClusterBuilder` + `HostBuilder`: programmatic cluster construction

Fluent builder mirroring `fdl.cluster.yml` 1:1 (same fields, same validation), for tests and for binaries that want to launch a cluster from inside `main()` without depending on a yml on disk.

- **`flodl::ClusterBuilder::new()`** → `.controller("controller.example.com").port(1337).path("/srv/project").done().host("worker-a").devices([0, 1]).nccl_socket_ifname("enp1s0").ssh("worker-a.example.com").ssh_port(22).ssh_user("ubuntu").ssh_identity_file("/path/to/key").done().build()` → `FullCluster`. `.controller()` returns a `ControllerBuilder` (`.port()` / `.path()` / `.done()`); each `.host()` returns a `HostBuilder`; `.done()` returns the parent `ClusterBuilder` for chaining.
- **`ClusterBuilder::all_local_gpus()`** ergonomic single-host helper: synthesizes a `FullCluster` from `sys::detect_gpus()` with a loopback controller and one worker pinning every visible CUDA device. The "I just want every local GPU as a rank, no yml" path.
- **`TrainerConfig::cluster(full)`** wires a `FullCluster` directly into `Trainer::run`. The launcher env-var contract (`FLODL_INTERNAL_FULL_CLUSTER_JSON`) is filled by `Trainer::run` if not already set by fdl-cli, so a programmatic cluster and an overlay-driven cluster reach the launcher the same way.

#### `flodl::sys::detect_gpus`: CUDA-free GPU detection

`sys::detect_gpus() -> Vec<GpuInfo>` shells out to `nvidia-smi` and returns `(index, name, sm_version, vram_bytes)` per visible device without loading libtorch. Honors `CUDA_VISIBLE_DEVICES` so the result matches the post-scope view that the auto-promote path and child processes will see.

This is the canonical pre-`Trainer::run` GPU query. The previous habit of calling `flodl::tensor::cuda_device_count()` from `main()` initializes libtorch's CUDA context in the launcher process; that context then poisons the spawned children's contexts on heterogeneous-GPU rigs ("no CUDA before `Trainer::run`" invariant). `detect_gpus` is the safe replacement.

The "no CUDA before `Trainer::run`" invariant is now hardened: `Trainer::builder` / `Trainer::run` / `Module::on_device(CUDA(_))` all defer device touches until inside the run path. User binaries that respected the invariant pre-0.6.0 are unaffected; binaries that didn't will now error at a clearer site instead of corrupting a spawned child's CUDA state silently.

#### Elastic membership + controller-driven checkpoint orchestration

Ranks can die without aborting the run: the controller evicts the dead rank for the remainder of the run and redistributes its work across survivors. The controller owns the lifecycle decisions; workers just report and follow. Membership only ever shrinks — a dead or new rank cannot join a formed world (elastic scale-up is designed but not yet implemented).

- **Heartbeat**: `HeartbeatWire` flows worker → controller; missed heartbeats past a configurable threshold transition the rank to `Dead` in the coordinator's per-rank state and elastically renormalize partition ratios across survivors.
- **`max_failure` threshold + `ShutdownWithSave`**: cluster aborts cleanly when the surviving rank count drops below `max_failure`. On abort the coordinator drives a final checkpoint save through whichever rank still has the freshest state (callback-role-aware), then signals every survivor to exit. Lone NCCL survivors short-circuit the wait and exit immediately.
- **`flodl::distributed::checkpoint_meta`**: `CheckpointMeta` writes a `<stem>.meta.json` sidecar carrying ElCheState (phase, calibration_count, anchor, partition_ratios, ring buffer) plus the `SaveReason` (GracefulShutdown / MaxFailureExceeded / SingleSurvivor / AllRanksLost / ReduceStall / Checkpoint) and the `CheckpointBundle` path helpers (`model_path` / `optim_path` / `meta_path` / `config_sidecar_path`). The controller writes meta atomically alongside the model + optimizer files.
- **Controller-driven checkpoint retry + role failover**: a save failure on the elected callback rank does not poison the run. The coordinator picks a new callback rank from survivors (cost-aware: lowest smoothed_ms_per_batch first, sticky within a run), re-issues the save, and resumes. Failed callbacks are time-excluded from rank-cost accounting so retry latency doesn't bias the next dispatch decision.
- **NCCL rendezvous-timeout retry with `survivors_ordered`**: if `ncclCommInitRank` doesn't quorum within the timeout, the coordinator picks the largest contiguous survivor subset, rebuilds the comm, and retries. Used at run start and after a mid-run rank death.
- **Every wait is bounded, every death is diagnosed**: bootstrap rendezvous has an idle deadline + a pre-auth reject cap; reduce cycles have stall ceilings on both backends (a wedge escalates to a diagnosed `ShutdownWithSave` instead of a silent overnight hang); a rank exiting non-zero writes a `RankDeathRecord` sidecar (`<save_path>.rank<N>.death.json`) so post-mortems don't start from a bare exit code.
- **`Trainer::builder(...).resume_from(stem)` / `TrainerConfig::resume_from`**: launcher kickoff loads the model / optim / meta bundle, restores `ElCheState` (preserving phase + calibration trajectory), and continues training from the resume epoch. Compatible with controller-written `.meta.json` from any prior run.
- **Coverage-granular checkpoint/resume** (`TrainerConfig::checkpoint_at_epoch` / `DdpBuilder::checkpoint_at_epoch`): a one-shot checkpoint fires on a progressive data-*coverage* boundary (the epoch where any rank reaches the target) rather than on a wall clock. `load_consensus_checkpoint` forges a coherent consensus model from the cluster's `RoundFrame`s; `ddp-bench` exposes `--save-path` / `--resume-from` / `--checkpoint-at-epoch` to exercise the round-trip.

#### ElChe: phase machine, divergence guard, LR-aware meta-controller, EASGD

The cadence balancer that shipped in 0.5.x grew load-bearing additions for production-grade heterogeneous training.

- **`Phase` lifecycle**: `Probe → Warmup → Stable → Mature`, monotonic and `>=`-comparable. Probe = no calibrations yet; Warmup = first few calibrations with sticky anchor; Stable = normal operation with overhead auto-tune + hysteresis; Mature = long-running steady state. Gates the more aggressive controllers (anchor swaps, relax-up) to `>= Stable`.
- **`relax_up`**: in `Phase::Stable` with passing convergence guard, ElChe is allowed to grow the anchor upward, amortizing AllReduce barrier cost over more local SGD when divergence stays bounded. Off by default; opt in via `ElCheConfig::relax_up(true)`.
- **Progressive warmup + 5% dead zone**: anchor decisions inside the dead zone (anchor differences smaller than 5% of current) are no-ops, reducing thrash on near-stable cadences.
- **Window-pressure anchor auto-tune**: the anchor is tuned from per-window *fixed overhead* (reduce + window-fill measured against the bottleneck rank's window wall) toward `ElCheConfig::overhead_target` (default `0.05`), with a growth latch that only grows the anchor while pressure stays below target. Replaces the compute-only overhead signal on the cadence path.
- **Delivered-cost scheduling**: ElChe calibrates on each rank's *delivered* cost — compute + data starvation + transport — not compute-only timing, so a rank on a slow link or a starving loader is no longer over-allocated and left idling at the barrier.
- **Work-weighted consensus averaging** (both backends): each rank's contribution is weighted by its realized work in the window (`w_k` proportional to steps delivered, shaped by `ElCheConfig::gamma`); ranks that delivered zero steps are excluded rather than diluting the consensus. On NCCL the weighting is applied inside the AllReduce via `PreMulSum` (NCCL >= 2.11 required), removing the bookend scale/divide kernels — and unlocking `gamma` on the NCCL backend. Non-learnable f32 buffers (BatchNorm running stats and the like) ride the same sync on both backends, averaged with equal weight among the ranks that stepped in the window — never `gamma`-weighted, so running statistics don't inherit a fast rank's dominance; non-f32 buffers (deterministic integer counters, updated identically on every rank) keep their local value.
- **Convergence guard (`flodl::distributed::ddp_run::convergence`)**: `ConvergenceGuard` trait with three implementations:
  - **`NoGuard`** (passive baseline; always reports `Stable`).
  - **`TrendGuard`** (production default; three-rises-above-threshold rule on weight-space divergence, default threshold 0.05).
  - **`LambdaEstimator`** (MSF λ-hat passive observation for instrumentation; doesn't influence cadence).
  - `ConvergenceAction::{Stable, Divergent, Tighten}` drives the coordinator's response (nudge anchor down on `Divergent`, no-op on `Stable`).
  - Guard state is part of `ElCheState` and round-trips through checkpoint resume.
- **LR-aware meta-controller (`flodl::distributed::lr_event_meta`)**: a layer above ElChe that watches the LR trajectory, anchor trend, and convergence-guard verdicts in a rolling window. Reactively nudges the anchor down on sharp LR drops or sustained divergence, and reports `is_settled()` once the metric stops moving. **On by default**; `ElCheConfig::meta_controller(false)` collects an unconditioned baseline.
- **EASGD elastic averaging**: `ElCheConfig::easgd_alpha(α)` enables EASGD-style elastic blending (0 < α ≤ 1.0) on the `CpuAsync` path. The CPU averaging backend receives a blend of local + center weights instead of a hard overwrite, smoothing divergence in long async runs. Ignored outside `CpuAsync`.

#### Pluggable outer optimizer: SlowMo / DiLoCo

A second optimization loop applied to the work-weighted **consensus** model between the reduce and the broadcast, on top of the inner per-rank optimizers. Opens the door to communication-efficient distributed methods (DiLoCo, SlowMo) without touching the inner training loop.

- **`flodl::distributed::OuterOptimizer`** trait + `OuterOptimizerFactory`: an `outer_step` plus `checkpoint_state` / `load_checkpoint_state` / `resets_inner`. Three impls, all re-exported at the crate root:
  - **`OuterAvg`** (default): stateless identity passthrough — reproduces plain work-weighted averaging exactly, no momentum, no artifact.
  - **`SlowMomentum::new(lr, mu)`**: SlowMo heavy-ball momentum on the pseudo-gradient, continuous inner loop.
  - **`NesterovMomentum::new(lr, mu)`**: DiLoCo-style Nesterov outer step with disposable inner state (`resets_inner()` makes each worker reset its inner optimizer per outer round).
- **`TrainerConfig::outer_optimizer(factory)`** / **`DdpBuilder::outer_optimizer(factory)`** select the variant; built once per site (controller-side on the CPU backend, per-rank replicated lock-step on NCCL).
- **`ElCheConfig::gamma(γ)`**: consensus allocation-weighting exponent (default `1.0` = pre-gamma behavior).
- **Outer-optimizer momentum checkpointing**: momentum-bearing variants persist their slow momentum to a `<stem>.outer.fdl` sidecar (one tensor per model parameter, positional). On CPU the controller writes it; on NCCL an elected rank writes it and every rank reloads its replicated copy on resume, so the resumed outer trajectory is faithful. Stateless `OuterAvg` writes no sidecar.
- **`ddp-bench`**: `--outer-optimizer none|slowmo|diloco` plus `--outer-lr` / `--outer-mu` / `--gamma`, with a loud-error guard on momentum flags passed to `none`.

#### `EpochCallbackPolicy::Fastest`: cost-aware free-compute callbacks

`Trainer::builder().epoch_callback_policy(EpochCallbackPolicy::Fastest)` dispatches per-epoch callbacks (`checkpoint_fn`, `eval_fn`, `metrics_fn`, `epoch_fn`) on the rank with the lowest smoothed_ms_per_batch instead of always pinning to rank 0. On heterogeneous rigs the fastest rank has the most idle time at the sync barrier, so the eval / save runs as free compute rather than stalling the slow rank's next batch.

- Sticky within a run: the dispatcher re-picks only on rank death.
- Supported on the via_coord cluster path; loud-errors on the single-host fallback.
- **`Fastest` is the default**; `Rank(n)` (the previous rank-0 convention) remains available for workflows that need a pinned callback rank.

#### `Trainer::builder().eval_fn(...)` + eval cadence: cluster-aware held-out evaluation

`eval_fn` registers an evaluation closure that the coordinator dispatches per the configured `EvalCadence`, and `eval_result_fn` receives the controller-side scalar result. Heterogeneous-rig-friendly via `EpochCallbackPolicy::Fastest`: eval runs on whichever rank is idle longest at the barrier.

- `EvalFn<M> = Arc<dyn Fn(&M, &Tensor, &Tensor) -> Result<f64> + Send + Sync>`.
- `EvalResultFn = Arc<dyn Fn(usize, f64) + Send + Sync>` (controller-side; receives the rank's reported scalar plus the epoch index).
- Wires through `TrainerConfig::eval_fn` / `eval_result_fn` / `eval_dataset` for the config-bag entry.

#### `Trainer::builder().metrics_fn(...)`: host-side per-epoch callback

The chained `Trainer::builder(...).run()?.join()?` shape was sold as the canonical "just train" form, but anything beyond final-weights-only (per-epoch logging, monitor updates, per-rank metric capture) forced users into a manual `let handle = run()?; while let Some(m) = handle.next_metrics() {...}; handle.join()?` polling loop. `metrics_fn` closes that gap.

- `flodl::MetricsFn` type alias: `Arc<dyn Fn(&EpochMetrics) -> Result<()> + Send + Sync>`. Mirrors the shape of `CheckpointFn`.
- `DdpBuilder::metrics_fn(f)` registers a host-side callback fired once per epoch with the aggregated `EpochMetrics`, after all ranks have reported. Errors are logged to stderr; training continues.
- Composes with `next_metrics()`: the same `EpochMetrics` reaches the callback (if registered) and the polling queue, so users can register `metrics_fn` and keep polling. No deprecation, no fork in docs.
- Transparent 1-or-N GPU: fires identically on the multi-GPU path (coordinator thread, per-epoch as ranks aggregate) and the single-GPU fallback (main thread, per-epoch as training progresses). Single-GPU `next_metrics()` previously returned `None` immediately; that pre-existing transparency gap is closed.
- Single-GPU `run_single` is synchronous by design: runs to completion before returning the `DdpHandle`, so explicit pollers see all queued metrics back-to-back rather than blocking per-epoch.
- The contract is observation-only: callback errors are logged, not surfaced to `DdpHandle::join()`. Early-stop semantics via callback errors is a future enhancement.

#### `fdl probe`: cluster readiness gate

New `fdl probe` subcommand (`flodl-cli/src/probe.rs`) audits a host or a whole cluster for distributed-training readiness before fan-out. Errors loudly on misconfig; the green path is silent enough to use as a CI smoke test.

- **Single-host (`fdl probe`)**: GPU inventory (count, name, sm version, VRAM), libtorch variant + linkage, NCCL availability (host libnccl or Docker-image-bundled), shared-data path resolution, dashboard port availability. Splits results into warnings (informational) vs errors (block dispatch).
- **Cluster (`fdl @cluster probe`)**: SSHes each worker, runs the per-host probe, aggregates. Validates per-worker `arch:` against actual GPU sm versions, checks libtorch variant arch coverage, surfaces NCCL major.minor skew across hosts (the common failure mode on heterogeneous rigs).
- **Docker-aware NCCL detection**: when a worker has `docker: <svc>` set in cluster.yml, the probe reports "via Docker image `<svc>`" instead of erroring on a missing host-level libnccl.so.
- **Output modes**: `--json` for tooling, default human-readable. `--skip-mount` / `--data-path` / `--libtorch-path` for targeted overrides.
- Returns non-zero on errors; zero on green or warnings-only.

#### `fdl nccl build`: libnccl source builder for the LD_PRELOAD bridge

`fdl nccl build` compiles NVIDIA's libnccl from source for the local GPU architectures + auto-detected target version, producing a `libnccl.so.2` that can be `LD_PRELOAD`-ed into libtorch to override the bundled version. Required on heterogeneous-rig clusters when one host's libtorch ships NCCL 2.27.x and another's ships 2.26.x: NCCL refuses handshake across major.minor skew, so the easier side rebuilds.

- Auto-detects the target NCCL tag from the active libtorch variant's `third_party/nccl` submodule version.
- Auto-detects architectures from local GPUs (multi-arch builds supported, e.g. `sm_61 + sm_120` for a Pascal + Blackwell rig).
- Containerized build via `Dockerfile.nccl.source`. 5-15 minutes depending on CPU cores and arch count.
- Output drops under `libtorch/nccl/builds/<tag>-<archs>/lib/libnccl.so.2`. Wire it into a worker via `env.LD_PRELOAD` in cluster.yml.

#### `fdl --gpus`: global GPU scope override

`fdl --gpus <spec> <cmd>` sets `CUDA_VISIBLE_DEVICES` for the dispatched command. Accepted at any argv position; spec is a comma-separated index list (`0,1`) or the `all` shorthand.

- Cluster-aware: on `fdl @cluster <cmd>`, `--gpus` overrides per-worker `local_devices` for the local controller host; remote workers continue to use their cluster.yml-declared devices.
- Non-cluster: maps `--gpus` directly to `CUDA_VISIBLE_DEVICES` for the dispatched subprocess.
- Loud errors on duplicate, missing value, or invalid spec.

#### cluster.yml schema: first-class `controller:` / `workers:` separation

`fdl.cluster.yml.example` and the matching `fdl.cluster-test.yml.example` (testing overlay) now codify the controller / worker separation. The orchestrator host fdl-cli runs on is never a NCCL rank; every rank-carrying host lives under `workers:`.

- `controller:` block: `host` (controller bind, was `master_addr`) / `port` (default 1337, replaces PyTorch's 29500 — the ONE port every cross-host channel shares; worker hosts derive `+4`/`+5` for their host-local rank↔relay loopbacks only) / `path` (controller's view of the shared project root) / optional `docker:` / `arch:` for pre-flight build context / optional `join:` quorum block (see dial-in membership).
- `workers[]`: per-host entries with `host` (worker identifier and default ssh target, was `name`) / `local_devices` (explicit list or `all` shorthand probed at dispatch) / `nccl_socket_ifname` (required for multi-host) / `path` (project checkout dir) / `arch` (libtorch variant subpath under `<path>/libtorch/`) / optional `docker:` / `env:` / `ssh:` sub-block.
- `ssh:` sub-block: groups `target` / `port` / `user` / `identity_file` / `options` (list of `-o Key=Value`) into one launcher-only block. Omit entirely to use system ssh + `~/.ssh/config` defaults.
- `env:` blocks: cluster-scope `env:` applies to every rank child on every worker; per-worker `env:` overrides matching keys. Per-host CUDA_VISIBLE_DEVICES scoping for heterogeneous rigs.
- Orchestrator-only host entries are permitted (worker entry with empty `ranks:`) for clusters where the controller is itself one of the SSH targets but owns no GPUs.
- `cluster-test` env (in-process topology source, no fan-out) replaces the previous test-discovery convention. Exports `FLODL_TESTING_CLUSTER_JSON` to `cargo test` so `flodl::distributed::testing::discover_test_cluster()` reads the same yml the production overlay does.

#### `fdl @<env>` environment selector

The `@<env>` sigil (scan-anywhere before `--`) selects an `fdl.<env>.yml` overlay: `fdl @cluster probe`, `fdl @cluster-test cuda-test-nccl`. It ranks equally with `--env <name>` and `FDL_ENV=<name>`; supplying two that disagree is a loud error, and all three forms must resolve to an existing overlay. Replaces the previous "first positional token matches an overlay" convention, so the first bare token is now always a command (an env may share a name with a command).

#### Variant-shaped `#[derive(FdlArgs)]`: enum subcommands

`#[derive(FdlArgs)]` now accepts an **enum of newtype variants**, turning each variant into a subcommand. Previously the derive only accepted a struct with named fields, so multi-mode binaries hand-rolled their own `while let` argv dispatch (and a separate hand-maintained `usage()` printer that drifted from it). A multi-mode CLI is now one enum whose `main` is an exhaustive `match` — adding a mode is a compile-time exhaustiveness obligation, not a dispatch-table edit.

```rust
#[derive(FdlArgs)]
enum Cli {
    /// Train a model on a dataset
    Train(TrainArgs),
    /// Evaluate a trained model on a test split
    Eval(EvalArgs),
    /// Generate samples (subcommand renamed from the variant ident)
    #[command(name = "gen")]
    Generate(GenArgs),
}

fn main() {
    match parse_or_schema::<Cli>() {
        Cli::Train(a) => { /* ... */ }
        Cli::Eval(a) => { /* ... */ }
        Cli::Generate(a) => { /* ... */ }
    }
}
```

- **Thin dispatcher, full reuse**: the enum derive peels the leading subcommand token and delegates parsing / schema / help to the wrapped type, which carries its own `#[derive(FdlArgs)]`. No new field-parsing — a subcommand *is* a struct. Only single-tuple (newtype) variants are accepted; unit, named-field, and multi-field variants fail at derive time with a pointed error.
- **Self-documenting**: subcommand name = the variant ident kebab-cased (`TrainSubscan` → `train-subscan`), overridable with `#[command(name = "...")]`. Variant doc-comments become the per-subcommand descriptions in the parent `--help` command list.
- **Contextual help**: `<bin> --help` lists the commands; `<bin> train --help` renders only train's flags. `<bin> --fdl-schema` emits the whole tree. Backed by a new defaulted `FdlArgsTrait::render_help_path` method (single-struct CLIs are unaffected — the default forwards to `render_help`).
- **`Schema` grew two additive fields** (`description`, `commands: BTreeMap<String, Schema>`), both `#[serde(skip_serializing_if)]`. A leaf schema serializes byte-identically to before, so existing single-struct consumers (`ddp-bench`, `flodl-hf`, `fdl`'s own commands) and inline `fdl.yml` schemas are untouched. A node is a leaf (`args`/`options`) **or** a branch (`commands`), never both — enforced by `validate_schema`. The shape mirrors the recursive `commands:` map `fdl.yml` already uses, closing the depth-1 asymmetry between the macro and yaml layers.
- **`fdl`-side consumers are tree-aware**: `fdl <bin> --help` lists the binary's subcommands; tail validation descends to the invoked subcommand's leaf (with "did you mean" on a mistyped subcommand); bash / zsh / fish completions complete subcommand names and per-subcommand flags.
- **Arbitrary depth for free**: a variant may wrap another `FdlArgs` enum; dispatch, schema, and help all recurse through tail-recursive delegation with no special-casing.

#### Heterogeneous-rig cluster support

These features came out of a forced heterogeneous topology: a rig crash and OS migration pushed two of three GPUs into a VM, producing a single-machine cluster whose VM rank ran a different libtorch variant from the bare-metal ranks. Every heterogeneous-rig pain point a multi-host deployment would hit (NCCL version skew, per-host libtorch arch, shared-mount conventions, per-host CUDA scoping) showed up inside one box, and shaped the design accordingly.

- **Per-case `libtorch/.active.<case>` pointers**: one libtorch checkout, multiple per-host pointers. The `FDL_LIBTORCH_CASE=<case>` env var selects which pointer file to read; cluster.yml's per-host `arch:` points at the per-host case. Single-host setups keep using bare `.active`.
- **Per-host pre-flight build (`flodl-cli/src/prebuild.rs`)**: `fdl @cluster <cmd>` and any `cluster: true` command auto-build the target binary locally for every remote host before fan-out. Per-host `CARGO_TARGET_DIR=target/cluster/<host>/`, libtorch resolved from each host's variant, CUDA feature derived from the host's `.arch` metadata. Builds run in parallel per host; first failure aborts fan-out. Remote dispatch invokes the prebuilt binary directly (no cargo, no rustc on remote).
- **`Dockerfile.cuda` cuda-rank service**: long-lived sshd container as a VM-equivalent cluster remote. Lets a developer simulate a remote NCCL rank without standing up a real second host. Drops authorized_keys, mounts the project root + libtorch as the production layout, listens on port 2222.
- **libtorch source builds bundle cuDNN sub-libs**: source-built libtorch variants now bundle the matching cuDNN sub-libraries (cudnn_cnn, cudnn_ops, cudnn_adv, cudnn_engines_*, cudnn_graph, cudnn_heuristic) so dlopen-based loading doesn't fall back to the system cuDNN at runtime. Closes a class of mysterious mismatches on heterogeneous-rig deployments where the host cuDNN version differs from what libtorch was built against.
- **`Dockerfile.nccl.source`**: dedicated NCCL build context (see `fdl nccl build` above).

#### `flowbuilder_residual` example

`flodl/examples/flowbuilder_residual/`: minimal residual-block example showing the canonical `fork().also(...).merge()` pattern. Generated SVG (`site/assets/images/flowbuilder-residual.svg`) ships with the site assets for inline embedding in docs.

#### Two-stage streaming prefetch: reader ring + `ram_max_usage`

The streaming `DataLoader` pipeline now runs two stages on CUDA targets: a reader thread fetches batches from the dataset into a bounded pageable-RAM ring while the transfer thread pins and copies to the device. Storage-read latency (network shares, slow disks) overlaps transfer work instead of adding to it, raising the prefetch throughput ceiling from `1/(t_read + t_transfer)` to `1/max(t_read, t_transfer)`; the ring absorbs read jitter. This is a ceiling raise for read-bound pipelines — a pipeline that already keeps up gains nothing, by design.

- **`DataLoaderBuilder::ram_max_usage(f)`** (default `0.50`, clamped `[0.0, 0.90]`, `0.0` = single-stage): fraction of the host's **available** RAM the reader may claim while staging ahead. The ring is sized at each `epoch()` from the kernel's `MemAvailable`, so every other process on the box — permanent fixtures like pinned VM memory and hugepages included — is accounted for automatically and the budget self-adjusts as the box fills or drains; when it cannot fit one batch, the pipeline falls back to single-stage. Per-loader ceiling: multiple CUDA-target loaders on one host should each get a divided fraction.
- **`flodl::sys::mem_info()`**: host RAM probe (`MemTotal` / `MemAvailable` from `/proc/meminfo`), CUDA-free like the rest of `flodl::sys`.
- The VRAM depth governor and the RAM ring bound different resources and stay orthogonal: the governor caps device in-flight (transfer stage), the ring caps host-RAM in-flight (reader stage). CPU-target loaders keep the single-stage pipeline (their batch channel already is the read-ahead buffer), as does the coordinator-paced distributed batch path.

#### Augmentation as deterministic repeated picks

Augmentation is now a first-class, reproducible schedule concept instead of per-call randomness hidden in the dataset. Two orthogonal knobs, on `DataLoaderBuilder` (solo) and `TrainerConfig` / `DdpBuilder` (DDP) alike:

- **`.augment(k)`** — each sample appears `k` times per epoch, spread by the shuffle. Pure scheduling: the epoch permutation runs over the pick space `len() * k`, and a pick decodes intrinsically as `(pick / k, pick % k)` = (sample, repeat) — so the decode survives re-partition, rank death, and resume with no side tables. Every pick fetches the same raw bytes (staged once across the tiers) and counts as one unit of realized work; ElChe, the coordinator ledger, the wire format, and coverage-granular resume all run in pick space unchanged. Composes with the built-in samplers; combining with a custom sampler errors loudly.
- **`.transform(f)`** — the deterministic delivery transform, the sanctioned home for augmentation. Receives each delivered batch (raw rows, already on the target device, freshly assembled) plus one `PickKey { sample, repeat, epoch, seed }` per row; derive per-view randomness from `PickKey::rng()` (stateless, frozen mixing constants — checkpointed runs reproduce their augmentation exactly). Runs live on every delivery and never writes back: the staging tiers (RAM cache, disk stage, VRAM pool) retain raw samples only, so a VRAM-pooled sample uploads once and derives its `k` views on device. Batch assembly always materializes fresh storage, so even an in-place transform cannot corrupt the retained raw bytes.

With `k = 1` and no transform, everything is byte-identical to before — the epoch permutation scheme is unchanged (and now lives in one place, `epoch_permutation`, shared by the solo sampler and the DDP partition expansion). The flow window's drop-behind became multiplicity-aware: a sample stays resident until its last advised pick instead of popping on first hit.

`ddp-bench` exposes the pair as `--augment <k>` (schedule multiplicity) and `--augment-noise <amp>` (a `PickKey`-keyed additive input-noise transform, so the k views carry distinct bytes — the A/B arm for the keyed delivery path under real multi-rank runs).

#### Sample cache: later epochs read from RAM, not storage

`DataSet`-backed streaming loaders now retain samples in a read-through RAM cache as epoch 1 reads them; later epochs hit RAM instead of re-reading storage. The cache is keyed by sample identity, which makes staged content reshuffle-proof: a reshuffle changes only the access order, never the content set. When the budget covers the whole dataset, storage is read exactly once for the entire run.

- **`DataLoaderBuilder::sample_cache(bool)`** (default enabled): the off switch for single-pass training over data far larger than RAM, where retained samples are never revisited. Opaque `BatchDataSet` loaders have no sample layer and are never cached (their batching is the dataset's own affair).
- **Admission: fill until full; evict only under re-partition.** On the solo path every epoch touches each sample exactly once in a fresh random order, so a cache holding K of N samples hits K/N of reads for any eviction policy; admit-until-full delivers that with zero eviction churn and the solo loader never evicts. On the DDP staging path the coordinator re-partitions the permutation every epoch, the K-set tie breaks, and the stager evicts in next-use order against its advisory — prior-epoch leftovers (absent from the visible future) go first, sooner-needed samples take their room, and the tier declines when everything held is needed sooner (the flow window's exact rule, now spanning both tiers). A shrinking RAM budget stops new admissions; retained content drops only through that next-use order, never as a blanket flush.
- Shares the `ram_max_usage` budget with the reader ring: the ring is capped to a small flow-buffer depth while the cache is active (jitter absorption saturates fast; retained samples pay again on every later epoch), the cache gets the remaining headroom, refreshed each `epoch()` against `MemAvailable`. Reads take an uncontended per-slot read lock; admission is set-if-empty under the slot's write lock, eviction empties the slot for reuse.

#### Disk stage: a local-drive tier under the sample cache

`DataLoaderBuilder::disk_stage(gb)` (default 0 = off) adds a local-disk overflow tier: samples the RAM cache declines are staged once in an append-only pack file, and later epochs read them at local-disk speed instead of source speed. With RAM + disk covering the dataset, a network-mounted source is read exactly once per run. Lookup cascades RAM → disk → source.

- **One pack file, not one file per sample**: sequential append is every drive's fast path, and the offset index reuses the same lock-free set-once-slot pattern as the RAM tier (positioned reads, no shared seek state). The per-tensor layout reuses the checkpoint codec — one serialization format in the codebase, not two.
- **`DataLoaderBuilder::disk_stage_dir(path)`** overrides the location (default: system temp dir). A RAM-backed directory (tmpfs — `/tmp` frequently is) triggers a loud warning, since a stage that spends RAM defeats its purpose. The pack file is ephemeral: removed when the loader drops.
- **Failure split**: a read error on the pack file surfaces (it is a real disk problem); a write error never fails training — the sample is already in hand — it latches the stage off loudly and the run continues source-backed. Budget-full is a plain decline, not a failure.
- Requires the sample layer: `build()` errors loudly on an opaque `BatchDataSet` loader or with `sample_cache(false)`. Pays exactly when the source is slower than local disk (network mounts) and data is revisited; for a dataset already on local SSD the source is the disk and the stage buys nothing.

#### Reservation staging: cluster ranks read their data ahead of training

Cluster workers now stage their upcoming training data ahead of the training frontier. At each progressive epoch start the coordinator sends every rank a `StageAdvisory` — its reservation view as certainty-ordered permutation spans: the rank's own reserved span first (deterministic for the whole epoch), then the other spans' window-sized tails (the truing margins, whose final owner can move under throughput drift). A background stager thread on each rank walks the advisory through the shared permutation and reads the samples into a sample-keyed staging tier that the live prefetch path shares: batch fetches become read-through (staged rows served from the tier, misses fetched in one bulk call and admitted on the way out, row order preserved).

- **Advisory, never authoritative**: chunk allocation remains the only execution authority. Staging may overlap across ranks near reservation boundaries — margins are staged by several ranks on purpose — but only allocated work executes, so staged-and-allocated-elsewhere data needs no invalidation protocol; it just ages out. Latest advisory wins.
- **Cross-epoch**: advisories carry run-stream segments — this epoch's spans plus the predicted next epoch's (same ratio table over the next permutation, computable before its pool exists) — so the stager walks into the next epoch while this one trains. An epoch is just where the order function switches, never a data-movement event; staged content is reshuffle-invariant (sample-keyed).
- **Refreshed on the window clock**: advisories re-emit at each reduce boundary (the same clock reservation truing rides), so boundary drift and ratio changes reach the stagers without a timer of their own.
- **Consumption-proportional host shares**: each advisory carries the current schedule; a worker takes the `ram_max_usage` share (default 0.50) of its host's live `MemAvailable` — anchored by what its tiers already retain, so the budget holds steady as they fill — and splits it among co-hosted ranks (from the cluster envelope) by their schedule counts — `budget ∝ consumption rate` gives every rank the same seconds of lookahead, which matters acutely on hosts whose combined VRAM exceeds RAM. Available-based on purpose: a total-anchored cap reads permanent fixtures (pinned VM memory, hugepages) as pressure and zeroes the budget on exactly the heterogeneous hosts staging targets. Budgets refresh with each advisory; a shrink stops new admissions, never drops staged content.
- **Dormant until advised**: the tier has budget 0 (pure pass-through, one atomic load per row) until the first advisory arrives, so single-device runs, tests, and non-progressive modes never pay for it.
- **A flow window with next-use-priority eviction rides beyond the pinned tier**: what the pinned tier declines lands in a bounded stream pool instead of wasting the read. Training consumption pops entries (the frontier passing IS the drop-behind), admission under pressure evicts the entry whose next use in the advised stream is farthest — keep what recurs soonest — and the stager pauses before fetching rather than reading past a full window, so no source read is ever spent on a sample nothing can retain. Next-use positions re-key from each advisory on the window clock.
- **`FLODL_STAGER=off` kill-switch + verbose observability**: setting `FLODL_STAGER=off` (or `0`) in a worker's `env:` block leaves the dataset unwrapped and spawns no stager — the clean A/B lever for measuring staging against the same binary. Under `-vv`, each advisory logs its segment count, sample count, pinned/stream budget split, and cumulative staged count, so a silent stager (zero budget, empty advisories) is visible instead of indistinguishable from a working one. User `FLODL_*` variables pass through worker `env:` blocks; only the framework's internal channel (`FLODL_INTERNAL_*`, `CUDA_VISIBLE_DEVICES`, and the other launcher-owned keys) is reserved.

#### Device-resident sample pool: the VRAM tier of the sample cascade

Streaming `DataLoader`s on CUDA targets now retain as many samples as fit in leftover VRAM and assemble batches by gathering retained rows on device instead of uploading them — H2D traffic shrinks by the hit rate. The middle ground between resident mode (whole dataset on device) and streaming (every byte crosses PCIe every epoch): one byte over the resident budget no longer costs the whole dataset's residency.

- **Admission is capture-at-delivery**: samples enter the pool by device-to-device copy out of batches that were uploaded anyway (transfer stream, raw pre-augmentation rows), so filling costs zero extra transfers — the first epoch populates the pool as a side effect. Sample-keyed and admit-until-full: under per-epoch reshuffle any K retained samples of N hit K/N of reads, the same argument as the RAM sample cache one tier up.
- **Sizing is automatic and conservative**: dormant until the honest post-first-step probe (the prefetch governor's latch, or the rank worker's explicit signal on coordinator-paced paths), then one budget decision from measured free VRAM minus a flow-buffer in-flight reserve minus a safety margin — with a capacity tier active, prefetch depth is a rate-matcher, not a capacity claim, the same arbitration as the reader ring's flow-buffer cap one tier down. No fraction knob; **`DataLoaderBuilder::vram_pool(false)`** is the off switch.
- **Never the reason a step OOMs**: storage is slab-chunked (partially freeable); on transient data-plane OOM the governor's target halving runs first and slab eviction is the last resort, after which the budget latches down. The training-step path is untouched.
- Batch assembly restores exact caller row order (pooled rows gathered, misses uploaded, stitched on the transfer stream so the delivery event covers everything); per-epoch `-v` telemetry reports rows served on-device, H2D bytes saved, and pool occupancy.
- **DDP rank workers pool too** — the path that never had a resident mode at all (cluster ranks re-uploaded every byte every epoch, whatever the dataset size). The worker signals the pool's budget moment itself at the first post-calibration plan boundary (measured activation peak = its honest probe). Knobs: `TrainerConfig::vram_pool` / `DdpBuilder::vram_pool` / `DdpRunConfig::with_vram_pool` (default on), plus a per-worker `FLODL_VRAM_POOL=off` env kill-switch for A/B runs. On a heterogeneous cluster the fast rank serves 100% of its rows on-device once the pool fills (~400MB of H2D saved per epoch on CIFAR-sized data); reservation-drift misses on slower ranks stay small and keep being captured.

#### One memory-budget policy: the data-plane knobs reach the trainer

All memory sizing across the data cascade — reader ring, sample cache, DDP stager tiers, prefetch depth, VRAM-pool flow reserve — now runs through one policy module (`flodl::data::budget`), so the solo loader and the DDP rank workers can never drift apart on how a machine's memory is priced. Previously the DDP side hardcoded its own copies (a fixed `0.90` VRAM fraction, a fixed `available/2` host-RAM share) with no user knob.

- **`TrainerConfig::with_vram_max_usage(f)` / `with_ram_max_usage(f)`** (and the chained `DdpBuilder` twins `.vram_max_usage()` / `.ram_max_usage()`): the same two memory knobs the solo `DataLoaderBuilder` has always had now govern each DDP rank's data plane — prefetch channel + device sample pool for VRAM, staging tiers for host RAM — with identical defaults (`0.90` / `0.50`) and clamps. Co-hosted ranks split the host-RAM share in proportion to their schedule.
- **`with_sample_cache(b)` / `with_disk_stage(gb)` / `with_disk_stage_dir(path)`** complete the knob parity: the solo loader's remaining staging knobs now reach the trainer too. `sample_cache(false)` pins each rank's retained cache at zero (the flow window keeps the whole staging share); `disk_stage(gb)` attaches the same RAM → disk → source overflow cascade under each rank's staging cache, spilling to an ephemeral pid-unique pack file per rank, so co-hosted ranks sharing a temp directory never collide. `ddp-bench` exposes both as `--sample-cache` / `--disk-stage`.
- **`FLODL_VRAM_POOL=off` now reaches the solo loader too**: the runtime kill-switch (and the pool's default) had one definition on the DDP path and none on the solo path, so a scripted A/B silently no-op'ed on single-GPU runs. One shared parse + one shared default now serve both.
- **One anchored budget law**: every host-RAM budget is `r × (MemAvailable + held)` — held bytes added back *before* taking the share, so the cap is a fixed point of the run's starting headroom. Both single-sided variants are real bugs this rules out: full add-back after the share ratchets toward 100% of RAM, no add-back self-starves the tier as it fills.
- **Honest retention pricing** (`Tensor::storage_nbytes`, new public API): a retained view is charged the whole backing buffer its clone pins, not its logical size — and a view whose backing storage exceeds twice its logical size is materialized into owned storage at admission instead (a 4KB row `select`ed from a 500MB row-group no longer pins the 500MB, in any tier). The stager also re-prices its room-check estimate after every fetch (running max) instead of trusting the first sample's size, so variable-size datasets (NLP sequences, mixed resolutions) can no longer slip past the room checks and waste source reads on samples nothing can retain.
- **Install-chunk truce**: on the plan boundary where a DDP rank's VRAM pool takes its one-shot budget, the prefetch channel collapses to the pool's flow-buffer reserve — previously both sized themselves against the same free VRAM on that one chunk (transient-OOM window); from the next boundary the probe sees pool bytes as used and the depth re-sizes honestly. The flow-reserve formula itself now has a single definition both budget-signal paths share.

#### Per-sample datasets end to end: `DataSet` trainer entries + disk-backed readers

The one-method `DataSet::get(index)` contract now reaches every training entry — implement how to read one sample (or use a shipped reader) and the framework owns batching, RAM caching, disk staging, reservation staging, and distribution. Storage-backed data (local files, network mounts, beds larger than RAM) trains through the same entry as RAM-resident tensors.

- **`TrainerConfig::from_dataset(ds)`** and **`DdpBuilder::sample_dataset(ds)`**: per-sample twins of the `BatchDataSet` entries. Rank workers read samples through the shared staging tier and stage them ahead of the training frontier per their reservations.
- **`flodl::data::batch_dataset_from(ds)`**: public promotion of any `DataSet` into an opaque `BatchDataSet` (position-wise stacking) for APIs without a native per-sample entry.
- **`flodl::data::FixedStrideRecords`**: a file as `count` fixed-size records with lock-free positioned reads (`read_exact_at`), optional leading header. Many raw dataset distributions are exactly this shape; a custom format needs a parse closure, not a loader. Misaligned file sizes error loudly.
- **`flodl::data::datasets::Cifar10Disk`**: CIFAR-10 read per sample from the raw batch files (the binary distribution is already a 3073-byte-stride record file — no repacking). `open(paths)` / `open_train(dir)` / `open_test(dir)`; sample output batch-stacks identically to the RAM parser.
- **ddp-bench `--data-source ram|disk`**: the CIFAR models (`resnet`, `resnet-graph`) can train from per-sample storage reads, exercising the staging tiers against real read paths; models without a per-sample reader reject the flag loudly.

#### Per-rank data reservations in progressive dispatch

The progressive chunk pool now partitions each epoch's permutation into contiguous per-rank spans sized by ElChe throughput ratios (equal until calibrated), and serves each rank's chunks from the front of its own span instead of a shared arrival-order cursor. Each rank's upcoming data is thereby deterministic for the whole epoch — the foundation for staging it ahead of the training frontier. Throughput drift is absorbed by reservation truing: a rank that out-runs its span steals from the tail of the largest-residue span (the boundary moves, the coverage books stay exact). Training semantics are unchanged — a reservation table is a deterministic partition of the globally reshuffled order where the old cursor produced a nondeterministic one — and the dispatch discipline (one chunk in flight, schedule-exact window sizing, reduce/epoch barriers, coverage-granular checkpoint/resume, dead-rank reclaim) is untouched.

#### Misc additions

- **`flodl/examples/auto_promote`**: plain-binary multi-GPU — a minimal `Trainer::builder().run()` binary demonstrating that the same code auto-promotes to process-per-rank on a multi-GPU host with zero cluster config.
- **`FLODL_NET_TIMEOUT_SCALE`**: one env knob scales the whole cluster deadline set (handshakes, rendezvous, heartbeat windows, join window) — for slow rigs, congested CI, or debugger-attached runs.
- **`flodl::distributed::chunk_pool`**: extracted from coordinator; reusable chunk-pool dispatch primitive that reclaims a dead rank's in-flight data chunks and redistributes them deterministically across the surviving ranks.
- **Network-aware logging (`flodl::log`)**: rank-scope prefixes that survive the cluster fan-in (`[rank=N]`-style markers preserved when controller forwards logs to launcher stdout).
- **Optimizer `state_dict_keys()`**: Adam / RMSprop / SGD expose state-dict key listings for checkpoint introspection.
- **Optimizer state persistence now covers every optimizer**: `Adagrad`, `RAdam`, and `NAdam` gained `Stateful` (save/load of moments, per-parameter step counts, and hyperparameters), so `Optimizer::save_state_to` no longer returns "unsupported" for them — the cluster save-on-unrecoverable-failure flow can now persist any optimizer's `.optim` sidecar and resume faithfully instead of silently restarting their moment estimates. Their state files are born under the new self-identifying `FDLO` header (see Fixed), so no migration ever applies to them.
- **Scaled CUDA NCCL communicator wiring**: `NcclComms` better handles N-rank topologies; `NcclRankComm::split` is the per-thread comm seam used by the `Ddp::wrap` / cluster-worker thread-test path (production ranks use per-process `NcclRankComm::init_rank`).
- **`Tensor::copy()`**: a deep-copy primitive that allocates fresh storage (the flodl spelling of PyTorch's deep `.clone()`). flodl's `Clone` is a *shallow* alias sharing storage (libtorch's copy constructor), so an in-place op through one handle is visible through every alias; `Tensor::copy()` returns an independent, owned duplicate for the cases that need it (optimizer state seeded from a gradient, snapshots held across later mutation). The `Clone` and `Variable::data` docs now flag the shallow-vs-PyTorch divergence and point to it.

### Changed

#### CUDA stream/event primitives moved to `tensor` (BC-transparent)

`CudaStream` / `StreamGuard` / `CudaEvent` / `CudaEventFlags` are device-runtime tools, not DDP machinery, and now live in `flodl::tensor::{cuda_stream, cuda_event}` — below every consumer, so the data layer no longer reaches into `distributed` for them. Every existing path keeps working: the crate-root re-exports are unchanged and `flodl::distributed::cuda_stream::…` / `flodl::distributed::{CudaStream, …}` remain valid re-exports of the same types. `Variable::id()` and `Buffer::id()` are also new: the stable shared-cell identity that parameter/buffer collection dedups on, previously hand-rolled as `Rc::as_ptr` casts at every site.

#### `Module` trait de-cycled from `graph` and `distributed`

The `nn::Module` trait no longer names types from higher layers — a `Linear` no longer transitively knows what a DDP metrics struct or a `Graph` is. Two changes, one of them BC-relevant:

- **`Module::as_graph` → `Module::as_any` + `graph::GraphExt`** (BC-relevant): the trait's `as_graph(&self) -> Option<&Graph>` hook is replaced by a graph-agnostic opt-in identity hook `as_any(&self) -> Option<&dyn Any>` (default `None`; composite types return `Some(self)`, transparent wrappers may present their inner composite). The ergonomic `.as_graph()` call survives unchanged as `flodl::graph::GraphExt` — a blanket extension over every `Module`, re-exported at the crate root, so `use flodl::*` code keeps compiling verbatim. Migration is only needed for external `Module` impls that *overrode* `as_graph`: override `as_any` instead (return the graph you used to return, as `&dyn Any`).
- **`EpochMetrics` moved to the leaf `flodl::metrics` module** (BC-transparent): the type `Module::aggregated_metrics_slot` is typed against is plain metrics data, not distributed vocabulary, and now lives below every consumer. `flodl::EpochMetrics` and `flodl::distributed::EpochMetrics` remain valid re-exports of the same type — no path breaks, no behavior change.

The last `graph` → `distributed` edge (`Graph`'s embedded `cluster_ddp` / `cluster_el_che` state and the cluster branches of `Graph::step`) was the engine of the self-driven `Trainer::setup` tier; it left wholesale when that tier was removed (see Removed). `Graph::step` is now single-device only, and `graph` no longer imports `distributed` DDP types.

- **`DataSet::get` / `BatchDataSet::get_batch` purity contract, stated and probed**: both traits now document that sample content must be a pure function of the index — the staging cascade (RAM sample cache, disk stage, VRAM sample pool, all on by default) retains samples by index and re-serves them on later epochs, so per-call randomness (augmentation inside the dataset, the PyTorch `__getitem__` convention) is silently frozen at its first realization and served for the rest of the run. Debug builds now probe the contract — the first staged fetch is fetched twice and compared (NaN-tolerant), panicking with the full explanation on divergence; release builds skip the probe entirely. Augmentation belongs downstream of the loader as a deterministic on-device transform (e.g. a graph `.map` stage); first-class support for augmentation as deterministic repeated picks is designed in `docs/design/data-cascade.md`.

#### Distributed layer: in-process thread-per-GPU DDP engine removed

The in-process multi-replica DDP machinery on `Ddp`, `Graph::distribute`, and the `DataLoader::distributed` mode is no longer the production multi-GPU path. The new launcher trampoline + cluster coordinator path is the canonical one and auto-promotes on 2+ visible GPUs. `Trainer::run` / `Trainer::builder` always go through the cluster path on `cfg(not(test))` when multiple GPUs are visible.

The in-process multi-GPU **orchestration engine** that briefly survived as the `cfg(test)` multi-GPU harness — the thread-based `Coordinator` plus the multi-worker branch of `DdpHandle::launch` — is now removed outright. `DdpHandle::launch` keeps only the launcher-trampoline path (auto-promote / cluster) and the single-device fallback; reaching it with 2+ GPUs and no cluster envelope (only possible under `cfg(test)`) is a loud error. flodl's own multi-GPU validation moved to the `ddp-bench` binary (real process-per-rank, exercised by `fdl cuda-test-nccl`); thread-based multi-GPU survives only as the lower-level `Ddp::wrap` primitive (GAN / RL / manual control) and the coordinator's own thread-driven control-protocol tests.

- **`Coordinator` / `CoordinatorBuilder` removed**: the in-process coordinator and its builder are gone (they were never part of the `flodl::distributed` public re-export — `Trainer` / `DdpHandle` / `DdpBuilder` / `GpuWorker` are the surface). The cross-process `ClusterCoordinator` is the only coordinator now.
- **Single training entry**: `Trainer::run` / `Trainer::builder().run()` work identically on 1 GPU, N GPUs on one host, or N GPUs across hosts. No code change to scale up.
- **`Graph::distribute` simplified**: the cross-replica gather pipeline, the per-replica `named_trace_buf` plumbing, the host-side `Rc<RefCell>` choreography are gone from the production path. Re-introduced only in `cfg(test)` for the `Ddp::wrap`-driven test suite.
- **`DataLoader::distributed` mode removed**: each rank child instantiates its own loader against its own dataset shard; proportional sharding is computed from `ElCheConfig::partition_ratios` (or auto-balanced) by the coordinator and pushed to workers as part of the epoch plan.

This is the largest pre-1.0 API break in flodl's history. The motivation is dead-simple: the threaded model could not survive a rank dying, could not span a host, could not give the user a per-rank log stream, and forced the entire process to share libtorch's per-process CUDA context. Multi-process solves all four at once.

**Migration note**: most call sites that drove `Ddp::*` or `Graph::distribute` directly migrate to `Trainer::run` / `Trainer::builder().run()` as a one-line swap. The same swap unlocks multi-host scaling at no extra cost (auto-promote on 2+ visible GPUs, opt-in to multi-host via `fdl.cluster.yml` or `TrainerConfig::cluster(FullCluster)`). `Ddp::wrap` remains available for callers that explicitly want the thread-per-GPU path (single-process testing, GAN / RL patterns that need direct replica control) — **with a new per-rank signature**: `Ddp::wrap(&model, device, global_rank, &rdv)` wraps ONE replica against a shared `TcpRendezvous` (the same primitive each cluster rank uses internally), replacing the old whole-process form that owned every replica at once.

#### cluster.yml: `master_addr` / `master_port` → `controller.host` / `controller.port`

The previous flat top-level `master_addr` / `master_port` keys are replaced by the structured `controller:` block (`host` / `port` / `path` / `arch` / `docker`). `name:` on each worker is renamed to `host:` to match the controller key. SSH knobs are grouped into an `ssh:` sub-block instead of living as flat `ssh_*` fields on the worker.

Migration: see `fdl.cluster.yml.example` and `fdl.cluster-test.yml.example` for the canonical layout. `fdl probe` warns on legacy keys.

#### `ddp-bench`: unified harness, eval-cost separation, cluster-aware loader

The benchmark crate was overhauled to drive the new cluster path and to surface ElChe / cadence behavior cleanly.

- `run_sync` collapsed into `run_unified`: one harness for every cadence mode, parametrized by the same `ElCheMode` enum as production.
- `run_baseline_solo`: single-GPU baseline with eval-cost separation so reported speedups don't smuggle eval overhead into the train-time denominator.
- `--partition-ratios`: explicit per-rank ratio passthrough (`flodl::DdpBuilder::partition_ratios`).
- `--epoch-callback-policy`: pick `Rank(n)` or `Fastest` from the CLI.
- Per-rank schedule reporting + train-only-aware speedup in the analyze + report passes.
- Cluster-aware: `ddp-bench` skips the dataset load on the launcher process (the launcher exits without training), so cluster fan-out doesn't pay the dataset cost twice.
- Analysis layer split into `analyze/{fit, log, msf, timeline}` + `report/{mod, elche, msf, tables}` submodules.

#### Internal: large modules split into per-file submodules

`cluster_coordinator.rs` (4227 LOC), `ddp_run/orchestrator.rs` (3000+ LOC), `ddp_run/worker.rs` (2796 LOC), `ddp_run/tests.rs` (4368 LOC), `cluster_coordinator/tests.rs` (2964 LOC), `graph/graph_tests.rs` (2670 LOC), `flodl-cli/src/config.rs` (2995 LOC), `flodl-cli/src/main.rs` (1958 LOC) are now multi-file submodules. Tests for `flodl::autograd`, `flodl::nn`, `flodl::tensor`, `flodl::nn::checkpoint`, `flodl::graph::tree`, `flodl::distributed::cluster`, `controller`, `cpu_reduce`, `nccl`, `wire`, `flodl-hf::models::{bert, distilbert, deberta_v2}`, `flodl-hf::safetensors_io` were extracted to sibling `*_tests.rs` files. No behavior change; per-file diffs become reviewable again.

#### Dashboard binds loopback by default

The live training dashboard (an unauthenticated HTTP server) previously bound `0.0.0.0`: anyone who could reach the host could read training metrics. It now binds `127.0.0.1` by default; view it remotely through an SSH tunnel (`ssh -L <port>:localhost:<port> <host>`), or set `FLODL_DASHBOARD_BIND=<addr>` to widen the bind explicitly — a non-loopback value prints a loud no-auth warning.

#### docs.rs gate: strict local pre-commit canonical check

`make docs-rs` now runs a CI-parity pass with `RUSTDOCFLAGS="-D warnings"` against every published crate (`flodl`, `flodl-cli`, `flodl-hf`, `flodl-cli-macros`) on stable + nightly. It is the canonical strict pre-commit gate; `fdl doc` is the in-Docker CI-strict gate; `fdl ci` is the full CPU job orchestrator. Mismatches between the three were a recurring source of "docs build locally, fail on docs.rs" surprises.

#### Mixed-precision (`amp`) API: `cast_parameters` now fallible; autocast query gains a device variant

`nn::cast_parameters` returns `Result<()>` (was `()`). It previously swallowed a per-parameter `to_dtype` failure silently (leaving a half-cast, mixed-dtype model that failed cryptically much later) and is now all-or-nothing: every conversion is computed first, so a failure returns `Err` with no parameter mutated. Callers add `?` / `.unwrap()`. `nn::is_autocast_enabled_for(device_type)` is new, the general form of `is_autocast_enabled()` (which stays as the CUDA shorthand), mirroring `AutocastGuard::for_device`.

### Deprecated

- The flat `cluster.yml` schema (`master_addr`, `master_port`, top-level `ssh_*` on workers) is deprecated in favor of the structured `controller:` / `workers[].ssh:` layout. `fdl probe` flags legacy keys with migration hints. Removal targeted for a future release.

### Removed

- **The self-driven setup tier** - `Trainer::setup()`, `Trainer::setup_with()`, `Trainer::setup_head()`, `Trainer::setup_head_with()`, and the `DdpConfig` config bag. It was the only path that scheduled without the controller (no convergence guard, meta-controller, outer optimizer, elastic membership, or checkpoint orchestration), so its self-driven replicated ElChe brain could drift from the controller's. Its user-owned-loop ergonomics return, controller-authoritative, as the **cooperative tier**: `Trainer::builder(model_factory, optim_factory, train_fn).into_worker()` yields a `Worker` you drive yourself (`next_plan` / `next_batch` / `step` / `finish`) while the controller owns cadence, partition, eval election, and checkpointing. For `flodl-hf` task heads (which now `impl Module` directly), `setup_head` is replaced by driving the head through `Trainer::builder(head_factory, optim, |head, batch| head.compute_loss(...)).into_worker()` or `.run()`. Also removed with the tier: the graph-embedded `cluster_ddp` / `cluster_el_che` state and cluster branches of `Graph::step` (now single-device only), and `Graph::set_loss_fn` / `has_loss_fn` + the `LossContext` type (the vestigial distributed-gather loss hook, which had no remaining driver). `HasGraph` stays. See `docs/design/trainer-execution-tiers.md`.
- The `0.3.0`-deprecated compatibility surface is gone: the `AsyncDdp` / `AsyncDdpBuilder` / `AsyncDdpConfig` type aliases and the `DdpHandle::auto()` / `DdpHandle::auto_with()` / `DdpHandle::builder()` constructors. All of them had pointed at `Trainer::builder()` for two minor releases; migrate any remaining call to `Trainer::builder(...)` (chained setters) or `Trainer::run(...)` (config bag).
- **The NCCL async mode** (`ddp-bench`'s `nccl-async`, the `Async` policy on the NCCL backend). Cross-epoch lookahead on NCCL delivered near-zero real-world speedup over `nccl-cadence` while complicating the dispatch path; `cpu_async` is the genuine async mode (decoupled averaging on a separate channel). `ElCheMode` never carries an `NcclAsync` variant.

### Fixed

Fixes below correct behavior that shipped in 0.5.x. Bugs born and fixed inside this release cycle carry no entry — the feature sections above describe the delivered state.

- **The NCCL backend never re-synced non-learnable buffers after startup**: the periodic NCCL sync AllReduced parameters only, while buffers (BatchNorm running mean/var and the like) got a single broadcast at formation and then drifted apart as each rank's forward accumulated statistics over its own data partition. Weights stayed in consensus — the visible signal — so the drift was silent: eval and the consensus checkpoint used the elected rank's own diverged running stats, and the CPU backend (which does average buffers) produced checkpoints with different buffer semantics than NCCL ones. f32 buffers now ride every NCCL sync as a second `PreMulSum` collective, averaged with equal weight among the ranks that stepped in the window (matching the CPU backend exactly); the abort-recovery retry restores buffers alongside params so a peer-death mid-collective can't leave torn running stats behind.
- **Streaming loader could deadlock at an epoch boundary (in-flight accounting race)**: the prefetch worker publishes a batch into the epoch channel *before* counting it against the depth governor, so a consumer that received the batch, finished the epoch, and started the next one inside that window had the straggling `sent` increment land after the new epoch's counter reset — phantom in-flight work that, at the pre-warm-up depth target of 1, parked the worker at the governor gate forever while the consumer blocked on an empty channel. Needed an unlucky preemption between two adjacent instructions, so it surfaced as a rare, load-dependent hang (constrained-core CI, oversubscribed test hosts) rather than anything reproducible. The per-epoch counter reset now rides the `StartEpoch` command into the worker itself, where the command channel orders it after any straggler by construction — the consumer arms the epoch (target + abandon latch), the worker owns the flight counters.
- **`flodl-sys` FFI strings typed `*mut c_char`, unbreaking Linux aarch64**: every extern fn returning a C `char*` (292 signatures — error strings, `flodl_free_string`, the device-name buffer) was hardcoded `*mut i8`. `c_char` is `i8` on x86_64 but `u8` on Linux aarch64, so every `CStr::from_ptr(err)` failed to compile on ARM (Apple-Silicon Docker, Graviton, Jetson-class hosts). The signatures now say `*mut c_char` at the source — correct by construction: on x86_64 this is a type-alias no-op (zero ABI or behavior change), on aarch64 the crate now simply compiles (`cargo check --target aarch64-unknown-linux-gnu` verified), and any future `CStr::from_ptr` call site is portable without remembering a cast. Supersedes the four per-site `as *const c_char` casts from the Apple-Silicon support PR (thanks @newQuery — the casts were exactly right; this moves the fix to the root so no fifth site can ever miss it).
- **The VRAM budget layer read the memory probe inverted, disabling adaptive prefetch and resident mode on mostly-free devices**: `cuda_memory_info_idx` returns `(used, total)` — as its documentation says — but `prefetch_depth_from_vram` and `can_fit_resident` destructured it as `(free, total)`. Every derived quantity flipped: the emptier the card, the smaller the computed budget. In practice a mostly-free device produced a ~zero prefetch budget (streaming loaders fell back to minimal depth; DDP rank workers never created their prefetch pipeline at all and ran the synchronous fetch-and-upload path), and `can_fit_resident` concluded that almost nothing fit, so resident mode silently never engaged on CUDA targets. Both call sites now read the contract; adaptive prefetch depth, resident-mode auto-detection, and the rank workers' async data pipeline all size from the device's real headroom.
- **`Monitor::new` initialized CUDA in every process, including the launcher**: constructing a monitor probed hardware and started resource sampling through the CUDA runtime (`cudaMemGetInfo` on every device creates a primary context, pinning VRAM for the life of the process). The documented "one `Monitor` at the top of the training code" pattern therefore violated the no-CUDA-before-`Trainer::run` rule in every user binary, and the launcher's dashboard sink did the same internally, squatting VRAM on all GPUs for the whole run. GPU identity now comes from nvidia-smi (`sys::detect_gpus`), live metrics from NVML, and the only context-dependent read (caching-allocator reserved bytes) is gated on `tensor::cuda_has_primary_context`, a new context-state query that never creates one. Constructing and polling a `Monitor` is now CUDA-free anywhere. Along the way: the NVML utilization poller was indexing cards by CUDA runtime index, but NVML ignores `CUDA_VISIBLE_DEVICES`, so a rank scoped to `CUDA_VISIBLE_DEVICES=1` was polling the wrong card's utilization; the monitor now resolves physical indices for NVML queries.
- **Autograd silently swallowed FFI failures**: `Variable::set_grad` and `zero_grad` discarded errors (a failed write let the optimizer step with the unclipped / still-scaled / stale gradient), `detach()` fell back to a still-attached clone, `Tensor::grad()` mapped errors to `None` (indistinguishable from "no gradient"), and `Variable::new(tensor, true)` on an integer dtype silently returned a non-tracking variable that trained nothing. All five now panic with a named message; the integer-dtype case matches PyTorch, which raises the same error. None of these paths can fail for a correct program with valid tensors.
- **C++ exceptions could unwind through the FFI boundary (undefined behavior)**: ~55 shim entry points had no try/catch at all — including the fused Adam/AdamW kernels, the CUDA graph/event/stream families, and the tensor accessors — so a `c10::Error` thrown there (bad device index, shape mismatch in a fused step, CUDA failure during a free) unwound into Rust as UB, and no function anywhere caught non-`std::exception` throws. Every `extern "C"` function is now exception-tight: functions with an error return report through it; functions without one contain the exception and abort with a named message (a defined, loud failure replacing UB — not a path working code could reach).
- **GRU/LSTM ran with stale weights after checkpoint load**: the cuDNN parameter cache pinned the tensors it was built from on first forward and was never invalidated, but `load_checkpoint`, `cast_parameters`, and `Graph::to_device` replace parameter tensors wholesale (`Variable::set_data`) — so any forward → load → forward sequence silently kept computing with the pre-load weights (in-place optimizer/DDP updates were unaffected). `Variable` now carries a data generation bumped on every `set_data`, and GRU/LSTM rebuild their cache when any parameter's generation changes; cache hits remain an integer compare with no FFI.
- **Graph epoch iteration use-after-free window**: `Graph::epoch(..).activate()` released its borrow on the data-loader binding while the returned iterator kept a raw pointer into it, so calling `set_data_loader` mid-iteration (or activating a second iterator) was undefined behavior from safe code. The loader now lives in its own cell whose exclusive borrow is held for the iterator's lifetime: a mid-iteration `set_data_loader` returns a loud error, a second activation panics with a named message, and `forward_batch` / `data_num_batches` / `data_batch_size` keep working during iteration (the scalars are cached at bind time).
- **`migrate_checkpoint_file` with equal source and destination destroyed the checkpoint**: the destination was opened with `File::create` before the source was read, so an in-place migration truncated the file it was about to migrate ("source and destination must be different" was documented but nothing enforced it). Every checkpoint file writer now shares the same atomic tmp + rename path (previously only `save_checkpoint_file` had it), which makes in-place migration safe by construction — the source is fully consumed before the rename replaces it — and gives the cluster-consensus writer crash-atomicity as well.
- **Optimizer state load trusted the group table**: `load_state` restored per-group LR ranges without validating them, so a corrupt or truncated `.optim` file could restore a range past the parameter count — an index-out-of-bounds panic at the next `step()` instead of an error at load — or a non-contiguous table that silently skipped parameters (`step()` only updates group-covered params). The group codec is now one shared reader/writer for SGD / Adam / AdamW / RMSprop that rejects any table that is not a contiguous partition of the optimizer's parameters, naming the offending group.
- **Checkpoint reads allocated whatever the header claimed**: entry name lengths, tensor ranks, and payload byte counts were trusted before any data arrived, so a corrupt or truncated `.fdl` file could abort the process on a 2^60-byte allocation instead of reporting an error. Payload reads are now bounded by the bytes actually present (a lying header errors as "payload truncated"), and name/rank fields carry sanity caps — corruption is a loud `Err` on every checkpoint read path (load, keys listing, migration, optimizer state).
- **`fdl` overlay libtorch resolution silently fell back to `.active`**: the per-host libtorch resolver hardcoded `fdl.yml` (config discovery accepts `fdl.yaml` / `fdl.json` too) and swallowed overlay load errors — so under `FDL_ENV` an `fdl.yaml` project, or any broken overlay, silently built against the `.active` variant instead of the host's declared arch (the wrong libtorch on a heterogeneous rig). A set `FDL_ENV` whose overlay cannot be resolved is now a loud failure; legitimately-not-applicable cases (host not listed, no `cluster:` block) still fall back.
- **`fdl probe --json` emitted invalid JSON for strings with control characters**: two divergent hand-rolled escapers (one missing `\t`/`\r`, the other silently deleting `\r`) meant a tab in a GPU name or mount path broke cluster probe fan-in. One RFC-8259-complete escaper now serves every JSON emitter in fdl. The probe fan-in parser also flags a remote report missing its schema's required keys as probable fdl version skew instead of parsing it as a healthy zero-GPU host.
- **`fdl` could silently adopt an example config in non-interactive contexts**: the "copy `fdl.yml.example` to `fdl.yml`? [Y/n]" prompt ran even without a terminal, where reading stdin hits EOF and the Y-default treated that as consent — CI, shell completions, and piped invocations could create a live config file as a side effect. Without a TTY the example is now used read-only, no prompt, no copy.
- **A project path containing a space shattered `fdl`'s docker dispatch**: the compose overlay `-f <path>`, the `-e FDL_PROJECT_ROOT=<root>` injection, the container-side `cd <root>/<workdir>`, and the remote probe's `cd`/argv quoting were spliced into `sh -c` strings unquoted (or through hand-rolled quote escapes). All now go through the one shared POSIX-quoting helper — which also replaces the three divergent private copies of that helper that had drifted apart.
- **Dashboard server shutdown leaked its port and its client threads**: `Monitor` shutdown stopped only the message thread — the acceptor kept the port bound until process exit and every connected SSE client's handler thread stayed blocked forever. Shutdown now closes the accept loop (freeing the port for the next run in the same process), disconnects every SSE client, and a connection racing shutdown can no longer register itself into the already-cleared client list. Dropping the server without an explicit shutdown does the same.
- **A dashboard client that stopped reading grew server memory without bound**: each SSE client's event queue was unbounded and pruned only on hard disconnect, so a stalled-but-connected browser tab accumulated every epoch event for the life of the run. Client queues are now bounded; a client that stops draining is disconnected instead of buffered forever.
- **Timeline profiler archives grew unbounded and an abandoned timeline could never be freed**: samples accumulate at poll rate (~864k/day at the default 100 ms) with no cap, and the poller thread held a strong `Arc` to its own timeline — dropping the last user handle without calling `stop()` leaked the thread and the whole archive permanently. The poller now holds a `Weak` reference (an abandoned timeline winds down on its next tick), and both archives are capped with oldest-first trimming and a one-time notice.
- **Dropping a data loader could hang forever behind a stalled consumer**: the prefetch worker's batch sends block, so a consumer that kept its receiver alive but stopped draining wedged the worker mid-send — and the loader's `Drop` then waited on the join indefinitely. Batch sends now block in bounded slices against a teardown latch, so dropping the loader always reclaims the worker.
- **A panic in user dataset code silently killed the data loader**: a `get_batch` panic took the prefetch worker thread down with it — the panic message was discarded, and every subsequent epoch failed with a generic "prefetch worker stopped unexpectedly" that pointed at flodl instead of the dataset. Dataset panics are now caught at the fetch boundary and delivered as an `Err` batch carrying the panic message, exactly like a dataset `Err`; the worker survives and later epochs keep working.
- **Resident-loader epoch setup panicked instead of erroring**: a failure while uploading the epoch permutation tensor (CUDA error or out-of-memory at epoch start) was an `.expect` panic. The epoch iterator's items are already `Result<Batch>`, so the failure is now delivered as the first item of the epoch — the same channel streaming-loader errors always used.
- **Conv constructors accepted invalid `groups`**: `groups = 0` was an i64 division-by-zero panic inside a `Result`-returning constructor, and a non-dividing `groups` silently truncated the weight shape (integer division) into a wrong-shaped kernel. All six conv / conv-transpose variants now validate through one shared check — positive and dividing both channel counts, matching PyTorch's errors.
- **`Tensor::unique(sorted: false)` deduplicated adjacent runs only**: the unsorted path ran `unique_consecutive`, so `unique([1, 2, 1])` returned `[1, 2, 1]` — a silent wrong answer for anyone porting PyTorch code, where `torch.unique` always deduplicates globally and `sorted` only controls output ordering. Both paths now run the global dedup kernel (the sorted path also stops paying a discarded `unique_consecutive` pass it always ran first), and the adjacent-run behavior is available deliberately as the new `Tensor::unique_consecutive`, matching PyTorch's op of the same name.
- **Shape ops copied or aliased depending on input layout**: `reshape` / `transpose` / `permute` / `select` / `narrow` / `squeeze` / `unsqueeze` / `flatten` appended a `.contiguous()` intending an owned copy, but `.contiguous()` is a no-op *view* when the result is already contiguous — so `narrow(0, ..)` aliased the source while `narrow(1, ..)` copied, and whether an in-place write on the result reached the source depended on dim and layout. The ops now return views wherever libtorch does (PyTorch parity, deterministic): slice-write idioms like `t.narrow(0, i, 1)?.copy_(&src)` now always write through, `transpose` / `permute` stop paying a copy, and every readout path makes its own contiguous copy at the point of use as before.
- **`Tensor::item()` / `to_i64_vec()` / `to_f64_vec()` dtype-blind reads**: `item()` returned garbage bit patterns as `Ok` for Float16/BFloat16/Int32 scalars (an f16 loss under autocast was the live case) and errored on Int64; `to_i64_vec()` on a non-Int64 tensor reinterpreted raw bytes as indices and returned `Ok`; `to_f64_vec()` routed integers through f32, silently truncating above 2^24. All three now cast on device to the target dtype before the host copy (floats truncate toward zero in `to_i64_vec`, matching PyTorch's `.long()`), the same pattern `to_f32_vec` always used.
- **`Tensor::from_f32` / `from_f64` / `from_i64` out-of-bounds read**: the typed constructors handed the slice pointer to libtorch without checking `data.len()` against the shape product, so a shape larger than the data was an out-of-bounds read from safe code (`from_blob` already validated; the typed paths did not). All four constructors now share one validation home; length mismatches, negative dimensions, and overflowing shape products error loudly, naming the constructor that was called.
- **Checkpoint load of `f16` / `bf16` / `i32` tensors out-of-bounds read**: while the `f32` / `f64` / `i64` restore paths built owned buffers routed through the validated typed constructors, the half-precision and `i32` path handed the raw byte pointer straight to libtorch, reading `numel x element_size` bytes on trust — so a truncated or corrupt checkpoint drove an out-of-bounds read. That path now goes through `Tensor::from_blob`, which enforces the same length check, so a short payload errors instead of reading past the buffer.
- **DDP activation-reserve calibration double-subtracted the resident batch**: the first-batch measurement (`peak - current`) already nets the batch out — both readings include it — but the worker subtracted one batch again, so whenever activations + gradients fit inside a single batch (small models) the reserve saturated to 0. And 0 doubles as the not-yet-measured sentinel, so every subsequent chunk re-calibrated with the same arithmetic and the rank ran the synchronous data path — no prefetch — for the entire run. The second subtraction is gone, and a completed measurement is floored to 1 byte so a degenerate reading can never collide with the sentinel and re-arm calibration.
- **NCCL communicator init from threads**: `ncclCommInitRank` must run on the main thread; the released thread-per-GPU engine (`Trainer::setup` multi-GPU) was occasionally calling it from a worker thread on heterogeneous-GPU rigs, corrupting the shared CUDA context. Production ranks are now separate processes (per-process `init_rank`); the surviving `Ddp::wrap` thread path uses init-on-main + `split()`.
- **`fdl` argv handling**: the parser scanned the whole argv for `--fdl-schema` / `--help`, hijacking tokens meant for the command after `--`; negative numbers were rejected as space-separated option values (`--offset -5`); and `fdl run` passed user args to `docker` unquoted, so shell metacharacters in an argument could be interpreted. All three scoped/quoted correctly now.
- **Streaming prefetch could OOM the first training step and killed the epoch on transient OOM**: the adaptive VRAM sizing filled to the 90% cap using a probe taken before any forward/backward had run, when activations, gradients, and lazily created optimizer state do not exist yet, so the first step competed with a full prefetch buffer (the `activation_reserve` deduction existed but every caller passed 0). And a prefetch-side CUDA OOM surfaced as one `Err` batch that aborted the epoch, even though the consumer drains continuously and the same allocation succeeds moments later. Prefetch depth is now a governed in-flight target instead of a fixed per-epoch channel capacity: the first fill takes a graduated share of the budget (full with a user-declared `activation_reserve`, 1/2 when `Graph::set_data_loader` derives 3x parameter bytes for gradients + optimizer state, 1/3 bare), then a one-shot honest resize raises it to the full budget as soon as the second batch is consumed, keyed to consumption rather than epoch boundaries so single-pass training benefits too. The worker retries transient OOM (empty cache, up to 10 x 100 ms) and halves the in-flight target so overcommit self-heals. `auto_resize` mid-epoch now actually takes effect immediately (its docs previously claimed it did).
- **`fdl` config files silently ignored unknown keys**: a mistyped key anywhere in `fdl.yml`, a sub-command `fdl.yml`, or a cluster overlay configured nothing (`dokcer: cuda` ran on the host, `epoch: 50` under `training:` trained with default epochs) while CLI flag typos got strict did-you-mean rejection. Every config struct now rejects unknown keys, naming the field and listing the valid ones with the merged-view location. Scoping is preserved: a bad key inside one `commands:` entry errors only when that command is invoked, so `--help` and sibling commands keep working. A user-supplied `ranks:` in a worker block, which never did anything (rank assignment is probe-computed), is now a loud load error saying so. A `--fdl-schema` probe emitted by a newer flodl-cli-macros than the installed `fdl` understands now falls back to the inline schema instead of silently dropping the unknown field.
- **`fdl` entry-command docker wrapper defeated its own argument quoting**: the `entry:`-kind docker path (e.g. `fdl ddp-bench`) wrapped the composed command as `bash -c "…"` inside the outer `sh -c`, where the host shell still expands `$` and backticks (an argument containing `$(cmd)` executed on the host despite being single-quoted for the container) and any `"` in an argument spliced the command line. The wrapper now POSIX-quotes the whole inner command, the same form the `run:`-kind path always used. The testing-cluster envelope also moved from an inline `-e NAME=VALUE` to a bare `-e NAME` pass-through, so its value never rides the command line at all.
- **Empty `fdl.yml`**: an empty or comments-only config file failed to parse with a cryptic serde error instead of loading as the default project config.
- **Adam / AdamW / RAdam / NAdam / Adagrad bias-corrected against a global step counter**: every adaptive optimizer advanced one step count shared by all parameters, but bias correction (and RAdam's variance rectification, NAdam's Nesterov schedule, Adagrad's lr-decay) is per-parameter by construction. A parameter unfrozen partway through a run — the fine-tuning workflow parameter freezing exists to serve — therefore bias-corrected its fresh `m=0`/`v=0` moments against the *global* step instead of its own first step, and its first few updates landed roughly 3x too large (the first-moment estimate is under-boosted ~10x while the second-moment denominator is under-boosted ~31x), a silent overshoot that decayed over ~1/(1-β₂) steps. Each optimizer now tracks a per-parameter step count incremented only when that parameter receives a gradient, matching PyTorch's `state_steps`; the fused Adam/AdamW CUDA kernel already accepted libtorch's native per-param `state_steps` vector and had merely been fed one broadcast scalar. The `.optim` state format changed to carry per-parameter steps and now leads with a self-identifying `FDLO` header (magic + version + optimizer kind), so a file can no longer be positionally misparsed by the wrong optimizer; a pre-header file written by flodl ≤ 0.5.x is rejected with a pointer to the new `flodl::nn::migrate_optim_state_file(src, dst, kind)` converter, which rewrites the file under the header and expands the old single global step into per-parameter steps.
- **`Module::move_to_device` default did nothing despite documenting that it moved parameters and buffers**: the trait default was an empty body, so `Graph::set_device` moved parameters itself and only *composite* modules worked — calling `move_to_device` on a bare leaf module (a lone `Linear`, a `BatchNorm` outside a graph) silently left it on its original device, and `BatchNorm`'s own override moved only its running-stat buffers while swallowing the errors. The default now moves everything reachable through `parameters()` and `buffers()` (detach → move → `set_data`, the recipe `Graph::set_device` already used, which also bumps parameter data-generation so cuDNN weight caches rebuild), skips already-on-device tensors so repeated moves stay cheap, and panics with a named message on a failed move rather than leaving a half-moved model to surface later as a confusing cross-device op error. `BatchNorm`'s redundant override is gone.
- **Dashboard `<script>` injection could break out of its tag on a `</script>` in run data**: both dashboard emitters inline run constants (graph label, structural hash, hardware string, metadata, GPU init) into a `<script>` block, but the escaping diverged — the static-report path neutralized `</script>` in some constants (data/svg/metadata) and the live server escaped only `\`/`"`, so a label or metadata value containing `</script>` closed the tag early and injected arbitrary markup into the page. The HTML parser scans for `</script` literally, ignorant of JS string quoting, so per-value quote-escaping never protected the tag. Both paths now route every injected constant through one shared `</script>` neutralization applied to the whole assembled script body (`<\/script`, transparent across JSON/JS-string/template-literal contexts) so the escape set cannot drift between them again.
- **A custom module that forgot to override `parameters()` trained nothing, silently**: `Module::parameters()` defaults to an empty list for a leaf with no sub-modules, so a user layer that held parameters but never overrode the accessor reached training with nothing to optimize and every step was a no-op — burning GPU hours to produce an unchanged model with no error. Every training entry (`Trainer::run`, `Trainer::builder().run()`, `DdpBuilder`, `Ddp::wrap` / `Ddp::from_comm`) now rejects a zero-parameter model loudly, naming the likely cause. Optimizer-level empty parameter lists stay supported (deliberate — a metrics-only probe optimizer is valid).
- **`GradScaler` could drive its loss scale to zero and silently kill training**: on a detected inf/NaN gradient the scale is multiplied by the backoff factor with no lower bound, so a run with sustained non-finite gradients (a genuinely diverging model) halved the scale every step until it underflowed to `0.0`, after which `scale(loss)` is always zero, the unscaled gradients are always finite zeros, no inf is ever detected again, and training continues forever having quietly stopped learning. The scale now floors at `min_scale` (default `1.0`, since loss scaling only ever scales *up*, so there is no legitimate reason to go below 1.0); if inf persists at the floor the divergence is real and surfaces as such instead of masquerading as a dead-but-"healthy" run.
- **`cuda_graph_capture` synchronized a hardcoded device 0 before capture, wrong on multi-GPU**: the pre-capture sync always targeted device 0 regardless of which device the captured closure actually runs on, so on a rig where capture happens on any other device the warmup work on the real capture device was not guaranteed to have drained before capture began. It now synchronizes the current device (`tensor::current_cuda_device()`).
- **List-returning shims could segfault on allocation failure and leaked on a mid-loop throw**: the four shims that return a tensor array (`meshgrid`, `chunk`, `split`, `unbind`) malloc the result array then wrap each tensor in a loop. `flodl_meshgrid` alone never null-checked the malloc (a failed allocation wrote through a null pointer), and all four leaked the array plus every already-wrapped tensor if a wrap threw partway through (an out-of-memory the caller could recover from, since it surfaces as a normal `Err`). All four now go through one leak-safe `wrap_list` helper: it null-checks the allocation, and on any mid-loop failure frees every element allocated so far and the array before the exception propagates.

## [0.5.3] - 2026-04-28

### Added

#### `LoopBody` + `TraceEmit`: multi-output per-iteration traces from loop bodies

Loop bodies can now publish multiple named auxiliary outputs per iteration without maintaining side-channel state. `Module::trace()` is unchanged and remains the convenience for single-output bodies.

- **`flodl::LoopBody`** trait: opt-in extension on top of `Module`. Bodies implement `step(input, refs, emit)` and call `emit.publish("name", value)` per iteration; the runner harvests each step's emit map and appends entries into per-name vectors on the loop node. No `Rc<RefCell>` field on the body, no `reset()` hook for traces, no `trace()` getter.
- **`flodl::TraceEmit`**: per-step emit channel handed in by the loop runner. `publish()` panics on duplicate names within a single step (per-step dedup, always-on). `TraceEmit::discard()` returns a no-op emitter for non-loop forward paths.
- **`flodl::forward_via_step`**: one-line helper for bodies that have no standalone `forward()` semantics: `fn forward(&self, x) -> Result<Variable> { forward_via_step(self, x) }`. Combined with `fn as_loop_body(&self) -> Option<&dyn LoopBody> { Some(self) }` on the body's `Module` impl, this is the canonical "loop-only body" pattern.
- **Read side unchanged**: emit-published names land in `Graph::traces(name)` and `LossContext::traces[name]` alongside legacy single-trace streams. `Graph::traces_named(name)` is available when you want to skip the legacy fallback.
- **Sparse emits supported**: a step that doesn't publish a given name simply doesn't grow that name's vector. `traces["name"].len() <= n_iter`, equal to the count of iterations where the name was published.
- **Collision detection**: trace namespace is validated once per graph (cached via `Cell<bool>`) on first observation, either via `Graph::traces`, `Graph::traces_named`, or `el_che_snapshot_traces` / `gather_tags_and_traces` on the El Che path. Panics on cross-loop emit-name reuse or emit-name vs post-loop-trace-tag collisions; tag and trace key spaces are otherwise separate (`LossContext::tags` vs `LossContext::traces`).
- **DDP integration via `Graph::distribute`**: full multi-trace support across replicas, not a single-GPU-only feature.
  - Each replica's body, built by the factory closure, owns its own `named_trace_buf` (per-replica emit storage). No host-side `Rc<RefCell>` for replicas to share - that was the failure mode the API was designed to avoid.
  - The gather pipeline (`el_che_snapshot_traces`, `gather_detached_traces`, `el_che_set_gathered_traces`) walks `named_trace_buf` alongside the legacy single-stream `trace_buf`, moving each step's `Variable` to the gather device and concatenating per `(emit_name, step_idx)` across ranks and batches. The loss closure observing `ctx.traces["name"]` sees one combined `Vec<Variable>` per step regardless of how many replicas published it.
  - `validate_trace_namespace` runs from the gather path too, so cross-loop emit-name collisions and emit-vs-tag collisions are caught under DDP - they don't silently merge into the same key on the host side.
  - Backward through gathered traces is supported: the loss closure can read `ctx.traces["name"]` and let its `Variable`s feed into the loss; the autograd graph spans the per-replica forward + the host-side loss.
  - Test coverage: `test_el_che_loop_body_emits_gathered_across_replicas` exercises the full path - two named emits per iteration, gather across replicas, loss-closure consumption with backward through grad-bearing parameters.

Motivation: `Module::trace() -> Option<Variable>` is one stream per loop, requires four pieces of side-channel state on the body (RefCell field + side-effect write in `forward` + getter + `reset()` cleanup), and the natural `Rc<RefCell>` workaround for multiple streams breaks under DDP because `Graph::distribute` builds fresh bodies per replica while loss closures registered on the host capture only the host's buffer. `LoopBody::step` removes the side-channel pattern entirely and makes multi-stream a free side effect.

#### `Trainer`: primary training entry point

`Trainer` is the new default API for training in flodl. It forwards to the same DDP machinery as `Ddp::*` but reads as "just train" rather than "set up DDP" - the one-liner works transparently on 1 or N GPUs.

- **`Trainer::setup(&model, builder, optimizer)`** - one-call setup for Graph-based models, replacing `Ddp::setup()`. Auto-detects hardware, distributes if multi-GPU, sets optimizer, enables training mode. Zero DDP overhead on single GPU / CPU.
- **`Trainer::setup_with(&model, builder, optimizer, config)`** - same but takes a `DdpConfig` for explicit El Che cadence / speed hints / overhead target. Replaces `Ddp::setup_with()`.
- **`Trainer::builder(model_factory, optim_factory, train_fn)`** - builder entry for framework-managed training. Replaces `Ddp::builder()`. Works identically on single or multi-GPU.
- `flodl::Trainer` re-exported from the crate root alongside `Ddp`.

Motivation: `Ddp::builder()` read as an opt-in for "when you have multiple GPUs," obscuring that the same entry is the sensible default for single-GPU training too. `Trainer::builder()` makes the intent explicit - reach for it by default; drop to `Ddp::wrap()` when you need explicit multi-GPU control (GAN, RL, progressive patterns).

#### flodl-hf: loss wiring on BERT-family task heads

All nine `*For{SequenceClassification,TokenClassification,QuestionAnswering}` heads across BERT, RoBERTa, and DistilBERT can now drive a training loop without any hand-rolled loss plumbing.

- **Free functions in `flodl_hf::task_heads`** (family-agnostic, compose with flodl's existing loss idiom):
  - `sequence_classification_loss(logits, labels)` - CE over `[batch, num_labels]` logits with `[batch]` indices or `[batch, num_labels]` soft labels.
  - `token_classification_loss(logits, labels)` - CE over flattened `[batch*seq_len, num_labels]` logits with `-100` ignore on `[batch, seq_len]` labels, matching HF Python's convention for specials / padding / non-first subwords.
  - `question_answering_loss(logits, start_positions, end_positions)` - averaged CE on the split `[batch, seq_len, 2]` logits.
- **`forward_encoded(enc) -> Variable` on every head** returns raw logits without touching train / eval mode, leaving the caller (or `Trainer::setup`) in charge of mode.
- **`compute_loss(enc, labels)` on every head** combines `forward_encoded` with the matching task-head loss. Mirrors HF Python's `model(..., labels=...).loss` one-call pattern for `/port`-friendly fine-tune loops.
- **New example**: `fdl flodl-hf example distilbert-finetune` - loads the SST-2 DistilBERT checkpoint, fine-tunes on an inline 10-example polarity dataset for 5 steps, prints the loss curve and a final eval probe. Self-contained (no dataset download at example-runtime), CPU-only, about 30 s after the one-time weight fetch.

#### `Trainer::setup_head` + `HasGraph` trait: transparent DDP for task-head wrappers

`Trainer::setup_head` extends the transparent 1-or-N-GPU story to wrapper types like `flodl-hf`'s `BertForSequenceClassification`. Same call shape as `Trainer::setup` (`head, factory, optimizer`), same training-loop code path, works identically on CPU / single GPU / multi-GPU.

- **`flodl::HasGraph` trait**: one method, `fn graph(&self) -> &Graph`. Any wrapper that owns a flodl `Graph` can opt into graph-aware DDP machinery by implementing it (3 lines). `Graph` itself implements it trivially.
- **`Trainer::setup_head(head, head_factory, optimizer)`** (and `setup_head_with` for explicit `DdpConfig`): prints device summary, distributes via the head factory for additional GPUs, wires the optimizer, enables training mode. On 1 GPU or CPU the factory is never called; on N GPUs each replica is a fresh head built at its device.
- Internally routes through a small `HeadReplica<H>` adapter that delegates `Module::parameters/buffers/set_training/as_graph` to the inner graph. Task heads stay free of a direct `impl Module` (their true forward is multi-input via `forward_multi` and doesn't fit the single-Variable `Module::forward` signature).
- All nine flodl-hf task heads (BERT / RoBERTa / DistilBERT × SeqCls / TokenCls / QA) now implement `HasGraph` and expose a `config()` accessor so callers can build replicas inside the factory closure.
- The `distilbert-finetune` example is rewritten to use `setup_head`: the loop is byte-identical to the multi-GPU path, so a user can scale to N GPUs without changing any training code.

#### `GELU` now carries an approximation form (BC-clean)

`GELU` gains an `approximate: GeluApprox` field so flodl can dispatch both the erf form (PyTorch `nn.GELU()`, HF `hidden_act="gelu"`) and the tanh approximation (HF `hidden_act` in {`"gelu_new"`, `"gelu_pytorch_tanh"`}) required by ALBERT, GPT-2, and derivative checkpoints. The bare-name usage `.through(GELU)` keeps compiling: `pub const GELU: GELU = GELU::exact();` re-exports the default-constructed value under the type name, so existing code is untouched.

- **`GELU::exact()`** - erf form, the default; same as bare `GELU`.
- **`GELU::tanh()`** - tanh approximation; pick this for ALBERT, GPT-2, and HF `gelu_new` / `gelu_pytorch_tanh` checkpoints.
- **`GELU::with_approximate(approx)`** - runtime-chosen form, used by `flodl-hf` config loaders that map `hidden_act` strings to a [`GeluApprox`] value at load time.
- **`GeluApprox` enum** - `Exact` (default) | `Tanh`. Adding a new variant later (e.g. a polynomial fit) makes every downstream `match` site fail to compile until handled, which is what we want for a numerically-distinct activation.

This is the canonical pattern in flodl for parametrising what was previously a unit-struct module without breaking BC: `pub struct GELU { … }` carries the field, a `pub const GELU: GELU = GELU::exact();` named identically (Rust puts types and consts in separate namespaces) keeps bare-name value usage working, and opt-in constructors cover the variants.

#### flodl-hf: ALBERT family + task heads

ALBERT (`albert-base-v1` / `albert-base-v2` reference checkpoints) joins the family roster, with both architecture deltas that distinguish ALBERT from BERT plumbed through end-to-end.

- **`AlbertConfig` / `AlbertModel`** - backbone with **factorised embeddings** (token / position / type embeddings live in a smaller `embedding_size` space, lifted into `hidden_size` via a single `embedding_hidden_mapping_in` projection; embedding LayerNorm runs in embedding space) and **cross-layer parameter sharing** (one transformer block re-applied `num_hidden_layers` times). The encoder block itself is mathematically identical to BERT (post-LN, GELU activation), so the shared `TransformerLayer` carries the implementation; only the weight-key suffixes differ.
- **`AlbertLayerStack`**: wraps the single shared block and forwards `num_hidden_layers` times inside one `Module`, surfacing parameters under the HF state_dict tag `albert.encoder.albert_layer_groups.0.albert_layers.0`. Configs with `num_hidden_groups > 1` or `inner_group_num > 1` are rejected at `from_json_str` time - every public `albert-*` checkpoint as of 0.5.3 sits at `1`/`1`; the axis can grow when a non-trivial checkpoint appears.
- **`AlbertPooler`**: tanh-activated `[CLS]` pooler, bit-exact against HF reference.
- **`AlbertMLMHeadTransform` + `AlbertForMaskedLM`**: dedicated `hidden -> dense -> activation -> LayerNorm -> embedding_size` transform feeding a tied decoder back to vocabulary, mirroring HF's `AlbertMLMHead`.
- **Full task-head set**: `AlbertForSequenceClassification`, `AlbertForTokenClassification`, `AlbertForQuestionAnswering`, `AlbertForMaskedLM` - type aliases over the family-agnostic task-head generics, all exposing `forward_encoded` / `compute_loss` like the BERT-family heads.
- **`hidden_act` dispatch**: ALBERT ships `gelu_new` (tanh approximation); `AlbertConfig::from_json_str` parses `hidden_act` into `GeluApprox::Tanh` and the encoder layer plus MLM head transform call the matching libtorch op. ALBERT was the integration that motivated the GELU approximation-form work above - picking the wrong form silently produces ~1e-2 max-abs diff.
- **Auto dispatch**: `AutoConfig::Albert` and `AutoModelFor*::Albert` variants extend the family dispatch.

#### flodl-hf: XLM-RoBERTa family + task heads

XLM-RoBERTa (`xlm-roberta-base` reference checkpoint and the multilingual fine-tunes built on top) joins as a structural sibling to RoBERTa.

- **`XlmRobertaConfig` / `XlmRobertaModel`** - architecturally identical to RoBERTa: same encoder layers, same `roberta.*` state_dict prefix, same tied-decoder MLM head, same position-id convention. HF's `XLMRobertaModel` subclasses `RobertaModel` without structural changes; this port follows suit, delegating to the RoBERTa graph builders after a trivial `From<&XlmRobertaConfig> for RobertaConfig` conversion. Loaded safetensors line up directly without any key renaming.
- **Distinct config struct, not a type alias**: keeps the HF `model_type: "xlm-roberta"` signal typed through `AutoConfig`, and leaves room for XLM-R-only fields to grow without churning `RobertaConfig`. Field layout mirrors `RobertaConfig` exactly.
- **Full task-head set**: `XlmRobertaForSequenceClassification`, `XlmRobertaForTokenClassification`, `XlmRobertaForQuestionAnswering`, `XlmRobertaForMaskedLM`, all over the family-agnostic generics with `forward_encoded` / `compute_loss`.
- **Tokenizer**: SentencePiece over the ~250k multilingual vocabulary (vs RoBERTa's 50k BPE) is handled by `HfTokenizer::from_pretrained` transparently - from the model's perspective, `input_ids` are `input_ids`.
- **Auto dispatch**: `AutoConfig::XlmRoberta` and `AutoModelFor*::XlmRoberta` variants extend the family dispatch.

#### flodl-hf: DeBERTa-v2 / DeBERTa-v3 family + task heads

DeBERTa-v2 / DeBERTa-v3 (`microsoft/deberta-v3-{xsmall,small,base,large}` and SQuAD / NLI fine-tunes on top) joins with disentangled self-attention. DeBERTa-v3 ships under HF's `deberta-v2` architecture name (the v3 distinction is a config knob, not a separate class), so this port covers both.

- **`DebertaV2Config` / `DebertaV2Model`** - backbone with three load-bearing departures from the BERT family:
  1. **Disentangled attention** - each layer computes content-to-content + content-to-position + position-to-content scores, scaled by `sqrt(head_dim * 3)`. Implemented in a dedicated `crate::models::deberta_transformer_layer`, separate from the shared `TransformerLayer` because the math is fundamentally different.
  2. **No absolute positional embedding** - position information is carried by the encoder's `rel_embeddings` table and threaded into every layer as a disentangled bias.
  3. **Mask-gated embeddings** - post-LayerNorm, the embedding output is multiplied element-wise by the padding mask, zeroing pad positions before they enter the encoder.
- **`DebertaV2Encoder`** + **`ContextPooler`** - DeBERTa-v2's `pooler_output` is `tanh(dropout(linear(last_hidden[:, 0])))`, distinct from BERT's tanh-only pooler. The sequence-classification head dispatches via the family-generic `ClassificationHead<DebertaV2Config>` over the `ContextPooler`.
- **`build_deberta_attention_mask`** - public helper exposed for callers wiring `forward_multi` directly outside the head wrappers.
- **Task-head coverage**: `DebertaV2ForSequenceClassification`, `DebertaV2ForTokenClassification`, `DebertaV2ForQuestionAnswering` are bit-exact against HF Python on pinned checkpoints. `DebertaV2ForMaskedLM` is wired in but does not have a working pinned reference: V3 checkpoints ship no MLM weights (V3 trains via Replaced-Token-Detection - the MLM head is random-init by design), and V2 xlarge ships real MLM weights but uses `conv_kernel_size=3`, which this port does not implement. The investigation is documented in `flodl-hf/tests/deberta_v2_parity.rs` module-doc; ConvLayer support would unblock V2 xlarge MLM parity.
- **Config strictness**: `from_json_str` rejects `share_att_key=false`, missing `c2p` / `p2c` in `pos_att_type`, `relative_attention=false`, `position_biased_input=true`, `norm_rel_ebd != "layer_norm"`, non-zero `conv_kernel_size`, `embedding_size != hidden_size`, and `legacy=true`. Each rejection names the failing knob. This matches every public `microsoft/deberta-v3-*` checkpoint; DeBERTa-v1 and other variants surface a specific parse-time error.
- **Auto dispatch**: `AutoConfig::DebertaV2` and `AutoModelFor*::DebertaV2` variants extend the family dispatch.

#### flodl-hf: Masked-language-modeling heads across all six families

A fourth task shape - masked-language modeling - joins sequence classification, token classification, and question answering. All six families ship a `*ForMaskedLM` wrapper that consumes raw text and returns the top-k fill-mask candidates with probabilities, mirroring HF Python's `pipeline("fill-mask")`.

- **Type aliases over `MaskedLmHead<Cfg>`**: `BertForMaskedLM`, `RobertaForMaskedLM`, `DistilBertForMaskedLM`, `XlmRobertaForMaskedLM`, `AlbertForMaskedLM`, `DebertaV2ForMaskedLM` - same `from_pretrained` / `forward_encoded` / `compute_loss` / `predict` shape as the other task heads.
- **`AutoModelForMaskedLM` enum**: family-agnostic dispatch, mirrors the existing `AutoModelFor*` entry points. `from_pretrained` reads `config.json`, builds the matching family head, and returns the dispatched enum; callers stay family-agnostic.
- **`fill_mask(text, top_k)` ergonomics**: pick the `[MASK]` (BERT-family) or `<mask>` (RoBERTa-family) token, run a single forward, and return the top-k vocabulary candidates with probabilities for the masked position. Tokenizer mask-token resolution is unified through `HfTokenizer`.
- **Per-family head shapes**: each family ships its native MLM head structure unchanged from HF reference - BERT's `transform + tied decoder`, RoBERTa's flat tied decoder with bias, DistilBERT's `vocab_layer_norm + vocab_projector`, ALBERT's `embedding_size`-factored decoder, DeBERTa-v2's V3 non-legacy layout. No structural unification; the family-agnostic surface lives at the `MaskedLmHead<Cfg>` generic level above the per-family graph.
- **Parity coverage**: five of six family MLM cells (BERT, RoBERTa, DistilBERT, XLM-RoBERTa, ALBERT) are bit-exact against HF Python on pinned checkpoints. The DeBERTa-v2 MLM gap is documented above; the wrapper compiles and runs against the available reference but the comparison surfaces the upstream RTD-vs-MLM mismatch rather than a flodl-hf bug.

#### `fdl flodl-hf verify-export <dir>` - generic export verifier (auto-detect)

A single Python verifier replaces the six per-family `verify-export-<family>` scripts. Reads `<dir>/config.json`, dispatches on `(model_type, architectures[0])` to the matching HF `AutoModelFor*`, then asserts (1) zero `missing_keys` / `unexpected_keys` on load and (2) bit-exact agreement on the head's primary forward output(s) for a fixed prompt.

- **`fdl flodl-hf verify-export <dir>`** - positional `<dir>` is the staged export. No `--family` / `--head` flag - both auto-detected from the suffix on `architectures[0]` (`Model`, `ForSequenceClassification`, `ForTokenClassification`, `ForQuestionAnswering`, `ForMaskedLM`).
- **Hub source recovery**: `fdl flodl-hf export --hub <repo>` now stamps `flodl_source_repo: <repo>` into the exported `config.json`, so `verify-export` recovers the source automatically. Override via `--hub-source <repo>` for hand-staged dirs or fall-through to `_name_or_path` if a Hub config still carries one.
- The six per-family `verify-export-{bert,roberta,distilbert,xlm-roberta,albert,deberta-v2}` commands are now thin wrappers that call the generic script with `--hub-source` baked in - same zero-arg ergonomics, one Python script behind them. Removed: `flodl-hf/scripts/verify_export_<family>.py` (×6) and `flodl-hf/scripts/_export_verify.py`.

The generic command extends coverage from base backbones to the full 30-cell head matrix (6 families × {base, seqcls, tokcls, qa, mlm}) - exactly the cases the Rust `_live` head-roundtrip tests already cover bit-exact at the safetensors layer. Run order: `fdl flodl-hf export --hub <repo> --out <dir>` (dev container), then `fdl flodl-hf verify-export <dir>` (hf-parity container).

#### `fdl flodl-hf export` - staged HuggingFace-compatible export

Re-emit any flodl-hf-supported model as an HF-compatible directory (`model.safetensors` + `config.json`) that loads back into HF Python's `AutoModelFor*.from_pretrained`. The companion to the verify-export entry above; together they form the round-trip gate that proves flodl-hf weight loaders, graph builders, and config writers all agree with the HF Python reference.

- **Two source modes, mutually exclusive**:
  - `--hub <repo>` - fetch from the HuggingFace Hub and re-emit. Auto-detects family from `model_type` (`bert`, `roberta`, `distilbert`, `xlm-roberta`, `albert`, `deberta-v2`).
  - `--checkpoint <path>` - re-emit a local `.fdl` checkpoint. Reads architecture from the sidecar `<stem>.config.json` (or `--config <path>` to override).
- **`--head <auto|base|seqcls|tokcls|qa|mlm>`** (Hub mode): force a specific head class instead of dispatching on the upstream `architectures[0]`. `auto` (default) reads the upstream architecture; `base` re-exports the bare backbone even when the upstream advertises a head - useful for treating a pretraining checkpoint as a feature-extraction encoder. The other four force the matching head wrapper.
- **`--out <dir>`** (required): output directory. Writes `<out>/model.safetensors` + `<out>/config.json` in HF-canonical layout.
- **`--force`**: overwrite existing files in `<out>` without prompting.
- **`--preserve-source-config`** (checkpoint mode): also write the loaded source config verbatim to `<out>/config.source.json` alongside the canonical `config.json` - for research / replication provenance, since the canonical `to_json_str` normalises some fields away.
- **Bit-exact round-trip on every supported family / head**: exported dir loads back into HF Python's `AutoModelFor*` with zero `missing_keys` / `unexpected_keys` and bit-identical forward outputs, validated by `fdl flodl-hf verify-matrix` across the 30-cell head matrix.
- **Tokenizer round-trip**: when a `--hub` source ships a fast tokenizer, `tokenizer.json` is also persisted to `<out>` via [`HfTokenizer::save`](#hftokenizersave--persist-a-loaded-tokenizer-back-to-disk) so the staged dir is fully self-contained for HF Python's `AutoTokenizer.from_pretrained`.

#### `fdl flodl-hf verify-matrix` - full head-matrix runner

Quarterly-manual gate that runs `fdl flodl-hf export` then `verify-export` across the full 30-cell head matrix (6 families × `{base, seqcls, tokcls, qa, mlm}`), then prints a PASS/FAIL grid.

- **Cell list comes from `flodl-hf/tests/fixtures/head_matrix.json`**: each entry pins a `{family, head, repo}` tuple. Adding a cell is a one-line JSON edit; the runner picks it up next invocation.
- **Filter forwarding**: `fdl flodl-hf verify-matrix -- --families bert,albert --heads base,seqcls` runs the matching subset.
- **Fail-soft**: each cell's PASS/FAIL is captured; the grid prints at the end. A red cell does not abort the run.
- **Heavyweight**: downloads ~10+ GB of Hub weights on a cold cache. Documented as a pre-release gate, not a per-PR gate.

#### Native-dtype safetensors round-trip + `Tensor::to_blob`

`flodl-hf` safetensors I/O now preserves the checkpoint's native dtype on the way out, instead of always casting to f32. Combined with the load-side dtype preservation, this means a `bf16` or `f16` checkpoint round-trips bit-exact through `flodl_hf::safetensors_io::{load_safetensors_file_into_graph, save_safetensors_file_from_graph}`.

- **`save_safetensors_from_graph` / `save_safetensors_file_from_graph`**: preserve `f32` / `f64` / `f16` / `bf16` end-to-end, written into the safetensors header's `dtype` field. Integer dtypes are still rejected at save (BERT-family and the other supported families only store floats).
- **`load_safetensors_into_graph` / `load_safetensors_file_into_graph`**: now construct the destination tensor in the checkpoint's dtype rather than force-casting to f32. Existing f32 checkpoints (BERT-base etc.) round-trip identically; f16 / bf16 / f64 checkpoints now load at native precision.
- **`flodl::Tensor::to_blob`** (new on `flodl`): copies a tensor's raw bytes in the current dtype's storage layout. The primitive that lets `safetensors_io` write any libtorch dtype straight into the safetensors payload without going through f32.
- **DeBERTa-v2 dtype hardening**: the disentangled-attention module gained an explicit `attention_mask_dtype` argument so `c2p` / `p2c` bias additions and the masked-softmax stay in the model's native dtype, rather than getting silently up-cast.

Surfaced by the round-trip gate: a save side that always wrote f32 made the export step lossy on f16 / bf16 checkpoints, and `verify-matrix` would have eventually flagged it as a numerics drift on those families.

#### `HfTokenizer::save` - persist a loaded tokenizer back to disk

`HfTokenizer` gained a `save(path)` method that writes the wrapped `tokenizers::Tokenizer` to a JSON file in the form HF Python's `AutoTokenizer.from_pretrained` reads back. Required for the export round-trip (`fdl flodl-hf export --hub` writes `tokenizer.json` alongside `model.safetensors` so the staged dir is self-contained). Standalone callers can use it to checkpoint tokenizer state at fine-tune save points.

```rust
let tok = HfTokenizer::from_pretrained("bert-base-uncased")?;
tok.save("./checkpoint/tokenizer.json")?;
```

#### `fdl add` evolution: `--playground` / `--install` mode split

`fdl add flodl-hf` from 0.5.2 dropped a sandbox playground under `./flodl-hf/`. It now exposes two modes (combinable) reflecting how a user actually wires an ecosystem crate into a project: try-it-out (sandbox) versus wire-it-in (root dependency).

- **`--playground`** - original behaviour: scaffolds `./flodl-hf/` as a standalone cargo crate with a one-file `AutoModel` example, an `fdl.yml` with runnable commands, and a `flodl-hf:` entry in the root `fdl.yml` so `fdl flodl-hf <cmd>` routes into the playground from the project root. The user's own `Cargo.toml` is untouched.
- **`--install`** - new: appends `flodl-hf = "=X.Y.Z"` to the root `Cargo.toml` `[dependencies]` (default features = `hub` + `tokenizer`). Wires the crate into the user's own code; nothing else mutated. Idempotent (already-present is a no-op). Version locked to the project's flodl version.
- **Combinable**: `fdl add flodl-hf --playground --install` does both.
- **No flag**: interactive `[Y/n]`-style prompt asking which mode(s). When stdin is non-tty (CI, piped input) the prompt errors loudly with the explicit-flag guidance instead of silently picking a default - per `feedback_loud_errors_over_silent.md`.
- **Internals**: two new flodl-cli utility modules, `util::cargo_toml` and `util::fdl_yml`, do the file mutations. They preserve formatting and comments where possible, surface conflicts loudly (path-only or git-only flodl deps in the host project's `Cargo.toml` error with actionable guidance instead of guessing a version), and are reusable for future `fdl add <crate>` targets.

#### `fdl run` argv forwarding: `--` separator + `append:` field

`run:`-kind commands in `fdl.yml` now accept user args after an explicit `--` separator on the CLI, and can declare literal trailing tokens via a new `append:` field. The composed shell command is `[run:] [user args after --] [append:]`.

- **`--` separator on the CLI**: `fdl test -- -p flodl-hf --test foo` splices `-p flodl-hf --test foo` into the run command. Path-kind sub-commands (those with sub-`fdl.yml`) keep forwarding every extra argv token; `run:`-kind commands forward only after `--`.
- **Loud error on stray args**: `fdl test -p flodl-hf` (without the `--`) errors with a hint pointing at the right form, instead of silently dropping the args.
- **`append:` field**: declares trailing tokens that always follow the user-supplied portion (typically the libtest `-- --nocapture --ignored` for cargo test). Example:
  ```yaml
  test:
    run: cargo test
    append: -- --nocapture
  ```
  `fdl test` runs `cargo test -- --nocapture`; `fdl test -- -p flodl-hf` runs `cargo test -p flodl-hf -- --nocapture`.
- **Same forwarding rule for path-kind commands' `run:` entry-point** (e.g. cargo run --example): args after `--` go between the entry and any `append:` tokens. Visible in every `fdl flodl-hf example <name>` invocation that takes args.

The change makes `fdl` a transparent forwarder for the underlying tool's flags rather than a wrapper that absorbs them - running `cargo test`-equivalent flows through `fdl` is now byte-identical to running `cargo test` directly, modulo Docker dispatch.

#### flodl-cli polish: schema probing, bare-project help, parity subcommand layout

Smaller user-facing fixes that smooth out the host-side `fdl` experience.

- **Docker-aware schema probing**: `fdl <command> --help` invokes the command's schema-probe (`<entry> --fdl-schema`) inside the appropriate Docker service when the command declares `docker:` and the host shell isn't already inside a container. Previously, schema probes ran on the host and silently failed when the binary lived in the container - leaving `--help` either incomplete or erroring on a missing executable. Now `fdl` walks up to the nearest `docker-compose.yml` and runs `docker compose run --rm <svc> bash -c '<entry> --fdl-schema'` from there, matching the dispatch path for the actual run.
- **Bare-project help fallthrough**: a path-kind sub-project (no top-level `entry:` but with `commands:` listed) used to error on `fdl <project>` with "no entry point defined". Now it prints help, mirroring the top-level `fdl` UX. Per `feedback_help_never_blocked.md`: `--help` must always render, validation lives on the exec path, scoped to the single thing invoked.
- **`fdl flodl-hf parity` subcommand layout**: the per-checkpoint parity regenerator commands now live under `flodl-hf/parity/fdl.yml.example` (a sub-project) instead of being flat under `flodl-hf/fdl.yml.example`. `fdl flodl-hf parity bert`, `fdl flodl-hf parity albert`, etc., remain the call shape; the underlying organisation is just cleaner. New `parity_all.py` runs every checkpoint in sequence (handy for contributors regenerating after sha bumps).
- **Doc-link sweep** (`b683ed0`): mass fix of stale doc links across `README.md`, `docs/ddp.md`, `docs/tutorials/13-data-loading.md`, plus a new `site/guide/index.html` landing page on flodl.dev.

#### `fdl` daily update check (opt-in by default, multi-axis opt-out)

`fdl` now probes crates.io once per day for newer versions of itself (`flodl-cli`) and, when run inside a Cargo project, the user-facing flodl crates the project depends on (`flodl`, `flodl-hf`). Outdated crates surface as one-line nudges at the end of the user's command - no extra latency on the work itself, no surprise network traffic, no blocking on a slow registry.

- **Cache + throttle**: results are cached in `<config-dir>/flodl/config.json` (XDG on Linux/BSD, `~/Library/Application Support` on macOS, `%APPDATA%` on Windows); the throttle window is 24 hours per machine.
- **Non-blocking probe**: HTTP via `curl --max-time 2`, fired from a `Drop` guard at process exit so the user-visible command output runs first. Every failure mode (offline, slow network, registry hiccup) is silent.
- **Opt-out hierarchy**:
  - `FDL_NO_UPDATE_CHECK=1` env var (wins over all else).
  - `update_check.enabled = false` in `<config-dir>/flodl/config.json`.
  - Auto-disabled when `CI=true` (any standard CI runner), or when `/.dockerenv` is present (container filesystems are ephemeral, cache would never warm).
- **Probe scope**: when run inside a Cargo project, reads `Cargo.lock` to identify which user-facing flodl crates are actually depended on. No probe for crates the user doesn't use.

This is the same UX pattern `cargo` and `rustup` use - minimal, opt-out-able, ergonomic. Surfaced from `feedback_ux_polish_is_adoption_lever.md` as a small UX win that compounds as flodl ships more often.

#### flodl-hf: internal consolidation pass

A round of structural refactors landed alongside the family expansion, with no user-facing API change. Listed for completeness so future archaeology has a single entry point; nothing here moves the public surface, so existing callers compile unchanged.

- **`hub.rs` split into per-family submodules**: the ~1900 LOC monofile is now `hub::{bert, roberta, distilbert, albert, xlm_roberta, deberta_v2}` plus a thin top-level shim, with each family owning its own config-and-weights fetcher.
- **6× `fetch_<family>_config_and_weights` wrappers collapsed into one generic** over the family config trait, removing the per-family copy-paste that had grown a clear seam.
- **Pooler-detection logic consolidated into `safetensors_io`**: `keys_have_pooler` / `weights_have_pooler` now have one source of truth across the loaders, replacing the mirrored helpers that had drifted enough to surface a real bug on the 0.5.2 → 0.5.3 fix path.
- **`HeadKind` enum + `From<&str>` impl** centralises head-suffix → kind dispatch (`"ForMaskedLM"` → `HeadKind::Mlm`, etc.) for `Auto*::from_pretrained`, the export path, and `verify-export`.
- **`Auto*::from_pretrained` dispatch unified with `export::build_<family>_for_export`** so the load path and the export path share one set of family-x-head builders instead of two near-duplicates.
- **`safetensors_io` unit tests** (dtype matrix + key-validation edge cases) join the existing `_live` integration tests.
- **`parity_common` test helper module** lifted out of the boilerplate every `<family>_*_parity.rs` was duplicating: max-abs-diff, fixture parsing, shape probes.

### Changed

- **Repository moved to `github.com/flodl-labs/flodl`**. `Cargo.toml` `repository = ...` metadata, in-repo doc links, and CI/release scripts updated. Maintainer handle (`fab2s`) unchanged. Old GitHub URLs redirect transparently, but newly published crate metadata points at the new org.

#### flodl-hf: `AutoConfig` and `AutoModelFor*` dispatch enums grew variants and are now `#[non_exhaustive]`

The four pre-existing dispatch enums - `AutoConfig`, `AutoModelForSequenceClassification`, `AutoModelForTokenClassification`, `AutoModelForQuestionAnswering` - gained variants for the three new families (`XlmRoberta`, `Albert`, `DebertaV2`) added this release. The brand-new `AutoModelForMaskedLM` enum ships with the same shape. All five are now marked `#[non_exhaustive]` so future family additions (ModernBERT, LLaMA, ViT, …) do not require another bump on this axis.

- **Documented usage is unaffected.** `AutoModelFor*::from_pretrained(...)?.predict(...)` and `AutoConfig::from_json_str(...)?.model_type()` do not pattern-match on the variant list, so the call-site shape stays identical.
- **Exhaustive `match` arms in caller code break.** A match on `AutoConfig { Bert, Roberta, DistilBert }` written against 0.5.2 will fail to compile against 0.5.3 - both because the variant set grew and because `#[non_exhaustive]` requires a `_ => …` arm even when all known variants are covered. Adding a wildcard arm (or coverage for the new variants) fixes it.
- **Pre-1.0 break, called out so adopters aren't surprised.** Strict cargo-semver would require a 0.6.0 bump. flodl-hf is on its second publish (first was 0.5.2 five days ago), the practical break radius is essentially "users who wrote exhaustive matches on a brand-new dispatch enum within a five-day window," and a 0.6.0 cycle for one shape of break would force the same bump on every future family addition. Shipping as 0.5.3 with `#[non_exhaustive]` installed once means subsequent family adds (ModernBERT, LLaMA, ViT, LoRA) are BC-clean by attribute, regardless of version policy.

### Deprecated

- `Ddp::setup()`, `Ddp::setup_with()`, `Ddp::builder()` - use the matching `Trainer::*` methods instead. Same behavior, clearer intent. Compile-time deprecation warnings guide migration. `Ddp::wrap()` remains on `Ddp` as the explicit multi-GPU control tier. Removal targeted for a future release.

### Fixed

- **flodl-hf `--checkpoint` re-export round-trip on base backbones.** `AutoModel::from_pretrained_for_export` was preserving the Hub config's `architectures` field verbatim (e.g. `["BertForMaskedLM"]` for `bert-base-uncased`) while the actual built graph mirrored HF's `AutoModel.from_pretrained` and dropped the head. The sidecar then drove `build_for_export` to rebuild the head class, producing a structural-hash mismatch on `Graph::load_checkpoint`. Now normalised to the base class name (`BertModel`, `RobertaModel`, `DistilBertModel`, `XLMRobertaModel`, `AlbertModel`, `DebertaV2Model`) so `--hub` and `--checkpoint` modes round-trip bit-identically.
- **`flodl_hf::export::keys_have_pooler`** misclassified saved checkpoints whose pooler keys carry a tag-qualified prefix (e.g. `bert.pooler/dense.weight`). The `starts_with("pooler/")` check only matched bare layouts and silently returned `false` for every BERT-family base checkpoint. Fixed to normalise the `/` tag separator and `ends_with` against the family pooler suffixes, mirroring the safetensors-side `weights_have_pooler`.
- **`DebertaV2Config::from_json_str`** now accepts `pos_att_type` as either the pipe-separated string (`"p2c|c2p"`, the v3 base convention) or a JSON array (`["p2c", "c2p"]`, what `transformers` re-emits when re-saving fine-tuned heads - `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, `deepset/deberta-v3-base-squad2`, etc.). Previously array configs failed parsing with an empty-string error.
- **`fdl flodl-hf export --hub <repo>`** was re-emitting the upstream Hub config's `architectures` field verbatim (e.g. `["BertForMaskedLM"]` for `bert-base-uncased`) while the loader, mirroring HF's `AutoModel.from_pretrained`, built the base backbone and silently dropped head keys. The stale architecture fooled HF Python's `AutoModelFor*` dispatch on the exported dir into building a head whose weights weren't there. The companion fix on the `--checkpoint` re-export path landed in `cf967f8`; this completes the matching fix on `--hub`. Now `examples/export_hf.rs::run_hub` reads the normalised config from `graph.source_config()` (which `from_pretrained_for_export` already stamps with the base class name on every supported family), so the staged dir's `architectures` reflects what was actually built. Surfaced by `fdl flodl-hf verify-export` on first run.

## [0.5.2] - 2026-04-22

### Added

#### flodl-hf: new sibling crate for HuggingFace integration
Scaffolded under `flodl-hf/` with feature-gated modules so downstream users can take only what they need. Transformer blocks build on flodl's `nn` module; the crate depends on `flodl` for `Tensor`, `Module`, and named-parameter machinery.

- **Three install profiles**:
  - *Full* (default): `safetensors` + `hf-hub` + `tokenizers`. `flodl-hf = "0.5.2"` loads `"bert-base-uncased"` out of the box.
  - *Vision-only*: `hub` feature only. For ViT, CLIP vision towers, or any image model that doesn't need tokenisation. Drops regex + unicode surface.
  - *Offline / minimal*: no default features. `safetensors`-only. For air-gapped environments, embedded training, or local-disk pipelines - no network, no async runtime, no TLS stack.
- **`cuda` feature** on `flodl-hf` re-exports `flodl/cuda`.
- **HTTP backend**: `ureq` + `rustls-tls` on `hf-hub = "0.4"`. Sync, no tokio, no openssl (dev Docker image has no `libssl-dev`, so rustls is now the convention for any HTTP dep).
- **ROADMAP**: HF fine-tuning moved to `In progress` with `[started]` marker; `flodl-manager CLI evolution` line added to Possibilities (gaps flagged while scaffolding: `fdl build` argv forwarding, `fdl add <crate>` command).

#### flodl-hf: HuggingFace-naming foundations

- **`flodl-hf::path::HfPath`** - immutable dotted-path builder that assembles HuggingFace-style keys segment by segment. Authors write short identifiers (`root.sub("encoder").sub("layer").sub(i).sub("attention").sub("self").leaf("query")`) instead of `format!` boilerplate. `sub` accepts anything `ToString`, so integer layer indices compose directly. `new`/`sub`/`leaf` panic on invalid segments (programmer error); `try_new`/`try_sub`/`try_leaf` return `Result` for user-supplied input (LoRA adapter names, custom head names from config, HF `get_submodule` paths). Validation rejects empty segments and embedded `.` / `/`.
- **`flodl-hf::path::hf_key_from_flodl_key`** - converts flodl's `"{tag}/{leaf}"` qualified names (from `Graph::named_parameters()`) to HuggingFace-dotted keys by swapping only the final `/` for `.`. Centralises the flodl ↔ HF boundary in one place.
- **`flodl-hf::safetensors_io::LoadValidation`** - three-bucket key-set diff (`missing`, `unused`, `shape_mismatches`) with stable sorted output. `into_result()` emits a loud `TensorError` listing up to 20 entries per bucket with a `"... and N more"` truncation tail, surfacing every disagreement in a single error instead of failing on the first mismatch. Catches the entire `"queri"` vs `"query"` typo class: the bad tag appears as `missing`, the real checkpoint key as `unused`, pointing straight at the fix.
- **`flodl-hf::safetensors_io::expected_from_graph`** - walks a `Graph`'s named parameters + buffers and returns the HF-key + shape list needed by `validate_keys`.

#### flodl-hf: BERT architecture

Full HuggingFace BERT stack under `flodl-hf/src/models/bert.rs`: `BertConfig` (with a `bert_base_uncased()` preset), `BertEmbeddings`, `BertSelfAttention` (fused `scaled_dot_product_attention` with in-kernel dropout), `BertSelfOutput`, `BertAttention`, `BertIntermediate`, `BertOutput`, `BertLayer`, `BertPooler`, `BertModel`.

- **`BertModel::build` / `BertModel::on_device`** - returns a flodl `Graph` with `embeddings → N encoder layers → pooler`. The graph takes **4 inputs**: `input_ids`, `position_ids`, `token_type_ids`, and a pre-computed additive `attention_mask` shared across all encoder layers via `.using()`.
- **Only `BertLayer` implements `Module`**; inner composites carry ad-hoc `forward` signatures matching their real semantics (residual inputs). Not pretending residuals are single-input. Parameter aggregation is explicit via `HfPath::prefix_params`.
- **`build_extended_attention_mask(mask)`** helper: raw `[B, S]` 0/1 → additive `[B, 1, 1, S]` f32 (`0.0` attend, `-1e4` mask, fp16-safe). Callers run this once before `forward_multi`, mirroring HF Python's explicit `get_extended_attention_mask` idiom.
- **HF-compatible parameter naming**: tags encode HF dotted paths directly; `Graph::named_parameters() + hf_key_from_flodl_key` yields `"bert.encoder.layer.0.attention.self.query.weight"` on the first run. BERT-base has **199 parameters** total - pinned by a test.

#### flodl-hf: safetensors weight loader

`flodl-hf/src/safetensors_io.rs` - `load_safetensors_into_graph(graph, bytes)` plus rename-aware and allow-unused variants (`*_with_rename`, `*_with_rename_allow_unused`, `*_file_*` path-based).

- **Strict-load semantics**: `validate_keys` runs first; any disagreement bails before mutating any parameter. Either the graph is fully loaded or fully untouched. Makes safe retry / fall-back possible.
- **`Variable::set_data` over `copy_`**: libtorch rejects in-place ops on leaf Variables that require grad. `set_data` swaps storage while preserving `requires_grad` - the documented "optimizer replacement" path. `Buffer::set` is the buffer equivalent.
- **Host-side dtype conversion** supports F32 / F64 / BF16 / F16 → f32, including a custom `f16_bits_to_f32` (normals, subnormals, Inf, NaN) so the loader doesn't drag in the `half` crate.
- **Integer dtypes rejected loudly** (`I8`/`I16`/`I32`/`I64`/`U*`/`BOOL`/`F8*`). Silent casts hide upstream bugs.
- **`bert_legacy_key_rename`** handles pre-2020 BERT checkpoints' legacy `LayerNorm.gamma` / `LayerNorm.beta` → `weight` / `bias`. The rename-aware loader checks injectivity and raises a loud error on collision.

#### flodl-hf: HuggingFace Hub integration

- **`BertModel::from_pretrained(repo_id)` / `::from_pretrained_on_device(repo_id, dev)`** - one-liner weight + config pull via `hf_hub::api::sync::Api`. Parses `config.json`, builds the matching `Graph`, loads safetensors weights via the allow-unused rename-aware loader. The 7 `cls.*` task-head keys in `bert-base-uncased` (it's a `BertForPreTraining` checkpoint) are logged and discarded (up to 20 with a truncation tail).
- **`HfTokenizer::from_pretrained(repo_id)`** - downloads `tokenizer.json` via the same `hf_hub` cache. Feature-gated on both `hub` and `tokenizer`.
- **`fdl test-live`** - root-level command that runs `cargo test live -- --nocapture --ignored`. Canonical runner for `_live`-suffixed `#[ignore]`'d tests that need network / external resources. See `feedback_live_test_naming.md`.

#### flodl-hf: `HfTokenizer` (model-agnostic wrapper)

`flodl-hf/src/tokenizer.rs` - thin façade over `tokenizers::Tokenizer`.

- **`from_file(path)` + `from_pretrained(repo_id)`** (the latter gated on the `hub` feature).
- **`encode(&[&str])` / `encode_on_device(&[&str], Device)`** return an `EncodedBatch` carrying `input_ids` / `attention_mask` / `token_type_ids` / `position_ids` as `i64 [B, S]` Variables.
- **Sensible padding defaults** installed on load when `tokenizer.json` hasn't configured padding itself: `BatchLongest`, direction `Right`, `pad_id = token_to_id("[PAD]").unwrap_or(0)`. **No default truncation** - oversized texts error loudly at the model rather than silently truncate.
- **Model-agnostic**: one wrapper serves BERT, GPT2, LLaMA, etc. The loaded `tokenizer.json` carries the model-specific pre-tokenizer and post-processor. For BERT, the raw 0/1 `attention_mask` still needs `build_extended_attention_mask` before `forward_multi`.

#### flodl-hf: PyTorch forward-parity infrastructure

`flodl-hf/` is now a self-contained sub-project with its own child `fdl.yml`. The root `fdl.yml` picks it up via the convention `flodl-hf:` entry (same shape as `ddp-bench:`).

- **`fdl flodl-hf parity-bert`** regenerates the committed parity fixture:
  - `flodl-hf/scripts/Dockerfile.parity` (`python:3.12-slim` + torch 2.8.0 CPU wheel + `transformers ~4.46` + `safetensors ~0.4` + `huggingface-hub ~0.26`).
  - `flodl-hf/scripts/parity_bert.py` - loads `bert-base-uncased`, forces `torch.nn.attention.SDPBackend.MATH` for determinism, writes inputs + outputs + provenance metadata (`source_model` / `source_sha` / `torch_version` / `sdpa_backend`) to `flodl-hf/tests/fixtures/bert_base_uncased_parity.safetensors` (~16 KB).
- **`flodl-hf/tests/bert_parity.rs`** → `bert_parity_vs_pytorch_live`. Asserts `max_abs_diff ≤ 1e-5` on `pooler_output` vs the HF Python reference. Observed on the reference host: **9.835e-7** (well under the 1e-5 tolerance, 10x headroom).
- **`flodl-hf/tests/tokenizer_parity.rs`** → `bert_tokenizer_matches_parity_fixture_live`. Asserts `HfTokenizer` reproduces the exact pinned `input_ids` + `attention_mask` + `token_type_ids` from the parity fixture - `"hello world"` → `[101, 7592, 2088, 102]`. Closes the `text → tokens → BertModel → HF reference` loop end-to-end.
- **`docker-compose.yml` gains the `hf-parity` service** (mounts workspace, `HF_HOME=/workspace/.hf-cache` for persistent weight / tokenizer cache; gitignored).
- Both parity gates run via `fdl test-live`.

#### flodl-hf: runnable examples

- **`flodl-hf/examples/`** with a child `fdl.yml`, surfaced as `fdl flodl-hf example <name>`. Cleanly separates user-facing demos from dev tooling (`parity-bert`).
- **`flodl-hf/examples/bert_embed.rs`** - closed-loop example: `HfTokenizer::from_pretrained` → `BertModel::from_pretrained` → `forward_multi` → per-sentence pooled embeddings. Prints `dim=768 L2=… head=[…]` for each input text in a batch.
- **Cargo `[[example]]` stanzas** carry `required-features = ["hub", "tokenizer"]` so `--no-default-features` builds skip the example cleanly. Adding an example is three yml lines + one Cargo stanza.

#### flodl-hf: BERT task heads

Three fine-tuned heads on top of `BertModel`, each with a Laravel-flavoured `predict()` / `answer()` API and live parity tests against real Hub checkpoints. All three load with one line (`from_pretrained(repo_id)`), pulling weights, config, and tokenizer in one go. No per-head tokenizer setup, no separate `AutoTokenizer` call.

- **`BertForSequenceClassification`** - `pooler_output → Dropout → Linear(hidden, num_labels)`. Parameter keys `classifier.{weight,bias}`. `predict(&[&str])` returns `Vec<Vec<(String, f32)>>` sorted descending by probability, with label names from the checkpoint's `id2label` (or `LABEL_k` fallback). Works out of the box with emotion / sentiment / toxicity / NLI fine-tunes such as `nateraw/bert-base-uncased-emotion`, `nlptown/bert-base-multilingual-uncased-sentiment`, `unitary/toxic-bert`.
- **`BertForTokenClassification`** - `last_hidden_state → Dropout → Linear(hidden, num_labels)`. Parameter keys `classifier.{weight,bias}`. `predict(&[&str])` returns `Vec<Vec<TokenPrediction>>` with `{ token, label, score, attends }` per sub-token; the `attends` flag mirrors the attention mask so padding drops cleanly. Works with `dslim/bert-base-NER`, `dbmdz/bert-large-cased-finetuned-conll03-english`, etc.
- **`BertForQuestionAnswering`** - `last_hidden_state → Linear(hidden, 2)` splitting into start/end logits. Parameter keys `qa_outputs.{weight,bias}`. `answer(question, context)` / `answer_batch(&[(q, c)])` return `Answer { text, start, end, score }` with the extracted span decoded through the attached tokenizer. Span search is restricted to context tokens (`token_type_id == 1`) so the question region can't be answered-with-itself. Works with `csarron/bert-base-uncased-squad-v1` and other SQuAD fine-tunes.
- **`BertConfig` extended** with `num_labels: Option<i64>` and `id2label: Option<Vec<String>>`, parsed from `config.json`. Non-contiguous label ids (gap, duplicate) error loudly - silently reindexing would misalign names with logits rows.
- **`BertModel::on_device_without_pooler`** - mirrors HF Python's `add_pooling_layer=False`. Emits `last_hidden_state` (`[B, S, H]`) instead of pooled output; the shape token-classification and QA heads consume. Backed by a shared private `bert_backbone_flow` helper so `BertModel` and every task head build on one source of truth.
- **`HfTokenizer::encode_pairs(&[(&str, &str)])`** - paired encoding with `token_type_ids == 1` on the second segment. Required for QA; also useful for NLI and sentence-pair classification.
- **Parity infrastructure per head**:
  - `fdl flodl-hf parity-bert-seqcls` / `parity-bert-tokencls` / `parity-bert-qa` regenerate fixtures under `flodl-hf/tests/fixtures/bert_{seqcls,tokencls,qa}_parity.safetensors` against `nateraw/bert-base-uncased-emotion` / `dslim/bert-base-NER` / `csarron/bert-base-uncased-squad-v1` respectively. Each script pins a text input, forces the MATH SDPA backend, records source SHA + torch version in metadata. The SeqCls script chains through `convert_bin_to_safetensors.py` first because the emotion checkpoint is `.bin`-only.
  - Matching `_live` integration tests (`bert_seqcls_parity_vs_pytorch_live`, `bert_tokencls_parity_vs_pytorch_live`, `bert_qa_parity_vs_pytorch_live`) assert `max_abs_diff ≤ 1e-5` on logits against the HF reference. Run via `fdl test-live`.
- **Runnable examples**: `fdl flodl-hf example bert-classify` / `bert-ner` / `bert-qa`. Each demo loads a real fine-tune, runs a small pinned batch, prints the top labels / entities / extracted spans.

#### flodl-hf: RoBERTa architecture + task heads

`flodl-hf/src/models/roberta.rs` - full RoBERTa stack (`RobertaConfig`, `RobertaEmbeddings`, encoder layer, pooler, three task heads). Same attention + FFN shape as BERT, four load-bearing deltas that make RoBERTa-family checkpoints load cleanly without per-model tokenizer or input plumbing.

- **`RobertaModel::from_pretrained(repo_id)`** - one-liner weight + config pull mirroring the BERT path. **Returns a pooler-free backbone by default** (`last_hidden_state` of shape `[B, S, hidden]`) since `roberta-base` and most fine-tunes don't ship pooler weights - RoBERTa pretraining drops BERT's NSP objective. HF Python silently random-initialises the pooler on load, which makes `pooler_output` non-reproducible; flodl-hf takes the opposite default and keeps the weight load strict. `RobertaModel::on_device` is still available for checkpoints that do carry their own pooler.
- **Position ids computed internally** from `input_ids` using HF's padding-offset convention (`padding_idx + cumsum(mask) * mask`; real tokens start at `padding_idx + 1`). The graph takes **3 named inputs** (`input_ids`, `token_type_ids`, `attention_mask`) - no `position_ids` in the signature, matching HF Python's `RobertaModel.forward`. Callers don't need to know the quirk exists.
- **`RobertaForSequenceClassification`** - uses the HF-native two-layer head on the `<s>` hidden state: `Dropout → dense → tanh → Dropout → out_proj`. Parameter keys `classifier.dense.{weight,bias}` + `classifier.out_proj.{weight,bias}` - not a single `classifier.{weight,bias}` like BERT. Works with `cardiffnlp/twitter-roberta-base-sentiment-latest`, `roberta-large-mnli`, `SamLowe/roberta-base-go_emotions`.
- **`RobertaForTokenClassification`** - same `Dropout → Linear` shape as BERT's token-classification head; loads `Jean-Baptiste/roberta-large-ner-english`, `obi/deid_roberta_i2b2`, etc. `predict(&[&str]) → Vec<Vec<TokenPrediction>>`.
- **`RobertaForQuestionAnswering`** - `qa_outputs.{weight,bias}` head. `answer(question, context)` / `answer_batch(&[(q, c)])` return `Answer { text, start, end, score }`. Span search is restricted to `sequence_id == 1` (see below), since RoBERTa's `token_type_ids` are uniformly zero and can't distinguish question from context. Works with `deepset/roberta-base-squad2`.
- **`RobertaConfig::from_json_str`** parses all shape + task-head fields. Defaults track HF's `RobertaConfig`: `layer_norm_eps = 1e-5` (not BERT's `1e-12`), `type_vocab_size = 1`, `pad_token_id = 1`, `max_position_embeddings = 514` (holds `padding_idx` row + 512 real positions).
- **Parity infrastructure per head**: `fdl flodl-hf parity-roberta` / `parity-roberta-seqcls` / `parity-roberta-tokencls` / `parity-roberta-qa` regenerate fixtures under `flodl-hf/tests/fixtures/roberta_*.safetensors` against `roberta-base`, `cardiffnlp/twitter-roberta-base-sentiment-latest`, `Jean-Baptiste/roberta-large-ner-english`, and `deepset/roberta-base-squad2`. Matching `_live` integration tests assert `max_abs_diff ≤ 1e-5` on pooled output / logits against the HF reference. Run via `fdl test-live`.
- **Runnable examples**: `fdl flodl-hf example roberta-embed` / `roberta-classify` / `roberta-ner` / `roberta-qa`.

#### flodl-hf: shared encoder layer + `LayerNaming` abstraction

`flodl-hf/src/models/transformer_layer.rs` introduces a single `TransformerLayer` module reused across BERT, RoBERTa, and DistilBERT. The three families share the same mathematical encoder block (self-attention + residual + LayerNorm, two-layer GELU FFN + residual + LayerNorm); only the HF weight-key suffixes differ. `LayerNaming` carries the per-family mapping as a `const` struct of 8 static strings, swapped at construction time.

- **`LayerNaming::BERT`** covers both BERT and RoBERTa (`attention.self.{query,key,value}`, `attention.output.dense`, `attention.output.LayerNorm`, `intermediate.dense`, `output.dense`, `output.LayerNorm`).
- **`LayerNaming::DISTILBERT`** maps to DistilBERT's flatter layout (`attention.{q_lin,k_lin,v_lin,out_lin}`, `sa_layer_norm`, `ffn.{lin1,lin2}`, `output_layer_norm`).
- `bert.rs` and `roberta.rs` collapsed from ~1800 + ~1250 lines to their embeddings + pooler + task heads; six duplicated encoder-layer structs per family (`*SelfAttention`, `*SelfOutput`, `*Attention`, `*Intermediate`, `*Output`, `*Layer`) replaced by one shared implementation. Existing parity tests gate the refactor at `max_abs_diff <= 1e-5` vs HF Python on 8 pinned checkpoints; numbers unchanged by the collapse.

#### flodl-hf: DistilBERT architecture + task heads

`flodl-hf/src/models/distilbert.rs` ships the 6-layer distilled BERT family (`DistilBertConfig`, `DistilBertEmbeddings`, `DistilBertModel`, three task heads). Encoder block shared with BERT / RoBERTa via `LayerNaming::DISTILBERT` (see above). Load-bearing deltas from the BERT port:

- **`DistilBertModel::from_pretrained(repo_id)`** returns a pooler-free `Graph` taking **2 named inputs**: `input_ids` (implicit) + `attention_mask`. No `token_type_ids` (DistilBERT is single-segment; the embedding table doesn't exist) and no `position_ids` (sequential `0..S` computed internally via `Tensor::arange + reshape + expand`). Callers ignore both quirks.
- **`DistilBertConfig::from_json_str`** reads HF's native field names exactly: `n_layers` / `n_heads` / `dim` / `hidden_dim` rather than BERT's `num_hidden_layers` / `num_attention_heads` / `hidden_size` / `intermediate_size`. HF docs cross-reference friction-free; the encoder instantiation pays a tiny adapter cost. Plus the two DistilBERT-specific dropouts `qa_dropout` (typical `0.1`) and `seq_classif_dropout` (typical `0.2`), and `sinusoidal_pos_embds` (parsed but unused: HF Python overwrites the sinusoidal init with the checkpoint's learned positions, so every public checkpoint ships a trained table).
- **`DistilBertForSequenceClassification`** uses HF's two-layer head on the first token's hidden state: `select(CLS) -> pre_classifier (dim -> dim) -> ReLU -> Dropout(seq_classif_dropout) -> classifier (dim -> num_labels)`. Parameter keys `pre_classifier.{weight,bias}` + `classifier.{weight,bias}` are siblings at the root level, not nested. Works with `lxyuan/distilbert-base-multilingual-cased-sentiments-student` (3-class sentiment, multilingual).
- **`DistilBertForTokenClassification`** - `last_hidden_state -> Dropout -> Linear(dim, num_labels)`. Parameter keys `classifier.{weight,bias}`. `predict(&[&str]) -> Vec<Vec<TokenPrediction>>`. Works with `dslim/distilbert-NER` (PER / ORG / LOC / MISC BIO, 9 labels).
- **`DistilBertForQuestionAnswering`** - `last_hidden_state -> Dropout(qa_dropout) -> Linear(dim, 2)`. Parameter keys `qa_outputs.{weight,bias}`. `answer(question, context)` / `answer_batch(&[(q, c)])` return `Answer { text, start, end, score }`; span search restricted to `sequence_ids == 1` (reuses the model-agnostic filter added with `EncodedBatch.sequence_ids`). Works with `distilbert/distilbert-base-cased-distilled-squad`.
- **Parity infrastructure per head**: `fdl flodl-hf parity-distilbert` / `parity-distilbert-seqcls` / `parity-distilbert-tokencls` / `parity-distilbert-qa` regenerate fixtures under `flodl-hf/tests/fixtures/distilbert_*.safetensors` against the four pinned checkpoints. Matching `_live` integration tests assert `max_abs_diff <= 1e-5` on logits / hidden state. Observed on the reference host: `distilbert-base-uncased` backbone **1.431e-6**, `lxyuan/*-sentiments-student` SeqCls **2.384e-7** (42x headroom), `dslim/distilbert-NER` TokenCls **3.815e-6**, `distilbert/distilbert-base-cased-distilled-squad` QA **2.623e-6**.
- **Runnable examples**: `fdl flodl-hf example distilbert-embed` / `distilbert-classify` / `distilbert-ner` / `distilbert-qa`.

#### flodl-hf: AutoModel family dispatch

One-liner Hub loading over the BERT / RoBERTa / DistilBERT families without the caller having to know which family the checkpoint belongs to. Dispatches on `config.json`'s `model_type` field, mirroring HF Python's `AutoModel` / `AutoModelForSequenceClassification` / … entry points.

- **`flodl-hf::models::auto::AutoConfig`** - enum over `BertConfig` / `RobertaConfig` / `DistilBertConfig`, parsed by `AutoConfig::from_json_str`. Dispatches on `model_type` (`bert` / `roberta` / `distilbert`). Unsupported values (`modernbert`, `xlm-roberta`, `electra`, …) surface a loud error naming the offending type and listing the supported set. A new `config_json::required_string` helper backs the dispatch read.
- **`AutoModel::from_pretrained(repo_id)` / `::from_pretrained_on_device`** - returns a `Graph`. Routes BERT through `BertModel::on_device_without_pooler` so the output is always `last_hidden_state` of shape `[batch, seq_len, hidden]`, consistent across the three families. Diverges intentionally from HF Python's `BertModel.from_pretrained` (which includes the pooler); when BERT's pooler output is specifically needed, use `BertModel::from_pretrained` directly. The returned graph's `forward_multi` input count still varies by family (BERT: 4, RoBERTa: 3, DistilBERT: 2); callers that run the graph directly need to match that, the task-head wrappers below hide it.
- **`AutoModelForSequenceClassification` / `AutoModelForTokenClassification` / `AutoModelForQuestionAnswering`** - enums over the per-family concrete heads. `from_pretrained(repo_id)` dispatches loading; `predict(&[&str])` / `answer(question, context)` / `answer_batch(&[(q, c)])` run inference with a unified signature. `with_tokenizer` and `graph()` / `labels()` accessors delegate to the inner head. The same code path serves `bert-base-uncased`, `roberta-base`, and `distilbert-base-uncased`.
- **Runnable example**: `fdl flodl-hf example auto-classify -- <repo_id>`. Default: `cardiffnlp/twitter-roberta-base-sentiment-latest`; pass any BERT / RoBERTa / DistilBERT classification checkpoint as `argv[1]`. Same three-line caller regardless of family.
- **No new parity fixtures**: AutoModel is a pure dispatch layer over already-tested per-family paths. Unit tests cover `AutoConfig::from_json_str` dispatch for all three families plus unknown-model-type and malformed-input error cases.

#### flodl-manager: `fdl add flodl-hf` scaffold + `fdl init --with-hf`

Closes the "very rustic" discovery gap. Before today, a user with a fresh flodl project couldn't find flodl-hf without reading docs, editing their `Cargo.toml` manually, and guessing the right feature flavors. Now one command drops a working playground.

- **`fdl add flodl-hf` (alias: `fdl add hf`)** - scaffolds a `./flodl-hf/` sub-crate inside the current flodl project. Standalone cargo crate with its own `Cargo.toml` + `src/main.rs` (a one-file `AutoModel` classifier that takes a repo id from argv) + `fdl.yml` with runnable commands (`classify`, `bert`, `distilbert-sentiment`, plus `build` / `check` / `shell`) + `README.md` documenting the three feature flavors (full / vision-only / offline), the `fdl flodl-hf convert` workflow for `.bin`-only repos, and how to wire flodl-hf into a main crate when the user is ready.
- **Version lockstep**: the scaffold parses the host project's `flodl = "X.Y.Z"` dependency (plain, table, or workspace-inherited form) and pins `flodl-hf` to the matching `=X.Y.Z`. Git-only and path-only flodl deps error with actionable guidance rather than silently picking a version.
- **Scope contract**: no mutation of the user's root `Cargo.toml` or `fdl.yml`. The playground is a side crate for hands-on discovery; wiring flodl-hf into the user's main code stays their call. The generated README walks through it.
- **Idempotent**: refuses to overwrite an existing `./flodl-hf/` directory. Users delete explicitly if they want a regenerate.
- **`fdl init --with-hf`** and **interactive prompt**: `fdl init` now asks "Include flodl-hf (HuggingFace: BERT/RoBERTa/DistilBERT, Hub loader, tokenizer)?" after the Docker/native choice. `--with-hf` bypasses the prompt for scripted invocations; any explicit `--docker` / `--native` / `--with-hf` flag puts init in non-interactive mode, respecting `--with-hf` verbatim.
- **Templates live in `flodl-cli/src/scaffold/`** - baked into the `fdl` binary via `include_str!` at compile time and travel inside the `flodl-cli` crate tarball, so `cargo install flodl-cli` from crates.io drops a fully functional `fdl add`. The scaffold `Cargo.toml` is stored as `Cargo.toml.in` to prevent cargo treating the sub-directory as a nested package during `cargo package`; it is written out as `Cargo.toml` when the scaffold runs.
- **Host-project mode detection**: `fdl add flodl-hf` inspects the parent dir to decide how to wire the scaffolded commands. `docker-compose.yml` present → Docker mode, scaffolded `fdl.yml` keeps `docker: dev` on each cargo command so `fdl classify` dispatches into the `dev` service. `docker-compose.yml` absent → Native mode, `docker:` lines stripped so `fdl classify` runs `cargo run --release` directly on the host. The invariant `fdl.yml` (or `fdl.yml.example`) must be present is enforced loudly: a missing fdl config aborts the scaffold with "expects an initialised flodl project". `.bin`-to-safetensors conversion is documented as a direct Python invocation in the scaffold README (`pip install torch transformers safetensors` + inline script) rather than assuming the rdl-repo-internal `fdl flodl-hf convert` Docker service is available in user projects.
- **First slice of the broader flodl-manager roadmap line**: deliberately narrow. `fdl add` supports only `flodl-hf` today; per-model feature flavors (`fdl add hf --for bert|vit|offline`), `fdl build` / `clippy` argv forwarding, and `fdl doctor` / `model-info` stay on the roadmap for follow-up arcs.

#### flodl-hf: `EncodedBatch.sequence_ids` + model-agnostic QA span filter

- **`EncodedBatch` gains `sequence_ids: Variable`** - per-token segment tag from the HF tokenizer (`0` = first sequence, `1` = second sequence, `-1` = special / padding). This is the canonical HF signal for "which part of a pair encoding does this token belong to"; it's model-agnostic, where `token_type_ids` is a model input whose semantics vary (BERT sets segment B to 1; RoBERTa keeps everything at zero).
- **`BertForQuestionAnswering::extract` switched** from `token_type_ids == 1` to `sequence_ids == 1` for context-region filtering. Behaviour is bit-identical on BERT (the tokenizer sets both equal), but the same code now works across the full BERT family.

#### flodl: `LayerNorm` with custom epsilon
- **`LayerNorm::with_eps`** and **`LayerNorm::on_device_with_eps`** - constructors accepting a custom epsilon, required for HuggingFace BERT (`eps = 1e-12`) and any architecture deviating from the PyTorch `1e-5` default.
- **`LayerNorm::DEFAULT_EPS`** associated constant.
- Hand-computed golden-value test anchors the eps-reaches-the-kernel claim (not just "doesn't panic").

#### flodl: Native `torch.embedding` FFI with `padding_idx`
- **FFI chain**: `flodl_embedding` shim in `flodl-sys/{shim.h, ops_training.cpp, src/lib.rs}` → `Tensor::embedding(weight, indices, padding_idx)` → `autograd::embedding(weight, indices, padding_idx)`. Delegates to libtorch's `at::embedding` directly, replacing the previous `index_select + reshape` manual path in `Embedding::forward`.
- **`Embedding::with_padding_idx`** and **`Embedding::on_device_with_padding_idx`** - constructors accepting `Option<i64>`. The gradient of the `padding_idx` row is masked to zero during backward by the native kernel, so the PAD embedding doesn't drift during fine-tuning. Range-checked at construction.
- **`Embedding::NO_PADDING = -1`** associated constant (sentinel matching `at::embedding`'s convention).
- For LLaMA-style checkpoints where `pad_token_id == eos_token_id`, pass `padding_idx = None` - otherwise the EOS row freezes, silently breaking fine-tuning.
- `Embedding::forward` now handles indices of any shape, returning `[*indices.shape, embedding_dim]` without manual reshape.

#### flodl: `scaled_dot_product_attention` FFI

Full FFI chain adding fused attention to flodl. Used internally by `BertSelfAttention`; available to any flodl model that wants fused softmax(QKᵀ/√d)V + optional masking + optional dropout.

- **`flodl_scaled_dot_product_attention`** shim in `flodl-sys/{shim.h, ops_nn.cpp, src/lib.rs}`.
- **`Tensor::scaled_dot_product_attention(q, k, v, attn_mask: Option<&Tensor>, dropout_p, is_causal, scale: Option<f64>)`** in `flodl/src/tensor/nn_ops.rs`.
- **`autograd::scaled_dot_product_attention(...)`** (re-exported as `flodl::scaled_dot_product_attention`) - backward via native libtorch autograd, same `Variable::wrap` pattern as `embedding`.
- Sentinel conventions: `attn_mask = None` for no mask; `scale = None` (or any `Some(x)` with `x <= 0.0`) selects the default `1/sqrt(E)`.
- Parity test `test_sdpa_parity_vs_naive` anchors the fused kernel against a hand-rolled `softmax(QKᵀ/√d)V` implementation; `test_sdpa_backward` covers the autograd path.
- libtorch 2.10.0; SDPA shipped in 2.0, so safe under any supported variant.

### Changed

- **`Embedding::forward` input dtype**: the preferred input is now `i64`. The legacy f32-indices path is kept as a fallback but emits a one-shot stderr deprecation warning (`"[flodl] deprecated: Embedding::forward received non-i64 indices; this fallback will be removed in a future release. Pass i64 tensors via Tensor::from_i64."`) the first time it fires per process. Internal tests that previously used `from_f32` indices have been migrated to `Tensor::from_i64`.

### Fixed

- **DDP test flake under full-suite CUDA contention**: `distributed::ddp_run::tests::test_epoch_fn_called_per_epoch` and `::test_epoch_fn_set_lr` now explicitly use `ApplyPolicy::Sync`. Both assumed `count == num_epochs * world_size`, which only holds in Sync mode: under the default `Cadence`, progressive dispatch lets a fast rank drain an epoch's pool past a slow rank's share, so the slow rank legitimately receives fewer `StartEpoch` events. Designed behaviour for progressive streaming; the test assumption was wrong.
- **`fdl cuda-test-all` / `cuda-test-serial` pulled in `_live` tests**: the "remaining ignored" leg ran `cargo test --ignored --skip nccl --skip graph_distribute`, which swept up the new HuggingFace `_live` parity tests along with the intended CUDA Graph / manual_seed / probe tests. `--skip _live` added to both commands in `fdl.yml` and `fdl.yml.example`. Live tests are the sole domain of `fdl test-live`.

### Removed

- **`Embedding` struct fields `num_embeddings` and `embedding_dim`** - both were stored but never read after the move to `at::embedding`. Fields were private; no user-visible impact.

## [0.5.1] - 2026-04-19

### Added

#### `fdl init --native` and interactive mode selection
- **Three scaffold modes**, mutually exclusive: default (Docker with host-mounted libtorch), `--docker` (Docker with libtorch baked into the image), `--native` (no Docker; libtorch and cargo on the host).
- **Interactive prompt** via `util::prompt::ask_yn` + `ask_choice` when no flag is passed and a TTY is available: asks whether to use Docker, then (if yes) whether libtorch should be host-mounted or baked in. Non-interactive invocations default to mounted.
- **Native scaffold** skips `Dockerfile` / `Dockerfile.cuda` / `docker-compose.yml`; `fdl.yml.example` omits the `docker:` field so every command runs directly on the host. Next-steps message points at `./fdl libtorch download --cpu` / `--cuda 12.8` for host-side libtorch provisioning.

#### Release-readiness suite (`make release-check`)
- **`ci/release/`** (new): eight self-contained shell scripts each verifying one release-gate invariant, plus a `run-all.sh` orchestrator. Scripts: `01-git` (clean tree, tag available), `02-version-sync` (Cargo.toml matches a dated CHANGELOG header), `03-lint-docs` (stale `make` refs, hardcoded user paths, dangling `fdl <cmd>` references in docs), `04-shell` (`sh -n` / `bash -n` picks interpreter from shebang, optional `shellcheck`), `05-ci` (delegates to `fdl ci`), `06-scaffold` (delegates to `make test-init`), `07-docs-rs` (delegates to `make docs-rs`), `08-publish-dry` (`cargo publish --dry-run` per workspace crate in dep order).
- **`make release-check`**: orchestrator target that prints a pass/fail summary and exits non-zero on any failure. Designed to catch the exact bug class this release fixed (removed `make bench*` / `bench-cpu` leftovers across docs and source code).
- **`docs/release.md`** (new): release process doc - pre-flight checklist, script table, common failures, post-tag steps (`git push --tags`, `cargo publish` dep order).
- **Side-fixes uncovered by the linter and folded in**: `flodl-cli/src/libtorch/{build,download}.rs` printing `Run 'make cuda-test' to verify.` → `fdl cuda-test`; 23 `#[ignore = "... run with: make cuda-test-*"]` test attribute messages across `flodl/src/distributed/*.rs` and `flodl/src/nn/cuda_graph.rs` → `fdl cuda-test-*`; `Dockerfile.cuda.source` + embedded copy comments referencing `make build-libtorch` → `fdl libtorch build`.

#### Post-init / post-setup "install globally?" prompt
- **`util::install_prompt::offer_global_install`**: new helper that fires at the end of `fdl init` and (interactive) `fdl setup`. Offers to promote the running binary to `~/.local/bin/fdl` so subsequent invocations can drop the `./` prefix. Skips itself when already installed at the target path, when a different `fdl` is already there, or when `HOME` is unresolvable. Declining prints a single-line reminder (`(later: ./fdl install)`).

#### Auto-probe for non-cargo entries
- **`flodl-cli/src/config.rs::load_command`**: when a sub-command's schema cache is stale or missing **and** its `entry:` is not a cargo command, `fdl` probes `<entry> --fdl-schema` automatically and caches the result under `<cmd-dir>/.fdl/schema-cache/<cmd>.json`. Scripts and pre-built binaries become first-class schema sources without an explicit `fdl schema refresh` round-trip on a fresh clone. Cargo entries remain explicit-only: `cargo run --fdl-schema` triggers a full compile, which is unacceptable latency for `fdl <cmd> --help`.
- Probe failures are swallowed: an entry that doesn't implement `--fdl-schema` simply falls through to the inline YAML schema (or no schema). Help always renders.
- New tests in `config::tests`: `load_command_auto_probes_non_cargo_entry_and_writes_cache`, `load_command_skips_auto_probe_for_cargo_entries`, `load_command_auto_probe_failure_falls_through_silently`.

### Changed

#### Scaffold is now fdl-native
- **`fdl.yml.example`** (new, committed): shipped by every scaffold mode with 8-10 commands (the exact set depends on mode). fdl auto-copies it to the gitignored `fdl.yml` on first run.
- **`./fdl` bootstrap** now shipped in **all three modes** (previously mounted-only): `./fdl install` promotes it to `~/.local/bin/fdl`.
- **Scaffold `.gitignore`** now ignores `fdl.yml` and `fdl.yaml` alongside the existing cargo/libtorch paths.
- **`fdl init` next-steps message** rewritten: `./fdl build / test / run / shell` replaces the old `make build / test / run / shell`, with a mode-specific first step (`./fdl setup`, `./fdl build`, or `./fdl libtorch download --cpu`).
- **`fdl setup`** post-install hints: `make cuda-test / cuda-build / cuda-shell` and `make test / build / shell` became the `fdl` equivalents.

#### `init.sh` reduced to a thin `fdl` proxy
- **Dropped**: the separate Docker/make dependency checks (fdl itself handles these where they still apply; scaffolded projects no longer need `make`), the hardcoded `--docker` flag, and the custom post-scaffold instructions.
- **Kept**: the "download the pre-compiled binary, fall back to `cargo build`" bootstrap for the `curl ... | sh -s <name>` path. After obtaining the binary the script simply `exec "$CLI" init "$@"`, so every flag (`--docker`, `--native`, the interactive prompt) behaves the same as running `fdl init` directly.
- **`$FDL_BIN`** (new, opt-in): when set to an executable path, `init.sh` skips the download and execs that binary instead. Used by `make test-init` to smoke-test the current checkout rather than the last-released binary on GitHub.
- **`make test-init`** rewritten: builds `flodl-cli` via cargo, scaffolds a `--docker` project through `init.sh` with `$FDL_BIN` pointed at the fresh binary, verifies every expected file is present, and runs `docker compose config` as a generated-config sanity check. Dropped the previous `make image` + live-container cargo-cache-write step (the scaffold no longer ships a `Makefile`, and the real integration path is exercised by `fdl test` separately).

#### `download-libtorch.sh` reduced to a thin `fdl libtorch download` proxy
- **Dropped**: the entire platform detection / URL construction / zip extraction / `.arch` and `.active` writer / shell-setup-instructions machinery (305 lines of logic duplicated from `flodl-cli/src/libtorch/download.rs`).
- **Kept**: the bootstrap-fdl-binary flow + `$FDL_BIN` override (same pattern as `init.sh`). After obtaining the binary: `exec "$CLI" libtorch download "$@"`.
- **Legacy `--project` flag**: filtered out with a `note:` to stderr. `fdl libtorch download` auto-detects whether to install into the project's `./libtorch/` or `$FLODL_HOME/libtorch/` based on where it's invoked from, so the explicit flag is redundant.

#### Benchmark pipeline: `fdl bench` is now the entry point
- **`benchmarks/fdl.yml`** (new): entry-kind sub-command with three presets: `publish` (10 interleaved rounds, 15s warmup), `cpu` (CPU-only quick run), and `cpu-publish`. Replaces the two root-level `run:`-kind commands.
- **`benchmarks/run.sh`** emits its option schema via `--fdl-schema`, handled at the top of the file before `set -euo pipefail`. `fdl bench --help` now lists `--rounds`, `--lock-clocks`, `--warmup-secs`, `--output`, `--cpu`, `--tier1`, `--tier2`, `--bench <NAME>`.
- **Root `fdl.yml` / `fdl.yml.example`**: `bench:` is now a path-kind pointer to `./benchmarks/`; `bench-cpu` removed (superseded by the `cpu` preset).
- **`benchmarks/bench-publish.ps1`** calls `fdl bench publish --rounds X --lock-clocks Y --output Z` instead of the removed `make bench-publish` target. Repo root inside WSL is discovered via `wsl wslpath -a (Resolve-Path "$PSScriptRoot\..").Path`, no hardcoded path.
- **`ddp-bench/run-missing.sh`**: hardcoded repo path replaced with `cd "$(dirname "$0")/.."`.
- **`docs/benchmark.md`**: four `make bench*` invocations rewritten as `fdl bench [<preset>]` plus a pointer to `fdl bench --help`.

#### Documentation and repo hygiene
- **`docs/cli.md`** `fdl init` section: three-mode invocation, updated file list (`fdl.yml.example` + `./fdl` bootstrap), removed the "scaffold ships a Makefile by default" caveat.
- **`docs/cli.md`** `fdl schema` / `--fdl-schema` section reframed around the two opt-in paths: `#[derive(FdlArgs)]` for Rust binaries, manual JSON emit for scripts and pre-built tools (with `benchmarks/run.sh` cited as the reference example). Clarifies that non-cargo entries auto-probe on first use, while cargo entries still require an explicit `fdl schema refresh` after rebuilds.
- **`docs/cli.md`** Benchmarks section (flodl-source-checkout context): updated to the `fdl bench [<preset>]` surface.
- **`ai/skills/port/guide.md`** + embedded copy **`flodl-cli/assets/skills/port-guide.md`**: Phase 0 rewritten to reflect the fdl-native scaffold (`./fdl build / test / cuda-test` instead of `make *`). "Option A" is now labelled "Mounted libtorch (recommended)".
- **`benchmarks/README.md`**: quick-start and publication-mode invocations rewritten for `fdl bench [<preset>]`.
- **`.github/pull_request_template.md`**: `make test` / `make clippy` checkboxes swapped for `fdl test` / `fdl clippy`.
- **Blog posts** (`site/_posts/2026-03-25-benchmarks.md`, `site/_posts/2026-03-31-benchmark-update.md`): short update notes added pointing at `docs/benchmark.md` for the current `fdl bench [<preset>]` invocations. Original prose preserved for historical accuracy.

### Removed

- Root `fdl.yml` / `fdl.yml.example`: `bench-cpu` command (use `fdl bench cpu` instead).
- **Scaffolded `Makefile`** (both `MAKEFILE_DOCKER` and `MAKEFILE_MOUNTED` in `flodl-cli/src/init.rs`): projects generated by `fdl init` are now fdl-native. The commands the Makefile used to wrap (`build`, `test`, `run`, `check`, `clippy`, `shell`, `cuda-*`) now live in the scaffolded `fdl.yml.example`. The libtorch env-var derivation that lived in the mounted Makefile is handled once inside `flodl-cli/src/run.rs::libtorch_env` for every dispatch.

## [0.5.0] - 2026-04-18

> Upgrading from 0.4.0? The only breaking changes live in `fdl.yml`
> (`scripts:` merged into `commands:`) and in `#[derive(FdlArgs)]`
> (a small set of reserved flag names). See
> [UPGRADE.md](UPGRADE.md) for the step-by-step migration.

### Added

#### New Crate: `flodl-cli-macros`
- **`flodl-cli-macros/`** (new workspace member): proc-macro derive crate exposing `#[derive(FdlArgs)]`, re-exported as `flodl_cli::FdlArgs`. Turns a plain struct into an argv parser plus schema and help renderer. Implements `flodl_cli::FdlArgsTrait` with `try_parse_from(&[String]) -> Result<Self, String>`, `schema() -> flodl_cli::Schema`, and `render_help() -> String`.
- **`#[option(...)]`** named-flag attribute: `short = 'c'`, `default = "..."`, `choices = &["a", "b"]`, `env = "VAR"`, `completer = "name"`. Supported field shapes: `bool` (absent = false, present = true), `T` (scalar, requires `default`), `Option<T>` (absent = None), `Vec<T>` (repeatable).
- **`#[arg(...)]`** positional attribute: `default`, `choices`, `variadic` (requires `Vec<T>`, must be last), `completer`.
- **Derive-time validation**: required positionals cannot follow optional ones; variadic must be last; reserved flags cannot be shadowed (see Global Flags for the authoritative list); duplicate long/short flags error at compile time.
- **Per-option env fallback**: `#[option(env = "WANDB_API_KEY")]` falls back to the environment variable when the flag is absent (argv > env > default). `bool` fields are exempt.
- **Typed help via Rust docs**: doc-comments on the struct and fields flow into `render_help()` output with ANSI colouring.

#### `fdl.yml` Manifest Overhaul
- **Unified `commands:` map**: replaces the separate `scripts:` + `commands:` pair from 0.4.0. Each entry is exactly one of three kinds, chosen by which fields are set.
- **`run:` kind**: inline shell script, optionally wrapped in `docker compose run --rm <service>` when `docker:` is set. Closed script: extra argv is **not** forwarded (use shell `$VAR` inside the script instead).
- **`path:` kind**: pointer to a nested directory with its own `fdl.yml`. Convention default: when the entry is empty and a sibling `<name>/` directory exists, `fdl` loads `<name>/fdl.yml`. Extra argv after `fdl <cmd> ...` flows through to the nested `entry:` and is validated against the `FdlArgs` schema.
- **preset kind**: neither `run:` nor `path:` set; inline `ddp:` / `training:` / `output:` / `options:` fields deep-merge over the enclosing sub-command's defaults and invoke its `entry:`. Only legal inside a path-kind sub-command's own `fdl.yml`.
- **Load-time validation**: `docker:` on non-`run:` entries is rejected; unknown keys error with a clear message; kind-mismatch (e.g. both `run:` and `path:`) errors loudly.
- **Auto-bootstrap**: when only `fdl.yml.example` or `fdl.yml.dist` is present, `fdl` offers to copy it to the real (gitignored) `fdl.yml`.

#### Environment Overlays (`--env`)
- **`--env <name>`** global flag: deep-merges `fdl.<name>.yml` over the base `fdl.yml` before resolving any command.
- **`FDL_ENV=<name>`**: equivalent environment-variable form.
- **First-arg convention**: `fdl ci test` applies the `ci` overlay when `fdl.ci.yml` exists AND the name does not collide with a command. Ambiguity errors loudly.
- **Loud vs. silent fallthrough**: explicit selectors (flag, env var) fail loudly when the overlay file is missing; the first-arg convention silently falls through so existing commands are never shadowed.
- **Per-layer origin annotations**: every merged field is tagged with the file and line that contributed it, visible via `fdl config show`.

#### New Top-Level Commands
- **`fdl config show [env]`**: prints the fully-resolved YAML config with per-layer origin annotations. Useful for previewing overlay behaviour before running a long job. Equivalent forms: `fdl config show ci`, `fdl --env ci config show`, `fdl ci config show`.
- **`fdl schema list`** / **`clear [<cmd>]`** / **`refresh [<cmd>]`**: manage the per-command schema cache that powers help, completion, and validation. `list --json` for machine-readable output. Fresh / stale / orphan status is reported for every cached entry.
- **`--fdl-schema`** (hidden probe flag): every binary built with `#[derive(FdlArgs)]` responds with a JSON description of its flags. `fdl` calls it as a subprocess and caches the result at `<cmd-dir>/.fdl/schema-cache/<cmd>.json`.
- **`--refresh-schema`** per-invocation flag: refreshes a single entry's cache on the next call without running `fdl schema refresh` explicitly. Handy during development.

#### Global Flags
- **`--env <name>`**: apply overlay (see above).
- **`--ansi`** / **`--no-ansi`**: force or disable ANSI color output, overriding TTY and `NO_COLOR` auto-detection.
- **Reserved flag set** (`--help`, `--version`, `--quiet`, `--env`, `-h`, `-V`, `-q`, `-v`, `-e`): cannot be shadowed by `FdlArgs`-derived structs. Enforced at derive time for clear errors.
- **`--help` is never blocked**: validation lives strictly on the exec path, scoped to the single command being invoked. Running `fdl <cmd> --help` never triggers manifest-wide validation.

#### Value-Aware Completions
- **`choices:` drives completion**: flag completion returns the declared set, e.g. `fdl libtorch download --cuda <TAB>` offers `12.6 12.8`; `fdl ddp-bench quick --model <TAB>` offers values from the `FdlArgs` schema.
- **Project-aware**: generated scripts reflect the current `fdl.yml`'s `commands:` (all three kinds) plus every sub-command's own nested entries.
- **`fdl autocomplete`**: one-shot installer that detects the user's shell and writes the completion script to the right location.

#### Styled Output
- **ANSI-coloured help**: `render_help()` assembles colour-annotated help from doc-comments and attribute metadata. Styles are centralised in `flodl-cli/src/style.rs`.
- **Help layout for presets**: preset sub-commands render under an **Arguments** heading as a single synthetic slot with values indented beneath (placeholder overridable via `arg-name:`); regular sub-commands render under **Commands** (run / path kinds only).

#### Schema Cache (`flodl-cli/src/schema_cache.rs`)
- Per-project cache at `<cmd-dir>/.fdl/schema-cache/<cmd>.json`, populated on first use of a `path:`-kind sub-command and refreshed on demand. Cache entries carry mtime + binary hash so `fdl schema list` can flag stale (binary newer than cache) and orphan (command removed from `fdl.yml`) states.

### Changed

#### Docs
- **`docs/cli.md`** rewritten: restructured around three contexts: standalone (no project), inside an `fdl.yml` project, inside the flodl source checkout. Standalone libtorch-manager examples now lead with PyTorch C++ (CMake / `CMAKE_PREFIX_PATH`) alongside the existing tch-rs walkthrough.
- **`docs/design/run-config.md`** expanded: formal schema for `fdl.yml`, sub-command resolution, overlay merge semantics, and the DDP / training / output to `DdpConfig` / `DdpRunConfig` mapping.
- **`docs/design/msf-cadence-control.md`** (new, 669 lines): design spec for the MSF cadence-control layer.
- **`flodl-cli/README.md`** rewritten: leads with "this is the flodl CLI"; standalone libtorch manager framed as a secondary use case.
- **`flodl-cli-macros/README.md`** (new): attribute reference for `#[derive(FdlArgs)]`.
- **Root `README.md`**: short pointer box advertising `fdl` as a standalone libtorch manager for tch-rs and PyTorch C++ users.

#### Dogfooding
- **`ddp-bench/src/main.rs`** ported to `#[derive(FdlArgs)]`: typed flags, shared schema with `fdl`, help / completion / validation all come from the derived parser. Replaces the hand-rolled argv handling.
- **`fdl.yml.example`** and **`ddp-bench/fdl.yml.example`** updated to the unified `commands:` shape with the three-kind distinction.

### Removed

- **`scripts:` key in `fdl.yml`**: merged into the unified `commands:` map. Any 0.4.0 `fdl.yml` that used `scripts:` must move its entries into `commands:` with an explicit `run:` field. The three-kind `commands:` model (`run:` / `path:` / preset) is now the long-term stable manifest surface; no further breaking changes to its shape are scheduled.
- **Shadowing of reserved CLI flags in `#[derive(FdlArgs)]` structs**: `--help`, `--version`, `--quiet`, `--env`, `-h`, `-V`, `-q`, `-v`, `-e` are now reserved and enforced at derive time. Structs in 0.4.0 that named fields with any of these flags silently overrode them; in 0.5.0 they fail to compile. Rename any affected fields.

## [0.4.0] - 2026-04-14

### Added

#### `ddp-bench` - DDP Validation Suite
- **New workspace member `ddp-bench/`**: End-to-end harness that reproduces published training setups to build scientifically valid solo baselines, then measures DDP/ElChe convergence quality against them.
- **8 reference models** (`ddp-bench/src/models/`):
  - `logistic` / `mlp` / `lenet` / `conv_ae` (MNIST)
  - `resnet` (ResNet-20 on CIFAR-10, He et al. 2015 - paper baseline 91.25%)
  - `resnet_graph` (FlowBuilder rewrite of ResNet-20: same parameter count, same accuracy, with graph-level observation, named parameters and tagged residual blocks)
  - `char_rnn` (Karpathy 2015 char-RNN on Shakespeare, LSTM-256x2)
  - `gpt_nano` (4-layer pre-norm Transformer on Shakespeare, warmup + cosine decay)
- **8 DDP modes**: `solo-0`, `solo-1`, `nccl-{sync,cadence,async}`, `cpu-{sync,cadence,async}`. Side-by-side validation across all backend × policy combinations.
- **Harness** (`harness.rs`): single-process and DDP launch paths, per-batch metric collection via `record_scalar`, per-epoch convergence summaries, baseline JSON I/O.
- **Analyzer** (`analyze.rs`): compares runs against committed baselines (`baselines/structured.json`, `baselines/baseline.json`, `baselines/sync.json`) with relative-error tolerances.
- **Reporter** (`report.rs`): generates Markdown convergence reports including loss curves and timing tables (`runs/report.md`, `ddp-bench/report.md`).
- **Dataset downloader** (`download.rs`): on-demand download + cache for MNIST, CIFAR-10, Shakespeare. Cache lives under `data/` (gitignored).
- CLI flags: `--list`, `--model <name|all>`, `--mode <mode|all>`, `--epochs N`, `--batch-size`, `--lr-scale F`, `--validate`, `--baseline <path>`, `--save-baseline`, `--report <path>`, `--seed`.

#### Built-in Standard Datasets - `flodl::data::datasets`
- **`Mnist`** (`data/datasets/mnist.rs`): parses IDX gzip into `[N,1,28,28]` Float32 + `[N]` Int64. `Mnist::parse(images_gz, labels_gz) -> Result<Self>`. Implements `BatchDataSet`.
- **`Cifar10`** (`data/datasets/cifar10.rs`): parses the binary batch format into `[N,3,32,32]` Float32 + `[N]` Int64 (10 classes). Implements `BatchDataSet`.
- **`Shakespeare`** (`data/datasets/shakespeare.rs`): char-level tokenizer for next-char prediction. `[N, seq_len]` Int64 over a 65-symbol vocabulary, plus a `decode(&[i64]) -> String` helper. Implements `BatchDataSet`.
- All three plug directly into `DataLoader::builder(dataset)` in single-GPU and DDP modes.

#### Convergence Guard - Unified Divergence Reaction
- **`convergence` module** (`flodl/src/distributed/ddp_run/convergence.rs`): unified weight-space divergence guard for both NCCL and CPU averaging paths.
- **`DivergenceReport`**: per-rank L2 deltas plus optional pre/post norms. Free decomposition into cosine similarities and magnitude shifts via the algebraic identity (no extra reductions).
- **`ConvergenceAction`**: `Stable` / `SuppressGrowth` / `NudgeDown { factor }` recommendations.
- **`ConvergenceGuard::new(policy, enabled, threshold)`**: 5-interval ring buffer. Detects 3-consecutive-rising trends above threshold and returns `SuppressGrowth` to freeze ElChe anchor/overshoot growth (rather than aggressively shrinking, which can kill convergence - overhead auto-tune handles loosening on its own).
- **Wired into `Coordinator`** for both NCCL and CPU paths (`Sync` is no-op, `Cadence`/`Async` use trend detection). Configurable via `DdpRunConfig::with_divergence_threshold(f64)`.
- Cross-rank divergence is now reset after every averaging event, fixing a stale-state bug that pinned the ElChe anchor at 1.

#### Timeline Profiler - `monitor::timeline`
- **`Timeline`** (`flodl/src/monitor/timeline.rs`): high-frequency (default 100ms poll, 1s broadcast) system + GPU profiler. Captures CPU, RAM, per-GPU compute utilization and VRAM as `TimelineSample`s, interleaved with training events.
- **`EventKind`**: `EpochStart` / `EpochEnd { loss }` / `SyncStart` / `SyncEnd { duration_ms }` / `CpuAvgStart` / `CpuAvgEnd { duration_ms }` / `AnchorChanged { from, to }` / `Throttle { rank }` / `Idle { device, duration_ms }` / `Custom { label }`.
- **API**: `Timeline::new(poll_ms)` / `with_intervals(poll_ms, broadcast_ms)` (returns `Arc<Timeline>`), `start()` / `stop()`, `event(EventKind)`, `subscribe()` for live `mpsc` updates, `summary()`, `idle_gaps(device, threshold_pct, min_ms)`, `drain()`, `sample_count()`.
- **Output**: `save_json(path)`, `save_csv(path)`, `save_html(path)` - the HTML view (`timeline.html`) renders a swimlane visualization of CPU/GPU utilization, sync/averaging events, anchor changes and detected idle gaps. Used by `ddp-bench` for every run (`runs/<model>/<mode>/timeline.html`).
- Enable per-job in `fdl.yaml` with `ddp.timeline: true` or `output.timeline: true`.

#### Verbosity-Gated Logging - `flodl::log`
- **`Verbosity` enum**: `Quiet (0)` / `Normal (1)` / `Verbose (2)` / `Debug (3)` / `Trace (4)`. Higher levels include lower.
- **Macros**: `flodl::msg!("...", args)` (Normal default, `@Verbose`/`@Debug`/`@Trace` for explicit level), plus `flodl::verbose!()`, `flodl::debug!()`, `flodl::trace!()`.
- **Routing**: Normal/Verbose go to **stdout**; Debug/Trace go to **stderr** so they remain unbuffered in Docker non-TTY environments. Errors keep using bare `eprintln!`.
- **Zero-code config**: `FLODL_VERBOSITY=verbose cargo run` (accepts integers 0-4 or names). Programmatic override via `flodl::log::set_verbosity(Verbosity)`.
- **CLI integration**: `fdl -v` / `-vv` / `-vvv` / `--quiet` set `FLODL_VERBOSITY` in the parent process so it flows into Docker child commands automatically.

#### FlowBuilder - `also_with`
- **`FlowBuilder::also_with(skip, main)`** (`flodl/src/graph/flow.rs`): residual connection with a custom skip path. Generalizes [`also`](../flodl/src/graph/flow.rs) for cases where the skip needs its own transform - e.g. ResNet downsample blocks where a 1×1 conv + BN matches channel/stride changes. Output is `skip(x) + main(x)`. Exercised by `ddp-bench/src/models/resnet_graph.rs` (ResNet-20 on CIFAR-10, full paper-accuracy baseline).

#### `AdaptiveAvgPool2d`
- **`AdaptiveAvgPool2d::new([h, w])`** (`flodl/src/nn/pooling.rs`): global / fixed-output-size average pooling. Counterpart to the existing `AdaptiveMaxPool2d`. `[1, 1]` gives global average pooling (common ResNet head before FC); arbitrary output sizes enable variable-size input support. Re-exported at crate root.

#### Metrics - `drain_scalars`
- **`flodl::drain_scalars() -> HashMap<String, (f64, usize)>`** (`flodl/src/distributed/ddp_run/mod.rs`): companion to the existing `record_scalar`. Flushes the thread-local accumulator and returns `(sum, count)` per tag so callers (monitors, custom loops) can average or log per-batch scalars outside the DDP coordinator path. Re-exported at crate root.

#### LR Scheduling - Cross-Mode Parity
- **`Graph::set_scheduler(Arc<dyn Scheduler>)`** and **`Graph::set_lr_scale(f64)`** (`flodl/src/graph/distributed.rs`): scheduler attached on the Graph DDP path drives the optimizer LR via `scheduler.lr(training_step) * lr_scale` on every `step()`. `training_step` advances per `step()` call. **`Graph::training_step()`** accessor exposed for monitoring.
- **`GpuWorker::set_scheduler` / `set_lr_scale` / `current_lr`** (`flodl/src/distributed/ddp_run/worker.rs`): same mechanism on the DDP-builder path. LR computed as `scheduler.lr(global_step + steps_since_avg) * lr_scale` per batch.
- **`DdpBuilder::scheduler(factory)`** (`flodl/src/distributed/ddp_run/orchestrator.rs:1219`): per-worker scheduler factory closure. Each rank instantiates its own scheduler (cheap to clone, no shared state). Pairs with `lr_scale_ratio` to keep all ranks in lockstep.
- **`DdpBuilder::lr_scale_ratio(f64)`** / **`DdpRunConfig::with_lr_scale_ratio(f64)`**: when set, the framework auto-computes the per-rank `lr_scale` from `world_size` (linear scaling rule, Goyal et al. 2017). Default `0.0` (= disabled, `lr_scale = 1.0`); set to `1.0` for full linear scaling, fractional values for sub-linear. Manual override stays available via `--lr-scale` in `ddp-bench`.
- **Cross-mode parity test** (`graph_tests.rs`): asserts that the same `MultiStepLR` produces identical LR trajectories across all three training paths - manual reference loop, `GpuWorker` (DDP builder), and `Graph::step()` - for both unscaled and `lr_scale != 1.0`.
- **Coordinator regression**: `SyncAck` no longer inflates `steps_since_avg` and now properly satisfies `nccl_ack`, fixing a scheduler drift across NCCL averaging events.

#### DDP - New Configuration Knobs
- **`DdpBuilder::no_divergence_guard()`** / **`DdpRunConfig::with_no_divergence_guard()`**: disable the convergence guard entirely. Use during calibration runs or when the divergence trend logging is more noise than signal. Default: enabled with `divergence_threshold = 0.05`.
- **`DdpBuilder::max_overshoot(usize)`** / **`DdpRunConfig::with_max_overshoot(usize)`**: cap how many extra batches the fastest rank can run past the slowest before the next averaging event in `Async` policy. Pairs with auto-tuning; set to bound the worst case explicitly. Async-only - the `Cadence` policy uses wall-time anchoring instead. The internal `overshoot_ceiling` (default ~3× anchor) gates the auto-tuner.
- **`DdpBuilder::timeline(Arc<Timeline>)`** / **`DdpRunConfig::with_timeline(Arc<Timeline>)`** / **`DdpConfig::timeline(Arc<Timeline>)`** / **`Graph::timeline(Arc<Timeline>)`**: attach a shared `monitor::Timeline` so the DDP runtime injects `EpochStart/End`, `SyncStart/End`, `CpuAvgStart/End`, `AnchorChanged`, `Throttle` events into the profiler stream. All four entry points (single-GPU Graph, manual `Ddp::wrap`, `Ddp::setup`, `DdpBuilder`) accept the same `Arc<Timeline>`. Used by `ddp-bench` to produce per-run swimlane HTML.
- **`Coordinator::builder()`** (`flodl/src/distributed/ddp_run/coordinator/mod.rs`): the coordinator now exposes a fluent builder (`progressive`, `batch_size`, `timeline`, `divergence_threshold`, `no_divergence_guard`, `overhead_target`, `max_anchor`, `checkpoint_every`, `snapshot_timeout_secs`, `epoch_metrics_tx`, `device_indices`, `num_epochs`, `partition_ratios`, `max_overshoot`, `overshoot_ceiling`, `build`). Internal - the user-facing surface is still `DdpBuilder`/`Ddp::setup` - but useful for writing custom orchestrators.
- **Note on `max_batch_diff`**: the field shipped in 0.3.0 (per-rank lockstep limit). What's new is `DdpBuilder::max_batch_diff(usize)` as a top-level fluent setter (was only reachable via `DdpRunConfig::with_max_batch_diff`).

#### CLI: `fdl run` and Project / Sub-command Manifests
- **`fdl.yaml`** (also `fdl.yml`, `fdl.json`): committed project manifest. Declares `description`, `scripts` (named shell commands with optional `docker:` service binding) and `commands` (paths to sub-command directories that have their own `fdl.yaml`). Example at the repo root: `fdl.yml.example` (84 lines).
- **Sub-command manifests** (e.g. `ddp-bench/fdl.yml.example`): declare `entry`, `docker`, structured `ddp` / `training` / `output` sections, and named `jobs` (presets that merge over the defaults). DDP section maps 1:1 to `DdpConfig` / `DdpRunConfig` (mode, policy, backend, anchor, max_anchor, overhead_target, divergence_threshold, max_batch_diff, speed_hint, partition_ratios, progressive, max_grad_norm, lr_scale_ratio, snapshot_timeout, checkpoint_every, timeline).
- **Auto-bootstrap**: when only `fdl.yml.example` (or `.dist`) is present, `fdl` offers to copy it into the real, gitignored `fdl.yml` so users can customize without polluting the repo.
- **Built-in script targets** (e.g. `fdl test`, `fdl cuda-test-all`, `fdl shell`, `fdl bench`, `fdl self-build`): any unknown command is resolved against the project's `scripts:` map and wrapped in `docker compose run --rm <service>` when a `docker:` field is set. Replaces the old `make` workflow.
- **Sub-command dispatch**: `fdl <cmd> [<job>] [--flag ...]` resolves `<cmd>` against `commands:`, picks the named job (or defaults), merges DDP/training/output sections and forwards everything as CLI flags to the configured `entry`. Pass-through for unknown flags is preserved.
- **Recursive help**: `fdl <cmd> --help` and `fdl <cmd> <job> --help` print resolved options and inherited defaults.

#### CLI: `fdl completions` / `fdl autocomplete`
- **`fdl completions <bash|zsh|fish>`**: emits a shell-completion script that knows about all built-in commands, the local project's `scripts:` and `commands:`, and per-sub-command jobs.
- **`fdl autocomplete`**: dynamic, project-aware completion suggestions for the current cwd.
- Designed to be sourced from `~/.bashrc` / `~/.zshrc` so completions update automatically as `fdl.yml` evolves.

#### CLI: `fdl diagnose --json`
- The diagnostics report now has a fully structured `--json` mode for CI pipelines and tooling: system, CUDA devices, libtorch variants, compatibility verdict.

#### Docs: PyTorch Porting Guide
- **`docs/porting.md`** (257 lines, full rewrite from the previous 7-line stub): user-facing porting guide that mirrors the AI skill (`ai/skills/port/guide.md`) and references `fdl api-ref` for the canonical type/method index.
- **`docs/cli.md`** (130 lines): full CLI reference (setup, libtorch, init, diagnose, api-ref, install, skill, run, completions, config, verbosity flags, fdl.yaml manifest).
- **`docs/design/run-config.md`** (296 lines): formal spec for `fdl.yaml` - schema, merge order, sub-command resolution, Docker integration, and how DDP/training/output map onto `DdpConfig` / `DdpRunConfig`.
- Updates to `docs/pytorch_migration.md` and the CLI section of the README.

#### CLI: API Reference Generator
- **`fdl api-ref`**: Generate a structured API reference from flodl source. Extracts all public types, constructors, methods, builder patterns, trait implementations, and doc examples.
  - Human-readable output (1700+ lines, 170 types) or `--json` for structured data.
  - `--path <dir>` for explicit source path.
  - Auto-discovers source: project checkout, cargo registry, or downloads latest release from GitHub.
  - Downloaded sources cached at `~/.flodl/api-ref-cache/<version>/` for instant re-use.
  - Designed for AI-assisted PyTorch-to-flodl porting: the reference provides everything an agent needs to map PyTorch patterns to flodl equivalents.

#### PyTorch Porting Skill
- **`ai/skills/port/`**: AI-assisted PyTorch-to-flodl porting framework. Universal porting guide (`guide.md`) and agent instructions (`instructions.md`) that work with any AI coding assistant. Covers the full journey from environment setup (`fdl init`) through model translation (FlowBuilder patterns, layer mapping, loss/optimizer/scheduler tables) to validation (`cargo check` loop).
- **`ai/adapters/claude/`**: Claude Code adapter (SKILL.md template) for `/port` slash command. Installed via `fdl skill install`.
- Guide includes: project scaffolding (native vs Docker), 30+ module mappings, FlowBuilder patterns (sequential, residual, skip connections, split/merge, loops, tags), training loop translation, data loading, checkpointing, device management, and Rust-specific idioms.

#### CLI: Global Install & Self-Update
- **`fdl install`**: Copy the current binary to `~/.local/bin/fdl` for global access. Downloads the latest release from GitHub if a newer version is available. Detects shell (bash/zsh) and prints PATH instructions if needed.
- **`fdl install --dev`**: Symlink to the current binary instead of copying. Global `fdl` tracks local builds automatically. Every `cargo build --release -p flodl-cli` updates the global command instantly. Ideal for developers.
- **`fdl install --check`**: Compare installed version against latest GitHub release. Shows install mode (dev symlink or copied binary).
- Version-aware: shows "Updating 0.3.0 -> 0.3.1" or "already installed".
- Platform detection for pre-compiled binaries (linux/darwin/windows, x86_64/aarch64/arm64).

#### CLI: Skill Management
- **`fdl skill install`**: Detect the user's AI coding tool (Claude Code, Cursor) and install flodl skills. Auto-detects `.claude/` or `.cursorrules`. Copies universal skill files (guide, instructions) plus tool-specific adapter. `--tool <name>` to force a tool, `--skill <name>` to install one skill.
- **`fdl skill list`**: Show available skills and detected tools with install status.
- Claude Code: installs `/port` slash command to `.claude/skills/port/`.
- Cursor: appends porting context to `.cursorrules`.
- Skill files embedded in the binary via `include_str!`, so it works without a repo checkout.
- Re-running `fdl skill install` updates existing skills in place.

### Changed

#### DDP - Streaming Epochs and NCCL Cadence Boundaries
- **Streaming epoch dispatch**: `Coordinator::dispatch_next_chunk` now streams sub-epoch chunks instead of full-epoch partitions in `Cadence` and `Async` modes, adapting to live throughput. Added a guard so the coordinator never recreates chunk pools for already-aggregated epochs (was causing a deadlock under heterogeneous cadences).
- **NCCL cadence boundary fixes**: per-rank epoch ack handling rewritten so that the slowest rank no longer stalls the next epoch's `SyncNow` broadcast. ElChe anchor + overshoot remain anchored to the slow rank's wall time.
- **`max_overshoot` is Async-only**: documented as such; the auto-tune is no longer evaluated for `Cadence`.
- **Convergence safety net**: divergence signals now reset after every NCCL averaging event (was leaking stale norms across intervals and pinning the anchor at 1).

#### Optimizer Module Layout
- **`flodl/src/nn/optim.rs` (1975 lines) split into a module**: `optim/{mod, sgd, adam, rmsprop, adagrad, radam, nadam}.rs`. Public API and behavior unchanged; navigation and review surface dramatically improved.

#### FFI Shim Layout
- **`flodl-sys/shim.cpp` (4517 lines) split into themed translation units**: `ops_tensor.cpp`, `ops_nn.cpp`, `ops_math_ext.cpp`, `ops_training.cpp`, `ops_cuda.cpp`, plus a shared `helpers.h`. `shim.cpp` is now a unity-build aggregator. No FFI surface change.

#### Other
- **Rust doc warnings**: Fixed all 32 documentation link warnings (unresolved cross-module references, private item links).
- **GitHub Actions**: Added `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24` env to silence Node.js 20 deprecation warnings.
- **Release workflow**: `gh release create` now falls back to `gh release upload --clobber` when the release already exists (tag push before workflow completes).
- **CLI help text**: Updated to reflect broader scope (API reference, global install). Added examples for `api-ref` and `install` commands.

### Fixed

#### CPU Averaging Race Condition
- **`snapshot_params()` stream sync**: Added `comm_stream.synchronize()` before reading GPU parameters for CPU averaging snapshots. Without this, `Update` + `RequestParams` messages processed in the same `handle_control()` call could read mid-copy GPU memory from a pending `load_averaged()` non-blocking transfer. The coordinator's `tick()` method can send both messages in the same tick when averaging completes and the next cycle triggers immediately.
- **CPU averaging convergence fixed**: The stream sync fix (above) resolved the CPU averaging convergence failure from 0.3.0. All three CPU policies (Sync/Cadence/Async) now converge correctly (91-92% on CIFAR-10 ResNet-20, matching NCCL). Both backends are production-ready.

#### Test Stability
- **`test_graph_loop_leak`**: removed quantitative assertions (`live_tensor_count`, RSS) that flake under parallel CI. The test's real value is exercising 500 iterations of graph+loop+optimizer without crashing (use-after-free, double-free, unbounded Rc chains). Diagnostics are logged for manual review.
- **NCCL/Graph distribute test isolation**: clarified ignore set so `fdl cuda-test-nccl` covers both `nccl` and `graph_distribute` patterns and `fdl cuda-test-serial` covers everything else.

#### libtorch `AccumulateGrad` Stream Mismatch (DDP Workers)
- **Warning eliminated**: `"AccumulateGrad node's stream does not match"` fired on every DDP backward pass when workers ran on a non-default training stream. Three stacked undocumented libtorch facts combined to produce it, and fixing any one of them alone was insufficient:
  1. `AccumulateGrad` nodes capture their stream into `input_metadata` at **construction time**, not at each runtime backward call.
  2. The node is created lazily on first `backward()` **inside the autograd engine's worker thread**, whose current stream is the device default (not the user's training stream).
  3. `AutogradMeta` holds a `weak_ptr` to the node, so without an external strong reference it is collected between iterations and re-created on the default stream on every backward pass.
- **`Tensor::ensure_grad_accumulator()`** (`flodl/src/tensor/mod.rs`) / **`Variable::ensure_grad_accumulator()`** (`flodl/src/autograd/variable.rs`): eagerly materialize the `AccumulateGrad` node for a leaf tensor with `requires_grad=true`, pinning its stream to the current CUDA stream at the moment of the call. Returns a `GradAccumulatorHandle` that keeps the node alive through a strong `shared_ptr<Node>` on the C++ side. No-op for non-leaf or non-`requires_grad` tensors.
- **`GradAccumulatorHandle`** (`flodl/src/tensor/mod.rs`): opaque `Send + Sync` strong-reference handle. `Drop` frees the node (unless a backward pass still holds its own reference). Intended to be held for the lifetime of the owner, typically a DDP worker.
- **FFI additions** (`flodl-sys/ops_training.cpp`, `shim.h`, `src/lib.rs`): `flodl_ensure_grad_accumulator(FlodlTensor, void**)` and `flodl_grad_accumulator_delete(void*)`. The C++ side calls the semi-internal libtorch API `torch::autograd::impl::grad_accumulator()` (found by reading libtorch source) and heap-allocates the returned `shared_ptr<Node>` so Rust owns its lifetime.
- **`GpuWorker` construction reordered** (`flodl/src/distributed/ddp_run/worker.rs`): CUDA streams are now created **before** `model_factory` so every leaf tensor (parameters, buffers, initial copies, optimizer state, `AccumulateGrad` nodes) is allocated under `StreamGuard(compute_stream)` and carries the training-stream affinity from birth. New `_grad_accumulators: Vec<GradAccumulatorHandle>` field on `GpuWorker` holds strong references to every parameter's accumulator for the worker's lifetime; explicitly documented as liveness-only ownership (never read at runtime, dropping it re-introduces the bug).
- **Validated**: 54 training runs across 6 architectures (`logistic`, `mlp`, `lenet`, `char-rnn`, `gpt-nano`, `conv-ae`) times 9 DDP modes with zero warnings in any `training.log`. Also validated across the earlier 6-mode 200-epoch `resnet_graph` run on CIFAR-10.
- **Side effect**: unblocks CUDA Graph capture for DDP workers. Graph capture fails loudly on stream mismatches between the training stream and the accumulator stream, so prior workarounds are no longer needed.

## [0.3.0] - 2026-04-08 - Multi-GPU & Infrastructure

### Added

#### Async GPU-CPU Foundation
- **`CudaEvent`**: Record/synchronize/elapsed_time on CUDA streams. `CudaEventFlags` (Default for timing, DisableTiming for pure sync). RAII Drop, Send. 14 FFI functions (7 event + 7 stream).
- **`CudaStream`**: Pool-managed streams per device. Synchronize, wait_event, is_complete. RAII Drop, Send.
- **`StreamGuard`**: RAII stream switching (sets on create, restores default on drop). Async copy pattern: `let _guard = StreamGuard::new(&stream); tensor.to_device_async(Device::CPU)?;`
- Enables zero-stall GPU-to-CPU pipeline: `training stream -> CudaEvent -> copy stream -> CPU`

#### NCCL Collective Operations
- **`NcclComms`**: RAII communicator group for multi-GPU collectives. 5 FFI functions wrapping raw NCCL (ncclCommInitAll, AllReduce, Broadcast via GroupStart/End).
- **`ReduceOp`**: Sum, Prod, Max, Min, Avg.
- **`all_reduce()`** / **`all_reduce_on_streams()`**: In-place AllReduce across all devices (default or explicit streams).
- **`broadcast()`** / **`broadcast_on_streams()`**: Broadcast from root rank to all devices.
- Raw NCCL (not c10d) for minimal overhead in single-process multi-GPU.

#### NCCL Per-Rank Communication
- **`NcclRankComm`**: Per-rank communicator for multi-threaded DDP. Each GPU thread owns one comm, runs collectives independently. `Send` so it can be moved into spawned threads.
  - `init_rank(rank, world_size, &uid)`: Direct per-rank init from a shared `NcclUniqueId`.
  - `all_reduce(&[&Tensor], ReduceOp)` / `all_reduce_on_stream(...)`: Rank-local AllReduce.
  - `broadcast(&[&Tensor], root)`: Rank-local broadcast.
- **`NcclComms::split()`**: Extracts per-rank `NcclRankComm` from a group-initialized `NcclComms`. Preferred over per-thread `init_rank` because `ncclCommInitRank` from worker threads corrupts CUDA context on heterogeneous GPUs. Init-on-main + split is the safe pattern.
- **`NcclAbortHandle`**: Arc-shared handle to abort a stuck `NcclRankComm`. Calling `abort()` unblocks any thread stuck in an AllReduce/Broadcast and makes the comm's Drop a no-op. Used by `DdpHandle` to recover from worker death without deadlocking surviving workers.
- **`NcclUniqueId`**: 128-byte unique ID for coordinating per-rank init. `NcclUniqueId::new()` generates on rank 0, then shared to all ranks.
- 7 per-rank FFI functions: `flodl_nccl_get_unique_id`, `flodl_nccl_init_rank`, `flodl_nccl_destroy_rank`, `flodl_nccl_all_reduce_rank`, `flodl_nccl_abort_rank`, `flodl_nccl_split_rank`.

#### Transparent Multi-GPU Training
- **`Graph::distribute()`**: Auto-detect GPUs, create replicas, broadcast params. Single line to enable multi-GPU. No-op on single GPU.
- **`Graph::set_optimizer()`**: Creates per-replica optimizers when distributed.
- **`Graph::step()`**: AllReduce gradients + sync buffers + optimizer step + zero_grad. One call replaces the manual loop.
- **`Graph::set_lr()`** / **`world_size()`** / **`is_distributed()`**: Multi-GPU aware API.
- **Cross-device autograd**: `Tensor::to_device()` preserves grad_fn (ToCopyBackward). Forward chunks input, forwards shards on their GPUs, gathers via to_device + cat. libtorch autograd naturally flows gradients back through device transfers.
- **`Ddp`**: Manual DDP coordinator for complex training patterns (GAN, RL, progressive). Explicit sync_params, all_reduce_gradients, sync_buffers.
- Training loop is identical for 1 or N GPUs; `distribute()` is the only difference.

#### Async Data Loading Pipeline
- **`DataSet` trait**: Per-item dataset (`get(index) -> Vec<Tensor>`). `Send + Sync` for background prefetch. Automatic batching via `DataSetAdapter` (pre-allocate + copy, O(1 sample) peak memory).
- **`BatchDataSet` trait**: Per-batch dataset (`get_batch(indices) -> Vec<Tensor>`) for bulk-efficient sources (mmap, database). `Send + Sync`.
- **`Sampler` trait**: Index ordering per epoch. Built-in: `RandomSampler` (deterministic per seed+epoch), `SequentialSampler`.
- **`Batch`**: Named tensor wrapper with `Index<usize>` and `Index<&str>` for clean destructuring (`let images = &b["image"]` or `&b[0]`). `.names()`, `.has()`, `.get_named()` for introspection. Owns its tensors.
- **`DataLoader`**: Builder pattern with auto-detection of resident vs streaming mode.
  - **Resident mode**: Dataset fits in VRAM (75% headroom). Loaded once via `pin_memory()` + `to_device()`. Per-epoch: GPU-side `index_select` with shuffled permutation. Zero CPU-GPU transfer after warmup.
  - **Streaming mode**: Persistent worker thread with dedicated `CudaStream`. Per-epoch fresh batch channel (no deadlock on mid-epoch drop). Worker: `get_batch` -> `pin_memory` -> `StreamGuard` + `to_device_async` -> `CudaEvent`. Consumer: `event.synchronize()` (typically instant due to prefetch depth).
  - **CUDA OOM fallback**: If resident load fails with OOM, automatically retries with streaming mode.
  - **Auto prefetch depth**: `clamp(free_vram * 10% / batch_bytes, 2, 4)`. Override with `.prefetch(n)` for high-latency cloud/NFS storage.
  - `.streaming()` to force streaming mode (preserve VRAM headroom, benchmarking).
  - `drop_last` defaults to `true` (BatchNorm safety: size-1 batches cause NaN variance).
  - `EpochIterator` implements `Iterator<Item = Result<Batch>>` + `ExactSizeIterator`.
- **`TensorError::is_cuda_oom()`**: Detect CUDA out-of-memory errors for graceful fallback.
- **`.names()`**: Builder method for named batch fields (`["image", "letter", "case", "origin"]`). Auto-generated positional names ("0", "1", ...) when unspecified. Validates name count against dataset tensor count.
- DDP-aware: loader yields pinned CPU data, `forward_distributed` scatters to devices efficiently.

#### Resident DDP
- **DDP-aware DataLoader**: Third internal mode `DistributedLoader` with per-device backends. Each GPU independently selects resident (data fits in VRAM) or streaming (prefetch worker) based on its own VRAM. No lowest-common-denominator constraint.
- **`DeviceBackend`**: Per-device data strategy. Resident: full dataset on GPU, index_select per batch. Streaming: dedicated PrefetchWorker with async H2D transfers.
- **`Graph::set_data_loader(loader, "input")`**: Attach DataLoader to model. When distributed: upgrades to per-device backends. Auto-wires batch names to graph `.input()` ports. Remaining names treated as targets for loss.
- **`Graph::epoch(epoch)`**: Returns `GraphEpochIterator` that produces per-rank shards and user-facing Batch. When distributed: each backend produces on-device data, shards stored for presharded forward. When single-GPU: delegates to DataLoader.
- **`Graph::forward_batch(&batch)`**: Batch-aware forward. Extracts named inputs, handles DDP presharding transparently. Coexists with `Module::forward(&Variable)`.
- **Presharded forward path**: `forward_distributed_presharded()` consumes per-rank shards from DataLoader via `.take_shards()`. Each replica forwards its local shard (zero cross-device input transfer). Outputs gathered to gather device. CudaEvent timing for auto-balancer.
- **Multi-input auto-wiring**: `set_data_loader()` precomputes `shard_input_map` matching graph `.input()` port names to batch tensor positions. `forward_distributed_presharded()` passes all inputs (primary + auxiliary) to each replica via `as_graph().forward_impl()`. Single-GPU `forward_batch()` also builds the full input vector. Enables multi-input models (FBRL with case/origin alongside image) in distributed training.
- **Efficient distributed streaming**: `StartDistributedEpoch` + `LoadBatch` worker commands. One channel per epoch instead of per-batch channel creation. Flat state machine in `worker_loop` (no nested loops). `PrefetchWorker::start_distributed_epoch()` opens the channel once, `load_batch()` sends indices per batch.
- **Gather device selection**: Prefers resident backend with most free VRAM. Falls back to CPU if all backends are streaming (targets fetched from dataset). No GPU 0 priority.
- **Auto-balancing integration**: Epoch iterator reads chunk_ratios fresh per batch. Shard sizes adapt as ratios change every 50 steps. Mixed resident/streaming backends handle dynamic ratios correctly.
- Training loop identical for 1 or N GPUs. `distribute()` + `set_data_loader()` are the only differences.

#### `Ddp::setup()` - One-Liner DDP Setup
- **`Ddp::setup(&model, builder, optimizer)`**: Single call to auto-detect GPUs, distribute the model, set per-replica optimizers, and enable training mode. No-op distribute for single GPU/CPU (still sets optimizer + training). Training loop identical for 1 or N GPUs.
- **`Ddp::setup_with(&model, builder, optimizer, config)`**: Same as `setup()` but accepts a `DdpConfig` for explicit El Che configuration (speed hints, overhead target, max anchor).
- **`Ddp::is_heterogeneous()`**: Detects mixed GPU models. `setup()` auto-enables El Che when heterogeneous GPUs are detected.
- **Hardware diagnostics**: Always prints detected hardware to stderr on call:
  - `ddp: 2 GPUs (heterogeneous) | RTX 5060 Ti (16.0 GB) | GTX 1060 (6.0 GB)`
  - `ddp: 1 GPU | RTX 5060 Ti (16.0 GB) | single-device mode`
  - `ddp: no CUDA available | CPU mode`

#### Multi-GPU Dashboard
- **Per-GPU tabs**: Tab bar appears when 2+ GPUs detected (hidden for single-GPU, zero visual regression). Each GPU tab shows 4 time-series charts: VRAM usage (bytes, with physical limit reference line), utilization (%), throughput (samples/ms), batch share (%).
- **GPU Overview card** (Home tab): Compact row per GPU with VRAM bar, utilization, throughput, and batch share. Fastest GPU highlighted green, slowest yellow.
- **JS data model**: `gpuSeries[deviceIndex]` with per-device VRAM, throughput, chunk, and utilization arrays. Populated from `d.gpus` in `processEpoch()`. Works in both live SSE and archive replay modes.

#### Multi-GPU Dashboard Data Pipeline
- **`GpuSnapshot`**: Per-device resource sampling (VRAM allocated/total, utilization, device name). `ResourceSampler` iterates all CUDA devices on each sample. Aggregate fields kept for backward compat with single-GPU dashboards.
- **`GpuMetrics`**: DDP metrics per device (EMA throughput, chunk_ratio, shard_size). Exposed via `Metrics::gpu_metrics()` trait method with default empty impl.
- **Per-GPU JSON in epoch records**: `"gpus":[...]` array merges hardware snapshots (from `GpuSnapshot`) with DDP metrics (from `GpuMetrics`). Flows through SSE live updates and HTML archives.
- **`Graph::auto_distribute()`**: Auto-detect usable CUDA devices and distribute. No-op on single GPU. Keeps the builder closure for user-controlled model construction.
- **`Graph::shard_sizes()`** / **`Graph::devices()`**: Public accessors for per-rank shard sizes and device list.

#### Auto-Balancing
- **Per-GPU throughput measurement**: CudaEvent-based timing around each replica's forward pass in `forward_distributed()`. Zero overhead (async GPU recording, no CPU sync).
- **EMA throughput tracking**: Exponentially smoothed samples/ms per device (alpha=0.3). First measurement initializes directly, subsequent measurements blend.
- **Adaptive batch sharding**: After 10 calibration steps with equal splits, `chunk_ratios` are recomputed proportional to measured throughput. Re-evaluated every 50 steps. `MIN_CHUNK_RATIO` (5%) prevents starving any GPU.
- **Weighted gradient averaging**: When chunk ratios are unequal, each replica's gradient is scaled by `(shard_size / batch_size)` then AllReduce Sum, producing the mathematically correct mean gradient regardless of shard distribution.
- **`Graph::chunk_ratios()`**: Query current batch distribution ratios (for logging/debugging).
- **`Graph::throughput()`**: Query per-device EMA throughput (samples/ms).
- All auto-balancing is internal to `forward_distributed()` and `step()`. Training loop is unchanged.

#### NCCL Device Safety
- **Device save/restore**: All `NcclComms` methods (`new`, `all_reduce`, `broadcast`, and stream variants) now save and restore the current CUDA device around FFI calls. Prevents NCCL operations from leaking device context changes to callers.
- **Shared `NCCL_LOCK`**: Single `pub(crate)` mutex in `ddp` module, used by both `nccl::tests` and `ddp::tests` to serialize NCCL communicator operations.

#### El Che - Heterogeneous DDP
- **`ElChe`**: Cadence strategy for mixed-GPU training. Slow device anchors the sync cadence, fast devices range ahead processing more batches per sync. Named after Che Guevara's marching principle: "the column marches at the slowest one's pace."
  - `ElChe::new(world_size, anchor)` with builder pattern.
  - `with_speed_ratio(slow_rank, ratio)`: Seed initial batch distribution from known speed differential. Self-corrects after first `report_timing()`.
  - `with_overhead_target(f64)`: Default 0.10 (10%). Auto-tunes anchor upward to keep AllReduce overhead below target.
  - `with_max_anchor(usize)`: Gradient staleness cap. Prevents unbounded accumulation.
  - `report_timing(&wall_ms, sync_ms)`: Discovers true speed ratios from CudaEvent measurements, recomputes batch counts, auto-tunes anchor.
  - `batch_counts() -> &[usize]`: Per-device batch counts for the current cadence step.
  - `clamp_total(max) -> Vec<usize>`: Proportional clamping for epoch-end alignment.
- **`DdpConfig`**: Configuration struct for `Ddp::setup_with()`.
  - `speed_hint(slow_rank, ratio)`: Initial speed estimate (optional, self-corrects).
  - `overhead_target(f64)`: AllReduce overhead ceiling.
  - `max_anchor(Option<usize>)`: `None` = auto (default), `Some(0)` = disable El Che (traditional DDP), `Some(n)` = fixed cap.
  - `max_grad_norm(f64)`: Per-rank gradient clipping before normalize-by-count and weighted AllReduce. Bounds accumulated gradients on all ranks (including replicas the caller cannot reach). Uses fused C++ kernel (`clip_grad_norm_fused`).
- **`Graph::step()` El Che branch**: Normalizes accumulated gradients by `1/count[rank]` (mean per device), weighted AllReduce by `count[rank]/total` (proportional contribution), reports timing to ElChe for adaptation. Per-rank gradient clipping when configured. Existing scatter and single-GPU paths unchanged.
- **`Graph::has_el_che()`** / **`Graph::configure_el_che()`**: Query and configure El Che state.
- **`weighted_all_reduce_gradients()`**: Scales each replica's gradient by batch contribution before AllReduce Sum. Produces the mathematically correct mean gradient regardless of per-device batch counts.

#### El Che Forward Path
- **`forward_distributed_el_che()`**: Multi-batch per-device forward. Each device processes `batch_counts[rank]` complete batches independently. Gradients accumulate naturally via libtorch autograd across all forward passes. CudaEvent timing per rank.
- **Tagged output gathering**: After each forward pass, tagged outputs (`Graph::tag()`) are captured from each device and concatenated across all batches and all devices. Custom loss functions work transparently on gathered intermediates: `model.tagged("scan_locations")` returns the catted value from all devices.
- **Loop trace gathering**: Per-step outputs from loop nodes (`trace_buf`) are gathered across all batches and all devices, keyed by `(tag_name, step_index)`. `model.traces("attn")` returns catted per-step traces. Enables transparent El Che training for models with loop-based attention (scan/read fixations, per-step losses). No-op when no loop nodes exist.
- **El Che data routing**: `DistributedEpochIterator` pulls `sum(batch_counts)` complete batches per iteration (not shards). Routes whole batches to each device via `load_batch_on_device()` (supports both Resident index_select and Streaming prefetch worker). Proportional clamping near epoch boundaries.
- **Epoch-end flush**: `ActiveGraphEpochIterator::drop()` detects accumulated un-synced gradients (forward without step) and forces a final `step()` to prevent silent gradient loss.
- **`Graph::epoch()`** seeds initial batch counts from `ElChe::batch_counts()`. **`Graph::step()`** feeds updated counts back to the loader after `report_timing()`.
- Training loop is identical for homogeneous and heterogeneous GPU setups. `Ddp::setup()` detects heterogeneous hardware and enables El Che automatically.

#### DDP Builder - Thread-Per-GPU Training
- **`DdpHandle`**: Thread-per-GPU training with Local SGD and adaptive parameter averaging. Each GPU runs its own training loop with a local optimizer. A lightweight coordinator thread triggers periodic parameter averaging. Two orthogonal knobs: [`ApplyPolicy`] (when to average) and [`AverageBackend`] (how to average).
- **`DdpBuilder`** (recommended entry point): Fluent API for configuring and launching training. Required: `.dataset()`, `.batch_size()`, `.num_epochs()`. Optional: `.policy()`, `.backend()`, `.overhead_target()`, `.max_anchor()`, `.anchor()`, `.divergence_threshold()`, `.max_batch_diff()`, `.checkpoint_every()`, `.checkpoint_fn()`, `.epoch_fn()`, `.progressive_dispatch()`.
  ```rust
  let ddp = Ddp::builder(model_factory, optim_factory, train_fn)
      .dataset(dataset)
      .batch_size(32)
      .num_epochs(10)
      .policy(ApplyPolicy::Cadence)
      .backend(AverageBackend::Nccl)
      .run()?;
  let state = ddp.join()?;
  ```
- **`Ddp::builder()`**: Quick-start alternative (replaces the former `AsyncDdp::auto()`/`auto_with()`).
- **`ApplyPolicy`**: Controls WHEN averaging occurs.
  - `Sync`: K=1 (every batch). Equivalent to standard DDP. Best convergence.
  - `Cadence`: K=N (ElChe anchor count). Slow GPU anchors the cadence, fast GPUs fill wall time. Uses wall-time trigger (fires when slowest rank's accumulated wall time reaches anchor wall-time). Recommended for heterogeneous hardware.
  - `Async`: K=adaptive. Uses batch-count trigger (fires when all ranks complete their assigned counts). Overshooting is intentional: each replica explores slightly different parameter neighborhoods between averaging events, producing diversity that benefits convergence. Auto-tunes interval from divergence monitoring. Maximum throughput.
- **`AverageBackend`**: Controls HOW averaging is performed. Orthogonal to policy, all combinations valid for A/B testing.
  - `Nccl`: In-place AllReduce on GPU. Zero extra memory, GPU-to-GPU DMA. All GPUs sync at collective barrier.
  - `Cpu`: Workers send parameter snapshots to coordinator, which averages on CPU and distributes. No GPU ever blocks. Uses O(world_size * model_size) CPU RAM. Non-blocking 3-phase state machine (Idle/Collecting/Computing) keeps coordinator responsive during averaging.
- **`GpuWorker<M>`**: Generic worker bound to a single GPU. Thread-local model + optimizer (Rc-based, not Send). CUDA streams for overlapped compute/communication. Handles `SyncNow` (NCCL), `RequestParams`/`Update` (CPU), `Throttle`, `StartEpoch`, `Checkpoint`, `Shutdown`.
- **`Coordinator`**: Lightweight scheduling thread. Collects timing from workers (for ElChe throughput ratios), triggers averaging, monitors divergence to auto-tune interval, rebalances data partitions. Builder pattern with configurable `divergence_threshold`, `overhead_target`, `max_anchor`, `checkpoint_every`, `snapshot_timeout_secs`.
- **`TrainedState`**: Return type from `DdpHandle::join()`. Contains averaged `params` and `buffers` as CPU tensors, ready for inference or checkpoint.
- **`DdpRunConfig`**: Configuration struct with builder methods: `with_overhead_target()`, `with_max_anchor()`, `with_anchor()`, `with_divergence_threshold()`, `with_max_batch_diff()`, `with_max_grad_norm()`, `with_checkpoint_every()`, `with_snapshot_timeout()`, `with_partition_ratios()`, `with_progressive_dispatch()`.
- **Per-worker gradient clipping**: `DdpBuilder::max_grad_norm(f64)` clips gradients between `backward()` and `optimizer.step()` on each GPU worker. Prevents gradient spikes on any single GPU from propagating through AllReduce averaging. Same fused kernel as El Che path.
- **`progressive_dispatch`**: When enabled, the coordinator streams work in small chunks instead of sending full epoch partitions, adapting to throughput continuously. Default: auto (true for Cadence/Async, false for Sync).
- **Global epoch management**: Coordinator owns epochs globally. Workers are mode-agnostic (wait for `EpochPlan`, run partition, report metrics). `EpochPlan { epoch, partition_offset, partition_size }` ensures deterministic, non-overlapping sample coverage. Throughput-proportional partition sizing when ElChe is calibrated; `partition_ratios` for fixed splits. Auto lookahead in `Async` mode (fast ranks may run 1 epoch ahead).
- **Single-GPU fallback**: With fewer than 2 CUDA devices, training runs on the main thread with no coordinator or averaging. API is identical; `join()` returns `TrainedState` in both cases.

#### DDP Builder - Robustness
- **`max_batch_diff`**: Hard limit on how far any GPU can run ahead of the slowest. Workers that exceed the limit are throttled (block on control channel) until the next averaging event. `Some(0)` = strict lockstep.
- **`drain_until_shutdown`**: After training, workers keep handling control messages (especially `SyncNow`) until the coordinator sends `Shutdown`. Prevents NCCL deadlock when workers finish at different times.
- **NCCL init-on-main + split()**: All NCCL communicators initialized from the main thread via `NcclComms::new()` then `split()` into per-rank `NcclRankComm`. Per-thread `ncclCommInitRank` corrupts CUDA context on heterogeneous GPUs.
- **NCCL abort handles**: If a worker dies mid-collective, `DdpHandle::abort_nccl()` calls `ncclCommAbort` on all communicators, unblocking surviving workers. Also triggered in `Drop`.
- **Worker error propagation**: Failed workers set the shared shutdown flag and send `TimingMsg::Exiting` so the coordinator stops including that rank in collectives.
- **CPU averaging timeout**: Configurable `snapshot_timeout_secs` (default 5s). If not all worker snapshots arrive in time, the round is soft-aborted (logged with missing rank IDs and abort count), stale snapshots drained, and retried on the next cycle.
- **CPU Update delivery logging**: Failed Update deliveries to dead workers are logged with the affected rank.
- **Shutdown cleanup**: `drain_avg_state()` logs and joins any in-progress CPU averaging (Collecting or Computing) before the coordinator exits, preventing detached threads from holding GPU resources.

#### DDP Builder - Observability
- **Averaging success logging**: Both paths log on successful averaging. NCCL: `"NCCL averaging #N complete (vV)"`. CPU: `"CPU averaging #N complete (vV, X.Xms)"` with timing.
- **Per-rank epoch metrics**: Worker epoch-end metrics (rank, epoch, loss, batches, wall time) forwarded to stderr from the coordinator loop.
- **Coordinator accessors**: `avg_count()`, `abort_count()`, `last_batch_ms()`, `last_avg_ms()`, `is_cpu_averaging()`, `version()`, `avg_interval()`, `is_calibrated()`, `steps_since_avg()` for external monitoring.
- **Divergence monitoring** (Async policy): Per-rank parameter L2 norms tracked. Relative norm difference triggers interval halving (diverging) or doubling (converging). Threshold configurable via `divergence_threshold` (default 0.05).
- **Hardware summary**: Prints GPU count, heterogeneous/homogeneous detection, per-GPU name + VRAM, policy, and backend at launch.

#### DDP Builder - Metrics Pipeline
- **`record_scalar(name, value)`**: Thread-local function callable from inside the train function. Records named scalar metrics (accuracy, custom losses, etc.) per batch. Metrics are aggregated per rank per epoch and forwarded to the coordinator.
- **`EpochMetrics`**: Aggregated metrics for one completed epoch. Fields: `epoch`, `avg_loss`, `batches_processed`, `epoch_ms`, `samples_processed`, `per_rank_loss`, `per_rank_time_ms`, `per_rank_scalars`, `scalars`.
- **`DdpHandle::poll_metrics()`**: Non-blocking poll for completed epoch metrics. Returns a `Vec<EpochMetrics>` of all epochs aggregated since the last poll. Enables external monitoring loops.
- **`DdpHandle::next_metrics()`**: Blocking call that returns the next available `EpochMetrics`. Useful for sequential metric processing.
- **`DdpHandle::setup_monitor(&self, &mut Monitor)`**: Wire the DDP handle's graph identity, architecture SVG, and training config into a training monitor. Enables the live dashboard and HTML archive for DDP Builder training runs.
- **`LossContext`**: Per-batch context passed to loss closures in distributed training. Provides batch metadata (shard sizes, device indices) for loss functions that need to weight contributions correctly.

#### DDP Builder - Epoch Callback
- **`EpochFn<M>`**: `Arc<dyn Fn(usize, &mut GpuWorker<M>) + Send + Sync>`. Called at the start of each epoch inside each worker thread, before `run_epoch_plan()`.
- **`.epoch_fn()`** on `DdpBuilder`: Set the callback. Typical uses: LR schedules, noise curricula, dynamic loss weights.
- **`GpuWorker::set_lr(f64)`**: Delegate to the worker's optimizer.
- **`GpuWorker::current_epoch()`**: Public accessor for the current epoch number.

#### DDP Builder - Checkpointing
- **`CheckpointFn<M>`**: `Arc<dyn Fn(u64, &M) -> Result<()> + Send + Sync>`. Called on rank 0 after averaging events (multi-GPU) or epoch boundaries (single-GPU). Errors are logged but do not stop training.
- **`checkpoint_every(n)`**: Save every N averaging events. Coordinated through `ControlMsg::Checkpoint` to rank 0's worker thread (which owns the model).
- **`TrainedState`** on partial failure: If some workers died, `collect_final_state()` averages surviving workers' snapshots. If averaging fails, falls back to the first snapshot's tensors. Returns `None` only if zero snapshots arrived.

#### Adaptive Data Pipeline
- **VRAM-aware prefetch depth**: `prefetch_depth_from_vram()` computes prefetch budget as the gap between current VRAM usage and a configurable cap. No manual tuning needed.
- **Bootstrap prefetch**: Initial depth of 4 batches during DataLoader construction. Real depth computed at `epoch(0)` after model is loaded and VRAM usage is stable.
- **Per-epoch VRAM probing**: `epoch(N)` re-probes VRAM usage and fills up to the cap. Adapts to VRAM fragmentation and activation memory changes across epochs.
- **`DataLoaderBuilder::vram_max_usage(f64)`**: Default 0.90 (use up to 90% of total VRAM). Clamped to [0.50, 0.99]. Remaining headroom covers activations, gradients, and CUDA overhead.
- **Manual override**: `.prefetch(n)` or `set_prefetch_depth()` disables automatic adaptation (`user_set_depth` flag).
- **`auto_resize()`**: Manual trigger for VRAM-based resize between epochs.

#### Module Builders
- **`ConvTranspose1dBuilder`**, **`ConvTranspose2dBuilder`**, **`ConvTranspose3dBuilder`**: Fluent builder APIs for transposed convolution layers (`with_stride`, `with_padding`, `with_output_padding`, `with_dilation`, `with_groups`, `with_bias`, `on_device`, `done`). Consistent with existing Conv1d/Conv2d/Conv3d builder pattern.

#### CLI Tool
- **`fdl`** (shell script): Zero-dependency entry point. Auto-detects libtorch, Docker, Rust, GPUs. Dispatches to the compiled binary (native or Docker) with shell fallback for diagnostics. Interactive setup wizard guides users through libtorch installation and build environment selection.
- **`flodl-cli`** (`cargo install flodl-cli`): Standalone Rust binary. Pure Rust, no libtorch dependency. Works inside floDl projects and standalone (system-wide libtorch management under `~/.flodl/`). Override global root with `$FLODL_HOME`. Commands:
  - `fdl setup`: Guided wizard. Detects project vs standalone mode. In a project: system detection, libtorch download, Docker image build. Standalone: system detection, libtorch download to `~/.flodl/`, prints shell export instructions.
  - `fdl libtorch download [--cpu | --cuda 12.6|12.8]`: Auto-detect GPUs and download matching libtorch variant. Project-local or global depending on context.
  - `fdl libtorch build [--docker | --native] [--archs "6.1;12.0"]`: Compile libtorch from source for custom GPU architectures.
  - `fdl libtorch list / info / activate / remove`: Manage installed variants.
  - `fdl init <name> [--docker]`: Scaffold a new floDl project. Default mode uses mounted libtorch (like the main repo). `--docker` bakes libtorch into the Docker image for standalone deployment. Generates Cargo.toml, Dockerfiles, docker-compose.yml, Makefile, and annotated src/main.rs.
  - `fdl diagnose [--json]`: System + GPU + libtorch + compatibility report. Shows context mode (project/global). Probes GPUs via nvidia-smi, verifies libtorch arch coverage, detects Docker containers.
  - `fdl help / version`
- Pre-compiled binaries published via GitHub Releases for Linux x86_64/aarch64, macOS arm64, Windows x86_64. Downloaded automatically by the `fdl` shell script on first use.

#### Small Additions
- **`Linear::no_bias_on_device()`**: Create a bias-free linear layer on a specific device. Previously `no_bias()` was CPU-only.
- **`AdamBuilder::betas()` / `.eps()`**: Customize beta1, beta2, and epsilon in Adam per-group builder. Previously hardcoded to (0.9, 0.999) and 1e-8.
- **`AdamWBuilder::betas()` / `.eps()`**: Same for AdamW per-group builder.
- Improved doc comments on all loss functions (dtype requirements), conv builders, and optimizer constructors.

### Changed

#### Unified DDP API
- **`Ddp` is now the single entry point** for all multi-GPU training modes: `setup()` (user owns the loop), `builder()` (framework owns the loop), `wrap()` (manual).
- **Renamed**: `AsyncDdp` -> `DdpHandle`, `AsyncDdpBuilder` -> `DdpBuilder`, `AsyncDdpConfig` -> `DdpRunConfig`, `Ddp::auto()` -> `Ddp::setup()`, `Ddp::auto_with()` -> `Ddp::setup_with()`.
- **Module renamed**: `nn::async_ddp` -> `nn::ddp_run`.
- **Log prefix**: `async-ddp:` -> `ddp:` in all runtime output.
- **Deprecated aliases** preserved for backward compatibility: `AsyncDdp`, `AsyncDdpBuilder`, `AsyncDdpConfig`, `Ddp::auto()`, `Ddp::auto_with()`.

#### Unified libtorch Management
- **`libtorch/` directory**: Single host-side directory for all libtorch variants.
  - `libtorch/precompiled/cpu|cu128|cu126/` for downloaded pre-built variants
  - `libtorch/builds/<arch>/` for source-compiled variants (e.g., `sm61-sm120`)
  - `libtorch/.active` points to the variant in use
  - `libtorch/<variant>/.arch` contains metadata (cuda version, torch version, architectures, source type)
- **Docker images are libtorch-agnostic**: No libtorch baked into images. Mounted at runtime via volume.
  - `Dockerfile` (new, replaces `Dockerfile.cpu`): Ubuntu + Rust, no libtorch
  - `Dockerfile.cuda`: parameterized `CUDA_VERSION`, cudnn-devel base, no libtorch
  - `Dockerfile.cuda.source`: builder-only (no Stage 2 runtime image), Makefile extracts via `docker cp`
  - `Dockerfile.bench`: removed libtorch download, kept Python + PyTorch pip install
- **docker-compose.yml simplified**: 5 services reduced to 3 (`dev`, `cuda`, `bench`). Removed `cuda-local` and `cuda-source`. All services mount `${LIBTORCH_HOST_PATH}:/usr/local/libtorch:ro`.
- **Makefile auto-detection**: Reads `libtorch/.active` and `.arch` to derive `CUDA_VERSION` and libtorch mount path. Override: `CUDA_VERSION=12.6.0 make cuda-test`.
- **`download-libtorch.sh --project`**: Downloads to `libtorch/precompiled/<variant>/`, writes `.arch` and `.active`. Existing `--path` mode for native installs unchanged.

#### Test Infrastructure
- **15 tests un-ignored**: `cuda_event` (3), `cuda_stream` (4), DDP cross-device autograd (2) tests now run in the normal `make cuda-test` flow. They have proper mutex serialization and early-return guards.
- **NCCL/DDP/Graph tests remain `#[ignore]`**: NCCL communicator init corrupts concurrent CUBLAS operations. Must run single-threaded.
- **Process-isolated test targets**: NCCL tests run in their own cargo process to prevent CUBLAS context poisoning. Fixes SIGABRT in `test_manual_seed_reproducible` when run after NCCL init.
  - **`make cuda-test-all`**: Three-pass target - parallel + NCCL (isolated) + remaining serial.
  - **`make cuda-test-nccl`**: NCCL/DDP tests only (isolated processes).
  - **`make cuda-test-serial`** (new): Remaining serial tests (CUDA Graphs, manual_seed, probes).

#### Build Targets
- **`make setup`**: Auto-detect hardware, download CPU libtorch + CUDA libtorch (or build from source), build Docker image. One command from zero to ready.
- **`make build-libtorch`**: Compile libtorch from source, extract to `libtorch/builds/<arch>/`, write `.arch`/`.active`.
- **`make cli`** / **`make cuda-cli`**: Build flodl-cli (CPU/CUDA). **`make run-cli`** / **`make cuda-run-cli`**: Run inside Docker.
- **CI updated**: CUDA job downloads libtorch separately and mounts into container (no longer baked into image).

### Removed
- `Dockerfile.cpu` (replaced by `Dockerfile`)
- `cuda-local` and `cuda-source` docker-compose services

## [0.2.2] - 2026-03-31

### Added
- `Tensor::nbytes()` - total size in bytes (`numel() * element_size()`), matches `torch.Tensor.nbytes`

#### Fused sequence RNN kernels
- **`LSTM::forward_seq`** now calls `at::lstm()` - single cuDNN kernel for the entire sequence across all layers, replacing per-timestep cell unrolling. Eliminates N×L kernel launches (N=timesteps, L=layers) per forward pass.
- **`GRU::forward_seq`** now calls `at::gru()` - same fused optimization. Also eliminates the cuDNN benchmark variance that caused ±270ms σ in per-cell dispatch.
- **`flatten_rnn_params`** (shim) - packs per-cell RNN weight tensors into cuDNN's expected contiguous layout using `at::_cudnn_rnn_flatten_weight`, the same function PyTorch's `nn.LSTM.flatten_parameters()` uses internally. Eliminates the "RNN module weights are not part of single contiguous chunk" warning on CUDA. Uses `set_()` under `NoGradGuard` to redirect parameter storage in-place - persists across training steps, self-corrects after checkpoint load or dtype cast.
- **Flatten cache** - LSTM and GRU cache the flattened param tensors after the first forward call, skipping both the per-forward param collection (8 tensors via `flat_map` + `collect`) and the cuDNN flatten FFI call on subsequent forwards. Same strategy as PyTorch's `flatten_parameters()` but without the pointer-validation overhead.
- **`RnnParams` C++ cache** - persistent `std::vector<at::Tensor>` on the C++ side behind an opaque handle (`flodl_rnn_params_create` / `flodl_lstm_cached` / `flodl_gru_cached`). After the first forward, subsequent calls pass a single pointer to the pre-built param vector, eliminating per-forward handle collection, FFI array marshalling, and `std::vector` reconstruction. Matches PyTorch's single-call `at::lstm()`/`at::gru()` pattern exactly.
- FFI chain: `flodl_lstm` / `flodl_gru` in shim → `Tensor::lstm_seq` / `Tensor::gru_seq` in nn_ops (new `flatten` flag skips redundant flatten calls). Cached path: `flodl_lstm_cached` / `flodl_gru_cached` → `Tensor::lstm_seq_cached` / `Tensor::gru_seq_cached`.
- `LSTMCell::forward_step` and `GRUCell::forward_step` unchanged - still available for single-step / streaming use cases

#### Benchmark suite extensions
- **`transformer`** benchmark - 4-layer encoder (MultiheadAttention + FFN + LayerNorm + residual), Embedding, cross-entropy loss. B=32, seq=128, d_model=512, 8 heads.
- **`lstm_seq`** benchmark - 2-layer LSTM + linear projection, directly comparable to gru_seq. B=128, seq=50.
- **`conv_autoenc`** benchmark - Conv2d encoder + ConvTranspose2d decoder (DCGAN-style), reconstruction with MSE loss. B=64, 64×64 images.

### Changed
- **Benchmark σ uses scaled MAD** - variance column now reports Median Absolute Deviation × 1.4826 (σ-equivalent for normal distributions) instead of standard deviation. Robust to OS scheduling outliers, GC pauses, and WSL2 thermal transients that inflated stdev on long runs (e.g. gru_seq Py σ: ±143 stdev → ±27 MAD).

### Fixed
- **Benchmark report generation**: Fix silent `set -e` exit caused by `[ "$ROUNDS" -gt 1 ] && echo 's'` returning exit code 1 inside command substitution when ROUNDS=1. Reports were never written for single-round runs.
- **Benchmark report rotation**: Previous report is now rotated to `report.YYYY-MM-DD-HH-MM-SS.txt` instead of being overwritten. All rotated reports are gitignored.

## [0.2.1] - 2026-03-29

### Added

#### PyTorch Parity - Tensor Operations
- **Math ops**: `log1p`, `expm1`, `log2`, `log10`, `tan`, `asin`, `acos`, `atan`, `erf`, `erfc`, `trunc`, `frac`, `fmod`, `fmod_tensor`, `remainder`, `remainder_tensor`, `lerp`, `lerp_tensor`, `isclose`, `addmm`, `addcmul`, `addcdiv`, `clamp_min`, `clamp_max`, `selu`, `hardswish`, `hardsigmoid`, `prelu`
- **Reductions**: `prod`, `prod_dim`, `cumsum`, `logsumexp`
- **Shape ops**: `flip`, `roll`, `diagonal`, `movedim`, `tile`, `split`, `unbind`, `contiguous`, `cat_many`, `unsqueeze_many`, `narrow_scatter`, `pad_mode` (constant/reflect/replicate/circular), `meshgrid`
- **NN tensor ops**: `conv1d`, `conv_transpose1d`, `conv3d`, `conv_transpose3d`, `avg_pool2d`, `avg_pool1d`, `max_pool1d`, `adaptive_max_pool2d`, `instance_norm`, `group_norm`, `linear` (fused), `pixel_shuffle`, `pixel_unshuffle`, `bilinear`, `embedding_bag`, `interpolate` (nearest/bilinear/bicubic/trilinear), `im2col`, `col2im`, `bce_loss`, `nll_loss`, `ctc_loss`
- **Comparison/similarity**: `maximum`, `minimum`, `atan2`, `masked_fill`, `normalize`, `cosine_similarity`

#### PyTorch Parity - Autograd
- **New differentiable ops**: `leaky_relu`, `elu`, `softplus`, `mish`, `selu`, `hardswish`, `hardsigmoid`, `prelu`, `clamp_min`, `clamp_max`, `log1p`, `expm1`, `log2`, `log10`, `atan2`, `maximum`, `minimum`, `masked_fill`, `normalize`, `cosine_similarity`, `prod`, `prod_dim`, `cumsum`, `logsumexp`, `unsqueeze_many`, `cat_many`, `stack`, `triu`, `tril`
- **NN autograd ops**: `conv1d`, `conv_transpose1d`, `conv3d`, `conv_transpose3d`, `avg_pool2d`, `avg_pool1d`, `max_pool1d`, `adaptive_max_pool2d`, `instance_norm`, `group_norm`, `pixel_shuffle`, `pixel_unshuffle`, `bilinear`, `embedding_bag`, `im2col`, `col2im`

#### PyTorch Parity - Modules
- **Convolutions**: `Conv1d` (with `Conv1dBuilder`), `Conv3d` (with `Conv3dBuilder`), `ConvTranspose1d`, `ConvTranspose3d`
- **Recurrent**: `GRU` (multi-layer sequence module), `LSTM` (multi-layer sequence module) - match `nn.GRU`/`nn.LSTM` interface with `forward_seq`, batch-first support
- **Normalization**: `GroupNorm`, `InstanceNorm`, `RMSNorm`
- **Pooling**: `AvgPool2d`, `MaxPool1d`, `AvgPool1d`, `AdaptiveMaxPool2d`, `PixelShuffle`, `PixelUnshuffle`, `Upsample`, `Unfold`, `Fold`
- **Attention**: `MultiheadAttention` - self-attention and cross-attention with optional masking
- **Bilinear**: `Bilinear` - bilinear transformation `y = x1^T A x2 + b`
- **Activations**: `LeakyReLU`, `ELU`, `Softplus`, `Mish`, `SELU`, `Hardswish`, `Hardsigmoid`, `PReLU` (learnable), `Softmax`, `LogSoftmax`, `Flatten`
- **Dropout**: `AlphaDropout` - maintains self-normalizing property for SELU networks
- **Embedding**: `EmbeddingBag` - bag-of-embeddings with sum/mean/max aggregation
- **Padding**: `ZeroPad2d`, `ReflectionPad2d` - symmetric and asymmetric padding modules

#### PyTorch Parity - Losses
- `bce_loss` (from probabilities), `nll_loss`, `ctc_loss`, `focal_loss` (class imbalance), `triplet_margin_loss`, `cosine_embedding_loss`, `hinge_embedding_loss`, `margin_ranking_loss`, `poisson_nll_loss`

#### PyTorch Parity - Optimizers
- `RMSprop` (with `RMSpropBuilder` for parameter groups)
- `Adagrad` (with `AdagradBuilder` for parameter groups)
- `RAdam` - Rectified Adam with variance-aware warmup
- `NAdam` - Nesterov-accelerated Adam

#### PyTorch Parity - LR Schedulers
- `ExponentialLR` - exponential decay (`lr = base_lr * gamma^step`)
- `MultiStepLR` - decay at specific milestones
- `OneCycleLR` - super-convergence schedule (warmup + cosine decay)
- `CyclicLR` - triangular wave between base and max LR (symmetric and asymmetric)

#### PyTorch Parity - Initialization
- `kaiming_uniform`, `kaiming_normal` now re-exported at crate root
- New: `uniform`, `normal`, `orthogonal`, `trunc_normal`, `uniform_bias`

#### Test Coverage (+165 tests, 769 total)
- **Autograd gradient verification** (55 tests): finite-difference checks for every new differentiable op - `leaky_relu`, `elu`, `softplus`, `mish`, `selu`, `hardswish`, `hardsigmoid`, `prelu`, `clamp_min`/`clamp_max`, `log1p`, `expm1`, `log2`, `log10`, `maximum`, `minimum`, `masked_fill`, `cosine_similarity`, `normalize`, `prod`, `cumsum`, `logsumexp`, `tril`, `flatten`; fused NN op gradients for all conv variants (1d/2d/3d + transpose), all pooling variants, `layer_norm`, `group_norm`, `instance_norm`, `bilinear`, `embedding_bag`, `pixel_shuffle`/`unshuffle`, `im2col`/`col2im`, `grid_sample`, `gru_cell`, `lstm_cell`; Variable API coverage (`set_grad`, `set_requires_grad`, `is_leaf`, `numel`, `zero_grad_set_to_none`, `set_data`, `to_device`)
- **Module forward/backward** (60+ tests): Conv1d (builder, groups, stride/padding, no-bias, gradient), Conv2d (builder, grouped, stride, no-bias, gradient), Conv3d, ConvTranspose1d/2d/3d (forward, gradient, stride, parameters), GroupNorm (batch-size-one, single-group, groups=channels, gradient), InstanceNorm (3D input, affine parameters, gradient), LayerNorm (3D, normalization, gradient), BatchNorm/BatchNorm2d (training, eval, running stats, rejects invalid dims, gradient), Bilinear (gradient, no-bias, rejects single input), Dropout (training, eval identity, p=0), ZeroPad2d/ReflectionPad2d (asymmetric, values, no-parameters)
- **Loss functions** (20+ tests): MSE (basic, zero loss), cross-entropy (class indices, wrong predictions, gradient), BCE/BCEWithLogits (gradient), L1, SmoothL1 (negative beta rejection), KLDiv, CTC, focal (reduces to CE at gamma=0), triplet margin (zero when far), cosine embedding (similar/dissimilar), hinge embedding (positive/negative), margin ranking (with margin), Poisson NLL (log/no-log)
- **Mixed precision** (7 tests): AutocastGuard lifecycle, autocast closure, GradScaler (defaults, scale, step finite/inf, update growth/backoff, state roundtrip), cast_parameters (basic, noop same dtype)
- **Gradient clipping** (6 tests): clip_grad_norm (scales down, no-op when small, multiple params), clip_grad_value (clamps, no-op, no-grad params)
- **Graph observation** (8 tests): collect/flush/trend pipeline, reduction modes (mean, sum, min, max, norm, scalar passthrough), rejects non-scalar, map operations (over tag, slices, batched, gradient, error cases)

## [0.2.0] - 2026-03-29

### Added

#### Graph Tree (hierarchical composition)
- **Label-path addressing**: Dot-separated paths (`"encoder.scan.hidden"`) for addressing subgraphs and tags across graph boundaries. Strict dot semantics - dots always mean subgraph boundaries, no fuzzy resolution.
- **Tree registration**: Labeled graphs nested via `FlowBuilder` are automatically detected as child subgraphs. `tree_children()`, `child_graph()`, `subgraph()` for navigation. `is_composed()` flag on child graphs.
- **Selective freeze/thaw**: `freeze("encoder.read")`, `thaw("encoder.scan")`, `is_frozen("encoder")` - declarative training phase control by label path.
- **Path-based parameter collection**: `parameters_at()`, `named_parameters_at()`, `named_buffers_at()` for per-subgraph optimizer groups. Target namespace used for checkpoint compatibility.
- **Subgraph checkpoint loading**: `load_subgraph_checkpoint("encoder", "encoder_v1.fdl.gz")` - loads a checkpoint into a specific subgraph using the child's own namespace and structural hash validation.
- **Cross-boundary observation**: `tagged_at()` (null/nil semantics), `collect_at()`, `record_at()`, `trend_at()` - read tagged outputs and metrics across graph boundaries.
- **Tree-aware flush and metrics**: `flush()` automatically recurses into labeled child subgraphs. `latest_metrics()` collects from the entire tree with dotted prefixes (`"encoder.loss"`). `Monitor::log()` sees the whole tree with zero extra code. `flush_local()` and `latest_metrics_local()` for independent per-subgraph observation cadences.
- **Internal tags**: Tags prefixed with `_` are auto-internal (hidden from parent resolution). Explicit `.internal("tag")` on FlowBuilder. Cross-boundary resolution rejects internal tags.
- **Training mode propagation**: `set_training_at("encoder", false)` for selective eval mode on subgraphs (BatchNorm running stats).
- **Verbose build output**: `.verbose(true)` on FlowBuilder prints tree structure, tag resolution, and parameter summary. `tree_summary()`, `param_summary()` methods.
- **Path validation**: `validate_path()` returns `PathKind::Subgraph` or `PathKind::Tag` for build-time wiring checks.
- **Module trait**: Added `as_graph()` method (default `None`, overridden in Graph) for subgraph detection.
- **Zero forward-path impact**: All tree metadata is build-time/query-time only. The pre-computed Vec routing in `forward_impl()` is untouched.

#### Modules
- **`GaussianBlur`**: Stateless `Module` wrapper around `gaussian_blur_2d()` for use in `FlowBuilder` graphs. Fixed sigma, no parameters. Kernel size auto-computed from sigma (`2 * ceil(3 * sigma) + 1`).

#### Checkpoint Migration
- **`migrate_checkpoint()`** / **`migrate_checkpoint_file()`**: Automatically remap parameter names from an older checkpoint to match a model's current naming. Matches by exact name first, then by shape+dtype in positional order. Handles params and buffers, supports `.gz` compression. Returns a `MigrateReport` with `unchanged`, `remapped`, `dropped`, `missing` fields and a `Display` impl for human-readable output.
- **`checkpoint_version()`**: Peek at a checkpoint file's version without loading it. Returns `1` for flodl 0.1.x, `2` for 0.2.0+.
- **`MigrateReport`**: Full accounting of a migration - `is_complete()` returns true when nothing was dropped or missing.

### Changed
- **Breaking**: Checkpoint format version bumped to v2. Checkpoints saved with 0.2.0+ write version 2; `load_checkpoint` accepts both v1 and v2 (binary layout is identical, only naming conventions differ). v1 checkpoints can be migrated with `migrate_checkpoint_file()`.
- **Breaking**: Restructuring a graph with `.label()` or renaming tags changes the parameter names that feed into `structural_hash()` - the hash algorithm is unchanged, but its inputs differ. Checkpoints saved before restructuring will fail architecture validation on load. Use `migrate_checkpoint_file()` to remap parameter names, or retrain.

## [0.1.5] - 2026-03-25

### Added
- `make docs-rs` - local docs.rs build validation via disposable Docker container (nightly Rust, `--cfg docsrs`, no libtorch). Catches docs.rs failures before publishing.

### Fixed
- Fix docs.rs build: `rand` 0.9.2 uses `feature(doc_auto_cfg)` removed in nightly 1.92+. Made `rand` an optional dependency (`rng` feature, on by default) so docs.rs can build without it.
- Fix flaky `test_clip_grad_norm` - seed RNG for deterministic weights.
- Fix rustdoc broken intra-doc links in `Tensor` (escaped shape brackets, qualified method paths).

## [0.1.4] - 2026-03-25

### Fixed
- Disable example scraping on docs.rs - examples require libtorch which the docs.rs sandbox doesn't have. The scraping failure corrupted dependency artifacts, breaking the doc build.

## [0.1.3] - 2026-03-25

### Added

#### GPU Performance
- **Fused Adam/AdamW**: `_fused_adamw_` single multi-tensor CUDA kernel for the complete optimizer step across all parameters. Reduces ~4N kernel launches to 1 per parameter group. Automatic on CUDA - no API change needed. `grad_scale`/`found_inf` params exposed for GradScaler integration.
- **Foreach operations**: 7 batched tensor ops that reduce CUDA kernel launches - `foreach_add_scalar_`, `foreach_mul_scalar_`, `foreach_zero_`, `foreach_add_list_`, `foreach_norm`, `foreach_lerp_scalar_`, `foreach_sqrt_`. Used internally by fused optimizers and gradient clipping.
- **Fused gradient clipping**: `clip_grad_norm` now uses `_foreach_norm` + `_foreach_mul_` internally (2 kernels instead of 2N).
- **CUDA Graphs**: `CudaGraph` struct with capture/replay/reset for zero CPU dispatch overhead. `cuda_graph_capture()` convenience helper with warmup. `MemPoolId`, `CaptureMode` (Global/ThreadLocal/Relaxed), `cuda_graph_pool_handle()` for memory pool sharing. 2-5x speedup for models with many small kernels.
- **Autocast (AMP)**: `AutocastGuard` RAII wrapper and `autocast()` closure helper for automatic mixed-precision dispatch. Eligible ops (matmul, conv, linear) run in Float16/BFloat16 on Tensor Core GPUs. Up to 3x speedup on RTX 30xx+.
- **GradScaler**: Dynamic loss scaling for mixed-precision training. Scale, unscale with inf/nan detection, step with skip-on-inf, update with growth/backoff.

#### Tensor Operations
- **Channels-last memory format**: `Tensor::to_channels_last()` and `is_channels_last()` for NHWC layout. 8-35% speedup for Conv2d on Tensor Core GPUs.
- **Non-blocking device transfer**: `Tensor::to_device_async()` for overlapped CPU-to-GPU transfer. Pair with `pin_memory()` for maximum overlap.
- **`Tensor::copy_()`**: In-place copy with `non_blocking` parameter for async CUDA transfers. Used by CUDA Graph capture for data loading.
- **`Tensor::pin_memory()`** and `is_pinned()`: Page-locked CPU memory for fast async GPU transfers.
- **Peak VRAM tracking**: `cuda_peak_active_bytes()`, `cuda_peak_reserved_bytes()`, `cuda_reset_peak_stats()` - matches `torch.cuda.max_memory_allocated()` / `max_memory_reserved()` / `reset_peak_memory_stats()` semantics. With `_idx` variants for multi-GPU.

#### Graph Engine
- **Pre-computed routing**: `Graph::build()` pre-computes a Vec-indexed routing table. Forward dispatch uses flat array indexing instead of HashMap lookups. Cached execution buffers reused across forward calls. Zero allocation during inference.
- **Vectorized gate combination**: Gate routing stacks all expert outputs and combines via broadcast multiply + sum (~3 kernel launches regardless of expert count, vs 3N with sequential accumulation).
- **Loop fast-path**: `for_n` loops detect at call time whether refs are needed and call `body.forward()` directly when no `.using()` is chained, skipping HashMap construction and `body_step` indirection.

#### Other
- **`MaxPool2d`** module: 2D max pooling with kernel size, stride, padding, dilation, and ceil mode.
- **`Rng`** struct: CPU-side RNG (SmallRng/Xoshiro256++) with seed, shuffle, bernoulli, range, normal.
- **`manual_seed(u64)`** / **`cuda_manual_seed_all(u64)`**: Seed libtorch RNGs for reproducibility.
- **`cuda_active_bytes()`**: Query bytes backing live tensors (matches `torch.cuda.memory_allocated()`).

### Fixed
- **VRAM monitoring**: `cuda_allocated_bytes()` now returns `reserved_bytes` from the allocator, making spill detection work.
- Removed unused `ResourceSample::vram_used_bytes` field.
- Dashboard uses `vram_alloc` as the sole VRAM metric.

### Changed
- **Benchmark suite**: Publication-grade methodology with interleaved multi-round execution (`--rounds N`), GPU clock locking (`--lock-clocks FREQ`), configurable warmup (`--warmup-secs`). 7 benchmarks (3 standard + 4 graph-builder). Peak VRAM tracking (not snapshots). WSL2 host-side clock management via `bench-publish.ps1`. `make bench-publish` for reproducible runs.
- **Docker**: `.dockerignore`, BuildKit cache for libtorch downloads, skip-if-exists image targets, dedicated bench image.

## [0.1.2] - 2026-03-19

### Added
- **VRAM spill detection**: New FFI function `flodl_cuda_alloc_bytes` queries libtorch's CUDA caching allocator. `cuda_allocated_bytes()` / `cuda_allocated_bytes_idx()` expose it in Rust. When allocated bytes exceed physical VRAM, the monitor shows spill in terminal output, live dashboard, CSV export, and epoch log.
- `ResourceSample::vram_allocated_bytes` field for allocator-level memory tracking.
- `vram_spill` column in CSV export.

### Fixed
- README links now use absolute GitHub URLs - fixes broken links on crates.io where relative paths don't resolve.

## [0.1.1] - 2026-03-18

### Fixed
- Replace `sha2` with `hmac-sha256` - fixes docs.rs build (sha2's asm feature doesn't compile on docs.rs).
- Widen leak test tolerance for CI parallel test jitter.

## [0.1.0] - 2026-03-18

### Added
- **Graph identity**: `Graph::structural_hash()` - deterministic SHA-256 hash of graph topology, module names, and parameter/buffer shapes. Any architecture change produces a different hash. `Graph::short_hash()` returns the first 8 chars. `FlowBuilder::label()` sets a human-readable name (does not affect hash).
- **Checkpoint architecture validation**: Checkpoint format v1 embeds a 32-byte structural hash. `load_checkpoint` / `load_checkpoint_file` accept an optional hash and error on architecture mismatch.
- **Dashboard metadata**: `Monitor::set_metadata(serde_json::Value)` attaches hyperparameters/config to the HTML archive. `watch()` / `watch_profiled()` capture graph label and hash. Dashboard header shows `"floDl - {label} [{hash8}]"`.
- **Parameter freezing**: `Parameter::freeze()`, `unfreeze()`, `is_frozen()` - disable/enable gradient tracking per parameter. Optimizers automatically skip frozen params (no grad). `Parameter::to_device()` now preserves frozen state.
- **Named checkpoints**: `Graph::named_parameters()` and `named_buffers()` return qualified names (`"tag/weight"` or `"node_id/running_mean"`). `save_checkpoint` / `load_checkpoint` persist both parameters and buffers (e.g., BatchNorm running stats), matching by name for partial loading. `LoadReport` reports what was loaded, skipped, and missing.
- **Optimizer parameter groups**: `Adam::with_groups()`, `SGD::with_groups()`, `AdamW::with_groups()` - builder API for per-group learning rates. `Optimizer::set_group_lr()` adjusts a single group; `set_lr()` updates all groups. Groups are persisted through `Stateful` save/load.

### Core Stack
- **Tensor**: Owned RAII tensors with Drop, ~72 operations. CPU and CUDA (feature-gated).
- **Autograd**: Reverse-mode AD backed by libtorch's native autograd engine. 37 differentiable operations with numerical gradient verification.
- **NN Modules**: Linear, Conv2d, ConvTranspose2d, LayerNorm, BatchNorm, Dropout, Embedding, GRUCell, LSTMCell.
- **Activations**: ReLU, Sigmoid, Tanh, GELU, SiLU.
- **Losses**: mse_loss, cross_entropy_loss, bce_with_logits_loss, l1_loss, smooth_l1_loss, kl_div_loss.
- **Optimizers**: SGD (with momentum), Adam, AdamW.

### Graph Builder
- Fluent API: from/through/build, split/merge, also (residual), tag/using (named refs).
- Loop constructs: for_n (fixed), while_cond (pre-condition), until_cond (post-condition).
- Routing: gate (soft, weighted), switch (hard, selected branch only).
- Map constructs: each, over, slices, with batched fast path.
- Input (auxiliary graph inputs), tag_group (auto-suffixed parallel branch names).

### Training Tools
- LR scheduling: StepDecay, CosineScheduler, WarmupScheduler (composable), PlateauScheduler.
- Mixed precision: Float16/BFloat16 dtype casting, GradScaler for loss scaling.
- Gradient clipping: clip_grad_norm, clip_grad_value.
- Checkpointing: save_checkpoint/load_checkpoint (named binary format with LoadReport, persists parameters + buffers, structural hash validation, file or io::Write).
- Weight initialization: kaiming_uniform/normal, xavier_uniform/normal.

### Training Monitor
- Human-readable ETA with adaptive formatting (hours/minutes/seconds/milliseconds).
- System resource tracking: CPU, RAM, GPU utilization (NVML), VRAM usage.
- Live web dashboard via embedded HTTP server with Server-Sent Events.
- Dashboard features: real-time training curves, resource usage charts, epoch log, graph SVG, label/hash header, metadata card.
- CSV and log file export.

### Observation & Visualization
- Tag-based metric collection: collect/flush/trend.
- Trend analysis: slope, stalled, improving, converged.
- Group trends with tag_group expansion.
- DOT/SVG graph visualization with parameter counts and node type shapes.
- Profiling: enable_profiling, profile, timing trends.
- Training curves: plot_html, export_trends, write_log.

### Infrastructure
- **CI**: GitHub Actions with CPU test matrix and CUDA build verification.
- **Docker**: CPU and CUDA Dockerfiles, docker-compose with GPU support.
- **Build**: Makefile with cpu/cuda targets (build, test, clippy, shell).

### Testing
- 389 library tests + showcase tests.
- Zero clippy warnings.
- Autograd numerical gradient checks.
- Module-level gradient checks.

### Key Design Decisions
- **Deterministic VRAM**: Rust's Drop trait replaces 5 phases of GC-based memory management.
- **No GC overhead**: No runtime.KeepAlive, no pending-free queues, no VRAM budget heuristics.
- **Variable**: `Rc<RefCell<VariableInner>>` for cheap Clone with interior mutability.
- **Module trait**: single-input forward + optional NamedInputModule for multi-input. `structural_hash()` for architecture identity.
- **Graph-as-Module**: Graph implements Module for hierarchical composition.
- **NamedInputModule on routers**: SoftmaxRouter and SigmoidRouter sum refs into input before projection.
- **Native FFI ops**: flodl_max, flodl_norm, flodl_cuda_mem_info, flodl_cuda_utilization.
