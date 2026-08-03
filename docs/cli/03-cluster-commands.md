# Cluster and Hardware Commands

## `fdl probe`

Cluster readiness audit. Single-host (default) probes the local box;
in cluster context (`fdl @cluster probe` or `FDL_ENV=cluster fdl probe`)
SSHes every host in `fdl.cluster.yml` and aggregates the report. Use
it as a CI gate before launching multi-hour training runs.

```bash
fdl probe                       # local host: GPU + libtorch + NCCL + shared-data
fdl @cluster probe               # multi-host: SSHes each worker, aggregates
fdl probe --json                # machine-readable for CI gating
fdl @cluster probe --json        # cluster JSON aggregate
fdl probe --skip-mount          # skip shared-data-mount check on single-host setups
fdl probe --data-path /flodl/data        # override the shared-data path
fdl probe --libtorch-path /opt/libtorch  # override the libtorch directory
fdl probe --docker cuda         # NCCL is provided by a Docker image (compose service)
```

Exit code: **0** when every checked component is green, **1** when
any issue was surfaced. The green path is silent enough to use as a
post-deploy smoke test.

**What it checks:**

- **GPU inventory**: count, name, sm version, VRAM per device (via
  `nvidia-smi`, no libtorch context touched).
- **libtorch variant**: active variant, version, CUDA version, arch
  coverage, source (precompiled vs source-built).
- **GPU/libtorch arch compatibility**: every visible GPU's sm version
  is covered by the active libtorch's `archs:` metadata.
- **NCCL availability**: host-level `libnccl.so` linkage, version. On
  workers with `docker: <svc>` declared in `fdl.cluster.yml`, the
  probe reports "via Docker image `<svc>`" instead of erroring on a
  missing host-level libnccl.
- **NCCL version skew (cluster mode)**: surfaces major.minor skew
  across hosts - the common failure mode on heterogeneous rigs.
  Resolution: [`fdl nccl build`](#fdl-nccl) to bridge.
- **Shared-data path**: convention default `/flodl/data` (override via
  `--data-path` or per-host `data_path:` in cluster.yml). Verifies
  every host can see the same mount.
- **`fdl.cluster.yml` schema**: warns on legacy keys (`master_addr`,
  `master_port`, top-level `ssh_*` on workers).
- **Dashboard port availability** (default 3000).

Output splits results into warnings (informational; do not block) and
errors (block dispatch). `fdl probe` is a manual readiness gate — run it
yourself before a cluster run; it is not invoked implicitly by fan-out.
(The automatic pre-flight step `fdl @cluster <cmd>` performs is the
per-host binary build, skippable with `--no-prebuild`.)

## `fdl status`

Live status of a running cluster: lifecycle phase (`waiting` /
`staging` / `forming` / `training` / `done` / `failed`), who has
joined with what hardware, the join-window countdowns while it is
still open, and — on `start: manual` / `hybrid` runs — the start
switch's state (a staging roster renders **"roster startable, fire
with `fdl start`"**). The controller serves the state as `state.json`
over plain HTTP on its training port, so no extra port or config is
involved — and `curl` works where fdl isn't installed.

```bash
fdl @cluster status              # controller from the overlay's cluster.yml
fdl status --addr host[:port]    # explicit controller (default port 1337)
fdl @cluster status --json       # raw state.json for scripts
curl http://<controller>:1337/state.json   # same truth, no fdl
```

Address resolution: `--addr` wins; otherwise the active env's
`cluster.controller` (with a loopback retry, so all-tunneled runs are
found when running on the controller box); otherwise the convention
default `127.0.0.1:1337` (single-host auto-promoted runs), noted on
stderr.

Exit code: **0** when the state was fetched and printed, **1** when no
endpoint answered. The endpoint lives exactly as long as the launcher
process — connection-refused after a run ends is the expected "no run
listening" signal, not a fault.

## `fdl start`

Fire the operator start switch of a **staging** cluster run. A join
window opened with `controller.join.start: manual` (or `hybrid`) holds
the roster once quorum is met instead of forming on the clock — the
scavenged-credit shape: launch instances until the money runs out,
watch `fdl status`, start when the roster looks full enough.

```bash
fdl @cluster start               # controller from the overlay's cluster.yml
fdl start --addr host[:port]     # explicit controller
fdl start --token <hex>          # non-loopback fire: the run's join token
```

Trust mirrors join admission: fired from the controller box (or
through the sshd tunnel) the loopback peer address is the credential
and no token is needed; from anywhere else `--token` must match the
run's `controller.join.token`. Address resolution is the same as
[`fdl status`](#fdl-status).

Refusals name their reason and are never queued — auto mode (no switch
to fire), quorum not met (with counts), window already closed, bad
token. Exit code: **0** when the start was armed (the world forms at
the next poll — watch `fdl status`), **1** otherwise.

## `fdl join`

Join a cluster run's window as a **self-deployed worker**: the
worker-side walk-in for discovery windows
(`controller.join.discovery: true`), where the window alone defines
the world and worker addresses need not exist in any roster. It dials
the controller, offers the box's GPUs, and runs your training binary
in agent role — the binary joins, then spawns and supervises this
host's relay and rank children itself; training code downstream is
byte-identical to the fan-out path.

```bash
# Direct dial (trusted segment), authenticated by the run token:
fdl join 10.0.0.1:1337 --token <hex> --bin target/release/train -- --model resnet

# Through a guardrailed sshd on the controller box (the controller
# binds loopback under `tunnel_only`; reachability = authentication):
fdl join --ssh flodl-join@ctrl.example.com --identity ~/.ssh/join_key \
         --bin target/release/train -- --model resnet
```

- `--ssh [user@]host[:port]` brings up a local `ssh -L` forward of the
  controller port (fresh per attempt, `ExitOnForwardFailure`, never a
  password prompt) and dials through it. The positional controller
  address is then as seen FROM the ssh host — default `127.0.0.1:1337`.
- Arguments after `--` go to the binary verbatim and must match the
  run: rank children re-enter the binary with them.
- `--devices 0,1` scopes the GPUs offered (default: all);
  `--host` names the worker in the roster (default: hostname).
- `--persist` re-dials with backoff (5s doubling to 60s) whenever the
  agent exits — no window open yet, run finished, controller rebooted —
  the systemd / golden-image mode.
- Inside a project, the active libtorch's `lib/` rides
  `LD_LIBRARY_PATH` onto the binary automatically (`FDL_LIBTORCH_CASE`
  honored) and its variant label rides the join hello.

Every flag defaults from a top-level `join:` block in `fdl.yml` (see
`fdl.yml.example`), so a golden image boots into bare `fdl join`.

### Preparation, before the dial

Admission starts a window deadline, so everything a box needs is
acquired before it dials, and re-acquired on every attempt — which is
what makes `--persist` a provisioning loop: a box picks up a changed
source on its next re-dial, with no reprovisioning.

- **The GPU gate.** No usable device at all, and the box does not dial.
  The agent already refuses an empty device list, but only *after*
  admission — by then this host has been counted into a quorum and takes
  the cohort's formation down with it. The bar is "any usable device",
  not "nothing to report": an unusable card beside working ones (an AMD
  iGPU with no ROCm runtime, say) is what `fdl probe` flags and what a
  perfectly trainable box looks like. Those findings become the
  *explanation* when there is genuinely nothing.
- **The dataset source root.** `--data-path` is the local path this
  box's ranks read from; it is verified, then shipped to them, so the
  training binary needs no data flag. `--data-source` mounts it first
  when it is not already there:

  ```bash
  fdl join --ssh flodl-join@ctrl --bin target/release/train \
           --data-source sshfs://flodl-join@ctrl:/flodl/data
  ```

  The mount goes up **read-only**: a rank reads the source root and
  never writes it (anything missing is acquired into `~/.flodl/data`
  instead), so the kernel enforces what was otherwise a convention. An
  already-mounted path is left alone and reused; a mount from a
  *different* source is reported and still reused, since unmounting
  behind the operator would be worse. Credentials come from the `ssh:`
  block — same box, same key, and that key has to permit sftp: a join
  key guardrailed with `command="/usr/sbin/nologin"` refuses it, so the
  tunnel comes up and the mount says permission denied. Either the key
  carries `command="internal-sftp -R -d /flodl/data"` instead, or the
  root is mounted during provisioning and `--data-path` is declared
  bare. Both are in the [guardrail
  recipe](../ddp/02-cluster-guide.md#dial-in-membership-the-join-window).
- **The local directories.** `~/.flodl/data` (the across-run dataset
  cache) and the temp dir (the within-run disk stage) are proven writable
  by writing. RAM-backed (`tmpfs`) or nearly-full volumes are reported,
  not refused.
- **libtorch**, when `--libtorch` names a variant: acquired into
  `~/.flodl/libtorch/` and made active. `auto` routes on the devices
  *this* box has, which is what lets one golden image serve NVIDIA and
  AMD instances. Never into the project tree, which on a walk-in is
  frequently a read-only mount.
- **The training binary**, when `--source` names a tree instead of
  `--bin` naming a path. The tree is fetched to local disk and built
  there, so it links against the libtorch this box holds and the ABI
  matches by construction:

  ```bash
  fdl join --ssh flodl-join@ctrl --libtorch auto \
           --source rsync://flodl@ctrl:/home/op/my-train \
           --source-bin target/release/my-train
  ```

  Building from a mount is never an option: cargo fingerprints by
  stat'ing every source file on every invocation, and the attribute
  caching that would hide that latency makes it serve a stale binary. The
  fetch preserves mtimes, so the build stays incremental across dials
  rather than being a cold rebuild in an incremental costume.

  `--source-cwd` is the project directory inside the tree and governs the
  build and the run both; `--source-build` is a shell line (default
  `cargo build --release`) and can be a script committed beside the code,
  so the recipe travels with the source while its invocation stays in the
  box's config. It receives `LIBTORCH_PATH`, `FDL_GPU_FEATURE` and
  `LD_LIBRARY_PATH`. There is deliberately no toolchain flag: the tree
  carries its own `rust-toolchain.toml` and lockfile when the operator
  pinned them, and `RUSTUP_TOOLCHAIN` set here would silently override
  that.

A source your provisioning already mounts needs no scheme at all: name
its path in `--data-path`. Nothing is checked and nothing is shipped
when neither field is set.

Exit codes:

| code | meaning | `--persist` |
|---|---|---|
| **0** | the agent's own: this host's ranks all finished cleanly | re-dials |
| **1** | transient failure — controller unreachable, mount or fetch attempt failed, **the source did not compile** | re-dials |
| **2** | permanent failure — no GPU, a spec that does not parse, a missing binary or toolchain, unwritable stage | **stops** |

A compile error being transient is deliberate rather than lenient. The
source is remote, so the fix is a push away, and the systemd recipe below
pairs code 2 with `poweroff`: a box that stopped permanently over a typo
would take the fleet with it.

fdl never powers a box off itself; 2 is how the thing that owns the
instance hears about it:

```ini
# /etc/systemd/system/flodl-join.service
Restart=always
RestartPreventExitStatus=2   # stop hot-looping a misprovisioned box
FailureAction=poweroff       # ... and self-deprovision it
```

Full protocol walkthrough, trust model, and the join-sshd guardrail
recipe: [DDP reference](../ddp/02-cluster-guide.md#dial-in-membership-the-join-window).

## `fdl nccl`

Build NVIDIA's `libnccl` from source. Required for heterogeneous-rig
clusters when the bundled NCCL versions across hosts don't match (NCCL
refuses handshake across major.minor skew). The build runs in a
dedicated Docker context (`Dockerfile.nccl.source`) so no host-level
NCCL/CUDA toolchain is required.

```bash
fdl nccl build                              # auto-detect target tag + local GPU archs
fdl nccl build --tag v2.27.5                # explicit NCCL git tag
fdl nccl build --archs "6.1;12.0"           # explicit archs (heterogeneous rig)
fdl nccl build --jobs 8                     # parallel compilation jobs (default 6)
fdl nccl build --dry-run                    # print build plan, do nothing
```

Auto-detection:

- **Target NCCL tag**: read from the active libtorch variant's
  `third_party/nccl` submodule version. Override with `--tag` for
  pre-release or version-pinned builds.
- **Archs**: from local GPUs (multi-arch builds supported, e.g.
  `sm_61 + sm_120` for a Pascal + Blackwell rig).

**Output path**: `libtorch/nccl/builds/v<version>-<archs>/lib/libnccl.so.2`.

Wire it into a worker via the `env: LD_PRELOAD:` block in
`fdl.cluster.yml`:

```yaml
workers:
  - host: node-b
    arch: builds/sm61-sm120
    env:
      LD_PRELOAD: /srv/flodl/libtorch/nccl/builds/v2.27.5-sm61/lib/libnccl.so.2
```

Build time: 5-15 minutes depending on CPU cores and arch count.

## `fdl --gpus`

Scope GPU visibility for a single command. Two forms:

```bash
fdl --gpus all <cmd>            # use every visible CUDA device
fdl --gpus 0,1 <cmd>            # explicit physical indices
```

Behavior depends on the command kind:

- **Cluster-aware commands** (`cluster: true` in `fdl.yml`, like
  `ddp-bench`): N ≥ 2 GPUs trigger synthesis of a single-host
  cluster envelope (loopback controller, one host with N ranks) and
  process-per-rank fan-out via the standard launcher. N = 1
  degenerates to a single-process run on that device.
- **Non-cluster commands** (`test`, `clippy`, …): `--gpus` sets
  `CUDA_VISIBLE_DEVICES` for the dispatched subprocess. No envelope
  synthesis, no spawning.

**Cluster context interaction**: on `fdl @cluster <cmd>`, `--gpus`
overrides per-worker `local_devices:` for the local controller host;
remote workers continue to use their cluster.yml-declared devices.
Loud-errors on duplicate, missing value, or invalid spec.

```bash
fdl --gpus 0 test                # CPU-style: scope tests to GPU 0
fdl --gpus 0,1 ddp-bench --mode nccl-cadence   # synthesize 2-rank single-host cluster
fdl --gpus all @cluster ddp-bench --mode nccl-cadence   # override local host devices in cluster mode
```

## `fdl @cluster <cmd>` - multi-host fan-out

`fdl @cluster <cmd>` selects the `cluster` env overlay via the `@`
sigil. When `fdl.cluster.yml` exists alongside `fdl.yml`, it
deep-merges as an overlay and triggers SSH fan-out for any command
marked `cluster: true`.

Three equivalent forms:

```bash
fdl @cluster <cmd>           # sigil form
fdl --env cluster <cmd>      # explicit flag
FDL_ENV=cluster fdl <cmd>    # env-var form
```

**Pre-flight per-host build**: before fan-out, `fdl @cluster <cmd>`
auto-builds the target binary locally for every remote host. Per-host
`CARGO_TARGET_DIR=target/cluster/<host>/<arch>/`, libtorch resolved from
each host's `arch:` declaration, CUDA feature derived from the host's
GPU arch metadata. (Keying the target dir on `arch` as well as `host`
means changing a host's `arch:` rebuilds cleanly instead of reusing a
binary linked against the old libtorch.) Builds run in parallel per
host; first failure aborts fan-out. Remote dispatch invokes the prebuilt binary directly -
no cargo, no rustc on remote.

Pass `--no-prebuild` to skip the pre-flight phase (when binaries are
known fresh, or when iterating on a build-only issue).

**Heterogeneous-rig flow** (a Blackwell host + a Pascal VM, say):

1. `fdl @cluster probe` - confirm GPU + libtorch + NCCL match per host.
2. `fdl nccl build` on the host with the older NCCL - produces a
   matching `libnccl.so.2` to wire via `env: LD_PRELOAD:`.
3. `fdl @cluster <cmd>` - fan out.

See [DDP Reference: Multi-host
clusters](../ddp/02-cluster-guide.md) for the `fdl.cluster.yml`
schema and conventions.

<!-- nav: generated by site/build_guide.py — do not edit below -->

---

Previous: [Project and libtorch Commands](02-setup-commands.md) | Next: [Introspection and Tooling](04-tooling-commands.md)
