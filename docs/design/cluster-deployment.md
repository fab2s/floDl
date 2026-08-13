# Cluster Deployment Architecture

flodl's distributed-training stack ships in three structurally
independent layers - **storage**, **orchestration**, and **compute** -
that production users can scale, replace, and pay for separately. The
default in-tree dev path (single host, controller + averager + ranks
in one process tree) is the special case where all three layers
collapse into one machine; the design accommodates that without
locking users into it.

This document captures the deployment shape, the principles that
constrain it, and the future slices that implement what's currently
design-only. It does not cover communication algorithms (see
[cloud-ddp.md](./cloud-ddp.md)) or training-loop semantics.

> **Status lines here are dated, the code is not.** Every "today" and
> "future" claim below was last reconciled against the tree on
> **2026-08-08**. For what a command actually does now, `fdl <cmd> -h`
> and [the CLI reference](../cli/03-cluster-commands.md) are
> authoritative over this file.

## The three layers

### Storage

A shared filesystem accessible by every node in the cluster, hosting:

- **Datasets** - read-mostly. NFS, S3-FUSE, virtiofs, NetApp, anything
  that mounts. Production users handle this exactly the way they'd
  mount any other ML dataset.
- **libtorch builds** - read-only after provisioning. One install per
  arch (or one multi-arch variant covering many) lives here.
- **Training binaries** - compiled artifacts the controller ships
  out per training run. Read-only at runtime.
- **Checkpoints** - written by ranks during training; read at resume.
  Bundle layout per
  [`CheckpointBundle`](../../flodl/src/distributed/checkpoint_meta.rs).
  The bundle is **split across hosts** and each piece is written on its
  writer's host: the user's `checkpoint_fn` runs on the controller host
  (CPU backend) or the elected rank's host (NCCL), each surviving worker
  writes its `.fdl` / `.optim` on its host, and the controller writes
  (and reads back at resume) the consensus `.fdl` + `.meta.json` on its
  host. So a
  `save_path` / `resume_from` that is *not* on this shared layer
  scatters the bundle and breaks resume - which is exactly why
  checkpoints belong here. flodl prints a one-time reminder on any
  genuine multi-host launch with a checkpoint path set.

This layer is **always on** and **cheap relative to compute**. A
small file server, a NAS appliance, or a $20/month object-storage
mount covers most clusters. It does NOT need a GPU. It does not
participate in any training collective.

For dev rigs (single host + a passthrough VM), the analog is a
virtiofs mount from host into VM. Same mental model, same code path.

### Orchestration

The **controller** runs `fdl-cli` in its launcher role: parses
cluster topology, fans out per-host child processes via SSH, tails
their logs, propagates exit codes, and (when the cluster-aware path
is active) hosts the `ClusterCoordinator` that drives heartbeat
detection, dead-rank declaration, NCCL re-rendezvous, ExtendPartition
dispatch, and the `ShutdownWithSave` broadcast.

This layer needs:

- An IP address and ports reachable by ranks (control channel +
  TCP averaging channel + NCCL rendezvous port).
- Enough CPU/RAM to handle control-plane message routing (negligible).
- Optionally: GPUs, if the controller also wants to participate as a
  rank. **Not required.**

A `t3.small` EC2, a Raspberry Pi, or an unused desktop is enough.
The orchestrator is **always-on capable** and **cheap**.

### Compute

The GPU nodes. Each runs one training-binary process per rank,
managed by `flodl::distributed::launcher` (the per-rank entry point).
Each rank holds a `GpuWorker` on its assigned device + bridge threads
to the control + data channels.

The CPU-averaging service (`ClusterController` in flodl's TCP
averaging path) is conceptually a compute-layer service too,
NOT an orchestration concern. It does real work: receives every
rank's parameter snapshots, computes `reduce_average_alive`, and
broadcasts the averaged params back. For large models the network
+ CPU budget is non-trivial.

Today these collapse into one process tree; tomorrow they should
be deployable independently. See "Controller / averager separation"
below.

### Network reachability (two planes, one is a full mesh)

flodl's own transport is **hub-and-spoke**: every worker dials the
controller, nothing else. From `controller.port` (base, default 1337):

| Port | Bind | Purpose |
|---|---|---|
| base | `0.0.0.0` | NCCL-UID bootstrap rendezvous |
| base+1 | — | reserved (dashboard side-channel) |
| base+2 | `0.0.0.0` | CPU-averaging `ClusterController` |
| base+3 | `0.0.0.0` | `ClusterCoordinator` control channel |
| base+4 / +5 | `127.0.0.1` | per-host relay loopback (rank ↔ relay) — **not** cross-host |

So for flodl's transport, only the controller needs inbound from
workers; workers need no inbound from each other, and the `+4/+5` relay
ports never leave `localhost`. This is NAT-friendly on the control side.

The **NCCL data plane is a separate, full-mesh network** and is the
cloud gotcha. On the NCCL backend, once the UID is exchanged via
rendezvous, NCCL forms its communicator with `ncclCommInitRank` and
opens **direct rank-to-rank connections on ephemeral ports** over the
interface named by each host's `nccl_socket_ifname` (required on any
multi-host cluster — see `rendezvous.rs`). This needs **all-to-all TCP
among the workers**, which flodl neither brokers nor sees.

On a cloud with default-deny security groups or NAT between workers,
this manifests as a silent hang inside `ncclCommInitRank` with no
flodl-level diagnostic (flodl's own hub-and-spoke channels connect
fine; only NCCL's mesh stalls). Deployment requirement: open all-to-all
TCP among the worker nodes on the `nccl_socket_ifname` interface, and
pin NCCL's port range (`NCCL_PORT_RANGE` / IB/socket env) if the
security group must enumerate ports. The CPU-averaging backend has no
mesh — it routes only through the controller — so it is the fallback
for topologies where an all-to-all NCCL mesh is not achievable.

### Network deadlines and `FLODL_NET_TIMEOUT_SCALE`

flodl's transport deadlines are LAN-tuned defaults that together define
**one coherent notion of "gone"** — a peer silent past ~30s:

| Budget | Default | Detects |
|---|---|---|
| TCP connect (all cluster dials) | 60 × 500ms ≈ 30s | controller not up yet |
| write-stall (every cluster socket) | 30s zero-progress | wedged peer / dead link |
| coord heartbeat staleness (coord side) | 30s | silent rank |
| coord-liveness (rank side) | 30s | wedged coordinator |
| CPU reduce read deadline | 120s per-read silence | vanished controller mid-round |

On a slower link (WAN, NAT hub-and-spoke — a declared target of the
CPU-averaging path) set `FLODL_NET_TIMEOUT_SCALE` where you run `fdl`
(e.g. `3` ≈ "gone" at 90s): it scales the whole set together and is
forwarded to every remote rank/relay automatically, so the cluster
keeps a single notion of "gone". Values below `1` (floor `0.1`) shrink
the deadlines for fast-failure test rigs. An explicit
`heartbeat_timeout_secs` on the trainer config overrides the heartbeat
pair unscaled. SSH keepalives are a separate axis (ssh transport, not
flodl wire): tune per host via `ssh.options:`
(`["ServerAliveInterval=30", ...]`) — user options win over flodl's
defaults.

## Principles

These principles together produced the testing-convention slice and
the elastic-membership work; restating them keeps future
extensions consistent.

### Ship binaries, not source

Remote nodes need glibc + libtorch.so + libnccl.so + the training
binary. No remote `cargo`, no remote `rustup`, no remote `git`. The
existing flodl-cli multi-arch GitHub-Actions CI already does this for
the `fdl` binary itself; the same pattern generalizes to user
training binaries via a future `fdl deploy` command.

Matches how every modern ML serving stack (Triton, TorchServe, vLLM)
already works. PyTorch's "ship source, install in venv, JIT-compile
at startup" pattern is research-grade, not production-grade.

### libtorch is a pre-provisioned asset, never a deploy-time concern

> "It would just be insane to start a cluster of H100 without the
> right libtorch."

Building libtorch from source takes hours. Downloading the official
~6GB build takes minutes on a slow link. **Neither belongs in the
critical path of starting a training run.** Each host has libtorch
installed at a known path **before** any deploy attempt. `fdl deploy`
probes for compatibility and refuses to start the run if anything's
missing, with a clear "provision libtorch on host X via `fdl libtorch
download <variant>`" error.

The same applies to libnccl and any other system-level dependency.

### Arch is declared in yml, validated by probe

Every host entry in the cluster topology declares its GPU compute
capability (`arch: sm_NN`). The deploy probe verifies the declared
arch matches the probed hardware:

- **Match** → proceed.
- **Mismatch, but a locally-provisioned libtorch variant covers the
  probed arch** → hot-swap with warning. The hardware was likely
  swapped out behind the controller's back; deploy adapts.
- **Mismatch, no variant covers** → loud error with the exact
  provisioning command. Production-safety mode (`--strict`) escalates
  any mismatch to error.

The arch field landed as an optional schema entry in
[`ClusterWorker`](../../flodl-cli/src/config/cluster.rs); the probe consumer
is future work.

### Shared storage > per-host source-of-truth

Data, libs, and binaries live on the shared layer. Each host
mounts them; nothing is rsync'd or git-pulled at training-start time.
This is how real datacenters work. NFS / S3-FUSE / NAS for prod;
virtiofs for dev rigs. Either way, the storage layer **is** the
source-of-truth - eliminates entire classes of "host X has a stale
checkout" failures.

## Controller / averager separation (future capability)

Today the launcher process hosts both `ClusterCoordinator` (cheap
orchestration) and `ClusterController` (heavy CPU averaging - receives
every rank's params, computes mean, broadcasts back). They share an
address space because they were sequenced together; nothing in their
contracts requires it.

Production users will want them separable:

| Role | Live duration | Hardware profile | Cost |
|------|---------------|------------------|------|
| Coordinator | Always-on capable | Cheap CPU, low RAM, no GPU | $5-20/mo cloud VM |
| Averager | Per training run | High bandwidth, moderate CPU | $$ per hour, on-demand |
| GPU ranks | Per training run | High GPU + interconnect | $$$ per hour, on-demand |

Cluster.yml gains optional separate addressing - `coordinator_addr`
+ `coordinator_port` distinct from `averager_addr` + `averager_port`.
Falling back to the controller's single mux port (current behavior:
both channels accept there, routed by channel-select magic) when both
unset preserves the single-host dev path.

This separation also makes the cheap orchestrator a viable
multi-tenant role: one always-on coordinator can sequence many
training runs over time, each spinning up its own ephemeral averager
+ GPU pool, freeing them after completion. Matches how Kubernetes
scheduler relates to pods.

Status: **design-only**. The wire frames and config fields are
already in place; what's missing is making the
coordinator run as its own process (no launcher fan-out) and the
averager run as its own process (no launcher fan-out). Each becomes
a long-running service or a one-shot spun up per training run.

## Cloud portability of `fdl libtorch build`

Most users won't need this: PyTorch's official precompiled libtorch
already covers cloud-typical archs (sm_70 V100, sm_75 T4, sm_80
A100, sm_86 A10, sm_89 L4, sm_90 H100, sm_100/120 Blackwell) on
manylinux2014 base with cudnn + cufile bundled. For those users,
"deploy is portable" requires nothing beyond using the official
precompiled variant.

Edge cases (one extreme: Pascal sm_61 + Blackwell sm_120 on the same
cluster - actual dev rig used to validate this arc) require custom
libtorch builds. Today `fdl libtorch build` produces native-glibc
binaries; for cloud portability of custom builds, add a `--portable`
flag that:

1. Builds in a **manylinux2014 base container** (glibc 2.17 floor;
   runs on Ubuntu 20.04+, AL2, RHEL 8+).
2. Defaults to a **broad arch list** (`sm_61;sm_70;sm_75;sm_80;sm_86;
   sm_89;sm_90;sm_100;sm_120`) covering Pascal through Blackwell.
3. **Bundles** libcudnn.so.9 + libcufile.so.0 + libcublas + libcudart
   in the output `lib/` dir, matching what the precompiled variant
   ships.
4. Names the output `libtorch/builds/<arch_set>-portable/` so it's
   distinguishable from native-only builds.

Cost: ~50-80 LOC in flodl-cli + a Dockerfile addition. Default
behavior unchanged; `--portable` is opt-in.

Status: **design-only**. Easy slice to land.

## Future slices

In rough sequencing order:

`fdl probe` **shipped** and is no longer on this list: it SSHes each
host, aggregates arch, libtorch variant, NCCL version, shared-data path
and prerequisites, and is loud on mismatch with `--json` for machine
use. It grew past the sketch below, notably the loader-compatibility
check that names a variant this host's glibc cannot load.

| Slice | Scope | LOC est |
|-------|-------|---------|
| `fdl deploy <env>` | Calls probe internally, ships binary via shared storage layer or rsync, runs. Hot-swap libtorch variant when probed arch ≠ declared but a local variant covers. `--strict` flag escalates any mismatch to error. | ~250 |
| `fdl libtorch build --portable` | Manylinux2014-base container build, broad arch list, bundled cudnn/cufile. | ~80 + Dockerfile |
| Controller/averager separation | Make `ClusterCoordinator` runnable as standalone process; same for `ClusterController`. yml gains optional separate addressing; launcher spawns them independently or talks to existing instances. | ~300 |

None of these are blocking: single-host and multi-host both work today,
by fan-out and by dial-in. They're the remaining steps from "runs on
the hosts you name" to "datacenter-grade deployment" without rewriting
the foundation.

## Today's state

Reasonable production user with `--gpus all` on a single multi-GPU
host: **works**. The launcher synthesizes a single-host cluster from
visible GPUs, the controller + averager run in-process, NCCL elastic
membership survives rank death, ShutdownWithSave persists state on
unrecoverable failure.

Multi-host production user with `fdl @cluster <cmd>`: **works**.
libtorch provisioning is still per host (`fdl libtorch download` on
each, separately), but `fdl probe` now answers whether that provisioning
is coherent before a run spends a window on finding out. Self-deployed
workers arrived too: a box can dial in with `fdl join` rather than being
fanned out to, `fdl publish` makes the controller the authority for what
a run is, and `fdl join-config` scaffolds the whole farm. Still deferred:
auto-deploy, controller/averager separation, and a shared-storage
abstraction in the cluster topology.

Test rig with `fdl @cluster-test <cmd>`: **works** with virtiofs or
sshfs shared mount. The end-to-end NCCL via-coord smoke test passed
on a 2-rank Pascal rig validating the elasticity + persistence work.
