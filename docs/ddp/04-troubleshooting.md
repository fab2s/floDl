# DDP Troubleshooting

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

### NCCL hangs on a cloud rig that formed fine

The most misattributed cluster failure, because the two planes have
different network shapes and only one of them is exercised before the
hang.

The **control plane is hub-and-spoke**: every worker talks to the
controller and to nobody else. So a cohort forms, `fdl status` looks
healthy, and admission passes with every worker reachable through a
single open port (or a single ssh forward).

The **NCCL data plane is a full mesh**. Every rank opens TCP to every
other rank on ephemeral ports, so an N-host run needs **all-to-all**
reachability among the workers, not just worker-to-controller. On a
default cloud security group, or any topology where workers reach the
controller but not each other, formation succeeds and then the first
collective blocks forever with no error: NCCL waits rather than
failing.

Three ways out:

- Open all-to-all TCP between workers in the security group / firewall.
- Use a `Cpu*` `ElCheMode`. CPU averaging routes through the controller,
  so it needs only the hub-and-spoke shape it already has, and it is the
  supported answer for tunneled workers and topologies where a mesh is
  not achievable.
- Confirm before blaming NCCL: if the same run completes under
  `ElCheConfig::cpu_async()` and hangs under an `Nccl*` mode, the
  difference is reachability, not the model or the data.

Tunneled workers (`--ssh`, `join.tunnel_only`) are CPU-mode only for
exactly this reason: a port forward carries the hub-and-spoke plane and
cannot carry a mesh.

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

### Address already in use

The bind error on a cluster port is self-diagnosing on Linux: it names
the process holding the port (pid, name) and how to clear it — and when
the holder is PID 1 inside a container's PID namespace, where `kill -9`
fails in a way that reads as a permissions problem, it says
`docker rm -f <container>` is the remedy. The diagnosis is advisory:
if `/proc` cannot answer, the plain error stands.

### Cluster progressive hangs

If `fdl @cluster` runs hang several epochs in, the cause is usually:

1. **Stale child processes** from a previous aborted run holding GPU
   memory or rendezvous ports. `fdl @cluster` cleans these up
   pre-spawn, but a kill -9 on the launcher bypasses cleanup.
2. **Shared-mount staleness** when the project mount is NFS or virtiofs
   and the controller and a worker see different file states. `fdl
   probe` flags mount-state divergence.

### Walk-in (`fdl join`) failures, by their messages

Every preparation failure is classed: transient exits 1 and re-dials
under `--persist`; permanent exits 2 and stops (the systemd recipe
pairs 2 with poweroff — see the [cluster
guide](02-cluster-guide.md#dial-in-membership-the-join-window)).

- **"no usable GPU"** with AMD cards installed — the two first-contact
  causes on a fresh box: no ROCm userspace runtime (the message names
  the install), or `/dev/kfd` / the DRM render node not openable by
  this user. The fix for the second is membership, then a re-login:
  `sudo usermod -aG render,video $USER`. In a container, pass the
  devices and groups (`--device /dev/kfd --device /dev/dri --group-add
  video --group-add render`). This is the predicted number-one AMD
  first-run failure; `rocm-capture.sh` at the repo root snapshots the
  whole stack for a bug report.
- **The tunnel works but the data mount says "permission denied"** — a
  forced `command=` covers subsystem requests, so a join key
  guardrailed with `/usr/sbin/nologin` refuses sftp while `ssh -N`
  sails through. Use the recipe's key A (`internal-sftp -R`), or mount
  during provisioning and declare a bare `data_path`.
- **The source fetch fails behind the same key** — rsync execs `rsync
  --server` on the far side, which the nologin key also refuses; the
  publish-flow key is recipe B (`command="rrsync -ro <served>"`), and
  behind it the worker's spec is `rsync://<host>:/tree`, never the
  absolute path (rrsync re-roots what it serves).
- **"the fetched source carries no run manifest and this box declares
  no artifact"** — transient by design: a publish is exactly what fixes
  it, including the window a publish opens on purpose while its gate
  builds. The box re-dials silently; if it does so forever, nobody has
  run `fdl publish` and no `--source-bin` names a local artifact.
- **"the source does not build"** — transient, the fix is a push away;
  the box picks it up on its next dial. Exception: the same failure
  with the vendor toolkit headers missing goes permanent and names the
  `apt` line, because re-dialing cannot install a package.
- **"the build succeeded but `bin:` is not there"** — permanent: the
  artifact path is relative to `cwd:`, and a workspace member's build
  lands in the WORKSPACE `target/`, not the member's.
- **"`cwd:` names no directory in the fetched source"** — permanent:
  it is a path inside the tree, not on the box.
- **"host X already joined this run"** — a stale worker from a
  previous launch still holds the name, or two boxes share a hostname;
  `--host` renames a walk-in.
- **"GPU vendor mismatch"** — a CUDA box and a ROCm box cannot share
  an NCCL/RCCL data plane; use a CPU ElChe mode (which mixes legally)
  or a one-vendor fleet.
- **"run identity mismatch"** — a publish landed between two boxes'
  fetches, so they hold two different runs. Nothing to fix: the stale
  side picks the new run up on its next dial.
- **"NCCL version skew"** — the cohort's libtorches load different
  NCCL major.minor versions, which refuse each other's handshake.
  Align the variants, or bridge with `fdl nccl build`.
- **"libtorch `…` ships no kernel for part of what this box offers"** —
  the resolved variant does not cover a card this box would train on
  (the first GPU op would die with `no kernel image is available`).
  `libtorch: auto` picks a covering variant when one exists;
  `--devices` scopes the offer to covered cards.
- **"model mismatch"** (at the join window) — the model this box
  builds differs from the cohort's (parameter names, shapes or
  dtypes), as probed by `fdl join` before the dial. Only this box's
  attempt is refused; fix it (a stale source tree, a wrong `bin:`) and
  `--persist` re-dials.
- **"model mismatch at formation"** — the named ranks constructed
  models whose parameter names, shapes or dtypes differ, so they
  cannot average each other. One box is running different model code:
  a stale source tree, a wrong `bin:`, or divergent arguments reaching
  the model factory. Re-publish (or fix the odd box) and relaunch.
  This is the backstop behind the join-window check above — it fires
  for boxes that joined without a signature (probe skipped or failed)
  and for stale fan-out binaries under `--no-prebuild`.

<!-- nav: generated by site/build_guide.py — do not edit below -->

---

Previous: [Internals and Expert APIs](03-internals.md) | Next: [Install and Global Flags](../cli/01-install.md)
