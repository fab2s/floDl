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

- **GPU inventory**: count, name, vendor, arch (sm or gfx), VRAM per
  device — free of any GPU runtime and of libtorch by construction (NVIDIA via
  nvidia-smi, AMD via the kernel's KFD topology), and the sweep's
  findings ride along: an AMD card present with no ROCm runtime is
  reported with the install command, not silently dropped.
- **libtorch variant**: active variant, version, toolkit version, arch
  coverage, source (precompiled vs source-built).
- **GPU/libtorch arch compatibility**: every visible GPU's arch is
  covered by the active libtorch's `archs:` metadata, both vendors
  (tokenized match — a bare `5` no longer matches the `5` inside `7.5`).
- **Host prerequisites**: the tools a native build needs (curl or wget,
  unzip, a C++ compiler) and the ACTIVE variant's vendor toolkit
  headers — the full set `flodl-sys/build.rs` demands (7 headers on
  ROCm, which has no metapackage), each with the install command.
- **NCCL availability**: host-level `libnccl.so` linkage, version. On
  workers with `docker: <svc>` declared in `fdl.cluster.yml`, the
  probe reports "via Docker image `<svc>`" instead of erroring on a
  missing host-level libnccl. Skipped on an AMD-only host (RCCL ships
  inside libtorch-rocm).
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

## `fdl publish`

Put a training run where a fleet can pull it. The controller side of
compiling on the node: it resolves a source spec into a served
directory, builds it once as a gate, and writes the run manifest workers
read.

```bash
# The operator's own crate, wherever it lives, with the run's arguments:
fdl publish file:///home/op/my-train --bin target/release/my-train \
            -- --model resnet --epochs 20

# A pinned checkout, fetched by the controller itself (a remote
# controller has no copy of your tree):
fdl publish git+https://github.com/me/train#v3 --bin target/release/train

# This repo's own vehicle: a workspace-excluded crate, so `--cwd`:
fdl publish file:///home/op/rdl --cwd ddp-bench \
            --build 'cargo build --release --features "$FDL_GPU_FEATURE" --bin ddp-bench' \
            --bin target/release/ddp-bench -- --model olmo-graph --epochs 1
```

The tree lands in `<served>/tree` (default `~/.flodl/run/tree`), which
is what a worker's `--source` points at, and the manifest sits at its
root so one fetch carries both.

Flags: `--bin` (required — the artifact relative to the project dir; a
workspace member's build lands in the WORKSPACE `target/`, so no rule
fdl invented would be right for everyone), `--cwd` (project dir inside
the tree), `--build` (a shell recipe, default `cargo build --release`;
it receives `LIBTORCH_PATH`, `FDL_GPU_FEATURE` and `LD_LIBRARY_PATH`,
with the system ROCm runtime — `$ROCM_PATH` / `$HIP_PATH` / `$HSA_PATH`,
else `/opt/rocm` — resolved ahead of libtorch's bundled copy), `--to
<dir>` (the served directory, default `~/.flodl/run`; the guardrail
key's `rrsync -ro` scopes to exactly this, so pick it deliberately),
`--identity <key>` (for an `rsync://` source the controller itself pulls
over ssh), `--no-build` (skip the gate — recorded in the manifest,
loudly), `--gate <variant>` (extra check-builds, below) and `--json`
(the report as JSON on stdout, notes on stderr — the machine twin of the
human report, same data). One deliberate asymmetry with `fdl join`:
publish owns the whole command so its flags are bare (`--cwd`), while
join prefixes its source flags (`--source-cwd`) because join also
carries data, tunnel and libtorch surfaces.

**A standing `publish:` block** in fdl.yml (or the active env overlay)
carries all of the above, so re-publishing a run is one bare `fdl
publish`. Flags win field by field, and a `--` tail replaces the block's
`args:` outright — even an empty tail, because the args belong to the
run and "explicitly none" must be sayable. `--no-build` has no block
field on purpose: a standing config that skips the gate would ship every
future typo to the fleet.

```yaml
publish:
  source: file:///home/op/rdl
  cwd: ddp-bench
  build: cargo build --release --features "$FDL_GPU_FEATURE" --bin ddp-bench
  bin: target/release/ddp-bench
  args: [--model, olmo-graph, --epochs, "1"]
```

**The manifest is `.fdl-run.yml`**, at the tree root:

| field | meaning |
|---|---|
| `cwd` | project directory inside the tree (default: its root); governs the build AND the run |
| `build` | build recipe, a shell line (default `cargo build --release`) |
| `bin` | built artifact, relative to `cwd` — what workers run |
| `args` | the binary's own arguments (everything after `--`) |
| `origin` | the source spec the controller resolved, for provenance |
| `rustc` | `rustc -V` on the controller — advisory; a worker reports a mismatch, never enforces it |
| `published_epoch` | unix seconds at publish, so a box can say how old its run is |
| `run` | this publish's identity nonce; it rides each worker's join hello, and the window refuses a cohort whose members hold different ids (two boxes that fetched across a publish boundary would train two different runs as one world) |
| `built` | `false` when `--no-build` skipped the gate; workers say so out loud |

Do not hand-edit it: the next publish overwrites it, and its *presence*
is what tells a worker the run is ready.

**Chaining runs on a standing fleet is then one command.** Publish
again and every box picks the new run up on its next dial, with nothing
to edit on any worker. That is the point of the manifest: a worker's own
config keeps only what is stable for that box (its credentials, its
libtorch policy, where to pull from), while `cwd` / `build` / `bin` /
`args` belong to the run and come from the controller. `args` is the
sharp case rather than a convenience: they must match the run, because
rank children re-enter the binary with them, so a fleet carrying its own
copy would train the next run with the previous one's hyperparameters.

**The build is validation, not an artifact.** One build gates the
publish; every worker still compiles its own, because a controller
producing binaries for N worker variants is the build matrix this design
deleted. It also needs no GPU libtorch — compiling without a GPU feature
against the cheap CPU variant catches user-code errors just as well — so
the cost of having it on by default is rustup plus `fdl libtorch
download --cpu`. What it buys is that a tree which cannot compile never
reaches the fleet, where N boxes would each discover it separately in
logs nobody is watching. It proves the tree for the *controller's*
variant only: a break that exists solely under `--features rocm` passes
a CUDA gate and lands on a worker. `--no-build` skips it and the
manifest records that nothing has compiled this tree.

`--gate <variant>` closes that per-vendor hole from the controller:
each one runs the same recipe as an extra check-build against a named
libtorch variant (`--gate precompiled/rocm70` on a CUDA controller, and
vice versa), under its own `CARGO_TARGET_DIR` so every variant's
incremental cache stays warm. No GPU is needed — linking is the proof —
but a flodl-linking crate does need that vendor's *dev headers* on the
controller (libtorch bundles runtime libraries, not headers); a gate on
a box without them fails loudly with the exact package line to install.
A failed check-build publishes nothing, exactly like the primary gate.

**The manifest's presence is the commit point.** `fdl publish` removes
it before it touches the tree and writes it only once the build has
passed, so a box that dials mid-publish, or after a publish whose build
failed, finds no manifest and waits for the next dial instead of
training something unvalidated. A failed gate publishes nothing, and the
fleet keeps running whatever it had.

The served directory is what a source key must be scoped to:
`command="rrsync -ro <served>"`. rrsync re-roots every requested path
under that directory, so a worker behind it points its `source.from` at
`rsync://<host>:/tree`, not the absolute path (which double-roots and
fails). `fdl publish` prints both spellings, each labelled with the key
it pairs with (see the
[guardrail recipe](../ddp/02-cluster-guide.md#dial-in-membership-the-join-window)).

Exit code: **0** when the run is published, **1** otherwise.

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
- **The integrated-GPU RAM share**, when `--gpu-ram-share` (or
  `join.gpu_ram_share:`) declares one: shipped to this box's ranks the
  same way `--data-path` is, where it overrides any cluster-scope
  default the controller declared and fills the training binary's
  config when that left the knob unset. APU boxes only; discrete GPUs
  ignore it.
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

  All three are optional when the tree came from [`fdl
  publish`](#fdl-publish): it carries a run manifest naming them, and
  that manifest is the authority, so a worker pointed at a published tree
  needs nothing but the pointer:

  ```bash
  # plain source key; behind a guardrailed rrsync key the spec is
  # `rsync://flodl-join@ctrl:/tree` instead (rrsync re-roots the path)
  fdl join --ssh flodl-join@ctrl --source rsync://flodl@ctrl:/home/op/.flodl/run/tree
  ```

  A tree with no manifest and no local artifact is a **transient**
  failure, not a permanent one: publishing is exactly what fixes it, and
  that includes the window a publish opens deliberately while its build
  runs.
- **The model signature** (default on, `--no-sig-probe` or
  `join.sig_probe: false` to skip). The resolved binary is re-run once,
  CPU-only, to print the signature of the model it builds (parameter
  names, shapes and dtypes); the signature rides the join hello and
  admission refuses a box whose model differs from the cohort's — at
  the door, where the refusal costs only this box's own dial and
  `--persist` re-dials once it is fixed. The probe's outcome is cached
  across re-dials, keyed on the binary's identity and the run's
  arguments, so an idle `--persist` box pays it once per actual change
  (a rebuild or a re-publish re-probes), not once per backoff tick. Without it the mismatch is
  still caught, but at formation, where it takes the whole cohort's
  attempt down. The probe is best-effort: a probe that fails or times
  out joins without a signature and says so. One probe outcome deserves
  attention beyond its warning: a binary that exits non-zero under the
  probe will usually fail the same way when rank children re-enter it
  with the same arguments after admission.

A source your provisioning already mounts needs no scheme at all: name
its path in `--data-path`. Nothing is checked and nothing is shipped
when neither field is set.

Exit codes:

| code | meaning | `--persist` |
|---|---|---|
| **0** | the agent's own: this host's ranks all finished cleanly | re-dials |
| **1** | transient failure — the controller unreachable or the agent exiting 1, a mount or fetch attempt failed, **the source did not compile** | re-dials |
| **2** | permanent failure — no GPU, a spec that does not parse, a missing binary or toolchain, unwritable stage | **stops** |

A compile error being transient is deliberate rather than lenient. The
source is remote, so the fix is a push away, and the systemd recipe below
pairs code 2 with `poweroff`: a box that stopped permanently over a typo
would take the fleet with it. One exception: a compile failure on a box
whose vendor toolkit headers are demonstrably missing is permanent —
re-dialing cannot install a package, and the error names the `apt` line.

One-shot mode (no `--persist`) passes the **agent's own exit code**
through verbatim, so a training binary that exits 2 for its own reasons
is indistinguishable from fdl's permanent class. The systemd recipe
below is therefore written for `--persist`, where agent exits re-dial
inside fdl and only classed preparation failures ever reach systemd —
pair the recipe with `persist: true` (or treat 2 as reserved in your
training binary).

fdl never powers a box off itself; 2 is how the thing that owns the
instance hears about it:

```ini
# /etc/systemd/system/flodl-join.service  (fdl join --persist ...)
Restart=always
RestartPreventExitStatus=2   # stop hot-looping a misprovisioned box
FailureAction=poweroff       # ... and self-deprovision it
```

Full protocol walkthrough, trust model, and the join-sshd guardrail
recipe: [DDP reference](../ddp/02-cluster-guide.md#dial-in-membership-the-join-window).

## `fdl join-config`

The once-per-farm wizard: everything the guardrail recipe asks an
operator to assemble by hand, produced in one pass on the controller.

```bash
fdl join-config b300                       # interactive: prompts have defaults
fdl join-config b300 --controller flodl-join@ctrl.example.com:2222 \
                     --install-key --cloud-init --yes    # scripted
fdl @b300 join-config --regen              # new farm instantiation: rotate credentials
```

A **farm is an env overlay**: the wizard scaffolds `fdl.<label>.yml`
(discovery window, `start: manual`, the admission token stamped) and
`fdl @<label> <cmd>` targets it afterwards with the machinery that
already exists: deep-merge onto the base fdl.yml, `inherit-from:` for
[sharing a base between farms](05-manifest.md#inherit-from), `fdl
config show` provenance. On an existing overlay the wizard only ever
touches the `token:` line (byte-preserving), and a user-authored
overlay without one is never edited; the snippet is printed instead.

One invocation produces, under `./.fdl/<label>/` (which gitignores
itself, since it holds keys):

- **an ed25519 join key**, born per farm so it cannot be shared across
  clusters by construction (a config referencing an identity outside
  the farm dir draws a warning), plus a fresh 32-hex token;
- **the composed `authorized_keys` line** for the chosen door
  (`--door b` rrsync source pull, the publish-then-join default; `a`
  read-only sftp data mount; `nologin` tunnel-only) and the sshd
  `Match` hardening block, saved to `install-notes.md`;
- **the paste-ready worker `fdl.yml`** speaking that door's dialect,
  `libtorch: auto`, `persist: true`, the token inside;
- **a publish recipe derived from the training crate's own manifest**:
  a path dep on flodl walks the `source:` up to the dep root with
  `cwd:` pointing back down (what stops the dep dangling outside the
  fetched tree); a registry dep ships the crate dir alone;
  `--features "$FDL_GPU_FEATURE"` appears only when the manifest
  declares `cuda`/`rocm`; a workspace above the crate earns an explicit
  `bin:` caveat instead of a silent wrong guess;
- **a freshness report**: whether `Cargo.lock` still describes the
  source about to ship;
- with `--cloud-init`, **a user-data file** embedding the worker yml,
  the private key and the systemd recipe (`Restart=always`,
  `RestartPreventExitStatus=2`, `FailureAction=poweroff`), so an
  instance boots straight into `fdl join`. A SECRET artifact (key and
  token inside), written 0600 and never printed.

The wizard also **offers to install** its `authorized_keys` line into
the invoking user's own `~/.ssh/authorized_keys`, because the composed
guardrail line is the artifact most likely to be mangled by hand.
Consent is explicit only: the prompt, or `--install-key` (`--yes`
deliberately does not count; it accepts ordinary defaults, and a
security-relevant mutation is not one). Only the wizard's own line is
ever touched (identity is the public key material; foreign lines are
preserved byte for byte), permissions are reported with the exact
`chmod` and fixed on confirm, the rewrite is atomic, and `/etc/ssh` is
never edited: the dedicated-user hardening stays in the notes. After
installing, the wizard checks something is listening where workers will
dial and says so if not (on macOS: Remote Login).

The wizard guides the setup, not just the artifacts. Before writing
anything it reports what THIS box is still missing for the chosen door:
an ssh daemon, something listening on the door port, `rrsync` for door
`b` or an sftp server for door `a`, the served directory, plus the two
traps that produce no useful error on their own (Debian hands the ssh
listener to `ssh.socket`, and while it holds it the `Port` directive is
ignored outright; SELinux refuses a non-standard ssh port with a message
that never mentions SELinux). Each gap is reported with the fix in this
platform's own spelling — apt, dnf or brew; `ssh.service` or
`sshd.service`; ufw or firewalld; `semanage` where it applies.

It then writes a ready-to-install `sshd_config.d` drop-in into the farm
directory and prints the setup as numbered steps whose commands read
those files rather than repeating their contents, so what you paste is a
copy of something you can review first. The last of the controller-side
steps is a door self-test specific to the door: every door must refuse a
shell and permit the forward, and only `b` must additionally list the
served tree, only `a` must open sftp.

The drop-in scopes its guardrail with `Match LocalPort`, not `Match
User`. Binding it to the exposed port confines every key that arrives
there, including ones added later, while leaving port 22 and your own
logins untouched — which also means the wizard can install its line into
your own `~/.ssh/authorized_keys` unaided, where a dedicated no-shell
account would have needed root. `ForceCommand` appears only for the
tunnel-only door: doors `a` and `b` carry their command in the key line,
and a daemon-level forced command would override it, leaving the tunnel
working while the mount or the source pull failed.

`--authorized-keys <path>` names a different door for the setups the
default cannot reach at all: an sshd in a container, whose key file is
a bind mount (writable only from the host, while the in-container run
sees a read-only filesystem), or a host configured with
`AuthorizedKeysFile /etc/ssh/authorized_keys.d/%u`. Every other rule
holds — the same explicit consent, only the wizard's own line, the same
atomic rewrite — except that the named file's parent directory is left
exactly as found, since it belongs to a layout the operator already
owns rather than to a `~/.ssh` the wizard may have to create. A path
under `/etc/ssh` is still refused by name.

The scaffolded overlay also carries a `commands:` entry, because a join
window only opens for a command running in launcher mode and `cluster:
true` is what puts it there. Without one, `fdl @<label> <cmd>` resolves
the base command and trains locally: no window, no walk-ins, and no
complaint, since training on that box is a legitimate thing to do. The
wizard names the entry after the training crate when it can read one,
and comments a placeholder when it would be guessing. Running a
non-cluster command under a farm overlay warns for the same reason.

Credentials are reused on re-runs, and so is the farm's shape: a re-run
without `--door` or `--controller` recovers both from the farm's own
worker yml rather than falling back to flag defaults, so reprinting the
`authorized_keys` line cannot silently re-render the farm around it.
Naming either flag is how you intend a change. `--regen` (or the
prompt) rotates key and token together for a new farm instantiation,
after which workers holding the old ones stop being admitted. `--json`
emits the machine twin; secrets appear as file paths, never payloads.

Two read-only companions serve scripts and interfaces built on the
wizard. `--list` enumerates the project's farms — the union of
`fdl.<label>.*` overlays and `.fdl/<label>/` farm dirs — with door,
controller and credential state; env overlays that are not farms are
reported apart rather than dressed as broken ones. `--dry-run` runs
the full pass with every write withheld and reports what an apply
would `create`, `update` or leave alone, per file; credentials an
apply would mint appear as placeholders, never as values the apply
will not reproduce. Neither ever prompts — a dry run reads the consent
flags exactly as given — and both combine with `--json`.

Exit code: **0** with the report, **1** on any refusal (contradictory
flags, a decision needed without a tty, an unfixable permission).

Platform notes: the token comes from OS entropy (cross-platform; the
wizard refuses rather than degrade if the system call fails) and the
key from `ssh-keygen`, which Linux and macOS ship natively (on macOS,
Remote Login is the sshd) and Windows carries with its OpenSSH
feature — though for cluster work on Windows the supported path stays
[WSL2](../windows-wsl.md), where the permission checks also actually
mean something. Running the wizard inside a container works for
*generation* — the farm lands in the mounted project — but the
`authorized_keys` install step must run where the workers' sshd
actually lives, which is not the container's `~/.ssh`.

## `fdl ui`

The local operations page: one loopback web page for the project's
farms, hardware probe, cluster run status and resolved config — the
browser counterpart of the walk-in CLI surface — plus the first
actions:

- **the join-config wizard as a form**: fill label/door/controller,
  Preview runs `--dry-run` and shows exactly what an apply would
  create or update, and Apply unlocks only after a preview of exactly
  that form state — the dry run IS the confirmation step. The applied
  report carries the authorized_keys line and the worker yml with copy
  buttons (cloud-init stays a file path: it embeds the private key and
  a secret artifact is never served over HTTP);
- **publish with the gate build streamed live**: the child's output
  arrives line by line with the exit code at the end. One job runs at
  a time (two concurrent publishes would race the manifest commit
  point), a closed tab never kills the build (a publish must reach or
  cleanly fail before its commit point), and "Follow last job" replays
  the stream from its first line and follows live.

```bash
fdl ui                # serve http://127.0.0.1:1338/
fdl ui --port 8040    # any free loopback port
```

The **run tab** is one slot whose backing follows the run's lifecycle:
before anything answers on the dashboard port it shows the admission
view (`fdl status`, re-probing quietly); the moment the run's live
dashboard comes up, the same slot becomes that dashboard, reverse-
proxied through the ui's own port — so a headless controller needs
exactly **one** `ssh -L` forward for the whole experience, ops page
and live dashboard together. The proxy forwards the dashboard's own
routes verbatim to a loopback port only (the host is not configurable,
so it cannot be aimed off-box). The **history tab** is training
history alone: the dashboards runs persisted on disk (`dashboard*.html`
/ `timeline*.html` under the project, rustdoc lookalikes and templates
excluded), grouped one row per run directory since artifacts sharing a
directory are one run, filtered by space-separated terms, newest 15 by
default, and served into the same kind of slot — which the browser list
yields to while you are viewing one. Archive serving is double-bounded
(the path must resolve inside the project root AND look like a run
artifact).

The **launch tab** runs the project's own configured commands (the
`fdl.yml` `commands:` tree, under a farm overlay when one is selected —
which is where a farm's `cluster: true` run command lives). A command
that resolves a schema — cached `--fdl-schema` output in its own
directory, or an inline `schema:` block — gets a **form generated from
it**: checkboxes for bools, selects for choices, typed hints and
defaults on everything else, and only the fields the operator actually
sets become argv (defaults stay the binary's own). Without one the form
degrades to a freeform args field, and the hint says which kind of
absence it is: a `run:` command is a shell line and never grows a form,
while a path command whose cache is stale or missing grows one after
`fdl <cmd> --refresh-schema` (a page load never triggers that compile
itself). **fdl's own
options ride alongside** — verbosity (`-q` / `-v` / `-vv` / `-vvv`),
`--gpus`, `--no-append`, `--no-prebuild` — grouped with the env select
because they are the same fdl-level scope and, like `--env`, precede
the command; they are no command's schema, so the form has to carry
them or they are unreachable from the page. A **help** button runs the
command with `--help`, which is the only way to see a schema-less
command's surface from the browser; it streams like a launch but never
enters the run ledger, since asking what something takes is not
running it. Launching streams
the run's output live through the same job machinery as publish (one
job at a time; a closed tab never kills the run; "Follow last job"
reconnects), a `--monitor <port>` in the args automatically points the
run tab's dashboard slot at the run, and **each completed launch
appends one line to the run ledger** (`.fdl/ui/runs.jsonl`: timestamp,
duration, farm, exact argv, exit code), which is listed **beside the
form** in a scrolling column: a command history belongs next to the
thing that runs commands. Clicking an entry *proposes* it again — the
recorded argv replayed verbatim behind an explicit confirm, never
launched by the click itself. Deliberately absent: a stop button. A killed cluster
run leaves ports held and remote ranks spinning (the rig-hygiene
protocol exists for a reason), and a button that pretends otherwise
would be a lie; stopping stays a deliberate act.

**The page drives the CLI, it never reimplements it.** Every panel that
runs something spawns `fdl` itself with `--json` and renders argv, exit
code and output verbatim — the exact command line sits above each
result with a copy button, so anything the page does is reproducible in
a terminal, and the page structurally cannot drift from the CLI. The
farm list is the one pure local read, and it calls the same function
`fdl join-config --list --json` prints from.

Security: binds `127.0.0.1` only, and on top of the bind every request
must carry the loopback `Host` it was served on (which stops
DNS-rebinding pages that resolve an attacker domain to 127.0.0.1), and
every API route requires the per-session token minted at startup and
injected into the served page (which stops blind cross-site requests).
Reaching the page from another box is an ssh forward — `ssh -L
1338:127.0.0.1:1338 <controller>` — the same trust story as the
cluster itself. There is no auth beyond that on purpose: the page has
exactly the authority of the user's own shell on that box, so the
boundary worth defending is the box, not the page.

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
