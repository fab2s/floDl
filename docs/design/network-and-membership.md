# Network & Membership - Dial-In Clusters, Tunnels, and One Address

**Status:** design, decided. Companion to
[cloud-ddp.md](cloud-ddp.md) (scaling vision) and
[../distributed/architecture.md](../distributed/architecture.md)
(current process/wire topology). This document specifies the
communication strategy for the next cluster iteration: how workers
find and join a training run, how the transport is secured, and how
the whole thing stays observable.

## Where the wire already is

Every cross-host TCP connection in a cluster run already dials
*toward* the controller host - and, with the single-port mux (the
first piece of this design, now landed), onto **one port**:

| Connection | Direction | Port |
|---|---|---|
| NCCL rendezvous | rank → controller | `port` (channel magic: rendezvous) |
| CPU-reduce data channel | relay → controller | `port` (channel magic: data) |
| Coordinator control channel | relay → controller | `port` (channel magic: control) |
| Rank loopback (data / control) | rank → local relay | `port+4` / `port+5` (127.0.0.1 only) |
| Dashboard (HTTP) | browser → controller | its own port |

The controller never dials out at the TCP level. What prevents the
"one address, any number of workers" deployment is not the wire
direction - it is three things this design removes:

1. **Several ports.** Before the mux, tunneling a run needed three or
   four `ssh -L` forwards, and config carried host+port pairs per
   concern. Solved: one forward covers all training traffic.
2. **Push membership.** The controller must know every worker up
   front (`cluster.yml`: ssh access, project path, arch, libtorch
   variant) and reaches out to spawn them. A worker the controller
   cannot SSH into - a NAT'd cloud instance, a spot VM - can never
   exist in this model.
3. **Cleartext transport with a pre-shared secret.** The session salt
   must be distributed to every worker out-of-band, and params cross
   the network in cleartext; fine on a controlled rig, undocumented
   and silent anywhere else.

## Goals

- **One address.** A valid worker deployment carries the controller
  address and a credential. Nothing else. "As many ranks as wished"
  attach to that one address.
- **Dial-in membership.** Workers join; the controller admits. How a
  worker process got started (SSH fan-out, cloud startup script, a
  human shell) is not the controller's concern.
- **Tunnel-first security.** SSH is the blessed transport for
  anything beyond a private network, with the deployment credential
  being an SSH key. TLS is a later opt-in with the same shape.
- **Observable membership.** One state source, visible from a log
  line, an HTTP endpoint, and the CLI.

## Single-port mux

All controller-side listeners fold onto **one accepting port**. Every
cross-host dial opens with a 4-byte **channel-select magic**
(rendezvous / data / control - the two relay legs were previously told
apart only by port number, so the connection had to become
self-describing); an accept dispatcher peeks the magic without
consuming it and hands the connection to the owning subsystem, which
validates it as its first read. The magic is deliberately
unauthenticated - routing is not security-sensitive, a spoofed magic
just lands at a subsystem whose frame authentication rejects it,
exactly as a wrong port number did before. The reserved `port+1..+3`
offsets disappear from the public surface; rank↔relay loopback offsets
remain (they never leave the host), and the dashboard keeps its own
HTTP port (browser-facing, tunneled separately when wanted).

Config becomes literally `controller: host:port`, and one
`ssh -L port:localhost:port controller-host` covers all training
traffic. This piece is independently shippable and useful before any
membership change.

## Membership: the join protocol

### Join flow

A worker dials the controller port and sends a **join hello**:
protocol version, host name, GPU inventory (count, arch), libtorch
variant, dataset signature, and the deployment credential when the
trust mode requires one (see below). What admission validates today,
immediately rather than at window close: protocol version, host-name
uniqueness, device-list sanity, GPU-vendor coherence when the data
plane is NCCL/RCCL (a mixed NCCL+RCCL cohort passes every structural
check and hangs at formation, so the libtorch label's vendor is checked
at the door; CPU averaging modes mix legally), **run identity** (the
`.fdl-run.yml` nonce `fdl publish` stamps — a cohort straddling a
publish boundary holds two different runs, and boxes carrying no id
gate nothing), **NCCL/RCCL major.minor** (read on each worker from the
library its binary actually loads; skew refuses the handshake at
formation, and a walk-in fleet has no roster for a probe to sweep, so
the window is the only place this check can live), and the dataset
signature — with one honest caveat on the last: `fdl join` does not yet
stamp a real dataset signature into the hello (workers send zeros), so
that check bites only for binaries that set it themselves. Every one of
these is first-member-seeded consistency among joiners, never an
authority the controller asserts. As-designed per-host prechecks beyond
these were deliberately NOT built: the controller cannot reach back
through a NAT'd tunnel, and the worker has the filesystem — that
precheck dissolved into `fdl join`'s prepare phase (which also gates
libtorch↔GPU arch coverage, the other half of the promised coherence,
where the `.arch` metadata lives). The next admission fact is the
cohort code signature (param names+shapes at formation).

The join reply carries the **session salt**, the assigned global rank
ids (one per local GPU, assigned in admission order), and the cluster
view the worker needs to proceed. Handing the salt out at join - over
a channel whose trust is established by reachability or credential -
removes the pre-shared-secret distribution problem entirely: every
subsequent channel (rendezvous, control, data) authenticates with the
salt received at join, which binds those connections to the admitted
identity.

### World formation: quorum knobs

Until elastic scale-up lands, the world is formed once, at start:

- **`min_rank_start`** - the quorum, counted in ranks (GPUs). The run
  cannot start below it.
- **`join_timeout`** (default 300 s) - the join window. Reaching
  quorum early does NOT close the window: late workers within it are
  still admitted. More capacity is never refused while the door is
  open.
- **`target_ranks`** (optional) - closes the window early the moment
  it is reached. When you launched exactly N workers, the run starts
  the moment all N are in instead of waiting out the window. Unset
  means the full window runs.
- **`max_join_timeout`** (default 600 s) - the hard cap. If quorum is
  still unmet when it expires, the run fails loudly. Between
  `join_timeout` and `max_join_timeout` the controller keeps waiting
  for quorum (but the settled window semantics above no longer
  apply: the first moment quorum is met in this range, the world
  forms).

All four are tunable; the timeouts scale with `FLODL_NET_TIMEOUT_SCALE`
like the rest of the deadline set. Shipped alongside them (documented
in the cluster guide, listed here so this section is not read as the
whole knob set): `discovery` (a roster-free window that requires an
explicit `min_rank_start`), `open_admission`, `token` (pre-shared
session salt), `tunnel_only`, and `start: auto|manual|hybrid` (the
staging hold). When the window closes, world_size
freezes: seed-derived sharding, the cadence scheduler, and the
window≤epoch invariant all see a static world. Elastic *death*
(dead-rank detection, membership shrink, NCCL rebuild) continues to
work exactly as today. Elastic *join* after training start is
explicitly out of scope for this iteration - it implies mid-flight
NCCL rebuild and resharding, and it gets its own design when it
comes.

### Push becomes sugar

`fdl @cluster <cmd>` fan-out does not go away - it becomes one
convenient way to *start* workers on a managed rig. The SSH fan-out
starts worker processes; those processes dial in and join like any
cloud worker would. Membership is decided by the join protocol in
both cases. `cluster.yml` keeps working as the description of a
managed rig; a pull-only deployment needs none of it.

## Trust model

The load-bearing insight: **on a tunneled deployment, reachability is
the authentication.** If the controller binds loopback only, the only
path to the port is through sshd on the controller host; a connection
arriving at all proves possession of an authorized SSH key - a
strictly stronger guarantee than any shared-secret signature (it is
asymmetric crypto plus an encrypted channel, versus a symmetric salt
that must be shipped around). Accept-any is sound there, and the salt
is handed out at join.

That inference is *only* valid under the loopback bind, so admission
is gated on bind scope:

| Bind scope | Admission | Rationale |
|---|---|---|
| `127.0.0.1` (tunnel mode) | **open** - any inbound join is accepted, salt handed out in the reply | reachability proves SSH-key possession |
| non-loopback (rig mode) | **pre-shared salt** as today (fdl fan-out delivers it), or explicit `open_admission: true` with a loud warning | reaching a LAN/WAN port proves nothing; silent open admission would let a network neighbor *participate in* (poison) training, which is strictly worse than observing cleartext |

Deployment hardening that falls out for free: the worker's SSH key
can be a restricted `authorized_keys` entry on the controller host,
granting exactly the tunnel and nothing else - `restrict`,
`port-forwarding`, `permitopen="127.0.0.1:<port>"` and a forced
`command=`, that last one load-bearing because `no-pty` alone still
leaves `ssh host <cmd>` wide open (full recipe, including what the
forced command costs a walk-in's data mount, in
[the cluster guide](../ddp/02-cluster-guide.md)). That makes the credential safe to bake into cloud
images: a leaked key lets someone join a training run if they can
also reach the controller's sshd, and nothing more.

### Cleartext guard

Whenever a non-loopback, non-private (not RFC1918 / link-local) peer
appears on an unencrypted channel, the controller and the worker both
emit a **loud warning, not an error**: the controlled-network
assumption is the documented contract, tunnels are the supported way
out of it, and the framework does not silently pretend otherwise.
(Explicit selectors error; conventions warn.)

### TLS, later

The TLS variant keeps the same deployment shape with one correction:
a pinned self-signed server certificate authenticates the
*controller to the worker* (plus confidentiality), but server-auth
TLS does not authenticate the worker back - anyone with the address
completes the handshake. The bundle is therefore
`{controller address, pinned cert, join token}`, with the token
checked inside the TLS channel at join (structurally equivalent:
mTLS with a client cert in the bundle). Everything else - salt in
the join reply, open admission gated on transport trust - carries
over unchanged. TLS is not scheduled; SSH tunnels cover the target
deployments with zero new dependencies.

## Observability

One small state struct on the controller, three renderings:

- **Phase**: `waiting(joined/quorum/target, window countdown, cap
  countdown)` → `staging` (manual/hybrid start: quorum met, roster
  held for the operator) → `forming` → `training` → `done/failed`.
- **Members**: per host - name, ranks, GPU inventory, join
  timestamp, precheck result.

Renderings, cheapest first:

1. **Log lines** on the controller's stderr for every transition and
   every join/reject - visible from any SSH session, greppable.
2. **`state.json`** served over plain HTTP on the mux port itself
   (an HTTP GET's `"GET "` bytes route like a channel magic). Landed
   here rather than on the dashboard server as first sketched: the
   dashboard binds lazily on the first rank-emitted register frame,
   on a port only rank user-code knows - pre-formation there are no
   ranks, so the `waiting`/`forming` phases would have been
   unobservable there. The mux is bound before the window opens.
3. **`fdl status`** - fetches that endpoint (through the tunnel when
   one is up) and pretty-prints it.

The log file alone would satisfy the requirement; the JSON endpoint
costs a few lines more and gives the CLI (and anything with curl) the
same truth for free.

## Host-tier scheduling (recursive cadence)

Deferred to its own arc, with the interface fixed here so the join
protocol and the fold tier don't need rework later.

The per-host relay currently folds the data plane (one summed frame
per host per round - the controller already accounts
per-connection). The remaining half is scheduling: the controller
treats each host as **one virtual device** with aggregate throughput,
hands it a per-window *host budget*, and the relay splits that budget
across its local GPUs using the same cadence logic one tier down -
exactly the recursive-scheduler shape sketched in
[cloud-ddp.md](cloud-ddp.md). Two boundaries are non-negotiable:

- **NCCL ranks stay structural.** The virtual-device abstraction
  applies to the CPU/WAN tier only; an NCCL world is flat by
  construction.
- **The determinism rule holds.** Coverage (which samples, which
  epoch) stays seed-derived and rank-granular; only *allocation*
  (how many batches per window) becomes hierarchical. Soft decisions
  may be distributed; deterministic coverage may not.

Related and equally deferred: on CPU-backend runs, a multi-GPU host
may pre-sum its local contributions over NCCL before the fold
(GPU-side reduce, one D2H instead of K). Same associative monoid, so
it composes with the relay fold and the controller unchanged. Parked
until profiles show D2H or relay-CPU pressure - the fold already
collapsed the wire bytes.

## Control-plane hardening (folded in)

- **Unilateral mid-collective exit rescue.** A rank whose `Exiting`
  arrives while a reduce is unsettled has left a collective hanging;
  the coordinator broadcasts `DeclareDead` for it (when not already
  shutting down) so survivors' watchdogs abort out of the stranded
  collective instead of spinning. Complements the settle-gated
  Shutdown that already covers the coordinator-driven side.
- **ssh option ordering.** `ssh.options` from `cluster.yml` must be
  placed so user options cannot override the launcher's
  session-critical flags; documented as part of the tunnel work
  where fdl starts managing SSH invocations that carry forwards.

## Sequencing

1. **Single-port mux** - small, self-contained, immediately makes
   tunneling one-liner-cheap. No membership change. **Landed.**
2. **Cleartext guard + fdl tunnel sugar** - loud warning whenever a
   cleartext channel touches a peer outside private address space, and
   `tunnel: true` per host. **Landed.** One mechanism note: for
   fan-out-managed workers the forward is a *remote* forward (`-R`) on
   the host's relay SSH session - the launcher→worker credential that
   fan-out already requires - not a worker-side `ssh -L` (which would
   need worker→controller SSH credentials on every host). The `-L`
   form stays the shape for self-deployed workers in the join-protocol
   world. Tunnel mode requires a CPU ElChe mode (NCCL's peer-to-peer
   data plane cannot ride a controller tunnel); all-remotes-tunneled
   flips the controller bind to loopback-only, the bind scope the
   trust model keys on.
3. **Join protocol + quorum knobs + observability** - the membership
   flip. Push fan-out reworked to start workers that dial in.
   **Landed.** Implementation notes, where the landed shape refined
   the sketch above:
   - **Ranks are assigned in admission order** - contiguous by
     construction, no holes, no compaction; `cluster.yml` rank lists
     demoted to a capacity declaration (feeds the default quorum /
     target). Rank ids are per-run dynamic.
   - **One agent per host** (`Role::Agent`, bootstrapped by
     `FLODL_INTERNAL_AGENT_JSON`) replaces per-rank SSH spawn; the
     minimal self-deploy spec is `{host, controller_host,
     controller_port}` - the worker resolves its own GPUs. As shipped
     the spec also carries what only that box knows: its libtorch
     variant label (the vendor-coherence fact), the session token, an
     explicit device scope, and its resolved `data_path`. The join
     connection stays open as the host control link (`RankExited` up,
     `Abort` down, EOF = host death).
   - **Launcher-local hosts join in-process** (a thread dialing
     loopback), so their children stay direct launcher children - no
     grandchild leak on kill-all, and the local test topology is
     unchanged.
   - **Controller infrastructure starts after formation**: the coord
     config is built by a factory keyed on the world size that
     actually formed (`CoordSpec`), not patched afterwards.
   - **Trust rides the join-frame HMAC key**: pre-shared salt keys
     the hello in rig mode (wrong key = dropped without reply); open
     admission (loopback bind, or the loudly-warned knob) accepts
     zero-keyed hellos and hands the salt out in the accept reply.
   - **`state.json` moved off the dashboard server and onto the mux
     itself**: an HTTP GET's leading `"GET "` bytes route like a
     channel magic to a tiny read-only responder. The dashboard could
     not host it - it binds lazily on the first rank-emitted register
     frame, on a port only rank user-code knows, so the
     `waiting`/`forming` phases would have been unobservable. On the
     mux the endpoint is up before the window opens, needs zero new
     config, and its reachability follows bind scope exactly like
     admission. `fdl status` fetches it (or curl does).
   - **User-binary contract**: a binary that gates before
     `Trainer::run` must call
     `flodl::distributed::launcher::exit_if_worker_role()` first,
     or its worker agents fall into the gating on the remote host and
     the window idles to its hard cap (found on the rig).
4. **Host-tier scheduling** - its own arc, interface fixed above.

Out of scope for this iteration: mid-training join, TLS transport,
channel encryption beyond the tunnel, NCCL pre-sum (parked on
profiling evidence).
