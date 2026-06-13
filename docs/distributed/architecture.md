# Distributed architecture

This document maps the **entire distributed pattern** -- the data, the
communication, and the logic that move parameters between heterogeneous GPUs
and hosts during DDP training.

The orchestration is not one flow but **three orthogonal flows over a shared
role topology**. A single mega-chart would be unreadable, so each flow gets its
own view, in the diagram type that fits it:

| View | What it answers | Diagram |
|---|---|---|
| [1. Role topology](#1-role-topology) | Who runs where? Process vs thread boundaries. | flowchart |
| [2. Training lifecycle](#2-training-lifecycle) | What happens end to end, in order? | sequence |
| [3. Coordinator state machine](#3-coordinator-state-machine) | How does CPU averaging stay non-blocking? | state |
| [4. Reduce backends](#4-reduce-backends) | NCCL inline vs CPU 3-phase -- the key divergence. | flowchart |
| [5. ElChe scheduling](#5-elche-scheduling-data-flow) | How does work get allocated per rank? | flowchart |
| [6. Message catalog](#6-message-catalog) | Which message carries what, between whom? | tables |

> Every diagram cites its source file(s). When the code moves, re-sync the
> diagram from the cited enum/struct -- the variant names below are pulled
> verbatim so drift is greppable.

The user-facing strategy is a single **`ElCheMode`** (`nccl_sync`,
`nccl_cadence`, `cpu_sync`, `cpu_cadence`, `cpu_async`) on
[`ElCheConfig`]. Internally each mode decomposes into the two axes that
parameterize everything below:

- **`ApplyPolicy`** -- the *pacing* clock: `Sync` (K=1), `Cadence` (K=N via
  ElChe), `Async` (Cadence + bounded lookahead). `Sync`/`Cadence` are
  *barrier-paced* (a rank is held at its window until the reduce resets it);
  `Async` opts out.
- **`AverageBackend`** -- the *transport*: `Nccl` (in-place GPU AllReduce) or
  `Cpu` (snapshot round-trip through the coordinator). **Orthogonal to
  pacing** -- five of the six combinations are valid modes (`Async + Nccl`
  was dropped and hard-errors at `.run()`), which is what makes NCCL-vs-CPU
  A/B testing possible.

> Source: `flodl/src/distributed/config.rs` (`ElCheMode`, `ElCheMode::split`),
> `flodl/src/distributed/ddp_run/mod.rs` (`ApplyPolicy`, `AverageBackend`).

[`ElCheConfig`]: ../../flodl/src/distributed/config.rs

---

## 1. Role topology

A run is a tree of processes. The **launcher** never touches CUDA (it exits
after fan-out); each **rank** is its own process owning one GPU; one **relay**
per remote host multiplexes that host's ranks onto a single connection to the
controller. The **coordinator** is the scheduler the ranks talk to.

```mermaid
flowchart TB
    subgraph launchbox["Launcher process (no CUDA)"]
        L["Role::Launcher<br/>reads FLODL_FULL_CLUSTER_JSON<br/>fan-out + exit"]
    end

    subgraph coordhost["Controller host"]
        C["cluster_coordinator<br/>(scheduler + averaging)<br/>control port +3 / data port +2"]
    end

    subgraph host0["Host 0 (local)"]
        W0["Role::Rank GPU0<br/>GpuWorker"]
        W1["Role::Rank GPU1<br/>GpuWorker"]
    end

    subgraph host1["Host 1 (remote)"]
        R1["Role::Relay<br/>mux, no CUDA"]
        W2["Role::Rank GPU0<br/>GpuWorker"]
        W3["Role::Rank GPU1<br/>GpuWorker"]
    end

    L -.->|SSH spawn| W0
    L -.->|SSH spawn| W1
    L -.->|SSH spawn| R1
    L -.->|SSH spawn| W2
    L -.->|SSH spawn| W3

    W0 <-->|in-process channels| C
    W1 <-->|in-process channels| C
    W2 <-->|loopback| R1
    W3 <-->|loopback| R1
    R1 <-->|one muxed TCP conn<br/>MuxRecord frames| C
```

Local ranks (same host as the controller) talk over in-process channels;
remote ranks talk loopback to their host's relay, which carries every local
rank's frames over one connection as `MuxRecord` blobs tagged by rank.

> Source: `launcher/mod.rs` (`Role`, `dispatch()`), `relay/agent.rs`
> (`ChannelKind`), `relay/mux.rs` (`MuxRecord`, `RelayControlMsg`).

---

## 2. Training lifecycle

End to end for one cluster run: bootstrap rendezvous, then the repeating
**dispatch -> train -> reduce -> average -> re-arm** window. Shown for the
`Cadence` pacing; the `Update.next_plan` atomic-dispatch fold (CPU path)
collapses the post-reduce control round-trip.

```mermaid
sequenceDiagram
    participant L as Launcher
    participant C as Coordinator
    participant W as GpuWorker (rank)

    Note over L,W: Bootstrap (control channel, MsgKind::Rendezvous)
    L->>W: spawn (cluster envelope in env)
    W->>C: Hello { dataset_sig, global_rank, host_name }
    C->>W: Role(Generate | Wait)
    alt rank is Generate
        W->>C: Uid { uid_bytes }
    else rank is Wait
        C->>W: Uid { uid_bytes }
    end

    Note over C,W: Per-window training loop
    C->>W: StartEpoch(EpochPlan { offset, size })
    loop batch_counts[rank] batches
        W->>W: train_step (fwd + bwd + opt)
        W->>C: Batch { rank, batch_ms, step_count, batch_loss, ... }
    end

    Note over C,W: Reduce (backend-dependent -- see view 4)
    alt AverageBackend::Cpu
        C->>W: RequestParams
        W->>C: SnapshotReady { rank }
        W-->>C: ParamSnapshot (data channel RoundFrame)
        C->>C: average_params (background thread)
        C->>W: Update { version, next_plan: Some(plan) }
    else AverageBackend::Nccl
        C->>W: SyncNow
        W->>W: in-place AllReduce on comm_stream
        W->>C: SyncAck { rank, step_count, divergence, ... }
        C->>W: StartEpoch(next plan)
    end

    Note over C,W: Epoch boundary
    W->>C: MetricsMsg { epoch, avg_loss, share_complete_ms, ... }
    C->>W: SetGlobalStep(n)
    W->>C: Exiting { rank }
```

The `share_complete_ms` in `MetricsMsg` (not `epoch_ms`) is the honest
balancer denominator; ElChe's per-batch signal is the coordinator-measured
**delivered** cost, not `batch_ms` (compute only) -- see view 5.

> Source: `wire.rs` (`RendezvousMsgWire`), `ddp_run/mod.rs` (`ControlMsg`,
> `TimingMsg`, `MetricsMsg`), `cluster_coordinator/epoch_dispatch.rs`,
> `ddp_run/worker/sync.rs`.

---

## 3. Coordinator state machine

CPU averaging must **never block the scheduler** -- the coordinator keeps
servicing `check_throttle` and timing reports while a reduce is in flight. It
does this with a 3-state machine that snapshots counters at trigger time and
runs the average on a background thread.

```mermaid
stateDiagram-v2
    [*] --> Idle
    Idle --> Collecting: should_average()<br/>broadcast RequestParams<br/>snapshot steps + wall_ms
    Collecting --> Collecting: try_recv param_rx each tick
    Collecting --> Computing: all snapshots in<br/>spawn average thread
    Collecting --> Idle: deadline passed<br/>(soft abort, stale drain)
    Computing --> Idle: thread joins<br/>broadcast Update(avg)<br/>re-arm
    Computing --> [*]: shutdown
```

**Invariants that keep this correct:**

- **subtract-not-zero**: `steps_snapshot` / `wall_ms_snapshot` are captured at
  trigger time, so timing is attributed to the right window even though work
  continues during the cycle.
- **single-consumer reuse**: the worker's `ParamSnapshot` shares storage with
  persistent pinned buffers; the `Idle -> Collecting -> Computing -> Idle`
  cycle issues exactly one `RequestParams` per cycle and the worker
  re-snapshots only after the `Update` round-trips back, so a snapshot is
  never overwritten while in flight.
- **re-arm via `cpu_avg_state`** (back to `Idle`), *not* `nccl_ack` -- the CPU
  `SyncAck` carries no meaningful `step_count`.

> Source: `ddp_run/coordinator/cpu_avg.rs` (`CpuAvgState`),
> `ddp_run/coordinator/mod.rs`.

---

## 4. Reduce backends

The single most important divergence in the codebase: **NCCL reduce is inline;
CPU reduce is the 3-phase state machine.** They are gated on `AverageBackend`,
independent of pacing.

```mermaid
flowchart LR
    subgraph nccl["AverageBackend::Nccl -- inline"]
        direction TB
        N1["Coord: SyncNow broadcast"] --> N2["Worker: AllReduce in-place<br/>on comm_stream"]
        N2 --> N3["Worker: record CudaEvent"]
        N3 --> N4["Worker: SyncAck<br/>{ divergence, pre/post_norm }"]
        N4 --> N5["finish_averaging_nccl<br/>INLINE in trigger_averaging<br/>(before metrics drain)"]
        N5 --> N6["ElChe fed wall_ms_accum<br/>(compute-only)"]
    end

    subgraph cpu["AverageBackend::Cpu -- 3-phase"]
        direction TB
        P1["Coord: RequestParams"] --> P2["Worker: snapshot_pinned_params<br/>async D2H, single sync"]
        P2 --> P3["Worker: SnapshotReady"]
        P3 --> P4["Worker: send ParamSnapshot<br/>(data channel)"]
        P4 --> P5["Coord: average_params<br/>background thread"]
        P5 --> P6["Coord: Update { version, next_plan }"]
        P6 --> P7["Worker: load_averaged<br/>async GPU writeback"]
        P7 --> P8["ElChe fed delivered_ms_accum<br/>(compute + data + transport)"]
    end
```

Key asymmetries (each a hard-won fix):

- **Memory**: NCCL is zero-extra (in-place); CPU is `O(world_size *
  model_size)` host RAM.
- **Blocking**: NCCL sync at a collective barrier (fast GPU waits); CPU never
  blocks a GPU.
- **Timing feed**: NCCL's `finish_averaging_nccl` runs *before* the metrics
  drain, so its delivered cost would be stale -- it keeps the compute-only
  `wall_ms_accum` feed. CPU+Cadence gets the transport-aware
  `delivered_ms_accum` feed (view 5).
- **Snapshot readout** (CPU): batched **async** D2H into reused **pinned** host
  buffers, then a single `synchronize()` per window -- not per-param
  synchronous copies.

> Source: `ddp_run/mod.rs` (`AverageBackend`),
> `cluster_coordinator/averaging.rs`, `ddp_run/worker/sync.rs`
> (`snapshot_pinned_params`, `load_averaged`).

---

## 5. ElChe scheduling data flow

ElChe turns per-rank timing into a `batch_counts[rank]` schedule vector (the
reduce window). N GPUs are treated as **one logical GPU partitioned into
heterogeneous per-rank step counts**. The window is capped at one epoch so
syncs never collapse below 1/epoch.

```mermaid
flowchart TB
    A["Per-rank timing signal"] --> B{backend + policy}
    B -->|Cpu + Cadence| C["delivered_ms_accum<br/>dispatch -> completion delta<br/>(compute + data + transport)"]
    B -->|Nccl / Sync / Async| D["wall_ms_accum<br/>(compute only)"]
    C --> E["report_timing"]
    D --> E
    E --> F["ms_per_batch_window<br/>(ring buffer, window-mean)"]
    F --> G["recompute_batch_counts<br/>slow device = anchor<br/>fast devices range ahead"]
    G --> H["batch_counts[rank]<br/>= reduce window"]
    H --> I["compute_chunk_batches<br/>dispatch exactly counts[rank]"]
    I --> J["reduce + epoch barriers<br/>(reduce_step_budget)"]

    subgraph autotune["anchor auto-tune"]
        K["overhead_target<br/>keep reduce under ~10% wall"] --> G
        M["convergence_guard<br/>SuppressGrowth / NudgeDown"] --> G
        N["set_max_total_batches<br/>cap: window at most 1 epoch"] --> H
    end

    P["Phase: Probe -> Warmup -> Stable -> Mature"] -.->|hysteresis| G
```

The delivered-cost feed is what closed the cpu-cadence idle prize: ranks pay
compute + data + transport, but ElChe was scheduling on compute-only timing, so
it over-allocated the fast RTX and left it idle at the barrier. Feeding the
coordinator-measured dispatch-to-completion delta (which excludes the
reduce-barrier wait) made cpu-cadence track nccl-cadence.

> Source: `el_che.rs` (`ElChe`, `Phase`, `recompute_batch_counts`),
> `cluster_coordinator/epoch_dispatch.rs` (`compute_chunk_batches`,
> `take_next_chunk_plan`), `cluster_coordinator/averaging.rs` (`timing_feed`).

---

## 6. Message catalog

Two communication layers. **In-process channels** carry Rust types (including
`Tensor` handles) between coordinator and local worker threads. **Wire frames**
strip tensor handles (data travels paired on a data channel) and are
HMAC-signed with the session salt for cross-host transport.

### Wire frames (cross-host)

Every frame is tagged with a `MsgKind` and carried in an HMAC-signed
`ControlFrame`.

| `MsgKind` | Direction | Payload | Carries |
|---|---|---|---|
| `Control` | coord -> worker | `ControlMsgWire` | RequestParams / Update{version,next_plan} / SyncNow / StartEpoch / DeclareDead / ... |
| `Timing` | worker -> coord | `TimingMsgWire` | Batch / SyncAck / SnapshotReady / Heartbeat / LrUpdate / Exiting / EvalResult |
| `Metrics` | worker -> coord | `MetricsMsgWire` | per-epoch avg_loss, share_complete_ms, samples |
| `ParamSnapshotMeta` | worker -> coord | `ParamSnapshotMetaWire` | pre-snapshot metadata, paired with a data-channel RoundFrame |
| `Heartbeat` | worker -> coord | `HeartbeatWire` | liveness (distinguishes "alive at barrier" from "dead") |
| `Rendezvous` | both | `RendezvousMsgWire` | Hello / Role / Uid bootstrap |

> Source: `wire.rs` (`MsgKind`, `ControlMsgWire`, `TimingMsgWire`,
> `RendezvousMsgWire`).

### In-process channels (local threads)

| Channel | Direction | Type | Notable variants |
|---|---|---|---|
| control | coord -> worker | `ControlMsg` | RequestParams, Update(AveragedParams), SyncNow, StartEpoch(EpochPlan), ExtendPartition, DeclareDead, NewNcclSession, RequestNewNcclId, Throttle, SetGlobalStep |
| timing | worker -> coord | `TimingMsg` | Batch, SyncAck, SnapshotReady, Heartbeat, LrUpdate, Exiting, EvalResult, CheckpointResult |
| metrics | worker -> coord | `MetricsMsg` | epoch, avg_loss, batches_processed, share_complete_ms, samples_processed |
| data | both | `RoundFrame` | tensor payloads (ParamSnapshot, AveragedParams) |

The wire types are 1:1 mirrors of the in-process types with tensor handles
removed; the relay's `MuxRecord` wraps these into one connection per host, and
`RelayControlMsg` (Hello / HelloAck / RankExit) manages the relay-to-controller
leg.

> Source: `ddp_run/mod.rs` (`ControlMsg`, `TimingMsg`, `MetricsMsg`),
> `controller.rs` (`RoundFrame`), `relay/mux.rs` (`MuxRecord`,
> `RelayControlMsg`).

---

## How to keep this in sync

Each diagram cites the enum/struct it was built from. When you touch one of
those types, re-render the affected view -- the variant names here are pulled
verbatim, so a `grep` for a renamed variant finds every stale diagram. Mermaid
renders natively on GitHub and on flodl.dev; no build step.
