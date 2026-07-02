# Trainer Execution Tiers: one authoritative controller, three ergonomics

flodl's distributed layer currently exposes two cluster entry points that
diverge in both feature set and control model:

- `Trainer::builder(...).run()` runs through the **`ClusterCoordinator`**
  (authoritative): the controller computes cadence, data partition, anchor,
  callback-role election, and drives elastic membership + checkpoint
  orchestration + the dashboard. Ranks execute the controller's plan.
- `Trainer::setup(&graph, ...)` runs a **self-driven, replicated** ElChe:
  ranks AllReduce their per-rank wall times and each computes an identical
  schedule locally, with no controller spawned. It keeps the user's own
  training loop, but only carries *base* ElChe (overhead/anchor/speed-ratio),
  not the convergence guard, LR-aware meta-controller, outer optimizer,
  EASGD, relax-up, or the CPU averaging backend, and gets none of the ops
  layer (elastic membership, checkpoint-on-failure, dashboard).

So there are effectively two scheduling brains, and they can drift: a
`setup` run and a `builder` run of the same configuration do **not**
necessarily produce the same model once the controller-only features engage.
This note proposes collapsing that to one engine with three ergonomic tiers.

## Principle: the controller stays authoritative

Cross-rank decisions belong in one place. Cadence, data partition,
work-weighted grad averaging, callback-role election, dead-rank consensus,
per-host relay topology, and sum-and-count aggregation across hosts are
genuinely centralized concerns. Replicating that logic onto every rank trades
one observer for an N-way consensus problem (membership agreement,
split-brain) and introduces a cross-rank floating-point determinism hazard
(see the [determinism rule](#the-determinism-rule-for-any-replicated-decision)
below) for no real gain. The per-host **relay** already exists as the
per-host transport/aggregation boundary; the controller already scales
through it. Keep the controller authoritative; do not push its brain onto the
ranks.

The corollary: "give the user their training loop back" must mean a
**cooperative** loop that participates in the controller's schedule, not a
loop that bypasses the controller.

## Three tiers on one engine

| Tier | Entry | Who owns the loop | Controller |
|---|---|---|---|
| Bypass | `Ddp::wrap` | user, fully | none (raw per-rank collectives) |
| Cooperative | `Trainer::builder(...)` + a rank handle | user owns the loop body | authoritative |
| Managed | `Trainer::builder(...).run()` | framework | authoritative |

The managed tier and the bypass primitive exist today. The **cooperative
tier is the missing middle** and is what closes the `setup` / `builder` gap.

A cooperative loop reads like single-device code; the cluster is presented as
one logical trainer ("the collective as a whole"):

```rust
let mut w = Trainer::builder(model_factory, optim_factory).into_worker()?;
// on the launcher: fans out + drives the controller, never returns past here.
// on each rank: connected to the controller, ready to drive.
for epoch in w.epochs() {              // controller pushes the epoch plan
    for batch in w.batches(epoch)? {   // iterates THIS rank's assigned shard
        let loss = train_step(w.model(), &batch)?;
        loss.backward()?;
        w.step()?;                     // see the cooperation contract below
    }
}
w.finish()?;
```

Because `w.step()` consults the same controller `run()` uses, the
cadence/partition/averaging decisions, and therefore the trained model, are
identical to the managed tier. The only difference is who writes the `for`.

## The cooperation contract

The framework loop performs a fixed set of actions, in a fixed order, that
the controller's schedule depends on. A cooperative loop must preserve all of
them, so they live **inside** `step()` / the handle, never in user hands:

1. **Delivered-cost timing report** every batch. ElChe schedules on per-rank
   delivered cost (compute + data + transport); without it the controller is
   blind and mis-allocates.
2. **Control draining** at the sync boundary, so `SyncNow` / `StartEpoch` /
   `ExtendPartition` / `DeclareDead` are seen in time. Missing it desyncs or
   wedges the cohort.
3. **Stream-ordered, fenced gated reduce.** The work-weighted AllReduce must
   run on the comm stream, fenced against the compute stream (a hand-rolled
   reduce reintroduces the intermittent-NaN / stall class of bug).
4. **Heartbeat.** A long user step that does not heartbeat gets the rank
   declared dead by the controller's failure detector.
5. **Coordinator-assigned shard + coverage accounting.** The loop consumes
   the controller's partition, not arbitrary batches, so the epoch (=one
   dataset pass) and `checkpoint_at_epoch` invariants hold.

The user supplies only forward/backward and intents. Anything the API exposes
rawly is something a user loop can get wrong; the contract is enforced by
encapsulation.

## Intent channel: tuning into the controller

The user steers controller-side events through a **request, not a command**:
"eval at the next occasion", "checkpoint now". The request flows rank/user ->
controller, and the controller folds it into its next *coherent* dispatch, on
the rank its policy elects (`Fastest` / `Rank(n)`), at the next sync boundary.
The user expresses intent; the controller decides *when* and *which rank*.
This gives the imperative control users actually want without surrendering
the controller's authority or coherence.

## When you must bypass: `Ddp::wrap`

The cooperative tier assumes the standard shape: one forward, one backward,
one gated sync, one optimizer step. The cases that genuinely need to bypass
the controller are the ones whose *step shape* differs:

- multi-model / multi-optimizer with distinct sync cadences (GAN, actor-critic),
- custom collectives (all-gather a statistic, not just AllReduce gradients),
- dynamic resharding mid-run.

These keep the raw per-rank primitive `Ddp::wrap` (manual collectives, you own
everything). Bypass is a deliberate escape hatch, not a fallback.

## Why "the collective as a whole" is enough (data parallel)

In data-parallel DDP the ranks are *replicas*: same model, different data,
identical parameters after each reduce. There is therefore no legitimate
per-rank-divergent *training* logic to express. The only single-rank actions,
eval, checkpoint, logging, are side tasks you specifically do **not** want
hand-coded per rank (that is N eval passes / N file writes); the controller
already elects one rank for them. So collective-as-a-whole plus controller
role-election covers every single-rank need. The user's cooperative loop can
present the whole cluster as one logical device.

## Forward-compatibility: model splitting as a meta-GPU layer

Splitting one model across several GPUs (tensor- or pipeline-parallel) does
**not** reframe any of the above. It is the standard 2D composition:
data-parallel *across* groups, model-split *within* a group. A group is
abstracted as one logical data-parallel replica, a "meta-GPU", and the
controller keeps operating at replica granularity (cadence, partition,
grad-averaging across replicas, role election are all unchanged whether a
replica is one physical GPU or a group of K).

The grouping has a natural home: the **per-host relay** already aggregates a
host's ranks and muxes them to the controller, and model-split groups live
intra-host on the fast interconnect (NVLink). So the relay is where a host's
GPUs can be presented to the controller as one (or a few) meta-GPU replicas,
no new topology machinery required at the controller.

Where it composes cleanly vs. where the layer does real work:

- **Tensor parallel** composes cleanly: the model's forward/backward does
  intra-group collectives internally; externally it is still one
  forward/backward. The cooperative loop and the contract are untouched.
- **Pipeline parallel** preserves the external contract (one batch -> grads)
  but its internal step is a different engine: a microbatch schedule (1F1B /
  GPipe with bubbles, activation stashing, stage-to-stage point-to-point). It
  leaks one knob to the user (microbatch count).
- **New knobs** the layer introduces: topology-aware placement (group on the
  fast interconnect, DP across groups), failure blast-radius (one physical GPU
  death takes the whole replica down, the controller's elastic logic still
  operates at replica granularity, but the operator-facing cost is K GPUs
  idle), and ElChe measuring the group's throughput rather than a single
  GPU's.

Net: model-splitting is an added abstraction layer plus a couple of knobs, not
a reframe of the controller or the cooperative loop.

## The determinism rule (for any replicated decision)

If any decision is ever computed redundantly on each rank rather than centrally
(as the self-driven path does today, and as a replicated guard/meta-controller
would), it may branch **only** on values explicitly reduced to a scalar that is
bit-identical across ranks. It may **not** branch on a quantity each rank
computes locally from its own post-AllReduce parameters, even one that "should"
be equal: NCCL `AllReduce(Avg)` does not guarantee bit-identical results across
ranks (floating-point non-associativity in the reduction tree), so two ranks
can read slightly different values and decide differently, desyncing the
cohort. This is not hypothetical: a post-reduce parameter-norm consensus check
trips on roughly `6.6e-4` relative cross-rank drift on a real 2-GPU rig. Base
ElChe is correct because it decides on the AllReduce'd timing vector (identical
on every rank); a divergence-keyed guard would have to AllReduce the divergence
scalar first.

This rule is the reason the design keeps scheduling **authoritative-central**
rather than replicated: it sidesteps the hazard entirely.

## Consequence for `Trainer::setup`

With the cooperative tier in place, `setup`'s self-driven replicated ElChe is
the odd one out, the only path that schedules without the controller. It can
be re-expressed as the cooperative tier (a cooperative loop plus the `Graph`
convenience) or retired. Either way the replicated scheduling brain goes away,
leaving **one authoritative controller, one rank engine, one cooperation
contract**, exposed at three ergonomic levels, with model-splitting as a
future layer on top.

## Scope

This is a design direction, not a description of shipped behavior. The managed
tier (`Trainer::builder().run()`) and the bypass primitive (`Ddp::wrap`) exist
today; the cooperative tier, the intent channel, and model-splitting are
proposed. None of them reframe the controller that ships now.
