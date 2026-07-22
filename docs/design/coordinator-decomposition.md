# Coordinator Decomposition

The cluster control plane has accreted three distinct roles into one
surface: scheduling policy (when a reduce window fires, how work is
allocated), transport state machines (arming, polling, and finishing
averaging cycles per backend), and controller feedback (feeding ElChe
timing, applying convergence verdicts). This document records the
decomposition that separates them, in dependency order, and the
invariants that bound it.

The motivation is composition. The hierarchical architecture sketched
in [Hierarchical scaling](cloud-ddp.md#hierarchical-scaling-to-massive-clusters-preliminary)
generalizes both planes to a tree: intermediate nodes that present one
virtual device upstream and allocate work downstream. That
architecture is hypothesis-gated and none of it is built here. What
this decomposition does is make the seams narrow enough that a tree
node *could* implement them on both faces, per the same doc's mandate
that near-term work be shaped to generalize.

## Vocabulary: three roles, one overloaded name

- **Averager** (data plane): the star reducer in
  `distributed/controller.rs` (`ClusterController`) plus the per-host
  relay fold. Every rank ships a pre-scaled contribution; relays fold
  their local ranks into one host frame (`sum_frames`, an associative
  monoid that never divides); the root sums host frames and divides
  exactly once by the accepted realized-work mass
  (`reduce_realized_work`). The math is deterministic and already
  isolated.
- **Coordinator** (control plane): `distributed/cluster_coordinator/`.
  Hosts ElChe and the convergence guard, owns membership, epoch
  dispatch, checkpoint arming, callback roles, and the averaging-cycle
  orchestration.
- **Relay** (fold tier): `distributed/relay/`. One per host; speaks
  the controller protocol toward its local ranks and the muxed
  protocol upstream. It is the depth-2 case of the aggregation tree,
  already in production.

The type named `ClusterController` is the *averager*; the component
that behaves like "the controller" in the control-theory sense is the
*coordinator*. Prose in this repo should use averager / coordinator /
relay for these three; "controller" unqualified is reserved for the
launcher-side process (which today hosts both the averager and the
coordinator).

## What is tangled today

- **Realized work exists in four disguises.** The coordinator counts
  per-rank steps per window (`steps_since_avg`); ElChe holds scheduled
  and actual window counts (`batch_counts`); the worker computes the
  `n_i^gamma` contribution weight at snapshot time (`gamma_weights`);
  the data plane carries and normalizes by the mass
  (`RoundFrame::weight`, `reduce_realized_work`). Same concept, four
  vocabularies, no shared module stating the algebra.
- **The coordinator holds ElChe's raw material.** Per-rank wall-time
  and delivered-cost accumulators live as loose coordinator fields;
  the fill/marginal decomposition of first-batch timing and the
  delivered-vs-compute coherence policy (`timing_feed`) are
  scheduling-model computations performed in orchestration code before
  the results are handed to ElChe.
- **Scheduling policy is inlined in the event loop.** The window
  firing condition (`should_average`) re-derives ElChe's schedule;
  callback-slack shaping and chunk sizing re-derive dispatch
  quantities from `batch_counts` with policy branches.
- **ElChe's surface is too fine-grained.** The coordinator drives
  anchor tuning verdict-by-verdict through a cluster of methods
  (`commit_proposed_anchor`, `veto_proposed_growth`,
  `nudge_anchor_down`, `relax_anchor_up`, `discard_proposed_anchor`),
  which welds the guard's verdict taxonomy to the coordinator's call
  sites.
- **`cluster_coordinator/averaging.rs` mixes all three roles**: firing
  policy, the CPU and NCCL cycle state machines (deadlines, stall
  ceilings, ack collection, re-rendezvous), and the post-cycle ElChe
  feedback, in one file.

Notably *not* tangled: the averaging math itself. The sum monoid, the
divide-once law, gamma weighting, the EASGD blend, and the outer
optimizer are each in one place. The CPU-async mode in particular has
no distinct averaging code path; its identity is an apply-side blend,
an overshoot dispatch budget, and a barrier exemption.

## The decomposition, in dependency order

### Realized-work vocabulary

A small pure module owning the semantics every plane shares:

- The **mass algebra**: contributions are pre-scaled by a mass, masses
  sum through any number of associative fold tiers, and the divide
  happens exactly once at the root, over exactly the accepted cohort.
  The load-bearing law, "the sum and its divisor come from the same
  accepted frames", is stated and tested here.
- **Mass is policy-supplied**, not definitionally `n^gamma`. Gamma
  weighting is the current policy; the signature leaves room for
  verdict-modulated masses (a unit doing many steps of low-value work
  can be downweighted without an API break).
- **Mover rules**: the 0/1 buffer-mover indicator and the all-idle
  (zero-mass) round semantics.

The worker's pre-scale, the relay fold, and the averager's final
reduce all import this vocabulary instead of restating it.

### Window ledger

A coordinator-side component absorbing the loose per-rank fields:
step counts since the last reduce, wall-time and delivered-cost
accumulators, first-batch delivery marks, and mover detection. The
firing check becomes a ledger-plus-schedule query instead of inline
event-loop arithmetic.

The ledger is **advisory scheduling state**, a distinct authority from
the frame mass (which is ground truth, computed rank-side at snapshot
time). They must not be unified: the schedule remains the single step
clock, and the mass remains what the reduce actually normalizes by.

### ElChe interface inversion

The coordinator currently accumulates timing and pushes digested
aggregates into ElChe; the inversion makes ElChe ingest events and
answer questions.

- **Events in**: batch delivered, first-batch fill, sync elapsed,
  callback cost, convergence verdict.
- **Queries out**: per-rank window counts, anchor, window-completion.
- The `timing_feed` coherence policy and the fill/marginal
  decomposition move inside ElChe, next to the model they feed.
- The verdict-application method cluster collapses into one
  **source-agnostic** `apply_verdict`: ElChe does not know whether a
  verdict came from the convergence guard, the meta-controller, or a
  future detector. Arbitration between verdict sources is the
  caller's, in one place.

This seam is the recursion enabler. A tree node implements it on both
faces: upward it is one rank (reporting aggregate delivered cost and
realized mass), downward it is a coordinator (allocating windows to
children). Without the narrow seam there is nothing to recurse on.

### Averaging-cycle extraction

A per-backend cycle component (CPU, NCCL) owning the transport
mechanics currently interleaved in `averaging.rs`: arming the cycle,
deadline and stall ceilings, ack collection, elastic re-rendezvous.
The coordinator keeps exactly two hooks: "should this window fire?"
(policy, from the ledger and schedule) and "here is the cycle report,
retune" (feedback, into ElChe and the guard).

Constraints on the extraction:

- Any interior wait must keep the coord-to-rank heartbeat beating; a
  cycle that blocks silently past the ranks' watchdog deadline
  self-destructs the cohort.
- The cycle owns no cadence state. The schedule is the single step
  clock; the cycle executes windows, it never decides them.
- The pre-fold moment stays tappable: the point where per-child frames
  coexist before summation must remain wrappable by an observer (the
  checkpoint forge tap is the precedent). The horizontal convergence
  check hypothesized in
  [cloud-ddp.md](cloud-ddp.md#horizontal-convergence-checks-hypothesis)
  lives exactly there if it ever lives anywhere.

### External averager decoupling (deferred)

The averager's wire protocol already assumes nothing about
co-location; what pins it to the launcher process is shared memory:
the dead-rank ledger, the checkpoint forge handle, and the in-process
outer optimizer. Replacing those with a membership and checkpoint
interface is mechanical, and deferred until an averager node (or an
averaging cluster) is actually on deck. Nothing in the earlier moves
may widen these couplings.

## Invariants that bound the refactor

Carried from the cluster sync invariants; restated because each one
vetoes an otherwise-tempting simplification:

- One step clock: coverage and synchronization never split into two
  racing clocks. The ledger and the schedule describe the same clock.
- Window never spans more than one dataset pass.
- ElChe schedules on delivered cost, not compute-only timing.
- Param-snapshot D2H stays single-consumer per window.
- The coordinator is never heartbeat-silent while alive.
- Advisory counts (ledger) and ground-truth mass (frames) are separate
  authorities; the reduce normalizes only by what it accepted.
- Control-plane telemetry stays per-rank and structured end to end;
  only the data plane folds.

## What this does not do

No rack, row, or root tier is built, and no horizontal detector. Those
remain gated by the hierarchy hypotheses (H1 foremost) in
[cloud-ddp.md](cloud-ddp.md#hierarchical-scaling-to-massive-clusters-preliminary).
This decomposition is the near-term shaping that document calls for:
after it, the seams are narrow enough that each future tier is an
implementation of existing interfaces rather than a redesign.
