# Epoch-Tail Allocation: Eliminating the Boundary Fallback

**Status:** implemented for the barrier-paced (Cadence) progressive path;
cpu-async is a deliberate follow-up (see [Scope](#scope-cadence-first)).
Motivated by a single delivered-feed `COMPUTE-FALLBACK` window observed at
every epoch boundary on the heterogeneous rig (3-GPU, RTX 5060 Ti + 2× Pascal
GP106 behind asymmetric PCIe links). The Cadence implementation lives in
`compute_chunk_batches` / `refresh_final_window_plan` / `final_window_alloc`
(`cluster_coordinator/epoch_dispatch.rs`).

---

## Symptom

On a `nccl-cadence` rig run (`resnet-graph`, 20 epochs) one reduce window
per epoch boundary fell back from the delivered timing feed to the
compute-only feed. The `[coord-prof]` line at the epoch 14 → 15 boundary:

```
feed=COMPUTE-FALLBACK missing=[1] pb_delivered_ms/batch=[4.6, 0.0, 0.0]
  steps=[71, 1, 0] pb_batches=[70, 0, 0] batch_counts=[71, 18, 15] reduce_ms=18.7
```

Rank 0 (fast) ran its full 71-batch share, rank 1 ran exactly **one** batch,
rank 2 ran zero. This is benign today (one short window per epoch on the
compute scale, the all-or-none coherence gate doing its job) but it is
avoidable, and removing it tightens the delivered feed at exactly the points
where ElChe is re-deriving the schedule for the next epoch.

## Root cause

`ClusterCoordinator::compute_chunk_batches` has an EDGE branch for the
epoch/run tail: once `remaining < Σ batch_counts`, each rank is dispatched
`batch_counts[rank].min(remaining)`. Because it is evaluated per rank as the
shared pool drains in dispatch order, the fast rank consumes its full share
off the top and the slow ranks receive whatever scraps remain. With 72
batches left and a `[71, 18, 15]` schedule, rank 0 took 71, leaving 1 for
rank 1 and 0 for rank 2 — the `[71, 1, 0]` above.

Coverage is unaffected: the pool still drains to exactly 0, every sample is
dispatched once, and a rank that finds the pool empty contributes `(0, 0)`
and is excluded from the weighted average (sum-and-count). The fallback only
changes **which timing scale** ElChe schedules on for that one window; it
does not touch the averaging weights or convergence.

## The actual trigger: a lone-1 window, not a 1-batch chunk

The delivered accumulator in the event loop is marginal — it skips the
window's first batch (`steps_since_avg[rank] > 1`) so the per-chunk fixed
fill cost never enters the per-batch rate. Consequently a rank's delivered
sample is zeroed **only when it runs exactly one step in the whole window**:

- 0 steps → not a mover, excluded by the all-or-none predicate (fine);
- ≥2 steps → has a delivered sample (fine);
- **exactly 1 step → mover with no sample → trips the all-or-none fallback.**

So the governing invariant for the redesign is:

> **No fired reduce window may leave any rank at exactly 1 step.**
> Every participating rank ends at 0 or ≥2.

## Design

Let `R = remaining_batches − Σ batch_counts` be the tail remainder once the
edge regime is entered, `ratio[r] = batch_counts[r] / Σ batch_counts`, and
"fastest" = the rank with the largest `batch_counts` (the rank whose
between-sync drift is the convergence-sensitive quantity, so the one we never
pad).

Two branches:

### 1. `R ≥ world_size` — fire the final window, proportionally, then consolidate

Split the remainder proportionally: `round(R · ratio[r])`, assigning the
integer rounding residual to a single rank so the dispatched total equals `R`
exactly (coverage stays exact). Then a **consolidation pass**: any rank
allocated exactly 1 has that batch moved onto the smallest peer that already
holds ≥ 1 (never onto a 0 — that would create a fresh lone-1 and walk the
problem around the ring); the orphan drops to 0.

Because `R ≥ world_size` there is always a ≥1 peer, so the pass terminates
with every nonzero rank at ≥2. Worked cases:

```
[1,1,1] → [0,3,0]      [2,1,1] → [2,0,2]      [2,2,1] → [3,2,0]
```

### 2. `R < world_size` — fold the crumb into the penultimate window

When fewer than `world_size` batches remain there is not enough to give every
rank even one, so no clean final window exists. Instead, fold all of `R` into
the **penultimate** window as +1 batch on `R` of the slow ranks. There are
`world_size − 1` slow slots and `R ≤ world_size − 1`, so there is always
exactly enough room, and the fastest rank is never touched. No final window
fires.

This is the piece that closes the last gap. Within a single window the
`[1,0,0]` case (R = 1) is irreducible — there is no peer to consolidate the
lone batch onto. But the **previous window is always a valid sink**: a +1 on
a slow rank in the penultimate costs a single batch of negligible drift, and
the degenerate final window simply never happens. Composed with branch 1,
the two branches remove precisely the remainders that branch 1 cannot repair,
so **no fired window ever contains a lone 1, and no boundary window ever
falls back.**

### Trigger band

The fold is a forward decision, not a retroactive mutation. It fires while
sizing the penultimate window, when

```
Σ batch_counts < remaining < Σ batch_counts + world_size
```

i.e. "one normal window plus a sub-cohort crumb remain." `R` and the ratios
are known at that point, so the coordinator dispatches `remaining` (the
normal window plus the ≤ `world_size − 1` crumb spread one-per-slow-rank)
instead of `Σ batch_counts`. This branch sits just **above** the existing
edge condition (`remaining < Σ batch_counts`); the two never overlap.

## Why this is safe

- **Coverage stays exact.** Branch 1's consolidation only *moves* batches
  between ranks; branch 2's fold only *moves* batches from the final window
  into the penultimate. The dispatched total is always conserved, the pool
  still drains to 0, and the sum-and-count average is unchanged.
- **Fast-rank drift is untouched.** The fastest rank is never padded in
  either branch. "Smallest ≥1 peer" in branch 1 naturally coalesces slow
  lone-1s onto each other rather than onto the fast rank; branch 2 pads only
  slow ranks. The one rank whose extra between-sync steps could matter for
  convergence keeps exactly its scheduled count.
- **A +1 on a slow rank is free, often better than free.** Slow ranks have
  drift headroom by construction. At an epoch boundary the designated rank is
  also running eval/checkpoint (whose time `callback_roles` already subtracts
  from `wall_ms_accum` / `pb_delivered`); if that rank is the fast one,
  pushing the tail onto slow ranks overlaps their compute with the fast
  rank's eval bubble — a net win, not merely benign.
- **The fallback remains as a backstop.** The all-or-none gate is untouched.
  A bug in the new branches can at worst reintroduce a one-window
  compute-scale fallback; it cannot corrupt coverage or the average.

## A note on wording

The guarantee is "every participating rank ends at **0 or ≥2**," not "every
GPU runs ≥2." Consolidation legitimately parks some slow ranks at 0 in the
final window — they sit out, excluded as non-movers, which is correct. The
property we actually enforce is the absence of any rank at exactly 1, because
that is the fallback trigger.

## Scope: Cadence first

This specification targets the barrier-paced (Cadence) path, where one chunk
equals one reduce window and "the penultimate window" is a well-defined
cohort-level object. That is also where the observed `[71, 1, 0]`
originated.

cpu-async is deferred. It is still delivered-capable, so it can in principle
hit the same lone-1 fallback, but the dynamics are softer and the right
mechanism is different:

- **Overshoot diffuses the drain.** Async ranks stream several chunks ahead
  under the overshoot budget, so the pool does not empty in the clean
  dispatch-order cascade the barrier path shows; a sharp `[N,1,0]` is far
  less likely to form at all.
- **EASGD softens the cost.** The elastic pull means a single compute-scale
  tail window barely perturbs the trajectory.

The natural async intervention is overshoot-aware — let the boundary residual
ride the overshoot rather than forcing a barrier-aligned crumb window — which
is a genuinely separate mechanism. It can reuse the `R < world_size`
predicate but needs async-aware timing, and is taken up in a follow-up once
the Cadence path is landed and validated.

## Validation

The two branches live in `compute_chunk_batches` (served from a cohort plan
above the per-rank edge path), `refresh_final_window_plan` (caches the plan at
window start), and the pure `final_window_alloc` (unit-tested: proportional
split, consolidation, fold band, dead ranks, the irreducible lone-1).

20-epoch Cadence rig runs (3-GPU heterogeneous, `resnet-graph`) confirm the
fix against the existing `[coord-prof]` dump, which already reports `feed=`,
`steps=`, and `pb_batches=` per window:

| mode | `feed=delivered` | `COMPUTE-FALLBACK` | lone-1 windows | RTX share | eval |
|------|------------------|--------------------|----------------|-----------|------|
| cpu-cadence | 45 / 45 | 0 | none | 0.688 | 0.856 |
| nccl-cadence | 105 / 105 | 0 | none | 0.690 | 0.850 |

The originally-observed `[71,1,0]` boundary is gone; a representative small
final window now reads `steps=[3,3,0]` (`feed=delivered`) — rank 2 sits out at
0 (excluded as a non-mover) and the other two land at ≥2.

**Follow-up:** the async treatment (overshoot-aware, reusing the
`R < world_size` predicate), per [Scope](#scope-cadence-first).
