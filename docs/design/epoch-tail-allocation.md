# Epoch-Tail Allocation: Eliminating the Boundary Fallback

**Status:** the barrier-paced (Cadence) epoch-tail fix is implemented. The
async **checkpoint/resume** mechanism (the [Async](#async) section: consensus
model plus exact data-coverage, both CPU and NCCL backends) is implemented and
unit-validated; rig kill-and-resume validation is pending. Eval-on-consensus
and the callback report-and-wait gate remain design.
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
rank 1 and 0 for rank 2 - the `[71, 1, 0]` above.

Coverage is unaffected: the pool still drains to exactly 0, every sample is
dispatched once, and a rank that finds the pool empty contributes `(0, 0)`
and is excluded from the weighted average (sum-and-count). The fallback only
changes **which timing scale** ElChe schedules on for that one window; it
does not touch the averaging weights or convergence.

## The actual trigger: a lone-1 window, not a 1-batch chunk

The delivered accumulator in the event loop is marginal - it skips the
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

### 1. `R ≥ world_size` - fire the final window, proportionally, then consolidate

Split the remainder proportionally: `round(R · ratio[r])`, assigning the
integer rounding residual to a single rank so the dispatched total equals `R`
exactly (coverage stays exact). Then a **consolidation pass**: any rank
allocated exactly 1 has that batch moved onto the smallest peer that already
holds ≥ 1 (never onto a 0 - that would create a fresh lone-1 and walk the
problem around the ring); the orphan drops to 0.

Because `R ≥ world_size` there is always a ≥1 peer, so the pass terminates
with every nonzero rank at ≥2. Worked cases:

```
[1,1,1] → [0,3,0]      [2,1,1] → [2,0,2]      [2,2,1] → [3,2,0]
```

### 2. `R < world_size` - fold the crumb into the penultimate window

When fewer than `world_size` batches remain there is not enough to give every
rank even one, so no clean final window exists. Instead, fold all of `R` into
the **penultimate** window as +1 batch on `R` of the slow ranks. There are
`world_size − 1` slow slots and `R ≤ world_size − 1`, so there is always
exactly enough room, and the fastest rank is never touched. No final window
fires.

This is the piece that closes the last gap. Within a single window the
`[1,0,0]` case (R = 1) is irreducible - there is no peer to consolidate the
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
  rank's eval bubble - a net win, not merely benign.
- **The fallback remains as a backstop.** The all-or-none gate is untouched.
  A bug in the new branches can at worst reintroduce a one-window
  compute-scale fallback; it cannot corrupt coverage or the average.

## A note on wording

The guarantee is "every participating rank ends at **0 or ≥2**," not "every
GPU runs ≥2." Consolidation legitimately parks some slow ranks at 0 in the
final window - they sit out, excluded as non-movers, which is correct. The
property we actually enforce is the absence of any rank at exactly 1, because
that is the fallback trigger.

## Scope: Cadence first

This specification targets the barrier-paced (Cadence) path, where one chunk
equals one reduce window and "the penultimate window" is a well-defined
cohort-level object. That is also where the observed `[71, 1, 0]`
originated.

cpu-async does **not** share this problem, and the reason is worth stating: its
reduces ride the overshoot cadence, not epoch-aligned windows, and
`pb_delivered` accumulates continuously across the overshoot (spanning chunks
and epochs). So a small epoch-tail chunk never produces a 1-step *reduce
window* - the `[N,1,0]` fallback is a Cadence artifact of barrier-aligned final
windows. Async needs no `final_window_alloc` analog.

Async's boundary question is a different one - coherent callbacks and
resumable checkpoints - and is specified in [Async](#async) below.

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
final window now reads `steps=[3,3,0]` (`feed=delivered`) - rank 2 sits out at
0 (excluded as a non-mover) and the other two land at ≥2.

---

## Async

**Status:** the **checkpoint/resume** mechanism described here is implemented
and unit-validated (see [What landed](#what-landed)); rig kill-and-resume
validation is pending. Eval-on-consensus and the callback report-and-wait gate
remain design. Async (CpuAsync - there is no
NcclAsync) has no lone-1 tail problem (see [Scope](#scope-cadence-first)), so
this section is not about chunk sizing. It is about the *other* thing the
epoch boundary touches in async: firing callbacks coherently and writing
checkpoints that resume without repeating data - while never introducing an
artificial barrier.

### The boundary is a bookkeeping point, not a pacing event

Async has no epoch barrier. A rank streams straight across the epoch boundary
into the next epoch's pool; the only cohort gate is the reduce barrier, which
holds a rank at `counts[rank] + max_overshoot` steps since the last reduce.
The single-step-clock invariant ("coverage and synchronization must not split
into two racing clocks") keeps the cohort within one window of the shared
reduce clock. Epochs are a data-coverage label, not a synchronization event -
and the design below keeps them that way.

LR scheduling, convergence, and ElChe are all step-denominated and don't care
where epochs land. Eval cares only about trend and the single final number
(which already gets its own consensus reduce before the canonical eval). So
the epoch boundary is load-bearing for exactly one consumer: **checkpoint**,
and only because an epoch has meaning for data coverage on resume.

### Callbacks, split by what they touch

- **`metrics_fn` - controller-side, model-free.** Already fires from the
  aggregate hook when every rank has crossed the epoch (`is_epoch_done`), with
  the aggregated metrics in hand. This *is* the epoch-report hook; no separate
  rank-side reporting callback is needed. No barrier.
- **`eval_fn` - elected rank, on the consensus.** Reads the model, so it must
  operate on the **consensus average**, not the elected rank's own (overshot,
  or EASGD-blended) weights.
- **`checkpoint_fn` - consensus by construction.** CPU backend: fires
  controller-side on the forge's consensus materialization (no rank stall, no
  slack accounting on that path). NCCL: elected rank, post-collective.
- **`epoch_fn` - elected rank, generic per-epoch model hook.** Unchanged; the
  rank-side callback that may touch the model for arbitrary user purposes.

### Eval/checkpoint observe the consensus non-destructively

There is no persistent center variable: the consensus each cycle is the
averaged param set the worker receives (`update.params`), which the EASGD
elastic blend reads but never mutates. So the elected rank **overshoots
normally** - no barrier, no hold, no unlock - and at the boundary reduce it
eval/checkpoints `update.params` (the consensus), captured *beside* the blend
into its own training weights:

- **checkpoint** serializes the received average directly;
- **eval** forwards on the staged average (`no_grad`), not the rank's blended
  weights.

The training trajectory is byte-identical to a no-eval run - the callback is a
pure observation. Two effects must be kept separate, because they have very
different review-risk:

- **(A) which average gets saved is non-deterministic in step count** (reflects
  N epochs ± up-to-`max_overshoot` batches). This is inherent to
  async-without-a-barrier and standard for the Local-SGD family; state it
  plainly ("step count varies by ≤ max_overshoot, sub-epoch noise").
- **(B) the elected rank's post-callback trajectory.** Do **not** take the
  "α=1 full-adopt" shortcut (elected rank jumps to the consensus to avoid a
  param-buffer): it perturbs that rank's EASGD trajectory for no necessary
  gain. Observe non-destructively instead. And never defend (B) with "noise is
  helpful" - that conflates an avoidable artifact with the intentional
  exploration noise, which is exactly what a careful reviewer flags.

### Checkpoint: when, and the resume contract

A checkpoint fires at the **first reduce after the epoch boundary is
crossed**, and captures, snapshotted atomically at that reduce:

1. **the consensus model** (`update.params`) - written by the elected rank
   (the reserved `target_rank == u64::MAX` sentinel already anticipates a
   controller-as-checkpointer variant for CpuAsync, where the coord holds the
   averaged tensors);
2. **resume counters** - `global_step`, epoch, ElChe + sync state (controller
   meta);
3. **exact data-coverage** - for each in-progress epoch pool: the *completed*
   ranges, the cursor, and the epoch's **shuffle seed**.

**Resume** reconstructs the pools to the recorded completed-coverage and
dispatches only the uncovered remainder. Two details are load-bearing for
"resume without repeating data":

- **Completed, not dispatched.** A chunk in-flight at the checkpoint reduce
  has *not* had its gradient applied into the consensus, so it is recorded as
  not-covered and **re-dispatched** on resume. That is first-coverage, not a
  repeat - and it is why the consensus↔coverage snapshot must be atomic at the
  reduce (the coverage recorded is exactly "what's in this average").
- **The in-progress shuffle seed.** "Cover only the remaining chunks" is only
  well-defined if the resumed epoch reuses the same permutation. Resume
  reconstructs the pool over the recorded seed and dispatches the uncovered
  ranges. Without it, a fresh reshuffle re-randomizes the index space and
  re-coverage becomes unavoidable.

With exact coverage recorded, **resumability is independent of how the cohort
was spread across epochs at snapshot time.** Even a consensus smeared across
several epochs is exactly resumable - you record the precise completed-set
that produced it. The redo is only the in-flight-at-R chunks (small, bounded),
never whole epochs. So the smear width is **not** a checkpoint-correctness
concern; it is purely a training-dynamics concern (averaging too-divergent
models), already owned by `max_overshoot` and the convergence guard.

### No optimizer momentum in the checkpoint (today), and the outer-optimizer arc

The cluster path param-averages (Local-SGD / EASGD): each rank's optimizer
momentum accumulates on its own local gradients and is **never synced**, so it
diverges per-rank. Unlike gradient-averaging DDP (where the optimizer state
stays bit-identical across ranks and any one rank's is canonical), there is no
canonical single-rank momentum to save. The checkpoint therefore stores the
**consensus model only**, and resume **re-warms the inner optimizer from
fresh**. This keeps the checkpoint topology-independent (no per-rank `.optim`).

**Rig finding (2026-06-18) revises the cost.** The earlier claim, that the
fresh-inner warm-up is "short and sub-noise," does **not** hold in the resume
regime. Resuming a partly-converged run (lenet, checkpoint at epoch 3/8, train
loss already ~0.05), fresh Adam near the optimum mis-scales: with tiny
gradients `m/sqrt(v)` approaches `sign(g)`, so each step moves about `lr`
regardless of gradient magnitude, and cpu-async takes *hundreds* of local steps
per window (one rank ran 611) before consensus pulls it back. The result was a
sustained ~1e10 local training loss, masked from eval (0.9857) only because the
work-weighted consensus stayed near the loaded optimum. So fresh-inner resume
is not benign for a checkpoint of a model already in training, which is the
normal case. **Caveat:** the tiny-model, near-convergence regime almost
certainly *amplifies* this (gradients are smallest exactly there). Severity on a
real workload is unknown, so the first step of this arc is to **re-measure** it.

This reclassifies the **outer optimizer** from "parked nice-to-have" to the
principled fix: a *canonical global* optimizer state that survives a resume by
construction, and a known convergence lever (it was already on the convergence
bucket list). It is an **optimization-method track**: it changes training, not
only checkpointing, so it ships **opt-in** behind one selector, defaulting to
today's exact behavior, so all three regimes A/B on the same harness.

#### Step 0: re-measure the resume cost on a real model

Before building, quantify what we are fixing. Run resnet (`--depth-n`) at a real
epoch count, checkpoint mid-run, resume, and measure post-resume train-loss
continuity (not just eval). If the spike is tiny-model-only and a real run's
warm-up is genuinely sub-noise, the soft fix (SlowMo) suffices for resume and
DiLoCo is wanted for convergence alone; if the spike persists, DiLoCo's strict
fix earns its keep on the resume axis too. The design below supports both; this
measurement sets how hard we lean on each.

#### Pluggable `OuterOptimizer` at the guard tier

The outer step lives at the **same tier as the convergence guard**: the
controller's reduce, between the work-weighted average and the scatter back to
ranks (exactly where the [consensus forge](#async) already taps). It transforms
the averaged consensus into the new global before broadcast.

- **Trait (averaged-consensus form):**
  `outer_step(prev_global, work_weighted_consensus) -> new_global`, momentum
  held internally. It consumes the consensus the reduce *already* produces (the
  sum-and-count weighted average), **not** per-worker deltas: the outer gradient
  `g = prev_global - consensus` equals `mean_k(prev_global - theta_k)` under
  that weighting, so no per-rank state is needed at the controller and the step
  composes with the relay's per-host fold.
- **Variants:** `OuterAvg` (default; identity passthrough = today, zero
  regression), `SlowMomentum` (SlowMo), `NesterovMomentum` (DiLoCo).
- **Selector:** one builder setter (`.outer_optimizer(..)`); absent = `OuterAvg`.

#### The selector spans two tiers (not just a coordinator knob)

DiLoCo's faithful-resume property comes from **disposable inner state**, which
is a *worker-side* behavior the coordinator knob cannot supply alone:

| regime | coordinator (outer step) | worker (inner policy) | new optimizer API |
|---|---|---|---|
| `OuterAvg` (today) | identity | continuous inner, EASGD blend | none |
| `SlowMomentum` | slow momentum on consensus | continuous inner, EASGD blend | none |
| `NesterovMomentum` (DiLoCo) | Nesterov on pseudo-grad | **full overwrite (alpha=1) + reset inner each round** | `Optimizer::reset_state()` |

So DiLoCo needs: (1) a new `Optimizer::reset_state()` on the trait + impls
(clear Adam's `m`/`v` to `None`, `step_count = 0`; trivial given the lazy
`None`-init, but it is real API surface across optimizers); (2) the worker sync
handler, in DiLoCo mode, applying the new global with no blend and calling
`reset_state()`; (3) the selected variant signaling that inner policy to the
workers. `SlowMomentum`/`OuterAvg` keep the inner loop exactly as today.

#### Backends: CPU forge (centralized) vs NCCL (replicated per rank)

The outer step's *site* differs by backend, because the consensus lives in
different places. The trait is the same; only where it runs, and where its
momentum lives, changes.

**CPU.** The consensus is forged host-side in the controller reduce thread, so
the outer step runs **once, at the controller**, on the averaged frame before
scatter. The momentum is a single host buffer, and the controller already holds
the last scattered global, so `prev_global` is free. The scattered frame becomes
the outer-stepped `new_global`.

**NCCL.** The in-place AllReduce leaves the consensus on **every rank's GPU**;
the controller never sees model bytes (the same reason NCCL checkpointing is
elected-rank). So the outer step runs **replicated, once per rank**: each rank
holds the outer momentum (replicated) plus a `prev_global` anchor (the global it
adopted at the end of the previous round) and computes
`outer_step(prev_global, consensus)` on its own GPU copies. Identical inputs and
a deterministic op give every rank the same `new_global` and the same momentum
update, so the cohort stays in lock-step **with no extra collective** (the outer
optimizer is replicated state, exactly like the model already is on the NCCL
path). Electing one rank to compute and broadcast would add a broadcast and a
single point of failure; replicated needs neither.

This makes the factory **per-site**: `.outer_optimizer(|| ..)` is instantiated
once at the controller for CPU, but once per rank for NCCL (each rank owns its
replicated instance).

| | CPU | NCCL |
|---|---|---|
| consensus produced | controller reduce (host) | in-place AllReduce (every GPU) |
| outer step runs | once at controller | replicated, per rank on GPU |
| momentum lives | one host buffer | replicated GPU buffer per rank |
| `prev_global` | controller holds last global | per-rank GPU anchor (new buffer) |
| momentum checkpoint | forge, host payload-direct | elected rank, D2H |

The model checkpoint already splits this way (forge host-side for CPU,
`SaveConsensusModel` elected-rank for NCCL); the outer momentum rides the same
split. "All ranks AllReduce, all apply the identical outer step, the elected
rank checkpoints its copy" holds by construction, since the replicated momentum
is canonical.

**VRAM:** NCCL DiLoCo adds **two param-sized GPU buffers per rank** (replicated
momentum + `prev_global` anchor) on top of model + optimizer. Standard DiLoCo
footprint, but asymmetric vs CPU (where the momentum sits once on the host) and
worth budgeting on small cards (the Pascal 6 GB rig).

#### Uneven allocation stays meaningful

ElChe sizes per-rank work unevenly, so a fast rank takes more inner steps and
drifts further. This composes correctly because the consensus the outer step
consumes is **work-weighted** (`sum_k w_k theta_k / sum_k w_k`, `w_k` = batch
count): a fast rank is weighted in proportion to the data it actually processed,
so the outer gradient leans toward more-data, not toward fast-hardware bias
(with IID sharding). The second-order concern is the **outer learning rate**:
its calibration assumes a per-round drift *scale*, which shifts as ElChe
re-allocates. ElChe already tracks per-rank step counts, so the pseudo-gradient
can be step-normalized, which is precisely the **MSF-as-DiLoCo-H-controller**
composition (MSF gives DiLoCo the principled `H`/scale schedule it leaves as a
tuned constant).

#### Checkpoint byproduct

The outer momentum is model-sized (one buffer per parameter), so it is a second
consensus artifact, `<stem>.outer.fdl`, written by the **same writer as the
consensus model** (forge host-side on CPU, elected rank D2H on NCCL; see the
backend table above), with the same atomic-rename. Inner state stays disposable
and uncheckpointed. Resume loads consensus + outer momentum (replicated back to
every rank on NCCL, as the model is); under DiLoCo the fresh inner is correct by
design, so the resume is faithful. `OuterAvg` resume is unchanged (no outer
artifact).

#### A/B plan

Three arms on one harness and seed: **no-outer (`OuterAvg`, current EASGD) /
SlowMo / DiLoCo**, measured on **two axes**: convergence (held-out eval) and
resume (post-resume train-loss continuity). The selector makes no-outer the
default codepath, so the baseline is the real production path, not a synthetic
one.

### VRAM: the elected rank is asymmetric

Observing the consensus is **not** a second full model - no duplicate
optimizer, no second module. Eval adds a param-sized average buffer (largely
reusable from the EASGD staging copy already allocated in the apply path) plus
forward-pass activations (`no_grad`, freeable layer-by-layer) plus eval-data
batches; checkpoint adds only the param-sized average. But that working set
competes with the adaptive prefetcher's ~90% ceiling, so the elected rank can
OOM at the boundary on a full-budget prefetch.

The elected rank is asymmetric in **time** (already handled - `apply_callback_slack`
pre-reserves its callback wall-time in ElChe's allocation) and now in **space**.
The space fix, deferred as a named follow-up, is to **predictively ramp the
prefetch margin down as the elected rank approaches the boundary** (more async
load around the boundary, sized from a rough eval-working-set estimate, then
ramp back) - composing with the prefetcher's auto-resize, rather than a
permanently lower ceiling or a full drain-and-cold-refill.

### What landed

Resume was epoch-granular (`start_epoch`, fresh pool). The async work added
**coverage-granular reconstruction**:

- **Coverage.** `ChunkPool::uncovered_ranges()` derives the holes (the
  unassigned tail plus the in-flight `outstanding` chunks plus `reclaimed`);
  `ChunkPool::from_coverage()` rebuilds a pool serving only those, reusing the
  existing reclaimed / `take_chunk` machinery (built for dead-rank
  redistribution) to re-dispatch the in-flight-at-reduce chunks as
  first-coverage. `CheckpointMeta` gained an optional `CoverageBlock` (the
  shuffle seed plus, per in-progress epoch, the uncovered offset ranges); older
  meta files still parse. Resume verifies the seed and rebuilds via
  `ClusterCoordinator::resume_progressive_from_coverage()`; the launcher
  kickoff falls back to fresh dispatch when no coverage is recorded. No
  barrier, no rendezvous, no `final_window_alloc` analog.

- **Atomicity.** The checkpoint is armed at `trigger_averaging`, before the
  `RequestParams` / `SyncNow` broadcast (before workers freeze their params for
  the reduce), so the captured coverage is a subset of the consensus. A
  finish-time capture would over-count under async overshoot and lose data; an
  early capture can only redo in-flight chunks (bounded), never lose covered
  ones. The `.meta.json` is written at `finish_averaging_*` with the final
  post-round counters. This confirms the fact flagged before building:
  `should_average` plus the `max_overshoot` reduce barrier keep the cohort
  within one window of the reduce clock, so the in-flight-at-reduce redo stays
  small.

- **Consensus model, at the forge** (the meta is always coord-side). CPU: the
  consensus is forged in the controller reduce thread, so a `CheckpointForge`
  holds the static model schema (param/buffer names captured once at launch
  from a CPU-built model) and writes a named `.fdl` from the averaged
  `RoundFrame` on a detached thread, keeping the controller a model-agnostic
  byte reducer. NCCL: the consensus is on-device after the collective, so a
  `SaveConsensusModel` control frame tells the elected rank to write
  `self.model` (no EASGD blend on the NCCL path).

- **No optimizer momentum** in the checkpoint (param-averaging has no canonical
  momentum); resume re-warms the inner optimizer fresh.

A one-shot trigger (`checkpoint_at_epoch`) drives validation; the recurring
cadence (and eval-on-consensus, the callback report-and-wait gate) is a
follow-on.

---

## Open: the sub-batch tail (partial final batches)

Separate from the lone-1 window above, and unsolved. An epoch trains whole
batches, so `epoch_samples % batch_size` picks fall outside its last batch.
Today they are excluded from the pool *before* allocation, which is what
makes the coverage argument in this document hold — the pool still drains to
exactly 0 and every allocated sample is realized. The cost is that those
picks are not trained on that pass.

`epoch_splits` raises the stakes: the drop applies once per epoch, so a pass
sheds it `epoch_splits` times instead of once.

Mitigated today rather than fixed:

- **Where the corpus size is a free parameter** (staged token shards),
  `ddp-bench`'s `--train-tokens` snaps it to a multiple of
  `epoch_splits * batch_size`, so the pass divides and nothing is dropped.
- **Where it is fixed** (a downloaded image dataset), the tail is re-drawn
  each pass, so multi-pass training averages it away: across `E` passes a
  sample is missed `E * dropped / N` times, i.e. ~0.3 times over a 200-epoch
  CIFAR-10 run at batch 128. A single-pass run averages with nothing and is
  warned.

The real fix is a short final batch (`drop_last = false`). Three things block
it, in increasing order of difficulty:

1. **The worker cannot emit one.** `partition[start..start + batch_size]` is
   unclamped in both the prefetch submission loop and the sync path, and
   `num_batches` floors at five sites.
2. **A final batch of 1 breaks BatchNorm training** (variance over one
   sample), so any implementation needs a batch-of-1 guard for models
   carrying BN buffers.
3. **It would bias ElChe.** The scheduler derives its ratios from per-batch
   *delivered cost*, and a systematically cheaper final batch every epoch
   skews them — `epoch_splits` times per pass. This is the open question: a
   short batch would have to be excluded from the delivered-cost sample, or
   normalized by its actual pick count, before the tail could be trained
   without distorting the schedule that allocates it.
