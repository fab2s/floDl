# Data Cascade: Raw-Chunk Residency and Augmentation-as-Picks

A design for the layered data plane (VRAM pool → RAM sample cache → disk stage
→ shared source) that keeps its retention correct under per-epoch reshuffle and
on-the-fly augmentation. It resolves three audit findings (see
[Findings resolved](#findings-resolved)) not by disabling the caches but by
fixing what they hold and when they evict.

**Status:** design. The residency tiers, the reservation/ChunkPool schedule,
the next-use flow window, and the cross-epoch advisory all ship today; this note
defines the contract and the eviction/augmentation model they should share. The
hard-boundary over-allocation half is already designed and partly implemented in
[Epoch-Tail Allocation](epoch-tail-allocation.md) and is referenced, not
re-derived, here.

---

## The core distinction

Two units, deliberately separate:

- **Residency unit — the raw chunk.** A sample identified by its stable id,
  resident as high in the cascade as it fits (VRAM → RAM → disk → source),
  sized to what fits. Content is a pure function of the chunk id.
- **Schedule unit — the pick.** One entry in the epoch's permutation, consumed
  as one unit of ElChe realized work. A pick names a chunk id; several picks may
  name the *same* chunk id (that is augmentation, below).

The residency tiers key by **chunk id** (deduped across repeats). The schedule
and the realized-work algebra key by **pick** (repeats counted). Keeping these
distinct is what makes the rest fall out.

## Contract: `get()` returns the raw chunk; it never augments

The VRAM tier already assumes this in its own words — it captures *"raw
pre-augmentation samples"* device-to-device out of each uploaded batch
(`vram_pool.rs`, capture-at-delivery). But the `DataSet::get()` contract does
not require it: today it promises only that each position has a consistent
meaning and shapes are consistent across calls. That gap is the whole bug — a
dataset that augments inside `get()` (the PyTorch `__getitem__` convention) has
its first-seen realization captured and served frozen on every later epoch.

The contract must state, and debug builds should check, that **`get(id)` is a
pure function of `id`**: it returns the raw chunk, and applies no per-call
randomness. Augmentation does not belong here.

## Augmentation is repeated pick indices, not per-call randomness

Augmentation is expressed as **multiplicity in the permutation**: chunk id `X`
appears at several pick positions, each realizing different bytes via a
deterministic transform keyed by the pick. Consequences:

- **ElChe stays augmentation-blind.** A pick is a pick. The realized-work
  constant is the augmented permutation length — deterministic, seed-computable
  arbitrarily ahead, partitioned by ratio exactly as today. `realized_work`
  (`gamma_mass` / `mover_mass` / `is_realized`) needs no change and never learns
  that some picks share a chunk.
- **Residency composes.** The raw chunk is fetched and resident once; each
  augmented pick derives its variant from the pick, on the resident bytes.
- **Determinism is required, not incidental.** The transform must be a pure
  function of a key like `hash(chunk_id, repeat_instance, epoch_seed)` — a
  stateless / functional RNG, not global mutable RNG. This is what keeps the
  augmented pick-stream seed-computable ahead, which the whole prefetch model
  depends on (the data plane is epoch-blind, computable from seed arbitrarily
  ahead). Deterministic seeding is statistically equivalent to stochastic
  augmentation when the key is well-distributed, and strictly better for
  reproducibility.
- **Device-expressible.** For a VRAM-resident chunk the transform must run as
  tensor/graph ops on device; otherwise the row is pulled back to host,
  transformed, and re-uploaded, defeating residency. This suits a GPU-first
  framework and is faster regardless. Augmentation as graph nodes (`FlowBuilder`
  `.map`) is the natural home; a collate/transform stage
  ([training-roadmap](training-roadmap.md), "Samplers and variable-length
  collate") is the other.

## Reshuffle is a global re-partition (already built)

Each epoch draws a fresh permutation and `ChunkPool` re-partitions it into fresh
per-rank spans (`chunk_pool.rs`, `new(epoch, …)`). This is the statistically
correct choice — every sample can land on any rank, full shuffle entropy — and
matches PyTorch's `DistributedSampler` with `set_epoch`. Residency is therefore
**opportunistic across epochs, not guaranteed**: a rank's resident set from one
epoch overlaps its next-epoch picks only partially. That partial overlap is
worth keeping, and that is exactly what a smart eviction *order* recovers.

## Eviction: next-use order, across all tiers, keyed by remaining picks

The flow window already does the right thing: `StreamPool` evicts the entry
whose next use is farthest, declines admission when everything held is needed
sooner (the caller pauses rather than fetch past a full window), and re-keys
held entries against each fresh advisory (`refresh_positions`), so entries
absent from the next stream go first. The cross-epoch advisory already carries
the next epoch's picks (*"segments walking into the next epoch's permutation"*).

The gap: the **persistent tiers (RAM sample cache, VRAM pool) never evict** —
they are fill-until-full, remove-nothing. That policy is only valid for a
*stable draw-set* (the same N reshuffled each epoch, i.e. solo mode). Under the
global re-partition above, a rank's assigned set changes every epoch, so the
persistent tiers must evict `(previous_resident − next_picks)` to admit
`(next_picks − resident)`, in the same next-use order the flow window uses. The
information to drive it already exists in the advisory; it simply is not wired
into the persistent tiers.

### One metric: distance to next use (Belady's MIN, made realizable)

Two situations look distinct but are the *same*: a chunk picked twice within an
epoch (augmentation, e.g. positions 5 and 900), and a chunk that survives into
the next reshuffled partition (cross-epoch reappearance). Both reduce to one
question — **what is this chunk's nearest upcoming pick in the forward
horizon?** — where the horizon spans the epoch boundary (the advisory already
walks into the next epoch's permutation). The boundary is irrelevant to the
metric; there is one ordered delete queue, not two mechanisms.

The eviction key is the chunk's **next-use** = its nearest upcoming pick.
Under pressure, evict the chunk whose next-use is *largest*; a chunk absent from
the horizon has next-use ∞ and goes first. The key is recomputed as picks are
consumed and on each advisory refresh: a chunk picked at {5, 900} has next-use 5
before position 5 (needed soon, keep) and next-use 900 after (now a prime
victim). Naive pop-on-hit is just the multiplicity-one special case.

This is **Belady's optimal replacement (MIN)**: for a known access sequence,
evicting the farthest-next-use item minimizes misses (exchange argument: swap any
other eviction for the farthest-next-use one without increasing misses). MIN is
normally unrealizable because the future is unknown — here the pick stream is
deterministic and seed-computable arbitrarily ahead, so the optimal policy is
directly implementable. `StreamPool` already is this (`offer` evicts
`max_by_key(next_use)`; `refresh_positions` re-keys absent entries to
`usize::MAX`); the work is running the persistent tiers on the same queue.

Two caveats keep it honest:

- **In a cascade the action is demote by default; delete only on *provable*
  terminality.** The victim is still chosen by Belady next-use, but what happens
  to it turns on a distinction that is easy to get wrong: the global pick stream
  is seed-deterministic, and per-*rank* assignment is *forecastable but not
  certain*. Two bounded sources of uncertainty, not a pervasive one:
  - **The boundary tail.** ElChe ratios are stable once calibrated (they drift
    slowly, they do not jump), so the ratios measured during epoch k set epoch
    k+1's spans up to the next barrier with good accuracy. The fuzz is
    concentrated in the last chunks near a hard boundary — the same tail variance
    the [over-allocation](#hard-boundaries-over-allocate-staging-see-epoch-tail-allocation)
    already absorbs — not across the whole next partition.
  - **Rank death.** A dead rank's *unconsumed* span is returned to the global
    queue and served to survivors as extra work (forfeit → reclaim), extending
    them "as if training were longer." Existing spans are untouched — death
    *appends*, it does not rewrite assignments — so it changes training order but
    keeps every survivor's existing forecast valid. By non-overlap a survivor
    receives the dead rank's *different* chunks, never re-sees its own consumed
    ones (so within-epoch terminality stays solid even under failure). How the
    dead span is split (whole tail to the last partition vs. distributed to
    preserve distributed-FIFO) is a locality nicety only — the global shuffle
    stays statistically correct under any redistribution order.

  So "this chunk never returns *to this rank*" is a confident forecast for the
  bulk of the next partition, uncertain only in the tail and under death-append —
  both bounded and statistically benign.

  Therefore **demote (VRAM→RAM→disk) is the default eviction action.** It
  amortizes that residual uncertainty in both directions: a cheap tier-local move
  if the chunk never comes back, and a cheap hit if it returns unpredictably — e.g. a
  chunk kept demoted from a prior epoch that this epoch falls in a dying rank's
  span and is reassigned here, a return no forecast would have kept. Demote is
  also self-cleaning: a demoted chunk whose next-use never materializes becomes
  the farthest-next-use victim at the lower tier and drops, so per-tier Belady
  auto-corrects and RAM does not fill with hopeful chunks. The single next-use
  ordering applies at each tier boundary independently.

  **Delete** is the specialization, gated on *provable* in-partition
  terminality: a chunk past all its picks in a fixed partition with no further
  epochs — end-of-training's last window, and single-epoch / streaming
  pretraining throughout (the only epoch is the last; spans are fixed; even under
  failure a survivor steals a dead rank's *different* chunks, never re-sees its
  own consumed ones). That is exactly the regime where delete pays most: huge
  data, nothing fits, so demote would otherwise churn bytes for data that
  provably never returns. Start with demote everywhere; add delete-on-terminal
  for these provable cases.

  This whole decision is pure data-plane residency: invisible to ElChe and the
  realized-work algebra, consistent with keeping augmentation and residency out
  of the scheduler's view.
- **Optimality assumes uniform item size.** Exact for fixed-resolution samples;
  for variable-size chunks (variable-length sequences, mixed resolution — the
  same datasets behind the view-sizing finding) it degrades to a heuristic
  (far-but-large vs several near-but-small is a knapsack choice). Farthest-next-
  use remains a sound default, just not provably optimal.

## Hard boundaries: over-allocate staging (see Epoch-Tail Allocation)

Per-rank reservations are a *forecast* from ElChe ratios, so at a hard boundary
(epoch end; end-of-training, where no next epoch absorbs drift) ElChe's actual
last picks can vary from the forecast. To avoid last-loop data starvation the
staging layer **over-allocates**: it speculatively stages the last chunks to all
candidate ranks before the coordinator actually assigns them. Whichever rank
gets the allocation already has the bytes resident; the losers' redundant
staging is discarded for free. This is *repeated data* (same bytes, staged N
times, allocated once, no extra realized work) — the opposite of augmentation
(*different data*, N distinct picks, N units of realized work).

This is safe by construction: staging may overlap across ranks, allocation never
does, and the allocation ledger is the staging invalidation (no revocation
protocol). It is the tail-margin behavior already in `ChunkPool` (*"prefetches
everyone's tails last, as margin"*); its delivered-feed motivation and the
Cadence implementation are covered in
[Epoch-Tail Allocation](epoch-tail-allocation.md). The margin should be sized to
the forecast variance at the boundary, not a fixed large chunk.

## What ships vs what to build

Ships today: the residency tiers; the reservation/ChunkPool schedule with fresh
per-epoch re-partition; `StreamPool` next-use eviction on the flow window; the
cross-epoch advisory carrying next-epoch picks; the tail-margin over-allocation.

To build:

1. **`DataSet::get()` = raw, pure contract** — document it, add a debug-mode
   purity check, and provide the augmentation seam (on-device `.map` / collate).
2. **Augmentation as deterministic repeated picks** — pick-keyed stateless RNG;
   the permutation carries the multiplicity; ElChe and `realized_work` unchanged.
3. **Wire next-use eviction into the persistent tiers** — reuse the advisory the
   flow window already consumes; key eviction by remaining-picks-per-chunk.
4. **Pick-vs-chunk accounting** — the window/coverage math counts picks (so
   augmentation multiplicity is honored by `window ≤ epoch` and `batch_counts`),
   while residency budgets count distinct resident chunks.

## Findings resolved

The post-2026-07-02 data-plane audit surfaced three findings that are all
consequences of the two defects this design closes (persistent tiers cache
`get()` output rather than raw chunks, and never evict under re-partition):

- **Cache RAM budget ratchets toward ~100% of MemAvailable across epochs**
  (`loader.rs`) — the persistent tier accumulating the union across
  re-partitions because it never evicts. Closed by next-use eviction (item 3);
  the cap-math fix becomes a small independent hardening rather than the sole
  guard.
- **VRAM pool / sample cache freeze the first-observed sample forever**, silently
  defeating stochastic augmentation and diverging cached-vs-uncached
  (`prefetch.rs`, `sample_cache.rs`) — closed by the raw-chunk contract plus
  augmentation-as-picks (items 1, 2): the tiers hold raw bytes, augmentation is
  re-derived per pick.

The flow window (`StreamPool`) was already correct on both counts; the work is
bringing the persistent tiers under the same policy.
