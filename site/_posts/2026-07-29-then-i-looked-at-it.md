---
title: "Then I looked at it"
subtitle: "flodl 0.7.0 'one-level-down': the training monitor becomes a recursive portal, one view repeated at root, host and rank, live or saved as a single self-contained file. Building it caught five of this project's own numbers measuring the wrong thing. Plus a memory pass that takes gigabytes of transients out of the sync barrier, and the flagship sweep re-run, seeded."
date: 2026-07-29
description: "flodl 0.7.0 turns the training dashboard into a recursive portal: every level of a cluster run, root, host and rank, renders through one view, addressable by path over HTTP, persistable as JSONL, savable as one self-contained HTML file. Building it exposed five reported numbers that were measuring the wrong thing, starting with a --seed that never reached model initialization. The flagship sweep was re-run seeded: ResNet-20 to 92.29% on CIFAR-10, 28% faster than the fast GPU alone, and every cluster mode 2 to 6% quicker than a week ago."
---

The [last post](/blog/then-the-cpu-died) ended on a promise: the floor is
rebuilt, from here it is incremental again. This is seven days later. 42
commits, 141 files, 20,789 lines added. The floor held.

0.6.0 built a cluster. It did not give you a way to look *into* it. The dashboard
showed one epoch feed for the whole run plus a tab per GPU, which is the right
shape for one box and the wrong shape for a cohort: a rank's loss, a host's
throughput, and the cohort's roll-up are three different questions, and there
was one answer. So the flagship of 0.7.0 is the layer that answers all three
with the same view.

And then I looked at it, which turned out to be the more interesting half of
this release.

## The dashboard is a portal

Every level of a run renders identically, and the breadcrumb is the record path:

```
root                          the cohort: work-weighted roll-up of every host
root/flodl-pascal             one host: roll-up of the ranks on it
root/flodl-pascal/rank1       one rank: its own measurements, nothing averaged
```

Click a child to descend, a breadcrumb segment to come back, and each level is
a real URL (`#path=root/flodl-pascal/rank1`) that the browser's Back button
walks. Every level draws the same way: its own metrics plotted, its own
resources plotted, its children compared on one metric and clickable, and its
log listing epoch and sub-epoch rows interleaved on the one axis the two
cadences share.

The property that makes it a portal rather than a tree view is that the page
subscribes to **one level**. A query returns the node's own aggregate plus one
record per direct child, so watching `root` on a 300-rank run costs exactly what
watching `root` on a 3-rank run costs. Drilling in is the same call, re-scoped.
Alerts are the deliberate exception: a subscription pinned at `root` carries
rank losses, drift and dropped control frames from the whole subtree, because you
should not have to be looking at the right level to learn a rank died.

Two details were needed to make the levels honest rather than merely pretty. At
an interior node every metric is a roll-up, so the legend names the reduction it
used (`loss (mean)`, `throughput (sum)`); at a leaf it is a raw measurement and
the legend is the bare key. And cross-rank means are weighted by realized work,
with every node carrying its own weight, so a hierarchical roll-up equals the
flat one exactly. That last one is checkable and was checked against the real
3-rank stream: the root's work-weighted loss equals the roll-up through the host
tier equals the flat weighted mean over all ranks, to 1e-9, at every tick.

The same record plane comes out three other ways, all off by default:

- **Over HTTP.** `GET /paths`, `/node?path=`, `/history?path=&n=`, and
  `/stream?path=` (SSE). `/history` returns exactly what `/stream` would have
  sent, so "read the history, then subscribe" has no gap and no duplicate at
  the handover.
- **On disk.** `record_log(dir, max_bytes)` writes a tree where a record's path
  *is* its filesystem path (`root.log`, `root/<host>.log`,
  `root/<host>/rank0.log`), one standalone JSON object per line. `jq`-able as
  is, ingestible by fluentd or Cloud Logging as is, and each file is a
  drop-oldest ring so a long run cannot fill a disk.
- **As one file.** `save_dashboard(path)` writes the whole portal as a single
  self-contained HTML artifact. Every level browsable, deep links working, the
  model graph SVG and the hyperparameters inline. No server, no sibling files,
  nothing fetched, and it needs no dashboard port: persisting a dashboard should
  not require serving one.

That last one is not a demo. The dashboard embedded on the
[benchmark page](/benchmark) is a roughly 850 KB `dashboard.html` written at
teardown by the flagship cell of the published sweep, 200 epochs of ResNet-20
under DiLoCo across three GPUs on two hosts. Drill into the hosts and the ranks.
Every level is the run's real data.

Because the archive is meant to end up in a paper or a ticket, it also grew a
light theme and follows the reader's desktop by default, and
`dashboard_theme("light")` pins it so a figure does not change appearance with
the reviewer's OS. The reader's own toggle still wins, because pinning sets a
default rather than removing a control.

## Then I looked at it

An observability layer's first real job is to catch its own instruments lying.
This one earned its keep before it shipped. Five numbers this project was
reporting, several of them in the last post's benchmark report, turned out to be
measuring something other than what they claimed.

**`--seed` never reached model initialization.** It was plumbed into the dataset
config and consumed by the synthetic generators, and nothing called
`manual_seed` before the model was built, so weights came from an unseeded
generator on every path. Three runs of `--model olmo --batches 1 --seed 42`
returned 11.0448, 10.9921 and 10.9241. That spread is wider than the
differences the benchmark was being used to measure: an eager-versus-graph
comparison showed 0.084 between the two arms against 0.12 *within* one arm. So
every single-seed differential in the previous report was noise-limited, and the
whole sweep has been re-run seeded. Initialization is now bit-reproducible
(`olmo` at 11.006310 three times). Training still is not, and two layers below
this remain nondeterministic by nature: GPU kernels drift, and the cadence and
async modes allocate work from delivered cost, which is coupled to the wall
clock, so those modes cannot be bit-reproducible however they are seeded.

**Resource readings were usually taken while the GPU was idle.** A resource
point on the dashboard was whichever roughly-500ms sample happened to be latest
when the report was published, and publication rides the reduce boundary, which
is precisely when a rank is sitting in a sync. A GPU busy for most of an interval
rendered as quiet. Points are now the min/mean/max envelope of every sample
taken since the previous report, with the reduction chosen per meaning: a
parent's utilization peak takes the max over children, because averaging peaks
makes one pegged rank beside three idle ones read quarter-busy, while VRAM peaks
sum, because sum-of-peaks is the conservative answer to "how much VRAM did the
cohort need available". On a 20-epoch cluster run, 195 of 306 GPU-carrying
records show a peak more than a point above the mean, and the early spread
(mean 19.3 against peak 64.1) narrows late. That narrowing is itself the signal
the single sample was hiding.

**`data_starve` was counting parameter-plane work as data starvation.** The
prefetch wait loop drains control messages while it is blocked, and control
handling can run a full parameter snapshot, an averaged-model writeback, or a
collective. All of it was timed as waiting for data, which is how a slow rank in
a CPU-averaging cohort read as I/O-starved for about 58s per epoch while it was
in fact working the parameter plane. Control time is now subtracted from the
diagnostic. The scheduler's own ledger, which prices the full wall on purpose,
is unchanged.

**Two of the three ranks had no data prefetch at all, for entire runs.** The DDP
prefetch governor sizes itself from a memory budget, and a budget below one
batch means depth zero, which routes every batch through the synchronous path. On
a GTX 1060 running OLMo-150M the probe reported 5592MB used against a 5459MB
cap, so the budget saturated to zero on every chunk and both Pascal ranks ran
fully synchronous while the 16GB rank beside them kept a depth of hundreds. Two
independent causes, both fixed: the activation reserve was being subtracted on
top of a `used` figure that already included it (the caching allocator frees
activations to its own cache, not to the driver, so they stay counted), and a
budget too small for one batch now stops disabling the feed outright, because a
two-batch flow window is a rate-matcher rather than a capacity claim and is
priced against physical free memory instead (32KB against 473MB free, in that
case).

This one deserves its honest ending: **it is not a throughput win.** Wall time
moved 420.7s to 419.8s and per-rank utilization did not move. A blocking
host-to-device copy from pageable memory cannot begin until previously queued
kernels on that stream drain, so much of what was reported as data starvation
was the previous step's GPU work waiting under another name. What the fix buys
is that a rank's compute-versus-data split now means what it says, and that a
silent, un-overridable degradation is gone. Treat `data_starve` on a rank
without a prefetch channel as an upper bound.

**Progressive-dispatch timings reported the largest chunk as the epoch total.**
The coordinator folded per-chunk durations with `max()`, on the assumption they
were cumulative from the epoch start. The worker's per-chunk state made every
duration per-chunk instead. So the dashboard reported one chunk, and per-rank
throughput divided epoch-summed samples by one chunk's time. The fold now sums.

None of these five were found by staring at the code. They were found by
plotting what the code said and noticing the plot was wrong.

## One more, from a user

[Issue #32](https://github.com/flodl-labs/flodl/issues/32) reported that graph
`switch` failed with `router selected branch 24 but only 2 branches exist`. Two
defects: the selector took an argmax over a flattened `[Batch, NumBranches]`
projection, so the "branch index" scaled with batch size, and `switch` routed
the whole batch to one branch anyway.

The obvious minimal fix, reducing the batch to a single index, is a dead end,
and it is worth saying why, because it generalizes. Averaging the branch logits
over a batch is the same as computing logits from the average sample, and the
average sample barely moves from batch to batch, so a batch-pooled selector
converges to a fixed one as the batch grows, and a router trained batched would
not transfer to single-sample inference. (For a linear projection this is exact
rather than approximate: `mean_i(W·x_i + c) = W·x̄ + c`, and `x̄` tends to
`E[x]`. Mean-of-softmax and majority vote collapse the same way.)

So `switch` now dispatches per sample: rows are grouped by the branch they
chose, only the branches that received rows run, and outputs are restored to the
original row order differentiably. Whole-batch switching stays available where
the batch genuinely is the decision unit, and it is still the cheaper path.

Every prior `switch` test used batch size 1, which is why this survived to a bug
report, so the graph control paths gained batched coverage across the board. That
sweep found no further defects. It also produced a project invariant: a control
decision derived from a batched activation must state its batch semantics, and
per-sample decisions must dispatch per sample. Related, and found in the same
pass: named-ref accumulation was summing in `HashMap` order, which Rust's
per-process hasher randomization makes differ between runs. Float addition is not
associative, so a graph with two refs into one node was not bit-reproducible at a
fixed seed. Harmless for soft routing weights. It can flip a hard-routing branch
outright.

## The memory pass

The other half of the release is a memory pass on the CPU averaging plane, driven
by trying to train OLMo-150M (190M parameters once the padded, untied head is
counted) on 6GB cards and a 10GB VM. Every one of these was a model-sized
transient landing on the sync barrier, which is exactly where every rank spikes
at once:

- The rank's reduce round-trip no longer materializes wire transients in either
  direction. The contribution streams from parameter staging to the socket one
  tensor at a time, with the realized-work scale fused into the encode at byte
  level, and the reply parses straight off the socket into its tensors. The send
  side went from about four model copies to one tensor's worth of wire bytes.
  About 2GB of per-rank transients gone at OLMo-190M.
- The averaged-model writeback became a true async copy. It was fully async on
  paper and defeated by its input: the reply decoded into fresh pageable memory
  every window, and an async copy from pageable memory silently degrades to a
  synchronous bounce. Decoding into reused pinned staging removes a 762MB
  per-window allocation and makes the overlap real.
- Formation no longer builds throwaway model copies. Aligning initial state
  rides a zero-weight reduce; both sides used to materialize a full model to do
  it, one of them purely to serialize it to zero bytes. The root now streams its
  live tensors and the others stream literal zeros from the shapes, byte-identical
  to before.
- The consensus scatter stopped cloning a model-sized buffer per connection
  (381MB per connection per window on a bf16 190M wire, pure churn), the relay
  folds each contribution as it arrives instead of holding every frame to the
  barrier (about 1GB per host at OLMo-190M), and the elastic blend stages one
  tensor at a time instead of a whole averaged model (762MB of VRAM).
- Attention routes through fused scaled-dot-product attention, so the two
  `[batch, heads, seq, seq]` tensors the explicit chain held through backward
  are never materialized. About 300MB of VRAM back at sequence 256 batch 4, and
  the memory-efficient backend covers sm50 and up, so Pascal is in.

Measured end to end on a 10GB two-rank VM at OLMo-190M: the worst-case
per-rank memory headroom floor went from 26MB to 341MB, with wall time down 15%.
26MB of headroom is not a configuration, it is a run that dies on the next
allocation.

There is also a scheduling fix with a similar flavor. Cadence-window growth was
gated behind five calibrations, so a cluster run spent its first six windows at
the minimum window size no matter how lopsided the sync-to-compute ratio was. On
the 3-rank OLMo rig that meant six reduces of about 15s each protecting about 4s
of compute apiece: 78% measured overhead, held for a quarter of the run, before
the controller was allowed to react to a signal that was 15 times its target
from the first window. Growth now starts at the second calibration under two
disciplines, one of which is that an early proposal must clear the overhead
target by the step-cap factor, which makes early action possible exactly where
the proposal is clamped by the cap regardless of measurement noise. Borderline
pressure near the amortization knee, where jitter genuinely could flip the
decision, keeps the full five-calibration patience it always had. Anchor
election, which is the thing cold-start skew actually endangers, is untouched.

## Two bugs worth naming

**A monitor poll could crash a rank during cluster formation.** This presented
as four rank-0 SIGSEGVs in the first 15 minutes of the overnight sweep and zero
in the 1h45m after, and as one "non-reproducible startup race" a week earlier.
Root-caused from a captured core: the crashing thread was in the allocator's
stats read, the main thread was mid-`ioctl` inside libtorch's lazy CUDA init.
The GPU pollers gated their read on a CUDA primary context existing, and that
gate is unsound during formation, because NCCL creates the primary context in
the middle of its own init while libtorch's allocator only initializes at the
process's first torch CUDA op moments later. Worse, torch's init publishes the
per-device table's size before constructing its entries, and each constructor
makes driver calls: about a millisecond warm, tens of milliseconds on a cold
driver, which is why the crashes clustered at the start of the night. A poller
tick landing in that window loaded a null allocator. Every allocator-stats entry
point is now guarded by a passive context pre-check (so processes that must stay
CUDA-free, like the cluster launcher, still trigger nothing) followed by the lazy
init that serializes with torch's own once-guard. Validated with the same
amplified repro that captured the core, poll interval 100ms to 1ms: unfixed
crashes by formation 7, fixed survives 30 of 30.

**A mid-epoch schedule change could deadlock a cohort at the epoch tail.** The
epoch-tail allocation pins the final window's per-rank split once, and the firing
gate excludes ranks that can take no more steps this epoch. Both are correct
alone. They are wrong together the moment the batch counts change between the
pin and the fire, which is exactly what a mid-epoch growth commit, a nudge, or
an election does: the gate then demands steps from a rank whose pinned slot is
zero, that rank can never be dispatched again, and its quiesced exclusion never
engages because the pool still holds a peer's unreachable crumb. The whole cohort
freezes. Reproduced at 4 in 30 once the warmup-growth change made mid-epoch
commits routine, though the race is latent at every heterogeneous epoch tail and
predates it. Two-sided cause, two-sided fix: a rank whose final slot is zero now
counts as quiesced, and the pinned plan carries the counts it was computed from
and re-pins from the live remainder whenever they diverge. Two deterministic
regression tests plus the smoke looped 100 times; at the observed 13% rate, an
unfixed gate surviving that has a probability around 1e-6.

The stall-watch dump now prints the pinned plan and each rank's true gate
verdict. This deadlock was diagnosed by inferring missing state from dispatch
timing. The next one gets stated.

## The numbers

The published sweep was re-run in full, seeded, on the same rig and the same
model code: 8 models, 63 runs, an RTX 5060 Ti on one host and two GTX 1060s in a
VM on the other, coordinated over TCP, the slowest card behind a PCIe x1 riser.
The flagship is 200 epochs of ResNet-20 on CIFAR-10, built through the graph
builder. Same epochs, same seed, and this time the seed actually applies.

| Mode | 0.6.0 | 0.7.0 | Change |
|------|-------|-------|--------|
| nccl-cadence | 547.7s | 512.2s | -6.5% |
| cpu-async | 529.4s | 497.9s | -6.0% |
| cpu-async-diloco | 524.1s | 499.7s | -4.7% |
| nccl-sync | 1172.6s | 1137.2s | -3.0% |
| cpu-sync | 1650.2s | 1607.3s | -2.6% |
| cpu-cadence | 511.8s | 503.0s | -1.7% |
| solo-0 (5060 Ti alone) | 698.7s | 696.1s | -0.4% |

The last row is the interesting one. This release barely touches the
single-process path on this model, so solo-0 is the closest thing the sweep has
to a control arm, and it moved 0.4%. Every cluster mode moved several times
that. The gains are engine-side, on the sync and wire and writeback paths, which
is where the work went.

DiLoCo over asynchronous CPU averaging still takes the accuracy crown, now as a
seeded number: **92.29% eval, against the 91.25% published reference and the
91.46% the fast GPU reaches alone, finishing 28% sooner than that GPU.** It gets
there while stopping at twice the training loss of the solo run, which is the
same story as last time: each replica only sees its own data partition, so
replica-private memorization is averaged away every round while shared structure
survives. Distributed training as built-in early stopping.

One caveat about the report, since it is a measurement change rather than a
result. The GPU columns now name every physical device on every host
(`exa-cuda:GPU0`, `flodl-pascal:GPU0`, `flodl-pascal:GPU1`), sourced from the
ranks' own samples, where 0.6.0 could only see the controller box and printed
`-` for the two Pascal cards. That is a real gain in coverage. The trade is that
a dedicated controller now correctly declines to poll GPUs it does not train on,
and idle-gap detection needs that dense local poll, so on this topology the idle
analysis has no data source for cluster runs at all. (Which caught one more
instrument lying, while writing this post: the Idle column was summing an empty
list and printing `0.0`, a measured zero for the one thing nothing had measured,
and it now prints `-` while both idle sections state their coverage. The
preamble spells out the rest, including why a dense figure and a `~` figure are
not interchangeable if you compare against the older report.)

Two cells are missing on purpose: the single-Pascal 200-epoch baselines were
deferred, because each one occupies the rig for close to an hour to reproduce a
number the previous sweep already has (that card alone needs 55 minutes for what
the cohort does in 8).

The rest of the sweep, per-epoch trajectories, per-rank schedules, VRAM, sync
cadence and calibration, is in
[docs/ddp-benchmark.md](https://github.com/flodl-labs/flodl/blob/main/docs/ddp-benchmark.md),
with the summary on the [benchmark page](/ddp-benchmark).

## Also in this release

Transformer pieces, because the OLMo work needed them: `RotaryEmbedding` with a
`MultiheadAttention` rotary seam, `SwiGLU` as a first-class layer, and
`TokenShards` for pre-tokenized corpora in NumPy shards or raw dumps. A bf16
wire format for the CPU averaging plane that halves the payload at every hop
while every accumulator stays f32. Cluster ranks got the two-stage reader ring
the solo loader already had, which matters most on the plainest configuration:
sync-mode cluster runs previously had no read-ahead of any kind, and clusters are
precisely where remote storage lives. An
[Apple Silicon path](https://github.com/flodl-labs/flodl/blob/main/docs/mac-apple-silicon.md)
documented end to end, thanks to [@newQuery](https://github.com/newQuery). And a small one that had been quietly
costing time: a new CLI flag now shows up in `fdl <cmd> -h` on its own, because
the help schema cache was invalidated against the command's config file and not
against the sources the schema is actually compiled from.

The full list, at the depth this post does not go to, is in the
[changelog](https://github.com/flodl-labs/flodl/blob/main/CHANGELOG.md). To run
any of it: the [monitor guide](/guide/monitor) covers the portal, the archive,
and the path API, the [distributed reference](/guide/ddp-reference) covers one
GPU to a cluster, and [`fdl setup`](/guide/cli) gets you from clone to training.

This time the instruments contributed the roadmap.
