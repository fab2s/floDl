# Training Capability Roadmap

Design sketches for the training-side capabilities that would make flodl
useful to a wider range of DL projects: memory levers for consumer
hardware, eval-driven run decisions, fine-tuning-era workflows, and data
ingestion without a shared mount. Companion to
[inference-roadmap.md](inference-roadmap.md), which owns generation loops,
export formats, and eval metrics for generative tasks.

**Status:** design sketch. No code deliverables in this doc; each section
is an independent work item with its own sizing.

**Sizing scale:** S = about a session of work, M = a few sessions,
L = a multi-session arc. Sizes describe scope, not schedule, and each
section names what actually dominates the effort.

---

## Why this exists

flodl's differentiator is heterogeneous, elastic distributed training.
The items here are chosen because they compound that differentiator
rather than chase feature parity: activation checkpointing, gradient
accumulation, and LoRA are what let mixed consumer rigs touch models
worth training; metrics-driven stopping is what makes long unattended
runs on such rigs trustworthy; streaming ingestion is what removes the
last shared-infrastructure assumption from the cluster story.

Guiding constraint throughout: every feature must hold on the ElChe
path, not just single-GPU. Each section has an explicit "ElChe
interaction" note; "none" is a valid answer but has to be argued.

---

## Activation checkpointing

**What:** trade compute for memory by discarding intermediate
activations during forward and recomputing them during backward. The
standard lever for training larger models on small-VRAM GPUs; PyTorch
ships it as `torch.utils.checkpoint` (Python only; libtorch has no C++
helper, so flodl implements the mechanism itself).

**Design.** Two tiers, built in order:

1. **Functional primitive.** `nn::checkpoint_segment(f, inputs)` runs
   `f` under `NoGradGuard`, keeps only the (detached) segment inputs,
   and returns outputs that participate in autograd via a splice: at
   backward time the segment re-runs with grad enabled from the stored
   inputs, backward runs inside the segment, and the input gradients
   are chained into the outer graph. Because flodl's backward already
   delegates to the C++ engine and then detaches, the splice can be
   orchestrated at the Rust layer: run the outer backward to the
   segment boundary, then re-forward + backward the segment, then
   continue. No custom C++ autograd node is required if segment
   boundaries are also backward boundaries.

2. **Graph-level auto-segmentation.** This is flodl's edge over the
   PyTorch UX: `FlowBuilder` knows the topology, so `Graph` can choose
   recompute segments automatically (the classic sqrt(N)-blocks
   heuristic over the node count, respecting fork/merge boundaries)
   instead of requiring per-call annotation. A `.checkpoint()` tag on
   FlowBuilder marks a subgraph; a `Graph::auto_checkpoint(budget)`
   picks segments to fit a VRAM budget. Frozen subtrees interact
   nicely: a frozen segment whose upstream is also frozen needs no
   recompute at all, which is exactly the shape frozen-component
   research workflows produce.

**Correctness hazards (this is where the effort goes):**

- **RNG:** dropout must see identical randomness in both forwards.
  Requires capturing and restoring the device RNG state around the
  segment (new FFI: generator state save/restore; libtorch exposes
  both).
- **Stateful buffers:** BatchNorm running stats must not update twice.
  The recompute forward runs with stats-update suppressed (eval-mode
  stats behavior with train-mode dropout is wrong, so this needs a
  dedicated "recompute pass" flag, not a blanket `eval()`).
- **CUDA Graphs:** capture/replay and re-forward do not compose in the
  first iteration; explicitly unsupported together until proven.

**ElChe interaction:** none by design, and that is the point of the
delivered-cost scheduler: recompute makes steps slower and lighter on
memory, ElChe measures delivered cost rather than modeling it, so
heterogeneous rigs where only the small-VRAM ranks enable checkpointing
rebalance automatically.

**Sizing:** M for the functional primitive (dominated by the
backward-splice design plus RNG/BatchNorm FFI and parity tests against
a non-checkpointed reference), then M for graph auto-segmentation on
top (dominated by segment-selection correctness across fork/merge/loop
constructs).

---

## Gradient accumulation in Trainer

**What:** process k micro-batches per optimizer step (loss scaled by
1/k, `zero_grad` + `step` every k), so effective batch size decouples
from VRAM. Users can hand-roll this in manual loops today; it belongs
in `TrainerConfig` because on the ElChe path the framework owns the
step loop and the accounting.

**Design.** `TrainerConfig.accumulation_steps: usize` (default 1),
threaded to the worker step loop. The accounting decisions are the
real content:

- **Coverage clock:** ElChe's schedule counts micro-batches. An epoch
  means one dataset pass; that is unchanged.
- **Realized-work mass:** frames keep weighting by micro-batches
  processed (samples seen), consistent with the realized-work reduce
  invariant: mass rides the frame, consensus divides by accepted mass.
- **Sync boundaries:** a reduce must never land mid-accumulation
  (gradients applied to params only at step time; averaging params
  with k-1 unapplied micro-grads silently drops them). The dispatcher
  rounds chunk allocations to multiples of k, and the worker defers
  `should_average` to the next step boundary as a backstop.
- **Interactions:** `max_grad_norm` clips the accumulated gradient
  once per step; `GradScaler` unscales once per step; schedulers tick
  per optimizer step, not per micro-batch.

**ElChe interaction:** as above; boundary alignment is the load-bearing
piece and gets its own tests (window edges, epoch tails where the
remaining allocation is not a multiple of k: final short step is legal,
its mass is its true micro-batch count).

**Sizing:** S for the mechanism, M total with the ElChe
boundary-alignment tests. Small code, care-dominated.

---

## Metrics module and eval-driven decisions

**What exists already:** `TrainerConfig` ships `eval_fn`,
`eval_dataset`, `eval_every`, `eval_result_fn`, `metrics_fn`; Monitor
ships `write_log` and `export_csv`. The gap is not eval plumbing, it is
(a) a metrics library so every project stops hand-rolling its
`eval_fn`, and (b) decisions driven by the eval signal.

**Design, metrics:** `flodl::metrics` with streaming accumulators:
`Accuracy`, `TopK(k)`, `Precision`/`Recall`/`F1` (micro/macro),
`ConfusionMatrix`, `MeanMetric` for plain averaged losses. Each is
`update(batch_output, batch_target)` + `merge(other)` + `compute()`.
The `merge` operation is the distributed story: rank-local accumulators
merge by summing counts, so cluster-mode eval reduces to shipping small
count vectors through the existing control channel rather than
averaging metric values (no mean-of-means bias on uneven shards). This
also gives the cluster final-eval item (launcher-side eval after
`TrainedState`) a clean payload. Generative metrics (perplexity, BLEU,
ROUGE) stay in the inference roadmap.

**Design, decisions:** two `TrainerConfig` knobs riding the existing
eval path:

- `early_stop: Option<EarlyStop>` with `metric`, `mode` (min/max),
  `patience`, `min_delta`. On trigger, the coordinator broadcasts a
  cooperative stop at the next sync boundary (the same graceful
  teardown the launcher shutdown path uses); ranks finish the window,
  final checkpoint saves, `join()` returns `TrainedState` marked
  stopped-early.
- `keep_best: Option<KeepBest>` with `metric`, `mode`: retain the
  best-scoring checkpoint alongside the periodic ones (atomic-rename
  discipline, `<stem>.best.fdl`), so a diverged tail never costs the
  run.

**ElChe interaction:** the stop signal must reach every rank at a
window boundary, not mid-window; it rides the coordinator control
channel that already sequences windows, so no new clock is introduced.

**Sizing:** S for the metrics module (mechanical, test-heavy in a good
way). M for early-stop/keep-best, dominated by the cluster stop-signal
path and its rig validation; the single-host case is trivial.

---

## LoRA adapters

**What:** low-rank adaptation for fine-tuning: freeze a base layer's
`W`, train `B·A` (rank r) added to its output, scaled `alpha/r`. The
consumer-hardware fine-tuning method, and the natural companion to the
HF loaders flodl-hf already ships.

**Design.** `nn::LoraLinear` wrapping an existing `Linear`: holds the
frozen base plus `a`/`b` parameters (`a` normal-init, `b` zero-init so
step 0 is an identity change); `parameters()` exposes only `a`/`b`;
`merge()`/`unmerge()` fold the product into `W` for inference-cost-free
deployment. Graph tree `freeze(path)` already provides the freeze
half; wrapping-in-place for graph nodes and for flodl-hf transformer
layers (q/k/v/out projections) is the integration surface. Checkpoint
side: adapter-only save/load rides the existing named/partial
checkpoint support and produces small artifacts; adapter export in the
HF `peft` naming layout belongs to flodl-hf's export module.

**ElChe interaction:** free win. The trainable set shrinks by orders
of magnitude, so reduce payloads shrink identically, sync cost
collapses, and the anchor auto-tune converges to tighter windows on
its own. Worth one benchmark cell to show the effect, since "cheap
sync changes the cadence landscape" is a claim the framework can
demonstrate empirically.

**Sizing:** S for the module and single-host training; M total with
graph/flodl-hf wrapping integration and peft-layout export.

---

## Samplers and variable-length collate

**What:** the sampler surface is Random/Sequential today. Two additions
unlock imbalanced-data and NLP workloads: `WeightedRandomSampler`
(per-sample weights, with/without replacement; the `multinomial`
tensor op already exists) and `BucketSampler` (group by length, batch
within buckets to minimize padding). Plus a padding collate helper that
produces `(padded_batch, attention_mask)` for token sequences, feeding
directly into the flodl-hf model inputs.

**ElChe interaction:** none; sampling happens inside a rank's allocated
chunk, and coverage accounting is unchanged. Distributed determinism
note: samplers must stay seedable per (seed, epoch, rank) exactly as
RandomSampler is today.

**Sizing:** S.

---

## Rank-sharded streaming ingestion

**What:** every cluster feature today assumes shared storage (each node
mounts the same logical paths). That is the single biggest deployment
constraint on the cloud path: cloud nodes have object storage, not a
common POSIX mount. The goal is ranks pulling their data directly over
the network with no shared filesystem.

**Design sketch.**

- **Shard manifest:** a dataset is a manifest (JSON) listing shard
  files with sample counts and checksums. Shards are plain files
  reachable by URL; the manifest is the unit the coordinator reasons
  about.
- **Source abstraction:** a `DataSource` trait (open a shard, read
  ranges) with `file://` and `https://` (range requests) backends
  first. S3 arrives via presigned URLs or an HTTP gateway initially,
  which sidesteps credential handling entirely; a native signed-S3
  backend is a later, separate decision.
- **Dispatch mapping:** ElChe's coordinator already dispatches dynamic
  chunk allocations per rank; chunks map to (shard, range) spans via
  the manifest's cumulative counts. The existing PrefetchWorker seam is
  where fetch-ahead lives: it is already a dedicated thread feeding a
  bounded channel, so a network-backed batch source slots behind the
  same interface.
- **Failure semantics:** fetch retry with backoff inside the prefetch
  worker; a shard that fails past the retry budget surfaces as the
  rank's error (elastic supervision already handles a dying rank), and
  checksums make corruption loud.

**Two-stage prefetch (storage → RAM → VRAM) — LANDED:** the streaming
prefetch pipeline runs two stages on CUDA targets: a reader thread
fetches batches from the dataset into a bounded pageable-RAM ring
while the transfer thread pins and copies to the device. The worker's
batch-throughput ceiling rises from `1/(t_read + t_transfer)` to
`1/max(t_read, t_transfer)` (the genuine overlap is between the
reader's I/O wait and the transfer stage's pin memcpy — pinning stays
transfer-side so the ring's currency is ordinary pageable RAM), and
the ring absorbs read jitter. Honest scope: the batch channel already
provides read-ahead, so this is a *ceiling raise* for read-bound
pipelines (network shares, slow disks), not a speedup for pipelines
that already keep up. Bounds are orthogonal by construction: the depth
governor bounds VRAM in-flight (transfer stage), the ring bounds RAM
in-flight (reader stage). The ring is sized per epoch from the host
RAM budget — `ram_max_usage` (default 0.50, `0.0` = off) caps total
system RAM usage, measured against `MemAvailable` so every other
process on the box is accounted for automatically. CPU-target loaders
stay single-stage (their batch channel *is* the read-ahead ring), as
does the coordinator-paced distributed `LoadBatch` path (no index
foresight, nothing to read ahead).

**Tiered data plane (the arc this seeds).** One rule unifies the
tiers: *staged bytes are reshuffle-invariant as long as staging is
keyed by sample identity, not epoch position.* A reshuffle changes
only the order function (the index stream, pure `seed + epoch`,
computable arbitrarily far ahead — to end of training), never the
content set. So each tier holds as much sample-keyed content as
genuinely reservable, permanently; only the order-dependent artifacts
(the in-flight stacked batches: reader ring + VRAM prefetch queue) are
rebuilt per epoch, and they are deliberately the smallest layer.
Resident mode is the existing embodiment of this rule at the VRAM tier
(preload everything, re-fetch never, re-permute per epoch); the
increments below extend it downward. Landing order; each increment
stands alone:

1. *Two-stage split + byte-budgeted ring* (landed, above).
2. *RAM sample cache* (**landed**): a read-through, sample-keyed cache
   at the `DataSet::get(index)` layer, inside the `DataSet` →
   `BatchDataSet` adapter. Batches are not reusable across epochs
   (reshuffle changes their composition); samples are. Admission is
   fill-until-full with no eviction: every epoch touches each sample
   exactly once in fresh random order, so a cache holding K of N
   samples hits K/N for ANY eviction policy — admit-until-full gets
   the same hit rate with zero churn. Shares the `ram_max_usage`
   budget with the reader ring (ring capped to a flow-buffer depth
   while the cache is active; budget refreshed per epoch against
   `MemAvailable`; shrinking headroom stops new admissions but never
   drops staged content). Lock-free (`OnceLock` slot per sample). Pure
   `BatchDataSet` implementors (opaque batching) are the explicit
   escape hatch and stay uncached; DDP rank workers drive their own
   `PrefetchWorker` without a `DataLoader` and get cache wiring with
   the reservation layer (increment 4), which owns per-rank budgets.
3. *Disk stage* (**landed**): a local-drive overflow tier under the
   RAM cache (`disk_stage(gb)`, `0` = off; `disk_stage_dir` override
   with a loud tmpfs warning). Nothing spills — the RAM tier never
   evicts — the disk tier admits what RAM *declined*, once, at first
   read: one append-only pack file (sequential append, lock-free
   positioned reads, offsets in set-once slots; per-tensor layout
   reuses the checkpoint codec) whose lookup cascades RAM → disk →
   source. Ephemeral (removed on loader drop): persistent cross-run
   staging needs dataset identity/invalidation and belongs to the
   host-scoped stage in increment 4. An earlier sketch here proposed
   Belady-clairvoyant eviction; the same K/N argument that settled the
   RAM tier retires it — under a uniformly reshuffled scan, WHICH
   samples sit in which tier does not change the hit profile, so
   clairvoyance only becomes real when access turns non-uniform
   (reservation-constrained windows, increment 4).
4. *Schedule-aware reservations (the distributed half; coordinator
   ledger **landed**, wire + stager pending):* the epoch permutation
   is partitioned into contiguous per-rank spans sized by ElChe
   throughput ratios (equal until calibrated), and each rank's chunks
   come from the front of its own span via a per-rank cursor — the
   shared arrival-order cursor is gone from progressive dispatch, so
   each rank's upcoming data is deterministic for the whole epoch.
   Drift is absorbed by **truing**: a rank that out-runs its span
   steals from the tail of the largest-residue span (the boundary
   moves, the books stay exact; donor consumes front-to-back, thief
   peels back-to-front, they can meet but never cross). Span tails are
   therefore the only owner-uncertain region, which is why the staging
   layer prefetches everyone's tails last, as margin — staging may
   overlap across ranks near boundaries, allocation never does, and
   staged-but-allocated-elsewhere bytes need **no invalidation
   protocol at all**: only allocated work executes, so the allocation
   ledger IS the invalidation, and unused staged data just ages out.
   Global reshuffle semantics are byte-identical (a reservation table
   is a deterministic partition of the shuffled order; the old cursor
   produced one nondeterministically). The data plane itself is
   epoch-blind: each rank's stream is the concatenation of its spans
   across epochs, computable from the seed arbitrarily far ahead —
   an epoch is where the order function switches, never a
   data-movement event. On that concatenated stream, next-use
   distances are known and non-uniform, so the streaming pool's
   eviction is next-use-priority (keep what recurs soonest) — the
   correctly-scoped return of the clairvoyance retired at the
   single-tier level. Dead ranks: the unconsumed span redistributes
   under the same truing rule. Keeping allocation inside staged data
   also keeps ElChe's delivered-cost signal clean: the data term goes
   uniformly cheap in steady state, so the scheduler measures true
   compute. The wire + stager half is **landed**: the coordinator
   emits per-rank `StageAdvisory` frames at progressive epoch start
   and re-emits them at every reduce boundary (reservation state
   changes ride the window clock — no timer of their own). Each frame
   carries the current schedule plus run-stream segments: this epoch's
   spans (own span first, window-sized margin tails last) and the
   predicted next epoch's (same ratio table over the next permutation,
   computable before its pool exists — margin-covered if ratios drift).
   Each rank's background stager walks the segments through the shared
   permutation into a sample-keyed tier the live prefetch path shares
   read-through; its budget refreshes with each advisory as the host's
   live RAM headroom split consumption-proportionally among co-hosted
   ranks (envelope-derived; `budget ∝ rate` = equal lookahead time).
   Dormant until the first advisory. Remaining slices:
   next-use-priority eviction for the beyond-budget stream, and the
   rig falsification.
5. *Partial VRAM sample tier* (candidate, after 4): the same rule one
   tier up — when a dataset almost fits on device, keep K of N samples
   VRAM-resident and stream the rest, completing the spectrum between
   resident and streaming modes. The batch-assembly machinery exists
   (resident mode's device-side gather); per-rank VRAM pools are what
   the controller's compute-ratio reservations would size, hence the
   sequencing after increment 4.

Cross-cutting invariants for the arc:

- *Reservation state changes ride the window boundary, never a
  separate timer* (single-step-clock rule, extended to the data
  plane); prefetch byte movement free-runs between boundaries, like
  the param-snapshot D2H clock.
- *Reserve-ahead is for data only.* Data buffers have exactly known
  sizes (`per_sample_bytes`); model memory (activations, lazy
  optimizer state) is unknowable before the first step, which is why
  the VRAM governor stays reactive (honest resize) while RAM/disk
  tiers plan ahead.
- *Staging is host-scoped, rings are rank-scoped.* Ranks on one host
  share the disk stage (deduplicated union of their reservations); RAM
  rings and VRAM governors stay per rank process.
- *Per-rank memory shares are consumption-proportional.* Splitting a
  host's RAM/disk budget by ElChe throughput ratios gives every rank
  the same seconds of lookahead (bytes_i / rate_i constant) — equal
  time, not equal bytes. This matters acutely on hosts whose combined
  VRAM exceeds host RAM, where pinned in-flight buffers alone are a
  serious RAM draw; RAM there is a rate-matching flow buffer, not a
  capacity tier (capacity lives on disk). The split belongs to the
  wiring layer (the `Graph::set_data_loader` auto-wiring seam), since
  a loader cannot see its siblings at sizing time.
- *Watch-point:* transient allocation oscillation from cache state
  (cold cache after reshuffle, network hiccup) feeding the
  delivered-cost signal — expected self-stabilizing, to be measured on
  the rig, not assumed.

**ElChe interaction:** this is the feature ElChe was shaped for. Fetch
latency lands in per-rank delivered cost (compute + data + transport
is already the scheduling signal), so a rank on a slow link gets a
smaller allocation automatically; no new tuning knob.

**Sizing:** L. Dominated by manifest/shard format design, failure-path
semantics, and rig validation over a genuinely slow link (the Pascal
VM's constrained PCIe/network topology is a realistic bed). Sequenced
after the current cluster-scale work stabilizes; the source trait and
manifest format can land first and independently.

---

## Structured metrics export

**What:** the monitor's HTML dashboard is self-contained by design, and
`write_log`/`export_csv` already exist for post-hoc export. The gap is
a streaming, machine-readable feed that external tooling can tail
while a run is live.

**Design:** `Monitor::stream_jsonl(path)`: append one JSON object per
epoch record (schema documented and stable: epoch, duration, metrics
map, hardware snapshot), written at `log()` time with the same
append-discipline the training log uses. A TensorBoard event-file
writer is deliberately deferred: it is a binary protobuf format,
hand-rolling it is real work, and a ten-line external script converts
JSONL to anything.

**ElChe interaction:** none new; in cluster mode the controller-hosted
monitor is the single writer, which the dashboard sink already
establishes.

**Sizing:** S.

---

## KV-cache attention (enabler, owned elsewhere)

Decoder-only models, the generation loop, and sampling strategies are
[inference-roadmap.md](inference-roadmap.md) territory. The one
nn-level enabler worth naming here so it is not designed twice:
`MultiheadAttention` needs a decode path that accepts cached K/V from
previous positions and appends the new position (`forward_decode(q_new,
cache) -> (out, cache)`), because `forward_ext` recomputes full-sequence
projections today. Sizing S-M, to be built when the generation loop
lands, not before.

---

## Sequencing view

| Item | Size | Depends on | Thesis served |
| --- | --- | --- | --- |
| Metrics module | S | nothing | every project |
| Samplers + collate | S | nothing | NLP/imbalanced data |
| JSONL metrics stream | S | nothing | tooling interop |
| Gradient accumulation | S-M | nothing | consumer hardware |
| Early stop / keep best | M | metrics module | unattended runs |
| LoRA adapters | S-M | nothing (export half: flodl-hf) | fine-tuning era |
| Activation checkpointing | M+M | nothing | consumer hardware |
| KV-cache attention | S-M | generation loop (inference roadmap) | LLM-era models |
| Streaming ingestion | L | cluster-scale work stabilized | cloud without shared mounts |

The three S items are independent and each removes a reason not to
adopt flodl; accumulation and LoRA together with checkpointing form the
"real models on mixed consumer rigs" story that compounds ElChe; the
streaming arc is the strategic one and the only L.
