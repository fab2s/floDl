# Cloud DDP - Communication-Efficient Distributed Training

The next DDP iteration targets **cloud and cross-datacenter** training,
where network latency (not GPU compute) dominates wall time, hardware
is heterogeneous across workers, and any single node can disappear
mid-epoch. Current flodl DDP is a single-host story: NCCL AllReduce
over PCIe / NVLink finishes in microseconds and `overhead_target=0.10`
is trivial. On a cross-region link, one AllReduce can cost *seconds*,
so sync-every-step becomes impractical and cadence-every-10-batches
still wastes time. `max_batch_diff` and `max_overshoot` are pairwise
guards that don't generalize to N>2 ranks, and a straggler or failed
node has no defined recovery path.

The design closes these gaps along two axes. A **communication
unlock** (outer optimizer on pseudo-gradients) widens how many local
steps each worker can take between sync rounds, cutting the number of
AllReduces per epoch by up to an order of magnitude. A **scaling
unlock** (meta-step rendezvous) replaces pairwise guards and
single-device anchor election with a participation-closed superstep
plus a predictive scheduler, addressing both heterogeneity and fault
handling. Together they make cloud DDP viable at N>2 heterogeneous
nodes with bounded convergence and graceful failure.

## Outer optimizer on pseudo-gradients

flodl's current `ElChe` cadence averages parameters directly (Local SGD
/ FedAvg semantics). This produces the implicit-regularization boost
already observed empirically (nccl-cadence reaches higher test accuracy
than solo on ResNet-20 CIFAR-10 at the same seed) but leaves the
communication-efficiency half on the table.

[DiLoCo (DeepMind, 2023)](https://arxiv.org/abs/2311.08105) closes that
gap with a two-level optimizer: inner AdamW for H local steps per
worker, outer Nesterov momentum running on the **parameter-space drift**
between sync rounds. The pseudo-gradient (`Δ_k = θ_global - θ_k`) is
treated as a gradient by the outer optimizer; its momentum buffer
smooths the consensus direction across rounds. This unlocks H up to
~500 without divergence, an order of magnitude wider than plain
parameter averaging tolerates.

Every round saved is one AllReduce not performed on the internet.

### API

```rust
Trainer::builder(...)
    .policy(ApplyPolicy::Cadence)
    .outer_optimizer(|| NesterovMomentum::new(lr = 0.7, mu = 0.9))
    .max_anchor(500)                    // H, widen aggressively
    .run()?;
```

Mechanically, the outer optimizer hooks in at the coordinator, between
the AllReduce (or CPU snapshot average) and the parameter broadcast
back to workers. flodl already snapshots parameters at averaging events
(CPU backend) and computes cross-worker deltas for divergence
monitoring, so the outer-optimizer state buffer is a small addition on
top.

### Design targets

> The concrete, current design for this increment (refined trait, the
> two-tier worker coupling, and the checkpoint integration now that the
> consensus checkpoint path exists) lives in
> [`epoch-tail-allocation.md`](epoch-tail-allocation.md). The sketch below is
> the originating vision; the trait was since collapsed to the
> averaged-consensus form so it composes with the relay sum-and-count.

- **`OuterOptimizer` trait**: `outer_step(&mut self, prev_global, work_weighted_consensus) -> new_global`.
  Consumes the consensus the reduce already produces (not per-worker deltas);
  the outer gradient `prev_global - consensus` equals `mean_k(prev_global - theta_k)`
  under work-weighting, so no per-rank state is needed at the controller.
- **Built-ins**: `NesterovMomentum` (DiLoCo default), `SlowMomentum`
  (SlowMo), `OuterSgd` (momentum=0 ablation), `OuterAdam` (sanity-check).
- **Stateless variant**: `OuterAvg` exactly replicates today's
  weighted-AllReduce behavior so existing code is unchanged when no
  outer optimizer is set (the default).
- **Two tiers**: the outer step is coordinator-side, but DiLoCo also needs a
  worker-side inner policy (full overwrite + `Optimizer::reset_state()` each
  round) for its disposable-inner / faithful-resume property. SlowMo and
  OuterAvg keep the inner loop as today. See the linked doc.
- **Checkpoints**: the outer-optimizer momentum is model-sized and persists as
  a `<stem>.outer.fdl` consensus artifact through the forge path; inner state
  stays disposable.

## Meta-step rendezvous

At 2 GPUs, `max_batch_diff` (Cadence) and `max_overshoot` (Async) are
pairwise guards and the anchor is a single-device election. Neither
generalizes cleanly to N>2 with heterogeneous links, where PCIe
contention, NCCL ring transit, and network jitter can shift the
effective bottleneck between rounds. The architecture that scales
splits the problem into **invariants** (the existing guards, unchanged)
and a **scheduler** (ElChe, now predictive).

### Participation-closed superstep

A *meta-step* is the interval between two instants at which every
active rank has completed at least one AllReduce since the previous
boundary. In full sync mode that is exactly one AllReduce. In async /
Cadence mode the meta-step length emerges from whichever participant
rejoined the group last. All tuning decisions (anchor weights, cadence
adjustment, dispersion evaluation) fire at meta-step boundaries;
between boundaries, policy is fixed.

This buys two properties. First, **measurement freshness**: dispersion
across ranks is only semantically clean when every rank has a recent
weight snapshot, which the participation-closure definition guarantees
at each boundary. Second, **natural hysteresis**: no single signal
moves the anchor mid-flight, because the only decision points are
after the full group has been heard from.

### Invariants stay pairwise, scheduler scales

`max_batch_diff` and `max_overshoot` remain local pairwise bounds,
empirically proven and retained unchanged. They apply *within* a
meta-step, where the participation-closure definition already bounds
heterogeneity of state. They act as safety limiters: under nominal
conditions, ElChe's scheduling keeps actuals well inside the guard
envelope; a guard firing is the signal that the predictor was wrong
and the next meta-step should retune.

All N>2 complexity moves into the scheduler, which is well-studied
territory (HPC job scheduling, work-stealing, BSP with fault
tolerance).

### Wall-time prediction

ElChe's job is to produce aligned close-times for every rank in the
next meta-step. The inputs are a joint prediction over `(device,
dispatch size, expected sync cost)`: EMA on per-batch compute
wall-time plus EMA on transfer cost per byte per link. Dispatch time
itself participates in the cost model, so the feedback loop closes:
biased schedules show up as biased predictions and self-correct.

Cold start dispatches uniformly in meta-step 0 to collect samples.
Subsequent meta-steps use the running prediction to size each rank's
chunk so that predicted close-times align within a target window.
Prediction error is surfaced as first-class telemetry: persistent
one-sided residuals indicate the scheduler is biasing the workload and
is a knob worth tuning on.

The per-link latency signal the predictor already needs subsumes a
"bandwidth-aware Cadence anchor" as a special case: when the binding
constraint is a slow link rather than a slow GPU, the EMA on transfer
cost per byte surfaces it naturally and the scheduler sizes around it.

### Reschedule on breach

If a rank runs past its predicted wall-time budget within a meta-step,
two options close the meta-step cleanly:

- **Work-steal** (default): the unfinished shard migrates to the
  fastest available GPU and replays. The meta-step sees the full data
  shard; convergence math is unchanged. Cost: re-dispatch machinery.
- **Graceful exclusion** (escape hatch): close the meta-step without
  the laggard and re-weight the average over actual participants.
  Used only when replay would itself exceed the meta-step budget.
  Introduces per-meta-step sample-weight variance; documented
  degradation.

Failure detection is the same signal: a rank that exceeds its
predicted budget by a threshold is promoted from *late* to *failed*
and reschedule kicks in. No separate health-ping channel needed.

### Two-layer control

The whole structure is a control-theoretic hierarchy: a *soft*
predictive scheduler (ElChe sizing work so close-times align) and
*hard* safety guards (`max_batch_diff`, `max_overshoot`) that bound
what can go wrong inside a single meta-step. The scheduler handles
adaptivity and scales with N; the guards handle worst-case bounds and
stay pairwise. Decoupling the two is what keeps the convergence
argument simple as the cluster grows.

### Positioning

This is a BSP superstep with the barrier relaxed from
wall-clock-global to participation-closure, plus an online wall-time
predictor and work-stealing on budget breach. Against the MSF lineage,
HetSeq and Cannikin fix cadence and let divergence emerge; the
meta-step architecture bounds divergence by construction (via the
retained guards) and lets cadence emerge from heterogeneity. Two
orthogonal axes, same problem.

## Cloud-specific primitives (follow-on)

Once outer optimizer and meta-step are in place, the remaining cloud
DDP stack is additive:

- **Gradient / delta compression**: top-K sparsification, 1-bit
  quantization, error-feedback accumulators. Works cleanly on
  pseudo-gradients because the outer optimizer absorbs the quantization
  noise.
- **Nested meta-steps (hierarchical ElChe)**: an intra-host meta-step
  with tight-cadence NCCL nested inside an inter-host meta-step with
  loose-cadence DiLoCo. One heterogeneous cluster, two
  participation-closure levels. The wall-time predictor operates at
  the outer level; the inner level reuses today's single-host ElChe.
  The recursive generalization (arbitrary depth, plane separation,
  resilience economics) is sketched as hypothesis-gated preliminary
  thinking under [Hierarchical scaling](#hierarchical-scaling-to-massive-clusters-preliminary).
- **Parameter-server / fully-async rounds**: drop the rendezvous
  barrier entirely and submit deltas with a staleness bound. Harder
  to reason about than meta-step async, so offered as an opt-in when
  the outer optimizer's noise tolerance is enough to absorb the
  staleness.
- **Byzantine-tolerant aggregation** (optional): trimmed mean /
  median instead of weighted mean for untrusted workers (federated /
  open-contribution training).

## Hierarchical scaling to massive clusters (preliminary)

This section is **preliminary thinking, not a committed design**. It records
the architecture the project is pointed at, so that near-term work
(window-pressure auto-tune, the outer optimizer) can be shaped to generalize
and so the assumptions are written down as falsifiable hypotheses rather than
carried implicitly. No code exists for any of it, and none should be built
until the governing hypothesis below is measured on real workloads.

### The governing hypothesis (the gate)

Everything here is subordinate to one precondition:

> **H1.** Scaling is only meaningful while convergence holds across the *full
> end-to-end averaging cycle*. Tree depth times per-level window sums to a
> total staleness budget `H_total`; the entire topology exists to keep every
> unit's work landing comfortably inside it. Outside `H_total` the workers
> decohere (the divergence / NaN regime) and no topology recovers that.

So `H_max(model, LR, data)`, the largest staleness a workload tolerates before
convergence degrades, is the quantity every other decision is subordinate to,
and it is currently unmeasured. The near-term window-pressure work and the
outer optimizer are the instruments that measure and widen `H`; hierarchical
scaling is unfounded until `H_max` is characterized.

### The unifying abstraction (hypothesis)

Under H1, a sub-cluster behaves like a single device with more gradient noise:
larger effective batch, bounded staleness, and the implicit-regularization
effect already observed at single-host scale. The noise has a sign that flips
at `H_max`: regularizing inside the window, destructive beyond it. "A cluster
is a bigger GPU, a datacenter is one huge GPU" is a useful frame precisely and
only while H1 holds.

### Two planes scale independently

flodl already separates the **averager** (data plane: the deterministic
sum-and-count reduce) from the **controller** (control plane: ElChe
scheduling). Both generalize to a tree, but differently.

- **Averager: recursive pre-summation.** Hosts sum their ranks; intermediate
  nodes sum hosts; the root sums sub-clusters. Sum-and-count is associative, so
  a reduction tree adds no averaging bias (no average-of-averages). The only
  cost of depth is added staleness, charged against `H_total`. A pure
  compute-versus-network tradeoff.
- **Controller: recursive ElChe.** Each level balances its children; the parent
  treats each sub-cluster as one virtual device with an aggregate throughput.
  Deterministic, seed-derived shard assignment keeps coverage well-defined at
  every level.

### Recursive ElChe stability (hypothesis)

Nested adaptive controllers can oscillate, but the topology enforces the
cascade-control stability condition structurally: the expensive link sits at
the top with large windows (slow adaptation), the fast local links at the
bottom with small windows (fast adaptation). Inner-fast / outer-slow is exactly
the separation that prevents the loops from fighting, and it falls out of
placing the costly link at the root rather than being engineered. Aggregation
also reduces variance: a sub-controller absorbs local throttling locally, so
the parent sees a more stationary signal than a flat controller would.
**Invariant to enforce:** per-level window must increase up the tree; inverting
it breaks the stability condition. The convergence guard is scale-invariant in
mechanism (tightening a rank's cadence and tightening a sub-cluster's cadence
are the same operation); only its thresholds scale with level, looser higher up
because the aggregate is pre-smoothed.

### Resilience economics (hypothesis)

- Aggregator nodes do CPU-side summation and scheduling, no GPU, so they are
  far cheaper than the compute they front. Redundancy is cheapest exactly where
  blast radius is largest.
- A sub-cluster failure is graceful, not a stop: the root's alive-count
  division (today's dead-rank ledger generalized to a dead meta-rank) drops the
  sub-cluster and continues, losing one window of its work, the same semantics
  as a single rank failing today.
- Expected failure exposure stays dominated by the GPU tier (`N_gpu * p_gpu`);
  the aggregator tier (fewer nodes, more reliable, replicable) adds a negligible
  term, and replication drives it to `p^2`.
- **Principle: replicate by (blast-radius * rate / cost).** Aggregators score
  high (large blast, cheap, rare), so replicate them; GPUs score low (1/N
  blast, expensive, common), so use elastic membership instead. Redundancy
  lands where it is cheap and impactful.
- The averager replicates **active-active**: a deterministic reduce means two
  replicas on the same inputs produce bit-identical output, so failover is
  seamless. The controller's allocation is a soft, timing-influenced decision,
  but coverage is reconstructable from seed-derived sharding plus the
  replicated count stream, so a standby recomputes valid coverage
  independently; the only requirement is leader fencing (one primary drives the
  workers at a time).
- The **root** is the genuine single point. It concentrates the hardening (ECC
  RAM, redundant power, RAID) and, critically, its entire state (coverage,
  outer-optimizer momentum, ElChe trajectory) is already checkpointable through
  the consensus-forge path, so recovery is promote-a-standby-and-resume with
  bounded cost.

### The determinism boundary

The recurring seam under all of the above: deterministic components
(sum-and-count, seed-derived sharding) replicate and resume trivially; the
single timing-driven component (ElChe scheduling) is where soft variance lives,
and it is tolerable because allocation is a soft decision while coverage stays
deterministic. Resilience, replication, and the transport split below all
cleave along this same boundary.

### Transport at extreme scale (hypothesis)

At millions of devices the connection mesh itself becomes the problem. A
durable log / stream substrate (for example Kafka) for the **control plane**
replaces point-to-point connections and gives replay-based fault tolerance; the
**data plane** (gigabyte-scale tensor payloads) stays on a high-bandwidth path
(RDMA, collectives, or an object store), since bulk blobs are not a log's sweet
spot. Broker latency is tolerable only in the large-`H` regime, where
infrequent syncs amortize it; it would not survive tight all-reduce. Replay
requires the reduce to be idempotent, keyed by `(round, unit)`.

### Where the abstraction leaks

These do not fall out of "a cluster is a bigger GPU" and must be designed
explicitly:

- **Failure blast radius grows up the tree** even as allocation variance
  smooths down it. Fault tolerance is the one place the single-GPU analogy
  fails outright; it needs its own subtree-failure and aggregator-redundancy
  design.
- **Sharding must become hierarchy-aware**: sub-clusters need disjoint,
  IID-balanced slices, or sub-aggregates correlate and the noise stops being
  benign.
- **The staleness-noise sign-flip** means the regularization benefit and the
  divergence cliff are the same mechanism at different magnitudes; the margin to
  the cliff is `H_max` and must be respected, not assumed.

### Hypotheses that must hold (checklist)

Before any of this is built, each must be validated, not assumed:

1. **H1 (gate):** `H_max` is large enough that a full end-to-end averaging
   cycle lands comfortably inside it. Currently unmeasured; the near-term `H`
   work is the instrument.
2. **Equivalence:** "sub-cluster = one noisier GPU" holds inside `H_max`, and
   the added noise is net neutral or beneficial there.
3. **Recursive control:** nested ElChe is stable given windows-grow-up-the-tree;
   to be demonstrated at depth 2 first.
4. **Sharding:** data assignment can be made hierarchy-aware without inducing
   cross-sub-cluster correlation.
5. **Resilience:** aggregator failure is graceful and cheaply made rare; root
   recovery via checkpoint/resume meets the availability target.
6. **Transport (extreme scale only):** control-plane streaming latency is
   amortized by large `H`, and a separate high-bandwidth data-plane path exists.

The depth-2 case of this structure already exists today: per-host relays
(level-1 aggregation) under a single controller, with the inner/outer optimizer
split as the two-level cadence. The recursive version is that same structure
unrolled, earned one level at a time as workloads demand it, never built ahead
of the hypothesis that gates it.

## Why this matters for flodl

Most DDP framework docs sell "faster multi-GPU." The empirical data
already gathered shows flodl's cadence mode is *already* winning on
generalization at the single-host scale, which is the same mechanism
frontier labs are reaching for at the trillion-parameter scale with
DiLoCo and its descendants. Pairing that outer-optimizer pattern with
the meta-step architecture positions flodl's `ElChe` as the complete
story: heterogeneous-hardware-aware, fault-tolerant,
network-efficient, generalization-improving DDP out of the box, from
2 GPUs on a workstation to a multi-region cluster.
