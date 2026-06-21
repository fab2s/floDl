# MSF Cadence Control for DDP

**Status:** research proposal, pre-implementation. This document proposes
a principled replacement for ElChe's current convergence heuristic,
grounded in the synchronization-of-coupled-systems literature (Master
Stability Function theory). Captures hypotheses, controllers, and a
phased experimental program with explicit kill criteria at each stage.

---

## Motivation

ElChe's current convergence guard uses `||pre - post|| / ||post||` as a
divergence proxy between synchronization points and reacts when three
consecutive readings rise above a threshold. It works (battle-tested across
200-epoch ResNet-20 runs), but it is an ad-hoc heuristic: the threshold is
hand-tuned, the "three rises" rule has no principled derivation, and the
metric itself has no clean interpretation in training-step units.

DDP is a network of identical dynamical systems (GPU replicas with the
same optimizer and architecture) driven by different data batches and
coupled through periodic AllReduce. This is the exact setup the
synchronization-of-coupled-systems literature studies, with a mature
theoretical framework (Pecora and Carroll, 1998; Arenas et al., 2008) and
1500+ citations of related work. ElChe's divergence metric is a scalar
proxy for a specific quantity in this framework: the transversal
deviation of replicas from the synchronization manifold.

The hypothesis: framing cadence control as synchronization-manifold
stability gives a principled controller that either matches the current
heuristic (formalizing why the heuristic works) or exceeds it (by
correctly balancing transversal deviation against exploration). Local SGD
has not, to our knowledge, been framed this way before, so even a
theoretical paper that formalizes the existing heuristic would be
publishable.

---

## Theoretical framework

### Coupled dynamical systems

DDP with `N` workers is a network of `N` identical dynamical systems
evolving in the joint state space:

- Each worker has state `W_i` (its weight vector).
- All workers share the same optimizer dynamics (SGD + optional momentum).
- Workers are driven by independent data perturbations (different batches).
- Periodic AllReduce couples them by projecting all `W_i` to their mean.

Two features distinguish this from the classical MSF setup:

1. **Pulse coupling, not continuous.** Classical MSF usually analyzes
   continuous diffusive coupling. DDP couples impulsively at sync events,
   with the strength and cadence of coupling under the controller's
   discretion.
2. **Non-identical forcing.** The data-driven perturbation differs across
   workers, so strictly speaking the nodes are not perfectly identical.
   The generalized-synchronization literature extends MSF to this case
   (less saturated, more room for contribution).

### Synchronization manifold

The synchronization manifold is the subspace where all replicas agree:

```
M_sync = { (W_1, W_2, ..., W_N) : W_1 = W_2 = ... = W_N }
```

Immediately after each AllReduce, the joint state is exactly on `M_sync`.
Between AllReduce events, data perturbation drives each worker along its
own trajectory, and the ensemble drifts away from `M_sync`. The geometry
of this drift is the object we care about.

This also connects cleanly to the existing SWA / centroid-in-wider-basin
story from [research-questions.md](research-questions.md): `M_sync` is
exactly the centroid subspace, and the averaging operation is a projection
onto it.

### Transversal Lyapunov exponent

The rate at which small perturbations grow transverse to `M_sync` is the
transversal Lyapunov exponent, denoted `lambda_T`.

- `lambda_T > 0`: replicas drift off `M_sync` exponentially (sync
  unstable, need to couple more often).
- `lambda_T < 0`: replicas contract back toward `M_sync` exponentially
  (sync stable, could couple less often).
- `lambda_T ~ 0`: marginal (steady state).

In classical MSF, `lambda_T` is a function of coupling strength and the
spectrum of the coupling Laplacian. For pulse-coupled DDP, `lambda_T`
becomes an effective quantity over the cadence interval, and the
controller's job is to regulate it by adjusting cadence.

### ElChe's metric as a transversal deviation

The divergence metric:

```
D_t = ||W_pre - W_post|| / ||W_post||
```

has a direct MSF interpretation. `W_pre` is a worker's weights just before
AllReduce. `W_post` is the averaged weights, which lie on `M_sync` by
construction. Therefore `W_pre - W_post` is exactly the worker's
transversal deviation from `M_sync` at the end of the coupling interval.
`D_t` is this deviation normalized by state scale, a dimensionless summary.

---

## Practical formulation

### Estimating `lambda_T` from end-of-cycle measurements

Within a single cadence interval, transversal deviation starts at zero
(right after AllReduce) and grows to `D_t`. We do not measure intermediate
samples, so we cannot compute a within-cycle exponent directly.

Across consecutive coupling events, if transversal deviation grows
approximately exponentially between events:

```
D_t ~= D_{t-1} * exp(lambda_T * k)
```

then `lambda_T` can be estimated from two consecutive end-of-cycle
measurements:

```
lambda_hat_t = (1/k) * log(D_t / D_{t-1})
```

where `k` is the cadence interval just completed. Both operands are
nonzero by construction (ElChe only measures at pre-AllReduce moments).
No epsilon floors, no assumed reference scale, no within-cycle
instrumentation.

**Interpretation.** `lambda_hat > 0` means transversal deviation is
growing faster across coupling events than the previous cycle
(synchronization margin deteriorating). `lambda_hat < 0` means the margin
is improving. `lambda_hat ~ 0` means steady state.

**Empirical sanity check.** Applying this formula to the data already in
[research-questions.md](research-questions.md) at `k = 8`:

| Phase | D_{t-1} -> D_t | lambda_hat | Interpretation |
|---|---|---|---|
| Warmup | 0.12 -> 0.04 | -0.14 | stabilizing |
| Steady state | 0.03 -> 0.03 | ~0 | equilibrium |
| LR drop | 0.025 -> 0.0002 | -0.59 | strong collapse |
| Late training | 0.0002 -> 0.00002 | -0.29 | stable attractor |

All interpretable, phase-aware without LR-change notifications, good
dynamic range across regimes. This is a sanity check, not a validation;
the formal validation lives in Phase 1 of the research program.

### What this is and is not

This is not the exact MSF quantity. The exact MSF `lambda_T` would
require either continuous coupling (not what DDP does) or within-cycle
sampling of transversal deviation (which we do not currently collect).
What we compute is an **across-event proxy** for `lambda_T`, using the
endpoint measurements already available.

The proxy is valid if:

1. Transversal deviation grows approximately exponentially across
   consecutive coupling events (hypothesis R1 below).
2. The proxy tracks the true `lambda_T` well enough to drive a controller
   that outperforms the current heuristic (hypothesis R3 below).

Both hypotheses are falsifiable by the Phase 1 plots.

---

## Refinements (hypotheses to test)

### R1: across-event exponential growth

**Hypothesis.** At fixed learning rate within a single training phase,
transversal deviation grows approximately exponentially across consecutive
coupling events, so `log(D_t)` is approximately linear in cumulative
training step.

**Test.** Sweep fixed-k runs at `k` in {2, 4, 8, 16, 32} at stable LR.
Plot `log(D_t)` vs step. Linear fit with `R^2 > 0.9` supports R1. Curved
or saturating behavior supports an alternative functional form
(power-law, logistic), in which case the controller formula must change.

**Why this matters.** The across-event proxy and the closed-form
controller (C1) both depend on this. If R1 fails, the research program
changes direction (see kill criterion in Phase 1).

### R2: phase-dependent setpoint

**Hypothesis.** The optimal `lambda_T` setpoint tracks learning rate.
A two-level controller (outer: `LR -> lambda_T*`, inner: `lambda_T* -> k`)
naturally encodes the observed empirical pattern from
[research-questions.md](research-questions.md) where divergence magnitude
drops 100x across LR annealing.

**Stronger claim.** Learning rate annealing is a transversal-stability
schedule in disguise: the operator driving the coupled system from
chaotic exploration toward the sync attractor. If provable, this is the
headline result of the paper.

### R3: controller matches or beats current guard

**Hypothesis.** A principled controller based on R1 and C1 matches the
current ElChe convergence guard's test accuracy on ResNet-20 CIFAR-10
within statistical uncertainty.

**Why matching is enough.** Matching means the principled version is
equivalent to the heuristic, which formalizes why the heuristic works
and connects Local SGD to MSF theory. Beating is a bonus, not a
requirement for the paper. Parity is validation.

### R4: noise floor in the stable regime

**Issue.** At very small divergence (late-training, `D ~ 1e-5`), the
signal is dominated by floating-point noise. `log(D_t / D_{t-1})` is
unreliable when both are near the noise floor.

**Hypothesis.** A simple clamp (freeze controller when `D` is below
threshold, hold last stable cadence) is sufficient. Second-order metrics
(variance of deviation) are overkill.

### R5: innovation detector as guard replacement

**Hypothesis.** The current "3 consecutive rises in D" rule is a coarse
derivative detector. A proper innovation signal
(`lambda_hat_t - lambda_hat_{t-1}`, CUSUM-style) fires earlier and with
fewer false positives.

Classical change-detection territory (Kalman filtering, CUSUM); not
expected to be novel, but potentially cleaner than the existing rule.

---

## Controllers

### C1: closed-form cadence inversion

Given current transversal deviation `D_t`, its smoothed growth rate
`lambda_hat_t`, and a target budget `D_max`, invert the exponential
growth law to solve for the next cadence:

```
k_next = floor(log(D_max / D_t) / lambda_hat_t)
```

This is a horizon-1 model-predictive controller with a closed-form plant
model. No PI gains. Reads as "given the current deviation and its current
growth rate, how many steps until we hit the budget."

**Safety rails.**

| Regime | Behavior | Action |
|---|---|---|
| `lambda_hat ~ 0` | steady state | clamp `k_next` to `k_max` |
| `lambda_hat < 0`, `D_t < D_max` | over-synchronized | loosen: `k_next = min(k_max, 2*k)` |
| `lambda_hat > 0`, `D_t > D_max` | destabilizing | formula gives small positive `k_next`; honor subject to `k_min` |
| `D_t` below noise floor (R4) | stable attractor | freeze controller, hold current `k` |
| Noisy `lambda_hat` | transient | EMA smoothing before inversion (C2) |
| Wild single-sample spike | robustness | multiplicative rate limit `k_next in [k/2, 2*k]` |
| `floor()` dead-band | avoid pointless re-floor | small hysteresis band |

### C2: EMA-smoothed estimator

```
lambda_hat_t = alpha * lambda_hat_{t-1} + (1 - alpha) * lambda_t
```

Default `alpha = 0.9`, effective window ~10 events.

- **Adam-style bias correction:** `lambda_hat_t / (1 - alpha^t)` for the
  first few events; startup bias otherwise dominates during the high-LR
  phase where `lambda` is most informative.
- **Memory in training-step units.** With fixed `alpha`, the filter's
  memory in training steps scales with `k`. Either document this as
  intended coupling or adapt `alpha` so memory stays constant in
  training-step units.
- **Phase-transition lag.** At LR drops, `lambda` can jump ~100x. With
  `alpha = 0.9`, filter needs ~20 events to catch up. During the lag the
  controller holds cadence tight when it could loosen. Partial mitigation
  via C3.

### C3: two-timescale filter

Run a fast estimator (`alpha_f ~ 0.5`) and slow estimator (`alpha_s ~ 0.9`)
in parallel. Use slow for the controller, fast for change detection. When
`|lambda_hat_f - lambda_hat_s| > threshold`, the system is in a
transition: switch controller to fast estimate until the two reconverge.

Standard in tracking / Kalman filtering. Avoids hand-tuning `alpha` per
training phase.

### C4: phase-aware budget

Couple `D_max` to learning rate:

```
D_max(LR) = D_max_base * (LR / LR_initial) ^ p
```

Exponent `p` fitted from R2 data. If R2 holds, `p` falls out of the same
experiment.

---

## Research program

Phases are gated. Do not advance without satisfying the previous stage's
success metric. Kill criteria at each stage are explicit.

### Phase 1: passive observation

**Goal.** Determine whether `lambda_hat` carries useful signal, or is
just a function of data we already plot.

**Implementation.** Instrument the current DDP pipeline to compute and
log `lambda_hat_t` alongside existing metrics. No behavior change.
Current convergence guard remains authoritative. Measurement addition
only.

**Runs.**

- ResNet-20 CIFAR-10, 200 epochs, 5 seeds, both CPU-sync and NCCL-sync.
  Standard training config from v0.4.0 benchmark.
- Fixed-`k` sweep at `k` in {2, 4, 8, 16, 32}, 50 epochs each, 3 seeds,
  NCCL-sync. Tests R1.

**Plots.**

- `lambda_hat_t`, `D_t`, `k_t`, `LR`, train loss, test acc on a shared
  timeline.
- `lambda_hat_t` distributions per training phase (pre-LR-drop,
  post-LR-drop, late training).
- `log(D_t)` vs step at fixed `k`, each `k` value (R1 validation).
- `lambda_hat_t` correlation with subsequent divergence events and with
  held-out test accuracy.
- `d(lambda_hat)/dt` alongside `lambda_hat`, current-guard fires marked
  (R5 comparison).

**Success metrics.**

- `lambda_hat` shows visibly more structure than raw `D` in at least one
  of: phase-transition alignment (especially at LR drops), dynamic range
  across phases, noise floor behavior, predictive value for subsequent
  divergence.
- AND/OR R1 holds: `log(D_t)` is approximately linear in step within
  training phases, `R^2 > 0.9`, `lambda_hat` has bounded variance within
  phases.

**Kill criterion.** If `lambda_hat` is indistinguishable from raw `D`
(same zero-crossings, same false-positive rate for the current guard)
AND R1 fails (no regularity in log-divergence), the reformulation adds
no information and the research program stops. This is the cheapest and
most honest kill.

### Phase 2: toy-problem controller validation

**Goal.** Confirm C1 works on a controlled problem with known dynamics
before touching ResNet.

**Implementation.** Two-worker synthetic problem with a known loss
landscape (quadratic with injected perturbation, or Rosenbrock) where
transversal deviation can be measured against ground truth. Implement
C1 + C2 and compare to fixed-`k` baselines.

**Success metrics.**

- C1 holds `lambda_hat` within 20% of setpoint across stable regimes.
- No oscillation beyond rate limits under step perturbations.
- Controller recovers from induced LR drops within 5 events.

**Kill criterion.** If the toy problem cannot be regulated cleanly, the
exponential-growth model is wrong and controllers built on it will fail
at scale. Back to Phase 1 with an alternative functional form.

### Phase 3: A/B vs current guard on ResNet-20

**Goal.** Determine whether any MSF-based controller matches or exceeds
the current guard on the benchmark.

**Implementation.** Ship controllers as opt-in research modes (see
pluggable architecture). Run full 200-epoch benchmarks.

**Success ladder.**

| Tier | Condition | Interpretation |
|---|---|---|
| 0 | matches convergence within CI overlap | theory validated, ship as pluggable alternative |
| 1 | matches convergence + speed | same, no tradeoff |
| 2 | matches convergence + beats speed | default-candidate |
| 3 | beats convergence | standalone result, paper writes itself |

**Statistical discipline.**

- `N >= 5` seeds per configuration.
- 95% confidence intervals on every headline number.
- Comparisons require non-overlapping CIs for strong claims; overlapping
  CIs with consistent mean direction count as weak signal needing more
  seeds.
- The current v0.4.0 benchmark gap between cpu-sync and nccl-sync is
  0.15pp, inside single-run variance. Single-run comparisons will fool
  the researcher at this scale.

**Kill criterion.** If the best-performing MSF controller underperforms
current guard by more than 0.3pp test accuracy across 5 seeds, the
principled version is strictly worse and the controller research stops.
The theoretical reframing may still stand as a descriptive result even
if the controller fails.

### Phase 4: phase-aware adaptation

**Goal.** Validate R2. Implement C4.

**Entry condition.** Phase 3 at Tier 0 or better.

**Success metric.** C4 outperforms constant `D_max` on at least one of
speed or convergence at the LR-drop boundary.

### Phase 5: paper

**Entry condition.** Phase 3 at Tier 0 or better, AND at least one of:

- R1 confirmed (exponential growth holds).
- R2 confirmed (phase-aware setpoint outperforms).
- Phase 3 reaches Tier 2 or better.

**Narrative spine.** "We frame ElChe's divergence heuristic as an
across-event proxy for the transversal Lyapunov exponent of the DDP
coupling dynamics. The resulting closed-form controller matches the
heuristic on ResNet-20 (validating the framing) and the phase-aware
variant connects learning rate annealing to synchronization-manifold
stability theory."

---

## Metrics discipline

- **Seed count.** All comparative claims require `N >= 5` seeds. No
  single-seed comparisons in the paper.
- **Confidence intervals.** 95% CI on every headline number.
- **Held-out split.** All accuracy from held-out test data, never
  training data. Project policy already (memory:
  `feedback_bench_test_split.md`).
- **Sync overhead accounting.** Report wall time and sync percentage
  separately. A controller that is faster by doing less is different
  from one that is faster by working smarter.
- **`lambda_hat` stability.** Track variance within a training phase.
  An unstable estimator cannot drive a stable controller.

---

## Pluggable architecture

Controllers must slot in without touching DistributedState internals.
Sketch of a trait boundary (subject to revision during Phase 1):

```rust
pub trait CadenceController: Send + Sync {
    /// Called at every averaging event. Returns the cadence k to use
    /// for the next interval.
    fn on_divergence(&mut self, ctx: &DivergenceContext) -> usize;

    /// Optional: called at each LR change for phase-aware controllers.
    fn on_lr_change(&mut self, _new_lr: f64) {}

    /// Snapshot for observability. Controller-specific content.
    fn telemetry(&self) -> Vec<(String, f64)> { vec![] }
}

pub struct DivergenceContext<'a> {
    pub d_raw: f64,         // ||pre - post|| / ||post||
    pub k_current: usize,
    pub step: usize,
    pub lr: f64,
    pub history: &'a [DivergenceSample],
}
```

Shipping configuration:

- **Default:** current ElChe convergence guard. Unchanged.
- **Research modes (opt-in, experimental-flagged):**
  - `Msf::Basic` (C1 + C2, constant `D_max`)
  - `Msf::TwoTimescale` (C1 + C3)
  - `Msf::PhaseAware` (C1 + C3 + C4)
- **Observational mode:** Phase 1 instrumentation, always-on
  `lambda_hat` logging with no controller influence. First deliverable
  so community users can contribute data.

Get the trait boundary right before the research iterations start; it
constrains what every future controller can see.

---

## Open questions

- Does the across-event proxy track the true `lambda_T` well enough to
  drive a controller? Phase 1 answers this directly.
- Does `lambda_hat` track LR drops sharply enough to act on without
  explicit LR-change notifications? If yes, R2's two-level controller
  is simpler than expected.
- Does the parameter-averaging-vs-gradient-averaging generalization gap
  (0.15pp in v0.4.0) correlate with `lambda_hat`? If yes, the SWA
  connection tightens into a testable claim.
- Is there an analog for the optimizer's momentum state? Current metric
  measures weight drift; momentum drift may be a separate (faster)
  signal.
- How much of the classical MSF toolkit (coupling Laplacian spectrum,
  stability conditions as functions of network topology) transfers to
  pulse-coupled DDP? Pulse regime has less theoretical machinery than
  continuous; may be a contribution area in itself.
- Heterogeneous hardware makes nodes non-identical (different speeds,
  different effective batch sizes). Classical MSF assumes identical
  oscillators; generalized-synchronization literature addresses
  non-identical nodes but is less saturated. ElChe's
  heterogeneous-GPU angle may land in that less-crowded corner.
- Does the framework generalize beyond DDP? Federated learning has
  structurally identical cadence-vs-drift tradeoffs. Out of scope
  here, noted for future work.

---

## Appendix: why not classical single-trajectory Lyapunov?

The initial intuition behind this research was that `||pre - post|| / ||post||`
looks like a discrete-time Lyapunov exponent for a single trajectory.
That intuition seeded the work, but on closer inspection it fails for
three reasons:

1. **Wrong object.** Classical Lyapunov describes how a single trajectory
   diverges from nearby initial conditions over time. DDP has `N`
   trajectories evolving in parallel; the relevant question is whether
   the *ensemble* stays synchronized, not whether any one trajectory is
   chaotic.
2. **Sign convention breaks.** Computing `log(D_t) / k` always yields a
   negative value for well-behaved training (drift stays small relative
   to weight norm). The textbook "positive lambda means chaotic"
   intuition does not apply. The current guard's "rising divergence"
   rule is a *derivative* signal, not a sign signal.
3. **Zero-reference problem.** The textbook form `log(||delta_n|| /
   ||delta_0||) / n` requires a nonzero `||delta_0||`. Right after
   AllReduce, all replicas agree, so transversal deviation is literally
   zero. Any within-cycle formulation blows up at the reference point.

The MSF framing resolves all three: the object is the synchronization
manifold (not a trajectory), the sign convention is well-defined (`lambda_T`
can be genuinely positive or negative depending on coupling strength),
and the across-event growth rate form sidesteps the zero-reference
problem by comparing endpoints of consecutive cycles.

---

## Related work

The closest venue-scale predecessor on the heterogeneous-DDP empirical
axis is **HetSeq** (Ding, Botzer, Weninger, AAAI 2021). HetSeq addresses
the same motivation - university clusters with mixed consumer GPUs that
stock DDP cannot use - by introducing a weighted loss-averaging shim on
top of unmodified step-synchronous PyTorch DDP: each GPU gets a batch
sized to its capacity, per-GPU losses are aggregated by the master with
weights proportional to batch size or token count, and the averaged
loss is broadcast back before an every-step gradient AllReduce.

Their reported configurations reach 5.94x speedup on BERT at 8
heterogeneous nodes (7.19 d to 1.21 d), but expansion decays to 0.60 on
the Transformer translation task at 8 nodes with Transformer BLEU
regressing from 25.09 to 18.74 and BERT training loss doubling from
0.026 to 0.055. The authors flag the scaling shortfall explicitly:
*"speedup scales at about one-half the linear rate, which can certainly
be improved with further development,"* and note *"future plans include
adapting ongoing research in distributed optimization to further improve
training performance on heterogeneous infrastructure."*

MSF cadence control is orthogonal to their mechanism. HetSeq's shim
operates at the loss-aggregation stage and leaves the synchronization
cadence untouched (every-step AllReduce). The MSF framing treats the
residual shortfall as a cadence problem: once replicas are modeled as
impulsively coupled dynamical systems, the transversal Lyapunov proxy
gives a principled signal to decouple fast replicas from slow ones
rather than paying the step-synchronous tax on every iteration.
HetSeq's weighted averaging composes as an ingredient underneath a
cadence-controlled AllReduce, not as a competitor. Note the asymmetry
of validation scale: HetSeq reports at 32 GPUs across 8 real cluster
nodes; the present work's empirical anchor is a 2-GPU heterogeneous
desktop (RTX 5060 Ti + GTX 1060). Closing that scale gap on cloud
infrastructure is a direct prerequisite for a full comparison against
the HetSeq configurations.

## References

- Pecora and Carroll, 1998. "Master Stability Functions for Synchronized
  Coupled Systems." Phys. Rev. Lett. 80, 2109.
  https://doi.org/10.1103/PhysRevLett.80.2109.
- Arenas, Diaz-Guilera, Kurths, Moreno, Zhou, 2008. "Synchronization in
  complex networks." Physics Reports 469(3), 93-153.
  https://doi.org/10.1016/j.physrep.2008.09.002. Modern review, covers
  pulse coupling and non-identical nodes.
- Izmailov et al., 2018. "Averaging Weights Leads to Wider Optima and
  Better Generalization." https://arxiv.org/abs/1803.05407. SWA
  foundation; centroid-in-wider-basin connects naturally to sync-manifold
  geometry.
- Ding, Botzer, Weninger, 2021. "HetSeq: Distributed GPU Training on
  Heterogeneous Infrastructure." AAAI 2021, pp. 15432-15438.
  https://ojs.aaai.org/index.php/AAAI/article/view/17813. Direct
  predecessor on the heterogeneous-DDP empirical axis; weighted-loss
  shim over step-synchronous PyTorch DDP.
- [research-questions.md](research-questions.md): empirical observations
  on divergence behavior, strange-attractor interpretation, and the
  phase-transition pattern at LR drops.

---

## Glossary

### Notation conventions

- **Greek letter alone** (e.g. `lambda`): the true underlying quantity
  of the system. Unobservable in general.
- **Greek letter with hat** (e.g. `lambda_hat`, typeset `λ̂`): an
  estimator of that quantity, computed from observed data. Standard
  statistics convention.
- **Subscript `t`** (e.g. `D_t`, `lambda_hat_t`): value at the t-th
  coupling event (i.e. the t-th AllReduce). Counts sync events, not
  training steps. Consecutive events are separated by `k` training steps.
- **Asterisk** (e.g. `lambda*`): target or setpoint value for the
  controller, not a measured value.

### Symbols used in this document

| Symbol | Meaning | Typical range |
|---|---|---|
| `W_i` | Weight vector of worker `i` | model-dependent |
| `W_pre` | Worker's weights just before AllReduce | |
| `W_post` | Averaged weights after AllReduce (on `M_sync`) | |
| `M_sync` | Synchronization manifold (subspace where all `W_i` agree) | |
| `N` | Number of workers (GPUs) | 2 to thousands |
| `k` | Cadence: training steps between consecutive AllReduce events | 2 to 32 practical |
| `k_min`, `k_max` | Clamp bounds for controller output | policy-dependent |
| `D_t` | Dimensionless transversal deviation at event `t`: `||W_pre - W_post|| / ||W_post||` | 1e-5 to 0.12 (observed) |
| `D_max` | Controller's target budget for `D_t` (upper bound) | 0.01 to 0.1 typical |
| `lambda_T` | True transversal Lyapunov exponent of the coupled system | unobservable directly |
| `lambda_hat_t` | Estimator of `lambda_T` at event `t`: `(1/k) * log(D_t / D_{t-1})` | -0.6 to ~0 observed at k=8 |
| `lambda_hat_f`, `lambda_hat_s` | Fast and slow EMA estimators (C3) | same range |
| `lambda*` | Target setpoint for the controller | policy-dependent |
| `alpha` | EMA smoothing coefficient (C2) | 0.5 to 0.95 |
| `alpha_f`, `alpha_s` | Fast and slow EMA coefficients (C3) | ~0.5 and ~0.9 |
| `p` | Exponent in phase-aware `D_max(LR)` (C4) | fitted empirically |
| `t` | Coupling event index (counts AllReduce events) | |
| `n` | Training step index (counts SGD steps) | |

### Sign and magnitude interpretation

`lambda_hat` carries both sign and magnitude meaning, which is important
because the sign convention here differs from textbook single-trajectory
Lyapunov.

| `lambda_hat` sign | Meaning | Controller action |
|---|---|---|
| `> 0` | Transversal deviation growing cycle-over-cycle; sync margin deteriorating | Tighten cadence |
| `~ 0` | Steady state; deviation neither growing nor shrinking | Hold cadence |
| `< 0` | Deviation decaying; sync margin improving | Can loosen cadence if `D_t < D_max` |

`|lambda_hat|` indicates *speed* of the change: large magnitude means a
fast transition (typical around LR drops), small magnitude means slow
drift (typical in steady-state training). The two-timescale filter (C3)
uses this distinction explicitly.

### Acronyms

| Acronym | Expansion | Notes |
|---|---|---|
| DDP | Distributed Data Parallel | Multi-GPU training via gradient or parameter averaging |
| MSF | Master Stability Function | Pecora-Carroll framework for sync of coupled systems |
| ElChe | El Che | Existing flodl cadence mechanism for heterogeneous DDP |
| SGD | Stochastic Gradient Descent | |
| SWA | Stochastic Weight Averaging | Izmailov et al. 2018 |
| AllReduce | Collective averaging across workers | NCCL or CPU-backed |
| NCCL | NVIDIA Collective Communications Library | GPU-native AllReduce |
| EMA | Exponential Moving Average | First-order low-pass filter |
| MPC | Model Predictive Control | Closed-form cadence inversion (C1) is horizon-1 MPC |
| PI | Proportional-Integral (control) | Classical controller; C1 avoids needing PI gains |
| CUSUM | Cumulative Sum | Change-detection algorithm, classical statistics |
| CI | Confidence Interval | 95% CI used throughout |
| LR | Learning Rate | |
| pp | Percentage points | e.g. "0.15pp" = 0.0015 absolute difference in accuracy |
| OLS | Ordinary Least Squares | Used for linear fit in R1 validation |
| R^2 | Coefficient of determination | Goodness-of-fit for linear regression |
| GPU | Graphics Processing Unit | |

### Observed value calibration

From 200-epoch ResNet-20 CIFAR-10 runs (see
[research-questions.md](research-questions.md)):

| Training phase | `D_t` range | `lambda_hat` at k=8 | Regime |
|---|---|---|---|
| Warmup (epoch 0-1) | 0.12 -> 0.04 | ~-0.14 | Chaotic, rapid stabilization |
| Dynamic equilibrium (epoch 2-99) | 0.025 to 0.04 | ~0 | Steady state (~98% of training) |
| LR drop boundary (epoch 100) | 0.025 -> 0.0002 | ~-0.59 | Collapse to attractor |
| Late training (epoch 130+) | 0.00002 | ~-0.29 | Stable attractor, near noise floor |

These numbers are reference points for what the controller sees in
practice, not targets. The controller regulates `D_t` toward `D_max` (a
budget), using `lambda_hat` as the plant model's growth rate.
