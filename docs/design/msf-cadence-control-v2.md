# MSF Cadence Control v2: Two-Scale Framing

**Status:** refinement of [msf-cadence-control.md](msf-cadence-control.md)
after Phase 1 passive observation (5-seed sweep, ResNet-20 / CIFAR-10 /
3-GPU heterogeneous, 2026-05-04). The original document framed DDP as a
single-scale Pecora-Carroll problem with `λ̂ = (1/k)·log(D_t/D_{t-1})` as
an across-event proxy for the transversal Lyapunov exponent. The data
falsified that proxy as a *predictor* (it tracks endpoint-to-endpoint
noise around an OU setpoint, not within-cycle exponential growth) but
strengthened the *theoretical hook*: training is genuinely a
synchronization-of-chaotic-oscillators system, just at two scales linked
by AllReduce. This document captures the two-scale framing, the Phase 1
verdict against the original R1–R5, the refined hypotheses, and the
sweeps needed to conclude the research program.

The original document is preserved as the historical proposal. This v2
is the working specification going forward.

---

## What changed

The Phase 1 sweep (10 nccl-async runs at N=5 seeds × 2 guards, plus 10
disqualified cpu-async runs) produced three structural refinements:

1. **Meta-oscillator confirmation.** Cross-rank Pearson correlation of
   `D` trajectories: `r > 0.99` across all five seeds, all rank pairs.
   Three GPUs collapse to one degree of freedom in transversal space.
   Per-rank dispersion noise was eating R² in the original cumulative-
   step OLS fits; the `D_mean` basis (per-event mean across ranks)
   recovers the meta-oscillator amplitude cleanly.

2. **R1 axis correction.** The original R1 hypothesis tested
   `log(D_t)` linearity in *cumulative training step*. This is the
   wrong axis. AllReduce sets `D ≈ 0` immediately post-sync (strictly,
   not numerically) so each cycle is an independent draw connected to
   the previous one only through the slowly-evolving weight state, not
   through `D` itself. The natural drift clock restarts at every
   coupling event. With the x-axis switched to `k_used` (steps since
   last sync), R² climbs from 0.49 ± 0.11 to 0.69 ± 0.05 in warmup and
   from 0.41 ± 0.25 to 0.52 ± 0.34 in late training. Within-cycle
   Lyapunov exponents become directly measurable: `λ_T ≈ +1.46e-3`
   ln(D)/step at warmup (5/5 seeds positive), `+2.07e-3` at late
   training (4/5 positive). Mid-LR is the OU-saturated regime where
   cycles reach `D*(LR)` before sync, so R² collapses on every basis,
   matching the marginal-stability prediction.

3. **Two-scale picture.** The combination of the meta-oscillator
   collapse and the within-cycle exponential growth resolves the
   apparent tension between "spiral toward consensus" (contracting,
   OU-style, R1 fails on cumulative step) and "chaotic noise to ride"
   (positive Lyapunov, R1 holds on by-k). They are not competing
   pictures. They are the same phenomenon at two scales linked by
   AllReduce.

   Per-GPU within a cycle: SGD with random batches drives chaotic
   exponential separation in transversal direction. λ_T > 0,
   measurable on the by-k axis. This is the "chaos to ride."

   Meta-oscillator across cycles: the cross-rank average `D_mean`
   evolves as an Ornstein-Uhlenbeck process with phase-dependent
   setpoint `D*(LR)`. Replicas spiral toward consensus, not away from
   it. λ at this scale is contracting, not expanding.

   AllReduce is the projection operator linking the scales. Each
   cycle: per-GPU chaos explores local geometry, then AllReduce snaps
   the ensemble onto the synchronization manifold. Per-rank chaos is
   bounded by periodic re-projection, which is the literal Pecora-
   Carroll synchronization condition. ElChe's cadence parameter `k` is
   the coupling-strength knob from the original 1998 paper.

The Phase 1 program also disqualified the cpu-async backend from MSF
analysis. The 3-phase Idle/Collecting/Computing state machine pipelines
parameter averaging across a wallclock window decoupled from `k`, so
the impulsive-coupling assumption (D resets exactly at the sync moment)
breaks. Cpu-async data is retained only as a foil. The valid backend
for this work is nccl-async, which is in practice nccl-cadence with the
relax-up extension. cpu-cadence remains a candidate for a separate
homogeneous-CPU-averaging study.

---

## The two-scale framing (unified picture)

### Hybrid stochastic model

Within a fixed-LR window:

```
Bottom scale (per-rank, within cycle, τ ∈ [0, k]):
    D_i(τ) ≈ ε_i · exp(λ_T(LR) · τ)        clipped at D*(LR)

Top scale (meta-oscillator, across cycles):
    D_mean(t) ≈ OU process with mean D*(LR) and noise σ²(LR)

Coupling:
    AllReduce projects (W_1, ..., W_N) → mean(W) at sync events,
    setting all D_i ← 0 and starting a new cycle.
```

Here:

- `τ` is steps since last sync (0 ≤ τ ≤ k). The within-cycle clock.
- `t` indexes coupling events (counts sync moments, not steps).
- `ε_i` is the random drift seed at cycle start (per-rank, varies cycle
  to cycle with batch ordering and weight state).
- `λ_T(LR)` is the within-cycle transversal Lyapunov exponent. Positive
  in chaotic regimes, ≈ 0 when the cycle saturates to `D*`.
- `D*(LR)` is the OU setpoint: the equilibrium between centrifugal
  gradient noise and centripetal AllReduce contraction. Function of LR
  (and probably batch size, model size, data distribution).

A cycle either ends sub-saturation (D < D*, exponential signal
dominates, by-k OLS R² high) or saturates (D ≈ D*, k-independent
endpoint, R² collapses on every axis). The transition between regimes
sits at roughly `k ≈ ln(D*/ε) / λ_T`. Below that k, the cycle is in the
chaotic-exploration regime; above it, the cycle is in the OU-equilibrium
regime.

### Why the two scales coexist

A single chaotic oscillator has positive Lyapunov and unbounded
transversal drift. A network of chaotic oscillators with sufficient
coupling synchronizes (Pecora-Carroll 1998): the cross-replica
correlation grows even though each replica's individual trajectory is
chaotic. The collective observable (here, `D_mean` after collapse)
inherits the contractive dynamics of the synchronization manifold,
while the individual observables (per-rank `D_i`) inherit the chaotic
dynamics of the underlying SGD.

This is not a metaphor for our setup. It is the literal Pecora-Carroll
problem with `k` as the discrete coupling-strength parameter. The
original v1 doc named the framing; v2 confirms it operates as designed.

### What this implies for guards and controllers

- **Trend guard's 55 fires per run is not a tuning issue, it is a scale
  error.** The "3 consecutive rises in D" rule operates on per-event
  data, which carries per-rank chaos signal regardless of whether the
  meta-oscillator is converging. Of course it fires often: cycles are
  individually chaotic by physics. The fires are mostly true positives
  *at the bottom scale* and mostly false positives *at the top scale.*
  The meta-oscillator framing says only top-scale events deserve a
  response.

- **MSF guard's 1 fire per run is correct by construction.** It runs
  on `λ_ema`, a smoothed signal that averages out per-rank chaos and
  fires only when the top-scale setpoint shifts. The 98% reduction in
  fires is architecturally inevitable, not lucky.

- **CUSUM lives at the meta scale.** The doc's R5 mention of CUSUM as
  "innovation detector" is the right tool, on the right signal: watch
  the OU residual `λ_ema_t - λ_ema_{t-1}` (or `D_mean(t) - D*(LR)`) for
  cumulative drift exceeding the OU stationary distribution's tail.
  Per-rank D never enters CUSUM.

- **The controller's job is regime detection, not stability
  enforcement.** The natural dynamics already converge. The controller
  should be passive in steady state, react sharply at LR drops (when
  D* shifts), and respect that the system is operating *above* the
  Pecora-Carroll synchronization threshold by default. ElChe's auto-
  tune already does some of this; the spiral framing makes the design
  intent explicit.

---

## Metrics scale convention

Three principles. Locked here so the analyzer, the report, and the controller
implementation all read the same metric in the same way.

### 1. The estimator split (v1 `λ̂` is overloaded; v2 splits into three)

The original v1 formula `λ̂_t = (1/k) · log(D_t / D_{t-1})` was intended as
a Lyapunov estimator. The data shows it is not: `r(λ_raw_t → ln D_{t+1})
= +0.006 ± 0.010` cross-seed (N=5), well within noise. The formula
overloads three physically distinct quantities into one symbol.

| Quantity | Scale | v2 estimator | Used by |
|---|---|---|---|
| Within-cycle Lyapunov `λ_T(LR)` | bottom (per-rank) | by-k OLS slope of `ln(D_mean)` over a rolling window within current LR phase. Per-rank version available as cross-scale consistency check. | C1' (cadence inversion plant model) |
| Regime-change signal | top (meta-oscillator) | CUSUM on `λ_ema_t − E[λ_ema | LR]` (OU residual against phase setpoint), normalized by stationary σ. Threshold in σ-units. | R5'/C5' (regime detector) |
| Smoothed phase indicator | top (meta-oscillator) | `λ_ema` survives but as visualization/coarse-detection only, not as a controller plant input. | R5 v1 result, observability dashboards |

The v1 formula is **deprecated** as a Lyapunov estimator. The analyzer
keeps it for backward compatibility on legacy timelines but new
controllers must use the by-k slope.

### 2. Convergence is exclusively a top-scale phenomenon

Per-replica trajectories are chaotic by construction (positive
within-cycle Lyapunov, measured at `λ_T ≈ +1.5e-3` ln(D)/step at
warmup). They do not converge individually. Asking "did rank 0
converge" is a category error.

The model we ship is the **centroid** (post-AllReduce consensus
weights), which lies on the synchronization manifold by construction.
The centroid converges; the metric we track is at the meta scale
(`final_eval` on consensus weights, training loss curve from
consensus state).

This connects to SWA tightly: `D_i(τ)` are samples from the OU
stationary distribution around the centroid; AllReduce is the running
SWA averaging operation; our standard `final_eval` is already a
SWA-style evaluation by construction.

Operational rule: **per-rank metrics are diagnostic, never target.**
The user-facing convergence story is entirely top-scale. Per-rank
numbers exist to validate the framing or to alarm when it breaks.

### 3. Cross-rank Pearson `r` is the formal framing-validity gate

The meta-oscillator framing requires `D_mean(t)` to faithfully collapse
the per-rank `D_i(t)` into one degree of freedom. The empirical anchor
is cross-rank Pearson `r > 0.99` on every rank pair across all 5 seeds
in the 2026-05-04 sweep. Below `r ≈ 0.95` the collapse breaks: ranks
become independent oscillators rather than coupled views of one
process, and the meta-oscillator framing no longer applies.

**Gate:** the analyzer warns when any rank-pair's Pearson `r < 0.95`.
Below this threshold, bottom-scale per-rank treatment is required and
top-scale signals (`D_mean`, `λ_ema`, guard fires) become unreliable.

A second gate sits at the per-rank by-k slope level: per-rank slopes
should match the meta-`D_mean` slope within seed-to-seed sd. The
analyzer warns when any per-rank slope diverges by more than 2× from
the meta slope on any LR window.

The `cpu-async` backend disqualification is a special case of the
first gate firing for backend reasons rather than dynamics reasons:
pipelined averaging means `D_post ≠ 0` at sync time, breaking the
impulsive-coupling assumption. Cross-rank `r` may still be high but
the meta-collapse becomes meaningless because the per-event D
measurement no longer corresponds to a clean cycle endpoint.

---

## Phase 1 verdict against original R1–R5

Reading original hypothesis numbers against N=5 nccl-async data:

| Original | Verdict | What survives |
|---|---|---|
| R1 (cumulative-step exponential) | **falsified** | the inquiry, not the form. Replaced by R1' (by-k axis) below. |
| R2 (LR-tracking λ* setpoint) | **untested** in passive observation; needs controller A/B. Reframed as D*(LR) tracking (R2'). |
| R3 (controller matches current guard) | **Tier 0 (parity)**. nccl-async msf 91.83% vs trend 91.71%, Δ +0.11pp, within seed range (sd ≈ 0.20pp each). |
| R4 (noise floor handling) | **moot.** Final D ≈ 4–5e-3, never reached the 1e-5 floor in 200 epochs. Untested at scale. |
| R5 (innovation detector beats 3-rises rule) | **strongly supported.** 55.4 → 1.0 fires per run, 98% reduction in false positives. λ_ema is decisively distinguishable from raw D. |

**Phase 1 kill criterion** ("λ_hat indistinguishable from raw D AND R1
fails → stop"): **passes (does not trigger kill)**. R1 fails on the
original axis but holds on the corrected axis (R² ≈ 0.7), and λ_ema is
demonstrably distinguishable from raw D as a guard signal. The
reformulation adds information; the research program continues.

The kill *avoidance* clause matters for the paper: the original v1
narrative was "MSF apparatus formalizes ElChe's heuristic." The v2
narrative is sharper: "training is a Pecora-Carroll synchronization
problem at two scales, ElChe's cadence is the coupling-strength
parameter, and the heuristic works because it operates above the
synchronization threshold." Same theory hook, more empirical leverage.

### Predictive correlations (for completeness)

Cross-seed N=5 on nccl-async msf:

- `r(λ_raw_t → ln D_{t+1}) = +0.006 ± 0.010`. Endpoint-to-endpoint
  λ_hat does not predict next D. This is consistent with the OU
  picture: `λ_hat` measures fluctuation around `D*`, which is mean-
  reverting noise, not drift. The original doc's λ̂ formula tracks the
  wrong quantity for prediction. The within-cycle by-k slope is the
  right Lyapunov estimator.
- `r(λ_mean per epoch → eval) = +0.160 ± 0.053`. Weak but non-zero.
- `r(λ_ema end-of-epoch → eval) = +0.125 ± 0.094`. Same regime.

The `→ eval` correlations are the design doc's "predictive value
against held-out test accuracy" Phase 1 success metric. They are
positive in mean but small in magnitude. Two scales explain why: the
meta-scale signal (which is what `λ_ema` tracks via smoothing) carries
*regime* information (LR drops), not *accuracy* information per se.
Eval correlation appears at LR drops, dilutes through the rest of
training. We do not expect this to be the headline result.

---

## Refined hypotheses

### R1' — within-cycle exponential growth on by-k axis

**Hypothesis.** At fixed LR, `ln(D_mean_t)` is approximately linear in
`k_used_t` (cycle length). Linear fit R² > 0.7 supports R1' in the
sub-saturation regime; R² < 0.3 indicates the cycle saturates to D*(LR)
before sync. Sign of slope is the within-cycle Lyapunov exponent
λ_T(LR).

**Status.** Provisionally supported on observational data (N=5, R² 0.69
± 0.05 warmup, 0.75 ± 0.05 post-transient). Needs cleaner test via
fixed-k sweeps (Sweep B below).

**Why this matters.** The closed-form controller's plant model is
exponential growth within a cycle. R1' is the cleaner statement of the
original R1 and gives a measurable λ_T(LR) per training phase.

### R2' — phase-dependent OU setpoint D*(LR)

**Hypothesis.** The meta-oscillator settles to a setpoint
`D*(LR) = D*_base · (LR / LR_init)^p` with `p ≈ 1` (linear in LR scale)
or some other simple functional form. LR annealing moves D* from the
warmup attractor to progressively smaller attractors.

**Status.** Heuristically supported in the data (D drops from ~0.07
during LR=0.3 phase to ~0.005 in late training). Functional form not
yet fit. Falsifiable: regress `log(D*(LR))` on `log(LR)` across the
three plateaus, fit p, check residuals.

**Why this matters.** The two-level controller in the original v1 was
"LR → λ*". In v2 it becomes "LR → D*", which is empirically observable
(unlike λ*, which is unobservable directly). The cadence controller's
job is to keep D < D*(LR), regulating against the within-cycle
exponential growth.

### R3' — by-k controller matches or beats current guard

**Hypothesis.** A controller using the by-k slope as the within-cycle
Lyapunov estimator (instead of the original endpoint-to-endpoint λ̂)
matches the current ElChe guard's test accuracy on ResNet-20.

**Status.** Untested. The current msf guard uses the original λ̂
formula, not the by-k slope. Running both back-to-back is the cleanest
way to compare.

### R4' — Pecora-Carroll threshold proximity

**Hypothesis.** ElChe's auto-tuned cadence `k` operates above the
synchronization threshold `k*(LR)` from Pecora-Carroll theory. Relaxing
`k` toward `k*` should preserve convergence with fewer syncs (faster
training); relaxing past `k*` should destabilize the meta-oscillator
and cause divergence.

**Test.** Sweep the relax-up flag (`--elche-relax-up`) and the upper
cadence bound (`k_max`) across orders of magnitude; observe where
accuracy degrades and sync count saturates. The threshold sits where
these two regimes meet.

**Why this matters.** This is the literal Pecora-Carroll experiment in
DDP form. If the threshold can be located, the paper has a quantitative
hook ("ElChe operates 2x above the synchronization threshold; relaxed-
anchor mode operates 1.5x above; below 1.0x convergence breaks").
Without it, the claim is qualitative.

### R5' — meta-CUSUM detector

**Hypothesis.** A CUSUM detector running on the meta-scale residual
(`λ_ema_t - λ_ema_target` or `D_mean_t - D*(LR)`, normalized by the OU
stationary σ) fires *only* at LR-drop boundaries with no false
positives elsewhere. This is the precise statement of v1's "fires
earlier with fewer false positives" claim.

**Status.** Not yet implemented. The MSF guard currently fires on
sustained `λ_ema > threshold`, which is a coarse approximation to
CUSUM. The full CUSUM accumulates the residual; the threshold-on-EMA
is a derivative-style detector. CUSUM should be more sensitive to slow
shifts (meta-scale setpoint movement) and less sensitive to fast
fluctuation (OU noise around D*).

---

## Refined controllers

### C1' — by-k cadence inversion

Plant model:

```
D(τ) ≈ ε · exp(λ_T(LR) · τ),   τ ∈ [0, k]
```

Given current `D_t`, the by-k slope estimate `λ̂_T(LR)`, and a target
budget `D_max`:

```
k_next = floor(log(D_max / ε̂) / λ̂_T(LR))
```

where `ε̂` is the early-cycle drift estimate (the y-intercept of the by-
k OLS, exponentiated). This is the v1 closed-form C1 with the right
plant model. Same safety rails apply.

**Differences from v1 C1:**

- The Lyapunov estimate is the by-k OLS slope per LR window, not the
  endpoint-to-endpoint `λ̂_t`. The slope is more stable across seeds (sd
  ±5e-5 on a +1.5e-3 mean for warmup, vs sd at order-of-mean for the
  original endpoint estimator).
- ε̂ is now an explicit parameter (the post-sync drift seed), not
  absorbed into D_{t-1}. Honest about the cycle reset.
- Per-LR-phase tuning is required because λ_T(LR) is regime-dependent;
  this composes naturally with C4'.

### C2' — by-k slope EMA

The slope λ_T(LR) is estimated from a rolling window of past by-k
points (the last N events within the current LR phase). EMA smooths the
slope estimate over time. Tuning parameter is the window size, not an
α coefficient.

The original v1 C2 used `α = 0.9` on per-event λ̂. The cross-seed
variance of that estimator was so high that smoothing didn't fix it.
Switching to slope-of-slopes (rolling OLS over the by-k axis) is a more
honest filter at this signal-to-noise ratio.

### C3' — two-scale filter

Run a fast estimator on per-event `λ_raw` (bottom scale, picks up
chaos) and a slow estimator on the by-k slope (top scale, picks up
regime). Use the slow estimator for cadence control. Use the fast
estimator's running residual against the slow estimator as the meta-
CUSUM input (R5').

This is the v1 C3 with the scale separation made explicit. Fast-vs-slow
EMA in v1 was hand-tuned with both filters running on the same signal.
In v2 they run on physically distinct signals (per-event vs per-window
slope), so the time-scale separation is principled.

### C4' — D*(LR) setpoint regulator

```
D_max(LR) := γ · D*(LR),   γ ∈ (0, 1)
```

with γ ≈ 0.5 as the design budget (target half the OU setpoint, keeping
cycles in the sub-saturation regime where the controller's plant model
holds). D*(LR) is fitted from R2' data per LR window.

This is the v1 C4 with `D*(LR)` replacing the never-defined `λ*(LR)`.
Empirically observable, fitting falls out of the same R2' experiment.

### C5' — threshold-aware cadence

If R4' locates the Pecora-Carroll threshold `k*(LR)`, expose a margin
parameter:

```
k_max := μ · k*(LR),   μ > 1
```

where μ is the safety margin above threshold (default 2x; relax-up
mode pushes toward 1.5x; experimental mode probes 1.1x).

This is new in v2. It treats cadence as a coupling-strength parameter
explicitly, with the synchronization threshold as the structural
reference point. ElChe's existing `--elche-relax-up` is a coarse
implementation of this idea (relax until overhead drops); the
threshold-aware version puts the relaxation under stability-theoretic
control rather than overhead-tuning.

---

## Research program (post-Phase-1)

The original v1 program is reset. Phase 1 is complete with the verdict
above. Subsequent phases are reordered and rescoped around the two-
scale framing.

### Phase 2: relaxed-anchor sweep (probes spiral framing + R4')

**Goal.** Test the spiral-framing prediction that loosening cadence
preserves accuracy, and locate the regime where it stops doing so.

**Implementation.** Re-run the overnight 2026-05-04 nccl-async matrix
with `--elche-relax-up` enabled. Compare to the existing default-anchor
data already on disk.

**Runs.** 5 seeds × 2 guards × nccl-async × `--elche-relax-up`. ~5
hours. Single overnight.

**Success metric.** Relaxed-anchor msf matches default-anchor accuracy
within seed range *and* shows a measurable sync-count reduction. Anti-
prediction (degrades by >0.3pp) falsifies the spiral framing and
reinstates the chaotic-noise picture.

**Side experiment.** Sweep `k_max` upward (default → 2× → 8×) on a
single seed to bracket the threshold. Cheap, falsifies extrapolations.

### Phase 3: fixed-k probe (validates R1' directly)

**Goal.** Confirm within-cycle exponential growth without the
controller in the loop. The original v1 Phase 1 plan included this and
we never ran it; the by-k axis correction makes it the correct probe.

**Implementation.** Disable the convergence guard. Force `k ∈ {2, 4, 8,
16, 32, 64, 128}`. 50 epochs each, 3 seeds, nccl backend. ~3 hours.

**Plots.**
- `D_t` vs `k_used_t` per fixed-k run, log-y axis.
- λ_T fit per (k, LR) cell with confidence intervals.
- Saturation onset: at what k does D_t plateau (cycle reaches D*)?

**Success metric.** R1' linear fit R² > 0.7 in sub-saturation regime,
with λ_T estimates concordant across fixed-k cells (same LR, different
k → same slope). Saturation onset matches `ln(D*/ε)/λ_T` from
independent R2' fit.

### Phase 4: D*(LR) functional form (R2')

**Goal.** Fit the OU setpoint as a function of LR. Closed-form D_max
budget for C4'.

**Implementation.** Re-analyze existing nccl-async data with extended
LR sweep (training-recipe variants: standard 0.3 → 0.03 → 0.003 multistep,
plus 0.1 → 0.01 → 0.001 and 1.0 → 0.1 → 0.01 to bracket). Fit log-log
regression of D*(LR) on LR.

**Runs.** 3 LR schedules × 3 seeds × 200 epochs × nccl-async. ~6 hours.

**Success metric.** Fitted exponent p with seed-to-seed sd < 30% of
mean. Residuals from log-log fit are zero-mean and bounded by OU
stationary σ.

### Phase 5: controller A/B (was v1 Phase 3)

**Goal.** A/B C1' (by-k cadence inversion) vs current ElChe guard on
the standard benchmark.

**Entry condition.** Phases 2-4 complete with at least Phase 2 at
parity.

**Runs.** 5 seeds × {current guard, msf v1, C1' v2} × nccl-async ×
200 epochs. ~7.5 hours.

**Success ladder.** v1 doc's tier table applies unchanged.

### Phase 6: paper

**Entry condition.** Phase 5 at Tier 0 or better, AND at least one of:

- R1' confirmed (Phase 3 fixed-k sweep R² > 0.7).
- R4' threshold located (Phase 2 side experiment finds the relax-up
  failure point).
- R5' meta-CUSUM beats sustained-EMA on a controlled regime-shift
  benchmark.

**Narrative spine.**

> Distributed data parallel training is a synchronization-of-coupled-
> chaotic-oscillators problem in the Pecora-Carroll sense. Each replica
> evolves chaotically between AllReduce events with positive
> within-cycle Lyapunov exponent λ_T(LR); the meta-oscillator (cross-
> rank average) collapses these into a contracting Ornstein-Uhlenbeck
> process around a phase-dependent setpoint D*(LR). AllReduce is the
> projection operator linking the scales, and cadence k is the
> coupling-strength parameter from the original 1998 framework. The
> ElChe heuristic works because it operates above the synchronization
> threshold; the principled controller (by-k cadence inversion against
> a D*(LR) budget) matches its test accuracy and clarifies the design
> intent.

---

## Run plan (concrete sweeps to launch)

In priority order, all on the `ddp-scale` branch from tree `321a401`
or descendants (matching the 2026-05-04 overnight). All under
`ddp-bench/runs/<sweep-name>/seed-N-<config>/`.

### Sweep A: relaxed-anchor overnight

```
5 seeds × 2 guards × nccl-async × --elche-relax-up
ResNet-20 / CIFAR-10 / 200 epochs / 3-GPU heterogeneous
```

10 runs, ~5 hours. Cheapest experiment with the highest expected
information yield (validates or breaks the spiral framing). Verify
analyzer uses `k_max` not `k_used` in the production λ̂ formula
before launch (memory says the fix landed in commit `14fbad6`; quick
source check first).

### Sweep B: fixed-k probe

```
7 k values × 3 seeds × 50 epochs × nccl-async × convergence guard off
k ∈ {2, 4, 8, 16, 32, 64, 128}
```

21 runs, ~3 hours. Direct test of R1' without controller interference.
Likely to be the paper's central R1 figure.

### Sweep C: threshold bracket

```
4 k_max values × 3 seeds × 200 epochs × nccl-async × msf guard
k_max ∈ {default, 2×default, 8×default, 32×default}
```

12 runs, ~6 hours. Locates the Pecora-Carroll synchronization threshold
empirically. If accuracy collapses between two k_max values, the
threshold sits between them.

### Sweep D: D*(LR) extended schedule

```
3 LR schedules × 3 seeds × 200 epochs × nccl-async × msf guard
schedules: {0.3 → 0.03 → 0.003, 0.1 → 0.01 → 0.001, 1.0 → 0.1 → 0.01}
```

9 runs, ~4.5 hours. Fits R2'.

### Sweep E: controller A/B (post-Sweep-A-D)

```
5 seeds × 3 guards × nccl-async × 200 epochs
guards: {current, msf v1, C1' v2}
```

15 runs, ~7.5 hours. The headline parity (or beating) experiment.

**Total budget across all sweeps:** ~26 hours across 5 overnights or
~3 days continuous. Sweeps A-D can run independently and merge into
sweep E only if all return non-killed. Sweep E is the gate to the
paper.

---

## Open questions (post-v2)

- **Where exactly does the Pecora-Carroll threshold sit for ResNet-20
  CIFAR-10 at the standard LR schedule?** Sweep C answers
  qualitatively; quantitative localization may need finer-grained
  k_max values.
- **Does D*(LR) follow a clean power law, or are there crossover
  scales?** Sweep D answers. If multiple LRs produce overlapping D*
  bands, the OU setpoint is more complex than the simple
  `D*_base · (LR/LR_init)^p` form.
- **Does the by-k slope estimator generalize across model
  architectures?** Phase 1 used ResNet-20. SmolLM-135M (per
  `project_smollm_bench_arc.md`) is the next bench model and would
  test transfer. Open until Sweep E validates the controller on
  ResNet, then re-test on SmolLM.
- **Heterogeneous-rank effect on the threshold.** Pecora-Carroll
  classical theory assumes identical oscillators; non-identical
  ranks (different speeds, different effective batch sizes via per-
  rank dispatch shares) may shift `k*` per rank. Worth investigating
  whether a single `k*` exists or whether each rank needs its own
  threshold (and the network threshold is the most conservative).
- **Is there a momentum analog?** Current D measures weight drift.
  Optimizer momentum state has its own (probably faster) drift
  pattern. Adding momentum-D to the analyzer is a single-event
  payload extension; whether it carries different signal is an open
  question.
- **CUSUM threshold-tuning under OU noise.** Standard CUSUM design
  assumes Gaussian residuals; the OU stationary distribution at the
  meta scale is Gaussian asymptotically but with a slow (LR-
  dependent) variance that needs tracking for tail-quantile
  thresholds. Probably 30 LOC, but the design needs validation
  before claiming the meta-CUSUM result.

---

## Notation deltas from v1

The v1 [glossary](msf-cadence-control.md#glossary) is mostly preserved.
v2 adds and reframes the following:

| Symbol | v1 meaning | v2 meaning |
|---|---|---|
| `λ_T` | True transversal Lyapunov, single scale | Within-cycle Lyapunov exponent at the bottom (per-rank) scale, function of LR |
| `λ̂_t` | Estimator of λ_T from endpoint-to-endpoint formula | Deprecated as Lyapunov estimator; surviving as a fluctuation-around-D* signal at the meta scale |
| `D*(LR)` | Not in v1 | OU setpoint of the meta-oscillator; the "consensus attractor" |
| `D_mean(t)` | Implicit (per-event mean across ranks) | Explicit: the meta-oscillator amplitude observable; canonical basis for top-scale analysis |
| `k_used` | Just a stat | First-class axis for R1'; the within-cycle clock |
| `k*(LR)` | Not in v1 | Pecora-Carroll synchronization threshold cadence; coupling-parameter reference point |
| `ε_i` | Not in v1 | Per-rank early-cycle drift seed; y-intercept of by-k OLS |
| `μ` | Not in v1 | Cadence safety margin above threshold (`k_max = μ · k*(LR)`) |

The v1 doc remains the historical reference; v2 supersedes it for the
go-forward research program.

---

## References

All v1 references preserved. Additional context for the two-scale
framing:

- Boccaletti, Kurths, Osipov, Valladares, Zhou, 2002. "The
  synchronization of chaotic systems." Physics Reports 366(1-2),
  1-101. https://doi.org/10.1016/S0370-1573(02)00137-0. Comprehensive
  review of synchronization regimes including phase, generalized,
  and complete sync; covers the chaotic-but-coupled regime that
  bottom-scale per-rank dynamics inhabit.
- McLachlan, Krishnan, 2008. "The EM Algorithm and Extensions" (2nd
  ed.), Wiley. Standard reference for OU stationary distributions
  used in the meta-scale residual analysis.
- Page, 1954. "Continuous Inspection Schemes." Biometrika 41(1/2),
  100-115. Original CUSUM paper; basis for the meta-CUSUM detector
  in R5'.

The v1 references for Pecora-Carroll, Arenas et al., Izmailov et al.,
and HetSeq remain the load-bearing citations.
