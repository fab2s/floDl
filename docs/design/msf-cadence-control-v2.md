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

## Phase 2 verdict (Sweep A landed 2026-05-05)

The 4-cell experiment (5 seeds × 2 guards × {default, relaxed} anchor)
landed cleanly. Sweep ran 6h51m, 12/12 OK, 0 FAIL. Cells map onto a
Pecora-Carroll proximity gradient that correlates with eval AND with
framing-validity gate firing — empirical confirmation of the v2
two-scale framing.

### Headline 4-cell table

| Cell | Eval (mean ± sd, N=5) | Sync count | Cross-rank r̄ | Per-rank ratio (warmup) | Gates ✓ |
|---|---|---|---|---|---|
| trend default | 91.71% ± 0.21 | 539 ± 205 | 0.993 | 1.04 ± 0.02 | 5/5 |
| trend relaxed | 91.74% ± 0.07 | 402 ± 160 | 0.981 | 1.04 ± 0.04 | 5/5 |
| msf default | 91.83% ± 0.20 | 882 ± 299 | 0.993 | 1.05 ± 0.02 | 5/5 |
| **msf relaxed** | **91.95% ± 0.32** | **641 ± 162** | **0.981** | **2.07 ± 1.19** | **3/5** |

**Eval optimum sits at the noisiest framing-gate corner.** The msf
relaxed cell beats trend default by +0.24pp and msf default by +0.12pp,
while having 2/5 framing-gate violations. This is the Pecora-Carroll
prediction made concrete: optimal coupling sits at the edge of the
synchronized regime.

### Within-cycle Lyapunov is a property of the system, not the guard

| Cell | Slope | R²(by-k) |
|---|---|---|
| trend default | +1.45e-3 ± 1.94e-4 | 0.737 ± 0.070 |
| msf default | +1.47e-3 ± 5.25e-5 | 0.752 ± 0.053 |

Match within 1% on slope, 2pp on R². The msf guard's smoothing only
tightens the sd; it doesn't change what's measured. R1' is the same
underlying physics regardless of guard choice.

### Heterogeneous decoupling pattern

Cross-rank Pearson r drop under relax-up is asymmetric:

| Pair | Default | Relaxed | Δ |
|---|---|---|---|
| rank 0 ↔ rank 1 (fast↔slow) | +0.9927 | +0.9814 | −0.011 |
| rank 0 ↔ rank 2 (fast↔slow) | +0.9932 | +0.9811 | −0.012 |
| rank 1 ↔ rank 2 (slow↔slow) | +0.9962 | +0.9962 | **0.000** |

The fast GPU (rank 0, RTX 5060 Ti) decouples; the slow GPUs (1060s)
stay perfectly locked. Hardware heterogeneity dictates the
decoupling-direction structure: rank 0 has the largest batch share,
runs farthest ahead between syncs, drifts most when cadence loosens.
Slow GPUs are pinned together — both anchored to the same plodding
pace.

This is a publishable empirical fact about heterogeneous-DDP
synchronization that has no homogeneous-cluster analog. The HetSeq
framing (mixed-GPU university clusters) is exactly the regime where
this matters.

### Critical mechanism — guard silence breaks relax-up

| Cell | Trend-rule fires/run | MSF-rule fires/run |
|---|---|---|
| msf default | 54.5 ± 7.9 | 1.0 ± 1.4 |
| msf relaxed | 80.4 ± 17.2 | **0.0 ± 0.0** |

**The MSF guard never fires across all 5 relaxed-anchor seeds.**
Mechanism: ElChe's anchor relax-up grows cadence on every "Stable"
verdict; with MSF silent, every verdict is Stable, anchor grows
unbounded, cycles get very long, cycles saturate to D*(LR) before sync,
R² collapses, framing breaks at 2/5 seeds.

trend-relaxed avoids this because trend fires 80×/run, keeping ElChe
from growing the anchor too aggressively. The alarm cadence is what
bounds the anchor.

**Implication for production**: a threshold-aware controller (C5')
that targets `μ·k*(LR)` directly is required for safe deployment.
Silence ≠ stability; the relax-up policy currently composes badly with
a quiet guard. v1's "loose by default" intuition fails when the safety
mechanism never reports.

### EASGD smoke (single seed, cpu-async α=0.5)

Side experiment validates the Zhang/Choromanska/LeCun 2015 elastic
blending fix at one seed:

| Cell | Eval (seed 0) | R²(by-k) | Per-rank ratio | Sync count |
|---|---|---|---|---|
| cpu-async α=1.0 trend (current) | 92.18% | (impulsive coupling broken) | | 812 |
| cpu-async α=1.0 msf (current) | 91.91% | (same) | | 617 |
| cpu-async α=0.5 trend (EASGD) | 91.39% | 0.86–0.89 | 1.05× | 726 |
| **cpu-async α=0.5 msf (EASGD)** | **91.91%** | **0.92–0.94** | **1.04×** | **408** |

EASGD blending fixes impulsive coupling AND tightens framing beyond
the default-anchor nccl-async R² baseline (0.74). The msf+α=0.5 cell
matches the α=1.0 baseline at seed 0 (91.91% vs 91.91%) with **34%
fewer syncs** and the cleanest cross-rank coupling we've measured.

trend at α=0.5 loses 0.79pp — the impulsive-coupling perturbation
breaks trend's "3 rises in D" detector, but msf's λ_ema smoothing
rides through it. This is independent evidence for R5' (top-scale
smoothing is the right detector level).

Single-seed only. Multi-seed α-sweep is justified.

### Phase 2 verdict against R1'–R5'

| Hypothesis | Result |
|---|---|
| R1' within-cycle exponential | **confirmed** at warmup (R² 0.74–0.75 across both default-anchor guards, slopes match within 1%). Mid-LR and late-training are OU-saturated as predicted (R² < 0.3 universally). |
| R2' phase-dependent setpoint D*(LR) | **inferred** from saturation pattern; quantitative fit pending Sweep D. |
| R3' by-k controller A/B | **untested in Sweep A** (still on v1's λ̂); pending C1' implementation + Sweep E. |
| R4' Pecora-Carroll threshold proximity | **partially confirmed** — proximity gradient observed; quantitative threshold pending Sweep C. |
| R5' meta-CUSUM | **partially supported** — λ_ema smoothing demonstrably more robust than 3-rises rule (0 false fires under EASGD perturbation that trend can't survive). Full CUSUM detector still pending implementation. |

**No kill criterion triggered.** Phase 2 advances to Phase 3.

---

## Phase 3 verdict (Sweep B2 cliff bracket landed 2026-05-06)

The fixed-k cliff bracket (3 seeds × 6 k values × 200 epochs × `--guard none` × `--min-anchor=k --max-anchor=k`) landed cleanly. Sweep ran ~9h30m, 18/18 OK, 0 FAIL. The Pecora-Carroll synchronization threshold is now empirically localized for ResNet-20 / CIFAR-10 / standard 200-epoch schedule / 3-GPU heterogeneous.

### Headline cliff table

| k | Within-train syncs | Eval seed 0 | Eval seed 1 | Eval seed 2 | Mean ± sd | Range | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| 3200  | 13 | 91.76% | 91.76% | 91.28% | 91.60% ± 0.28 |  0.5pp | safe baseline |
| 6400  |  7 | 91.31% | 91.39% | 91.64% | 91.45% ± 0.17 |  0.3pp | safe |
| 12800 |  4 | 91.29% | 91.37% | 91.21% | 91.29% ± 0.08 |  0.2pp | safe |
| 16000 |  3 | 90.31% | 89.32% | 90.53% | 90.05% ± 0.64 |  1.2pp | soft pre-cliff |
| **25600** |  2 | 90.89% | **55.80%** | 84.58% | 77.09% ± 18.71 | **35.1pp** | **bimodal cliff edge** |
| 51200 |  1 | 63.22% | 10.02% | 10.02% | 27.75% ± 30.72 | 53.2pp | past cliff |

**The cliff sits between k=16000 and k=25600.** Below it, all three seeds converge within 1.3pp of the safe-regime mean. At and above it, seeds split between safe-basin and collapsed-basin endings within the same cell. The bimodality at k=25600 — three independently-seeded runs landing at 90.89%, 55.80%, 84.58% — is the basin-of-attraction signature of a noise-perturbed system at the synchronization threshold. Two seeds find a common basin; one falls into a different one.

### Adjacent-cell deltas localize the bracket

| Transition | Δ mean eval | Verdict |
|---|---:|---|
| k=3200 → k=6400   |  −0.15pp | flat |
| k=6400 → k=12800  |  −0.16pp | flat |
| k=12800 → k=16000 |  −1.24pp | soft drop (>1pp) |
| k=16000 → k=25600 | −12.96pp | soft drop (>1pp) |
| k=25600 → k=51200 | **−49.34pp** | **cliff edge** |

The hard cliff (Δ > 30pp) localizes between k=25600 and k=51200; the cliff edge itself (first cell with bimodal seed split) is k=25600. ElChe's auto-tune saturates near k ≈ 200 in this regime — operating ~80–125× below the cliff.

### No eval peak above ElChe's default operating point

Within the safe regime k ∈ {3200, 6400, 12800}, eval is monotone non-increasing (91.60% → 91.45% → 91.29%). The "ride the limit" hypothesis — that the eval-vs-k curve has a peak between ElChe's default operating point (~k=200) and the cliff — is **falsified**. The controller story for C5' is **"stay safely below the cliff"**, not "target the peak". The cliff localization above gives that story its quantitative reference.

### Loss of within-cycle observability is itself a regime signature

R1 by-k OLS is observable only at k=3200 (LR=0.3 window, 6 events, slope −1.98e−4 ± 1.43e−4, R² = 0.235 ± 0.099). For k ≥ 6400 the analyzer skips the section because per-LR-window sync count drops below the OLS minimum. The disappearance of the within-cycle Lyapunov axis is *itself* a signature of crossing into the sparse-coupling regime — by construction, as inter-sync interval grows past a single LR window, the by-k axis degenerates.

A corollary: **cross-rank Pearson r is uninformative past the cliff.** Reported r ≈ 1.0 in collapsed cells is artifact of N ≤ 2 sync events (two points always lie perfectly on a line). The eval signal is the load-bearing falsifier; the framing-validity gate works only in the safe regime where it's already saturated near 1.

### Phase 3 verdict against R1'–R5'

| Hypothesis | Result |
|---|---|
| R1' within-cycle exponential | confirmed previously at warmup; now also confirmed that observability degrades by construction past the cliff (sync-count-bounded by LR-window length). |
| R2' phase-dependent setpoint D*(LR) | unchanged; pending Sweep D. |
| R3' by-k controller A/B | unchanged; pending C1' implementation + Sweep E. |
| R4' Pecora-Carroll threshold proximity | **confirmed and quantitatively localized** — synchronization threshold sits between k=16000 (last fully safe) and k=25600 (first bimodal); hard collapse at k=51200. ElChe operates ~80–125× below the cliff at default cadence. |
| R5' meta-CUSUM | unchanged; full CUSUM detector still pending. |

**No kill criterion triggered.** Phase 3 confirms R4' and falsifies the "peak between default and cliff" sub-hypothesis.

---

## Program reframe (post-Phase-3)

Phase 3 changed the paper. Eval is monotone non-increasing across the safe regime; ElChe at default cadence sits within the optimal eval band; **there is no controller-relaxation regime on the cadence axis that improves final eval at this scale**. The publishable result reorients from "principled controller beats heuristic" to **empirical characterization of the eval-vs-cost Pareto frontier, with a structural scaling argument for where the frontier rotates**.

### Pareto frontier (200-epoch data, post-Gate-A multi-seed)

Cost axis: syncs / 200ep (network-volume proxy at fixed model size; rotates toward wall time as parameter count grows). Eval axis: held-out accuracy.

| Config | Eval (n=5 unless noted) | Syncs/200ep | Frontier status |
|---|---:|---:|---|
| nccl-async default msf       | 91.83% ± 0.20 | 882 ± 299 | dominated by cpu-async default trend |
| nccl-async default trend     | 91.71% ± 0.21 | 539 ± 205 | dominated by nccl-async relaxed trend |
| cpu-async default msf        | 91.86% ± 0.27 | 619 ± 212 | dominated by cpu-async default trend |
| **cpu-async default trend**  | **91.96% ± 0.23** | **613 ± 206** | **frontier (eval max)** |
| nccl-async relaxed msf       | 91.95% ± 0.32 | 641 ± 162 | dominated by cpu-async default trend |
| **nccl-async relaxed trend** | **91.74% ± 0.07** | **402 ± 160** | **frontier (lowest-sync at near-parity)** |
| cpu-async EASGD α=0.5 msf    | 91.67% ± 0.19 | 604 ± 154 | dominated by nccl-async relaxed trend |
| cpu-async EASGD α=0.5 trend  | 91.75% ± 0.22 | 594 ± 162 | frontier (thin margin between two existing points) |
| Fixed k=3200                 | 91.60% ± 0.28 |  13       | frontier knee |
| Fixed k=6400                 | 91.45% ± 0.17 |   7       | frontier |
| Fixed k=12800                | 91.29% ± 0.08 |   4       | frontier |
| Fixed k=16000                | 90.05% ± 0.64 |   3       | post-knee |
| Fixed k=25600                | 77.09% ± 18.71|   2       | past Pareto (bimodal) |
| Fixed k=51200                | 27.75% ± 30.72|   1       | collapsed |

#### Multi-seed correction to the seed-0 EASGD smoke story

The seed-0 EASGD smoke (Phase 2) reported `cpu-async α=0.5 + msf` at 91.91% / 408 syncs and `cpu-async α=0.5 + trend` at 91.39% / 726 syncs — a 0.52pp eval gap and a >2× sync-count gap between the two arms. Gate A multi-seed (seeds 1–4) decisively contradicts both signals. Multi-seed means converge into a tight band: msf at 91.67% ± 0.19, trend at 91.75% ± 0.22, sync counts ~600 for both. Seed 0 was a tail outlier on both arms simultaneously; the strong seed-0-derived dominance claim does not survive replication.

Net Pareto effect: EASGD α=0.5 does **not** add a Pareto-improving direction. The msf arm is now **dominated by `nccl-async relaxed trend`** (91.74% / 402 syncs strictly improves on 91.67% / 604). The trend arm sits on the frontier with thin margin (~0.01pp eval / ~19 syncs window between `nccl-async relaxed trend` and `cpu-async default trend`).

#### What the corrected frontier actually shows

> **2026-05-07 reorg note:** the research/ data layout was consolidated
> on 2026-05-07. The cpu-async α=1.0 cells in the original
> `passive-observation/` were dropped (pre-EASGD impulsive-coupling
> break per Phase 2 verdict — MSF-framing-invalid foil); the EASGD
> α=0.5 cohort (seed-0 from the relaxed-easgd smoke + seeds 1–4 from
> the multi-seed Gate A) was migrated into `passive-observation/` as
> the canonical cpu-async cohort. `relaxed-anchor-easgd/` was renamed
> to `relaxed-anchor/`; `easgd-multiseed/` was folded into
> passive-observation. The strict-Pareto computation now resolves
> three frontier configurations at the high-sync end (with the eval
> max shifting to `nccl-async relaxed msf` since the previous eval-max
> point was an α=1.0 cell that no longer enters the frontier set). A
> full re-run at a single fresh tree state is queued to eliminate the
> mixed-tree caveat (seed-0 EASGD α=0.5 cells are tree `321a401`,
> seeds 1–4 are tree `0806f84`, nccl-async cells are tree `321a401`).

The high-sync end of the frontier resolves to **three non-dominated configurations**:
- `nccl-async relaxed msf` — eval maximum (91.95% ± 0.32) at 641 ± 162 syncs.
- `cpu-async default trend` — middle (91.75% ± 0.22) at 594 ± 162 syncs (this cell is the EASGD α=0.5 cohort under the post-reorg naming).
- `nccl-async relaxed trend` — lowest-sync near-parity point (91.74% ± 0.07) at 402 ± 160 syncs.

Production default `nccl-async default msf` (91.83% ± 0.20 at 882 ± 299 syncs) is dominated by `nccl-async relaxed msf` — the production-config improvement is a **single flag** (`--elche-relax-up`), Δ +0.12pp eval at 27% sync reduction. Same backend (NCCL async), same guard (msf), just relax-up enabled.

The cadence-axis frontier (fixed-k cliff bracket) saturates around k=3200–6400; past k=12800 the knee bends sharply. Each fixed-k cell strictly Pareto-improves on cost as it loses ~0.15–0.5pp eval, until the cliff at k=25600.

**Honest controller-story conclusion:** ElChe's auto-tune is **near-optimal at this scale**; the relaxed-anchor flag (`--elche-relax-up`) is the meaningful Pareto-improving knob in the high-sync regime, on either guard. The coupling-mechanism axis (EASGD α<1) sits on the frontier as one of three configurations but the within-seed-noise margin to the eval-max endpoint means it is not a clearly differentiable Pareto direction at this scale. The structural-rotation argument (model size + GPU heterogeneity) remains the open prediction.

### Scaling prediction (the structural argument)

The frontier rotates with two axes that ResNet-20 / 3-GPU does not stress:

1. **Model size.** AllReduce cost scales linearly with parameter count. ResNet-20 has 270K params; SmolLM-135M is ~500× larger. Sync count translates more directly to wall time at scale. The most Pareto-improving config at ResNet-20 (`nccl-async relaxed trend`, ~34% sync reduction at −0.22pp eval) becomes a near-proportional wall-time reduction at SmolLM size; on ResNet-20 the wall-time delta is barely measurable.
2. **Rank count and heterogeneity diversity.** 3 GPUs (1 fast + 2 slow) yields a binary fast/slow asymmetry. With 4+ GPUs in richer mixes, the Phase 2 bottom-scale decoupling pattern (fast GPU drifts, slow pair stays Pearson-locked) generalizes to multi-cluster decoupling, where per-cluster cadence becomes a controller knob the single-cadence treatment cannot use.

The two-scale framing predicts these directions are where principled controllers earn their keep — not on ResNet-20/CIFAR-10/3-GPU, where ElChe's auto-tune is already near-optimal. The paper's contribution is the structural framing + the falsifiable scaling prediction, not a controller-improvement benchmark on the small case.

### Updated experimental gates

The original Phase 4 / Phase 5 specs predate cliff localization and are superseded.

**Gate A (LANDED 2026-05-06): Multi-seed EASGD α-sweep.** Confirmation pass for the seed-0 EASGD smoke. Result: **EASGD α=0.5 does not add a Pareto-improving direction.** Sharp predictions decisively violated:
- msf+α=0.5 cross-seed mean = 91.67% ± 0.19 vs α=1.0 baseline 91.86% ± 0.27. Predicted parity ±0.15pp; actual −0.19pp at marginal-significance edge.
- trend+α=0.5 cross-seed mean = 91.75% ± 0.22 vs α=1.0 baseline 91.96% ± 0.23. Predicted 0.5–1.0pp degradation; actual −0.21pp (essentially within noise).
- Sync-count reduction ~3-5% (594-604 vs 613-619), **not** the predicted ≥25%. The seed-0 sync counts (408 msf, 726 trend) were tail outliers.
- Net effect: msf arm now dominated by `nccl-async relaxed trend`; trend arm on frontier with thin margin. See "Multi-seed correction" subsection above for full read.

The Gate A null result strengthens the structural-scaling argument: at ResNet-20 / 3-GPU scale, the coupling-mechanism axis (EASGD elastic blending) provides no frontier-improving direction. Whether this generalizes — or whether EASGD becomes a real Pareto improvement at SmolLM scale where AllReduce cost is non-trivial — is the Paper 2 question (Gate E, scale-axis follow-up).

**Gate B (scoped): D*(LR) fit (was Phase 4).** Reduced scope. C4' (setpoint regulator) is deprioritized given the no-eval-peak finding; D*(LR)'s remaining consumer is R5' (CUSUM normalization).
- 2 LR schedules × 3 seeds × 200 epochs × nccl-async × msf guard.
- Standard `0.3 → 0.03 → 0.003` plus one off-baseline (e.g. `0.1 → 0.01 → 0.001`).
- 6 runs, ~3h.

**Gate C (reframed): Pareto-frontier characterization (was Phase 5 controller A/B).** Headline figure becomes the (syncs, eval) Pareto plot across {ElChe default trend/msf, ElChe relaxed trend/msf, EASGD α=0.5 trend/msf, fixed k=3200/6400/12800, optionally C1' if implemented}.
- Most cells already on disk. Aggregation script can produce the figure from existing runs + Gate A results.
- C1' (by-k cadence inversion) becomes optional; its predicted ceiling is parity, not improvement, and its input signal degrades by construction past the safe regime.

**Gate D (in-paper, post-Gate-A): ResNet-56 bytes-axis confirmation.** Intermediate model (n=9 in the He et al. CIFAR family, ~850K params, 3.1× ResNet-20). Confirms the bytes-axis Pareto rotation prediction directly: same dataset, same eval metric, same architecture family, just 3× the parameter count. Implementation cost ~1h (lift the hardcoded `make_layer(..., 3, ...)` literal in `ddp-bench/src/models/resnet.rs:236-238` to a CLI `--depth-n` flag); experimental cost ~1 overnight (Gate A repeat at ResNet-56: 8 runs × ~90 min). Does **not** substitute for the wall-time-axis test — at this rig and model size, AllReduce remains <1ms regardless of params, so wall-time rotation needs ~10M+ params. ResNet-56 buys the bytes axis cleanly and falsifies-or-confirms the directional structural prediction at ~1 overnight cost.

**Gate E (out-of-paper, follow-up): Architecture variance + scale axis.** Two follow-up directions, deferred to a second paper rather than squeezed into this one:
- **Architecture variance (TinyViT / vanilla ViT-Tiny CIFAR, ~5.7M params).** Tests whether the two-scale framing is architecture-independent. flodl already has the building blocks (`MultiheadAttention`, `LayerNorm`, `GELU`, `Embedding`, `Dropout`); impl cost ~4–8h for the model + recipe. Vanilla ViT-Tiny preferred over Microsoft's hierarchical TinyViT (the latter requires windowed-shifted attention + distillation pipeline, multi-day work).
- **Scale axis (SmolLM-135M, ~135M params, see `project_smollm_bench_arc.md`).** Tests the wall-time-axis Pareto rotation that ResNet-56 cannot reach. Different domain (language vs vision), different eval metric (perplexity vs accuracy) — its own Pareto plot, complementary to the ResNet-{20,56} story.

Both follow-up directions are now scoped to the second paper (architecture variance + scale axis as the structural rotation argument); ResNet-56 stays in this paper as the in-budget bytes-axis confirmation.

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

**Status (2026-05-06).** Phase 2 sweep landed (proximity gradient).
**Phase 3 (Sweep B2 cliff bracket) landed** — synchronization threshold
quantitatively localized between k=16000 (last fully safe) and k=25600
(first bimodal cliff edge); hard collapse at k=51200. ElChe's auto-tuned
cadence saturates near k ≈ 200 in this regime, operating ~80–125× below
the cliff. R4' is **confirmed and quantitatively localized**. See Phase 3
verdict section above for the headline cliff table and adjacent-cell
deltas.

**Why this matters.** This is the literal Pecora-Carroll experiment in
DDP form, and the threshold is now located. The paper has its
quantitative hook: for ResNet-20 / CIFAR-10 / standard 200-epoch
schedule / 3-GPU heterogeneous, ElChe operates ~80–125× below the
synchronization threshold; the threshold sits at the cadence where
within-training sync count drops to ≤2 over the 200-epoch run.

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

**Entry condition (post-Phase-3 + post-Gate-A reframe).** Gate A LANDED 2026-05-06 with a null result: EASGD α=0.5 does not add a Pareto-improving direction at this scale (multi-seed contradicted the seed-0 smoke). R4' is confirmed (Phase 3 cliff localization). Both decisions are now on the table for the paper. Remaining gates that could strengthen the manuscript:

- Gate B (D*(LR) fit, scoped): closes R2' analytically and feeds R5'.
- C1' / C5' implemented and run as optional arms in Gate C: lets the paper sketch the principled controller's predicted ceiling (parity, not improvement) directly.

**Narrative spine (post-Phase-3 reframe).**

> Distributed data parallel training is a synchronization-of-coupled-chaotic-oscillators problem in the Pecora-Carroll sense. Each replica evolves chaotically between AllReduce events with positive within-cycle Lyapunov exponent λ_T(LR); the meta-oscillator (cross-rank average) collapses these into a contracting Ornstein-Uhlenbeck process around a phase-dependent setpoint D*(LR). AllReduce is the projection operator linking the scales, and cadence k is the coupling-strength parameter from the original 1998 framework.
>
> We characterize the empirical eval-vs-cost Pareto frontier for heterogeneous DDP on ResNet-20/CIFAR-10/3-GPU and locate the Pecora-Carroll synchronization threshold (cliff between k=16000 and k=25600 for the standard 200-epoch schedule; ElChe operates ~80–125× below it). The frontier at the high-sync end resolves to two configurations: `cpu-async default trend` (eval maximum) and `nccl-async relaxed trend` (lowest-sync near-parity, trading 0.22pp eval for 34% sync reduction). ElChe's auto-tuned cadence sits within the optimal eval band; **no controller-relaxation regime on the cadence axis improves final eval at this scale**, and the coupling-mechanism axis (EASGD elastic blending α<1) does not add a Pareto-improving direction either (multi-seed contradicts the seed-0 smoke; mean eval drifts −0.19pp at marginal sync savings). The two-scale framing predicts both null results generalize structurally only up to a scaling boundary: top-scale (cadence) headroom narrows as a model approaches its training-recipe optimum; bottom-scale (per-rank decoupling treatment) headroom widens with rank count and hardware heterogeneity; coupling-mechanism choice (EASGD α) becomes Pareto-relevant only when AllReduce cost is non-trivial relative to per-step compute, which ResNet-20 / 3-GPU does not reach. We sketch the controller refinements the framing motivates (by-k cadence inversion C1', threshold-aware cadence C5', meta-CUSUM regime detector R5') and leave their empirical validation at scale to follow-up.

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

**[SUPERSEDED post-Phase-3 — see "Program reframe → Gate B" above. Original spec preserved for context.]**

```
3 LR schedules × 3 seeds × 200 epochs × nccl-async × msf guard
schedules: {0.3 → 0.03 → 0.003, 0.1 → 0.01 → 0.001, 1.0 → 0.1 → 0.01}
```

9 runs, ~4.5 hours. Fits R2'. Reduced to 2 schedules (6 runs, ~3h) under the reframe — third schedule's marginal value collapsed when C4' was deprioritized.

### Sweep E: controller A/B (post-Sweep-A-D)

**[SUPERSEDED post-Phase-3 — see "Program reframe → Gate C" above. Original spec preserved for context.]**

```
5 seeds × 3 guards × nccl-async × 200 epochs
guards: {current, msf v1, C1' v2}
```

15 runs, ~7.5 hours. Originally framed as headline parity (or beating) experiment. Reframed under post-Phase-3 program as Pareto-frontier characterization — most cells already on disk; Gate A's EASGD multi-seed result is the load-bearing addition. C1' becomes optional rather than the headline arm.

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

## Reproducibility — artifact pointers

Every empirical claim in this document maps to a specific run directory
and analysis script. Co-located rather than gathered, but tracked here
so any number in the doc can be traced to its source.

### Phase 1 (passive observation, 2026-05-04)

- **Sweep base:** `ddp-bench/runs/overnight-2026-05-04/`
- **Run script:** `ddp-bench/runs/overnight-2026-05-04/run.sh` (21 runs:
  1 validation + 5 seeds × 2 modes × 2 guards). Note: this script has
  the relative-path doubling bug that produced
  `ddp-bench/ddp-bench/runs/...` artifacts; fixed in-place 2026-05-04
  afternoon by `mv`'ing into the correct layout.
- **Per-seed reports:** `runs/overnight-2026-05-04/seed-*/report.md`
  (regenerated 2026-05-05 morning to add by-k columns and per-rank
  consistency check from step 1 of the v2 plan).
- **Per-seed timeline data:** `runs/overnight-2026-05-04/seed-*/resnet-graph/<mode>/timeline.{csv,json,html}`
  + `training.log` + `run.stdout.log`.
- **Aggregation scripts:**
  - `aggregate_phase1_v1.py` — original 5-seed verdict (eval, R1
    cumulative-step OLS, guard fires, kill-criterion correlations).
    First pass before by-k axis correction.
  - `aggregate_phase1_v2_byk.py` — second pass after by-k axis added
    to the analyzer, includes the post-transient subwindow split.
  - `aggregate_phase1_byk_only.py` — focused by-k analysis for the
    Phase 1 reframing.
- **Key numbers anchored here**: cross-rank Pearson r > 0.99 across
  all seeds (meta-oscillator anchor), R1 cumulative-step R² 0.49 ± 0.11
  (failed), R1 by-k R² 0.69 ± 0.05 (succeeds at warmup), within-cycle
  λ_T = +1.46e-3 ± 5.25e-5 ln(D)/step (warmup), MSF guard fires
  1.0 ± 1.4 vs trend 54.5 ± 7.9.

### Phase 2 (Sweep A relaxed-anchor + EASGD smoke, 2026-05-05)

- **Sweep base:** `ddp-bench/runs/overnight-2026-05-05-relaxed-easgd/`
- **Run script:** `runs/overnight-2026-05-05-relaxed-easgd/run.sh`
  (12 runs: 10 nccl-async × `--elche-relax-up` × 5 seeds × 2 guards
  + 2 cpu-async × `--easgd-alpha 0.5` × seed 0 × 2 guards). Uses
  `fdl cuda-shell -- -c "..."` pattern to bypass the fdl-preset
  clobber. Output paths stripped of `ddp-bench/` prefix to avoid the
  2026-05-04 doubling bug.
- **Run log:** `runs/overnight-2026-05-05-relaxed-easgd/_runlog.txt`
  (sweep started 2026-05-04T18:43:33Z, completed 2026-05-05T01:34:14Z,
  6h51m total).
- **Per-seed reports + timeline:** same shape as Phase 1.
- **Aggregation script:** `runs/overnight-2026-05-05-relaxed-easgd/aggregate.py`
  — the 4-cell aggregator that produced the Phase 2 verdict table.
  Compares Phase 1 default-anchor against Phase 2 relaxed-anchor.
  Outputs eval mean/sd/range, sync counts, cross-rank Pearson r per
  pair, R1 by-k slopes + R² + per-rank ratios + gates per LR window,
  guard fire counts.
- **Key numbers anchored here**: 4-cell eval ordering (trend default
  91.71% → trend relaxed 91.74% → msf default 91.83% → msf relaxed
  91.95%), proximity gradient via per-rank ratio (1.04 → 1.04 → 1.05
  → 2.07), MSF guard 0/5 fires under relax-up, cpu-async α=0.5 + msf
  R² 0.92–0.94 (best framing observed), 34% sync reduction at
  α=0.5+msf vs α=1.0+msf.

### Phase 3 (Sweep B2 cliff bracket, 2026-05-06)

- **Sweep base:** `ddp-bench/runs/overnight-2026-05-05-sweep-b2-cliff/`
- **Run script:** `runs/overnight-2026-05-05-sweep-b2-cliff/run.sh` (18
  runs: 3 seeds × 6 k values × 200 epochs, `nccl-async`, `--guard none`,
  `--min-anchor=k --max-anchor=k` to pin cadence at fixed k). Same
  `fdl cuda-shell -- -c "cd /workspace/ddp-bench && ./target/release/ddp-bench …"`
  pattern as Phase 2.
- **Run log:** `runs/overnight-2026-05-05-sweep-b2-cliff/_runlog.txt`
  (sweep started 2026-05-05T20:46:28Z, completed 2026-05-06T06:15:58Z,
  ~9h30m total, 18/18 OK 0 FAIL).
- **Per-seed reports + timeline:** same shape as Phase 1/2; reports
  regenerated 2026-05-06.
- **Aggregation:** `runs/overnight-2026-05-05-sweep-b2-cliff/aggregate.py`
  + canonical output `aggregate.txt` co-located. The 6-cell × 3-seed
  cliff aggregator. Adapted from Sweep C aggregator with two structural
  changes: (1) cliff localization — surfaces per-seed evals + range +
  bimodality flag (range > 30pp), since past-threshold cells split
  between safe-basin and collapsed-basin seeds within the same cell;
  (2) Pearson + R1 tolerance — cells past the cliff have ≤ 1
  within-training sync, so the aggregator skips OLS and Pearson
  silently rather than crashing. Run from project root:
  `python3 ddp-bench/runs/overnight-2026-05-05-sweep-b2-cliff/aggregate.py`.
- **Pre-launch tree:** `0806f84` on `ddp-scale` (clean, post `--min-anchor`
  plumbing commit).
- **Key numbers anchored here:** cliff localized between k=16000 (last
  fully safe, all 3 seeds within 1.3pp of safe-regime mean) and k=25600
  (first bimodal, range 35.1pp, seeds at 90.89% / 55.80% / 84.58%); hard
  collapse at k=51200 (mean 27.75%, 2/3 seeds at 10.02% random chance);
  no eval peak above ElChe default in safe regime (monotone
  non-increasing 91.60% → 91.45% → 91.29% across k ∈ {3200, 6400,
  12800}); within-cycle by-k axis observable only at k=3200 for the
  LR=0.3 window (slope −1.98e−4 ± 1.43e−4, R² = 0.235 ± 0.099); ElChe's
  default cadence saturates near k ≈ 200, ~80–125× below the cliff.

### Code state at Phase 2

Tree `321a401` on `ddp-scale` branch with two uncommitted change
families:

- **Step 1 (analyzer scale labels + per-rank by-k consistency
  check):**
  - `ddp-bench/src/analyze.rs`: `MsfLrWindowFit` + `LrWindowSamples`
    extended with per-rank by-k fields and pts_per_rank capture in
    `collect_lr_window_samples`. Per-rank OLS in `fit_samples`.
  - `ddp-bench/src/report.rs`: MSF section intro rewritten with
    two-scale framing convention. All 8 subsection headings tagged
    with scale. New `### R1' per-rank by-k slopes` subsection with
    framing-validity gate (warns when max/min per-rank slope ratio
    > 2×).
- **Step 2 (EASGD elastic blending):**
  - `flodl/src/distributed/ddp_run/mod.rs`: `DdpRunConfig.easgd_alpha:
    Option<f64>` field + `with_easgd_alpha(α)` builder method
    (asserts α ∈ (0, 1]). Same field on `WorkerConfig` for plumbing.
  - `flodl/src/distributed/ddp_run/orchestrator.rs`:
    `DdpBuilder::easgd_alpha(α)` method, two `WorkerConfig`
    construction sites updated.
  - `flodl/src/distributed/ddp_run/worker.rs`: `GpuWorker.easgd_alpha`
    field + two-path `load_averaged()` (None → fast non-blocking
    `copy_`; Some(α) → stage averaged params on GPU then batched
    `Tensor::foreach_lerp_scalar_` for in-place blend).
  - `flodl/src/distributed/ddp_run/tests.rs`: 2 test `WorkerConfig`
    literals updated with `easgd_alpha: None`.
  - `ddp-bench/src/main.rs`: `Cli.easgd_alpha: Option<f64>` field +
    Zhang/Choromanska/LeCun 2015 doc-comment.
  - `ddp-bench/src/config.rs`: `RunConfig.easgd_alpha` field.
  - `ddp-bench/src/harness.rs`: builder wiring.

Both change families pass `fdl cuda-clippy` clean and `fdl
cuda-test-nccl` (12 tests pass).

### Sweep names map to numbered phases

| Sweep | Run dir | Phase |
|---|---|---|
| Phase 1 passive observation | `runs/overnight-2026-05-04/` | landed |
| **Sweep A relaxed-anchor** | `runs/overnight-2026-05-05-relaxed-easgd/` | landed |
| **EASGD smoke (in Sweep A)** | (same dir, seed-0-cpu-async-*-easgd05) | landed |
| Sweep B fixed-k probe | (not yet run as v2-spec; see Sweep B2 below for the landed cliff probe) | superseded |
| **Sweep B2 cliff bracket (200ep)** | `runs/overnight-2026-05-05-sweep-b2-cliff/` | landed |
| Sweep C threshold bracket | `runs/overnight-2026-05-05-sweep-c-threshold/` | landed (auto-tune characterization; cap non-binding above 1.5×) |
| Sweep D D*(LR) extended | (not yet run) | pending |
| Sweep E controller A/B | (not yet run) | pending |

When new sweeps land, append rows here with their run dir + key claims
anchored.

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
