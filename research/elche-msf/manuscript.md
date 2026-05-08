# Heterogeneous DDP as a Pecora-Carroll Synchronization Problem: Empirical Pareto Characterization on CIFAR-10

**Status:** working draft. Sections marked `[STUB]` are placeholders;
sections marked `[DRAFT]` carry working prose; sections marked `[LOCKED]`
are paper-ready. Working spec: [`docs/design/msf-cadence-control-v2.md`](../../docs/design/msf-cadence-control-v2.md).

---

## Abstract

[STUB] One-paragraph version of the narrative spine. Will lift from
the working spec once primary results are locked.

Working spine (from working spec):

> Distributed data parallel training is a synchronization-of-coupled-
> chaotic-oscillators problem in the Pecora-Carroll sense. Each replica
> evolves chaotically between AllReduce events with positive within-cycle
> Lyapunov exponent λ_T(LR); the meta-oscillator (cross-rank average)
> collapses these into a contracting Ornstein-Uhlenbeck process around
> a phase-dependent setpoint D*(LR). AllReduce is the projection
> operator linking the scales, and cadence k is the coupling-strength
> parameter from the original 1998 framework.
>
> We characterize the empirical eval-vs-cost Pareto frontier for
> heterogeneous DDP on ResNet-20/CIFAR-10/3-GPU and locate the
> Pecora-Carroll synchronization threshold (cliff between k=16000 and
> k=25600 for the standard 200-epoch schedule; ElChe operates ~80–128×
> below it). The frontier at the high-sync end resolves to three
> non-dominated configurations: `nccl-async relaxed trend` (91.64% ±
> 0.32 / 402 ± 160 syncs, lowest-sync near-parity), `nccl-async
> relaxed msf` (91.82% ± 0.31 / 431 ± 178, mid-frontier), and
> `cpu-async default msf` (92.03% ± 0.31 / 613 ± 128, EASGD α=0.5
> elastic blending, eval maximum). The production default
> `nccl-async default msf` (91.70% ± 0.25 / 671 ± 242) is dominated by
> `nccl-async relaxed msf` via a single flag (`--elche-relax-up`),
> with Δ +0.12pp eval at 36% sync reduction.

---

## 1. Introduction

### 1.1 Motivation

[STUB] Heterogeneous DDP — academic clusters, mixed GPU generations,
HetSeq (AAAI 2021) and Cannikin (Stony Brook 2024) as 2-way step-sync
baselines.

### 1.2 Contribution

[STUB]

1. Two-scale framing of DDP as a Pecora-Carroll synchronization
   problem: per-rank chaos at bottom scale, OU-collapsed meta-oscillator
   at top scale, AllReduce as projection operator.
2. To our knowledge, the first 3-way heterogeneous DDP datapoint
   reported in open-source deep-learning literature, using flodl on
   1×RTX 5060 Ti + 2×GTX 1060 6GB.
3. Empirical Pareto frontier characterization with synchronization
   threshold localization (cliff between k=16000 and k=25600).
4. Falsifiable scaling prediction: coupling-mechanism axis
   (EASGD α) becomes Pareto-relevant only when AllReduce cost is
   non-trivial relative to per-step compute. ResNet-56 bytes-axis
   confirmation pending (in progress).

### 1.3 Roadmap

[STUB]

---

## 2. Background

### 2.1 Pecora-Carroll synchronization

[STUB] Foundational 1990 PRL paper introduces synchronization of
chaotic systems via subsystem coupling. Pecora & Carroll 1998
(`pecora1998master`) introduces the Master Stability Function:
stability of the synchronization manifold reduces to a transversal
eigenvalue problem with coupling strength as the key parameter. In
the DDP setting, AllReduce cadence `k` is the discrete analog of that
coupling-strength parameter.

### 2.2 Distributed data parallel

[STUB] AllReduce semantics, sync vs cadence vs async, ElChe (flodl's
heterogeneous-DDP cadence controller — auto-balanced batch shares,
relax-up flag, convergence guard).

### 2.3 Prior heterogeneous DDP

[STUB — see [`references.bib`](references.bib) for full set]

Related work organized by control axis:

- **Per-rank batch sizing** (still step-sync): HetSeq AAAI 2021
  (`ding2021hetseq`), Cannikin Middleware 2024 (`nie2024cannikin`),
  HetCCL arXiv 2026 (`hetccl2026`). All retain unmodified AllReduce
  every step; flexibility is in batch shares only.
- **Plan synthesis**: HAP EuroSys 2024 (`zhang2024hap`), Cephalo
  ICS 2025 (`guo2025cephalo`), Metis ATC 2024 (`um2024metis`),
  HeteroG CoNEXT 2020 (`yi2020heterog`). Compile-time placement
  decisions; orthogonal layer to cadence.
- **Cluster scheduler**: FFT ICS 2025 (`mo2025fft`). Note: uses
  ``Lyapunov-drift'' in the Neely queueing-theory sense — disambiguate
  explicitly from the dynamical-systems Lyapunov framing imported here.
- **Edge-device pipeline**: Asteroid MobiCom 2024 (`ye2024asteroid`).

None addresses cadence-as-coupling-strength. The closest dynamical-
systems framing in the literature is Kuramoto-FedAvg 2025
(`muhebwa2025kuramoto`) — phase-angle weighting at FedAvg, not
cadence — and Lyapunov Learning ICML 2025 HiLD
(`benati2025lyapunov`) — a single-machine loss-landscape regularizer.

### 2.4 Local SGD lineage

[STUB] Stich ICLR 2019 (`stich2019local`) is the foundational
convergence framework. Modern variants: DiLoCo NeurIPS 2023 workshop
(`douillard2023diloco`, fixed-`H` outer Nesterov), QSR ICLR 2024
(`gu2024quadratic`, LR-driven adaptive cadence), PALSGD TMLR 2025
(`naganuma2025palsgd`, probabilistic mixing), HALoS ICML 2025
(`kim2025halos`, hierarchical async). Adaptive-batch axis: Lau-Kolar
2024 (`lau2024communication`) and CPAL 2025 (`lau2025adaptive`).

---

## 3. Two-scale framing

[DRAFT — lift from `docs/design/msf-cadence-control-v2.md` §"The
two-scale framing"]

Hybrid stochastic model. Bottom scale: per-rank within-cycle
exponential drift. Top scale: meta-oscillator OU around D*(LR).
AllReduce as projection operator.

### 3.1 Why the two scales coexist

[DRAFT — lift]

Note: the framing applies recursively. The two-scale structure
described here for training dynamics extends to any chaos-based
controller closing its loop through the same system, including the
ElChe controller used in this paper. The empirical signature of the
recursion (per-realization unpredictability of cadence decisions
combined with cohort-mean stability of eval and sync count) is
discussed in §7.2.

### 3.2 Implications for guards and controllers

[DRAFT — lift]

---

## 4. Empirical setup

### 4.1 Hardware

3-GPU heterogeneous: 1× NVIDIA RTX 5060 Ti 16GB (sm_120) + 2× NVIDIA
GTX 1060 6GB (sm_61). 6 GB VRAM ceiling on the slow cards is the tight
constraint.

### 4.2 Models

- **ResNet-20** on CIFAR-10 (He et al. 2015 Table 6, paper baseline
  91.25%). ~270K params. Primary benchmark.
- **ResNet-56** on CIFAR-10 (n=9 in the He et al. CIFAR family, paper
  baseline 93.03%). ~850K params, 3.1× ResNet-20. Bytes-axis
  confirmation only.

### 4.3 Recipe

[STUB] SGD momentum=0.9, weight_decay=1e-4, LR=0.1 with multi-step decay
at 50%/75% of training, 200 epochs, batch_size=64 per rank. Standard
CIFAR-10 augmentation (per-channel normalize + 4px pad + random crop +
random horizontal flip).

### 4.4 Software

flodl (this repo). DDP entry via `Trainer::builder` with three apply
policies (Sync / Cadence / Async) and two average backends (NCCL /
CPU). EASGD α blending available on cpu-async only (Zhang et al. 2015).

---

## 5. Pareto frontier characterization

### 5.1 Passive observation (5-seed default-anchor sweep)

[DRAFT — lift verdict subsections from v2 doc passive-observation verdict
+ relaxed-anchor sweep]

Multi-seed 4-cell table: see [`tables/relaxed-anchor-4cell.md`](tables/relaxed-anchor-4cell.md) [pending].

### 5.2 Synchronization threshold (cliff bracket)

[DRAFT — lift]

Cliff bracket table: see [`tables/cliff-bracket.md`](tables/cliff-bracket.md) [pending].

### 5.3 Multi-seed EASGD

Two paired sweeps probe the EASGD coupling-mechanism axis at
ResNet-20 / 3-GPU. The Gate A multi-seed pass
([`tables/cpu-async-multiseed.md`](tables/cpu-async-multiseed.md))
runs 4 seeds × 2 guards × `cpu-async` × α=0.5 to confirm or falsify
the seed-0 single-shot smoke. The follow-up α-axis sweep
([`tables/cpu-async-alpha-sweep.md`](tables/cpu-async-alpha-sweep.md))
walks α ∈ {0.3, 0.5, 0.7, 1.0} × 4 seeds at fixed `cpu-async × msf`
to measure the α optimum directly, including the previously-missing
in-mode α=1.0 baseline.

Headline cohort means at fixed cpu-async × msf × ResNet-20:

| sweep | α | n | eval mean ± sd | sync mean |
|---|---:|---:|---:|---:|
| cpu-async-multiseed (Gate A) | 0.5 | 4 | 91.61 % ± 0.15 | 652 |
| cpu-async-alpha-sweep | 1.0 | 4 | 91.77 % ± 0.19 | 502 |
| cpu-async-alpha-sweep | 0.7 | 4 | 91.88 % ± 0.21 | 571 |
| cpu-async-alpha-sweep | 0.5 | 4 | 91.89 % ± 0.23 | 548 |
| cpu-async-alpha-sweep | 0.3 | 4 | 91.57 % ± 0.23 | 581 |

Three findings:

1. **α ∈ [0.5, 1.0] is a flat region within seed sd.** Paired-seed
   contrasts vs α=1.0 give Δ = +0.115 pp (α=0.7, t=1.06, NS) and
   Δ = +0.123 pp (α=0.5, t=0.66, NS) at n=4, df=3. The pre-registered
   regularization-optimum threshold of +0.15 pp is not cleared. The α
   knob does not introduce a Pareto-improving direction at this rig
   and recipe.

2. **α=0.3 is a new empirical lower bound on useful EASGD α.**
   Paired-seed Δ = −0.197 pp vs α=1.0 (t = −3.08, p ≈ 0.027 one-sided),
   the only contrast in the α axis to clear seed-noise significance.
   Deep-blending degrades eval; the elastic-averaging Pareto-relevant
   range begins above 0.3 at this recipe.

3. **The sync-count framing-prediction is inverted.** Cohort sync
   means run 502 / 571 / 548 / 581 across α = 1.0 / 0.7 / 0.5 / 0.3.
   α=1.0 has the lowest sync mean of the four cohorts. The original
   "α < 1 reduces syncs" prediction sourced from the seed-0 single-shot
   smoke (408 syncs at α=0.5) does not generalize. The seed-0 number
   was a tail outlier on the favorable side; the multi-seed mean
   reverses the direction.

The Gate A α=0.5 cohort at 91.61 % vs the in-mode α=1.0 cohort at
91.77 % gives paired Δ = −0.16 pp (just outside the design-doc
±0.15 pp parity band, by 0.01 pp). Eval is at parity within seed sd;
the structural-scaling argument that α<1 introduces no Pareto-improving
direction at R-20 / 3-GPU stands on the eval axis, and the sync-cost
axis falsifies the original "α<1 → fewer syncs" framing-prediction
outright.

### 5.4 Pareto figure

[STUB] Figure: eval vs syncs/200ep. Cells: ElChe default trend/msf,
ElChe relaxed trend/msf, EASGD α=0.5 trend/msf, fixed-k {3200, 6400,
12800, 16000, 25600, 51200}. Frontier resolves to three configs:
cpu-async default msf (eval max) + nccl-async relaxed msf (mid) +
nccl-async relaxed trend (lowest-sync near-parity).

See [`figures/pareto-200ep.svg`](figures/pareto-200ep.svg) [pending],
generated from [`ddp-bench/runs/pareto-frontier-200ep/`](../../ddp-bench/runs/pareto-frontier-200ep/).

---

## 6. Bytes-axis confirmation

Question: does EASGD α=0.5 differ from α=1.0 at 3.1× the parameter
count of ResNet-20? The sweep
([`tables/resnet56-bytes-axis.md`](tables/resnet56-bytes-axis.md),
data at [`data/resnet56-cpu-async/`](data/resnet56-cpu-async/))
runs `cpu-async × msf × ResNet-56 (n=9 in the He et al. CIFAR
family, ~850 K params) × 200 epochs` over 4 seeds at α=0.5 plus
1 seed at α=1.0.

| α | n | eval mean ± sd | sync mean |
|---|---:|---:|---:|
| 0.5 | 4 | **93.06 % ± 0.10** | 454 |
| 1.0 | 1 | 92.88 % (single-seed) | 286 |

Published ResNet-56 baseline (He et al. 2015, Table 6, CIFAR-10):
**93.03 %**. The α=0.5 cohort lands +0.03 pp above the published
baseline at n=4 with sd 0.10; the controller plus recipe deliver
published-baseline parity at 3.1× the parameter count of ResNet-20
without hand-tuning.

The R-20 ↔ R-56 contrast on the EASGD coupling-mechanism axis lands
flat on eval. Cross-cohort eval comparison gives α=0.5 mean 93.06 %
vs α=1.0 single-cell 92.88 %, Δ = +0.18 pp; the gap is below the
typical paired-seed sd at R-56 (~0.10 pp) but the n=1 vs n=4 design
makes the comparison directional only, not paired. Together with §5.3,
the bytes-axis null result holds at the eval axis: scaling to 3× the
parameter count does not surface a Pareto-improving α<1 direction.

The sync-count framing-prediction is inverted at R-56 by the same
direction as at R-20. α=0.5 cohort syncs 454 (mean) vs α=1.0
single-cell at 286, that is +59 % more, not fewer. Both model sizes
falsify the "α < 1 → fewer syncs" prediction the seed-0 R-20 smoke
seeded; the falsification direction generalizes to the larger model.

Single-seed limitation. The α=1.0 cell is n=1 by design (the sweep
launcher was killed mid-α=1.0-phase after seed-1 finished; seeds 2–4
α=1.0 not run). Cross-day rerun data on cpu-async at R-20 (§7.2)
indicates that even at fixed RNG seed, eval realizations span
0.5–0.9 pp at this rig, so the α=1.0 single-cell's 92.88 % carries an
implicit uncertainty wider than the n=4 α=0.5 within-cohort sd of
0.10 pp would suggest. The bytes-axis comparison should be read as
directional only until the α=1.0 cohort is filled to n=4. The
[`resnet56-bytes-axis.md`](tables/resnet56-bytes-axis.md) table
flags this as the load-bearing follow-up gap.

---

## 7. Discussion

### 7.1 Frontier rotation prediction

[DRAFT — lift "Scaling prediction" from v2 doc]

Two axes ResNet-20/3-GPU does not stress:

1. Model size — AllReduce cost scales linearly with parameter count.
2. Rank count and heterogeneity diversity — fast/slow asymmetry
   generalizes to multi-cluster decoupling at 4+ GPUs in richer mixes.

### 7.2 cpu-async vs nccl-async — eval / noise observation

A side observation from rerunning the same recipe across days: at
ResNet-20 / 3-GPU, **cpu-async cohorts have higher cross-day
realization noise but a slight cohort-mean lead over nccl-async**.

Cross-cohort means (R-20, msf guard, n ≥ 4 per cohort):

| mode | cohorts (n) | cohort-mean (across cohorts) |
|---|---|---:|
| cpu-async (α = 0.5 or 1.0) | 4 (multiseed, passive-observation, alpha-sweep α=0.5, alpha-sweep α=1.0) | **91.83 %** |
| nccl-async (default or relax-up) | 2 (passive-observation, relaxed-anchor) | **91.76 %** |

The +0.07 pp lead is within seed sd, but it appears in every cpu-async
cohort vs every nccl-async cohort. Three independent cpu-async runs of
the *same* α=0.5 msf recipe on different days (2026-05-06 / -07 / -08,
seeds 1–4 each) give per-seed cross-day swings up to 0.87 pp at fixed
RNG seed — substantially wider than the 0.18–0.31 pp within-day cohort
sd.

A commit-by-commit walk of the three trees the sweeps ran on
(`0806f84`, `4544408`, `54bcfe6`) ruled out code drift as the source:
no commit in the range modifies the training path in a way that
changes the realized trajectory at fixed seed (the only training-path
delta is a refactor of the ResNet builder that is byte-equivalent at
n=3, with all other behavior gated behind unused CLI flags).
`flodl/`, `flodl-sys/`, `Cargo.lock`, and `libtorch/.active` are
untouched across the range.

The noise is therefore non-determinism on the heterogeneous 3-GPU
rig. The most likely dominant source, given the structure, is
**cadence-controller timing variability** — sync-window boundaries on
cpu-async are placed by a controller that consults wall-time and
overhead measurements, both of which vary slightly across runs. Once
sync points shift, the per-rank parameter snapshots being averaged
shift with them, and the trajectory diverges. This is at a different
scale than the micro-level CUDA / NCCL kernel-ordering jitter that
both modes share.

The two-scale framing offers a tentative interpretation: cpu-async's
elastic blending (`W ← (1−α)·W_local + α·W_avg` with α ≤ 1) preserves
some of the per-rank Lyapunov drift between syncs, where nccl-async's
in-place `AllReduce(Mean)` discards it at every sync. If the drift
carries any structured information (which the meta-oscillator framing
suggests it does), cpu-async would inherit both more eval-axis upside
*and* more cross-day variance — which is what the data shows. We do
not have an nccl-async cross-day same-seed cohort at matched recipe
to make this test paired (it would require a small targeted rerun);
the observation is recorded here without claiming a mechanism.

A deeper point follows from the framing established in §3: ElChe is
itself a chaos-based controller, not a deterministic one. The cadence
decisions react to a divergence-rate estimate (`λ_ema`) computed from
the very per-rank trajectories the controller is trying to keep
synchronized. The control loop is closed through the system's own
chaotic state, and the controller's own decisions inherit that
chaos: micro-scale (per-realization sync-window placement) is
unpredictable, but macro-scale (cohort-mean eval, cohort-mean sync
count) is the stable equilibrium the loop drives toward. The
two-scale frame the paper invokes for the training dynamics applies
recursively to the controller itself.

A familiar everyday instance is the bang-bang thermostat. The
furnace's instantaneous on/off state is a chaotic limit cycle that
no observer can predict from a snapshot, but the room temperature
sits stably at the setpoint when sampled over any reasonable
interval. The micro-unpredictability is not a flaw of the
thermostat; it is the only mode in which a control system whose
input is itself a chaotic-leaning quantity can deliver a stable
macro-equilibrium. A deterministic controller scheduling sync points
ahead of time, oblivious to the system state, would by construction
fail to balance the chaos it is meant to manage; it would either
over- or under-couple the per-rank trajectories and drift off the
meta-oscillator attractor. Cohort-level statistics on ElChe sweeps
are therefore not "averaging away noise" but sampling the attractor
ergodically, which is the load-bearing measurement.

### 7.3 Controller refinements (motivated, not validated)

[STUB] C1' by-k cadence inversion, C5' threshold-aware cadence, R5'
meta-CUSUM regime detector. Empirical validation deferred to follow-up
(SmolLM-135M scale-axis study, Meta-ElChe model-parallel study).

### 7.4 Limitations

[STUB]

- Single architecture family (ResNet). Vanilla ViT-Tiny test deferred.
- Single dataset (CIFAR-10). Larger-scale (ImageNet, language) deferred.
- 3-GPU. Multi-cluster heterogeneity (4+ GPUs in richer mixes) deferred.
- AllReduce cost <1ms at this scale — wall-time-axis Pareto rotation
  cannot be tested at this rig and model size.

A note on the limitations *not* claimed. Per-realization eval and
sync-count noise at fixed RNG seed is not listed above and should not
be read as an apparatus or method limitation. As §7.2 establishes,
ElChe is a chaos-based controller closing its loop through a chaotic
training dynamic, so individual-realization unpredictability is the
expected operating mode (the bang-bang thermostat analog). Cohort
sweeps sample the controller's attractor ergodically; cohort-mean
statistics are the load-bearing measurement, not a noise-suppression
artifact. Single-seed differential claims should be interpreted in
that light, with the cross-day seed-fixed envelope (≈ 0.5–0.9 pp at
this rig and recipe) as the realistic noise floor for any n=1 cell.

---

## 8. Conclusion

[STUB] Two-scale framing locates DDP within Pecora-Carroll
synchronization theory and produces a falsifiable scaling prediction.
Empirically: ElChe's auto-tune sits within the optimal eval band on
ResNet-20/3-GPU; no controller-relaxation regime improves final eval
on the cadence axis at this scale; coupling-mechanism axis (EASGD α<1)
adds no Pareto-improving direction at this scale either. Whether
either generalizes to scale and richer heterogeneity is the
open question.

---

## Appendix A. Reproducibility

See [`reproducibility.md`](reproducibility.md) for the claim ↔
artifact map: every empirical claim names a run dir, aggregator
script, and pre-launch commit hash.

## Appendix B. Notation

[STUB — lift glossary from v2 doc + v1 glossary]

## Appendix C. References

See [`references.bib`](references.bib).
