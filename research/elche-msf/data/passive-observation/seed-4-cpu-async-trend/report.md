# DDP Benchmark Report

## Hardware

- gpu0: NVIDIA GeForce RTX 5060 Ti (15GB, sm_120)
- gpu1: NVIDIA GeForce GTX 1060 6GB (5GB, sm_61)
- gpu2: NVIDIA GeForce GTX 1060 6GB (5GB, sm_61)

- **Models**: 1
- **Modes**: 1

## Notes

DDP modes are expected to show slightly lower eval than solo on small models with few epochs. Distributed training converges slower in early epochs due to gradient averaging across devices with different data views, and ElChe (cadence/async) modes need calibration time to find the optimal sync interval, which further penalizes short runs. On longer training (200 epochs), every DDP mode surpasses solo convergence while completing faster -- the whole point of multi-GPU training.

## Incomplete Runs

- resnet-graph/solo-0
- resnet-graph/solo-1
- resnet-graph/solo-2
- resnet-graph/nccl-sync
- resnet-graph/nccl-cadence
- resnet-graph/nccl-async
- resnet-graph/cpu-sync
- resnet-graph/cpu-cadence

## Per-Model Results

GPU0/GPU1 = compute utilization % (not load). Idle = total time with <5% utilization.

### resnet-graph

> Published: CIFAR-10 91.25% acc, Graph builder ([He et al. 2015](https://arxiv.org/abs/1512.03385), Table 6)

| Mode | Loss | Eval | vs Ref | Total (s) | Syncs | Avg Sync (ms) | GPU0 | GPU1 | Idle (s) |
|------|------|------|--------|-----------|-------|--------------|------|------|----------|
| cpu-async | 0.068892 | 0.9155 | +0.0030 | 1872.5 | 260 | 94.4 | 100% | 100% | 10.2 |

## Best Mode per Model

| Model | Best Eval | Mode | Fastest (within 2% of solo-0) | Mode |
|-------|-----------|------|-------------------------------|------|
| resnet-graph | 0.9155 | cpu-async | - | - |

## Eval Quality (vs solo-0)

| Model | cpu-async |
|-------|-----------|
| resnet-graph | - |

## Convergence Quality (loss ratio vs solo-0)

| Model | cpu-async |
|-------|-----------|
| resnet-graph | - |

## Per-Epoch Loss Trajectory

### resnet-graph (sampled, 200 epochs)

| Mode | E0 | E10 | E21 | E31 | E42 | E52 | E63 | E73 | E84 | E94 | E105 | E115 | E126 | E136 | E147 | E157 | E168 | E178 | E189 | E199 |
|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| cpu-async | 1.9813 | 0.7937 | 0.6285 | 0.5640 | 0.5292 | 0.5194 | 0.5029 | 0.4862 | 0.4792 | 0.4742 | 0.2221 | 0.1845 | 0.1694 | 0.1566 | 0.1483 | 0.0915 | 0.0819 | 0.0779 | 0.0739 | 0.0689 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | cpu-async | 0 | cuda:0 | 0.3998 | 2.6 |
| resnet-graph | cpu-async | 1 | cuda:1 | 0.3016 | 3.4 |
| resnet-graph | cpu-async | 2 | cuda:2 | 0.2985 | 3.5 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|
| resnet-graph | cpu-async | 358 | 357 | 386 | 383 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| cpu-async | gpu2 | 645.9 | 1.3 | epoch-boundary(68) |
| cpu-async | gpu1 | 542.7 | 1.2 | epoch-boundary(57) |
| cpu-async | gpu1 | 646.1 | 1.1 | epoch-boundary(68) |
| cpu-async | gpu1 | 533.8 | 0.9 | epoch-boundary(56) |
| cpu-async | gpu1 | 861.9 | 0.8 | epoch-boundary(91) |
| cpu-async | gpu2 | 1758.9 | 0.7 | epoch-boundary(187) |
| cpu-async | gpu1 | 580.7 | 0.7 | epoch-boundary(61) |
| cpu-async | gpu2 | 862.0 | 0.7 | epoch-boundary(91) |
| cpu-async | gpu1 | 1759.0 | 0.6 | epoch-boundary(187) |
| cpu-async | gpu2 | 533.9 | 0.6 | epoch-boundary(56) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | cpu-async | gpu1 | 5.2s | 0.0s | 0.0s | 0.0s | 6.0s |
| resnet-graph | cpu-async | gpu2 | 3.3s | 0.0s | 0.0s | 0.0s | 4.1s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | cpu-async | 227 | 0 | 260 | 94.4 | 8194/10808 | 260 | 94.4 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | cpu-async | 199.2 | 10.6% |

## MSF Passive Observation

Per the v2 framing (`docs/design/msf-cadence-control-v2.md`), DDP is a synchronization-of-coupled-chaotic-oscillators problem at **two scales** linked by AllReduce. Each subsection below is tagged by the scale it operates at:

- **Top scale (meta-oscillator)**: the cross-rank-collapsed observable `D_mean(t)`, the OU process the system spirals toward. The model we ship is the centroid that sits on the synchronization manifold; convergence is exclusively a top-scale phenomenon.
- **Bottom scale (per-GPU)**: per-rank `D_i(τ)` within a cycle, chaotic by construction with positive within-cycle Lyapunov `λ_T(LR)`. Per-replica trajectories don't converge — that's by design.
- **Cross-scale consistency**: cross-rank Pearson `r` and per-rank vs meta slope agreement. The gate that validates the meta-oscillator framing — when `r < 0.95` for any rank pair, the framing has broken and bottom-scale per-rank treatment is required (e.g. cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons).

Historical proxy `λ̂ = (1/k) * log(D_t / D_{t-1})` from v1 doc survives only as a coarse phase indicator; the v2 estimators are the by-k OLS slope (within-cycle Lyapunov, bottom-scale) and CUSUM-on-OU-residual (regime detection, top-scale).

Phase candidates flag epochs where `λ_min < -1e-2` AND `D_end / prev_D_end < 1/3` (collapse signature, e.g. LR drop boundary).

### Summary (top scale)

| Model | Mode | Epochs | Div Events | Phase Candidates | Final D | Final λ_ema |
|-------|------|--------|-----------:|-----------------:|--------:|------------:|
| resnet-graph | cpu-async | 183 | 260 | 0 | 7.19e-3 | +1.70e-5 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | cpu-async | 0 | 260 | 1.13e-1 | 8.65e-2 | 5.73e-3 | 3.37e-1 | 25.0 | -1.62e-4 | 1.10e-3 |
| resnet-graph | cpu-async | 1 | 260 | 1.14e-1 | 8.80e-2 | 5.64e-3 | 3.39e-1 | 37.7 | -1.68e-4 | 1.10e-3 |
| resnet-graph | cpu-async | 2 | 260 | 1.14e-1 | 8.86e-2 | 5.67e-3 | 3.63e-1 | 37.3 | -1.59e-4 | 1.13e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | cpu-async | rank0 ↔ rank1 | +0.9966 |
| resnet-graph | cpu-async | rank0 ↔ rank2 | +0.9957 |
| resnet-graph | cpu-async | rank1 ↔ rank2 | +0.9955 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | cpu-async | 33 (2,5,21,23,37,42,43,44…144,149) | 0 (—) | — | 2,5,21,23,37,42,43,44…144,149 | — |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | cpu-async | 0e0 | 3 | 13 | 13 |
| resnet-graph | cpu-async | 0e0 | 5 | 6 | 6 |
| resnet-graph | cpu-async | 0e0 | 10 | 2 | 2 |
| resnet-graph | cpu-async | 1e-4 | 3 | 0 | 0 |
| resnet-graph | cpu-async | 1e-4 | 5 | 0 | 0 |
| resnet-graph | cpu-async | 1e-4 | 10 | 0 | 0 |
| resnet-graph | cpu-async | 1e-3 | 3 | 0 | 0 |
| resnet-graph | cpu-async | 1e-3 | 5 | 0 | 0 |
| resnet-graph | cpu-async | 1e-3 | 10 | 0 | 0 |
| resnet-graph | cpu-async | 1e-2 | 3 | 0 | 0 |
| resnet-graph | cpu-async | 1e-2 | 5 | 0 | 0 |
| resnet-graph | cpu-async | 1e-2 | 10 | 0 | 0 |
| resnet-graph | cpu-async | 1e-1 | 3 | 0 | 0 |
| resnet-graph | cpu-async | 1e-1 | 5 | 0 | 0 |
| resnet-graph | cpu-async | 1e-1 | 10 | 0 | 0 |

### Predictive Value by LR Window (top scale)

Per-LR-window Pearson `r(λ_raw_t, ln(D_{t+1}))`. Pairs that straddle a window boundary are excluded so the LR-drop collapse cannot leak in as artefactual signal. Reading: a clean R1 (exponential growth at fixed LR) shows up as *non-zero* `r` within each window, with sign and magnitude that may differ between warmup, post-drop transient, and late-training phases.

| Model | Mode | LR | Epochs | n_pairs | r(λ → ln D_{t+1}) |
|-------|------|---:|--------|--------:|----------------:|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 141 | +0.175 |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | 54 | +0.126 |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | 61 | -0.384 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | cpu-async | 258 | -0.068 | 182 | +0.368 | +0.562 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | cpu-async | 259 | 3.31e1–7.93e1 | 6.44e1 | 4.62e-3 | 6.47e-3 | 5.75e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 143 | 32–77770 | +6.221e-6 | 0.367 | +6.562e-6 | 0.392 | 90 | +3.719e-6 | 0.289 | 32–996 | +6.548e-4 | 0.658 |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 134 | 881–77770 | +7.186e-6 | 0.507 | +7.425e-6 | 0.519 | 89 | +3.813e-6 | 0.294 | 90–996 | +7.915e-4 | 0.932 |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | 55 | 78497–116912 | +1.381e-5 | 0.177 | +1.417e-5 | 0.185 | 45 | +1.323e-5 | 0.160 | 486–952 | +1.641e-3 | 0.231 |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | 62 | 117632–155840 | -1.258e-5 | 0.158 | -1.294e-5 | 0.166 | 48 | -1.507e-5 | 0.183 | 421–795 | +1.065e-3 | 0.079 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | +6.548e-4 | r0: +6.231e-4, r1: +6.737e-4, r2: +6.716e-4 | r0: 0.650, r1: 0.659, r2: 0.635 | 1.08× | ✓ |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +7.915e-4 | r0: +7.433e-4, r1: +8.098e-4, r2: +8.243e-4 | r0: 0.909, r1: 0.913, r2: 0.932 | 1.11× | ✓ |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | +1.641e-3 | r0: +1.631e-3, r1: +1.629e-3, r2: +1.663e-3 | r0: 0.226, r1: 0.228, r2: 0.236 | 1.02× | ✓ |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | +1.065e-3 | r0: +1.021e-3, r1: +1.032e-3, r2: +1.145e-3 | r0: 0.071, r1: 0.075, r2: 0.089 | 1.12× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | cpu-async | `█▇█████████████████████▄▄▄▄▄▅▅▅▅▅▅▂▁▁▁▁▁▁▁▁▁▁▁` | `▁▆▇▇███████████████████▇▇▇████████▇▇▇█████████` |

### Per-Epoch Detail: resnet-graph / cpu-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 9 | 1.27e-1 | 3.63e-1 | 2.20e-1 | 1.90e-1 | 53 | -2.35e-2 | +8.68e-3 | -8.46e-3 | -6.83e-3 |
| 1 | 3.00e-1 | 6 | 1.20e-1 | 2.10e-1 | 1.54e-1 | 1.20e-1 | 46 | -4.21e-3 | +1.07e-3 | -1.84e-3 | -3.81e-3 |
| 2 | 3.00e-1 | 7 | 1.28e-1 | 1.60e-1 | 1.35e-1 | 1.29e-1 | 50 | -4.54e-3 | +3.25e-3 | -1.66e-4 | -1.69e-3 |
| 3 | 3.00e-1 | 4 | 1.16e-1 | 1.61e-1 | 1.31e-1 | 1.16e-1 | 40 | -6.21e-3 | +2.58e-3 | -1.43e-3 | -1.62e-3 |
| 4 | 3.00e-1 | 6 | 1.11e-1 | 1.58e-1 | 1.25e-1 | 1.11e-1 | 36 | -6.24e-3 | +3.45e-3 | -9.22e-4 | -1.31e-3 |
| 5 | 3.00e-1 | 8 | 1.15e-1 | 1.56e-1 | 1.23e-1 | 1.16e-1 | 40 | -5.50e-3 | +4.28e-3 | -3.89e-4 | -7.66e-4 |
| 6 | 3.00e-1 | 4 | 1.13e-1 | 1.47e-1 | 1.27e-1 | 1.28e-1 | 49 | -5.02e-3 | +3.31e-3 | -2.80e-4 | -5.88e-4 |
| 7 | 3.00e-1 | 6 | 1.18e-1 | 1.57e-1 | 1.28e-1 | 1.19e-1 | 40 | -4.34e-3 | +2.69e-3 | -6.45e-4 | -6.28e-4 |
| 8 | 3.00e-1 | 1 | 1.18e-1 | 1.18e-1 | 1.18e-1 | 1.18e-1 | 40 | -2.33e-4 | -2.33e-4 | -2.33e-4 | -5.89e-4 |
| 9 | 3.00e-1 | 2 | 2.19e-1 | 2.20e-1 | 2.20e-1 | 2.19e-1 | 275 | -1.43e-5 | +1.82e-3 | +9.04e-4 | -3.13e-4 |
| 11 | 3.00e-1 | 1 | 2.33e-1 | 2.33e-1 | 2.33e-1 | 2.33e-1 | 360 | +1.66e-4 | +1.66e-4 | +1.66e-4 | -2.65e-4 |
| 12 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 309 | -1.79e-4 | -1.79e-4 | -1.79e-4 | -2.56e-4 |
| 13 | 3.00e-1 | 1 | 2.31e-1 | 2.31e-1 | 2.31e-1 | 2.31e-1 | 364 | +1.28e-4 | +1.28e-4 | +1.28e-4 | -2.18e-4 |
| 14 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 287 | -2.51e-4 | -2.51e-4 | -2.51e-4 | -2.21e-4 |
| 15 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 298 | -6.14e-5 | -6.14e-5 | -6.14e-5 | -2.05e-4 |
| 16 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 266 | -9.18e-5 | -9.18e-5 | -9.18e-5 | -1.94e-4 |
| 17 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 300 | +1.69e-4 | +1.69e-4 | +1.69e-4 | -1.57e-4 |
| 18 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 282 | -1.02e-4 | -1.02e-4 | -1.02e-4 | -1.52e-4 |
| 19 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 268 | -4.68e-5 | -4.68e-5 | -4.68e-5 | -1.41e-4 |
| 20 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 269 | +1.57e-5 | +1.57e-5 | +1.57e-5 | -1.26e-4 |
| 21 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 275 | +3.77e-5 | +3.77e-5 | +3.77e-5 | -1.09e-4 |
| 23 | 3.00e-1 | 2 | 2.09e-1 | 2.20e-1 | 2.14e-1 | 2.09e-1 | 265 | -2.11e-4 | +1.46e-4 | -3.25e-5 | -9.64e-5 |
| 25 | 3.00e-1 | 2 | 2.10e-1 | 2.21e-1 | 2.16e-1 | 2.10e-1 | 265 | -2.04e-4 | +1.77e-4 | -1.34e-5 | -8.26e-5 |
| 27 | 3.00e-1 | 1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 321 | +1.84e-4 | +1.84e-4 | +1.84e-4 | -5.59e-5 |
| 28 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 286 | -1.89e-4 | -1.89e-4 | -1.89e-4 | -6.92e-5 |
| 29 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 301 | +4.69e-5 | +4.69e-5 | +4.69e-5 | -5.76e-5 |
| 30 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 275 | -6.17e-5 | -6.17e-5 | -6.17e-5 | -5.80e-5 |
| 31 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 266 | -2.96e-5 | -2.96e-5 | -2.96e-5 | -5.51e-5 |
| 32 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 272 | +3.88e-5 | +3.88e-5 | +3.88e-5 | -4.57e-5 |
| 33 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 255 | -7.98e-6 | -7.98e-6 | -7.98e-6 | -4.20e-5 |
| 34 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 265 | +3.19e-5 | +3.19e-5 | +3.19e-5 | -3.46e-5 |
| 35 | 3.00e-1 | 2 | 2.09e-1 | 2.11e-1 | 2.10e-1 | 2.11e-1 | 247 | -5.04e-5 | +3.01e-5 | -1.02e-5 | -2.95e-5 |
| 37 | 3.00e-1 | 2 | 2.08e-1 | 2.25e-1 | 2.16e-1 | 2.08e-1 | 247 | -3.09e-4 | +2.07e-4 | -5.09e-5 | -3.62e-5 |
| 39 | 3.00e-1 | 2 | 2.09e-1 | 2.25e-1 | 2.17e-1 | 2.09e-1 | 247 | -3.01e-4 | +2.48e-4 | -2.64e-5 | -3.71e-5 |
| 40 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 233 | -1.14e-4 | -1.14e-4 | -1.14e-4 | -4.48e-5 |
| 41 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 233 | +7.38e-6 | +7.38e-6 | +7.38e-6 | -3.95e-5 |
| 42 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 250 | +1.10e-4 | +1.10e-4 | +1.10e-4 | -2.46e-5 |
| 43 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 286 | +1.54e-4 | +1.54e-4 | +1.54e-4 | -6.75e-6 |
| 44 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 296 | +1.14e-5 | +1.14e-5 | +1.14e-5 | -4.93e-6 |
| 45 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 265 | -8.46e-5 | -8.46e-5 | -8.46e-5 | -1.29e-5 |
| 46 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 264 | +6.32e-6 | +6.32e-6 | +6.32e-6 | -1.10e-5 |
| 47 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 246 | -9.98e-5 | -9.98e-5 | -9.98e-5 | -1.99e-5 |
| 48 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 241 | -7.17e-5 | -7.17e-5 | -7.17e-5 | -2.50e-5 |
| 49 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 261 | +1.51e-4 | +1.51e-4 | +1.51e-4 | -7.47e-6 |
| 50 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 260 | -2.67e-6 | -2.67e-6 | -2.67e-6 | -6.99e-6 |
| 51 | 3.00e-1 | 2 | 2.01e-1 | 2.02e-1 | 2.01e-1 | 2.01e-1 | 204 | -2.55e-4 | -3.76e-5 | -1.46e-4 | -3.24e-5 |
| 52 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 212 | +1.75e-6 | +1.75e-6 | +1.75e-6 | -2.90e-5 |
| 53 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 236 | +7.85e-5 | +7.85e-5 | +7.85e-5 | -1.82e-5 |
| 54 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 270 | +2.72e-4 | +2.72e-4 | +2.72e-4 | +1.08e-5 |
| 55 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 247 | -1.39e-4 | -1.39e-4 | -1.39e-4 | -4.17e-6 |
| 56 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 262 | +2.89e-5 | +2.89e-5 | +2.89e-5 | -8.60e-7 |
| 57 | 3.00e-1 | 2 | 2.01e-1 | 2.11e-1 | 2.06e-1 | 2.01e-1 | 202 | -2.56e-4 | -5.77e-5 | -1.57e-4 | -3.15e-5 |
| 58 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 200 | -6.53e-5 | -6.53e-5 | -6.53e-5 | -3.48e-5 |
| 59 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 199 | +1.36e-5 | +1.36e-5 | +1.36e-5 | -3.00e-5 |
| 60 | 3.00e-1 | 1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 180 | -1.34e-4 | -1.34e-4 | -1.34e-4 | -4.04e-5 |
| 61 | 3.00e-1 | 2 | 2.10e-1 | 2.18e-1 | 2.14e-1 | 2.18e-1 | 276 | +1.33e-4 | +3.22e-4 | +2.27e-4 | +9.52e-6 |
| 63 | 3.00e-1 | 1 | 2.36e-1 | 2.36e-1 | 2.36e-1 | 2.36e-1 | 360 | +2.11e-4 | +2.11e-4 | +2.11e-4 | +2.97e-5 |
| 64 | 3.00e-1 | 1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 312 | -1.53e-4 | -1.53e-4 | -1.53e-4 | +1.14e-5 |
| 65 | 3.00e-1 | 1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 331 | +1.38e-5 | +1.38e-5 | +1.38e-5 | +1.16e-5 |
| 66 | 3.00e-1 | 1 | 2.31e-1 | 2.31e-1 | 2.31e-1 | 2.31e-1 | 351 | +6.14e-5 | +6.14e-5 | +6.14e-5 | +1.66e-5 |
| 67 | 3.00e-1 | 1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 318 | -8.57e-5 | -8.57e-5 | -8.57e-5 | +6.36e-6 |
| 68 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 279 | -7.23e-5 | -7.23e-5 | -7.23e-5 | -1.51e-6 |
| 70 | 3.00e-1 | 2 | 2.16e-1 | 2.28e-1 | 2.22e-1 | 2.16e-1 | 279 | -1.80e-4 | +1.02e-4 | -3.90e-5 | -1.00e-5 |
| 72 | 3.00e-1 | 1 | 2.39e-1 | 2.39e-1 | 2.39e-1 | 2.39e-1 | 358 | +2.80e-4 | +2.80e-4 | +2.80e-4 | +1.89e-5 |
| 73 | 3.00e-1 | 1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 336 | -1.51e-4 | -1.51e-4 | -1.51e-4 | +1.90e-6 |
| 74 | 3.00e-1 | 1 | 2.37e-1 | 2.37e-1 | 2.37e-1 | 2.37e-1 | 347 | +1.22e-4 | +1.22e-4 | +1.22e-4 | +1.39e-5 |
| 75 | 3.00e-1 | 1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 311 | -1.36e-4 | -1.36e-4 | -1.36e-4 | -1.06e-6 |
| 76 | 3.00e-1 | 1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 279 | -9.44e-5 | -9.44e-5 | -9.44e-5 | -1.04e-5 |
| 77 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 251 | -1.94e-4 | -1.94e-4 | -1.94e-4 | -2.87e-5 |
| 78 | 3.00e-1 | 1 | 2.28e-1 | 2.28e-1 | 2.28e-1 | 2.28e-1 | 302 | +2.59e-4 | +2.59e-4 | +2.59e-4 | +3.09e-8 |
| 79 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 261 | -2.14e-4 | -2.14e-4 | -2.14e-4 | -2.14e-5 |
| 80 | 3.00e-1 | 1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 270 | +8.41e-5 | +8.41e-5 | +8.41e-5 | -1.08e-5 |
| 81 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 266 | -7.92e-5 | -7.92e-5 | -7.92e-5 | -1.77e-5 |
| 82 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 252 | -3.56e-5 | -3.56e-5 | -3.56e-5 | -1.95e-5 |
| 83 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 244 | -4.78e-5 | -4.78e-5 | -4.78e-5 | -2.23e-5 |
| 84 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 290 | +1.30e-4 | +1.30e-4 | +1.30e-4 | -7.09e-6 |
| 85 | 3.00e-1 | 1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 292 | +1.85e-5 | +1.85e-5 | +1.85e-5 | -4.53e-6 |
| 86 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 282 | -3.16e-5 | -3.16e-5 | -3.16e-5 | -7.24e-6 |
| 87 | 3.00e-1 | 1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 316 | +7.35e-5 | +7.35e-5 | +7.35e-5 | +8.35e-7 |
| 88 | 3.00e-1 | 1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 309 | +1.92e-5 | +1.92e-5 | +1.92e-5 | +2.67e-6 |
| 89 | 3.00e-1 | 1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 312 | +9.16e-5 | +9.16e-5 | +9.16e-5 | +1.16e-5 |
| 91 | 3.00e-1 | 2 | 2.15e-1 | 2.24e-1 | 2.19e-1 | 2.15e-1 | 244 | -1.62e-4 | -1.29e-4 | -1.46e-4 | -1.85e-5 |
| 92 | 3.00e-1 | 1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 281 | +1.23e-4 | +1.23e-4 | +1.23e-4 | -4.37e-6 |
| 93 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 256 | -9.89e-5 | -9.89e-5 | -9.89e-5 | -1.38e-5 |
| 94 | 3.00e-1 | 1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 294 | +1.22e-4 | +1.22e-4 | +1.22e-4 | -2.07e-7 |
| 95 | 3.00e-1 | 1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 299 | +3.64e-5 | +3.64e-5 | +3.64e-5 | +3.46e-6 |
| 96 | 3.00e-1 | 1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 290 | +3.93e-6 | +3.93e-6 | +3.93e-6 | +3.51e-6 |
| 97 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 253 | -2.25e-4 | -2.25e-4 | -2.25e-4 | -1.93e-5 |
| 98 | 3.00e-1 | 1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 270 | +1.31e-4 | +1.31e-4 | +1.31e-4 | -4.25e-6 |
| 99 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 236 | -3.02e-4 | -3.02e-4 | -3.02e-4 | -3.41e-5 |
| 100 | 3.00e-2 | 1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 260 | +2.40e-4 | +2.40e-4 | +2.40e-4 | -6.61e-6 |
| 101 | 3.00e-2 | 1 | 1.07e-1 | 1.07e-1 | 1.07e-1 | 1.07e-1 | 241 | -2.99e-3 | -2.99e-3 | -2.99e-3 | -3.05e-4 |
| 102 | 3.00e-2 | 2 | 3.43e-2 | 5.48e-2 | 4.46e-2 | 3.43e-2 | 207 | -2.97e-3 | -2.27e-3 | -2.62e-3 | -7.42e-4 |
| 103 | 3.00e-2 | 1 | 2.83e-2 | 2.83e-2 | 2.83e-2 | 2.83e-2 | 253 | -7.58e-4 | -7.58e-4 | -7.58e-4 | -7.43e-4 |
| 104 | 3.00e-2 | 1 | 2.60e-2 | 2.60e-2 | 2.60e-2 | 2.60e-2 | 232 | -3.67e-4 | -3.67e-4 | -3.67e-4 | -7.05e-4 |
| 105 | 3.00e-2 | 1 | 2.77e-2 | 2.77e-2 | 2.77e-2 | 2.77e-2 | 264 | +2.40e-4 | +2.40e-4 | +2.40e-4 | -6.11e-4 |
| 106 | 3.00e-2 | 1 | 2.78e-2 | 2.78e-2 | 2.78e-2 | 2.78e-2 | 243 | +1.10e-5 | +1.10e-5 | +1.10e-5 | -5.49e-4 |
| 107 | 3.00e-2 | 2 | 2.84e-2 | 2.88e-2 | 2.86e-2 | 2.88e-2 | 240 | +5.24e-5 | +9.90e-5 | +7.57e-5 | -4.30e-4 |
| 109 | 3.00e-2 | 2 | 2.95e-2 | 3.25e-2 | 3.10e-2 | 2.95e-2 | 190 | -5.00e-4 | +3.83e-4 | -5.85e-5 | -3.64e-4 |
| 110 | 3.00e-2 | 1 | 3.11e-2 | 3.11e-2 | 3.11e-2 | 3.11e-2 | 243 | +2.19e-4 | +2.19e-4 | +2.19e-4 | -3.06e-4 |
| 111 | 3.00e-2 | 1 | 3.01e-2 | 3.01e-2 | 3.01e-2 | 3.01e-2 | 216 | -1.51e-4 | -1.51e-4 | -1.51e-4 | -2.90e-4 |
| 112 | 3.00e-2 | 1 | 3.15e-2 | 3.15e-2 | 3.15e-2 | 3.15e-2 | 232 | +1.93e-4 | +1.93e-4 | +1.93e-4 | -2.42e-4 |
| 113 | 3.00e-2 | 2 | 3.07e-2 | 3.20e-2 | 3.14e-2 | 3.07e-2 | 190 | -2.16e-4 | +6.89e-5 | -7.35e-5 | -2.11e-4 |
| 114 | 3.00e-2 | 1 | 3.35e-2 | 3.35e-2 | 3.35e-2 | 3.35e-2 | 237 | +3.63e-4 | +3.63e-4 | +3.63e-4 | -1.54e-4 |
| 115 | 3.00e-2 | 1 | 3.46e-2 | 3.46e-2 | 3.46e-2 | 3.46e-2 | 234 | +1.43e-4 | +1.43e-4 | +1.43e-4 | -1.24e-4 |
| 116 | 3.00e-2 | 1 | 3.45e-2 | 3.45e-2 | 3.45e-2 | 3.45e-2 | 230 | -1.89e-5 | -1.89e-5 | -1.89e-5 | -1.14e-4 |
| 117 | 3.00e-2 | 2 | 3.27e-2 | 3.42e-2 | 3.34e-2 | 3.42e-2 | 212 | -2.96e-4 | +2.11e-4 | -4.23e-5 | -9.77e-5 |
| 118 | 3.00e-2 | 1 | 3.58e-2 | 3.58e-2 | 3.58e-2 | 3.58e-2 | 216 | +2.17e-4 | +2.17e-4 | +2.17e-4 | -6.62e-5 |
| 119 | 3.00e-2 | 1 | 3.67e-2 | 3.67e-2 | 3.67e-2 | 3.67e-2 | 229 | +1.06e-4 | +1.06e-4 | +1.06e-4 | -4.90e-5 |
| 120 | 3.00e-2 | 2 | 3.39e-2 | 3.62e-2 | 3.51e-2 | 3.39e-2 | 178 | -3.68e-4 | -6.37e-5 | -2.16e-4 | -8.22e-5 |
| 121 | 3.00e-2 | 1 | 3.87e-2 | 3.87e-2 | 3.87e-2 | 3.87e-2 | 236 | +5.55e-4 | +5.55e-4 | +5.55e-4 | -1.85e-5 |
| 122 | 3.00e-2 | 1 | 3.90e-2 | 3.90e-2 | 3.90e-2 | 3.90e-2 | 224 | +3.46e-5 | +3.46e-5 | +3.46e-5 | -1.32e-5 |
| 123 | 3.00e-2 | 2 | 3.68e-2 | 4.09e-2 | 3.88e-2 | 4.09e-2 | 256 | -3.28e-4 | +4.19e-4 | +4.54e-5 | +1.67e-6 |
| 124 | 3.00e-2 | 1 | 4.45e-2 | 4.45e-2 | 4.45e-2 | 4.45e-2 | 280 | +2.98e-4 | +2.98e-4 | +2.98e-4 | +3.13e-5 |
| 126 | 3.00e-2 | 2 | 4.52e-2 | 4.96e-2 | 4.74e-2 | 4.52e-2 | 256 | -3.60e-4 | +3.02e-4 | -2.88e-5 | +1.65e-5 |
| 128 | 3.00e-2 | 2 | 4.64e-2 | 4.94e-2 | 4.79e-2 | 4.64e-2 | 256 | -2.45e-4 | +2.75e-4 | +1.47e-5 | +1.36e-5 |
| 130 | 3.00e-2 | 2 | 4.82e-2 | 4.99e-2 | 4.91e-2 | 4.82e-2 | 256 | -1.39e-4 | +2.36e-4 | +4.83e-5 | +1.83e-5 |
| 132 | 3.00e-2 | 1 | 5.35e-2 | 5.35e-2 | 5.35e-2 | 5.35e-2 | 330 | +3.18e-4 | +3.18e-4 | +3.18e-4 | +4.83e-5 |
| 133 | 3.00e-2 | 1 | 5.10e-2 | 5.10e-2 | 5.10e-2 | 5.10e-2 | 283 | -1.73e-4 | -1.73e-4 | -1.73e-4 | +2.62e-5 |
| 134 | 3.00e-2 | 1 | 5.07e-2 | 5.07e-2 | 5.07e-2 | 5.07e-2 | 278 | -1.51e-5 | -1.51e-5 | -1.51e-5 | +2.20e-5 |
| 135 | 3.00e-2 | 1 | 5.26e-2 | 5.26e-2 | 5.26e-2 | 5.26e-2 | 291 | +1.26e-4 | +1.26e-4 | +1.26e-4 | +3.24e-5 |
| 136 | 3.00e-2 | 1 | 5.27e-2 | 5.27e-2 | 5.27e-2 | 5.27e-2 | 271 | +5.28e-6 | +5.28e-6 | +5.28e-6 | +2.97e-5 |
| 137 | 3.00e-2 | 1 | 5.43e-2 | 5.43e-2 | 5.43e-2 | 5.43e-2 | 285 | +1.04e-4 | +1.04e-4 | +1.04e-4 | +3.71e-5 |
| 138 | 3.00e-2 | 1 | 5.78e-2 | 5.78e-2 | 5.78e-2 | 5.78e-2 | 311 | +2.04e-4 | +2.04e-4 | +2.04e-4 | +5.38e-5 |
| 139 | 3.00e-2 | 1 | 5.61e-2 | 5.61e-2 | 5.61e-2 | 5.61e-2 | 287 | -1.04e-4 | -1.04e-4 | -1.04e-4 | +3.80e-5 |
| 140 | 3.00e-2 | 1 | 5.51e-2 | 5.51e-2 | 5.51e-2 | 5.51e-2 | 263 | -7.46e-5 | -7.46e-5 | -7.46e-5 | +2.67e-5 |
| 141 | 3.00e-2 | 1 | 5.74e-2 | 5.74e-2 | 5.74e-2 | 5.74e-2 | 282 | +1.47e-4 | +1.47e-4 | +1.47e-4 | +3.87e-5 |
| 142 | 3.00e-2 | 1 | 5.70e-2 | 5.70e-2 | 5.70e-2 | 5.70e-2 | 255 | -2.43e-5 | -2.43e-5 | -2.43e-5 | +3.24e-5 |
| 143 | 3.00e-2 | 1 | 5.91e-2 | 5.91e-2 | 5.91e-2 | 5.91e-2 | 286 | +1.25e-4 | +1.25e-4 | +1.25e-4 | +4.17e-5 |
| 144 | 3.00e-2 | 1 | 6.20e-2 | 6.20e-2 | 6.20e-2 | 6.20e-2 | 299 | +1.58e-4 | +1.58e-4 | +1.58e-4 | +5.33e-5 |
| 145 | 3.00e-2 | 1 | 6.05e-2 | 6.05e-2 | 6.05e-2 | 6.05e-2 | 278 | -8.55e-5 | -8.55e-5 | -8.55e-5 | +3.94e-5 |
| 146 | 3.00e-2 | 1 | 6.34e-2 | 6.34e-2 | 6.34e-2 | 6.34e-2 | 306 | +1.51e-4 | +1.51e-4 | +1.51e-4 | +5.06e-5 |
| 147 | 3.00e-2 | 1 | 5.82e-2 | 5.82e-2 | 5.82e-2 | 5.82e-2 | 250 | -3.40e-4 | -3.40e-4 | -3.40e-4 | +1.15e-5 |
| 148 | 3.00e-2 | 1 | 5.89e-2 | 5.89e-2 | 5.89e-2 | 5.89e-2 | 249 | +5.00e-5 | +5.00e-5 | +5.00e-5 | +1.53e-5 |
| 149 | 3.00e-2 | 1 | 6.17e-2 | 6.17e-2 | 6.17e-2 | 6.17e-2 | 266 | +1.75e-4 | +1.75e-4 | +1.75e-4 | +3.13e-5 |
| 150 | 3.00e-3 | 1 | 6.08e-2 | 6.08e-2 | 6.08e-2 | 6.08e-2 | 257 | -5.83e-5 | -5.83e-5 | -5.83e-5 | +2.23e-5 |
| 151 | 3.00e-3 | 1 | 3.01e-2 | 3.01e-2 | 3.01e-2 | 3.01e-2 | 247 | -2.85e-3 | -2.85e-3 | -2.85e-3 | -2.65e-4 |
| 152 | 3.00e-3 | 1 | 1.56e-2 | 1.56e-2 | 1.56e-2 | 1.56e-2 | 251 | -2.60e-3 | -2.60e-3 | -2.60e-3 | -4.99e-4 |
| 153 | 3.00e-3 | 1 | 9.33e-3 | 9.33e-3 | 9.33e-3 | 9.33e-3 | 245 | -2.11e-3 | -2.11e-3 | -2.11e-3 | -6.60e-4 |
| 154 | 3.00e-3 | 1 | 7.19e-3 | 7.19e-3 | 7.19e-3 | 7.19e-3 | 272 | -9.55e-4 | -9.55e-4 | -9.55e-4 | -6.90e-4 |
| 155 | 3.00e-3 | 1 | 6.76e-3 | 6.76e-3 | 6.76e-3 | 6.76e-3 | 287 | -2.14e-4 | -2.14e-4 | -2.14e-4 | -6.42e-4 |
| 156 | 3.00e-3 | 1 | 6.64e-3 | 6.64e-3 | 6.64e-3 | 6.64e-3 | 277 | -6.78e-5 | -6.78e-5 | -6.78e-5 | -5.85e-4 |
| 157 | 3.00e-3 | 1 | 6.47e-3 | 6.47e-3 | 6.47e-3 | 6.47e-3 | 270 | -9.57e-5 | -9.57e-5 | -9.57e-5 | -5.36e-4 |
| 158 | 3.00e-3 | 1 | 6.49e-3 | 6.49e-3 | 6.49e-3 | 6.49e-3 | 271 | +1.23e-5 | +1.23e-5 | +1.23e-5 | -4.81e-4 |
| 159 | 3.00e-3 | 2 | 6.20e-3 | 6.44e-3 | 6.32e-3 | 6.20e-3 | 220 | -1.75e-4 | -3.08e-5 | -1.03e-4 | -4.10e-4 |
| 160 | 3.00e-3 | 1 | 6.40e-3 | 6.40e-3 | 6.40e-3 | 6.40e-3 | 258 | +1.24e-4 | +1.24e-4 | +1.24e-4 | -3.56e-4 |
| 161 | 3.00e-3 | 1 | 6.48e-3 | 6.48e-3 | 6.48e-3 | 6.48e-3 | 261 | +4.74e-5 | +4.74e-5 | +4.74e-5 | -3.16e-4 |
| 162 | 3.00e-3 | 1 | 6.33e-3 | 6.33e-3 | 6.33e-3 | 6.33e-3 | 235 | -9.64e-5 | -9.64e-5 | -9.64e-5 | -2.94e-4 |
| 163 | 3.00e-3 | 1 | 6.27e-3 | 6.27e-3 | 6.27e-3 | 6.27e-3 | 225 | -4.30e-5 | -4.30e-5 | -4.30e-5 | -2.69e-4 |
| 164 | 3.00e-3 | 2 | 6.20e-3 | 6.29e-3 | 6.25e-3 | 6.20e-3 | 220 | -6.40e-5 | +1.44e-5 | -2.48e-5 | -2.23e-4 |
| 166 | 3.00e-3 | 2 | 6.38e-3 | 6.90e-3 | 6.64e-3 | 6.38e-3 | 220 | -3.57e-4 | +3.96e-4 | +1.93e-5 | -1.81e-4 |
| 167 | 3.00e-3 | 1 | 6.55e-3 | 6.55e-3 | 6.55e-3 | 6.55e-3 | 242 | +1.07e-4 | +1.07e-4 | +1.07e-4 | -1.52e-4 |
| 168 | 3.00e-3 | 1 | 6.77e-3 | 6.77e-3 | 6.77e-3 | 6.77e-3 | 262 | +1.28e-4 | +1.28e-4 | +1.28e-4 | -1.24e-4 |
| 169 | 3.00e-3 | 1 | 6.93e-3 | 6.93e-3 | 6.93e-3 | 6.93e-3 | 281 | +8.07e-5 | +8.07e-5 | +8.07e-5 | -1.04e-4 |
| 170 | 3.00e-3 | 1 | 6.77e-3 | 6.77e-3 | 6.77e-3 | 6.77e-3 | 234 | -9.84e-5 | -9.84e-5 | -9.84e-5 | -1.03e-4 |
| 171 | 3.00e-3 | 1 | 6.51e-3 | 6.51e-3 | 6.51e-3 | 6.51e-3 | 228 | -1.69e-4 | -1.69e-4 | -1.69e-4 | -1.10e-4 |
| 172 | 3.00e-3 | 2 | 6.38e-3 | 6.55e-3 | 6.46e-3 | 6.38e-3 | 208 | -1.23e-4 | +2.23e-5 | -5.05e-5 | -9.91e-5 |
| 173 | 3.00e-3 | 1 | 6.39e-3 | 6.39e-3 | 6.39e-3 | 6.39e-3 | 220 | +5.70e-6 | +5.70e-6 | +5.70e-6 | -8.86e-5 |
| 174 | 3.00e-3 | 1 | 6.42e-3 | 6.42e-3 | 6.42e-3 | 6.42e-3 | 220 | +2.24e-5 | +2.24e-5 | +2.24e-5 | -7.75e-5 |
| 175 | 3.00e-3 | 1 | 6.80e-3 | 6.80e-3 | 6.80e-3 | 6.80e-3 | 244 | +2.36e-4 | +2.36e-4 | +2.36e-4 | -4.62e-5 |
| 176 | 3.00e-3 | 2 | 6.50e-3 | 6.83e-3 | 6.66e-3 | 6.50e-3 | 208 | -2.35e-4 | +1.40e-5 | -1.10e-4 | -5.96e-5 |
| 177 | 3.00e-3 | 1 | 6.84e-3 | 6.84e-3 | 6.84e-3 | 6.84e-3 | 254 | +2.01e-4 | +2.01e-4 | +2.01e-4 | -3.36e-5 |
| 178 | 3.00e-3 | 1 | 6.71e-3 | 6.71e-3 | 6.71e-3 | 6.71e-3 | 227 | -8.44e-5 | -8.44e-5 | -8.44e-5 | -3.86e-5 |
| 179 | 3.00e-3 | 1 | 6.63e-3 | 6.63e-3 | 6.63e-3 | 6.63e-3 | 219 | -5.65e-5 | -5.65e-5 | -5.65e-5 | -4.04e-5 |
| 180 | 3.00e-3 | 1 | 6.70e-3 | 6.70e-3 | 6.70e-3 | 6.70e-3 | 225 | +4.53e-5 | +4.53e-5 | +4.53e-5 | -3.18e-5 |
| 181 | 3.00e-3 | 2 | 6.56e-3 | 7.36e-3 | 6.96e-3 | 6.56e-3 | 195 | -5.87e-4 | +3.57e-4 | -1.15e-4 | -5.23e-5 |
| 182 | 3.00e-3 | 1 | 6.97e-3 | 6.97e-3 | 6.97e-3 | 6.97e-3 | 252 | +2.41e-4 | +2.41e-4 | +2.41e-4 | -2.30e-5 |
| 183 | 3.00e-3 | 1 | 6.54e-3 | 6.54e-3 | 6.54e-3 | 6.54e-3 | 198 | -3.23e-4 | -3.23e-4 | -3.23e-4 | -5.30e-5 |
| 184 | 3.00e-3 | 2 | 6.34e-3 | 6.59e-3 | 6.47e-3 | 6.34e-3 | 182 | -2.07e-4 | +3.25e-5 | -8.73e-5 | -6.07e-5 |
| 185 | 3.00e-3 | 1 | 6.42e-3 | 6.42e-3 | 6.42e-3 | 6.42e-3 | 211 | +5.38e-5 | +5.38e-5 | +5.38e-5 | -4.93e-5 |
| 186 | 3.00e-3 | 2 | 6.44e-3 | 6.56e-3 | 6.50e-3 | 6.56e-3 | 211 | +1.87e-5 | +8.80e-5 | +5.33e-5 | -2.94e-5 |
| 188 | 3.00e-3 | 2 | 6.41e-3 | 7.14e-3 | 6.77e-3 | 6.41e-3 | 168 | -6.40e-4 | +3.25e-4 | -1.58e-4 | -5.86e-5 |
| 189 | 3.00e-3 | 1 | 6.88e-3 | 6.88e-3 | 6.88e-3 | 6.88e-3 | 211 | +3.37e-4 | +3.37e-4 | +3.37e-4 | -1.90e-5 |
| 190 | 3.00e-3 | 2 | 6.00e-3 | 6.57e-3 | 6.29e-3 | 6.00e-3 | 159 | -5.65e-4 | -2.44e-4 | -4.04e-4 | -9.39e-5 |
| 191 | 3.00e-3 | 1 | 6.24e-3 | 6.24e-3 | 6.24e-3 | 6.24e-3 | 176 | +2.20e-4 | +2.20e-4 | +2.20e-4 | -6.25e-5 |
| 192 | 3.00e-3 | 2 | 6.31e-3 | 6.69e-3 | 6.50e-3 | 6.31e-3 | 177 | -3.27e-4 | +3.27e-4 | -2.41e-7 | -5.39e-5 |
| 193 | 3.00e-3 | 1 | 6.78e-3 | 6.78e-3 | 6.78e-3 | 6.78e-3 | 229 | +3.12e-4 | +3.12e-4 | +3.12e-4 | -1.73e-5 |
| 194 | 3.00e-3 | 2 | 6.14e-3 | 7.05e-3 | 6.59e-3 | 6.14e-3 | 156 | -8.85e-4 | +1.57e-4 | -3.64e-4 | -8.84e-5 |
| 195 | 3.00e-3 | 1 | 6.37e-3 | 6.37e-3 | 6.37e-3 | 6.37e-3 | 192 | +1.91e-4 | +1.91e-4 | +1.91e-4 | -6.05e-5 |
| 196 | 3.00e-3 | 2 | 6.09e-3 | 6.33e-3 | 6.21e-3 | 6.09e-3 | 156 | -2.49e-4 | -3.17e-5 | -1.40e-4 | -7.68e-5 |
| 197 | 3.00e-3 | 1 | 6.47e-3 | 6.47e-3 | 6.47e-3 | 6.47e-3 | 192 | +3.14e-4 | +3.14e-4 | +3.14e-4 | -3.77e-5 |
| 198 | 3.00e-3 | 2 | 6.39e-3 | 6.60e-3 | 6.49e-3 | 6.39e-3 | 164 | -1.95e-4 | +1.05e-4 | -4.50e-5 | -4.06e-5 |
| 199 | 3.00e-3 | 1 | 7.19e-3 | 7.19e-3 | 7.19e-3 | 7.19e-3 | 221 | +5.35e-4 | +5.35e-4 | +5.35e-4 | +1.70e-5 |

