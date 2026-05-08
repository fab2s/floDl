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
| cpu-async | 0.015464 | 0.9303 | +0.0178 | 5061.7 | 409 | 234.8 | 100% | 100% | 8.1 |

## Best Mode per Model

| Model | Best Eval | Mode | Fastest (within 2% of solo-0) | Mode |
|-------|-----------|------|-------------------------------|------|
| resnet-graph | 0.9303 | cpu-async | - | - |

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
| cpu-async | 2.4187 | 1.2005 | 0.6917 | 0.6052 | 0.5466 | 0.5154 | 0.4972 | 0.4798 | 0.4664 | 0.4555 | 0.1801 | 0.1351 | 0.1084 | 0.0915 | 0.0850 | 0.0299 | 0.0238 | 0.0186 | 0.0171 | 0.0155 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | cpu-async | 0 | cuda:0 | 0.3510 | 1.0 |
| resnet-graph | cpu-async | 1 | cuda:1 | 0.3271 | 1.2 |
| resnet-graph | cpu-async | 2 | cuda:2 | 0.3219 | 1.2 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|
| resnet-graph | cpu-async | 514 | 513 | 588 | 586 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| cpu-async | gpu2 | 5060.0 | 1.7 | epoch-boundary(199) |
| cpu-async | gpu1 | 5060.1 | 1.6 | epoch-boundary(199) |
| cpu-async | gpu1 | 698.4 | 1.4 | epoch-boundary(27) |
| cpu-async | gpu2 | 698.4 | 1.4 | epoch-boundary(27) |
| cpu-async | gpu1 | 4189.1 | 0.9 | epoch-boundary(165) |
| cpu-async | gpu0 | 0.4 | 0.5 | unexplained |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | cpu-async | gpu0 | 0.0s | 0.0s | 0.0s | 0.5s | 0.5s |
| resnet-graph | cpu-async | gpu1 | 3.8s | 0.0s | 0.0s | 0.0s | 4.5s |
| resnet-graph | cpu-async | gpu2 | 3.0s | 0.0s | 0.0s | 0.0s | 3.0s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | cpu-async | 288 | 0 | 409 | 234.8 | 13619/24276 | 409 | 234.8 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | cpu-async | 339.7 | 6.7% |

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
| resnet-graph | cpu-async | 199 | 409 | 0 | 4.79e-3 | -5.74e-5 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | cpu-async | 0 | 409 | 9.46e-2 | 1.01e-1 | 4.68e-3 | 1.08e0 | 22.0 | -2.45e-4 | 1.80e-3 |
| resnet-graph | cpu-async | 1 | 409 | 9.59e-2 | 1.07e-1 | 4.65e-3 | 1.32e0 | 31.5 | -2.85e-4 | 1.93e-3 |
| resnet-graph | cpu-async | 2 | 409 | 9.69e-2 | 1.07e-1 | 4.63e-3 | 1.35e0 | 46.5 | -3.13e-4 | 2.18e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | cpu-async | rank0 ↔ rank1 | +0.9930 |
| resnet-graph | cpu-async | rank0 ↔ rank2 | +0.9845 |
| resnet-graph | cpu-async | rank1 ↔ rank2 | +0.9919 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | cpu-async | 48 (1,2,3,4,5,6,7,8…140,143) | 0 (—) | — | 1,2,3,4,5,6,7,8…140,143 | — |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | cpu-async | 0e0 | 3 | 3 | 3 |
| resnet-graph | cpu-async | 0e0 | 5 | 1 | 1 |
| resnet-graph | cpu-async | 0e0 | 10 | 0 | 0 |
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
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 271 | +0.050 |
| resnet-graph | cpu-async | 3.00e-2 | 100–148 | 71 | +0.037 |
| resnet-graph | cpu-async | 3.00e-3 | 149–199 | 63 | -0.138 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | cpu-async | 407 | -0.037 | 198 | +0.370 | +0.488 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | cpu-async | 408 | 6.22e1–1.51e2 | 8.22e1 | 3.86e-3 | 1.41e-2 | 2.64e-1 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 273 | 32–77932 | +2.724e-5 | 0.527 | +2.758e-5 | 0.534 | 100 | +2.033e-5 | 0.573 | 32–794 | +2.083e-3 | 0.497 |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 260 | 917–77932 | +2.846e-5 | 0.659 | +2.859e-5 | 0.661 | 99 | +2.036e-5 | 0.567 | 73–794 | +2.142e-3 | 0.617 |
| resnet-graph | cpu-async | 3.00e-2 | 100–148 | 72 | 78559–116350 | +1.167e-5 | 0.173 | +1.164e-5 | 0.176 | 49 | +1.018e-5 | 0.106 | 345–723 | -1.294e-3 | 0.142 |
| resnet-graph | cpu-async | 3.00e-3 | 149–199 | 64 | 116779–156174 | -1.977e-5 | 0.239 | -1.961e-5 | 0.237 | 50 | -1.660e-5 | 0.206 | 329–789 | -2.076e-3 | 0.209 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | +2.083e-3 | r0: +2.071e-3, r1: +2.092e-3, r2: +2.089e-3 | r0: 0.490, r1: 0.495, r2: 0.502 | 1.01× | ✓ |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +2.142e-3 | r0: +2.128e-3, r1: +2.151e-3, r2: +2.150e-3 | r0: 0.612, r1: 0.612, r2: 0.623 | 1.01× | ✓ |
| resnet-graph | cpu-async | 3.00e-2 | 100–148 | -1.294e-3 | r0: -1.309e-3, r1: -1.336e-3, r2: -1.238e-3 | r0: 0.145, r1: 0.150, r2: 0.129 | 1.08× | ✓ |
| resnet-graph | cpu-async | 3.00e-3 | 149–199 | -2.076e-3 | r0: -2.146e-3, r1: -2.060e-3, r2: -2.020e-3 | r0: 0.222, r1: 0.204, r2: 0.199 | 1.06× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | cpu-async | `▆▅▆▇▇▇▇██████████████████▆▄▄▄▅▅▅▅▅▅▅▅▅▁▁▁▁▁▁▁▁▁▁▁▁` | `▁▇▇▇▇▇▇██████████████████▇▇██████████▇▇▇▇█████████` |

### Per-Epoch Detail: resnet-graph / cpu-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 13 | 2.17e-2 | 1.35e0 | 2.75e-1 | 2.40e-2 | 26 | -4.84e-2 | +3.90e-3 | -1.97e-2 | -1.41e-2 |
| 1 | 3.00e-1 | 8 | 1.53e-2 | 3.46e-2 | 2.00e-2 | 2.11e-2 | 36 | -2.35e-2 | +5.42e-3 | -2.04e-3 | -5.69e-3 |
| 2 | 3.00e-1 | 7 | 2.19e-2 | 3.17e-2 | 2.48e-2 | 2.36e-2 | 42 | -1.09e-2 | +5.37e-3 | -4.78e-4 | -2.79e-3 |
| 3 | 3.00e-1 | 6 | 2.62e-2 | 3.45e-2 | 2.86e-2 | 2.86e-2 | 39 | -5.35e-3 | +5.09e-3 | +2.29e-5 | -1.42e-3 |
| 4 | 3.00e-1 | 8 | 2.68e-2 | 3.76e-2 | 3.02e-2 | 3.24e-2 | 37 | -8.28e-3 | +3.75e-3 | -1.85e-4 | -5.40e-4 |
| 5 | 3.00e-1 | 8 | 3.29e-2 | 4.78e-2 | 3.80e-2 | 3.91e-2 | 36 | -1.01e-2 | +4.96e-3 | -2.70e-4 | -3.02e-4 |
| 6 | 3.00e-1 | 5 | 3.96e-2 | 5.72e-2 | 4.42e-2 | 3.96e-2 | 33 | -7.71e-3 | +4.39e-3 | -1.14e-3 | -6.64e-4 |
| 7 | 3.00e-1 | 7 | 4.48e-2 | 6.06e-2 | 4.86e-2 | 4.65e-2 | 31 | -9.00e-3 | +5.46e-3 | -4.72e-4 | -5.60e-4 |
| 8 | 3.00e-1 | 8 | 5.09e-2 | 6.85e-2 | 5.42e-2 | 5.56e-2 | 33 | -9.88e-3 | +5.09e-3 | -2.67e-4 | -2.78e-4 |
| 9 | 3.00e-1 | 9 | 5.69e-2 | 8.51e-2 | 6.21e-2 | 6.09e-2 | 32 | -1.00e-2 | +5.08e-3 | -5.45e-4 | -3.12e-4 |
| 10 | 3.00e-1 | 6 | 6.13e-2 | 9.09e-2 | 6.91e-2 | 6.13e-2 | 31 | -8.77e-3 | +5.29e-3 | -1.25e-3 | -7.61e-4 |
| 11 | 3.00e-1 | 10 | 6.52e-2 | 8.89e-2 | 7.01e-2 | 6.57e-2 | 28 | -7.11e-3 | +5.24e-3 | -5.03e-4 | -6.05e-4 |
| 12 | 3.00e-1 | 6 | 6.89e-2 | 1.02e-1 | 7.60e-2 | 7.05e-2 | 27 | -1.11e-2 | +6.44e-3 | -1.02e-3 | -7.68e-4 |
| 13 | 3.00e-1 | 9 | 7.41e-2 | 1.00e-1 | 8.35e-2 | 7.88e-2 | 32 | -6.40e-3 | +5.33e-3 | -3.46e-4 | -5.59e-4 |
| 14 | 3.00e-1 | 6 | 8.40e-2 | 1.20e-1 | 9.08e-2 | 8.73e-2 | 37 | -1.27e-2 | +5.01e-3 | -1.11e-3 | -7.35e-4 |
| 15 | 3.00e-1 | 6 | 9.83e-2 | 1.25e-1 | 1.06e-1 | 9.83e-2 | 40 | -3.92e-3 | +4.11e-3 | -2.04e-4 | -5.42e-4 |
| 16 | 3.00e-1 | 5 | 9.77e-2 | 1.29e-1 | 1.06e-1 | 9.79e-2 | 37 | -5.67e-3 | +3.62e-3 | -6.66e-4 | -6.06e-4 |
| 17 | 3.00e-1 | 7 | 9.91e-2 | 1.39e-1 | 1.08e-1 | 1.05e-1 | 42 | -9.63e-3 | +4.03e-3 | -6.54e-4 | -5.72e-4 |
| 18 | 3.00e-1 | 5 | 1.02e-1 | 1.45e-1 | 1.14e-1 | 1.02e-1 | 41 | -6.71e-3 | +3.53e-3 | -1.08e-3 | -7.88e-4 |
| 19 | 3.00e-1 | 6 | 9.94e-2 | 1.44e-1 | 1.12e-1 | 1.08e-1 | 40 | -4.86e-3 | +3.82e-3 | -4.23e-4 | -5.84e-4 |
| 20 | 3.00e-1 | 6 | 1.05e-1 | 1.37e-1 | 1.14e-1 | 1.10e-1 | 39 | -5.99e-3 | +3.11e-3 | -4.42e-4 | -5.00e-4 |
| 21 | 3.00e-1 | 7 | 1.07e-1 | 1.43e-1 | 1.16e-1 | 1.07e-1 | 40 | -4.89e-3 | +3.42e-3 | -5.35e-4 | -5.28e-4 |
| 22 | 3.00e-1 | 5 | 1.08e-1 | 1.48e-1 | 1.19e-1 | 1.10e-1 | 42 | -7.02e-3 | +4.01e-3 | -7.91e-4 | -6.46e-4 |
| 23 | 3.00e-1 | 7 | 1.04e-1 | 1.47e-1 | 1.15e-1 | 1.13e-1 | 42 | -6.07e-3 | +3.82e-3 | -4.11e-4 | -4.65e-4 |
| 24 | 3.00e-1 | 8 | 1.05e-1 | 1.55e-1 | 1.16e-1 | 1.05e-1 | 33 | -6.97e-3 | +3.62e-3 | -9.11e-4 | -6.90e-4 |
| 25 | 3.00e-1 | 5 | 1.08e-1 | 1.50e-1 | 1.19e-1 | 1.08e-1 | 35 | -7.02e-3 | +4.51e-3 | -8.36e-4 | -7.76e-4 |
| 26 | 3.00e-1 | 7 | 1.02e-1 | 1.50e-1 | 1.13e-1 | 1.05e-1 | 32 | -8.30e-3 | +4.45e-3 | -9.83e-4 | -8.24e-4 |
| 27 | 3.00e-1 | 2 | 1.07e-1 | 2.19e-1 | 1.63e-1 | 2.19e-1 | 255 | +4.08e-4 | +2.83e-3 | +1.62e-3 | -3.48e-4 |
| 28 | 3.00e-1 | 1 | 2.36e-1 | 2.36e-1 | 2.36e-1 | 2.36e-1 | 269 | +2.71e-4 | +2.71e-4 | +2.71e-4 | -2.86e-4 |
| 29 | 3.00e-1 | 1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 254 | -1.52e-4 | -1.52e-4 | -1.52e-4 | -2.73e-4 |
| 30 | 3.00e-1 | 1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 265 | +3.50e-5 | +3.50e-5 | +3.50e-5 | -2.42e-4 |
| 31 | 3.00e-1 | 1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 261 | -3.17e-5 | -3.17e-5 | -3.17e-5 | -2.21e-4 |
| 32 | 3.00e-1 | 1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 264 | +3.98e-5 | +3.98e-5 | +3.98e-5 | -1.95e-4 |
| 33 | 3.00e-1 | 1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 269 | +2.64e-6 | +2.64e-6 | +2.64e-6 | -1.75e-4 |
| 34 | 3.00e-1 | 1 | 2.36e-1 | 2.36e-1 | 2.36e-1 | 2.36e-1 | 289 | +9.36e-5 | +9.36e-5 | +9.36e-5 | -1.48e-4 |
| 35 | 3.00e-1 | 1 | 2.33e-1 | 2.33e-1 | 2.33e-1 | 2.33e-1 | 286 | -4.39e-5 | -4.39e-5 | -4.39e-5 | -1.38e-4 |
| 36 | 3.00e-1 | 1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 269 | -2.34e-5 | -2.34e-5 | -2.34e-5 | -1.26e-4 |
| 37 | 3.00e-1 | 1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 247 | -8.68e-5 | -8.68e-5 | -8.68e-5 | -1.22e-4 |
| 38 | 3.00e-1 | 2 | 2.13e-1 | 2.23e-1 | 2.18e-1 | 2.13e-1 | 210 | -2.19e-4 | -6.37e-5 | -1.42e-4 | -1.27e-4 |
| 39 | 3.00e-1 | 1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 240 | +2.14e-4 | +2.14e-4 | +2.14e-4 | -9.27e-5 |
| 40 | 3.00e-1 | 1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 249 | +5.44e-5 | +5.44e-5 | +5.44e-5 | -7.80e-5 |
| 41 | 3.00e-1 | 1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 251 | -7.28e-5 | -7.28e-5 | -7.28e-5 | -7.75e-5 |
| 42 | 3.00e-1 | 1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 251 | +6.51e-5 | +6.51e-5 | +6.51e-5 | -6.32e-5 |
| 43 | 3.00e-1 | 1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 265 | +5.41e-5 | +5.41e-5 | +5.41e-5 | -5.15e-5 |
| 44 | 3.00e-1 | 1 | 2.35e-1 | 2.35e-1 | 2.35e-1 | 2.35e-1 | 263 | +7.84e-5 | +7.84e-5 | +7.84e-5 | -3.85e-5 |
| 45 | 3.00e-1 | 1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 246 | -1.35e-4 | -1.35e-4 | -1.35e-4 | -4.82e-5 |
| 46 | 3.00e-1 | 1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 243 | +2.15e-5 | +2.15e-5 | +2.15e-5 | -4.12e-5 |
| 47 | 3.00e-1 | 2 | 2.18e-1 | 2.23e-1 | 2.20e-1 | 2.23e-1 | 262 | -2.29e-4 | +8.29e-5 | -7.28e-5 | -4.57e-5 |
| 48 | 3.00e-1 | 1 | 2.40e-1 | 2.40e-1 | 2.40e-1 | 2.40e-1 | 271 | +2.80e-4 | +2.80e-4 | +2.80e-4 | -1.31e-5 |
| 49 | 3.00e-1 | 1 | 2.34e-1 | 2.34e-1 | 2.34e-1 | 2.34e-1 | 258 | -9.75e-5 | -9.75e-5 | -9.75e-5 | -2.15e-5 |
| 50 | 3.00e-1 | 1 | 2.47e-1 | 2.47e-1 | 2.47e-1 | 2.47e-1 | 282 | +1.84e-4 | +1.84e-4 | +1.84e-4 | -9.54e-7 |
| 51 | 3.00e-1 | 1 | 2.39e-1 | 2.39e-1 | 2.39e-1 | 2.39e-1 | 273 | -1.19e-4 | -1.19e-4 | -1.19e-4 | -1.28e-5 |
| 52 | 3.00e-1 | 1 | 2.39e-1 | 2.39e-1 | 2.39e-1 | 2.39e-1 | 270 | -1.66e-6 | -1.66e-6 | -1.66e-6 | -1.17e-5 |
| 53 | 3.00e-1 | 1 | 2.34e-1 | 2.34e-1 | 2.34e-1 | 2.34e-1 | 253 | -8.52e-5 | -8.52e-5 | -8.52e-5 | -1.90e-5 |
| 54 | 3.00e-1 | 1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 247 | -8.17e-5 | -8.17e-5 | -8.17e-5 | -2.53e-5 |
| 55 | 3.00e-1 | 1 | 2.43e-1 | 2.43e-1 | 2.43e-1 | 2.43e-1 | 280 | +2.07e-4 | +2.07e-4 | +2.07e-4 | -2.09e-6 |
| 56 | 3.00e-1 | 1 | 2.45e-1 | 2.45e-1 | 2.45e-1 | 2.45e-1 | 278 | +3.63e-5 | +3.63e-5 | +3.63e-5 | +1.75e-6 |
| 57 | 3.00e-1 | 1 | 2.41e-1 | 2.41e-1 | 2.41e-1 | 2.41e-1 | 275 | -6.02e-5 | -6.02e-5 | -6.02e-5 | -4.44e-6 |
| 58 | 3.00e-1 | 1 | 2.34e-1 | 2.34e-1 | 2.34e-1 | 2.34e-1 | 252 | -1.21e-4 | -1.21e-4 | -1.21e-4 | -1.61e-5 |
| 59 | 3.00e-1 | 1 | 2.35e-1 | 2.35e-1 | 2.35e-1 | 2.35e-1 | 248 | +2.32e-5 | +2.32e-5 | +2.32e-5 | -1.22e-5 |
| 60 | 3.00e-1 | 1 | 2.28e-1 | 2.28e-1 | 2.28e-1 | 2.28e-1 | 238 | -1.35e-4 | -1.35e-4 | -1.35e-4 | -2.45e-5 |
| 61 | 3.00e-1 | 2 | 2.20e-1 | 2.39e-1 | 2.29e-1 | 2.39e-1 | 250 | -1.69e-4 | +3.38e-4 | +8.43e-5 | -1.29e-6 |
| 62 | 3.00e-1 | 1 | 2.40e-1 | 2.40e-1 | 2.40e-1 | 2.40e-1 | 258 | +2.18e-5 | +2.18e-5 | +2.18e-5 | +1.02e-6 |
| 63 | 3.00e-1 | 1 | 2.44e-1 | 2.44e-1 | 2.44e-1 | 2.44e-1 | 273 | +6.19e-5 | +6.19e-5 | +6.19e-5 | +7.11e-6 |
| 64 | 3.00e-1 | 1 | 2.38e-1 | 2.38e-1 | 2.38e-1 | 2.38e-1 | 261 | -1.02e-4 | -1.02e-4 | -1.02e-4 | -3.77e-6 |
| 65 | 3.00e-1 | 1 | 2.37e-1 | 2.37e-1 | 2.37e-1 | 2.37e-1 | 246 | -2.14e-5 | -2.14e-5 | -2.14e-5 | -5.53e-6 |
| 66 | 3.00e-1 | 1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 229 | -1.32e-4 | -1.32e-4 | -1.32e-4 | -1.82e-5 |
| 67 | 3.00e-1 | 2 | 2.20e-1 | 2.26e-1 | 2.23e-1 | 2.20e-1 | 199 | -1.51e-4 | -6.88e-5 | -1.10e-4 | -3.60e-5 |
| 68 | 3.00e-1 | 1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 233 | +2.39e-4 | +2.39e-4 | +2.39e-4 | -8.51e-6 |
| 69 | 3.00e-1 | 1 | 2.37e-1 | 2.37e-1 | 2.37e-1 | 2.37e-1 | 244 | +7.77e-5 | +7.77e-5 | +7.77e-5 | +1.05e-7 |
| 70 | 3.00e-1 | 1 | 2.35e-1 | 2.35e-1 | 2.35e-1 | 2.35e-1 | 239 | -3.56e-5 | -3.56e-5 | -3.56e-5 | -3.46e-6 |
| 71 | 3.00e-1 | 1 | 2.34e-1 | 2.34e-1 | 2.34e-1 | 2.34e-1 | 227 | -4.65e-6 | -4.65e-6 | -4.65e-6 | -3.58e-6 |
| 72 | 3.00e-1 | 2 | 2.28e-1 | 2.38e-1 | 2.33e-1 | 2.28e-1 | 220 | -2.08e-4 | +6.40e-5 | -7.21e-5 | -1.80e-5 |
| 73 | 3.00e-1 | 1 | 2.37e-1 | 2.37e-1 | 2.37e-1 | 2.37e-1 | 253 | +1.55e-4 | +1.55e-4 | +1.55e-4 | -6.50e-7 |
| 74 | 3.00e-1 | 1 | 2.36e-1 | 2.36e-1 | 2.36e-1 | 2.36e-1 | 233 | -1.25e-5 | -1.25e-5 | -1.25e-5 | -1.84e-6 |
| 75 | 3.00e-1 | 1 | 2.37e-1 | 2.37e-1 | 2.37e-1 | 2.37e-1 | 235 | +2.81e-5 | +2.81e-5 | +2.81e-5 | +1.16e-6 |
| 76 | 3.00e-1 | 1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 222 | -1.36e-4 | -1.36e-4 | -1.36e-4 | -1.26e-5 |
| 77 | 3.00e-1 | 2 | 2.24e-1 | 2.38e-1 | 2.31e-1 | 2.38e-1 | 244 | -1.42e-4 | +2.43e-4 | +5.03e-5 | +1.30e-6 |
| 78 | 3.00e-1 | 1 | 2.35e-1 | 2.35e-1 | 2.35e-1 | 2.35e-1 | 240 | -4.56e-5 | -4.56e-5 | -4.56e-5 | -3.39e-6 |
| 79 | 3.00e-1 | 1 | 2.35e-1 | 2.35e-1 | 2.35e-1 | 2.35e-1 | 226 | -7.70e-6 | -7.70e-6 | -7.70e-6 | -3.82e-6 |
| 80 | 3.00e-1 | 1 | 2.41e-1 | 2.41e-1 | 2.41e-1 | 2.41e-1 | 249 | +1.05e-4 | +1.05e-4 | +1.05e-4 | +7.07e-6 |
| 81 | 3.00e-1 | 2 | 2.15e-1 | 2.31e-1 | 2.23e-1 | 2.31e-1 | 251 | -6.33e-4 | +2.85e-4 | -1.74e-4 | -2.28e-5 |
| 82 | 3.00e-1 | 1 | 2.45e-1 | 2.45e-1 | 2.45e-1 | 2.45e-1 | 262 | +2.18e-4 | +2.18e-4 | +2.18e-4 | +1.33e-6 |
| 83 | 3.00e-1 | 1 | 2.40e-1 | 2.40e-1 | 2.40e-1 | 2.40e-1 | 242 | -7.86e-5 | -7.86e-5 | -7.86e-5 | -6.66e-6 |
| 84 | 3.00e-1 | 1 | 2.49e-1 | 2.49e-1 | 2.49e-1 | 2.49e-1 | 257 | +1.30e-4 | +1.30e-4 | +1.30e-4 | +6.97e-6 |
| 85 | 3.00e-1 | 1 | 2.54e-1 | 2.54e-1 | 2.54e-1 | 2.54e-1 | 289 | +8.10e-5 | +8.10e-5 | +8.10e-5 | +1.44e-5 |
| 86 | 3.00e-1 | 1 | 2.42e-1 | 2.42e-1 | 2.42e-1 | 2.42e-1 | 256 | -2.01e-4 | -2.01e-4 | -2.01e-4 | -7.15e-6 |
| 87 | 3.00e-1 | 1 | 2.49e-1 | 2.49e-1 | 2.49e-1 | 2.49e-1 | 278 | +1.02e-4 | +1.02e-4 | +1.02e-4 | +3.78e-6 |
| 88 | 3.00e-1 | 1 | 2.47e-1 | 2.47e-1 | 2.47e-1 | 2.47e-1 | 266 | -3.01e-5 | -3.01e-5 | -3.01e-5 | +3.89e-7 |
| 89 | 3.00e-1 | 1 | 2.46e-1 | 2.46e-1 | 2.46e-1 | 2.46e-1 | 258 | -5.49e-6 | -5.49e-6 | -5.49e-6 | -1.99e-7 |
| 90 | 3.00e-1 | 1 | 2.45e-1 | 2.45e-1 | 2.45e-1 | 2.45e-1 | 258 | -2.39e-5 | -2.39e-5 | -2.39e-5 | -2.57e-6 |
| 91 | 3.00e-1 | 2 | 2.27e-1 | 2.40e-1 | 2.33e-1 | 2.27e-1 | 204 | -2.73e-4 | -8.25e-5 | -1.78e-4 | -3.68e-5 |
| 92 | 3.00e-1 | 1 | 2.38e-1 | 2.38e-1 | 2.38e-1 | 2.38e-1 | 238 | +1.99e-4 | +1.99e-4 | +1.99e-4 | -1.32e-5 |
| 93 | 3.00e-1 | 1 | 2.47e-1 | 2.47e-1 | 2.47e-1 | 2.47e-1 | 261 | +1.39e-4 | +1.39e-4 | +1.39e-4 | +2.01e-6 |
| 94 | 3.00e-1 | 1 | 2.48e-1 | 2.48e-1 | 2.48e-1 | 2.48e-1 | 264 | +1.43e-5 | +1.43e-5 | +1.43e-5 | +3.24e-6 |
| 95 | 3.00e-1 | 1 | 2.48e-1 | 2.48e-1 | 2.48e-1 | 2.48e-1 | 269 | +5.91e-6 | +5.91e-6 | +5.91e-6 | +3.51e-6 |
| 96 | 3.00e-1 | 1 | 2.41e-1 | 2.41e-1 | 2.41e-1 | 2.41e-1 | 243 | -1.12e-4 | -1.12e-4 | -1.12e-4 | -8.07e-6 |
| 97 | 3.00e-1 | 1 | 2.58e-1 | 2.58e-1 | 2.58e-1 | 2.58e-1 | 285 | +2.36e-4 | +2.36e-4 | +2.36e-4 | +1.64e-5 |
| 98 | 3.00e-1 | 2 | 2.27e-1 | 2.39e-1 | 2.33e-1 | 2.27e-1 | 207 | -3.37e-4 | -2.46e-4 | -2.92e-4 | -4.17e-5 |
| 99 | 3.00e-1 | 1 | 2.34e-1 | 2.34e-1 | 2.34e-1 | 2.34e-1 | 222 | +1.32e-4 | +1.32e-4 | +1.32e-4 | -2.43e-5 |
| 100 | 3.00e-2 | 1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 222 | -2.65e-4 | -2.65e-4 | -2.65e-4 | -4.84e-5 |
| 101 | 3.00e-2 | 1 | 1.08e-1 | 1.08e-1 | 1.08e-1 | 1.08e-1 | 220 | -3.23e-3 | -3.23e-3 | -3.23e-3 | -3.67e-4 |
| 102 | 3.00e-2 | 2 | 3.40e-2 | 5.74e-2 | 4.57e-2 | 3.40e-2 | 210 | -2.71e-3 | -2.50e-3 | -2.60e-3 | -7.90e-4 |
| 103 | 3.00e-2 | 1 | 2.97e-2 | 2.97e-2 | 2.97e-2 | 2.97e-2 | 265 | -5.05e-4 | -5.05e-4 | -5.05e-4 | -7.62e-4 |
| 104 | 3.00e-2 | 1 | 2.82e-2 | 2.82e-2 | 2.82e-2 | 2.82e-2 | 243 | -2.24e-4 | -2.24e-4 | -2.24e-4 | -7.08e-4 |
| 105 | 3.00e-2 | 1 | 2.79e-2 | 2.79e-2 | 2.79e-2 | 2.79e-2 | 241 | -3.19e-5 | -3.19e-5 | -3.19e-5 | -6.40e-4 |
| 106 | 3.00e-2 | 1 | 2.96e-2 | 2.96e-2 | 2.96e-2 | 2.96e-2 | 253 | +2.22e-4 | +2.22e-4 | +2.22e-4 | -5.54e-4 |
| 107 | 3.00e-2 | 1 | 2.92e-2 | 2.92e-2 | 2.92e-2 | 2.92e-2 | 238 | -4.44e-5 | -4.44e-5 | -4.44e-5 | -5.03e-4 |
| 108 | 3.00e-2 | 2 | 2.91e-2 | 3.00e-2 | 2.95e-2 | 2.91e-2 | 185 | -1.80e-4 | +1.14e-4 | -3.30e-5 | -4.15e-4 |
| 109 | 3.00e-2 | 1 | 3.08e-2 | 3.08e-2 | 3.08e-2 | 3.08e-2 | 215 | +2.79e-4 | +2.79e-4 | +2.79e-4 | -3.46e-4 |
| 110 | 3.00e-2 | 1 | 3.13e-2 | 3.13e-2 | 3.13e-2 | 3.13e-2 | 222 | +6.74e-5 | +6.74e-5 | +6.74e-5 | -3.05e-4 |
| 111 | 3.00e-2 | 1 | 3.37e-2 | 3.37e-2 | 3.37e-2 | 3.37e-2 | 243 | +3.01e-4 | +3.01e-4 | +3.01e-4 | -2.44e-4 |
| 112 | 3.00e-2 | 2 | 3.21e-2 | 3.43e-2 | 3.32e-2 | 3.21e-2 | 184 | -3.66e-4 | +7.54e-5 | -1.45e-4 | -2.27e-4 |
| 113 | 3.00e-2 | 1 | 3.64e-2 | 3.64e-2 | 3.64e-2 | 3.64e-2 | 243 | +5.25e-4 | +5.25e-4 | +5.25e-4 | -1.52e-4 |
| 114 | 3.00e-2 | 2 | 3.33e-2 | 3.52e-2 | 3.42e-2 | 3.33e-2 | 172 | -3.24e-4 | -1.62e-4 | -2.43e-4 | -1.70e-4 |
| 115 | 3.00e-2 | 1 | 3.60e-2 | 3.60e-2 | 3.60e-2 | 3.60e-2 | 210 | +3.69e-4 | +3.69e-4 | +3.69e-4 | -1.16e-4 |
| 116 | 3.00e-2 | 1 | 3.74e-2 | 3.74e-2 | 3.74e-2 | 3.74e-2 | 214 | +1.80e-4 | +1.80e-4 | +1.80e-4 | -8.67e-5 |
| 117 | 3.00e-2 | 1 | 4.02e-2 | 4.02e-2 | 4.02e-2 | 4.02e-2 | 236 | +3.05e-4 | +3.05e-4 | +3.05e-4 | -4.75e-5 |
| 118 | 3.00e-2 | 2 | 3.71e-2 | 3.93e-2 | 3.82e-2 | 3.71e-2 | 179 | -3.25e-4 | -1.01e-4 | -2.13e-4 | -8.01e-5 |
| 119 | 3.00e-2 | 1 | 4.11e-2 | 4.11e-2 | 4.11e-2 | 4.11e-2 | 219 | +4.68e-4 | +4.68e-4 | +4.68e-4 | -2.53e-5 |
| 120 | 3.00e-2 | 1 | 4.12e-2 | 4.12e-2 | 4.12e-2 | 4.12e-2 | 219 | +1.65e-5 | +1.65e-5 | +1.65e-5 | -2.11e-5 |
| 121 | 3.00e-2 | 2 | 3.92e-2 | 4.23e-2 | 4.08e-2 | 3.92e-2 | 172 | -4.34e-4 | +1.17e-4 | -1.59e-4 | -5.00e-5 |
| 122 | 3.00e-2 | 1 | 4.13e-2 | 4.13e-2 | 4.13e-2 | 4.13e-2 | 193 | +2.69e-4 | +2.69e-4 | +2.69e-4 | -1.81e-5 |
| 123 | 3.00e-2 | 2 | 4.02e-2 | 4.33e-2 | 4.17e-2 | 4.02e-2 | 161 | -4.63e-4 | +2.22e-4 | -1.20e-4 | -4.10e-5 |
| 124 | 3.00e-2 | 1 | 4.43e-2 | 4.43e-2 | 4.43e-2 | 4.43e-2 | 213 | +4.58e-4 | +4.58e-4 | +4.58e-4 | +8.93e-6 |
| 125 | 3.00e-2 | 2 | 4.39e-2 | 4.66e-2 | 4.53e-2 | 4.39e-2 | 164 | -3.64e-4 | +2.31e-4 | -6.65e-5 | -8.38e-6 |
| 126 | 3.00e-2 | 1 | 4.74e-2 | 4.74e-2 | 4.74e-2 | 4.74e-2 | 212 | +3.59e-4 | +3.59e-4 | +3.59e-4 | +2.84e-5 |
| 127 | 3.00e-2 | 2 | 4.38e-2 | 4.65e-2 | 4.52e-2 | 4.38e-2 | 161 | -3.72e-4 | -9.06e-5 | -2.31e-4 | -2.24e-5 |
| 128 | 3.00e-2 | 1 | 4.62e-2 | 4.62e-2 | 4.62e-2 | 4.62e-2 | 201 | +2.65e-4 | +2.65e-4 | +2.65e-4 | +6.35e-6 |
| 129 | 3.00e-2 | 2 | 4.61e-2 | 4.75e-2 | 4.68e-2 | 4.61e-2 | 166 | -1.78e-4 | +1.43e-4 | -1.75e-5 | +2.14e-7 |
| 130 | 3.00e-2 | 1 | 4.93e-2 | 4.93e-2 | 4.93e-2 | 4.93e-2 | 214 | +3.16e-4 | +3.16e-4 | +3.16e-4 | +3.18e-5 |
| 131 | 3.00e-2 | 1 | 5.32e-2 | 5.32e-2 | 5.32e-2 | 5.32e-2 | 212 | +3.60e-4 | +3.60e-4 | +3.60e-4 | +6.46e-5 |
| 132 | 3.00e-2 | 2 | 4.63e-2 | 5.06e-2 | 4.85e-2 | 4.63e-2 | 151 | -5.80e-4 | -2.67e-4 | -4.23e-4 | -2.97e-5 |
| 133 | 3.00e-2 | 1 | 5.61e-2 | 5.61e-2 | 5.61e-2 | 5.61e-2 | 224 | +8.57e-4 | +8.57e-4 | +8.57e-4 | +5.89e-5 |
| 134 | 3.00e-2 | 2 | 4.79e-2 | 5.41e-2 | 5.10e-2 | 4.79e-2 | 143 | -8.46e-4 | -1.79e-4 | -5.13e-4 | -5.30e-5 |
| 135 | 3.00e-2 | 2 | 5.05e-2 | 5.27e-2 | 5.16e-2 | 5.05e-2 | 151 | -2.77e-4 | +5.23e-4 | +1.23e-4 | -2.35e-5 |
| 136 | 3.00e-2 | 1 | 5.20e-2 | 5.20e-2 | 5.20e-2 | 5.20e-2 | 181 | +1.61e-4 | +1.61e-4 | +1.61e-4 | -5.06e-6 |
| 137 | 3.00e-2 | 2 | 5.06e-2 | 5.33e-2 | 5.19e-2 | 5.06e-2 | 143 | -3.51e-4 | +1.35e-4 | -1.08e-4 | -2.71e-5 |
| 138 | 3.00e-2 | 2 | 4.93e-2 | 5.46e-2 | 5.19e-2 | 4.93e-2 | 143 | -7.09e-4 | +4.07e-4 | -1.51e-4 | -5.63e-5 |
| 139 | 3.00e-2 | 1 | 5.69e-2 | 5.69e-2 | 5.69e-2 | 5.69e-2 | 195 | +7.37e-4 | +7.37e-4 | +7.37e-4 | +2.31e-5 |
| 140 | 3.00e-2 | 2 | 5.12e-2 | 5.99e-2 | 5.55e-2 | 5.12e-2 | 143 | -1.09e-3 | +2.69e-4 | -4.12e-4 | -6.65e-5 |
| 141 | 3.00e-2 | 2 | 5.16e-2 | 5.49e-2 | 5.33e-2 | 5.16e-2 | 143 | -4.29e-4 | +3.87e-4 | -2.11e-5 | -6.19e-5 |
| 142 | 3.00e-2 | 1 | 5.88e-2 | 5.88e-2 | 5.88e-2 | 5.88e-2 | 175 | +7.41e-4 | +7.41e-4 | +7.41e-4 | +1.84e-5 |
| 143 | 3.00e-2 | 2 | 5.51e-2 | 6.35e-2 | 5.93e-2 | 5.51e-2 | 140 | -1.01e-3 | +4.02e-4 | -3.04e-4 | -4.99e-5 |
| 144 | 3.00e-2 | 2 | 5.26e-2 | 5.65e-2 | 5.45e-2 | 5.26e-2 | 134 | -5.39e-4 | +1.52e-4 | -1.94e-4 | -8.06e-5 |
| 145 | 3.00e-2 | 2 | 5.29e-2 | 5.84e-2 | 5.57e-2 | 5.29e-2 | 126 | -7.82e-4 | +5.99e-4 | -9.16e-5 | -8.96e-5 |
| 146 | 3.00e-2 | 2 | 5.63e-2 | 5.94e-2 | 5.78e-2 | 5.63e-2 | 137 | -3.90e-4 | +6.62e-4 | +1.36e-4 | -5.20e-5 |
| 147 | 3.00e-2 | 2 | 5.38e-2 | 5.91e-2 | 5.65e-2 | 5.38e-2 | 126 | -7.42e-4 | +2.97e-4 | -2.23e-4 | -8.96e-5 |
| 148 | 3.00e-2 | 2 | 5.57e-2 | 5.89e-2 | 5.73e-2 | 5.57e-2 | 118 | -4.71e-4 | +5.40e-4 | +3.49e-5 | -7.10e-5 |
| 149 | 3.00e-3 | 2 | 5.54e-2 | 5.69e-2 | 5.62e-2 | 5.54e-2 | 126 | -2.12e-4 | +1.31e-4 | -4.09e-5 | -6.70e-5 |
| 150 | 3.00e-3 | 1 | 3.34e-2 | 3.34e-2 | 3.34e-2 | 3.34e-2 | 180 | -2.82e-3 | -2.82e-3 | -2.82e-3 | -3.42e-4 |
| 151 | 3.00e-3 | 2 | 8.63e-3 | 1.65e-2 | 1.26e-2 | 8.63e-3 | 117 | -5.55e-3 | -4.34e-3 | -4.94e-3 | -1.22e-3 |
| 152 | 3.00e-3 | 2 | 5.31e-3 | 5.38e-3 | 5.34e-3 | 5.31e-3 | 244 | -4.05e-3 | -5.44e-5 | -2.05e-3 | -1.36e-3 |
| 153 | 3.00e-3 | 1 | 5.63e-3 | 5.63e-3 | 5.63e-3 | 5.63e-3 | 262 | +2.23e-4 | +2.23e-4 | +2.23e-4 | -1.20e-3 |
| 154 | 3.00e-3 | 1 | 5.71e-3 | 5.71e-3 | 5.71e-3 | 5.71e-3 | 277 | +5.48e-5 | +5.48e-5 | +5.48e-5 | -1.08e-3 |
| 155 | 3.00e-3 | 1 | 5.70e-3 | 5.70e-3 | 5.70e-3 | 5.70e-3 | 271 | -1.03e-5 | -1.03e-5 | -1.03e-5 | -9.69e-4 |
| 156 | 3.00e-3 | 1 | 5.59e-3 | 5.59e-3 | 5.59e-3 | 5.59e-3 | 273 | -6.72e-5 | -6.72e-5 | -6.72e-5 | -8.79e-4 |
| 157 | 3.00e-3 | 1 | 5.53e-3 | 5.53e-3 | 5.53e-3 | 5.53e-3 | 261 | -4.58e-5 | -4.58e-5 | -4.58e-5 | -7.96e-4 |
| 158 | 3.00e-3 | 1 | 5.32e-3 | 5.32e-3 | 5.32e-3 | 5.32e-3 | 258 | -1.46e-4 | -1.46e-4 | -1.46e-4 | -7.31e-4 |
| 159 | 3.00e-3 | 1 | 5.64e-3 | 5.64e-3 | 5.64e-3 | 5.64e-3 | 267 | +2.17e-4 | +2.17e-4 | +2.17e-4 | -6.36e-4 |
| 160 | 3.00e-3 | 1 | 5.64e-3 | 5.64e-3 | 5.64e-3 | 5.64e-3 | 267 | +1.71e-6 | +1.71e-6 | +1.71e-6 | -5.72e-4 |
| 161 | 3.00e-3 | 1 | 5.65e-3 | 5.65e-3 | 5.65e-3 | 5.65e-3 | 264 | +6.63e-6 | +6.63e-6 | +6.63e-6 | -5.14e-4 |
| 162 | 3.00e-3 | 1 | 5.81e-3 | 5.81e-3 | 5.81e-3 | 5.81e-3 | 273 | +1.02e-4 | +1.02e-4 | +1.02e-4 | -4.53e-4 |
| 163 | 3.00e-3 | 2 | 5.21e-3 | 5.76e-3 | 5.49e-3 | 5.21e-3 | 208 | -4.81e-4 | -3.47e-5 | -2.58e-4 | -4.18e-4 |
| 165 | 3.00e-3 | 2 | 5.30e-3 | 5.87e-3 | 5.59e-3 | 5.30e-3 | 206 | -4.98e-4 | +4.04e-4 | -4.67e-5 | -3.52e-4 |
| 166 | 3.00e-3 | 1 | 5.52e-3 | 5.52e-3 | 5.52e-3 | 5.52e-3 | 276 | +1.45e-4 | +1.45e-4 | +1.45e-4 | -3.02e-4 |
| 167 | 3.00e-3 | 1 | 5.66e-3 | 5.66e-3 | 5.66e-3 | 5.66e-3 | 269 | +9.07e-5 | +9.07e-5 | +9.07e-5 | -2.63e-4 |
| 168 | 3.00e-3 | 1 | 5.70e-3 | 5.70e-3 | 5.70e-3 | 5.70e-3 | 252 | +3.13e-5 | +3.13e-5 | +3.13e-5 | -2.33e-4 |
| 169 | 3.00e-3 | 1 | 5.54e-3 | 5.54e-3 | 5.54e-3 | 5.54e-3 | 248 | -1.16e-4 | -1.16e-4 | -1.16e-4 | -2.22e-4 |
| 170 | 3.00e-3 | 1 | 5.42e-3 | 5.42e-3 | 5.42e-3 | 5.42e-3 | 257 | -8.46e-5 | -8.46e-5 | -8.46e-5 | -2.08e-4 |
| 171 | 3.00e-3 | 1 | 5.73e-3 | 5.73e-3 | 5.73e-3 | 5.73e-3 | 266 | +2.07e-4 | +2.07e-4 | +2.07e-4 | -1.66e-4 |
| 172 | 3.00e-3 | 1 | 5.52e-3 | 5.52e-3 | 5.52e-3 | 5.52e-3 | 256 | -1.43e-4 | -1.43e-4 | -1.43e-4 | -1.64e-4 |
| 173 | 3.00e-3 | 2 | 5.08e-3 | 5.41e-3 | 5.24e-3 | 5.08e-3 | 202 | -3.07e-4 | -8.39e-5 | -1.95e-4 | -1.71e-4 |
| 174 | 3.00e-3 | 1 | 5.32e-3 | 5.32e-3 | 5.32e-3 | 5.32e-3 | 234 | +1.93e-4 | +1.93e-4 | +1.93e-4 | -1.35e-4 |
| 175 | 3.00e-3 | 1 | 5.55e-3 | 5.55e-3 | 5.55e-3 | 5.55e-3 | 235 | +1.83e-4 | +1.83e-4 | +1.83e-4 | -1.03e-4 |
| 176 | 3.00e-3 | 1 | 5.52e-3 | 5.52e-3 | 5.52e-3 | 5.52e-3 | 252 | -2.08e-5 | -2.08e-5 | -2.08e-5 | -9.48e-5 |
| 177 | 3.00e-3 | 1 | 5.34e-3 | 5.34e-3 | 5.34e-3 | 5.34e-3 | 241 | -1.35e-4 | -1.35e-4 | -1.35e-4 | -9.88e-5 |
| 178 | 3.00e-3 | 2 | 4.99e-3 | 5.21e-3 | 5.10e-3 | 4.99e-3 | 194 | -2.26e-4 | -1.06e-4 | -1.66e-4 | -1.12e-4 |
| 179 | 3.00e-3 | 1 | 5.37e-3 | 5.37e-3 | 5.37e-3 | 5.37e-3 | 232 | +3.16e-4 | +3.16e-4 | +3.16e-4 | -6.94e-5 |
| 180 | 3.00e-3 | 1 | 5.26e-3 | 5.26e-3 | 5.26e-3 | 5.26e-3 | 219 | -8.96e-5 | -8.96e-5 | -8.96e-5 | -7.14e-5 |
| 181 | 3.00e-3 | 1 | 5.55e-3 | 5.55e-3 | 5.55e-3 | 5.55e-3 | 241 | +2.22e-4 | +2.22e-4 | +2.22e-4 | -4.21e-5 |
| 182 | 3.00e-3 | 2 | 5.16e-3 | 5.69e-3 | 5.43e-3 | 5.16e-3 | 185 | -5.31e-4 | +1.01e-4 | -2.15e-4 | -7.81e-5 |
| 183 | 3.00e-3 | 1 | 5.37e-3 | 5.37e-3 | 5.37e-3 | 5.37e-3 | 244 | +1.64e-4 | +1.64e-4 | +1.64e-4 | -5.39e-5 |
| 184 | 3.00e-3 | 1 | 5.28e-3 | 5.28e-3 | 5.28e-3 | 5.28e-3 | 219 | -8.02e-5 | -8.02e-5 | -8.02e-5 | -5.65e-5 |
| 185 | 3.00e-3 | 1 | 5.42e-3 | 5.42e-3 | 5.42e-3 | 5.42e-3 | 236 | +1.13e-4 | +1.13e-4 | +1.13e-4 | -3.96e-5 |
| 186 | 3.00e-3 | 2 | 5.00e-3 | 5.80e-3 | 5.40e-3 | 5.00e-3 | 184 | -8.12e-4 | +2.73e-4 | -2.69e-4 | -8.87e-5 |
| 187 | 3.00e-3 | 1 | 5.45e-3 | 5.45e-3 | 5.45e-3 | 5.45e-3 | 223 | +3.86e-4 | +3.86e-4 | +3.86e-4 | -4.12e-5 |
| 188 | 3.00e-3 | 1 | 5.15e-3 | 5.15e-3 | 5.15e-3 | 5.15e-3 | 210 | -2.69e-4 | -2.69e-4 | -2.69e-4 | -6.40e-5 |
| 189 | 3.00e-3 | 2 | 4.93e-3 | 5.07e-3 | 5.00e-3 | 4.93e-3 | 174 | -1.57e-4 | -7.13e-5 | -1.14e-4 | -7.39e-5 |
| 190 | 3.00e-3 | 1 | 5.10e-3 | 5.10e-3 | 5.10e-3 | 5.10e-3 | 218 | +1.55e-4 | +1.55e-4 | +1.55e-4 | -5.11e-5 |
| 191 | 3.00e-3 | 1 | 5.26e-3 | 5.26e-3 | 5.26e-3 | 5.26e-3 | 226 | +1.39e-4 | +1.39e-4 | +1.39e-4 | -3.20e-5 |
| 192 | 3.00e-3 | 2 | 4.90e-3 | 5.32e-3 | 5.11e-3 | 4.90e-3 | 174 | -4.70e-4 | +4.73e-5 | -2.11e-4 | -6.87e-5 |
| 193 | 3.00e-3 | 1 | 5.28e-3 | 5.28e-3 | 5.28e-3 | 5.28e-3 | 226 | +3.29e-4 | +3.29e-4 | +3.29e-4 | -2.89e-5 |
| 194 | 3.00e-3 | 2 | 4.86e-3 | 5.02e-3 | 4.94e-3 | 4.86e-3 | 165 | -2.57e-4 | -1.99e-4 | -2.28e-4 | -6.65e-5 |
| 195 | 3.00e-3 | 1 | 5.15e-3 | 5.15e-3 | 5.15e-3 | 5.15e-3 | 213 | +2.74e-4 | +2.74e-4 | +2.74e-4 | -3.24e-5 |
| 196 | 3.00e-3 | 1 | 5.09e-3 | 5.09e-3 | 5.09e-3 | 5.09e-3 | 200 | -5.98e-5 | -5.98e-5 | -5.98e-5 | -3.51e-5 |
| 197 | 3.00e-3 | 2 | 4.95e-3 | 5.08e-3 | 5.02e-3 | 4.95e-3 | 170 | -1.44e-4 | -8.70e-6 | -7.63e-5 | -4.36e-5 |
| 198 | 3.00e-3 | 1 | 5.18e-3 | 5.18e-3 | 5.18e-3 | 5.18e-3 | 213 | +2.13e-4 | +2.13e-4 | +2.13e-4 | -1.79e-5 |
| 199 | 3.00e-3 | 2 | 4.79e-3 | 5.08e-3 | 4.94e-3 | 4.79e-3 | 167 | -3.49e-4 | -8.89e-5 | -2.19e-4 | -5.74e-5 |

