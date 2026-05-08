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

GPU columns = compute utilization % (not load). Idle = total time with <5% utilization.

### resnet-graph

> Published: CIFAR-10 91.25% acc, Graph builder ([He et al. 2015](https://arxiv.org/abs/1512.03385), Table 6)

| Mode | Loss | Eval | vs Ref | Total (s) | Syncs | Avg Sync (ms) | GPU0 | GPU1 | GPU2 | Idle (s) |
|------|------|------|--------|-----------|-------|--------------|------|------|------|----------|
| cpu-async | 0.011964 | 0.9297 | +0.0172 | 5002.2 | 587 | 217.1 | 100% | 100% | 100% | 6.5 |

## Best Mode per Model

| Model | Best Eval | Mode | Fastest (within 2% of solo-0) | Mode |
|-------|-----------|------|-------------------------------|------|
| resnet-graph | 0.9297 | cpu-async | - | - |

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
| cpu-async | 2.3170 | 0.8594 | 0.6259 | 0.5480 | 0.5044 | 0.4747 | 0.4520 | 0.4410 | 0.4275 | 0.4127 | 0.1526 | 0.1055 | 0.0825 | 0.0734 | 0.0655 | 0.0223 | 0.0174 | 0.0145 | 0.0117 | 0.0120 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | cpu-async | 0 | cuda:0 | 0.3512 | 1.1 |
| resnet-graph | cpu-async | 1 | cuda:1 | 0.3280 | 1.3 |
| resnet-graph | cpu-async | 2 | cuda:2 | 0.3208 | 1.2 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) | GPU2 Peak (MB) | GPU2 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|---------------|---------------|
| resnet-graph | cpu-async | 514 | 513 | 582 | 580 | 576 | 567 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| cpu-async | gpu1 | 5000.6 | 1.6 | epoch-boundary(199) |
| cpu-async | gpu2 | 5000.6 | 1.6 | epoch-boundary(199) |
| cpu-async | gpu1 | 1056.6 | 1.1 | epoch-boundary(41) |
| cpu-async | gpu2 | 1056.8 | 0.9 | epoch-boundary(41) |
| cpu-async | gpu1 | 4084.3 | 0.7 | epoch-boundary(162) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | cpu-async | gpu1 | 3.4s | 0.0s | 0.0s | 0.0s | 4.0s |
| resnet-graph | cpu-async | gpu2 | 2.5s | 0.0s | 0.0s | 0.0s | 2.5s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | cpu-async | 351 | 0 | 587 | 217.1 | 4830/21856 | 587 | 217.1 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | cpu-async | 312.4 | 6.2% |

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
| resnet-graph | cpu-async | 199 | 587 | 0 | 5.18e-3 | -2.88e-5 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | cpu-async | 0 | 587 | 7.54e-2 | 8.43e-2 | 1.93e-3 | 9.65e-1 | 21.8 | -2.90e-4 | 1.78e-3 |
| resnet-graph | cpu-async | 1 | 587 | 7.60e-2 | 8.54e-2 | 1.96e-3 | 9.45e-1 | 34.2 | -2.97e-4 | 1.81e-3 |
| resnet-graph | cpu-async | 2 | 587 | 7.70e-2 | 8.52e-2 | 2.03e-3 | 9.12e-1 | 44.0 | -3.10e-4 | 1.92e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | cpu-async | rank0 ↔ rank1 | +0.9981 |
| resnet-graph | cpu-async | rank0 ↔ rank2 | +0.9972 |
| resnet-graph | cpu-async | rank1 ↔ rank2 | +0.9971 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | cpu-async | 62 (1,2,4,5,6,7,8,9…149,150) | 0 (—) | — | 1,2,4,5,6,7,8,9…149,150 | — |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | cpu-async | 0e0 | 3 | 0 | 0 |
| resnet-graph | cpu-async | 0e0 | 5 | 0 | 0 |
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
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 263 | +0.082 |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | 245 | +0.111 |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | 75 | +0.051 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | cpu-async | 585 | -0.004 | 198 | +0.290 | +0.412 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | cpu-async | 586 | 6.04e1–8.80e1 | 7.21e1 | 1.91e-3 | 8.01e-3 | 1.81e-1 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 265 | 30–77966 | +1.484e-5 | 0.439 | +1.489e-5 | 0.444 | 100 | +1.048e-5 | 0.388 | 30–754 | +1.831e-3 | 0.511 |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 252 | 949–77966 | +1.517e-5 | 0.576 | +1.520e-5 | 0.582 | 99 | +1.035e-5 | 0.375 | 74–754 | +1.903e-3 | 0.695 |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | 246 | 78312–117058 | +1.879e-5 | 0.356 | +1.862e-5 | 0.369 | 50 | +1.729e-5 | 0.350 | 89–370 | +1.923e-3 | 0.138 |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | 76 | 117274–155935 | +6.742e-6 | 0.034 | +7.051e-6 | 0.038 | 49 | +2.768e-6 | 0.025 | 104–783 | +7.638e-4 | 0.146 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | +1.831e-3 | r0: +1.816e-3, r1: +1.837e-3, r2: +1.842e-3 | r0: 0.504, r1: 0.511, r2: 0.513 | 1.01× | ✓ |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +1.903e-3 | r0: +1.886e-3, r1: +1.911e-3, r2: +1.912e-3 | r0: 0.686, r1: 0.696, r2: 0.694 | 1.01× | ✓ |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | +1.923e-3 | r0: +1.786e-3, r1: +1.951e-3, r2: +2.016e-3 | r0: 0.121, r1: 0.142, r2: 0.146 | 1.13× | ✓ |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | +7.638e-4 | r0: +7.216e-4, r1: +7.960e-4, r2: +7.771e-4 | r0: 0.134, r1: 0.146, r2: 0.157 | 1.10× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | cpu-async | `▇▇▇▇▇████████████████████▅▄▄▄▄▄▄▅▅▅▅▅▄▁▁▁▁▁▁▁▁▁▁▁▁` | `▁▇▇▇▇████████████████████▆▇█▇▇▇▇▇▇▇██▅▆▇▇█████████` |

### Per-Epoch Detail: resnet-graph / cpu-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 13 | 2.78e-2 | 9.65e-1 | 2.23e-1 | 3.40e-2 | 39 | -5.65e-2 | +5.21e-3 | -2.13e-2 | -1.52e-2 |
| 1 | 3.00e-1 | 6 | 3.25e-2 | 4.37e-2 | 3.64e-2 | 3.70e-2 | 37 | -8.10e-3 | +3.40e-3 | -2.64e-4 | -6.91e-3 |
| 2 | 3.00e-1 | 9 | 4.01e-2 | 5.66e-2 | 4.60e-2 | 4.72e-2 | 33 | -9.61e-3 | +5.99e-3 | -3.94e-5 | -2.39e-3 |
| 3 | 3.00e-1 | 6 | 4.99e-2 | 7.69e-2 | 5.53e-2 | 5.22e-2 | 30 | -1.49e-2 | +5.68e-3 | -1.30e-3 | -1.76e-3 |
| 4 | 3.00e-1 | 9 | 5.77e-2 | 7.95e-2 | 6.37e-2 | 6.54e-2 | 35 | -1.00e-2 | +5.83e-3 | -1.12e-4 | -6.67e-4 |
| 5 | 3.00e-1 | 6 | 6.67e-2 | 9.31e-2 | 7.38e-2 | 7.11e-2 | 31 | -9.41e-3 | +4.53e-3 | -7.81e-4 | -6.73e-4 |
| 6 | 3.00e-1 | 7 | 7.30e-2 | 1.09e-1 | 8.11e-2 | 7.80e-2 | 33 | -1.01e-2 | +4.98e-3 | -8.50e-4 | -6.69e-4 |
| 7 | 3.00e-1 | 7 | 7.28e-2 | 1.08e-1 | 8.67e-2 | 9.13e-2 | 43 | -9.83e-3 | +4.76e-3 | -7.21e-4 | -5.58e-4 |
| 8 | 3.00e-1 | 7 | 8.60e-2 | 1.24e-1 | 9.50e-2 | 8.60e-2 | 35 | -7.34e-3 | +3.52e-3 | -9.30e-4 | -7.33e-4 |
| 9 | 3.00e-1 | 7 | 9.53e-2 | 1.25e-1 | 1.02e-1 | 9.93e-2 | 43 | -7.66e-3 | +4.19e-3 | -3.93e-4 | -5.26e-4 |
| 10 | 3.00e-1 | 5 | 9.35e-2 | 1.25e-1 | 1.02e-1 | 9.73e-2 | 41 | -7.10e-3 | +2.99e-3 | -7.68e-4 | -6.02e-4 |
| 11 | 3.00e-1 | 8 | 9.39e-2 | 1.36e-1 | 1.01e-1 | 9.53e-2 | 36 | -9.54e-3 | +3.82e-3 | -8.78e-4 | -6.65e-4 |
| 12 | 3.00e-1 | 5 | 9.46e-2 | 1.42e-1 | 1.06e-1 | 9.46e-2 | 35 | -9.31e-3 | +4.20e-3 | -1.44e-3 | -9.67e-4 |
| 13 | 3.00e-1 | 8 | 9.81e-2 | 1.36e-1 | 1.06e-1 | 1.02e-1 | 38 | -8.55e-3 | +4.58e-3 | -4.91e-4 | -6.47e-4 |
| 14 | 3.00e-1 | 6 | 9.66e-2 | 1.41e-1 | 1.11e-1 | 9.66e-2 | 32 | -6.92e-3 | +4.08e-3 | -1.17e-3 | -9.53e-4 |
| 15 | 3.00e-1 | 7 | 9.51e-2 | 1.39e-1 | 1.05e-1 | 9.86e-2 | 34 | -7.33e-3 | +4.63e-3 | -6.65e-4 | -7.61e-4 |
| 16 | 3.00e-1 | 6 | 1.14e-1 | 1.41e-1 | 1.23e-1 | 1.17e-1 | 48 | -2.78e-3 | +4.18e-3 | +6.55e-5 | -4.21e-4 |
| 17 | 3.00e-1 | 4 | 1.12e-1 | 1.49e-1 | 1.23e-1 | 1.12e-1 | 43 | -5.09e-3 | +2.64e-3 | -8.97e-4 | -6.00e-4 |
| 18 | 3.00e-1 | 6 | 1.09e-1 | 1.57e-1 | 1.20e-1 | 1.13e-1 | 41 | -7.97e-3 | +3.39e-3 | -8.36e-4 | -6.64e-4 |
| 19 | 3.00e-1 | 6 | 1.07e-1 | 1.49e-1 | 1.18e-1 | 1.07e-1 | 37 | -4.67e-3 | +3.40e-3 | -7.14e-4 | -6.97e-4 |
| 20 | 3.00e-1 | 7 | 1.03e-1 | 1.54e-1 | 1.15e-1 | 1.12e-1 | 43 | -9.89e-3 | +3.76e-3 | -8.58e-4 | -6.72e-4 |
| 21 | 3.00e-1 | 1 | 1.11e-1 | 1.11e-1 | 1.11e-1 | 1.11e-1 | 39 | -1.38e-4 | -1.38e-4 | -1.38e-4 | -6.18e-4 |
| 22 | 3.00e-1 | 2 | 2.08e-1 | 2.10e-1 | 2.09e-1 | 2.08e-1 | 218 | -3.83e-5 | +2.20e-3 | +1.08e-3 | -3.06e-4 |
| 23 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 233 | +1.05e-5 | +1.05e-5 | +1.05e-5 | -2.75e-4 |
| 24 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 235 | -5.53e-6 | -5.53e-6 | -5.53e-6 | -2.48e-4 |
| 25 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 249 | +8.16e-5 | +8.16e-5 | +8.16e-5 | -2.15e-4 |
| 26 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 246 | -3.85e-5 | -3.85e-5 | -3.85e-5 | -1.97e-4 |
| 27 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 244 | -2.28e-5 | -2.28e-5 | -2.28e-5 | -1.80e-4 |
| 28 | 3.00e-1 | 1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 277 | +2.11e-4 | +2.11e-4 | +2.11e-4 | -1.41e-4 |
| 29 | 3.00e-1 | 2 | 2.04e-1 | 2.13e-1 | 2.08e-1 | 2.04e-1 | 206 | -2.15e-4 | -1.62e-4 | -1.88e-4 | -1.50e-4 |
| 30 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 249 | +1.99e-4 | +1.99e-4 | +1.99e-4 | -1.15e-4 |
| 31 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 246 | +2.38e-6 | +2.38e-6 | +2.38e-6 | -1.03e-4 |
| 32 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 261 | +1.30e-5 | +1.30e-5 | +1.30e-5 | -9.17e-5 |
| 33 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 251 | +3.32e-5 | +3.32e-5 | +3.32e-5 | -7.93e-5 |
| 34 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 233 | -1.59e-4 | -1.59e-4 | -1.59e-4 | -8.72e-5 |
| 35 | 3.00e-1 | 2 | 2.03e-1 | 2.15e-1 | 2.09e-1 | 2.03e-1 | 203 | -2.73e-4 | +1.06e-4 | -8.35e-5 | -8.84e-5 |
| 36 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 249 | +2.47e-4 | +2.47e-4 | +2.47e-4 | -5.49e-5 |
| 37 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 258 | +5.22e-5 | +5.22e-5 | +5.22e-5 | -4.41e-5 |
| 38 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 252 | -1.16e-5 | -1.16e-5 | -1.16e-5 | -4.09e-5 |
| 39 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 224 | -1.24e-4 | -1.24e-4 | -1.24e-4 | -4.92e-5 |
| 40 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 240 | +1.44e-4 | +1.44e-4 | +1.44e-4 | -2.98e-5 |
| 41 | 3.00e-1 | 2 | 2.02e-1 | 2.16e-1 | 2.09e-1 | 2.02e-1 | 197 | -3.27e-4 | -7.57e-5 | -2.02e-4 | -6.37e-5 |
| 42 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 229 | +2.55e-4 | +2.55e-4 | +2.55e-4 | -3.19e-5 |
| 43 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 224 | -6.23e-5 | -6.23e-5 | -6.23e-5 | -3.49e-5 |
| 44 | 3.00e-1 | 2 | 2.00e-1 | 2.16e-1 | 2.08e-1 | 2.00e-1 | 200 | -3.83e-4 | +9.45e-5 | -1.44e-4 | -5.81e-5 |
| 45 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 232 | +3.04e-4 | +3.04e-4 | +3.04e-4 | -2.19e-5 |
| 46 | 3.00e-1 | 1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 245 | +1.22e-4 | +1.22e-4 | +1.22e-4 | -7.43e-6 |
| 47 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 210 | -2.27e-4 | -2.27e-4 | -2.27e-4 | -2.94e-5 |
| 48 | 3.00e-1 | 2 | 2.00e-1 | 2.12e-1 | 2.06e-1 | 2.00e-1 | 184 | -3.23e-4 | +1.10e-5 | -1.56e-4 | -5.51e-5 |
| 49 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 213 | +2.41e-4 | +2.41e-4 | +2.41e-4 | -2.55e-5 |
| 50 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 221 | +6.53e-5 | +6.53e-5 | +6.53e-5 | -1.64e-5 |
| 51 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 227 | +1.29e-4 | +1.29e-4 | +1.29e-4 | -1.92e-6 |
| 52 | 3.00e-1 | 2 | 2.01e-1 | 2.12e-1 | 2.06e-1 | 2.01e-1 | 180 | -3.00e-4 | -1.62e-4 | -2.31e-4 | -4.61e-5 |
| 53 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 220 | +1.67e-4 | +1.67e-4 | +1.67e-4 | -2.49e-5 |
| 54 | 3.00e-1 | 2 | 2.05e-1 | 2.15e-1 | 2.10e-1 | 2.05e-1 | 189 | -2.39e-4 | +1.45e-4 | -4.71e-5 | -3.10e-5 |
| 55 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 211 | +2.06e-4 | +2.06e-4 | +2.06e-4 | -7.33e-6 |
| 56 | 3.00e-1 | 1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 236 | +1.57e-4 | +1.57e-4 | +1.57e-4 | +9.10e-6 |
| 57 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 212 | -2.38e-4 | -2.38e-4 | -2.38e-4 | -1.56e-5 |
| 58 | 3.00e-1 | 2 | 2.01e-1 | 2.26e-1 | 2.14e-1 | 2.01e-1 | 181 | -6.32e-4 | +2.80e-4 | -1.76e-4 | -5.07e-5 |
| 59 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 215 | +2.19e-4 | +2.19e-4 | +2.19e-4 | -2.37e-5 |
| 60 | 3.00e-1 | 1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 222 | +2.19e-4 | +2.19e-4 | +2.19e-4 | +5.69e-7 |
| 61 | 3.00e-1 | 2 | 1.95e-1 | 2.15e-1 | 2.05e-1 | 1.95e-1 | 162 | -6.11e-4 | -1.43e-4 | -3.77e-4 | -7.35e-5 |
| 62 | 3.00e-1 | 2 | 1.95e-1 | 2.08e-1 | 2.01e-1 | 1.95e-1 | 159 | -3.89e-4 | +3.29e-4 | -3.03e-5 | -6.89e-5 |
| 63 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 200 | +3.94e-4 | +3.94e-4 | +3.94e-4 | -2.26e-5 |
| 64 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 210 | +6.31e-5 | +6.31e-5 | +6.31e-5 | -1.40e-5 |
| 65 | 3.00e-1 | 2 | 1.98e-1 | 2.11e-1 | 2.05e-1 | 1.98e-1 | 161 | -3.93e-4 | -5.97e-5 | -2.26e-4 | -5.60e-5 |
| 66 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 188 | +2.89e-4 | +2.89e-4 | +2.89e-4 | -2.15e-5 |
| 67 | 3.00e-1 | 2 | 1.98e-1 | 2.14e-1 | 2.06e-1 | 1.98e-1 | 159 | -4.67e-4 | +1.03e-4 | -1.82e-4 | -5.49e-5 |
| 68 | 3.00e-1 | 1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 213 | +4.94e-4 | +4.94e-4 | +4.94e-4 | -1.22e-8 |
| 69 | 3.00e-1 | 2 | 1.97e-1 | 2.16e-1 | 2.06e-1 | 1.97e-1 | 159 | -5.76e-4 | -1.07e-4 | -3.42e-4 | -6.73e-5 |
| 70 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 202 | +4.91e-4 | +4.91e-4 | +4.91e-4 | -1.14e-5 |
| 71 | 3.00e-1 | 2 | 1.95e-1 | 2.15e-1 | 2.05e-1 | 1.95e-1 | 155 | -6.27e-4 | -4.79e-5 | -3.38e-4 | -7.63e-5 |
| 72 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 181 | +3.16e-4 | +3.16e-4 | +3.16e-4 | -3.70e-5 |
| 73 | 3.00e-1 | 2 | 1.93e-1 | 2.07e-1 | 2.00e-1 | 1.93e-1 | 150 | -4.62e-4 | -5.81e-6 | -2.34e-4 | -7.68e-5 |
| 74 | 3.00e-1 | 2 | 1.96e-1 | 2.07e-1 | 2.01e-1 | 1.96e-1 | 150 | -3.67e-4 | +3.69e-4 | +1.16e-6 | -6.56e-5 |
| 75 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 181 | +3.46e-4 | +3.46e-4 | +3.46e-4 | -2.44e-5 |
| 76 | 3.00e-1 | 2 | 2.00e-1 | 2.11e-1 | 2.05e-1 | 2.00e-1 | 157 | -3.50e-4 | +5.96e-5 | -1.45e-4 | -4.94e-5 |
| 77 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 169 | +2.38e-5 | +2.38e-5 | +2.38e-5 | -4.21e-5 |
| 78 | 3.00e-1 | 2 | 1.94e-1 | 2.24e-1 | 2.09e-1 | 1.94e-1 | 140 | -1.02e-3 | +4.73e-4 | -2.73e-4 | -9.33e-5 |
| 79 | 3.00e-1 | 2 | 1.87e-1 | 1.98e-1 | 1.93e-1 | 1.87e-1 | 131 | -4.34e-4 | +1.39e-4 | -1.48e-4 | -1.07e-4 |
| 80 | 3.00e-1 | 2 | 1.88e-1 | 2.09e-1 | 1.98e-1 | 1.88e-1 | 132 | -8.22e-4 | +6.24e-4 | -9.88e-5 | -1.12e-4 |
| 81 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 197 | +7.60e-4 | +7.60e-4 | +7.60e-4 | -2.51e-5 |
| 82 | 3.00e-1 | 2 | 1.89e-1 | 2.07e-1 | 1.98e-1 | 1.89e-1 | 134 | -6.78e-4 | -3.06e-4 | -4.92e-4 | -1.16e-4 |
| 83 | 3.00e-1 | 2 | 1.90e-1 | 2.04e-1 | 1.97e-1 | 1.90e-1 | 131 | -5.48e-4 | +4.44e-4 | -5.18e-5 | -1.08e-4 |
| 84 | 3.00e-1 | 2 | 1.96e-1 | 2.04e-1 | 2.00e-1 | 1.96e-1 | 134 | -3.07e-4 | +4.22e-4 | +5.79e-5 | -8.05e-5 |
| 85 | 3.00e-1 | 2 | 1.82e-1 | 1.99e-1 | 1.91e-1 | 1.82e-1 | 125 | -7.36e-4 | +1.31e-4 | -3.02e-4 | -1.27e-4 |
| 86 | 3.00e-1 | 2 | 1.91e-1 | 2.02e-1 | 1.97e-1 | 1.91e-1 | 139 | -4.06e-4 | +6.85e-4 | +1.40e-4 | -8.17e-5 |
| 87 | 3.00e-1 | 1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 203 | +7.31e-4 | +7.31e-4 | +7.31e-4 | -4.09e-7 |
| 88 | 3.00e-1 | 3 | 1.77e-1 | 2.13e-1 | 1.93e-1 | 1.77e-1 | 117 | -9.82e-4 | -2.29e-4 | -5.93e-4 | -1.64e-4 |
| 89 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 160 | +8.43e-4 | +8.43e-4 | +8.43e-4 | -6.34e-5 |
| 90 | 3.00e-1 | 3 | 1.81e-1 | 2.01e-1 | 1.88e-1 | 1.83e-1 | 123 | -9.52e-4 | +8.63e-5 | -2.97e-4 | -1.25e-4 |
| 91 | 3.00e-1 | 2 | 1.86e-1 | 1.99e-1 | 1.93e-1 | 1.86e-1 | 121 | -5.51e-4 | +5.63e-4 | +5.68e-6 | -1.06e-4 |
| 92 | 3.00e-1 | 2 | 1.85e-1 | 1.99e-1 | 1.92e-1 | 1.85e-1 | 114 | -6.63e-4 | +4.26e-4 | -1.19e-4 | -1.14e-4 |
| 93 | 3.00e-1 | 2 | 1.79e-1 | 1.95e-1 | 1.87e-1 | 1.79e-1 | 115 | -7.38e-4 | +3.87e-4 | -1.75e-4 | -1.31e-4 |
| 94 | 3.00e-1 | 2 | 1.84e-1 | 1.92e-1 | 1.88e-1 | 1.84e-1 | 122 | -3.26e-4 | +4.99e-4 | +8.67e-5 | -9.39e-5 |
| 95 | 3.00e-1 | 3 | 1.79e-1 | 2.01e-1 | 1.89e-1 | 1.79e-1 | 107 | -5.87e-4 | +5.82e-4 | -1.41e-4 | -1.16e-4 |
| 96 | 3.00e-1 | 1 | 1.91e-1 | 1.91e-1 | 1.91e-1 | 1.91e-1 | 134 | +4.97e-4 | +4.97e-4 | +4.97e-4 | -5.46e-5 |
| 97 | 3.00e-1 | 3 | 1.67e-1 | 1.98e-1 | 1.80e-1 | 1.67e-1 | 96 | -1.12e-3 | +2.58e-4 | -4.82e-4 | -1.78e-4 |
| 98 | 3.00e-1 | 2 | 1.71e-1 | 1.94e-1 | 1.83e-1 | 1.71e-1 | 96 | -1.32e-3 | +1.05e-3 | -1.37e-4 | -1.82e-4 |
| 99 | 3.00e-1 | 3 | 1.66e-1 | 1.92e-1 | 1.75e-1 | 1.66e-1 | 92 | -1.56e-3 | +8.26e-4 | -2.60e-4 | -2.11e-4 |
| 100 | 3.00e-2 | 3 | 4.78e-2 | 1.93e-1 | 1.12e-1 | 4.78e-2 | 91 | -7.78e-3 | +1.14e-3 | -4.70e-3 | -1.51e-3 |
| 101 | 3.00e-2 | 2 | 1.92e-2 | 2.86e-2 | 2.39e-2 | 1.92e-2 | 89 | -4.46e-3 | -3.92e-3 | -4.19e-3 | -2.02e-3 |
| 102 | 3.00e-2 | 3 | 1.64e-2 | 2.02e-2 | 1.79e-2 | 1.64e-2 | 84 | -1.67e-3 | +3.34e-4 | -6.48e-4 | -1.66e-3 |
| 103 | 3.00e-2 | 3 | 1.69e-2 | 2.05e-2 | 1.83e-2 | 1.69e-2 | 80 | -2.03e-3 | +1.66e-3 | -2.45e-4 | -1.29e-3 |
| 104 | 3.00e-2 | 3 | 1.78e-2 | 2.01e-2 | 1.87e-2 | 1.83e-2 | 86 | -1.60e-3 | +1.45e-3 | +6.45e-5 | -9.34e-4 |
| 105 | 3.00e-2 | 3 | 1.75e-2 | 2.10e-2 | 1.88e-2 | 1.79e-2 | 78 | -2.38e-3 | +1.05e-3 | -3.64e-4 | -7.86e-4 |
| 106 | 3.00e-2 | 4 | 1.72e-2 | 2.05e-2 | 1.86e-2 | 1.86e-2 | 75 | -1.71e-3 | +1.72e-3 | -5.55e-5 | -5.39e-4 |
| 107 | 3.00e-2 | 3 | 1.82e-2 | 2.20e-2 | 1.95e-2 | 1.82e-2 | 67 | -2.51e-3 | +1.40e-3 | -4.32e-4 | -5.24e-4 |
| 108 | 3.00e-2 | 4 | 1.77e-2 | 2.21e-2 | 1.90e-2 | 1.77e-2 | 66 | -3.21e-3 | +1.87e-3 | -4.02e-4 | -4.97e-4 |
| 109 | 3.00e-2 | 4 | 1.83e-2 | 2.19e-2 | 1.95e-2 | 1.83e-2 | 59 | -2.37e-3 | +2.03e-3 | -2.15e-4 | -4.21e-4 |
| 110 | 3.00e-2 | 4 | 1.78e-2 | 2.33e-2 | 1.96e-2 | 1.78e-2 | 54 | -3.82e-3 | +2.12e-3 | -6.88e-4 | -5.31e-4 |
| 111 | 3.00e-2 | 4 | 1.85e-2 | 2.25e-2 | 1.97e-2 | 1.91e-2 | 59 | -3.65e-3 | +2.48e-3 | -1.62e-4 | -4.15e-4 |
| 112 | 3.00e-2 | 5 | 1.68e-2 | 2.30e-2 | 1.87e-2 | 1.74e-2 | 45 | -3.68e-3 | +1.93e-3 | -7.35e-4 | -5.34e-4 |
| 113 | 3.00e-2 | 5 | 1.75e-2 | 2.42e-2 | 1.92e-2 | 1.77e-2 | 46 | -6.00e-3 | +3.56e-3 | -6.90e-4 | -6.02e-4 |
| 114 | 3.00e-2 | 7 | 1.70e-2 | 2.48e-2 | 1.91e-2 | 1.70e-2 | 43 | -4.59e-3 | +3.30e-3 | -6.53e-4 | -6.42e-4 |
| 115 | 3.00e-2 | 4 | 1.73e-2 | 2.36e-2 | 1.93e-2 | 1.78e-2 | 39 | -6.17e-3 | +3.50e-3 | -8.94e-4 | -7.39e-4 |
| 116 | 3.00e-2 | 6 | 1.78e-2 | 2.34e-2 | 1.93e-2 | 1.85e-2 | 40 | -5.24e-3 | +3.76e-3 | -3.68e-4 | -5.65e-4 |
| 117 | 3.00e-2 | 7 | 1.75e-2 | 2.62e-2 | 1.98e-2 | 1.92e-2 | 42 | -1.07e-2 | +3.87e-3 | -7.18e-4 | -5.66e-4 |
| 118 | 3.00e-2 | 5 | 1.87e-2 | 2.60e-2 | 2.09e-2 | 1.99e-2 | 46 | -6.17e-3 | +3.28e-3 | -7.09e-4 | -6.03e-4 |
| 119 | 3.00e-2 | 6 | 1.81e-2 | 2.70e-2 | 2.02e-2 | 1.84e-2 | 36 | -6.77e-3 | +3.26e-3 | -1.17e-3 | -8.26e-4 |
| 120 | 3.00e-2 | 7 | 1.65e-2 | 2.67e-2 | 1.89e-2 | 1.78e-2 | 33 | -8.05e-3 | +4.40e-3 | -9.90e-4 | -8.13e-4 |
| 121 | 3.00e-2 | 7 | 1.82e-2 | 2.73e-2 | 2.06e-2 | 1.96e-2 | 34 | -7.06e-3 | +5.03e-3 | -5.88e-4 | -6.40e-4 |
| 122 | 3.00e-2 | 6 | 1.90e-2 | 2.85e-2 | 2.15e-2 | 2.32e-2 | 55 | -1.13e-2 | +4.73e-3 | -7.50e-4 | -5.55e-4 |
| 123 | 3.00e-2 | 5 | 2.24e-2 | 3.25e-2 | 2.55e-2 | 2.24e-2 | 48 | -4.41e-3 | +3.24e-3 | -8.48e-4 | -7.02e-4 |
| 124 | 3.00e-2 | 6 | 2.25e-2 | 2.85e-2 | 2.41e-2 | 2.28e-2 | 41 | -4.42e-3 | +2.98e-3 | -3.51e-4 | -5.54e-4 |
| 125 | 3.00e-2 | 6 | 2.13e-2 | 2.97e-2 | 2.38e-2 | 2.28e-2 | 40 | -4.69e-3 | +3.92e-3 | -5.76e-4 | -5.57e-4 |
| 126 | 3.00e-2 | 6 | 2.14e-2 | 3.03e-2 | 2.41e-2 | 2.28e-2 | 36 | -4.50e-3 | +3.33e-3 | -5.96e-4 | -5.48e-4 |
| 127 | 3.00e-2 | 6 | 2.07e-2 | 3.00e-2 | 2.31e-2 | 2.23e-2 | 37 | -7.65e-3 | +3.22e-3 | -8.83e-4 | -6.30e-4 |
| 128 | 3.00e-2 | 7 | 2.07e-2 | 3.26e-2 | 2.37e-2 | 2.16e-2 | 35 | -7.93e-3 | +4.55e-3 | -9.93e-4 | -7.65e-4 |
| 129 | 3.00e-2 | 7 | 2.14e-2 | 3.27e-2 | 2.38e-2 | 2.19e-2 | 35 | -7.98e-3 | +5.00e-3 | -9.25e-4 | -8.01e-4 |
| 130 | 3.00e-2 | 7 | 2.15e-2 | 3.25e-2 | 2.43e-2 | 2.56e-2 | 44 | -9.24e-3 | +4.50e-3 | -6.49e-4 | -5.63e-4 |
| 131 | 3.00e-2 | 6 | 2.48e-2 | 3.61e-2 | 2.84e-2 | 2.64e-2 | 40 | -5.23e-3 | +3.60e-3 | -6.06e-4 | -5.84e-4 |
| 132 | 3.00e-2 | 4 | 2.63e-2 | 3.57e-2 | 2.94e-2 | 2.68e-2 | 51 | -4.70e-3 | +3.48e-3 | -8.18e-4 | -6.91e-4 |
| 133 | 3.00e-2 | 5 | 2.93e-2 | 3.68e-2 | 3.16e-2 | 3.06e-2 | 53 | -2.64e-3 | +3.44e-3 | +5.34e-5 | -4.06e-4 |
| 134 | 3.00e-2 | 5 | 2.82e-2 | 3.94e-2 | 3.12e-2 | 2.94e-2 | 44 | -5.47e-3 | +2.68e-3 | -6.87e-4 | -5.00e-4 |
| 135 | 3.00e-2 | 7 | 2.63e-2 | 3.88e-2 | 2.92e-2 | 2.63e-2 | 41 | -5.40e-3 | +3.09e-3 | -7.77e-4 | -6.28e-4 |
| 136 | 3.00e-2 | 4 | 2.66e-2 | 3.97e-2 | 3.10e-2 | 2.66e-2 | 35 | -7.52e-3 | +3.99e-3 | -1.54e-3 | -9.88e-4 |
| 137 | 3.00e-2 | 7 | 2.39e-2 | 3.73e-2 | 2.78e-2 | 2.64e-2 | 38 | -7.63e-3 | +3.91e-3 | -8.65e-4 | -8.47e-4 |
| 138 | 3.00e-2 | 8 | 2.50e-2 | 3.74e-2 | 2.78e-2 | 2.56e-2 | 33 | -7.87e-3 | +4.25e-3 | -8.00e-4 | -7.76e-4 |
| 139 | 3.00e-2 | 3 | 2.94e-2 | 3.85e-2 | 3.53e-2 | 3.80e-2 | 82 | -7.75e-3 | +4.64e-3 | +1.72e-5 | -5.71e-4 |
| 140 | 3.00e-2 | 4 | 3.90e-2 | 4.68e-2 | 4.14e-2 | 3.92e-2 | 76 | -1.88e-3 | +1.87e-3 | -1.28e-4 | -4.35e-4 |
| 141 | 3.00e-2 | 2 | 3.99e-2 | 4.77e-2 | 4.38e-2 | 3.99e-2 | 77 | -2.32e-3 | +1.59e-3 | -3.63e-4 | -4.41e-4 |
| 142 | 3.00e-2 | 4 | 3.90e-2 | 4.95e-2 | 4.23e-2 | 3.90e-2 | 72 | -2.64e-3 | +1.79e-3 | -4.18e-4 | -4.47e-4 |
| 143 | 3.00e-2 | 4 | 3.46e-2 | 4.77e-2 | 3.92e-2 | 3.70e-2 | 64 | -3.75e-3 | +1.71e-3 | -5.73e-4 | -4.84e-4 |
| 144 | 3.00e-2 | 3 | 3.76e-2 | 4.51e-2 | 4.12e-2 | 3.76e-2 | 64 | -1.47e-3 | +1.91e-3 | -3.17e-4 | -4.70e-4 |
| 145 | 3.00e-2 | 5 | 3.33e-2 | 4.42e-2 | 3.69e-2 | 3.52e-2 | 55 | -3.15e-3 | +1.64e-3 | -4.78e-4 | -4.60e-4 |
| 146 | 3.00e-2 | 4 | 3.52e-2 | 4.48e-2 | 3.81e-2 | 3.52e-2 | 57 | -3.47e-3 | +2.57e-3 | -4.67e-4 | -4.83e-4 |
| 147 | 3.00e-2 | 6 | 3.25e-2 | 4.30e-2 | 3.51e-2 | 3.25e-2 | 48 | -4.06e-3 | +2.29e-3 | -5.20e-4 | -5.01e-4 |
| 148 | 3.00e-2 | 5 | 3.24e-2 | 4.40e-2 | 3.58e-2 | 3.37e-2 | 45 | -3.66e-3 | +3.33e-3 | -4.04e-4 | -4.66e-4 |
| 149 | 3.00e-2 | 5 | 2.99e-2 | 4.62e-2 | 3.49e-2 | 3.26e-2 | 42 | -6.49e-3 | +3.44e-3 | -9.71e-4 | -6.36e-4 |
| 150 | 3.00e-3 | 6 | 2.82e-3 | 4.95e-2 | 1.62e-2 | 2.82e-3 | 38 | -1.89e-2 | +4.69e-3 | -1.08e-2 | -5.52e-3 |
| 151 | 3.00e-3 | 7 | 2.05e-3 | 3.37e-3 | 2.42e-3 | 2.21e-3 | 42 | -8.17e-3 | +1.91e-3 | -1.30e-3 | -3.22e-3 |
| 152 | 3.00e-3 | 5 | 2.17e-3 | 2.88e-3 | 2.37e-3 | 2.21e-3 | 39 | -4.82e-3 | +3.55e-3 | -6.39e-4 | -2.17e-3 |
| 153 | 3.00e-3 | 2 | 2.13e-3 | 4.70e-3 | 3.41e-3 | 4.70e-3 | 239 | -9.84e-4 | +3.31e-3 | +1.17e-3 | -1.52e-3 |
| 154 | 3.00e-3 | 1 | 5.02e-3 | 5.02e-3 | 5.02e-3 | 5.02e-3 | 242 | +2.73e-4 | +2.73e-4 | +2.73e-4 | -1.34e-3 |
| 155 | 3.00e-3 | 1 | 5.14e-3 | 5.14e-3 | 5.14e-3 | 5.14e-3 | 246 | +9.48e-5 | +9.48e-5 | +9.48e-5 | -1.19e-3 |
| 156 | 3.00e-3 | 1 | 5.29e-3 | 5.29e-3 | 5.29e-3 | 5.29e-3 | 251 | +1.16e-4 | +1.16e-4 | +1.16e-4 | -1.06e-3 |
| 157 | 3.00e-3 | 1 | 5.48e-3 | 5.48e-3 | 5.48e-3 | 5.48e-3 | 252 | +1.40e-4 | +1.40e-4 | +1.40e-4 | -9.43e-4 |
| 158 | 3.00e-3 | 1 | 5.25e-3 | 5.25e-3 | 5.25e-3 | 5.25e-3 | 251 | -1.67e-4 | -1.67e-4 | -1.67e-4 | -8.66e-4 |
| 159 | 3.00e-3 | 1 | 5.47e-3 | 5.47e-3 | 5.47e-3 | 5.47e-3 | 260 | +1.54e-4 | +1.54e-4 | +1.54e-4 | -7.64e-4 |
| 160 | 3.00e-3 | 2 | 5.08e-3 | 5.77e-3 | 5.42e-3 | 5.08e-3 | 245 | -5.21e-4 | +1.94e-4 | -1.63e-4 | -6.53e-4 |
| 162 | 3.00e-3 | 2 | 5.14e-3 | 5.90e-3 | 5.52e-3 | 5.14e-3 | 219 | -6.30e-4 | +5.30e-4 | -4.97e-5 | -5.44e-4 |
| 163 | 3.00e-3 | 1 | 5.36e-3 | 5.36e-3 | 5.36e-3 | 5.36e-3 | 247 | +1.74e-4 | +1.74e-4 | +1.74e-4 | -4.72e-4 |
| 164 | 3.00e-3 | 1 | 5.28e-3 | 5.28e-3 | 5.28e-3 | 5.28e-3 | 233 | -6.98e-5 | -6.98e-5 | -6.98e-5 | -4.32e-4 |
| 165 | 3.00e-3 | 1 | 5.57e-3 | 5.57e-3 | 5.57e-3 | 5.57e-3 | 264 | +2.01e-4 | +2.01e-4 | +2.01e-4 | -3.69e-4 |
| 166 | 3.00e-3 | 1 | 5.09e-3 | 5.09e-3 | 5.09e-3 | 5.09e-3 | 243 | -3.68e-4 | -3.68e-4 | -3.68e-4 | -3.69e-4 |
| 167 | 3.00e-3 | 1 | 5.70e-3 | 5.70e-3 | 5.70e-3 | 5.70e-3 | 258 | +4.40e-4 | +4.40e-4 | +4.40e-4 | -2.88e-4 |
| 168 | 3.00e-3 | 1 | 5.50e-3 | 5.50e-3 | 5.50e-3 | 5.50e-3 | 251 | -1.41e-4 | -1.41e-4 | -1.41e-4 | -2.73e-4 |
| 169 | 3.00e-3 | 2 | 5.31e-3 | 5.38e-3 | 5.34e-3 | 5.31e-3 | 217 | -9.51e-5 | -6.46e-5 | -7.98e-5 | -2.36e-4 |
| 170 | 3.00e-3 | 1 | 5.33e-3 | 5.33e-3 | 5.33e-3 | 5.33e-3 | 248 | +1.49e-5 | +1.49e-5 | +1.49e-5 | -2.11e-4 |
| 171 | 3.00e-3 | 1 | 5.32e-3 | 5.32e-3 | 5.32e-3 | 5.32e-3 | 235 | -3.86e-6 | -3.86e-6 | -3.86e-6 | -1.90e-4 |
| 172 | 3.00e-3 | 1 | 5.67e-3 | 5.67e-3 | 5.67e-3 | 5.67e-3 | 258 | +2.46e-4 | +2.46e-4 | +2.46e-4 | -1.47e-4 |
| 173 | 3.00e-3 | 1 | 5.53e-3 | 5.53e-3 | 5.53e-3 | 5.53e-3 | 253 | -9.62e-5 | -9.62e-5 | -9.62e-5 | -1.42e-4 |
| 174 | 3.00e-3 | 1 | 5.18e-3 | 5.18e-3 | 5.18e-3 | 5.18e-3 | 237 | -2.76e-4 | -2.76e-4 | -2.76e-4 | -1.55e-4 |
| 175 | 3.00e-3 | 1 | 5.55e-3 | 5.55e-3 | 5.55e-3 | 5.55e-3 | 248 | +2.73e-4 | +2.73e-4 | +2.73e-4 | -1.12e-4 |
| 176 | 3.00e-3 | 2 | 5.29e-3 | 5.31e-3 | 5.30e-3 | 5.31e-3 | 211 | -1.98e-4 | +1.40e-5 | -9.19e-5 | -1.07e-4 |
| 177 | 3.00e-3 | 1 | 5.38e-3 | 5.38e-3 | 5.38e-3 | 5.38e-3 | 244 | +5.61e-5 | +5.61e-5 | +5.61e-5 | -9.10e-5 |
| 178 | 3.00e-3 | 1 | 5.23e-3 | 5.23e-3 | 5.23e-3 | 5.23e-3 | 232 | -1.24e-4 | -1.24e-4 | -1.24e-4 | -9.44e-5 |
| 179 | 3.00e-3 | 1 | 5.84e-3 | 5.84e-3 | 5.84e-3 | 5.84e-3 | 244 | +4.56e-4 | +4.56e-4 | +4.56e-4 | -3.93e-5 |
| 180 | 3.00e-3 | 2 | 4.90e-3 | 5.21e-3 | 5.06e-3 | 4.90e-3 | 187 | -5.44e-4 | -3.38e-4 | -4.41e-4 | -1.15e-4 |
| 181 | 3.00e-3 | 1 | 4.99e-3 | 4.99e-3 | 4.99e-3 | 4.99e-3 | 211 | +9.22e-5 | +9.22e-5 | +9.22e-5 | -9.40e-5 |
| 182 | 3.00e-3 | 1 | 4.98e-3 | 4.98e-3 | 4.98e-3 | 4.98e-3 | 213 | -9.30e-6 | -9.30e-6 | -9.30e-6 | -8.55e-5 |
| 183 | 3.00e-3 | 2 | 5.08e-3 | 5.18e-3 | 5.13e-3 | 5.18e-3 | 196 | +9.55e-5 | +9.63e-5 | +9.59e-5 | -5.10e-5 |
| 184 | 3.00e-3 | 1 | 5.50e-3 | 5.50e-3 | 5.50e-3 | 5.50e-3 | 242 | +2.43e-4 | +2.43e-4 | +2.43e-4 | -2.16e-5 |
| 185 | 3.00e-3 | 1 | 5.57e-3 | 5.57e-3 | 5.57e-3 | 5.57e-3 | 260 | +5.52e-5 | +5.52e-5 | +5.52e-5 | -1.39e-5 |
| 186 | 3.00e-3 | 1 | 5.17e-3 | 5.17e-3 | 5.17e-3 | 5.17e-3 | 224 | -3.38e-4 | -3.38e-4 | -3.38e-4 | -4.63e-5 |
| 187 | 3.00e-3 | 1 | 5.71e-3 | 5.71e-3 | 5.71e-3 | 5.71e-3 | 245 | +4.10e-4 | +4.10e-4 | +4.10e-4 | -7.08e-7 |
| 188 | 3.00e-3 | 2 | 4.65e-3 | 5.55e-3 | 5.10e-3 | 4.65e-3 | 174 | -1.01e-3 | -1.20e-4 | -5.66e-4 | -1.13e-4 |
| 189 | 3.00e-3 | 1 | 5.10e-3 | 5.10e-3 | 5.10e-3 | 5.10e-3 | 244 | +3.77e-4 | +3.77e-4 | +3.77e-4 | -6.36e-5 |
| 190 | 3.00e-3 | 1 | 4.90e-3 | 4.90e-3 | 4.90e-3 | 4.90e-3 | 209 | -1.94e-4 | -1.94e-4 | -1.94e-4 | -7.66e-5 |
| 191 | 3.00e-3 | 2 | 5.00e-3 | 5.36e-3 | 5.18e-3 | 5.00e-3 | 179 | -3.86e-4 | +3.80e-4 | -2.89e-6 | -6.64e-5 |
| 192 | 3.00e-3 | 1 | 5.43e-3 | 5.43e-3 | 5.43e-3 | 5.43e-3 | 212 | +3.88e-4 | +3.88e-4 | +3.88e-4 | -2.10e-5 |
| 193 | 3.00e-3 | 2 | 4.95e-3 | 5.14e-3 | 5.05e-3 | 4.95e-3 | 168 | -2.52e-4 | -2.25e-4 | -2.38e-4 | -6.22e-5 |
| 194 | 3.00e-3 | 1 | 5.04e-3 | 5.04e-3 | 5.04e-3 | 5.04e-3 | 198 | +8.43e-5 | +8.43e-5 | +8.43e-5 | -4.75e-5 |
| 195 | 3.00e-3 | 1 | 4.99e-3 | 4.99e-3 | 4.99e-3 | 4.99e-3 | 208 | -4.38e-5 | -4.38e-5 | -4.38e-5 | -4.71e-5 |
| 196 | 3.00e-3 | 2 | 4.80e-3 | 4.94e-3 | 4.87e-3 | 4.94e-3 | 172 | -1.92e-4 | +1.72e-4 | -1.02e-5 | -3.83e-5 |
| 197 | 3.00e-3 | 1 | 5.12e-3 | 5.12e-3 | 5.12e-3 | 5.12e-3 | 236 | +1.57e-4 | +1.57e-4 | +1.57e-4 | -1.88e-5 |
| 198 | 3.00e-3 | 2 | 4.63e-3 | 5.51e-3 | 5.07e-3 | 4.63e-3 | 161 | -1.09e-3 | +3.60e-4 | -3.63e-4 | -9.14e-5 |
| 199 | 3.00e-3 | 1 | 5.18e-3 | 5.18e-3 | 5.18e-3 | 5.18e-3 | 211 | +5.35e-4 | +5.35e-4 | +5.35e-4 | -2.88e-5 |

