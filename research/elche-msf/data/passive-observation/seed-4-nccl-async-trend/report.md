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
- resnet-graph/cpu-sync
- resnet-graph/cpu-cadence
- resnet-graph/cpu-async

## Per-Model Results

GPU0/GPU1 = compute utilization % (not load). Idle = total time with <5% utilization.

### resnet-graph

> Published: CIFAR-10 91.25% acc, Graph builder ([He et al. 2015](https://arxiv.org/abs/1512.03385), Table 6)

| Mode | Loss | Eval | vs Ref | Total (s) | Syncs | Avg Sync (ms) | GPU0 | GPU1 | Idle (s) |
|------|------|------|--------|-----------|-------|--------------|------|------|----------|
| nccl-async | 0.060114 | 0.9157 | +0.0032 | 2010.9 | 356 | 48.0 | 100% | 100% | 6.9 |

## Best Mode per Model

| Model | Best Eval | Mode | Fastest (within 2% of solo-0) | Mode |
|-------|-----------|------|-------------------------------|------|
| resnet-graph | - | - | - | - |

## Eval Quality (vs solo-0)

| Model | nccl-async |
|-------|------------|
| resnet-graph | - |

## Convergence Quality (loss ratio vs solo-0)

| Model | nccl-async |
|-------|------------|
| resnet-graph | - |

## Per-Epoch Loss Trajectory

### resnet-graph (sampled, 200 epochs)

| Mode | E0 | E10 | E21 | E31 | E42 | E52 | E63 | E73 | E84 | E94 | E105 | E115 | E126 | E136 | E147 | E157 | E168 | E178 | E189 | E199 |
|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| nccl-async | 1.9728 | 0.8298 | 0.6156 | 0.5385 | 0.5121 | 0.4478 | 0.4976 | 0.4696 | 0.4673 | 0.4608 | 0.2072 | 0.1672 | 0.1536 | 0.1439 | 0.1360 | 0.0777 | 0.0732 | 0.0634 | 0.0627 | 0.0601 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | nccl-async | 0 | cuda:0 | 0.4013 | 2.5 |
| resnet-graph | nccl-async | 1 | cuda:1 | 0.3034 | 3.2 |
| resnet-graph | nccl-async | 2 | cuda:2 | 0.2954 | 3.2 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|
| resnet-graph | nccl-async | 356 | 355 | 388 | 384 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| nccl-async | gpu1 | 2009.6 | 1.2 | epoch-boundary(199) |
| nccl-async | gpu2 | 2009.6 | 1.2 | epoch-boundary(199) |
| nccl-async | gpu0 | 2009.6 | 0.6 | epoch-boundary(199) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | nccl-async | gpu0 | 0.6s | 0.0s | 0.0s | 0.0s | 2.7s |
| resnet-graph | nccl-async | gpu1 | 1.2s | 0.0s | 0.0s | 0.0s | 2.3s |
| resnet-graph | nccl-async | gpu2 | 1.2s | 0.0s | 0.0s | 0.0s | 1.8s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | nccl-async | 304 | 0 | 356 | 48.0 | 6252/10757 | 0 | 0.0 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | nccl-async | 195.4 | 9.7% |

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
| resnet-graph | nccl-async | 191 | 356 | 0 | 6.66e-3 | +3.97e-5 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | nccl-async | 0 | 356 | 9.56e-2 | 7.09e-2 | 0.00e0 | 4.33e-1 | 34.0 | -2.07e-4 | 2.47e-3 |
| resnet-graph | nccl-async | 1 | 356 | 9.68e-2 | 7.35e-2 | 0.00e0 | 5.21e-1 | 43.3 | -2.26e-4 | 3.02e-3 |
| resnet-graph | nccl-async | 2 | 356 | 9.60e-2 | 7.26e-2 | 0.00e0 | 4.90e-1 | 22.8 | -2.27e-4 | 3.10e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | nccl-async | rank0 ↔ rank1 | +0.9904 |
| resnet-graph | nccl-async | rank0 ↔ rank2 | +0.9903 |
| resnet-graph | nccl-async | rank1 ↔ rank2 | +0.9990 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | nccl-async | 34 (0,2,3,4,18,22,26,40…149,150) | 0 (—) | — | 0,2,3,4,18,22,26,40…149,150 | — |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | nccl-async | 0e0 | 3 | 27 | 27 |
| resnet-graph | nccl-async | 0e0 | 5 | 13 | 13 |
| resnet-graph | nccl-async | 0e0 | 10 | 5 | 5 |
| resnet-graph | nccl-async | 1e-4 | 3 | 3 | 3 |
| resnet-graph | nccl-async | 1e-4 | 5 | 1 | 1 |
| resnet-graph | nccl-async | 1e-4 | 10 | 0 | 0 |
| resnet-graph | nccl-async | 1e-3 | 3 | 0 | 0 |
| resnet-graph | nccl-async | 1e-3 | 5 | 0 | 0 |
| resnet-graph | nccl-async | 1e-3 | 10 | 0 | 0 |
| resnet-graph | nccl-async | 1e-2 | 3 | 0 | 0 |
| resnet-graph | nccl-async | 1e-2 | 5 | 0 | 0 |
| resnet-graph | nccl-async | 1e-2 | 10 | 0 | 0 |
| resnet-graph | nccl-async | 1e-1 | 3 | 0 | 0 |
| resnet-graph | nccl-async | 1e-1 | 5 | 0 | 0 |
| resnet-graph | nccl-async | 1e-1 | 10 | 0 | 0 |

### Predictive Value by LR Window (top scale)

Per-LR-window Pearson `r(λ_raw_t, ln(D_{t+1}))`. Pairs that straddle a window boundary are excluded so the LR-drop collapse cannot leak in as artefactual signal. Reading: a clean R1 (exponential growth at fixed LR) shows up as *non-zero* `r` within each window, with sign and magnitude that may differ between warmup, post-drop transient, and late-training phases.

| Model | Mode | LR | Epochs | n_pairs | r(λ → ln D_{t+1}) |
|-------|------|---:|--------|--------:|----------------:|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 228 | +0.153 |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | 62 | +0.188 |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | 61 | +0.047 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | nccl-async | 353 | +0.009 | 190 | +0.187 | +0.645 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | nccl-async | 354 | 3.43e1–7.96e1 | 6.84e1 | 3.64e-3 | 8.13e-3 | 6.16e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 230 | 79–77623 | +4.163e-6 | 0.032 | +4.441e-6 | 0.035 | 99 | +3.840e-6 | 0.083 | 30–919 | +1.406e-3 | 0.603 |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 213 | 832–77623 | -1.457e-8 | 0.000 | -8.664e-8 | 0.000 | 98 | +3.240e-6 | 0.063 | 39–919 | +1.323e-3 | 0.672 |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | 63 | 78201–116673 | +2.000e-5 | 0.251 | +2.013e-5 | 0.253 | 48 | +2.284e-5 | 0.384 | 373–883 | +1.514e-3 | 0.198 |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | 62 | 117416–155740 | -6.464e-6 | 0.025 | -6.552e-6 | 0.026 | 44 | -1.172e-5 | 0.061 | 394–1082 | +9.619e-4 | 0.099 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | +1.406e-3 | r0: +1.351e-3, r1: +1.430e-3, r2: +1.444e-3 | r0: 0.605, r1: 0.597, r2: 0.602 | 1.07× | ✓ |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +1.323e-3 | r0: +1.275e-3, r1: +1.351e-3, r2: +1.349e-3 | r0: 0.701, r1: 0.655, r2: 0.655 | 1.06× | ✓ |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | +1.514e-3 | r0: +1.538e-3, r1: +1.498e-3, r2: +1.507e-3 | r0: 0.206, r1: 0.192, r2: 0.194 | 1.03× | ✓ |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | +9.619e-4 | r0: +9.895e-4, r1: +9.291e-4, r2: +9.683e-4 | r0: 0.106, r1: 0.091, r2: 0.099 | 1.07× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | nccl-async | `██████████▇▇▇▇▇██████████▄▄▄▄▄▄▅▅▅▅▅▅▂▁▁▁▁▁▁▁▁▂▂` | `▁▄▆▇▇███████▇▇███████████▆▇▇█████████▇▇▇████████` |

### Per-Epoch Detail: resnet-graph / nccl-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 18 | 0.00e0 | 5.21e-1 | 9.54e-2 | 6.48e-2 | 18 | -8.38e-2 | +1.42e-2 | -1.10e-2 | -4.03e-3 |
| 1 | 3.00e-1 | 2 | 5.61e-2 | 6.59e-2 | 6.10e-2 | 6.59e-2 | 190 | -8.04e-3 | +8.50e-4 | -3.59e-3 | -3.88e-3 |
| 2 | 3.00e-1 | 1 | 1.70e-1 | 1.70e-1 | 1.70e-1 | 1.70e-1 | 228 | +4.15e-3 | +4.15e-3 | +4.15e-3 | -2.95e-3 |
| 3 | 3.00e-1 | 1 | 1.73e-1 | 1.73e-1 | 1.73e-1 | 1.73e-1 | 230 | +9.61e-5 | +9.61e-5 | +9.61e-5 | -2.60e-3 |
| 4 | 3.00e-1 | 2 | 1.80e-1 | 1.90e-1 | 1.85e-1 | 1.90e-1 | 172 | +1.56e-4 | +3.31e-4 | +2.43e-4 | -2.00e-3 |
| 5 | 3.00e-1 | 1 | 1.75e-1 | 1.75e-1 | 1.75e-1 | 1.75e-1 | 262 | -3.19e-4 | -3.19e-4 | -3.19e-4 | -1.82e-3 |
| 6 | 3.00e-1 | 2 | 1.86e-1 | 1.95e-1 | 1.90e-1 | 1.86e-1 | 164 | -2.85e-4 | +4.50e-4 | +8.24e-5 | -1.43e-3 |
| 7 | 3.00e-1 | 1 | 1.71e-1 | 1.71e-1 | 1.71e-1 | 1.71e-1 | 190 | -4.49e-4 | -4.49e-4 | -4.49e-4 | -1.33e-3 |
| 8 | 3.00e-1 | 1 | 1.76e-1 | 1.76e-1 | 1.76e-1 | 1.76e-1 | 180 | +1.84e-4 | +1.84e-4 | +1.84e-4 | -1.17e-3 |
| 9 | 3.00e-1 | 2 | 1.75e-1 | 1.76e-1 | 1.76e-1 | 1.76e-1 | 164 | -2.80e-5 | +2.50e-5 | -1.49e-6 | -9.34e-4 |
| 10 | 3.00e-1 | 2 | 1.67e-1 | 1.80e-1 | 1.74e-1 | 1.80e-1 | 176 | -2.40e-4 | +4.07e-4 | +8.33e-5 | -7.30e-4 |
| 11 | 3.00e-1 | 1 | 1.69e-1 | 1.69e-1 | 1.69e-1 | 1.69e-1 | 218 | -2.81e-4 | -2.81e-4 | -2.81e-4 | -6.84e-4 |
| 12 | 3.00e-1 | 1 | 1.81e-1 | 1.81e-1 | 1.81e-1 | 1.81e-1 | 194 | +3.51e-4 | +3.51e-4 | +3.51e-4 | -5.77e-4 |
| 13 | 3.00e-1 | 2 | 1.76e-1 | 1.79e-1 | 1.77e-1 | 1.79e-1 | 169 | -1.32e-4 | +9.35e-5 | -1.93e-5 | -4.67e-4 |
| 14 | 3.00e-1 | 1 | 1.66e-1 | 1.66e-1 | 1.66e-1 | 1.66e-1 | 210 | -3.58e-4 | -3.58e-4 | -3.58e-4 | -4.56e-4 |
| 15 | 3.00e-1 | 2 | 1.73e-1 | 1.78e-1 | 1.76e-1 | 1.73e-1 | 159 | -1.64e-4 | +3.47e-4 | +9.14e-5 | -3.53e-4 |
| 16 | 3.00e-1 | 1 | 1.64e-1 | 1.64e-1 | 1.64e-1 | 1.64e-1 | 181 | -2.96e-4 | -2.96e-4 | -2.96e-4 | -3.47e-4 |
| 17 | 3.00e-1 | 1 | 1.72e-1 | 1.72e-1 | 1.72e-1 | 1.72e-1 | 231 | +2.05e-4 | +2.05e-4 | +2.05e-4 | -2.91e-4 |
| 18 | 3.00e-1 | 2 | 1.82e-1 | 1.83e-1 | 1.82e-1 | 1.82e-1 | 150 | -4.01e-5 | +2.78e-4 | +1.19e-4 | -2.14e-4 |
| 19 | 3.00e-1 | 2 | 1.62e-1 | 1.72e-1 | 1.67e-1 | 1.72e-1 | 142 | -5.92e-4 | +4.26e-4 | -8.29e-5 | -1.83e-4 |
| 20 | 3.00e-1 | 2 | 1.60e-1 | 1.67e-1 | 1.63e-1 | 1.67e-1 | 137 | -4.57e-4 | +3.43e-4 | -5.72e-5 | -1.55e-4 |
| 21 | 3.00e-1 | 1 | 1.58e-1 | 1.58e-1 | 1.58e-1 | 1.58e-1 | 174 | -3.41e-4 | -3.41e-4 | -3.41e-4 | -1.74e-4 |
| 22 | 3.00e-1 | 2 | 1.70e-1 | 1.70e-1 | 1.70e-1 | 1.70e-1 | 144 | +3.18e-6 | +4.22e-4 | +2.13e-4 | -1.02e-4 |
| 23 | 3.00e-1 | 2 | 1.61e-1 | 1.76e-1 | 1.69e-1 | 1.76e-1 | 157 | -2.73e-4 | +5.60e-4 | +1.44e-4 | -5.10e-5 |
| 24 | 3.00e-1 | 2 | 1.65e-1 | 1.75e-1 | 1.70e-1 | 1.75e-1 | 157 | -3.53e-4 | +3.63e-4 | +4.96e-6 | -3.68e-5 |
| 25 | 3.00e-1 | 1 | 1.62e-1 | 1.62e-1 | 1.62e-1 | 1.62e-1 | 172 | -4.52e-4 | -4.52e-4 | -4.52e-4 | -7.84e-5 |
| 26 | 3.00e-1 | 2 | 1.75e-1 | 1.76e-1 | 1.76e-1 | 1.76e-1 | 149 | +2.37e-5 | +4.34e-4 | +2.29e-4 | -2.19e-5 |
| 27 | 3.00e-1 | 1 | 1.64e-1 | 1.64e-1 | 1.64e-1 | 1.64e-1 | 190 | -3.68e-4 | -3.68e-4 | -3.68e-4 | -5.66e-5 |
| 28 | 3.00e-1 | 2 | 1.77e-1 | 1.78e-1 | 1.77e-1 | 1.77e-1 | 141 | -7.70e-5 | +4.47e-4 | +1.85e-4 | -1.32e-5 |
| 29 | 3.00e-1 | 2 | 1.62e-1 | 1.70e-1 | 1.66e-1 | 1.70e-1 | 135 | -5.24e-4 | +3.61e-4 | -8.18e-5 | -2.18e-5 |
| 30 | 3.00e-1 | 2 | 1.61e-1 | 1.77e-1 | 1.69e-1 | 1.77e-1 | 115 | -2.60e-4 | +8.25e-4 | +2.82e-4 | +4.15e-5 |
| 31 | 3.00e-1 | 2 | 1.51e-1 | 1.66e-1 | 1.59e-1 | 1.66e-1 | 115 | -1.12e-3 | +8.47e-4 | -1.35e-4 | +1.77e-5 |
| 32 | 3.00e-1 | 2 | 1.49e-1 | 1.66e-1 | 1.58e-1 | 1.66e-1 | 115 | -7.22e-4 | +9.37e-4 | +1.07e-4 | +4.30e-5 |
| 33 | 3.00e-1 | 2 | 1.53e-1 | 1.69e-1 | 1.61e-1 | 1.69e-1 | 122 | -5.45e-4 | +8.59e-4 | +1.57e-4 | +7.17e-5 |
| 34 | 3.00e-1 | 2 | 1.53e-1 | 1.73e-1 | 1.63e-1 | 1.73e-1 | 110 | -5.67e-4 | +1.10e-3 | +2.66e-4 | +1.17e-4 |
| 35 | 3.00e-1 | 2 | 1.48e-1 | 1.64e-1 | 1.56e-1 | 1.64e-1 | 110 | -1.08e-3 | +9.26e-4 | -7.54e-5 | +9.04e-5 |
| 36 | 3.00e-1 | 3 | 1.46e-1 | 1.68e-1 | 1.54e-1 | 1.46e-1 | 100 | -1.42e-3 | +1.30e-3 | -2.78e-4 | -1.69e-5 |
| 37 | 3.00e-1 | 3 | 1.43e-1 | 1.69e-1 | 1.53e-1 | 1.43e-1 | 100 | -1.68e-3 | +1.54e-3 | -5.09e-5 | -4.27e-5 |
| 38 | 3.00e-1 | 2 | 1.42e-1 | 1.61e-1 | 1.51e-1 | 1.61e-1 | 100 | -9.02e-5 | +1.26e-3 | +5.85e-4 | +8.33e-5 |
| 39 | 3.00e-1 | 3 | 1.43e-1 | 1.59e-1 | 1.49e-1 | 1.47e-1 | 100 | -9.23e-4 | +1.09e-3 | -2.07e-4 | +5.20e-6 |
| 40 | 3.00e-1 | 2 | 1.48e-1 | 1.64e-1 | 1.56e-1 | 1.64e-1 | 100 | +6.21e-5 | +9.97e-4 | +5.30e-4 | +1.10e-4 |
| 41 | 3.00e-1 | 3 | 1.44e-1 | 1.60e-1 | 1.49e-1 | 1.45e-1 | 96 | -1.01e-3 | +1.13e-3 | -2.76e-4 | +3.66e-6 |
| 42 | 3.00e-1 | 2 | 1.41e-1 | 1.65e-1 | 1.53e-1 | 1.65e-1 | 101 | -1.78e-4 | +1.56e-3 | +6.91e-4 | +1.43e-4 |
| 43 | 3.00e-1 | 3 | 1.38e-1 | 1.59e-1 | 1.48e-1 | 1.38e-1 | 90 | -1.58e-3 | +7.80e-4 | -5.37e-4 | -4.94e-5 |
| 44 | 3.00e-1 | 3 | 1.41e-1 | 1.60e-1 | 1.48e-1 | 1.42e-1 | 89 | -1.33e-3 | +1.40e-3 | +7.85e-5 | -2.97e-5 |
| 45 | 3.00e-1 | 3 | 1.34e-1 | 1.61e-1 | 1.45e-1 | 1.34e-1 | 79 | -2.28e-3 | +1.78e-3 | -2.07e-4 | -9.93e-5 |
| 46 | 3.00e-1 | 4 | 1.32e-1 | 1.55e-1 | 1.38e-1 | 1.34e-1 | 79 | -2.08e-3 | +1.96e-3 | -3.70e-6 | -7.99e-5 |
| 47 | 3.00e-1 | 2 | 1.33e-1 | 1.59e-1 | 1.46e-1 | 1.59e-1 | 68 | -3.86e-5 | +2.64e-3 | +1.30e-3 | +1.96e-4 |
| 48 | 3.00e-1 | 4 | 1.19e-1 | 1.50e-1 | 1.29e-1 | 1.22e-1 | 58 | -3.58e-3 | +3.05e-3 | -5.92e-4 | -6.79e-5 |
| 49 | 3.00e-1 | 4 | 1.15e-1 | 1.46e-1 | 1.25e-1 | 1.16e-1 | 58 | -4.05e-3 | +2.92e-3 | -2.58e-4 | -1.64e-4 |
| 50 | 3.00e-1 | 7 | 1.04e-1 | 1.48e-1 | 1.19e-1 | 1.06e-1 | 50 | -3.60e-3 | +3.92e-3 | -3.37e-4 | -3.13e-4 |
| 51 | 3.00e-1 | 3 | 1.11e-1 | 1.47e-1 | 1.25e-1 | 1.11e-1 | 43 | -6.53e-3 | +5.16e-3 | -1.64e-4 | -3.46e-4 |
| 52 | 3.00e-1 | 8 | 8.78e-2 | 1.42e-1 | 1.01e-1 | 8.87e-2 | 30 | -1.01e-2 | +8.54e-3 | -5.72e-4 | -5.44e-4 |
| 53 | 3.00e-1 | 9 | 7.58e-2 | 1.33e-1 | 8.99e-2 | 7.58e-2 | 25 | -1.35e-2 | +1.50e-2 | -5.71e-4 | -7.87e-4 |
| 54 | 3.00e-1 | 12 | 6.56e-2 | 1.32e-1 | 7.71e-2 | 7.25e-2 | 20 | -3.47e-2 | +2.46e-2 | -3.55e-4 | -4.63e-4 |
| 55 | 3.00e-1 | 18 | 5.59e-2 | 1.38e-1 | 7.22e-2 | 7.71e-2 | 20 | -4.22e-2 | +3.50e-2 | -4.21e-5 | +1.93e-4 |
| 56 | 3.00e-1 | 11 | 6.01e-2 | 1.43e-1 | 7.32e-2 | 7.61e-2 | 17 | -4.80e-2 | +3.79e-2 | +9.02e-5 | +5.40e-4 |
| 57 | 3.00e-1 | 1 | 6.56e-2 | 6.56e-2 | 6.56e-2 | 6.56e-2 | 20 | -7.41e-3 | -7.41e-3 | -7.41e-3 | -2.55e-4 |
| 58 | 3.00e-1 | 1 | 6.96e-2 | 6.96e-2 | 6.96e-2 | 6.96e-2 | 330 | +1.78e-4 | +1.78e-4 | +1.78e-4 | -2.12e-4 |
| 59 | 3.00e-1 | 1 | 2.45e-1 | 2.45e-1 | 2.45e-1 | 2.45e-1 | 322 | +3.91e-3 | +3.91e-3 | +3.91e-3 | +2.00e-4 |
| 60 | 3.00e-1 | 1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 292 | -3.25e-4 | -3.25e-4 | -3.25e-4 | +1.48e-4 |
| 61 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 323 | -1.32e-4 | -1.32e-4 | -1.32e-4 | +1.20e-4 |
| 62 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 282 | +3.64e-5 | +3.64e-5 | +3.64e-5 | +1.11e-4 |
| 63 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 294 | -1.78e-4 | -1.78e-4 | -1.78e-4 | +8.24e-5 |
| 64 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 294 | +6.89e-5 | +6.89e-5 | +6.89e-5 | +8.10e-5 |
| 65 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 281 | -4.78e-5 | -4.78e-5 | -4.78e-5 | +6.81e-5 |
| 66 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 306 | -2.08e-6 | -2.08e-6 | -2.08e-6 | +6.11e-5 |
| 67 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 257 | +1.27e-4 | +1.27e-4 | +1.27e-4 | +6.77e-5 |
| 68 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 300 | -2.33e-4 | -2.33e-4 | -2.33e-4 | +3.76e-5 |
| 69 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 280 | +1.71e-4 | +1.71e-4 | +1.71e-4 | +5.09e-5 |
| 70 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 301 | -5.65e-5 | -5.65e-5 | -5.65e-5 | +4.02e-5 |
| 71 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 266 | +5.26e-5 | +5.26e-5 | +5.26e-5 | +4.14e-5 |
| 72 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 270 | -1.07e-4 | -1.07e-4 | -1.07e-4 | +2.66e-5 |
| 73 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 259 | +1.71e-5 | +1.71e-5 | +1.71e-5 | +2.56e-5 |
| 74 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 241 | -8.24e-5 | -8.24e-5 | -8.24e-5 | +1.48e-5 |
| 75 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 263 | -8.70e-6 | -8.70e-6 | -8.70e-6 | +1.25e-5 |
| 76 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 293 | +4.07e-5 | +4.07e-5 | +4.07e-5 | +1.53e-5 |
| 77 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 294 | +1.51e-4 | +1.51e-4 | +1.51e-4 | +2.88e-5 |
| 78 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 300 | -1.55e-5 | -1.55e-5 | -1.55e-5 | +2.44e-5 |
| 79 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 244 | -3.83e-5 | -3.83e-5 | -3.83e-5 | +1.81e-5 |
| 80 | 3.00e-1 | 2 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 219 | -2.46e-4 | +1.58e-5 | -1.15e-4 | -5.88e-6 |
| 82 | 3.00e-1 | 1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 324 | -9.28e-5 | -9.28e-5 | -9.28e-5 | -1.46e-5 |
| 83 | 3.00e-1 | 2 | 1.94e-1 | 2.12e-1 | 2.03e-1 | 1.94e-1 | 210 | -4.17e-4 | +4.61e-4 | +2.23e-5 | -1.20e-5 |
| 84 | 3.00e-1 | 1 | 1.88e-1 | 1.88e-1 | 1.88e-1 | 1.88e-1 | 240 | -1.33e-4 | -1.33e-4 | -1.33e-4 | -2.41e-5 |
| 85 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 272 | +1.93e-4 | +1.93e-4 | +1.93e-4 | -2.40e-6 |
| 86 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 272 | +8.95e-5 | +8.95e-5 | +8.95e-5 | +6.79e-6 |
| 87 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 236 | +8.69e-5 | +8.69e-5 | +8.69e-5 | +1.48e-5 |
| 88 | 3.00e-1 | 1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 270 | -2.49e-4 | -2.49e-4 | -2.49e-4 | -1.15e-5 |
| 89 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 263 | +1.14e-4 | +1.14e-4 | +1.14e-4 | +9.94e-7 |
| 90 | 3.00e-1 | 2 | 1.97e-1 | 2.01e-1 | 1.99e-1 | 1.97e-1 | 172 | -1.12e-4 | +2.12e-5 | -4.54e-5 | -8.48e-6 |
| 91 | 3.00e-1 | 1 | 1.75e-1 | 1.75e-1 | 1.75e-1 | 1.75e-1 | 183 | -6.31e-4 | -6.31e-4 | -6.31e-4 | -7.07e-5 |
| 92 | 3.00e-1 | 2 | 1.80e-1 | 1.84e-1 | 1.82e-1 | 1.80e-1 | 182 | -1.30e-4 | +2.72e-4 | +7.10e-5 | -4.58e-5 |
| 93 | 3.00e-1 | 1 | 1.81e-1 | 1.81e-1 | 1.81e-1 | 1.81e-1 | 267 | +1.13e-5 | +1.13e-5 | +1.13e-5 | -4.01e-5 |
| 94 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 227 | +5.51e-4 | +5.51e-4 | +5.51e-4 | +1.91e-5 |
| 95 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 242 | -1.86e-4 | -1.86e-4 | -1.86e-4 | -1.46e-6 |
| 96 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 242 | +6.52e-5 | +6.52e-5 | +6.52e-5 | +5.20e-6 |
| 97 | 3.00e-1 | 2 | 1.91e-1 | 1.95e-1 | 1.93e-1 | 1.91e-1 | 195 | -1.06e-4 | -7.64e-5 | -9.14e-5 | -1.33e-5 |
| 98 | 3.00e-1 | 1 | 1.83e-1 | 1.83e-1 | 1.83e-1 | 1.83e-1 | 212 | -2.03e-4 | -2.03e-4 | -2.03e-4 | -3.23e-5 |
| 99 | 3.00e-1 | 1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 214 | +2.08e-4 | +2.08e-4 | +2.08e-4 | -8.29e-6 |
| 100 | 3.00e-2 | 2 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 162 | -4.70e-5 | +8.78e-6 | -1.91e-5 | -1.01e-5 |
| 101 | 3.00e-2 | 2 | 1.82e-2 | 1.84e-2 | 1.83e-2 | 1.84e-2 | 163 | -1.23e-2 | +5.72e-5 | -6.14e-3 | -1.11e-3 |
| 102 | 3.00e-2 | 1 | 1.84e-2 | 1.84e-2 | 1.84e-2 | 1.84e-2 | 189 | +1.63e-5 | +1.63e-5 | +1.63e-5 | -1.00e-3 |
| 103 | 3.00e-2 | 2 | 2.01e-2 | 2.20e-2 | 2.11e-2 | 2.20e-2 | 169 | +3.95e-4 | +5.36e-4 | +4.66e-4 | -7.21e-4 |
| 104 | 3.00e-2 | 1 | 2.10e-2 | 2.10e-2 | 2.10e-2 | 2.10e-2 | 200 | -2.41e-4 | -2.41e-4 | -2.41e-4 | -6.73e-4 |
| 105 | 3.00e-2 | 1 | 2.29e-2 | 2.29e-2 | 2.29e-2 | 2.29e-2 | 216 | +4.10e-4 | +4.10e-4 | +4.10e-4 | -5.65e-4 |
| 106 | 3.00e-2 | 2 | 2.33e-2 | 2.48e-2 | 2.41e-2 | 2.48e-2 | 176 | +7.83e-5 | +3.37e-4 | +2.08e-4 | -4.17e-4 |
| 107 | 3.00e-2 | 1 | 2.21e-2 | 2.21e-2 | 2.21e-2 | 2.21e-2 | 215 | -5.28e-4 | -5.28e-4 | -5.28e-4 | -4.28e-4 |
| 108 | 3.00e-2 | 2 | 2.49e-2 | 2.55e-2 | 2.52e-2 | 2.49e-2 | 179 | -1.42e-4 | +7.25e-4 | +2.92e-4 | -2.95e-4 |
| 109 | 3.00e-2 | 1 | 2.47e-2 | 2.47e-2 | 2.47e-2 | 2.47e-2 | 210 | -4.29e-5 | -4.29e-5 | -4.29e-5 | -2.70e-4 |
| 110 | 3.00e-2 | 1 | 2.67e-2 | 2.67e-2 | 2.67e-2 | 2.67e-2 | 207 | +3.80e-4 | +3.80e-4 | +3.80e-4 | -2.05e-4 |
| 111 | 3.00e-2 | 2 | 2.63e-2 | 2.65e-2 | 2.64e-2 | 2.65e-2 | 166 | -6.29e-5 | +3.79e-5 | -1.25e-5 | -1.68e-4 |
| 112 | 3.00e-2 | 1 | 2.55e-2 | 2.55e-2 | 2.55e-2 | 2.55e-2 | 200 | -1.91e-4 | -1.91e-4 | -1.91e-4 | -1.70e-4 |
| 113 | 3.00e-2 | 2 | 2.75e-2 | 2.81e-2 | 2.78e-2 | 2.81e-2 | 171 | +1.10e-4 | +3.78e-4 | +2.44e-4 | -9.29e-5 |
| 114 | 3.00e-2 | 1 | 2.71e-2 | 2.71e-2 | 2.71e-2 | 2.71e-2 | 238 | -1.41e-4 | -1.41e-4 | -1.41e-4 | -9.77e-5 |
| 115 | 3.00e-2 | 2 | 2.78e-2 | 3.07e-2 | 2.92e-2 | 2.78e-2 | 151 | -6.66e-4 | +7.06e-4 | +2.01e-5 | -8.22e-5 |
| 116 | 3.00e-2 | 1 | 2.68e-2 | 2.68e-2 | 2.68e-2 | 2.68e-2 | 185 | -1.91e-4 | -1.91e-4 | -1.91e-4 | -9.30e-5 |
| 117 | 3.00e-2 | 2 | 2.85e-2 | 2.95e-2 | 2.90e-2 | 2.85e-2 | 143 | -2.23e-4 | +5.27e-4 | +1.52e-4 | -5.01e-5 |
| 118 | 3.00e-2 | 2 | 2.69e-2 | 3.08e-2 | 2.89e-2 | 3.08e-2 | 159 | -3.22e-4 | +8.51e-4 | +2.65e-4 | +1.55e-5 |
| 119 | 3.00e-2 | 1 | 2.91e-2 | 2.91e-2 | 2.91e-2 | 2.91e-2 | 191 | -2.91e-4 | -2.91e-4 | -2.91e-4 | -1.51e-5 |
| 120 | 3.00e-2 | 2 | 3.21e-2 | 3.21e-2 | 3.21e-2 | 3.21e-2 | 160 | -2.09e-6 | +5.04e-4 | +2.51e-4 | +3.28e-5 |
| 121 | 3.00e-2 | 2 | 3.00e-2 | 3.21e-2 | 3.11e-2 | 3.21e-2 | 140 | -3.49e-4 | +4.89e-4 | +6.97e-5 | +4.40e-5 |
| 122 | 3.00e-2 | 1 | 2.96e-2 | 2.96e-2 | 2.96e-2 | 2.96e-2 | 134 | -6.16e-4 | -6.16e-4 | -6.16e-4 | -2.19e-5 |
| 123 | 3.00e-2 | 1 | 2.83e-2 | 2.83e-2 | 2.83e-2 | 2.83e-2 | 276 | -1.57e-4 | -1.57e-4 | -1.57e-4 | -3.54e-5 |
| 124 | 3.00e-2 | 1 | 4.08e-2 | 4.08e-2 | 4.08e-2 | 4.08e-2 | 295 | +1.23e-3 | +1.23e-3 | +1.23e-3 | +9.14e-5 |
| 125 | 3.00e-2 | 1 | 4.24e-2 | 4.24e-2 | 4.24e-2 | 4.24e-2 | 281 | +1.39e-4 | +1.39e-4 | +1.39e-4 | +9.62e-5 |
| 126 | 3.00e-2 | 1 | 4.20e-2 | 4.20e-2 | 4.20e-2 | 4.20e-2 | 280 | -3.17e-5 | -3.17e-5 | -3.17e-5 | +8.34e-5 |
| 127 | 3.00e-2 | 1 | 4.22e-2 | 4.22e-2 | 4.22e-2 | 4.22e-2 | 260 | +1.99e-5 | +1.99e-5 | +1.99e-5 | +7.70e-5 |
| 128 | 3.00e-2 | 1 | 4.14e-2 | 4.14e-2 | 4.14e-2 | 4.14e-2 | 271 | -7.05e-5 | -7.05e-5 | -7.05e-5 | +6.23e-5 |
| 129 | 3.00e-2 | 1 | 4.23e-2 | 4.23e-2 | 4.23e-2 | 4.23e-2 | 262 | +7.74e-5 | +7.74e-5 | +7.74e-5 | +6.38e-5 |
| 130 | 3.00e-2 | 1 | 4.35e-2 | 4.35e-2 | 4.35e-2 | 4.35e-2 | 280 | +1.02e-4 | +1.02e-4 | +1.02e-4 | +6.76e-5 |
| 131 | 3.00e-2 | 1 | 4.53e-2 | 4.53e-2 | 4.53e-2 | 4.53e-2 | 275 | +1.51e-4 | +1.51e-4 | +1.51e-4 | +7.59e-5 |
| 132 | 3.00e-2 | 1 | 4.64e-2 | 4.64e-2 | 4.64e-2 | 4.64e-2 | 289 | +8.30e-5 | +8.30e-5 | +8.30e-5 | +7.66e-5 |
| 133 | 3.00e-2 | 1 | 4.77e-2 | 4.77e-2 | 4.77e-2 | 4.77e-2 | 254 | +1.01e-4 | +1.01e-4 | +1.01e-4 | +7.91e-5 |
| 134 | 3.00e-2 | 1 | 4.59e-2 | 4.59e-2 | 4.59e-2 | 4.59e-2 | 279 | -1.31e-4 | -1.31e-4 | -1.31e-4 | +5.81e-5 |
| 135 | 3.00e-2 | 1 | 4.70e-2 | 4.70e-2 | 4.70e-2 | 4.70e-2 | 295 | +7.79e-5 | +7.79e-5 | +7.79e-5 | +6.01e-5 |
| 136 | 3.00e-2 | 1 | 4.97e-2 | 4.97e-2 | 4.97e-2 | 4.97e-2 | 281 | +1.97e-4 | +1.97e-4 | +1.97e-4 | +7.38e-5 |
| 137 | 3.00e-2 | 2 | 4.91e-2 | 4.96e-2 | 4.93e-2 | 4.96e-2 | 241 | -4.49e-5 | +4.15e-5 | -1.73e-6 | +5.99e-5 |
| 139 | 3.00e-2 | 2 | 4.84e-2 | 5.53e-2 | 5.18e-2 | 5.53e-2 | 241 | -7.04e-5 | +5.47e-4 | +2.38e-4 | +9.68e-5 |
| 141 | 3.00e-2 | 1 | 4.96e-2 | 4.96e-2 | 4.96e-2 | 4.96e-2 | 314 | -3.42e-4 | -3.42e-4 | -3.42e-4 | +5.29e-5 |
| 142 | 3.00e-2 | 1 | 5.61e-2 | 5.61e-2 | 5.61e-2 | 5.61e-2 | 270 | +4.56e-4 | +4.56e-4 | +4.56e-4 | +9.33e-5 |
| 143 | 3.00e-2 | 1 | 5.33e-2 | 5.33e-2 | 5.33e-2 | 5.33e-2 | 269 | -1.93e-4 | -1.93e-4 | -1.93e-4 | +6.46e-5 |
| 144 | 3.00e-2 | 2 | 5.28e-2 | 5.30e-2 | 5.29e-2 | 5.28e-2 | 212 | -2.51e-5 | -1.41e-5 | -1.96e-5 | +4.87e-5 |
| 145 | 3.00e-2 | 1 | 4.93e-2 | 4.93e-2 | 4.93e-2 | 4.93e-2 | 240 | -2.89e-4 | -2.89e-4 | -2.89e-4 | +1.49e-5 |
| 146 | 3.00e-2 | 1 | 5.26e-2 | 5.26e-2 | 5.26e-2 | 5.26e-2 | 224 | +2.92e-4 | +2.92e-4 | +2.92e-4 | +4.26e-5 |
| 147 | 3.00e-2 | 1 | 5.18e-2 | 5.18e-2 | 5.18e-2 | 5.18e-2 | 262 | -5.88e-5 | -5.88e-5 | -5.88e-5 | +3.25e-5 |
| 148 | 3.00e-2 | 1 | 5.47e-2 | 5.47e-2 | 5.47e-2 | 5.47e-2 | 244 | +2.22e-4 | +2.22e-4 | +2.22e-4 | +5.15e-5 |
| 149 | 3.00e-2 | 1 | 5.47e-2 | 5.47e-2 | 5.47e-2 | 5.47e-2 | 269 | +1.10e-6 | +1.10e-6 | +1.10e-6 | +4.64e-5 |
| 150 | 3.00e-3 | 1 | 5.97e-2 | 5.97e-2 | 5.97e-2 | 5.97e-2 | 260 | +3.35e-4 | +3.35e-4 | +3.35e-4 | +7.53e-5 |
| 151 | 3.00e-3 | 1 | 5.80e-2 | 5.80e-2 | 5.80e-2 | 5.80e-2 | 280 | -1.04e-4 | -1.04e-4 | -1.04e-4 | +5.74e-5 |
| 152 | 3.00e-3 | 1 | 6.02e-3 | 6.02e-3 | 6.02e-3 | 6.02e-3 | 268 | -8.45e-3 | -8.45e-3 | -8.45e-3 | -7.93e-4 |
| 153 | 3.00e-3 | 2 | 5.50e-3 | 5.69e-3 | 5.59e-3 | 5.50e-3 | 187 | -2.24e-4 | -1.81e-4 | -2.03e-4 | -6.81e-4 |
| 154 | 3.00e-3 | 1 | 4.79e-3 | 4.79e-3 | 4.79e-3 | 4.79e-3 | 226 | -6.13e-4 | -6.13e-4 | -6.13e-4 | -6.74e-4 |
| 155 | 3.00e-3 | 2 | 5.16e-3 | 5.19e-3 | 5.18e-3 | 5.19e-3 | 207 | +2.82e-5 | +3.24e-4 | +1.76e-4 | -5.14e-4 |
| 157 | 3.00e-3 | 2 | 5.08e-3 | 5.98e-3 | 5.53e-3 | 5.98e-3 | 229 | -7.19e-5 | +7.13e-4 | +3.21e-4 | -3.52e-4 |
| 158 | 3.00e-3 | 1 | 5.24e-3 | 5.24e-3 | 5.24e-3 | 5.24e-3 | 221 | -5.98e-4 | -5.98e-4 | -5.98e-4 | -3.76e-4 |
| 159 | 3.00e-3 | 1 | 5.28e-3 | 5.28e-3 | 5.28e-3 | 5.28e-3 | 206 | +4.29e-5 | +4.29e-5 | +4.29e-5 | -3.34e-4 |
| 160 | 3.00e-3 | 2 | 5.14e-3 | 5.25e-3 | 5.19e-3 | 5.14e-3 | 187 | -1.09e-4 | -3.42e-5 | -7.14e-5 | -2.85e-4 |
| 161 | 3.00e-3 | 1 | 4.93e-3 | 4.93e-3 | 4.93e-3 | 4.93e-3 | 208 | -2.00e-4 | -2.00e-4 | -2.00e-4 | -2.76e-4 |
| 162 | 3.00e-3 | 1 | 5.28e-3 | 5.28e-3 | 5.28e-3 | 5.28e-3 | 218 | +3.09e-4 | +3.09e-4 | +3.09e-4 | -2.18e-4 |
| 163 | 3.00e-3 | 1 | 5.46e-3 | 5.46e-3 | 5.46e-3 | 5.46e-3 | 228 | +1.48e-4 | +1.48e-4 | +1.48e-4 | -1.81e-4 |
| 164 | 3.00e-3 | 2 | 5.42e-3 | 5.59e-3 | 5.51e-3 | 5.59e-3 | 198 | -2.96e-5 | +1.54e-4 | +6.20e-5 | -1.34e-4 |
| 165 | 3.00e-3 | 1 | 5.15e-3 | 5.15e-3 | 5.15e-3 | 5.15e-3 | 223 | -3.70e-4 | -3.70e-4 | -3.70e-4 | -1.58e-4 |
| 166 | 3.00e-3 | 1 | 5.63e-3 | 5.63e-3 | 5.63e-3 | 5.63e-3 | 240 | +3.70e-4 | +3.70e-4 | +3.70e-4 | -1.05e-4 |
| 167 | 3.00e-3 | 1 | 5.78e-3 | 5.78e-3 | 5.78e-3 | 5.78e-3 | 226 | +1.22e-4 | +1.22e-4 | +1.22e-4 | -8.21e-5 |
| 168 | 3.00e-3 | 1 | 5.34e-3 | 5.34e-3 | 5.34e-3 | 5.34e-3 | 247 | -3.22e-4 | -3.22e-4 | -3.22e-4 | -1.06e-4 |
| 169 | 3.00e-3 | 2 | 5.90e-3 | 6.13e-3 | 6.02e-3 | 6.13e-3 | 188 | +2.00e-4 | +3.67e-4 | +2.84e-4 | -3.30e-5 |
| 170 | 3.00e-3 | 2 | 5.06e-3 | 5.58e-3 | 5.32e-3 | 5.58e-3 | 165 | -8.82e-4 | +5.94e-4 | -1.44e-4 | -4.67e-5 |
| 171 | 3.00e-3 | 1 | 4.89e-3 | 4.89e-3 | 4.89e-3 | 4.89e-3 | 191 | -6.93e-4 | -6.93e-4 | -6.93e-4 | -1.11e-4 |
| 172 | 3.00e-3 | 1 | 5.37e-3 | 5.37e-3 | 5.37e-3 | 5.37e-3 | 184 | +5.12e-4 | +5.12e-4 | +5.12e-4 | -4.90e-5 |
| 173 | 3.00e-3 | 2 | 5.31e-3 | 5.38e-3 | 5.35e-3 | 5.38e-3 | 179 | -6.07e-5 | +7.07e-5 | +5.01e-6 | -3.81e-5 |
| 174 | 3.00e-3 | 1 | 5.26e-3 | 5.26e-3 | 5.26e-3 | 5.26e-3 | 204 | -1.16e-4 | -1.16e-4 | -1.16e-4 | -4.59e-5 |
| 175 | 3.00e-3 | 1 | 5.43e-3 | 5.43e-3 | 5.43e-3 | 5.43e-3 | 198 | +1.67e-4 | +1.67e-4 | +1.67e-4 | -2.46e-5 |
| 176 | 3.00e-3 | 2 | 5.39e-3 | 5.66e-3 | 5.52e-3 | 5.66e-3 | 175 | -3.45e-5 | +2.73e-4 | +1.19e-4 | +4.23e-6 |
| 177 | 3.00e-3 | 1 | 5.23e-3 | 5.23e-3 | 5.23e-3 | 5.23e-3 | 194 | -4.00e-4 | -4.00e-4 | -4.00e-4 | -3.62e-5 |
| 178 | 3.00e-3 | 2 | 5.40e-3 | 5.65e-3 | 5.52e-3 | 5.65e-3 | 169 | +1.44e-4 | +2.74e-4 | +2.09e-4 | +1.10e-5 |
| 179 | 3.00e-3 | 1 | 5.17e-3 | 5.17e-3 | 5.17e-3 | 5.17e-3 | 203 | -4.38e-4 | -4.38e-4 | -4.38e-4 | -3.39e-5 |
| 180 | 3.00e-3 | 2 | 5.61e-3 | 5.76e-3 | 5.69e-3 | 5.76e-3 | 145 | +1.81e-4 | +3.60e-4 | +2.71e-4 | +2.31e-5 |
| 181 | 3.00e-3 | 1 | 4.92e-3 | 4.92e-3 | 4.92e-3 | 4.92e-3 | 179 | -8.76e-4 | -8.76e-4 | -8.76e-4 | -6.68e-5 |
| 182 | 3.00e-3 | 2 | 5.27e-3 | 5.30e-3 | 5.28e-3 | 5.27e-3 | 145 | -3.28e-5 | +3.99e-4 | +1.83e-4 | -2.15e-5 |
| 183 | 3.00e-3 | 2 | 4.88e-3 | 5.53e-3 | 5.20e-3 | 5.53e-3 | 145 | -4.01e-4 | +8.64e-4 | +2.31e-4 | +3.28e-5 |
| 184 | 3.00e-3 | 2 | 4.93e-3 | 5.32e-3 | 5.13e-3 | 5.32e-3 | 145 | -6.69e-4 | +5.17e-4 | -7.61e-5 | +1.81e-5 |
| 185 | 3.00e-3 | 1 | 4.77e-3 | 4.77e-3 | 4.77e-3 | 4.77e-3 | 155 | -7.05e-4 | -7.05e-4 | -7.05e-4 | -5.43e-5 |
| 186 | 3.00e-3 | 1 | 4.92e-3 | 4.92e-3 | 4.92e-3 | 4.92e-3 | 305 | +1.02e-4 | +1.02e-4 | +1.02e-4 | -3.87e-5 |
| 187 | 3.00e-3 | 1 | 6.81e-3 | 6.81e-3 | 6.81e-3 | 6.81e-3 | 276 | +1.18e-3 | +1.18e-3 | +1.18e-3 | +8.32e-5 |
| 188 | 3.00e-3 | 1 | 6.77e-3 | 6.77e-3 | 6.77e-3 | 6.77e-3 | 300 | -2.22e-5 | -2.22e-5 | -2.22e-5 | +7.27e-5 |
| 190 | 3.00e-3 | 2 | 6.86e-3 | 7.95e-3 | 7.41e-3 | 7.95e-3 | 246 | +3.45e-5 | +5.97e-4 | +3.16e-4 | +1.22e-4 |
| 192 | 3.00e-3 | 1 | 6.34e-3 | 6.34e-3 | 6.34e-3 | 6.34e-3 | 318 | -7.10e-4 | -7.10e-4 | -7.10e-4 | +3.85e-5 |
| 193 | 3.00e-3 | 2 | 6.69e-3 | 7.10e-3 | 6.89e-3 | 6.69e-3 | 246 | -2.40e-4 | +3.88e-4 | +7.39e-5 | +4.21e-5 |
| 195 | 3.00e-3 | 2 | 6.63e-3 | 7.18e-3 | 6.90e-3 | 7.18e-3 | 246 | -2.97e-5 | +3.28e-4 | +1.49e-4 | +6.42e-5 |
| 197 | 3.00e-3 | 2 | 6.40e-3 | 7.46e-3 | 6.93e-3 | 7.46e-3 | 249 | -3.42e-4 | +6.13e-4 | +1.36e-4 | +8.26e-5 |
| 199 | 3.00e-3 | 1 | 6.66e-3 | 6.66e-3 | 6.66e-3 | 6.66e-3 | 328 | -3.46e-4 | -3.46e-4 | -3.46e-4 | +3.97e-5 |

