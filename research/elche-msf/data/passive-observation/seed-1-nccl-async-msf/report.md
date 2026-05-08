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

GPU columns = compute utilization % (not load). Idle = total time with <5% utilization.

### resnet-graph

> Published: CIFAR-10 91.25% acc, Graph builder ([He et al. 2015](https://arxiv.org/abs/1512.03385), Table 6)

| Mode | Loss | Eval | vs Ref | Total (s) | Syncs | Avg Sync (ms) | GPU0 | GPU1 | GPU2 | Idle (s) |
|------|------|------|--------|-----------|-------|--------------|------|------|------|----------|
| nccl-async | 0.061583 | 0.9173 | +0.0048 | 1934.6 | 464 | 42.8 | 100% | 100% | 100% | 7.0 |

## Best Mode per Model

| Model | Best Eval | Mode | Fastest (within 2% of solo-0) | Mode |
|-------|-----------|------|-------------------------------|------|
| resnet-graph | 0.9173 | nccl-async | - | - |

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
| nccl-async | 1.9563 | 0.7756 | 0.6215 | 0.5549 | 0.5238 | 0.5102 | 0.4932 | 0.4692 | 0.4618 | 0.4515 | 0.2025 | 0.1703 | 0.1549 | 0.1445 | 0.1419 | 0.0795 | 0.0712 | 0.0651 | 0.0624 | 0.0616 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | nccl-async | 0 | cuda:0 | 0.4030 | 2.5 |
| resnet-graph | nccl-async | 1 | cuda:1 | 0.3010 | 3.4 |
| resnet-graph | nccl-async | 2 | cuda:2 | 0.2959 | 3.4 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) | GPU2 Peak (MB) | GPU2 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|---------------|---------------|
| resnet-graph | nccl-async | 356 | 355 | 398 | 395 | 390 | 378 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| nccl-async | gpu2 | 1933.5 | 1.0 | epoch-boundary(199) |
| nccl-async | gpu1 | 1933.6 | 0.9 | epoch-boundary(199) |
| nccl-async | gpu0 | 1933.4 | 0.6 | epoch-boundary(199) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | nccl-async | gpu0 | 0.6s | 0.0s | 0.0s | 0.0s | 2.7s |
| resnet-graph | nccl-async | gpu1 | 0.9s | 0.0s | 0.0s | 0.0s | 2.3s |
| resnet-graph | nccl-async | gpu2 | 1.0s | 0.0s | 0.0s | 0.0s | 2.0s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | nccl-async | 340 | 0 | 464 | 42.8 | 3759/9604 | 0 | 0.0 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | nccl-async | 195.6 | 10.1% |

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
| resnet-graph | nccl-async | 194 | 464 | 0 | 6.33e-3 | +7.38e-4 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | nccl-async | 0 | 464 | 6.88e-2 | 7.38e-2 | 0.00e0 | 3.59e-1 | 35.3 | -1.69e-4 | 2.51e-3 |
| resnet-graph | nccl-async | 1 | 464 | 6.96e-2 | 7.56e-2 | 0.00e0 | 3.94e-1 | 38.4 | -1.77e-4 | 3.64e-3 |
| resnet-graph | nccl-async | 2 | 464 | 6.89e-2 | 7.47e-2 | 0.00e0 | 3.53e-1 | 26.3 | -1.60e-4 | 3.49e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | nccl-async | rank0 ↔ rank1 | +0.9981 |
| resnet-graph | nccl-async | rank0 ↔ rank2 | +0.9983 |
| resnet-graph | nccl-async | rank1 ↔ rank2 | +0.9991 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | nccl-async | 31 (1,2,3,4,5,6,7,9…145,150) | 0 (—) | — | 1,2,3,4,5,6,7,9…145,150 | — |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | nccl-async | 0e0 | 3 | 17 | 17 |
| resnet-graph | nccl-async | 0e0 | 5 | 10 | 10 |
| resnet-graph | nccl-async | 0e0 | 10 | 3 | 3 |
| resnet-graph | nccl-async | 1e-4 | 3 | 4 | 4 |
| resnet-graph | nccl-async | 1e-4 | 5 | 2 | 2 |
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
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 212 | +0.080 |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | 67 | +0.174 |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | 180 | +0.012 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | nccl-async | 461 | -0.010 | 193 | +0.166 | +0.299 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | nccl-async | 462 | 3.33e1–7.85e1 | 6.13e1 | 2.61e-3 | 5.46e-3 | 7.92e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 214 | 61–78028 | +1.335e-5 | 0.512 | +1.378e-5 | 0.517 | 94 | +5.220e-6 | 0.148 | 30–1037 | +1.383e-3 | 0.731 |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 199 | 892–78028 | +1.298e-5 | 0.531 | +1.334e-5 | 0.534 | 93 | +4.646e-6 | 0.124 | 36–1037 | +1.369e-3 | 0.783 |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | 68 | 78484–116825 | +2.198e-5 | 0.366 | +2.219e-5 | 0.368 | 50 | +2.120e-5 | 0.404 | 270–835 | +1.208e-3 | 0.159 |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | 181 | 117434–155794 | -4.433e-5 | 0.556 | -4.539e-5 | 0.565 | 50 | -2.596e-5 | 0.302 | 39–708 | +2.681e-3 | 0.639 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | +1.383e-3 | r0: +1.346e-3, r1: +1.397e-3, r2: +1.410e-3 | r0: 0.754, r1: 0.712, r2: 0.723 | 1.05× | ✓ |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +1.369e-3 | r0: +1.336e-3, r1: +1.380e-3, r2: +1.395e-3 | r0: 0.810, r1: 0.763, r2: 0.771 | 1.04× | ✓ |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | +1.208e-3 | r0: +1.203e-3, r1: +1.206e-3, r2: +1.215e-3 | r0: 0.156, r1: 0.158, r2: 0.160 | 1.01× | ✓ |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | +2.681e-3 | r0: +2.684e-3, r1: +2.707e-3, r2: +2.662e-3 | r0: 0.653, r1: 0.627, r2: 0.629 | 1.02× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | nccl-async | `█▇▇████████████████████▇▄▄▅▅▅▅▅▅▅▆▆▆▃▂▂▂▂▂▂▂▁▁▁▁▂` | `▁▄▄▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▆▅▅▆▆▄▃▅▅▅▆▆▆▆▆▆▆▆▃▄▅▅▆▆▆▅▅▄▃▇█` |

### Per-Epoch Detail: resnet-graph / nccl-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 16 | 0.00e0 | 3.94e-1 | 8.87e-2 | 5.97e-2 | 20 | -6.42e-2 | +8.85e-3 | -1.01e-2 | -5.10e-3 |
| 1 | 3.00e-1 | 14 | 4.87e-2 | 1.20e-1 | 6.67e-2 | 7.19e-2 | 20 | -4.16e-2 | +3.60e-2 | -5.04e-5 | -6.03e-4 |
| 2 | 3.00e-1 | 15 | 5.94e-2 | 1.10e-1 | 7.04e-2 | 6.81e-2 | 16 | -2.32e-2 | +2.15e-2 | -3.93e-4 | -3.38e-4 |
| 3 | 3.00e-1 | 12 | 6.17e-2 | 1.30e-1 | 9.03e-2 | 1.06e-1 | 37 | -4.98e-2 | +4.00e-2 | +7.25e-4 | +4.09e-4 |
| 4 | 3.00e-1 | 7 | 9.15e-2 | 1.49e-1 | 1.04e-1 | 9.51e-2 | 25 | -1.67e-2 | +9.33e-3 | -7.71e-4 | -1.98e-4 |
| 5 | 3.00e-1 | 12 | 6.78e-2 | 1.35e-1 | 8.22e-2 | 8.11e-2 | 22 | -2.86e-2 | +1.83e-2 | -8.03e-4 | -3.17e-4 |
| 6 | 3.00e-1 | 12 | 6.91e-2 | 1.35e-1 | 8.41e-2 | 8.37e-2 | 18 | -3.72e-2 | +2.95e-2 | +5.71e-5 | -6.07e-5 |
| 7 | 3.00e-1 | 15 | 6.38e-2 | 1.35e-1 | 7.69e-2 | 6.38e-2 | 15 | -3.33e-2 | +3.36e-2 | -5.71e-4 | -9.13e-4 |
| 8 | 3.00e-1 | 1 | 6.47e-2 | 6.47e-2 | 6.47e-2 | 6.47e-2 | 17 | +8.06e-4 | +8.06e-4 | +8.06e-4 | -7.41e-4 |
| 9 | 3.00e-1 | 1 | 6.79e-2 | 6.79e-2 | 6.79e-2 | 6.79e-2 | 340 | +1.45e-4 | +1.45e-4 | +1.45e-4 | -6.52e-4 |
| 10 | 3.00e-1 | 1 | 2.39e-1 | 2.39e-1 | 2.39e-1 | 2.39e-1 | 342 | +3.68e-3 | +3.68e-3 | +3.68e-3 | -2.19e-4 |
| 11 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 365 | -2.85e-4 | -2.85e-4 | -2.85e-4 | -2.26e-4 |
| 13 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 372 | +2.77e-5 | +2.77e-5 | +2.77e-5 | -2.00e-4 |
| 14 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 340 | -5.66e-5 | -5.66e-5 | -5.66e-5 | -1.86e-4 |
| 15 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 294 | -6.09e-5 | -6.09e-5 | -6.09e-5 | -1.73e-4 |
| 16 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 306 | -1.71e-4 | -1.71e-4 | -1.71e-4 | -1.73e-4 |
| 17 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 298 | +1.43e-5 | +1.43e-5 | +1.43e-5 | -1.54e-4 |
| 18 | 3.00e-1 | 1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 305 | -4.20e-5 | -4.20e-5 | -4.20e-5 | -1.43e-4 |
| 20 | 3.00e-1 | 2 | 2.02e-1 | 2.12e-1 | 2.07e-1 | 2.12e-1 | 260 | +5.97e-5 | +1.92e-4 | +1.26e-4 | -9.14e-5 |
| 22 | 3.00e-1 | 2 | 1.91e-1 | 2.04e-1 | 1.98e-1 | 2.04e-1 | 260 | -3.27e-4 | +2.45e-4 | -4.11e-5 | -7.90e-5 |
| 24 | 3.00e-1 | 2 | 1.91e-1 | 2.06e-1 | 1.99e-1 | 2.06e-1 | 260 | -2.00e-4 | +2.91e-4 | +4.57e-5 | -5.28e-5 |
| 26 | 3.00e-1 | 1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 345 | -1.93e-4 | -1.93e-4 | -1.93e-4 | -6.68e-5 |
| 27 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 288 | +2.70e-4 | +2.70e-4 | +2.70e-4 | -3.31e-5 |
| 28 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 287 | -1.58e-4 | -1.58e-4 | -1.58e-4 | -4.57e-5 |
| 29 | 3.00e-1 | 1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 286 | -3.03e-5 | -3.03e-5 | -3.03e-5 | -4.41e-5 |
| 30 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 303 | +2.24e-5 | +2.24e-5 | +2.24e-5 | -3.75e-5 |
| 31 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 262 | +1.10e-4 | +1.10e-4 | +1.10e-4 | -2.27e-5 |
| 32 | 3.00e-1 | 1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 313 | -1.54e-4 | -1.54e-4 | -1.54e-4 | -3.58e-5 |
| 33 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 244 | +2.16e-4 | +2.16e-4 | +2.16e-4 | -1.07e-5 |
| 34 | 3.00e-1 | 1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 280 | -2.37e-4 | -2.37e-4 | -2.37e-4 | -3.33e-5 |
| 35 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 270 | +1.32e-4 | +1.32e-4 | +1.32e-4 | -1.68e-5 |
| 36 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 300 | -1.15e-5 | -1.15e-5 | -1.15e-5 | -1.63e-5 |
| 37 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 228 | +1.81e-4 | +1.81e-4 | +1.81e-4 | +3.46e-6 |
| 38 | 3.00e-1 | 2 | 1.88e-1 | 1.89e-1 | 1.89e-1 | 1.89e-1 | 213 | -4.00e-4 | +2.08e-5 | -1.89e-4 | -3.11e-5 |
| 39 | 3.00e-1 | 1 | 1.83e-1 | 1.83e-1 | 1.83e-1 | 1.83e-1 | 276 | -1.21e-4 | -1.21e-4 | -1.21e-4 | -4.00e-5 |
| 40 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 280 | +3.29e-4 | +3.29e-4 | +3.29e-4 | -3.10e-6 |
| 41 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 274 | +1.89e-5 | +1.89e-5 | +1.89e-5 | -9.07e-7 |
| 42 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 256 | +7.58e-6 | +7.58e-6 | +7.58e-6 | -5.79e-8 |
| 43 | 3.00e-1 | 1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 249 | -1.39e-4 | -1.39e-4 | -1.39e-4 | -1.40e-5 |
| 44 | 3.00e-1 | 1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 260 | -2.81e-5 | -2.81e-5 | -2.81e-5 | -1.54e-5 |
| 45 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 263 | +1.41e-4 | +1.41e-4 | +1.41e-4 | +3.01e-7 |
| 46 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 300 | -3.69e-5 | -3.69e-5 | -3.69e-5 | -3.42e-6 |
| 47 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 263 | +9.21e-5 | +9.21e-5 | +9.21e-5 | +6.13e-6 |
| 48 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 270 | -6.85e-5 | -6.85e-5 | -6.85e-5 | -1.33e-6 |
| 49 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 270 | +6.53e-6 | +6.53e-6 | +6.53e-6 | -5.43e-7 |
| 50 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 260 | -3.57e-5 | -3.57e-5 | -3.57e-5 | -4.06e-6 |
| 51 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 260 | +8.21e-6 | +8.21e-6 | +8.21e-6 | -2.83e-6 |
| 52 | 3.00e-1 | 1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 256 | -4.96e-5 | -4.96e-5 | -4.96e-5 | -7.52e-6 |
| 53 | 3.00e-1 | 1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 261 | +8.95e-6 | +8.95e-6 | +8.95e-6 | -5.87e-6 |
| 54 | 3.00e-1 | 2 | 1.95e-1 | 1.98e-1 | 1.97e-1 | 1.95e-1 | 214 | -5.64e-5 | +1.50e-5 | -2.07e-5 | -9.05e-6 |
| 55 | 3.00e-1 | 1 | 1.83e-1 | 1.83e-1 | 1.83e-1 | 1.83e-1 | 277 | -2.31e-4 | -2.31e-4 | -2.31e-4 | -3.13e-5 |
| 56 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 229 | +4.79e-4 | +4.79e-4 | +4.79e-4 | +1.98e-5 |
| 57 | 3.00e-1 | 1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 245 | -3.04e-4 | -3.04e-4 | -3.04e-4 | -1.26e-5 |
| 58 | 3.00e-1 | 1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 241 | +1.55e-4 | +1.55e-4 | +1.55e-4 | +4.17e-6 |
| 59 | 3.00e-1 | 2 | 1.92e-1 | 1.98e-1 | 1.95e-1 | 1.98e-1 | 200 | -1.14e-4 | +1.68e-4 | +2.70e-5 | +9.91e-6 |
| 60 | 3.00e-1 | 1 | 1.81e-1 | 1.81e-1 | 1.81e-1 | 1.81e-1 | 218 | -4.15e-4 | -4.15e-4 | -4.15e-4 | -3.26e-5 |
| 61 | 3.00e-1 | 1 | 1.88e-1 | 1.88e-1 | 1.88e-1 | 1.88e-1 | 225 | +1.64e-4 | +1.64e-4 | +1.64e-4 | -1.29e-5 |
| 62 | 3.00e-1 | 1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 262 | +5.49e-5 | +5.49e-5 | +5.49e-5 | -6.15e-6 |
| 63 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 283 | +1.88e-4 | +1.88e-4 | +1.88e-4 | +1.33e-5 |
| 64 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 222 | +3.58e-5 | +3.58e-5 | +3.58e-5 | +1.55e-5 |
| 65 | 3.00e-1 | 2 | 1.92e-1 | 1.94e-1 | 1.93e-1 | 1.92e-1 | 201 | -1.83e-4 | -6.45e-5 | -1.24e-4 | -1.04e-5 |
| 66 | 3.00e-1 | 1 | 1.84e-1 | 1.84e-1 | 1.84e-1 | 1.84e-1 | 230 | -1.93e-4 | -1.93e-4 | -1.93e-4 | -2.87e-5 |
| 67 | 3.00e-1 | 1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 229 | +2.50e-4 | +2.50e-4 | +2.50e-4 | -8.06e-7 |
| 68 | 3.00e-1 | 2 | 1.91e-1 | 1.94e-1 | 1.93e-1 | 1.94e-1 | 190 | -7.13e-5 | +8.49e-5 | +6.80e-6 | +1.42e-6 |
| 69 | 3.00e-1 | 1 | 1.83e-1 | 1.83e-1 | 1.83e-1 | 1.83e-1 | 219 | -2.89e-4 | -2.89e-4 | -2.89e-4 | -2.76e-5 |
| 70 | 3.00e-1 | 1 | 1.89e-1 | 1.89e-1 | 1.89e-1 | 1.89e-1 | 233 | +1.39e-4 | +1.39e-4 | +1.39e-4 | -1.09e-5 |
| 71 | 3.00e-1 | 1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 217 | +1.55e-4 | +1.55e-4 | +1.55e-4 | +5.65e-6 |
| 72 | 3.00e-1 | 2 | 1.91e-1 | 1.94e-1 | 1.93e-1 | 1.94e-1 | 193 | -8.70e-5 | +8.39e-5 | -1.57e-6 | +5.13e-6 |
| 73 | 3.00e-1 | 1 | 1.83e-1 | 1.83e-1 | 1.83e-1 | 1.83e-1 | 200 | -2.83e-4 | -2.83e-4 | -2.83e-4 | -2.36e-5 |
| 74 | 3.00e-1 | 2 | 1.83e-1 | 1.86e-1 | 1.85e-1 | 1.83e-1 | 166 | -9.85e-5 | +7.30e-5 | -1.27e-5 | -2.24e-5 |
| 75 | 3.00e-1 | 1 | 1.74e-1 | 1.74e-1 | 1.74e-1 | 1.74e-1 | 220 | -2.37e-4 | -2.37e-4 | -2.37e-4 | -4.39e-5 |
| 76 | 3.00e-1 | 2 | 1.81e-1 | 1.91e-1 | 1.86e-1 | 1.81e-1 | 188 | -2.83e-4 | +4.72e-4 | +9.44e-5 | -2.14e-5 |
| 77 | 3.00e-1 | 1 | 1.84e-1 | 1.84e-1 | 1.84e-1 | 1.84e-1 | 226 | +7.75e-5 | +7.75e-5 | +7.75e-5 | -1.15e-5 |
| 78 | 3.00e-1 | 2 | 1.91e-1 | 1.94e-1 | 1.92e-1 | 1.91e-1 | 190 | -7.66e-5 | +2.26e-4 | +7.49e-5 | +3.41e-6 |
| 80 | 3.00e-1 | 2 | 1.83e-1 | 1.96e-1 | 1.89e-1 | 1.96e-1 | 169 | -1.76e-4 | +4.04e-4 | +1.14e-4 | +2.74e-5 |
| 81 | 3.00e-1 | 2 | 1.75e-1 | 1.89e-1 | 1.82e-1 | 1.89e-1 | 159 | -5.43e-4 | +4.75e-4 | -3.44e-5 | +2.07e-5 |
| 82 | 3.00e-1 | 1 | 1.68e-1 | 1.68e-1 | 1.68e-1 | 1.68e-1 | 186 | -6.22e-4 | -6.22e-4 | -6.22e-4 | -4.36e-5 |
| 83 | 3.00e-1 | 2 | 1.77e-1 | 1.82e-1 | 1.80e-1 | 1.77e-1 | 167 | -1.70e-4 | +4.64e-4 | +1.47e-4 | -1.06e-5 |
| 84 | 3.00e-1 | 1 | 1.79e-1 | 1.79e-1 | 1.79e-1 | 1.79e-1 | 197 | +5.99e-5 | +5.99e-5 | +5.99e-5 | -3.55e-6 |
| 85 | 3.00e-1 | 2 | 1.85e-1 | 1.87e-1 | 1.86e-1 | 1.85e-1 | 167 | -4.41e-5 | +1.95e-4 | +7.54e-5 | +1.03e-5 |
| 86 | 3.00e-1 | 1 | 1.70e-1 | 1.70e-1 | 1.70e-1 | 1.70e-1 | 193 | -4.30e-4 | -4.30e-4 | -4.30e-4 | -3.38e-5 |
| 87 | 3.00e-1 | 2 | 1.84e-1 | 1.86e-1 | 1.85e-1 | 1.84e-1 | 167 | -6.10e-5 | +4.52e-4 | +1.95e-4 | +7.16e-6 |
| 88 | 3.00e-1 | 1 | 1.74e-1 | 1.74e-1 | 1.74e-1 | 1.74e-1 | 201 | -2.73e-4 | -2.73e-4 | -2.73e-4 | -2.09e-5 |
| 89 | 3.00e-1 | 2 | 1.84e-1 | 1.88e-1 | 1.86e-1 | 1.84e-1 | 170 | -1.10e-4 | +4.16e-4 | +1.53e-4 | +9.56e-6 |
| 90 | 3.00e-1 | 1 | 1.79e-1 | 1.79e-1 | 1.79e-1 | 1.79e-1 | 189 | -1.45e-4 | -1.45e-4 | -1.45e-4 | -5.93e-6 |
| 91 | 3.00e-1 | 2 | 1.81e-1 | 1.83e-1 | 1.82e-1 | 1.81e-1 | 141 | -5.90e-5 | +1.05e-4 | +2.29e-5 | -1.27e-6 |
| 92 | 3.00e-1 | 2 | 1.67e-1 | 1.76e-1 | 1.72e-1 | 1.76e-1 | 141 | -4.75e-4 | +3.85e-4 | -4.51e-5 | -5.30e-6 |
| 93 | 3.00e-1 | 2 | 1.63e-1 | 1.85e-1 | 1.74e-1 | 1.85e-1 | 150 | -4.12e-4 | +8.38e-4 | +2.13e-4 | +4.24e-5 |
| 94 | 3.00e-1 | 2 | 1.71e-1 | 1.87e-1 | 1.79e-1 | 1.87e-1 | 141 | -4.18e-4 | +6.43e-4 | +1.13e-4 | +6.11e-5 |
| 95 | 3.00e-1 | 1 | 1.62e-1 | 1.62e-1 | 1.62e-1 | 1.62e-1 | 179 | -7.89e-4 | -7.89e-4 | -7.89e-4 | -2.40e-5 |
| 96 | 3.00e-1 | 2 | 1.79e-1 | 1.83e-1 | 1.81e-1 | 1.79e-1 | 139 | -1.48e-4 | +6.55e-4 | +2.53e-4 | +2.47e-5 |
| 97 | 3.00e-1 | 2 | 1.66e-1 | 1.76e-1 | 1.71e-1 | 1.76e-1 | 127 | -4.56e-4 | +4.69e-4 | +6.65e-6 | +2.59e-5 |
| 98 | 3.00e-1 | 2 | 1.61e-1 | 1.75e-1 | 1.68e-1 | 1.75e-1 | 124 | -5.38e-4 | +6.76e-4 | +6.91e-5 | +4.02e-5 |
| 99 | 3.00e-1 | 2 | 1.58e-1 | 1.75e-1 | 1.67e-1 | 1.75e-1 | 124 | -6.46e-4 | +8.51e-4 | +1.03e-4 | +5.95e-5 |
| 100 | 3.00e-2 | 2 | 1.25e-1 | 1.60e-1 | 1.42e-1 | 1.25e-1 | 130 | -1.90e-3 | -5.42e-4 | -1.22e-3 | -1.90e-4 |
| 101 | 3.00e-2 | 2 | 1.69e-2 | 1.86e-2 | 1.78e-2 | 1.86e-2 | 115 | -1.25e-2 | +8.54e-4 | -5.82e-3 | -1.19e-3 |
| 102 | 3.00e-2 | 2 | 1.67e-2 | 1.85e-2 | 1.76e-2 | 1.85e-2 | 107 | -7.77e-4 | +9.16e-4 | +6.98e-5 | -9.45e-4 |
| 103 | 3.00e-2 | 2 | 1.70e-2 | 1.89e-2 | 1.79e-2 | 1.89e-2 | 107 | -6.14e-4 | +1.03e-3 | +2.08e-4 | -7.18e-4 |
| 104 | 3.00e-2 | 3 | 1.81e-2 | 2.03e-2 | 1.91e-2 | 1.91e-2 | 116 | -5.40e-4 | +1.00e-3 | +4.78e-5 | -5.13e-4 |
| 105 | 3.00e-2 | 2 | 1.88e-2 | 2.13e-2 | 2.01e-2 | 2.13e-2 | 116 | -7.29e-5 | +1.04e-3 | +4.86e-4 | -3.18e-4 |
| 106 | 3.00e-2 | 2 | 2.01e-2 | 2.15e-2 | 2.08e-2 | 2.15e-2 | 108 | -4.12e-4 | +6.26e-4 | +1.07e-4 | -2.32e-4 |
| 107 | 3.00e-2 | 2 | 1.90e-2 | 1.95e-2 | 1.93e-2 | 1.95e-2 | 100 | -1.13e-3 | +2.78e-4 | -4.25e-4 | -2.61e-4 |
| 108 | 3.00e-2 | 1 | 1.88e-2 | 1.88e-2 | 1.88e-2 | 1.88e-2 | 276 | -1.36e-4 | -1.36e-4 | -1.36e-4 | -2.49e-4 |
| 109 | 3.00e-2 | 1 | 3.06e-2 | 3.06e-2 | 3.06e-2 | 3.06e-2 | 300 | +1.63e-3 | +1.63e-3 | +1.63e-3 | -6.15e-5 |
| 110 | 3.00e-2 | 1 | 3.22e-2 | 3.22e-2 | 3.22e-2 | 3.22e-2 | 260 | +1.96e-4 | +1.96e-4 | +1.96e-4 | -3.57e-5 |
| 111 | 3.00e-2 | 1 | 3.14e-2 | 3.14e-2 | 3.14e-2 | 3.14e-2 | 250 | -1.05e-4 | -1.05e-4 | -1.05e-4 | -4.27e-5 |
| 112 | 3.00e-2 | 1 | 3.19e-2 | 3.19e-2 | 3.19e-2 | 3.19e-2 | 254 | +6.01e-5 | +6.01e-5 | +6.01e-5 | -3.24e-5 |
| 113 | 3.00e-2 | 1 | 3.24e-2 | 3.24e-2 | 3.24e-2 | 3.24e-2 | 258 | +5.78e-5 | +5.78e-5 | +5.78e-5 | -2.34e-5 |
| 114 | 3.00e-2 | 1 | 3.33e-2 | 3.33e-2 | 3.33e-2 | 3.33e-2 | 242 | +1.17e-4 | +1.17e-4 | +1.17e-4 | -9.32e-6 |
| 115 | 3.00e-2 | 2 | 3.29e-2 | 3.30e-2 | 3.29e-2 | 3.29e-2 | 215 | -4.51e-5 | -1.81e-6 | -2.35e-5 | -1.18e-5 |
| 116 | 3.00e-2 | 1 | 3.28e-2 | 3.28e-2 | 3.28e-2 | 3.28e-2 | 284 | -2.01e-5 | -2.01e-5 | -2.01e-5 | -1.26e-5 |
| 117 | 3.00e-2 | 1 | 3.71e-2 | 3.71e-2 | 3.71e-2 | 3.71e-2 | 269 | +4.66e-4 | +4.66e-4 | +4.66e-4 | +3.52e-5 |
| 118 | 3.00e-2 | 1 | 3.68e-2 | 3.68e-2 | 3.68e-2 | 3.68e-2 | 240 | -4.10e-5 | -4.10e-5 | -4.10e-5 | +2.76e-5 |
| 119 | 3.00e-2 | 1 | 3.65e-2 | 3.65e-2 | 3.65e-2 | 3.65e-2 | 236 | -2.71e-5 | -2.71e-5 | -2.71e-5 | +2.21e-5 |
| 120 | 3.00e-2 | 1 | 3.60e-2 | 3.60e-2 | 3.60e-2 | 3.60e-2 | 252 | -5.43e-5 | -5.43e-5 | -5.43e-5 | +1.45e-5 |
| 121 | 3.00e-2 | 1 | 3.76e-2 | 3.76e-2 | 3.76e-2 | 3.76e-2 | 250 | +1.70e-4 | +1.70e-4 | +1.70e-4 | +3.00e-5 |
| 122 | 3.00e-2 | 1 | 3.75e-2 | 3.75e-2 | 3.75e-2 | 3.75e-2 | 237 | -4.67e-6 | -4.67e-6 | -4.67e-6 | +2.65e-5 |
| 123 | 3.00e-2 | 1 | 3.73e-2 | 3.73e-2 | 3.73e-2 | 3.73e-2 | 290 | -2.23e-5 | -2.23e-5 | -2.23e-5 | +2.17e-5 |
| 124 | 3.00e-2 | 2 | 4.09e-2 | 4.18e-2 | 4.14e-2 | 4.09e-2 | 199 | -1.04e-4 | +4.34e-4 | +1.65e-4 | +4.62e-5 |
| 125 | 3.00e-2 | 1 | 3.74e-2 | 3.74e-2 | 3.74e-2 | 3.74e-2 | 247 | -3.64e-4 | -3.64e-4 | -3.64e-4 | +5.14e-6 |
| 126 | 3.00e-2 | 1 | 4.19e-2 | 4.19e-2 | 4.19e-2 | 4.19e-2 | 240 | +4.72e-4 | +4.72e-4 | +4.72e-4 | +5.18e-5 |
| 127 | 3.00e-2 | 1 | 4.11e-2 | 4.11e-2 | 4.11e-2 | 4.11e-2 | 224 | -8.41e-5 | -8.41e-5 | -8.41e-5 | +3.82e-5 |
| 128 | 3.00e-2 | 2 | 4.02e-2 | 4.08e-2 | 4.05e-2 | 4.02e-2 | 186 | -7.83e-5 | -4.17e-5 | -6.00e-5 | +1.94e-5 |
| 129 | 3.00e-2 | 1 | 3.86e-2 | 3.86e-2 | 3.86e-2 | 3.86e-2 | 204 | -2.02e-4 | -2.02e-4 | -2.02e-4 | -2.71e-6 |
| 130 | 3.00e-2 | 1 | 4.09e-2 | 4.09e-2 | 4.09e-2 | 4.09e-2 | 234 | +2.57e-4 | +2.57e-4 | +2.57e-4 | +2.33e-5 |
| 131 | 3.00e-2 | 2 | 4.27e-2 | 4.30e-2 | 4.29e-2 | 4.27e-2 | 217 | -3.51e-5 | +2.27e-4 | +9.59e-5 | +3.58e-5 |
| 132 | 3.00e-2 | 1 | 4.29e-2 | 4.29e-2 | 4.29e-2 | 4.29e-2 | 246 | +2.29e-5 | +2.29e-5 | +2.29e-5 | +3.45e-5 |
| 133 | 3.00e-2 | 1 | 4.58e-2 | 4.58e-2 | 4.58e-2 | 4.58e-2 | 234 | +2.81e-4 | +2.81e-4 | +2.81e-4 | +5.91e-5 |
| 134 | 3.00e-2 | 1 | 4.59e-2 | 4.59e-2 | 4.59e-2 | 4.59e-2 | 237 | +7.18e-6 | +7.18e-6 | +7.18e-6 | +5.39e-5 |
| 135 | 3.00e-2 | 2 | 4.74e-2 | 4.83e-2 | 4.79e-2 | 4.83e-2 | 206 | +9.27e-5 | +1.33e-4 | +1.13e-4 | +6.48e-5 |
| 136 | 3.00e-2 | 1 | 4.51e-2 | 4.51e-2 | 4.51e-2 | 4.51e-2 | 198 | -3.52e-4 | -3.52e-4 | -3.52e-4 | +2.32e-5 |
| 137 | 3.00e-2 | 1 | 4.52e-2 | 4.52e-2 | 4.52e-2 | 4.52e-2 | 213 | +1.35e-5 | +1.35e-5 | +1.35e-5 | +2.22e-5 |
| 138 | 3.00e-2 | 1 | 4.74e-2 | 4.74e-2 | 4.74e-2 | 4.74e-2 | 231 | +2.06e-4 | +2.06e-4 | +2.06e-4 | +4.05e-5 |
| 139 | 3.00e-2 | 2 | 4.86e-2 | 5.03e-2 | 4.95e-2 | 5.03e-2 | 200 | +1.01e-4 | +1.66e-4 | +1.33e-4 | +5.85e-5 |
| 140 | 3.00e-2 | 1 | 4.74e-2 | 4.74e-2 | 4.74e-2 | 4.74e-2 | 240 | -2.47e-4 | -2.47e-4 | -2.47e-4 | +2.79e-5 |
| 141 | 3.00e-2 | 1 | 5.27e-2 | 5.27e-2 | 5.27e-2 | 5.27e-2 | 210 | +5.03e-4 | +5.03e-4 | +5.03e-4 | +7.55e-5 |
| 142 | 3.00e-2 | 2 | 4.84e-2 | 4.95e-2 | 4.89e-2 | 4.95e-2 | 176 | -4.06e-4 | +1.30e-4 | -1.38e-4 | +3.76e-5 |
| 143 | 3.00e-2 | 1 | 4.61e-2 | 4.61e-2 | 4.61e-2 | 4.61e-2 | 188 | -3.79e-4 | -3.79e-4 | -3.79e-4 | -4.04e-6 |
| 144 | 3.00e-2 | 1 | 4.80e-2 | 4.80e-2 | 4.80e-2 | 4.80e-2 | 191 | +2.17e-4 | +2.17e-4 | +2.17e-4 | +1.81e-5 |
| 145 | 3.00e-2 | 2 | 4.86e-2 | 5.25e-2 | 5.06e-2 | 5.25e-2 | 187 | +5.57e-5 | +4.13e-4 | +2.34e-4 | +6.10e-5 |
| 146 | 3.00e-2 | 1 | 4.91e-2 | 4.91e-2 | 4.91e-2 | 4.91e-2 | 210 | -3.26e-4 | -3.26e-4 | -3.26e-4 | +2.23e-5 |
| 147 | 3.00e-2 | 1 | 5.30e-2 | 5.30e-2 | 5.30e-2 | 5.30e-2 | 190 | +4.10e-4 | +4.10e-4 | +4.10e-4 | +6.11e-5 |
| 148 | 3.00e-2 | 2 | 5.07e-2 | 5.25e-2 | 5.16e-2 | 5.25e-2 | 187 | -2.09e-4 | +1.83e-4 | -1.26e-5 | +4.90e-5 |
| 149 | 3.00e-2 | 1 | 4.98e-2 | 4.98e-2 | 4.98e-2 | 4.98e-2 | 205 | -2.57e-4 | -2.57e-4 | -2.57e-4 | +1.84e-5 |
| 150 | 3.00e-3 | 2 | 5.31e-2 | 5.58e-2 | 5.45e-2 | 5.58e-2 | 199 | +2.53e-4 | +2.92e-4 | +2.73e-4 | +6.65e-5 |
| 151 | 3.00e-3 | 1 | 5.32e-3 | 5.32e-3 | 5.32e-3 | 5.32e-3 | 198 | -1.19e-2 | -1.19e-2 | -1.19e-2 | -1.13e-3 |
| 152 | 3.00e-3 | 1 | 5.01e-3 | 5.01e-3 | 5.01e-3 | 5.01e-3 | 208 | -2.89e-4 | -2.89e-4 | -2.89e-4 | -1.04e-3 |
| 153 | 3.00e-3 | 2 | 5.07e-3 | 5.07e-3 | 5.07e-3 | 5.07e-3 | 173 | +6.42e-6 | +5.22e-5 | +2.93e-5 | -8.40e-4 |
| 154 | 3.00e-3 | 1 | 4.71e-3 | 4.71e-3 | 4.71e-3 | 4.71e-3 | 194 | -3.86e-4 | -3.86e-4 | -3.86e-4 | -7.95e-4 |
| 155 | 3.00e-3 | 2 | 5.12e-3 | 5.12e-3 | 5.12e-3 | 5.12e-3 | 152 | -8.17e-6 | +4.15e-4 | +2.04e-4 | -6.07e-4 |
| 156 | 3.00e-3 | 1 | 4.37e-3 | 4.37e-3 | 4.37e-3 | 4.37e-3 | 190 | -8.27e-4 | -8.27e-4 | -8.27e-4 | -6.29e-4 |
| 157 | 3.00e-3 | 2 | 4.91e-3 | 5.02e-3 | 4.96e-3 | 4.91e-3 | 163 | -1.33e-4 | +7.55e-4 | +3.11e-4 | -4.55e-4 |
| 158 | 3.00e-3 | 2 | 4.80e-3 | 5.10e-3 | 4.95e-3 | 5.10e-3 | 144 | -1.10e-4 | +4.11e-4 | +1.50e-4 | -3.37e-4 |
| 159 | 3.00e-3 | 1 | 4.54e-3 | 4.54e-3 | 4.54e-3 | 4.54e-3 | 191 | -6.02e-4 | -6.02e-4 | -6.02e-4 | -3.64e-4 |
| 160 | 3.00e-3 | 2 | 4.98e-3 | 5.01e-3 | 4.99e-3 | 4.98e-3 | 130 | -5.34e-5 | +5.20e-4 | +2.33e-4 | -2.53e-4 |
| 161 | 3.00e-3 | 2 | 4.25e-3 | 5.08e-3 | 4.66e-3 | 5.08e-3 | 128 | -8.49e-4 | +1.39e-3 | +2.69e-4 | -1.43e-4 |
| 162 | 3.00e-3 | 2 | 4.40e-3 | 4.54e-3 | 4.47e-3 | 4.54e-3 | 128 | -9.35e-4 | +2.40e-4 | -3.48e-4 | -1.76e-4 |
| 163 | 3.00e-3 | 2 | 4.37e-3 | 4.66e-3 | 4.52e-3 | 4.66e-3 | 128 | -2.38e-4 | +5.06e-4 | +1.34e-4 | -1.13e-4 |
| 164 | 3.00e-3 | 2 | 4.30e-3 | 4.73e-3 | 4.51e-3 | 4.73e-3 | 128 | -5.13e-4 | +7.40e-4 | +1.14e-4 | -6.39e-5 |
| 165 | 3.00e-3 | 2 | 4.48e-3 | 4.98e-3 | 4.73e-3 | 4.98e-3 | 128 | -3.00e-4 | +8.16e-4 | +2.58e-4 | +2.77e-6 |
| 166 | 3.00e-3 | 2 | 4.31e-3 | 4.90e-3 | 4.61e-3 | 4.90e-3 | 128 | -9.55e-4 | +1.00e-3 | +2.36e-5 | +1.65e-5 |
| 167 | 3.00e-3 | 2 | 4.24e-3 | 4.71e-3 | 4.47e-3 | 4.71e-3 | 128 | -9.20e-4 | +8.16e-4 | -5.19e-5 | +1.22e-5 |
| 168 | 3.00e-3 | 2 | 4.49e-3 | 4.81e-3 | 4.65e-3 | 4.81e-3 | 128 | -2.91e-4 | +5.41e-4 | +1.25e-4 | +3.78e-5 |
| 169 | 3.00e-3 | 2 | 4.38e-3 | 4.48e-3 | 4.43e-3 | 4.48e-3 | 131 | -6.88e-4 | +1.69e-4 | -2.60e-4 | -1.44e-5 |
| 170 | 3.00e-3 | 2 | 4.64e-3 | 5.29e-3 | 4.96e-3 | 5.29e-3 | 120 | +1.94e-4 | +1.09e-3 | +6.40e-4 | +1.14e-4 |
| 171 | 3.00e-3 | 2 | 4.37e-3 | 5.08e-3 | 4.73e-3 | 5.08e-3 | 126 | -1.18e-3 | +1.19e-3 | +2.67e-6 | +1.05e-4 |
| 172 | 3.00e-3 | 2 | 4.49e-3 | 4.90e-3 | 4.69e-3 | 4.90e-3 | 104 | -7.82e-4 | +8.31e-4 | +2.47e-5 | +9.78e-5 |
| 173 | 3.00e-3 | 3 | 3.87e-3 | 5.04e-3 | 4.40e-3 | 3.87e-3 | 90 | -2.92e-3 | +1.67e-3 | -6.70e-4 | -1.32e-4 |
| 174 | 3.00e-3 | 2 | 3.85e-3 | 4.43e-3 | 4.14e-3 | 4.43e-3 | 96 | -5.56e-5 | +1.47e-3 | +7.09e-4 | +3.54e-5 |
| 175 | 3.00e-3 | 2 | 4.16e-3 | 4.64e-3 | 4.40e-3 | 4.64e-3 | 96 | -4.93e-4 | +1.15e-3 | +3.28e-4 | +9.93e-5 |
| 176 | 3.00e-3 | 3 | 3.86e-3 | 4.64e-3 | 4.13e-3 | 3.88e-3 | 102 | -1.77e-3 | +1.81e-3 | -4.28e-4 | -4.89e-5 |
| 177 | 3.00e-3 | 2 | 4.24e-3 | 4.67e-3 | 4.46e-3 | 4.67e-3 | 95 | +6.33e-4 | +1.04e-3 | +8.34e-4 | +1.21e-4 |
| 178 | 3.00e-3 | 3 | 4.00e-3 | 4.40e-3 | 4.14e-3 | 4.03e-3 | 86 | -1.15e-3 | +1.02e-3 | -3.89e-4 | -1.71e-5 |
| 179 | 3.00e-3 | 3 | 3.83e-3 | 4.42e-3 | 4.06e-3 | 3.83e-3 | 85 | -1.68e-3 | +1.39e-3 | -1.71e-4 | -7.34e-5 |
| 180 | 3.00e-3 | 4 | 3.76e-3 | 4.62e-3 | 4.00e-3 | 3.79e-3 | 89 | -2.16e-3 | +2.42e-3 | -1.33e-5 | -7.38e-5 |
| 181 | 3.00e-3 | 2 | 3.87e-3 | 4.63e-3 | 4.25e-3 | 4.63e-3 | 83 | +1.69e-4 | +2.16e-3 | +1.16e-3 | +1.71e-4 |
| 182 | 3.00e-3 | 3 | 3.51e-3 | 4.24e-3 | 3.85e-3 | 3.51e-3 | 64 | -2.93e-3 | +1.51e-3 | -1.03e-3 | -1.67e-4 |
| 183 | 3.00e-3 | 4 | 3.07e-3 | 4.20e-3 | 3.56e-3 | 3.62e-3 | 76 | -3.08e-3 | +4.92e-3 | +3.24e-4 | -1.94e-6 |
| 184 | 3.00e-3 | 4 | 3.19e-3 | 4.18e-3 | 3.63e-3 | 3.19e-3 | 60 | -2.72e-3 | +2.02e-3 | -6.42e-4 | -2.68e-4 |
| 185 | 3.00e-3 | 5 | 2.85e-3 | 3.82e-3 | 3.12e-3 | 2.90e-3 | 54 | -4.82e-3 | +4.01e-3 | -3.16e-4 | -3.15e-4 |
| 186 | 3.00e-3 | 4 | 2.98e-3 | 4.05e-3 | 3.35e-3 | 3.08e-3 | 60 | -3.34e-3 | +5.12e-3 | +2.10e-4 | -1.94e-4 |
| 187 | 3.00e-3 | 5 | 2.88e-3 | 4.11e-3 | 3.23e-3 | 3.06e-3 | 44 | -6.60e-3 | +4.60e-3 | -5.42e-5 | -1.55e-4 |
| 188 | 3.00e-3 | 6 | 2.39e-3 | 3.45e-3 | 2.75e-3 | 2.39e-3 | 31 | -7.11e-3 | +4.98e-3 | -1.06e-3 | -6.52e-4 |
| 189 | 3.00e-3 | 10 | 1.81e-3 | 3.33e-3 | 2.12e-3 | 1.92e-3 | 22 | -1.64e-2 | +1.48e-2 | -8.79e-4 | -7.56e-4 |
| 190 | 3.00e-3 | 13 | 1.46e-3 | 3.12e-3 | 1.72e-3 | 1.63e-3 | 15 | -3.80e-2 | +2.77e-2 | -5.03e-4 | -4.13e-4 |
| 191 | 3.00e-3 | 15 | 1.21e-3 | 3.19e-3 | 1.55e-3 | 1.21e-3 | 14 | -4.53e-2 | +5.15e-2 | -1.16e-3 | -1.62e-3 |
| 192 | 3.00e-3 | 13 | 1.23e-3 | 3.05e-3 | 1.72e-3 | 1.44e-3 | 22 | -3.62e-2 | +4.55e-2 | +5.78e-4 | -7.63e-4 |
| 193 | 3.00e-3 | 9 | 1.68e-3 | 3.42e-3 | 2.10e-3 | 1.88e-3 | 20 | -1.52e-2 | +2.46e-2 | +5.11e-4 | -2.86e-4 |
| 194 | 3.00e-3 | 18 | 1.21e-3 | 3.03e-3 | 1.58e-3 | 1.45e-3 | 20 | -2.82e-2 | +2.77e-2 | -5.77e-4 | -2.84e-4 |
| 195 | 3.00e-3 | 2 | 1.49e-3 | 1.68e-3 | 1.58e-3 | 1.68e-3 | 20 | +1.38e-3 | +5.98e-3 | +3.68e-3 | +4.93e-4 |
| 196 | 3.00e-3 | 2 | 1.43e-3 | 6.52e-3 | 3.98e-3 | 6.52e-3 | 221 | -6.55e-4 | +6.88e-3 | +3.11e-3 | +1.03e-3 |
| 197 | 3.00e-3 | 1 | 6.33e-3 | 6.33e-3 | 6.33e-3 | 6.33e-3 | 250 | -1.20e-4 | -1.20e-4 | -1.20e-4 | +9.13e-4 |
| 198 | 3.00e-3 | 1 | 6.62e-3 | 6.62e-3 | 6.62e-3 | 6.62e-3 | 226 | +1.99e-4 | +1.99e-4 | +1.99e-4 | +8.42e-4 |
| 199 | 3.00e-3 | 1 | 6.33e-3 | 6.33e-3 | 6.33e-3 | 6.33e-3 | 230 | -1.99e-4 | -1.99e-4 | -1.99e-4 | +7.38e-4 |

