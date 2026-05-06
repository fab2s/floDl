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
| nccl-async | 0.044022 | 0.9161 | +0.0036 | 1909.7 | 767 | 39.2 | 100% | 100% | 9.5 |

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
| nccl-async | 2.0353 | 0.7800 | 0.6143 | 0.5613 | 0.5271 | 0.5112 | 0.4804 | 0.4568 | 0.4637 | 0.4505 | 0.1946 | 0.1553 | 0.1258 | 0.1143 | 0.1004 | 0.0592 | 0.0547 | 0.0494 | 0.0475 | 0.0440 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | nccl-async | 0 | cuda:0 | 0.3966 | 2.6 |
| resnet-graph | nccl-async | 1 | cuda:1 | 0.3065 | 3.4 |
| resnet-graph | nccl-async | 2 | cuda:2 | 0.2969 | 3.4 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|
| resnet-graph | nccl-async | 356 | 355 | 396 | 384 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| nccl-async | gpu1 | 1906.7 | 3.0 | epoch-boundary(199) |
| nccl-async | gpu2 | 1906.7 | 3.0 | epoch-boundary(199) |
| nccl-async | gpu0 | 1908.7 | 0.5 | epoch-boundary(199) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | nccl-async | gpu0 | 0.5s | 0.0s | 0.0s | 0.0s | 1.6s |
| resnet-graph | nccl-async | gpu1 | 3.0s | 0.0s | 0.0s | 0.0s | 4.2s |
| resnet-graph | nccl-async | gpu2 | 3.0s | 0.0s | 0.0s | 0.0s | 3.6s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | nccl-async | 377 | 0 | 767 | 39.2 | 680/9059 | 0 | 0.0 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | nccl-async | 185.8 | 9.7% |

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
| resnet-graph | nccl-async | 197 | 767 | 0 | 5.90e-3 | +4.41e-5 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | nccl-async | 0 | 767 | 4.36e-2 | 6.53e-2 | 0.00e0 | 4.87e-1 | 42.9 | -1.81e-4 | 4.23e-3 |
| resnet-graph | nccl-async | 1 | 767 | 4.46e-2 | 6.69e-2 | 0.00e0 | 4.63e-1 | 38.5 | -1.88e-4 | 6.13e-3 |
| resnet-graph | nccl-async | 2 | 767 | 4.39e-2 | 6.57e-2 | 0.00e0 | 3.83e-1 | 18.6 | -1.75e-4 | 6.17e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | nccl-async | rank0 ↔ rank1 | +0.9988 |
| resnet-graph | nccl-async | rank0 ↔ rank2 | +0.9970 |
| resnet-graph | nccl-async | rank1 ↔ rank2 | +0.9988 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | nccl-async | 51 (0,1,2,3,14,17,22,29…149,150) | 2 (135,148) | 148 | 0,1,2,3,14,17,22,29…149,150 | 135 |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | nccl-async | 0e0 | 3 | 47 | 47 |
| resnet-graph | nccl-async | 0e0 | 5 | 25 | 25 |
| resnet-graph | nccl-async | 0e0 | 10 | 9 | 9 |
| resnet-graph | nccl-async | 1e-4 | 3 | 20 | 20 |
| resnet-graph | nccl-async | 1e-4 | 5 | 11 | 11 |
| resnet-graph | nccl-async | 1e-4 | 10 | 3 | 3 |
| resnet-graph | nccl-async | 1e-3 | 3 | 2 | 2 |
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
| resnet-graph | nccl-async | 3.00e-1 | 0–98 | 170 | +0.189 |
| resnet-graph | nccl-async | 3.00e-2 | 99–149 | 498 | +0.024 |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | 94 | +0.094 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | nccl-async | 764 | +0.017 | 196 | +0.217 | +0.160 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | nccl-async | 765 | 3.38e1–7.93e1 | 6.44e1 | 1.63e-3 | 6.29e-3 | 1.11e-1 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | nccl-async | 3.00e-1 | 0–98 | 172 | 64–76965 | +1.269e-5 | 0.416 | +1.299e-5 | 0.421 | 97 | +2.848e-6 | 0.065 | 29–960 | +1.475e-3 | 0.673 |
| resnet-graph | nccl-async | 3.00e-1 | 1–98 (post-transient, skipped 1) | 157 | 870–76965 | +1.128e-5 | 0.425 | +1.151e-5 | 0.425 | 96 | +2.102e-6 | 0.041 | 42–960 | +1.412e-3 | 0.751 |
| resnet-graph | nccl-async | 3.00e-2 | 99–149 | 499 | 77421–117102 | -4.818e-8 | 0.000 | -4.701e-7 | 0.000 | 51 | -1.564e-5 | 0.144 | 32–456 | +2.649e-3 | 0.254 |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | 95 | 117237–156038 | +3.344e-5 | 0.367 | +3.434e-5 | 0.376 | 49 | +9.390e-6 | 0.107 | 35–739 | +2.273e-3 | 0.630 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | nccl-async | 3.00e-1 | 0–98 | +1.475e-3 | r0: +1.454e-3, r1: +1.479e-3, r2: +1.494e-3 | r0: 0.671, r1: 0.667, r2: 0.678 | 1.03× | ✓ |
| resnet-graph | nccl-async | 3.00e-1 | 1–98 (post-transient, skipped 1) | +1.412e-3 | r0: +1.392e-3, r1: +1.423e-3, r2: +1.425e-3 | r0: 0.765, r1: 0.744, r2: 0.739 | 1.02× | ✓ |
| resnet-graph | nccl-async | 3.00e-2 | 99–149 | +2.649e-3 | r0: +2.619e-3, r1: +2.652e-3, r2: +2.693e-3 | r0: 0.290, r1: 0.232, r2: 0.238 | 1.03× | ✓ |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | +2.273e-3 | r0: +2.198e-3, r1: +2.310e-3, r2: +2.321e-3 | r0: 0.633, r1: 0.626, r2: 0.631 | 1.06× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | nccl-async | `█▇██████████████████████▇▃▄▄▄▄▄▄▄▄▄▄▄▁▁▁▁▁▁▁▁▁▁▁▁▁` | `▁███▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▅▄▆▇▆▆▅▆▆█▆▇█▃▇▇▇▇▇▇▇▇▇▇▇▇` |

### Per-Epoch Detail: resnet-graph / nccl-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 16 | 0.00e0 | 4.87e-1 | 9.19e-2 | 5.72e-2 | 20 | -7.21e-2 | +3.05e-2 | -1.43e-2 | -7.02e-3 |
| 1 | 3.00e-1 | 16 | 5.00e-2 | 1.34e-1 | 6.41e-2 | 6.98e-2 | 21 | -2.84e-2 | +3.77e-2 | +1.75e-4 | -4.07e-4 |
| 2 | 3.00e-1 | 10 | 6.83e-2 | 1.14e-1 | 7.56e-2 | 6.96e-2 | 16 | -2.35e-2 | +2.21e-2 | -2.12e-4 | -4.64e-4 |
| 3 | 3.00e-1 | 14 | 6.46e-2 | 1.31e-1 | 7.90e-2 | 8.08e-2 | 18 | -3.51e-2 | +3.91e-2 | +6.64e-4 | +3.14e-4 |
| 4 | 3.00e-1 | 1 | 7.72e-2 | 7.72e-2 | 7.72e-2 | 7.72e-2 | 18 | -2.48e-3 | -2.48e-3 | -2.48e-3 | +3.35e-5 |
| 5 | 3.00e-1 | 1 | 7.35e-2 | 7.35e-2 | 7.35e-2 | 7.35e-2 | 308 | -1.62e-4 | -1.62e-4 | -1.62e-4 | +1.39e-5 |
| 6 | 3.00e-1 | 1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 315 | +3.57e-3 | +3.57e-3 | +3.57e-3 | +3.70e-4 |
| 7 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 323 | -2.50e-4 | -2.50e-4 | -2.50e-4 | +3.08e-4 |
| 8 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 322 | +6.29e-6 | +6.29e-6 | +6.29e-6 | +2.78e-4 |
| 9 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 302 | -1.05e-4 | -1.05e-4 | -1.05e-4 | +2.40e-4 |
| 10 | 3.00e-1 | 1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 298 | -8.82e-5 | -8.82e-5 | -8.82e-5 | +2.07e-4 |
| 11 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 310 | -2.45e-5 | -2.45e-5 | -2.45e-5 | +1.84e-4 |
| 13 | 3.00e-1 | 1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 365 | +1.37e-5 | +1.37e-5 | +1.37e-5 | +1.67e-4 |
| 14 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 307 | +1.14e-4 | +1.14e-4 | +1.14e-4 | +1.61e-4 |
| 15 | 3.00e-1 | 1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 303 | -1.41e-4 | -1.41e-4 | -1.41e-4 | +1.31e-4 |
| 16 | 3.00e-1 | 1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 304 | +1.89e-5 | +1.89e-5 | +1.89e-5 | +1.20e-4 |
| 17 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 286 | +2.96e-5 | +2.96e-5 | +2.96e-5 | +1.11e-4 |
| 18 | 3.00e-1 | 1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 287 | -6.93e-5 | -6.93e-5 | -6.93e-5 | +9.28e-5 |
| 19 | 3.00e-1 | 1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 257 | -5.28e-5 | -5.28e-5 | -5.28e-5 | +7.83e-5 |
| 20 | 3.00e-1 | 1 | 1.88e-1 | 1.88e-1 | 1.88e-1 | 1.88e-1 | 275 | -7.33e-5 | -7.33e-5 | -7.33e-5 | +6.31e-5 |
| 21 | 3.00e-1 | 1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 283 | +8.93e-5 | +8.93e-5 | +8.93e-5 | +6.57e-5 |
| 22 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 280 | +6.04e-5 | +6.04e-5 | +6.04e-5 | +6.52e-5 |
| 23 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 301 | -2.49e-6 | -2.49e-6 | -2.49e-6 | +5.84e-5 |
| 24 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 296 | +1.05e-4 | +1.05e-4 | +1.05e-4 | +6.31e-5 |
| 25 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 295 | -3.61e-5 | -3.61e-5 | -3.61e-5 | +5.31e-5 |
| 26 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 277 | -3.39e-5 | -3.39e-5 | -3.39e-5 | +4.44e-5 |
| 27 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 289 | -3.79e-5 | -3.79e-5 | -3.79e-5 | +3.62e-5 |
| 28 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 296 | +5.85e-5 | +5.85e-5 | +5.85e-5 | +3.84e-5 |
| 29 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 290 | +4.05e-5 | +4.05e-5 | +4.05e-5 | +3.87e-5 |
| 30 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 258 | -7.17e-6 | -7.17e-6 | -7.17e-6 | +3.41e-5 |
| 31 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 284 | -9.49e-5 | -9.49e-5 | -9.49e-5 | +2.12e-5 |
| 32 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 295 | +1.37e-4 | +1.37e-4 | +1.37e-4 | +3.28e-5 |
| 33 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 272 | -4.40e-5 | -4.40e-5 | -4.40e-5 | +2.51e-5 |
| 34 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 280 | -5.07e-5 | -5.07e-5 | -5.07e-5 | +1.75e-5 |
| 35 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 300 | +1.57e-5 | +1.57e-5 | +1.57e-5 | +1.73e-5 |
| 36 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 269 | +6.64e-5 | +6.64e-5 | +6.64e-5 | +2.22e-5 |
| 37 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 273 | -8.46e-5 | -8.46e-5 | -8.46e-5 | +1.16e-5 |
| 38 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 269 | +8.18e-6 | +8.18e-6 | +8.18e-6 | +1.12e-5 |
| 39 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 246 | -1.69e-5 | -1.69e-5 | -1.69e-5 | +8.40e-6 |
| 40 | 3.00e-1 | 1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 263 | -9.88e-5 | -9.88e-5 | -9.88e-5 | -2.32e-6 |
| 41 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 267 | +1.26e-4 | +1.26e-4 | +1.26e-4 | +1.05e-5 |
| 42 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 250 | +5.55e-5 | +5.55e-5 | +5.55e-5 | +1.50e-5 |
| 43 | 3.00e-1 | 1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 262 | -1.56e-4 | -1.56e-4 | -1.56e-4 | -2.09e-6 |
| 44 | 3.00e-1 | 2 | 1.96e-1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 216 | +2.87e-5 | +3.83e-5 | +3.35e-5 | +4.72e-6 |
| 45 | 3.00e-1 | 1 | 1.89e-1 | 1.89e-1 | 1.89e-1 | 1.89e-1 | 254 | -1.77e-4 | -1.77e-4 | -1.77e-4 | -1.34e-5 |
| 46 | 3.00e-1 | 1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 264 | +1.66e-4 | +1.66e-4 | +1.66e-4 | +4.54e-6 |
| 47 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 265 | +6.89e-5 | +6.89e-5 | +6.89e-5 | +1.10e-5 |
| 48 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 263 | -3.36e-5 | -3.36e-5 | -3.36e-5 | +6.52e-6 |
| 49 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 247 | -6.97e-6 | -6.97e-6 | -6.97e-6 | +5.17e-6 |
| 50 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 252 | +6.03e-6 | +6.03e-6 | +6.03e-6 | +5.26e-6 |
| 51 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 277 | +1.83e-5 | +1.83e-5 | +1.83e-5 | +6.56e-6 |
| 52 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 252 | +7.35e-5 | +7.35e-5 | +7.35e-5 | +1.33e-5 |
| 53 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 253 | -6.21e-5 | -6.21e-5 | -6.21e-5 | +5.72e-6 |
| 54 | 3.00e-1 | 2 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 204 | -1.07e-5 | +1.23e-6 | -4.71e-6 | +3.80e-6 |
| 55 | 3.00e-1 | 1 | 1.85e-1 | 1.85e-1 | 1.85e-1 | 1.85e-1 | 244 | -3.27e-4 | -3.27e-4 | -3.27e-4 | -2.92e-5 |
| 56 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 245 | +3.28e-4 | +3.28e-4 | +3.28e-4 | +6.49e-6 |
| 57 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 242 | -2.09e-5 | -2.09e-5 | -2.09e-5 | +3.76e-6 |
| 58 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 240 | -6.88e-5 | -6.88e-5 | -6.88e-5 | -3.50e-6 |
| 59 | 3.00e-1 | 2 | 1.94e-1 | 1.95e-1 | 1.95e-1 | 1.94e-1 | 192 | -2.89e-5 | -1.87e-5 | -2.38e-5 | -7.40e-6 |
| 60 | 3.00e-1 | 1 | 1.82e-1 | 1.82e-1 | 1.82e-1 | 1.82e-1 | 220 | -2.86e-4 | -2.86e-4 | -2.86e-4 | -3.52e-5 |
| 61 | 3.00e-1 | 1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 228 | +2.78e-4 | +2.78e-4 | +2.78e-4 | -3.94e-6 |
| 62 | 3.00e-1 | 1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 240 | -2.98e-5 | -2.98e-5 | -2.98e-5 | -6.53e-6 |
| 63 | 3.00e-1 | 2 | 2.00e-1 | 2.02e-1 | 2.01e-1 | 2.00e-1 | 204 | -4.23e-5 | +1.77e-4 | +6.75e-5 | +6.43e-6 |
| 64 | 3.00e-1 | 1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 230 | -2.27e-4 | -2.27e-4 | -2.27e-4 | -1.69e-5 |
| 65 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 222 | +1.50e-4 | +1.50e-4 | +1.50e-4 | -2.67e-7 |
| 66 | 3.00e-1 | 2 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 180 | -6.19e-5 | -4.36e-6 | -3.31e-5 | -6.23e-6 |
| 67 | 3.00e-1 | 1 | 1.78e-1 | 1.78e-1 | 1.78e-1 | 1.78e-1 | 221 | -3.66e-4 | -3.66e-4 | -3.66e-4 | -4.22e-5 |
| 68 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 213 | +4.55e-4 | +4.55e-4 | +4.55e-4 | +7.51e-6 |
| 69 | 3.00e-1 | 2 | 1.92e-1 | 1.97e-1 | 1.94e-1 | 1.97e-1 | 180 | -1.01e-4 | +1.42e-4 | +2.06e-5 | +1.12e-5 |
| 70 | 3.00e-1 | 1 | 1.79e-1 | 1.79e-1 | 1.79e-1 | 1.79e-1 | 234 | -4.16e-4 | -4.16e-4 | -4.16e-4 | -3.15e-5 |
| 71 | 3.00e-1 | 2 | 1.89e-1 | 1.98e-1 | 1.94e-1 | 1.89e-1 | 180 | -2.49e-4 | +5.24e-4 | +1.37e-4 | -3.27e-6 |
| 73 | 3.00e-1 | 2 | 1.82e-1 | 2.08e-1 | 1.95e-1 | 2.08e-1 | 180 | -1.48e-4 | +7.56e-4 | +3.04e-4 | +5.96e-5 |
| 74 | 3.00e-1 | 2 | 1.83e-1 | 1.91e-1 | 1.87e-1 | 1.91e-1 | 168 | -6.07e-4 | +2.54e-4 | -1.77e-4 | +1.90e-5 |
| 75 | 3.00e-1 | 1 | 1.76e-1 | 1.76e-1 | 1.76e-1 | 1.76e-1 | 210 | -3.93e-4 | -3.93e-4 | -3.93e-4 | -2.21e-5 |
| 76 | 3.00e-1 | 1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 220 | +4.44e-4 | +4.44e-4 | +4.44e-4 | +2.44e-5 |
| 77 | 3.00e-1 | 2 | 1.94e-1 | 1.99e-1 | 1.96e-1 | 1.99e-1 | 168 | -4.79e-6 | +1.57e-4 | +7.61e-5 | +3.51e-5 |
| 78 | 3.00e-1 | 1 | 1.81e-1 | 1.81e-1 | 1.81e-1 | 1.81e-1 | 239 | -3.97e-4 | -3.97e-4 | -3.97e-4 | -8.17e-6 |
| 79 | 3.00e-1 | 2 | 1.93e-1 | 2.02e-1 | 1.98e-1 | 1.93e-1 | 155 | -3.04e-4 | +5.34e-4 | +1.15e-4 | +1.11e-5 |
| 80 | 3.00e-1 | 1 | 1.76e-1 | 1.76e-1 | 1.76e-1 | 1.76e-1 | 190 | -4.75e-4 | -4.75e-4 | -4.75e-4 | -3.76e-5 |
| 81 | 3.00e-1 | 2 | 1.79e-1 | 1.88e-1 | 1.84e-1 | 1.79e-1 | 139 | -3.78e-4 | +3.69e-4 | -4.49e-6 | -3.50e-5 |
| 82 | 3.00e-1 | 2 | 1.71e-1 | 1.93e-1 | 1.82e-1 | 1.93e-1 | 149 | -2.36e-4 | +8.38e-4 | +3.01e-4 | +3.42e-5 |
| 83 | 3.00e-1 | 1 | 1.72e-1 | 1.72e-1 | 1.72e-1 | 1.72e-1 | 194 | -6.06e-4 | -6.06e-4 | -6.06e-4 | -2.99e-5 |
| 84 | 3.00e-1 | 1 | 1.88e-1 | 1.88e-1 | 1.88e-1 | 1.88e-1 | 194 | +4.75e-4 | +4.75e-4 | +4.75e-4 | +2.06e-5 |
| 85 | 3.00e-1 | 3 | 1.72e-1 | 1.92e-1 | 1.83e-1 | 1.72e-1 | 147 | -7.28e-4 | +2.28e-4 | -1.94e-4 | -4.39e-5 |
| 86 | 3.00e-1 | 1 | 1.71e-1 | 1.71e-1 | 1.71e-1 | 1.71e-1 | 175 | -4.17e-5 | -4.17e-5 | -4.17e-5 | -4.37e-5 |
| 87 | 3.00e-1 | 1 | 1.86e-1 | 1.86e-1 | 1.86e-1 | 1.86e-1 | 187 | +4.41e-4 | +4.41e-4 | +4.41e-4 | +4.84e-6 |
| 88 | 3.00e-1 | 2 | 1.93e-1 | 1.94e-1 | 1.93e-1 | 1.94e-1 | 149 | +2.45e-5 | +1.82e-4 | +1.03e-4 | +2.27e-5 |
| 89 | 3.00e-1 | 2 | 1.74e-1 | 1.87e-1 | 1.81e-1 | 1.87e-1 | 137 | -5.50e-4 | +5.44e-4 | -2.66e-6 | +2.34e-5 |
| 90 | 3.00e-1 | 2 | 1.73e-1 | 1.82e-1 | 1.78e-1 | 1.82e-1 | 130 | -4.51e-4 | +3.72e-4 | -3.95e-5 | +1.55e-5 |
| 91 | 3.00e-1 | 1 | 1.68e-1 | 1.68e-1 | 1.68e-1 | 1.68e-1 | 168 | -4.84e-4 | -4.84e-4 | -4.84e-4 | -3.44e-5 |
| 92 | 3.00e-1 | 2 | 1.76e-1 | 1.88e-1 | 1.82e-1 | 1.88e-1 | 130 | +2.63e-4 | +4.83e-4 | +3.73e-4 | +4.40e-5 |
| 93 | 3.00e-1 | 2 | 1.65e-1 | 1.89e-1 | 1.77e-1 | 1.89e-1 | 130 | -6.71e-4 | +1.06e-3 | +1.96e-4 | +8.16e-5 |
| 94 | 3.00e-1 | 2 | 1.66e-1 | 1.86e-1 | 1.76e-1 | 1.86e-1 | 135 | -7.11e-4 | +8.24e-4 | +5.67e-5 | +8.46e-5 |
| 95 | 3.00e-1 | 2 | 1.69e-1 | 1.88e-1 | 1.79e-1 | 1.88e-1 | 122 | -5.30e-4 | +8.58e-4 | +1.64e-4 | +1.07e-4 |
| 96 | 3.00e-1 | 2 | 1.61e-1 | 1.83e-1 | 1.72e-1 | 1.83e-1 | 122 | -9.12e-4 | +1.07e-3 | +8.01e-5 | +1.11e-4 |
| 97 | 3.00e-1 | 2 | 1.62e-1 | 1.85e-1 | 1.73e-1 | 1.85e-1 | 122 | -7.23e-4 | +1.10e-3 | +1.90e-4 | +1.36e-4 |
| 98 | 3.00e-1 | 1 | 1.60e-1 | 1.60e-1 | 1.60e-1 | 1.60e-1 | 171 | -8.54e-4 | -8.54e-4 | -8.54e-4 | +3.66e-5 |
| 99 | 3.00e-2 | 3 | 1.59e-1 | 1.84e-1 | 1.75e-1 | 1.59e-1 | 115 | -1.27e-3 | +7.88e-4 | -1.56e-4 | -3.52e-5 |
| 100 | 3.00e-2 | 2 | 3.73e-2 | 1.60e-1 | 9.88e-2 | 3.73e-2 | 107 | -1.36e-2 | +5.18e-5 | -6.78e-3 | -1.39e-3 |
| 101 | 3.00e-2 | 2 | 1.48e-2 | 1.74e-2 | 1.61e-2 | 1.74e-2 | 107 | -6.19e-3 | +1.51e-3 | -2.34e-3 | -1.53e-3 |
| 102 | 3.00e-2 | 2 | 1.61e-2 | 1.88e-2 | 1.75e-2 | 1.88e-2 | 106 | -5.25e-4 | +1.47e-3 | +4.73e-4 | -1.14e-3 |
| 103 | 3.00e-2 | 3 | 1.64e-2 | 1.92e-2 | 1.73e-2 | 1.64e-2 | 106 | -1.51e-3 | +1.55e-3 | -3.04e-4 | -9.19e-4 |
| 104 | 3.00e-2 | 2 | 1.78e-2 | 1.94e-2 | 1.86e-2 | 1.94e-2 | 107 | +5.95e-4 | +8.46e-4 | +7.20e-4 | -6.06e-4 |
| 105 | 3.00e-2 | 3 | 1.65e-2 | 1.92e-2 | 1.79e-2 | 1.65e-2 | 88 | -1.73e-3 | +7.21e-4 | -5.38e-4 | -5.99e-4 |
| 106 | 3.00e-2 | 3 | 1.69e-2 | 2.04e-2 | 1.82e-2 | 1.74e-2 | 88 | -1.76e-3 | +2.13e-3 | +1.77e-4 | -4.07e-4 |
| 107 | 3.00e-2 | 3 | 1.74e-2 | 1.97e-2 | 1.85e-2 | 1.83e-2 | 88 | -8.48e-4 | +1.41e-3 | +1.85e-4 | -2.56e-4 |
| 108 | 3.00e-2 | 2 | 1.76e-2 | 2.23e-2 | 2.00e-2 | 2.23e-2 | 84 | -2.57e-4 | +2.81e-3 | +1.28e-3 | +5.06e-5 |
| 109 | 3.00e-2 | 3 | 1.70e-2 | 2.12e-2 | 1.87e-2 | 1.70e-2 | 79 | -2.76e-3 | +2.21e-3 | -7.58e-4 | -1.80e-4 |
| 110 | 3.00e-2 | 4 | 1.65e-2 | 2.15e-2 | 1.82e-2 | 1.67e-2 | 65 | -4.08e-3 | +2.55e-3 | -1.97e-4 | -2.20e-4 |
| 111 | 3.00e-2 | 4 | 1.68e-2 | 2.07e-2 | 1.82e-2 | 1.83e-2 | 69 | -3.06e-3 | +3.13e-3 | +3.75e-4 | -2.62e-5 |
| 112 | 3.00e-2 | 3 | 1.76e-2 | 2.22e-2 | 1.94e-2 | 1.84e-2 | 76 | -2.48e-3 | +3.26e-3 | +1.50e-4 | -2.90e-7 |
| 113 | 3.00e-2 | 4 | 1.79e-2 | 2.21e-2 | 1.93e-2 | 1.83e-2 | 59 | -3.21e-3 | +2.33e-3 | -4.55e-5 | -3.92e-5 |
| 114 | 3.00e-2 | 5 | 1.62e-2 | 2.22e-2 | 1.76e-2 | 1.62e-2 | 51 | -5.29e-3 | +5.29e-3 | -2.60e-4 | -1.63e-4 |
| 115 | 3.00e-2 | 7 | 1.55e-2 | 2.21e-2 | 1.70e-2 | 1.58e-2 | 46 | -6.97e-3 | +5.14e-3 | -1.41e-4 | -1.93e-4 |
| 116 | 3.00e-2 | 4 | 1.39e-2 | 2.21e-2 | 1.65e-2 | 1.43e-2 | 36 | -1.22e-2 | +8.88e-3 | -6.29e-4 | -4.24e-4 |
| 117 | 3.00e-2 | 11 | 1.15e-2 | 1.93e-2 | 1.34e-2 | 1.21e-2 | 23 | -1.14e-2 | +8.57e-3 | -5.43e-4 | -5.15e-4 |
| 118 | 3.00e-2 | 10 | 7.96e-3 | 2.04e-2 | 1.02e-2 | 9.66e-3 | 15 | -3.68e-2 | +3.26e-2 | -1.01e-3 | -3.74e-4 |
| 119 | 3.00e-2 | 12 | 7.50e-3 | 1.77e-2 | 1.06e-2 | 1.31e-2 | 26 | -5.33e-2 | +5.35e-2 | +1.09e-3 | +7.51e-4 |
| 120 | 3.00e-2 | 12 | 9.56e-3 | 2.02e-2 | 1.16e-2 | 9.62e-3 | 17 | -2.98e-2 | +2.21e-2 | -1.11e-3 | -7.94e-4 |
| 121 | 3.00e-2 | 13 | 8.10e-3 | 2.10e-2 | 1.02e-2 | 8.82e-3 | 18 | -4.52e-2 | +5.55e-2 | +3.71e-4 | -5.06e-4 |
| 122 | 3.00e-2 | 18 | 8.26e-3 | 2.08e-2 | 1.04e-2 | 1.07e-2 | 18 | -5.48e-2 | +4.90e-2 | +1.29e-4 | +8.78e-5 |
| 123 | 3.00e-2 | 10 | 9.47e-3 | 2.01e-2 | 1.13e-2 | 1.08e-2 | 18 | -3.76e-2 | +3.76e-2 | +2.54e-4 | +1.48e-4 |
| 124 | 3.00e-2 | 17 | 7.58e-3 | 1.93e-2 | 1.00e-2 | 7.58e-3 | 12 | -4.33e-2 | +3.39e-2 | -1.99e-3 | -2.45e-3 |
| 125 | 3.00e-2 | 16 | 8.03e-3 | 2.31e-2 | 1.03e-2 | 9.84e-3 | 15 | -7.38e-2 | +7.55e-2 | +4.46e-4 | -5.33e-4 |
| 126 | 3.00e-2 | 16 | 8.58e-3 | 2.17e-2 | 1.08e-2 | 1.06e-2 | 17 | -4.52e-2 | +6.19e-2 | +1.21e-3 | +4.52e-4 |
| 127 | 3.00e-2 | 19 | 8.66e-3 | 2.15e-2 | 1.07e-2 | 1.01e-2 | 17 | -4.57e-2 | +4.28e-2 | -1.08e-4 | -4.63e-5 |
| 128 | 3.00e-2 | 11 | 9.23e-3 | 2.32e-2 | 1.25e-2 | 1.13e-2 | 19 | -6.16e-2 | +4.88e-2 | -6.11e-4 | -6.25e-4 |
| 129 | 3.00e-2 | 15 | 9.75e-3 | 2.25e-2 | 1.18e-2 | 1.03e-2 | 16 | -3.46e-2 | +3.55e-2 | -6.35e-4 | -6.91e-4 |
| 130 | 3.00e-2 | 14 | 9.13e-3 | 2.33e-2 | 1.16e-2 | 1.14e-2 | 32 | -5.06e-2 | +4.90e-2 | +2.41e-4 | +1.86e-4 |
| 131 | 3.00e-2 | 9 | 1.58e-2 | 2.86e-2 | 1.86e-2 | 1.58e-2 | 24 | -1.78e-2 | +1.45e-2 | +4.10e-5 | -2.45e-4 |
| 132 | 3.00e-2 | 18 | 1.12e-2 | 2.65e-2 | 1.32e-2 | 1.27e-2 | 17 | -3.62e-2 | +2.42e-2 | -8.52e-4 | -1.96e-4 |
| 133 | 3.00e-2 | 11 | 9.71e-3 | 2.59e-2 | 1.30e-2 | 9.71e-3 | 17 | -4.11e-2 | +4.77e-2 | -1.20e-3 | -1.60e-3 |
| 134 | 3.00e-2 | 12 | 8.54e-3 | 2.45e-2 | 1.27e-2 | 2.29e-2 | 38 | -7.84e-2 | +6.15e-2 | +1.55e-3 | +1.47e-3 |
| 135 | 3.00e-2 | 8 | 1.91e-2 | 3.00e-2 | 2.16e-2 | 2.07e-2 | 32 | -1.05e-2 | +1.02e-2 | -1.48e-4 | +5.48e-4 |
| 136 | 3.00e-2 | 9 | 1.60e-2 | 3.05e-2 | 1.90e-2 | 1.68e-2 | 22 | -1.71e-2 | +1.57e-2 | -9.30e-4 | -3.52e-4 |
| 137 | 3.00e-2 | 14 | 1.22e-2 | 2.92e-2 | 1.47e-2 | 1.36e-2 | 19 | -4.74e-2 | +3.18e-2 | -9.87e-4 | -5.83e-4 |
| 138 | 3.00e-2 | 13 | 1.10e-2 | 2.91e-2 | 1.46e-2 | 1.29e-2 | 23 | -6.48e-2 | +4.96e-2 | -1.77e-4 | -4.61e-4 |
| 139 | 3.00e-2 | 12 | 1.41e-2 | 2.88e-2 | 1.72e-2 | 1.41e-2 | 17 | -2.62e-2 | +2.29e-2 | -6.77e-4 | -1.02e-3 |
| 140 | 3.00e-2 | 17 | 1.13e-2 | 2.82e-2 | 1.58e-2 | 1.63e-2 | 20 | -4.56e-2 | +5.05e-2 | +1.34e-3 | +4.76e-4 |
| 141 | 3.00e-2 | 10 | 1.31e-2 | 3.12e-2 | 1.65e-2 | 1.50e-2 | 16 | -2.94e-2 | +3.08e-2 | -7.01e-4 | -2.39e-4 |
| 142 | 3.00e-2 | 13 | 1.18e-2 | 3.39e-2 | 1.46e-2 | 1.48e-2 | 18 | -7.36e-2 | +5.53e-2 | -8.58e-4 | -7.34e-5 |
| 143 | 3.00e-2 | 17 | 1.07e-2 | 3.30e-2 | 1.43e-2 | 1.26e-2 | 15 | -6.64e-2 | +4.91e-2 | -1.52e-3 | -1.02e-3 |
| 144 | 3.00e-2 | 15 | 1.20e-2 | 2.90e-2 | 1.49e-2 | 1.49e-2 | 18 | -5.01e-2 | +5.78e-2 | +5.84e-4 | -9.49e-5 |
| 145 | 3.00e-2 | 13 | 1.50e-2 | 3.28e-2 | 1.79e-2 | 1.97e-2 | 20 | -3.57e-2 | +3.83e-2 | +9.25e-4 | +8.10e-4 |
| 146 | 3.00e-2 | 17 | 1.45e-2 | 3.49e-2 | 1.69e-2 | 1.59e-2 | 18 | -4.17e-2 | +3.95e-2 | -3.78e-4 | -4.37e-5 |
| 147 | 3.00e-2 | 10 | 1.47e-2 | 3.12e-2 | 1.74e-2 | 1.58e-2 | 17 | -3.92e-2 | +4.11e-2 | -7.28e-5 | -2.53e-4 |
| 148 | 3.00e-2 | 15 | 1.15e-2 | 3.12e-2 | 1.63e-2 | 1.96e-2 | 20 | -5.64e-2 | +4.92e-2 | +7.56e-4 | +1.02e-3 |
| 149 | 3.00e-2 | 13 | 1.40e-2 | 3.04e-2 | 1.72e-2 | 1.81e-2 | 20 | -4.27e-2 | +3.57e-2 | -1.47e-4 | +4.49e-4 |
| 150 | 3.00e-3 | 15 | 1.28e-3 | 3.05e-2 | 4.52e-3 | 1.28e-3 | 15 | -1.53e-1 | +2.87e-2 | -8.73e-3 | -4.09e-3 |
| 151 | 3.00e-3 | 15 | 1.06e-3 | 2.95e-3 | 1.44e-3 | 1.37e-3 | 18 | -5.85e-2 | +5.54e-2 | -5.35e-5 | -1.03e-3 |
| 152 | 3.00e-3 | 2 | 1.48e-3 | 1.52e-3 | 1.50e-3 | 1.52e-3 | 212 | +1.36e-4 | +4.38e-3 | +2.26e-3 | -4.29e-4 |
| 153 | 3.00e-3 | 1 | 5.80e-3 | 5.80e-3 | 5.80e-3 | 5.80e-3 | 257 | +5.21e-3 | +5.21e-3 | +5.21e-3 | +1.35e-4 |
| 154 | 3.00e-3 | 1 | 6.40e-3 | 6.40e-3 | 6.40e-3 | 6.40e-3 | 250 | +3.94e-4 | +3.94e-4 | +3.94e-4 | +1.61e-4 |
| 155 | 3.00e-3 | 1 | 6.28e-3 | 6.28e-3 | 6.28e-3 | 6.28e-3 | 238 | -8.08e-5 | -8.08e-5 | -8.08e-5 | +1.37e-4 |
| 156 | 3.00e-3 | 1 | 6.19e-3 | 6.19e-3 | 6.19e-3 | 6.19e-3 | 253 | -5.47e-5 | -5.47e-5 | -5.47e-5 | +1.18e-4 |
| 157 | 3.00e-3 | 1 | 6.13e-3 | 6.13e-3 | 6.13e-3 | 6.13e-3 | 257 | -3.88e-5 | -3.88e-5 | -3.88e-5 | +1.02e-4 |
| 158 | 3.00e-3 | 1 | 6.29e-3 | 6.29e-3 | 6.29e-3 | 6.29e-3 | 264 | +1.01e-4 | +1.01e-4 | +1.01e-4 | +1.02e-4 |
| 159 | 3.00e-3 | 2 | 6.11e-3 | 6.44e-3 | 6.28e-3 | 6.11e-3 | 218 | -2.37e-4 | +9.66e-5 | -7.00e-5 | +6.76e-5 |
| 160 | 3.00e-3 | 1 | 6.02e-3 | 6.02e-3 | 6.02e-3 | 6.02e-3 | 237 | -6.43e-5 | -6.43e-5 | -6.43e-5 | +5.44e-5 |
| 161 | 3.00e-3 | 1 | 6.10e-3 | 6.10e-3 | 6.10e-3 | 6.10e-3 | 260 | +4.95e-5 | +4.95e-5 | +4.95e-5 | +5.39e-5 |
| 162 | 3.00e-3 | 1 | 6.55e-3 | 6.55e-3 | 6.55e-3 | 6.55e-3 | 260 | +2.73e-4 | +2.73e-4 | +2.73e-4 | +7.59e-5 |
| 163 | 3.00e-3 | 1 | 6.46e-3 | 6.46e-3 | 6.46e-3 | 6.46e-3 | 263 | -5.37e-5 | -5.37e-5 | -5.37e-5 | +6.29e-5 |
| 164 | 3.00e-3 | 1 | 6.69e-3 | 6.69e-3 | 6.69e-3 | 6.69e-3 | 263 | +1.34e-4 | +1.34e-4 | +1.34e-4 | +7.00e-5 |
| 165 | 3.00e-3 | 1 | 6.38e-3 | 6.38e-3 | 6.38e-3 | 6.38e-3 | 250 | -1.87e-4 | -1.87e-4 | -1.87e-4 | +4.44e-5 |
| 166 | 3.00e-3 | 2 | 6.13e-3 | 6.28e-3 | 6.21e-3 | 6.13e-3 | 182 | -1.31e-4 | -7.29e-5 | -1.02e-4 | +1.63e-5 |
| 167 | 3.00e-3 | 1 | 5.79e-3 | 5.79e-3 | 5.79e-3 | 5.79e-3 | 229 | -2.53e-4 | -2.53e-4 | -2.53e-4 | -1.06e-5 |
| 168 | 3.00e-3 | 1 | 6.02e-3 | 6.02e-3 | 6.02e-3 | 6.02e-3 | 208 | +1.91e-4 | +1.91e-4 | +1.91e-4 | +9.53e-6 |
| 169 | 3.00e-3 | 2 | 5.94e-3 | 6.11e-3 | 6.03e-3 | 5.94e-3 | 194 | -1.47e-4 | +7.01e-5 | -3.84e-5 | -6.73e-7 |
| 170 | 3.00e-3 | 1 | 5.55e-3 | 5.55e-3 | 5.55e-3 | 5.55e-3 | 226 | -2.97e-4 | -2.97e-4 | -2.97e-4 | -3.03e-5 |
| 171 | 3.00e-3 | 1 | 6.14e-3 | 6.14e-3 | 6.14e-3 | 6.14e-3 | 249 | +4.01e-4 | +4.01e-4 | +4.01e-4 | +1.29e-5 |
| 172 | 3.00e-3 | 2 | 6.32e-3 | 6.68e-3 | 6.50e-3 | 6.68e-3 | 216 | +1.13e-4 | +2.51e-4 | +1.82e-4 | +4.57e-5 |
| 174 | 3.00e-3 | 2 | 6.24e-3 | 6.75e-3 | 6.50e-3 | 6.75e-3 | 199 | -2.46e-4 | +3.95e-4 | +7.42e-5 | +5.43e-5 |
| 175 | 3.00e-3 | 1 | 6.06e-3 | 6.06e-3 | 6.06e-3 | 6.06e-3 | 207 | -5.23e-4 | -5.23e-4 | -5.23e-4 | -3.44e-6 |
| 176 | 3.00e-3 | 1 | 6.18e-3 | 6.18e-3 | 6.18e-3 | 6.18e-3 | 206 | +9.82e-5 | +9.82e-5 | +9.82e-5 | +6.72e-6 |
| 177 | 3.00e-3 | 2 | 5.90e-3 | 6.03e-3 | 5.96e-3 | 5.90e-3 | 173 | -1.29e-4 | -1.26e-4 | -1.28e-4 | -1.88e-5 |
| 178 | 3.00e-3 | 1 | 5.75e-3 | 5.75e-3 | 5.75e-3 | 5.75e-3 | 223 | -1.15e-4 | -1.15e-4 | -1.15e-4 | -2.85e-5 |
| 179 | 3.00e-3 | 2 | 6.06e-3 | 6.16e-3 | 6.11e-3 | 6.06e-3 | 173 | -9.89e-5 | +3.32e-4 | +1.17e-4 | -3.04e-6 |
| 180 | 3.00e-3 | 1 | 5.66e-3 | 5.66e-3 | 5.66e-3 | 5.66e-3 | 210 | -3.20e-4 | -3.20e-4 | -3.20e-4 | -3.47e-5 |
| 181 | 3.00e-3 | 2 | 6.28e-3 | 6.59e-3 | 6.44e-3 | 6.28e-3 | 175 | -2.77e-4 | +6.66e-4 | +1.94e-4 | +4.14e-6 |
| 182 | 3.00e-3 | 1 | 5.94e-3 | 5.94e-3 | 5.94e-3 | 5.94e-3 | 197 | -2.82e-4 | -2.82e-4 | -2.82e-4 | -2.45e-5 |
| 183 | 3.00e-3 | 1 | 6.18e-3 | 6.18e-3 | 6.18e-3 | 6.18e-3 | 201 | +1.93e-4 | +1.93e-4 | +1.93e-4 | -2.77e-6 |
| 184 | 3.00e-3 | 2 | 6.19e-3 | 6.28e-3 | 6.23e-3 | 6.28e-3 | 173 | +1.08e-5 | +8.41e-5 | +4.74e-5 | +7.13e-6 |
| 185 | 3.00e-3 | 1 | 5.70e-3 | 5.70e-3 | 5.70e-3 | 5.70e-3 | 208 | -4.65e-4 | -4.65e-4 | -4.65e-4 | -4.01e-5 |
| 186 | 3.00e-3 | 1 | 6.09e-3 | 6.09e-3 | 6.09e-3 | 6.09e-3 | 222 | +3.01e-4 | +3.01e-4 | +3.01e-4 | -6.01e-6 |
| 187 | 3.00e-3 | 2 | 6.12e-3 | 6.40e-3 | 6.26e-3 | 6.12e-3 | 173 | -2.64e-4 | +2.55e-4 | -4.50e-6 | -8.32e-6 |
| 188 | 3.00e-3 | 2 | 5.73e-3 | 6.39e-3 | 6.06e-3 | 6.39e-3 | 160 | -3.12e-4 | +6.89e-4 | +1.89e-4 | +3.41e-5 |
| 189 | 3.00e-3 | 1 | 5.73e-3 | 5.73e-3 | 5.73e-3 | 5.73e-3 | 196 | -5.56e-4 | -5.56e-4 | -5.56e-4 | -2.48e-5 |
| 190 | 3.00e-3 | 1 | 6.01e-3 | 6.01e-3 | 6.01e-3 | 6.01e-3 | 205 | +2.26e-4 | +2.26e-4 | +2.26e-4 | +2.61e-7 |
| 191 | 3.00e-3 | 2 | 6.20e-3 | 6.34e-3 | 6.27e-3 | 6.34e-3 | 151 | +1.46e-4 | +1.56e-4 | +1.51e-4 | +2.89e-5 |
| 192 | 3.00e-3 | 2 | 5.40e-3 | 6.25e-3 | 5.83e-3 | 6.25e-3 | 151 | -7.47e-4 | +9.72e-4 | +1.12e-4 | +5.33e-5 |
| 193 | 3.00e-3 | 1 | 5.43e-3 | 5.43e-3 | 5.43e-3 | 5.43e-3 | 203 | -6.97e-4 | -6.97e-4 | -6.97e-4 | -2.17e-5 |
| 194 | 3.00e-3 | 1 | 6.22e-3 | 6.22e-3 | 6.22e-3 | 6.22e-3 | 192 | +7.10e-4 | +7.10e-4 | +7.10e-4 | +5.15e-5 |
| 195 | 3.00e-3 | 2 | 5.83e-3 | 5.90e-3 | 5.87e-3 | 5.83e-3 | 151 | -2.80e-4 | -8.10e-5 | -1.81e-4 | +8.38e-6 |
| 196 | 3.00e-3 | 2 | 5.66e-3 | 6.07e-3 | 5.86e-3 | 6.07e-3 | 141 | -1.51e-4 | +4.98e-4 | +1.73e-4 | +4.30e-5 |
| 197 | 3.00e-3 | 1 | 5.29e-3 | 5.29e-3 | 5.29e-3 | 5.29e-3 | 170 | -8.10e-4 | -8.10e-4 | -8.10e-4 | -4.23e-5 |
| 198 | 3.00e-3 | 2 | 5.60e-3 | 5.95e-3 | 5.77e-3 | 5.95e-3 | 141 | +3.01e-4 | +4.32e-4 | +3.66e-4 | +3.60e-5 |
| 199 | 3.00e-3 | 2 | 5.38e-3 | 5.90e-3 | 5.64e-3 | 5.90e-3 | 141 | -5.58e-4 | +6.53e-4 | +4.71e-5 | +4.41e-5 |

