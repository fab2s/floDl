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
| nccl-async | 0.062318 | 0.9163 | +0.0038 | 1741.9 | 669 | 37.6 | 100% | 100% | 100% | 4.0 |

## Best Mode per Model

| Model | Best Eval | Mode | Fastest (within 2% of solo-0) | Mode |
|-------|-----------|------|-------------------------------|------|
| resnet-graph | 0.9163 | nccl-async | - | - |

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
| nccl-async | 1.9811 | 0.7928 | 0.6071 | 0.5528 | 0.5307 | 0.5113 | 0.4920 | 0.4835 | 0.4767 | 0.4686 | 0.2148 | 0.1756 | 0.1608 | 0.1509 | 0.1461 | 0.0820 | 0.0747 | 0.0679 | 0.0650 | 0.0623 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | nccl-async | 0 | cuda:0 | 0.4035 | 2.9 |
| resnet-graph | nccl-async | 1 | cuda:1 | 0.3005 | 3.7 |
| resnet-graph | nccl-async | 2 | cuda:2 | 0.2960 | 3.7 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) | GPU2 Peak (MB) | GPU2 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|---------------|---------------|
| resnet-graph | nccl-async | 356 | 355 | 394 | 379 | 390 | 380 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| nccl-async | gpu1 | 1741.0 | 0.9 | epoch-boundary(199) |
| nccl-async | gpu2 | 1741.0 | 0.9 | epoch-boundary(199) |
| nccl-async | gpu0 | 1741.0 | 0.6 | epoch-boundary(199) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | nccl-async | gpu0 | 0.6s | 0.0s | 0.0s | 0.0s | 1.5s |
| resnet-graph | nccl-async | gpu1 | 0.9s | 0.0s | 0.0s | 0.0s | 0.9s |
| resnet-graph | nccl-async | gpu2 | 0.9s | 0.0s | 0.0s | 0.0s | 1.6s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | nccl-async | 355 | 0 | 669 | 37.6 | 673/9000 | 0 | 0.0 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | nccl-async | 167.3 | 9.6% |

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
| resnet-graph | nccl-async | 192 | 669 | 0 | 1.58e-3 | -5.56e-4 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | nccl-async | 0 | 669 | 4.73e-2 | 6.83e-2 | 0.00e0 | 4.08e-1 | 42.9 | -1.40e-4 | 3.33e-3 |
| resnet-graph | nccl-async | 1 | 669 | 4.76e-2 | 6.95e-2 | 0.00e0 | 3.87e-1 | 35.7 | -1.62e-4 | 5.17e-3 |
| resnet-graph | nccl-async | 2 | 669 | 4.75e-2 | 6.94e-2 | 0.00e0 | 4.25e-1 | 21.4 | -1.62e-4 | 5.15e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | nccl-async | rank0 ↔ rank1 | +0.9982 |
| resnet-graph | nccl-async | rank0 ↔ rank2 | +0.9985 |
| resnet-graph | nccl-async | rank1 ↔ rank2 | +0.9993 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | nccl-async | 32 (0,2,3,4,5,7,8,9…141,149) | 0 (—) | — | 0,2,3,4,5,7,8,9…141,149 | — |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | nccl-async | 0e0 | 3 | 45 | 45 |
| resnet-graph | nccl-async | 0e0 | 5 | 20 | 20 |
| resnet-graph | nccl-async | 0e0 | 10 | 7 | 7 |
| resnet-graph | nccl-async | 1e-4 | 3 | 16 | 16 |
| resnet-graph | nccl-async | 1e-4 | 5 | 5 | 5 |
| resnet-graph | nccl-async | 1e-4 | 10 | 2 | 2 |
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
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 224 | +0.058 |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | 67 | +0.216 |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | 373 | +0.014 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | nccl-async | 666 | -0.004 | 191 | +0.055 | -0.040 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | nccl-async | 667 | 3.36e1–7.89e1 | 6.00e1 | 1.79e-3 | 4.14e-3 | 6.10e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 226 | 69–77947 | +1.691e-5 | 0.576 | +1.748e-5 | 0.584 | 93 | +7.654e-6 | 0.283 | 34–970 | +1.394e-3 | 0.800 |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 208 | 877–77947 | +1.636e-5 | 0.613 | +1.688e-5 | 0.619 | 92 | +7.033e-6 | 0.260 | 37–970 | +1.368e-3 | 0.872 |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | 68 | 78493–117042 | +1.391e-5 | 0.185 | +1.371e-5 | 0.181 | 49 | +1.322e-5 | 0.192 | 399–771 | -1.099e-3 | 0.055 |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | 374 | 117540–156183 | -3.887e-5 | 0.550 | -3.999e-5 | 0.566 | 50 | -4.589e-5 | 0.713 | 34–524 | +3.671e-3 | 0.660 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | +1.394e-3 | r0: +1.355e-3, r1: +1.424e-3, r2: +1.408e-3 | r0: 0.818, r1: 0.789, r2: 0.787 | 1.05× | ✓ |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +1.368e-3 | r0: +1.332e-3, r1: +1.398e-3, r2: +1.379e-3 | r0: 0.895, r1: 0.854, r2: 0.861 | 1.05× | ✓ |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | -1.099e-3 | r0: -1.034e-3, r1: -1.141e-3, r2: -1.120e-3 | r0: 0.048, r1: 0.058, r2: 0.057 | 1.10× | ✓ |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | +3.671e-3 | r0: +3.606e-3, r1: +3.683e-3, r2: +3.743e-3 | r0: 0.706, r1: 0.628, r2: 0.631 | 1.04× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | nccl-async | `█▇▇████████████████████▇▄▅▅▅▅▅▅▅▅▅▆▅▂▂▂▂▂▁▁▁▁▁▁▁` | `▁▇▅█▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▃▂▅▆▆▇▇▇▇▇▇▇▅▂▅▆▆▆▅▄▅▃▄▇█` |

### Per-Epoch Detail: resnet-graph / nccl-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 19 | 0.00e0 | 4.25e-1 | 8.08e-2 | 5.52e-2 | 15 | -6.39e-2 | +1.60e-2 | -8.16e-3 | -3.01e-3 |
| 1 | 3.00e-1 | 18 | 5.48e-2 | 1.12e-1 | 6.65e-2 | 6.65e-2 | 20 | -3.81e-2 | +4.20e-2 | +7.66e-4 | -1.54e-4 |
| 2 | 3.00e-1 | 10 | 6.26e-2 | 1.15e-1 | 7.14e-2 | 6.74e-2 | 18 | -3.59e-2 | +3.19e-2 | +6.70e-5 | -1.33e-4 |
| 3 | 3.00e-1 | 15 | 6.19e-2 | 1.26e-1 | 7.47e-2 | 6.73e-2 | 18 | -4.50e-2 | +3.78e-2 | -2.97e-4 | -5.33e-4 |
| 4 | 3.00e-1 | 14 | 6.19e-2 | 1.24e-1 | 8.28e-2 | 1.15e-1 | 41 | -3.88e-2 | +2.94e-2 | +1.02e-3 | +9.19e-4 |
| 5 | 3.00e-1 | 6 | 9.57e-2 | 1.41e-1 | 1.11e-1 | 1.01e-1 | 32 | -7.14e-3 | +4.99e-3 | -6.70e-4 | +1.48e-4 |
| 6 | 3.00e-1 | 10 | 7.83e-2 | 1.43e-1 | 9.08e-2 | 9.12e-2 | 25 | -1.61e-2 | +1.36e-2 | -3.83e-4 | +2.92e-5 |
| 7 | 3.00e-1 | 15 | 5.84e-2 | 1.29e-1 | 7.18e-2 | 6.63e-2 | 18 | -3.60e-2 | +2.03e-2 | -1.34e-3 | -5.27e-4 |
| 8 | 3.00e-1 | 14 | 6.64e-2 | 1.37e-1 | 7.80e-2 | 6.64e-2 | 20 | -4.12e-2 | +3.60e-2 | -2.49e-4 | -7.33e-4 |
| 9 | 3.00e-1 | 14 | 6.58e-2 | 1.35e-1 | 7.62e-2 | 7.04e-2 | 19 | -4.23e-2 | +2.88e-2 | -4.28e-4 | -5.09e-4 |
| 10 | 3.00e-1 | 1 | 7.50e-2 | 7.50e-2 | 7.50e-2 | 7.50e-2 | 18 | +3.53e-3 | +3.53e-3 | +3.53e-3 | -1.05e-4 |
| 11 | 3.00e-1 | 2 | 7.16e-2 | 2.30e-1 | 1.51e-1 | 2.30e-1 | 271 | -1.58e-4 | +4.30e-3 | +2.07e-3 | +3.31e-4 |
| 13 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 369 | -2.81e-4 | -2.81e-4 | -2.81e-4 | +2.70e-4 |
| 14 | 3.00e-1 | 1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 294 | +2.49e-4 | +2.49e-4 | +2.49e-4 | +2.67e-4 |
| 15 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 283 | -2.83e-4 | -2.83e-4 | -2.83e-4 | +2.12e-4 |
| 16 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 277 | -4.97e-5 | -4.97e-5 | -4.97e-5 | +1.86e-4 |
| 17 | 3.00e-1 | 1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 286 | -1.03e-4 | -1.03e-4 | -1.03e-4 | +1.57e-4 |
| 18 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 305 | +2.77e-5 | +2.77e-5 | +2.77e-5 | +1.44e-4 |
| 19 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 309 | +1.13e-4 | +1.13e-4 | +1.13e-4 | +1.41e-4 |
| 20 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 312 | -4.01e-5 | -4.01e-5 | -4.01e-5 | +1.23e-4 |
| 21 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 315 | -8.75e-6 | -8.75e-6 | -8.75e-6 | +1.10e-4 |
| 22 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 289 | +2.02e-5 | +2.02e-5 | +2.02e-5 | +1.01e-4 |
| 24 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 345 | -3.73e-5 | -3.73e-5 | -3.73e-5 | +8.71e-5 |
| 25 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 304 | +1.22e-4 | +1.22e-4 | +1.22e-4 | +9.06e-5 |
| 26 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 292 | -5.25e-5 | -5.25e-5 | -5.25e-5 | +7.63e-5 |
| 27 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 277 | +4.20e-6 | +4.20e-6 | +4.20e-6 | +6.91e-5 |
| 28 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 317 | -9.31e-5 | -9.31e-5 | -9.31e-5 | +5.28e-5 |
| 29 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 305 | +1.17e-4 | +1.17e-4 | +1.17e-4 | +5.93e-5 |
| 30 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 305 | -1.08e-5 | -1.08e-5 | -1.08e-5 | +5.23e-5 |
| 31 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 277 | -3.88e-5 | -3.88e-5 | -3.88e-5 | +4.32e-5 |
| 32 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 308 | -9.67e-5 | -9.67e-5 | -9.67e-5 | +2.92e-5 |
| 33 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 257 | +1.59e-4 | +1.59e-4 | +1.59e-4 | +4.21e-5 |
| 34 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 306 | -1.72e-4 | -1.72e-4 | -1.72e-4 | +2.07e-5 |
| 35 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 271 | +1.97e-4 | +1.97e-4 | +1.97e-4 | +3.84e-5 |
| 36 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 299 | -1.37e-4 | -1.37e-4 | -1.37e-4 | +2.09e-5 |
| 37 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 270 | +1.24e-4 | +1.24e-4 | +1.24e-4 | +3.12e-5 |
| 39 | 3.00e-1 | 2 | 1.97e-1 | 2.17e-1 | 2.07e-1 | 2.17e-1 | 243 | -1.17e-4 | +3.97e-4 | +1.40e-4 | +5.45e-5 |
| 40 | 3.00e-1 | 1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 243 | -4.86e-4 | -4.86e-4 | -4.86e-4 | +4.36e-7 |
| 41 | 3.00e-1 | 1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 280 | +1.17e-5 | +1.17e-5 | +1.17e-5 | +1.56e-6 |
| 43 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 340 | +1.65e-4 | +1.65e-4 | +1.65e-4 | +1.79e-5 |
| 44 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 326 | +1.73e-4 | +1.73e-4 | +1.73e-4 | +3.34e-5 |
| 45 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 313 | -4.89e-5 | -4.89e-5 | -4.89e-5 | +2.52e-5 |
| 46 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 290 | -8.61e-5 | -8.61e-5 | -8.61e-5 | +1.40e-5 |
| 47 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 273 | -4.84e-5 | -4.84e-5 | -4.84e-5 | +7.79e-6 |
| 48 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 275 | -4.33e-5 | -4.33e-5 | -4.33e-5 | +2.68e-6 |
| 49 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 306 | +6.33e-6 | +6.33e-6 | +6.33e-6 | +3.05e-6 |
| 50 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 294 | +8.62e-5 | +8.62e-5 | +8.62e-5 | +1.14e-5 |
| 51 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 300 | -3.85e-5 | -3.85e-5 | -3.85e-5 | +6.38e-6 |
| 52 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 307 | +9.65e-5 | +9.65e-5 | +9.65e-5 | +1.54e-5 |
| 54 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 340 | -7.11e-5 | -7.11e-5 | -7.11e-5 | +6.74e-6 |
| 55 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 280 | +9.49e-5 | +9.49e-5 | +9.49e-5 | +1.56e-5 |
| 56 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 260 | -2.26e-4 | -2.26e-4 | -2.26e-4 | -8.59e-6 |
| 57 | 3.00e-1 | 2 | 2.02e-1 | 2.03e-1 | 2.02e-1 | 2.03e-1 | 260 | +1.49e-5 | +2.54e-5 | +2.02e-5 | -3.18e-6 |
| 59 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 310 | -3.46e-5 | -3.46e-5 | -3.46e-5 | -6.33e-6 |
| 60 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 293 | +1.83e-4 | +1.83e-4 | +1.83e-4 | +1.26e-5 |
| 61 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 281 | -9.77e-5 | -9.77e-5 | -9.77e-5 | +1.59e-6 |
| 62 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 280 | -1.47e-5 | -1.47e-5 | -1.47e-5 | -3.73e-8 |
| 63 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 293 | +2.07e-6 | +2.07e-6 | +2.07e-6 | +1.74e-7 |
| 64 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 295 | +2.99e-5 | +2.99e-5 | +2.99e-5 | +3.15e-6 |
| 65 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 292 | -7.72e-6 | -7.72e-6 | -7.72e-6 | +2.06e-6 |
| 66 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 305 | +3.09e-5 | +3.09e-5 | +3.09e-5 | +4.94e-6 |
| 67 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 264 | +3.09e-5 | +3.09e-5 | +3.09e-5 | +7.53e-6 |
| 68 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 297 | -1.49e-4 | -1.49e-4 | -1.49e-4 | -8.14e-6 |
| 69 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 282 | +1.34e-4 | +1.34e-4 | +1.34e-4 | +6.08e-6 |
| 70 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 247 | -1.13e-4 | -1.13e-4 | -1.13e-4 | -5.81e-6 |
| 71 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 262 | -1.29e-4 | -1.29e-4 | -1.29e-4 | -1.81e-5 |
| 72 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 267 | +1.25e-4 | +1.25e-4 | +1.25e-4 | -3.82e-6 |
| 73 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 295 | +2.16e-5 | +2.16e-5 | +2.16e-5 | -1.28e-6 |
| 74 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 248 | +5.89e-5 | +5.89e-5 | +5.89e-5 | +4.74e-6 |
| 75 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 257 | -1.38e-4 | -1.38e-4 | -1.38e-4 | -9.58e-6 |
| 76 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 242 | +4.16e-5 | +4.16e-5 | +4.16e-5 | -4.46e-6 |
| 77 | 3.00e-1 | 2 | 1.93e-1 | 2.09e-1 | 2.01e-1 | 2.09e-1 | 242 | -1.68e-4 | +3.42e-4 | +8.71e-5 | +1.55e-5 |
| 79 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 294 | -1.80e-4 | -1.80e-4 | -1.80e-4 | -4.05e-6 |
| 80 | 3.00e-1 | 2 | 2.03e-1 | 2.09e-1 | 2.06e-1 | 2.03e-1 | 198 | -1.56e-4 | +2.09e-4 | +2.69e-5 | -2.36e-9 |
| 81 | 3.00e-1 | 1 | 1.85e-1 | 1.85e-1 | 1.85e-1 | 1.85e-1 | 217 | -4.23e-4 | -4.23e-4 | -4.23e-4 | -4.23e-5 |
| 82 | 3.00e-1 | 1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 206 | +1.28e-4 | +1.28e-4 | +1.28e-4 | -2.53e-5 |
| 83 | 3.00e-1 | 1 | 1.87e-1 | 1.87e-1 | 1.87e-1 | 1.87e-1 | 230 | -6.95e-5 | -6.95e-5 | -6.95e-5 | -2.97e-5 |
| 84 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 248 | +1.92e-4 | +1.92e-4 | +1.92e-4 | -7.55e-6 |
| 85 | 3.00e-1 | 2 | 2.01e-1 | 2.07e-1 | 2.04e-1 | 2.07e-1 | 206 | +8.91e-5 | +1.28e-4 | +1.08e-4 | +1.47e-5 |
| 86 | 3.00e-1 | 1 | 1.87e-1 | 1.87e-1 | 1.87e-1 | 1.87e-1 | 225 | -4.54e-4 | -4.54e-4 | -4.54e-4 | -3.22e-5 |
| 87 | 3.00e-1 | 1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 234 | +1.32e-4 | +1.32e-4 | +1.32e-4 | -1.58e-5 |
| 88 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 220 | +7.43e-5 | +7.43e-5 | +7.43e-5 | -6.77e-6 |
| 89 | 3.00e-1 | 1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 228 | -4.47e-5 | -4.47e-5 | -4.47e-5 | -1.06e-5 |
| 90 | 3.00e-1 | 2 | 1.96e-1 | 2.00e-1 | 1.98e-1 | 2.00e-1 | 206 | +5.11e-5 | +1.05e-4 | +7.81e-5 | +6.55e-6 |
| 91 | 3.00e-1 | 1 | 1.86e-1 | 1.86e-1 | 1.86e-1 | 1.86e-1 | 250 | -2.93e-4 | -2.93e-4 | -2.93e-4 | -2.34e-5 |
| 92 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 251 | +2.87e-4 | +2.87e-4 | +2.87e-4 | +7.65e-6 |
| 93 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 234 | +4.20e-5 | +4.20e-5 | +4.20e-5 | +1.11e-5 |
| 94 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 226 | -1.28e-4 | -1.28e-4 | -1.28e-4 | -2.80e-6 |
| 95 | 3.00e-1 | 2 | 1.89e-1 | 1.92e-1 | 1.91e-1 | 1.89e-1 | 202 | -9.83e-5 | -7.33e-5 | -8.58e-5 | -1.84e-5 |
| 96 | 3.00e-1 | 1 | 1.89e-1 | 1.89e-1 | 1.89e-1 | 1.89e-1 | 216 | -2.09e-6 | -2.09e-6 | -2.09e-6 | -1.68e-5 |
| 97 | 3.00e-1 | 1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 227 | +6.03e-5 | +6.03e-5 | +6.03e-5 | -9.09e-6 |
| 98 | 3.00e-1 | 1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 252 | +2.49e-5 | +2.49e-5 | +2.49e-5 | -5.69e-6 |
| 99 | 3.00e-1 | 2 | 1.93e-1 | 1.98e-1 | 1.95e-1 | 1.93e-1 | 174 | -1.62e-4 | +1.14e-4 | -2.43e-5 | -1.06e-5 |
| 100 | 3.00e-2 | 1 | 1.77e-1 | 1.77e-1 | 1.77e-1 | 1.77e-1 | 193 | -4.26e-4 | -4.26e-4 | -4.26e-4 | -5.21e-5 |
| 101 | 3.00e-2 | 2 | 1.90e-2 | 1.74e-1 | 9.63e-2 | 1.90e-2 | 170 | -1.30e-2 | -1.04e-4 | -6.56e-3 | -1.35e-3 |
| 102 | 3.00e-2 | 1 | 1.85e-2 | 1.85e-2 | 1.85e-2 | 1.85e-2 | 202 | -1.25e-4 | -1.25e-4 | -1.25e-4 | -1.23e-3 |
| 103 | 3.00e-2 | 1 | 2.11e-2 | 2.11e-2 | 2.11e-2 | 2.11e-2 | 220 | +5.95e-4 | +5.95e-4 | +5.95e-4 | -1.05e-3 |
| 104 | 3.00e-2 | 2 | 2.25e-2 | 2.34e-2 | 2.30e-2 | 2.34e-2 | 176 | +2.18e-4 | +2.82e-4 | +2.50e-4 | -8.02e-4 |
| 105 | 3.00e-2 | 2 | 2.14e-2 | 2.37e-2 | 2.26e-2 | 2.37e-2 | 176 | -4.48e-4 | +5.77e-4 | +6.41e-5 | -6.32e-4 |
| 106 | 3.00e-2 | 1 | 2.22e-2 | 2.22e-2 | 2.22e-2 | 2.22e-2 | 194 | -3.48e-4 | -3.48e-4 | -3.48e-4 | -6.04e-4 |
| 107 | 3.00e-2 | 2 | 2.37e-2 | 2.51e-2 | 2.44e-2 | 2.51e-2 | 187 | +3.05e-4 | +3.21e-4 | +3.13e-4 | -4.30e-4 |
| 108 | 3.00e-2 | 1 | 2.49e-2 | 2.49e-2 | 2.49e-2 | 2.49e-2 | 215 | -4.03e-5 | -4.03e-5 | -4.03e-5 | -3.91e-4 |
| 109 | 3.00e-2 | 1 | 2.65e-2 | 2.65e-2 | 2.65e-2 | 2.65e-2 | 187 | +3.38e-4 | +3.38e-4 | +3.38e-4 | -3.18e-4 |
| 110 | 3.00e-2 | 1 | 2.55e-2 | 2.55e-2 | 2.55e-2 | 2.55e-2 | 271 | -1.39e-4 | -1.39e-4 | -1.39e-4 | -3.00e-4 |
| 111 | 3.00e-2 | 1 | 3.05e-2 | 3.05e-2 | 3.05e-2 | 3.05e-2 | 240 | +7.45e-4 | +7.45e-4 | +7.45e-4 | -1.96e-4 |
| 112 | 3.00e-2 | 2 | 2.97e-2 | 3.00e-2 | 2.99e-2 | 2.97e-2 | 218 | -6.78e-5 | -4.11e-5 | -5.45e-5 | -1.69e-4 |
| 114 | 3.00e-2 | 2 | 2.96e-2 | 3.36e-2 | 3.16e-2 | 3.36e-2 | 218 | -2.27e-5 | +5.87e-4 | +2.82e-4 | -8.00e-5 |
| 115 | 3.00e-2 | 1 | 3.08e-2 | 3.08e-2 | 3.08e-2 | 3.08e-2 | 244 | -3.55e-4 | -3.55e-4 | -3.55e-4 | -1.08e-4 |
| 116 | 3.00e-2 | 1 | 3.28e-2 | 3.28e-2 | 3.28e-2 | 3.28e-2 | 257 | +2.47e-4 | +2.47e-4 | +2.47e-4 | -7.20e-5 |
| 117 | 3.00e-2 | 1 | 3.39e-2 | 3.39e-2 | 3.39e-2 | 3.39e-2 | 244 | +1.36e-4 | +1.36e-4 | +1.36e-4 | -5.12e-5 |
| 118 | 3.00e-2 | 1 | 3.39e-2 | 3.39e-2 | 3.39e-2 | 3.39e-2 | 266 | -1.06e-6 | -1.06e-6 | -1.06e-6 | -4.62e-5 |
| 119 | 3.00e-2 | 2 | 3.54e-2 | 3.60e-2 | 3.57e-2 | 3.54e-2 | 205 | -8.07e-5 | +2.43e-4 | +8.12e-5 | -2.36e-5 |
| 120 | 3.00e-2 | 1 | 3.30e-2 | 3.30e-2 | 3.30e-2 | 3.30e-2 | 242 | -2.90e-4 | -2.90e-4 | -2.90e-4 | -5.03e-5 |
| 121 | 3.00e-2 | 1 | 3.70e-2 | 3.70e-2 | 3.70e-2 | 3.70e-2 | 250 | +4.57e-4 | +4.57e-4 | +4.57e-4 | +4.07e-7 |
| 122 | 3.00e-2 | 1 | 3.85e-2 | 3.85e-2 | 3.85e-2 | 3.85e-2 | 232 | +1.70e-4 | +1.70e-4 | +1.70e-4 | +1.74e-5 |
| 123 | 3.00e-2 | 2 | 3.73e-2 | 3.74e-2 | 3.73e-2 | 3.74e-2 | 193 | -1.50e-4 | +1.82e-5 | -6.61e-5 | +2.37e-6 |
| 124 | 3.00e-2 | 1 | 3.53e-2 | 3.53e-2 | 3.53e-2 | 3.53e-2 | 213 | -2.64e-4 | -2.64e-4 | -2.64e-4 | -2.42e-5 |
| 125 | 3.00e-2 | 1 | 3.79e-2 | 3.79e-2 | 3.79e-2 | 3.79e-2 | 203 | +3.43e-4 | +3.43e-4 | +3.43e-4 | +1.24e-5 |
| 126 | 3.00e-2 | 2 | 3.76e-2 | 3.80e-2 | 3.78e-2 | 3.80e-2 | 193 | -4.06e-5 | +5.53e-5 | +7.34e-6 | +1.20e-5 |
| 127 | 3.00e-2 | 1 | 3.72e-2 | 3.72e-2 | 3.72e-2 | 3.72e-2 | 205 | -1.02e-4 | -1.02e-4 | -1.02e-4 | +5.27e-7 |
| 128 | 3.00e-2 | 1 | 3.96e-2 | 3.96e-2 | 3.96e-2 | 3.96e-2 | 203 | +3.08e-4 | +3.08e-4 | +3.08e-4 | +3.13e-5 |
| 129 | 3.00e-2 | 1 | 3.91e-2 | 3.91e-2 | 3.91e-2 | 3.91e-2 | 210 | -6.48e-5 | -6.48e-5 | -6.48e-5 | +2.17e-5 |
| 130 | 3.00e-2 | 2 | 4.02e-2 | 4.21e-2 | 4.12e-2 | 4.21e-2 | 193 | +1.24e-4 | +2.41e-4 | +1.82e-4 | +5.28e-5 |
| 131 | 3.00e-2 | 1 | 3.98e-2 | 3.98e-2 | 3.98e-2 | 3.98e-2 | 227 | -2.54e-4 | -2.54e-4 | -2.54e-4 | +2.21e-5 |
| 132 | 3.00e-2 | 1 | 4.34e-2 | 4.34e-2 | 4.34e-2 | 4.34e-2 | 223 | +3.89e-4 | +3.89e-4 | +3.89e-4 | +5.88e-5 |
| 133 | 3.00e-2 | 2 | 4.28e-2 | 4.31e-2 | 4.30e-2 | 4.28e-2 | 209 | -3.18e-5 | -2.89e-5 | -3.03e-5 | +4.19e-5 |
| 134 | 3.00e-2 | 1 | 4.36e-2 | 4.36e-2 | 4.36e-2 | 4.36e-2 | 194 | +9.54e-5 | +9.54e-5 | +9.54e-5 | +4.73e-5 |
| 135 | 3.00e-2 | 1 | 4.26e-2 | 4.26e-2 | 4.26e-2 | 4.26e-2 | 192 | -1.21e-4 | -1.21e-4 | -1.21e-4 | +3.04e-5 |
| 136 | 3.00e-2 | 2 | 4.24e-2 | 4.42e-2 | 4.33e-2 | 4.42e-2 | 199 | -1.90e-5 | +2.06e-4 | +9.36e-5 | +4.35e-5 |
| 137 | 3.00e-2 | 1 | 4.44e-2 | 4.44e-2 | 4.44e-2 | 4.44e-2 | 228 | +1.78e-5 | +1.78e-5 | +1.78e-5 | +4.10e-5 |
| 138 | 3.00e-2 | 1 | 4.68e-2 | 4.68e-2 | 4.68e-2 | 4.68e-2 | 204 | +2.62e-4 | +2.62e-4 | +2.62e-4 | +6.30e-5 |
| 139 | 3.00e-2 | 2 | 4.46e-2 | 4.72e-2 | 4.59e-2 | 4.72e-2 | 158 | -2.24e-4 | +3.67e-4 | +7.13e-5 | +6.76e-5 |
| 140 | 3.00e-2 | 1 | 4.21e-2 | 4.21e-2 | 4.21e-2 | 4.21e-2 | 202 | -5.67e-4 | -5.67e-4 | -5.67e-4 | +4.08e-6 |
| 141 | 3.00e-2 | 2 | 4.60e-2 | 5.02e-2 | 4.81e-2 | 5.02e-2 | 166 | +3.89e-4 | +5.28e-4 | +4.58e-4 | +9.11e-5 |
| 142 | 3.00e-2 | 2 | 4.32e-2 | 4.67e-2 | 4.50e-2 | 4.67e-2 | 164 | -7.79e-4 | +4.82e-4 | -1.48e-4 | +5.19e-5 |
| 143 | 3.00e-2 | 1 | 4.53e-2 | 4.53e-2 | 4.53e-2 | 4.53e-2 | 193 | -1.61e-4 | -1.61e-4 | -1.61e-4 | +3.06e-5 |
| 144 | 3.00e-2 | 2 | 4.65e-2 | 4.78e-2 | 4.72e-2 | 4.65e-2 | 151 | -1.84e-4 | +3.09e-4 | +6.25e-5 | +3.42e-5 |
| 145 | 3.00e-2 | 2 | 4.32e-2 | 4.70e-2 | 4.51e-2 | 4.70e-2 | 151 | -4.36e-4 | +5.58e-4 | +6.09e-5 | +4.42e-5 |
| 146 | 3.00e-2 | 1 | 4.53e-2 | 4.53e-2 | 4.53e-2 | 4.53e-2 | 198 | -1.77e-4 | -1.77e-4 | -1.77e-4 | +2.21e-5 |
| 147 | 3.00e-2 | 2 | 4.85e-2 | 5.11e-2 | 4.98e-2 | 4.85e-2 | 165 | -3.18e-4 | +6.53e-4 | +1.68e-4 | +4.49e-5 |
| 148 | 3.00e-2 | 1 | 4.81e-2 | 4.81e-2 | 4.81e-2 | 4.81e-2 | 187 | -3.49e-5 | -3.49e-5 | -3.49e-5 | +3.70e-5 |
| 149 | 3.00e-2 | 2 | 4.98e-2 | 5.07e-2 | 5.02e-2 | 5.07e-2 | 143 | +1.26e-4 | +1.81e-4 | +1.53e-4 | +5.88e-5 |
| 150 | 3.00e-3 | 2 | 4.05e-2 | 4.49e-2 | 4.27e-2 | 4.05e-2 | 143 | -7.29e-4 | -6.49e-4 | -6.89e-4 | -8.37e-5 |
| 151 | 3.00e-3 | 1 | 4.54e-3 | 4.54e-3 | 4.54e-3 | 4.54e-3 | 167 | -1.31e-2 | -1.31e-2 | -1.31e-2 | -1.39e-3 |
| 152 | 3.00e-3 | 2 | 4.68e-3 | 4.75e-3 | 4.71e-3 | 4.68e-3 | 143 | -9.99e-5 | +2.53e-4 | +7.66e-5 | -1.11e-3 |
| 153 | 3.00e-3 | 2 | 4.42e-3 | 5.14e-3 | 4.78e-3 | 5.14e-3 | 128 | -2.86e-4 | +1.17e-3 | +4.43e-4 | -8.07e-4 |
| 154 | 3.00e-3 | 2 | 4.45e-3 | 4.97e-3 | 4.71e-3 | 4.97e-3 | 120 | -7.79e-4 | +9.26e-4 | +7.33e-5 | -6.31e-4 |
| 155 | 3.00e-3 | 2 | 3.98e-3 | 4.45e-3 | 4.22e-3 | 4.45e-3 | 114 | -1.39e-3 | +9.63e-4 | -2.14e-4 | -5.40e-4 |
| 156 | 3.00e-3 | 2 | 3.93e-3 | 4.34e-3 | 4.13e-3 | 4.34e-3 | 114 | -9.33e-4 | +8.67e-4 | -3.30e-5 | -4.35e-4 |
| 157 | 3.00e-3 | 2 | 3.96e-3 | 4.56e-3 | 4.26e-3 | 4.56e-3 | 120 | -6.13e-4 | +1.18e-3 | +2.86e-4 | -2.89e-4 |
| 158 | 3.00e-3 | 3 | 4.04e-3 | 4.59e-3 | 4.25e-3 | 4.14e-3 | 120 | -8.57e-4 | +1.06e-3 | -1.81e-4 | -2.61e-4 |
| 159 | 3.00e-3 | 1 | 4.24e-3 | 4.24e-3 | 4.24e-3 | 4.24e-3 | 178 | +1.43e-4 | +1.43e-4 | +1.43e-4 | -2.21e-4 |
| 160 | 3.00e-3 | 2 | 4.82e-3 | 4.98e-3 | 4.90e-3 | 4.82e-3 | 110 | -3.01e-4 | +9.94e-4 | +3.47e-4 | -1.20e-4 |
| 161 | 3.00e-3 | 3 | 4.00e-3 | 4.56e-3 | 4.21e-3 | 4.07e-3 | 116 | -1.24e-3 | +1.18e-3 | -3.47e-4 | -1.80e-4 |
| 162 | 3.00e-3 | 2 | 4.06e-3 | 4.72e-3 | 4.39e-3 | 4.72e-3 | 116 | -6.25e-6 | +1.29e-3 | +6.44e-4 | -1.67e-5 |
| 163 | 3.00e-3 | 2 | 4.14e-3 | 4.55e-3 | 4.34e-3 | 4.55e-3 | 110 | -9.70e-4 | +8.67e-4 | -5.15e-5 | -1.41e-5 |
| 164 | 3.00e-3 | 3 | 4.06e-3 | 4.28e-3 | 4.16e-3 | 4.06e-3 | 103 | -7.74e-4 | +3.12e-4 | -3.28e-4 | -9.71e-5 |
| 165 | 3.00e-3 | 2 | 3.99e-3 | 4.57e-3 | 4.28e-3 | 4.57e-3 | 95 | -1.12e-4 | +1.43e-3 | +6.58e-4 | +5.40e-5 |
| 166 | 3.00e-3 | 3 | 3.94e-3 | 4.52e-3 | 4.17e-3 | 4.06e-3 | 96 | -1.11e-3 | +1.44e-3 | -2.60e-4 | -3.19e-5 |
| 167 | 3.00e-3 | 3 | 3.67e-3 | 4.44e-3 | 4.02e-3 | 3.67e-3 | 83 | -2.32e-3 | +1.41e-3 | -3.68e-4 | -1.44e-4 |
| 168 | 3.00e-3 | 2 | 3.65e-3 | 4.39e-3 | 4.02e-3 | 4.39e-3 | 89 | -4.04e-5 | +2.08e-3 | +1.02e-3 | +8.81e-5 |
| 169 | 3.00e-3 | 4 | 3.56e-3 | 4.25e-3 | 3.76e-3 | 3.58e-3 | 82 | -2.11e-3 | +1.85e-3 | -4.48e-4 | -9.23e-5 |
| 170 | 3.00e-3 | 2 | 3.77e-3 | 4.52e-3 | 4.14e-3 | 4.52e-3 | 81 | +4.12e-4 | +2.24e-3 | +1.33e-3 | +1.86e-4 |
| 171 | 3.00e-3 | 5 | 3.11e-3 | 4.30e-3 | 3.61e-3 | 3.11e-3 | 68 | -3.26e-3 | +1.82e-3 | -9.01e-4 | -2.73e-4 |
| 172 | 3.00e-3 | 3 | 3.22e-3 | 3.85e-3 | 3.50e-3 | 3.42e-3 | 72 | -1.64e-3 | +2.48e-3 | +3.94e-4 | -1.12e-4 |
| 173 | 3.00e-3 | 5 | 3.10e-3 | 4.03e-3 | 3.41e-3 | 3.10e-3 | 60 | -3.21e-3 | +2.32e-3 | -3.96e-4 | -2.60e-4 |
| 174 | 3.00e-3 | 3 | 3.02e-3 | 3.74e-3 | 3.40e-3 | 3.02e-3 | 55 | -3.88e-3 | +1.51e-3 | -4.24e-4 | -3.53e-4 |
| 175 | 3.00e-3 | 6 | 2.60e-3 | 3.67e-3 | 3.07e-3 | 2.60e-3 | 47 | -3.84e-3 | +3.88e-3 | -4.86e-4 | -5.21e-4 |
| 176 | 3.00e-3 | 4 | 2.45e-3 | 3.64e-3 | 2.83e-3 | 2.56e-3 | 37 | -1.07e-2 | +7.41e-3 | -4.17e-4 | -5.53e-4 |
| 177 | 3.00e-3 | 9 | 2.06e-3 | 3.60e-3 | 2.47e-3 | 2.27e-3 | 32 | -8.11e-3 | +1.27e-2 | -1.23e-4 | -3.38e-4 |
| 178 | 3.00e-3 | 8 | 1.78e-3 | 3.24e-3 | 2.11e-3 | 1.88e-3 | 23 | -1.87e-2 | +1.45e-2 | -8.62e-4 | -6.62e-4 |
| 179 | 3.00e-3 | 12 | 1.29e-3 | 3.24e-3 | 1.73e-3 | 1.68e-3 | 18 | -2.65e-2 | +2.71e-2 | -4.98e-4 | -3.05e-4 |
| 180 | 3.00e-3 | 19 | 1.16e-3 | 3.04e-3 | 1.53e-3 | 1.42e-3 | 20 | -5.73e-2 | +4.85e-2 | -2.83e-4 | -3.86e-4 |
| 181 | 3.00e-3 | 10 | 1.37e-3 | 2.82e-3 | 1.69e-3 | 1.63e-3 | 19 | -2.90e-2 | +3.60e-2 | +7.76e-4 | +1.06e-4 |
| 182 | 3.00e-3 | 18 | 1.08e-3 | 2.89e-3 | 1.59e-3 | 1.62e-3 | 20 | -4.82e-2 | +3.08e-2 | -7.54e-4 | -1.36e-4 |
| 183 | 3.00e-3 | 8 | 1.39e-3 | 3.18e-3 | 1.75e-3 | 1.67e-3 | 23 | -4.57e-2 | +3.13e-2 | -1.05e-3 | -5.62e-4 |
| 184 | 3.00e-3 | 14 | 1.29e-3 | 3.36e-3 | 1.66e-3 | 1.37e-3 | 17 | -6.28e-2 | +4.10e-2 | -1.34e-3 | -1.39e-3 |
| 185 | 3.00e-3 | 14 | 1.18e-3 | 3.21e-3 | 1.53e-3 | 1.57e-3 | 16 | -5.71e-2 | +5.49e-2 | +7.97e-4 | +3.78e-4 |
| 186 | 3.00e-3 | 15 | 1.34e-3 | 3.00e-3 | 1.61e-3 | 1.54e-3 | 24 | -3.98e-2 | +4.70e-2 | +5.84e-4 | +1.97e-4 |
| 187 | 3.00e-3 | 11 | 1.25e-3 | 3.27e-3 | 1.76e-3 | 1.25e-3 | 15 | -3.11e-2 | +2.47e-2 | -2.10e-3 | -1.92e-3 |
| 188 | 3.00e-3 | 13 | 1.22e-3 | 3.00e-3 | 1.67e-3 | 1.67e-3 | 24 | -4.69e-2 | +6.01e-2 | +1.63e-3 | +8.79e-5 |
| 189 | 3.00e-3 | 10 | 1.70e-3 | 3.50e-3 | 1.99e-3 | 1.70e-3 | 25 | -2.69e-2 | +2.59e-2 | -3.69e-4 | -4.49e-4 |
| 190 | 3.00e-3 | 13 | 1.22e-3 | 3.28e-3 | 1.68e-3 | 1.25e-3 | 17 | -2.46e-2 | +2.10e-2 | -2.06e-3 | -1.88e-3 |
| 191 | 3.00e-3 | 14 | 1.30e-3 | 3.24e-3 | 1.79e-3 | 1.87e-3 | 24 | -3.60e-2 | +4.33e-2 | +1.13e-3 | +2.36e-4 |
| 192 | 3.00e-3 | 12 | 1.29e-3 | 3.27e-3 | 1.65e-3 | 1.31e-3 | 16 | -3.25e-2 | +2.82e-2 | -1.98e-3 | -1.42e-3 |
| 193 | 3.00e-3 | 18 | 1.25e-3 | 3.04e-3 | 1.57e-3 | 1.75e-3 | 25 | -5.55e-2 | +5.54e-2 | +9.62e-4 | +4.79e-4 |
| 194 | 3.00e-3 | 10 | 1.23e-3 | 3.25e-3 | 1.77e-3 | 2.06e-3 | 23 | -4.97e-2 | +3.31e-2 | +5.16e-4 | +9.72e-4 |
| 195 | 3.00e-3 | 14 | 1.36e-3 | 3.26e-3 | 1.68e-3 | 1.55e-3 | 15 | -3.01e-2 | +2.33e-2 | -9.04e-4 | -4.59e-5 |
| 196 | 3.00e-3 | 16 | 1.11e-3 | 2.97e-3 | 1.45e-3 | 1.71e-3 | 20 | -5.38e-2 | +5.37e-2 | +4.15e-4 | +8.86e-4 |
| 197 | 3.00e-3 | 15 | 1.09e-3 | 3.16e-3 | 1.55e-3 | 1.57e-3 | 19 | -4.26e-2 | +3.80e-2 | -5.64e-4 | +3.15e-4 |
| 198 | 3.00e-3 | 15 | 1.18e-3 | 3.13e-3 | 1.54e-3 | 1.50e-3 | 23 | -6.97e-2 | +4.89e-2 | -3.05e-4 | +1.07e-5 |
| 199 | 3.00e-3 | 13 | 1.52e-3 | 3.27e-3 | 1.81e-3 | 1.58e-3 | 28 | -3.53e-2 | +2.77e-2 | -4.91e-4 | -5.56e-4 |

