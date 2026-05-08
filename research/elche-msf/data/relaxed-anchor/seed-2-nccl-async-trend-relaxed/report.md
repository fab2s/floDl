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
| nccl-async | 0.060994 | 0.9139 | +0.0014 | 1859.6 | 293 | 40.6 | 100% | 100% | 100% | 5.6 |

## Best Mode per Model

| Model | Best Eval | Mode | Fastest (within 2% of solo-0) | Mode |
|-------|-----------|------|-------------------------------|------|
| resnet-graph | 0.9139 | nccl-async | - | - |

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
| nccl-async | 2.0021 | 0.7184 | 0.6134 | 0.5536 | 0.5267 | 0.5058 | 0.4901 | 0.4793 | 0.4811 | 0.4633 | 0.2125 | 0.1752 | 0.1605 | 0.1510 | 0.1396 | 0.0785 | 0.0737 | 0.0698 | 0.0630 | 0.0610 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | nccl-async | 0 | cuda:0 | 0.3990 | 2.7 |
| resnet-graph | nccl-async | 1 | cuda:1 | 0.3025 | 3.5 |
| resnet-graph | nccl-async | 2 | cuda:2 | 0.2985 | 3.5 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) | GPU2 Peak (MB) | GPU2 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|---------------|---------------|
| resnet-graph | nccl-async | 356 | 355 | 390 | 386 | 382 | 375 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| nccl-async | gpu1 | 1858.5 | 1.1 | epoch-boundary(199) |
| nccl-async | gpu2 | 1858.6 | 1.0 | epoch-boundary(199) |
| nccl-async | gpu0 | 1858.6 | 0.6 | epoch-boundary(199) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | nccl-async | gpu0 | 0.6s | 0.0s | 0.0s | 0.0s | 1.4s |
| resnet-graph | nccl-async | gpu1 | 1.1s | 0.0s | 0.0s | 0.0s | 2.4s |
| resnet-graph | nccl-async | gpu2 | 1.0s | 0.0s | 0.0s | 0.0s | 1.8s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | nccl-async | 69 | 0 | 293 | 40.6 | 8795/10865 | 0 | 0.0 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | nccl-async | 183.1 | 9.8% |

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
| resnet-graph | nccl-async | 178 | 293 | 0 | 6.55e-3 | -8.21e-6 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | nccl-async | 0 | 293 | 1.01e-1 | 7.38e-2 | 0.00e0 | 3.59e-1 | 27.6 | -1.47e-4 | 1.66e-3 |
| resnet-graph | nccl-async | 1 | 293 | 1.02e-1 | 7.59e-2 | 0.00e0 | 3.84e-1 | 46.4 | -1.45e-4 | 2.16e-3 |
| resnet-graph | nccl-async | 2 | 293 | 1.01e-1 | 7.51e-2 | 0.00e0 | 3.62e-1 | 25.9 | -1.46e-4 | 2.08e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | nccl-async | rank0 ↔ rank1 | +0.9975 |
| resnet-graph | nccl-async | rank0 ↔ rank2 | +0.9976 |
| resnet-graph | nccl-async | rank1 ↔ rank2 | +0.9990 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | nccl-async | 36 (0,2,3,4,5,6,7,8…143,147) | 0 (—) | — | 0,2,3,4,5,6,7,8…143,147 | — |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | nccl-async | 0e0 | 3 | 19 | 19 |
| resnet-graph | nccl-async | 0e0 | 5 | 11 | 11 |
| resnet-graph | nccl-async | 0e0 | 10 | 2 | 2 |
| resnet-graph | nccl-async | 1e-4 | 3 | 0 | 0 |
| resnet-graph | nccl-async | 1e-4 | 5 | 0 | 0 |
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
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 190 | +0.076 |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | 48 | +0.292 |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | 50 | +0.089 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | nccl-async | 290 | -0.005 | 177 | +0.201 | +0.369 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | nccl-async | 291 | 3.34e1–7.98e1 | 6.24e1 | 4.11e-3 | 4.76e-3 | 4.11e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 192 | 62–78021 | +1.398e-5 | 0.583 | +1.441e-5 | 0.594 | 92 | +7.835e-6 | 0.397 | 31–1004 | +1.061e-3 | 0.800 |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 180 | 902–78021 | +1.384e-5 | 0.640 | +1.423e-5 | 0.648 | 91 | +7.481e-6 | 0.377 | 56–1004 | +1.063e-3 | 0.894 |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | 49 | 78883–117047 | +1.157e-5 | 0.124 | +1.162e-5 | 0.126 | 41 | +1.117e-5 | 0.113 | 661–1056 | -2.549e-4 | 0.002 |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | 51 | 117775–155937 | -7.161e-6 | 0.042 | -7.360e-6 | 0.044 | 45 | -7.687e-6 | 0.047 | 637–882 | -7.846e-4 | 0.013 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | +1.061e-3 | r0: +1.042e-3, r1: +1.067e-3, r2: +1.077e-3 | r0: 0.819, r1: 0.783, r2: 0.791 | 1.03× | ✓ |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +1.063e-3 | r0: +1.046e-3, r1: +1.068e-3, r2: +1.079e-3 | r0: 0.916, r1: 0.878, r2: 0.882 | 1.03× | ✓ |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | -2.549e-4 | r0: -1.552e-4, r1: -2.896e-4, r2: -3.188e-4 | r0: 0.001, r1: 0.003, r2: 0.004 | 2.05× | ⚠ framing breaking |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | -7.846e-4 | r0: -7.947e-4, r1: -7.714e-4, r2: -7.869e-4 | r0: 0.013, r1: 0.013, r2: 0.013 | 1.03× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | nccl-async | `█▇▇▇▇██████████████████████████▅▄▄▄▅▅▅▅▅▅▅▅▅▅▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁` | `▁▇▇████████████████████████████▆▆▇▇▇█████████▆▆▇▇▇▇█████████` |

### Per-Epoch Detail: resnet-graph / nccl-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 13 | 0.00e0 | 3.84e-1 | 1.05e-1 | 7.83e-2 | 23 | -4.11e-2 | +1.74e-2 | -9.08e-3 | -5.00e-3 |
| 1 | 3.00e-1 | 10 | 6.06e-2 | 1.27e-1 | 7.57e-2 | 8.33e-2 | 31 | -3.36e-2 | +2.13e-2 | -3.44e-4 | -1.22e-3 |
| 2 | 3.00e-1 | 8 | 8.00e-2 | 1.10e-1 | 8.71e-2 | 9.13e-2 | 30 | -7.56e-3 | +8.54e-3 | +2.17e-4 | -2.88e-4 |
| 3 | 3.00e-1 | 8 | 9.07e-2 | 1.24e-1 | 9.88e-2 | 9.39e-2 | 34 | -9.60e-3 | +1.03e-2 | +1.42e-4 | -1.44e-4 |
| 4 | 3.00e-1 | 8 | 9.68e-2 | 1.40e-1 | 1.07e-1 | 9.68e-2 | 31 | -1.03e-2 | +7.97e-3 | -1.47e-4 | -2.82e-4 |
| 5 | 3.00e-1 | 9 | 8.44e-2 | 1.42e-1 | 9.93e-2 | 8.57e-2 | 26 | -9.88e-3 | +1.04e-2 | -8.25e-4 | -7.35e-4 |
| 6 | 3.00e-1 | 10 | 8.94e-2 | 1.38e-1 | 1.00e-1 | 9.50e-2 | 29 | -1.28e-2 | +1.55e-2 | +2.86e-4 | -2.44e-4 |
| 7 | 3.00e-1 | 7 | 9.11e-2 | 1.40e-1 | 1.01e-1 | 9.38e-2 | 29 | -1.31e-2 | +1.34e-2 | -6.91e-5 | -2.45e-4 |
| 8 | 3.00e-1 | 10 | 7.95e-2 | 1.42e-1 | 9.58e-2 | 8.92e-2 | 24 | -1.99e-2 | +1.10e-2 | -4.99e-4 | -3.86e-4 |
| 9 | 3.00e-1 | 10 | 8.31e-2 | 1.35e-1 | 9.92e-2 | 9.84e-2 | 29 | -1.73e-2 | +1.72e-2 | +4.23e-4 | +1.60e-5 |
| 10 | 3.00e-1 | 7 | 8.59e-2 | 1.37e-1 | 9.75e-2 | 9.35e-2 | 30 | -1.62e-2 | +1.23e-2 | -1.70e-4 | -9.13e-5 |
| 11 | 3.00e-1 | 9 | 8.06e-2 | 1.37e-1 | 9.62e-2 | 9.05e-2 | 25 | -1.39e-2 | +1.33e-2 | -2.43e-4 | -2.14e-4 |
| 12 | 3.00e-1 | 2 | 8.37e-2 | 9.49e-2 | 8.93e-2 | 9.49e-2 | 196 | -2.51e-3 | +6.37e-4 | -9.36e-4 | -3.36e-4 |
| 13 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 212 | +3.54e-3 | +3.54e-3 | +3.54e-3 | +5.24e-5 |
| 14 | 3.00e-1 | 1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 240 | -1.30e-4 | -1.30e-4 | -1.30e-4 | +3.42e-5 |
| 15 | 3.00e-1 | 2 | 1.77e-1 | 1.95e-1 | 1.86e-1 | 1.77e-1 | 189 | -5.31e-4 | +1.03e-5 | -2.60e-4 | -2.44e-5 |
| 16 | 3.00e-1 | 1 | 1.80e-1 | 1.80e-1 | 1.80e-1 | 1.80e-1 | 337 | +5.18e-5 | +5.18e-5 | +5.18e-5 | -1.68e-5 |
| 17 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 322 | +5.91e-4 | +5.91e-4 | +5.91e-4 | +4.39e-5 |
| 19 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 360 | -1.34e-4 | -1.34e-4 | -1.34e-4 | +2.62e-5 |
| 20 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 354 | +6.01e-5 | +6.01e-5 | +6.01e-5 | +2.96e-5 |
| 21 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 363 | +3.00e-5 | +3.00e-5 | +3.00e-5 | +2.96e-5 |
| 22 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 261 | +1.24e-5 | +1.24e-5 | +1.24e-5 | +2.79e-5 |
| 23 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 267 | -3.43e-4 | -3.43e-4 | -3.43e-4 | -9.17e-6 |
| 24 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 302 | -1.38e-6 | -1.38e-6 | -1.38e-6 | -8.39e-6 |
| 25 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 300 | +1.82e-4 | +1.82e-4 | +1.82e-4 | +1.06e-5 |
| 26 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 277 | -5.28e-5 | -5.28e-5 | -5.28e-5 | +4.28e-6 |
| 28 | 3.00e-1 | 2 | 2.00e-1 | 2.11e-1 | 2.05e-1 | 2.11e-1 | 289 | -5.76e-5 | +1.82e-4 | +6.24e-5 | +1.65e-5 |
| 30 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 345 | -1.34e-4 | -1.34e-4 | -1.34e-4 | +1.42e-6 |
| 31 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 316 | +2.02e-4 | +2.02e-4 | +2.02e-4 | +2.14e-5 |
| 32 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 294 | -1.71e-4 | -1.71e-4 | -1.71e-4 | +2.19e-6 |
| 33 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 289 | +3.67e-6 | +3.67e-6 | +3.67e-6 | +2.33e-6 |
| 34 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 279 | +2.43e-5 | +2.43e-5 | +2.43e-5 | +4.53e-6 |
| 35 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 309 | -9.25e-5 | -9.25e-5 | -9.25e-5 | -5.17e-6 |
| 36 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 306 | +1.29e-4 | +1.29e-4 | +1.29e-4 | +8.22e-6 |
| 38 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 338 | -9.15e-6 | -9.15e-6 | -9.15e-6 | +6.48e-6 |
| 39 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 316 | +8.31e-5 | +8.31e-5 | +8.31e-5 | +1.41e-5 |
| 40 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 317 | -2.85e-5 | -2.85e-5 | -2.85e-5 | +9.88e-6 |
| 41 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 288 | -3.80e-5 | -3.80e-5 | -3.80e-5 | +5.09e-6 |
| 42 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 288 | -5.86e-5 | -5.86e-5 | -5.86e-5 | -1.28e-6 |
| 43 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 294 | +2.54e-5 | +2.54e-5 | +2.54e-5 | +1.39e-6 |
| 44 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 295 | +1.31e-5 | +1.31e-5 | +1.31e-5 | +2.56e-6 |
| 45 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 290 | +1.77e-5 | +1.77e-5 | +1.77e-5 | +4.07e-6 |
| 46 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 310 | -6.47e-5 | -6.47e-5 | -6.47e-5 | -2.80e-6 |
| 48 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 316 | +9.15e-5 | +9.15e-5 | +9.15e-5 | +6.63e-6 |
| 49 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 319 | -4.25e-5 | -4.25e-5 | -4.25e-5 | +1.72e-6 |
| 50 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 306 | +7.34e-5 | +7.34e-5 | +7.34e-5 | +8.89e-6 |
| 51 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 310 | -4.57e-5 | -4.57e-5 | -4.57e-5 | +3.43e-6 |
| 52 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 277 | +5.47e-6 | +5.47e-6 | +5.47e-6 | +3.63e-6 |
| 53 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 286 | -1.29e-4 | -1.29e-4 | -1.29e-4 | -9.62e-6 |
| 54 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 268 | +7.09e-5 | +7.09e-5 | +7.09e-5 | -1.56e-6 |
| 55 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 306 | -9.90e-5 | -9.90e-5 | -9.90e-5 | -1.13e-5 |
| 56 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 306 | +2.04e-4 | +2.04e-4 | +2.04e-4 | +1.02e-5 |
| 57 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 276 | -2.41e-5 | -2.41e-5 | -2.41e-5 | +6.78e-6 |
| 58 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 283 | -1.43e-4 | -1.43e-4 | -1.43e-4 | -8.23e-6 |
| 59 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 278 | +4.91e-5 | +4.91e-5 | +4.91e-5 | -2.49e-6 |
| 60 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 314 | -1.24e-5 | -1.24e-5 | -1.24e-5 | -3.49e-6 |
| 61 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 288 | +1.40e-4 | +1.40e-4 | +1.40e-4 | +1.08e-5 |
| 62 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 285 | -8.24e-5 | -8.24e-5 | -8.24e-5 | +1.50e-6 |
| 63 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 254 | -8.53e-5 | -8.53e-5 | -8.53e-5 | -7.18e-6 |
| 64 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 286 | -5.98e-5 | -5.98e-5 | -5.98e-5 | -1.24e-5 |
| 65 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 270 | +2.07e-4 | +2.07e-4 | +2.07e-4 | +9.52e-6 |
| 66 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 272 | -1.55e-4 | -1.55e-4 | -1.55e-4 | -6.91e-6 |
| 67 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 320 | +4.16e-5 | +4.16e-5 | +4.16e-5 | -2.06e-6 |
| 69 | 3.00e-1 | 2 | 2.12e-1 | 2.16e-1 | 2.14e-1 | 2.16e-1 | 260 | +7.50e-5 | +8.74e-5 | +8.12e-5 | +1.37e-5 |
| 71 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 328 | -2.50e-4 | -2.50e-4 | -2.50e-4 | -1.27e-5 |
| 72 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 281 | +2.53e-4 | +2.53e-4 | +2.53e-4 | +1.39e-5 |
| 73 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 278 | -1.20e-4 | -1.20e-4 | -1.20e-4 | +5.97e-7 |
| 74 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 280 | -6.07e-7 | -6.07e-7 | -6.07e-7 | +4.76e-7 |
| 75 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 310 | +4.36e-5 | +4.36e-5 | +4.36e-5 | +4.79e-6 |
| 76 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 260 | -4.29e-6 | -4.29e-6 | -4.29e-6 | +3.89e-6 |
| 77 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 294 | -1.49e-4 | -1.49e-4 | -1.49e-4 | -1.14e-5 |
| 78 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 287 | +1.98e-4 | +1.98e-4 | +1.98e-4 | +9.50e-6 |
| 79 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 288 | -2.56e-5 | -2.56e-5 | -2.56e-5 | +5.99e-6 |
| 80 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 290 | -6.43e-5 | -6.43e-5 | -6.43e-5 | -1.04e-6 |
| 81 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 304 | +3.32e-5 | +3.32e-5 | +3.32e-5 | +2.38e-6 |
| 82 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 307 | +6.74e-5 | +6.74e-5 | +6.74e-5 | +8.88e-6 |
| 83 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 310 | -1.65e-5 | -1.65e-5 | -1.65e-5 | +6.35e-6 |
| 84 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 307 | -7.43e-7 | -7.43e-7 | -7.43e-7 | +5.64e-6 |
| 85 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 294 | +2.19e-5 | +2.19e-5 | +2.19e-5 | +7.26e-6 |
| 86 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 276 | -8.22e-5 | -8.22e-5 | -8.22e-5 | -1.69e-6 |
| 87 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 285 | -1.19e-4 | -1.19e-4 | -1.19e-4 | -1.34e-5 |
| 89 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 387 | +9.24e-5 | +9.24e-5 | +9.24e-5 | -2.86e-6 |
| 90 | 3.00e-1 | 1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 308 | +2.40e-4 | +2.40e-4 | +2.40e-4 | +2.14e-5 |
| 91 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 322 | -2.37e-4 | -2.37e-4 | -2.37e-4 | -4.36e-6 |
| 92 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 302 | +9.28e-5 | +9.28e-5 | +9.28e-5 | +5.35e-6 |
| 93 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 291 | -7.24e-5 | -7.24e-5 | -7.24e-5 | -2.43e-6 |
| 94 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 304 | -8.11e-5 | -8.11e-5 | -8.11e-5 | -1.03e-5 |
| 95 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 287 | +1.24e-4 | +1.24e-4 | +1.24e-4 | +3.11e-6 |
| 96 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 268 | -5.50e-5 | -5.50e-5 | -5.50e-5 | -2.70e-6 |
| 97 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 289 | -6.21e-5 | -6.21e-5 | -6.21e-5 | -8.64e-6 |
| 98 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 280 | +9.80e-5 | +9.80e-5 | +9.80e-5 | +2.03e-6 |
| 99 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 293 | -8.06e-5 | -8.06e-5 | -8.06e-5 | -6.23e-6 |
| 100 | 3.00e-2 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 319 | +1.50e-5 | +1.50e-5 | +1.50e-5 | -4.11e-6 |
| 102 | 3.00e-2 | 1 | 1.20e-1 | 1.20e-1 | 1.20e-1 | 1.20e-1 | 336 | -1.63e-3 | -1.63e-3 | -1.63e-3 | -1.67e-4 |
| 103 | 3.00e-2 | 1 | 2.48e-2 | 2.48e-2 | 2.48e-2 | 2.48e-2 | 278 | -5.66e-3 | -5.66e-3 | -5.66e-3 | -7.17e-4 |
| 104 | 3.00e-2 | 1 | 2.45e-2 | 2.45e-2 | 2.45e-2 | 2.45e-2 | 286 | -4.21e-5 | -4.21e-5 | -4.21e-5 | -6.49e-4 |
| 105 | 3.00e-2 | 1 | 2.57e-2 | 2.57e-2 | 2.57e-2 | 2.57e-2 | 305 | +1.58e-4 | +1.58e-4 | +1.58e-4 | -5.68e-4 |
| 106 | 3.00e-2 | 1 | 2.70e-2 | 2.70e-2 | 2.70e-2 | 2.70e-2 | 289 | +1.78e-4 | +1.78e-4 | +1.78e-4 | -4.94e-4 |
| 107 | 3.00e-2 | 1 | 2.77e-2 | 2.77e-2 | 2.77e-2 | 2.77e-2 | 322 | +7.16e-5 | +7.16e-5 | +7.16e-5 | -4.37e-4 |
| 108 | 3.00e-2 | 1 | 2.97e-2 | 2.97e-2 | 2.97e-2 | 2.97e-2 | 315 | +2.28e-4 | +2.28e-4 | +2.28e-4 | -3.71e-4 |
| 109 | 3.00e-2 | 1 | 3.07e-2 | 3.07e-2 | 3.07e-2 | 3.07e-2 | 296 | +1.05e-4 | +1.05e-4 | +1.05e-4 | -3.23e-4 |
| 110 | 3.00e-2 | 1 | 3.00e-2 | 3.00e-2 | 3.00e-2 | 3.00e-2 | 278 | -7.57e-5 | -7.57e-5 | -7.57e-5 | -2.98e-4 |
| 111 | 3.00e-2 | 1 | 3.09e-2 | 3.09e-2 | 3.09e-2 | 3.09e-2 | 296 | +9.79e-5 | +9.79e-5 | +9.79e-5 | -2.59e-4 |
| 112 | 3.00e-2 | 1 | 3.17e-2 | 3.17e-2 | 3.17e-2 | 3.17e-2 | 290 | +8.44e-5 | +8.44e-5 | +8.44e-5 | -2.24e-4 |
| 113 | 3.00e-2 | 1 | 3.28e-2 | 3.28e-2 | 3.28e-2 | 3.28e-2 | 286 | +1.19e-4 | +1.19e-4 | +1.19e-4 | -1.90e-4 |
| 115 | 3.00e-2 | 1 | 3.31e-2 | 3.31e-2 | 3.31e-2 | 3.31e-2 | 398 | +2.23e-5 | +2.23e-5 | +2.23e-5 | -1.69e-4 |
| 116 | 3.00e-2 | 2 | 3.45e-2 | 3.83e-2 | 3.64e-2 | 3.45e-2 | 251 | -4.08e-4 | +5.13e-4 | +5.22e-5 | -1.32e-4 |
| 118 | 3.00e-2 | 2 | 3.43e-2 | 3.68e-2 | 3.55e-2 | 3.68e-2 | 231 | -2.15e-5 | +3.04e-4 | +1.41e-4 | -7.80e-5 |
| 120 | 3.00e-2 | 1 | 3.42e-2 | 3.42e-2 | 3.42e-2 | 3.42e-2 | 306 | -2.40e-4 | -2.40e-4 | -2.40e-4 | -9.42e-5 |
| 121 | 3.00e-2 | 2 | 3.67e-2 | 3.88e-2 | 3.77e-2 | 3.67e-2 | 231 | -2.32e-4 | +4.88e-4 | +1.28e-4 | -5.56e-5 |
| 123 | 3.00e-2 | 1 | 3.60e-2 | 3.60e-2 | 3.60e-2 | 3.60e-2 | 330 | -5.96e-5 | -5.96e-5 | -5.96e-5 | -5.60e-5 |
| 124 | 3.00e-2 | 1 | 4.21e-2 | 4.21e-2 | 4.21e-2 | 4.21e-2 | 259 | +6.01e-4 | +6.01e-4 | +6.01e-4 | +9.66e-6 |
| 125 | 3.00e-2 | 2 | 3.96e-2 | 4.23e-2 | 4.09e-2 | 4.23e-2 | 245 | -2.11e-4 | +2.66e-4 | +2.72e-5 | +1.54e-5 |
| 127 | 3.00e-2 | 2 | 4.08e-2 | 4.44e-2 | 4.26e-2 | 4.44e-2 | 245 | -1.18e-4 | +3.47e-4 | +1.15e-4 | +3.66e-5 |
| 128 | 3.00e-2 | 1 | 4.13e-2 | 4.13e-2 | 4.13e-2 | 4.13e-2 | 268 | -2.75e-4 | -2.75e-4 | -2.75e-4 | +5.37e-6 |
| 129 | 3.00e-2 | 1 | 4.30e-2 | 4.30e-2 | 4.30e-2 | 4.30e-2 | 270 | +1.52e-4 | +1.52e-4 | +1.52e-4 | +2.00e-5 |
| 130 | 3.00e-2 | 1 | 4.43e-2 | 4.43e-2 | 4.43e-2 | 4.43e-2 | 287 | +1.06e-4 | +1.06e-4 | +1.06e-4 | +2.86e-5 |
| 131 | 3.00e-2 | 1 | 4.56e-2 | 4.56e-2 | 4.56e-2 | 4.56e-2 | 297 | +9.35e-5 | +9.35e-5 | +9.35e-5 | +3.51e-5 |
| 133 | 3.00e-2 | 2 | 4.71e-2 | 4.97e-2 | 4.84e-2 | 4.97e-2 | 245 | +1.03e-4 | +2.19e-4 | +1.61e-4 | +5.96e-5 |
| 135 | 3.00e-2 | 2 | 4.52e-2 | 5.18e-2 | 4.85e-2 | 5.18e-2 | 245 | -2.92e-4 | +5.59e-4 | +1.33e-4 | +7.79e-5 |
| 137 | 3.00e-2 | 1 | 4.74e-2 | 4.74e-2 | 4.74e-2 | 4.74e-2 | 328 | -2.70e-4 | -2.70e-4 | -2.70e-4 | +4.31e-5 |
| 138 | 3.00e-2 | 1 | 5.31e-2 | 5.31e-2 | 5.31e-2 | 5.31e-2 | 277 | +4.10e-4 | +4.10e-4 | +4.10e-4 | +7.97e-5 |
| 139 | 3.00e-2 | 2 | 5.04e-2 | 5.12e-2 | 5.08e-2 | 5.04e-2 | 245 | -1.46e-4 | -6.80e-5 | -1.07e-4 | +4.46e-5 |
| 140 | 3.00e-2 | 1 | 5.14e-2 | 5.14e-2 | 5.14e-2 | 5.14e-2 | 250 | +8.37e-5 | +8.37e-5 | +8.37e-5 | +4.85e-5 |
| 141 | 3.00e-2 | 1 | 5.15e-2 | 5.15e-2 | 5.15e-2 | 5.15e-2 | 257 | +7.61e-6 | +7.61e-6 | +7.61e-6 | +4.44e-5 |
| 142 | 3.00e-2 | 1 | 5.19e-2 | 5.19e-2 | 5.19e-2 | 5.19e-2 | 296 | +2.34e-5 | +2.34e-5 | +2.34e-5 | +4.23e-5 |
| 143 | 3.00e-2 | 1 | 5.66e-2 | 5.66e-2 | 5.66e-2 | 5.66e-2 | 267 | +3.24e-4 | +3.24e-4 | +3.24e-4 | +7.05e-5 |
| 144 | 3.00e-2 | 1 | 5.48e-2 | 5.48e-2 | 5.48e-2 | 5.48e-2 | 267 | -1.21e-4 | -1.21e-4 | -1.21e-4 | +5.13e-5 |
| 145 | 3.00e-2 | 1 | 5.47e-2 | 5.47e-2 | 5.47e-2 | 5.47e-2 | 265 | -5.52e-6 | -5.52e-6 | -5.52e-6 | +4.56e-5 |
| 146 | 3.00e-2 | 1 | 5.58e-2 | 5.58e-2 | 5.58e-2 | 5.58e-2 | 287 | +7.11e-5 | +7.11e-5 | +7.11e-5 | +4.82e-5 |
| 147 | 3.00e-2 | 1 | 5.72e-2 | 5.72e-2 | 5.72e-2 | 5.72e-2 | 264 | +9.59e-5 | +9.59e-5 | +9.59e-5 | +5.30e-5 |
| 148 | 3.00e-2 | 1 | 5.62e-2 | 5.62e-2 | 5.62e-2 | 5.62e-2 | 270 | -6.92e-5 | -6.92e-5 | -6.92e-5 | +4.08e-5 |
| 149 | 3.00e-2 | 1 | 5.70e-2 | 5.70e-2 | 5.70e-2 | 5.70e-2 | 247 | +6.02e-5 | +6.02e-5 | +6.02e-5 | +4.27e-5 |
| 150 | 3.00e-3 | 1 | 5.66e-2 | 5.66e-2 | 5.66e-2 | 5.66e-2 | 249 | -2.88e-5 | -2.88e-5 | -2.88e-5 | +3.56e-5 |
| 151 | 3.00e-3 | 1 | 3.60e-2 | 3.60e-2 | 3.60e-2 | 3.60e-2 | 248 | -1.82e-3 | -1.82e-3 | -1.82e-3 | -1.50e-4 |
| 152 | 3.00e-3 | 1 | 5.60e-3 | 5.60e-3 | 5.60e-3 | 5.60e-3 | 273 | -6.82e-3 | -6.82e-3 | -6.82e-3 | -8.17e-4 |
| 153 | 3.00e-3 | 1 | 5.66e-3 | 5.66e-3 | 5.66e-3 | 5.66e-3 | 278 | +3.90e-5 | +3.90e-5 | +3.90e-5 | -7.31e-4 |
| 154 | 3.00e-3 | 1 | 5.74e-3 | 5.74e-3 | 5.74e-3 | 5.74e-3 | 287 | +4.97e-5 | +4.97e-5 | +4.97e-5 | -6.53e-4 |
| 155 | 3.00e-3 | 1 | 5.82e-3 | 5.82e-3 | 5.82e-3 | 5.82e-3 | 318 | +3.85e-5 | +3.85e-5 | +3.85e-5 | -5.84e-4 |
| 156 | 3.00e-3 | 1 | 6.23e-3 | 6.23e-3 | 6.23e-3 | 6.23e-3 | 272 | +2.56e-4 | +2.56e-4 | +2.56e-4 | -5.00e-4 |
| 157 | 3.00e-3 | 1 | 5.78e-3 | 5.78e-3 | 5.78e-3 | 5.78e-3 | 252 | -3.03e-4 | -3.03e-4 | -3.03e-4 | -4.80e-4 |
| 158 | 3.00e-3 | 1 | 5.62e-3 | 5.62e-3 | 5.62e-3 | 5.62e-3 | 260 | -1.09e-4 | -1.09e-4 | -1.09e-4 | -4.43e-4 |
| 159 | 3.00e-3 | 1 | 5.67e-3 | 5.67e-3 | 5.67e-3 | 5.67e-3 | 255 | +3.65e-5 | +3.65e-5 | +3.65e-5 | -3.95e-4 |
| 160 | 3.00e-3 | 1 | 6.03e-3 | 6.03e-3 | 6.03e-3 | 6.03e-3 | 277 | +2.26e-4 | +2.26e-4 | +2.26e-4 | -3.33e-4 |
| 161 | 3.00e-3 | 1 | 5.90e-3 | 5.90e-3 | 5.90e-3 | 5.90e-3 | 292 | -7.71e-5 | -7.71e-5 | -7.71e-5 | -3.08e-4 |
| 162 | 3.00e-3 | 1 | 6.21e-3 | 6.21e-3 | 6.21e-3 | 6.21e-3 | 282 | +1.84e-4 | +1.84e-4 | +1.84e-4 | -2.58e-4 |
| 163 | 3.00e-3 | 1 | 6.10e-3 | 6.10e-3 | 6.10e-3 | 6.10e-3 | 280 | -6.43e-5 | -6.43e-5 | -6.43e-5 | -2.39e-4 |
| 164 | 3.00e-3 | 1 | 6.22e-3 | 6.22e-3 | 6.22e-3 | 6.22e-3 | 241 | +7.68e-5 | +7.68e-5 | +7.68e-5 | -2.07e-4 |
| 165 | 3.00e-3 | 1 | 5.66e-3 | 5.66e-3 | 5.66e-3 | 5.66e-3 | 250 | -3.78e-4 | -3.78e-4 | -3.78e-4 | -2.24e-4 |
| 166 | 3.00e-3 | 1 | 5.93e-3 | 5.93e-3 | 5.93e-3 | 5.93e-3 | 235 | +1.98e-4 | +1.98e-4 | +1.98e-4 | -1.82e-4 |
| 167 | 3.00e-3 | 2 | 5.87e-3 | 6.07e-3 | 5.97e-3 | 6.07e-3 | 241 | -4.01e-5 | +1.42e-4 | +5.09e-5 | -1.37e-4 |
| 169 | 3.00e-3 | 2 | 5.85e-3 | 6.37e-3 | 6.11e-3 | 6.37e-3 | 241 | -1.21e-4 | +3.57e-4 | +1.18e-4 | -8.62e-5 |
| 171 | 3.00e-3 | 1 | 5.81e-3 | 5.81e-3 | 5.81e-3 | 5.81e-3 | 319 | -2.86e-4 | -2.86e-4 | -2.86e-4 | -1.06e-4 |
| 172 | 3.00e-3 | 2 | 6.51e-3 | 6.56e-3 | 6.53e-3 | 6.51e-3 | 254 | -2.85e-5 | +4.18e-4 | +1.95e-4 | -5.12e-5 |
| 174 | 3.00e-3 | 1 | 6.00e-3 | 6.00e-3 | 6.00e-3 | 6.00e-3 | 326 | -2.49e-4 | -2.49e-4 | -2.49e-4 | -7.10e-5 |
| 175 | 3.00e-3 | 2 | 6.46e-3 | 6.74e-3 | 6.60e-3 | 6.46e-3 | 259 | -1.63e-4 | +4.41e-4 | +1.39e-4 | -3.41e-5 |
| 177 | 3.00e-3 | 1 | 6.47e-3 | 6.47e-3 | 6.47e-3 | 6.47e-3 | 320 | +3.14e-6 | +3.14e-6 | +3.14e-6 | -3.04e-5 |
| 178 | 3.00e-3 | 1 | 7.09e-3 | 7.09e-3 | 7.09e-3 | 7.09e-3 | 282 | +3.26e-4 | +3.26e-4 | +3.26e-4 | +5.29e-6 |
| 179 | 3.00e-3 | 1 | 6.59e-3 | 6.59e-3 | 6.59e-3 | 6.59e-3 | 265 | -2.77e-4 | -2.77e-4 | -2.77e-4 | -2.29e-5 |
| 180 | 3.00e-3 | 1 | 6.58e-3 | 6.58e-3 | 6.58e-3 | 6.58e-3 | 278 | -7.20e-7 | -7.20e-7 | -7.20e-7 | -2.07e-5 |
| 181 | 3.00e-3 | 1 | 6.47e-3 | 6.47e-3 | 6.47e-3 | 6.47e-3 | 293 | -5.79e-5 | -5.79e-5 | -5.79e-5 | -2.44e-5 |
| 182 | 3.00e-3 | 1 | 6.64e-3 | 6.64e-3 | 6.64e-3 | 6.64e-3 | 255 | +1.01e-4 | +1.01e-4 | +1.01e-4 | -1.19e-5 |
| 183 | 3.00e-3 | 1 | 6.30e-3 | 6.30e-3 | 6.30e-3 | 6.30e-3 | 252 | -2.09e-4 | -2.09e-4 | -2.09e-4 | -3.16e-5 |
| 184 | 3.00e-3 | 2 | 6.41e-3 | 6.43e-3 | 6.42e-3 | 6.43e-3 | 241 | +8.98e-6 | +6.47e-5 | +3.68e-5 | -1.89e-5 |
| 186 | 3.00e-3 | 2 | 6.27e-3 | 6.86e-3 | 6.56e-3 | 6.86e-3 | 235 | -8.46e-5 | +3.83e-4 | +1.49e-4 | +1.54e-5 |
| 187 | 3.00e-3 | 1 | 6.27e-3 | 6.27e-3 | 6.27e-3 | 6.27e-3 | 247 | -3.66e-4 | -3.66e-4 | -3.66e-4 | -2.28e-5 |
| 188 | 3.00e-3 | 1 | 6.37e-3 | 6.37e-3 | 6.37e-3 | 6.37e-3 | 260 | +6.52e-5 | +6.52e-5 | +6.52e-5 | -1.40e-5 |
| 189 | 3.00e-3 | 1 | 6.61e-3 | 6.61e-3 | 6.61e-3 | 6.61e-3 | 281 | +1.28e-4 | +1.28e-4 | +1.28e-4 | +2.59e-7 |
| 190 | 3.00e-3 | 1 | 6.71e-3 | 6.71e-3 | 6.71e-3 | 6.71e-3 | 309 | +4.84e-5 | +4.84e-5 | +4.84e-5 | +5.08e-6 |
| 191 | 3.00e-3 | 1 | 7.06e-3 | 7.06e-3 | 7.06e-3 | 7.06e-3 | 241 | +2.12e-4 | +2.12e-4 | +2.12e-4 | +2.58e-5 |
| 192 | 3.00e-3 | 1 | 6.41e-3 | 6.41e-3 | 6.41e-3 | 6.41e-3 | 260 | -3.69e-4 | -3.69e-4 | -3.69e-4 | -1.37e-5 |
| 193 | 3.00e-3 | 1 | 6.47e-3 | 6.47e-3 | 6.47e-3 | 6.47e-3 | 266 | +3.49e-5 | +3.49e-5 | +3.49e-5 | -8.80e-6 |
| 194 | 3.00e-3 | 1 | 6.70e-3 | 6.70e-3 | 6.70e-3 | 6.70e-3 | 270 | +1.28e-4 | +1.28e-4 | +1.28e-4 | +4.91e-6 |
| 195 | 3.00e-3 | 1 | 6.79e-3 | 6.79e-3 | 6.79e-3 | 6.79e-3 | 264 | +5.30e-5 | +5.30e-5 | +5.30e-5 | +9.72e-6 |
| 196 | 3.00e-3 | 1 | 6.59e-3 | 6.59e-3 | 6.59e-3 | 6.59e-3 | 292 | -1.06e-4 | -1.06e-4 | -1.06e-4 | -1.85e-6 |
| 197 | 3.00e-3 | 1 | 7.21e-3 | 7.21e-3 | 7.21e-3 | 7.21e-3 | 278 | +3.26e-4 | +3.26e-4 | +3.26e-4 | +3.09e-5 |
| 198 | 3.00e-3 | 1 | 6.92e-3 | 6.92e-3 | 6.92e-3 | 6.92e-3 | 257 | -1.59e-4 | -1.59e-4 | -1.59e-4 | +1.19e-5 |
| 199 | 3.00e-3 | 1 | 6.55e-3 | 6.55e-3 | 6.55e-3 | 6.55e-3 | 289 | -1.90e-4 | -1.90e-4 | -1.90e-4 | -8.21e-6 |

