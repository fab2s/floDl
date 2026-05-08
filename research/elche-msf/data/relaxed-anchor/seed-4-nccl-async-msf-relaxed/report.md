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
| nccl-async | 0.061020 | 0.9132 | +0.0007 | 1846.8 | 442 | 37.5 | 100% | 100% | 100% | 9.7 |

## Best Mode per Model

| Model | Best Eval | Mode | Fastest (within 2% of solo-0) | Mode |
|-------|-----------|------|-------------------------------|------|
| resnet-graph | 0.9132 | nccl-async | - | - |

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
| nccl-async | 1.9450 | 0.7385 | 0.5651 | 0.4982 | 0.4731 | 0.4987 | 0.4895 | 0.4742 | 0.4706 | 0.4610 | 0.2134 | 0.1788 | 0.1600 | 0.1505 | 0.1430 | 0.0822 | 0.0753 | 0.0679 | 0.0632 | 0.0610 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | nccl-async | 0 | cuda:0 | 0.4073 | 2.7 |
| resnet-graph | nccl-async | 1 | cuda:1 | 0.2993 | 3.5 |
| resnet-graph | nccl-async | 2 | cuda:2 | 0.2934 | 3.6 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) | GPU2 Peak (MB) | GPU2 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|---------------|---------------|
| resnet-graph | nccl-async | 356 | 355 | 388 | 384 | 384 | 380 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| nccl-async | gpu1 | 1844.1 | 2.7 | epoch-boundary(199) |
| nccl-async | gpu2 | 1844.1 | 2.7 | epoch-boundary(199) |
| nccl-async | gpu0 | 1845.8 | 0.6 | epoch-boundary(199) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | nccl-async | gpu0 | 0.6s | 0.0s | 0.0s | 0.0s | 2.0s |
| resnet-graph | nccl-async | gpu1 | 2.7s | 0.0s | 0.0s | 0.0s | 4.2s |
| resnet-graph | nccl-async | gpu2 | 2.7s | 0.0s | 0.0s | 0.0s | 3.5s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | nccl-async | 16 | 0 | 442 | 37.5 | 1367/10733 | 0 | 0.0 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | nccl-async | 186.7 | 10.1% |

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
| resnet-graph | nccl-async | 180 | 442 | 0 | 7.29e-3 | +1.56e-5 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | nccl-async | 0 | 442 | 9.72e-2 | 5.29e-2 | 0.00e0 | 3.90e-1 | 42.8 | -9.96e-5 | 1.45e-3 |
| resnet-graph | nccl-async | 1 | 442 | 9.82e-2 | 5.55e-2 | 0.00e0 | 4.49e-1 | 39.1 | -1.07e-4 | 2.02e-3 |
| resnet-graph | nccl-async | 2 | 442 | 9.74e-2 | 5.51e-2 | 0.00e0 | 4.41e-1 | 18.1 | -1.16e-4 | 1.98e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | nccl-async | rank0 ↔ rank1 | +0.9909 |
| resnet-graph | nccl-async | rank0 ↔ rank2 | +0.9915 |
| resnet-graph | nccl-async | rank1 ↔ rank2 | +0.9983 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | nccl-async | 63 (0,1,2,3,4,5,6,7…138,142) | 0 (—) | — | 0,1,2,3,4,5,6,7…138,142 | — |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | nccl-async | 0e0 | 3 | 28 | 28 |
| resnet-graph | nccl-async | 0e0 | 5 | 11 | 11 |
| resnet-graph | nccl-async | 0e0 | 10 | 4 | 4 |
| resnet-graph | nccl-async | 1e-4 | 3 | 2 | 2 |
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
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 348 | +0.081 |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | 44 | +0.197 |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | 45 | +0.105 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | nccl-async | 439 | +0.012 | 179 | +0.216 | +0.426 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | nccl-async | 440 | 3.39e1–7.97e1 | 6.59e1 | 2.79e-3 | 3.86e-3 | 3.91e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 350 | 71–77882 | +1.245e-5 | 0.542 | +1.285e-5 | 0.563 | 97 | +1.410e-5 | 0.774 | 34–977 | +9.754e-4 | 0.681 |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 339 | 876–77882 | +1.251e-5 | 0.628 | +1.287e-5 | 0.646 | 96 | +1.408e-5 | 0.768 | 72–977 | +9.688e-4 | 0.798 |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | 45 | 78689–116903 | +1.125e-5 | 0.091 | +1.105e-5 | 0.088 | 42 | +1.044e-5 | 0.073 | 719–982 | -2.648e-3 | 0.108 |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | 46 | 117702–156021 | -9.096e-6 | 0.052 | -9.179e-6 | 0.053 | 41 | -1.026e-5 | 0.061 | 733–1021 | -1.381e-3 | 0.037 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | +9.754e-4 | r0: +9.614e-4, r1: +9.804e-4, r2: +9.871e-4 | r0: 0.734, r1: 0.646, r2: 0.650 | 1.03× | ✓ |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +9.688e-4 | r0: +9.555e-4, r1: +9.750e-4, r2: +9.784e-4 | r0: 0.874, r1: 0.748, r2: 0.758 | 1.02× | ✓ |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | -2.648e-3 | r0: -2.573e-3, r1: -2.695e-3, r2: -2.675e-3 | r0: 0.102, r1: 0.111, r2: 0.110 | 1.05× | ✓ |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | -1.381e-3 | r0: -1.381e-3, r1: -1.352e-3, r2: -1.410e-3 | r0: 0.037, r1: 0.036, r2: 0.039 | 1.04× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | nccl-async | `█▇▇▇▇▇▇▇▇▇▇▇▇▇▇██████████████████▄▄▄▄▅▅▅▅▅▅▅▅▅▅▁▁▁▁▁▁▁▁▁▁▁▁▁` | `▁█▇█▇█▇▇███▇█████████████████████▇▇▇▇██████████▆▇▇▇█████████` |

### Per-Epoch Detail: resnet-graph / nccl-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 12 | 0.00e0 | 4.49e-1 | 1.07e-1 | 8.75e-2 | 38 | -4.66e-2 | +9.96e-3 | -1.14e-2 | -6.86e-3 |
| 1 | 3.00e-1 | 7 | 7.53e-2 | 1.10e-1 | 8.61e-2 | 8.26e-2 | 38 | -8.93e-3 | +4.91e-3 | -3.68e-4 | -2.78e-3 |
| 2 | 3.00e-1 | 7 | 8.44e-2 | 1.14e-1 | 9.25e-2 | 9.67e-2 | 36 | -7.30e-3 | +7.99e-3 | +5.75e-4 | -8.79e-4 |
| 3 | 3.00e-1 | 7 | 8.67e-2 | 1.22e-1 | 9.90e-2 | 1.05e-1 | 39 | -9.79e-3 | +7.98e-3 | +3.98e-4 | -1.57e-4 |
| 4 | 3.00e-1 | 9 | 8.89e-2 | 1.28e-1 | 9.79e-2 | 9.24e-2 | 39 | -9.29e-3 | +6.93e-3 | -3.53e-4 | -2.80e-4 |
| 5 | 3.00e-1 | 5 | 9.50e-2 | 1.27e-1 | 1.05e-1 | 1.02e-1 | 41 | -6.03e-3 | +7.54e-3 | +5.00e-4 | -2.44e-5 |
| 6 | 3.00e-1 | 11 | 8.23e-2 | 1.27e-1 | 9.70e-2 | 8.89e-2 | 34 | -6.21e-3 | +5.81e-3 | -4.66e-4 | -3.48e-4 |
| 7 | 3.00e-1 | 5 | 9.21e-2 | 1.40e-1 | 1.08e-1 | 1.06e-1 | 34 | -1.17e-2 | +1.01e-2 | +7.08e-4 | +2.80e-5 |
| 8 | 3.00e-1 | 7 | 8.60e-2 | 1.39e-1 | 1.01e-1 | 9.74e-2 | 35 | -1.09e-2 | +9.67e-3 | -4.93e-4 | -2.11e-4 |
| 9 | 3.00e-1 | 9 | 9.49e-2 | 1.28e-1 | 1.02e-1 | 1.01e-1 | 34 | -8.11e-3 | +6.92e-3 | +5.72e-5 | -5.37e-5 |
| 10 | 3.00e-1 | 5 | 8.96e-2 | 1.32e-1 | 1.02e-1 | 9.86e-2 | 35 | -1.14e-2 | +1.02e-2 | -1.00e-4 | -9.73e-5 |
| 11 | 3.00e-1 | 7 | 9.19e-2 | 1.37e-1 | 1.02e-1 | 1.03e-1 | 36 | -1.24e-2 | +1.17e-2 | +2.49e-4 | +6.83e-5 |
| 12 | 3.00e-1 | 7 | 9.06e-2 | 1.37e-1 | 1.00e-1 | 9.48e-2 | 35 | -1.17e-2 | +1.05e-2 | -1.10e-4 | -6.31e-5 |
| 13 | 3.00e-1 | 8 | 8.41e-2 | 1.35e-1 | 9.49e-2 | 8.91e-2 | 39 | -1.45e-2 | +1.02e-2 | -6.19e-4 | -3.67e-4 |
| 14 | 3.00e-1 | 9 | 8.80e-2 | 1.37e-1 | 9.82e-2 | 9.18e-2 | 34 | -8.87e-3 | +7.72e-3 | -2.66e-4 | -3.27e-4 |
| 15 | 3.00e-1 | 5 | 9.44e-2 | 1.34e-1 | 1.05e-1 | 1.01e-1 | 41 | -8.98e-3 | +9.84e-3 | +5.77e-4 | -2.48e-5 |
| 16 | 3.00e-1 | 6 | 9.89e-2 | 1.39e-1 | 1.08e-1 | 1.07e-1 | 44 | -7.43e-3 | +7.53e-3 | +1.95e-4 | +5.60e-5 |
| 17 | 3.00e-1 | 9 | 9.84e-2 | 1.39e-1 | 1.08e-1 | 9.84e-2 | 40 | -7.82e-3 | +5.87e-3 | -3.16e-4 | -2.41e-4 |
| 18 | 3.00e-1 | 4 | 8.78e-2 | 1.40e-1 | 1.05e-1 | 8.78e-2 | 36 | -1.41e-2 | +9.29e-3 | -1.18e-3 | -6.81e-4 |
| 19 | 3.00e-1 | 6 | 9.31e-2 | 1.28e-1 | 1.04e-1 | 9.97e-2 | 40 | -8.04e-3 | +7.52e-3 | +4.15e-4 | -2.48e-4 |
| 20 | 3.00e-1 | 9 | 9.38e-2 | 1.38e-1 | 1.04e-1 | 1.03e-1 | 40 | -8.72e-3 | +8.16e-3 | -2.20e-5 | -1.01e-4 |
| 21 | 3.00e-1 | 4 | 9.76e-2 | 1.39e-1 | 1.09e-1 | 1.00e-1 | 41 | -8.66e-3 | +8.45e-3 | -4.14e-5 | -1.38e-4 |
| 22 | 3.00e-1 | 6 | 9.30e-2 | 1.34e-1 | 1.04e-1 | 1.03e-1 | 39 | -9.84e-3 | +6.90e-3 | -8.11e-6 | -7.86e-5 |
| 23 | 3.00e-1 | 9 | 9.42e-2 | 1.35e-1 | 1.04e-1 | 9.49e-2 | 32 | -7.50e-3 | +7.39e-3 | -2.86e-4 | -2.99e-4 |
| 24 | 3.00e-1 | 5 | 9.16e-2 | 1.30e-1 | 1.05e-1 | 1.03e-1 | 41 | -7.25e-3 | +9.31e-3 | +5.89e-4 | +5.35e-6 |
| 25 | 3.00e-1 | 6 | 9.94e-2 | 1.39e-1 | 1.09e-1 | 1.01e-1 | 36 | -8.36e-3 | +7.41e-3 | -1.59e-4 | -1.51e-4 |
| 26 | 3.00e-1 | 6 | 9.41e-2 | 1.30e-1 | 1.04e-1 | 1.07e-1 | 41 | -8.40e-3 | +8.87e-3 | +4.33e-4 | +1.08e-4 |
| 27 | 3.00e-1 | 9 | 9.46e-2 | 1.45e-1 | 1.06e-1 | 9.87e-2 | 38 | -8.32e-3 | +7.20e-3 | -2.67e-4 | -1.43e-4 |
| 28 | 3.00e-1 | 4 | 9.69e-2 | 1.30e-1 | 1.08e-1 | 1.07e-1 | 40 | -6.91e-3 | +7.03e-3 | +6.37e-4 | +9.75e-5 |
| 29 | 3.00e-1 | 8 | 9.82e-2 | 1.41e-1 | 1.09e-1 | 1.07e-1 | 43 | -6.71e-3 | +8.37e-3 | +1.25e-4 | +7.98e-5 |
| 30 | 3.00e-1 | 4 | 1.02e-1 | 1.41e-1 | 1.13e-1 | 1.04e-1 | 40 | -8.06e-3 | +7.43e-3 | -2.06e-4 | -7.61e-5 |
| 31 | 3.00e-1 | 6 | 9.32e-2 | 1.42e-1 | 1.03e-1 | 9.48e-2 | 38 | -1.24e-2 | +9.42e-3 | -4.64e-4 | -3.03e-4 |
| 32 | 3.00e-1 | 8 | 9.80e-2 | 1.40e-1 | 1.09e-1 | 1.11e-1 | 47 | -7.99e-3 | +8.77e-3 | +4.03e-4 | +5.72e-5 |
| 33 | 3.00e-1 | 5 | 1.00e-1 | 1.31e-1 | 1.10e-1 | 1.03e-1 | 44 | -6.57e-3 | +4.29e-3 | -3.43e-4 | -1.35e-4 |
| 34 | 3.00e-1 | 6 | 1.02e-1 | 1.40e-1 | 1.11e-1 | 1.02e-1 | 40 | -6.25e-3 | +5.53e-3 | -1.93e-4 | -2.30e-4 |
| 35 | 3.00e-1 | 6 | 9.76e-2 | 1.40e-1 | 1.07e-1 | 1.04e-1 | 34 | -9.06e-3 | +8.48e-3 | +4.84e-5 | -1.26e-4 |
| 36 | 3.00e-1 | 6 | 9.13e-2 | 1.45e-1 | 1.05e-1 | 9.47e-2 | 37 | -1.10e-2 | +1.35e-2 | +2.11e-4 | -8.88e-5 |
| 37 | 3.00e-1 | 7 | 8.98e-2 | 1.35e-1 | 1.03e-1 | 1.03e-1 | 45 | -1.12e-2 | +9.20e-3 | +1.95e-4 | +3.38e-5 |
| 38 | 3.00e-1 | 6 | 1.02e-1 | 1.32e-1 | 1.11e-1 | 1.14e-1 | 45 | -5.74e-3 | +4.85e-3 | +3.38e-4 | +1.73e-4 |
| 39 | 3.00e-1 | 6 | 9.92e-2 | 1.42e-1 | 1.10e-1 | 9.92e-2 | 40 | -8.35e-3 | +5.62e-3 | -5.49e-4 | -2.04e-4 |
| 40 | 3.00e-1 | 8 | 1.00e-1 | 1.42e-1 | 1.09e-1 | 1.06e-1 | 47 | -6.75e-3 | +7.48e-3 | +1.04e-4 | -6.99e-5 |
| 41 | 3.00e-1 | 4 | 1.02e-1 | 1.49e-1 | 1.18e-1 | 1.07e-1 | 38 | -9.06e-3 | +6.70e-3 | -1.15e-4 | -1.47e-4 |
| 42 | 3.00e-1 | 6 | 9.08e-2 | 1.41e-1 | 1.08e-1 | 1.09e-1 | 47 | -1.11e-2 | +9.58e-3 | +3.57e-4 | +8.28e-5 |
| 43 | 3.00e-1 | 8 | 9.62e-2 | 1.43e-1 | 1.08e-1 | 9.99e-2 | 35 | -8.66e-3 | +6.55e-3 | -1.22e-4 | -8.57e-5 |
| 44 | 3.00e-1 | 2 | 9.47e-2 | 9.73e-2 | 9.60e-2 | 9.73e-2 | 40 | -1.42e-3 | +6.75e-4 | -3.70e-4 | -1.29e-4 |
| 45 | 3.00e-1 | 1 | 1.04e-1 | 1.04e-1 | 1.04e-1 | 1.04e-1 | 301 | +2.08e-4 | +2.08e-4 | +2.08e-4 | -9.56e-5 |
| 46 | 3.00e-1 | 1 | 2.31e-1 | 2.31e-1 | 2.31e-1 | 2.31e-1 | 321 | +2.49e-3 | +2.49e-3 | +2.49e-3 | +1.63e-4 |
| 47 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 352 | -1.62e-4 | -1.62e-4 | -1.62e-4 | +1.31e-4 |
| 49 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 363 | -4.03e-6 | -4.03e-6 | -4.03e-6 | +1.17e-4 |
| 50 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 299 | -1.14e-5 | -1.14e-5 | -1.14e-5 | +1.04e-4 |
| 51 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 258 | -1.18e-4 | -1.18e-4 | -1.18e-4 | +8.22e-5 |
| 52 | 3.00e-1 | 1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 287 | -2.31e-4 | -2.31e-4 | -2.31e-4 | +5.09e-5 |
| 53 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 302 | +9.30e-5 | +9.30e-5 | +9.30e-5 | +5.51e-5 |
| 54 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 319 | +4.17e-5 | +4.17e-5 | +4.17e-5 | +5.38e-5 |
| 55 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 279 | +1.04e-4 | +1.04e-4 | +1.04e-4 | +5.88e-5 |
| 56 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 310 | -1.33e-4 | -1.33e-4 | -1.33e-4 | +3.96e-5 |
| 57 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 272 | +9.95e-5 | +9.95e-5 | +9.95e-5 | +4.56e-5 |
| 58 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 268 | -1.46e-4 | -1.46e-4 | -1.46e-4 | +2.64e-5 |
| 59 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 247 | -4.37e-5 | -4.37e-5 | -4.37e-5 | +1.94e-5 |
| 60 | 3.00e-1 | 1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 257 | -1.23e-4 | -1.23e-4 | -1.23e-4 | +5.15e-6 |
| 61 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 295 | +1.31e-4 | +1.31e-4 | +1.31e-4 | +1.77e-5 |
| 62 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 337 | +7.42e-5 | +7.42e-5 | +7.42e-5 | +2.34e-5 |
| 63 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 314 | +1.37e-4 | +1.37e-4 | +1.37e-4 | +3.47e-5 |
| 64 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 252 | -1.22e-4 | -1.22e-4 | -1.22e-4 | +1.91e-5 |
| 65 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 273 | -1.68e-4 | -1.68e-4 | -1.68e-4 | +3.77e-7 |
| 66 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 267 | +6.27e-5 | +6.27e-5 | +6.27e-5 | +6.61e-6 |
| 67 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 256 | -2.27e-5 | -2.27e-5 | -2.27e-5 | +3.68e-6 |
| 68 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 287 | +1.19e-5 | +1.19e-5 | +1.19e-5 | +4.50e-6 |
| 69 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 316 | +9.25e-5 | +9.25e-5 | +9.25e-5 | +1.33e-5 |
| 70 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 305 | +1.08e-4 | +1.08e-4 | +1.08e-4 | +2.28e-5 |
| 71 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 299 | -1.14e-4 | -1.14e-4 | -1.14e-4 | +9.07e-6 |
| 73 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 308 | +7.52e-5 | +7.52e-5 | +7.52e-5 | +1.57e-5 |
| 74 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 271 | -4.94e-5 | -4.94e-5 | -4.94e-5 | +9.18e-6 |
| 75 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 270 | -1.48e-4 | -1.48e-4 | -1.48e-4 | -6.50e-6 |
| 76 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 274 | +4.53e-5 | +4.53e-5 | +4.53e-5 | -1.32e-6 |
| 77 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 285 | +2.10e-5 | +2.10e-5 | +2.10e-5 | +9.16e-7 |
| 78 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 268 | +4.10e-5 | +4.10e-5 | +4.10e-5 | +4.93e-6 |
| 79 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 321 | -7.77e-5 | -7.77e-5 | -7.77e-5 | -3.33e-6 |
| 80 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 308 | +1.58e-4 | +1.58e-4 | +1.58e-4 | +1.28e-5 |
| 81 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 295 | -2.15e-5 | -2.15e-5 | -2.15e-5 | +9.37e-6 |
| 82 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 268 | -5.21e-5 | -5.21e-5 | -5.21e-5 | +3.23e-6 |
| 83 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 279 | -9.10e-5 | -9.10e-5 | -9.10e-5 | -6.19e-6 |
| 84 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 270 | +5.44e-5 | +5.44e-5 | +5.44e-5 | -1.35e-7 |
| 85 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 295 | +1.82e-8 | +1.82e-8 | +1.82e-8 | -1.19e-7 |
| 86 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 298 | +8.40e-6 | +8.40e-6 | +8.40e-6 | +7.32e-7 |
| 87 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 310 | +1.57e-5 | +1.57e-5 | +1.57e-5 | +2.23e-6 |
| 88 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 296 | +5.40e-5 | +5.40e-5 | +5.40e-5 | +7.40e-6 |
| 89 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 290 | +2.18e-5 | +2.18e-5 | +2.18e-5 | +8.85e-6 |
| 90 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 347 | -5.28e-5 | -5.28e-5 | -5.28e-5 | +2.68e-6 |
| 92 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 320 | +1.72e-4 | +1.72e-4 | +1.72e-4 | +1.96e-5 |
| 93 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 265 | -1.06e-4 | -1.06e-4 | -1.06e-4 | +7.05e-6 |
| 94 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 302 | -1.66e-4 | -1.66e-4 | -1.66e-4 | -1.02e-5 |
| 95 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 297 | +1.11e-4 | +1.11e-4 | +1.11e-4 | +1.92e-6 |
| 96 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 303 | -5.28e-5 | -5.28e-5 | -5.28e-5 | -3.56e-6 |
| 97 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 294 | +8.31e-5 | +8.31e-5 | +8.31e-5 | +5.11e-6 |
| 98 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 283 | -6.77e-5 | -6.77e-5 | -6.77e-5 | -2.16e-6 |
| 99 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 282 | -1.72e-5 | -1.72e-5 | -1.72e-5 | -3.67e-6 |
| 100 | 3.00e-2 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 277 | -3.61e-5 | -3.61e-5 | -3.61e-5 | -6.91e-6 |
| 101 | 3.00e-2 | 1 | 1.87e-1 | 1.87e-1 | 1.87e-1 | 1.87e-1 | 264 | -3.18e-4 | -3.18e-4 | -3.18e-4 | -3.80e-5 |
| 102 | 3.00e-2 | 1 | 2.10e-2 | 2.10e-2 | 2.10e-2 | 2.10e-2 | 319 | -6.87e-3 | -6.87e-3 | -6.87e-3 | -7.21e-4 |
| 104 | 3.00e-2 | 1 | 2.49e-2 | 2.49e-2 | 2.49e-2 | 2.49e-2 | 340 | +5.02e-4 | +5.02e-4 | +5.02e-4 | -5.99e-4 |
| 105 | 3.00e-2 | 1 | 2.58e-2 | 2.58e-2 | 2.58e-2 | 2.58e-2 | 308 | +1.18e-4 | +1.18e-4 | +1.18e-4 | -5.27e-4 |
| 106 | 3.00e-2 | 1 | 2.60e-2 | 2.60e-2 | 2.60e-2 | 2.60e-2 | 320 | +2.26e-5 | +2.26e-5 | +2.26e-5 | -4.72e-4 |
| 107 | 3.00e-2 | 1 | 2.75e-2 | 2.75e-2 | 2.75e-2 | 2.75e-2 | 293 | +2.00e-4 | +2.00e-4 | +2.00e-4 | -4.05e-4 |
| 108 | 3.00e-2 | 1 | 2.73e-2 | 2.73e-2 | 2.73e-2 | 2.73e-2 | 293 | -2.92e-5 | -2.92e-5 | -2.92e-5 | -3.67e-4 |
| 109 | 3.00e-2 | 1 | 2.88e-2 | 2.88e-2 | 2.88e-2 | 2.88e-2 | 281 | +1.95e-4 | +1.95e-4 | +1.95e-4 | -3.11e-4 |
| 110 | 3.00e-2 | 1 | 2.87e-2 | 2.87e-2 | 2.87e-2 | 2.87e-2 | 331 | -1.06e-5 | -1.06e-5 | -1.06e-5 | -2.81e-4 |
| 111 | 3.00e-2 | 1 | 3.09e-2 | 3.09e-2 | 3.09e-2 | 3.09e-2 | 298 | +2.43e-4 | +2.43e-4 | +2.43e-4 | -2.29e-4 |
| 112 | 3.00e-2 | 1 | 3.10e-2 | 3.10e-2 | 3.10e-2 | 3.10e-2 | 281 | +1.65e-5 | +1.65e-5 | +1.65e-5 | -2.04e-4 |
| 114 | 3.00e-2 | 1 | 3.16e-2 | 3.16e-2 | 3.16e-2 | 3.16e-2 | 343 | +4.85e-5 | +4.85e-5 | +4.85e-5 | -1.79e-4 |
| 115 | 3.00e-2 | 2 | 3.39e-2 | 3.40e-2 | 3.39e-2 | 3.39e-2 | 281 | -1.52e-5 | +2.50e-4 | +1.17e-4 | -1.24e-4 |
| 117 | 3.00e-2 | 1 | 3.38e-2 | 3.38e-2 | 3.38e-2 | 3.38e-2 | 332 | -8.86e-6 | -8.86e-6 | -8.86e-6 | -1.12e-4 |
| 118 | 3.00e-2 | 1 | 3.69e-2 | 3.69e-2 | 3.69e-2 | 3.69e-2 | 313 | +2.85e-4 | +2.85e-4 | +2.85e-4 | -7.26e-5 |
| 119 | 3.00e-2 | 1 | 3.64e-2 | 3.64e-2 | 3.64e-2 | 3.64e-2 | 301 | -4.53e-5 | -4.53e-5 | -4.53e-5 | -6.98e-5 |
| 120 | 3.00e-2 | 1 | 3.65e-2 | 3.65e-2 | 3.65e-2 | 3.65e-2 | 298 | +7.47e-6 | +7.47e-6 | +7.47e-6 | -6.21e-5 |
| 121 | 3.00e-2 | 1 | 3.79e-2 | 3.79e-2 | 3.79e-2 | 3.79e-2 | 296 | +1.25e-4 | +1.25e-4 | +1.25e-4 | -4.34e-5 |
| 122 | 3.00e-2 | 1 | 3.79e-2 | 3.79e-2 | 3.79e-2 | 3.79e-2 | 281 | -5.56e-7 | -5.56e-7 | -5.56e-7 | -3.91e-5 |
| 124 | 3.00e-2 | 1 | 3.89e-2 | 3.89e-2 | 3.89e-2 | 3.89e-2 | 354 | +7.34e-5 | +7.34e-5 | +7.34e-5 | -2.79e-5 |
| 125 | 3.00e-2 | 1 | 4.28e-2 | 4.28e-2 | 4.28e-2 | 4.28e-2 | 295 | +3.24e-4 | +3.24e-4 | +3.24e-4 | +7.35e-6 |
| 126 | 3.00e-2 | 1 | 4.10e-2 | 4.10e-2 | 4.10e-2 | 4.10e-2 | 313 | -1.38e-4 | -1.38e-4 | -1.38e-4 | -7.18e-6 |
| 127 | 3.00e-2 | 1 | 4.29e-2 | 4.29e-2 | 4.29e-2 | 4.29e-2 | 312 | +1.46e-4 | +1.46e-4 | +1.46e-4 | +8.14e-6 |
| 128 | 3.00e-2 | 1 | 4.27e-2 | 4.27e-2 | 4.27e-2 | 4.27e-2 | 296 | -1.29e-5 | -1.29e-5 | -1.29e-5 | +6.03e-6 |
| 129 | 3.00e-2 | 1 | 4.30e-2 | 4.30e-2 | 4.30e-2 | 4.30e-2 | 303 | +2.02e-5 | +2.02e-5 | +2.02e-5 | +7.45e-6 |
| 130 | 3.00e-2 | 1 | 4.49e-2 | 4.49e-2 | 4.49e-2 | 4.49e-2 | 349 | +1.27e-4 | +1.27e-4 | +1.27e-4 | +1.94e-5 |
| 131 | 3.00e-2 | 1 | 4.81e-2 | 4.81e-2 | 4.81e-2 | 4.81e-2 | 299 | +2.30e-4 | +2.30e-4 | +2.30e-4 | +4.05e-5 |
| 133 | 3.00e-2 | 1 | 4.72e-2 | 4.72e-2 | 4.72e-2 | 4.72e-2 | 332 | -5.48e-5 | -5.48e-5 | -5.48e-5 | +3.10e-5 |
| 134 | 3.00e-2 | 1 | 4.90e-2 | 4.90e-2 | 4.90e-2 | 4.90e-2 | 302 | +1.23e-4 | +1.23e-4 | +1.23e-4 | +4.02e-5 |
| 135 | 3.00e-2 | 1 | 4.93e-2 | 4.93e-2 | 4.93e-2 | 4.93e-2 | 284 | +1.78e-5 | +1.78e-5 | +1.78e-5 | +3.80e-5 |
| 136 | 3.00e-2 | 1 | 4.85e-2 | 4.85e-2 | 4.85e-2 | 4.85e-2 | 289 | -5.63e-5 | -5.63e-5 | -5.63e-5 | +2.85e-5 |
| 137 | 3.00e-2 | 1 | 4.92e-2 | 4.92e-2 | 4.92e-2 | 4.92e-2 | 340 | +4.22e-5 | +4.22e-5 | +4.22e-5 | +2.99e-5 |
| 138 | 3.00e-2 | 1 | 5.34e-2 | 5.34e-2 | 5.34e-2 | 5.34e-2 | 334 | +2.43e-4 | +2.43e-4 | +2.43e-4 | +5.13e-5 |
| 139 | 3.00e-2 | 1 | 5.32e-2 | 5.32e-2 | 5.32e-2 | 5.32e-2 | 303 | -8.91e-6 | -8.91e-6 | -8.91e-6 | +4.52e-5 |
| 140 | 3.00e-2 | 1 | 5.18e-2 | 5.18e-2 | 5.18e-2 | 5.18e-2 | 323 | -8.48e-5 | -8.48e-5 | -8.48e-5 | +3.22e-5 |
| 142 | 3.00e-2 | 2 | 5.45e-2 | 5.50e-2 | 5.48e-2 | 5.50e-2 | 263 | +3.59e-5 | +1.66e-4 | +1.01e-4 | +4.46e-5 |
| 144 | 3.00e-2 | 2 | 5.13e-2 | 5.79e-2 | 5.46e-2 | 5.79e-2 | 270 | -2.01e-4 | +4.45e-4 | +1.22e-4 | +6.26e-5 |
| 146 | 3.00e-2 | 1 | 5.35e-2 | 5.35e-2 | 5.35e-2 | 5.35e-2 | 360 | -2.16e-4 | -2.16e-4 | -2.16e-4 | +3.47e-5 |
| 147 | 3.00e-2 | 1 | 6.05e-2 | 6.05e-2 | 6.05e-2 | 6.05e-2 | 305 | +4.01e-4 | +4.01e-4 | +4.01e-4 | +7.13e-5 |
| 148 | 3.00e-2 | 1 | 5.68e-2 | 5.68e-2 | 5.68e-2 | 5.68e-2 | 268 | -2.36e-4 | -2.36e-4 | -2.36e-4 | +4.06e-5 |
| 149 | 3.00e-2 | 1 | 5.58e-2 | 5.58e-2 | 5.58e-2 | 5.58e-2 | 292 | -6.10e-5 | -6.10e-5 | -6.10e-5 | +3.04e-5 |
| 150 | 3.00e-3 | 1 | 5.92e-2 | 5.92e-2 | 5.92e-2 | 5.92e-2 | 269 | +2.20e-4 | +2.20e-4 | +2.20e-4 | +4.94e-5 |
| 151 | 3.00e-3 | 1 | 5.64e-2 | 5.64e-2 | 5.64e-2 | 5.64e-2 | 263 | -1.87e-4 | -1.87e-4 | -1.87e-4 | +2.58e-5 |
| 152 | 3.00e-3 | 1 | 5.65e-3 | 5.65e-3 | 5.65e-3 | 5.65e-3 | 291 | -7.91e-3 | -7.91e-3 | -7.91e-3 | -7.67e-4 |
| 153 | 3.00e-3 | 1 | 5.71e-3 | 5.71e-3 | 5.71e-3 | 5.71e-3 | 307 | +3.73e-5 | +3.73e-5 | +3.73e-5 | -6.87e-4 |
| 154 | 3.00e-3 | 1 | 5.78e-3 | 5.78e-3 | 5.78e-3 | 5.78e-3 | 290 | +4.04e-5 | +4.04e-5 | +4.04e-5 | -6.14e-4 |
| 156 | 3.00e-3 | 2 | 5.89e-3 | 6.18e-3 | 6.04e-3 | 6.18e-3 | 281 | +5.59e-5 | +1.67e-4 | +1.12e-4 | -4.76e-4 |
| 158 | 3.00e-3 | 1 | 5.83e-3 | 5.83e-3 | 5.83e-3 | 5.83e-3 | 340 | -1.71e-4 | -1.71e-4 | -1.71e-4 | -4.45e-4 |
| 159 | 3.00e-3 | 1 | 6.17e-3 | 6.17e-3 | 6.17e-3 | 6.17e-3 | 307 | +1.86e-4 | +1.86e-4 | +1.86e-4 | -3.82e-4 |
| 160 | 3.00e-3 | 1 | 6.15e-3 | 6.15e-3 | 6.15e-3 | 6.15e-3 | 336 | -9.46e-6 | -9.46e-6 | -9.46e-6 | -3.45e-4 |
| 161 | 3.00e-3 | 1 | 6.22e-3 | 6.22e-3 | 6.22e-3 | 6.22e-3 | 276 | +4.19e-5 | +4.19e-5 | +4.19e-5 | -3.06e-4 |
| 162 | 3.00e-3 | 1 | 6.03e-3 | 6.03e-3 | 6.03e-3 | 6.03e-3 | 251 | -1.23e-4 | -1.23e-4 | -1.23e-4 | -2.88e-4 |
| 163 | 3.00e-3 | 1 | 5.77e-3 | 5.77e-3 | 5.77e-3 | 5.77e-3 | 252 | -1.78e-4 | -1.78e-4 | -1.78e-4 | -2.77e-4 |
| 164 | 3.00e-3 | 1 | 5.87e-3 | 5.87e-3 | 5.87e-3 | 5.87e-3 | 272 | +6.33e-5 | +6.33e-5 | +6.33e-5 | -2.43e-4 |
| 165 | 3.00e-3 | 1 | 5.95e-3 | 5.95e-3 | 5.95e-3 | 5.95e-3 | 316 | +4.31e-5 | +4.31e-5 | +4.31e-5 | -2.14e-4 |
| 166 | 3.00e-3 | 1 | 6.50e-3 | 6.50e-3 | 6.50e-3 | 6.50e-3 | 310 | +2.83e-4 | +2.83e-4 | +2.83e-4 | -1.65e-4 |
| 167 | 3.00e-3 | 1 | 6.38e-3 | 6.38e-3 | 6.38e-3 | 6.38e-3 | 320 | -5.52e-5 | -5.52e-5 | -5.52e-5 | -1.54e-4 |
| 169 | 3.00e-3 | 1 | 6.64e-3 | 6.64e-3 | 6.64e-3 | 6.64e-3 | 317 | +1.24e-4 | +1.24e-4 | +1.24e-4 | -1.26e-4 |
| 170 | 3.00e-3 | 2 | 6.21e-3 | 6.58e-3 | 6.39e-3 | 6.21e-3 | 272 | -2.15e-4 | -3.23e-5 | -1.23e-4 | -1.26e-4 |
| 172 | 3.00e-3 | 1 | 6.02e-3 | 6.02e-3 | 6.02e-3 | 6.02e-3 | 331 | -9.25e-5 | -9.25e-5 | -9.25e-5 | -1.23e-4 |
| 173 | 3.00e-3 | 1 | 6.69e-3 | 6.69e-3 | 6.69e-3 | 6.69e-3 | 284 | +3.72e-4 | +3.72e-4 | +3.72e-4 | -7.34e-5 |
| 174 | 3.00e-3 | 1 | 6.31e-3 | 6.31e-3 | 6.31e-3 | 6.31e-3 | 304 | -1.92e-4 | -1.92e-4 | -1.92e-4 | -8.52e-5 |
| 175 | 3.00e-3 | 1 | 6.62e-3 | 6.62e-3 | 6.62e-3 | 6.62e-3 | 286 | +1.69e-4 | +1.69e-4 | +1.69e-4 | -5.98e-5 |
| 176 | 3.00e-3 | 1 | 6.44e-3 | 6.44e-3 | 6.44e-3 | 6.44e-3 | 353 | -7.97e-5 | -7.97e-5 | -7.97e-5 | -6.18e-5 |
| 177 | 3.00e-3 | 1 | 7.16e-3 | 7.16e-3 | 7.16e-3 | 7.16e-3 | 303 | +3.50e-4 | +3.50e-4 | +3.50e-4 | -2.06e-5 |
| 178 | 3.00e-3 | 1 | 6.61e-3 | 6.61e-3 | 6.61e-3 | 6.61e-3 | 264 | -3.04e-4 | -3.04e-4 | -3.04e-4 | -4.89e-5 |
| 180 | 3.00e-3 | 1 | 6.25e-3 | 6.25e-3 | 6.25e-3 | 6.25e-3 | 312 | -1.79e-4 | -1.79e-4 | -1.79e-4 | -6.20e-5 |
| 181 | 3.00e-3 | 2 | 6.45e-3 | 6.90e-3 | 6.67e-3 | 6.45e-3 | 264 | -2.57e-4 | +3.62e-4 | +5.26e-5 | -4.33e-5 |
| 183 | 3.00e-3 | 2 | 6.35e-3 | 6.90e-3 | 6.63e-3 | 6.90e-3 | 324 | -4.58e-5 | +2.57e-4 | +1.06e-4 | -1.35e-5 |
| 185 | 3.00e-3 | 1 | 6.79e-3 | 6.79e-3 | 6.79e-3 | 6.79e-3 | 384 | -4.21e-5 | -4.21e-5 | -4.21e-5 | -1.63e-5 |
| 186 | 3.00e-3 | 1 | 7.49e-3 | 7.49e-3 | 7.49e-3 | 7.49e-3 | 312 | +3.12e-4 | +3.12e-4 | +3.12e-4 | +1.65e-5 |
| 187 | 3.00e-3 | 1 | 6.67e-3 | 6.67e-3 | 6.67e-3 | 6.67e-3 | 303 | -3.80e-4 | -3.80e-4 | -3.80e-4 | -2.31e-5 |
| 188 | 3.00e-3 | 1 | 6.70e-3 | 6.70e-3 | 6.70e-3 | 6.70e-3 | 297 | +1.14e-5 | +1.14e-5 | +1.14e-5 | -1.97e-5 |
| 189 | 3.00e-3 | 1 | 6.64e-3 | 6.64e-3 | 6.64e-3 | 6.64e-3 | 313 | -2.83e-5 | -2.83e-5 | -2.83e-5 | -2.06e-5 |
| 191 | 3.00e-3 | 2 | 6.82e-3 | 7.15e-3 | 6.98e-3 | 7.15e-3 | 266 | +8.10e-5 | +1.76e-4 | +1.29e-4 | +8.27e-6 |
| 193 | 3.00e-3 | 1 | 6.56e-3 | 6.56e-3 | 6.56e-3 | 6.56e-3 | 321 | -2.70e-4 | -2.70e-4 | -2.70e-4 | -1.95e-5 |
| 194 | 3.00e-3 | 1 | 6.86e-3 | 6.86e-3 | 6.86e-3 | 6.86e-3 | 267 | +1.69e-4 | +1.69e-4 | +1.69e-4 | -7.10e-7 |
| 195 | 3.00e-3 | 1 | 6.71e-3 | 6.71e-3 | 6.71e-3 | 6.71e-3 | 286 | -7.50e-5 | -7.50e-5 | -7.50e-5 | -8.14e-6 |
| 196 | 3.00e-3 | 1 | 6.78e-3 | 6.78e-3 | 6.78e-3 | 6.78e-3 | 299 | +3.55e-5 | +3.55e-5 | +3.55e-5 | -3.78e-6 |
| 197 | 3.00e-3 | 1 | 6.95e-3 | 6.95e-3 | 6.95e-3 | 6.95e-3 | 328 | +7.18e-5 | +7.18e-5 | +7.18e-5 | +3.78e-6 |
| 198 | 3.00e-3 | 1 | 7.42e-3 | 7.42e-3 | 7.42e-3 | 7.42e-3 | 315 | +2.08e-4 | +2.08e-4 | +2.08e-4 | +2.42e-5 |
| 199 | 3.00e-3 | 1 | 7.29e-3 | 7.29e-3 | 7.29e-3 | 7.29e-3 | 280 | -6.21e-5 | -6.21e-5 | -6.21e-5 | +1.56e-5 |

