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
| nccl-async | 0.057255 | 0.9158 | +0.0033 | 1999.3 | 872 | 40.4 | 100% | 100% | 5.1 |

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
| nccl-async | 1.9712 | 0.7185 | 0.5509 | 0.4753 | 0.5266 | 0.5111 | 0.4896 | 0.4878 | 0.4624 | 0.4686 | 0.2120 | 0.1736 | 0.1550 | 0.1448 | 0.1365 | 0.0777 | 0.0687 | 0.0665 | 0.0604 | 0.0573 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | nccl-async | 0 | cuda:0 | 0.4018 | 2.5 |
| resnet-graph | nccl-async | 1 | cuda:1 | 0.3018 | 3.3 |
| resnet-graph | nccl-async | 2 | cuda:2 | 0.2964 | 3.1 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|
| resnet-graph | nccl-async | 356 | 355 | 394 | 391 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| nccl-async | gpu1 | 1998.1 | 1.1 | epoch-boundary(199) |
| nccl-async | gpu2 | 1998.2 | 1.0 | epoch-boundary(199) |
| nccl-async | gpu0 | 1998.1 | 0.6 | epoch-boundary(199) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | nccl-async | gpu0 | 0.6s | 0.0s | 0.0s | 0.0s | 1.4s |
| resnet-graph | nccl-async | gpu1 | 1.1s | 0.0s | 0.0s | 0.0s | 1.9s |
| resnet-graph | nccl-async | gpu2 | 1.0s | 0.0s | 0.0s | 0.0s | 1.8s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | nccl-async | 408 | 0 | 872 | 40.4 | 713/9582 | 0 | 0.0 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | nccl-async | 202.1 | 10.1% |

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
| resnet-graph | nccl-async | 192 | 872 | 0 | 1.27e-3 | -7.05e-4 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | nccl-async | 0 | 872 | 5.77e-2 | 5.70e-2 | 0.00e0 | 5.22e-1 | 49.3 | -1.22e-4 | 3.76e-3 |
| resnet-graph | nccl-async | 1 | 872 | 5.82e-2 | 5.90e-2 | 0.00e0 | 5.12e-1 | 32.7 | -1.02e-4 | 5.84e-3 |
| resnet-graph | nccl-async | 2 | 872 | 5.76e-2 | 5.89e-2 | 0.00e0 | 5.37e-1 | 18.0 | -1.01e-4 | 5.90e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | nccl-async | rank0 ↔ rank1 | +0.9933 |
| resnet-graph | nccl-async | rank0 ↔ rank2 | +0.9930 |
| resnet-graph | nccl-async | rank1 ↔ rank2 | +0.9984 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | nccl-async | 59 (0,1,2,3,4,5,6,9…146,148) | 7 (20,21,33,34,191,192,193) | 20,21,33,34 | 0,1,2,3,4,5,6,9…146,148 | 191,192,193 |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | nccl-async | 0e0 | 3 | 54 | 54 |
| resnet-graph | nccl-async | 0e0 | 5 | 28 | 28 |
| resnet-graph | nccl-async | 0e0 | 10 | 8 | 8 |
| resnet-graph | nccl-async | 1e-4 | 3 | 32 | 32 |
| resnet-graph | nccl-async | 1e-4 | 5 | 16 | 16 |
| resnet-graph | nccl-async | 1e-4 | 10 | 6 | 6 |
| resnet-graph | nccl-async | 1e-3 | 3 | 7 | 7 |
| resnet-graph | nccl-async | 1e-3 | 5 | 4 | 4 |
| resnet-graph | nccl-async | 1e-3 | 10 | 1 | 1 |
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
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 511 | +0.043 |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | 61 | +0.173 |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | 295 | -0.025 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | nccl-async | 869 | -0.005 | 191 | +0.207 | +0.171 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | nccl-async | 870 | 3.43e1–8.09e1 | 6.33e1 | 1.45e-3 | 3.71e-3 | 8.18e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 513 | 79–77545 | +1.769e-5 | 0.467 | +1.828e-5 | 0.484 | 94 | +1.819e-5 | 0.677 | 33–991 | +1.428e-3 | 0.661 |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 497 | 906–77545 | +1.811e-5 | 0.520 | +1.870e-5 | 0.536 | 93 | +1.819e-5 | 0.671 | 33–991 | +1.428e-3 | 0.725 |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | 62 | 78315–117013 | +9.869e-6 | 0.087 | +9.661e-6 | 0.083 | 48 | +1.365e-5 | 0.201 | 446–863 | -1.200e-3 | 0.094 |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | 296 | 117545–156192 | -4.918e-5 | 0.686 | -5.075e-5 | 0.705 | 50 | -4.971e-5 | 0.705 | 34–554 | +3.483e-3 | 0.720 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | +1.428e-3 | r0: +1.399e-3, r1: +1.440e-3, r2: +1.450e-3 | r0: 0.713, r1: 0.629, r2: 0.633 | 1.04× | ✓ |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +1.428e-3 | r0: +1.399e-3, r1: +1.440e-3, r2: +1.451e-3 | r0: 0.786, r1: 0.686, r2: 0.695 | 1.04× | ✓ |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | -1.200e-3 | r0: -1.189e-3, r1: -1.190e-3, r2: -1.221e-3 | r0: 0.091, r1: 0.093, r2: 0.096 | 1.03× | ✓ |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | +3.483e-3 | r0: +3.434e-3, r1: +3.523e-3, r2: +3.510e-3 | r0: 0.760, r1: 0.694, r2: 0.692 | 1.03× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | nccl-async | `█▇▇▇▇▇▇▇▇██████████████▇▄▅▅▅▅▅▅▅▅▅▆▆▂▂▂▂▂▂▂▁▁▁▁▁` | `▁▆▅▅▆▇▇▅███▇▇▇▇▇▇▇▇▇▇▇▇▆▅▆▆▆▇▇▇▇▇▇▇▅▃▅▆▆▆▆▆▅▅▆█▅` |

### Per-Epoch Detail: resnet-graph / nccl-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 17 | 0.00e0 | 5.37e-1 | 1.02e-1 | 5.45e-2 | 19 | -7.76e-2 | +1.14e-2 | -1.17e-2 | -5.81e-3 |
| 1 | 3.00e-1 | 14 | 5.05e-2 | 1.03e-1 | 6.25e-2 | 6.08e-2 | 18 | -3.22e-2 | +3.18e-2 | +2.13e-4 | -1.05e-3 |
| 2 | 3.00e-1 | 16 | 4.96e-2 | 1.18e-1 | 6.42e-2 | 6.51e-2 | 21 | -4.26e-2 | +4.09e-2 | -1.37e-5 | +1.10e-5 |
| 3 | 3.00e-1 | 15 | 5.55e-2 | 1.29e-1 | 7.27e-2 | 7.82e-2 | 20 | -4.91e-2 | +3.82e-2 | +3.28e-4 | +5.47e-4 |
| 4 | 3.00e-1 | 21 | 6.38e-2 | 1.40e-1 | 7.72e-2 | 7.42e-2 | 17 | -2.79e-2 | +2.68e-2 | -2.84e-4 | -8.25e-5 |
| 5 | 3.00e-1 | 8 | 5.35e-2 | 1.41e-1 | 8.41e-2 | 8.05e-2 | 20 | -6.81e-2 | +4.92e-2 | -1.12e-3 | -7.83e-4 |
| 6 | 3.00e-1 | 11 | 7.74e-2 | 1.36e-1 | 8.98e-2 | 8.42e-2 | 18 | -2.48e-2 | +2.82e-2 | +4.57e-4 | -2.72e-4 |
| 7 | 3.00e-1 | 8 | 7.43e-2 | 1.52e-1 | 1.13e-1 | 1.14e-1 | 37 | -7.08e-3 | +1.74e-2 | +1.08e-3 | +3.08e-4 |
| 8 | 3.00e-1 | 7 | 7.73e-2 | 1.50e-1 | 9.55e-2 | 7.73e-2 | 24 | -2.09e-2 | +1.15e-2 | -2.65e-3 | -1.24e-3 |
| 9 | 3.00e-1 | 10 | 7.49e-2 | 1.41e-1 | 8.92e-2 | 8.56e-2 | 24 | -3.00e-2 | +2.35e-2 | -1.70e-5 | -4.91e-4 |
| 10 | 3.00e-1 | 12 | 7.51e-2 | 1.32e-1 | 8.51e-2 | 7.51e-2 | 16 | -2.18e-2 | +2.05e-2 | -4.30e-4 | -6.83e-4 |
| 11 | 3.00e-1 | 19 | 5.48e-2 | 1.29e-1 | 7.07e-2 | 8.31e-2 | 21 | -5.45e-2 | +5.27e-2 | +6.20e-4 | +7.56e-4 |
| 12 | 3.00e-1 | 7 | 7.06e-2 | 1.38e-1 | 8.89e-2 | 9.11e-2 | 27 | -3.07e-2 | +2.47e-2 | +1.55e-4 | +4.71e-4 |
| 13 | 3.00e-1 | 13 | 6.25e-2 | 1.45e-1 | 7.77e-2 | 7.24e-2 | 20 | -3.83e-2 | +2.48e-2 | -9.04e-4 | -3.03e-4 |
| 14 | 3.00e-1 | 18 | 6.42e-2 | 1.34e-1 | 7.40e-2 | 6.62e-2 | 15 | -3.31e-2 | +2.94e-2 | -5.74e-4 | -6.50e-4 |
| 15 | 3.00e-1 | 12 | 5.04e-2 | 1.33e-1 | 6.58e-2 | 5.73e-2 | 15 | -6.81e-2 | +5.83e-2 | -1.32e-3 | -1.30e-3 |
| 16 | 3.00e-1 | 16 | 5.93e-2 | 1.26e-1 | 6.95e-2 | 5.93e-2 | 18 | -3.87e-2 | +4.89e-2 | +5.02e-4 | -6.93e-4 |
| 17 | 3.00e-1 | 13 | 5.53e-2 | 1.39e-1 | 7.80e-2 | 7.50e-2 | 17 | -6.45e-2 | +4.87e-2 | +9.46e-5 | -3.42e-4 |
| 18 | 3.00e-1 | 17 | 5.85e-2 | 1.35e-1 | 6.80e-2 | 6.04e-2 | 16 | -5.60e-2 | +4.78e-2 | -4.95e-4 | -6.61e-4 |
| 19 | 3.00e-1 | 16 | 5.47e-2 | 1.34e-1 | 6.96e-2 | 7.78e-2 | 16 | -5.51e-2 | +4.69e-2 | +5.88e-4 | +8.80e-4 |
| 20 | 3.00e-1 | 13 | 5.47e-2 | 1.38e-1 | 7.67e-2 | 7.41e-2 | 16 | -5.13e-2 | +5.01e-2 | +8.76e-4 | +4.83e-4 |
| 21 | 3.00e-1 | 10 | 5.68e-2 | 1.35e-1 | 8.71e-2 | 9.26e-2 | 30 | -4.55e-2 | +5.07e-2 | +1.96e-3 | +1.07e-3 |
| 22 | 3.00e-1 | 14 | 6.80e-2 | 1.50e-1 | 8.40e-2 | 8.09e-2 | 25 | -2.24e-2 | +1.79e-2 | -5.41e-4 | +4.12e-5 |
| 23 | 3.00e-1 | 9 | 7.37e-2 | 1.35e-1 | 8.70e-2 | 7.37e-2 | 20 | -2.19e-2 | +1.51e-2 | -8.26e-4 | -6.78e-4 |
| 24 | 3.00e-1 | 14 | 6.11e-2 | 1.43e-1 | 8.36e-2 | 8.61e-2 | 30 | -4.27e-2 | +3.28e-2 | -4.04e-4 | -3.78e-4 |
| 25 | 3.00e-1 | 8 | 7.12e-2 | 1.42e-1 | 8.73e-2 | 9.34e-2 | 20 | -3.38e-2 | +2.12e-2 | +1.10e-4 | +2.14e-4 |
| 26 | 3.00e-1 | 11 | 6.57e-2 | 1.29e-1 | 7.86e-2 | 7.22e-2 | 23 | -2.15e-2 | +2.76e-2 | +1.17e-4 | -7.03e-6 |
| 27 | 3.00e-1 | 15 | 6.04e-2 | 1.28e-1 | 8.16e-2 | 8.60e-2 | 26 | -2.28e-2 | +2.24e-2 | +2.98e-4 | +3.25e-4 |
| 28 | 3.00e-1 | 9 | 6.84e-2 | 1.38e-1 | 8.26e-2 | 6.84e-2 | 18 | -2.57e-2 | +1.65e-2 | -2.04e-3 | -1.17e-3 |
| 29 | 3.00e-1 | 15 | 5.49e-2 | 1.40e-1 | 6.85e-2 | 5.99e-2 | 23 | -5.57e-2 | +4.90e-2 | -3.02e-4 | -7.42e-4 |
| 30 | 3.00e-1 | 17 | 6.18e-2 | 1.39e-1 | 7.63e-2 | 7.56e-2 | 20 | -2.88e-2 | +2.51e-2 | +4.04e-5 | +1.08e-4 |
| 31 | 3.00e-1 | 10 | 5.70e-2 | 1.46e-1 | 7.36e-2 | 7.12e-2 | 15 | -5.19e-2 | +3.58e-2 | -1.24e-3 | -3.17e-4 |
| 32 | 3.00e-1 | 14 | 6.18e-2 | 1.40e-1 | 7.42e-2 | 6.61e-2 | 17 | -4.31e-2 | +4.71e-2 | +4.79e-4 | -2.23e-4 |
| 33 | 3.00e-1 | 14 | 5.53e-2 | 1.38e-1 | 8.46e-2 | 1.12e-1 | 41 | -4.99e-2 | +4.61e-2 | +1.60e-3 | +1.22e-3 |
| 34 | 3.00e-1 | 4 | 9.65e-2 | 1.51e-1 | 1.15e-1 | 9.65e-2 | 33 | -1.09e-2 | +8.42e-3 | -1.06e-3 | +3.26e-4 |
| 35 | 3.00e-1 | 2 | 9.57e-2 | 1.03e-1 | 9.91e-2 | 9.57e-2 | 27 | -2.60e-3 | +2.06e-3 | -2.68e-4 | +1.90e-4 |
| 36 | 3.00e-1 | 1 | 8.34e-2 | 8.34e-2 | 8.34e-2 | 8.34e-2 | 289 | -4.74e-4 | -4.74e-4 | -4.74e-4 | +1.24e-4 |
| 37 | 3.00e-1 | 1 | 2.37e-1 | 2.37e-1 | 2.37e-1 | 2.37e-1 | 290 | +3.60e-3 | +3.60e-3 | +3.60e-3 | +4.71e-4 |
| 38 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 299 | -2.70e-4 | -2.70e-4 | -2.70e-4 | +3.97e-4 |
| 39 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 307 | -5.75e-5 | -5.75e-5 | -5.75e-5 | +3.52e-4 |
| 40 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 316 | -1.31e-5 | -1.31e-5 | -1.31e-5 | +3.15e-4 |
| 41 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 312 | +4.68e-5 | +4.68e-5 | +4.68e-5 | +2.88e-4 |
| 42 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 289 | -4.74e-5 | -4.74e-5 | -4.74e-5 | +2.55e-4 |
| 44 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 371 | -5.28e-5 | -5.28e-5 | -5.28e-5 | +2.24e-4 |
| 45 | 3.00e-1 | 1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 320 | +1.90e-4 | +1.90e-4 | +1.90e-4 | +2.21e-4 |
| 46 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 309 | -1.68e-4 | -1.68e-4 | -1.68e-4 | +1.82e-4 |
| 47 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 265 | -7.94e-6 | -7.94e-6 | -7.94e-6 | +1.63e-4 |
| 48 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 278 | -1.46e-4 | -1.46e-4 | -1.46e-4 | +1.32e-4 |
| 49 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 302 | +1.53e-5 | +1.53e-5 | +1.53e-5 | +1.20e-4 |
| 50 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 276 | +1.63e-4 | +1.63e-4 | +1.63e-4 | +1.25e-4 |
| 51 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 265 | -1.75e-4 | -1.75e-4 | -1.75e-4 | +9.45e-5 |
| 52 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 275 | +2.85e-5 | +2.85e-5 | +2.85e-5 | +8.79e-5 |
| 53 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 269 | +1.60e-5 | +1.60e-5 | +1.60e-5 | +8.07e-5 |
| 54 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 281 | -2.89e-5 | -2.89e-5 | -2.89e-5 | +6.98e-5 |
| 55 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 282 | +6.45e-5 | +6.45e-5 | +6.45e-5 | +6.92e-5 |
| 56 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 300 | +1.30e-5 | +1.30e-5 | +1.30e-5 | +6.36e-5 |
| 57 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 290 | +6.13e-5 | +6.13e-5 | +6.13e-5 | +6.34e-5 |
| 58 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 318 | -6.03e-5 | -6.03e-5 | -6.03e-5 | +5.10e-5 |
| 59 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 273 | +1.38e-4 | +1.38e-4 | +1.38e-4 | +5.98e-5 |
| 60 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 238 | -3.50e-4 | -3.50e-4 | -3.50e-4 | +1.88e-5 |
| 61 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 267 | -1.04e-5 | -1.04e-5 | -1.04e-5 | +1.58e-5 |
| 62 | 3.00e-1 | 2 | 2.02e-1 | 2.06e-1 | 2.04e-1 | 2.06e-1 | 226 | +6.78e-5 | +7.53e-5 | +7.15e-5 | +2.65e-5 |
| 64 | 3.00e-1 | 2 | 1.99e-1 | 2.14e-1 | 2.06e-1 | 2.14e-1 | 239 | -1.19e-4 | +3.16e-4 | +9.87e-5 | +4.24e-5 |
| 66 | 3.00e-1 | 2 | 1.98e-1 | 2.17e-1 | 2.08e-1 | 2.17e-1 | 211 | -2.35e-4 | +4.42e-4 | +1.04e-4 | +5.74e-5 |
| 67 | 3.00e-1 | 1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 250 | -5.35e-4 | -5.35e-4 | -5.35e-4 | -1.81e-6 |
| 68 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 252 | +2.45e-4 | +2.45e-4 | +2.45e-4 | +2.28e-5 |
| 69 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 231 | +1.01e-5 | +1.01e-5 | +1.01e-5 | +2.16e-5 |
| 70 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 249 | -1.04e-4 | -1.04e-4 | -1.04e-4 | +8.99e-6 |
| 71 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 310 | +6.18e-5 | +6.18e-5 | +6.18e-5 | +1.43e-5 |
| 72 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 307 | +2.21e-4 | +2.21e-4 | +2.21e-4 | +3.50e-5 |
| 73 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 253 | -3.61e-5 | -3.61e-5 | -3.61e-5 | +2.79e-5 |
| 74 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 271 | -2.07e-4 | -2.07e-4 | -2.07e-4 | +4.37e-6 |
| 75 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 260 | +9.11e-5 | +9.11e-5 | +9.11e-5 | +1.30e-5 |
| 76 | 3.00e-1 | 2 | 1.96e-1 | 2.03e-1 | 1.99e-1 | 1.96e-1 | 213 | -1.64e-4 | -8.21e-5 | -1.23e-4 | -1.32e-5 |
| 78 | 3.00e-1 | 2 | 1.94e-1 | 2.16e-1 | 2.05e-1 | 2.16e-1 | 213 | -3.69e-5 | +5.26e-4 | +2.44e-4 | +3.85e-5 |
| 79 | 3.00e-1 | 1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 246 | -4.46e-4 | -4.46e-4 | -4.46e-4 | -9.93e-6 |
| 80 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 230 | +1.54e-4 | +1.54e-4 | +1.54e-4 | +6.43e-6 |
| 81 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 240 | -9.48e-5 | -9.48e-5 | -9.48e-5 | -3.70e-6 |
| 82 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 259 | +1.23e-4 | +1.23e-4 | +1.23e-4 | +9.00e-6 |
| 83 | 3.00e-1 | 2 | 1.98e-1 | 2.05e-1 | 2.02e-1 | 1.98e-1 | 213 | -1.51e-4 | +4.20e-5 | -5.46e-5 | -4.04e-6 |
| 84 | 3.00e-1 | 1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 228 | -1.16e-4 | -1.16e-4 | -1.16e-4 | -1.52e-5 |
| 85 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 290 | +1.06e-4 | +1.06e-4 | +1.06e-4 | -3.03e-6 |
| 86 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 287 | +2.17e-4 | +2.17e-4 | +2.17e-4 | +1.90e-5 |
| 87 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 280 | -1.62e-5 | -1.62e-5 | -1.62e-5 | +1.54e-5 |
| 89 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 385 | +2.58e-5 | +2.58e-5 | +2.58e-5 | +1.65e-5 |
| 90 | 3.00e-1 | 2 | 2.12e-1 | 2.28e-1 | 2.20e-1 | 2.12e-1 | 262 | -2.82e-4 | +2.37e-4 | -2.25e-5 | +6.49e-6 |
| 92 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 361 | -9.81e-5 | -9.81e-5 | -9.81e-5 | -3.97e-6 |
| 93 | 3.00e-1 | 1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 295 | +3.50e-4 | +3.50e-4 | +3.50e-4 | +3.15e-5 |
| 94 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 260 | -2.91e-4 | -2.91e-4 | -2.91e-4 | -7.57e-7 |
| 95 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 280 | -2.16e-5 | -2.16e-5 | -2.16e-5 | -2.84e-6 |
| 96 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 279 | +1.17e-5 | +1.17e-5 | +1.17e-5 | -1.39e-6 |
| 97 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 254 | -1.54e-5 | -1.54e-5 | -1.54e-5 | -2.78e-6 |
| 98 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 269 | -6.08e-5 | -6.08e-5 | -6.08e-5 | -8.58e-6 |
| 99 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 247 | +3.14e-5 | +3.14e-5 | +3.14e-5 | -4.58e-6 |
| 100 | 3.00e-2 | 2 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 245 | -1.71e-5 | -1.08e-5 | -1.40e-5 | -6.40e-6 |
| 102 | 3.00e-2 | 1 | 2.06e-2 | 2.06e-2 | 2.06e-2 | 2.06e-2 | 316 | -7.28e-3 | -7.28e-3 | -7.28e-3 | -7.33e-4 |
| 103 | 3.00e-2 | 1 | 2.38e-2 | 2.38e-2 | 2.38e-2 | 2.38e-2 | 296 | +4.86e-4 | +4.86e-4 | +4.86e-4 | -6.12e-4 |
| 104 | 3.00e-2 | 1 | 2.48e-2 | 2.48e-2 | 2.48e-2 | 2.48e-2 | 271 | +1.56e-4 | +1.56e-4 | +1.56e-4 | -5.35e-4 |
| 105 | 3.00e-2 | 1 | 2.48e-2 | 2.48e-2 | 2.48e-2 | 2.48e-2 | 250 | -6.34e-6 | -6.34e-6 | -6.34e-6 | -4.82e-4 |
| 106 | 3.00e-2 | 2 | 2.46e-2 | 2.49e-2 | 2.47e-2 | 2.49e-2 | 217 | -3.70e-5 | +6.79e-5 | +1.55e-5 | -3.87e-4 |
| 107 | 3.00e-2 | 1 | 2.48e-2 | 2.48e-2 | 2.48e-2 | 2.48e-2 | 282 | -1.23e-5 | -1.23e-5 | -1.23e-5 | -3.49e-4 |
| 108 | 3.00e-2 | 1 | 2.79e-2 | 2.79e-2 | 2.79e-2 | 2.79e-2 | 255 | +4.62e-4 | +4.62e-4 | +4.62e-4 | -2.68e-4 |
| 109 | 3.00e-2 | 1 | 2.74e-2 | 2.74e-2 | 2.74e-2 | 2.74e-2 | 244 | -7.54e-5 | -7.54e-5 | -7.54e-5 | -2.49e-4 |
| 110 | 3.00e-2 | 1 | 2.74e-2 | 2.74e-2 | 2.74e-2 | 2.74e-2 | 246 | -5.81e-6 | -5.81e-6 | -5.81e-6 | -2.25e-4 |
| 111 | 3.00e-2 | 1 | 2.86e-2 | 2.86e-2 | 2.86e-2 | 2.86e-2 | 226 | +1.85e-4 | +1.85e-4 | +1.85e-4 | -1.84e-4 |
| 112 | 3.00e-2 | 1 | 2.84e-2 | 2.84e-2 | 2.84e-2 | 2.84e-2 | 269 | -1.83e-5 | -1.83e-5 | -1.83e-5 | -1.67e-4 |
| 113 | 3.00e-2 | 1 | 3.09e-2 | 3.09e-2 | 3.09e-2 | 3.09e-2 | 261 | +3.22e-4 | +3.22e-4 | +3.22e-4 | -1.18e-4 |
| 114 | 3.00e-2 | 2 | 3.08e-2 | 3.34e-2 | 3.21e-2 | 3.34e-2 | 231 | -1.57e-5 | +3.54e-4 | +1.69e-4 | -6.18e-5 |
| 116 | 3.00e-2 | 1 | 3.06e-2 | 3.06e-2 | 3.06e-2 | 3.06e-2 | 295 | -2.93e-4 | -2.93e-4 | -2.93e-4 | -8.49e-5 |
| 117 | 3.00e-2 | 2 | 3.45e-2 | 3.49e-2 | 3.47e-2 | 3.45e-2 | 215 | -4.51e-5 | +4.86e-4 | +2.21e-4 | -2.95e-5 |
| 118 | 3.00e-2 | 1 | 3.23e-2 | 3.23e-2 | 3.23e-2 | 3.23e-2 | 250 | -2.69e-4 | -2.69e-4 | -2.69e-4 | -5.34e-5 |
| 119 | 3.00e-2 | 1 | 3.39e-2 | 3.39e-2 | 3.39e-2 | 3.39e-2 | 250 | +1.99e-4 | +1.99e-4 | +1.99e-4 | -2.81e-5 |
| 120 | 3.00e-2 | 1 | 3.44e-2 | 3.44e-2 | 3.44e-2 | 3.44e-2 | 270 | +5.22e-5 | +5.22e-5 | +5.22e-5 | -2.01e-5 |
| 121 | 3.00e-2 | 1 | 3.60e-2 | 3.60e-2 | 3.60e-2 | 3.60e-2 | 276 | +1.61e-4 | +1.61e-4 | +1.61e-4 | -2.02e-6 |
| 122 | 3.00e-2 | 1 | 3.76e-2 | 3.76e-2 | 3.76e-2 | 3.76e-2 | 261 | +1.63e-4 | +1.63e-4 | +1.63e-4 | +1.45e-5 |
| 123 | 3.00e-2 | 2 | 3.69e-2 | 3.77e-2 | 3.73e-2 | 3.69e-2 | 204 | -1.06e-4 | +1.33e-5 | -4.65e-5 | +2.30e-6 |
| 124 | 3.00e-2 | 1 | 3.58e-2 | 3.58e-2 | 3.58e-2 | 3.58e-2 | 226 | -1.27e-4 | -1.27e-4 | -1.27e-4 | -1.07e-5 |
| 125 | 3.00e-2 | 1 | 3.73e-2 | 3.73e-2 | 3.73e-2 | 3.73e-2 | 227 | +1.80e-4 | +1.80e-4 | +1.80e-4 | +8.40e-6 |
| 126 | 3.00e-2 | 1 | 3.80e-2 | 3.80e-2 | 3.80e-2 | 3.80e-2 | 258 | +7.48e-5 | +7.48e-5 | +7.48e-5 | +1.50e-5 |
| 127 | 3.00e-2 | 1 | 4.09e-2 | 4.09e-2 | 4.09e-2 | 4.09e-2 | 261 | +2.82e-4 | +2.82e-4 | +2.82e-4 | +4.17e-5 |
| 128 | 3.00e-2 | 1 | 4.12e-2 | 4.12e-2 | 4.12e-2 | 4.12e-2 | 249 | +2.35e-5 | +2.35e-5 | +2.35e-5 | +3.99e-5 |
| 129 | 3.00e-2 | 2 | 4.09e-2 | 4.12e-2 | 4.10e-2 | 4.09e-2 | 197 | -4.03e-5 | +3.39e-8 | -2.02e-5 | +2.83e-5 |
| 130 | 3.00e-2 | 1 | 3.93e-2 | 3.93e-2 | 3.93e-2 | 3.93e-2 | 247 | -1.58e-4 | -1.58e-4 | -1.58e-4 | +9.68e-6 |
| 131 | 3.00e-2 | 2 | 3.89e-2 | 4.26e-2 | 4.08e-2 | 3.89e-2 | 174 | -5.26e-4 | +4.36e-4 | -4.51e-5 | -5.53e-6 |
| 132 | 3.00e-2 | 1 | 3.87e-2 | 3.87e-2 | 3.87e-2 | 3.87e-2 | 201 | -2.39e-5 | -2.39e-5 | -2.39e-5 | -7.37e-6 |
| 133 | 3.00e-2 | 1 | 4.12e-2 | 4.12e-2 | 4.12e-2 | 4.12e-2 | 205 | +2.98e-4 | +2.98e-4 | +2.98e-4 | +2.31e-5 |
| 134 | 3.00e-2 | 1 | 4.20e-2 | 4.20e-2 | 4.20e-2 | 4.20e-2 | 220 | +9.02e-5 | +9.02e-5 | +9.02e-5 | +2.98e-5 |
| 135 | 3.00e-2 | 2 | 4.39e-2 | 4.78e-2 | 4.58e-2 | 4.78e-2 | 194 | +1.79e-4 | +4.33e-4 | +3.06e-4 | +8.36e-5 |
| 136 | 3.00e-2 | 1 | 4.26e-2 | 4.26e-2 | 4.26e-2 | 4.26e-2 | 225 | -5.14e-4 | -5.14e-4 | -5.14e-4 | +2.38e-5 |
| 137 | 3.00e-2 | 2 | 4.22e-2 | 4.59e-2 | 4.41e-2 | 4.22e-2 | 164 | -5.12e-4 | +4.22e-4 | -4.50e-5 | +6.05e-6 |
| 138 | 3.00e-2 | 1 | 3.99e-2 | 3.99e-2 | 3.99e-2 | 3.99e-2 | 195 | -2.90e-4 | -2.90e-4 | -2.90e-4 | -2.36e-5 |
| 139 | 3.00e-2 | 1 | 4.49e-2 | 4.49e-2 | 4.49e-2 | 4.49e-2 | 195 | +6.04e-4 | +6.04e-4 | +6.04e-4 | +3.92e-5 |
| 140 | 3.00e-2 | 2 | 4.59e-2 | 4.76e-2 | 4.68e-2 | 4.76e-2 | 174 | +1.08e-4 | +1.97e-4 | +1.52e-4 | +6.12e-5 |
| 141 | 3.00e-2 | 2 | 4.48e-2 | 4.49e-2 | 4.49e-2 | 4.49e-2 | 174 | -3.19e-4 | +2.02e-5 | -1.49e-4 | +2.29e-5 |
| 142 | 3.00e-2 | 1 | 4.52e-2 | 4.52e-2 | 4.52e-2 | 4.52e-2 | 218 | +2.82e-5 | +2.82e-5 | +2.82e-5 | +2.34e-5 |
| 143 | 3.00e-2 | 1 | 4.81e-2 | 4.81e-2 | 4.81e-2 | 4.81e-2 | 230 | +2.69e-4 | +2.69e-4 | +2.69e-4 | +4.80e-5 |
| 144 | 3.00e-2 | 2 | 4.80e-2 | 5.10e-2 | 4.95e-2 | 4.80e-2 | 174 | -3.59e-4 | +2.90e-4 | -3.47e-5 | +2.90e-5 |
| 145 | 3.00e-2 | 1 | 4.54e-2 | 4.54e-2 | 4.54e-2 | 4.54e-2 | 197 | -2.74e-4 | -2.74e-4 | -2.74e-4 | -1.26e-6 |
| 146 | 3.00e-2 | 2 | 4.95e-2 | 4.98e-2 | 4.97e-2 | 4.98e-2 | 184 | +2.86e-5 | +4.33e-4 | +2.31e-4 | +4.08e-5 |
| 147 | 3.00e-2 | 1 | 4.84e-2 | 4.84e-2 | 4.84e-2 | 4.84e-2 | 223 | -1.31e-4 | -1.31e-4 | -1.31e-4 | +2.36e-5 |
| 148 | 3.00e-2 | 2 | 5.29e-2 | 5.35e-2 | 5.32e-2 | 5.35e-2 | 159 | +7.06e-5 | +4.27e-4 | +2.49e-4 | +6.46e-5 |
| 149 | 3.00e-2 | 1 | 4.84e-2 | 4.84e-2 | 4.84e-2 | 4.84e-2 | 217 | -4.62e-4 | -4.62e-4 | -4.62e-4 | +1.19e-5 |
| 150 | 3.00e-3 | 1 | 5.37e-2 | 5.37e-2 | 5.37e-2 | 5.37e-2 | 187 | +5.59e-4 | +5.59e-4 | +5.59e-4 | +6.66e-5 |
| 151 | 3.00e-3 | 2 | 5.01e-3 | 4.38e-2 | 2.44e-2 | 5.01e-3 | 150 | -1.45e-2 | -1.10e-3 | -7.78e-3 | -1.49e-3 |
| 152 | 3.00e-3 | 2 | 4.31e-3 | 4.46e-3 | 4.39e-3 | 4.46e-3 | 150 | -9.33e-4 | +2.32e-4 | -3.50e-4 | -1.27e-3 |
| 153 | 3.00e-3 | 1 | 4.25e-3 | 4.25e-3 | 4.25e-3 | 4.25e-3 | 180 | -2.68e-4 | -2.68e-4 | -2.68e-4 | -1.17e-3 |
| 154 | 3.00e-3 | 2 | 4.78e-3 | 5.00e-3 | 4.89e-3 | 5.00e-3 | 158 | +2.88e-4 | +5.73e-4 | +4.31e-4 | -8.66e-4 |
| 155 | 3.00e-3 | 2 | 4.56e-3 | 4.75e-3 | 4.66e-3 | 4.75e-3 | 158 | -5.06e-4 | +2.53e-4 | -1.27e-4 | -7.22e-4 |
| 156 | 3.00e-3 | 1 | 4.42e-3 | 4.42e-3 | 4.42e-3 | 4.42e-3 | 187 | -3.80e-4 | -3.80e-4 | -3.80e-4 | -6.88e-4 |
| 157 | 3.00e-3 | 3 | 4.34e-3 | 4.79e-3 | 4.60e-3 | 4.34e-3 | 149 | -6.61e-4 | +3.19e-4 | -6.79e-5 | -5.29e-4 |
| 158 | 3.00e-3 | 1 | 4.32e-3 | 4.32e-3 | 4.32e-3 | 4.32e-3 | 179 | -1.91e-5 | -1.91e-5 | -1.91e-5 | -4.78e-4 |
| 159 | 3.00e-3 | 2 | 4.84e-3 | 5.01e-3 | 4.93e-3 | 5.01e-3 | 140 | +2.34e-4 | +6.00e-4 | +4.17e-4 | -3.10e-4 |
| 160 | 3.00e-3 | 1 | 4.43e-3 | 4.43e-3 | 4.43e-3 | 4.43e-3 | 159 | -7.71e-4 | -7.71e-4 | -7.71e-4 | -3.56e-4 |
| 161 | 3.00e-3 | 3 | 4.19e-3 | 4.76e-3 | 4.44e-3 | 4.19e-3 | 124 | -1.02e-3 | +6.45e-4 | -1.47e-4 | -3.09e-4 |
| 162 | 3.00e-3 | 1 | 4.08e-3 | 4.08e-3 | 4.08e-3 | 4.08e-3 | 164 | -1.57e-4 | -1.57e-4 | -1.57e-4 | -2.93e-4 |
| 163 | 3.00e-3 | 3 | 4.23e-3 | 4.80e-3 | 4.51e-3 | 4.23e-3 | 133 | -9.43e-4 | +5.63e-4 | +4.16e-5 | -2.17e-4 |
| 164 | 3.00e-3 | 1 | 4.19e-3 | 4.19e-3 | 4.19e-3 | 4.19e-3 | 170 | -6.56e-5 | -6.56e-5 | -6.56e-5 | -2.02e-4 |
| 165 | 3.00e-3 | 2 | 4.82e-3 | 4.86e-3 | 4.84e-3 | 4.86e-3 | 133 | +6.04e-5 | +8.07e-4 | +4.34e-4 | -8.50e-5 |
| 166 | 3.00e-3 | 2 | 4.15e-3 | 4.73e-3 | 4.44e-3 | 4.73e-3 | 133 | -1.01e-3 | +9.71e-4 | -2.09e-5 | -6.29e-5 |
| 167 | 3.00e-3 | 2 | 4.54e-3 | 4.62e-3 | 4.58e-3 | 4.62e-3 | 122 | -2.45e-4 | +1.35e-4 | -5.49e-5 | -5.95e-5 |
| 168 | 3.00e-3 | 3 | 4.10e-3 | 4.82e-3 | 4.34e-3 | 4.10e-3 | 112 | -1.44e-3 | +1.31e-3 | -2.82e-4 | -1.27e-4 |
| 169 | 3.00e-3 | 1 | 3.97e-3 | 3.97e-3 | 3.97e-3 | 3.97e-3 | 134 | -2.34e-4 | -2.34e-4 | -2.34e-4 | -1.38e-4 |
| 170 | 3.00e-3 | 2 | 4.39e-3 | 4.66e-3 | 4.53e-3 | 4.66e-3 | 116 | +5.16e-4 | +7.01e-4 | +6.09e-4 | +2.92e-6 |
| 171 | 3.00e-3 | 3 | 4.01e-3 | 4.60e-3 | 4.25e-3 | 4.14e-3 | 112 | -1.00e-3 | +1.22e-3 | -2.44e-4 | -6.41e-5 |
| 172 | 3.00e-3 | 2 | 4.13e-3 | 4.63e-3 | 4.38e-3 | 4.63e-3 | 119 | -1.44e-5 | +9.71e-4 | +4.78e-4 | +4.39e-5 |
| 173 | 3.00e-3 | 2 | 4.30e-3 | 4.33e-3 | 4.32e-3 | 4.33e-3 | 91 | -5.85e-4 | +5.87e-5 | -2.63e-4 | -1.12e-5 |
| 174 | 3.00e-3 | 3 | 3.47e-3 | 4.38e-3 | 3.92e-3 | 3.47e-3 | 85 | -2.76e-3 | +1.31e-3 | -7.25e-4 | -2.25e-4 |
| 175 | 3.00e-3 | 3 | 3.63e-3 | 4.50e-3 | 3.96e-3 | 3.63e-3 | 85 | -2.55e-3 | +2.17e-3 | +7.70e-5 | -1.74e-4 |
| 176 | 3.00e-3 | 3 | 3.65e-3 | 4.25e-3 | 3.89e-3 | 3.65e-3 | 87 | -1.73e-3 | +1.40e-3 | +4.10e-6 | -1.46e-4 |
| 177 | 3.00e-3 | 3 | 3.50e-3 | 4.21e-3 | 3.84e-3 | 3.50e-3 | 79 | -2.31e-3 | +1.29e-3 | -2.40e-4 | -1.97e-4 |
| 178 | 3.00e-3 | 4 | 3.46e-3 | 3.87e-3 | 3.61e-3 | 3.57e-3 | 79 | -1.13e-3 | +1.39e-3 | +6.50e-5 | -1.15e-4 |
| 179 | 3.00e-3 | 3 | 3.33e-3 | 4.20e-3 | 3.73e-3 | 3.33e-3 | 75 | -3.10e-3 | +1.84e-3 | -3.47e-4 | -2.11e-4 |
| 180 | 3.00e-3 | 5 | 2.95e-3 | 4.33e-3 | 3.41e-3 | 2.95e-3 | 57 | -5.34e-3 | +3.43e-3 | -4.49e-4 | -3.62e-4 |
| 181 | 3.00e-3 | 3 | 3.08e-3 | 3.63e-3 | 3.27e-3 | 3.08e-3 | 57 | -2.89e-3 | +2.77e-3 | +1.34e-4 | -2.61e-4 |
| 182 | 3.00e-3 | 5 | 2.91e-3 | 3.95e-3 | 3.24e-3 | 2.91e-3 | 51 | -4.21e-3 | +3.76e-3 | -2.26e-4 | -3.04e-4 |
| 183 | 3.00e-3 | 5 | 2.55e-3 | 4.02e-3 | 3.02e-3 | 2.65e-3 | 42 | -5.84e-3 | +7.87e-3 | -3.79e-4 | -4.09e-4 |
| 184 | 3.00e-3 | 9 | 2.08e-3 | 3.44e-3 | 2.40e-3 | 2.34e-3 | 36 | -1.22e-2 | +8.32e-3 | -3.74e-4 | -2.98e-4 |
| 185 | 3.00e-3 | 7 | 1.70e-3 | 3.31e-3 | 2.22e-3 | 1.70e-3 | 23 | -1.58e-2 | +1.18e-2 | -1.97e-3 | -1.39e-3 |
| 186 | 3.00e-3 | 17 | 1.25e-3 | 3.26e-3 | 1.52e-3 | 1.50e-3 | 19 | -4.68e-2 | +4.11e-2 | +2.32e-4 | +1.08e-4 |
| 187 | 3.00e-3 | 11 | 1.25e-3 | 3.01e-3 | 1.53e-3 | 1.38e-3 | 17 | -4.46e-2 | +4.17e-2 | -3.84e-4 | -2.59e-4 |
| 188 | 3.00e-3 | 15 | 1.14e-3 | 2.86e-3 | 1.43e-3 | 1.24e-3 | 20 | -4.86e-2 | +4.48e-2 | -4.25e-4 | -3.97e-4 |
| 189 | 3.00e-3 | 13 | 1.28e-3 | 3.28e-3 | 1.57e-3 | 1.51e-3 | 23 | -3.47e-2 | +4.16e-2 | +3.43e-4 | +1.25e-4 |
| 190 | 3.00e-3 | 21 | 1.06e-3 | 3.09e-3 | 1.52e-3 | 1.06e-3 | 12 | -3.67e-2 | +2.87e-2 | -1.37e-3 | -1.98e-3 |
| 191 | 3.00e-3 | 9 | 1.03e-3 | 3.26e-3 | 1.59e-3 | 1.33e-3 | 20 | -5.74e-2 | +9.36e-2 | +5.63e-3 | +1.22e-3 |
| 192 | 3.00e-3 | 12 | 1.32e-3 | 2.91e-3 | 1.59e-3 | 1.43e-3 | 20 | -2.93e-2 | +3.30e-2 | +1.72e-4 | +2.62e-4 |
| 193 | 3.00e-3 | 12 | 1.05e-3 | 3.14e-3 | 1.70e-3 | 1.64e-3 | 23 | -4.98e-2 | +5.42e-2 | +2.32e-3 | +1.04e-3 |
| 194 | 3.00e-3 | 12 | 1.49e-3 | 3.08e-3 | 1.74e-3 | 1.56e-3 | 16 | -3.42e-2 | +2.95e-2 | -3.58e-4 | -1.33e-4 |
| 195 | 3.00e-3 | 14 | 1.07e-3 | 2.95e-3 | 1.54e-3 | 1.62e-3 | 20 | -7.76e-2 | +6.68e-2 | +3.53e-4 | +2.94e-4 |
| 196 | 3.00e-3 | 14 | 1.24e-3 | 3.12e-3 | 1.52e-3 | 1.32e-3 | 18 | -6.15e-2 | +3.94e-2 | -1.24e-3 | -7.87e-4 |
| 197 | 3.00e-3 | 16 | 1.10e-3 | 2.94e-3 | 1.42e-3 | 1.19e-3 | 15 | -5.97e-2 | +5.02e-2 | -5.31e-4 | -1.00e-3 |
| 198 | 3.00e-3 | 19 | 1.14e-3 | 2.76e-3 | 1.43e-3 | 1.54e-3 | 19 | -5.96e-2 | +5.48e-2 | +2.43e-4 | +2.72e-4 |
| 199 | 3.00e-3 | 13 | 1.10e-3 | 3.06e-3 | 1.48e-3 | 1.27e-3 | 16 | -3.24e-2 | +3.25e-2 | -1.20e-3 | -7.05e-4 |

