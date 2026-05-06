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
| nccl-async | 0.054700 | 0.9167 | +0.0042 | 1843.1 | 1356 | 36.2 | 100% | 100% | 6.2 |

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
| nccl-async | 1.9932 | 0.8046 | 0.6257 | 0.5695 | 0.5361 | 0.5111 | 0.4963 | 0.4757 | 0.4614 | 0.4562 | 0.2087 | 0.1674 | 0.1416 | 0.1234 | 0.1114 | 0.0677 | 0.0625 | 0.0601 | 0.0563 | 0.0547 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | nccl-async | 0 | cuda:0 | 0.4063 | 2.8 |
| resnet-graph | nccl-async | 1 | cuda:1 | 0.3017 | 3.6 |
| resnet-graph | nccl-async | 2 | cuda:2 | 0.2920 | 3.5 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|
| resnet-graph | nccl-async | 356 | 355 | 404 | 389 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| nccl-async | gpu2 | 1841.4 | 1.8 | epoch-boundary(199) |
| nccl-async | gpu1 | 1841.5 | 1.7 | epoch-boundary(199) |
| nccl-async | gpu0 | 1842.0 | 0.6 | epoch-boundary(199) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | nccl-async | gpu0 | 0.6s | 0.0s | 0.0s | 0.0s | 1.6s |
| resnet-graph | nccl-async | gpu1 | 1.7s | 0.0s | 0.0s | 0.0s | 2.3s |
| resnet-graph | nccl-async | gpu2 | 1.8s | 0.0s | 0.0s | 0.0s | 2.4s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | nccl-async | 511 | 0 | 1356 | 36.2 | 577/7061 | 0 | 0.0 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | nccl-async | 159.6 | 8.7% |

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
| resnet-graph | nccl-async | 197 | 1356 | 0 | 1.64e-3 | -2.96e-4 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | nccl-async | 0 | 1356 | 2.48e-2 | 5.41e-2 | 0.00e0 | 5.44e-1 | 51.0 | -1.31e-4 | 4.22e-3 |
| resnet-graph | nccl-async | 1 | 1356 | 2.54e-2 | 5.72e-2 | 0.00e0 | 7.54e-1 | 30.6 | -1.56e-4 | 6.67e-3 |
| resnet-graph | nccl-async | 2 | 1356 | 2.48e-2 | 5.46e-2 | 0.00e0 | 5.29e-1 | 18.4 | -1.49e-4 | 6.71e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | nccl-async | rank0 ↔ rank1 | +0.9942 |
| resnet-graph | nccl-async | rank0 ↔ rank2 | +0.9990 |
| resnet-graph | nccl-async | rank1 ↔ rank2 | +0.9933 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | nccl-async | 56 (0,1,2,28,35,38,43,56…148,149) | 3 (132,158,194) | 132 | 0,1,2,28,35,38,43,56…148,149 | 158,194 |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | nccl-async | 0e0 | 3 | 58 | 58 |
| resnet-graph | nccl-async | 0e0 | 5 | 29 | 29 |
| resnet-graph | nccl-async | 0e0 | 10 | 6 | 6 |
| resnet-graph | nccl-async | 1e-4 | 3 | 39 | 39 |
| resnet-graph | nccl-async | 1e-4 | 5 | 16 | 16 |
| resnet-graph | nccl-async | 1e-4 | 10 | 2 | 2 |
| resnet-graph | nccl-async | 1e-3 | 3 | 3 | 3 |
| resnet-graph | nccl-async | 1e-3 | 5 | 1 | 1 |
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
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 163 | +0.130 |
| resnet-graph | nccl-async | 3.00e-2 | 100–148 | 475 | +0.025 |
| resnet-graph | nccl-async | 3.00e-3 | 149–199 | 713 | +0.018 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | nccl-async | 1353 | +0.003 | 196 | +0.071 | -0.036 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | nccl-async | 1354 | 3.61e1–7.88e1 | 6.19e1 | 8.66e-4 | 5.23e-3 | 1.05e-1 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 165 | 75–77782 | +1.151e-5 | 0.334 | +1.210e-5 | 0.352 | 97 | +1.809e-6 | 0.026 | 38–1007 | +1.533e-3 | 0.648 |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 149 | 879–77782 | +9.601e-6 | 0.325 | +9.977e-6 | 0.326 | 96 | +1.001e-6 | 0.009 | 38–1007 | +1.452e-3 | 0.722 |
| resnet-graph | nccl-async | 3.00e-2 | 100–148 | 476 | 78230–116325 | -1.664e-6 | 0.002 | -2.209e-6 | 0.004 | 49 | -1.282e-5 | 0.196 | 31–448 | +2.029e-3 | 0.219 |
| resnet-graph | nccl-async | 3.00e-3 | 149–199 | 714 | 116460–156153 | -3.864e-6 | 0.011 | -4.044e-6 | 0.013 | 51 | -2.938e-6 | 0.011 | 34–200 | +2.160e-3 | 0.018 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | +1.533e-3 | r0: +1.485e-3, r1: +1.517e-3, r2: +1.606e-3 | r0: 0.665, r1: 0.616, r2: 0.662 | 1.08× | ✓ |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +1.452e-3 | r0: +1.399e-3, r1: +1.453e-3, r2: +1.507e-3 | r0: 0.738, r1: 0.713, r2: 0.714 | 1.08× | ✓ |
| resnet-graph | nccl-async | 3.00e-2 | 100–148 | +2.029e-3 | r0: +1.990e-3, r1: +2.048e-3, r2: +2.066e-3 | r0: 0.260, r1: 0.194, r2: 0.202 | 1.04× | ✓ |
| resnet-graph | nccl-async | 3.00e-3 | 149–199 | +2.160e-3 | r0: +2.223e-3, r1: +2.175e-3, r2: +2.108e-3 | r0: 0.021, r1: 0.016, r2: 0.015 | 1.05× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | nccl-async | `████████████████████████▆▄▄▄▄▄▄▄▅▅▅▅▄▁▁▁▁▁▁▁▁▁▁▁▁▁` | `▅██▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▆▆▇▇▆▇█▇█▇▆▇▁██▇▇▇▇▇▇▇▇█▇▇` |

### Per-Epoch Detail: resnet-graph / nccl-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 17 | 0.00e0 | 7.54e-1 | 1.08e-1 | 4.84e-2 | 19 | -5.51e-2 | +5.22e-3 | -1.06e-2 | -5.18e-3 |
| 1 | 3.00e-1 | 14 | 5.10e-2 | 9.95e-2 | 5.97e-2 | 5.43e-2 | 18 | -4.13e-2 | +3.91e-2 | +2.94e-5 | -1.40e-3 |
| 2 | 3.00e-1 | 13 | 5.61e-2 | 1.09e-1 | 7.31e-2 | 7.53e-2 | 19 | -3.48e-2 | +3.50e-2 | +9.18e-4 | +1.38e-4 |
| 3 | 3.00e-1 | 2 | 6.88e-2 | 7.47e-2 | 7.18e-2 | 7.47e-2 | 17 | -5.00e-3 | +4.80e-3 | -9.90e-5 | +1.42e-4 |
| 4 | 3.00e-1 | 1 | 6.93e-2 | 6.93e-2 | 6.93e-2 | 6.93e-2 | 322 | -2.31e-4 | -2.31e-4 | -2.31e-4 | +1.04e-4 |
| 5 | 3.00e-1 | 1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 348 | +3.44e-3 | +3.44e-3 | +3.44e-3 | +4.40e-4 |
| 6 | 3.00e-1 | 1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 350 | -7.56e-5 | -7.56e-5 | -7.56e-5 | +3.88e-4 |
| 7 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 328 | -9.83e-5 | -9.83e-5 | -9.83e-5 | +3.39e-4 |
| 9 | 3.00e-1 | 2 | 2.08e-1 | 2.11e-1 | 2.10e-1 | 2.08e-1 | 253 | -6.84e-5 | -5.56e-5 | -6.20e-5 | +2.63e-4 |
| 11 | 3.00e-1 | 1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 314 | -2.37e-4 | -2.37e-4 | -2.37e-4 | +2.12e-4 |
| 12 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 292 | +1.77e-4 | +1.77e-4 | +1.77e-4 | +2.09e-4 |
| 13 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 288 | -4.52e-5 | -4.52e-5 | -4.52e-5 | +1.83e-4 |
| 14 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 278 | -8.02e-6 | -8.02e-6 | -8.02e-6 | +1.64e-4 |
| 15 | 3.00e-1 | 1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 314 | -1.08e-4 | -1.08e-4 | -1.08e-4 | +1.37e-4 |
| 16 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 269 | +1.90e-4 | +1.90e-4 | +1.90e-4 | +1.42e-4 |
| 17 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 334 | -7.77e-5 | -7.77e-5 | -7.77e-5 | +1.20e-4 |
| 18 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 307 | +1.76e-4 | +1.76e-4 | +1.76e-4 | +1.26e-4 |
| 19 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 300 | -3.87e-5 | -3.87e-5 | -3.87e-5 | +1.09e-4 |
| 20 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 293 | -4.38e-5 | -4.38e-5 | -4.38e-5 | +9.39e-5 |
| 21 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 297 | -6.57e-5 | -6.57e-5 | -6.57e-5 | +7.79e-5 |
| 22 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 251 | +3.16e-5 | +3.16e-5 | +3.16e-5 | +7.33e-5 |
| 23 | 3.00e-1 | 1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 253 | -1.49e-4 | -1.49e-4 | -1.49e-4 | +5.10e-5 |
| 24 | 3.00e-1 | 1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 258 | -2.78e-6 | -2.78e-6 | -2.78e-6 | +4.56e-5 |
| 25 | 3.00e-1 | 1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 252 | +5.18e-5 | +5.18e-5 | +5.18e-5 | +4.63e-5 |
| 26 | 3.00e-1 | 1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 280 | -1.41e-5 | -1.41e-5 | -1.41e-5 | +4.02e-5 |
| 27 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 287 | +6.77e-5 | +6.77e-5 | +6.77e-5 | +4.30e-5 |
| 28 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 249 | +9.88e-5 | +9.88e-5 | +9.88e-5 | +4.86e-5 |
| 29 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 264 | -1.32e-4 | -1.32e-4 | -1.32e-4 | +3.05e-5 |
| 30 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 259 | +1.27e-5 | +1.27e-5 | +1.27e-5 | +2.87e-5 |
| 31 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 302 | -4.71e-6 | -4.71e-6 | -4.71e-6 | +2.54e-5 |
| 32 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 282 | +1.38e-4 | +1.38e-4 | +1.38e-4 | +3.67e-5 |
| 33 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 280 | -4.25e-5 | -4.25e-5 | -4.25e-5 | +2.87e-5 |
| 34 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 290 | +1.23e-5 | +1.23e-5 | +1.23e-5 | +2.71e-5 |
| 35 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 257 | +1.24e-5 | +1.24e-5 | +1.24e-5 | +2.56e-5 |
| 36 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 259 | -1.26e-4 | -1.26e-4 | -1.26e-4 | +1.04e-5 |
| 37 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 284 | +2.61e-6 | +2.61e-6 | +2.61e-6 | +9.63e-6 |
| 38 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 262 | +1.29e-4 | +1.29e-4 | +1.29e-4 | +2.16e-5 |
| 39 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 240 | -1.33e-4 | -1.33e-4 | -1.33e-4 | +6.06e-6 |
| 40 | 3.00e-1 | 2 | 1.92e-1 | 1.94e-1 | 1.93e-1 | 1.92e-1 | 213 | -1.28e-4 | -3.49e-5 | -8.12e-5 | -1.01e-5 |
| 41 | 3.00e-1 | 1 | 1.89e-1 | 1.89e-1 | 1.89e-1 | 1.89e-1 | 243 | -7.69e-5 | -7.69e-5 | -7.69e-5 | -1.67e-5 |
| 42 | 3.00e-1 | 1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 250 | +1.10e-4 | +1.10e-4 | +1.10e-4 | -4.08e-6 |
| 43 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 240 | +1.27e-4 | +1.27e-4 | +1.27e-4 | +9.07e-6 |
| 44 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 276 | -6.73e-5 | -6.73e-5 | -6.73e-5 | +1.43e-6 |
| 45 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 271 | +1.12e-4 | +1.12e-4 | +1.12e-4 | +1.25e-5 |
| 46 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 252 | -3.76e-5 | -3.76e-5 | -3.76e-5 | +7.46e-6 |
| 47 | 3.00e-1 | 2 | 1.97e-1 | 2.03e-1 | 2.00e-1 | 2.03e-1 | 211 | -5.68e-5 | +1.27e-4 | +3.52e-5 | +1.36e-5 |
| 49 | 3.00e-1 | 2 | 1.87e-1 | 2.02e-1 | 1.95e-1 | 2.02e-1 | 211 | -2.89e-4 | +3.60e-4 | +3.54e-5 | +2.10e-5 |
| 50 | 3.00e-1 | 1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 289 | -2.20e-4 | -2.20e-4 | -2.20e-4 | -3.11e-6 |
| 51 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 236 | +3.62e-4 | +3.62e-4 | +3.62e-4 | +3.34e-5 |
| 52 | 3.00e-1 | 1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 223 | -2.51e-4 | -2.51e-4 | -2.51e-4 | +4.93e-6 |
| 53 | 3.00e-1 | 1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 258 | -4.47e-6 | -4.47e-6 | -4.47e-6 | +3.99e-6 |
| 54 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 241 | +9.45e-5 | +9.45e-5 | +9.45e-5 | +1.30e-5 |
| 55 | 3.00e-1 | 2 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 230 | -8.95e-5 | +2.40e-6 | -4.36e-5 | +2.75e-6 |
| 56 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 252 | +4.79e-6 | +4.79e-6 | +4.79e-6 | +2.95e-6 |
| 57 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 255 | +5.64e-5 | +5.64e-5 | +5.64e-5 | +8.29e-6 |
| 58 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 252 | +8.17e-5 | +8.17e-5 | +8.17e-5 | +1.56e-5 |
| 59 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 219 | -5.46e-5 | -5.46e-5 | -5.46e-5 | +8.61e-6 |
| 60 | 3.00e-1 | 2 | 1.91e-1 | 1.97e-1 | 1.94e-1 | 1.97e-1 | 190 | -1.82e-4 | +1.56e-4 | -1.31e-5 | +6.19e-6 |
| 61 | 3.00e-1 | 1 | 1.81e-1 | 1.81e-1 | 1.81e-1 | 1.81e-1 | 217 | -4.01e-4 | -4.01e-4 | -4.01e-4 | -3.45e-5 |
| 62 | 3.00e-1 | 1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 228 | +2.80e-4 | +2.80e-4 | +2.80e-4 | -3.09e-6 |
| 63 | 3.00e-1 | 1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 218 | +1.07e-4 | +1.07e-4 | +1.07e-4 | +7.90e-6 |
| 64 | 3.00e-1 | 2 | 1.92e-1 | 1.93e-1 | 1.92e-1 | 1.93e-1 | 190 | -1.30e-4 | +3.41e-5 | -4.80e-5 | -1.91e-6 |
| 65 | 3.00e-1 | 1 | 1.85e-1 | 1.85e-1 | 1.85e-1 | 1.85e-1 | 224 | -1.77e-4 | -1.77e-4 | -1.77e-4 | -1.94e-5 |
| 66 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 237 | +2.40e-4 | +2.40e-4 | +2.40e-4 | +6.53e-6 |
| 67 | 3.00e-1 | 2 | 1.95e-1 | 1.97e-1 | 1.96e-1 | 1.95e-1 | 179 | -7.75e-5 | +2.39e-5 | -2.68e-5 | -3.05e-7 |
| 68 | 3.00e-1 | 1 | 1.74e-1 | 1.74e-1 | 1.74e-1 | 1.74e-1 | 179 | -6.17e-4 | -6.17e-4 | -6.17e-4 | -6.20e-5 |
| 69 | 3.00e-1 | 1 | 1.82e-1 | 1.82e-1 | 1.82e-1 | 1.82e-1 | 231 | +1.87e-4 | +1.87e-4 | +1.87e-4 | -3.71e-5 |
| 70 | 3.00e-1 | 2 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 189 | -8.81e-6 | +3.05e-4 | +1.48e-4 | -3.53e-6 |
| 71 | 3.00e-1 | 1 | 1.82e-1 | 1.82e-1 | 1.82e-1 | 1.82e-1 | 243 | -2.85e-4 | -2.85e-4 | -2.85e-4 | -3.17e-5 |
| 72 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 211 | +3.86e-4 | +3.86e-4 | +3.86e-4 | +1.01e-5 |
| 73 | 3.00e-1 | 2 | 1.87e-1 | 1.92e-1 | 1.89e-1 | 1.92e-1 | 175 | -2.63e-4 | +1.39e-4 | -6.18e-5 | -1.57e-6 |
| 74 | 3.00e-1 | 1 | 1.78e-1 | 1.78e-1 | 1.78e-1 | 1.78e-1 | 220 | -3.39e-4 | -3.39e-4 | -3.39e-4 | -3.53e-5 |
| 75 | 3.00e-1 | 1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 220 | +3.44e-4 | +3.44e-4 | +3.44e-4 | +2.66e-6 |
| 76 | 3.00e-1 | 2 | 1.88e-1 | 1.94e-1 | 1.91e-1 | 1.88e-1 | 175 | -1.81e-4 | +6.11e-5 | -5.97e-5 | -1.04e-5 |
| 77 | 3.00e-1 | 1 | 1.75e-1 | 1.75e-1 | 1.75e-1 | 1.75e-1 | 210 | -3.57e-4 | -3.57e-4 | -3.57e-4 | -4.51e-5 |
| 78 | 3.00e-1 | 2 | 1.84e-1 | 1.93e-1 | 1.88e-1 | 1.84e-1 | 165 | -3.01e-4 | +5.14e-4 | +1.07e-4 | -2.03e-5 |
| 79 | 3.00e-1 | 1 | 1.78e-1 | 1.78e-1 | 1.78e-1 | 1.78e-1 | 177 | -1.89e-4 | -1.89e-4 | -1.89e-4 | -3.72e-5 |
| 80 | 3.00e-1 | 2 | 1.83e-1 | 1.86e-1 | 1.85e-1 | 1.86e-1 | 165 | +8.08e-5 | +1.71e-4 | +1.26e-4 | -6.62e-6 |
| 81 | 3.00e-1 | 1 | 1.74e-1 | 1.74e-1 | 1.74e-1 | 1.74e-1 | 210 | -3.22e-4 | -3.22e-4 | -3.22e-4 | -3.81e-5 |
| 82 | 3.00e-1 | 2 | 1.90e-1 | 2.00e-1 | 1.95e-1 | 2.00e-1 | 165 | +3.27e-4 | +3.82e-4 | +3.55e-4 | +3.62e-5 |
| 83 | 3.00e-1 | 1 | 1.69e-1 | 1.69e-1 | 1.69e-1 | 1.69e-1 | 205 | -8.42e-4 | -8.42e-4 | -8.42e-4 | -5.16e-5 |
| 84 | 3.00e-1 | 2 | 1.87e-1 | 1.93e-1 | 1.90e-1 | 1.87e-1 | 145 | -2.24e-4 | +6.94e-4 | +2.35e-4 | -1.78e-6 |
| 85 | 3.00e-1 | 2 | 1.68e-1 | 1.84e-1 | 1.76e-1 | 1.84e-1 | 145 | -5.62e-4 | +6.46e-4 | +4.22e-5 | +1.26e-5 |
| 86 | 3.00e-1 | 2 | 1.66e-1 | 1.78e-1 | 1.72e-1 | 1.78e-1 | 154 | -5.65e-4 | +4.50e-4 | -5.75e-5 | +4.37e-6 |
| 87 | 3.00e-1 | 1 | 1.72e-1 | 1.72e-1 | 1.72e-1 | 1.72e-1 | 196 | -1.79e-4 | -1.79e-4 | -1.79e-4 | -1.39e-5 |
| 88 | 3.00e-1 | 1 | 1.86e-1 | 1.86e-1 | 1.86e-1 | 1.86e-1 | 204 | +3.67e-4 | +3.67e-4 | +3.67e-4 | +2.42e-5 |
| 89 | 3.00e-1 | 2 | 1.85e-1 | 1.89e-1 | 1.87e-1 | 1.85e-1 | 138 | -1.22e-4 | +8.46e-5 | -1.85e-5 | +1.50e-5 |
| 90 | 3.00e-1 | 2 | 1.63e-1 | 1.85e-1 | 1.74e-1 | 1.85e-1 | 131 | -6.92e-4 | +9.49e-4 | +1.28e-4 | +4.48e-5 |
| 91 | 3.00e-1 | 2 | 1.60e-1 | 1.85e-1 | 1.72e-1 | 1.85e-1 | 139 | -8.08e-4 | +1.03e-3 | +1.13e-4 | +6.70e-5 |
| 92 | 3.00e-1 | 1 | 1.66e-1 | 1.66e-1 | 1.66e-1 | 1.66e-1 | 168 | -6.46e-4 | -6.46e-4 | -6.46e-4 | -4.24e-6 |
| 93 | 3.00e-1 | 3 | 1.59e-1 | 1.84e-1 | 1.74e-1 | 1.59e-1 | 113 | -1.28e-3 | +4.31e-4 | -2.19e-4 | -7.88e-5 |
| 94 | 3.00e-1 | 2 | 1.51e-1 | 1.72e-1 | 1.61e-1 | 1.72e-1 | 126 | -3.73e-4 | +1.02e-3 | +3.26e-4 | +5.06e-6 |
| 95 | 3.00e-1 | 2 | 1.63e-1 | 1.75e-1 | 1.69e-1 | 1.75e-1 | 129 | -3.43e-4 | +5.45e-4 | +1.01e-4 | +2.78e-5 |
| 96 | 3.00e-1 | 2 | 1.60e-1 | 1.81e-1 | 1.70e-1 | 1.81e-1 | 131 | -5.46e-4 | +9.38e-4 | +1.96e-4 | +6.72e-5 |
| 97 | 3.00e-1 | 2 | 1.66e-1 | 1.82e-1 | 1.74e-1 | 1.82e-1 | 131 | -5.11e-4 | +7.44e-4 | +1.16e-4 | +8.28e-5 |
| 98 | 3.00e-1 | 2 | 1.63e-1 | 1.75e-1 | 1.69e-1 | 1.75e-1 | 131 | -6.96e-4 | +5.38e-4 | -7.93e-5 | +5.81e-5 |
| 99 | 3.00e-1 | 1 | 1.63e-1 | 1.63e-1 | 1.63e-1 | 1.63e-1 | 170 | -4.46e-4 | -4.46e-4 | -4.46e-4 | +7.75e-6 |
| 100 | 3.00e-2 | 3 | 1.59e-2 | 1.83e-1 | 1.25e-1 | 1.59e-2 | 112 | -2.16e-2 | +6.91e-4 | -7.03e-3 | -2.11e-3 |
| 101 | 3.00e-2 | 2 | 1.48e-2 | 1.77e-2 | 1.62e-2 | 1.77e-2 | 112 | -4.79e-4 | +1.61e-3 | +5.65e-4 | -1.60e-3 |
| 102 | 3.00e-2 | 2 | 1.59e-2 | 1.88e-2 | 1.74e-2 | 1.88e-2 | 112 | -6.93e-4 | +1.49e-3 | +4.01e-4 | -1.21e-3 |
| 103 | 3.00e-2 | 3 | 1.73e-2 | 1.93e-2 | 1.81e-2 | 1.78e-2 | 112 | -7.33e-4 | +1.00e-3 | -1.02e-4 | -9.08e-4 |
| 104 | 3.00e-2 | 2 | 1.84e-2 | 1.98e-2 | 1.91e-2 | 1.98e-2 | 106 | +2.32e-4 | +7.09e-4 | +4.71e-4 | -6.44e-4 |
| 105 | 3.00e-2 | 2 | 1.76e-2 | 2.04e-2 | 1.90e-2 | 2.04e-2 | 106 | -8.36e-4 | +1.41e-3 | +2.89e-4 | -4.56e-4 |
| 106 | 3.00e-2 | 3 | 1.82e-2 | 2.04e-2 | 1.92e-2 | 1.82e-2 | 91 | -1.29e-3 | +6.80e-4 | -3.63e-4 | -4.39e-4 |
| 107 | 3.00e-2 | 3 | 1.75e-2 | 2.11e-2 | 1.89e-2 | 1.81e-2 | 85 | -1.80e-3 | +2.11e-3 | +1.11e-5 | -3.32e-4 |
| 108 | 3.00e-2 | 2 | 1.84e-2 | 2.09e-2 | 1.96e-2 | 2.09e-2 | 86 | +1.01e-4 | +1.49e-3 | +7.95e-4 | -1.11e-4 |
| 109 | 3.00e-2 | 3 | 1.82e-2 | 2.42e-2 | 2.02e-2 | 1.82e-2 | 78 | -3.66e-3 | +3.27e-3 | -4.26e-4 | -2.25e-4 |
| 110 | 3.00e-2 | 4 | 1.79e-2 | 2.05e-2 | 1.88e-2 | 1.84e-2 | 73 | -1.79e-3 | +1.36e-3 | +1.16e-5 | -1.54e-4 |
| 111 | 3.00e-2 | 3 | 1.72e-2 | 2.15e-2 | 1.93e-2 | 1.91e-2 | 86 | -1.38e-3 | +2.64e-3 | +2.25e-4 | -6.01e-5 |
| 112 | 3.00e-2 | 3 | 1.88e-2 | 2.23e-2 | 2.03e-2 | 1.88e-2 | 71 | -2.43e-3 | +1.70e-3 | -1.66e-4 | -1.15e-4 |
| 113 | 3.00e-2 | 5 | 1.78e-2 | 2.27e-2 | 1.92e-2 | 1.78e-2 | 64 | -3.20e-3 | +2.75e-3 | -2.02e-4 | -1.79e-4 |
| 114 | 3.00e-2 | 3 | 1.75e-2 | 2.28e-2 | 1.97e-2 | 1.88e-2 | 53 | -3.60e-3 | +4.09e-3 | +1.01e-4 | -1.38e-4 |
| 115 | 3.00e-2 | 4 | 1.63e-2 | 2.12e-2 | 1.76e-2 | 1.64e-2 | 50 | -4.90e-3 | +4.61e-3 | -3.79e-4 | -2.44e-4 |
| 116 | 3.00e-2 | 6 | 1.54e-2 | 2.21e-2 | 1.73e-2 | 1.66e-2 | 47 | -7.65e-3 | +5.25e-3 | -5.24e-5 | -1.75e-4 |
| 117 | 3.00e-2 | 6 | 1.27e-2 | 2.18e-2 | 1.62e-2 | 1.27e-2 | 30 | -8.59e-3 | +6.26e-3 | -1.45e-3 | -9.41e-4 |
| 118 | 3.00e-2 | 11 | 1.01e-2 | 2.03e-2 | 1.23e-2 | 1.01e-2 | 20 | -1.94e-2 | +1.70e-2 | -9.12e-4 | -1.13e-3 |
| 119 | 3.00e-2 | 14 | 9.27e-3 | 1.90e-2 | 1.14e-2 | 1.17e-2 | 25 | -3.29e-2 | +3.28e-2 | +3.97e-4 | -7.87e-5 |
| 120 | 3.00e-2 | 9 | 1.01e-2 | 1.91e-2 | 1.18e-2 | 1.02e-2 | 17 | -2.98e-2 | +1.87e-2 | -1.16e-3 | -8.19e-4 |
| 121 | 3.00e-2 | 10 | 7.89e-3 | 1.82e-2 | 1.26e-2 | 1.29e-2 | 25 | -4.64e-2 | +4.57e-2 | +1.31e-3 | +1.89e-4 |
| 122 | 3.00e-2 | 14 | 9.16e-3 | 1.99e-2 | 1.12e-2 | 9.42e-3 | 16 | -3.08e-2 | +2.17e-2 | -1.09e-3 | -8.13e-4 |
| 123 | 3.00e-2 | 16 | 7.19e-3 | 1.99e-2 | 1.01e-2 | 1.09e-2 | 18 | -5.64e-2 | +5.19e-2 | +4.87e-4 | +4.82e-4 |
| 124 | 3.00e-2 | 15 | 8.33e-3 | 2.19e-2 | 1.07e-2 | 1.02e-2 | 16 | -4.22e-2 | +4.44e-2 | -1.10e-4 | +3.18e-5 |
| 125 | 3.00e-2 | 14 | 8.25e-3 | 2.18e-2 | 1.10e-2 | 1.23e-2 | 18 | -4.17e-2 | +4.07e-2 | +1.50e-4 | +6.55e-4 |
| 126 | 3.00e-2 | 13 | 9.39e-3 | 2.35e-2 | 1.21e-2 | 1.22e-2 | 20 | -6.12e-2 | +5.41e-2 | +4.05e-4 | +4.54e-4 |
| 127 | 3.00e-2 | 12 | 1.09e-2 | 2.29e-2 | 1.35e-2 | 1.33e-2 | 22 | -2.70e-2 | +2.79e-2 | +5.71e-5 | +1.95e-4 |
| 128 | 3.00e-2 | 14 | 1.07e-2 | 2.20e-2 | 1.25e-2 | 1.11e-2 | 19 | -3.09e-2 | +2.61e-2 | -6.61e-4 | -5.63e-4 |
| 129 | 3.00e-2 | 15 | 9.65e-3 | 2.30e-2 | 1.17e-2 | 1.10e-2 | 16 | -4.62e-2 | +4.22e-2 | -3.71e-5 | -1.20e-4 |
| 130 | 3.00e-2 | 22 | 1.01e-2 | 2.46e-2 | 1.19e-2 | 1.07e-2 | 15 | -4.86e-2 | +4.57e-2 | -1.88e-4 | -3.88e-4 |
| 131 | 3.00e-2 | 7 | 9.53e-3 | 2.22e-2 | 1.38e-2 | 1.99e-2 | 36 | -5.77e-2 | +5.48e-2 | +3.02e-3 | +1.49e-3 |
| 132 | 3.00e-2 | 9 | 1.50e-2 | 2.84e-2 | 1.82e-2 | 1.65e-2 | 28 | -1.65e-2 | +1.05e-2 | -9.62e-4 | +7.27e-5 |
| 133 | 3.00e-2 | 13 | 1.16e-2 | 2.54e-2 | 1.46e-2 | 1.33e-2 | 16 | -1.84e-2 | +1.61e-2 | -9.28e-4 | -4.15e-4 |
| 134 | 3.00e-2 | 15 | 1.04e-2 | 2.57e-2 | 1.30e-2 | 1.32e-2 | 18 | -6.07e-2 | +6.16e-2 | +7.00e-4 | +1.99e-4 |
| 135 | 3.00e-2 | 16 | 9.68e-3 | 2.49e-2 | 1.25e-2 | 1.25e-2 | 19 | -5.53e-2 | +4.90e-2 | +2.69e-4 | +3.53e-4 |
| 136 | 3.00e-2 | 15 | 1.15e-2 | 2.71e-2 | 1.37e-2 | 1.23e-2 | 17 | -4.50e-2 | +4.37e-2 | -2.73e-4 | -3.12e-4 |
| 137 | 3.00e-2 | 13 | 1.16e-2 | 2.66e-2 | 1.49e-2 | 1.58e-2 | 26 | -4.46e-2 | +3.91e-2 | +7.31e-4 | +4.12e-4 |
| 138 | 3.00e-2 | 15 | 1.18e-2 | 3.15e-2 | 1.72e-2 | 1.82e-2 | 27 | -3.36e-2 | +3.35e-2 | -2.92e-4 | -1.19e-4 |
| 139 | 3.00e-2 | 7 | 1.41e-2 | 2.95e-2 | 1.82e-2 | 1.41e-2 | 22 | -2.24e-2 | +1.73e-2 | -2.32e-3 | -1.46e-3 |
| 140 | 3.00e-2 | 13 | 1.20e-2 | 3.04e-2 | 1.57e-2 | 1.20e-2 | 17 | -3.11e-2 | +2.97e-2 | -8.58e-4 | -1.48e-3 |
| 141 | 3.00e-2 | 20 | 1.06e-2 | 2.69e-2 | 1.42e-2 | 1.58e-2 | 21 | -4.36e-2 | +4.67e-2 | +5.64e-4 | +5.64e-4 |
| 142 | 3.00e-2 | 9 | 1.19e-2 | 3.39e-2 | 1.62e-2 | 1.54e-2 | 20 | -7.39e-2 | +5.40e-2 | -9.24e-4 | -2.38e-4 |
| 143 | 3.00e-2 | 13 | 1.35e-2 | 3.12e-2 | 1.70e-2 | 1.50e-2 | 18 | -3.18e-2 | +3.59e-2 | -2.05e-4 | -5.11e-4 |
| 144 | 3.00e-2 | 19 | 1.29e-2 | 3.11e-2 | 1.56e-2 | 1.49e-2 | 19 | -5.53e-2 | +4.71e-2 | -1.14e-4 | -1.59e-4 |
| 145 | 3.00e-2 | 10 | 1.26e-2 | 2.97e-2 | 1.60e-2 | 1.46e-2 | 19 | -4.11e-2 | +4.07e-2 | -1.91e-4 | -2.91e-4 |
| 146 | 3.00e-2 | 13 | 1.32e-2 | 3.17e-2 | 1.64e-2 | 1.54e-2 | 19 | -5.03e-2 | +3.94e-2 | -3.53e-4 | -3.04e-4 |
| 147 | 3.00e-2 | 16 | 1.01e-2 | 2.98e-2 | 1.44e-2 | 1.38e-2 | 20 | -9.11e-2 | +5.28e-2 | -1.38e-3 | -6.37e-4 |
| 148 | 3.00e-2 | 15 | 1.26e-2 | 3.27e-2 | 1.61e-2 | 1.75e-2 | 16 | -4.99e-2 | +3.77e-2 | -1.16e-4 | +4.46e-4 |
| 149 | 3.00e-3 | 18 | 1.49e-3 | 3.47e-2 | 1.55e-2 | 1.62e-3 | 20 | -1.19e-1 | +5.88e-2 | -8.20e-3 | -1.29e-2 |
| 150 | 3.00e-3 | 11 | 1.20e-3 | 3.13e-3 | 1.58e-3 | 1.20e-3 | 15 | -3.79e-2 | +2.88e-2 | -2.55e-3 | -5.95e-3 |
| 151 | 3.00e-3 | 17 | 1.03e-3 | 2.76e-3 | 1.37e-3 | 1.59e-3 | 16 | -6.19e-2 | +4.96e-2 | +1.00e-3 | +4.32e-4 |
| 152 | 3.00e-3 | 16 | 1.11e-3 | 3.15e-3 | 1.46e-3 | 1.45e-3 | 15 | -6.94e-2 | +5.05e-2 | -3.74e-4 | +7.77e-6 |
| 153 | 3.00e-3 | 14 | 1.16e-3 | 3.07e-3 | 1.54e-3 | 1.60e-3 | 20 | -5.04e-2 | +5.12e-2 | +1.88e-4 | +2.70e-4 |
| 154 | 3.00e-3 | 15 | 1.12e-3 | 3.14e-3 | 1.43e-3 | 1.40e-3 | 19 | -5.54e-2 | +3.99e-2 | -4.56e-4 | +2.86e-4 |
| 155 | 3.00e-3 | 19 | 1.08e-3 | 3.24e-3 | 1.41e-3 | 1.33e-3 | 19 | -6.70e-2 | +5.49e-2 | -1.38e-4 | +1.34e-4 |
| 156 | 3.00e-3 | 10 | 1.28e-3 | 3.11e-3 | 1.61e-3 | 1.28e-3 | 16 | -4.42e-2 | +4.09e-2 | -7.67e-4 | -9.74e-4 |
| 157 | 3.00e-3 | 20 | 1.19e-3 | 3.17e-3 | 1.43e-3 | 1.40e-3 | 16 | -3.80e-2 | +4.52e-2 | +1.81e-5 | +2.91e-4 |
| 158 | 3.00e-3 | 10 | 1.13e-3 | 2.97e-3 | 1.55e-3 | 1.71e-3 | 22 | -5.41e-2 | +6.19e-2 | +2.10e-3 | +1.27e-3 |
| 159 | 3.00e-3 | 14 | 1.28e-3 | 3.15e-3 | 1.59e-3 | 1.40e-3 | 38 | -5.68e-2 | +4.09e-2 | -1.12e-3 | -5.33e-4 |
| 160 | 3.00e-3 | 7 | 2.15e-3 | 3.57e-3 | 2.53e-3 | 2.25e-3 | 30 | -1.11e-2 | +1.05e-2 | +5.65e-4 | -2.04e-4 |
| 161 | 3.00e-3 | 9 | 1.71e-3 | 3.31e-3 | 2.12e-3 | 1.71e-3 | 20 | -1.65e-2 | +1.76e-2 | -1.21e-3 | -1.14e-3 |
| 162 | 3.00e-3 | 15 | 1.16e-3 | 3.24e-3 | 1.51e-3 | 1.44e-3 | 19 | -5.40e-2 | +4.82e-2 | -2.34e-5 | -1.41e-4 |
| 163 | 3.00e-3 | 14 | 1.31e-3 | 3.08e-3 | 1.63e-3 | 1.56e-3 | 18 | -4.49e-2 | +3.81e-2 | -1.66e-7 | -7.36e-5 |
| 164 | 3.00e-3 | 22 | 1.10e-3 | 2.99e-3 | 1.47e-3 | 1.22e-3 | 14 | -4.02e-2 | +3.84e-2 | -1.40e-3 | -1.54e-3 |
| 165 | 3.00e-3 | 10 | 1.06e-3 | 2.95e-3 | 1.44e-3 | 1.42e-3 | 17 | -5.68e-2 | +7.31e-2 | +1.73e-3 | +3.02e-4 |
| 166 | 3.00e-3 | 16 | 1.09e-3 | 2.90e-3 | 1.46e-3 | 1.45e-3 | 18 | -4.64e-2 | +3.67e-2 | -8.90e-4 | -1.87e-4 |
| 167 | 3.00e-3 | 16 | 1.22e-3 | 3.01e-3 | 1.51e-3 | 1.70e-3 | 18 | -5.88e-2 | +5.07e-2 | +5.26e-4 | +7.27e-4 |
| 168 | 3.00e-3 | 14 | 1.29e-3 | 2.86e-3 | 1.88e-3 | 1.77e-3 | 22 | -4.66e-2 | +3.35e-2 | -2.02e-4 | -2.88e-4 |
| 169 | 3.00e-3 | 9 | 1.55e-3 | 3.15e-3 | 1.83e-3 | 1.61e-3 | 17 | -3.65e-2 | +3.74e-2 | -8.09e-5 | -4.94e-4 |
| 170 | 3.00e-3 | 16 | 1.14e-3 | 3.13e-3 | 1.56e-3 | 1.24e-3 | 15 | -3.24e-2 | +4.49e-2 | -4.31e-4 | -9.44e-4 |
| 171 | 3.00e-3 | 13 | 1.07e-3 | 3.33e-3 | 1.70e-3 | 1.61e-3 | 18 | -5.57e-2 | +4.97e-2 | -8.69e-4 | -8.82e-4 |
| 172 | 3.00e-3 | 11 | 1.42e-3 | 3.45e-3 | 1.92e-3 | 1.92e-3 | 23 | -3.87e-2 | +4.00e-2 | +1.26e-3 | +3.38e-4 |
| 173 | 3.00e-3 | 12 | 1.28e-3 | 3.11e-3 | 1.84e-3 | 1.61e-3 | 20 | -2.81e-2 | +2.11e-2 | -8.68e-4 | -6.31e-4 |
| 174 | 3.00e-3 | 11 | 1.76e-3 | 3.40e-3 | 2.02e-3 | 1.81e-3 | 19 | -2.03e-2 | +2.52e-2 | +1.52e-4 | -3.70e-4 |
| 175 | 3.00e-3 | 21 | 1.09e-3 | 3.10e-3 | 1.58e-3 | 1.53e-3 | 20 | -5.10e-2 | +3.54e-2 | -8.19e-4 | -3.07e-4 |
| 176 | 3.00e-3 | 8 | 1.36e-3 | 3.52e-3 | 1.85e-3 | 1.62e-3 | 20 | -3.83e-2 | +3.36e-2 | -1.43e-3 | -9.53e-4 |
| 177 | 3.00e-3 | 14 | 1.39e-3 | 3.11e-3 | 1.68e-3 | 1.41e-3 | 16 | -4.50e-2 | +4.25e-2 | -4.72e-4 | -1.11e-3 |
| 178 | 3.00e-3 | 17 | 1.10e-3 | 3.07e-3 | 1.52e-3 | 1.74e-3 | 43 | -6.11e-2 | +5.33e-2 | +6.66e-5 | +2.14e-4 |
| 179 | 3.00e-3 | 6 | 2.26e-3 | 4.13e-3 | 2.76e-3 | 2.27e-3 | 32 | -1.37e-2 | +9.42e-3 | -3.16e-4 | -2.33e-4 |
| 180 | 3.00e-3 | 11 | 1.64e-3 | 4.22e-3 | 2.11e-3 | 1.84e-3 | 17 | -3.08e-2 | +2.34e-2 | -1.40e-3 | -8.87e-4 |
| 181 | 3.00e-3 | 16 | 1.34e-3 | 3.32e-3 | 1.74e-3 | 1.70e-3 | 15 | -3.92e-2 | +4.35e-2 | +2.12e-4 | +1.11e-4 |
| 182 | 3.00e-3 | 13 | 1.29e-3 | 3.20e-3 | 1.57e-3 | 1.42e-3 | 19 | -6.29e-2 | +6.13e-2 | -1.08e-4 | -2.93e-4 |
| 183 | 3.00e-3 | 15 | 1.10e-3 | 3.34e-3 | 1.63e-3 | 1.97e-3 | 24 | -6.62e-2 | +5.01e-2 | +3.67e-4 | +9.50e-4 |
| 184 | 3.00e-3 | 16 | 1.34e-3 | 3.19e-3 | 1.64e-3 | 1.34e-3 | 20 | -4.36e-2 | +2.76e-2 | -1.52e-3 | -9.68e-4 |
| 185 | 3.00e-3 | 14 | 1.33e-3 | 3.31e-3 | 1.71e-3 | 1.73e-3 | 19 | -5.68e-2 | +5.05e-2 | +9.97e-4 | +3.85e-4 |
| 186 | 3.00e-3 | 13 | 1.37e-3 | 3.36e-3 | 1.75e-3 | 1.89e-3 | 22 | -4.72e-2 | +3.74e-2 | -4.69e-4 | +1.05e-4 |
| 187 | 3.00e-3 | 15 | 1.61e-3 | 3.39e-3 | 1.96e-3 | 1.73e-3 | 18 | -3.60e-2 | +2.69e-2 | -3.79e-4 | -4.85e-4 |
| 188 | 3.00e-3 | 11 | 1.39e-3 | 3.54e-3 | 1.75e-3 | 1.43e-3 | 17 | -3.39e-2 | +3.77e-2 | -4.85e-4 | -7.98e-4 |
| 189 | 3.00e-3 | 15 | 1.37e-3 | 3.32e-3 | 1.74e-3 | 1.75e-3 | 18 | -4.40e-2 | +4.64e-2 | +9.61e-4 | +5.41e-4 |
| 190 | 3.00e-3 | 19 | 1.35e-3 | 3.01e-3 | 1.74e-3 | 1.74e-3 | 23 | -4.38e-2 | +3.77e-2 | -9.83e-5 | -3.76e-5 |
| 191 | 3.00e-3 | 9 | 1.45e-3 | 3.82e-3 | 1.88e-3 | 1.71e-3 | 16 | -5.63e-2 | +4.45e-2 | -6.56e-4 | -3.99e-4 |
| 192 | 3.00e-3 | 16 | 1.18e-3 | 3.09e-3 | 1.54e-3 | 1.70e-3 | 16 | -5.00e-2 | +5.30e-2 | +4.94e-4 | +9.35e-4 |
| 193 | 3.00e-3 | 17 | 1.13e-3 | 3.34e-3 | 1.49e-3 | 1.85e-3 | 18 | -4.93e-2 | +5.59e-2 | +4.47e-4 | +1.60e-3 |
| 194 | 3.00e-3 | 20 | 1.28e-3 | 3.30e-3 | 1.58e-3 | 1.50e-3 | 20 | -3.64e-2 | +5.11e-2 | -2.50e-5 | +3.22e-5 |
| 195 | 3.00e-3 | 12 | 1.24e-3 | 3.47e-3 | 1.65e-3 | 1.24e-3 | 20 | -5.01e-2 | +3.82e-2 | -1.63e-3 | -1.40e-3 |
| 196 | 3.00e-3 | 18 | 1.42e-3 | 4.05e-3 | 1.82e-3 | 1.63e-3 | 18 | -3.78e-2 | +3.80e-2 | -3.06e-4 | -4.79e-4 |
| 197 | 3.00e-3 | 9 | 1.38e-3 | 3.42e-3 | 1.79e-3 | 1.80e-3 | 20 | -4.43e-2 | +4.17e-2 | +3.66e-4 | +1.59e-4 |
| 198 | 3.00e-3 | 20 | 1.21e-3 | 3.66e-3 | 1.70e-3 | 1.55e-3 | 19 | -2.71e-2 | +3.85e-2 | -5.84e-4 | -2.70e-4 |
| 199 | 3.00e-3 | 10 | 1.36e-3 | 3.36e-3 | 1.76e-3 | 1.64e-3 | 18 | -3.54e-2 | +3.59e-2 | -5.05e-4 | -2.96e-4 |

