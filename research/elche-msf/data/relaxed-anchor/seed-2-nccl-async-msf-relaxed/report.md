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
| nccl-async | 0.048799 | 0.9204 | +0.0079 | 1967.5 | 790 | 38.3 | 100% | 100% | 6.6 |

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
| nccl-async | 1.9657 | 0.7091 | 0.5483 | 0.5001 | 0.4735 | 0.4454 | 0.4408 | 0.4318 | 0.4171 | 0.4199 | 0.1756 | 0.1434 | 0.1414 | 0.1328 | 0.1307 | 0.0678 | 0.0602 | 0.0548 | 0.0505 | 0.0488 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | nccl-async | 0 | cuda:0 | 0.3977 | 2.5 |
| resnet-graph | nccl-async | 1 | cuda:1 | 0.3070 | 3.2 |
| resnet-graph | nccl-async | 2 | cuda:2 | 0.2953 | 3.4 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|
| resnet-graph | nccl-async | 356 | 355 | 398 | 394 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| nccl-async | gpu2 | 1966.3 | 1.3 | epoch-boundary(199) |
| nccl-async | gpu1 | 1966.4 | 1.2 | epoch-boundary(199) |
| nccl-async | gpu0 | 1966.6 | 0.5 | epoch-boundary(199) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | nccl-async | gpu0 | 0.5s | 0.0s | 0.0s | 0.0s | 2.8s |
| resnet-graph | nccl-async | gpu1 | 1.2s | 0.0s | 0.0s | 0.0s | 2.0s |
| resnet-graph | nccl-async | gpu2 | 1.3s | 0.0s | 0.0s | 0.0s | 1.9s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | nccl-async | 25 | 0 | 790 | 38.3 | 1570/9908 | 0 | 0.0 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | nccl-async | 160.5 | 8.2% |

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
| resnet-graph | nccl-async | 193 | 790 | 0 | 6.62e-3 | -5.34e-7 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | nccl-async | 0 | 790 | 8.50e-2 | 4.33e-2 | 0.00e0 | 4.40e-1 | 43.0 | -7.31e-5 | 1.47e-3 |
| resnet-graph | nccl-async | 1 | 790 | 8.67e-2 | 4.65e-2 | 0.00e0 | 5.43e-1 | 40.0 | -8.96e-5 | 2.28e-3 |
| resnet-graph | nccl-async | 2 | 790 | 8.56e-2 | 4.55e-2 | 0.00e0 | 4.91e-1 | 17.0 | -8.65e-5 | 2.15e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | nccl-async | rank0 ↔ rank1 | +0.9832 |
| resnet-graph | nccl-async | rank0 ↔ rank2 | +0.9847 |
| resnet-graph | nccl-async | rank1 ↔ rank2 | +0.9963 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | nccl-async | 98 (0,1,2,3,5,6,7,8…146,147) | 0 (—) | — | 0,1,2,3,5,6,7,8…146,147 | — |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | nccl-async | 0e0 | 3 | 29 | 29 |
| resnet-graph | nccl-async | 0e0 | 5 | 14 | 14 |
| resnet-graph | nccl-async | 0e0 | 10 | 3 | 3 |
| resnet-graph | nccl-async | 1e-4 | 3 | 7 | 7 |
| resnet-graph | nccl-async | 1e-4 | 5 | 2 | 2 |
| resnet-graph | nccl-async | 1e-4 | 10 | 1 | 1 |
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
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 582 | +0.097 |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | 156 | +0.069 |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | 47 | +0.040 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | nccl-async | 787 | +0.015 | 192 | +0.203 | +0.356 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | nccl-async | 788 | 3.41e1–7.81e1 | 6.99e1 | 1.75e-3 | 2.76e-3 | 4.06e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 584 | 83–78059 | +2.526e-6 | 0.115 | +2.540e-6 | 0.127 | 100 | +2.371e-6 | 0.574 | 35–266 | +5.909e-4 | 0.020 |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 572 | 931–78059 | +2.288e-6 | 0.135 | +2.274e-6 | 0.152 | 99 | +2.218e-6 | 0.584 | 70–266 | +4.287e-4 | 0.015 |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | 157 | 78286–116865 | +4.416e-5 | 0.618 | +4.453e-5 | 0.622 | 49 | +4.987e-5 | 0.864 | 109–878 | +1.726e-3 | 0.629 |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | 48 | 117647–155621 | -1.063e-5 | 0.066 | -1.056e-5 | 0.065 | 44 | -1.125e-5 | 0.073 | 639–976 | -4.206e-4 | 0.004 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | +5.909e-4 | r0: +7.797e-4, r1: +4.989e-4, r2: +5.100e-4 | r0: 0.050, r1: 0.012, r2: 0.012 | 1.56× | ✓ |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +4.287e-4 | r0: +6.474e-4, r1: +3.248e-4, r2: +3.277e-4 | r0: 0.061, r1: 0.007, r2: 0.007 | 1.99× | ✓ |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | +1.726e-3 | r0: +1.729e-3, r1: +1.715e-3, r2: +1.736e-3 | r0: 0.646, r1: 0.614, r2: 0.621 | 1.01× | ✓ |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | -4.206e-4 | r0: -3.859e-4, r1: -4.401e-4, r2: -4.344e-4 | r0: 0.003, r1: 0.004, r2: 0.004 | 1.14× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | nccl-async | `██▇█▇▇▇██████████████████▄▃▃▄▄▄▄▅▅▆▆▆▅▁▁▁▁▁▁▁▁▁▁▁` | `▁█▇▇▇▇▇▇▆▇▇▇█▇▇▇▇▇▇▇▇▇▇▇▇▃▇▇█▇█▇█████▇▆▇▇▇▇▇█████` |

### Per-Epoch Detail: resnet-graph / nccl-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 13 | 0.00e0 | 5.43e-1 | 1.09e-1 | 6.86e-2 | 25 | -8.06e-2 | +1.08e-2 | -1.25e-2 | -7.28e-3 |
| 1 | 3.00e-1 | 9 | 7.19e-2 | 1.18e-1 | 8.46e-2 | 8.23e-2 | 29 | -1.85e-2 | +1.83e-2 | +7.36e-4 | -1.90e-3 |
| 2 | 3.00e-1 | 8 | 8.24e-2 | 1.23e-1 | 9.31e-2 | 8.89e-2 | 30 | -1.33e-2 | +1.34e-2 | +1.57e-4 | -7.82e-4 |
| 3 | 3.00e-1 | 9 | 8.45e-2 | 1.32e-1 | 9.91e-2 | 9.95e-2 | 28 | -1.60e-2 | +1.43e-2 | +5.50e-4 | -2.93e-5 |
| 4 | 3.00e-1 | 8 | 9.16e-2 | 1.41e-1 | 1.04e-1 | 9.96e-2 | 30 | -1.25e-2 | +1.43e-2 | +1.68e-4 | -2.20e-5 |
| 5 | 3.00e-1 | 7 | 9.27e-2 | 1.37e-1 | 1.04e-1 | 9.87e-2 | 36 | -1.24e-2 | +1.18e-2 | +6.83e-5 | -5.52e-5 |
| 6 | 3.00e-1 | 7 | 9.94e-2 | 1.46e-1 | 1.10e-1 | 1.06e-1 | 33 | -9.93e-3 | +1.00e-2 | +3.50e-4 | +8.97e-5 |
| 7 | 3.00e-1 | 7 | 9.68e-2 | 1.41e-1 | 1.09e-1 | 1.09e-1 | 35 | -1.01e-2 | +1.08e-2 | +2.53e-4 | +1.40e-4 |
| 8 | 3.00e-1 | 10 | 8.78e-2 | 1.43e-1 | 1.02e-1 | 9.06e-2 | 34 | -1.57e-2 | +1.11e-2 | -4.76e-4 | -3.85e-4 |
| 9 | 3.00e-1 | 4 | 9.37e-2 | 1.38e-1 | 1.07e-1 | 9.93e-2 | 37 | -9.81e-3 | +1.11e-2 | +6.61e-4 | -1.13e-4 |
| 10 | 3.00e-1 | 8 | 9.90e-2 | 1.37e-1 | 1.11e-1 | 1.06e-1 | 42 | -7.87e-3 | +8.05e-3 | +1.51e-4 | -6.05e-5 |
| 11 | 3.00e-1 | 5 | 9.52e-2 | 1.32e-1 | 1.06e-1 | 1.03e-1 | 43 | -9.06e-3 | +8.15e-3 | +5.86e-5 | -3.99e-5 |
| 12 | 3.00e-1 | 6 | 9.25e-2 | 1.45e-1 | 1.05e-1 | 9.96e-2 | 50 | -1.41e-2 | +9.63e-3 | -4.18e-4 | -2.44e-4 |
| 13 | 3.00e-1 | 7 | 9.21e-2 | 1.44e-1 | 1.05e-1 | 9.84e-2 | 35 | -1.01e-2 | +6.66e-3 | -4.10e-4 | -3.37e-4 |
| 14 | 3.00e-1 | 6 | 9.53e-2 | 1.37e-1 | 1.07e-1 | 1.05e-1 | 37 | -9.72e-3 | +9.70e-3 | +3.31e-4 | -8.61e-5 |
| 15 | 3.00e-1 | 7 | 9.64e-2 | 1.38e-1 | 1.04e-1 | 9.64e-2 | 36 | -8.46e-3 | +9.07e-3 | -1.69e-4 | -1.98e-4 |
| 16 | 3.00e-1 | 6 | 9.74e-2 | 1.44e-1 | 1.11e-1 | 1.07e-1 | 42 | -7.97e-3 | +8.86e-3 | +3.05e-4 | -3.61e-5 |
| 17 | 3.00e-1 | 7 | 8.97e-2 | 1.39e-1 | 1.04e-1 | 1.08e-1 | 37 | -1.26e-2 | +7.88e-3 | -1.53e-4 | -7.56e-6 |
| 18 | 3.00e-1 | 5 | 9.78e-2 | 1.42e-1 | 1.13e-1 | 1.04e-1 | 46 | -6.12e-3 | +8.65e-3 | +3.00e-4 | +4.26e-5 |
| 19 | 3.00e-1 | 7 | 9.41e-2 | 1.36e-1 | 1.06e-1 | 9.98e-2 | 34 | -5.68e-3 | +4.58e-3 | -4.60e-4 | -2.14e-4 |
| 20 | 3.00e-1 | 7 | 9.34e-2 | 1.36e-1 | 1.06e-1 | 1.14e-1 | 45 | -1.04e-2 | +9.18e-3 | +4.31e-4 | +1.49e-4 |
| 21 | 3.00e-1 | 5 | 1.01e-1 | 1.39e-1 | 1.12e-1 | 1.01e-1 | 42 | -6.76e-3 | +6.18e-3 | -4.61e-4 | -1.64e-4 |
| 22 | 3.00e-1 | 6 | 1.03e-1 | 1.40e-1 | 1.13e-1 | 1.03e-1 | 45 | -6.07e-3 | +6.12e-3 | +3.41e-5 | -1.70e-4 |
| 23 | 3.00e-1 | 6 | 9.51e-2 | 1.45e-1 | 1.14e-1 | 9.51e-2 | 40 | -4.99e-3 | +5.77e-3 | -5.29e-4 | -4.66e-4 |
| 24 | 3.00e-1 | 6 | 9.23e-2 | 1.36e-1 | 1.05e-1 | 1.02e-1 | 43 | -1.02e-2 | +8.56e-3 | +2.28e-4 | -1.85e-4 |
| 25 | 3.00e-1 | 7 | 1.04e-1 | 1.37e-1 | 1.14e-1 | 1.15e-1 | 51 | -5.88e-3 | +6.01e-3 | +3.38e-4 | +6.01e-5 |
| 26 | 3.00e-1 | 4 | 9.63e-2 | 1.38e-1 | 1.13e-1 | 9.63e-2 | 42 | -7.94e-3 | +4.70e-3 | -1.25e-3 | -4.67e-4 |
| 27 | 3.00e-1 | 7 | 8.83e-2 | 1.44e-1 | 1.04e-1 | 9.68e-2 | 44 | -1.37e-2 | +8.21e-3 | -2.39e-4 | -3.79e-4 |
| 28 | 3.00e-1 | 5 | 1.03e-1 | 1.46e-1 | 1.17e-1 | 1.17e-1 | 45 | -6.74e-3 | +5.83e-3 | +5.56e-4 | -2.53e-5 |
| 29 | 3.00e-1 | 6 | 1.07e-1 | 1.45e-1 | 1.15e-1 | 1.09e-1 | 45 | -6.66e-3 | +6.60e-3 | -1.05e-4 | -9.52e-5 |
| 30 | 3.00e-1 | 7 | 1.03e-1 | 1.54e-1 | 1.14e-1 | 1.15e-1 | 48 | -8.37e-3 | +7.84e-3 | +1.36e-4 | +2.75e-5 |
| 31 | 3.00e-1 | 4 | 1.06e-1 | 1.45e-1 | 1.19e-1 | 1.06e-1 | 45 | -5.96e-3 | +5.43e-3 | -4.38e-4 | -1.93e-4 |
| 32 | 3.00e-1 | 6 | 9.14e-2 | 1.54e-1 | 1.10e-1 | 9.14e-2 | 44 | -9.98e-3 | +7.78e-3 | -9.62e-4 | -6.45e-4 |
| 33 | 3.00e-1 | 6 | 9.62e-2 | 1.45e-1 | 1.08e-1 | 9.62e-2 | 37 | -9.14e-3 | +7.63e-3 | -3.16e-4 | -5.98e-4 |
| 34 | 3.00e-1 | 7 | 9.75e-2 | 1.46e-1 | 1.11e-1 | 1.02e-1 | 43 | -1.01e-2 | +8.86e-3 | +8.76e-6 | -3.95e-4 |
| 35 | 3.00e-1 | 4 | 9.12e-2 | 1.41e-1 | 1.10e-1 | 1.06e-1 | 35 | -9.74e-3 | +8.99e-3 | +9.32e-4 | +3.54e-5 |
| 36 | 3.00e-1 | 7 | 9.77e-2 | 1.45e-1 | 1.09e-1 | 1.05e-1 | 44 | -1.07e-2 | +1.07e-2 | +1.21e-4 | +2.88e-5 |
| 37 | 3.00e-1 | 5 | 1.07e-1 | 1.46e-1 | 1.16e-1 | 1.07e-1 | 53 | -6.30e-3 | +6.89e-3 | +1.40e-4 | +3.91e-6 |
| 38 | 3.00e-1 | 6 | 9.45e-2 | 1.49e-1 | 1.14e-1 | 9.45e-2 | 36 | -8.03e-3 | +5.70e-3 | -7.60e-4 | -4.82e-4 |
| 39 | 3.00e-1 | 5 | 9.59e-2 | 1.46e-1 | 1.10e-1 | 1.08e-1 | 41 | -1.03e-2 | +1.11e-2 | +6.36e-4 | -7.93e-5 |
| 40 | 3.00e-1 | 6 | 1.02e-1 | 1.53e-1 | 1.13e-1 | 1.04e-1 | 45 | -9.91e-3 | +8.46e-3 | -1.46e-4 | -1.65e-4 |
| 41 | 3.00e-1 | 6 | 1.01e-1 | 1.47e-1 | 1.14e-1 | 1.01e-1 | 40 | -6.27e-3 | +5.81e-3 | -3.32e-4 | -3.33e-4 |
| 42 | 3.00e-1 | 5 | 1.05e-1 | 1.42e-1 | 1.14e-1 | 1.09e-1 | 48 | -6.43e-3 | +7.00e-3 | +3.70e-4 | -1.02e-4 |
| 43 | 3.00e-1 | 8 | 1.02e-1 | 1.47e-1 | 1.13e-1 | 1.07e-1 | 42 | -6.36e-3 | +5.49e-3 | -1.37e-4 | -1.47e-4 |
| 44 | 3.00e-1 | 4 | 1.04e-1 | 1.44e-1 | 1.17e-1 | 1.09e-1 | 40 | -5.72e-3 | +7.15e-3 | +1.96e-4 | -8.84e-5 |
| 45 | 3.00e-1 | 6 | 9.83e-2 | 1.38e-1 | 1.06e-1 | 9.84e-2 | 40 | -7.54e-3 | +7.80e-3 | -2.39e-4 | -2.08e-4 |
| 46 | 3.00e-1 | 6 | 9.74e-2 | 1.48e-1 | 1.10e-1 | 1.09e-1 | 42 | -9.89e-3 | +9.69e-3 | +5.05e-4 | +8.58e-5 |
| 47 | 3.00e-1 | 6 | 1.04e-1 | 1.51e-1 | 1.16e-1 | 1.08e-1 | 47 | -8.42e-3 | +7.98e-3 | +3.91e-5 | -1.58e-7 |
| 48 | 3.00e-1 | 5 | 1.06e-1 | 1.46e-1 | 1.17e-1 | 1.17e-1 | 44 | -6.59e-3 | +6.29e-3 | +2.65e-4 | +9.22e-5 |
| 49 | 3.00e-1 | 6 | 1.01e-1 | 1.42e-1 | 1.13e-1 | 1.10e-1 | 50 | -7.66e-3 | +6.19e-3 | -1.11e-4 | -1.06e-5 |
| 50 | 3.00e-1 | 5 | 1.16e-1 | 1.45e-1 | 1.23e-1 | 1.22e-1 | 45 | -4.34e-3 | +4.45e-3 | +3.75e-4 | +1.19e-4 |
| 51 | 3.00e-1 | 6 | 1.01e-1 | 1.42e-1 | 1.11e-1 | 1.02e-1 | 47 | -8.45e-3 | +6.79e-3 | -4.68e-4 | -1.86e-4 |
| 52 | 3.00e-1 | 6 | 1.02e-1 | 1.44e-1 | 1.13e-1 | 1.04e-1 | 41 | -7.55e-3 | +5.67e-3 | -1.65e-4 | -2.34e-4 |
| 53 | 3.00e-1 | 5 | 1.00e-1 | 1.51e-1 | 1.16e-1 | 1.00e-1 | 38 | -5.94e-3 | +6.12e-3 | -8.41e-4 | -5.81e-4 |
| 54 | 3.00e-1 | 7 | 9.45e-2 | 1.36e-1 | 1.09e-1 | 1.09e-1 | 55 | -8.87e-3 | +7.83e-3 | +3.11e-4 | -1.55e-4 |
| 55 | 3.00e-1 | 5 | 1.04e-1 | 1.42e-1 | 1.16e-1 | 1.07e-1 | 43 | -6.94e-3 | +3.80e-3 | -2.72e-4 | -2.42e-4 |
| 56 | 3.00e-1 | 5 | 1.08e-1 | 1.46e-1 | 1.19e-1 | 1.18e-1 | 42 | -5.26e-3 | +6.00e-3 | +3.20e-4 | -3.91e-5 |
| 57 | 3.00e-1 | 5 | 1.01e-1 | 1.54e-1 | 1.15e-1 | 1.05e-1 | 57 | -8.42e-3 | +9.72e-3 | -9.91e-5 | -1.28e-4 |
| 58 | 3.00e-1 | 7 | 1.05e-1 | 1.52e-1 | 1.18e-1 | 1.05e-1 | 45 | -6.40e-3 | +3.71e-3 | -3.20e-4 | -2.94e-4 |
| 59 | 3.00e-1 | 4 | 1.05e-1 | 1.44e-1 | 1.19e-1 | 1.16e-1 | 52 | -5.55e-3 | +6.59e-3 | +5.97e-4 | -2.32e-5 |
| 60 | 3.00e-1 | 5 | 1.09e-1 | 1.44e-1 | 1.19e-1 | 1.09e-1 | 47 | -4.49e-3 | +4.37e-3 | -2.76e-4 | -1.74e-4 |
| 61 | 3.00e-1 | 6 | 1.05e-1 | 1.48e-1 | 1.16e-1 | 1.12e-1 | 47 | -7.44e-3 | +6.88e-3 | +9.16e-5 | -9.24e-5 |
| 62 | 3.00e-1 | 6 | 1.06e-1 | 1.52e-1 | 1.18e-1 | 1.11e-1 | 50 | -7.78e-3 | +5.57e-3 | -9.85e-5 | -1.32e-4 |
| 63 | 3.00e-1 | 4 | 1.08e-1 | 1.57e-1 | 1.23e-1 | 1.08e-1 | 44 | -7.90e-3 | +6.23e-3 | -2.87e-4 | -2.54e-4 |
| 64 | 3.00e-1 | 6 | 1.09e-1 | 1.47e-1 | 1.17e-1 | 1.10e-1 | 45 | -4.88e-3 | +5.99e-3 | -2.37e-5 | -2.06e-4 |
| 65 | 3.00e-1 | 5 | 9.86e-2 | 1.52e-1 | 1.14e-1 | 9.86e-2 | 40 | -7.07e-3 | +7.73e-3 | -5.59e-4 | -4.45e-4 |
| 66 | 3.00e-1 | 7 | 9.87e-2 | 1.37e-1 | 1.14e-1 | 1.15e-1 | 48 | -7.44e-3 | +6.10e-3 | +4.27e-4 | -3.70e-5 |
| 67 | 3.00e-1 | 4 | 1.06e-1 | 1.56e-1 | 1.22e-1 | 1.09e-1 | 49 | -8.36e-3 | +6.47e-3 | -3.25e-4 | -1.95e-4 |
| 68 | 3.00e-1 | 7 | 1.11e-1 | 1.49e-1 | 1.20e-1 | 1.12e-1 | 47 | -5.72e-3 | +5.42e-3 | +5.26e-5 | -1.38e-4 |
| 69 | 3.00e-1 | 3 | 1.07e-1 | 1.54e-1 | 1.26e-1 | 1.07e-1 | 44 | -8.32e-3 | +5.96e-3 | -6.07e-4 | -3.53e-4 |
| 70 | 3.00e-1 | 6 | 1.04e-1 | 1.41e-1 | 1.17e-1 | 1.15e-1 | 50 | -5.32e-3 | +6.56e-3 | +3.47e-4 | -7.44e-5 |
| 71 | 3.00e-1 | 5 | 1.02e-1 | 1.53e-1 | 1.20e-1 | 1.04e-1 | 43 | -4.75e-3 | +5.22e-3 | -7.56e-4 | -4.23e-4 |
| 72 | 3.00e-1 | 5 | 1.01e-1 | 1.50e-1 | 1.17e-1 | 1.11e-1 | 50 | -8.44e-3 | +8.39e-3 | +4.73e-4 | -1.15e-4 |
| 73 | 3.00e-1 | 5 | 1.07e-1 | 1.55e-1 | 1.21e-1 | 1.07e-1 | 46 | -5.89e-3 | +6.07e-3 | -2.37e-4 | -2.48e-4 |
| 74 | 3.00e-1 | 7 | 1.08e-1 | 1.42e-1 | 1.15e-1 | 1.08e-1 | 49 | -4.52e-3 | +5.58e-3 | +9.32e-6 | -1.82e-4 |
| 75 | 3.00e-1 | 3 | 1.13e-1 | 1.50e-1 | 1.29e-1 | 1.23e-1 | 55 | -3.56e-3 | +4.89e-3 | +6.36e-4 | -1.47e-6 |
| 76 | 3.00e-1 | 5 | 1.10e-1 | 1.42e-1 | 1.22e-1 | 1.17e-1 | 50 | -5.13e-3 | +2.21e-3 | -2.21e-4 | -9.43e-5 |
| 77 | 3.00e-1 | 5 | 1.04e-1 | 1.60e-1 | 1.21e-1 | 1.14e-1 | 49 | -7.97e-3 | +7.02e-3 | -2.23e-4 | -1.77e-4 |
| 78 | 3.00e-1 | 7 | 1.04e-1 | 1.45e-1 | 1.14e-1 | 1.12e-1 | 52 | -6.42e-3 | +5.42e-3 | -9.68e-5 | -1.28e-4 |
| 79 | 3.00e-1 | 4 | 1.09e-1 | 1.49e-1 | 1.22e-1 | 1.09e-1 | 41 | -6.55e-3 | +4.76e-3 | -3.79e-4 | -2.83e-4 |
| 80 | 3.00e-1 | 7 | 1.03e-1 | 1.45e-1 | 1.12e-1 | 1.09e-1 | 47 | -7.83e-3 | +8.10e-3 | +1.67e-4 | -9.25e-5 |
| 81 | 3.00e-1 | 4 | 1.06e-1 | 1.45e-1 | 1.19e-1 | 1.15e-1 | 53 | -6.72e-3 | +5.92e-3 | +2.25e-4 | -1.79e-5 |
| 82 | 3.00e-1 | 5 | 1.15e-1 | 1.48e-1 | 1.23e-1 | 1.16e-1 | 47 | -5.10e-3 | +4.20e-3 | -5.83e-5 | -7.25e-5 |
| 83 | 3.00e-1 | 6 | 1.06e-1 | 1.41e-1 | 1.15e-1 | 1.13e-1 | 47 | -6.89e-3 | +5.09e-3 | -1.40e-4 | -1.03e-4 |
| 84 | 3.00e-1 | 6 | 9.93e-2 | 1.38e-1 | 1.13e-1 | 1.10e-1 | 51 | -7.67e-3 | +4.63e-3 | -6.94e-5 | -9.61e-5 |
| 85 | 3.00e-1 | 4 | 1.14e-1 | 1.52e-1 | 1.26e-1 | 1.17e-1 | 46 | -3.91e-3 | +5.41e-3 | +1.94e-4 | -5.78e-5 |
| 86 | 3.00e-1 | 7 | 1.01e-1 | 1.46e-1 | 1.12e-1 | 1.15e-1 | 49 | -9.50e-3 | +8.28e-3 | +6.46e-5 | +2.61e-5 |
| 87 | 3.00e-1 | 4 | 1.11e-1 | 1.56e-1 | 1.23e-1 | 1.13e-1 | 45 | -7.09e-3 | +7.04e-3 | -3.46e-5 | -5.04e-5 |
| 88 | 3.00e-1 | 6 | 1.08e-1 | 1.48e-1 | 1.17e-1 | 1.15e-1 | 50 | -6.91e-3 | +6.13e-3 | +6.52e-5 | -9.09e-6 |
| 89 | 3.00e-1 | 5 | 1.08e-1 | 1.44e-1 | 1.19e-1 | 1.17e-1 | 53 | -5.77e-3 | +4.73e-3 | +1.17e-4 | +3.05e-5 |
| 90 | 3.00e-1 | 5 | 1.08e-1 | 1.56e-1 | 1.23e-1 | 1.14e-1 | 50 | -6.96e-3 | +5.05e-3 | -1.52e-4 | -8.93e-5 |
| 91 | 3.00e-1 | 5 | 1.07e-1 | 1.65e-1 | 1.23e-1 | 1.07e-1 | 41 | -7.95e-3 | +6.51e-3 | -4.50e-4 | -3.16e-4 |
| 92 | 3.00e-1 | 5 | 1.02e-1 | 1.41e-1 | 1.16e-1 | 1.17e-1 | 60 | -6.70e-3 | +6.06e-3 | +3.81e-4 | -4.80e-5 |
| 93 | 3.00e-1 | 5 | 1.10e-1 | 1.51e-1 | 1.24e-1 | 1.10e-1 | 48 | -6.36e-3 | +3.06e-3 | -4.54e-4 | -2.69e-4 |
| 94 | 3.00e-1 | 7 | 1.09e-1 | 1.44e-1 | 1.17e-1 | 1.12e-1 | 48 | -4.37e-3 | +4.83e-3 | -6.36e-6 | -1.77e-4 |
| 95 | 3.00e-1 | 3 | 1.08e-1 | 1.46e-1 | 1.24e-1 | 1.08e-1 | 49 | -6.14e-3 | +4.44e-3 | -3.44e-4 | -2.89e-4 |
| 96 | 3.00e-1 | 6 | 1.11e-1 | 1.45e-1 | 1.19e-1 | 1.19e-1 | 51 | -5.57e-3 | +4.99e-3 | +2.40e-4 | -5.85e-5 |
| 97 | 3.00e-1 | 6 | 1.13e-1 | 1.47e-1 | 1.21e-1 | 1.14e-1 | 48 | -4.90e-3 | +5.22e-3 | -7.85e-5 | -1.06e-4 |
| 98 | 3.00e-1 | 4 | 1.14e-1 | 1.46e-1 | 1.23e-1 | 1.15e-1 | 51 | -4.56e-3 | +4.78e-3 | +4.41e-5 | -9.81e-5 |
| 99 | 3.00e-1 | 5 | 1.09e-1 | 1.54e-1 | 1.19e-1 | 1.11e-1 | 44 | -7.28e-3 | +6.20e-3 | -1.39e-4 | -1.54e-4 |
| 100 | 3.00e-2 | 6 | 1.06e-2 | 1.10e-1 | 4.31e-2 | 1.07e-2 | 49 | -4.78e-2 | +6.18e-4 | -8.13e-3 | -3.63e-3 |
| 101 | 3.00e-2 | 5 | 9.89e-3 | 1.43e-2 | 1.15e-2 | 1.12e-2 | 42 | -8.16e-3 | +5.90e-3 | +2.35e-4 | -2.07e-3 |
| 102 | 3.00e-2 | 6 | 1.06e-2 | 1.60e-2 | 1.22e-2 | 1.20e-2 | 45 | -6.48e-3 | +8.32e-3 | +1.05e-4 | -1.09e-3 |
| 103 | 3.00e-2 | 4 | 1.17e-2 | 1.66e-2 | 1.33e-2 | 1.25e-2 | 45 | -6.02e-3 | +7.16e-3 | +2.79e-4 | -6.71e-4 |
| 104 | 3.00e-2 | 6 | 1.18e-2 | 1.66e-2 | 1.33e-2 | 1.32e-2 | 46 | -7.42e-3 | +8.00e-3 | +2.96e-4 | -2.62e-4 |
| 105 | 3.00e-2 | 5 | 1.30e-2 | 1.67e-2 | 1.39e-2 | 1.30e-2 | 42 | -4.56e-3 | +4.90e-3 | -1.82e-4 | -2.77e-4 |
| 106 | 3.00e-2 | 7 | 1.19e-2 | 1.69e-2 | 1.37e-2 | 1.19e-2 | 41 | -5.63e-3 | +6.88e-3 | -6.50e-5 | -3.11e-4 |
| 107 | 3.00e-2 | 4 | 1.23e-2 | 1.84e-2 | 1.43e-2 | 1.30e-2 | 47 | -6.90e-3 | +9.61e-3 | +5.41e-4 | -1.10e-4 |
| 108 | 3.00e-2 | 5 | 1.32e-2 | 1.79e-2 | 1.48e-2 | 1.42e-2 | 44 | -6.50e-3 | +5.49e-3 | +2.99e-4 | +7.08e-6 |
| 109 | 3.00e-2 | 8 | 1.31e-2 | 1.88e-2 | 1.49e-2 | 1.31e-2 | 50 | -4.54e-3 | +5.94e-3 | -2.07e-4 | -2.30e-4 |
| 110 | 3.00e-2 | 3 | 1.43e-2 | 1.99e-2 | 1.65e-2 | 1.43e-2 | 40 | -8.38e-3 | +6.58e-3 | -1.10e-6 | -2.68e-4 |
| 111 | 3.00e-2 | 7 | 1.42e-2 | 1.97e-2 | 1.58e-2 | 1.54e-2 | 47 | -5.92e-3 | +7.23e-3 | +2.70e-4 | -4.71e-5 |
| 112 | 3.00e-2 | 4 | 1.52e-2 | 1.97e-2 | 1.65e-2 | 1.55e-2 | 46 | -5.58e-3 | +5.10e-3 | +1.04e-5 | -7.06e-5 |
| 113 | 3.00e-2 | 6 | 1.52e-2 | 2.19e-2 | 1.74e-2 | 1.68e-2 | 50 | -5.27e-3 | +7.21e-3 | +2.87e-4 | +3.52e-5 |
| 114 | 3.00e-2 | 4 | 1.63e-2 | 2.30e-2 | 1.82e-2 | 1.71e-2 | 50 | -6.81e-3 | +6.66e-3 | +1.35e-4 | +2.37e-5 |
| 115 | 3.00e-2 | 4 | 1.72e-2 | 2.23e-2 | 1.90e-2 | 1.91e-2 | 52 | -4.20e-3 | +4.86e-3 | +5.89e-4 | +1.99e-4 |
| 116 | 3.00e-2 | 5 | 1.71e-2 | 2.20e-2 | 1.90e-2 | 1.85e-2 | 64 | -4.62e-3 | +3.70e-3 | +4.81e-5 | +1.24e-4 |
| 117 | 3.00e-2 | 5 | 1.64e-2 | 2.32e-2 | 1.84e-2 | 1.64e-2 | 47 | -7.87e-3 | +3.93e-3 | -6.70e-4 | -2.40e-4 |
| 118 | 3.00e-2 | 6 | 1.71e-2 | 2.34e-2 | 1.87e-2 | 1.80e-2 | 41 | -6.77e-3 | +6.62e-3 | +3.05e-4 | -4.51e-5 |
| 119 | 3.00e-2 | 5 | 1.69e-2 | 2.52e-2 | 1.93e-2 | 1.84e-2 | 46 | -7.48e-3 | +8.31e-3 | +2.47e-4 | +2.69e-5 |
| 120 | 3.00e-2 | 6 | 1.63e-2 | 2.65e-2 | 1.88e-2 | 1.75e-2 | 43 | -1.14e-2 | +8.03e-3 | -3.21e-4 | -1.57e-4 |
| 121 | 3.00e-2 | 5 | 1.84e-2 | 2.69e-2 | 2.07e-2 | 2.00e-2 | 46 | -7.31e-3 | +7.80e-3 | +4.45e-4 | +3.57e-5 |
| 122 | 3.00e-2 | 5 | 1.80e-2 | 2.52e-2 | 2.04e-2 | 2.03e-2 | 49 | -4.91e-3 | +6.93e-3 | +2.40e-4 | +1.02e-4 |
| 123 | 3.00e-2 | 5 | 1.86e-2 | 2.55e-2 | 2.09e-2 | 2.02e-2 | 46 | -4.71e-3 | +5.52e-3 | +3.65e-6 | +4.05e-5 |
| 124 | 3.00e-2 | 5 | 1.95e-2 | 2.88e-2 | 2.29e-2 | 1.99e-2 | 44 | -4.11e-3 | +6.38e-3 | -1.64e-4 | -1.55e-4 |
| 125 | 3.00e-2 | 2 | 1.87e-2 | 1.94e-2 | 1.90e-2 | 1.94e-2 | 41 | -1.39e-3 | +8.40e-4 | -2.75e-4 | -1.67e-4 |
| 126 | 3.00e-2 | 1 | 1.91e-2 | 1.91e-2 | 1.91e-2 | 1.91e-2 | 299 | -3.84e-5 | -3.84e-5 | -3.84e-5 | -1.54e-4 |
| 127 | 3.00e-2 | 1 | 4.97e-2 | 4.97e-2 | 4.97e-2 | 4.97e-2 | 252 | +3.78e-3 | +3.78e-3 | +3.78e-3 | +2.40e-4 |
| 128 | 3.00e-2 | 1 | 4.65e-2 | 4.65e-2 | 4.65e-2 | 4.65e-2 | 282 | -2.30e-4 | -2.30e-4 | -2.30e-4 | +1.93e-4 |
| 129 | 3.00e-2 | 2 | 4.88e-2 | 4.94e-2 | 4.91e-2 | 4.94e-2 | 251 | +4.81e-5 | +1.72e-4 | +1.10e-4 | +1.76e-4 |
| 131 | 3.00e-2 | 1 | 4.75e-2 | 4.75e-2 | 4.75e-2 | 4.75e-2 | 328 | -1.25e-4 | -1.25e-4 | -1.25e-4 | +1.46e-4 |
| 132 | 3.00e-2 | 1 | 5.33e-2 | 5.33e-2 | 5.33e-2 | 5.33e-2 | 291 | +3.96e-4 | +3.96e-4 | +3.96e-4 | +1.71e-4 |
| 133 | 3.00e-2 | 1 | 5.12e-2 | 5.12e-2 | 5.12e-2 | 5.12e-2 | 279 | -1.42e-4 | -1.42e-4 | -1.42e-4 | +1.40e-4 |
| 134 | 3.00e-2 | 1 | 5.02e-2 | 5.02e-2 | 5.02e-2 | 5.02e-2 | 294 | -6.38e-5 | -6.38e-5 | -6.38e-5 | +1.20e-4 |
| 135 | 3.00e-2 | 1 | 5.26e-2 | 5.26e-2 | 5.26e-2 | 5.26e-2 | 318 | +1.42e-4 | +1.42e-4 | +1.42e-4 | +1.22e-4 |
| 136 | 3.00e-2 | 1 | 5.44e-2 | 5.44e-2 | 5.44e-2 | 5.44e-2 | 257 | +1.35e-4 | +1.35e-4 | +1.35e-4 | +1.23e-4 |
| 137 | 3.00e-2 | 1 | 5.12e-2 | 5.12e-2 | 5.12e-2 | 5.12e-2 | 290 | -2.09e-4 | -2.09e-4 | -2.09e-4 | +8.99e-5 |
| 138 | 3.00e-2 | 1 | 5.57e-2 | 5.57e-2 | 5.57e-2 | 5.57e-2 | 275 | +3.09e-4 | +3.09e-4 | +3.09e-4 | +1.12e-4 |
| 139 | 3.00e-2 | 1 | 5.42e-2 | 5.42e-2 | 5.42e-2 | 5.42e-2 | 278 | -9.96e-5 | -9.96e-5 | -9.96e-5 | +9.06e-5 |
| 140 | 3.00e-2 | 1 | 5.48e-2 | 5.48e-2 | 5.48e-2 | 5.48e-2 | 237 | +4.24e-5 | +4.24e-5 | +4.24e-5 | +8.58e-5 |
| 141 | 3.00e-2 | 1 | 5.20e-2 | 5.20e-2 | 5.20e-2 | 5.20e-2 | 263 | -1.95e-4 | -1.95e-4 | -1.95e-4 | +5.77e-5 |
| 142 | 3.00e-2 | 1 | 5.50e-2 | 5.50e-2 | 5.50e-2 | 5.50e-2 | 250 | +2.23e-4 | +2.23e-4 | +2.23e-4 | +7.42e-5 |
| 143 | 3.00e-2 | 1 | 5.44e-2 | 5.44e-2 | 5.44e-2 | 5.44e-2 | 266 | -4.46e-5 | -4.46e-5 | -4.46e-5 | +6.23e-5 |
| 144 | 3.00e-2 | 1 | 5.63e-2 | 5.63e-2 | 5.63e-2 | 5.63e-2 | 285 | +1.24e-4 | +1.24e-4 | +1.24e-4 | +6.85e-5 |
| 145 | 3.00e-2 | 1 | 5.78e-2 | 5.78e-2 | 5.78e-2 | 5.78e-2 | 300 | +8.76e-5 | +8.76e-5 | +8.76e-5 | +7.04e-5 |
| 146 | 3.00e-2 | 1 | 6.01e-2 | 6.01e-2 | 6.01e-2 | 6.01e-2 | 325 | +1.16e-4 | +1.16e-4 | +1.16e-4 | +7.50e-5 |
| 147 | 3.00e-2 | 1 | 6.20e-2 | 6.20e-2 | 6.20e-2 | 6.20e-2 | 287 | +1.13e-4 | +1.13e-4 | +1.13e-4 | +7.88e-5 |
| 148 | 3.00e-2 | 1 | 5.98e-2 | 5.98e-2 | 5.98e-2 | 5.98e-2 | 306 | -1.20e-4 | -1.20e-4 | -1.20e-4 | +5.90e-5 |
| 149 | 3.00e-2 | 1 | 6.22e-2 | 6.22e-2 | 6.22e-2 | 6.22e-2 | 274 | +1.42e-4 | +1.42e-4 | +1.42e-4 | +6.73e-5 |
| 150 | 3.00e-3 | 1 | 6.07e-2 | 6.07e-2 | 6.07e-2 | 6.07e-2 | 270 | -9.11e-5 | -9.11e-5 | -9.11e-5 | +5.15e-5 |
| 151 | 3.00e-3 | 1 | 6.18e-2 | 6.18e-2 | 6.18e-2 | 6.18e-2 | 275 | +6.67e-5 | +6.67e-5 | +6.67e-5 | +5.30e-5 |
| 152 | 3.00e-3 | 1 | 6.02e-3 | 6.02e-3 | 6.02e-3 | 6.02e-3 | 295 | -7.90e-3 | -7.90e-3 | -7.90e-3 | -7.42e-4 |
| 153 | 3.00e-3 | 1 | 5.83e-3 | 5.83e-3 | 5.83e-3 | 5.83e-3 | 285 | -1.09e-4 | -1.09e-4 | -1.09e-4 | -6.79e-4 |
| 154 | 3.00e-3 | 1 | 5.80e-3 | 5.80e-3 | 5.80e-3 | 5.80e-3 | 290 | -2.11e-5 | -2.11e-5 | -2.11e-5 | -6.13e-4 |
| 155 | 3.00e-3 | 1 | 5.95e-3 | 5.95e-3 | 5.95e-3 | 5.95e-3 | 345 | +7.49e-5 | +7.49e-5 | +7.49e-5 | -5.44e-4 |
| 157 | 3.00e-3 | 1 | 6.28e-3 | 6.28e-3 | 6.28e-3 | 6.28e-3 | 352 | +1.55e-4 | +1.55e-4 | +1.55e-4 | -4.74e-4 |
| 158 | 3.00e-3 | 1 | 6.33e-3 | 6.33e-3 | 6.33e-3 | 6.33e-3 | 290 | +2.37e-5 | +2.37e-5 | +2.37e-5 | -4.24e-4 |
| 159 | 3.00e-3 | 1 | 5.82e-3 | 5.82e-3 | 5.82e-3 | 5.82e-3 | 300 | -2.79e-4 | -2.79e-4 | -2.79e-4 | -4.10e-4 |
| 160 | 3.00e-3 | 1 | 6.01e-3 | 6.01e-3 | 6.01e-3 | 6.01e-3 | 270 | +1.21e-4 | +1.21e-4 | +1.21e-4 | -3.57e-4 |
| 161 | 3.00e-3 | 1 | 6.12e-3 | 6.12e-3 | 6.12e-3 | 6.12e-3 | 280 | +6.52e-5 | +6.52e-5 | +6.52e-5 | -3.15e-4 |
| 162 | 3.00e-3 | 1 | 5.77e-3 | 5.77e-3 | 5.77e-3 | 5.77e-3 | 300 | -1.99e-4 | -1.99e-4 | -1.99e-4 | -3.03e-4 |
| 163 | 3.00e-3 | 1 | 5.94e-3 | 5.94e-3 | 5.94e-3 | 5.94e-3 | 270 | +1.08e-4 | +1.08e-4 | +1.08e-4 | -2.62e-4 |
| 164 | 3.00e-3 | 2 | 5.85e-3 | 5.87e-3 | 5.86e-3 | 5.87e-3 | 229 | -5.82e-5 | +1.83e-5 | -1.99e-5 | -2.16e-4 |
| 166 | 3.00e-3 | 1 | 5.63e-3 | 5.63e-3 | 5.63e-3 | 5.63e-3 | 312 | -1.33e-4 | -1.33e-4 | -1.33e-4 | -2.07e-4 |
| 167 | 3.00e-3 | 2 | 6.02e-3 | 6.37e-3 | 6.20e-3 | 6.02e-3 | 241 | -2.34e-4 | +4.60e-4 | +1.13e-4 | -1.50e-4 |
| 169 | 3.00e-3 | 2 | 5.72e-3 | 6.56e-3 | 6.14e-3 | 6.56e-3 | 241 | -1.58e-4 | +5.73e-4 | +2.07e-4 | -7.85e-5 |
| 170 | 3.00e-3 | 1 | 5.65e-3 | 5.65e-3 | 5.65e-3 | 5.65e-3 | 260 | -5.79e-4 | -5.79e-4 | -5.79e-4 | -1.29e-4 |
| 172 | 3.00e-3 | 1 | 6.14e-3 | 6.14e-3 | 6.14e-3 | 6.14e-3 | 362 | +2.33e-4 | +2.33e-4 | +2.33e-4 | -9.24e-5 |
| 173 | 3.00e-3 | 1 | 6.90e-3 | 6.90e-3 | 6.90e-3 | 6.90e-3 | 280 | +4.16e-4 | +4.16e-4 | +4.16e-4 | -4.16e-5 |
| 174 | 3.00e-3 | 2 | 6.26e-3 | 6.58e-3 | 6.42e-3 | 6.58e-3 | 257 | -3.29e-4 | +2.00e-4 | -6.46e-5 | -4.33e-5 |
| 176 | 3.00e-3 | 1 | 6.11e-3 | 6.11e-3 | 6.11e-3 | 6.11e-3 | 328 | -2.27e-4 | -2.27e-4 | -2.27e-4 | -6.17e-5 |
| 177 | 3.00e-3 | 1 | 6.58e-3 | 6.58e-3 | 6.58e-3 | 6.58e-3 | 254 | +2.92e-4 | +2.92e-4 | +2.92e-4 | -2.63e-5 |
| 178 | 3.00e-3 | 1 | 6.09e-3 | 6.09e-3 | 6.09e-3 | 6.09e-3 | 272 | -2.84e-4 | -2.84e-4 | -2.84e-4 | -5.21e-5 |
| 179 | 3.00e-3 | 1 | 6.17e-3 | 6.17e-3 | 6.17e-3 | 6.17e-3 | 316 | +4.20e-5 | +4.20e-5 | +4.20e-5 | -4.27e-5 |
| 180 | 3.00e-3 | 1 | 6.64e-3 | 6.64e-3 | 6.64e-3 | 6.64e-3 | 310 | +2.32e-4 | +2.32e-4 | +2.32e-4 | -1.52e-5 |
| 181 | 3.00e-3 | 1 | 6.59e-3 | 6.59e-3 | 6.59e-3 | 6.59e-3 | 304 | -2.42e-5 | -2.42e-5 | -2.42e-5 | -1.61e-5 |
| 182 | 3.00e-3 | 1 | 6.73e-3 | 6.73e-3 | 6.73e-3 | 6.73e-3 | 266 | +8.03e-5 | +8.03e-5 | +8.03e-5 | -6.45e-6 |
| 183 | 3.00e-3 | 1 | 6.31e-3 | 6.31e-3 | 6.31e-3 | 6.31e-3 | 336 | -1.92e-4 | -1.92e-4 | -1.92e-4 | -2.51e-5 |
| 184 | 3.00e-3 | 1 | 6.82e-3 | 6.82e-3 | 6.82e-3 | 6.82e-3 | 265 | +2.92e-4 | +2.92e-4 | +2.92e-4 | +6.66e-6 |
| 185 | 3.00e-3 | 1 | 6.36e-3 | 6.36e-3 | 6.36e-3 | 6.36e-3 | 253 | -2.72e-4 | -2.72e-4 | -2.72e-4 | -2.12e-5 |
| 186 | 3.00e-3 | 1 | 6.39e-3 | 6.39e-3 | 6.39e-3 | 6.39e-3 | 281 | +1.61e-5 | +1.61e-5 | +1.61e-5 | -1.74e-5 |
| 187 | 3.00e-3 | 1 | 6.72e-3 | 6.72e-3 | 6.72e-3 | 6.72e-3 | 301 | +1.67e-4 | +1.67e-4 | +1.67e-4 | +1.03e-6 |
| 188 | 3.00e-3 | 1 | 6.73e-3 | 6.73e-3 | 6.73e-3 | 6.73e-3 | 259 | +4.04e-6 | +4.04e-6 | +4.04e-6 | +1.33e-6 |
| 189 | 3.00e-3 | 1 | 6.26e-3 | 6.26e-3 | 6.26e-3 | 6.26e-3 | 274 | -2.66e-4 | -2.66e-4 | -2.66e-4 | -2.54e-5 |
| 190 | 3.00e-3 | 1 | 6.42e-3 | 6.42e-3 | 6.42e-3 | 6.42e-3 | 284 | +9.18e-5 | +9.18e-5 | +9.18e-5 | -1.37e-5 |
| 191 | 3.00e-3 | 1 | 6.67e-3 | 6.67e-3 | 6.67e-3 | 6.67e-3 | 283 | +1.36e-4 | +1.36e-4 | +1.36e-4 | +1.32e-6 |
| 192 | 3.00e-3 | 1 | 6.65e-3 | 6.65e-3 | 6.65e-3 | 6.65e-3 | 307 | -1.27e-5 | -1.27e-5 | -1.27e-5 | -8.10e-8 |
| 193 | 3.00e-3 | 1 | 6.92e-3 | 6.92e-3 | 6.92e-3 | 6.92e-3 | 276 | +1.43e-4 | +1.43e-4 | +1.43e-4 | +1.43e-5 |
| 194 | 3.00e-3 | 1 | 6.57e-3 | 6.57e-3 | 6.57e-3 | 6.57e-3 | 291 | -1.77e-4 | -1.77e-4 | -1.77e-4 | -4.90e-6 |
| 195 | 3.00e-3 | 1 | 6.64e-3 | 6.64e-3 | 6.64e-3 | 6.64e-3 | 284 | +3.57e-5 | +3.57e-5 | +3.57e-5 | -8.36e-7 |
| 196 | 3.00e-3 | 1 | 6.83e-3 | 6.83e-3 | 6.83e-3 | 6.83e-3 | 280 | +1.01e-4 | +1.01e-4 | +1.01e-4 | +9.36e-6 |
| 197 | 3.00e-3 | 1 | 6.88e-3 | 6.88e-3 | 6.88e-3 | 6.88e-3 | 275 | +2.95e-5 | +2.95e-5 | +2.95e-5 | +1.14e-5 |
| 199 | 3.00e-3 | 1 | 6.62e-3 | 6.62e-3 | 6.62e-3 | 6.62e-3 | 356 | -1.08e-4 | -1.08e-4 | -1.08e-4 | -5.34e-7 |

