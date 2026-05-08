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

GPU0/GPU1 = compute utilization % (not load). Idle = total time with <5% utilization.

### resnet-graph

> Published: CIFAR-10 91.25% acc, Graph builder ([He et al. 2015](https://arxiv.org/abs/1512.03385), Table 6)

| Mode | Loss | Eval | vs Ref | Total (s) | Syncs | Avg Sync (ms) | GPU0 | GPU1 | Idle (s) |
|------|------|------|--------|-----------|-------|--------------|------|------|----------|
| cpu-async | 0.055338 | 0.9183 | +0.0058 | 1943.9 | 745 | 83.4 | 100% | 100% | 8.5 |

## Best Mode per Model

| Model | Best Eval | Mode | Fastest (within 2% of solo-0) | Mode |
|-------|-----------|------|-------------------------------|------|
| resnet-graph | 0.9183 | cpu-async | - | - |

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
| cpu-async | 2.0347 | 0.7541 | 0.6319 | 0.5616 | 0.5396 | 0.5196 | 0.5072 | 0.4903 | 0.4779 | 0.4710 | 0.2096 | 0.1753 | 0.1558 | 0.1348 | 0.1320 | 0.0742 | 0.0677 | 0.0629 | 0.0588 | 0.0553 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | cpu-async | 0 | cuda:0 | 0.3993 | 2.6 |
| resnet-graph | cpu-async | 1 | cuda:1 | 0.3046 | 3.3 |
| resnet-graph | cpu-async | 2 | cuda:2 | 0.2961 | 3.4 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|
| resnet-graph | cpu-async | 358 | 357 | 396 | 392 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| cpu-async | gpu2 | 356.2 | 1.1 | epoch-boundary(36) |
| cpu-async | gpu1 | 681.6 | 0.9 | epoch-boundary(70) |
| cpu-async | gpu1 | 356.3 | 0.8 | epoch-boundary(36) |
| cpu-async | gpu1 | 385.6 | 0.7 | epoch-boundary(39) |
| cpu-async | gpu2 | 758.5 | 0.7 | epoch-boundary(78) |
| cpu-async | gpu2 | 681.7 | 0.6 | epoch-boundary(70) |
| cpu-async | gpu1 | 758.3 | 0.6 | epoch-boundary(78) |
| cpu-async | gpu1 | 1943.3 | 0.6 | epoch-boundary(199) |
| cpu-async | gpu2 | 1943.3 | 0.6 | epoch-boundary(199) |
| cpu-async | gpu2 | 385.5 | 0.6 | epoch-boundary(39) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | cpu-async | gpu1 | 3.6s | 0.0s | 0.0s | 0.0s | 4.2s |
| resnet-graph | cpu-async | gpu2 | 3.6s | 0.0s | 0.0s | 0.0s | 4.2s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | cpu-async | 317 | 0 | 745 | 83.4 | 1419/8584 | 745 | 83.4 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | cpu-async | 184.2 | 9.5% |

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
| resnet-graph | cpu-async | 196 | 745 | 0 | 3.03e-3 | -8.46e-4 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | cpu-async | 0 | 745 | 5.64e-2 | 7.14e-2 | 2.21e-3 | 3.51e-1 | 29.3 | -1.85e-4 | 1.08e-3 |
| resnet-graph | cpu-async | 1 | 745 | 5.67e-2 | 7.34e-2 | 2.21e-3 | 5.03e-1 | 33.3 | -2.14e-4 | 1.44e-3 |
| resnet-graph | cpu-async | 2 | 745 | 5.68e-2 | 7.23e-2 | 2.33e-3 | 3.51e-1 | 37.4 | -2.19e-4 | 1.37e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | cpu-async | rank0 ↔ rank1 | +0.9920 |
| resnet-graph | cpu-async | rank0 ↔ rank2 | +0.9951 |
| resnet-graph | cpu-async | rank1 ↔ rank2 | +0.9903 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | cpu-async | 48 (0,2,3,4,5,6,7,8…149,150) | 0 (—) | — | 0,2,3,4,5,6,7,8…149,150 | — |

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
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 230 | +0.168 |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | 191 | +0.131 |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | 320 | +0.047 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | cpu-async | 743 | +0.049 | 195 | +0.011 | +0.120 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | cpu-async | 744 | 3.33e1–7.93e1 | 6.24e1 | 1.63e-3 | 3.40e-3 | 4.55e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 232 | 30–77655 | +8.735e-6 | 0.418 | +9.161e-6 | 0.478 | 96 | +6.386e-6 | 0.357 | 30–930 | +9.354e-4 | 0.733 |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 221 | 891–77655 | +9.915e-6 | 0.604 | +1.014e-5 | 0.624 | 95 | +6.584e-6 | 0.368 | 79–930 | +1.021e-3 | 0.937 |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | 192 | 78099–117084 | -8.568e-7 | 0.002 | -8.602e-7 | 0.002 | 50 | -2.579e-7 | 0.000 | 93–559 | +3.999e-4 | 0.055 |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | 321 | 117289–156171 | +1.048e-6 | 0.002 | +1.127e-6 | 0.003 | 50 | +1.504e-6 | 0.013 | 78–238 | +3.521e-3 | 0.294 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | +9.354e-4 | r0: +9.219e-4, r1: +9.484e-4, r2: +9.421e-4 | r0: 0.751, r1: 0.699, r2: 0.726 | 1.03× | ✓ |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +1.021e-3 | r0: +1.005e-3, r1: +1.034e-3, r2: +1.027e-3 | r0: 0.929, r1: 0.930, r2: 0.923 | 1.03× | ✓ |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | +3.999e-4 | r0: +3.564e-4, r1: +3.717e-4, r2: +4.692e-4 | r0: 0.044, r1: 0.045, r2: 0.072 | 1.32× | ✓ |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | +3.521e-3 | r0: +3.118e-3, r1: +3.722e-3, r2: +3.688e-3 | r0: 0.236, r1: 0.314, r2: 0.307 | 1.19× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | cpu-async | `█▇▇▇▇███████████████████▅▄▄▅▅▅▅▅▅▅▅▅▄▁▁▁▁▁▁▁▁▁▁▁▁` | `▁▆▇▆▇▇██████████████████▆▇▇███▇▇▇▇▇▇▃▄▆▆▇▆▆▆▆▆▆▆▆` |

### Per-Epoch Detail: resnet-graph / cpu-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 11 | 1.25e-1 | 5.03e-1 | 2.35e-1 | 1.43e-1 | 40 | -4.84e-2 | +3.28e-2 | -6.61e-3 | -5.80e-3 |
| 1 | 3.00e-1 | 8 | 8.77e-2 | 1.53e-1 | 1.08e-1 | 9.45e-2 | 31 | -7.57e-3 | +2.41e-3 | -1.82e-3 | -2.94e-3 |
| 2 | 3.00e-1 | 8 | 1.01e-1 | 1.37e-1 | 1.12e-1 | 1.15e-1 | 38 | -1.06e-2 | +5.31e-3 | -1.88e-4 | -1.18e-3 |
| 3 | 3.00e-1 | 9 | 1.13e-1 | 1.51e-1 | 1.19e-1 | 1.15e-1 | 38 | -6.85e-3 | +3.51e-3 | -4.53e-4 | -6.48e-4 |
| 4 | 3.00e-1 | 4 | 1.16e-1 | 1.52e-1 | 1.26e-1 | 1.16e-1 | 38 | -6.71e-3 | +3.64e-3 | -9.49e-4 | -7.72e-4 |
| 5 | 3.00e-1 | 8 | 1.17e-1 | 1.50e-1 | 1.24e-1 | 1.21e-1 | 43 | -5.42e-3 | +3.65e-3 | -2.24e-4 | -4.33e-4 |
| 6 | 3.00e-1 | 5 | 1.12e-1 | 1.55e-1 | 1.24e-1 | 1.13e-1 | 40 | -5.75e-3 | +2.91e-3 | -1.10e-3 | -7.08e-4 |
| 7 | 3.00e-1 | 7 | 1.15e-1 | 1.49e-1 | 1.24e-1 | 1.25e-1 | 49 | -5.53e-3 | +3.55e-3 | -1.99e-4 | -3.99e-4 |
| 8 | 3.00e-1 | 5 | 1.18e-1 | 1.57e-1 | 1.30e-1 | 1.23e-1 | 45 | -4.67e-3 | +2.57e-3 | -5.55e-4 | -4.56e-4 |
| 9 | 3.00e-1 | 6 | 1.10e-1 | 1.61e-1 | 1.24e-1 | 1.24e-1 | 43 | -6.86e-3 | +2.96e-3 | -6.14e-4 | -4.48e-4 |
| 10 | 3.00e-1 | 6 | 1.06e-1 | 1.53e-1 | 1.18e-1 | 1.14e-1 | 39 | -7.29e-3 | +2.47e-3 | -9.16e-4 | -5.78e-4 |
| 11 | 3.00e-1 | 10 | 1.05e-1 | 1.49e-1 | 1.15e-1 | 1.05e-1 | 32 | -6.46e-3 | +3.31e-3 | -6.54e-4 | -5.96e-4 |
| 12 | 3.00e-1 | 4 | 1.12e-1 | 1.46e-1 | 1.22e-1 | 1.16e-1 | 40 | -7.48e-3 | +4.32e-3 | -5.98e-4 | -6.03e-4 |
| 13 | 3.00e-1 | 6 | 1.09e-1 | 1.62e-1 | 1.24e-1 | 1.09e-1 | 34 | -6.80e-3 | +3.85e-3 | -1.10e-3 | -8.60e-4 |
| 14 | 3.00e-1 | 9 | 9.78e-2 | 1.48e-1 | 1.10e-1 | 1.12e-1 | 40 | -9.09e-3 | +4.24e-3 | -5.98e-4 | -5.14e-4 |
| 15 | 3.00e-1 | 5 | 1.15e-1 | 1.53e-1 | 1.26e-1 | 1.16e-1 | 40 | -5.22e-3 | +3.54e-3 | -5.64e-4 | -5.55e-4 |
| 16 | 3.00e-1 | 6 | 1.12e-1 | 1.53e-1 | 1.21e-1 | 1.14e-1 | 40 | -5.55e-3 | +3.40e-3 | -6.51e-4 | -5.75e-4 |
| 17 | 3.00e-1 | 7 | 1.11e-1 | 1.43e-1 | 1.18e-1 | 1.16e-1 | 40 | -4.94e-3 | +3.04e-3 | -3.07e-4 | -3.94e-4 |
| 18 | 3.00e-1 | 7 | 1.06e-1 | 1.55e-1 | 1.18e-1 | 1.06e-1 | 34 | -7.53e-3 | +3.30e-3 | -1.02e-3 | -6.95e-4 |
| 19 | 3.00e-1 | 9 | 9.67e-2 | 1.46e-1 | 1.11e-1 | 1.05e-1 | 31 | -4.91e-3 | +4.63e-3 | -4.45e-4 | -5.04e-4 |
| 20 | 3.00e-1 | 1 | 1.03e-1 | 1.03e-1 | 1.03e-1 | 1.03e-1 | 30 | -6.07e-4 | -6.07e-4 | -6.07e-4 | -5.14e-4 |
| 21 | 3.00e-1 | 1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 314 | +2.47e-3 | +2.47e-3 | +2.47e-3 | -2.16e-4 |
| 22 | 3.00e-1 | 1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 276 | -2.65e-5 | -2.65e-5 | -2.65e-5 | -1.97e-4 |
| 23 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 255 | -1.17e-4 | -1.17e-4 | -1.17e-4 | -1.89e-4 |
| 24 | 3.00e-1 | 1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 297 | +9.54e-5 | +9.54e-5 | +9.54e-5 | -1.61e-4 |
| 25 | 3.00e-1 | 1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 331 | +1.04e-4 | +1.04e-4 | +1.04e-4 | -1.34e-4 |
| 26 | 3.00e-1 | 1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 305 | -9.23e-5 | -9.23e-5 | -9.23e-5 | -1.30e-4 |
| 27 | 3.00e-1 | 1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 297 | -2.13e-6 | -2.13e-6 | -2.13e-6 | -1.17e-4 |
| 28 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 302 | -3.43e-5 | -3.43e-5 | -3.43e-5 | -1.09e-4 |
| 29 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 254 | -2.11e-4 | -2.11e-4 | -2.11e-4 | -1.19e-4 |
| 30 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 297 | +1.79e-4 | +1.79e-4 | +1.79e-4 | -8.93e-5 |
| 31 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 256 | -1.91e-4 | -1.91e-4 | -1.91e-4 | -9.95e-5 |
| 32 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 295 | +1.43e-4 | +1.43e-4 | +1.43e-4 | -7.52e-5 |
| 33 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 310 | -7.96e-6 | -7.96e-6 | -7.96e-6 | -6.85e-5 |
| 34 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 280 | -3.61e-5 | -3.61e-5 | -3.61e-5 | -6.52e-5 |
| 35 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 274 | -1.24e-5 | -1.24e-5 | -1.24e-5 | -5.99e-5 |
| 36 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 273 | +7.53e-5 | +7.53e-5 | +7.53e-5 | -4.64e-5 |
| 37 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 304 | -5.37e-6 | -5.37e-6 | -5.37e-6 | -4.23e-5 |
| 38 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 273 | -1.16e-5 | -1.16e-5 | -1.16e-5 | -3.92e-5 |
| 39 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 273 | -6.13e-5 | -6.13e-5 | -6.13e-5 | -4.14e-5 |
| 41 | 3.00e-1 | 2 | 2.19e-1 | 2.27e-1 | 2.23e-1 | 2.19e-1 | 288 | -1.33e-4 | +1.69e-4 | +1.77e-5 | -3.17e-5 |
| 43 | 3.00e-1 | 2 | 2.16e-1 | 2.32e-1 | 2.24e-1 | 2.16e-1 | 273 | -2.60e-4 | +1.79e-4 | -4.03e-5 | -3.55e-5 |
| 45 | 3.00e-1 | 1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 336 | +2.11e-4 | +2.11e-4 | +2.11e-4 | -1.08e-5 |
| 46 | 3.00e-1 | 1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 275 | -1.75e-4 | -1.75e-4 | -1.75e-4 | -2.73e-5 |
| 47 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 232 | -3.05e-4 | -3.05e-4 | -3.05e-4 | -5.51e-5 |
| 48 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 231 | -6.21e-6 | -6.21e-6 | -6.21e-6 | -5.02e-5 |
| 49 | 3.00e-1 | 2 | 2.05e-1 | 2.10e-1 | 2.08e-1 | 2.10e-1 | 250 | -1.36e-5 | +9.66e-5 | +4.15e-5 | -3.22e-5 |
| 50 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 260 | +4.61e-5 | +4.61e-5 | +4.61e-5 | -2.44e-5 |
| 51 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 250 | +4.24e-6 | +4.24e-6 | +4.24e-6 | -2.15e-5 |
| 52 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 261 | -3.63e-6 | -3.63e-6 | -3.63e-6 | -1.97e-5 |
| 53 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 267 | +1.03e-5 | +1.03e-5 | +1.03e-5 | -1.67e-5 |
| 54 | 3.00e-1 | 1 | 2.28e-1 | 2.28e-1 | 2.28e-1 | 2.28e-1 | 313 | +2.16e-4 | +2.16e-4 | +2.16e-4 | +6.51e-6 |
| 55 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 256 | -2.28e-4 | -2.28e-4 | -2.28e-4 | -1.69e-5 |
| 56 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 249 | -7.16e-5 | -7.16e-5 | -7.16e-5 | -2.24e-5 |
| 57 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 281 | +1.32e-4 | +1.32e-4 | +1.32e-4 | -6.94e-6 |
| 58 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 269 | -3.28e-5 | -3.28e-5 | -3.28e-5 | -9.53e-6 |
| 59 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 261 | -2.74e-5 | -2.74e-5 | -2.74e-5 | -1.13e-5 |
| 60 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 266 | -3.63e-5 | -3.63e-5 | -3.63e-5 | -1.38e-5 |
| 61 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 227 | -9.75e-5 | -9.75e-5 | -9.75e-5 | -2.22e-5 |
| 62 | 3.00e-1 | 2 | 2.10e-1 | 2.21e-1 | 2.15e-1 | 2.10e-1 | 242 | -2.19e-4 | +2.00e-4 | -9.51e-6 | -2.19e-5 |
| 64 | 3.00e-1 | 2 | 2.04e-1 | 2.33e-1 | 2.19e-1 | 2.04e-1 | 212 | -6.19e-4 | +3.44e-4 | -1.37e-4 | -4.86e-5 |
| 65 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 224 | +3.60e-5 | +3.60e-5 | +3.60e-5 | -4.01e-5 |
| 66 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 227 | +5.62e-5 | +5.62e-5 | +5.62e-5 | -3.05e-5 |
| 67 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 224 | -9.98e-5 | -9.98e-5 | -9.98e-5 | -3.74e-5 |
| 68 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 240 | +1.07e-4 | +1.07e-4 | +1.07e-4 | -2.30e-5 |
| 69 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 239 | -4.34e-5 | -4.34e-5 | -4.34e-5 | -2.50e-5 |
| 70 | 3.00e-1 | 2 | 2.04e-1 | 2.10e-1 | 2.07e-1 | 2.04e-1 | 212 | -1.37e-4 | +6.30e-5 | -3.72e-5 | -2.83e-5 |
| 71 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 246 | +1.65e-4 | +1.65e-4 | +1.65e-4 | -8.97e-6 |
| 72 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 219 | -1.48e-4 | -1.48e-4 | -1.48e-4 | -2.29e-5 |
| 73 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 232 | +4.25e-5 | +4.25e-5 | +4.25e-5 | -1.63e-5 |
| 74 | 3.00e-1 | 2 | 1.99e-1 | 2.11e-1 | 2.05e-1 | 1.99e-1 | 199 | -2.73e-4 | +5.42e-5 | -1.10e-4 | -3.57e-5 |
| 75 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 231 | +2.85e-4 | +2.85e-4 | +2.85e-4 | -3.66e-6 |
| 76 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 228 | -5.39e-5 | -5.39e-5 | -5.39e-5 | -8.68e-6 |
| 77 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 221 | -9.69e-5 | -9.69e-5 | -9.69e-5 | -1.75e-5 |
| 78 | 3.00e-1 | 2 | 1.96e-1 | 2.05e-1 | 2.01e-1 | 1.96e-1 | 189 | -2.28e-4 | -2.14e-5 | -1.25e-4 | -3.89e-5 |
| 79 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 220 | +2.49e-4 | +2.49e-4 | +2.49e-4 | -1.01e-5 |
| 80 | 3.00e-1 | 2 | 2.00e-1 | 2.06e-1 | 2.03e-1 | 2.00e-1 | 189 | -1.76e-4 | -1.99e-5 | -9.77e-5 | -2.75e-5 |
| 81 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 215 | +2.26e-4 | +2.26e-4 | +2.26e-4 | -2.14e-6 |
| 82 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 204 | -1.20e-4 | -1.20e-4 | -1.20e-4 | -1.40e-5 |
| 83 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 229 | +7.81e-5 | +7.81e-5 | +7.81e-5 | -4.74e-6 |
| 84 | 3.00e-1 | 2 | 1.94e-1 | 2.08e-1 | 2.01e-1 | 1.94e-1 | 189 | -3.65e-4 | -8.43e-6 | -1.87e-4 | -4.11e-5 |
| 85 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 217 | +3.21e-4 | +3.21e-4 | +3.21e-4 | -4.92e-6 |
| 86 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 231 | +2.10e-5 | +2.10e-5 | +2.10e-5 | -2.33e-6 |
| 87 | 3.00e-1 | 2 | 1.99e-1 | 2.12e-1 | 2.06e-1 | 1.99e-1 | 178 | -3.67e-4 | +6.46e-5 | -1.51e-4 | -3.27e-5 |
| 88 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 211 | +1.57e-4 | +1.57e-4 | +1.57e-4 | -1.37e-5 |
| 89 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 200 | -5.23e-5 | -5.23e-5 | -5.23e-5 | -1.76e-5 |
| 90 | 3.00e-1 | 2 | 1.94e-1 | 2.02e-1 | 1.98e-1 | 1.94e-1 | 178 | -2.33e-4 | -3.56e-5 | -1.34e-4 | -4.08e-5 |
| 91 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 228 | +3.96e-4 | +3.96e-4 | +3.96e-4 | +2.92e-6 |
| 92 | 3.00e-1 | 2 | 1.90e-1 | 2.12e-1 | 2.01e-1 | 1.90e-1 | 167 | -6.66e-4 | +8.79e-6 | -3.29e-4 | -6.35e-5 |
| 93 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 210 | +3.47e-4 | +3.47e-4 | +3.47e-4 | -2.24e-5 |
| 94 | 3.00e-1 | 2 | 1.83e-1 | 1.97e-1 | 1.90e-1 | 1.83e-1 | 153 | -4.80e-4 | -2.07e-4 | -3.43e-4 | -8.47e-5 |
| 95 | 3.00e-1 | 1 | 1.91e-1 | 1.91e-1 | 1.91e-1 | 1.91e-1 | 178 | +2.48e-4 | +2.48e-4 | +2.48e-4 | -5.15e-5 |
| 96 | 3.00e-1 | 2 | 1.89e-1 | 1.95e-1 | 1.92e-1 | 1.89e-1 | 165 | -1.93e-4 | +1.12e-4 | -4.08e-5 | -5.10e-5 |
| 97 | 3.00e-1 | 2 | 1.91e-1 | 1.94e-1 | 1.93e-1 | 1.91e-1 | 174 | -9.86e-5 | +1.52e-4 | +2.68e-5 | -3.75e-5 |
| 98 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 202 | +3.87e-4 | +3.87e-4 | +3.87e-4 | +4.96e-6 |
| 99 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 197 | -7.17e-5 | -7.17e-5 | -7.17e-5 | -2.70e-6 |
| 100 | 3.00e-2 | 3 | 4.92e-2 | 1.89e-1 | 1.13e-1 | 4.92e-2 | 141 | -5.07e-3 | -4.43e-4 | -2.86e-3 | -8.22e-4 |
| 101 | 3.00e-2 | 1 | 3.22e-2 | 3.22e-2 | 3.22e-2 | 3.22e-2 | 191 | -2.23e-3 | -2.23e-3 | -2.23e-3 | -9.63e-4 |
| 102 | 3.00e-2 | 1 | 2.51e-2 | 2.51e-2 | 2.51e-2 | 2.51e-2 | 183 | -1.36e-3 | -1.36e-3 | -1.36e-3 | -1.00e-3 |
| 103 | 3.00e-2 | 3 | 2.14e-2 | 2.40e-2 | 2.23e-2 | 2.14e-2 | 133 | -8.41e-4 | -1.65e-5 | -3.74e-4 | -8.30e-4 |
| 104 | 3.00e-2 | 1 | 2.38e-2 | 2.38e-2 | 2.38e-2 | 2.38e-2 | 169 | +6.31e-4 | +6.31e-4 | +6.31e-4 | -6.84e-4 |
| 105 | 3.00e-2 | 2 | 2.19e-2 | 2.39e-2 | 2.29e-2 | 2.19e-2 | 127 | -6.73e-4 | +2.73e-5 | -3.23e-4 | -6.19e-4 |
| 106 | 3.00e-2 | 2 | 2.33e-2 | 2.55e-2 | 2.44e-2 | 2.33e-2 | 137 | -6.36e-4 | +8.57e-4 | +1.11e-4 | -4.87e-4 |
| 107 | 3.00e-2 | 2 | 2.43e-2 | 2.47e-2 | 2.45e-2 | 2.43e-2 | 137 | -1.21e-4 | +3.51e-4 | +1.15e-4 | -3.75e-4 |
| 108 | 3.00e-2 | 2 | 2.55e-2 | 2.60e-2 | 2.57e-2 | 2.55e-2 | 147 | -1.52e-4 | +3.85e-4 | +1.17e-4 | -2.85e-4 |
| 109 | 3.00e-2 | 1 | 2.86e-2 | 2.86e-2 | 2.86e-2 | 2.86e-2 | 190 | +6.07e-4 | +6.07e-4 | +6.07e-4 | -1.95e-4 |
| 110 | 3.00e-2 | 3 | 2.60e-2 | 2.95e-2 | 2.76e-2 | 2.60e-2 | 136 | -5.78e-4 | +1.66e-4 | -2.54e-4 | -2.16e-4 |
| 111 | 3.00e-2 | 1 | 2.72e-2 | 2.72e-2 | 2.72e-2 | 2.72e-2 | 143 | +3.30e-4 | +3.30e-4 | +3.30e-4 | -1.61e-4 |
| 112 | 3.00e-2 | 3 | 2.70e-2 | 2.80e-2 | 2.75e-2 | 2.76e-2 | 138 | -2.93e-4 | +1.80e-4 | +1.17e-5 | -1.15e-4 |
| 113 | 3.00e-2 | 1 | 2.84e-2 | 2.84e-2 | 2.84e-2 | 2.84e-2 | 150 | +2.01e-4 | +2.01e-4 | +2.01e-4 | -8.31e-5 |
| 114 | 3.00e-2 | 2 | 2.75e-2 | 3.21e-2 | 2.98e-2 | 2.75e-2 | 119 | -1.29e-3 | +6.62e-4 | -3.15e-4 | -1.37e-4 |
| 115 | 3.00e-2 | 3 | 2.60e-2 | 3.11e-2 | 2.82e-2 | 2.60e-2 | 104 | -1.19e-3 | +7.37e-4 | -3.33e-4 | -2.02e-4 |
| 116 | 3.00e-2 | 2 | 2.70e-2 | 2.97e-2 | 2.83e-2 | 2.70e-2 | 98 | -9.84e-4 | +9.40e-4 | -2.17e-5 | -1.77e-4 |
| 117 | 3.00e-2 | 2 | 2.83e-2 | 2.93e-2 | 2.88e-2 | 2.83e-2 | 105 | -3.13e-4 | +6.27e-4 | +1.57e-4 | -1.18e-4 |
| 118 | 3.00e-2 | 3 | 2.82e-2 | 3.11e-2 | 2.93e-2 | 2.82e-2 | 99 | -7.34e-4 | +6.56e-4 | -8.00e-5 | -1.15e-4 |
| 119 | 3.00e-2 | 2 | 2.82e-2 | 3.08e-2 | 2.95e-2 | 2.82e-2 | 94 | -9.61e-4 | +6.91e-4 | -1.35e-4 | -1.27e-4 |
| 120 | 3.00e-2 | 3 | 2.85e-2 | 3.20e-2 | 3.03e-2 | 3.04e-2 | 115 | -1.17e-3 | +8.72e-4 | +8.37e-5 | -7.27e-5 |
| 121 | 3.00e-2 | 2 | 2.88e-2 | 3.36e-2 | 3.12e-2 | 2.88e-2 | 89 | -1.73e-3 | +6.75e-4 | -5.29e-4 | -1.71e-4 |
| 122 | 3.00e-2 | 4 | 2.76e-2 | 3.13e-2 | 2.87e-2 | 2.76e-2 | 83 | -1.23e-3 | +6.68e-4 | -1.91e-4 | -1.82e-4 |
| 123 | 3.00e-2 | 2 | 2.98e-2 | 3.20e-2 | 3.09e-2 | 2.98e-2 | 98 | -7.27e-4 | +1.19e-3 | +2.31e-4 | -1.13e-4 |
| 124 | 3.00e-2 | 4 | 2.66e-2 | 3.08e-2 | 2.77e-2 | 2.66e-2 | 69 | -2.00e-3 | +3.27e-4 | -4.36e-4 | -2.20e-4 |
| 125 | 3.00e-2 | 4 | 2.80e-2 | 3.09e-2 | 2.93e-2 | 2.80e-2 | 71 | -8.14e-4 | +1.37e-3 | +1.46e-5 | -1.62e-4 |
| 126 | 3.00e-2 | 2 | 2.97e-2 | 3.12e-2 | 3.05e-2 | 2.97e-2 | 71 | -7.18e-4 | +1.04e-3 | +1.63e-4 | -1.09e-4 |
| 127 | 3.00e-2 | 5 | 2.60e-2 | 3.44e-2 | 2.91e-2 | 2.60e-2 | 59 | -2.31e-3 | +1.24e-3 | -6.27e-4 | -3.40e-4 |
| 128 | 3.00e-2 | 3 | 2.64e-2 | 3.11e-2 | 2.82e-2 | 2.71e-2 | 64 | -2.51e-3 | +1.81e-3 | -1.10e-4 | -2.91e-4 |
| 129 | 3.00e-2 | 6 | 2.56e-2 | 3.40e-2 | 2.83e-2 | 2.64e-2 | 52 | -2.09e-3 | +2.10e-3 | -2.89e-4 | -2.97e-4 |
| 130 | 3.00e-2 | 3 | 2.57e-2 | 3.10e-2 | 2.77e-2 | 2.57e-2 | 51 | -3.17e-3 | +1.96e-3 | -5.78e-4 | -3.95e-4 |
| 131 | 3.00e-2 | 5 | 2.36e-2 | 3.22e-2 | 2.62e-2 | 2.40e-2 | 43 | -4.85e-3 | +2.55e-3 | -7.32e-4 | -5.40e-4 |
| 132 | 3.00e-2 | 6 | 2.36e-2 | 3.22e-2 | 2.56e-2 | 2.50e-2 | 48 | -6.26e-3 | +3.43e-3 | -5.62e-4 | -5.06e-4 |
| 133 | 3.00e-2 | 6 | 2.53e-2 | 3.25e-2 | 2.68e-2 | 2.59e-2 | 46 | -5.21e-3 | +3.01e-3 | -3.14e-4 | -4.04e-4 |
| 134 | 3.00e-2 | 7 | 2.44e-2 | 3.29e-2 | 2.68e-2 | 2.44e-2 | 39 | -4.33e-3 | +2.95e-3 | -5.52e-4 | -5.04e-4 |
| 135 | 3.00e-2 | 5 | 2.34e-2 | 3.19e-2 | 2.58e-2 | 2.38e-2 | 40 | -6.04e-3 | +3.48e-3 | -9.47e-4 | -6.87e-4 |
| 136 | 3.00e-2 | 6 | 2.37e-2 | 3.17e-2 | 2.65e-2 | 2.49e-2 | 39 | -3.81e-3 | +3.69e-3 | -3.56e-4 | -5.47e-4 |
| 137 | 3.00e-2 | 6 | 2.40e-2 | 3.42e-2 | 2.66e-2 | 2.59e-2 | 44 | -6.89e-3 | +3.84e-3 | -6.95e-4 | -5.61e-4 |
| 138 | 3.00e-2 | 8 | 2.43e-2 | 3.57e-2 | 2.74e-2 | 2.62e-2 | 41 | -5.07e-3 | +3.33e-3 | -5.62e-4 | -4.93e-4 |
| 139 | 3.00e-2 | 5 | 2.70e-2 | 3.51e-2 | 2.94e-2 | 2.78e-2 | 44 | -4.14e-3 | +3.56e-3 | -4.30e-4 | -4.78e-4 |
| 140 | 3.00e-2 | 6 | 2.68e-2 | 3.66e-2 | 2.91e-2 | 2.69e-2 | 41 | -5.44e-3 | +3.59e-3 | -5.76e-4 | -5.27e-4 |
| 141 | 3.00e-2 | 6 | 2.51e-2 | 3.57e-2 | 2.88e-2 | 2.79e-2 | 43 | -5.28e-3 | +3.49e-3 | -6.15e-4 | -5.48e-4 |
| 142 | 3.00e-2 | 6 | 2.78e-2 | 3.56e-2 | 3.02e-2 | 2.78e-2 | 40 | -3.26e-3 | +2.90e-3 | -4.37e-4 | -5.29e-4 |
| 143 | 3.00e-2 | 6 | 2.69e-2 | 3.58e-2 | 3.06e-2 | 2.69e-2 | 39 | -3.04e-3 | +3.50e-3 | -5.21e-4 | -6.04e-4 |
| 144 | 3.00e-2 | 6 | 2.53e-2 | 3.60e-2 | 2.86e-2 | 2.89e-2 | 43 | -9.26e-3 | +3.92e-3 | -5.12e-4 | -4.81e-4 |
| 145 | 3.00e-2 | 6 | 2.70e-2 | 3.62e-2 | 2.97e-2 | 2.97e-2 | 43 | -5.54e-3 | +2.74e-3 | -4.98e-4 | -4.21e-4 |
| 146 | 3.00e-2 | 6 | 2.94e-2 | 3.97e-2 | 3.23e-2 | 3.13e-2 | 43 | -4.71e-3 | +3.67e-3 | -3.25e-4 | -3.58e-4 |
| 147 | 3.00e-2 | 6 | 2.97e-2 | 4.13e-2 | 3.23e-2 | 3.07e-2 | 44 | -5.93e-3 | +3.24e-3 | -5.85e-4 | -4.30e-4 |
| 148 | 3.00e-2 | 6 | 2.71e-2 | 4.26e-2 | 3.15e-2 | 2.74e-2 | 35 | -6.27e-3 | +3.76e-3 | -1.22e-3 | -7.98e-4 |
| 149 | 3.00e-2 | 6 | 2.84e-2 | 3.87e-2 | 3.18e-2 | 3.32e-2 | 48 | -5.22e-3 | +4.28e-3 | -9.46e-5 | -4.16e-4 |
| 150 | 3.00e-3 | 7 | 3.21e-3 | 4.18e-2 | 1.28e-2 | 3.21e-3 | 56 | -1.53e-2 | +2.90e-3 | -7.47e-3 | -3.90e-3 |
| 151 | 3.00e-3 | 4 | 2.68e-3 | 3.92e-3 | 3.13e-3 | 2.84e-3 | 41 | -5.83e-3 | +2.63e-3 | -1.24e-3 | -2.98e-3 |
| 152 | 3.00e-3 | 6 | 2.54e-3 | 4.13e-3 | 3.03e-3 | 2.54e-3 | 37 | -6.42e-3 | +4.15e-3 | -1.35e-3 | -2.25e-3 |
| 153 | 3.00e-3 | 7 | 2.62e-3 | 3.85e-3 | 2.88e-3 | 2.67e-3 | 38 | -8.91e-3 | +4.80e-3 | -7.71e-4 | -1.45e-3 |
| 154 | 3.00e-3 | 7 | 2.58e-3 | 3.54e-3 | 2.81e-3 | 2.70e-3 | 38 | -7.27e-3 | +3.71e-3 | -5.81e-4 | -9.49e-4 |
| 155 | 3.00e-3 | 8 | 2.33e-3 | 3.62e-3 | 2.61e-3 | 2.46e-3 | 32 | -9.53e-3 | +4.02e-3 | -1.03e-3 | -8.63e-4 |
| 156 | 3.00e-3 | 7 | 2.41e-3 | 3.52e-3 | 2.72e-3 | 2.67e-3 | 37 | -1.05e-2 | +5.04e-3 | -6.69e-4 | -6.77e-4 |
| 157 | 3.00e-3 | 7 | 2.56e-3 | 3.83e-3 | 2.85e-3 | 2.70e-3 | 39 | -7.86e-3 | +4.13e-3 | -8.10e-4 | -6.77e-4 |
| 158 | 3.00e-3 | 6 | 2.88e-3 | 3.66e-3 | 3.10e-3 | 2.94e-3 | 42 | -4.17e-3 | +3.66e-3 | -2.61e-4 | -5.03e-4 |
| 159 | 3.00e-3 | 6 | 2.66e-3 | 3.58e-3 | 2.92e-3 | 2.66e-3 | 36 | -5.99e-3 | +2.77e-3 | -8.67e-4 | -6.80e-4 |
| 160 | 3.00e-3 | 7 | 2.48e-3 | 3.73e-3 | 2.96e-3 | 3.04e-3 | 42 | -8.23e-3 | +3.99e-3 | -5.21e-4 | -4.92e-4 |
| 161 | 3.00e-3 | 8 | 2.62e-3 | 4.01e-3 | 2.99e-3 | 2.87e-3 | 42 | -5.27e-3 | +3.26e-3 | -6.88e-4 | -5.00e-4 |
| 162 | 3.00e-3 | 4 | 2.98e-3 | 4.17e-3 | 3.32e-3 | 2.98e-3 | 39 | -7.61e-3 | +4.14e-3 | -1.11e-3 | -7.34e-4 |
| 163 | 3.00e-3 | 6 | 2.84e-3 | 3.84e-3 | 3.08e-3 | 2.96e-3 | 41 | -5.99e-3 | +2.93e-3 | -6.38e-4 | -6.50e-4 |
| 164 | 3.00e-3 | 6 | 2.94e-3 | 3.91e-3 | 3.18e-3 | 2.94e-3 | 38 | -5.38e-3 | +3.37e-3 | -6.02e-4 | -6.37e-4 |
| 165 | 3.00e-3 | 7 | 3.13e-3 | 3.67e-3 | 3.41e-3 | 3.49e-3 | 53 | -3.22e-3 | +2.94e-3 | +2.00e-4 | -2.14e-4 |
| 166 | 3.00e-3 | 3 | 3.20e-3 | 4.09e-3 | 3.61e-3 | 3.20e-3 | 46 | -2.90e-3 | +1.64e-3 | -1.16e-3 | -5.06e-4 |
| 167 | 3.00e-3 | 6 | 2.86e-3 | 4.23e-3 | 3.25e-3 | 2.92e-3 | 35 | -4.14e-3 | +2.76e-3 | -8.68e-4 | -6.66e-4 |
| 168 | 3.00e-3 | 8 | 2.63e-3 | 4.18e-3 | 3.06e-3 | 2.78e-3 | 42 | -7.45e-3 | +4.15e-3 | -8.22e-4 | -7.07e-4 |
| 169 | 3.00e-3 | 5 | 2.76e-3 | 4.18e-3 | 3.19e-3 | 2.90e-3 | 36 | -7.32e-3 | +4.59e-3 | -1.03e-3 | -8.35e-4 |
| 170 | 3.00e-3 | 6 | 2.71e-3 | 3.78e-3 | 3.10e-3 | 2.71e-3 | 36 | -4.88e-3 | +3.52e-3 | -8.78e-4 | -9.18e-4 |
| 171 | 3.00e-3 | 7 | 2.86e-3 | 3.97e-3 | 3.09e-3 | 2.98e-3 | 41 | -7.29e-3 | +4.80e-3 | -4.02e-4 | -6.16e-4 |
| 172 | 3.00e-3 | 7 | 2.73e-3 | 4.14e-3 | 3.07e-3 | 2.77e-3 | 33 | -8.03e-3 | +3.95e-3 | -1.01e-3 | -7.90e-4 |
| 173 | 3.00e-3 | 7 | 2.52e-3 | 3.97e-3 | 2.99e-3 | 3.03e-3 | 39 | -8.89e-3 | +4.35e-3 | -6.27e-4 | -5.78e-4 |
| 174 | 3.00e-3 | 6 | 2.73e-3 | 3.99e-3 | 3.16e-3 | 2.92e-3 | 37 | -5.25e-3 | +3.35e-3 | -8.28e-4 | -6.86e-4 |
| 175 | 3.00e-3 | 7 | 2.72e-3 | 4.20e-3 | 3.06e-3 | 2.89e-3 | 39 | -9.60e-3 | +4.29e-3 | -1.18e-3 | -8.37e-4 |
| 176 | 3.00e-3 | 7 | 2.68e-3 | 4.26e-3 | 3.04e-3 | 2.73e-3 | 39 | -1.07e-2 | +4.78e-3 | -1.33e-3 | -1.03e-3 |
| 177 | 3.00e-3 | 6 | 3.11e-3 | 4.07e-3 | 3.32e-3 | 3.11e-3 | 44 | -4.75e-3 | +4.90e-3 | -2.49e-4 | -6.96e-4 |
| 178 | 3.00e-3 | 7 | 2.81e-3 | 4.43e-3 | 3.19e-3 | 2.96e-3 | 34 | -7.82e-3 | +3.89e-3 | -9.16e-4 | -7.44e-4 |
| 179 | 3.00e-3 | 6 | 2.87e-3 | 4.08e-3 | 3.21e-3 | 2.92e-3 | 36 | -6.41e-3 | +3.91e-3 | -8.80e-4 | -8.06e-4 |
| 180 | 3.00e-3 | 9 | 2.79e-3 | 3.98e-3 | 3.05e-3 | 2.81e-3 | 32 | -6.99e-3 | +4.17e-3 | -6.42e-4 | -6.87e-4 |
| 181 | 3.00e-3 | 6 | 2.68e-3 | 3.70e-3 | 3.04e-3 | 3.02e-3 | 39 | -6.60e-3 | +4.67e-3 | -5.45e-4 | -6.00e-4 |
| 182 | 3.00e-3 | 8 | 3.01e-3 | 3.99e-3 | 3.30e-3 | 3.34e-3 | 41 | -4.51e-3 | +3.66e-3 | -7.74e-5 | -2.49e-4 |
| 183 | 3.00e-3 | 4 | 3.05e-3 | 4.52e-3 | 3.48e-3 | 3.06e-3 | 35 | -8.46e-3 | +3.24e-3 | -1.80e-3 | -7.89e-4 |
| 184 | 3.00e-3 | 9 | 2.80e-3 | 3.97e-3 | 3.12e-3 | 3.17e-3 | 40 | -6.20e-3 | +3.25e-3 | -4.60e-4 | -4.67e-4 |
| 185 | 3.00e-3 | 4 | 3.12e-3 | 4.25e-3 | 3.54e-3 | 3.12e-3 | 40 | -3.92e-3 | +3.53e-3 | -1.03e-3 | -7.08e-4 |
| 186 | 3.00e-3 | 6 | 3.03e-3 | 4.30e-3 | 3.41e-3 | 3.12e-3 | 40 | -5.84e-3 | +4.30e-3 | -6.75e-4 | -7.13e-4 |
| 187 | 3.00e-3 | 6 | 3.07e-3 | 4.10e-3 | 3.41e-3 | 3.39e-3 | 40 | -4.63e-3 | +3.52e-3 | -2.04e-4 | -4.48e-4 |
| 188 | 3.00e-3 | 9 | 3.05e-3 | 4.40e-3 | 3.36e-3 | 3.10e-3 | 37 | -6.06e-3 | +3.20e-3 | -6.22e-4 | -5.18e-4 |
| 189 | 3.00e-3 | 4 | 3.01e-3 | 4.07e-3 | 3.38e-3 | 3.01e-3 | 34 | -5.62e-3 | +3.37e-3 | -1.32e-3 | -8.35e-4 |
| 190 | 3.00e-3 | 7 | 3.00e-3 | 4.20e-3 | 3.26e-3 | 3.19e-3 | 40 | -7.48e-3 | +4.20e-3 | -4.96e-4 | -5.92e-4 |
| 191 | 3.00e-3 | 6 | 3.10e-3 | 4.13e-3 | 3.34e-3 | 3.31e-3 | 42 | -7.23e-3 | +3.18e-3 | -4.63e-4 | -4.78e-4 |
| 192 | 3.00e-3 | 7 | 2.96e-3 | 3.99e-3 | 3.24e-3 | 2.96e-3 | 37 | -3.81e-3 | +2.42e-3 | -7.68e-4 | -6.22e-4 |
| 193 | 3.00e-3 | 8 | 3.02e-3 | 4.05e-3 | 3.32e-3 | 3.04e-3 | 35 | -5.23e-3 | +4.03e-3 | -5.04e-4 | -5.94e-4 |
| 194 | 3.00e-3 | 5 | 2.90e-3 | 4.62e-3 | 3.39e-3 | 2.98e-3 | 36 | -9.24e-3 | +4.87e-3 | -1.69e-3 | -1.04e-3 |
| 195 | 3.00e-3 | 8 | 2.74e-3 | 4.06e-3 | 3.26e-3 | 3.16e-3 | 41 | -6.06e-3 | +4.14e-3 | -5.42e-4 | -7.06e-4 |
| 196 | 3.00e-3 | 4 | 3.27e-3 | 4.48e-3 | 3.68e-3 | 3.27e-3 | 41 | -4.42e-3 | +4.12e-3 | -7.37e-4 | -7.62e-4 |
| 197 | 3.00e-3 | 9 | 2.96e-3 | 4.80e-3 | 3.41e-3 | 3.32e-3 | 40 | -4.71e-3 | +4.30e-3 | -5.09e-4 | -4.84e-4 |
| 198 | 3.00e-3 | 4 | 3.16e-3 | 4.76e-3 | 3.67e-3 | 3.19e-3 | 39 | -7.28e-3 | +4.51e-3 | -1.42e-3 | -8.39e-4 |
| 199 | 3.00e-3 | 7 | 2.86e-3 | 4.43e-3 | 3.31e-3 | 3.03e-3 | 36 | -6.60e-3 | +3.80e-3 | -9.52e-4 | -8.46e-4 |

