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

GPU columns = compute utilization % (not load). Idle = total time with <5% utilization.

### resnet-graph

> Published: CIFAR-10 91.25% acc, Graph builder ([He et al. 2015](https://arxiv.org/abs/1512.03385), Table 6)

| Mode | Loss | Eval | vs Ref | Total (s) | Syncs | Avg Sync (ms) | GPU0 | GPU1 | GPU2 | Idle (s) |
|------|------|------|--------|-----------|-------|--------------|------|------|------|----------|
| cpu-async | 0.052117 | 0.9159 | +0.0034 | 1778.4 | 588 | 78.1 | 100% | 100% | 100% | 2.8 |

## Best Mode per Model

| Model | Best Eval | Mode | Fastest (within 2% of solo-0) | Mode |
|-------|-----------|------|-------------------------------|------|
| resnet-graph | 0.9159 | cpu-async | - | - |

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
| cpu-async | 2.0161 | 0.7455 | 0.6180 | 0.5662 | 0.5252 | 0.5096 | 0.4945 | 0.4839 | 0.4665 | 0.4654 | 0.2029 | 0.1667 | 0.1432 | 0.1304 | 0.1163 | 0.0683 | 0.0627 | 0.0591 | 0.0558 | 0.0521 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | cpu-async | 0 | cuda:0 | 0.3990 | 2.7 |
| resnet-graph | cpu-async | 1 | cuda:1 | 0.3044 | 3.7 |
| resnet-graph | cpu-async | 2 | cuda:2 | 0.2967 | 3.7 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) | GPU2 Peak (MB) | GPU2 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|---------------|---------------|
| resnet-graph | cpu-async | 358 | 357 | 400 | 398 | 386 | 384 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| cpu-async | gpu1 | 1590.1 | 0.6 | epoch-boundary(178) |
| cpu-async | gpu1 | 1777.8 | 0.6 | epoch-boundary(199) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | cpu-async | gpu1 | 1.3s | 0.0s | 0.0s | 0.0s | 1.9s |
| resnet-graph | cpu-async | gpu2 | 0.0s | 0.0s | 0.0s | 0.0s | 0.9s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | cpu-async | 314 | 0 | 588 | 78.1 | 1386/9176 | 588 | 78.1 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | cpu-async | 187.1 | 10.5% |

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
| resnet-graph | cpu-async | 195 | 588 | 0 | 7.11e-3 | -1.66e-5 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | cpu-async | 0 | 588 | 6.52e-2 | 7.28e-2 | 1.89e-3 | 4.65e-1 | 29.4 | -2.69e-4 | 1.69e-3 |
| resnet-graph | cpu-async | 1 | 588 | 6.56e-2 | 7.35e-2 | 1.91e-3 | 4.22e-1 | 32.3 | -2.91e-4 | 1.85e-3 |
| resnet-graph | cpu-async | 2 | 588 | 6.56e-2 | 7.29e-2 | 1.76e-3 | 4.05e-1 | 38.3 | -2.78e-4 | 1.72e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | cpu-async | rank0 ↔ rank1 | +0.9968 |
| resnet-graph | cpu-async | rank0 ↔ rank2 | +0.9975 |
| resnet-graph | cpu-async | rank1 ↔ rank2 | +0.9971 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | cpu-async | 61 (1,2,3,4,5,6,8,9…148,149) | 0 (—) | — | 1,2,3,4,5,6,8,9…148,149 | — |

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
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 218 | +0.206 |
| resnet-graph | cpu-async | 3.00e-2 | 100–148 | 199 | +0.185 |
| resnet-graph | cpu-async | 3.00e-3 | 149–199 | 167 | +0.063 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | cpu-async | 586 | +0.030 | 194 | +0.241 | +0.303 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | cpu-async | 587 | 3.36e1–7.96e1 | 6.39e1 | 2.09e-3 | 3.83e-3 | 4.15e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 220 | 31–78024 | +9.486e-6 | 0.459 | +9.618e-6 | 0.483 | 97 | +6.144e-6 | 0.316 | 31–1009 | +1.019e-3 | 0.757 |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 209 | 914–78024 | +9.825e-6 | 0.557 | +9.902e-6 | 0.573 | 96 | +6.055e-6 | 0.304 | 76–1009 | +1.058e-3 | 0.915 |
| resnet-graph | cpu-async | 3.00e-2 | 100–148 | 200 | 78527–116292 | -4.720e-7 | 0.001 | -6.221e-7 | 0.001 | 49 | -3.359e-6 | 0.022 | 78–542 | +4.754e-4 | 0.088 |
| resnet-graph | cpu-async | 3.00e-3 | 149–199 | 168 | 116465–155664 | +2.039e-5 | 0.089 | +2.101e-5 | 0.094 | 49 | +3.296e-5 | 0.470 | 76–1042 | +1.331e-3 | 0.317 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | +1.019e-3 | r0: +1.006e-3, r1: +1.028e-3, r2: +1.024e-3 | r0: 0.749, r1: 0.741, r2: 0.766 | 1.02× | ✓ |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +1.058e-3 | r0: +1.041e-3, r1: +1.072e-3, r2: +1.061e-3 | r0: 0.910, r1: 0.907, r2: 0.907 | 1.03× | ✓ |
| resnet-graph | cpu-async | 3.00e-2 | 100–148 | +4.754e-4 | r0: +4.092e-4, r1: +4.966e-4, r2: +5.199e-4 | r0: 0.070, r1: 0.088, r2: 0.097 | 1.27× | ✓ |
| resnet-graph | cpu-async | 3.00e-3 | 149–199 | +1.331e-3 | r0: +1.326e-3, r1: +1.333e-3, r2: +1.334e-3 | r0: 0.319, r1: 0.316, r2: 0.314 | 1.01× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | cpu-async | `█▇▇▇▇███████████████████▆▄▄▄▅▅▅▅▅▅▅▅▄▁▁▁▁▁▂▂▂▂▂▂▂` | `▁▇▇▇▇▇██████████████████▇▇▇███▇▇▇▇▇▇▄▅▇▇▇▇███████` |

### Per-Epoch Detail: resnet-graph / cpu-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 11 | 5.80e-2 | 4.65e-1 | 1.65e-1 | 9.91e-2 | 38 | -4.15e-2 | +1.56e-2 | -1.41e-2 | -1.00e-2 |
| 1 | 3.00e-1 | 9 | 8.80e-2 | 1.33e-1 | 1.00e-1 | 9.83e-2 | 31 | -8.67e-3 | +4.03e-3 | -4.09e-4 | -3.08e-3 |
| 2 | 3.00e-1 | 5 | 9.58e-2 | 1.31e-1 | 1.05e-1 | 1.03e-1 | 34 | -9.32e-3 | +4.34e-3 | -6.79e-4 | -1.96e-3 |
| 3 | 3.00e-1 | 7 | 9.36e-2 | 1.50e-1 | 1.11e-1 | 1.22e-1 | 49 | -1.35e-2 | +4.89e-3 | -6.36e-4 | -1.03e-3 |
| 4 | 3.00e-1 | 8 | 1.06e-1 | 1.52e-1 | 1.24e-1 | 1.06e-1 | 35 | -4.81e-3 | +2.53e-3 | -7.57e-4 | -9.60e-4 |
| 5 | 3.00e-1 | 5 | 1.03e-1 | 1.45e-1 | 1.14e-1 | 1.11e-1 | 37 | -1.01e-2 | +4.01e-3 | -8.44e-4 | -8.64e-4 |
| 6 | 3.00e-1 | 7 | 1.11e-1 | 1.49e-1 | 1.26e-1 | 1.24e-1 | 47 | -8.01e-3 | +4.13e-3 | -2.07e-4 | -5.05e-4 |
| 7 | 3.00e-1 | 5 | 1.06e-1 | 1.64e-1 | 1.22e-1 | 1.09e-1 | 39 | -7.80e-3 | +3.04e-3 | -1.38e-3 | -8.45e-4 |
| 8 | 3.00e-1 | 8 | 9.68e-2 | 1.61e-1 | 1.13e-1 | 9.68e-2 | 33 | -7.81e-3 | +4.21e-3 | -1.13e-3 | -9.98e-4 |
| 9 | 3.00e-1 | 4 | 1.00e-1 | 1.44e-1 | 1.16e-1 | 1.12e-1 | 41 | -6.60e-3 | +5.04e-3 | +1.67e-5 | -7.06e-4 |
| 10 | 3.00e-1 | 7 | 1.00e-1 | 1.56e-1 | 1.13e-1 | 1.10e-1 | 40 | -7.05e-3 | +3.62e-3 | -6.36e-4 | -5.68e-4 |
| 11 | 3.00e-1 | 9 | 9.21e-2 | 1.42e-1 | 1.03e-1 | 9.21e-2 | 31 | -8.16e-3 | +3.24e-3 | -1.09e-3 | -8.40e-4 |
| 12 | 3.00e-1 | 4 | 1.15e-1 | 1.47e-1 | 1.24e-1 | 1.19e-1 | 48 | -5.80e-3 | +5.95e-3 | +2.20e-4 | -5.16e-4 |
| 13 | 3.00e-1 | 8 | 1.01e-1 | 1.62e-1 | 1.15e-1 | 1.02e-1 | 39 | -5.42e-3 | +2.93e-3 | -8.69e-4 | -6.73e-4 |
| 14 | 3.00e-1 | 4 | 1.08e-1 | 1.46e-1 | 1.20e-1 | 1.12e-1 | 45 | -6.09e-3 | +4.35e-3 | -5.44e-4 | -6.50e-4 |
| 15 | 3.00e-1 | 7 | 9.46e-2 | 1.57e-1 | 1.09e-1 | 9.92e-2 | 31 | -7.47e-3 | +3.80e-3 | -1.10e-3 | -8.08e-4 |
| 16 | 3.00e-1 | 7 | 9.42e-2 | 1.47e-1 | 1.04e-1 | 1.02e-1 | 36 | -1.30e-2 | +4.58e-3 | -1.04e-3 | -7.78e-4 |
| 17 | 3.00e-1 | 9 | 9.78e-2 | 1.44e-1 | 1.07e-1 | 9.78e-2 | 32 | -8.89e-3 | +4.46e-3 | -7.43e-4 | -7.37e-4 |
| 18 | 3.00e-1 | 1 | 9.66e-2 | 9.66e-2 | 9.66e-2 | 9.66e-2 | 32 | -3.70e-4 | -3.70e-4 | -3.70e-4 | -7.01e-4 |
| 19 | 3.00e-1 | 1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 334 | +2.58e-3 | +2.58e-3 | +2.58e-3 | -3.72e-4 |
| 20 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 282 | -1.50e-4 | -1.50e-4 | -1.50e-4 | -3.50e-4 |
| 21 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 283 | -9.06e-5 | -9.06e-5 | -9.06e-5 | -3.24e-4 |
| 22 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 305 | +2.97e-5 | +2.97e-5 | +2.97e-5 | -2.89e-4 |
| 23 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 298 | -1.97e-5 | -1.97e-5 | -1.97e-5 | -2.62e-4 |
| 24 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 310 | +2.38e-5 | +2.38e-5 | +2.38e-5 | -2.33e-4 |
| 26 | 3.00e-1 | 2 | 2.06e-1 | 2.34e-1 | 2.20e-1 | 2.06e-1 | 277 | -4.59e-4 | +2.14e-4 | -1.22e-4 | -2.16e-4 |
| 28 | 3.00e-1 | 1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 318 | +2.14e-4 | +2.14e-4 | +2.14e-4 | -1.73e-4 |
| 29 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 311 | -6.61e-5 | -6.61e-5 | -6.61e-5 | -1.62e-4 |
| 30 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 315 | -1.04e-6 | -1.04e-6 | -1.04e-6 | -1.46e-4 |
| 31 | 3.00e-1 | 1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 309 | +8.10e-5 | +8.10e-5 | +8.10e-5 | -1.23e-4 |
| 32 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 296 | -1.04e-4 | -1.04e-4 | -1.04e-4 | -1.21e-4 |
| 33 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 253 | -2.12e-4 | -2.12e-4 | -2.12e-4 | -1.30e-4 |
| 34 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 271 | +1.24e-5 | +1.24e-5 | +1.24e-5 | -1.16e-4 |
| 35 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 270 | +4.00e-5 | +4.00e-5 | +4.00e-5 | -1.00e-4 |
| 36 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 265 | +2.10e-5 | +2.10e-5 | +2.10e-5 | -8.83e-5 |
| 37 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 305 | +1.66e-4 | +1.66e-4 | +1.66e-4 | -6.28e-5 |
| 38 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 265 | -2.03e-4 | -2.03e-4 | -2.03e-4 | -7.68e-5 |
| 39 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 286 | +4.20e-5 | +4.20e-5 | +4.20e-5 | -6.50e-5 |
| 40 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 274 | -3.22e-5 | -3.22e-5 | -3.22e-5 | -6.17e-5 |
| 41 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 287 | +9.36e-5 | +9.36e-5 | +9.36e-5 | -4.62e-5 |
| 42 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 293 | -1.80e-5 | -1.80e-5 | -1.80e-5 | -4.33e-5 |
| 43 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 287 | -2.24e-5 | -2.24e-5 | -2.24e-5 | -4.12e-5 |
| 44 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 292 | +4.64e-5 | +4.64e-5 | +4.64e-5 | -3.25e-5 |
| 45 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 314 | +4.01e-5 | +4.01e-5 | +4.01e-5 | -2.52e-5 |
| 46 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 279 | -6.59e-5 | -6.59e-5 | -6.59e-5 | -2.93e-5 |
| 47 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 302 | +1.13e-4 | +1.13e-4 | +1.13e-4 | -1.50e-5 |
| 48 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 270 | -1.27e-4 | -1.27e-4 | -1.27e-4 | -2.63e-5 |
| 49 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 258 | -6.87e-5 | -6.87e-5 | -6.87e-5 | -3.05e-5 |
| 50 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 292 | +1.83e-4 | +1.83e-4 | +1.83e-4 | -9.14e-6 |
| 51 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 266 | -9.62e-5 | -9.62e-5 | -9.62e-5 | -1.79e-5 |
| 52 | 3.00e-1 | 1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 211 | -4.69e-4 | -4.69e-4 | -4.69e-4 | -6.29e-5 |
| 53 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 241 | +1.45e-4 | +1.45e-4 | +1.45e-4 | -4.22e-5 |
| 54 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 285 | +3.18e-4 | +3.18e-4 | +3.18e-4 | -6.19e-6 |
| 55 | 3.00e-1 | 2 | 1.93e-1 | 2.04e-1 | 1.99e-1 | 1.93e-1 | 215 | -3.25e-4 | -2.49e-4 | -2.87e-4 | -5.91e-5 |
| 57 | 3.00e-1 | 2 | 1.94e-1 | 2.22e-1 | 2.08e-1 | 1.94e-1 | 215 | -6.41e-4 | +4.84e-4 | -7.82e-5 | -6.84e-5 |
| 58 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 266 | +3.80e-4 | +3.80e-4 | +3.80e-4 | -2.36e-5 |
| 59 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 263 | -2.34e-5 | -2.34e-5 | -2.34e-5 | -2.35e-5 |
| 60 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 258 | -5.02e-5 | -5.02e-5 | -5.02e-5 | -2.62e-5 |
| 61 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 261 | -3.12e-5 | -3.12e-5 | -3.12e-5 | -2.67e-5 |
| 62 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 235 | -1.23e-4 | -1.23e-4 | -1.23e-4 | -3.63e-5 |
| 63 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 231 | +1.55e-5 | +1.55e-5 | +1.55e-5 | -3.11e-5 |
| 64 | 3.00e-1 | 2 | 1.91e-1 | 2.01e-1 | 1.96e-1 | 1.91e-1 | 202 | -2.42e-4 | -4.97e-5 | -1.46e-4 | -5.39e-5 |
| 65 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 246 | +3.28e-4 | +3.28e-4 | +3.28e-4 | -1.57e-5 |
| 66 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 235 | -8.36e-5 | -8.36e-5 | -8.36e-5 | -2.25e-5 |
| 67 | 3.00e-1 | 1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 219 | -1.97e-4 | -1.97e-4 | -1.97e-4 | -4.00e-5 |
| 68 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 231 | +7.77e-5 | +7.77e-5 | +7.77e-5 | -2.82e-5 |
| 69 | 3.00e-1 | 2 | 1.95e-1 | 2.01e-1 | 1.98e-1 | 1.95e-1 | 202 | -1.42e-4 | +4.98e-5 | -4.62e-5 | -3.26e-5 |
| 70 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 220 | +1.27e-4 | +1.27e-4 | +1.27e-4 | -1.67e-5 |
| 71 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 253 | +2.26e-4 | +2.26e-4 | +2.26e-4 | +7.60e-6 |
| 72 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 233 | -1.80e-4 | -1.80e-4 | -1.80e-4 | -1.12e-5 |
| 73 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 233 | +1.63e-5 | +1.63e-5 | +1.63e-5 | -8.46e-6 |
| 74 | 3.00e-1 | 2 | 1.89e-1 | 2.14e-1 | 2.01e-1 | 1.89e-1 | 188 | -6.71e-4 | +1.75e-4 | -2.48e-4 | -5.81e-5 |
| 75 | 3.00e-1 | 1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 217 | +1.96e-4 | +1.96e-4 | +1.96e-4 | -3.27e-5 |
| 76 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 220 | +7.39e-5 | +7.39e-5 | +7.39e-5 | -2.21e-5 |
| 77 | 3.00e-1 | 2 | 1.87e-1 | 2.02e-1 | 1.95e-1 | 1.87e-1 | 188 | -3.95e-4 | +4.52e-5 | -1.75e-4 | -5.33e-5 |
| 78 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 247 | +4.68e-4 | +4.68e-4 | +4.68e-4 | -1.13e-6 |
| 79 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 221 | -1.94e-4 | -1.94e-4 | -1.94e-4 | -2.04e-5 |
| 80 | 3.00e-1 | 2 | 1.90e-1 | 1.96e-1 | 1.93e-1 | 1.90e-1 | 188 | -1.78e-4 | -1.38e-4 | -1.58e-4 | -4.67e-5 |
| 81 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 215 | +1.93e-4 | +1.93e-4 | +1.93e-4 | -2.28e-5 |
| 82 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 237 | +2.53e-4 | +2.53e-4 | +2.53e-4 | +4.75e-6 |
| 83 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 219 | -2.14e-4 | -2.14e-4 | -2.14e-4 | -1.72e-5 |
| 84 | 3.00e-1 | 2 | 1.85e-1 | 2.13e-1 | 1.99e-1 | 1.85e-1 | 174 | -8.31e-4 | +2.55e-4 | -2.88e-4 | -7.40e-5 |
| 85 | 3.00e-1 | 1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 204 | +2.44e-4 | +2.44e-4 | +2.44e-4 | -4.22e-5 |
| 86 | 3.00e-1 | 2 | 1.85e-1 | 2.05e-1 | 1.95e-1 | 1.85e-1 | 181 | -5.59e-4 | +2.52e-4 | -1.54e-4 | -6.75e-5 |
| 87 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 212 | +4.28e-4 | +4.28e-4 | +4.28e-4 | -1.80e-5 |
| 88 | 3.00e-1 | 2 | 1.86e-1 | 1.94e-1 | 1.90e-1 | 1.86e-1 | 164 | -2.75e-4 | -2.42e-4 | -2.58e-4 | -6.38e-5 |
| 89 | 3.00e-1 | 1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 198 | +1.98e-4 | +1.98e-4 | +1.98e-4 | -3.76e-5 |
| 90 | 3.00e-1 | 2 | 1.83e-1 | 1.99e-1 | 1.91e-1 | 1.83e-1 | 164 | -4.94e-4 | +1.46e-4 | -1.74e-4 | -6.67e-5 |
| 91 | 3.00e-1 | 1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 197 | +3.09e-4 | +3.09e-4 | +3.09e-4 | -2.91e-5 |
| 92 | 3.00e-1 | 2 | 1.84e-1 | 1.93e-1 | 1.88e-1 | 1.84e-1 | 164 | -3.05e-4 | -4.79e-5 | -1.76e-4 | -5.83e-5 |
| 93 | 3.00e-1 | 1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 199 | +2.62e-4 | +2.62e-4 | +2.62e-4 | -2.63e-5 |
| 94 | 3.00e-1 | 2 | 1.87e-1 | 1.94e-1 | 1.90e-1 | 1.87e-1 | 170 | -2.27e-4 | +1.08e-5 | -1.08e-4 | -4.31e-5 |
| 95 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 196 | +3.09e-4 | +3.09e-4 | +3.09e-4 | -7.91e-6 |
| 96 | 3.00e-1 | 2 | 1.78e-1 | 2.04e-1 | 1.91e-1 | 1.78e-1 | 151 | -8.84e-4 | +1.35e-4 | -3.75e-4 | -8.27e-5 |
| 97 | 3.00e-1 | 1 | 1.85e-1 | 1.85e-1 | 1.85e-1 | 1.85e-1 | 175 | +1.99e-4 | +1.99e-4 | +1.99e-4 | -5.45e-5 |
| 98 | 3.00e-1 | 2 | 1.73e-1 | 1.89e-1 | 1.81e-1 | 1.73e-1 | 147 | -5.96e-4 | +1.32e-4 | -2.32e-4 | -9.18e-5 |
| 99 | 3.00e-1 | 2 | 1.77e-1 | 1.94e-1 | 1.85e-1 | 1.77e-1 | 147 | -6.38e-4 | +6.24e-4 | -6.91e-6 | -8.20e-5 |
| 100 | 3.00e-2 | 1 | 1.34e-1 | 1.34e-1 | 1.34e-1 | 1.34e-1 | 182 | -1.51e-3 | -1.51e-3 | -1.51e-3 | -2.25e-4 |
| 101 | 3.00e-2 | 3 | 1.92e-2 | 4.42e-2 | 2.84e-2 | 1.92e-2 | 148 | -5.70e-3 | -8.21e-4 | -3.77e-3 | -1.14e-3 |
| 102 | 3.00e-2 | 1 | 2.17e-2 | 2.17e-2 | 2.17e-2 | 2.17e-2 | 173 | +6.92e-4 | +6.92e-4 | +6.92e-4 | -9.56e-4 |
| 103 | 3.00e-2 | 1 | 2.04e-2 | 2.04e-2 | 2.04e-2 | 2.04e-2 | 162 | -3.69e-4 | -3.69e-4 | -3.69e-4 | -8.97e-4 |
| 104 | 3.00e-2 | 3 | 1.98e-2 | 2.22e-2 | 2.08e-2 | 1.98e-2 | 130 | -5.59e-4 | +4.89e-4 | -1.21e-4 | -6.94e-4 |
| 105 | 3.00e-2 | 1 | 2.25e-2 | 2.25e-2 | 2.25e-2 | 2.25e-2 | 160 | +8.00e-4 | +8.00e-4 | +8.00e-4 | -5.45e-4 |
| 106 | 3.00e-2 | 2 | 2.11e-2 | 2.34e-2 | 2.23e-2 | 2.11e-2 | 130 | -8.01e-4 | +2.37e-4 | -2.82e-4 | -5.00e-4 |
| 107 | 3.00e-2 | 2 | 2.24e-2 | 2.28e-2 | 2.26e-2 | 2.24e-2 | 138 | -1.10e-4 | +5.23e-4 | +2.06e-4 | -3.69e-4 |
| 108 | 3.00e-2 | 2 | 2.21e-2 | 2.44e-2 | 2.33e-2 | 2.21e-2 | 127 | -7.91e-4 | +5.03e-4 | -1.44e-4 | -3.33e-4 |
| 109 | 3.00e-2 | 2 | 2.21e-2 | 2.34e-2 | 2.27e-2 | 2.21e-2 | 127 | -4.59e-4 | +3.89e-4 | -3.49e-5 | -2.80e-4 |
| 110 | 3.00e-2 | 3 | 2.30e-2 | 2.48e-2 | 2.41e-2 | 2.30e-2 | 118 | -6.53e-4 | +7.04e-4 | +3.75e-5 | -2.07e-4 |
| 111 | 3.00e-2 | 1 | 2.50e-2 | 2.50e-2 | 2.50e-2 | 2.50e-2 | 145 | +5.96e-4 | +5.96e-4 | +5.96e-4 | -1.27e-4 |
| 112 | 3.00e-2 | 2 | 2.26e-2 | 2.53e-2 | 2.40e-2 | 2.26e-2 | 110 | -1.02e-3 | +7.29e-5 | -4.72e-4 | -1.98e-4 |
| 113 | 3.00e-2 | 2 | 2.29e-2 | 2.64e-2 | 2.47e-2 | 2.29e-2 | 104 | -1.38e-3 | +1.06e-3 | -1.57e-4 | -2.02e-4 |
| 114 | 3.00e-2 | 4 | 2.32e-2 | 2.50e-2 | 2.39e-2 | 2.37e-2 | 110 | -7.06e-4 | +3.88e-4 | +5.72e-5 | -1.20e-4 |
| 115 | 3.00e-2 | 1 | 2.72e-2 | 2.72e-2 | 2.72e-2 | 2.72e-2 | 148 | +9.32e-4 | +9.32e-4 | +9.32e-4 | -1.44e-5 |
| 116 | 3.00e-2 | 3 | 2.41e-2 | 2.79e-2 | 2.53e-2 | 2.41e-2 | 98 | -1.53e-3 | +1.54e-4 | -4.58e-4 | -1.36e-4 |
| 117 | 3.00e-2 | 2 | 2.47e-2 | 2.78e-2 | 2.63e-2 | 2.47e-2 | 101 | -1.19e-3 | +9.94e-4 | -9.96e-5 | -1.40e-4 |
| 118 | 3.00e-2 | 3 | 2.39e-2 | 2.98e-2 | 2.64e-2 | 2.56e-2 | 102 | -2.45e-3 | +1.34e-3 | -1.49e-4 | -1.47e-4 |
| 119 | 3.00e-2 | 3 | 2.38e-2 | 2.99e-2 | 2.60e-2 | 2.43e-2 | 86 | -2.74e-3 | +1.09e-3 | -4.68e-4 | -2.41e-4 |
| 120 | 3.00e-2 | 3 | 2.34e-2 | 2.87e-2 | 2.53e-2 | 2.34e-2 | 80 | -2.39e-3 | +1.25e-3 | -4.49e-4 | -3.10e-4 |
| 121 | 3.00e-2 | 3 | 2.32e-2 | 2.77e-2 | 2.50e-2 | 2.41e-2 | 77 | -2.31e-3 | +1.43e-3 | -1.36e-4 | -2.71e-4 |
| 122 | 3.00e-2 | 4 | 2.42e-2 | 2.79e-2 | 2.54e-2 | 2.43e-2 | 77 | -1.15e-3 | +1.31e-3 | -1.08e-4 | -2.28e-4 |
| 123 | 3.00e-2 | 2 | 2.45e-2 | 2.91e-2 | 2.68e-2 | 2.45e-2 | 77 | -2.24e-3 | +1.45e-3 | -3.92e-4 | -2.78e-4 |
| 124 | 3.00e-2 | 3 | 2.49e-2 | 3.00e-2 | 2.66e-2 | 2.49e-2 | 81 | -2.43e-3 | +1.67e-3 | -2.51e-4 | -2.85e-4 |
| 125 | 3.00e-2 | 4 | 2.42e-2 | 3.02e-2 | 2.61e-2 | 2.42e-2 | 67 | -2.50e-3 | +1.56e-3 | -3.68e-4 | -3.29e-4 |
| 126 | 3.00e-2 | 5 | 2.16e-2 | 2.91e-2 | 2.41e-2 | 2.16e-2 | 59 | -3.18e-3 | +1.75e-3 | -6.33e-4 | -4.74e-4 |
| 127 | 3.00e-2 | 3 | 2.33e-2 | 2.76e-2 | 2.47e-2 | 2.33e-2 | 60 | -2.84e-3 | +2.76e-3 | -2.00e-5 | -3.76e-4 |
| 128 | 3.00e-2 | 5 | 2.21e-2 | 2.86e-2 | 2.41e-2 | 2.21e-2 | 52 | -2.83e-3 | +2.23e-3 | -4.35e-4 | -4.21e-4 |
| 129 | 3.00e-2 | 7 | 2.07e-2 | 2.97e-2 | 2.27e-2 | 2.07e-2 | 44 | -5.15e-3 | +3.40e-3 | -5.55e-4 | -4.96e-4 |
| 130 | 3.00e-2 | 4 | 2.07e-2 | 3.05e-2 | 2.35e-2 | 2.07e-2 | 41 | -8.38e-3 | +4.05e-3 | -1.24e-3 | -7.75e-4 |
| 131 | 3.00e-2 | 6 | 1.99e-2 | 2.62e-2 | 2.20e-2 | 2.15e-2 | 42 | -4.18e-3 | +2.95e-3 | -2.40e-4 | -5.12e-4 |
| 132 | 3.00e-2 | 7 | 1.89e-2 | 2.93e-2 | 2.21e-2 | 1.89e-2 | 40 | -7.79e-3 | +3.54e-3 | -1.06e-3 | -8.25e-4 |
| 133 | 3.00e-2 | 4 | 2.21e-2 | 2.75e-2 | 2.38e-2 | 2.25e-2 | 53 | -4.35e-3 | +4.84e-3 | -3.45e-5 | -5.94e-4 |
| 134 | 3.00e-2 | 9 | 1.95e-2 | 3.28e-2 | 2.36e-2 | 2.02e-2 | 35 | -4.17e-3 | +4.25e-3 | -6.79e-4 | -6.78e-4 |
| 135 | 3.00e-2 | 3 | 2.33e-2 | 2.83e-2 | 2.55e-2 | 2.33e-2 | 42 | -2.87e-3 | +4.36e-3 | -3.04e-5 | -5.57e-4 |
| 136 | 3.00e-2 | 6 | 2.26e-2 | 3.08e-2 | 2.47e-2 | 2.41e-2 | 45 | -5.71e-3 | +3.71e-3 | -3.58e-4 | -4.34e-4 |
| 137 | 3.00e-2 | 6 | 2.01e-2 | 3.33e-2 | 2.33e-2 | 2.20e-2 | 38 | -1.15e-2 | +3.85e-3 | -1.37e-3 | -7.55e-4 |
| 138 | 3.00e-2 | 9 | 1.96e-2 | 3.21e-2 | 2.28e-2 | 2.26e-2 | 40 | -9.47e-3 | +4.24e-3 | -6.87e-4 | -5.43e-4 |
| 139 | 3.00e-2 | 5 | 2.20e-2 | 3.25e-2 | 2.60e-2 | 2.59e-2 | 48 | -7.32e-3 | +4.51e-3 | -3.70e-4 | -4.49e-4 |
| 140 | 3.00e-2 | 8 | 2.18e-2 | 3.30e-2 | 2.50e-2 | 2.34e-2 | 40 | -6.53e-3 | +2.95e-3 | -6.33e-4 | -4.95e-4 |
| 141 | 3.00e-2 | 4 | 2.31e-2 | 3.59e-2 | 2.77e-2 | 2.31e-2 | 34 | -5.52e-3 | +4.92e-3 | -1.40e-3 | -8.76e-4 |
| 142 | 3.00e-2 | 7 | 2.10e-2 | 3.07e-2 | 2.31e-2 | 2.23e-2 | 34 | -1.07e-2 | +4.27e-3 | -8.81e-4 | -7.81e-4 |
| 143 | 3.00e-2 | 7 | 2.10e-2 | 3.53e-2 | 2.46e-2 | 2.20e-2 | 38 | -1.06e-2 | +5.80e-3 | -1.20e-3 | -9.55e-4 |
| 144 | 3.00e-2 | 8 | 2.44e-2 | 3.41e-2 | 2.67e-2 | 2.62e-2 | 41 | -5.34e-3 | +5.21e-3 | -1.01e-4 | -4.53e-4 |
| 145 | 3.00e-2 | 5 | 2.29e-2 | 3.41e-2 | 2.65e-2 | 2.29e-2 | 33 | -8.33e-3 | +3.34e-3 | -1.56e-3 | -9.34e-4 |
| 146 | 3.00e-2 | 8 | 1.96e-2 | 3.30e-2 | 2.45e-2 | 2.51e-2 | 39 | -1.59e-2 | +5.23e-3 | -7.05e-4 | -6.30e-4 |
| 147 | 3.00e-2 | 6 | 2.36e-2 | 3.48e-2 | 2.68e-2 | 2.76e-2 | 39 | -7.87e-3 | +4.26e-3 | -2.96e-4 | -3.77e-4 |
| 148 | 3.00e-2 | 7 | 2.36e-2 | 3.66e-2 | 2.78e-2 | 2.62e-2 | 35 | -8.73e-3 | +3.91e-3 | -8.01e-4 | -5.60e-4 |
| 149 | 3.00e-3 | 10 | 4.76e-3 | 3.59e-2 | 2.27e-2 | 4.76e-3 | 35 | -3.13e-2 | +4.14e-3 | -5.38e-3 | -5.04e-3 |
| 150 | 3.00e-3 | 5 | 2.19e-3 | 3.70e-3 | 2.58e-3 | 2.32e-3 | 33 | -1.21e-2 | +9.28e-4 | -3.37e-3 | -4.17e-3 |
| 151 | 3.00e-3 | 7 | 2.04e-3 | 3.23e-3 | 2.37e-3 | 2.35e-3 | 43 | -1.48e-2 | +4.62e-3 | -9.40e-4 | -2.35e-3 |
| 152 | 3.00e-3 | 8 | 1.98e-3 | 3.42e-3 | 2.28e-3 | 2.11e-3 | 29 | -1.53e-2 | +6.00e-3 | -1.88e-3 | -1.92e-3 |
| 153 | 3.00e-3 | 8 | 2.02e-3 | 3.56e-3 | 2.35e-3 | 2.44e-3 | 38 | -1.65e-2 | +6.88e-3 | -8.14e-4 | -1.04e-3 |
| 154 | 3.00e-3 | 6 | 2.25e-3 | 3.38e-3 | 2.55e-3 | 2.33e-3 | 38 | -8.54e-3 | +4.16e-3 | -9.77e-4 | -9.90e-4 |
| 155 | 3.00e-3 | 6 | 2.67e-3 | 3.75e-3 | 2.91e-3 | 2.67e-3 | 43 | -5.16e-3 | +5.22e-3 | -2.69e-4 | -6.90e-4 |
| 156 | 3.00e-3 | 6 | 2.33e-3 | 3.41e-3 | 2.59e-3 | 2.33e-3 | 40 | -8.30e-3 | +3.13e-3 | -1.06e-3 | -8.42e-4 |
| 157 | 3.00e-3 | 6 | 2.51e-3 | 3.21e-3 | 2.72e-3 | 2.61e-3 | 45 | -5.06e-3 | +4.03e-3 | -2.23e-4 | -5.71e-4 |
| 158 | 3.00e-3 | 8 | 2.33e-3 | 3.44e-3 | 2.62e-3 | 2.39e-3 | 43 | -6.89e-3 | +3.48e-3 | -6.94e-4 | -6.29e-4 |
| 159 | 3.00e-3 | 4 | 2.40e-3 | 3.36e-3 | 2.81e-3 | 2.74e-3 | 43 | -4.73e-3 | +4.25e-3 | -1.15e-4 | -4.53e-4 |
| 160 | 3.00e-3 | 7 | 2.27e-3 | 3.28e-3 | 2.48e-3 | 2.33e-3 | 38 | -8.87e-3 | +2.58e-3 | -9.83e-4 | -6.41e-4 |
| 161 | 3.00e-3 | 7 | 2.25e-3 | 3.34e-3 | 2.50e-3 | 2.42e-3 | 36 | -9.04e-3 | +4.60e-3 | -7.00e-4 | -5.82e-4 |
| 162 | 3.00e-3 | 7 | 2.13e-3 | 3.60e-3 | 2.49e-3 | 2.13e-3 | 30 | -1.27e-2 | +5.21e-3 | -1.50e-3 | -1.06e-3 |
| 163 | 3.00e-3 | 7 | 2.26e-3 | 3.43e-3 | 2.46e-3 | 2.33e-3 | 37 | -1.33e-2 | +5.51e-3 | -7.40e-4 | -8.92e-4 |
| 164 | 3.00e-3 | 7 | 2.37e-3 | 3.51e-3 | 2.65e-3 | 2.55e-3 | 40 | -1.03e-2 | +5.53e-3 | -5.35e-4 | -6.64e-4 |
| 165 | 3.00e-3 | 7 | 2.20e-3 | 3.49e-3 | 2.63e-3 | 2.46e-3 | 31 | -4.74e-3 | +3.98e-3 | -7.20e-4 | -6.33e-4 |
| 166 | 3.00e-3 | 9 | 2.19e-3 | 3.59e-3 | 2.57e-3 | 2.31e-3 | 32 | -1.39e-2 | +5.20e-3 | -1.02e-3 | -8.14e-4 |
| 167 | 3.00e-3 | 5 | 2.20e-3 | 3.26e-3 | 2.47e-3 | 2.27e-3 | 34 | -1.03e-2 | +4.71e-3 | -1.32e-3 | -9.98e-4 |
| 168 | 3.00e-3 | 8 | 2.07e-3 | 3.41e-3 | 2.41e-3 | 2.77e-3 | 49 | -1.60e-2 | +5.50e-3 | -7.77e-4 | -5.67e-4 |
| 169 | 3.00e-3 | 1 | 2.98e-3 | 2.98e-3 | 2.98e-3 | 2.98e-3 | 53 | +1.44e-3 | +1.44e-3 | +1.44e-3 | -3.66e-4 |
| 170 | 3.00e-3 | 1 | 7.11e-3 | 7.11e-3 | 7.11e-3 | 7.11e-3 | 363 | +2.39e-3 | +2.39e-3 | +2.39e-3 | -9.01e-5 |
| 171 | 3.00e-3 | 1 | 7.38e-3 | 7.38e-3 | 7.38e-3 | 7.38e-3 | 325 | +1.16e-4 | +1.16e-4 | +1.16e-4 | -6.95e-5 |
| 172 | 3.00e-3 | 1 | 7.05e-3 | 7.05e-3 | 7.05e-3 | 7.05e-3 | 288 | -1.62e-4 | -1.62e-4 | -1.62e-4 | -7.88e-5 |
| 173 | 3.00e-3 | 1 | 7.16e-3 | 7.16e-3 | 7.16e-3 | 7.16e-3 | 304 | +5.03e-5 | +5.03e-5 | +5.03e-5 | -6.59e-5 |
| 174 | 3.00e-3 | 1 | 6.82e-3 | 6.82e-3 | 6.82e-3 | 6.82e-3 | 279 | -1.74e-4 | -1.74e-4 | -1.74e-4 | -7.67e-5 |
| 175 | 3.00e-3 | 1 | 6.96e-3 | 6.96e-3 | 6.96e-3 | 6.96e-3 | 267 | +7.68e-5 | +7.68e-5 | +7.68e-5 | -6.13e-5 |
| 176 | 3.00e-3 | 1 | 6.96e-3 | 6.96e-3 | 6.96e-3 | 6.96e-3 | 277 | +1.42e-6 | +1.42e-6 | +1.42e-6 | -5.50e-5 |
| 177 | 3.00e-3 | 1 | 7.21e-3 | 7.21e-3 | 7.21e-3 | 7.21e-3 | 287 | +1.23e-4 | +1.23e-4 | +1.23e-4 | -3.72e-5 |
| 178 | 3.00e-3 | 1 | 7.17e-3 | 7.17e-3 | 7.17e-3 | 7.17e-3 | 284 | -1.86e-5 | -1.86e-5 | -1.86e-5 | -3.53e-5 |
| 180 | 3.00e-3 | 2 | 7.14e-3 | 8.13e-3 | 7.64e-3 | 7.14e-3 | 281 | -4.61e-4 | +3.19e-4 | -7.11e-5 | -4.60e-5 |
| 182 | 3.00e-3 | 1 | 7.41e-3 | 7.41e-3 | 7.41e-3 | 7.41e-3 | 307 | +1.19e-4 | +1.19e-4 | +1.19e-4 | -2.95e-5 |
| 183 | 3.00e-3 | 1 | 7.18e-3 | 7.18e-3 | 7.18e-3 | 7.18e-3 | 262 | -1.17e-4 | -1.17e-4 | -1.17e-4 | -3.83e-5 |
| 184 | 3.00e-3 | 1 | 7.53e-3 | 7.53e-3 | 7.53e-3 | 7.53e-3 | 304 | +1.53e-4 | +1.53e-4 | +1.53e-4 | -1.92e-5 |
| 185 | 3.00e-3 | 1 | 7.24e-3 | 7.24e-3 | 7.24e-3 | 7.24e-3 | 279 | -1.42e-4 | -1.42e-4 | -1.42e-4 | -3.14e-5 |
| 186 | 3.00e-3 | 1 | 7.33e-3 | 7.33e-3 | 7.33e-3 | 7.33e-3 | 312 | +4.37e-5 | +4.37e-5 | +4.37e-5 | -2.39e-5 |
| 187 | 3.00e-3 | 1 | 7.12e-3 | 7.12e-3 | 7.12e-3 | 7.12e-3 | 285 | -1.04e-4 | -1.04e-4 | -1.04e-4 | -3.19e-5 |
| 188 | 3.00e-3 | 1 | 7.12e-3 | 7.12e-3 | 7.12e-3 | 7.12e-3 | 261 | -9.85e-8 | -9.85e-8 | -9.85e-8 | -2.87e-5 |
| 189 | 3.00e-3 | 1 | 6.88e-3 | 6.88e-3 | 6.88e-3 | 6.88e-3 | 249 | -1.37e-4 | -1.37e-4 | -1.37e-4 | -3.95e-5 |
| 190 | 3.00e-3 | 1 | 7.15e-3 | 7.15e-3 | 7.15e-3 | 7.15e-3 | 267 | +1.43e-4 | +1.43e-4 | +1.43e-4 | -2.13e-5 |
| 191 | 3.00e-3 | 1 | 7.16e-3 | 7.16e-3 | 7.16e-3 | 7.16e-3 | 260 | +3.80e-6 | +3.80e-6 | +3.80e-6 | -1.88e-5 |
| 192 | 3.00e-3 | 1 | 6.92e-3 | 6.92e-3 | 6.92e-3 | 6.92e-3 | 276 | -1.23e-4 | -1.23e-4 | -1.23e-4 | -2.92e-5 |
| 193 | 3.00e-3 | 1 | 6.94e-3 | 6.94e-3 | 6.94e-3 | 6.94e-3 | 258 | +1.37e-5 | +1.37e-5 | +1.37e-5 | -2.49e-5 |
| 194 | 3.00e-3 | 1 | 7.45e-3 | 7.45e-3 | 7.45e-3 | 7.45e-3 | 281 | +2.52e-4 | +2.52e-4 | +2.52e-4 | +2.79e-6 |
| 195 | 3.00e-3 | 1 | 7.78e-3 | 7.78e-3 | 7.78e-3 | 7.78e-3 | 295 | +1.47e-4 | +1.47e-4 | +1.47e-4 | +1.72e-5 |
| 196 | 3.00e-3 | 1 | 7.17e-3 | 7.17e-3 | 7.17e-3 | 7.17e-3 | 277 | -2.93e-4 | -2.93e-4 | -2.93e-4 | -1.39e-5 |
| 197 | 3.00e-3 | 1 | 7.30e-3 | 7.30e-3 | 7.30e-3 | 7.30e-3 | 290 | +5.99e-5 | +5.99e-5 | +5.99e-5 | -6.47e-6 |
| 198 | 3.00e-3 | 1 | 7.50e-3 | 7.50e-3 | 7.50e-3 | 7.50e-3 | 267 | +1.03e-4 | +1.03e-4 | +1.03e-4 | +4.51e-6 |
| 199 | 3.00e-3 | 1 | 7.11e-3 | 7.11e-3 | 7.11e-3 | 7.11e-3 | 258 | -2.07e-4 | -2.07e-4 | -2.07e-4 | -1.66e-5 |

