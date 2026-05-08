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
| cpu-async | 0.064709 | 0.9204 | +0.0079 | 1795.4 | 470 | 82.7 | 100% | 100% | 100% | 12.9 |

## Best Mode per Model

| Model | Best Eval | Mode | Fastest (within 2% of solo-0) | Mode |
|-------|-----------|------|-------------------------------|------|
| resnet-graph | 0.9204 | cpu-async | - | - |

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
| cpu-async | 1.9984 | 0.7481 | 0.6138 | 0.5599 | 0.5297 | 0.5116 | 0.4980 | 0.4908 | 0.4748 | 0.4758 | 0.2164 | 0.1821 | 0.1612 | 0.1467 | 0.1332 | 0.0824 | 0.0755 | 0.0728 | 0.0676 | 0.0647 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | cpu-async | 0 | cuda:0 | 0.4013 | 2.8 |
| resnet-graph | cpu-async | 1 | cuda:1 | 0.3045 | 3.7 |
| resnet-graph | cpu-async | 2 | cuda:2 | 0.2942 | 3.6 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) | GPU2 Peak (MB) | GPU2 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|---------------|---------------|
| resnet-graph | cpu-async | 358 | 357 | 392 | 385 | 388 | 383 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| cpu-async | gpu1 | 923.9 | 1.2 | epoch-boundary(103) |
| cpu-async | gpu2 | 924.0 | 1.1 | epoch-boundary(103) |
| cpu-async | gpu1 | 222.8 | 1.1 | epoch-boundary(24) |
| cpu-async | gpu2 | 222.7 | 1.0 | epoch-boundary(24) |
| cpu-async | gpu1 | 1713.3 | 0.9 | epoch-boundary(190) |
| cpu-async | gpu2 | 1713.3 | 0.8 | epoch-boundary(190) |
| cpu-async | gpu1 | 1067.6 | 0.8 | epoch-boundary(119) |
| cpu-async | gpu1 | 852.6 | 0.7 | epoch-boundary(95) |
| cpu-async | gpu0 | 1747.9 | 0.7 | epoch-boundary(194) |
| cpu-async | gpu1 | 666.2 | 0.6 | epoch-boundary(74) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | cpu-async | gpu0 | 0.7s | 0.0s | 0.0s | 0.0s | 0.7s |
| resnet-graph | cpu-async | gpu1 | 6.4s | 0.0s | 0.0s | 0.0s | 7.0s |
| resnet-graph | cpu-async | gpu2 | 4.7s | 0.0s | 0.0s | 0.0s | 5.2s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | cpu-async | 298 | 0 | 470 | 82.7 | 2347/10246 | 470 | 82.7 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | cpu-async | 199.1 | 11.1% |

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
| resnet-graph | cpu-async | 182 | 470 | 0 | 7.52e-3 | -6.65e-5 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | cpu-async | 0 | 470 | 7.03e-2 | 7.76e-2 | 1.87e-3 | 4.18e-1 | 30.2 | -2.35e-4 | 1.58e-3 |
| resnet-graph | cpu-async | 1 | 470 | 7.09e-2 | 7.84e-2 | 1.96e-3 | 4.11e-1 | 34.3 | -2.42e-4 | 1.53e-3 |
| resnet-graph | cpu-async | 2 | 470 | 7.07e-2 | 7.79e-2 | 1.91e-3 | 4.04e-1 | 35.5 | -2.44e-4 | 1.55e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | cpu-async | rank0 ↔ rank1 | +0.9973 |
| resnet-graph | cpu-async | rank0 ↔ rank2 | +0.9977 |
| resnet-graph | cpu-async | rank1 ↔ rank2 | +0.9973 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | cpu-async | 37 (1,2,3,4,5,6,7,8…146,148) | 0 (—) | — | 1,2,3,4,5,6,7,8…146,148 | — |

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
| resnet-graph | cpu-async | 3.00e-1 | 0–98 | 182 | +0.177 |
| resnet-graph | cpu-async | 3.00e-2 | 99–149 | 110 | +0.062 |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | 174 | +0.146 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | cpu-async | 468 | +0.037 | 181 | +0.200 | +0.284 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | cpu-async | 469 | 3.36e1–7.94e1 | 6.26e1 | 2.59e-3 | 4.11e-3 | 3.20e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | cpu-async | 3.00e-1 | 0–98 | 184 | 36–76893 | +1.024e-5 | 0.512 | +1.044e-5 | 0.529 | 86 | +6.288e-6 | 0.372 | 36–1083 | +9.263e-4 | 0.792 |
| resnet-graph | cpu-async | 3.00e-1 | 1–98 (post-transient, skipped 1) | 173 | 865–76893 | +1.047e-5 | 0.585 | +1.059e-5 | 0.601 | 85 | +6.121e-6 | 0.354 | 86–1083 | +9.583e-4 | 0.926 |
| resnet-graph | cpu-async | 3.00e-2 | 99–149 | 111 | 77526–117121 | -2.129e-6 | 0.007 | -2.221e-6 | 0.007 | 51 | -1.648e-6 | 0.003 | 109–661 | +2.931e-4 | 0.019 |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | 175 | 117331–155797 | +2.440e-5 | 0.279 | +2.495e-5 | 0.297 | 45 | +3.616e-5 | 0.669 | 75–1041 | +1.416e-3 | 0.755 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | cpu-async | 3.00e-1 | 0–98 | +9.263e-4 | r0: +9.186e-4, r1: +9.367e-4, r2: +9.251e-4 | r0: 0.781, r1: 0.789, r2: 0.786 | 1.02× | ✓ |
| resnet-graph | cpu-async | 3.00e-1 | 1–98 (post-transient, skipped 1) | +9.583e-4 | r0: +9.434e-4, r1: +9.748e-4, r2: +9.577e-4 | r0: 0.925, r1: 0.911, r2: 0.922 | 1.03× | ✓ |
| resnet-graph | cpu-async | 3.00e-2 | 99–149 | +2.931e-4 | r0: +2.850e-4, r1: +2.881e-4, r2: +3.062e-4 | r0: 0.018, r1: 0.018, r2: 0.020 | 1.07× | ✓ |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | +1.416e-3 | r0: +1.393e-3, r1: +1.430e-3, r2: +1.426e-3 | r0: 0.769, r1: 0.762, r2: 0.723 | 1.03× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | cpu-async | `█▇▇███████████████████▅▄▅▅▅▅▅▅▅▅▅▅▃▁▁▁▁▁▁▂▂▂▂▂` | `▁▆▆▆▇▇████████████████▆▇▇███████▇▇▂▅▆▇▇▆▆▇████` |

### Per-Epoch Detail: resnet-graph / cpu-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 11 | 8.65e-2 | 4.18e-1 | 1.51e-1 | 8.65e-2 | 29 | -3.92e-2 | +5.23e-3 | -9.53e-3 | -6.72e-3 |
| 1 | 3.00e-1 | 10 | 8.56e-2 | 1.36e-1 | 9.62e-2 | 1.10e-1 | 38 | -1.09e-2 | +5.62e-3 | -9.57e-5 | -1.47e-3 |
| 2 | 3.00e-1 | 5 | 1.02e-1 | 1.48e-1 | 1.13e-1 | 1.08e-1 | 38 | -9.30e-3 | +3.94e-3 | -9.68e-4 | -1.21e-3 |
| 3 | 3.00e-1 | 6 | 1.05e-1 | 1.56e-1 | 1.17e-1 | 1.13e-1 | 38 | -1.07e-2 | +4.48e-3 | -7.52e-4 | -9.10e-4 |
| 4 | 3.00e-1 | 7 | 1.04e-1 | 1.48e-1 | 1.14e-1 | 1.08e-1 | 35 | -7.18e-3 | +3.44e-3 | -6.69e-4 | -7.27e-4 |
| 5 | 3.00e-1 | 7 | 1.03e-1 | 1.46e-1 | 1.12e-1 | 1.10e-1 | 36 | -8.03e-3 | +4.11e-3 | -6.09e-4 | -5.77e-4 |
| 6 | 3.00e-1 | 6 | 1.10e-1 | 1.55e-1 | 1.26e-1 | 1.23e-1 | 50 | -9.41e-3 | +4.16e-3 | -4.84e-4 | -5.06e-4 |
| 7 | 3.00e-1 | 8 | 1.02e-1 | 1.54e-1 | 1.14e-1 | 1.06e-1 | 36 | -9.33e-3 | +2.94e-3 | -8.69e-4 | -6.26e-4 |
| 8 | 3.00e-1 | 5 | 1.02e-1 | 1.44e-1 | 1.15e-1 | 1.02e-1 | 33 | -8.29e-3 | +4.32e-3 | -1.09e-3 | -8.59e-4 |
| 9 | 3.00e-1 | 8 | 9.73e-2 | 1.44e-1 | 1.10e-1 | 1.07e-1 | 38 | -1.26e-2 | +4.41e-3 | -6.87e-4 | -6.62e-4 |
| 10 | 3.00e-1 | 5 | 1.03e-1 | 1.60e-1 | 1.21e-1 | 1.24e-1 | 46 | -1.05e-2 | +4.52e-3 | -5.99e-4 | -5.55e-4 |
| 11 | 3.00e-1 | 7 | 1.03e-1 | 1.56e-1 | 1.15e-1 | 1.09e-1 | 38 | -7.73e-3 | +2.59e-3 | -8.71e-4 | -6.30e-4 |
| 12 | 3.00e-1 | 8 | 1.05e-1 | 1.55e-1 | 1.16e-1 | 1.12e-1 | 40 | -7.88e-3 | +3.83e-3 | -5.76e-4 | -5.19e-4 |
| 13 | 3.00e-1 | 4 | 9.49e-2 | 1.55e-1 | 1.12e-1 | 1.01e-1 | 37 | -1.41e-2 | +3.53e-3 | -2.54e-3 | -1.17e-3 |
| 14 | 3.00e-1 | 2 | 9.78e-2 | 2.11e-1 | 1.54e-1 | 2.11e-1 | 239 | -9.84e-4 | +3.22e-3 | +1.12e-3 | -7.17e-4 |
| 15 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 268 | +5.76e-5 | +5.76e-5 | +5.76e-5 | -6.40e-4 |
| 17 | 3.00e-1 | 2 | 2.09e-1 | 2.33e-1 | 2.21e-1 | 2.09e-1 | 273 | -4.03e-4 | +2.24e-4 | -8.92e-5 | -5.38e-4 |
| 19 | 3.00e-1 | 1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 346 | +1.58e-4 | +1.58e-4 | +1.58e-4 | -4.69e-4 |
| 20 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 312 | -1.29e-4 | -1.29e-4 | -1.29e-4 | -4.35e-4 |
| 21 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 310 | -1.01e-5 | -1.01e-5 | -1.01e-5 | -3.92e-4 |
| 22 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 273 | -7.51e-5 | -7.51e-5 | -7.51e-5 | -3.60e-4 |
| 23 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 311 | +8.94e-5 | +8.94e-5 | +8.94e-5 | -3.15e-4 |
| 24 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 307 | -9.38e-5 | -9.38e-5 | -9.38e-5 | -2.93e-4 |
| 26 | 3.00e-1 | 2 | 2.02e-1 | 2.29e-1 | 2.15e-1 | 2.02e-1 | 263 | -4.63e-4 | +2.54e-4 | -1.04e-4 | -2.61e-4 |
| 28 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 318 | +2.48e-4 | +2.48e-4 | +2.48e-4 | -2.10e-4 |
| 29 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 289 | -1.03e-4 | -1.03e-4 | -1.03e-4 | -1.99e-4 |
| 30 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 277 | -1.68e-4 | -1.68e-4 | -1.68e-4 | -1.96e-4 |
| 31 | 3.00e-1 | 2 | 2.01e-1 | 2.04e-1 | 2.03e-1 | 2.01e-1 | 237 | -6.54e-5 | +2.56e-5 | -1.99e-5 | -1.63e-4 |
| 33 | 3.00e-1 | 2 | 2.04e-1 | 2.19e-1 | 2.11e-1 | 2.04e-1 | 253 | -2.90e-4 | +2.73e-4 | -8.65e-6 | -1.37e-4 |
| 35 | 3.00e-1 | 2 | 2.11e-1 | 2.12e-1 | 2.11e-1 | 2.11e-1 | 275 | -2.49e-5 | +1.30e-4 | +5.25e-5 | -1.01e-4 |
| 37 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 340 | +9.26e-5 | +9.26e-5 | +9.26e-5 | -8.21e-5 |
| 38 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 275 | -8.89e-5 | -8.89e-5 | -8.89e-5 | -8.28e-5 |
| 39 | 3.00e-1 | 1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 400 | +1.51e-4 | +1.51e-4 | +1.51e-4 | -5.94e-5 |
| 40 | 3.00e-1 | 1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 360 | -3.15e-5 | -3.15e-5 | -3.15e-5 | -5.66e-5 |
| 42 | 3.00e-1 | 1 | 2.33e-1 | 2.33e-1 | 2.33e-1 | 2.33e-1 | 384 | +1.16e-4 | +1.16e-4 | +1.16e-4 | -3.93e-5 |
| 43 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 325 | -2.17e-4 | -2.17e-4 | -2.17e-4 | -5.71e-5 |
| 44 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 331 | +1.78e-5 | +1.78e-5 | +1.78e-5 | -4.96e-5 |
| 45 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 307 | -4.84e-5 | -4.84e-5 | -4.84e-5 | -4.95e-5 |
| 46 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 300 | -1.98e-6 | -1.98e-6 | -1.98e-6 | -4.47e-5 |
| 48 | 3.00e-1 | 2 | 2.14e-1 | 2.33e-1 | 2.24e-1 | 2.14e-1 | 298 | -2.74e-4 | +2.14e-4 | -3.03e-5 | -4.44e-5 |
| 50 | 3.00e-1 | 1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 318 | +1.36e-4 | +1.36e-4 | +1.36e-4 | -2.64e-5 |
| 51 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 260 | -3.61e-4 | -3.61e-4 | -3.61e-4 | -5.99e-5 |
| 52 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 285 | +1.87e-4 | +1.87e-4 | +1.87e-4 | -3.51e-5 |
| 53 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 294 | +3.24e-5 | +3.24e-5 | +3.24e-5 | -2.84e-5 |
| 54 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 268 | -9.06e-5 | -9.06e-5 | -9.06e-5 | -3.46e-5 |
| 55 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 247 | -1.32e-4 | -1.32e-4 | -1.32e-4 | -4.43e-5 |
| 56 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 276 | +8.26e-5 | +8.26e-5 | +8.26e-5 | -3.16e-5 |
| 57 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 275 | +6.09e-5 | +6.09e-5 | +6.09e-5 | -2.24e-5 |
| 58 | 3.00e-1 | 2 | 2.10e-1 | 2.12e-1 | 2.11e-1 | 2.12e-1 | 256 | -5.72e-5 | +2.87e-5 | -1.42e-5 | -2.04e-5 |
| 60 | 3.00e-1 | 1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 352 | +1.86e-4 | +1.86e-4 | +1.86e-4 | +2.27e-7 |
| 61 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 267 | -3.48e-4 | -3.48e-4 | -3.48e-4 | -3.46e-5 |
| 62 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 281 | +8.71e-5 | +8.71e-5 | +8.71e-5 | -2.24e-5 |
| 63 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 292 | +6.85e-5 | +6.85e-5 | +6.85e-5 | -1.33e-5 |
| 64 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 282 | -1.89e-5 | -1.89e-5 | -1.89e-5 | -1.39e-5 |
| 65 | 3.00e-1 | 1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 313 | +9.97e-5 | +9.97e-5 | +9.97e-5 | -2.50e-6 |
| 66 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 290 | -6.49e-5 | -6.49e-5 | -6.49e-5 | -8.73e-6 |
| 67 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 262 | -1.66e-4 | -1.66e-4 | -1.66e-4 | -2.45e-5 |
| 68 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 265 | +3.63e-5 | +3.63e-5 | +3.63e-5 | -1.84e-5 |
| 69 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 277 | +1.03e-4 | +1.03e-4 | +1.03e-4 | -6.31e-6 |
| 70 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 228 | -3.60e-4 | -3.60e-4 | -3.60e-4 | -4.17e-5 |
| 71 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 251 | +2.42e-4 | +2.42e-4 | +2.42e-4 | -1.33e-5 |
| 72 | 3.00e-1 | 2 | 2.01e-1 | 2.08e-1 | 2.04e-1 | 2.01e-1 | 219 | -1.60e-4 | -5.88e-5 | -1.09e-4 | -3.20e-5 |
| 74 | 3.00e-1 | 2 | 1.97e-1 | 2.31e-1 | 2.14e-1 | 1.97e-1 | 219 | -7.29e-4 | +4.36e-4 | -1.47e-4 | -5.96e-5 |
| 75 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 262 | +3.12e-4 | +3.12e-4 | +3.12e-4 | -2.24e-5 |
| 76 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 288 | +4.80e-5 | +4.80e-5 | +4.80e-5 | -1.54e-5 |
| 77 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 238 | -3.01e-4 | -3.01e-4 | -3.01e-4 | -4.40e-5 |
| 78 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 227 | -4.77e-5 | -4.77e-5 | -4.77e-5 | -4.43e-5 |
| 79 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 261 | +2.21e-4 | +2.21e-4 | +2.21e-4 | -1.79e-5 |
| 80 | 3.00e-1 | 1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 280 | +1.68e-4 | +1.68e-4 | +1.68e-4 | +7.20e-7 |
| 81 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 241 | -3.37e-4 | -3.37e-4 | -3.37e-4 | -3.31e-5 |
| 82 | 3.00e-1 | 2 | 1.92e-1 | 2.10e-1 | 2.01e-1 | 1.92e-1 | 207 | -4.43e-4 | +1.21e-4 | -1.61e-4 | -6.02e-5 |
| 83 | 3.00e-1 | 1 | 1.91e-1 | 1.91e-1 | 1.91e-1 | 1.91e-1 | 201 | -1.29e-5 | -1.29e-5 | -1.29e-5 | -5.55e-5 |
| 84 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 230 | +2.35e-4 | +2.35e-4 | +2.35e-4 | -2.64e-5 |
| 85 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 236 | +7.19e-5 | +7.19e-5 | +7.19e-5 | -1.66e-5 |
| 86 | 3.00e-1 | 2 | 1.97e-1 | 2.15e-1 | 2.06e-1 | 1.97e-1 | 212 | -4.28e-4 | +1.70e-4 | -1.29e-4 | -4.09e-5 |
| 87 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 278 | +3.75e-4 | +3.75e-4 | +3.75e-4 | +7.43e-7 |
| 88 | 3.00e-1 | 1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 202 | -5.77e-4 | -5.77e-4 | -5.77e-4 | -5.71e-5 |
| 89 | 3.00e-1 | 1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 210 | +2.37e-5 | +2.37e-5 | +2.37e-5 | -4.90e-5 |
| 90 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 227 | +2.42e-4 | +2.42e-4 | +2.42e-4 | -1.99e-5 |
| 91 | 3.00e-1 | 2 | 1.89e-1 | 2.02e-1 | 1.96e-1 | 1.89e-1 | 191 | -3.42e-4 | -9.07e-5 | -2.16e-4 | -5.85e-5 |
| 92 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 229 | +2.92e-4 | +2.92e-4 | +2.92e-4 | -2.34e-5 |
| 93 | 3.00e-1 | 2 | 1.99e-1 | 2.03e-1 | 2.01e-1 | 1.99e-1 | 214 | -1.05e-4 | +2.03e-5 | -4.24e-5 | -2.76e-5 |
| 95 | 3.00e-1 | 2 | 1.98e-1 | 2.17e-1 | 2.08e-1 | 1.98e-1 | 214 | -4.27e-4 | +3.29e-4 | -4.91e-5 | -3.55e-5 |
| 96 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 249 | +1.58e-4 | +1.58e-4 | +1.58e-4 | -1.61e-5 |
| 97 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 229 | -2.28e-5 | -2.28e-5 | -2.28e-5 | -1.68e-5 |
| 98 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 241 | +7.78e-6 | +7.78e-6 | +7.78e-6 | -1.43e-5 |
| 99 | 3.00e-2 | 2 | 1.89e-1 | 2.02e-1 | 1.95e-1 | 1.89e-1 | 190 | -3.47e-4 | -8.25e-5 | -2.15e-4 | -5.37e-5 |
| 100 | 3.00e-2 | 1 | 1.31e-1 | 1.31e-1 | 1.31e-1 | 1.31e-1 | 221 | -1.67e-3 | -1.67e-3 | -1.67e-3 | -2.15e-4 |
| 101 | 3.00e-2 | 1 | 4.40e-2 | 4.40e-2 | 4.40e-2 | 4.40e-2 | 238 | -4.58e-3 | -4.58e-3 | -4.58e-3 | -6.51e-4 |
| 102 | 3.00e-2 | 1 | 2.62e-2 | 2.62e-2 | 2.62e-2 | 2.62e-2 | 240 | -2.15e-3 | -2.15e-3 | -2.15e-3 | -8.01e-4 |
| 103 | 3.00e-2 | 2 | 2.28e-2 | 2.45e-2 | 2.37e-2 | 2.28e-2 | 190 | -3.82e-4 | -2.88e-4 | -3.35e-4 | -7.13e-4 |
| 104 | 3.00e-2 | 1 | 2.48e-2 | 2.48e-2 | 2.48e-2 | 2.48e-2 | 211 | +3.93e-4 | +3.93e-4 | +3.93e-4 | -6.02e-4 |
| 105 | 3.00e-2 | 1 | 2.42e-2 | 2.42e-2 | 2.42e-2 | 2.42e-2 | 194 | -1.18e-4 | -1.18e-4 | -1.18e-4 | -5.54e-4 |
| 106 | 3.00e-2 | 2 | 2.40e-2 | 2.60e-2 | 2.50e-2 | 2.60e-2 | 206 | -5.02e-5 | +3.82e-4 | +1.66e-4 | -4.15e-4 |
| 107 | 3.00e-2 | 1 | 2.52e-2 | 2.52e-2 | 2.52e-2 | 2.52e-2 | 184 | -1.69e-4 | -1.69e-4 | -1.69e-4 | -3.90e-4 |
| 108 | 3.00e-2 | 2 | 2.58e-2 | 2.61e-2 | 2.59e-2 | 2.58e-2 | 175 | -7.13e-5 | +1.86e-4 | +5.75e-5 | -3.07e-4 |
| 109 | 3.00e-2 | 1 | 2.75e-2 | 2.75e-2 | 2.75e-2 | 2.75e-2 | 202 | +3.23e-4 | +3.23e-4 | +3.23e-4 | -2.44e-4 |
| 110 | 3.00e-2 | 2 | 2.58e-2 | 2.86e-2 | 2.72e-2 | 2.58e-2 | 154 | -6.76e-4 | +1.82e-4 | -2.47e-4 | -2.48e-4 |
| 111 | 3.00e-2 | 2 | 2.55e-2 | 2.64e-2 | 2.59e-2 | 2.55e-2 | 154 | -2.18e-4 | +1.37e-4 | -4.02e-5 | -2.11e-4 |
| 112 | 3.00e-2 | 2 | 2.73e-2 | 2.87e-2 | 2.80e-2 | 2.73e-2 | 154 | -3.09e-4 | +6.39e-4 | +1.65e-4 | -1.44e-4 |
| 113 | 3.00e-2 | 1 | 3.01e-2 | 3.01e-2 | 3.01e-2 | 3.01e-2 | 193 | +5.07e-4 | +5.07e-4 | +5.07e-4 | -7.89e-5 |
| 114 | 3.00e-2 | 1 | 3.14e-2 | 3.14e-2 | 3.14e-2 | 3.14e-2 | 211 | +1.88e-4 | +1.88e-4 | +1.88e-4 | -5.22e-5 |
| 115 | 3.00e-2 | 2 | 2.92e-2 | 3.08e-2 | 3.00e-2 | 2.92e-2 | 152 | -3.54e-4 | -8.63e-5 | -2.20e-4 | -8.54e-5 |
| 116 | 3.00e-2 | 2 | 2.99e-2 | 3.00e-2 | 3.00e-2 | 3.00e-2 | 160 | +1.07e-5 | +1.32e-4 | +7.15e-5 | -5.62e-5 |
| 117 | 3.00e-2 | 1 | 3.33e-2 | 3.33e-2 | 3.33e-2 | 3.33e-2 | 193 | +5.37e-4 | +5.37e-4 | +5.37e-4 | +3.09e-6 |
| 118 | 3.00e-2 | 2 | 2.91e-2 | 3.32e-2 | 3.11e-2 | 2.91e-2 | 144 | -9.04e-4 | -1.50e-5 | -4.60e-4 | -8.93e-5 |
| 119 | 3.00e-2 | 2 | 3.04e-2 | 3.09e-2 | 3.07e-2 | 3.04e-2 | 144 | -1.21e-4 | +3.73e-4 | +1.26e-4 | -5.09e-5 |
| 120 | 3.00e-2 | 1 | 3.21e-2 | 3.21e-2 | 3.21e-2 | 3.21e-2 | 174 | +3.03e-4 | +3.03e-4 | +3.03e-4 | -1.55e-5 |
| 121 | 3.00e-2 | 2 | 3.21e-2 | 3.55e-2 | 3.38e-2 | 3.21e-2 | 157 | -6.43e-4 | +4.85e-4 | -7.89e-5 | -3.32e-5 |
| 122 | 3.00e-2 | 1 | 3.56e-2 | 3.56e-2 | 3.56e-2 | 3.56e-2 | 192 | +5.40e-4 | +5.40e-4 | +5.40e-4 | +2.41e-5 |
| 123 | 3.00e-2 | 3 | 3.22e-2 | 3.63e-2 | 3.42e-2 | 3.22e-2 | 136 | -4.07e-4 | +1.14e-4 | -2.32e-4 | -5.01e-5 |
| 124 | 3.00e-2 | 1 | 3.42e-2 | 3.42e-2 | 3.42e-2 | 3.42e-2 | 157 | +3.80e-4 | +3.80e-4 | +3.80e-4 | -7.11e-6 |
| 125 | 3.00e-2 | 2 | 3.25e-2 | 3.71e-2 | 3.48e-2 | 3.25e-2 | 127 | -1.03e-3 | +4.58e-4 | -2.87e-4 | -6.78e-5 |
| 126 | 3.00e-2 | 2 | 3.15e-2 | 3.41e-2 | 3.28e-2 | 3.15e-2 | 118 | -6.66e-4 | +3.16e-4 | -1.75e-4 | -9.31e-5 |
| 127 | 3.00e-2 | 2 | 3.28e-2 | 3.42e-2 | 3.35e-2 | 3.28e-2 | 118 | -3.50e-4 | +5.46e-4 | +9.81e-5 | -6.13e-5 |
| 128 | 3.00e-2 | 2 | 3.28e-2 | 3.66e-2 | 3.47e-2 | 3.28e-2 | 118 | -9.31e-4 | +7.19e-4 | -1.06e-4 | -7.80e-5 |
| 129 | 3.00e-2 | 3 | 3.31e-2 | 3.93e-2 | 3.53e-2 | 3.31e-2 | 112 | -1.35e-3 | +1.10e-3 | -1.31e-4 | -1.04e-4 |
| 130 | 3.00e-2 | 1 | 3.66e-2 | 3.66e-2 | 3.66e-2 | 3.66e-2 | 149 | +6.72e-4 | +6.72e-4 | +6.72e-4 | -2.61e-5 |
| 131 | 3.00e-2 | 3 | 3.20e-2 | 3.95e-2 | 3.45e-2 | 3.20e-2 | 106 | -1.92e-3 | +4.70e-4 | -4.94e-4 | -1.57e-4 |
| 132 | 3.00e-2 | 2 | 3.44e-2 | 3.54e-2 | 3.49e-2 | 3.44e-2 | 114 | -2.62e-4 | +8.06e-4 | +2.72e-4 | -8.07e-5 |
| 133 | 3.00e-2 | 2 | 3.45e-2 | 3.88e-2 | 3.66e-2 | 3.45e-2 | 114 | -1.02e-3 | +7.87e-4 | -1.19e-4 | -9.71e-5 |
| 134 | 3.00e-2 | 2 | 3.66e-2 | 3.90e-2 | 3.78e-2 | 3.66e-2 | 120 | -5.21e-4 | +8.37e-4 | +1.58e-4 | -5.54e-5 |
| 135 | 3.00e-2 | 3 | 3.46e-2 | 3.93e-2 | 3.65e-2 | 3.46e-2 | 112 | -8.18e-4 | +5.14e-4 | -2.07e-4 | -1.04e-4 |
| 136 | 3.00e-2 | 2 | 3.56e-2 | 3.85e-2 | 3.71e-2 | 3.56e-2 | 105 | -7.40e-4 | +8.20e-4 | +4.02e-5 | -8.44e-5 |
| 137 | 3.00e-2 | 3 | 3.36e-2 | 4.00e-2 | 3.65e-2 | 3.36e-2 | 99 | -1.06e-3 | +8.23e-4 | -2.91e-4 | -1.54e-4 |
| 138 | 3.00e-2 | 2 | 3.64e-2 | 4.18e-2 | 3.91e-2 | 3.64e-2 | 105 | -1.31e-3 | +1.42e-3 | +5.51e-5 | -1.28e-4 |
| 139 | 3.00e-2 | 3 | 3.60e-2 | 3.89e-2 | 3.70e-2 | 3.61e-2 | 94 | -7.88e-4 | +5.21e-4 | -8.08e-5 | -1.19e-4 |
| 140 | 3.00e-2 | 2 | 3.58e-2 | 4.30e-2 | 3.94e-2 | 3.58e-2 | 94 | -1.96e-3 | +1.25e-3 | -3.58e-4 | -1.81e-4 |
| 141 | 3.00e-2 | 3 | 3.48e-2 | 4.15e-2 | 3.79e-2 | 3.48e-2 | 94 | -1.14e-3 | +1.15e-3 | -2.43e-4 | -2.15e-4 |
| 142 | 3.00e-2 | 3 | 3.54e-2 | 4.22e-2 | 3.84e-2 | 3.54e-2 | 87 | -1.28e-3 | +1.49e-3 | -1.38e-4 | -2.14e-4 |
| 143 | 3.00e-2 | 3 | 3.46e-2 | 4.01e-2 | 3.70e-2 | 3.46e-2 | 82 | -1.15e-3 | +1.04e-3 | -2.27e-4 | -2.32e-4 |
| 144 | 3.00e-2 | 3 | 3.32e-2 | 4.22e-2 | 3.64e-2 | 3.32e-2 | 75 | -2.98e-3 | +1.49e-3 | -5.69e-4 | -3.38e-4 |
| 145 | 3.00e-2 | 4 | 2.94e-2 | 4.08e-2 | 3.33e-2 | 3.16e-2 | 59 | -4.29e-3 | +1.74e-3 | -6.40e-4 | -4.32e-4 |
| 146 | 3.00e-2 | 5 | 3.04e-2 | 3.72e-2 | 3.21e-2 | 3.04e-2 | 62 | -2.81e-3 | +1.84e-3 | -3.14e-4 | -3.90e-4 |
| 147 | 3.00e-2 | 3 | 3.05e-2 | 3.96e-2 | 3.36e-2 | 3.08e-2 | 58 | -4.50e-3 | +2.59e-3 | -5.88e-4 | -4.65e-4 |
| 148 | 3.00e-2 | 8 | 2.53e-2 | 3.88e-2 | 3.04e-2 | 2.53e-2 | 40 | -3.71e-3 | +2.40e-3 | -7.61e-4 | -6.88e-4 |
| 149 | 3.00e-2 | 3 | 2.61e-2 | 3.54e-2 | 2.97e-2 | 2.61e-2 | 40 | -6.80e-3 | +4.00e-3 | -1.39e-3 | -9.27e-4 |
| 150 | 3.00e-3 | 6 | 2.36e-3 | 2.92e-2 | 7.88e-3 | 2.49e-3 | 41 | -3.09e-2 | +1.33e-3 | -9.80e-3 | -4.60e-3 |
| 151 | 3.00e-3 | 7 | 2.31e-3 | 3.45e-3 | 2.68e-3 | 2.86e-3 | 49 | -7.08e-3 | +4.10e-3 | -2.42e-4 | -2.21e-3 |
| 152 | 3.00e-3 | 4 | 2.71e-3 | 3.45e-3 | 2.94e-3 | 2.80e-3 | 49 | -4.24e-3 | +2.06e-3 | -5.43e-4 | -1.64e-3 |
| 153 | 3.00e-3 | 5 | 2.69e-3 | 3.85e-3 | 3.09e-3 | 3.11e-3 | 53 | -5.57e-3 | +3.18e-3 | -2.56e-4 | -1.03e-3 |
| 154 | 3.00e-3 | 5 | 2.64e-3 | 3.76e-3 | 2.96e-3 | 2.64e-3 | 43 | -5.97e-3 | +2.00e-3 | -1.06e-3 | -1.05e-3 |
| 155 | 3.00e-3 | 7 | 2.12e-3 | 3.76e-3 | 2.64e-3 | 2.12e-3 | 37 | -4.57e-3 | +3.99e-3 | -1.38e-3 | -1.27e-3 |
| 156 | 3.00e-3 | 8 | 2.33e-3 | 3.10e-3 | 2.53e-3 | 2.40e-3 | 34 | -7.30e-3 | +5.61e-3 | -1.41e-4 | -6.64e-4 |
| 157 | 3.00e-3 | 5 | 2.21e-3 | 3.07e-3 | 2.50e-3 | 2.57e-3 | 42 | -8.59e-3 | +3.45e-3 | -5.73e-4 | -5.48e-4 |
| 158 | 3.00e-3 | 6 | 2.47e-3 | 3.51e-3 | 2.71e-3 | 2.61e-3 | 39 | -9.48e-3 | +3.63e-3 | -7.34e-4 | -5.77e-4 |
| 159 | 3.00e-3 | 7 | 2.34e-3 | 3.60e-3 | 2.63e-3 | 2.34e-3 | 33 | -8.97e-3 | +4.21e-3 | -1.12e-3 | -8.29e-4 |
| 160 | 3.00e-3 | 5 | 2.43e-3 | 3.25e-3 | 2.95e-3 | 3.09e-3 | 60 | -8.12e-3 | +4.48e-3 | +7.39e-5 | -4.52e-4 |
| 161 | 3.00e-3 | 4 | 3.30e-3 | 4.10e-3 | 3.57e-3 | 3.30e-3 | 64 | -2.80e-3 | +2.73e-3 | -1.68e-4 | -3.85e-4 |
| 162 | 3.00e-3 | 4 | 2.99e-3 | 4.09e-3 | 3.41e-3 | 2.99e-3 | 55 | -3.81e-3 | +2.08e-3 | -8.55e-4 | -5.78e-4 |
| 163 | 3.00e-3 | 5 | 2.90e-3 | 3.91e-3 | 3.21e-3 | 2.90e-3 | 49 | -3.92e-3 | +2.76e-3 | -5.72e-4 | -6.00e-4 |
| 164 | 3.00e-3 | 6 | 2.81e-3 | 3.87e-3 | 3.05e-3 | 2.83e-3 | 43 | -6.99e-3 | +3.18e-3 | -6.19e-4 | -6.09e-4 |
| 165 | 3.00e-3 | 6 | 2.26e-3 | 3.65e-3 | 2.70e-3 | 2.88e-3 | 46 | -9.49e-3 | +3.15e-3 | -6.07e-4 | -4.52e-4 |
| 166 | 3.00e-3 | 6 | 2.55e-3 | 3.64e-3 | 2.81e-3 | 2.66e-3 | 43 | -6.94e-3 | +2.89e-3 | -7.41e-4 | -5.46e-4 |
| 167 | 3.00e-3 | 5 | 2.69e-3 | 3.67e-3 | 3.04e-3 | 3.06e-3 | 50 | -8.03e-3 | +4.00e-3 | -2.90e-4 | -4.17e-4 |
| 168 | 3.00e-3 | 7 | 2.92e-3 | 3.98e-3 | 3.13e-3 | 3.02e-3 | 47 | -5.93e-3 | +2.97e-3 | -3.67e-4 | -3.74e-4 |
| 169 | 3.00e-3 | 4 | 2.60e-3 | 3.79e-3 | 2.95e-3 | 2.60e-3 | 41 | -7.70e-3 | +2.63e-3 | -1.73e-3 | -8.43e-4 |
| 170 | 3.00e-3 | 8 | 2.46e-3 | 3.81e-3 | 2.79e-3 | 2.99e-3 | 48 | -8.91e-3 | +4.70e-3 | -2.10e-4 | -3.38e-4 |
| 171 | 3.00e-3 | 5 | 2.53e-3 | 3.66e-3 | 2.88e-3 | 2.66e-3 | 39 | -6.26e-3 | +2.63e-3 | -1.03e-3 | -6.00e-4 |
| 172 | 3.00e-3 | 8 | 2.56e-3 | 3.74e-3 | 2.84e-3 | 2.71e-3 | 41 | -7.44e-3 | +4.46e-3 | -4.52e-4 | -4.73e-4 |
| 173 | 3.00e-3 | 5 | 2.37e-3 | 3.81e-3 | 2.75e-3 | 2.37e-3 | 37 | -1.05e-2 | +4.79e-3 | -1.60e-3 | -9.37e-4 |
| 174 | 3.00e-3 | 11 | 2.16e-3 | 3.68e-3 | 2.47e-3 | 2.25e-3 | 31 | -1.24e-2 | +5.09e-3 | -8.93e-4 | -7.58e-4 |
| 175 | 3.00e-3 | 4 | 2.28e-3 | 3.56e-3 | 2.73e-3 | 2.69e-3 | 43 | -1.39e-2 | +5.96e-3 | -1.40e-3 | -9.40e-4 |
| 176 | 3.00e-3 | 1 | 2.90e-3 | 2.90e-3 | 2.90e-3 | 2.90e-3 | 43 | +1.77e-3 | +1.77e-3 | +1.77e-3 | -6.69e-4 |
| 177 | 3.00e-3 | 1 | 7.49e-3 | 7.49e-3 | 7.49e-3 | 7.49e-3 | 347 | +2.73e-3 | +2.73e-3 | +2.73e-3 | -3.29e-4 |
| 178 | 3.00e-3 | 1 | 7.49e-3 | 7.49e-3 | 7.49e-3 | 7.49e-3 | 306 | +5.96e-7 | +5.96e-7 | +5.96e-7 | -2.96e-4 |
| 179 | 3.00e-3 | 1 | 7.95e-3 | 7.95e-3 | 7.95e-3 | 7.95e-3 | 329 | +1.82e-4 | +1.82e-4 | +1.82e-4 | -2.49e-4 |
| 180 | 3.00e-3 | 1 | 7.74e-3 | 7.74e-3 | 7.74e-3 | 7.74e-3 | 348 | -7.52e-5 | -7.52e-5 | -7.52e-5 | -2.31e-4 |
| 182 | 3.00e-3 | 2 | 7.52e-3 | 8.27e-3 | 7.89e-3 | 7.52e-3 | 288 | -3.30e-4 | +1.66e-4 | -8.19e-5 | -2.05e-4 |
| 184 | 3.00e-3 | 1 | 7.94e-3 | 7.94e-3 | 7.94e-3 | 7.94e-3 | 349 | +1.58e-4 | +1.58e-4 | +1.58e-4 | -1.69e-4 |
| 185 | 3.00e-3 | 1 | 7.92e-3 | 7.92e-3 | 7.92e-3 | 7.92e-3 | 330 | -7.53e-6 | -7.53e-6 | -7.53e-6 | -1.53e-4 |
| 186 | 3.00e-3 | 1 | 7.81e-3 | 7.81e-3 | 7.81e-3 | 7.81e-3 | 305 | -4.78e-5 | -4.78e-5 | -4.78e-5 | -1.42e-4 |
| 187 | 3.00e-3 | 1 | 7.88e-3 | 7.88e-3 | 7.88e-3 | 7.88e-3 | 313 | +2.96e-5 | +2.96e-5 | +2.96e-5 | -1.25e-4 |
| 189 | 3.00e-3 | 2 | 7.56e-3 | 8.33e-3 | 7.95e-3 | 7.56e-3 | 266 | -3.67e-4 | +1.52e-4 | -1.08e-4 | -1.24e-4 |
| 191 | 3.00e-3 | 2 | 7.23e-3 | 8.22e-3 | 7.72e-3 | 7.23e-3 | 243 | -5.29e-4 | +2.35e-4 | -1.47e-4 | -1.33e-4 |
| 193 | 3.00e-3 | 1 | 7.97e-3 | 7.97e-3 | 7.97e-3 | 7.97e-3 | 312 | +3.12e-4 | +3.12e-4 | +3.12e-4 | -8.81e-5 |
| 194 | 3.00e-3 | 1 | 7.23e-3 | 7.23e-3 | 7.23e-3 | 7.23e-3 | 260 | -3.71e-4 | -3.71e-4 | -3.71e-4 | -1.16e-4 |
| 195 | 3.00e-3 | 1 | 7.69e-3 | 7.69e-3 | 7.69e-3 | 7.69e-3 | 283 | +2.15e-4 | +2.15e-4 | +2.15e-4 | -8.34e-5 |
| 196 | 3.00e-3 | 1 | 8.12e-3 | 8.12e-3 | 8.12e-3 | 8.12e-3 | 303 | +1.80e-4 | +1.80e-4 | +1.80e-4 | -5.70e-5 |
| 197 | 3.00e-3 | 1 | 7.42e-3 | 7.42e-3 | 7.42e-3 | 7.42e-3 | 279 | -3.23e-4 | -3.23e-4 | -3.23e-4 | -8.36e-5 |
| 198 | 3.00e-3 | 1 | 7.83e-3 | 7.83e-3 | 7.83e-3 | 7.83e-3 | 293 | +1.83e-4 | +1.83e-4 | +1.83e-4 | -5.69e-5 |
| 199 | 3.00e-3 | 1 | 7.52e-3 | 7.52e-3 | 7.52e-3 | 7.52e-3 | 267 | -1.53e-4 | -1.53e-4 | -1.53e-4 | -6.65e-5 |

