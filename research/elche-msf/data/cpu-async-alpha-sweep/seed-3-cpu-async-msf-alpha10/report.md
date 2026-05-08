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
| cpu-async | 0.052263 | 0.9186 | +0.0061 | 1815.4 | 430 | 79.9 | 100% | 100% | 100% | 9.9 |

## Best Mode per Model

| Model | Best Eval | Mode | Fastest (within 2% of solo-0) | Mode |
|-------|-----------|------|-------------------------------|------|
| resnet-graph | 0.9186 | cpu-async | - | - |

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
| cpu-async | 2.0085 | 0.8257 | 0.6297 | 0.5733 | 0.5419 | 0.5191 | 0.5061 | 0.4957 | 0.4624 | 0.4522 | 0.2012 | 0.1590 | 0.1341 | 0.1199 | 0.1392 | 0.0702 | 0.0623 | 0.0576 | 0.0556 | 0.0523 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | cpu-async | 0 | cuda:0 | 0.3993 | 2.6 |
| resnet-graph | cpu-async | 1 | cuda:1 | 0.3019 | 3.7 |
| resnet-graph | cpu-async | 2 | cuda:2 | 0.2988 | 3.6 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) | GPU2 Peak (MB) | GPU2 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|---------------|---------------|
| resnet-graph | cpu-async | 358 | 357 | 394 | 385 | 392 | 383 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| cpu-async | gpu1 | 1436.1 | 1.3 | epoch-boundary(158) |
| cpu-async | gpu2 | 1318.5 | 1.2 | epoch-boundary(145) |
| cpu-async | gpu2 | 1436.2 | 1.2 | epoch-boundary(158) |
| cpu-async | gpu1 | 1318.6 | 1.1 | epoch-boundary(145) |
| cpu-async | gpu1 | 1574.9 | 0.8 | epoch-boundary(173) |
| cpu-async | gpu2 | 1670.3 | 0.7 | epoch-boundary(183) |
| cpu-async | gpu2 | 1300.5 | 0.7 | epoch-boundary(143) |
| cpu-async | gpu1 | 523.4 | 0.6 | epoch-boundary(57) |
| cpu-async | gpu1 | 1282.2 | 0.6 | epoch-boundary(141) |
| cpu-async | gpu2 | 1282.2 | 0.6 | epoch-boundary(141) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | cpu-async | gpu1 | 5.0s | 0.0s | 0.0s | 0.0s | 5.0s |
| resnet-graph | cpu-async | gpu2 | 4.9s | 0.0s | 0.0s | 0.0s | 4.9s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | cpu-async | 282 | 0 | 430 | 79.9 | 2756/9630 | 430 | 79.9 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | cpu-async | 201.8 | 11.1% |

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
| resnet-graph | cpu-async | 189 | 430 | 0 | 6.00e-3 | -4.71e-5 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | cpu-async | 0 | 430 | 7.35e-2 | 7.55e-2 | 5.66e-3 | 4.20e-1 | 24.0 | -2.30e-4 | 1.94e-3 |
| resnet-graph | cpu-async | 1 | 430 | 7.42e-2 | 7.68e-2 | 5.33e-3 | 4.45e-1 | 35.1 | -2.48e-4 | 2.05e-3 |
| resnet-graph | cpu-async | 2 | 430 | 7.46e-2 | 7.75e-2 | 5.42e-3 | 4.66e-1 | 40.9 | -2.51e-4 | 2.02e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | cpu-async | rank0 ↔ rank1 | +0.9985 |
| resnet-graph | cpu-async | rank0 ↔ rank2 | +0.9982 |
| resnet-graph | cpu-async | rank1 ↔ rank2 | +0.9978 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | cpu-async | 55 (0,1,2,3,4,5,6,14…140,142) | 0 (—) | — | 0,1,2,3,4,5,6,14…140,142 | — |

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
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 170 | +0.220 |
| resnet-graph | cpu-async | 3.00e-2 | 100–147 | 203 | +0.096 |
| resnet-graph | cpu-async | 3.00e-3 | 149–198 | 53 | +0.087 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | cpu-async | 428 | +0.007 | 189 | +0.199 | +0.300 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | cpu-async | 429 | 3.39e1–7.97e1 | 6.60e1 | 2.81e-3 | 6.00e-3 | 7.40e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 172 | 36–78013 | +9.178e-6 | 0.395 | +9.306e-6 | 0.400 | 94 | +2.999e-6 | 0.105 | 36–957 | +1.194e-3 | 0.709 |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 160 | 896–78013 | +8.272e-6 | 0.402 | +8.339e-6 | 0.409 | 93 | +2.455e-6 | 0.079 | 76–957 | +1.158e-3 | 0.810 |
| resnet-graph | cpu-async | 3.00e-2 | 100–147 | 204 | 78418–115660 | +1.255e-5 | 0.124 | +1.281e-5 | 0.132 | 47 | +2.130e-5 | 0.321 | 76–950 | +1.529e-3 | 0.493 |
| resnet-graph | cpu-async | 3.00e-3 | 149–198 | 54 | 116563–155657 | -1.268e-5 | 0.108 | -1.289e-5 | 0.113 | 48 | -7.215e-6 | 0.058 | 539–977 | +9.822e-4 | 0.041 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | +1.194e-3 | r0: +1.181e-3, r1: +1.197e-3, r2: +1.204e-3 | r0: 0.709, r1: 0.702, r2: 0.707 | 1.02× | ✓ |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +1.158e-3 | r0: +1.142e-3, r1: +1.158e-3, r2: +1.176e-3 | r0: 0.807, r1: 0.801, r2: 0.808 | 1.03× | ✓ |
| resnet-graph | cpu-async | 3.00e-2 | 100–147 | +1.529e-3 | r0: +1.499e-3, r1: +1.534e-3, r2: +1.553e-3 | r0: 0.478, r1: 0.494, r2: 0.494 | 1.04× | ✓ |
| resnet-graph | cpu-async | 3.00e-3 | 149–198 | +9.822e-4 | r0: +9.835e-4, r1: +9.437e-4, r2: +1.019e-3 | r0: 0.040, r1: 0.037, r2: 0.045 | 1.08× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | cpu-async | `█▇█████████████████████▆▃▄▄▄▄▄▄▄▄▅▆▃▁▁▁▁▁▁▁▁▁▁▁▁` | `▁▆▇████████████████████▆▆▇▇▇▆▆▇▆▆▇█▇▇▇▇█████████` |

### Per-Epoch Detail: resnet-graph / cpu-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 12 | 5.72e-2 | 4.66e-1 | 1.12e-1 | 7.91e-2 | 32 | -6.76e-2 | +1.05e-2 | -1.15e-2 | -6.38e-3 |
| 1 | 3.00e-1 | 8 | 6.25e-2 | 1.13e-1 | 7.69e-2 | 7.16e-2 | 33 | -1.31e-2 | +4.82e-3 | -1.19e-3 | -2.81e-3 |
| 2 | 3.00e-1 | 11 | 7.65e-2 | 1.19e-1 | 8.57e-2 | 7.95e-2 | 29 | -1.20e-2 | +6.64e-3 | -5.50e-4 | -1.16e-3 |
| 3 | 3.00e-1 | 5 | 7.81e-2 | 1.34e-1 | 9.94e-2 | 1.01e-1 | 38 | -1.86e-2 | +7.67e-3 | -6.16e-4 | -8.44e-4 |
| 4 | 3.00e-1 | 9 | 9.00e-2 | 1.54e-1 | 1.02e-1 | 1.02e-1 | 36 | -1.15e-2 | +4.90e-3 | -6.28e-4 | -4.91e-4 |
| 5 | 3.00e-1 | 4 | 9.40e-2 | 1.42e-1 | 1.07e-1 | 9.76e-2 | 37 | -1.17e-2 | +3.95e-3 | -1.69e-3 | -8.91e-4 |
| 6 | 3.00e-1 | 6 | 9.27e-2 | 1.33e-1 | 1.10e-1 | 1.09e-1 | 44 | -9.86e-3 | +4.17e-3 | -3.44e-4 | -6.03e-4 |
| 7 | 3.00e-1 | 1 | 1.05e-1 | 1.05e-1 | 1.05e-1 | 1.05e-1 | 44 | -7.44e-4 | -7.44e-4 | -7.44e-4 | -6.17e-4 |
| 8 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 330 | +2.17e-3 | +2.17e-3 | +2.17e-3 | -3.38e-4 |
| 9 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 334 | -4.32e-5 | -4.32e-5 | -4.32e-5 | -3.08e-4 |
| 10 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 315 | -1.06e-4 | -1.06e-4 | -1.06e-4 | -2.88e-4 |
| 11 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 309 | -1.02e-4 | -1.02e-4 | -1.02e-4 | -2.69e-4 |
| 12 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 331 | +2.11e-5 | +2.11e-5 | +2.11e-5 | -2.40e-4 |
| 14 | 3.00e-1 | 2 | 1.90e-1 | 2.08e-1 | 1.99e-1 | 1.90e-1 | 267 | -3.42e-4 | +1.07e-4 | -1.18e-4 | -2.19e-4 |
| 16 | 3.00e-1 | 2 | 1.87e-1 | 2.04e-1 | 1.96e-1 | 1.87e-1 | 267 | -3.27e-4 | +2.18e-4 | -5.45e-5 | -1.91e-4 |
| 18 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 352 | +3.58e-4 | +3.58e-4 | +3.58e-4 | -1.36e-4 |
| 19 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 347 | -3.52e-5 | -3.52e-5 | -3.52e-5 | -1.26e-4 |
| 20 | 3.00e-1 | 1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 271 | -3.33e-4 | -3.33e-4 | -3.33e-4 | -1.46e-4 |
| 21 | 3.00e-1 | 1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 275 | +1.90e-7 | +1.90e-7 | +1.90e-7 | -1.32e-4 |
| 22 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 290 | +1.35e-4 | +1.35e-4 | +1.35e-4 | -1.05e-4 |
| 23 | 3.00e-1 | 1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 263 | -1.22e-4 | -1.22e-4 | -1.22e-4 | -1.07e-4 |
| 24 | 3.00e-1 | 1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 261 | +1.56e-5 | +1.56e-5 | +1.56e-5 | -9.45e-5 |
| 25 | 3.00e-1 | 1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 269 | +1.09e-5 | +1.09e-5 | +1.09e-5 | -8.39e-5 |
| 26 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 291 | +8.60e-5 | +8.60e-5 | +8.60e-5 | -6.69e-5 |
| 27 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 295 | +5.61e-5 | +5.61e-5 | +5.61e-5 | -5.46e-5 |
| 28 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 278 | -7.22e-5 | -7.22e-5 | -7.22e-5 | -5.64e-5 |
| 29 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 291 | +3.39e-5 | +3.39e-5 | +3.39e-5 | -4.73e-5 |
| 30 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 292 | +7.03e-5 | +7.03e-5 | +7.03e-5 | -3.56e-5 |
| 31 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 296 | +7.00e-5 | +7.00e-5 | +7.00e-5 | -2.50e-5 |
| 32 | 3.00e-1 | 1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 259 | -2.25e-4 | -2.25e-4 | -2.25e-4 | -4.50e-5 |
| 33 | 3.00e-1 | 1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 247 | -5.21e-5 | -5.21e-5 | -5.21e-5 | -4.58e-5 |
| 34 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 262 | +7.99e-5 | +7.99e-5 | +7.99e-5 | -3.32e-5 |
| 35 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 261 | +1.30e-5 | +1.30e-5 | +1.30e-5 | -2.86e-5 |
| 36 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 260 | +1.66e-5 | +1.66e-5 | +1.66e-5 | -2.41e-5 |
| 37 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 275 | +4.25e-5 | +4.25e-5 | +4.25e-5 | -1.74e-5 |
| 38 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 274 | -2.63e-6 | -2.63e-6 | -2.63e-6 | -1.59e-5 |
| 39 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 254 | -6.06e-5 | -6.06e-5 | -6.06e-5 | -2.04e-5 |
| 40 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 276 | +1.67e-4 | +1.67e-4 | +1.67e-4 | -1.63e-6 |
| 41 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 272 | -8.68e-5 | -8.68e-5 | -8.68e-5 | -1.01e-5 |
| 42 | 3.00e-1 | 1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 242 | -2.25e-4 | -2.25e-4 | -2.25e-4 | -3.17e-5 |
| 43 | 3.00e-1 | 2 | 1.88e-1 | 2.04e-1 | 1.96e-1 | 1.88e-1 | 210 | -3.86e-4 | +2.05e-4 | -9.05e-5 | -4.58e-5 |
| 44 | 3.00e-1 | 1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 240 | +1.66e-4 | +1.66e-4 | +1.66e-4 | -2.46e-5 |
| 45 | 3.00e-1 | 1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 228 | -1.24e-4 | -1.24e-4 | -1.24e-4 | -3.46e-5 |
| 46 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 260 | +2.28e-4 | +2.28e-4 | +2.28e-4 | -8.34e-6 |
| 47 | 3.00e-1 | 2 | 1.90e-1 | 1.99e-1 | 1.94e-1 | 1.90e-1 | 231 | -1.94e-4 | -5.66e-5 | -1.25e-4 | -3.12e-5 |
| 49 | 3.00e-1 | 2 | 1.95e-1 | 2.22e-1 | 2.08e-1 | 1.95e-1 | 231 | -5.65e-4 | +4.75e-4 | -4.48e-5 | -3.90e-5 |
| 51 | 3.00e-1 | 2 | 1.89e-1 | 2.19e-1 | 2.04e-1 | 1.89e-1 | 215 | -6.86e-4 | +3.89e-4 | -1.49e-4 | -6.52e-5 |
| 52 | 3.00e-1 | 1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 217 | +8.53e-5 | +8.53e-5 | +8.53e-5 | -5.02e-5 |
| 53 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 248 | +2.27e-4 | +2.27e-4 | +2.27e-4 | -2.25e-5 |
| 54 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 249 | -5.83e-5 | -5.83e-5 | -5.83e-5 | -2.61e-5 |
| 55 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 243 | -4.61e-5 | -4.61e-5 | -4.61e-5 | -2.81e-5 |
| 56 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 225 | -2.04e-5 | -2.04e-5 | -2.04e-5 | -2.73e-5 |
| 57 | 3.00e-1 | 2 | 1.87e-1 | 2.05e-1 | 1.96e-1 | 1.87e-1 | 200 | -4.54e-4 | +1.38e-4 | -1.58e-4 | -5.51e-5 |
| 58 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 235 | +3.04e-4 | +3.04e-4 | +3.04e-4 | -1.92e-5 |
| 59 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 250 | +8.75e-5 | +8.75e-5 | +8.75e-5 | -8.51e-6 |
| 60 | 3.00e-1 | 1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 220 | -3.07e-4 | -3.07e-4 | -3.07e-4 | -3.84e-5 |
| 61 | 3.00e-1 | 2 | 1.85e-1 | 1.93e-1 | 1.89e-1 | 1.85e-1 | 187 | -2.51e-4 | +3.42e-5 | -1.08e-4 | -5.31e-5 |
| 62 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 234 | +4.18e-4 | +4.18e-4 | +4.18e-4 | -5.98e-6 |
| 63 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 246 | -5.69e-5 | -5.69e-5 | -5.69e-5 | -1.11e-5 |
| 64 | 3.00e-1 | 1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 210 | -1.94e-4 | -1.94e-4 | -1.94e-4 | -2.94e-5 |
| 65 | 3.00e-1 | 2 | 1.90e-1 | 1.95e-1 | 1.92e-1 | 1.90e-1 | 198 | -1.35e-4 | +5.38e-5 | -4.04e-5 | -3.24e-5 |
| 66 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 252 | +3.53e-4 | +3.53e-4 | +3.53e-4 | +6.11e-6 |
| 67 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 224 | -2.60e-4 | -2.60e-4 | -2.60e-4 | -2.05e-5 |
| 68 | 3.00e-1 | 2 | 1.85e-1 | 2.04e-1 | 1.94e-1 | 1.85e-1 | 185 | -5.15e-4 | +1.70e-4 | -1.73e-4 | -5.28e-5 |
| 69 | 3.00e-1 | 1 | 1.87e-1 | 1.87e-1 | 1.87e-1 | 1.87e-1 | 185 | +4.71e-5 | +4.71e-5 | +4.71e-5 | -4.28e-5 |
| 70 | 3.00e-1 | 2 | 1.79e-1 | 1.97e-1 | 1.88e-1 | 1.79e-1 | 172 | -5.32e-4 | +2.50e-4 | -1.41e-4 | -6.54e-5 |
| 71 | 3.00e-1 | 1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 200 | +3.90e-4 | +3.90e-4 | +3.90e-4 | -1.98e-5 |
| 72 | 3.00e-1 | 2 | 1.89e-1 | 1.98e-1 | 1.93e-1 | 1.89e-1 | 185 | -2.54e-4 | +9.45e-5 | -7.96e-5 | -3.29e-5 |
| 74 | 3.00e-1 | 2 | 1.85e-1 | 2.18e-1 | 2.01e-1 | 1.85e-1 | 182 | -9.13e-4 | +5.18e-4 | -1.98e-4 | -7.14e-5 |
| 75 | 3.00e-1 | 1 | 1.89e-1 | 1.89e-1 | 1.89e-1 | 1.89e-1 | 197 | +1.31e-4 | +1.31e-4 | +1.31e-4 | -5.12e-5 |
| 76 | 3.00e-1 | 2 | 1.78e-1 | 1.97e-1 | 1.87e-1 | 1.78e-1 | 168 | -5.80e-4 | +1.81e-4 | -1.99e-4 | -8.32e-5 |
| 77 | 3.00e-1 | 2 | 1.80e-1 | 1.90e-1 | 1.85e-1 | 1.80e-1 | 168 | -2.97e-4 | +3.13e-4 | +8.26e-6 | -6.88e-5 |
| 78 | 3.00e-1 | 1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 197 | +3.35e-4 | +3.35e-4 | +3.35e-4 | -2.85e-5 |
| 79 | 3.00e-1 | 1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 205 | +5.32e-5 | +5.32e-5 | +5.32e-5 | -2.03e-5 |
| 80 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 228 | +1.37e-4 | +1.37e-4 | +1.37e-4 | -4.60e-6 |
| 81 | 3.00e-1 | 3 | 1.70e-1 | 1.96e-1 | 1.81e-1 | 1.70e-1 | 145 | -9.79e-4 | +4.92e-4 | -4.22e-4 | -1.20e-4 |
| 82 | 3.00e-1 | 1 | 1.81e-1 | 1.81e-1 | 1.81e-1 | 1.81e-1 | 166 | +3.87e-4 | +3.87e-4 | +3.87e-4 | -6.93e-5 |
| 83 | 3.00e-1 | 1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 186 | +3.06e-4 | +3.06e-4 | +3.06e-4 | -3.18e-5 |
| 84 | 3.00e-1 | 2 | 1.76e-1 | 1.84e-1 | 1.80e-1 | 1.76e-1 | 153 | -3.02e-4 | -2.42e-4 | -2.72e-4 | -7.77e-5 |
| 85 | 3.00e-1 | 2 | 1.75e-1 | 1.86e-1 | 1.81e-1 | 1.75e-1 | 153 | -4.03e-4 | +3.39e-4 | -3.20e-5 | -7.27e-5 |
| 86 | 3.00e-1 | 1 | 1.91e-1 | 1.91e-1 | 1.91e-1 | 1.91e-1 | 202 | +4.32e-4 | +4.32e-4 | +4.32e-4 | -2.22e-5 |
| 87 | 3.00e-1 | 2 | 1.77e-1 | 1.95e-1 | 1.86e-1 | 1.77e-1 | 153 | -6.06e-4 | +9.64e-5 | -2.55e-4 | -6.99e-5 |
| 88 | 3.00e-1 | 2 | 1.75e-1 | 1.89e-1 | 1.82e-1 | 1.89e-1 | 179 | -8.80e-5 | +4.25e-4 | +1.68e-4 | -2.21e-5 |
| 89 | 3.00e-1 | 1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 190 | +1.75e-4 | +1.75e-4 | +1.75e-4 | -2.41e-6 |
| 90 | 3.00e-1 | 2 | 1.66e-1 | 1.82e-1 | 1.74e-1 | 1.66e-1 | 133 | -6.92e-4 | -4.53e-4 | -5.72e-4 | -1.12e-4 |
| 91 | 3.00e-1 | 2 | 1.71e-1 | 1.83e-1 | 1.77e-1 | 1.71e-1 | 133 | -4.92e-4 | +5.77e-4 | +4.26e-5 | -8.79e-5 |
| 92 | 3.00e-1 | 2 | 1.70e-1 | 1.82e-1 | 1.76e-1 | 1.70e-1 | 149 | -4.44e-4 | +3.67e-4 | -3.82e-5 | -8.25e-5 |
| 93 | 3.00e-1 | 2 | 1.68e-1 | 1.94e-1 | 1.81e-1 | 1.68e-1 | 141 | -1.04e-3 | +7.08e-4 | -1.64e-4 | -1.07e-4 |
| 94 | 3.00e-1 | 2 | 1.67e-1 | 1.84e-1 | 1.75e-1 | 1.67e-1 | 141 | -6.95e-4 | +5.63e-4 | -6.62e-5 | -1.05e-4 |
| 95 | 3.00e-1 | 1 | 1.82e-1 | 1.82e-1 | 1.82e-1 | 1.82e-1 | 162 | +5.47e-4 | +5.47e-4 | +5.47e-4 | -4.00e-5 |
| 96 | 3.00e-1 | 2 | 1.69e-1 | 1.84e-1 | 1.77e-1 | 1.69e-1 | 141 | -6.02e-4 | +6.17e-5 | -2.70e-4 | -8.70e-5 |
| 97 | 3.00e-1 | 2 | 1.74e-1 | 1.91e-1 | 1.82e-1 | 1.74e-1 | 141 | -6.96e-4 | +6.29e-4 | -3.32e-5 | -8.34e-5 |
| 98 | 3.00e-1 | 2 | 1.64e-1 | 1.82e-1 | 1.73e-1 | 1.64e-1 | 123 | -8.27e-4 | +2.79e-4 | -2.74e-4 | -1.25e-4 |
| 99 | 3.00e-1 | 2 | 1.59e-1 | 1.87e-1 | 1.73e-1 | 1.59e-1 | 113 | -1.44e-3 | +8.13e-4 | -3.12e-4 | -1.72e-4 |
| 100 | 3.00e-2 | 2 | 1.56e-2 | 1.54e-1 | 8.48e-2 | 1.56e-2 | 120 | -1.91e-2 | -2.11e-4 | -9.66e-3 | -2.07e-3 |
| 101 | 3.00e-2 | 3 | 1.55e-2 | 1.65e-2 | 1.61e-2 | 1.62e-2 | 113 | -5.33e-4 | +4.23e-4 | +9.52e-5 | -1.48e-3 |
| 102 | 3.00e-2 | 1 | 1.87e-2 | 1.87e-2 | 1.87e-2 | 1.87e-2 | 156 | +8.94e-4 | +8.94e-4 | +8.94e-4 | -1.24e-3 |
| 103 | 3.00e-2 | 2 | 1.84e-2 | 2.01e-2 | 1.93e-2 | 1.84e-2 | 128 | -6.82e-4 | +4.76e-4 | -1.03e-4 | -1.03e-3 |
| 104 | 3.00e-2 | 3 | 1.70e-2 | 1.96e-2 | 1.80e-2 | 1.75e-2 | 104 | -1.36e-3 | +4.22e-4 | -2.19e-4 | -8.14e-4 |
| 105 | 3.00e-2 | 2 | 1.70e-2 | 1.96e-2 | 1.83e-2 | 1.70e-2 | 95 | -1.51e-3 | +8.90e-4 | -3.11e-4 | -7.30e-4 |
| 106 | 3.00e-2 | 3 | 1.76e-2 | 2.05e-2 | 1.90e-2 | 1.89e-2 | 101 | -1.53e-3 | +1.43e-3 | +2.06e-4 | -4.83e-4 |
| 107 | 3.00e-2 | 2 | 1.78e-2 | 2.17e-2 | 1.98e-2 | 1.78e-2 | 95 | -2.09e-3 | +9.72e-4 | -5.59e-4 | -5.12e-4 |
| 108 | 3.00e-2 | 4 | 1.77e-2 | 2.28e-2 | 1.94e-2 | 1.77e-2 | 85 | -1.98e-3 | +1.67e-3 | -2.36e-4 | -4.36e-4 |
| 109 | 3.00e-2 | 2 | 1.75e-2 | 2.21e-2 | 1.98e-2 | 1.75e-2 | 80 | -2.93e-3 | +1.77e-3 | -5.79e-4 | -4.87e-4 |
| 110 | 3.00e-2 | 3 | 1.91e-2 | 2.11e-2 | 1.97e-2 | 1.91e-2 | 88 | -1.14e-3 | +1.52e-3 | +1.31e-4 | -3.33e-4 |
| 111 | 3.00e-2 | 3 | 1.93e-2 | 2.18e-2 | 2.04e-2 | 2.00e-2 | 88 | -1.37e-3 | +1.10e-3 | +3.60e-5 | -2.39e-4 |
| 112 | 3.00e-2 | 3 | 1.89e-2 | 2.30e-2 | 2.03e-2 | 1.89e-2 | 73 | -2.30e-3 | +1.16e-3 | -4.31e-4 | -3.03e-4 |
| 113 | 3.00e-2 | 4 | 1.79e-2 | 2.15e-2 | 1.91e-2 | 1.85e-2 | 68 | -2.31e-3 | +1.25e-3 | -2.47e-4 | -2.83e-4 |
| 114 | 3.00e-2 | 3 | 1.81e-2 | 2.16e-2 | 1.97e-2 | 1.81e-2 | 68 | -1.66e-3 | +1.52e-3 | -3.71e-4 | -3.30e-4 |
| 115 | 3.00e-2 | 5 | 1.77e-2 | 2.26e-2 | 1.93e-2 | 1.77e-2 | 63 | -3.15e-3 | +2.27e-3 | -2.84e-4 | -3.40e-4 |
| 116 | 3.00e-2 | 3 | 1.82e-2 | 2.29e-2 | 2.01e-2 | 1.82e-2 | 59 | -2.83e-3 | +2.69e-3 | -3.49e-4 | -3.75e-4 |
| 117 | 3.00e-2 | 7 | 1.65e-2 | 2.54e-2 | 1.89e-2 | 1.65e-2 | 52 | -6.40e-3 | +3.18e-3 | -6.40e-4 | -5.39e-4 |
| 118 | 3.00e-2 | 3 | 1.65e-2 | 2.27e-2 | 1.92e-2 | 1.65e-2 | 46 | -3.78e-3 | +3.64e-3 | -8.22e-4 | -6.71e-4 |
| 119 | 3.00e-2 | 5 | 1.56e-2 | 2.38e-2 | 1.76e-2 | 1.66e-2 | 45 | -9.48e-3 | +3.98e-3 | -9.54e-4 | -7.43e-4 |
| 120 | 3.00e-2 | 8 | 1.54e-2 | 2.32e-2 | 1.71e-2 | 1.54e-2 | 37 | -4.78e-3 | +3.90e-3 | -6.91e-4 | -7.15e-4 |
| 121 | 3.00e-2 | 4 | 1.56e-2 | 2.36e-2 | 1.82e-2 | 1.56e-2 | 35 | -7.97e-3 | +4.83e-3 | -1.26e-3 | -9.55e-4 |
| 122 | 3.00e-2 | 6 | 1.49e-2 | 2.24e-2 | 1.69e-2 | 1.80e-2 | 44 | -1.13e-2 | +4.73e-3 | -3.75e-4 | -5.60e-4 |
| 123 | 3.00e-2 | 6 | 1.74e-2 | 2.48e-2 | 1.91e-2 | 1.77e-2 | 41 | -7.98e-3 | +3.90e-3 | -6.11e-4 | -5.73e-4 |
| 124 | 3.00e-2 | 7 | 1.56e-2 | 2.46e-2 | 1.83e-2 | 1.80e-2 | 38 | -1.10e-2 | +4.06e-3 | -7.02e-4 | -5.44e-4 |
| 125 | 3.00e-2 | 7 | 1.44e-2 | 2.58e-2 | 1.75e-2 | 1.58e-2 | 35 | -1.08e-2 | +3.98e-3 | -1.33e-3 | -8.56e-4 |
| 126 | 3.00e-2 | 6 | 1.69e-2 | 2.36e-2 | 1.86e-2 | 1.78e-2 | 39 | -8.71e-3 | +5.80e-3 | -3.76e-4 | -6.34e-4 |
| 127 | 3.00e-2 | 7 | 1.49e-2 | 2.81e-2 | 1.86e-2 | 1.79e-2 | 34 | -1.49e-2 | +5.35e-3 | -1.17e-3 | -7.60e-4 |
| 128 | 3.00e-2 | 7 | 1.59e-2 | 2.48e-2 | 1.80e-2 | 1.77e-2 | 35 | -1.07e-2 | +4.44e-3 | -6.65e-4 | -6.06e-4 |
| 129 | 3.00e-2 | 10 | 1.50e-2 | 2.71e-2 | 1.87e-2 | 1.87e-2 | 40 | -8.74e-3 | +5.48e-3 | -5.13e-4 | -3.61e-4 |
| 130 | 3.00e-2 | 5 | 1.50e-2 | 2.47e-2 | 1.76e-2 | 1.58e-2 | 27 | -1.84e-2 | +3.78e-3 | -2.68e-3 | -1.24e-3 |
| 131 | 3.00e-2 | 6 | 1.98e-2 | 2.81e-2 | 2.16e-2 | 2.08e-2 | 43 | -8.76e-3 | +6.69e-3 | -1.55e-4 | -7.42e-4 |
| 132 | 3.00e-2 | 7 | 1.97e-2 | 2.97e-2 | 2.22e-2 | 1.97e-2 | 43 | -8.12e-3 | +4.41e-3 | -6.64e-4 | -7.23e-4 |
| 133 | 3.00e-2 | 4 | 2.12e-2 | 2.94e-2 | 2.36e-2 | 2.22e-2 | 43 | -7.58e-3 | +5.05e-3 | -3.80e-4 | -6.21e-4 |
| 134 | 3.00e-2 | 7 | 1.91e-2 | 3.04e-2 | 2.13e-2 | 1.94e-2 | 34 | -1.15e-2 | +3.52e-3 | -1.21e-3 | -8.43e-4 |
| 135 | 3.00e-2 | 7 | 2.10e-2 | 3.27e-2 | 2.35e-2 | 2.13e-2 | 42 | -7.66e-3 | +4.94e-3 | -7.11e-4 | -7.65e-4 |
| 136 | 3.00e-2 | 4 | 2.03e-2 | 3.32e-2 | 2.39e-2 | 2.11e-2 | 38 | -1.23e-2 | +4.93e-3 | -1.84e-3 | -1.13e-3 |
| 137 | 3.00e-2 | 7 | 1.89e-2 | 3.00e-2 | 2.22e-2 | 2.06e-2 | 39 | -1.25e-2 | +5.00e-3 | -7.26e-4 | -8.90e-4 |
| 138 | 3.00e-2 | 6 | 2.14e-2 | 3.21e-2 | 2.39e-2 | 2.23e-2 | 39 | -1.05e-2 | +5.10e-3 | -7.71e-4 | -8.12e-4 |
| 139 | 3.00e-2 | 7 | 1.91e-2 | 3.35e-2 | 2.29e-2 | 2.34e-2 | 39 | -1.22e-2 | +4.54e-3 | -7.81e-4 | -6.09e-4 |
| 140 | 3.00e-2 | 5 | 2.27e-2 | 3.24e-2 | 2.55e-2 | 2.41e-2 | 41 | -8.61e-3 | +3.95e-3 | -6.64e-4 | -6.22e-4 |
| 141 | 3.00e-2 | 2 | 2.37e-2 | 5.97e-2 | 4.17e-2 | 5.97e-2 | 248 | -4.62e-4 | +3.73e-3 | +1.63e-3 | -1.72e-4 |
| 142 | 3.00e-2 | 1 | 6.47e-2 | 6.47e-2 | 6.47e-2 | 6.47e-2 | 289 | +2.81e-4 | +2.81e-4 | +2.81e-4 | -1.27e-4 |
| 143 | 3.00e-2 | 1 | 6.15e-2 | 6.15e-2 | 6.15e-2 | 6.15e-2 | 256 | -1.98e-4 | -1.98e-4 | -1.98e-4 | -1.34e-4 |
| 144 | 3.00e-2 | 1 | 6.28e-2 | 6.28e-2 | 6.28e-2 | 6.28e-2 | 282 | +7.51e-5 | +7.51e-5 | +7.51e-5 | -1.13e-4 |
| 145 | 3.00e-2 | 1 | 6.19e-2 | 6.19e-2 | 6.19e-2 | 6.19e-2 | 267 | -5.42e-5 | -5.42e-5 | -5.42e-5 | -1.07e-4 |
| 147 | 3.00e-2 | 2 | 6.11e-2 | 6.80e-2 | 6.46e-2 | 6.11e-2 | 267 | -4.00e-4 | +2.67e-4 | -6.66e-5 | -1.03e-4 |
| 149 | 3.00e-3 | 2 | 6.15e-2 | 6.74e-2 | 6.44e-2 | 6.15e-2 | 267 | -3.40e-4 | +2.99e-4 | -2.03e-5 | -9.03e-5 |
| 151 | 3.00e-3 | 1 | 6.93e-3 | 6.93e-3 | 6.93e-3 | 6.93e-3 | 358 | -6.10e-3 | -6.10e-3 | -6.10e-3 | -6.91e-4 |
| 152 | 3.00e-3 | 1 | 5.95e-3 | 5.95e-3 | 5.95e-3 | 5.95e-3 | 306 | -4.97e-4 | -4.97e-4 | -4.97e-4 | -6.72e-4 |
| 153 | 3.00e-3 | 1 | 6.49e-3 | 6.49e-3 | 6.49e-3 | 6.49e-3 | 327 | +2.64e-4 | +2.64e-4 | +2.64e-4 | -5.78e-4 |
| 154 | 3.00e-3 | 1 | 6.20e-3 | 6.20e-3 | 6.20e-3 | 6.20e-3 | 311 | -1.47e-4 | -1.47e-4 | -1.47e-4 | -5.35e-4 |
| 155 | 3.00e-3 | 1 | 5.95e-3 | 5.95e-3 | 5.95e-3 | 5.95e-3 | 279 | -1.45e-4 | -1.45e-4 | -1.45e-4 | -4.96e-4 |
| 156 | 3.00e-3 | 1 | 5.83e-3 | 5.83e-3 | 5.83e-3 | 5.83e-3 | 265 | -7.82e-5 | -7.82e-5 | -7.82e-5 | -4.54e-4 |
| 157 | 3.00e-3 | 1 | 6.23e-3 | 6.23e-3 | 6.23e-3 | 6.23e-3 | 307 | +2.15e-4 | +2.15e-4 | +2.15e-4 | -3.87e-4 |
| 158 | 3.00e-3 | 1 | 5.93e-3 | 5.93e-3 | 5.93e-3 | 5.93e-3 | 267 | -1.82e-4 | -1.82e-4 | -1.82e-4 | -3.67e-4 |
| 159 | 3.00e-3 | 1 | 5.84e-3 | 5.84e-3 | 5.84e-3 | 5.84e-3 | 259 | -6.07e-5 | -6.07e-5 | -6.07e-5 | -3.36e-4 |
| 160 | 3.00e-3 | 1 | 5.97e-3 | 5.97e-3 | 5.97e-3 | 5.97e-3 | 272 | +7.86e-5 | +7.86e-5 | +7.86e-5 | -2.95e-4 |
| 161 | 3.00e-3 | 1 | 6.14e-3 | 6.14e-3 | 6.14e-3 | 6.14e-3 | 275 | +1.01e-4 | +1.01e-4 | +1.01e-4 | -2.55e-4 |
| 162 | 3.00e-3 | 1 | 6.26e-3 | 6.26e-3 | 6.26e-3 | 6.26e-3 | 284 | +7.11e-5 | +7.11e-5 | +7.11e-5 | -2.22e-4 |
| 163 | 3.00e-3 | 1 | 6.04e-3 | 6.04e-3 | 6.04e-3 | 6.04e-3 | 259 | -1.41e-4 | -1.41e-4 | -1.41e-4 | -2.14e-4 |
| 164 | 3.00e-3 | 1 | 6.11e-3 | 6.11e-3 | 6.11e-3 | 6.11e-3 | 271 | +4.35e-5 | +4.35e-5 | +4.35e-5 | -1.89e-4 |
| 165 | 3.00e-3 | 1 | 6.22e-3 | 6.22e-3 | 6.22e-3 | 6.22e-3 | 285 | +6.37e-5 | +6.37e-5 | +6.37e-5 | -1.63e-4 |
| 166 | 3.00e-3 | 1 | 6.16e-3 | 6.16e-3 | 6.16e-3 | 6.16e-3 | 270 | -3.61e-5 | -3.61e-5 | -3.61e-5 | -1.51e-4 |
| 167 | 3.00e-3 | 1 | 6.34e-3 | 6.34e-3 | 6.34e-3 | 6.34e-3 | 285 | +9.97e-5 | +9.97e-5 | +9.97e-5 | -1.26e-4 |
| 168 | 3.00e-3 | 1 | 6.14e-3 | 6.14e-3 | 6.14e-3 | 6.14e-3 | 277 | -1.16e-4 | -1.16e-4 | -1.16e-4 | -1.25e-4 |
| 169 | 3.00e-3 | 1 | 6.30e-3 | 6.30e-3 | 6.30e-3 | 6.30e-3 | 271 | +9.49e-5 | +9.49e-5 | +9.49e-5 | -1.03e-4 |
| 170 | 3.00e-3 | 1 | 6.15e-3 | 6.15e-3 | 6.15e-3 | 6.15e-3 | 273 | -8.62e-5 | -8.62e-5 | -8.62e-5 | -1.01e-4 |
| 171 | 3.00e-3 | 1 | 6.60e-3 | 6.60e-3 | 6.60e-3 | 6.60e-3 | 302 | +2.34e-4 | +2.34e-4 | +2.34e-4 | -6.75e-5 |
| 173 | 3.00e-3 | 2 | 5.71e-3 | 7.14e-3 | 6.42e-3 | 5.71e-3 | 234 | -9.58e-4 | +2.33e-4 | -3.63e-4 | -1.30e-4 |
| 174 | 3.00e-3 | 1 | 6.45e-3 | 6.45e-3 | 6.45e-3 | 6.45e-3 | 272 | +4.50e-4 | +4.50e-4 | +4.50e-4 | -7.16e-5 |
| 175 | 3.00e-3 | 1 | 6.09e-3 | 6.09e-3 | 6.09e-3 | 6.09e-3 | 252 | -2.26e-4 | -2.26e-4 | -2.26e-4 | -8.71e-5 |
| 176 | 3.00e-3 | 1 | 6.31e-3 | 6.31e-3 | 6.31e-3 | 6.31e-3 | 263 | +1.33e-4 | +1.33e-4 | +1.33e-4 | -6.50e-5 |
| 177 | 3.00e-3 | 1 | 6.59e-3 | 6.59e-3 | 6.59e-3 | 6.59e-3 | 303 | +1.45e-4 | +1.45e-4 | +1.45e-4 | -4.41e-5 |
| 178 | 3.00e-3 | 1 | 6.50e-3 | 6.50e-3 | 6.50e-3 | 6.50e-3 | 284 | -4.83e-5 | -4.83e-5 | -4.83e-5 | -4.45e-5 |
| 179 | 3.00e-3 | 1 | 6.36e-3 | 6.36e-3 | 6.36e-3 | 6.36e-3 | 278 | -7.97e-5 | -7.97e-5 | -7.97e-5 | -4.80e-5 |
| 180 | 3.00e-3 | 1 | 6.11e-3 | 6.11e-3 | 6.11e-3 | 6.11e-3 | 239 | -1.65e-4 | -1.65e-4 | -1.65e-4 | -5.97e-5 |
| 181 | 3.00e-3 | 1 | 5.90e-3 | 5.90e-3 | 5.90e-3 | 5.90e-3 | 237 | -1.50e-4 | -1.50e-4 | -1.50e-4 | -6.88e-5 |
| 182 | 3.00e-3 | 1 | 5.96e-3 | 5.96e-3 | 5.96e-3 | 5.96e-3 | 222 | +4.45e-5 | +4.45e-5 | +4.45e-5 | -5.75e-5 |
| 183 | 3.00e-3 | 2 | 5.87e-3 | 6.44e-3 | 6.16e-3 | 5.87e-3 | 208 | -4.51e-4 | +3.06e-4 | -7.24e-5 | -6.41e-5 |
| 184 | 3.00e-3 | 1 | 5.98e-3 | 5.98e-3 | 5.98e-3 | 5.98e-3 | 221 | +8.62e-5 | +8.62e-5 | +8.62e-5 | -4.91e-5 |
| 185 | 3.00e-3 | 1 | 6.74e-3 | 6.74e-3 | 6.74e-3 | 6.74e-3 | 275 | +4.36e-4 | +4.36e-4 | +4.36e-4 | -5.13e-7 |
| 186 | 3.00e-3 | 1 | 6.12e-3 | 6.12e-3 | 6.12e-3 | 6.12e-3 | 243 | -3.95e-4 | -3.95e-4 | -3.95e-4 | -4.00e-5 |
| 187 | 3.00e-3 | 1 | 6.09e-3 | 6.09e-3 | 6.09e-3 | 6.09e-3 | 236 | -2.63e-5 | -2.63e-5 | -2.63e-5 | -3.86e-5 |
| 188 | 3.00e-3 | 1 | 6.58e-3 | 6.58e-3 | 6.58e-3 | 6.58e-3 | 261 | +2.98e-4 | +2.98e-4 | +2.98e-4 | -4.98e-6 |
| 189 | 3.00e-3 | 2 | 5.76e-3 | 6.57e-3 | 6.16e-3 | 5.76e-3 | 197 | -6.70e-4 | -6.40e-6 | -3.38e-4 | -7.16e-5 |
| 190 | 3.00e-3 | 1 | 6.46e-3 | 6.46e-3 | 6.46e-3 | 6.46e-3 | 249 | +4.66e-4 | +4.66e-4 | +4.66e-4 | -1.78e-5 |
| 191 | 3.00e-3 | 1 | 6.60e-3 | 6.60e-3 | 6.60e-3 | 6.60e-3 | 243 | +8.79e-5 | +8.79e-5 | +8.79e-5 | -7.25e-6 |
| 192 | 3.00e-3 | 1 | 6.46e-3 | 6.46e-3 | 6.46e-3 | 6.46e-3 | 261 | -8.20e-5 | -8.20e-5 | -8.20e-5 | -1.47e-5 |
| 193 | 3.00e-3 | 1 | 6.17e-3 | 6.17e-3 | 6.17e-3 | 6.17e-3 | 251 | -1.86e-4 | -1.86e-4 | -1.86e-4 | -3.19e-5 |
| 194 | 3.00e-3 | 1 | 6.47e-3 | 6.47e-3 | 6.47e-3 | 6.47e-3 | 247 | +1.92e-4 | +1.92e-4 | +1.92e-4 | -9.52e-6 |
| 195 | 3.00e-3 | 2 | 6.06e-3 | 6.63e-3 | 6.35e-3 | 6.06e-3 | 208 | -4.32e-4 | +9.45e-5 | -1.69e-4 | -4.24e-5 |
| 196 | 3.00e-3 | 1 | 5.97e-3 | 5.97e-3 | 5.97e-3 | 5.97e-3 | 211 | -7.30e-5 | -7.30e-5 | -7.30e-5 | -4.55e-5 |
| 197 | 3.00e-3 | 1 | 6.14e-3 | 6.14e-3 | 6.14e-3 | 6.14e-3 | 224 | +1.23e-4 | +1.23e-4 | +1.23e-4 | -2.86e-5 |
| 198 | 3.00e-3 | 2 | 6.00e-3 | 6.65e-3 | 6.32e-3 | 6.00e-3 | 195 | -5.28e-4 | +3.21e-4 | -1.04e-4 | -4.71e-5 |

