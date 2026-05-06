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
| cpu-async | 0.062863 | 0.9183 | +0.0058 | 1739.9 | 470 | 80.6 | 100% | 100% | 7.0 |

## Best Mode per Model

| Model | Best Eval | Mode | Fastest (within 2% of solo-0) | Mode |
|-------|-----------|------|-------------------------------|------|
| resnet-graph | - | - | - | - |

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
| cpu-async | 2.0719 | 0.7560 | 0.5987 | 0.5263 | 0.5430 | 0.5000 | 0.4975 | 0.4877 | 0.4743 | 0.4691 | 0.2166 | 0.1803 | 0.1616 | 0.1486 | 0.1309 | 0.0789 | 0.0719 | 0.0683 | 0.0662 | 0.0629 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | cpu-async | 0 | cuda:0 | 0.4031 | 2.8 |
| resnet-graph | cpu-async | 1 | cuda:1 | 0.3012 | 3.8 |
| resnet-graph | cpu-async | 2 | cuda:2 | 0.2958 | 3.7 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|
| resnet-graph | cpu-async | 358 | 357 | 394 | 390 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| cpu-async | gpu1 | 418.9 | 1.3 | epoch-boundary(47) |
| cpu-async | gpu1 | 720.9 | 0.9 | epoch-boundary(82) |
| cpu-async | gpu2 | 720.8 | 0.9 | epoch-boundary(82) |
| cpu-async | gpu1 | 357.7 | 0.6 | epoch-boundary(40) |
| cpu-async | gpu1 | 1633.2 | 0.5 | epoch-boundary(187) |
| cpu-async | gpu2 | 1739.3 | 0.5 | epoch-boundary(199) |
| cpu-async | gpu2 | 1459.4 | 0.5 | epoch-boundary(167) |
| cpu-async | gpu0 | 0.4 | 0.5 | cpu-avg |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | cpu-async | gpu0 | 0.0s | 0.0s | 0.5s | 0.0s | 0.5s |
| resnet-graph | cpu-async | gpu1 | 3.5s | 0.0s | 0.0s | 0.0s | 4.0s |
| resnet-graph | cpu-async | gpu2 | 2.0s | 0.0s | 0.0s | 0.0s | 2.5s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | cpu-async | 287 | 0 | 470 | 80.6 | 2283/9473 | 470 | 80.6 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | cpu-async | 188.4 | 10.8% |

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
| resnet-graph | cpu-async | 185 | 470 | 0 | 7.69e-3 | -2.87e-5 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | cpu-async | 0 | 470 | 1.03e-1 | 6.52e-2 | 6.79e-3 | 4.43e-1 | 25.1 | -1.86e-4 | 1.18e-3 |
| resnet-graph | cpu-async | 1 | 470 | 1.04e-1 | 6.75e-2 | 6.55e-3 | 5.50e-1 | 35.5 | -2.11e-4 | 1.73e-3 |
| resnet-graph | cpu-async | 2 | 470 | 1.04e-1 | 6.73e-2 | 6.64e-3 | 5.11e-1 | 39.4 | -2.21e-4 | 1.52e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | cpu-async | rank0 ↔ rank1 | +0.9872 |
| resnet-graph | cpu-async | rank0 ↔ rank2 | +0.9927 |
| resnet-graph | cpu-async | rank1 ↔ rank2 | +0.9844 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | cpu-async | 47 (0,2,3,4,5,8,9,11…137,145) | 0 (—) | — | 0,2,3,4,5,8,9,11…137,145 | — |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | cpu-async | 0e0 | 3 | 1 | 1 |
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
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 318 | +0.078 |
| resnet-graph | cpu-async | 3.00e-2 | 100–148 | 96 | -0.067 |
| resnet-graph | cpu-async | 3.00e-3 | 149–199 | 52 | -0.136 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | cpu-async | 468 | -0.049 | 184 | +0.332 | +0.552 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | cpu-async | 469 | 3.38e1–8.01e1 | 6.64e1 | 2.69e-3 | 4.26e-3 | 4.47e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 320 | 34–77890 | +9.265e-6 | 0.386 | +9.563e-6 | 0.433 | 95 | +1.043e-5 | 0.679 | 28–979 | +9.245e-4 | 0.698 |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 309 | 962–77890 | +1.061e-5 | 0.591 | +1.081e-5 | 0.630 | 94 | +1.083e-5 | 0.711 | 82–979 | +9.795e-4 | 0.920 |
| resnet-graph | cpu-async | 3.00e-2 | 100–148 | 97 | 78582–116306 | +3.039e-6 | 0.017 | +2.794e-6 | 0.015 | 49 | -5.654e-7 | 0.000 | 159–692 | +2.548e-5 | 0.000 |
| resnet-graph | cpu-async | 3.00e-3 | 149–199 | 53 | 116566–155953 | -1.803e-5 | 0.240 | -1.809e-5 | 0.245 | 41 | -6.679e-6 | 0.091 | 152–1055 | -1.712e-3 | 0.724 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | +9.245e-4 | r0: +9.117e-4, r1: +9.320e-4, r2: +9.315e-4 | r0: 0.703, r1: 0.683, r2: 0.678 | 1.02× | ✓ |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +9.795e-4 | r0: +9.673e-4, r1: +9.857e-4, r2: +9.862e-4 | r0: 0.916, r1: 0.909, r2: 0.889 | 1.02× | ✓ |
| resnet-graph | cpu-async | 3.00e-2 | 100–148 | +2.548e-5 | r0: -1.372e-5, r1: +9.528e-5, r2: -5.174e-6 | r0: 0.000, r1: 0.002, r2: 0.000 | 18.42× | ⚠ framing breaking |
| resnet-graph | cpu-async | 3.00e-3 | 149–199 | -1.712e-3 | r0: -1.735e-3, r1: -1.693e-3, r2: -1.708e-3 | r0: 0.739, r1: 0.717, r2: 0.711 | 1.02× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | cpu-async | `█▇▇▇▇▇▇▇▇▇██████████████▅▄▄▄▄▄▄▄▄▅▅▅▂▁▁▁▁▁▁▁▁▁▁` | `▁▆▇▇▇█▇▇▇▇██████████████▇▇█████████▇▆▆▇▇███████` |

### Per-Epoch Detail: resnet-graph / cpu-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 11 | 1.10e-1 | 5.50e-1 | 2.36e-1 | 1.37e-1 | 45 | -3.94e-2 | +8.13e-3 | -1.22e-2 | -9.49e-3 |
| 1 | 3.00e-1 | 6 | 9.68e-2 | 1.79e-1 | 1.18e-1 | 9.99e-2 | 37 | -1.05e-2 | +2.91e-3 | -2.09e-3 | -5.12e-3 |
| 2 | 3.00e-1 | 7 | 1.00e-1 | 1.41e-1 | 1.10e-1 | 1.05e-1 | 33 | -7.20e-3 | +3.88e-3 | -6.11e-4 | -2.48e-3 |
| 3 | 3.00e-1 | 6 | 1.12e-1 | 1.56e-1 | 1.26e-1 | 1.30e-1 | 46 | -8.10e-3 | +5.21e-3 | -1.99e-4 | -1.30e-3 |
| 4 | 3.00e-1 | 6 | 1.08e-1 | 1.75e-1 | 1.27e-1 | 1.14e-1 | 38 | -6.14e-3 | +2.77e-3 | -1.20e-3 | -1.21e-3 |
| 5 | 3.00e-1 | 8 | 1.16e-1 | 1.55e-1 | 1.25e-1 | 1.16e-1 | 37 | -6.38e-3 | +4.07e-3 | -4.73e-4 | -7.95e-4 |
| 6 | 3.00e-1 | 5 | 1.15e-1 | 1.65e-1 | 1.27e-1 | 1.16e-1 | 36 | -9.02e-3 | +4.34e-3 | -1.14e-3 | -9.26e-4 |
| 7 | 3.00e-1 | 8 | 1.05e-1 | 1.67e-1 | 1.17e-1 | 1.05e-1 | 30 | -1.03e-2 | +3.94e-3 | -1.30e-3 | -1.06e-3 |
| 8 | 3.00e-1 | 7 | 1.05e-1 | 1.56e-1 | 1.18e-1 | 1.16e-1 | 38 | -1.23e-2 | +5.35e-3 | -8.37e-4 | -8.45e-4 |
| 9 | 3.00e-1 | 7 | 1.12e-1 | 1.57e-1 | 1.22e-1 | 1.15e-1 | 40 | -6.44e-3 | +3.75e-3 | -6.15e-4 | -6.87e-4 |
| 10 | 3.00e-1 | 6 | 1.08e-1 | 1.62e-1 | 1.21e-1 | 1.12e-1 | 37 | -7.87e-3 | +4.12e-3 | -9.65e-4 | -7.71e-4 |
| 11 | 3.00e-1 | 7 | 1.10e-1 | 1.52e-1 | 1.21e-1 | 1.23e-1 | 44 | -7.00e-3 | +3.71e-3 | -3.27e-4 | -4.89e-4 |
| 12 | 3.00e-1 | 7 | 1.06e-1 | 1.56e-1 | 1.17e-1 | 1.07e-1 | 33 | -9.98e-3 | +2.75e-3 | -1.40e-3 | -9.04e-4 |
| 13 | 3.00e-1 | 9 | 1.01e-1 | 1.45e-1 | 1.12e-1 | 1.07e-1 | 33 | -8.74e-3 | +4.45e-3 | -6.79e-4 | -6.93e-4 |
| 14 | 3.00e-1 | 5 | 1.11e-1 | 1.51e-1 | 1.22e-1 | 1.23e-1 | 46 | -9.12e-3 | +4.59e-3 | -4.80e-4 | -5.62e-4 |
| 15 | 3.00e-1 | 7 | 1.08e-1 | 1.48e-1 | 1.20e-1 | 1.16e-1 | 41 | -4.41e-3 | +2.30e-3 | -5.09e-4 | -4.82e-4 |
| 16 | 3.00e-1 | 8 | 1.08e-1 | 1.55e-1 | 1.21e-1 | 1.12e-1 | 39 | -4.81e-3 | +3.39e-3 | -6.02e-4 | -5.31e-4 |
| 17 | 3.00e-1 | 4 | 1.18e-1 | 1.46e-1 | 1.26e-1 | 1.18e-1 | 40 | -4.48e-3 | +3.69e-3 | -4.24e-4 | -5.29e-4 |
| 18 | 3.00e-1 | 6 | 1.14e-1 | 1.59e-1 | 1.25e-1 | 1.14e-1 | 37 | -6.31e-3 | +3.51e-3 | -8.14e-4 | -6.66e-4 |
| 19 | 3.00e-1 | 8 | 1.08e-1 | 1.45e-1 | 1.19e-1 | 1.30e-1 | 56 | -6.57e-3 | +3.36e-3 | -1.50e-4 | -2.51e-4 |
| 20 | 3.00e-1 | 3 | 1.36e-1 | 1.56e-1 | 1.43e-1 | 1.36e-1 | 61 | -2.28e-3 | +2.22e-3 | -3.62e-5 | -2.13e-4 |
| 21 | 3.00e-1 | 5 | 1.28e-1 | 1.65e-1 | 1.40e-1 | 1.28e-1 | 50 | -3.05e-3 | +2.07e-3 | -4.69e-4 | -3.43e-4 |
| 22 | 3.00e-1 | 6 | 1.21e-1 | 1.54e-1 | 1.30e-1 | 1.29e-1 | 50 | -3.94e-3 | +2.23e-3 | -2.64e-4 | -2.73e-4 |
| 23 | 3.00e-1 | 4 | 1.20e-1 | 1.55e-1 | 1.31e-1 | 1.22e-1 | 49 | -3.56e-3 | +1.98e-3 | -7.11e-4 | -4.34e-4 |
| 24 | 3.00e-1 | 6 | 1.11e-1 | 1.62e-1 | 1.23e-1 | 1.17e-1 | 42 | -7.16e-3 | +2.97e-3 | -8.24e-4 | -5.57e-4 |
| 25 | 3.00e-1 | 6 | 1.14e-1 | 1.53e-1 | 1.23e-1 | 1.18e-1 | 42 | -5.07e-3 | +3.01e-3 | -5.33e-4 | -5.20e-4 |
| 26 | 3.00e-1 | 8 | 1.09e-1 | 1.57e-1 | 1.21e-1 | 1.09e-1 | 37 | -6.23e-3 | +3.21e-3 | -7.06e-4 | -6.25e-4 |
| 27 | 3.00e-1 | 5 | 1.09e-1 | 1.56e-1 | 1.22e-1 | 1.09e-1 | 37 | -6.73e-3 | +4.41e-3 | -9.59e-4 | -7.87e-4 |
| 28 | 3.00e-1 | 8 | 1.06e-1 | 1.48e-1 | 1.17e-1 | 1.21e-1 | 41 | -7.83e-3 | +3.95e-3 | -2.78e-4 | -3.87e-4 |
| 29 | 3.00e-1 | 5 | 1.08e-1 | 1.51e-1 | 1.20e-1 | 1.10e-1 | 37 | -7.26e-3 | +3.05e-3 | -1.16e-3 | -6.92e-4 |
| 30 | 3.00e-1 | 8 | 1.02e-1 | 1.52e-1 | 1.12e-1 | 1.03e-1 | 32 | -7.64e-3 | +4.25e-3 | -8.86e-4 | -7.80e-4 |
| 31 | 3.00e-1 | 7 | 1.09e-1 | 1.48e-1 | 1.25e-1 | 1.29e-1 | 52 | -8.03e-3 | +5.15e-3 | -3.34e-5 | -3.59e-4 |
| 32 | 3.00e-1 | 4 | 1.20e-1 | 1.60e-1 | 1.34e-1 | 1.20e-1 | 45 | -4.20e-3 | +2.59e-3 | -8.54e-4 | -5.55e-4 |
| 33 | 3.00e-1 | 7 | 1.07e-1 | 1.55e-1 | 1.23e-1 | 1.17e-1 | 40 | -4.59e-3 | +3.12e-3 | -5.58e-4 | -5.35e-4 |
| 34 | 3.00e-1 | 6 | 1.14e-1 | 1.50e-1 | 1.24e-1 | 1.18e-1 | 44 | -3.73e-3 | +3.01e-3 | -4.97e-4 | -5.06e-4 |
| 35 | 3.00e-1 | 7 | 1.05e-1 | 1.59e-1 | 1.18e-1 | 1.07e-1 | 31 | -7.05e-3 | +3.48e-3 | -9.66e-4 | -6.98e-4 |
| 36 | 3.00e-1 | 7 | 1.04e-1 | 1.44e-1 | 1.15e-1 | 1.11e-1 | 33 | -8.23e-3 | +4.31e-3 | -7.18e-4 | -6.60e-4 |
| 37 | 3.00e-1 | 8 | 1.08e-1 | 1.54e-1 | 1.23e-1 | 1.33e-1 | 51 | -7.11e-3 | +4.25e-3 | -1.96e-4 | -2.73e-4 |
| 38 | 3.00e-1 | 4 | 1.31e-1 | 1.59e-1 | 1.39e-1 | 1.31e-1 | 51 | -3.52e-3 | +2.12e-3 | -4.05e-4 | -3.30e-4 |
| 39 | 3.00e-1 | 4 | 1.30e-1 | 1.61e-1 | 1.40e-1 | 1.30e-1 | 54 | -2.78e-3 | +2.23e-3 | -4.50e-4 | -3.91e-4 |
| 40 | 3.00e-1 | 2 | 1.23e-1 | 2.17e-1 | 1.70e-1 | 2.17e-1 | 245 | -1.39e-3 | +2.33e-3 | +4.70e-4 | -2.09e-4 |
| 41 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 245 | +6.68e-5 | +6.68e-5 | +6.68e-5 | -1.81e-4 |
| 42 | 3.00e-1 | 1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 274 | +1.04e-4 | +1.04e-4 | +1.04e-4 | -1.53e-4 |
| 43 | 3.00e-1 | 1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 269 | -6.51e-5 | -6.51e-5 | -6.51e-5 | -1.44e-4 |
| 45 | 3.00e-1 | 2 | 2.14e-1 | 2.46e-1 | 2.30e-1 | 2.14e-1 | 245 | -5.73e-4 | +2.66e-4 | -1.53e-4 | -1.50e-4 |
| 47 | 3.00e-1 | 2 | 2.15e-1 | 2.30e-1 | 2.22e-1 | 2.15e-1 | 245 | -2.82e-4 | +2.25e-4 | -2.80e-5 | -1.29e-4 |
| 48 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 251 | -5.65e-5 | -5.65e-5 | -5.65e-5 | -1.22e-4 |
| 49 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 266 | +9.73e-5 | +9.73e-5 | +9.73e-5 | -1.00e-4 |
| 50 | 3.00e-1 | 1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 300 | +1.06e-4 | +1.06e-4 | +1.06e-4 | -7.96e-5 |
| 52 | 3.00e-1 | 2 | 2.18e-1 | 2.42e-1 | 2.30e-1 | 2.18e-1 | 263 | -3.93e-4 | +2.14e-4 | -8.98e-5 | -8.45e-5 |
| 54 | 3.00e-1 | 1 | 2.40e-1 | 2.40e-1 | 2.40e-1 | 2.40e-1 | 373 | +2.60e-4 | +2.60e-4 | +2.60e-4 | -5.01e-5 |
| 55 | 3.00e-1 | 1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 328 | -1.39e-4 | -1.39e-4 | -1.39e-4 | -5.90e-5 |
| 56 | 3.00e-1 | 1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 285 | -1.10e-4 | -1.10e-4 | -1.10e-4 | -6.41e-5 |
| 57 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 260 | -1.12e-4 | -1.12e-4 | -1.12e-4 | -6.89e-5 |
| 58 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 288 | +5.83e-5 | +5.83e-5 | +5.83e-5 | -5.62e-5 |
| 59 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 273 | -9.04e-5 | -9.04e-5 | -9.04e-5 | -5.96e-5 |
| 60 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 269 | +3.55e-5 | +3.55e-5 | +3.55e-5 | -5.01e-5 |
| 61 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 263 | -2.19e-5 | -2.19e-5 | -2.19e-5 | -4.73e-5 |
| 62 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 270 | +7.75e-5 | +7.75e-5 | +7.75e-5 | -3.48e-5 |
| 63 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 271 | -3.25e-5 | -3.25e-5 | -3.25e-5 | -3.46e-5 |
| 64 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 276 | -1.43e-5 | -1.43e-5 | -1.43e-5 | -3.26e-5 |
| 65 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 272 | +7.39e-6 | +7.39e-6 | +7.39e-6 | -2.86e-5 |
| 66 | 3.00e-1 | 1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 299 | +1.18e-4 | +1.18e-4 | +1.18e-4 | -1.39e-5 |
| 67 | 3.00e-1 | 1 | 2.37e-1 | 2.37e-1 | 2.37e-1 | 2.37e-1 | 341 | +1.47e-4 | +1.47e-4 | +1.47e-4 | +2.15e-6 |
| 68 | 3.00e-1 | 1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 302 | -1.71e-4 | -1.71e-4 | -1.71e-4 | -1.52e-5 |
| 69 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 272 | -8.32e-5 | -8.32e-5 | -8.32e-5 | -2.20e-5 |
| 70 | 3.00e-1 | 1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 300 | +6.35e-5 | +6.35e-5 | +6.35e-5 | -1.35e-5 |
| 71 | 3.00e-1 | 1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 287 | -3.93e-5 | -3.93e-5 | -3.93e-5 | -1.60e-5 |
| 72 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 256 | -7.29e-5 | -7.29e-5 | -7.29e-5 | -2.17e-5 |
| 73 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 249 | -8.36e-5 | -8.36e-5 | -8.36e-5 | -2.79e-5 |
| 74 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 265 | +9.65e-5 | +9.65e-5 | +9.65e-5 | -1.55e-5 |
| 75 | 3.00e-1 | 1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 280 | +9.70e-5 | +9.70e-5 | +9.70e-5 | -4.23e-6 |
| 76 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 257 | -1.22e-4 | -1.22e-4 | -1.22e-4 | -1.60e-5 |
| 77 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 272 | -1.02e-5 | -1.02e-5 | -1.02e-5 | -1.54e-5 |
| 78 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 250 | -6.36e-5 | -6.36e-5 | -6.36e-5 | -2.02e-5 |
| 79 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 225 | -1.08e-4 | -1.08e-4 | -1.08e-4 | -2.90e-5 |
| 80 | 3.00e-1 | 2 | 2.12e-1 | 2.29e-1 | 2.21e-1 | 2.12e-1 | 225 | -3.42e-4 | +3.26e-4 | -7.97e-6 | -2.84e-5 |
| 82 | 3.00e-1 | 2 | 2.14e-1 | 2.19e-1 | 2.17e-1 | 2.14e-1 | 244 | -8.29e-5 | +1.08e-4 | +1.26e-5 | -2.15e-5 |
| 83 | 3.00e-1 | 1 | 2.28e-1 | 2.28e-1 | 2.28e-1 | 2.28e-1 | 316 | +1.93e-4 | +1.93e-4 | +1.93e-4 | -9.78e-8 |
| 84 | 3.00e-1 | 1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 278 | -7.21e-5 | -7.21e-5 | -7.21e-5 | -7.30e-6 |
| 85 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 253 | -1.32e-4 | -1.32e-4 | -1.32e-4 | -1.98e-5 |
| 86 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 232 | -9.91e-5 | -9.91e-5 | -9.91e-5 | -2.77e-5 |
| 87 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 221 | -1.65e-4 | -1.65e-4 | -1.65e-4 | -4.15e-5 |
| 88 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 226 | +1.24e-4 | +1.24e-4 | +1.24e-4 | -2.49e-5 |
| 89 | 3.00e-1 | 2 | 1.99e-1 | 2.06e-1 | 2.03e-1 | 1.99e-1 | 200 | -1.78e-4 | -7.28e-5 | -1.25e-4 | -4.45e-5 |
| 90 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 244 | +2.62e-4 | +2.62e-4 | +2.62e-4 | -1.38e-5 |
| 91 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 256 | +1.43e-4 | +1.43e-4 | +1.43e-4 | +1.86e-6 |
| 92 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 259 | -3.93e-5 | -3.93e-5 | -3.93e-5 | -2.26e-6 |
| 93 | 3.00e-1 | 2 | 2.04e-1 | 2.13e-1 | 2.08e-1 | 2.04e-1 | 199 | -2.30e-4 | -8.71e-5 | -1.59e-4 | -3.27e-5 |
| 94 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 219 | +1.42e-4 | +1.42e-4 | +1.42e-4 | -1.53e-5 |
| 95 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 226 | +3.96e-6 | +3.96e-6 | +3.96e-6 | -1.33e-5 |
| 96 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 212 | -1.15e-4 | -1.15e-4 | -1.15e-4 | -2.35e-5 |
| 97 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 238 | +1.93e-4 | +1.93e-4 | +1.93e-4 | -1.85e-6 |
| 98 | 3.00e-1 | 2 | 2.04e-1 | 2.10e-1 | 2.07e-1 | 2.04e-1 | 213 | -1.45e-4 | -8.46e-5 | -1.15e-4 | -2.36e-5 |
| 99 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 246 | +1.86e-4 | +1.86e-4 | +1.86e-4 | -2.61e-6 |
| 100 | 3.00e-2 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 240 | -4.20e-5 | -4.20e-5 | -4.20e-5 | -6.55e-6 |
| 101 | 3.00e-2 | 1 | 1.07e-1 | 1.07e-1 | 1.07e-1 | 1.07e-1 | 255 | -2.66e-3 | -2.66e-3 | -2.66e-3 | -2.72e-4 |
| 102 | 3.00e-2 | 1 | 5.55e-2 | 5.55e-2 | 5.55e-2 | 5.55e-2 | 248 | -2.66e-3 | -2.66e-3 | -2.66e-3 | -5.11e-4 |
| 103 | 3.00e-2 | 2 | 2.60e-2 | 3.54e-2 | 3.07e-2 | 2.60e-2 | 184 | -1.86e-3 | -1.67e-3 | -1.77e-3 | -7.48e-4 |
| 104 | 3.00e-2 | 1 | 2.44e-2 | 2.44e-2 | 2.44e-2 | 2.44e-2 | 197 | -3.41e-4 | -3.41e-4 | -3.41e-4 | -7.08e-4 |
| 105 | 3.00e-2 | 2 | 2.44e-2 | 2.49e-2 | 2.47e-2 | 2.44e-2 | 184 | -1.10e-4 | +1.19e-4 | +4.38e-6 | -5.73e-4 |
| 106 | 3.00e-2 | 1 | 2.76e-2 | 2.76e-2 | 2.76e-2 | 2.76e-2 | 243 | +4.95e-4 | +4.95e-4 | +4.95e-4 | -4.67e-4 |
| 107 | 3.00e-2 | 1 | 2.75e-2 | 2.75e-2 | 2.75e-2 | 2.75e-2 | 221 | -1.10e-5 | -1.10e-5 | -1.10e-5 | -4.21e-4 |
| 108 | 3.00e-2 | 2 | 2.73e-2 | 2.80e-2 | 2.76e-2 | 2.73e-2 | 176 | -1.50e-4 | +8.05e-5 | -3.45e-5 | -3.49e-4 |
| 109 | 3.00e-2 | 1 | 2.94e-2 | 2.94e-2 | 2.94e-2 | 2.94e-2 | 222 | +3.39e-4 | +3.39e-4 | +3.39e-4 | -2.80e-4 |
| 110 | 3.00e-2 | 1 | 2.89e-2 | 2.89e-2 | 2.89e-2 | 2.89e-2 | 210 | -8.03e-5 | -8.03e-5 | -8.03e-5 | -2.60e-4 |
| 111 | 3.00e-2 | 2 | 2.81e-2 | 2.82e-2 | 2.81e-2 | 2.82e-2 | 165 | -1.56e-4 | +1.96e-5 | -6.84e-5 | -2.23e-4 |
| 112 | 3.00e-2 | 2 | 2.86e-2 | 2.96e-2 | 2.91e-2 | 2.86e-2 | 163 | -2.13e-4 | +2.48e-4 | +1.74e-5 | -1.79e-4 |
| 113 | 3.00e-2 | 1 | 3.04e-2 | 3.04e-2 | 3.04e-2 | 3.04e-2 | 186 | +3.29e-4 | +3.29e-4 | +3.29e-4 | -1.29e-4 |
| 114 | 3.00e-2 | 2 | 2.97e-2 | 3.04e-2 | 3.01e-2 | 3.04e-2 | 163 | -1.42e-4 | +1.56e-4 | +7.01e-6 | -1.01e-4 |
| 115 | 3.00e-2 | 1 | 3.22e-2 | 3.22e-2 | 3.22e-2 | 3.22e-2 | 193 | +2.88e-4 | +2.88e-4 | +2.88e-4 | -6.23e-5 |
| 116 | 3.00e-2 | 2 | 3.15e-2 | 3.25e-2 | 3.20e-2 | 3.15e-2 | 163 | -1.92e-4 | +5.56e-5 | -6.82e-5 | -6.47e-5 |
| 117 | 3.00e-2 | 2 | 3.20e-2 | 3.38e-2 | 3.29e-2 | 3.20e-2 | 163 | -3.39e-4 | +3.35e-4 | -2.13e-6 | -5.62e-5 |
| 118 | 3.00e-2 | 1 | 3.17e-2 | 3.17e-2 | 3.17e-2 | 3.17e-2 | 168 | -5.47e-5 | -5.47e-5 | -5.47e-5 | -5.60e-5 |
| 119 | 3.00e-2 | 1 | 3.35e-2 | 3.35e-2 | 3.35e-2 | 3.35e-2 | 175 | +3.30e-4 | +3.30e-4 | +3.30e-4 | -1.74e-5 |
| 120 | 3.00e-2 | 3 | 3.28e-2 | 3.39e-2 | 3.32e-2 | 3.29e-2 | 157 | -2.25e-4 | +5.97e-5 | -4.60e-5 | -2.54e-5 |
| 121 | 3.00e-2 | 1 | 3.66e-2 | 3.66e-2 | 3.66e-2 | 3.66e-2 | 192 | +5.49e-4 | +5.49e-4 | +5.49e-4 | +3.20e-5 |
| 122 | 3.00e-2 | 1 | 3.56e-2 | 3.56e-2 | 3.56e-2 | 3.56e-2 | 184 | -1.50e-4 | -1.50e-4 | -1.50e-4 | +1.38e-5 |
| 123 | 3.00e-2 | 2 | 3.42e-2 | 3.51e-2 | 3.46e-2 | 3.42e-2 | 141 | -2.00e-4 | -7.56e-5 | -1.38e-4 | -1.56e-5 |
| 124 | 3.00e-2 | 2 | 3.44e-2 | 3.85e-2 | 3.64e-2 | 3.44e-2 | 141 | -7.99e-4 | +6.10e-4 | -9.48e-5 | -3.77e-5 |
| 125 | 3.00e-2 | 2 | 3.48e-2 | 3.58e-2 | 3.53e-2 | 3.48e-2 | 141 | -2.04e-4 | +2.57e-4 | +2.65e-5 | -2.78e-5 |
| 126 | 3.00e-2 | 2 | 3.62e-2 | 3.71e-2 | 3.66e-2 | 3.62e-2 | 141 | -1.78e-4 | +3.89e-4 | +1.06e-4 | -5.31e-6 |
| 127 | 3.00e-2 | 2 | 3.53e-2 | 3.72e-2 | 3.63e-2 | 3.53e-2 | 141 | -3.75e-4 | +1.72e-4 | -1.01e-4 | -2.63e-5 |
| 128 | 3.00e-2 | 1 | 4.16e-2 | 4.16e-2 | 4.16e-2 | 4.16e-2 | 208 | +7.83e-4 | +7.83e-4 | +7.83e-4 | +5.47e-5 |
| 129 | 3.00e-2 | 2 | 3.78e-2 | 4.01e-2 | 3.90e-2 | 3.78e-2 | 131 | -4.40e-4 | -2.19e-4 | -3.30e-4 | -1.95e-5 |
| 130 | 3.00e-2 | 2 | 3.64e-2 | 4.01e-2 | 3.83e-2 | 3.64e-2 | 117 | -8.31e-4 | +3.66e-4 | -2.32e-4 | -6.59e-5 |
| 131 | 3.00e-2 | 2 | 3.51e-2 | 3.72e-2 | 3.61e-2 | 3.51e-2 | 117 | -4.99e-4 | +1.57e-4 | -1.71e-4 | -8.91e-5 |
| 132 | 3.00e-2 | 2 | 3.67e-2 | 4.14e-2 | 3.91e-2 | 3.67e-2 | 123 | -9.78e-4 | +1.00e-3 | +1.11e-5 | -7.99e-5 |
| 133 | 3.00e-2 | 3 | 3.71e-2 | 3.96e-2 | 3.82e-2 | 3.71e-2 | 123 | -3.85e-4 | +4.96e-4 | -1.44e-5 | -6.82e-5 |
| 134 | 3.00e-2 | 1 | 4.12e-2 | 4.12e-2 | 4.12e-2 | 4.12e-2 | 158 | +6.64e-4 | +6.64e-4 | +6.64e-4 | +5.08e-6 |
| 135 | 3.00e-2 | 2 | 3.89e-2 | 4.28e-2 | 4.08e-2 | 3.89e-2 | 116 | -8.31e-4 | +2.32e-4 | -2.99e-4 | -5.81e-5 |
| 136 | 3.00e-2 | 3 | 3.74e-2 | 4.04e-2 | 3.86e-2 | 3.79e-2 | 107 | -7.18e-4 | +2.92e-4 | -9.91e-5 | -7.04e-5 |
| 137 | 3.00e-2 | 2 | 3.77e-2 | 4.05e-2 | 3.91e-2 | 3.77e-2 | 107 | -6.66e-4 | +4.90e-4 | -8.78e-5 | -7.95e-5 |
| 138 | 3.00e-2 | 3 | 3.85e-2 | 4.23e-2 | 4.00e-2 | 3.85e-2 | 106 | -7.08e-4 | +7.94e-4 | -2.84e-5 | -7.45e-5 |
| 139 | 3.00e-2 | 2 | 3.69e-2 | 4.38e-2 | 4.03e-2 | 3.69e-2 | 90 | -1.88e-3 | +9.19e-4 | -4.81e-4 | -1.66e-4 |
| 140 | 3.00e-2 | 3 | 3.80e-2 | 4.25e-2 | 4.00e-2 | 3.80e-2 | 90 | -7.24e-4 | +1.05e-3 | -3.94e-5 | -1.45e-4 |
| 141 | 3.00e-2 | 3 | 3.71e-2 | 4.15e-2 | 3.88e-2 | 3.71e-2 | 90 | -1.05e-3 | +6.75e-4 | -1.88e-4 | -1.65e-4 |
| 142 | 3.00e-2 | 2 | 3.85e-2 | 4.32e-2 | 4.09e-2 | 3.85e-2 | 90 | -1.28e-3 | +1.17e-3 | -5.41e-5 | -1.56e-4 |
| 143 | 3.00e-2 | 3 | 3.83e-2 | 4.64e-2 | 4.15e-2 | 3.83e-2 | 90 | -1.72e-3 | +1.27e-3 | -2.86e-4 | -2.06e-4 |
| 144 | 3.00e-2 | 3 | 3.81e-2 | 4.37e-2 | 4.00e-2 | 3.81e-2 | 87 | -1.67e-3 | +1.09e-3 | -1.85e-4 | -2.10e-4 |
| 145 | 3.00e-2 | 3 | 3.54e-2 | 4.41e-2 | 3.89e-2 | 3.54e-2 | 69 | -2.31e-3 | +1.19e-3 | -6.16e-4 | -3.37e-4 |
| 146 | 3.00e-2 | 5 | 3.49e-2 | 4.22e-2 | 3.70e-2 | 3.49e-2 | 67 | -2.29e-3 | +1.66e-3 | -2.22e-4 | -3.06e-4 |
| 147 | 3.00e-2 | 3 | 3.37e-2 | 4.07e-2 | 3.65e-2 | 3.37e-2 | 59 | -2.47e-3 | +1.45e-3 | -5.45e-4 | -3.89e-4 |
| 148 | 3.00e-2 | 4 | 3.45e-2 | 3.91e-2 | 3.58e-2 | 3.47e-2 | 60 | -2.12e-3 | +1.49e-3 | -1.25e-4 | -3.08e-4 |
| 149 | 3.00e-3 | 5 | 3.11e-2 | 4.22e-2 | 3.50e-2 | 3.11e-2 | 55 | -2.69e-3 | +1.91e-3 | -6.78e-4 | -4.84e-4 |
| 150 | 3.00e-3 | 1 | 1.62e-2 | 1.62e-2 | 1.62e-2 | 1.62e-2 | 55 | -1.18e-2 | -1.18e-2 | -1.18e-2 | -1.62e-3 |
| 151 | 3.00e-3 | 1 | 9.90e-3 | 9.90e-3 | 9.90e-3 | 9.90e-3 | 336 | -1.47e-3 | -1.47e-3 | -1.47e-3 | -1.60e-3 |
| 152 | 3.00e-3 | 1 | 7.81e-3 | 7.81e-3 | 7.81e-3 | 7.81e-3 | 306 | -7.73e-4 | -7.73e-4 | -7.73e-4 | -1.52e-3 |
| 153 | 3.00e-3 | 1 | 7.06e-3 | 7.06e-3 | 7.06e-3 | 7.06e-3 | 284 | -3.56e-4 | -3.56e-4 | -3.56e-4 | -1.40e-3 |
| 154 | 3.00e-3 | 1 | 7.20e-3 | 7.20e-3 | 7.20e-3 | 7.20e-3 | 305 | +6.49e-5 | +6.49e-5 | +6.49e-5 | -1.26e-3 |
| 155 | 3.00e-3 | 1 | 7.10e-3 | 7.10e-3 | 7.10e-3 | 7.10e-3 | 298 | -4.63e-5 | -4.63e-5 | -4.63e-5 | -1.14e-3 |
| 157 | 3.00e-3 | 2 | 7.01e-3 | 7.46e-3 | 7.24e-3 | 7.01e-3 | 275 | -2.26e-4 | +1.45e-4 | -4.02e-5 | -9.30e-4 |
| 159 | 3.00e-3 | 1 | 7.82e-3 | 7.82e-3 | 7.82e-3 | 7.82e-3 | 362 | +3.02e-4 | +3.02e-4 | +3.02e-4 | -8.07e-4 |
| 160 | 3.00e-3 | 1 | 6.93e-3 | 6.93e-3 | 6.93e-3 | 6.93e-3 | 259 | -4.67e-4 | -4.67e-4 | -4.67e-4 | -7.73e-4 |
| 161 | 3.00e-3 | 1 | 7.10e-3 | 7.10e-3 | 7.10e-3 | 7.10e-3 | 332 | +7.14e-5 | +7.14e-5 | +7.14e-5 | -6.88e-4 |
| 162 | 3.00e-3 | 1 | 7.31e-3 | 7.31e-3 | 7.31e-3 | 7.31e-3 | 303 | +9.55e-5 | +9.55e-5 | +9.55e-5 | -6.10e-4 |
| 163 | 3.00e-3 | 1 | 7.32e-3 | 7.32e-3 | 7.32e-3 | 7.32e-3 | 296 | +5.53e-6 | +5.53e-6 | +5.53e-6 | -5.48e-4 |
| 164 | 3.00e-3 | 1 | 7.44e-3 | 7.44e-3 | 7.44e-3 | 7.44e-3 | 296 | +5.54e-5 | +5.54e-5 | +5.54e-5 | -4.88e-4 |
| 165 | 3.00e-3 | 1 | 7.26e-3 | 7.26e-3 | 7.26e-3 | 7.26e-3 | 299 | -8.25e-5 | -8.25e-5 | -8.25e-5 | -4.47e-4 |
| 166 | 3.00e-3 | 1 | 7.21e-3 | 7.21e-3 | 7.21e-3 | 7.21e-3 | 292 | -2.12e-5 | -2.12e-5 | -2.12e-5 | -4.05e-4 |
| 167 | 3.00e-3 | 1 | 7.14e-3 | 7.14e-3 | 7.14e-3 | 7.14e-3 | 279 | -3.89e-5 | -3.89e-5 | -3.89e-5 | -3.68e-4 |
| 168 | 3.00e-3 | 1 | 7.04e-3 | 7.04e-3 | 7.04e-3 | 7.04e-3 | 269 | -5.13e-5 | -5.13e-5 | -5.13e-5 | -3.37e-4 |
| 170 | 3.00e-3 | 2 | 7.48e-3 | 7.69e-3 | 7.59e-3 | 7.48e-3 | 290 | -9.45e-5 | +2.45e-4 | +7.54e-5 | -2.60e-4 |
| 172 | 3.00e-3 | 1 | 8.11e-3 | 8.11e-3 | 8.11e-3 | 8.11e-3 | 379 | +2.11e-4 | +2.11e-4 | +2.11e-4 | -2.13e-4 |
| 173 | 3.00e-3 | 1 | 7.82e-3 | 7.82e-3 | 7.82e-3 | 7.82e-3 | 322 | -1.13e-4 | -1.13e-4 | -1.13e-4 | -2.03e-4 |
| 174 | 3.00e-3 | 1 | 7.74e-3 | 7.74e-3 | 7.74e-3 | 7.74e-3 | 302 | -3.07e-5 | -3.07e-5 | -3.07e-5 | -1.86e-4 |
| 175 | 3.00e-3 | 1 | 7.70e-3 | 7.70e-3 | 7.70e-3 | 7.70e-3 | 291 | -1.85e-5 | -1.85e-5 | -1.85e-5 | -1.69e-4 |
| 177 | 3.00e-3 | 2 | 7.54e-3 | 8.62e-3 | 8.08e-3 | 7.54e-3 | 261 | -5.10e-4 | +2.92e-4 | -1.09e-4 | -1.61e-4 |
| 179 | 3.00e-3 | 2 | 7.55e-3 | 8.20e-3 | 7.88e-3 | 7.55e-3 | 289 | -2.86e-4 | +2.39e-4 | -2.33e-5 | -1.38e-4 |
| 181 | 3.00e-3 | 1 | 8.35e-3 | 8.35e-3 | 8.35e-3 | 8.35e-3 | 337 | +2.96e-4 | +2.96e-4 | +2.96e-4 | -9.44e-5 |
| 182 | 3.00e-3 | 1 | 7.79e-3 | 7.79e-3 | 7.79e-3 | 7.79e-3 | 275 | -2.53e-4 | -2.53e-4 | -2.53e-4 | -1.10e-4 |
| 183 | 3.00e-3 | 1 | 7.17e-3 | 7.17e-3 | 7.17e-3 | 7.17e-3 | 232 | -3.56e-4 | -3.56e-4 | -3.56e-4 | -1.35e-4 |
| 184 | 3.00e-3 | 2 | 6.98e-3 | 7.18e-3 | 7.08e-3 | 6.98e-3 | 225 | -1.28e-4 | +6.27e-6 | -6.07e-5 | -1.21e-4 |
| 185 | 3.00e-3 | 1 | 7.28e-3 | 7.28e-3 | 7.28e-3 | 7.28e-3 | 244 | +1.76e-4 | +1.76e-4 | +1.76e-4 | -9.17e-5 |
| 186 | 3.00e-3 | 1 | 7.51e-3 | 7.51e-3 | 7.51e-3 | 7.51e-3 | 264 | +1.16e-4 | +1.16e-4 | +1.16e-4 | -7.10e-5 |
| 187 | 3.00e-3 | 1 | 7.65e-3 | 7.65e-3 | 7.65e-3 | 7.65e-3 | 274 | +6.89e-5 | +6.89e-5 | +6.89e-5 | -5.70e-5 |
| 188 | 3.00e-3 | 1 | 7.98e-3 | 7.98e-3 | 7.98e-3 | 7.98e-3 | 299 | +1.41e-4 | +1.41e-4 | +1.41e-4 | -3.72e-5 |
| 190 | 3.00e-3 | 2 | 7.83e-3 | 8.44e-3 | 8.13e-3 | 7.83e-3 | 260 | -2.90e-4 | +1.61e-4 | -6.43e-5 | -4.46e-5 |
| 192 | 3.00e-3 | 2 | 7.61e-3 | 8.52e-3 | 8.06e-3 | 7.61e-3 | 261 | -4.35e-4 | +2.60e-4 | -8.77e-5 | -5.63e-5 |
| 194 | 3.00e-3 | 2 | 7.39e-3 | 7.79e-3 | 7.59e-3 | 7.39e-3 | 228 | -2.30e-4 | +8.34e-5 | -7.31e-5 | -6.10e-5 |
| 195 | 3.00e-3 | 1 | 7.65e-3 | 7.65e-3 | 7.65e-3 | 7.65e-3 | 248 | +1.36e-4 | +1.36e-4 | +1.36e-4 | -4.13e-5 |
| 196 | 3.00e-3 | 1 | 7.58e-3 | 7.58e-3 | 7.58e-3 | 7.58e-3 | 250 | -3.34e-5 | -3.34e-5 | -3.34e-5 | -4.05e-5 |
| 197 | 3.00e-3 | 1 | 7.73e-3 | 7.73e-3 | 7.73e-3 | 7.73e-3 | 263 | +7.36e-5 | +7.36e-5 | +7.36e-5 | -2.91e-5 |
| 198 | 3.00e-3 | 1 | 8.00e-3 | 8.00e-3 | 8.00e-3 | 8.00e-3 | 280 | +1.22e-4 | +1.22e-4 | +1.22e-4 | -1.40e-5 |
| 199 | 3.00e-3 | 1 | 7.69e-3 | 7.69e-3 | 7.69e-3 | 7.69e-3 | 247 | -1.62e-4 | -1.62e-4 | -1.62e-4 | -2.87e-5 |

