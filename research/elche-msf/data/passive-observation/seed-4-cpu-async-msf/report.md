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
| cpu-async | 0.059510 | 0.9177 | +0.0052 | 1746.7 | 467 | 78.5 | 100% | 100% | 14.6 |

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
| cpu-async | 2.0386 | 0.7812 | 0.5745 | 0.5217 | 0.5302 | 0.5200 | 0.4982 | 0.4861 | 0.4836 | 0.4768 | 0.2147 | 0.1790 | 0.1572 | 0.1469 | 0.1308 | 0.0755 | 0.0691 | 0.0663 | 0.0617 | 0.0595 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | cpu-async | 0 | cuda:0 | 0.4019 | 2.8 |
| resnet-graph | cpu-async | 1 | cuda:1 | 0.3030 | 3.9 |
| resnet-graph | cpu-async | 2 | cuda:2 | 0.2951 | 3.6 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|
| resnet-graph | cpu-async | 358 | 357 | 396 | 386 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| cpu-async | gpu2 | 1334.9 | 3.4 | epoch-boundary(152) |
| cpu-async | gpu1 | 1735.6 | 2.1 | epoch-boundary(198) |
| cpu-async | gpu2 | 325.3 | 1.4 | epoch-boundary(36) |
| cpu-async | gpu1 | 325.2 | 1.4 | epoch-boundary(36) |
| cpu-async | gpu2 | 1422.7 | 1.0 | epoch-boundary(162) |
| cpu-async | gpu1 | 811.6 | 1.0 | epoch-boundary(92) |
| cpu-async | gpu1 | 1422.8 | 0.7 | epoch-boundary(162) |
| cpu-async | gpu0 | 0.4 | 0.7 | cpu-avg |
| cpu-async | gpu1 | 586.2 | 0.7 | epoch-boundary(66) |
| cpu-async | gpu2 | 586.2 | 0.5 | epoch-boundary(66) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | cpu-async | gpu0 | 0.0s | 0.0s | 0.7s | 0.0s | 0.7s |
| resnet-graph | cpu-async | gpu1 | 5.8s | 0.0s | 0.0s | 0.0s | 6.9s |
| resnet-graph | cpu-async | gpu2 | 6.3s | 0.0s | 0.0s | 0.0s | 6.9s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | cpu-async | 291 | 0 | 467 | 78.5 | 2307/9078 | 467 | 78.5 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | cpu-async | 186.4 | 10.7% |

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
| resnet-graph | cpu-async | 187 | 467 | 0 | 7.66e-3 | -2.72e-5 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | cpu-async | 0 | 467 | 9.67e-2 | 6.60e-2 | 2.41e-3 | 4.08e-1 | 27.0 | -1.95e-4 | 1.13e-3 |
| resnet-graph | cpu-async | 1 | 467 | 9.75e-2 | 6.69e-2 | 2.64e-3 | 3.89e-1 | 35.5 | -2.20e-4 | 1.32e-3 |
| resnet-graph | cpu-async | 2 | 467 | 9.82e-2 | 6.81e-2 | 2.71e-3 | 4.36e-1 | 37.5 | -2.27e-4 | 1.38e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | cpu-async | rank0 ↔ rank1 | +0.9947 |
| resnet-graph | cpu-async | rank0 ↔ rank2 | +0.9913 |
| resnet-graph | cpu-async | rank1 ↔ rank2 | +0.9914 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | cpu-async | 51 (0,1,4,5,6,7,8,9…143,145) | 0 (—) | — | 0,1,4,5,6,7,8,9…143,145 | — |

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
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 293 | +0.110 |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | 112 | +0.023 |
| resnet-graph | cpu-async | 3.00e-3 | 150–198 | 58 | +0.312 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | cpu-async | 465 | +0.027 | 187 | +0.379 | +0.398 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | cpu-async | 466 | 3.38e1–7.97e1 | 6.53e1 | 2.61e-3 | 3.51e-3 | 2.69e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 295 | 36–78059 | +9.953e-6 | 0.473 | +1.031e-5 | 0.522 | 96 | +1.011e-5 | 0.660 | 32–1042 | +9.427e-4 | 0.753 |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 285 | 913–78059 | +1.112e-5 | 0.650 | +1.137e-5 | 0.686 | 95 | +1.046e-5 | 0.687 | 82–1042 | +9.929e-4 | 0.925 |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | 113 | 78619–117089 | +2.700e-6 | 0.026 | +2.431e-6 | 0.022 | 49 | +2.559e-6 | 0.015 | 122–722 | -7.100e-5 | 0.003 |
| resnet-graph | cpu-async | 3.00e-3 | 150–198 | 59 | 117291–155431 | +9.079e-6 | 0.077 | +9.527e-6 | 0.087 | 42 | +6.727e-6 | 0.207 | 109–1040 | +6.090e-4 | 0.168 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | +9.427e-4 | r0: +9.367e-4, r1: +9.591e-4, r2: +9.345e-4 | r0: 0.764, r1: 0.756, r2: 0.706 | 1.03× | ✓ |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +9.929e-4 | r0: +9.843e-4, r1: +1.005e-3, r2: +9.908e-4 | r0: 0.926, r1: 0.906, r2: 0.900 | 1.02× | ✓ |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | -7.100e-5 | r0: -1.048e-4, r1: -3.192e-5, r2: -7.497e-5 | r0: 0.007, r1: 0.001, r2: 0.004 | 3.28× | ⚠ framing breaking |
| resnet-graph | cpu-async | 3.00e-3 | 150–198 | +6.090e-4 | r0: +6.071e-4, r1: +6.130e-4, r2: +6.104e-4 | r0: 0.164, r1: 0.176, r2: 0.164 | 1.01× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | cpu-async | `█▇▇▇▇▇▇▇▇███████████████▅▄▄▄▄▄▄▅▅▅▅▅▂▁▁▁▁▁▁▁▁▁▁` | `▁▇▇▇▇▇▇▇▇███████████████▇▇█████████▇▃▅▆▇▇▇█████` |

### Per-Epoch Detail: resnet-graph / cpu-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 10 | 1.08e-1 | 4.36e-1 | 2.17e-1 | 1.61e-1 | 48 | -3.53e-2 | +6.49e-3 | -1.01e-2 | -7.78e-3 |
| 1 | 3.00e-1 | 6 | 9.92e-2 | 1.94e-1 | 1.29e-1 | 9.92e-2 | 37 | -7.19e-3 | +2.02e-3 | -2.51e-3 | -4.65e-3 |
| 2 | 3.00e-1 | 6 | 1.05e-1 | 1.29e-1 | 1.13e-1 | 1.11e-1 | 46 | -3.11e-3 | +3.82e-3 | +4.19e-6 | -2.22e-3 |
| 3 | 3.00e-1 | 6 | 1.08e-1 | 1.43e-1 | 1.17e-1 | 1.08e-1 | 37 | -4.57e-3 | +3.10e-3 | -5.80e-4 | -1.41e-3 |
| 4 | 3.00e-1 | 7 | 1.07e-1 | 1.48e-1 | 1.18e-1 | 1.14e-1 | 42 | -5.13e-3 | +4.07e-3 | -4.00e-4 | -8.71e-4 |
| 5 | 3.00e-1 | 7 | 1.09e-1 | 1.51e-1 | 1.18e-1 | 1.13e-1 | 39 | -5.98e-3 | +3.54e-3 | -5.14e-4 | -6.33e-4 |
| 6 | 3.00e-1 | 7 | 1.06e-1 | 1.45e-1 | 1.14e-1 | 1.13e-1 | 38 | -7.78e-3 | +3.16e-3 | -6.11e-4 | -5.32e-4 |
| 7 | 3.00e-1 | 6 | 1.09e-1 | 1.51e-1 | 1.19e-1 | 1.09e-1 | 36 | -6.54e-3 | +3.45e-3 | -8.54e-4 | -6.78e-4 |
| 8 | 3.00e-1 | 8 | 1.07e-1 | 1.54e-1 | 1.18e-1 | 1.13e-1 | 33 | -8.05e-3 | +4.13e-3 | -7.16e-4 | -6.20e-4 |
| 9 | 3.00e-1 | 6 | 1.09e-1 | 1.46e-1 | 1.22e-1 | 1.22e-1 | 46 | -5.10e-3 | +3.90e-3 | -3.08e-4 | -4.61e-4 |
| 10 | 3.00e-1 | 8 | 1.06e-1 | 1.53e-1 | 1.19e-1 | 1.06e-1 | 34 | -4.64e-3 | +2.84e-3 | -7.74e-4 | -6.49e-4 |
| 11 | 3.00e-1 | 5 | 1.07e-1 | 1.47e-1 | 1.19e-1 | 1.07e-1 | 37 | -3.73e-3 | +4.07e-3 | -5.91e-4 | -7.04e-4 |
| 12 | 3.00e-1 | 6 | 1.07e-1 | 1.44e-1 | 1.16e-1 | 1.13e-1 | 38 | -8.52e-3 | +4.09e-3 | -5.46e-4 | -5.97e-4 |
| 13 | 3.00e-1 | 7 | 1.11e-1 | 1.47e-1 | 1.19e-1 | 1.17e-1 | 41 | -6.64e-3 | +3.73e-3 | -3.30e-4 | -4.17e-4 |
| 14 | 3.00e-1 | 6 | 1.11e-1 | 1.51e-1 | 1.22e-1 | 1.11e-1 | 42 | -4.06e-3 | +2.82e-3 | -6.84e-4 | -5.46e-4 |
| 15 | 3.00e-1 | 10 | 9.66e-2 | 1.51e-1 | 1.11e-1 | 1.13e-1 | 39 | -6.99e-3 | +3.53e-3 | -4.67e-4 | -3.20e-4 |
| 16 | 3.00e-1 | 3 | 1.13e-1 | 1.53e-1 | 1.29e-1 | 1.13e-1 | 41 | -5.68e-3 | +3.74e-3 | -1.24e-3 | -6.19e-4 |
| 17 | 3.00e-1 | 6 | 1.16e-1 | 1.43e-1 | 1.23e-1 | 1.16e-1 | 45 | -3.43e-3 | +2.96e-3 | -3.38e-4 | -4.99e-4 |
| 18 | 3.00e-1 | 6 | 1.03e-1 | 1.56e-1 | 1.20e-1 | 1.03e-1 | 34 | -5.81e-3 | +3.17e-3 | -1.24e-3 | -8.80e-4 |
| 19 | 3.00e-1 | 9 | 9.59e-2 | 1.49e-1 | 1.09e-1 | 1.08e-1 | 38 | -9.32e-3 | +4.41e-3 | -6.75e-4 | -5.86e-4 |
| 20 | 3.00e-1 | 6 | 1.05e-1 | 1.48e-1 | 1.15e-1 | 1.06e-1 | 35 | -7.68e-3 | +3.85e-3 | -9.58e-4 | -7.38e-4 |
| 21 | 3.00e-1 | 9 | 1.02e-1 | 1.43e-1 | 1.09e-1 | 1.07e-1 | 34 | -7.25e-3 | +4.42e-3 | -4.65e-4 | -4.69e-4 |
| 22 | 3.00e-1 | 5 | 1.03e-1 | 1.47e-1 | 1.16e-1 | 1.08e-1 | 37 | -8.10e-3 | +4.75e-3 | -9.91e-4 | -7.07e-4 |
| 23 | 3.00e-1 | 8 | 9.86e-2 | 1.47e-1 | 1.13e-1 | 1.24e-1 | 49 | -1.00e-2 | +4.13e-3 | -5.43e-4 | -4.08e-4 |
| 24 | 3.00e-1 | 4 | 1.21e-1 | 1.67e-1 | 1.36e-1 | 1.26e-1 | 50 | -4.47e-3 | +3.21e-3 | -5.93e-4 | -4.87e-4 |
| 25 | 3.00e-1 | 6 | 1.11e-1 | 1.52e-1 | 1.21e-1 | 1.11e-1 | 40 | -6.29e-3 | +2.08e-3 | -9.98e-4 | -6.99e-4 |
| 26 | 3.00e-1 | 8 | 1.09e-1 | 1.60e-1 | 1.21e-1 | 1.15e-1 | 44 | -5.56e-3 | +3.89e-3 | -5.19e-4 | -5.44e-4 |
| 27 | 3.00e-1 | 4 | 1.17e-1 | 1.49e-1 | 1.28e-1 | 1.17e-1 | 44 | -3.83e-3 | +3.29e-3 | -5.65e-4 | -5.90e-4 |
| 28 | 3.00e-1 | 6 | 1.10e-1 | 1.49e-1 | 1.20e-1 | 1.10e-1 | 36 | -6.55e-3 | +3.11e-3 | -8.29e-4 | -7.21e-4 |
| 29 | 3.00e-1 | 6 | 9.93e-2 | 1.49e-1 | 1.21e-1 | 1.34e-1 | 56 | -9.46e-3 | +3.90e-3 | -4.21e-4 | -4.53e-4 |
| 30 | 3.00e-1 | 6 | 1.11e-1 | 1.59e-1 | 1.24e-1 | 1.11e-1 | 39 | -4.73e-3 | +1.93e-3 | -9.79e-4 | -6.82e-4 |
| 31 | 3.00e-1 | 6 | 1.05e-1 | 1.56e-1 | 1.19e-1 | 1.13e-1 | 40 | -7.89e-3 | +3.84e-3 | -8.39e-4 | -7.04e-4 |
| 32 | 3.00e-1 | 6 | 1.11e-1 | 1.52e-1 | 1.34e-1 | 1.41e-1 | 68 | -5.31e-3 | +3.82e-3 | -9.33e-5 | -3.94e-4 |
| 33 | 3.00e-1 | 3 | 1.34e-1 | 1.61e-1 | 1.44e-1 | 1.34e-1 | 62 | -2.53e-3 | +1.45e-3 | -5.19e-4 | -4.45e-4 |
| 34 | 3.00e-1 | 4 | 1.33e-1 | 1.71e-1 | 1.44e-1 | 1.33e-1 | 58 | -3.93e-3 | +2.14e-3 | -5.40e-4 | -4.89e-4 |
| 35 | 3.00e-1 | 4 | 1.28e-1 | 1.64e-1 | 1.38e-1 | 1.28e-1 | 54 | -3.93e-3 | +2.07e-3 | -6.33e-4 | -5.50e-4 |
| 36 | 3.00e-1 | 2 | 1.28e-1 | 2.16e-1 | 1.72e-1 | 2.16e-1 | 271 | -5.19e-6 | +1.94e-3 | +9.68e-4 | -2.52e-4 |
| 38 | 3.00e-1 | 2 | 2.22e-1 | 2.51e-1 | 2.36e-1 | 2.22e-1 | 285 | -4.36e-4 | +3.83e-4 | -2.62e-5 | -2.13e-4 |
| 40 | 3.00e-1 | 1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 308 | +3.84e-5 | +3.84e-5 | +3.84e-5 | -1.88e-4 |
| 41 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 291 | -7.71e-5 | -7.71e-5 | -7.71e-5 | -1.77e-4 |
| 42 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 278 | -1.23e-4 | -1.23e-4 | -1.23e-4 | -1.71e-4 |
| 43 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 283 | +7.76e-5 | +7.76e-5 | +7.76e-5 | -1.46e-4 |
| 44 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 271 | -8.05e-5 | -8.05e-5 | -8.05e-5 | -1.40e-4 |
| 45 | 3.00e-1 | 1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 322 | +2.17e-4 | +2.17e-4 | +2.17e-4 | -1.04e-4 |
| 46 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 295 | -1.37e-4 | -1.37e-4 | -1.37e-4 | -1.07e-4 |
| 47 | 3.00e-1 | 1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 317 | +1.05e-4 | +1.05e-4 | +1.05e-4 | -8.62e-5 |
| 48 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 289 | -1.37e-4 | -1.37e-4 | -1.37e-4 | -9.12e-5 |
| 49 | 3.00e-1 | 1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 290 | +7.70e-5 | +7.70e-5 | +7.70e-5 | -7.44e-5 |
| 50 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 244 | -2.24e-4 | -2.24e-4 | -2.24e-4 | -8.94e-5 |
| 51 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 251 | -3.28e-5 | -3.28e-5 | -3.28e-5 | -8.37e-5 |
| 52 | 3.00e-1 | 1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 292 | +2.00e-4 | +2.00e-4 | +2.00e-4 | -5.54e-5 |
| 53 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 261 | -1.33e-4 | -1.33e-4 | -1.33e-4 | -6.32e-5 |
| 54 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 284 | +3.72e-5 | +3.72e-5 | +3.72e-5 | -5.31e-5 |
| 55 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 266 | -7.87e-5 | -7.87e-5 | -7.87e-5 | -5.57e-5 |
| 56 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 263 | +8.38e-6 | +8.38e-6 | +8.38e-6 | -4.93e-5 |
| 57 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 302 | +1.28e-4 | +1.28e-4 | +1.28e-4 | -3.16e-5 |
| 58 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 269 | -9.64e-5 | -9.64e-5 | -9.64e-5 | -3.81e-5 |
| 59 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 257 | -9.00e-5 | -9.00e-5 | -9.00e-5 | -4.33e-5 |
| 60 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 275 | +9.44e-5 | +9.44e-5 | +9.44e-5 | -2.95e-5 |
| 61 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 296 | +3.15e-5 | +3.15e-5 | +3.15e-5 | -2.34e-5 |
| 62 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 296 | +2.57e-5 | +2.57e-5 | +2.57e-5 | -1.85e-5 |
| 64 | 3.00e-1 | 2 | 2.11e-1 | 2.32e-1 | 2.21e-1 | 2.11e-1 | 253 | -3.82e-4 | +1.67e-4 | -1.07e-4 | -3.81e-5 |
| 66 | 3.00e-1 | 2 | 2.12e-1 | 2.16e-1 | 2.14e-1 | 2.12e-1 | 253 | -8.98e-5 | +9.53e-5 | +2.73e-6 | -3.13e-5 |
| 67 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 269 | +8.33e-5 | +8.33e-5 | +8.33e-5 | -1.98e-5 |
| 68 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 254 | +3.04e-6 | +3.04e-6 | +3.04e-6 | -1.75e-5 |
| 69 | 3.00e-1 | 1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 307 | +1.30e-4 | +1.30e-4 | +1.30e-4 | -2.80e-6 |
| 70 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 249 | -2.37e-4 | -2.37e-4 | -2.37e-4 | -2.62e-5 |
| 71 | 3.00e-1 | 1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 299 | +2.24e-4 | +2.24e-4 | +2.24e-4 | -1.17e-6 |
| 72 | 3.00e-1 | 1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 278 | -4.88e-5 | -4.88e-5 | -4.88e-5 | -5.94e-6 |
| 73 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 233 | -3.11e-4 | -3.11e-4 | -3.11e-4 | -3.64e-5 |
| 74 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 239 | -9.79e-5 | -9.79e-5 | -9.79e-5 | -4.26e-5 |
| 75 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 250 | +1.03e-4 | +1.03e-4 | +1.03e-4 | -2.81e-5 |
| 76 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 234 | -1.82e-5 | -1.82e-5 | -1.82e-5 | -2.71e-5 |
| 77 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 255 | +4.72e-5 | +4.72e-5 | +4.72e-5 | -1.96e-5 |
| 78 | 3.00e-1 | 2 | 2.05e-1 | 2.12e-1 | 2.09e-1 | 2.05e-1 | 227 | -1.63e-4 | +3.41e-5 | -6.42e-5 | -2.91e-5 |
| 79 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 267 | +2.19e-4 | +2.19e-4 | +2.19e-4 | -4.29e-6 |
| 80 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 272 | +3.52e-5 | +3.52e-5 | +3.52e-5 | -3.39e-7 |
| 81 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 258 | -1.37e-4 | -1.37e-4 | -1.37e-4 | -1.40e-5 |
| 82 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 257 | +2.47e-5 | +2.47e-5 | +2.47e-5 | -1.01e-5 |
| 83 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 264 | +6.13e-6 | +6.13e-6 | +6.13e-6 | -8.51e-6 |
| 84 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 268 | +9.36e-5 | +9.36e-5 | +9.36e-5 | +1.70e-6 |
| 85 | 3.00e-1 | 1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 279 | +6.52e-5 | +6.52e-5 | +6.52e-5 | +8.05e-6 |
| 86 | 3.00e-1 | 2 | 1.99e-1 | 2.10e-1 | 2.04e-1 | 1.99e-1 | 209 | -2.74e-4 | -2.41e-4 | -2.58e-4 | -4.26e-5 |
| 87 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 242 | +2.31e-4 | +2.31e-4 | +2.31e-4 | -1.52e-5 |
| 88 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 221 | -8.66e-5 | -8.66e-5 | -8.66e-5 | -2.24e-5 |
| 89 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 232 | +6.97e-5 | +6.97e-5 | +6.97e-5 | -1.32e-5 |
| 90 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 221 | -8.00e-5 | -8.00e-5 | -8.00e-5 | -1.98e-5 |
| 91 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 241 | +1.08e-4 | +1.08e-4 | +1.08e-4 | -7.09e-6 |
| 92 | 3.00e-1 | 2 | 2.04e-1 | 2.15e-1 | 2.10e-1 | 2.04e-1 | 209 | -2.34e-4 | +7.02e-5 | -8.21e-5 | -2.29e-5 |
| 93 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 247 | +1.52e-4 | +1.52e-4 | +1.52e-4 | -5.41e-6 |
| 94 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 239 | +2.21e-5 | +2.21e-5 | +2.21e-5 | -2.66e-6 |
| 95 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 241 | +3.63e-5 | +3.63e-5 | +3.63e-5 | +1.24e-6 |
| 96 | 3.00e-1 | 2 | 1.97e-1 | 2.05e-1 | 2.01e-1 | 1.97e-1 | 185 | -2.31e-4 | -2.15e-4 | -2.23e-4 | -4.13e-5 |
| 97 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 242 | +3.39e-4 | +3.39e-4 | +3.39e-4 | -3.27e-6 |
| 98 | 3.00e-1 | 2 | 2.01e-1 | 2.04e-1 | 2.02e-1 | 2.01e-1 | 185 | -2.14e-4 | -6.69e-5 | -1.40e-4 | -2.86e-5 |
| 99 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 216 | +6.07e-5 | +6.07e-5 | +6.07e-5 | -1.97e-5 |
| 100 | 3.00e-2 | 1 | 1.32e-1 | 1.32e-1 | 1.32e-1 | 1.32e-1 | 208 | -2.10e-3 | -2.10e-3 | -2.10e-3 | -2.28e-4 |
| 101 | 3.00e-2 | 2 | 3.75e-2 | 6.45e-2 | 5.10e-2 | 3.75e-2 | 198 | -3.04e-3 | -2.75e-3 | -2.90e-3 | -7.33e-4 |
| 103 | 3.00e-2 | 2 | 2.50e-2 | 2.97e-2 | 2.73e-2 | 2.50e-2 | 198 | -8.81e-4 | -8.67e-4 | -8.74e-4 | -7.60e-4 |
| 104 | 3.00e-2 | 1 | 2.58e-2 | 2.58e-2 | 2.58e-2 | 2.58e-2 | 230 | +1.41e-4 | +1.41e-4 | +1.41e-4 | -6.70e-4 |
| 105 | 3.00e-2 | 2 | 2.53e-2 | 2.59e-2 | 2.56e-2 | 2.53e-2 | 184 | -1.13e-4 | +1.05e-5 | -5.11e-5 | -5.53e-4 |
| 106 | 3.00e-2 | 1 | 2.67e-2 | 2.67e-2 | 2.67e-2 | 2.67e-2 | 215 | +2.53e-4 | +2.53e-4 | +2.53e-4 | -4.72e-4 |
| 107 | 3.00e-2 | 1 | 2.72e-2 | 2.72e-2 | 2.72e-2 | 2.72e-2 | 212 | +8.29e-5 | +8.29e-5 | +8.29e-5 | -4.17e-4 |
| 108 | 3.00e-2 | 2 | 2.78e-2 | 2.83e-2 | 2.81e-2 | 2.78e-2 | 174 | -1.22e-4 | +1.83e-4 | +3.05e-5 | -3.33e-4 |
| 109 | 3.00e-2 | 1 | 2.88e-2 | 2.88e-2 | 2.88e-2 | 2.88e-2 | 202 | +1.89e-4 | +1.89e-4 | +1.89e-4 | -2.81e-4 |
| 110 | 3.00e-2 | 2 | 2.79e-2 | 3.05e-2 | 2.92e-2 | 2.79e-2 | 165 | -5.43e-4 | +2.50e-4 | -1.47e-4 | -2.59e-4 |
| 111 | 3.00e-2 | 1 | 3.04e-2 | 3.04e-2 | 3.04e-2 | 3.04e-2 | 214 | +4.00e-4 | +4.00e-4 | +4.00e-4 | -1.94e-4 |
| 112 | 3.00e-2 | 1 | 2.98e-2 | 2.98e-2 | 2.98e-2 | 2.98e-2 | 186 | -1.11e-4 | -1.11e-4 | -1.11e-4 | -1.85e-4 |
| 113 | 3.00e-2 | 2 | 2.76e-2 | 2.96e-2 | 2.86e-2 | 2.76e-2 | 145 | -4.82e-4 | -3.81e-5 | -2.60e-4 | -2.02e-4 |
| 114 | 3.00e-2 | 2 | 2.96e-2 | 3.08e-2 | 3.02e-2 | 2.96e-2 | 145 | -2.61e-4 | +5.88e-4 | +1.64e-4 | -1.37e-4 |
| 115 | 3.00e-2 | 1 | 3.15e-2 | 3.15e-2 | 3.15e-2 | 3.15e-2 | 182 | +3.39e-4 | +3.39e-4 | +3.39e-4 | -8.90e-5 |
| 116 | 3.00e-2 | 3 | 2.95e-2 | 3.46e-2 | 3.21e-2 | 2.95e-2 | 141 | -6.07e-4 | +4.62e-4 | -1.99e-4 | -1.29e-4 |
| 117 | 3.00e-2 | 1 | 3.42e-2 | 3.42e-2 | 3.42e-2 | 3.42e-2 | 195 | +7.56e-4 | +7.56e-4 | +7.56e-4 | -4.04e-5 |
| 118 | 3.00e-2 | 1 | 3.37e-2 | 3.37e-2 | 3.37e-2 | 3.37e-2 | 173 | -7.36e-5 | -7.36e-5 | -7.36e-5 | -4.37e-5 |
| 119 | 3.00e-2 | 2 | 3.04e-2 | 3.26e-2 | 3.15e-2 | 3.04e-2 | 130 | -5.20e-4 | -2.14e-4 | -3.67e-4 | -1.07e-4 |
| 120 | 3.00e-2 | 2 | 3.18e-2 | 3.53e-2 | 3.36e-2 | 3.18e-2 | 137 | -7.83e-4 | +7.90e-4 | +3.29e-6 | -9.37e-5 |
| 121 | 3.00e-2 | 2 | 3.33e-2 | 3.59e-2 | 3.46e-2 | 3.33e-2 | 142 | -5.16e-4 | +6.48e-4 | +6.59e-5 | -6.92e-5 |
| 122 | 3.00e-2 | 1 | 3.74e-2 | 3.74e-2 | 3.74e-2 | 3.74e-2 | 188 | +6.12e-4 | +6.12e-4 | +6.12e-4 | -1.06e-6 |
| 123 | 3.00e-2 | 2 | 3.54e-2 | 3.84e-2 | 3.69e-2 | 3.54e-2 | 155 | -5.32e-4 | +1.37e-4 | -1.98e-4 | -4.18e-5 |
| 124 | 3.00e-2 | 2 | 3.22e-2 | 3.57e-2 | 3.40e-2 | 3.22e-2 | 117 | -8.88e-4 | +6.89e-5 | -4.10e-4 | -1.16e-4 |
| 125 | 3.00e-2 | 2 | 3.31e-2 | 3.50e-2 | 3.41e-2 | 3.31e-2 | 116 | -4.89e-4 | +5.62e-4 | +3.63e-5 | -9.27e-5 |
| 126 | 3.00e-2 | 3 | 3.38e-2 | 3.72e-2 | 3.51e-2 | 3.45e-2 | 123 | -8.26e-4 | +7.29e-4 | +2.31e-5 | -6.62e-5 |
| 127 | 3.00e-2 | 1 | 3.57e-2 | 3.57e-2 | 3.57e-2 | 3.57e-2 | 142 | +2.47e-4 | +2.47e-4 | +2.47e-4 | -3.49e-5 |
| 128 | 3.00e-2 | 3 | 3.65e-2 | 3.86e-2 | 3.74e-2 | 3.65e-2 | 131 | -3.34e-4 | +4.83e-4 | +2.24e-5 | -2.46e-5 |
| 129 | 3.00e-2 | 1 | 4.02e-2 | 4.02e-2 | 4.02e-2 | 4.02e-2 | 169 | +5.69e-4 | +5.69e-4 | +5.69e-4 | +3.48e-5 |
| 130 | 3.00e-2 | 2 | 4.01e-2 | 4.22e-2 | 4.12e-2 | 4.01e-2 | 149 | -3.40e-4 | +2.55e-4 | -4.26e-5 | +1.71e-5 |
| 131 | 3.00e-2 | 2 | 3.68e-2 | 4.01e-2 | 3.85e-2 | 3.68e-2 | 112 | -7.75e-4 | +1.02e-6 | -3.87e-4 | -6.36e-5 |
| 132 | 3.00e-2 | 3 | 3.64e-2 | 3.97e-2 | 3.76e-2 | 3.65e-2 | 112 | -8.02e-4 | +5.21e-4 | -9.19e-5 | -7.58e-5 |
| 133 | 3.00e-2 | 2 | 3.78e-2 | 4.21e-2 | 3.99e-2 | 3.78e-2 | 112 | -9.65e-4 | +9.20e-4 | -2.28e-5 | -7.51e-5 |
| 134 | 3.00e-2 | 3 | 3.87e-2 | 4.11e-2 | 3.96e-2 | 3.89e-2 | 118 | -5.27e-4 | +5.70e-4 | +2.45e-5 | -5.30e-5 |
| 135 | 3.00e-2 | 1 | 4.36e-2 | 4.36e-2 | 4.36e-2 | 4.36e-2 | 153 | +7.48e-4 | +7.48e-4 | +7.48e-4 | +2.71e-5 |
| 136 | 3.00e-2 | 3 | 3.49e-2 | 4.29e-2 | 3.83e-2 | 3.49e-2 | 82 | -1.50e-3 | -1.10e-4 | -8.06e-4 | -2.05e-4 |
| 137 | 3.00e-2 | 3 | 3.45e-2 | 4.09e-2 | 3.71e-2 | 3.45e-2 | 82 | -1.55e-3 | +1.23e-3 | -2.83e-4 | -2.42e-4 |
| 138 | 3.00e-2 | 3 | 3.51e-2 | 4.02e-2 | 3.69e-2 | 3.51e-2 | 82 | -1.50e-3 | +1.24e-3 | -1.16e-4 | -2.20e-4 |
| 139 | 3.00e-2 | 3 | 3.67e-2 | 3.95e-2 | 3.77e-2 | 3.67e-2 | 86 | -8.69e-4 | +9.65e-4 | +2.38e-5 | -1.63e-4 |
| 140 | 3.00e-2 | 3 | 3.46e-2 | 4.17e-2 | 3.74e-2 | 3.46e-2 | 81 | -1.91e-3 | +1.07e-3 | -4.34e-4 | -2.50e-4 |
| 141 | 3.00e-2 | 3 | 3.52e-2 | 4.05e-2 | 3.71e-2 | 3.52e-2 | 76 | -1.72e-3 | +1.42e-3 | -1.36e-4 | -2.33e-4 |
| 142 | 3.00e-2 | 4 | 3.66e-2 | 4.10e-2 | 3.81e-2 | 3.68e-2 | 76 | -1.07e-3 | +1.46e-3 | +2.43e-5 | -1.59e-4 |
| 143 | 3.00e-2 | 3 | 3.63e-2 | 4.35e-2 | 3.90e-2 | 3.63e-2 | 76 | -2.26e-3 | +1.34e-3 | -4.01e-4 | -2.39e-4 |
| 144 | 3.00e-2 | 4 | 3.37e-2 | 3.86e-2 | 3.53e-2 | 3.42e-2 | 68 | -1.79e-3 | +6.56e-4 | -3.13e-4 | -2.62e-4 |
| 145 | 3.00e-2 | 3 | 3.71e-2 | 4.21e-2 | 3.94e-2 | 3.71e-2 | 68 | -9.82e-4 | +1.80e-3 | +2.46e-5 | -2.08e-4 |
| 146 | 3.00e-2 | 6 | 3.27e-2 | 4.21e-2 | 3.47e-2 | 3.28e-2 | 56 | -3.38e-3 | +1.21e-3 | -4.93e-4 | -3.24e-4 |
| 147 | 3.00e-2 | 4 | 3.07e-2 | 3.81e-2 | 3.35e-2 | 3.07e-2 | 48 | -2.39e-3 | +1.82e-3 | -6.16e-4 | -4.50e-4 |
| 148 | 3.00e-2 | 5 | 3.13e-2 | 4.13e-2 | 3.41e-2 | 3.13e-2 | 45 | -3.80e-3 | +2.92e-3 | -4.83e-4 | -4.86e-4 |
| 149 | 3.00e-2 | 5 | 3.00e-2 | 4.02e-2 | 3.31e-2 | 3.00e-2 | 45 | -4.31e-3 | +2.61e-3 | -7.63e-4 | -6.15e-4 |
| 150 | 3.00e-3 | 7 | 2.78e-3 | 3.90e-2 | 1.16e-2 | 2.78e-3 | 45 | -1.69e-2 | +3.23e-3 | -8.46e-3 | -4.51e-3 |
| 151 | 3.00e-3 | 4 | 2.85e-3 | 3.51e-3 | 3.07e-3 | 3.00e-3 | 51 | -4.30e-3 | +2.88e-3 | -2.52e-4 | -3.05e-3 |
| 152 | 3.00e-3 | 2 | 2.93e-3 | 6.87e-3 | 4.90e-3 | 6.87e-3 | 364 | -4.38e-4 | +2.34e-3 | +9.51e-4 | -2.28e-3 |
| 154 | 3.00e-3 | 1 | 7.53e-3 | 7.53e-3 | 7.53e-3 | 7.53e-3 | 373 | +2.47e-4 | +2.47e-4 | +2.47e-4 | -2.03e-3 |
| 155 | 3.00e-3 | 1 | 7.22e-3 | 7.22e-3 | 7.22e-3 | 7.22e-3 | 292 | -1.45e-4 | -1.45e-4 | -1.45e-4 | -1.84e-3 |
| 156 | 3.00e-3 | 1 | 7.16e-3 | 7.16e-3 | 7.16e-3 | 7.16e-3 | 315 | -2.40e-5 | -2.40e-5 | -2.40e-5 | -1.66e-3 |
| 157 | 3.00e-3 | 1 | 7.30e-3 | 7.30e-3 | 7.30e-3 | 7.30e-3 | 294 | +6.27e-5 | +6.27e-5 | +6.27e-5 | -1.48e-3 |
| 158 | 3.00e-3 | 1 | 6.77e-3 | 6.77e-3 | 6.77e-3 | 6.77e-3 | 271 | -2.76e-4 | -2.76e-4 | -2.76e-4 | -1.36e-3 |
| 159 | 3.00e-3 | 1 | 7.14e-3 | 7.14e-3 | 7.14e-3 | 7.14e-3 | 297 | +1.79e-4 | +1.79e-4 | +1.79e-4 | -1.21e-3 |
| 160 | 3.00e-3 | 1 | 7.24e-3 | 7.24e-3 | 7.24e-3 | 7.24e-3 | 312 | +4.30e-5 | +4.30e-5 | +4.30e-5 | -1.08e-3 |
| 161 | 3.00e-3 | 1 | 7.22e-3 | 7.22e-3 | 7.22e-3 | 7.22e-3 | 292 | -7.34e-6 | -7.34e-6 | -7.34e-6 | -9.76e-4 |
| 162 | 3.00e-3 | 1 | 7.01e-3 | 7.01e-3 | 7.01e-3 | 7.01e-3 | 261 | -1.17e-4 | -1.17e-4 | -1.17e-4 | -8.90e-4 |
| 164 | 3.00e-3 | 2 | 7.50e-3 | 7.70e-3 | 7.60e-3 | 7.50e-3 | 298 | -8.80e-5 | +2.75e-4 | +9.36e-5 | -7.05e-4 |
| 166 | 3.00e-3 | 1 | 7.34e-3 | 7.34e-3 | 7.34e-3 | 7.34e-3 | 299 | -7.23e-5 | -7.23e-5 | -7.23e-5 | -6.42e-4 |
| 167 | 3.00e-3 | 1 | 7.24e-3 | 7.24e-3 | 7.24e-3 | 7.24e-3 | 272 | -4.70e-5 | -4.70e-5 | -4.70e-5 | -5.82e-4 |
| 168 | 3.00e-3 | 1 | 7.95e-3 | 7.95e-3 | 7.95e-3 | 7.95e-3 | 343 | +2.72e-4 | +2.72e-4 | +2.72e-4 | -4.97e-4 |
| 169 | 3.00e-3 | 1 | 7.21e-3 | 7.21e-3 | 7.21e-3 | 7.21e-3 | 282 | -3.50e-4 | -3.50e-4 | -3.50e-4 | -4.82e-4 |
| 170 | 3.00e-3 | 1 | 7.28e-3 | 7.28e-3 | 7.28e-3 | 7.28e-3 | 287 | +3.73e-5 | +3.73e-5 | +3.73e-5 | -4.30e-4 |
| 171 | 3.00e-3 | 1 | 7.26e-3 | 7.26e-3 | 7.26e-3 | 7.26e-3 | 274 | -1.26e-5 | -1.26e-5 | -1.26e-5 | -3.89e-4 |
| 172 | 3.00e-3 | 1 | 7.84e-3 | 7.84e-3 | 7.84e-3 | 7.84e-3 | 323 | +2.40e-4 | +2.40e-4 | +2.40e-4 | -3.26e-4 |
| 173 | 3.00e-3 | 1 | 7.34e-3 | 7.34e-3 | 7.34e-3 | 7.34e-3 | 282 | -2.33e-4 | -2.33e-4 | -2.33e-4 | -3.16e-4 |
| 174 | 3.00e-3 | 1 | 7.54e-3 | 7.54e-3 | 7.54e-3 | 7.54e-3 | 296 | +8.90e-5 | +8.90e-5 | +8.90e-5 | -2.76e-4 |
| 175 | 3.00e-3 | 1 | 7.28e-3 | 7.28e-3 | 7.28e-3 | 7.28e-3 | 271 | -1.29e-4 | -1.29e-4 | -1.29e-4 | -2.61e-4 |
| 176 | 3.00e-3 | 1 | 7.08e-3 | 7.08e-3 | 7.08e-3 | 7.08e-3 | 244 | -1.18e-4 | -1.18e-4 | -1.18e-4 | -2.47e-4 |
| 177 | 3.00e-3 | 1 | 7.27e-3 | 7.27e-3 | 7.27e-3 | 7.27e-3 | 262 | +1.06e-4 | +1.06e-4 | +1.06e-4 | -2.12e-4 |
| 178 | 3.00e-3 | 1 | 7.11e-3 | 7.11e-3 | 7.11e-3 | 7.11e-3 | 256 | -9.04e-5 | -9.04e-5 | -9.04e-5 | -2.00e-4 |
| 179 | 3.00e-3 | 1 | 7.19e-3 | 7.19e-3 | 7.19e-3 | 7.19e-3 | 251 | +4.60e-5 | +4.60e-5 | +4.60e-5 | -1.75e-4 |
| 180 | 3.00e-3 | 1 | 7.73e-3 | 7.73e-3 | 7.73e-3 | 7.73e-3 | 279 | +2.59e-4 | +2.59e-4 | +2.59e-4 | -1.32e-4 |
| 181 | 3.00e-3 | 1 | 7.49e-3 | 7.49e-3 | 7.49e-3 | 7.49e-3 | 272 | -1.13e-4 | -1.13e-4 | -1.13e-4 | -1.30e-4 |
| 182 | 3.00e-3 | 2 | 7.16e-3 | 7.50e-3 | 7.33e-3 | 7.16e-3 | 247 | -1.92e-4 | +4.93e-6 | -9.34e-5 | -1.24e-4 |
| 184 | 3.00e-3 | 2 | 7.44e-3 | 7.82e-3 | 7.63e-3 | 7.44e-3 | 247 | -2.00e-4 | +2.95e-4 | +4.79e-5 | -9.37e-5 |
| 186 | 3.00e-3 | 2 | 7.42e-3 | 7.92e-3 | 7.67e-3 | 7.42e-3 | 247 | -2.61e-4 | +2.09e-4 | -2.64e-5 | -8.32e-5 |
| 188 | 3.00e-3 | 2 | 7.22e-3 | 8.04e-3 | 7.63e-3 | 7.22e-3 | 230 | -4.69e-4 | +2.54e-4 | -1.07e-4 | -9.14e-5 |
| 189 | 3.00e-3 | 1 | 7.67e-3 | 7.67e-3 | 7.67e-3 | 7.67e-3 | 259 | +2.37e-4 | +2.37e-4 | +2.37e-4 | -5.86e-5 |
| 190 | 3.00e-3 | 1 | 7.49e-3 | 7.49e-3 | 7.49e-3 | 7.49e-3 | 271 | -8.81e-5 | -8.81e-5 | -8.81e-5 | -6.16e-5 |
| 191 | 3.00e-3 | 1 | 7.68e-3 | 7.68e-3 | 7.68e-3 | 7.68e-3 | 273 | +9.22e-5 | +9.22e-5 | +9.22e-5 | -4.62e-5 |
| 192 | 3.00e-3 | 1 | 7.56e-3 | 7.56e-3 | 7.56e-3 | 7.56e-3 | 255 | -6.61e-5 | -6.61e-5 | -6.61e-5 | -4.82e-5 |
| 193 | 3.00e-3 | 1 | 7.64e-3 | 7.64e-3 | 7.64e-3 | 7.64e-3 | 267 | +4.17e-5 | +4.17e-5 | +4.17e-5 | -3.92e-5 |
| 194 | 3.00e-3 | 1 | 7.70e-3 | 7.70e-3 | 7.70e-3 | 7.70e-3 | 258 | +3.09e-5 | +3.09e-5 | +3.09e-5 | -3.22e-5 |
| 195 | 3.00e-3 | 1 | 7.66e-3 | 7.66e-3 | 7.66e-3 | 7.66e-3 | 270 | -1.76e-5 | -1.76e-5 | -1.76e-5 | -3.07e-5 |
| 196 | 3.00e-3 | 2 | 7.35e-3 | 7.76e-3 | 7.55e-3 | 7.35e-3 | 230 | -2.34e-4 | +4.71e-5 | -9.36e-5 | -4.41e-5 |
| 198 | 3.00e-3 | 2 | 7.66e-3 | 7.94e-3 | 7.80e-3 | 7.66e-3 | 243 | -1.46e-4 | +2.57e-4 | +5.52e-5 | -2.72e-5 |

