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
| cpu-async | 0.062496 | 0.9169 | +0.0044 | 1920.0 | 556 | 84.5 | 100% | 100% | 9.0 |

## Best Mode per Model

| Model | Best Eval | Mode | Fastest (within 2% of solo-0) | Mode |
|-------|-----------|------|-------------------------------|------|
| resnet-graph | 0.9169 | cpu-async | - | - |

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
| cpu-async | 1.9674 | 0.7536 | 0.5791 | 0.5173 | 0.4865 | 0.4661 | 0.4973 | 0.4811 | 0.4805 | 0.4670 | 0.2132 | 0.1766 | 0.1543 | 0.1428 | 0.1335 | 0.0802 | 0.0731 | 0.0703 | 0.0642 | 0.0625 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | cpu-async | 0 | cuda:0 | 0.4028 | 2.5 |
| resnet-graph | cpu-async | 1 | cuda:1 | 0.3020 | 3.5 |
| resnet-graph | cpu-async | 2 | cuda:2 | 0.2952 | 3.4 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|
| resnet-graph | cpu-async | 358 | 357 | 386 | 382 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| cpu-async | gpu1 | 1572.7 | 2.0 | epoch-boundary(163) |
| cpu-async | gpu2 | 1572.6 | 2.0 | epoch-boundary(163) |
| cpu-async | gpu2 | 1114.5 | 0.9 | epoch-boundary(115) |
| cpu-async | gpu1 | 1707.6 | 0.7 | epoch-boundary(177) |
| cpu-async | gpu1 | 1783.7 | 0.6 | epoch-boundary(185) |
| cpu-async | gpu2 | 1783.8 | 0.6 | epoch-boundary(185) |
| cpu-async | gpu2 | 1707.6 | 0.5 | epoch-boundary(177) |
| cpu-async | gpu2 | 1919.4 | 0.5 | epoch-boundary(199) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | cpu-async | gpu1 | 3.3s | 0.0s | 0.0s | 0.0s | 3.8s |
| resnet-graph | cpu-async | gpu2 | 4.6s | 0.0s | 0.0s | 0.0s | 5.2s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | cpu-async | 296 | 0 | 556 | 84.5 | 1526/9784 | 556 | 84.5 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | cpu-async | 207.8 | 10.8% |

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
| resnet-graph | cpu-async | 187 | 556 | 0 | 6.96e-3 | -2.63e-5 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | cpu-async | 0 | 556 | 1.02e-1 | 5.55e-2 | 3.60e-3 | 3.52e-1 | 26.6 | -1.76e-4 | 1.07e-3 |
| resnet-graph | cpu-async | 1 | 556 | 1.02e-1 | 5.61e-2 | 3.68e-3 | 3.63e-1 | 32.2 | -2.01e-4 | 1.24e-3 |
| resnet-graph | cpu-async | 2 | 556 | 1.03e-1 | 5.72e-2 | 3.41e-3 | 3.64e-1 | 41.2 | -2.19e-4 | 1.33e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | cpu-async | rank0 ↔ rank1 | +0.9918 |
| resnet-graph | cpu-async | rank0 ↔ rank2 | +0.9911 |
| resnet-graph | cpu-async | rank1 ↔ rank2 | +0.9902 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | cpu-async | 68 (2,4,5,6,7,8,9,10…147,148) | 0 (—) | — | 2,4,5,6,7,8,9,10…147,148 | — |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | cpu-async | 0e0 | 3 | 4 | 4 |
| resnet-graph | cpu-async | 0e0 | 5 | 1 | 1 |
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
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 414 | +0.119 |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | 71 | +0.067 |
| resnet-graph | cpu-async | 3.00e-3 | 150–198 | 67 | -0.064 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | cpu-async | 554 | -0.040 | 187 | +0.444 | +0.623 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | cpu-async | 555 | 3.34e1–8.06e1 | 6.80e1 | 2.37e-3 | 3.71e-3 | 3.59e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 416 | 31–77741 | +6.403e-6 | 0.232 | +6.642e-6 | 0.259 | 93 | +1.107e-5 | 0.649 | 31–1000 | +9.586e-4 | 0.722 |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 406 | 887–77741 | +7.447e-6 | 0.336 | +7.709e-6 | 0.377 | 92 | +1.152e-5 | 0.683 | 74–1000 | +9.958e-4 | 0.884 |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | 72 | 78536–116910 | +7.596e-6 | 0.079 | +7.513e-6 | 0.079 | 49 | +5.741e-6 | 0.035 | 341–795 | -3.819e-4 | 0.017 |
| resnet-graph | cpu-async | 3.00e-3 | 150–198 | 68 | 117348–155499 | +7.602e-6 | 0.047 | +8.014e-6 | 0.052 | 45 | +7.510e-6 | 0.065 | 221–1006 | +7.397e-4 | 0.190 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | +9.586e-4 | r0: +9.409e-4, r1: +9.645e-4, r2: +9.710e-4 | r0: 0.721, r1: 0.705, r2: 0.687 | 1.03× | ✓ |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +9.958e-4 | r0: +9.777e-4, r1: +1.002e-3, r2: +1.008e-3 | r0: 0.885, r1: 0.859, r2: 0.835 | 1.03× | ✓ |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | -3.819e-4 | r0: -4.517e-4, r1: -3.623e-4, r2: -3.317e-4 | r0: 0.023, r1: 0.015, r2: 0.012 | 1.36× | ✓ |
| resnet-graph | cpu-async | 3.00e-3 | 150–198 | +7.397e-4 | r0: +7.468e-4, r1: +7.253e-4, r2: +7.488e-4 | r0: 0.191, r1: 0.179, r2: 0.200 | 1.03× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | cpu-async | `█▇▇▇▇▇▇▇▇▇▇▇▇▇▇████████▇▄▄▄▅▅▅▅▅▅▅▅▅▁▁▁▂▂▂▂▂▂▂▂` | `▁▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇████████▇▇█████████▇▆▇█████████` |

### Per-Epoch Detail: resnet-graph / cpu-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 10 | 1.10e-1 | 3.64e-1 | 1.91e-1 | 1.21e-1 | 50 | -3.11e-2 | +8.13e-3 | -1.11e-2 | -8.93e-3 |
| 1 | 3.00e-1 | 6 | 9.81e-2 | 1.50e-1 | 1.14e-1 | 1.00e-1 | 44 | -4.21e-3 | +2.63e-3 | -1.10e-3 | -4.29e-3 |
| 2 | 3.00e-1 | 8 | 1.09e-1 | 1.30e-1 | 1.15e-1 | 1.11e-1 | 37 | -3.08e-3 | +3.29e-3 | -5.21e-5 | -1.67e-3 |
| 3 | 3.00e-1 | 4 | 1.15e-1 | 1.48e-1 | 1.27e-1 | 1.15e-1 | 38 | -4.38e-3 | +4.04e-3 | -6.24e-4 | -1.35e-3 |
| 4 | 3.00e-1 | 7 | 1.09e-1 | 1.63e-1 | 1.23e-1 | 1.23e-1 | 44 | -8.65e-3 | +3.97e-3 | -7.45e-4 | -9.03e-4 |
| 5 | 3.00e-1 | 6 | 1.18e-1 | 1.58e-1 | 1.29e-1 | 1.18e-1 | 40 | -5.09e-3 | +2.89e-3 | -7.38e-4 | -8.33e-4 |
| 6 | 3.00e-1 | 5 | 1.15e-1 | 1.56e-1 | 1.30e-1 | 1.29e-1 | 50 | -5.26e-3 | +3.51e-3 | -4.02e-4 | -6.32e-4 |
| 7 | 3.00e-1 | 5 | 1.29e-1 | 1.58e-1 | 1.37e-1 | 1.30e-1 | 53 | -3.19e-3 | +2.36e-3 | -3.21e-4 | -5.20e-4 |
| 8 | 3.00e-1 | 6 | 1.25e-1 | 1.60e-1 | 1.34e-1 | 1.25e-1 | 50 | -3.71e-3 | +2.30e-3 | -4.17e-4 | -4.99e-4 |
| 9 | 3.00e-1 | 5 | 1.15e-1 | 1.52e-1 | 1.26e-1 | 1.21e-1 | 50 | -4.17e-3 | +2.53e-3 | -5.29e-4 | -4.98e-4 |
| 10 | 3.00e-1 | 6 | 1.19e-1 | 1.56e-1 | 1.28e-1 | 1.19e-1 | 44 | -2.94e-3 | +2.75e-3 | -4.82e-4 | -4.95e-4 |
| 11 | 3.00e-1 | 6 | 1.18e-1 | 1.51e-1 | 1.25e-1 | 1.18e-1 | 44 | -4.66e-3 | +2.84e-3 | -4.88e-4 | -4.89e-4 |
| 12 | 3.00e-1 | 5 | 1.13e-1 | 1.45e-1 | 1.24e-1 | 1.21e-1 | 47 | -3.71e-3 | +2.45e-3 | -3.63e-4 | -4.30e-4 |
| 13 | 3.00e-1 | 8 | 1.14e-1 | 1.54e-1 | 1.22e-1 | 1.16e-1 | 40 | -4.88e-3 | +2.82e-3 | -4.65e-4 | -4.20e-4 |
| 14 | 3.00e-1 | 5 | 1.08e-1 | 1.49e-1 | 1.18e-1 | 1.09e-1 | 35 | -7.57e-3 | +3.19e-3 | -1.18e-3 | -7.13e-4 |
| 15 | 3.00e-1 | 7 | 1.07e-1 | 1.43e-1 | 1.14e-1 | 1.12e-1 | 39 | -6.75e-3 | +3.81e-3 | -4.75e-4 | -5.37e-4 |
| 16 | 3.00e-1 | 7 | 1.07e-1 | 1.44e-1 | 1.18e-1 | 1.15e-1 | 38 | -5.46e-3 | +3.44e-3 | -4.25e-4 | -4.51e-4 |
| 17 | 3.00e-1 | 10 | 9.78e-2 | 1.54e-1 | 1.09e-1 | 1.03e-1 | 31 | -1.09e-2 | +3.42e-3 | -1.08e-3 | -6.84e-4 |
| 18 | 3.00e-1 | 6 | 1.02e-1 | 1.41e-1 | 1.12e-1 | 1.02e-1 | 33 | -8.07e-3 | +4.50e-3 | -9.97e-4 | -8.36e-4 |
| 19 | 3.00e-1 | 7 | 9.94e-2 | 1.45e-1 | 1.13e-1 | 9.94e-2 | 28 | -3.48e-3 | +4.71e-3 | -8.21e-4 | -8.72e-4 |
| 20 | 3.00e-1 | 8 | 1.07e-1 | 1.48e-1 | 1.18e-1 | 1.19e-1 | 43 | -8.39e-3 | +5.05e-3 | -2.65e-4 | -4.62e-4 |
| 21 | 3.00e-1 | 5 | 1.10e-1 | 1.48e-1 | 1.22e-1 | 1.10e-1 | 40 | -4.05e-3 | +3.03e-3 | -7.88e-4 | -6.28e-4 |
| 22 | 3.00e-1 | 6 | 1.13e-1 | 1.58e-1 | 1.25e-1 | 1.13e-1 | 41 | -4.95e-3 | +4.57e-3 | -5.29e-4 | -6.20e-4 |
| 23 | 3.00e-1 | 7 | 1.04e-1 | 1.53e-1 | 1.15e-1 | 1.08e-1 | 32 | -6.73e-3 | +3.76e-3 | -8.53e-4 | -6.82e-4 |
| 24 | 3.00e-1 | 7 | 1.10e-1 | 1.50e-1 | 1.19e-1 | 1.15e-1 | 37 | -8.39e-3 | +4.50e-3 | -5.12e-4 | -5.47e-4 |
| 25 | 3.00e-1 | 8 | 1.01e-1 | 1.53e-1 | 1.18e-1 | 1.24e-1 | 46 | -7.01e-3 | +3.47e-3 | -4.71e-4 | -3.44e-4 |
| 26 | 3.00e-1 | 5 | 1.09e-1 | 1.57e-1 | 1.24e-1 | 1.13e-1 | 40 | -6.13e-3 | +2.70e-3 | -1.15e-3 | -6.64e-4 |
| 27 | 3.00e-1 | 6 | 1.11e-1 | 1.50e-1 | 1.23e-1 | 1.15e-1 | 40 | -5.22e-3 | +3.49e-3 | -4.92e-4 | -5.85e-4 |
| 28 | 3.00e-1 | 7 | 1.08e-1 | 1.47e-1 | 1.15e-1 | 1.08e-1 | 39 | -6.92e-3 | +3.29e-3 | -7.82e-4 | -6.43e-4 |
| 29 | 3.00e-1 | 6 | 1.16e-1 | 1.48e-1 | 1.24e-1 | 1.18e-1 | 43 | -5.20e-3 | +4.29e-3 | -2.63e-4 | -4.86e-4 |
| 30 | 3.00e-1 | 7 | 1.07e-1 | 1.59e-1 | 1.20e-1 | 1.07e-1 | 37 | -6.23e-3 | +3.31e-3 | -9.56e-4 | -7.17e-4 |
| 31 | 3.00e-1 | 7 | 1.07e-1 | 1.52e-1 | 1.15e-1 | 1.07e-1 | 33 | -9.09e-3 | +4.22e-3 | -8.93e-4 | -7.50e-4 |
| 32 | 3.00e-1 | 7 | 1.05e-1 | 1.52e-1 | 1.15e-1 | 1.07e-1 | 33 | -8.12e-3 | +4.59e-3 | -8.26e-4 | -7.45e-4 |
| 33 | 3.00e-1 | 8 | 1.00e-1 | 1.53e-1 | 1.11e-1 | 1.06e-1 | 30 | -1.18e-2 | +4.56e-3 | -9.65e-4 | -7.46e-4 |
| 34 | 3.00e-1 | 9 | 9.64e-2 | 1.52e-1 | 1.15e-1 | 1.18e-1 | 40 | -1.27e-2 | +4.55e-3 | -7.44e-4 | -5.50e-4 |
| 35 | 3.00e-1 | 5 | 1.16e-1 | 1.61e-1 | 1.29e-1 | 1.20e-1 | 40 | -6.05e-3 | +3.76e-3 | -6.97e-4 | -6.09e-4 |
| 36 | 3.00e-1 | 6 | 1.02e-1 | 1.59e-1 | 1.19e-1 | 1.10e-1 | 36 | -6.41e-3 | +3.31e-3 | -1.01e-3 | -7.59e-4 |
| 37 | 3.00e-1 | 6 | 1.13e-1 | 1.59e-1 | 1.25e-1 | 1.13e-1 | 42 | -5.55e-3 | +4.19e-3 | -6.16e-4 | -7.18e-4 |
| 38 | 3.00e-1 | 5 | 1.20e-1 | 1.54e-1 | 1.31e-1 | 1.26e-1 | 50 | -3.77e-3 | +3.99e-3 | -7.58e-5 | -4.71e-4 |
| 39 | 3.00e-1 | 5 | 1.25e-1 | 1.65e-1 | 1.36e-1 | 1.25e-1 | 45 | -5.27e-3 | +2.89e-3 | -6.36e-4 | -5.51e-4 |
| 40 | 3.00e-1 | 7 | 1.14e-1 | 1.53e-1 | 1.23e-1 | 1.23e-1 | 47 | -5.52e-3 | +2.50e-3 | -4.25e-4 | -4.12e-4 |
| 41 | 3.00e-1 | 5 | 1.21e-1 | 1.56e-1 | 1.31e-1 | 1.24e-1 | 44 | -4.03e-3 | +2.95e-3 | -3.81e-4 | -4.03e-4 |
| 42 | 3.00e-1 | 8 | 1.08e-1 | 1.57e-1 | 1.23e-1 | 1.08e-1 | 36 | -5.18e-3 | +2.82e-3 | -7.60e-4 | -6.37e-4 |
| 43 | 3.00e-1 | 5 | 1.10e-1 | 1.56e-1 | 1.21e-1 | 1.10e-1 | 36 | -8.67e-3 | +4.51e-3 | -1.08e-3 | -8.12e-4 |
| 44 | 3.00e-1 | 7 | 1.09e-1 | 1.58e-1 | 1.19e-1 | 1.10e-1 | 37 | -8.24e-3 | +4.64e-3 | -8.21e-4 | -7.84e-4 |
| 45 | 3.00e-1 | 7 | 1.10e-1 | 1.49e-1 | 1.18e-1 | 1.13e-1 | 37 | -6.97e-3 | +3.86e-3 | -5.13e-4 | -5.99e-4 |
| 46 | 3.00e-1 | 6 | 1.09e-1 | 1.56e-1 | 1.32e-1 | 1.33e-1 | 56 | -8.20e-3 | +4.07e-3 | -5.42e-4 | -5.40e-4 |
| 47 | 3.00e-1 | 4 | 1.25e-1 | 1.61e-1 | 1.39e-1 | 1.25e-1 | 50 | -3.00e-3 | +2.11e-3 | -6.60e-4 | -6.12e-4 |
| 48 | 3.00e-1 | 5 | 1.20e-1 | 1.68e-1 | 1.33e-1 | 1.26e-1 | 48 | -6.09e-3 | +3.16e-3 | -6.86e-4 | -6.22e-4 |
| 49 | 3.00e-1 | 6 | 1.12e-1 | 1.57e-1 | 1.26e-1 | 1.22e-1 | 47 | -5.54e-3 | +2.40e-3 | -6.35e-4 | -5.85e-4 |
| 50 | 3.00e-1 | 6 | 1.06e-1 | 1.64e-1 | 1.25e-1 | 1.06e-1 | 34 | -5.31e-3 | +3.07e-3 | -1.24e-3 | -9.30e-4 |
| 51 | 3.00e-1 | 7 | 1.10e-1 | 1.54e-1 | 1.20e-1 | 1.12e-1 | 36 | -7.39e-3 | +4.60e-3 | -6.43e-4 | -7.67e-4 |
| 52 | 3.00e-1 | 9 | 1.03e-1 | 1.61e-1 | 1.16e-1 | 1.10e-1 | 38 | -8.45e-3 | +4.38e-3 | -7.45e-4 | -6.46e-4 |
| 53 | 3.00e-1 | 4 | 1.09e-1 | 1.51e-1 | 1.23e-1 | 1.15e-1 | 42 | -6.85e-3 | +4.48e-3 | -8.39e-4 | -7.30e-4 |
| 54 | 3.00e-1 | 7 | 1.14e-1 | 1.54e-1 | 1.25e-1 | 1.23e-1 | 44 | -4.93e-3 | +3.80e-3 | -2.44e-4 | -4.50e-4 |
| 55 | 3.00e-1 | 6 | 1.08e-1 | 1.57e-1 | 1.25e-1 | 1.08e-1 | 32 | -6.19e-3 | +2.88e-3 | -1.26e-3 | -8.79e-4 |
| 56 | 3.00e-1 | 8 | 9.78e-2 | 1.52e-1 | 1.12e-1 | 1.11e-1 | 34 | -1.10e-2 | +4.40e-3 | -9.39e-4 | -7.46e-4 |
| 57 | 3.00e-1 | 8 | 1.04e-1 | 1.52e-1 | 1.21e-1 | 1.26e-1 | 42 | -1.05e-2 | +4.02e-3 | -5.86e-4 | -5.08e-4 |
| 58 | 3.00e-1 | 1 | 1.17e-1 | 1.17e-1 | 1.17e-1 | 1.17e-1 | 41 | -1.69e-3 | -1.69e-3 | -1.69e-3 | -6.27e-4 |
| 59 | 3.00e-1 | 1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 310 | +2.20e-3 | +2.20e-3 | +2.20e-3 | -3.44e-4 |
| 60 | 3.00e-1 | 1 | 2.31e-1 | 2.31e-1 | 2.31e-1 | 2.31e-1 | 271 | -1.19e-5 | -1.19e-5 | -1.19e-5 | -3.11e-4 |
| 61 | 3.00e-1 | 1 | 2.31e-1 | 2.31e-1 | 2.31e-1 | 2.31e-1 | 287 | -4.20e-6 | -4.20e-6 | -4.20e-6 | -2.80e-4 |
| 62 | 3.00e-1 | 1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 302 | -2.81e-5 | -2.81e-5 | -2.81e-5 | -2.55e-4 |
| 63 | 3.00e-1 | 1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 301 | -5.28e-5 | -5.28e-5 | -5.28e-5 | -2.35e-4 |
| 64 | 3.00e-1 | 1 | 2.39e-1 | 2.39e-1 | 2.39e-1 | 2.39e-1 | 331 | +1.74e-4 | +1.74e-4 | +1.74e-4 | -1.94e-4 |
| 65 | 3.00e-1 | 1 | 2.31e-1 | 2.31e-1 | 2.31e-1 | 2.31e-1 | 310 | -1.06e-4 | -1.06e-4 | -1.06e-4 | -1.85e-4 |
| 66 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 273 | -1.96e-4 | -1.96e-4 | -1.96e-4 | -1.86e-4 |
| 68 | 3.00e-1 | 2 | 2.17e-1 | 2.45e-1 | 2.31e-1 | 2.17e-1 | 258 | -4.73e-4 | +2.99e-4 | -8.71e-5 | -1.71e-4 |
| 70 | 3.00e-1 | 1 | 2.42e-1 | 2.42e-1 | 2.42e-1 | 2.42e-1 | 349 | +3.11e-4 | +3.11e-4 | +3.11e-4 | -1.23e-4 |
| 71 | 3.00e-1 | 1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 292 | -2.39e-4 | -2.39e-4 | -2.39e-4 | -1.35e-4 |
| 72 | 3.00e-1 | 1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 272 | -6.27e-5 | -6.27e-5 | -6.27e-5 | -1.27e-4 |
| 73 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 269 | -1.95e-5 | -1.95e-5 | -1.95e-5 | -1.17e-4 |
| 74 | 3.00e-1 | 1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 290 | +1.05e-4 | +1.05e-4 | +1.05e-4 | -9.45e-5 |
| 75 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 255 | -2.01e-4 | -2.01e-4 | -2.01e-4 | -1.05e-4 |
| 76 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 280 | +2.82e-5 | +2.82e-5 | +2.82e-5 | -9.18e-5 |
| 77 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 283 | +1.89e-5 | +1.89e-5 | +1.89e-5 | -8.08e-5 |
| 78 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 270 | -2.33e-5 | -2.33e-5 | -2.33e-5 | -7.50e-5 |
| 79 | 3.00e-1 | 1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 282 | +1.54e-4 | +1.54e-4 | +1.54e-4 | -5.21e-5 |
| 80 | 3.00e-1 | 1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 292 | -5.11e-5 | -5.11e-5 | -5.11e-5 | -5.20e-5 |
| 81 | 3.00e-1 | 1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 273 | -1.64e-5 | -1.64e-5 | -1.64e-5 | -4.85e-5 |
| 82 | 3.00e-1 | 1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 296 | +5.09e-5 | +5.09e-5 | +5.09e-5 | -3.85e-5 |
| 83 | 3.00e-1 | 2 | 2.12e-1 | 2.16e-1 | 2.14e-1 | 2.12e-1 | 240 | -1.72e-4 | -8.44e-5 | -1.28e-4 | -5.51e-5 |
| 85 | 3.00e-1 | 2 | 2.17e-1 | 2.32e-1 | 2.24e-1 | 2.17e-1 | 253 | -2.49e-4 | +2.73e-4 | +1.20e-5 | -4.50e-5 |
| 87 | 3.00e-1 | 2 | 2.19e-1 | 2.34e-1 | 2.26e-1 | 2.19e-1 | 253 | -2.56e-4 | +2.30e-4 | -1.29e-5 | -4.13e-5 |
| 89 | 3.00e-1 | 2 | 2.16e-1 | 2.31e-1 | 2.24e-1 | 2.16e-1 | 240 | -2.75e-4 | +1.72e-4 | -5.12e-5 | -4.54e-5 |
| 91 | 3.00e-1 | 2 | 2.19e-1 | 2.31e-1 | 2.25e-1 | 2.19e-1 | 240 | -2.14e-4 | +2.10e-4 | -1.69e-6 | -3.92e-5 |
| 93 | 3.00e-1 | 2 | 2.11e-1 | 2.34e-1 | 2.23e-1 | 2.11e-1 | 240 | -4.37e-4 | +2.15e-4 | -1.11e-4 | -5.61e-5 |
| 94 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 230 | -3.90e-5 | -3.90e-5 | -3.90e-5 | -5.44e-5 |
| 95 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 238 | +3.03e-5 | +3.03e-5 | +3.03e-5 | -4.59e-5 |
| 96 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 260 | +1.31e-4 | +1.31e-4 | +1.31e-4 | -2.83e-5 |
| 97 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 251 | -2.46e-5 | -2.46e-5 | -2.46e-5 | -2.79e-5 |
| 98 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 242 | -4.20e-5 | -4.20e-5 | -4.20e-5 | -2.93e-5 |
| 99 | 3.00e-1 | 1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 268 | +1.66e-4 | +1.66e-4 | +1.66e-4 | -9.82e-6 |
| 100 | 3.00e-2 | 1 | 2.28e-1 | 2.28e-1 | 2.28e-1 | 2.28e-1 | 294 | +5.76e-5 | +5.76e-5 | +5.76e-5 | -3.08e-6 |
| 101 | 3.00e-2 | 1 | 1.10e-1 | 1.10e-1 | 1.10e-1 | 1.10e-1 | 249 | -2.93e-3 | -2.93e-3 | -2.93e-3 | -2.95e-4 |
| 102 | 3.00e-2 | 2 | 3.45e-2 | 5.66e-2 | 4.55e-2 | 3.45e-2 | 208 | -2.85e-3 | -2.37e-3 | -2.61e-3 | -7.33e-4 |
| 103 | 3.00e-2 | 1 | 2.73e-2 | 2.73e-2 | 2.73e-2 | 2.73e-2 | 246 | -9.59e-4 | -9.59e-4 | -9.59e-4 | -7.55e-4 |
| 104 | 3.00e-2 | 1 | 2.65e-2 | 2.65e-2 | 2.65e-2 | 2.65e-2 | 241 | -1.19e-4 | -1.19e-4 | -1.19e-4 | -6.92e-4 |
| 105 | 3.00e-2 | 1 | 2.71e-2 | 2.71e-2 | 2.71e-2 | 2.71e-2 | 248 | +9.31e-5 | +9.31e-5 | +9.31e-5 | -6.13e-4 |
| 106 | 3.00e-2 | 1 | 2.81e-2 | 2.81e-2 | 2.81e-2 | 2.81e-2 | 259 | +1.36e-4 | +1.36e-4 | +1.36e-4 | -5.38e-4 |
| 107 | 3.00e-2 | 1 | 2.65e-2 | 2.65e-2 | 2.65e-2 | 2.65e-2 | 213 | -2.78e-4 | -2.78e-4 | -2.78e-4 | -5.12e-4 |
| 108 | 3.00e-2 | 2 | 2.68e-2 | 2.83e-2 | 2.75e-2 | 2.68e-2 | 197 | -2.77e-4 | +2.82e-4 | +2.85e-6 | -4.17e-4 |
| 109 | 3.00e-2 | 1 | 2.86e-2 | 2.86e-2 | 2.86e-2 | 2.86e-2 | 228 | +2.85e-4 | +2.85e-4 | +2.85e-4 | -3.47e-4 |
| 110 | 3.00e-2 | 2 | 2.89e-2 | 2.97e-2 | 2.93e-2 | 2.97e-2 | 218 | +4.83e-5 | +1.28e-4 | +8.82e-5 | -2.64e-4 |
| 112 | 3.00e-2 | 2 | 3.07e-2 | 3.29e-2 | 3.18e-2 | 3.07e-2 | 201 | -3.31e-4 | +3.93e-4 | +3.11e-5 | -2.12e-4 |
| 113 | 3.00e-2 | 1 | 3.08e-2 | 3.08e-2 | 3.08e-2 | 3.08e-2 | 209 | +2.62e-6 | +2.62e-6 | +2.62e-6 | -1.90e-4 |
| 114 | 3.00e-2 | 1 | 3.16e-2 | 3.16e-2 | 3.16e-2 | 3.16e-2 | 207 | +1.25e-4 | +1.25e-4 | +1.25e-4 | -1.59e-4 |
| 115 | 3.00e-2 | 2 | 3.28e-2 | 3.30e-2 | 3.29e-2 | 3.30e-2 | 215 | +3.48e-5 | +1.68e-4 | +1.02e-4 | -1.10e-4 |
| 116 | 3.00e-2 | 1 | 3.54e-2 | 3.54e-2 | 3.54e-2 | 3.54e-2 | 251 | +2.74e-4 | +2.74e-4 | +2.74e-4 | -7.14e-5 |
| 117 | 3.00e-2 | 1 | 3.64e-2 | 3.64e-2 | 3.64e-2 | 3.64e-2 | 237 | +1.23e-4 | +1.23e-4 | +1.23e-4 | -5.20e-5 |
| 118 | 3.00e-2 | 1 | 3.69e-2 | 3.69e-2 | 3.69e-2 | 3.69e-2 | 261 | +4.84e-5 | +4.84e-5 | +4.84e-5 | -4.20e-5 |
| 119 | 3.00e-2 | 1 | 3.70e-2 | 3.70e-2 | 3.70e-2 | 3.70e-2 | 241 | +1.18e-5 | +1.18e-5 | +1.18e-5 | -3.66e-5 |
| 120 | 3.00e-2 | 2 | 3.49e-2 | 3.83e-2 | 3.66e-2 | 3.49e-2 | 179 | -5.17e-4 | +1.37e-4 | -1.90e-4 | -6.90e-5 |
| 121 | 3.00e-2 | 1 | 3.53e-2 | 3.53e-2 | 3.53e-2 | 3.53e-2 | 193 | +6.00e-5 | +6.00e-5 | +6.00e-5 | -5.61e-5 |
| 122 | 3.00e-2 | 2 | 3.55e-2 | 3.68e-2 | 3.61e-2 | 3.55e-2 | 179 | -2.01e-4 | +1.87e-4 | -7.11e-6 | -4.87e-5 |
| 123 | 3.00e-2 | 1 | 3.76e-2 | 3.76e-2 | 3.76e-2 | 3.76e-2 | 211 | +2.82e-4 | +2.82e-4 | +2.82e-4 | -1.57e-5 |
| 124 | 3.00e-2 | 1 | 3.84e-2 | 3.84e-2 | 3.84e-2 | 3.84e-2 | 208 | +9.36e-5 | +9.36e-5 | +9.36e-5 | -4.77e-6 |
| 125 | 3.00e-2 | 1 | 4.03e-2 | 4.03e-2 | 4.03e-2 | 4.03e-2 | 222 | +2.21e-4 | +2.21e-4 | +2.21e-4 | +1.78e-5 |
| 126 | 3.00e-2 | 2 | 3.79e-2 | 4.13e-2 | 3.96e-2 | 3.79e-2 | 167 | -5.12e-4 | +1.06e-4 | -2.03e-4 | -2.72e-5 |
| 127 | 3.00e-2 | 1 | 3.97e-2 | 3.97e-2 | 3.97e-2 | 3.97e-2 | 212 | +2.19e-4 | +2.19e-4 | +2.19e-4 | -2.59e-6 |
| 128 | 3.00e-2 | 2 | 3.95e-2 | 4.05e-2 | 4.00e-2 | 3.95e-2 | 170 | -1.55e-4 | +9.50e-5 | -3.02e-5 | -9.09e-6 |
| 129 | 3.00e-2 | 1 | 4.22e-2 | 4.22e-2 | 4.22e-2 | 4.22e-2 | 213 | +3.16e-4 | +3.16e-4 | +3.16e-4 | +2.34e-5 |
| 130 | 3.00e-2 | 2 | 4.06e-2 | 4.08e-2 | 4.07e-2 | 4.08e-2 | 167 | -2.16e-4 | +4.32e-5 | -8.66e-5 | +3.83e-6 |
| 131 | 3.00e-2 | 1 | 4.38e-2 | 4.38e-2 | 4.38e-2 | 4.38e-2 | 199 | +3.45e-4 | +3.45e-4 | +3.45e-4 | +3.80e-5 |
| 132 | 3.00e-2 | 2 | 4.23e-2 | 4.47e-2 | 4.35e-2 | 4.23e-2 | 162 | -3.40e-4 | +1.07e-4 | -1.16e-4 | +6.41e-6 |
| 133 | 3.00e-2 | 2 | 4.02e-2 | 4.13e-2 | 4.07e-2 | 4.02e-2 | 157 | -1.68e-4 | -1.43e-4 | -1.56e-4 | -2.45e-5 |
| 134 | 3.00e-2 | 1 | 4.36e-2 | 4.36e-2 | 4.36e-2 | 4.36e-2 | 172 | +4.67e-4 | +4.67e-4 | +4.67e-4 | +2.46e-5 |
| 135 | 3.00e-2 | 2 | 4.29e-2 | 4.43e-2 | 4.36e-2 | 4.29e-2 | 157 | -2.03e-4 | +9.01e-5 | -5.66e-5 | +7.72e-6 |
| 136 | 3.00e-2 | 1 | 4.57e-2 | 4.57e-2 | 4.57e-2 | 4.57e-2 | 193 | +3.29e-4 | +3.29e-4 | +3.29e-4 | +3.98e-5 |
| 137 | 3.00e-2 | 2 | 4.37e-2 | 4.53e-2 | 4.45e-2 | 4.37e-2 | 157 | -2.26e-4 | -5.51e-5 | -1.41e-4 | +4.71e-6 |
| 138 | 3.00e-2 | 1 | 4.94e-2 | 4.94e-2 | 4.94e-2 | 4.94e-2 | 206 | +5.97e-4 | +5.97e-4 | +5.97e-4 | +6.39e-5 |
| 139 | 3.00e-2 | 2 | 4.41e-2 | 4.83e-2 | 4.62e-2 | 4.41e-2 | 147 | -6.15e-4 | -1.21e-4 | -3.68e-4 | -2.07e-5 |
| 140 | 3.00e-2 | 2 | 4.47e-2 | 4.73e-2 | 4.60e-2 | 4.47e-2 | 147 | -3.75e-4 | +3.99e-4 | +1.17e-5 | -1.84e-5 |
| 141 | 3.00e-2 | 2 | 4.63e-2 | 4.71e-2 | 4.67e-2 | 4.63e-2 | 147 | -1.11e-4 | +3.02e-4 | +9.55e-5 | +1.20e-6 |
| 142 | 3.00e-2 | 1 | 5.03e-2 | 5.03e-2 | 5.03e-2 | 5.03e-2 | 190 | +4.27e-4 | +4.27e-4 | +4.27e-4 | +4.37e-5 |
| 143 | 3.00e-2 | 2 | 4.57e-2 | 5.23e-2 | 4.90e-2 | 4.57e-2 | 135 | -1.01e-3 | +2.00e-4 | -4.03e-4 | -4.72e-5 |
| 144 | 3.00e-2 | 2 | 4.57e-2 | 4.67e-2 | 4.62e-2 | 4.57e-2 | 135 | -1.62e-4 | +1.44e-4 | -9.19e-6 | -4.15e-5 |
| 145 | 3.00e-2 | 2 | 4.62e-2 | 4.96e-2 | 4.79e-2 | 4.62e-2 | 135 | -5.15e-4 | +4.72e-4 | -2.14e-5 | -4.26e-5 |
| 146 | 3.00e-2 | 1 | 4.77e-2 | 4.77e-2 | 4.77e-2 | 4.77e-2 | 151 | +2.11e-4 | +2.11e-4 | +2.11e-4 | -1.73e-5 |
| 147 | 3.00e-2 | 3 | 4.61e-2 | 4.98e-2 | 4.75e-2 | 4.65e-2 | 126 | -5.70e-4 | +2.52e-4 | -7.98e-5 | -3.56e-5 |
| 148 | 3.00e-2 | 1 | 5.14e-2 | 5.14e-2 | 5.14e-2 | 5.14e-2 | 162 | +6.06e-4 | +6.06e-4 | +6.06e-4 | +2.85e-5 |
| 149 | 3.00e-2 | 2 | 4.55e-2 | 5.09e-2 | 4.82e-2 | 4.55e-2 | 118 | -9.63e-4 | -4.77e-5 | -5.05e-4 | -7.74e-5 |
| 150 | 3.00e-3 | 2 | 2.58e-2 | 5.22e-2 | 3.90e-2 | 2.58e-2 | 118 | -5.96e-3 | +8.38e-4 | -2.56e-3 | -5.83e-4 |
| 151 | 3.00e-3 | 3 | 5.33e-3 | 1.31e-2 | 8.65e-3 | 5.33e-3 | 118 | -4.74e-3 | -2.93e-3 | -4.12e-3 | -1.53e-3 |
| 152 | 3.00e-3 | 1 | 5.08e-3 | 5.08e-3 | 5.08e-3 | 5.08e-3 | 171 | -2.78e-4 | -2.78e-4 | -2.78e-4 | -1.40e-3 |
| 153 | 3.00e-3 | 3 | 4.15e-3 | 5.06e-3 | 4.58e-3 | 4.15e-3 | 107 | -9.67e-4 | -1.94e-5 | -5.98e-4 | -1.19e-3 |
| 154 | 3.00e-3 | 2 | 4.24e-3 | 4.70e-3 | 4.47e-3 | 4.24e-3 | 107 | -9.67e-4 | +8.30e-4 | -6.85e-5 | -9.86e-4 |
| 155 | 3.00e-3 | 2 | 4.44e-3 | 4.76e-3 | 4.60e-3 | 4.44e-3 | 120 | -5.74e-4 | +7.49e-4 | +8.77e-5 | -7.89e-4 |
| 156 | 3.00e-3 | 3 | 4.25e-3 | 4.79e-3 | 4.51e-3 | 4.25e-3 | 113 | -5.73e-4 | +4.98e-4 | -1.86e-4 | -6.35e-4 |
| 157 | 3.00e-3 | 2 | 4.25e-3 | 4.53e-3 | 4.39e-3 | 4.25e-3 | 104 | -6.16e-4 | +5.02e-4 | -5.70e-5 | -5.31e-4 |
| 158 | 3.00e-3 | 3 | 4.15e-3 | 4.44e-3 | 4.25e-3 | 4.16e-3 | 96 | -7.02e-4 | +3.42e-4 | -1.08e-4 | -4.19e-4 |
| 159 | 3.00e-3 | 2 | 4.35e-3 | 4.76e-3 | 4.55e-3 | 4.35e-3 | 96 | -9.25e-4 | +9.35e-4 | +4.58e-6 | -3.47e-4 |
| 160 | 3.00e-3 | 3 | 4.16e-3 | 4.72e-3 | 4.39e-3 | 4.16e-3 | 108 | -1.05e-3 | +5.75e-4 | -2.35e-4 | -3.24e-4 |
| 161 | 3.00e-3 | 2 | 3.98e-3 | 4.41e-3 | 4.20e-3 | 3.98e-3 | 91 | -1.13e-3 | +4.72e-4 | -3.30e-4 | -3.33e-4 |
| 162 | 3.00e-3 | 3 | 3.85e-3 | 4.49e-3 | 4.07e-3 | 3.86e-3 | 80 | -1.82e-3 | +1.04e-3 | -2.46e-4 | -3.18e-4 |
| 163 | 3.00e-3 | 2 | 3.94e-3 | 6.30e-3 | 5.12e-3 | 6.30e-3 | 298 | +2.56e-4 | +1.57e-3 | +9.15e-4 | -7.75e-5 |
| 165 | 3.00e-3 | 2 | 7.03e-3 | 7.41e-3 | 7.22e-3 | 7.03e-3 | 292 | -1.83e-4 | +4.37e-4 | +1.27e-4 | -4.17e-5 |
| 167 | 3.00e-3 | 1 | 7.63e-3 | 7.63e-3 | 7.63e-3 | 7.63e-3 | 371 | +2.22e-4 | +2.22e-4 | +2.22e-4 | -1.54e-5 |
| 168 | 3.00e-3 | 1 | 6.88e-3 | 6.88e-3 | 6.88e-3 | 6.88e-3 | 278 | -3.69e-4 | -3.69e-4 | -3.69e-4 | -5.08e-5 |
| 169 | 3.00e-3 | 1 | 6.79e-3 | 6.79e-3 | 6.79e-3 | 6.79e-3 | 284 | -4.84e-5 | -4.84e-5 | -4.84e-5 | -5.05e-5 |
| 170 | 3.00e-3 | 1 | 6.92e-3 | 6.92e-3 | 6.92e-3 | 6.92e-3 | 275 | +6.79e-5 | +6.79e-5 | +6.79e-5 | -3.87e-5 |
| 171 | 3.00e-3 | 1 | 6.77e-3 | 6.77e-3 | 6.77e-3 | 6.77e-3 | 279 | -7.55e-5 | -7.55e-5 | -7.55e-5 | -4.24e-5 |
| 172 | 3.00e-3 | 1 | 6.87e-3 | 6.87e-3 | 6.87e-3 | 6.87e-3 | 262 | +5.47e-5 | +5.47e-5 | +5.47e-5 | -3.27e-5 |
| 173 | 3.00e-3 | 1 | 6.96e-3 | 6.96e-3 | 6.96e-3 | 6.96e-3 | 284 | +4.38e-5 | +4.38e-5 | +4.38e-5 | -2.50e-5 |
| 174 | 3.00e-3 | 1 | 7.40e-3 | 7.40e-3 | 7.40e-3 | 7.40e-3 | 312 | +1.98e-4 | +1.98e-4 | +1.98e-4 | -2.71e-6 |
| 175 | 3.00e-3 | 1 | 7.29e-3 | 7.29e-3 | 7.29e-3 | 7.29e-3 | 294 | -4.98e-5 | -4.98e-5 | -4.98e-5 | -7.41e-6 |
| 176 | 3.00e-3 | 1 | 7.11e-3 | 7.11e-3 | 7.11e-3 | 7.11e-3 | 281 | -9.12e-5 | -9.12e-5 | -9.12e-5 | -1.58e-5 |
| 177 | 3.00e-3 | 1 | 6.99e-3 | 6.99e-3 | 6.99e-3 | 6.99e-3 | 255 | -6.67e-5 | -6.67e-5 | -6.67e-5 | -2.09e-5 |
| 179 | 3.00e-3 | 2 | 7.13e-3 | 8.09e-3 | 7.61e-3 | 7.13e-3 | 234 | -5.42e-4 | +3.84e-4 | -7.92e-5 | -3.66e-5 |
| 181 | 3.00e-3 | 2 | 6.93e-3 | 7.38e-3 | 7.16e-3 | 6.93e-3 | 234 | -2.71e-4 | +1.14e-4 | -7.88e-5 | -4.65e-5 |
| 182 | 3.00e-3 | 1 | 6.96e-3 | 6.96e-3 | 6.96e-3 | 6.96e-3 | 271 | +1.68e-5 | +1.68e-5 | +1.68e-5 | -4.02e-5 |
| 183 | 3.00e-3 | 1 | 6.88e-3 | 6.88e-3 | 6.88e-3 | 6.88e-3 | 245 | -4.96e-5 | -4.96e-5 | -4.96e-5 | -4.11e-5 |
| 184 | 3.00e-3 | 1 | 7.03e-3 | 7.03e-3 | 7.03e-3 | 7.03e-3 | 257 | +8.71e-5 | +8.71e-5 | +8.71e-5 | -2.83e-5 |
| 185 | 3.00e-3 | 1 | 7.12e-3 | 7.12e-3 | 7.12e-3 | 7.12e-3 | 277 | +4.28e-5 | +4.28e-5 | +4.28e-5 | -2.12e-5 |
| 186 | 3.00e-3 | 1 | 7.30e-3 | 7.30e-3 | 7.30e-3 | 7.30e-3 | 259 | +9.78e-5 | +9.78e-5 | +9.78e-5 | -9.31e-6 |
| 187 | 3.00e-3 | 1 | 7.33e-3 | 7.33e-3 | 7.33e-3 | 7.33e-3 | 282 | +1.66e-5 | +1.66e-5 | +1.66e-5 | -6.72e-6 |
| 188 | 3.00e-3 | 1 | 7.24e-3 | 7.24e-3 | 7.24e-3 | 7.24e-3 | 253 | -5.03e-5 | -5.03e-5 | -5.03e-5 | -1.11e-5 |
| 189 | 3.00e-3 | 1 | 7.28e-3 | 7.28e-3 | 7.28e-3 | 7.28e-3 | 270 | +2.19e-5 | +2.19e-5 | +2.19e-5 | -7.78e-6 |
| 190 | 3.00e-3 | 1 | 7.06e-3 | 7.06e-3 | 7.06e-3 | 7.06e-3 | 276 | -1.10e-4 | -1.10e-4 | -1.10e-4 | -1.80e-5 |
| 191 | 3.00e-3 | 1 | 7.49e-3 | 7.49e-3 | 7.49e-3 | 7.49e-3 | 289 | +2.03e-4 | +2.03e-4 | +2.03e-4 | +4.13e-6 |
| 192 | 3.00e-3 | 1 | 7.34e-3 | 7.34e-3 | 7.34e-3 | 7.34e-3 | 267 | -7.49e-5 | -7.49e-5 | -7.49e-5 | -3.77e-6 |
| 193 | 3.00e-3 | 1 | 7.35e-3 | 7.35e-3 | 7.35e-3 | 7.35e-3 | 260 | +4.88e-7 | +4.88e-7 | +4.88e-7 | -3.34e-6 |
| 194 | 3.00e-3 | 1 | 7.08e-3 | 7.08e-3 | 7.08e-3 | 7.08e-3 | 225 | -1.65e-4 | -1.65e-4 | -1.65e-4 | -1.96e-5 |
| 195 | 3.00e-3 | 1 | 7.26e-3 | 7.26e-3 | 7.26e-3 | 7.26e-3 | 255 | +9.75e-5 | +9.75e-5 | +9.75e-5 | -7.85e-6 |
| 196 | 3.00e-3 | 1 | 7.21e-3 | 7.21e-3 | 7.21e-3 | 7.21e-3 | 246 | -2.56e-5 | -2.56e-5 | -2.56e-5 | -9.63e-6 |
| 197 | 3.00e-3 | 1 | 7.31e-3 | 7.31e-3 | 7.31e-3 | 7.31e-3 | 270 | +5.17e-5 | +5.17e-5 | +5.17e-5 | -3.49e-6 |
| 198 | 3.00e-3 | 2 | 6.96e-3 | 7.39e-3 | 7.17e-3 | 6.96e-3 | 224 | -2.69e-4 | +3.83e-5 | -1.15e-4 | -2.63e-5 |

