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
| cpu-async | 0.063506 | 0.9220 | +0.0095 | 1913.4 | 725 | 82.8 | 100% | 100% | 7.3 |

## Best Mode per Model

| Model | Best Eval | Mode | Fastest (within 2% of solo-0) | Mode |
|-------|-----------|------|-------------------------------|------|
| resnet-graph | 0.9220 | cpu-async | - | - |

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
| cpu-async | 2.0091 | 0.7335 | 0.5729 | 0.5098 | 0.4729 | 0.4707 | 0.4481 | 0.4407 | 0.4450 | 0.4628 | 0.2121 | 0.1776 | 0.1617 | 0.1466 | 0.1434 | 0.0834 | 0.0753 | 0.0714 | 0.0687 | 0.0635 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | cpu-async | 0 | cuda:0 | 0.4027 | 2.6 |
| resnet-graph | cpu-async | 1 | cuda:1 | 0.3042 | 3.3 |
| resnet-graph | cpu-async | 2 | cuda:2 | 0.2931 | 3.4 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|
| resnet-graph | cpu-async | 358 | 357 | 398 | 393 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| cpu-async | gpu1 | 1529.5 | 1.5 | epoch-boundary(159) |
| cpu-async | gpu2 | 1529.6 | 1.4 | epoch-boundary(159) |
| cpu-async | gpu1 | 1072.7 | 0.9 | epoch-boundary(111) |
| cpu-async | gpu2 | 1072.7 | 0.9 | epoch-boundary(111) |
| cpu-async | gpu1 | 1912.7 | 0.6 | epoch-boundary(199) |
| cpu-async | gpu2 | 1912.7 | 0.6 | epoch-boundary(199) |
| cpu-async | gpu0 | 0.3 | 0.6 | cpu-avg |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | cpu-async | gpu0 | 0.0s | 0.0s | 0.6s | 0.0s | 0.6s |
| resnet-graph | cpu-async | gpu1 | 3.0s | 0.0s | 0.0s | 0.0s | 3.8s |
| resnet-graph | cpu-async | gpu2 | 2.9s | 0.0s | 0.0s | 0.0s | 2.9s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | cpu-async | 289 | 0 | 725 | 82.8 | 1327/9555 | 725 | 82.8 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | cpu-async | 167.6 | 8.8% |

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
| resnet-graph | cpu-async | 193 | 725 | 0 | 6.24e-3 | -6.08e-5 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | cpu-async | 0 | 725 | 1.06e-1 | 3.90e-2 | 5.92e-3 | 3.44e-1 | 31.3 | -1.84e-4 | 1.22e-3 |
| resnet-graph | cpu-async | 1 | 725 | 1.06e-1 | 3.97e-2 | 5.80e-3 | 3.49e-1 | 32.3 | -1.99e-4 | 1.19e-3 |
| resnet-graph | cpu-async | 2 | 725 | 1.07e-1 | 4.10e-2 | 5.89e-3 | 4.00e-1 | 36.4 | -2.11e-4 | 1.29e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | cpu-async | rank0 ↔ rank1 | +0.9804 |
| resnet-graph | cpu-async | rank0 ↔ rank2 | +0.9742 |
| resnet-graph | cpu-async | rank1 ↔ rank2 | +0.9776 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | cpu-async | 87 (0,2,3,4,5,6,7,9…141,149) | 0 (—) | — | 0,2,3,4,5,6,7,9…141,149 | — |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | cpu-async | 0e0 | 3 | 6 | 6 |
| resnet-graph | cpu-async | 0e0 | 5 | 3 | 3 |
| resnet-graph | cpu-async | 0e0 | 10 | 1 | 1 |
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
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 617 | +0.117 |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | 50 | +0.094 |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | 54 | -0.404 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | cpu-async | 723 | -0.039 | 192 | +0.486 | +0.759 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | cpu-async | 724 | 3.34e1–8.06e1 | 7.07e1 | 2.02e-3 | 3.61e-3 | 3.31e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 619 | 33–77846 | -1.486e-7 | 0.000 | +8.642e-9 | 0.000 | 100 | +2.177e-6 | 0.096 | 31–955 | +1.279e-3 | 0.445 |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 608 | 960–77846 | +5.500e-7 | 0.007 | +6.313e-7 | 0.011 | 99 | +2.535e-6 | 0.133 | 74–955 | +1.393e-3 | 0.691 |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | 51 | 78628–116956 | +8.827e-6 | 0.083 | +9.048e-6 | 0.087 | 44 | +6.234e-6 | 0.040 | 524–961 | +5.951e-4 | 0.025 |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | 55 | 117736–155928 | -1.276e-5 | 0.160 | -1.276e-5 | 0.165 | 49 | -1.372e-5 | 0.169 | 481–939 | +6.096e-4 | 0.025 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | +1.279e-3 | r0: +1.188e-3, r1: +1.313e-3, r2: +1.328e-3 | r0: 0.414, r1: 0.429, r2: 0.409 | 1.12× | ✓ |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +1.393e-3 | r0: +1.297e-3, r1: +1.428e-3, r2: +1.445e-3 | r0: 0.647, r1: 0.634, r2: 0.633 | 1.11× | ✓ |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | +5.951e-4 | r0: +5.644e-4, r1: +5.534e-4, r2: +6.650e-4 | r0: 0.023, r1: 0.022, r2: 0.031 | 1.20× | ✓ |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | +6.096e-4 | r0: +6.087e-4, r1: +6.373e-4, r2: +5.821e-4 | r0: 0.025, r1: 0.027, r2: 0.022 | 1.09× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | cpu-async | `█▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇█▆▄▄▅▅▅▅▅▅▅▅▃▁▁▁▁▁▁▁▁▁▁▁▁` | `▁▆▇▇▇▇▇▇▇▆▆▇▇▆▇▆▆▇▇▇▇▇▇▇█▇▇▇████████▇▇▇▇█████████` |

### Per-Epoch Detail: resnet-graph / cpu-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 11 | 1.22e-1 | 4.00e-1 | 2.08e-1 | 1.45e-1 | 36 | -3.46e-2 | +1.01e-2 | -8.63e-3 | -6.49e-3 |
| 1 | 3.00e-1 | 4 | 1.29e-1 | 1.84e-1 | 1.48e-1 | 1.29e-1 | 61 | -4.48e-3 | +2.28e-3 | -9.56e-4 | -4.05e-3 |
| 2 | 3.00e-1 | 5 | 1.33e-1 | 1.49e-1 | 1.42e-1 | 1.48e-1 | 62 | -1.04e-3 | +1.57e-3 | +2.82e-4 | -1.99e-3 |
| 3 | 3.00e-1 | 5 | 1.26e-1 | 1.75e-1 | 1.40e-1 | 1.29e-1 | 47 | -4.78e-3 | +1.70e-3 | -8.88e-4 | -1.48e-3 |
| 4 | 3.00e-1 | 5 | 1.23e-1 | 1.57e-1 | 1.33e-1 | 1.24e-1 | 42 | -2.82e-3 | +2.50e-3 | -4.92e-4 | -1.07e-3 |
| 5 | 3.00e-1 | 6 | 1.20e-1 | 1.52e-1 | 1.28e-1 | 1.27e-1 | 45 | -4.36e-3 | +2.81e-3 | -2.58e-4 | -6.48e-4 |
| 6 | 3.00e-1 | 7 | 1.11e-1 | 1.60e-1 | 1.22e-1 | 1.11e-1 | 38 | -6.60e-3 | +2.64e-3 | -1.03e-3 | -8.13e-4 |
| 7 | 3.00e-1 | 6 | 1.16e-1 | 1.62e-1 | 1.27e-1 | 1.21e-1 | 41 | -7.35e-3 | +4.34e-3 | -5.82e-4 | -6.85e-4 |
| 8 | 3.00e-1 | 6 | 1.17e-1 | 1.61e-1 | 1.29e-1 | 1.24e-1 | 41 | -6.01e-3 | +3.37e-3 | -4.68e-4 | -5.55e-4 |
| 9 | 3.00e-1 | 6 | 1.13e-1 | 1.58e-1 | 1.24e-1 | 1.16e-1 | 40 | -5.61e-3 | +3.10e-3 | -7.75e-4 | -6.30e-4 |
| 10 | 3.00e-1 | 8 | 1.03e-1 | 1.54e-1 | 1.16e-1 | 1.12e-1 | 39 | -6.46e-3 | +3.51e-3 | -6.88e-4 | -5.70e-4 |
| 11 | 3.00e-1 | 6 | 1.09e-1 | 1.55e-1 | 1.22e-1 | 1.20e-1 | 42 | -7.11e-3 | +4.09e-3 | -5.79e-4 | -5.23e-4 |
| 12 | 3.00e-1 | 6 | 1.18e-1 | 1.50e-1 | 1.26e-1 | 1.22e-1 | 46 | -4.66e-3 | +2.95e-3 | -3.32e-4 | -4.21e-4 |
| 13 | 3.00e-1 | 6 | 1.19e-1 | 1.51e-1 | 1.29e-1 | 1.20e-1 | 42 | -3.52e-3 | +2.78e-3 | -3.83e-4 | -4.23e-4 |
| 14 | 3.00e-1 | 7 | 1.10e-1 | 1.54e-1 | 1.18e-1 | 1.10e-1 | 37 | -6.93e-3 | +2.86e-3 | -8.85e-4 | -6.14e-4 |
| 15 | 3.00e-1 | 6 | 1.04e-1 | 1.63e-1 | 1.21e-1 | 1.04e-1 | 32 | -7.61e-3 | +4.09e-3 | -1.35e-3 | -9.80e-4 |
| 16 | 3.00e-1 | 8 | 1.02e-1 | 1.48e-1 | 1.14e-1 | 1.09e-1 | 35 | -8.49e-3 | +4.65e-3 | -8.21e-4 | -8.29e-4 |
| 17 | 3.00e-1 | 8 | 9.84e-2 | 1.33e-1 | 1.09e-1 | 1.12e-1 | 35 | -1.14e-2 | +3.37e-3 | -5.36e-4 | -5.22e-4 |
| 18 | 3.00e-1 | 8 | 1.14e-1 | 1.53e-1 | 1.21e-1 | 1.20e-1 | 43 | -8.30e-3 | +4.17e-3 | -3.67e-4 | -3.66e-4 |
| 19 | 3.00e-1 | 5 | 1.08e-1 | 1.55e-1 | 1.21e-1 | 1.16e-1 | 36 | -7.78e-3 | +3.06e-3 | -1.13e-3 | -6.24e-4 |
| 20 | 3.00e-1 | 7 | 1.06e-1 | 1.51e-1 | 1.17e-1 | 1.16e-1 | 38 | -6.97e-3 | +3.50e-3 | -5.56e-4 | -4.94e-4 |
| 21 | 3.00e-1 | 9 | 1.07e-1 | 1.50e-1 | 1.16e-1 | 1.13e-1 | 36 | -6.39e-3 | +3.29e-3 | -4.46e-4 | -3.73e-4 |
| 22 | 3.00e-1 | 5 | 1.08e-1 | 1.53e-1 | 1.20e-1 | 1.08e-1 | 34 | -7.50e-3 | +3.93e-3 | -1.30e-3 | -7.53e-4 |
| 23 | 3.00e-1 | 7 | 1.10e-1 | 1.44e-1 | 1.18e-1 | 1.14e-1 | 39 | -7.28e-3 | +4.09e-3 | -4.27e-4 | -5.53e-4 |
| 24 | 3.00e-1 | 8 | 1.04e-1 | 1.59e-1 | 1.21e-1 | 1.26e-1 | 47 | -8.80e-3 | +3.95e-3 | -5.70e-4 | -4.18e-4 |
| 25 | 3.00e-1 | 4 | 1.19e-1 | 1.60e-1 | 1.32e-1 | 1.19e-1 | 46 | -5.90e-3 | +2.82e-3 | -9.98e-4 | -6.39e-4 |
| 26 | 3.00e-1 | 6 | 1.18e-1 | 1.54e-1 | 1.28e-1 | 1.23e-1 | 45 | -3.46e-3 | +2.96e-3 | -2.87e-4 | -4.75e-4 |
| 27 | 3.00e-1 | 6 | 1.10e-1 | 1.49e-1 | 1.20e-1 | 1.24e-1 | 47 | -7.77e-3 | +2.73e-3 | -5.49e-4 | -4.10e-4 |
| 28 | 3.00e-1 | 8 | 1.14e-1 | 1.57e-1 | 1.24e-1 | 1.17e-1 | 40 | -5.16e-3 | +2.89e-3 | -5.55e-4 | -4.58e-4 |
| 29 | 3.00e-1 | 4 | 1.23e-1 | 1.48e-1 | 1.31e-1 | 1.26e-1 | 47 | -4.35e-3 | +3.21e-3 | -1.19e-4 | -3.58e-4 |
| 30 | 3.00e-1 | 6 | 1.12e-1 | 1.56e-1 | 1.25e-1 | 1.14e-1 | 39 | -4.60e-3 | +2.52e-3 | -8.01e-4 | -5.56e-4 |
| 31 | 3.00e-1 | 7 | 1.02e-1 | 1.52e-1 | 1.16e-1 | 1.09e-1 | 36 | -7.35e-3 | +3.58e-3 | -9.48e-4 | -6.92e-4 |
| 32 | 3.00e-1 | 8 | 1.10e-1 | 1.63e-1 | 1.23e-1 | 1.16e-1 | 41 | -4.90e-3 | +4.37e-3 | -3.85e-4 | -4.88e-4 |
| 33 | 3.00e-1 | 5 | 1.10e-1 | 1.54e-1 | 1.23e-1 | 1.17e-1 | 40 | -5.65e-3 | +3.77e-3 | -6.06e-4 | -5.19e-4 |
| 34 | 3.00e-1 | 7 | 1.08e-1 | 1.51e-1 | 1.17e-1 | 1.09e-1 | 34 | -5.62e-3 | +3.20e-3 | -8.60e-4 | -6.51e-4 |
| 35 | 3.00e-1 | 6 | 1.04e-1 | 1.58e-1 | 1.20e-1 | 1.21e-1 | 44 | -9.67e-3 | +4.62e-3 | -6.93e-4 | -5.83e-4 |
| 36 | 3.00e-1 | 7 | 1.12e-1 | 1.57e-1 | 1.21e-1 | 1.19e-1 | 45 | -9.02e-3 | +3.22e-3 | -7.52e-4 | -5.74e-4 |
| 37 | 3.00e-1 | 7 | 1.14e-1 | 1.61e-1 | 1.27e-1 | 1.14e-1 | 41 | -4.73e-3 | +3.36e-3 | -6.22e-4 | -6.17e-4 |
| 38 | 3.00e-1 | 6 | 1.03e-1 | 1.70e-1 | 1.20e-1 | 1.03e-1 | 31 | -1.06e-2 | +3.82e-3 | -1.93e-3 | -1.21e-3 |
| 39 | 3.00e-1 | 7 | 1.09e-1 | 1.43e-1 | 1.16e-1 | 1.09e-1 | 34 | -9.05e-3 | +5.06e-3 | -6.05e-4 | -8.90e-4 |
| 40 | 3.00e-1 | 7 | 1.09e-1 | 1.52e-1 | 1.20e-1 | 1.21e-1 | 42 | -9.40e-3 | +4.53e-3 | -4.90e-4 | -5.95e-4 |
| 41 | 3.00e-1 | 6 | 1.12e-1 | 1.53e-1 | 1.22e-1 | 1.12e-1 | 35 | -5.34e-3 | +2.93e-3 | -8.93e-4 | -7.29e-4 |
| 42 | 3.00e-1 | 8 | 1.01e-1 | 1.57e-1 | 1.12e-1 | 1.01e-1 | 29 | -8.34e-3 | +3.90e-3 | -1.13e-3 | -8.79e-4 |
| 43 | 3.00e-1 | 7 | 1.06e-1 | 1.55e-1 | 1.15e-1 | 1.09e-1 | 32 | -9.27e-3 | +5.25e-3 | -8.01e-4 | -7.73e-4 |
| 44 | 3.00e-1 | 8 | 9.73e-2 | 1.54e-1 | 1.10e-1 | 1.08e-1 | 37 | -1.14e-2 | +4.27e-3 | -1.06e-3 | -7.40e-4 |
| 45 | 3.00e-1 | 10 | 9.83e-2 | 1.51e-1 | 1.16e-1 | 1.18e-1 | 41 | -4.66e-3 | +4.61e-3 | -2.64e-4 | -3.04e-4 |
| 46 | 3.00e-1 | 4 | 1.17e-1 | 1.60e-1 | 1.33e-1 | 1.17e-1 | 44 | -4.68e-3 | +3.60e-3 | -8.70e-4 | -5.47e-4 |
| 47 | 3.00e-1 | 6 | 1.12e-1 | 1.61e-1 | 1.24e-1 | 1.12e-1 | 36 | -6.48e-3 | +3.63e-3 | -9.96e-4 | -7.62e-4 |
| 48 | 3.00e-1 | 8 | 9.74e-2 | 1.61e-1 | 1.11e-1 | 9.86e-2 | 27 | -8.23e-3 | +4.13e-3 | -1.27e-3 | -9.76e-4 |
| 49 | 3.00e-1 | 7 | 1.06e-1 | 1.43e-1 | 1.17e-1 | 1.13e-1 | 36 | -6.36e-3 | +5.41e-3 | -2.87e-4 | -6.15e-4 |
| 50 | 3.00e-1 | 8 | 1.07e-1 | 1.56e-1 | 1.16e-1 | 1.12e-1 | 39 | -1.00e-2 | +4.36e-3 | -7.74e-4 | -5.91e-4 |
| 51 | 3.00e-1 | 9 | 9.74e-2 | 1.62e-1 | 1.16e-1 | 1.15e-1 | 40 | -1.16e-2 | +4.26e-3 | -8.75e-4 | -5.70e-4 |
| 52 | 3.00e-1 | 5 | 1.08e-1 | 1.67e-1 | 1.25e-1 | 1.08e-1 | 34 | -5.93e-3 | +4.05e-3 | -1.36e-3 | -9.22e-4 |
| 53 | 3.00e-1 | 7 | 1.04e-1 | 1.49e-1 | 1.17e-1 | 1.17e-1 | 38 | -1.03e-2 | +4.51e-3 | -5.67e-4 | -6.39e-4 |
| 54 | 3.00e-1 | 7 | 1.04e-1 | 1.62e-1 | 1.19e-1 | 1.18e-1 | 40 | -9.52e-3 | +3.83e-3 | -8.53e-4 | -6.13e-4 |
| 55 | 3.00e-1 | 6 | 1.06e-1 | 1.66e-1 | 1.24e-1 | 1.19e-1 | 41 | -9.13e-3 | +3.74e-3 | -9.90e-4 | -7.09e-4 |
| 56 | 3.00e-1 | 7 | 1.11e-1 | 1.44e-1 | 1.22e-1 | 1.21e-1 | 41 | -6.13e-3 | +2.86e-3 | -2.75e-4 | -4.39e-4 |
| 57 | 3.00e-1 | 6 | 1.08e-1 | 1.60e-1 | 1.23e-1 | 1.22e-1 | 41 | -9.10e-3 | +3.28e-3 | -8.94e-4 | -5.63e-4 |
| 58 | 3.00e-1 | 8 | 1.12e-1 | 1.60e-1 | 1.21e-1 | 1.13e-1 | 39 | -8.61e-3 | +3.52e-3 | -8.07e-4 | -6.26e-4 |
| 59 | 3.00e-1 | 5 | 1.09e-1 | 1.58e-1 | 1.23e-1 | 1.14e-1 | 39 | -8.51e-3 | +3.97e-3 | -1.21e-3 | -8.40e-4 |
| 60 | 3.00e-1 | 6 | 1.05e-1 | 1.63e-1 | 1.25e-1 | 1.05e-1 | 32 | -4.43e-3 | +3.79e-3 | -1.23e-3 | -1.10e-3 |
| 61 | 3.00e-1 | 7 | 1.07e-1 | 1.53e-1 | 1.19e-1 | 1.17e-1 | 41 | -9.02e-3 | +4.81e-3 | -5.53e-4 | -7.45e-4 |
| 62 | 3.00e-1 | 5 | 1.21e-1 | 1.62e-1 | 1.30e-1 | 1.21e-1 | 44 | -6.29e-3 | +3.63e-3 | -6.08e-4 | -6.96e-4 |
| 63 | 3.00e-1 | 10 | 1.07e-1 | 1.55e-1 | 1.18e-1 | 1.16e-1 | 38 | -5.32e-3 | +3.15e-3 | -4.25e-4 | -3.96e-4 |
| 64 | 3.00e-1 | 4 | 1.06e-1 | 1.59e-1 | 1.25e-1 | 1.06e-1 | 34 | -6.90e-3 | +3.88e-3 | -1.77e-3 | -9.21e-4 |
| 65 | 3.00e-1 | 9 | 1.05e-1 | 1.47e-1 | 1.15e-1 | 1.16e-1 | 39 | -9.10e-3 | +4.68e-3 | -3.67e-4 | -4.68e-4 |
| 66 | 3.00e-1 | 5 | 1.11e-1 | 1.54e-1 | 1.24e-1 | 1.11e-1 | 37 | -5.10e-3 | +3.47e-3 | -9.76e-4 | -6.94e-4 |
| 67 | 3.00e-1 | 7 | 1.00e-1 | 1.62e-1 | 1.21e-1 | 1.03e-1 | 31 | -3.48e-3 | +4.50e-3 | -9.44e-4 | -9.11e-4 |
| 68 | 3.00e-1 | 6 | 1.20e-1 | 1.55e-1 | 1.31e-1 | 1.36e-1 | 58 | -5.47e-3 | +4.98e-3 | +2.37e-4 | -3.61e-4 |
| 69 | 3.00e-1 | 4 | 1.39e-1 | 1.66e-1 | 1.47e-1 | 1.39e-1 | 62 | -2.43e-3 | +2.19e-3 | -1.66e-4 | -3.16e-4 |
| 70 | 3.00e-1 | 7 | 1.26e-1 | 1.66e-1 | 1.35e-1 | 1.28e-1 | 48 | -3.56e-3 | +2.05e-3 | -4.27e-4 | -3.49e-4 |
| 71 | 3.00e-1 | 3 | 1.27e-1 | 1.60e-1 | 1.41e-1 | 1.27e-1 | 46 | -3.09e-3 | +2.58e-3 | -6.51e-4 | -4.68e-4 |
| 72 | 3.00e-1 | 7 | 1.05e-1 | 1.59e-1 | 1.19e-1 | 1.10e-1 | 35 | -9.72e-3 | +2.68e-3 | -1.22e-3 | -7.82e-4 |
| 73 | 3.00e-1 | 7 | 1.14e-1 | 1.48e-1 | 1.21e-1 | 1.14e-1 | 36 | -5.89e-3 | +4.34e-3 | -4.04e-4 | -6.03e-4 |
| 74 | 3.00e-1 | 7 | 1.14e-1 | 1.49e-1 | 1.21e-1 | 1.14e-1 | 34 | -7.04e-3 | +3.69e-3 | -6.24e-4 | -6.05e-4 |
| 75 | 3.00e-1 | 6 | 1.10e-1 | 1.54e-1 | 1.21e-1 | 1.19e-1 | 40 | -1.08e-2 | +3.97e-3 | -8.17e-4 | -6.27e-4 |
| 76 | 3.00e-1 | 6 | 1.23e-1 | 1.60e-1 | 1.32e-1 | 1.23e-1 | 43 | -3.84e-3 | +3.40e-3 | -3.79e-4 | -5.28e-4 |
| 77 | 3.00e-1 | 9 | 1.08e-1 | 1.59e-1 | 1.18e-1 | 1.08e-1 | 33 | -7.61e-3 | +3.11e-3 | -8.31e-4 | -6.45e-4 |
| 78 | 3.00e-1 | 5 | 1.11e-1 | 1.57e-1 | 1.23e-1 | 1.11e-1 | 36 | -7.28e-3 | +4.66e-3 | -9.71e-4 | -7.91e-4 |
| 79 | 3.00e-1 | 10 | 1.01e-1 | 1.57e-1 | 1.13e-1 | 1.07e-1 | 32 | -8.04e-3 | +4.57e-3 | -6.97e-4 | -6.08e-4 |
| 80 | 3.00e-1 | 4 | 1.09e-1 | 1.50e-1 | 1.23e-1 | 1.19e-1 | 41 | -1.18e-2 | +4.78e-3 | -1.22e-3 | -8.10e-4 |
| 81 | 3.00e-1 | 6 | 1.16e-1 | 1.65e-1 | 1.27e-1 | 1.18e-1 | 40 | -6.90e-3 | +3.90e-3 | -7.46e-4 | -7.62e-4 |
| 82 | 3.00e-1 | 7 | 1.19e-1 | 1.59e-1 | 1.27e-1 | 1.22e-1 | 43 | -5.10e-3 | +3.83e-3 | -3.26e-4 | -5.13e-4 |
| 83 | 3.00e-1 | 5 | 1.21e-1 | 1.57e-1 | 1.30e-1 | 1.25e-1 | 51 | -5.17e-3 | +3.05e-3 | -4.93e-4 | -4.97e-4 |
| 84 | 3.00e-1 | 5 | 1.22e-1 | 1.63e-1 | 1.35e-1 | 1.22e-1 | 43 | -3.72e-3 | +3.04e-3 | -5.69e-4 | -5.58e-4 |
| 85 | 3.00e-1 | 8 | 1.17e-1 | 1.61e-1 | 1.27e-1 | 1.29e-1 | 53 | -5.96e-3 | +3.31e-3 | -3.34e-4 | -3.71e-4 |
| 86 | 3.00e-1 | 4 | 1.16e-1 | 1.54e-1 | 1.30e-1 | 1.23e-1 | 46 | -4.38e-3 | +2.34e-3 | -8.22e-4 | -5.27e-4 |
| 87 | 3.00e-1 | 6 | 1.21e-1 | 1.63e-1 | 1.29e-1 | 1.22e-1 | 43 | -6.06e-3 | +3.30e-3 | -5.91e-4 | -5.38e-4 |
| 88 | 3.00e-1 | 6 | 1.16e-1 | 1.60e-1 | 1.27e-1 | 1.23e-1 | 44 | -5.51e-3 | +3.30e-3 | -5.63e-4 | -5.08e-4 |
| 89 | 3.00e-1 | 6 | 1.15e-1 | 1.60e-1 | 1.26e-1 | 1.15e-1 | 37 | -4.27e-3 | +3.17e-3 | -7.34e-4 | -6.15e-4 |
| 90 | 3.00e-1 | 7 | 1.13e-1 | 1.56e-1 | 1.20e-1 | 1.14e-1 | 39 | -8.60e-3 | +3.77e-3 | -7.24e-4 | -6.11e-4 |
| 91 | 3.00e-1 | 8 | 1.10e-1 | 1.56e-1 | 1.21e-1 | 1.10e-1 | 35 | -6.03e-3 | +3.93e-3 | -6.65e-4 | -6.45e-4 |
| 92 | 3.00e-1 | 5 | 1.13e-1 | 1.55e-1 | 1.27e-1 | 1.26e-1 | 44 | -5.52e-3 | +4.42e-3 | -1.47e-4 | -4.25e-4 |
| 93 | 3.00e-1 | 8 | 1.07e-1 | 1.64e-1 | 1.23e-1 | 1.12e-1 | 36 | -5.72e-3 | +3.14e-3 | -7.93e-4 | -5.86e-4 |
| 94 | 3.00e-1 | 1 | 1.13e-1 | 1.13e-1 | 1.13e-1 | 1.13e-1 | 37 | +2.91e-4 | +2.91e-4 | +2.91e-4 | -4.99e-4 |
| 95 | 3.00e-1 | 1 | 2.44e-1 | 2.44e-1 | 2.44e-1 | 2.44e-1 | 354 | +2.17e-3 | +2.17e-3 | +2.17e-3 | -2.31e-4 |
| 96 | 3.00e-1 | 1 | 2.37e-1 | 2.37e-1 | 2.37e-1 | 2.37e-1 | 289 | -1.09e-4 | -1.09e-4 | -1.09e-4 | -2.19e-4 |
| 97 | 3.00e-1 | 1 | 2.33e-1 | 2.33e-1 | 2.33e-1 | 2.33e-1 | 300 | -5.36e-5 | -5.36e-5 | -5.36e-5 | -2.03e-4 |
| 98 | 3.00e-1 | 1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 287 | -8.57e-5 | -8.57e-5 | -8.57e-5 | -1.91e-4 |
| 99 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 262 | -1.38e-4 | -1.38e-4 | -1.38e-4 | -1.86e-4 |
| 100 | 3.00e-2 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 266 | +4.30e-6 | +4.30e-6 | +4.30e-6 | -1.67e-4 |
| 101 | 3.00e-2 | 1 | 1.13e-1 | 1.13e-1 | 1.13e-1 | 1.13e-1 | 316 | -2.11e-3 | -2.11e-3 | -2.11e-3 | -3.61e-4 |
| 102 | 3.00e-2 | 1 | 5.88e-2 | 5.88e-2 | 5.88e-2 | 5.88e-2 | 305 | -2.13e-3 | -2.13e-3 | -2.13e-3 | -5.38e-4 |
| 103 | 3.00e-2 | 1 | 3.74e-2 | 3.74e-2 | 3.74e-2 | 3.74e-2 | 309 | -1.47e-3 | -1.47e-3 | -1.47e-3 | -6.31e-4 |
| 105 | 3.00e-2 | 2 | 2.87e-2 | 3.33e-2 | 3.10e-2 | 2.87e-2 | 244 | -6.12e-4 | -3.25e-4 | -4.68e-4 | -6.01e-4 |
| 107 | 3.00e-2 | 2 | 2.97e-2 | 3.22e-2 | 3.10e-2 | 2.97e-2 | 244 | -3.20e-4 | +3.49e-4 | +1.44e-5 | -4.88e-4 |
| 109 | 3.00e-2 | 2 | 3.05e-2 | 3.18e-2 | 3.12e-2 | 3.05e-2 | 244 | -1.80e-4 | +2.36e-4 | +2.80e-5 | -3.92e-4 |
| 110 | 3.00e-2 | 1 | 3.17e-2 | 3.17e-2 | 3.17e-2 | 3.17e-2 | 262 | +1.52e-4 | +1.52e-4 | +1.52e-4 | -3.37e-4 |
| 111 | 3.00e-2 | 1 | 3.24e-2 | 3.24e-2 | 3.24e-2 | 3.24e-2 | 262 | +7.91e-5 | +7.91e-5 | +7.91e-5 | -2.96e-4 |
| 112 | 3.00e-2 | 1 | 3.35e-2 | 3.35e-2 | 3.35e-2 | 3.35e-2 | 260 | +1.34e-4 | +1.34e-4 | +1.34e-4 | -2.53e-4 |
| 113 | 3.00e-2 | 1 | 3.55e-2 | 3.55e-2 | 3.55e-2 | 3.55e-2 | 289 | +2.01e-4 | +2.01e-4 | +2.01e-4 | -2.07e-4 |
| 115 | 3.00e-2 | 2 | 3.69e-2 | 3.97e-2 | 3.83e-2 | 3.69e-2 | 260 | -2.80e-4 | +3.10e-4 | +1.50e-5 | -1.68e-4 |
| 117 | 3.00e-2 | 2 | 3.73e-2 | 4.01e-2 | 3.87e-2 | 3.73e-2 | 240 | -3.04e-4 | +2.62e-4 | -2.08e-5 | -1.43e-4 |
| 118 | 3.00e-2 | 1 | 3.62e-2 | 3.62e-2 | 3.62e-2 | 3.62e-2 | 227 | -1.29e-4 | -1.29e-4 | -1.29e-4 | -1.42e-4 |
| 119 | 3.00e-2 | 1 | 3.70e-2 | 3.70e-2 | 3.70e-2 | 3.70e-2 | 244 | +8.74e-5 | +8.74e-5 | +8.74e-5 | -1.19e-4 |
| 120 | 3.00e-2 | 1 | 4.17e-2 | 4.17e-2 | 4.17e-2 | 4.17e-2 | 302 | +4.00e-4 | +4.00e-4 | +4.00e-4 | -6.67e-5 |
| 121 | 3.00e-2 | 1 | 4.03e-2 | 4.03e-2 | 4.03e-2 | 4.03e-2 | 268 | -1.33e-4 | -1.33e-4 | -1.33e-4 | -7.34e-5 |
| 122 | 3.00e-2 | 1 | 4.10e-2 | 4.10e-2 | 4.10e-2 | 4.10e-2 | 250 | +7.69e-5 | +7.69e-5 | +7.69e-5 | -5.83e-5 |
| 123 | 3.00e-2 | 1 | 4.10e-2 | 4.10e-2 | 4.10e-2 | 4.10e-2 | 251 | -2.12e-6 | -2.12e-6 | -2.12e-6 | -5.27e-5 |
| 124 | 3.00e-2 | 1 | 4.28e-2 | 4.28e-2 | 4.28e-2 | 4.28e-2 | 255 | +1.64e-4 | +1.64e-4 | +1.64e-4 | -3.10e-5 |
| 125 | 3.00e-2 | 1 | 4.34e-2 | 4.34e-2 | 4.34e-2 | 4.34e-2 | 271 | +5.32e-5 | +5.32e-5 | +5.32e-5 | -2.26e-5 |
| 126 | 3.00e-2 | 1 | 4.62e-2 | 4.62e-2 | 4.62e-2 | 4.62e-2 | 282 | +2.24e-4 | +2.24e-4 | +2.24e-4 | +2.09e-6 |
| 127 | 3.00e-2 | 1 | 4.58e-2 | 4.58e-2 | 4.58e-2 | 4.58e-2 | 277 | -2.98e-5 | -2.98e-5 | -2.98e-5 | -1.10e-6 |
| 128 | 3.00e-2 | 1 | 4.54e-2 | 4.54e-2 | 4.54e-2 | 4.54e-2 | 258 | -3.71e-5 | -3.71e-5 | -3.71e-5 | -4.71e-6 |
| 129 | 3.00e-2 | 1 | 4.54e-2 | 4.54e-2 | 4.54e-2 | 4.54e-2 | 258 | +1.80e-6 | +1.80e-6 | +1.80e-6 | -4.06e-6 |
| 130 | 3.00e-2 | 1 | 4.53e-2 | 4.53e-2 | 4.53e-2 | 4.53e-2 | 244 | -8.67e-6 | -8.67e-6 | -8.67e-6 | -4.52e-6 |
| 131 | 3.00e-2 | 2 | 4.45e-2 | 4.51e-2 | 4.48e-2 | 4.45e-2 | 212 | -6.81e-5 | -2.04e-5 | -4.43e-5 | -1.23e-5 |
| 132 | 3.00e-2 | 1 | 4.72e-2 | 4.72e-2 | 4.72e-2 | 4.72e-2 | 237 | +2.56e-4 | +2.56e-4 | +2.56e-4 | +1.45e-5 |
| 133 | 3.00e-2 | 1 | 5.02e-2 | 5.02e-2 | 5.02e-2 | 5.02e-2 | 257 | +2.39e-4 | +2.39e-4 | +2.39e-4 | +3.69e-5 |
| 134 | 3.00e-2 | 1 | 5.15e-2 | 5.15e-2 | 5.15e-2 | 5.15e-2 | 268 | +8.98e-5 | +8.98e-5 | +8.98e-5 | +4.22e-5 |
| 135 | 3.00e-2 | 1 | 5.28e-2 | 5.28e-2 | 5.28e-2 | 5.28e-2 | 284 | +8.82e-5 | +8.82e-5 | +8.82e-5 | +4.68e-5 |
| 136 | 3.00e-2 | 1 | 4.96e-2 | 4.96e-2 | 4.96e-2 | 4.96e-2 | 232 | -2.64e-4 | -2.64e-4 | -2.64e-4 | +1.57e-5 |
| 137 | 3.00e-2 | 1 | 4.89e-2 | 4.89e-2 | 4.89e-2 | 4.89e-2 | 218 | -6.43e-5 | -6.43e-5 | -6.43e-5 | +7.73e-6 |
| 138 | 3.00e-2 | 1 | 4.83e-2 | 4.83e-2 | 4.83e-2 | 4.83e-2 | 195 | -6.44e-5 | -6.44e-5 | -6.44e-5 | +5.17e-7 |
| 139 | 3.00e-2 | 2 | 5.44e-2 | 5.47e-2 | 5.45e-2 | 5.47e-2 | 260 | +1.50e-5 | +3.65e-4 | +1.90e-4 | +3.48e-5 |
| 141 | 3.00e-2 | 1 | 5.94e-2 | 5.94e-2 | 5.94e-2 | 5.94e-2 | 332 | +2.53e-4 | +2.53e-4 | +2.53e-4 | +5.66e-5 |
| 142 | 3.00e-2 | 1 | 5.90e-2 | 5.90e-2 | 5.90e-2 | 5.90e-2 | 285 | -2.67e-5 | -2.67e-5 | -2.67e-5 | +4.83e-5 |
| 143 | 3.00e-2 | 1 | 5.99e-2 | 5.99e-2 | 5.99e-2 | 5.99e-2 | 324 | +4.81e-5 | +4.81e-5 | +4.81e-5 | +4.83e-5 |
| 144 | 3.00e-2 | 1 | 5.95e-2 | 5.95e-2 | 5.95e-2 | 5.95e-2 | 288 | -2.22e-5 | -2.22e-5 | -2.22e-5 | +4.12e-5 |
| 145 | 3.00e-2 | 1 | 6.29e-2 | 6.29e-2 | 6.29e-2 | 6.29e-2 | 323 | +1.68e-4 | +1.68e-4 | +1.68e-4 | +5.40e-5 |
| 146 | 3.00e-2 | 1 | 6.25e-2 | 6.25e-2 | 6.25e-2 | 6.25e-2 | 309 | -1.71e-5 | -1.71e-5 | -1.71e-5 | +4.69e-5 |
| 147 | 3.00e-2 | 1 | 6.07e-2 | 6.07e-2 | 6.07e-2 | 6.07e-2 | 277 | -1.05e-4 | -1.05e-4 | -1.05e-4 | +3.16e-5 |
| 148 | 3.00e-2 | 1 | 6.20e-2 | 6.20e-2 | 6.20e-2 | 6.20e-2 | 286 | +7.35e-5 | +7.35e-5 | +7.35e-5 | +3.58e-5 |
| 149 | 3.00e-2 | 1 | 6.27e-2 | 6.27e-2 | 6.27e-2 | 6.27e-2 | 287 | +3.98e-5 | +3.98e-5 | +3.98e-5 | +3.62e-5 |
| 150 | 3.00e-3 | 1 | 5.89e-2 | 5.89e-2 | 5.89e-2 | 5.89e-2 | 288 | -2.20e-4 | -2.20e-4 | -2.20e-4 | +1.06e-5 |
| 151 | 3.00e-3 | 1 | 2.86e-2 | 2.86e-2 | 2.86e-2 | 2.86e-2 | 241 | -3.00e-3 | -3.00e-3 | -3.00e-3 | -2.90e-4 |
| 152 | 3.00e-3 | 1 | 1.50e-2 | 1.50e-2 | 1.50e-2 | 1.50e-2 | 295 | -2.18e-3 | -2.18e-3 | -2.18e-3 | -4.79e-4 |
| 153 | 3.00e-3 | 1 | 9.17e-3 | 9.17e-3 | 9.17e-3 | 9.17e-3 | 284 | -1.74e-3 | -1.74e-3 | -1.74e-3 | -6.05e-4 |
| 154 | 3.00e-3 | 1 | 7.08e-3 | 7.08e-3 | 7.08e-3 | 7.08e-3 | 262 | -9.90e-4 | -9.90e-4 | -9.90e-4 | -6.44e-4 |
| 155 | 3.00e-3 | 1 | 6.44e-3 | 6.44e-3 | 6.44e-3 | 6.44e-3 | 282 | -3.34e-4 | -3.34e-4 | -3.34e-4 | -6.13e-4 |
| 156 | 3.00e-3 | 1 | 6.58e-3 | 6.58e-3 | 6.58e-3 | 6.58e-3 | 307 | +7.02e-5 | +7.02e-5 | +7.02e-5 | -5.44e-4 |
| 157 | 3.00e-3 | 1 | 6.46e-3 | 6.46e-3 | 6.46e-3 | 6.46e-3 | 276 | -6.88e-5 | -6.88e-5 | -6.88e-5 | -4.97e-4 |
| 158 | 3.00e-3 | 1 | 6.60e-3 | 6.60e-3 | 6.60e-3 | 6.60e-3 | 308 | +7.43e-5 | +7.43e-5 | +7.43e-5 | -4.40e-4 |
| 159 | 3.00e-3 | 1 | 6.43e-3 | 6.43e-3 | 6.43e-3 | 6.43e-3 | 259 | -1.03e-4 | -1.03e-4 | -1.03e-4 | -4.06e-4 |
| 160 | 3.00e-3 | 1 | 6.39e-3 | 6.39e-3 | 6.39e-3 | 6.39e-3 | 269 | -2.37e-5 | -2.37e-5 | -2.37e-5 | -3.68e-4 |
| 161 | 3.00e-3 | 1 | 6.49e-3 | 6.49e-3 | 6.49e-3 | 6.49e-3 | 276 | +5.86e-5 | +5.86e-5 | +5.86e-5 | -3.25e-4 |
| 163 | 3.00e-3 | 2 | 6.33e-3 | 7.27e-3 | 6.80e-3 | 6.33e-3 | 235 | -5.88e-4 | +3.28e-4 | -1.30e-4 | -2.93e-4 |
| 164 | 3.00e-3 | 1 | 6.47e-3 | 6.47e-3 | 6.47e-3 | 6.47e-3 | 257 | +8.34e-5 | +8.34e-5 | +8.34e-5 | -2.55e-4 |
| 165 | 3.00e-3 | 1 | 6.86e-3 | 6.86e-3 | 6.86e-3 | 6.86e-3 | 284 | +2.07e-4 | +2.07e-4 | +2.07e-4 | -2.09e-4 |
| 166 | 3.00e-3 | 1 | 6.65e-3 | 6.65e-3 | 6.65e-3 | 6.65e-3 | 253 | -1.19e-4 | -1.19e-4 | -1.19e-4 | -2.00e-4 |
| 167 | 3.00e-3 | 1 | 7.13e-3 | 7.13e-3 | 7.13e-3 | 7.13e-3 | 323 | +2.14e-4 | +2.14e-4 | +2.14e-4 | -1.58e-4 |
| 168 | 3.00e-3 | 1 | 6.51e-3 | 6.51e-3 | 6.51e-3 | 6.51e-3 | 252 | -3.60e-4 | -3.60e-4 | -3.60e-4 | -1.79e-4 |
| 169 | 3.00e-3 | 1 | 6.45e-3 | 6.45e-3 | 6.45e-3 | 6.45e-3 | 255 | -3.62e-5 | -3.62e-5 | -3.62e-5 | -1.64e-4 |
| 170 | 3.00e-3 | 1 | 6.74e-3 | 6.74e-3 | 6.74e-3 | 6.74e-3 | 271 | +1.59e-4 | +1.59e-4 | +1.59e-4 | -1.32e-4 |
| 171 | 3.00e-3 | 1 | 6.57e-3 | 6.57e-3 | 6.57e-3 | 6.57e-3 | 262 | -9.33e-5 | -9.33e-5 | -9.33e-5 | -1.28e-4 |
| 172 | 3.00e-3 | 1 | 6.73e-3 | 6.73e-3 | 6.73e-3 | 6.73e-3 | 281 | +8.54e-5 | +8.54e-5 | +8.54e-5 | -1.07e-4 |
| 173 | 3.00e-3 | 1 | 7.22e-3 | 7.22e-3 | 7.22e-3 | 7.22e-3 | 307 | +2.28e-4 | +2.28e-4 | +2.28e-4 | -7.34e-5 |
| 174 | 3.00e-3 | 1 | 7.10e-3 | 7.10e-3 | 7.10e-3 | 7.10e-3 | 292 | -5.68e-5 | -5.68e-5 | -5.68e-5 | -7.18e-5 |
| 175 | 3.00e-3 | 1 | 6.75e-3 | 6.75e-3 | 6.75e-3 | 6.75e-3 | 259 | -1.98e-4 | -1.98e-4 | -1.98e-4 | -8.44e-5 |
| 176 | 3.00e-3 | 1 | 6.69e-3 | 6.69e-3 | 6.69e-3 | 6.69e-3 | 249 | -3.14e-5 | -3.14e-5 | -3.14e-5 | -7.91e-5 |
| 177 | 3.00e-3 | 1 | 6.75e-3 | 6.75e-3 | 6.75e-3 | 6.75e-3 | 252 | +3.23e-5 | +3.23e-5 | +3.23e-5 | -6.80e-5 |
| 178 | 3.00e-3 | 1 | 6.60e-3 | 6.60e-3 | 6.60e-3 | 6.60e-3 | 251 | -8.86e-5 | -8.86e-5 | -8.86e-5 | -7.00e-5 |
| 179 | 3.00e-3 | 1 | 6.74e-3 | 6.74e-3 | 6.74e-3 | 6.74e-3 | 248 | +8.60e-5 | +8.60e-5 | +8.60e-5 | -5.44e-5 |
| 180 | 3.00e-3 | 2 | 6.50e-3 | 6.80e-3 | 6.65e-3 | 6.50e-3 | 223 | -2.00e-4 | +3.49e-5 | -8.28e-5 | -6.10e-5 |
| 181 | 3.00e-3 | 1 | 6.85e-3 | 6.85e-3 | 6.85e-3 | 6.85e-3 | 249 | +2.05e-4 | +2.05e-4 | +2.05e-4 | -3.44e-5 |
| 182 | 3.00e-3 | 1 | 6.73e-3 | 6.73e-3 | 6.73e-3 | 6.73e-3 | 234 | -6.97e-5 | -6.97e-5 | -6.97e-5 | -3.79e-5 |
| 183 | 3.00e-3 | 1 | 6.99e-3 | 6.99e-3 | 6.99e-3 | 6.99e-3 | 246 | +1.50e-4 | +1.50e-4 | +1.50e-4 | -1.91e-5 |
| 184 | 3.00e-3 | 1 | 6.92e-3 | 6.92e-3 | 6.92e-3 | 6.92e-3 | 239 | -4.27e-5 | -4.27e-5 | -4.27e-5 | -2.15e-5 |
| 185 | 3.00e-3 | 1 | 6.77e-3 | 6.77e-3 | 6.77e-3 | 6.77e-3 | 234 | -9.20e-5 | -9.20e-5 | -9.20e-5 | -2.85e-5 |
| 186 | 3.00e-3 | 2 | 6.61e-3 | 6.91e-3 | 6.76e-3 | 6.61e-3 | 211 | -2.10e-4 | +7.87e-5 | -6.56e-5 | -3.70e-5 |
| 187 | 3.00e-3 | 1 | 6.83e-3 | 6.83e-3 | 6.83e-3 | 6.83e-3 | 232 | +1.41e-4 | +1.41e-4 | +1.41e-4 | -1.92e-5 |
| 188 | 3.00e-3 | 1 | 6.93e-3 | 6.93e-3 | 6.93e-3 | 6.93e-3 | 246 | +5.78e-5 | +5.78e-5 | +5.78e-5 | -1.15e-5 |
| 189 | 3.00e-3 | 1 | 7.06e-3 | 7.06e-3 | 7.06e-3 | 7.06e-3 | 254 | +7.38e-5 | +7.38e-5 | +7.38e-5 | -2.97e-6 |
| 190 | 3.00e-3 | 1 | 7.22e-3 | 7.22e-3 | 7.22e-3 | 7.22e-3 | 268 | +8.42e-5 | +8.42e-5 | +8.42e-5 | +5.75e-6 |
| 191 | 3.00e-3 | 1 | 7.11e-3 | 7.11e-3 | 7.11e-3 | 7.11e-3 | 245 | -5.98e-5 | -5.98e-5 | -5.98e-5 | -8.10e-7 |
| 192 | 3.00e-3 | 2 | 6.20e-3 | 6.95e-3 | 6.58e-3 | 6.20e-3 | 184 | -6.21e-4 | -1.05e-4 | -3.63e-4 | -7.22e-5 |
| 193 | 3.00e-3 | 1 | 6.51e-3 | 6.51e-3 | 6.51e-3 | 6.51e-3 | 209 | +2.35e-4 | +2.35e-4 | +2.35e-4 | -4.15e-5 |
| 194 | 3.00e-3 | 1 | 6.58e-3 | 6.58e-3 | 6.58e-3 | 6.58e-3 | 218 | +4.44e-5 | +4.44e-5 | +4.44e-5 | -3.29e-5 |
| 195 | 3.00e-3 | 2 | 6.48e-3 | 6.88e-3 | 6.68e-3 | 6.48e-3 | 187 | -3.15e-4 | +1.89e-4 | -6.32e-5 | -4.12e-5 |
| 196 | 3.00e-3 | 1 | 6.83e-3 | 6.83e-3 | 6.83e-3 | 6.83e-3 | 231 | +2.24e-4 | +2.24e-4 | +2.24e-4 | -1.47e-5 |
| 197 | 3.00e-3 | 1 | 6.91e-3 | 6.91e-3 | 6.91e-3 | 6.91e-3 | 234 | +4.91e-5 | +4.91e-5 | +4.91e-5 | -8.28e-6 |
| 198 | 3.00e-3 | 2 | 6.19e-3 | 6.95e-3 | 6.57e-3 | 6.19e-3 | 171 | -6.78e-4 | +2.88e-5 | -3.25e-4 | -7.19e-5 |
| 199 | 3.00e-3 | 1 | 6.24e-3 | 6.24e-3 | 6.24e-3 | 6.24e-3 | 192 | +3.91e-5 | +3.91e-5 | +3.91e-5 | -6.08e-5 |

