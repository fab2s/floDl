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
| cpu-async | 0.063331 | 0.9165 | +0.0040 | 1738.0 | 681 | 76.3 | 100% | 100% | 7.6 |

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
| cpu-async | 2.0402 | 0.7878 | 0.5942 | 0.5227 | 0.4928 | 0.4735 | 0.4776 | 0.4515 | 0.4886 | 0.4709 | 0.2145 | 0.1761 | 0.1604 | 0.1509 | 0.1377 | 0.0814 | 0.0747 | 0.0710 | 0.0631 | 0.0633 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | cpu-async | 0 | cuda:0 | 0.3997 | 2.8 |
| resnet-graph | cpu-async | 1 | cuda:1 | 0.3045 | 3.8 |
| resnet-graph | cpu-async | 2 | cuda:2 | 0.2958 | 3.7 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|
| resnet-graph | cpu-async | 358 | 357 | 396 | 391 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| cpu-async | gpu1 | 862.1 | 1.2 | epoch-boundary(98) |
| cpu-async | gpu2 | 862.1 | 1.2 | epoch-boundary(98) |
| cpu-async | gpu1 | 888.9 | 0.8 | epoch-boundary(101) |
| cpu-async | gpu2 | 888.7 | 0.7 | epoch-boundary(101) |
| cpu-async | gpu2 | 1079.1 | 0.6 | epoch-boundary(123) |
| cpu-async | gpu0 | 0.5 | 0.6 | cpu-avg |
| cpu-async | gpu2 | 906.2 | 0.5 | epoch-boundary(103) |
| cpu-async | gpu1 | 1079.2 | 0.5 | epoch-boundary(123) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | cpu-async | gpu0 | 0.0s | 0.0s | 0.6s | 0.0s | 0.6s |
| resnet-graph | cpu-async | gpu1 | 2.5s | 0.0s | 0.0s | 0.0s | 3.3s |
| resnet-graph | cpu-async | gpu2 | 3.1s | 0.0s | 0.0s | 0.0s | 3.7s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | cpu-async | 346 | 0 | 681 | 76.3 | 1327/8394 | 681 | 76.3 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | cpu-async | 172.4 | 9.9% |

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
| resnet-graph | cpu-async | 198 | 681 | 0 | 7.94e-3 | -5.90e-5 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | cpu-async | 0 | 681 | 9.35e-2 | 5.47e-2 | 2.46e-3 | 3.81e-1 | 26.9 | -1.83e-4 | 1.19e-3 |
| resnet-graph | cpu-async | 1 | 681 | 9.41e-2 | 5.56e-2 | 2.56e-3 | 4.05e-1 | 35.2 | -2.07e-4 | 1.29e-3 |
| resnet-graph | cpu-async | 2 | 681 | 9.45e-2 | 5.60e-2 | 2.38e-3 | 3.89e-1 | 37.9 | -2.11e-4 | 1.32e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | cpu-async | rank0 ↔ rank1 | +0.9926 |
| resnet-graph | cpu-async | rank0 ↔ rank2 | +0.9915 |
| resnet-graph | cpu-async | rank1 ↔ rank2 | +0.9911 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | cpu-async | 77 (4,5,6,7,9,10,11,12…148,150) | 0 (—) | — | 4,5,6,7,9,10,11,12…148,150 | — |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | cpu-async | 0e0 | 3 | 5 | 5 |
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
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 490 | +0.111 |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | 59 | +0.055 |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | 128 | +0.027 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | cpu-async | 679 | -0.040 | 197 | +0.478 | +0.649 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | cpu-async | 680 | 3.36e1–8.10e1 | 6.83e1 | 2.03e-3 | 3.43e-3 | 3.38e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 492 | 31–78019 | +2.902e-6 | 0.071 | +3.031e-6 | 0.084 | 98 | +8.877e-6 | 0.519 | 27–991 | +9.524e-4 | 0.627 |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 481 | 917–78019 | +3.826e-6 | 0.137 | +3.945e-6 | 0.161 | 97 | +9.317e-6 | 0.557 | 64–991 | +9.955e-4 | 0.813 |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | 60 | 78845–116867 | +1.034e-5 | 0.186 | +9.892e-6 | 0.177 | 50 | +9.479e-6 | 0.143 | 421–839 | -8.119e-4 | 0.087 |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | 129 | 117433–156144 | -2.040e-5 | 0.246 | -2.040e-5 | 0.245 | 50 | -7.193e-6 | 0.040 | 92–873 | +1.643e-3 | 0.549 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | +9.524e-4 | r0: +9.186e-4, r1: +9.624e-4, r2: +9.755e-4 | r0: 0.613, r1: 0.606, r2: 0.600 | 1.06× | ✓ |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +9.955e-4 | r0: +9.602e-4, r1: +1.005e-3, r2: +1.020e-3 | r0: 0.805, r1: 0.775, r2: 0.771 | 1.06× | ✓ |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | -8.119e-4 | r0: -8.020e-4, r1: -8.163e-4, r2: -8.152e-4 | r0: 0.084, r1: 0.087, r2: 0.087 | 1.02× | ✓ |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | +1.643e-3 | r0: +1.626e-3, r1: +1.648e-3, r2: +1.656e-3 | r0: 0.543, r1: 0.546, r2: 0.547 | 1.02× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | cpu-async | `█▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇█████▇▅▅▅▅▅▅▅▅▅▅▅▆▃▂▂▂▂▁▁▁▁▁▂▂▂` | `▁▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇██████▇▇▇█████████▆▇▇████▇▇▇▇██` |

### Per-Epoch Detail: resnet-graph / cpu-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 11 | 1.16e-1 | 4.05e-1 | 1.92e-1 | 1.38e-1 | 46 | -4.02e-2 | +1.01e-2 | -1.16e-2 | -8.31e-3 |
| 1 | 3.00e-1 | 6 | 9.85e-2 | 1.71e-1 | 1.21e-1 | 9.85e-2 | 41 | -6.01e-3 | +2.46e-3 | -1.68e-3 | -4.49e-3 |
| 2 | 3.00e-1 | 8 | 9.81e-2 | 1.34e-1 | 1.08e-1 | 1.08e-1 | 41 | -6.14e-3 | +3.83e-3 | -2.49e-4 | -1.80e-3 |
| 3 | 3.00e-1 | 5 | 1.04e-1 | 1.43e-1 | 1.17e-1 | 1.07e-1 | 33 | -7.11e-3 | +4.24e-3 | -1.13e-3 | -1.57e-3 |
| 4 | 3.00e-1 | 8 | 1.07e-1 | 1.49e-1 | 1.18e-1 | 1.13e-1 | 34 | -8.03e-3 | +4.61e-3 | -6.33e-4 | -1.00e-3 |
| 5 | 3.00e-1 | 8 | 9.88e-2 | 1.55e-1 | 1.13e-1 | 1.11e-1 | 34 | -9.36e-3 | +4.28e-3 | -9.08e-4 | -7.88e-4 |
| 6 | 3.00e-1 | 7 | 9.06e-2 | 1.58e-1 | 1.15e-1 | 1.27e-1 | 48 | -8.54e-3 | +6.87e-3 | -1.01e-3 | -6.85e-4 |
| 7 | 3.00e-1 | 6 | 1.26e-1 | 1.78e-1 | 1.38e-1 | 1.26e-1 | 46 | -3.81e-3 | +3.22e-3 | -4.84e-4 | -5.99e-4 |
| 8 | 3.00e-1 | 4 | 1.17e-1 | 1.63e-1 | 1.33e-1 | 1.17e-1 | 39 | -5.59e-3 | +2.65e-3 | -1.27e-3 | -8.57e-4 |
| 9 | 3.00e-1 | 5 | 1.18e-1 | 1.48e-1 | 1.29e-1 | 1.29e-1 | 53 | -4.33e-3 | +3.10e-3 | -6.45e-5 | -5.22e-4 |
| 10 | 3.00e-1 | 5 | 1.17e-1 | 1.72e-1 | 1.33e-1 | 1.17e-1 | 43 | -6.01e-3 | +2.62e-3 | -1.12e-3 | -7.71e-4 |
| 11 | 3.00e-1 | 6 | 1.13e-1 | 1.50e-1 | 1.22e-1 | 1.22e-1 | 50 | -6.76e-3 | +2.85e-3 | -4.66e-4 | -5.76e-4 |
| 12 | 3.00e-1 | 5 | 1.18e-1 | 1.55e-1 | 1.29e-1 | 1.25e-1 | 54 | -4.68e-3 | +2.81e-3 | -4.21e-4 | -4.98e-4 |
| 13 | 3.00e-1 | 5 | 1.31e-1 | 1.56e-1 | 1.36e-1 | 1.33e-1 | 58 | -2.99e-3 | +2.42e-3 | -6.40e-5 | -3.29e-4 |
| 14 | 3.00e-1 | 5 | 1.16e-1 | 1.68e-1 | 1.29e-1 | 1.17e-1 | 42 | -6.60e-3 | +2.08e-3 | -1.23e-3 | -6.80e-4 |
| 15 | 3.00e-1 | 7 | 1.13e-1 | 1.55e-1 | 1.21e-1 | 1.13e-1 | 42 | -6.25e-3 | +3.11e-3 | -6.08e-4 | -6.08e-4 |
| 16 | 3.00e-1 | 5 | 1.10e-1 | 1.47e-1 | 1.21e-1 | 1.17e-1 | 42 | -4.34e-3 | +3.31e-3 | -4.49e-4 | -5.33e-4 |
| 17 | 3.00e-1 | 6 | 1.07e-1 | 1.51e-1 | 1.18e-1 | 1.09e-1 | 35 | -7.73e-3 | +2.88e-3 | -9.62e-4 | -7.01e-4 |
| 18 | 3.00e-1 | 8 | 1.06e-1 | 1.41e-1 | 1.14e-1 | 1.16e-1 | 42 | -6.45e-3 | +3.60e-3 | -2.84e-4 | -3.76e-4 |
| 19 | 3.00e-1 | 5 | 1.15e-1 | 1.50e-1 | 1.23e-1 | 1.15e-1 | 40 | -5.64e-3 | +3.24e-3 | -6.64e-4 | -4.94e-4 |
| 20 | 3.00e-1 | 6 | 1.15e-1 | 1.51e-1 | 1.23e-1 | 1.16e-1 | 42 | -5.61e-3 | +3.37e-3 | -5.52e-4 | -5.13e-4 |
| 21 | 3.00e-1 | 6 | 1.04e-1 | 1.54e-1 | 1.21e-1 | 1.06e-1 | 36 | -4.84e-3 | +3.37e-3 | -1.02e-3 | -7.81e-4 |
| 22 | 3.00e-1 | 6 | 1.15e-1 | 1.42e-1 | 1.22e-1 | 1.18e-1 | 45 | -3.95e-3 | +4.06e-3 | -8.67e-5 | -4.77e-4 |
| 23 | 3.00e-1 | 8 | 1.12e-1 | 1.49e-1 | 1.23e-1 | 1.13e-1 | 40 | -3.35e-3 | +2.94e-3 | -3.82e-4 | -4.39e-4 |
| 24 | 3.00e-1 | 4 | 1.12e-1 | 1.48e-1 | 1.24e-1 | 1.12e-1 | 43 | -4.61e-3 | +3.38e-3 | -8.68e-4 | -6.19e-4 |
| 25 | 3.00e-1 | 7 | 1.03e-1 | 1.50e-1 | 1.17e-1 | 1.06e-1 | 33 | -3.46e-3 | +3.59e-3 | -7.02e-4 | -6.54e-4 |
| 26 | 3.00e-1 | 7 | 9.85e-2 | 1.56e-1 | 1.11e-1 | 1.01e-1 | 32 | -1.04e-2 | +4.54e-3 | -1.42e-3 | -9.75e-4 |
| 27 | 3.00e-1 | 7 | 1.02e-1 | 1.44e-1 | 1.16e-1 | 1.16e-1 | 44 | -1.33e-2 | +4.94e-3 | -7.26e-4 | -7.54e-4 |
| 28 | 3.00e-1 | 8 | 1.14e-1 | 1.61e-1 | 1.24e-1 | 1.14e-1 | 42 | -5.12e-3 | +3.40e-3 | -5.62e-4 | -6.24e-4 |
| 29 | 3.00e-1 | 4 | 1.08e-1 | 1.53e-1 | 1.23e-1 | 1.08e-1 | 40 | -6.64e-3 | +3.50e-3 | -1.40e-3 | -9.19e-4 |
| 30 | 3.00e-1 | 7 | 1.09e-1 | 1.49e-1 | 1.17e-1 | 1.09e-1 | 36 | -5.87e-3 | +3.94e-3 | -6.01e-4 | -7.39e-4 |
| 31 | 3.00e-1 | 7 | 1.06e-1 | 1.48e-1 | 1.17e-1 | 1.13e-1 | 41 | -6.45e-3 | +3.85e-3 | -5.43e-4 | -5.87e-4 |
| 32 | 3.00e-1 | 6 | 9.99e-2 | 1.63e-1 | 1.19e-1 | 9.99e-2 | 31 | -8.17e-3 | +3.93e-3 | -1.58e-3 | -1.08e-3 |
| 33 | 3.00e-1 | 9 | 9.95e-2 | 1.45e-1 | 1.16e-1 | 1.08e-1 | 35 | -4.65e-3 | +5.05e-3 | -3.35e-4 | -6.51e-4 |
| 34 | 3.00e-1 | 4 | 1.12e-1 | 1.49e-1 | 1.23e-1 | 1.12e-1 | 37 | -6.89e-3 | +3.99e-3 | -1.07e-3 | -8.24e-4 |
| 35 | 3.00e-1 | 7 | 1.07e-1 | 1.50e-1 | 1.18e-1 | 1.20e-1 | 41 | -5.97e-3 | +3.78e-3 | -3.32e-4 | -4.88e-4 |
| 36 | 3.00e-1 | 6 | 1.09e-1 | 1.62e-1 | 1.22e-1 | 1.09e-1 | 38 | -6.64e-3 | +3.31e-3 | -1.15e-3 | -7.78e-4 |
| 37 | 3.00e-1 | 7 | 1.14e-1 | 1.49e-1 | 1.21e-1 | 1.15e-1 | 40 | -6.02e-3 | +4.37e-3 | -3.67e-4 | -5.64e-4 |
| 38 | 3.00e-1 | 8 | 1.04e-1 | 1.59e-1 | 1.18e-1 | 1.10e-1 | 34 | -6.19e-3 | +3.92e-3 | -6.80e-4 | -5.68e-4 |
| 39 | 3.00e-1 | 5 | 1.07e-1 | 1.50e-1 | 1.19e-1 | 1.07e-1 | 34 | -8.01e-3 | +4.16e-3 | -1.17e-3 | -8.26e-4 |
| 40 | 3.00e-1 | 8 | 1.03e-1 | 1.45e-1 | 1.17e-1 | 1.21e-1 | 45 | -8.16e-3 | +4.18e-3 | -3.49e-4 | -4.54e-4 |
| 41 | 3.00e-1 | 5 | 1.19e-1 | 1.61e-1 | 1.30e-1 | 1.23e-1 | 46 | -4.71e-3 | +3.42e-3 | -4.55e-4 | -4.52e-4 |
| 42 | 3.00e-1 | 6 | 1.05e-1 | 1.61e-1 | 1.20e-1 | 1.10e-1 | 37 | -9.80e-3 | +3.05e-3 | -1.42e-3 | -8.43e-4 |
| 43 | 3.00e-1 | 6 | 1.13e-1 | 1.55e-1 | 1.24e-1 | 1.17e-1 | 46 | -5.15e-3 | +4.36e-3 | -4.53e-4 | -6.53e-4 |
| 44 | 3.00e-1 | 5 | 1.29e-1 | 1.58e-1 | 1.37e-1 | 1.29e-1 | 52 | -2.83e-3 | +3.15e-3 | -1.34e-4 | -4.75e-4 |
| 45 | 3.00e-1 | 8 | 1.08e-1 | 1.60e-1 | 1.21e-1 | 1.08e-1 | 35 | -5.55e-3 | +2.36e-3 | -8.41e-4 | -6.56e-4 |
| 46 | 3.00e-1 | 5 | 1.10e-1 | 1.45e-1 | 1.20e-1 | 1.13e-1 | 42 | -5.43e-3 | +4.11e-3 | -6.91e-4 | -6.79e-4 |
| 47 | 3.00e-1 | 6 | 1.12e-1 | 1.51e-1 | 1.23e-1 | 1.16e-1 | 39 | -4.51e-3 | +3.61e-3 | -4.28e-4 | -5.59e-4 |
| 48 | 3.00e-1 | 6 | 1.04e-1 | 1.53e-1 | 1.19e-1 | 1.07e-1 | 34 | -5.50e-3 | +3.35e-3 | -9.48e-4 | -7.42e-4 |
| 49 | 3.00e-1 | 8 | 1.11e-1 | 1.44e-1 | 1.18e-1 | 1.11e-1 | 37 | -5.55e-3 | +4.28e-3 | -3.79e-4 | -5.38e-4 |
| 50 | 3.00e-1 | 6 | 1.13e-1 | 1.49e-1 | 1.21e-1 | 1.18e-1 | 41 | -5.97e-3 | +3.87e-3 | -3.61e-4 | -4.33e-4 |
| 51 | 3.00e-1 | 6 | 1.05e-1 | 1.51e-1 | 1.19e-1 | 1.05e-1 | 34 | -5.25e-3 | +3.00e-3 | -1.07e-3 | -7.54e-4 |
| 52 | 3.00e-1 | 9 | 1.06e-1 | 1.51e-1 | 1.17e-1 | 1.30e-1 | 49 | -7.77e-3 | +4.82e-3 | -1.49e-4 | -2.16e-4 |
| 53 | 3.00e-1 | 4 | 1.14e-1 | 1.56e-1 | 1.27e-1 | 1.14e-1 | 40 | -5.83e-3 | +2.40e-3 | -1.25e-3 | -5.86e-4 |
| 54 | 3.00e-1 | 7 | 1.06e-1 | 1.52e-1 | 1.16e-1 | 1.12e-1 | 37 | -7.78e-3 | +3.71e-3 | -7.36e-4 | -5.86e-4 |
| 55 | 3.00e-1 | 7 | 1.09e-1 | 1.59e-1 | 1.20e-1 | 1.17e-1 | 40 | -8.17e-3 | +3.86e-3 | -6.91e-4 | -5.39e-4 |
| 56 | 3.00e-1 | 8 | 9.61e-2 | 1.55e-1 | 1.10e-1 | 1.09e-1 | 36 | -9.37e-3 | +3.49e-3 | -9.14e-4 | -5.58e-4 |
| 57 | 3.00e-1 | 6 | 1.10e-1 | 1.61e-1 | 1.22e-1 | 1.11e-1 | 37 | -1.03e-2 | +4.39e-3 | -1.26e-3 | -8.61e-4 |
| 58 | 3.00e-1 | 7 | 1.07e-1 | 1.58e-1 | 1.21e-1 | 1.15e-1 | 39 | -8.84e-3 | +4.57e-3 | -6.78e-4 | -7.08e-4 |
| 59 | 3.00e-1 | 9 | 1.03e-1 | 1.54e-1 | 1.15e-1 | 1.15e-1 | 39 | -6.12e-3 | +3.53e-3 | -4.83e-4 | -4.29e-4 |
| 60 | 3.00e-1 | 5 | 1.09e-1 | 1.61e-1 | 1.25e-1 | 1.09e-1 | 33 | -5.69e-3 | +3.94e-3 | -1.10e-3 | -7.37e-4 |
| 61 | 3.00e-1 | 9 | 1.02e-1 | 1.54e-1 | 1.11e-1 | 1.03e-1 | 29 | -1.13e-2 | +4.71e-3 | -1.05e-3 | -8.34e-4 |
| 62 | 3.00e-1 | 5 | 1.08e-1 | 1.49e-1 | 1.19e-1 | 1.13e-1 | 37 | -9.97e-3 | +4.95e-3 | -9.91e-4 | -8.80e-4 |
| 63 | 3.00e-1 | 5 | 1.38e-1 | 1.69e-1 | 1.46e-1 | 1.39e-1 | 63 | -2.41e-3 | +4.03e-3 | +1.58e-4 | -4.96e-4 |
| 64 | 3.00e-1 | 4 | 1.33e-1 | 1.70e-1 | 1.46e-1 | 1.33e-1 | 58 | -2.84e-3 | +1.94e-3 | -5.06e-4 | -5.24e-4 |
| 65 | 3.00e-1 | 4 | 1.30e-1 | 1.63e-1 | 1.40e-1 | 1.30e-1 | 55 | -3.19e-3 | +2.21e-3 | -4.76e-4 | -5.25e-4 |
| 66 | 3.00e-1 | 8 | 1.23e-1 | 1.57e-1 | 1.32e-1 | 1.23e-1 | 45 | -2.94e-3 | +2.30e-3 | -3.03e-4 | -4.09e-4 |
| 67 | 3.00e-1 | 4 | 1.12e-1 | 1.55e-1 | 1.27e-1 | 1.12e-1 | 36 | -5.85e-3 | +2.86e-3 | -1.38e-3 | -7.78e-4 |
| 68 | 3.00e-1 | 6 | 1.09e-1 | 1.48e-1 | 1.19e-1 | 1.22e-1 | 43 | -7.14e-3 | +3.92e-3 | -3.30e-4 | -4.91e-4 |
| 69 | 3.00e-1 | 7 | 1.05e-1 | 1.62e-1 | 1.17e-1 | 1.12e-1 | 35 | -8.64e-3 | +3.25e-3 | -9.40e-4 | -6.09e-4 |
| 70 | 3.00e-1 | 6 | 1.07e-1 | 1.40e-1 | 1.23e-1 | 1.29e-1 | 49 | -6.95e-3 | +3.28e-3 | -1.72e-4 | -3.49e-4 |
| 71 | 3.00e-1 | 6 | 1.11e-1 | 1.63e-1 | 1.25e-1 | 1.17e-1 | 45 | -8.14e-3 | +2.50e-3 | -1.07e-3 | -6.24e-4 |
| 72 | 3.00e-1 | 9 | 1.07e-1 | 1.56e-1 | 1.18e-1 | 1.24e-1 | 48 | -6.01e-3 | +3.53e-3 | -3.44e-4 | -2.99e-4 |
| 73 | 3.00e-1 | 3 | 1.25e-1 | 1.54e-1 | 1.36e-1 | 1.25e-1 | 47 | -4.17e-3 | +2.68e-3 | -6.39e-4 | -4.19e-4 |
| 74 | 3.00e-1 | 1 | 1.24e-1 | 1.24e-1 | 1.24e-1 | 1.24e-1 | 45 | -1.59e-4 | -1.59e-4 | -1.59e-4 | -3.93e-4 |
| 75 | 3.00e-1 | 1 | 2.43e-1 | 2.43e-1 | 2.43e-1 | 2.43e-1 | 365 | +1.84e-3 | +1.84e-3 | +1.84e-3 | -1.70e-4 |
| 76 | 3.00e-1 | 1 | 2.37e-1 | 2.37e-1 | 2.37e-1 | 2.37e-1 | 314 | -7.88e-5 | -7.88e-5 | -7.88e-5 | -1.61e-4 |
| 77 | 3.00e-1 | 1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 286 | -1.02e-4 | -1.02e-4 | -1.02e-4 | -1.55e-4 |
| 78 | 3.00e-1 | 1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 261 | -1.50e-4 | -1.50e-4 | -1.50e-4 | -1.54e-4 |
| 79 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 273 | -8.74e-5 | -8.74e-5 | -8.74e-5 | -1.48e-4 |
| 80 | 3.00e-1 | 1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 292 | +1.41e-4 | +1.41e-4 | +1.41e-4 | -1.19e-4 |
| 81 | 3.00e-1 | 1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 298 | -1.15e-5 | -1.15e-5 | -1.15e-5 | -1.08e-4 |
| 82 | 3.00e-1 | 1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 332 | +5.72e-5 | +5.72e-5 | +5.72e-5 | -9.16e-5 |
| 83 | 3.00e-1 | 1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 311 | +9.56e-6 | +9.56e-6 | +9.56e-6 | -8.15e-5 |
| 84 | 3.00e-1 | 1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 325 | -1.34e-5 | -1.34e-5 | -1.34e-5 | -7.47e-5 |
| 86 | 3.00e-1 | 2 | 2.28e-1 | 2.37e-1 | 2.32e-1 | 2.28e-1 | 281 | -1.33e-4 | +9.84e-5 | -1.71e-5 | -6.49e-5 |
| 88 | 3.00e-1 | 1 | 2.40e-1 | 2.40e-1 | 2.40e-1 | 2.40e-1 | 365 | +1.37e-4 | +1.37e-4 | +1.37e-4 | -4.47e-5 |
| 89 | 3.00e-1 | 1 | 2.31e-1 | 2.31e-1 | 2.31e-1 | 2.31e-1 | 323 | -1.20e-4 | -1.20e-4 | -1.20e-4 | -5.22e-5 |
| 90 | 3.00e-1 | 1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 278 | -1.49e-4 | -1.49e-4 | -1.49e-4 | -6.19e-5 |
| 91 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 263 | -1.50e-5 | -1.50e-5 | -1.50e-5 | -5.72e-5 |
| 92 | 3.00e-1 | 1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 284 | +2.22e-5 | +2.22e-5 | +2.22e-5 | -4.93e-5 |
| 93 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 262 | -6.47e-5 | -6.47e-5 | -6.47e-5 | -5.08e-5 |
| 94 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 274 | +1.55e-5 | +1.55e-5 | +1.55e-5 | -4.42e-5 |
| 95 | 3.00e-1 | 1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 304 | +7.43e-5 | +7.43e-5 | +7.43e-5 | -3.23e-5 |
| 96 | 3.00e-1 | 1 | 2.36e-1 | 2.36e-1 | 2.36e-1 | 2.36e-1 | 332 | +1.61e-4 | +1.61e-4 | +1.61e-4 | -1.30e-5 |
| 97 | 3.00e-1 | 1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 293 | -1.35e-4 | -1.35e-4 | -1.35e-4 | -2.53e-5 |
| 98 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 258 | -2.04e-4 | -2.04e-4 | -2.04e-4 | -4.32e-5 |
| 99 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 267 | +5.23e-5 | +5.23e-5 | +5.23e-5 | -3.36e-5 |
| 100 | 3.00e-2 | 1 | 1.58e-1 | 1.58e-1 | 1.58e-1 | 1.58e-1 | 287 | -1.12e-3 | -1.12e-3 | -1.12e-3 | -1.43e-4 |
| 101 | 3.00e-2 | 1 | 7.82e-2 | 7.82e-2 | 7.82e-2 | 7.82e-2 | 257 | -2.75e-3 | -2.75e-3 | -2.75e-3 | -4.03e-4 |
| 102 | 3.00e-2 | 1 | 4.34e-2 | 4.34e-2 | 4.34e-2 | 4.34e-2 | 244 | -2.41e-3 | -2.41e-3 | -2.41e-3 | -6.04e-4 |
| 103 | 3.00e-2 | 1 | 3.26e-2 | 3.26e-2 | 3.26e-2 | 3.26e-2 | 284 | -1.01e-3 | -1.01e-3 | -1.01e-3 | -6.45e-4 |
| 104 | 3.00e-2 | 1 | 2.91e-2 | 2.91e-2 | 2.91e-2 | 2.91e-2 | 274 | -4.06e-4 | -4.06e-4 | -4.06e-4 | -6.21e-4 |
| 105 | 3.00e-2 | 1 | 2.72e-2 | 2.72e-2 | 2.72e-2 | 2.72e-2 | 233 | -3.03e-4 | -3.03e-4 | -3.03e-4 | -5.89e-4 |
| 106 | 3.00e-2 | 1 | 2.81e-2 | 2.81e-2 | 2.81e-2 | 2.81e-2 | 242 | +1.39e-4 | +1.39e-4 | +1.39e-4 | -5.16e-4 |
| 107 | 3.00e-2 | 1 | 2.96e-2 | 2.96e-2 | 2.96e-2 | 2.96e-2 | 255 | +2.01e-4 | +2.01e-4 | +2.01e-4 | -4.44e-4 |
| 108 | 3.00e-2 | 1 | 3.06e-2 | 3.06e-2 | 3.06e-2 | 3.06e-2 | 278 | +1.22e-4 | +1.22e-4 | +1.22e-4 | -3.88e-4 |
| 109 | 3.00e-2 | 1 | 3.33e-2 | 3.33e-2 | 3.33e-2 | 3.33e-2 | 309 | +2.79e-4 | +2.79e-4 | +2.79e-4 | -3.21e-4 |
| 110 | 3.00e-2 | 1 | 3.30e-2 | 3.30e-2 | 3.30e-2 | 3.30e-2 | 287 | -3.51e-5 | -3.51e-5 | -3.51e-5 | -2.93e-4 |
| 111 | 3.00e-2 | 1 | 3.27e-2 | 3.27e-2 | 3.27e-2 | 3.27e-2 | 252 | -3.90e-5 | -3.90e-5 | -3.90e-5 | -2.67e-4 |
| 112 | 3.00e-2 | 1 | 3.20e-2 | 3.20e-2 | 3.20e-2 | 3.20e-2 | 240 | -8.35e-5 | -8.35e-5 | -8.35e-5 | -2.49e-4 |
| 113 | 3.00e-2 | 1 | 3.28e-2 | 3.28e-2 | 3.28e-2 | 3.28e-2 | 250 | +1.02e-4 | +1.02e-4 | +1.02e-4 | -2.14e-4 |
| 114 | 3.00e-2 | 1 | 3.34e-2 | 3.34e-2 | 3.34e-2 | 3.34e-2 | 247 | +6.53e-5 | +6.53e-5 | +6.53e-5 | -1.86e-4 |
| 115 | 3.00e-2 | 2 | 3.51e-2 | 3.59e-2 | 3.55e-2 | 3.51e-2 | 232 | -8.97e-5 | +2.72e-4 | +9.10e-5 | -1.35e-4 |
| 116 | 3.00e-2 | 1 | 3.65e-2 | 3.65e-2 | 3.65e-2 | 3.65e-2 | 248 | +1.54e-4 | +1.54e-4 | +1.54e-4 | -1.06e-4 |
| 117 | 3.00e-2 | 1 | 3.67e-2 | 3.67e-2 | 3.67e-2 | 3.67e-2 | 249 | +2.30e-5 | +2.30e-5 | +2.30e-5 | -9.32e-5 |
| 118 | 3.00e-2 | 1 | 3.81e-2 | 3.81e-2 | 3.81e-2 | 3.81e-2 | 256 | +1.51e-4 | +1.51e-4 | +1.51e-4 | -6.88e-5 |
| 119 | 3.00e-2 | 1 | 3.75e-2 | 3.75e-2 | 3.75e-2 | 3.75e-2 | 237 | -7.06e-5 | -7.06e-5 | -7.06e-5 | -6.90e-5 |
| 120 | 3.00e-2 | 1 | 3.73e-2 | 3.73e-2 | 3.73e-2 | 3.73e-2 | 216 | -2.68e-5 | -2.68e-5 | -2.68e-5 | -6.48e-5 |
| 121 | 3.00e-2 | 1 | 3.85e-2 | 3.85e-2 | 3.85e-2 | 3.85e-2 | 231 | +1.39e-4 | +1.39e-4 | +1.39e-4 | -4.44e-5 |
| 122 | 3.00e-2 | 1 | 3.96e-2 | 3.96e-2 | 3.96e-2 | 3.96e-2 | 241 | +1.13e-4 | +1.13e-4 | +1.13e-4 | -2.87e-5 |
| 123 | 3.00e-2 | 2 | 4.03e-2 | 4.12e-2 | 4.07e-2 | 4.03e-2 | 230 | -9.62e-5 | +1.60e-4 | +3.18e-5 | -1.85e-5 |
| 124 | 3.00e-2 | 1 | 4.31e-2 | 4.31e-2 | 4.31e-2 | 4.31e-2 | 261 | +2.58e-4 | +2.58e-4 | +2.58e-4 | +9.17e-6 |
| 125 | 3.00e-2 | 1 | 4.12e-2 | 4.12e-2 | 4.12e-2 | 4.12e-2 | 219 | -2.01e-4 | -2.01e-4 | -2.01e-4 | -1.18e-5 |
| 126 | 3.00e-2 | 1 | 4.21e-2 | 4.21e-2 | 4.21e-2 | 4.21e-2 | 232 | +8.88e-5 | +8.88e-5 | +8.88e-5 | -1.78e-6 |
| 127 | 3.00e-2 | 1 | 4.51e-2 | 4.51e-2 | 4.51e-2 | 4.51e-2 | 256 | +2.67e-4 | +2.67e-4 | +2.67e-4 | +2.51e-5 |
| 128 | 3.00e-2 | 1 | 4.52e-2 | 4.52e-2 | 4.52e-2 | 4.52e-2 | 252 | +1.37e-5 | +1.37e-5 | +1.37e-5 | +2.40e-5 |
| 129 | 3.00e-2 | 2 | 4.21e-2 | 4.63e-2 | 4.42e-2 | 4.63e-2 | 257 | -3.53e-4 | +3.64e-4 | +5.36e-6 | +2.40e-5 |
| 130 | 3.00e-2 | 1 | 4.70e-2 | 4.70e-2 | 4.70e-2 | 4.70e-2 | 254 | +6.63e-5 | +6.63e-5 | +6.63e-5 | +2.83e-5 |
| 131 | 3.00e-2 | 1 | 4.64e-2 | 4.64e-2 | 4.64e-2 | 4.64e-2 | 236 | -6.22e-5 | -6.22e-5 | -6.22e-5 | +1.92e-5 |
| 132 | 3.00e-2 | 1 | 4.79e-2 | 4.79e-2 | 4.79e-2 | 4.79e-2 | 238 | +1.36e-4 | +1.36e-4 | +1.36e-4 | +3.09e-5 |
| 133 | 3.00e-2 | 2 | 4.59e-2 | 4.84e-2 | 4.71e-2 | 4.59e-2 | 188 | -2.87e-4 | +4.47e-5 | -1.21e-4 | +3.90e-7 |
| 134 | 3.00e-2 | 1 | 4.84e-2 | 4.84e-2 | 4.84e-2 | 4.84e-2 | 241 | +2.24e-4 | +2.24e-4 | +2.24e-4 | +2.27e-5 |
| 135 | 3.00e-2 | 1 | 4.78e-2 | 4.78e-2 | 4.78e-2 | 4.78e-2 | 231 | -5.28e-5 | -5.28e-5 | -5.28e-5 | +1.52e-5 |
| 136 | 3.00e-2 | 2 | 4.68e-2 | 4.73e-2 | 4.70e-2 | 4.68e-2 | 186 | -5.80e-5 | -5.73e-5 | -5.77e-5 | +1.34e-6 |
| 137 | 3.00e-2 | 1 | 5.03e-2 | 5.03e-2 | 5.03e-2 | 5.03e-2 | 229 | +3.14e-4 | +3.14e-4 | +3.14e-4 | +3.26e-5 |
| 138 | 3.00e-2 | 1 | 5.11e-2 | 5.11e-2 | 5.11e-2 | 5.11e-2 | 229 | +7.62e-5 | +7.62e-5 | +7.62e-5 | +3.70e-5 |
| 139 | 3.00e-2 | 2 | 4.98e-2 | 5.17e-2 | 5.07e-2 | 4.98e-2 | 194 | -1.85e-4 | +4.39e-5 | -7.08e-5 | +1.53e-5 |
| 140 | 3.00e-2 | 1 | 5.38e-2 | 5.38e-2 | 5.38e-2 | 5.38e-2 | 231 | +3.29e-4 | +3.29e-4 | +3.29e-4 | +4.67e-5 |
| 141 | 3.00e-2 | 1 | 5.42e-2 | 5.42e-2 | 5.42e-2 | 5.42e-2 | 230 | +3.63e-5 | +3.63e-5 | +3.63e-5 | +4.57e-5 |
| 142 | 3.00e-2 | 1 | 5.29e-2 | 5.29e-2 | 5.29e-2 | 5.29e-2 | 204 | -1.22e-4 | -1.22e-4 | -1.22e-4 | +2.89e-5 |
| 143 | 3.00e-2 | 2 | 4.75e-2 | 5.07e-2 | 4.91e-2 | 5.07e-2 | 193 | -7.13e-4 | +3.42e-4 | -1.86e-4 | -6.59e-6 |
| 144 | 3.00e-2 | 2 | 4.98e-2 | 5.04e-2 | 5.01e-2 | 4.98e-2 | 175 | -6.40e-5 | -3.96e-5 | -5.18e-5 | -1.53e-5 |
| 145 | 3.00e-2 | 1 | 5.59e-2 | 5.59e-2 | 5.59e-2 | 5.59e-2 | 225 | +5.09e-4 | +5.09e-4 | +5.09e-4 | +3.72e-5 |
| 146 | 3.00e-2 | 2 | 5.02e-2 | 5.55e-2 | 5.29e-2 | 5.02e-2 | 160 | -6.24e-4 | -2.88e-5 | -3.26e-4 | -3.49e-5 |
| 147 | 3.00e-2 | 1 | 5.35e-2 | 5.35e-2 | 5.35e-2 | 5.35e-2 | 198 | +3.20e-4 | +3.20e-4 | +3.20e-4 | +6.44e-7 |
| 148 | 3.00e-2 | 2 | 5.24e-2 | 5.44e-2 | 5.34e-2 | 5.24e-2 | 160 | -2.34e-4 | +8.18e-5 | -7.61e-5 | -1.55e-5 |
| 149 | 3.00e-2 | 1 | 5.62e-2 | 5.62e-2 | 5.62e-2 | 5.62e-2 | 208 | +3.36e-4 | +3.36e-4 | +3.36e-4 | +1.97e-5 |
| 150 | 3.00e-3 | 2 | 2.81e-2 | 5.77e-2 | 4.29e-2 | 2.81e-2 | 165 | -4.36e-3 | +1.32e-4 | -2.12e-3 | -4.09e-4 |
| 151 | 3.00e-3 | 2 | 8.50e-3 | 1.44e-2 | 1.14e-2 | 8.50e-3 | 141 | -4.41e-3 | -3.73e-3 | -4.07e-3 | -1.10e-3 |
| 152 | 3.00e-3 | 1 | 6.20e-3 | 6.20e-3 | 6.20e-3 | 6.20e-3 | 181 | -1.75e-3 | -1.75e-3 | -1.75e-3 | -1.16e-3 |
| 153 | 3.00e-3 | 2 | 5.00e-3 | 5.58e-3 | 5.29e-3 | 5.00e-3 | 152 | -7.19e-4 | -5.48e-4 | -6.33e-4 | -1.06e-3 |
| 154 | 3.00e-3 | 1 | 5.42e-3 | 5.42e-3 | 5.42e-3 | 5.42e-3 | 182 | +4.37e-4 | +4.37e-4 | +4.37e-4 | -9.15e-4 |
| 155 | 3.00e-3 | 2 | 4.83e-3 | 5.64e-3 | 5.24e-3 | 4.83e-3 | 143 | -1.08e-3 | +1.86e-4 | -4.45e-4 | -8.32e-4 |
| 156 | 3.00e-3 | 2 | 4.94e-3 | 5.31e-3 | 5.13e-3 | 4.94e-3 | 142 | -5.06e-4 | +5.01e-4 | -2.31e-6 | -6.79e-4 |
| 157 | 3.00e-3 | 2 | 4.81e-3 | 5.35e-3 | 5.08e-3 | 4.81e-3 | 132 | -8.06e-4 | +4.31e-4 | -1.87e-4 | -5.92e-4 |
| 158 | 3.00e-3 | 1 | 5.24e-3 | 5.24e-3 | 5.24e-3 | 5.24e-3 | 177 | +4.74e-4 | +4.74e-4 | +4.74e-4 | -4.85e-4 |
| 159 | 3.00e-3 | 2 | 4.91e-3 | 5.16e-3 | 5.03e-3 | 4.91e-3 | 132 | -3.69e-4 | -8.37e-5 | -2.26e-4 | -4.37e-4 |
| 160 | 3.00e-3 | 2 | 4.83e-3 | 5.16e-3 | 4.99e-3 | 4.83e-3 | 132 | -5.07e-4 | +2.87e-4 | -1.10e-4 | -3.79e-4 |
| 161 | 3.00e-3 | 2 | 4.94e-3 | 5.24e-3 | 5.09e-3 | 4.94e-3 | 132 | -4.50e-4 | +5.33e-4 | +4.15e-5 | -3.04e-4 |
| 162 | 3.00e-3 | 2 | 5.04e-3 | 5.30e-3 | 5.17e-3 | 5.04e-3 | 135 | -3.74e-4 | +4.42e-4 | +3.38e-5 | -2.44e-4 |
| 163 | 3.00e-3 | 2 | 5.03e-3 | 5.23e-3 | 5.13e-3 | 5.03e-3 | 132 | -2.97e-4 | +2.25e-4 | -3.61e-5 | -2.07e-4 |
| 164 | 3.00e-3 | 2 | 4.90e-3 | 5.01e-3 | 4.96e-3 | 4.90e-3 | 132 | -1.69e-4 | -1.67e-5 | -9.29e-5 | -1.86e-4 |
| 165 | 3.00e-3 | 2 | 4.86e-3 | 5.29e-3 | 5.07e-3 | 4.86e-3 | 132 | -6.32e-4 | +4.57e-4 | -8.74e-5 | -1.73e-4 |
| 166 | 3.00e-3 | 1 | 5.17e-3 | 5.17e-3 | 5.17e-3 | 5.17e-3 | 165 | +3.77e-4 | +3.77e-4 | +3.77e-4 | -1.18e-4 |
| 167 | 3.00e-3 | 3 | 4.61e-3 | 5.28e-3 | 4.88e-3 | 4.61e-3 | 120 | -8.80e-4 | +1.26e-4 | -3.34e-4 | -1.80e-4 |
| 168 | 3.00e-3 | 1 | 5.44e-3 | 5.44e-3 | 5.44e-3 | 5.44e-3 | 161 | +1.02e-3 | +1.02e-3 | +1.02e-3 | -5.95e-5 |
| 169 | 3.00e-3 | 4 | 4.36e-3 | 5.67e-3 | 4.91e-3 | 4.36e-3 | 95 | -9.23e-4 | +2.36e-4 | -4.90e-4 | -2.17e-4 |
| 170 | 3.00e-3 | 1 | 4.98e-3 | 4.98e-3 | 4.98e-3 | 4.98e-3 | 137 | +9.70e-4 | +9.70e-4 | +9.70e-4 | -9.80e-5 |
| 171 | 3.00e-3 | 2 | 4.42e-3 | 4.91e-3 | 4.67e-3 | 4.42e-3 | 95 | -1.12e-3 | -1.09e-4 | -6.15e-4 | -2.01e-4 |
| 172 | 3.00e-3 | 3 | 4.48e-3 | 5.07e-3 | 4.71e-3 | 4.59e-3 | 105 | -1.18e-3 | +9.95e-4 | +1.48e-5 | -1.49e-4 |
| 173 | 3.00e-3 | 2 | 4.71e-3 | 5.10e-3 | 4.90e-3 | 4.71e-3 | 106 | -7.58e-4 | +7.47e-4 | -5.62e-6 | -1.30e-4 |
| 174 | 3.00e-3 | 3 | 4.55e-3 | 4.97e-3 | 4.70e-3 | 4.55e-3 | 96 | -8.77e-4 | +3.83e-4 | -1.74e-4 | -1.45e-4 |
| 175 | 3.00e-3 | 3 | 4.31e-3 | 4.51e-3 | 4.40e-3 | 4.38e-3 | 101 | -4.74e-4 | +1.60e-4 | -1.30e-4 | -1.39e-4 |
| 176 | 3.00e-3 | 3 | 4.43e-3 | 4.93e-3 | 4.62e-3 | 4.43e-3 | 101 | -9.33e-4 | +9.36e-4 | -4.32e-5 | -1.22e-4 |
| 177 | 3.00e-3 | 2 | 4.54e-3 | 5.21e-3 | 4.88e-3 | 4.54e-3 | 95 | -1.45e-3 | +1.08e-3 | -1.85e-4 | -1.47e-4 |
| 178 | 3.00e-3 | 3 | 4.02e-3 | 5.22e-3 | 4.56e-3 | 4.02e-3 | 76 | -1.80e-3 | +9.87e-4 | -7.05e-4 | -3.20e-4 |
| 179 | 3.00e-3 | 3 | 3.90e-3 | 4.76e-3 | 4.31e-3 | 3.90e-3 | 67 | -1.42e-3 | +1.43e-3 | -4.61e-4 | -3.84e-4 |
| 180 | 3.00e-3 | 5 | 3.71e-3 | 4.89e-3 | 4.05e-3 | 3.81e-3 | 73 | -3.00e-3 | +1.85e-3 | -3.65e-4 | -3.77e-4 |
| 181 | 3.00e-3 | 2 | 3.88e-3 | 4.37e-3 | 4.13e-3 | 3.88e-3 | 62 | -1.91e-3 | +1.30e-3 | -3.08e-4 | -3.80e-4 |
| 182 | 3.00e-3 | 5 | 3.46e-3 | 4.30e-3 | 3.72e-3 | 3.57e-3 | 62 | -2.65e-3 | +1.04e-3 | -3.84e-4 | -3.72e-4 |
| 183 | 3.00e-3 | 4 | 3.73e-3 | 4.18e-3 | 3.86e-3 | 3.73e-3 | 69 | -1.69e-3 | +1.60e-3 | -5.64e-5 | -2.78e-4 |
| 184 | 3.00e-3 | 4 | 3.52e-3 | 4.58e-3 | 3.90e-3 | 3.52e-3 | 55 | -2.69e-3 | +1.96e-3 | -5.52e-4 | -3.97e-4 |
| 185 | 3.00e-3 | 5 | 3.07e-3 | 4.03e-3 | 3.37e-3 | 3.21e-3 | 50 | -3.44e-3 | +1.61e-3 | -6.31e-4 | -4.74e-4 |
| 186 | 3.00e-3 | 7 | 2.95e-3 | 3.91e-3 | 3.21e-3 | 3.09e-3 | 45 | -5.07e-3 | +2.44e-3 | -4.73e-4 | -4.36e-4 |
| 187 | 3.00e-3 | 5 | 2.92e-3 | 4.16e-3 | 3.29e-3 | 2.97e-3 | 38 | -5.81e-3 | +3.48e-3 | -8.58e-4 | -6.21e-4 |
| 188 | 3.00e-3 | 7 | 2.72e-3 | 3.60e-3 | 3.01e-3 | 2.72e-3 | 36 | -5.21e-3 | +2.52e-3 | -7.35e-4 | -7.03e-4 |
| 189 | 3.00e-3 | 6 | 2.56e-3 | 3.83e-3 | 2.89e-3 | 2.56e-3 | 35 | -7.36e-3 | +4.53e-3 | -1.18e-3 | -9.23e-4 |
| 190 | 3.00e-3 | 6 | 2.90e-3 | 3.79e-3 | 3.14e-3 | 3.06e-3 | 43 | -5.96e-3 | +5.59e-3 | +8.09e-6 | -5.09e-4 |
| 191 | 3.00e-3 | 2 | 2.90e-3 | 6.49e-3 | 4.70e-3 | 6.49e-3 | 256 | -1.60e-3 | +3.15e-3 | +7.77e-4 | -2.41e-4 |
| 192 | 3.00e-3 | 1 | 7.23e-3 | 7.23e-3 | 7.23e-3 | 7.23e-3 | 267 | +4.05e-4 | +4.05e-4 | +4.05e-4 | -1.76e-4 |
| 193 | 3.00e-3 | 1 | 7.66e-3 | 7.66e-3 | 7.66e-3 | 7.66e-3 | 291 | +1.97e-4 | +1.97e-4 | +1.97e-4 | -1.39e-4 |
| 194 | 3.00e-3 | 1 | 7.73e-3 | 7.73e-3 | 7.73e-3 | 7.73e-3 | 281 | +3.23e-5 | +3.23e-5 | +3.23e-5 | -1.22e-4 |
| 195 | 3.00e-3 | 1 | 7.34e-3 | 7.34e-3 | 7.34e-3 | 7.34e-3 | 271 | -1.92e-4 | -1.92e-4 | -1.92e-4 | -1.29e-4 |
| 196 | 3.00e-3 | 1 | 7.42e-3 | 7.42e-3 | 7.42e-3 | 7.42e-3 | 257 | +4.13e-5 | +4.13e-5 | +4.13e-5 | -1.12e-4 |
| 197 | 3.00e-3 | 1 | 7.39e-3 | 7.39e-3 | 7.39e-3 | 7.39e-3 | 253 | -1.27e-5 | -1.27e-5 | -1.27e-5 | -1.02e-4 |
| 198 | 3.00e-3 | 1 | 7.57e-3 | 7.57e-3 | 7.57e-3 | 7.57e-3 | 269 | +8.55e-5 | +8.55e-5 | +8.55e-5 | -8.30e-5 |
| 199 | 3.00e-3 | 1 | 7.94e-3 | 7.94e-3 | 7.94e-3 | 7.94e-3 | 305 | +1.58e-4 | +1.58e-4 | +1.58e-4 | -5.90e-5 |

