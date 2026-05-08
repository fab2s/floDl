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
| cpu-async | 0.062366 | 0.9162 | +0.0037 | 1745.9 | 728 | 75.1 | 100% | 100% | 100% | 9.2 |

## Best Mode per Model

| Model | Best Eval | Mode | Fastest (within 2% of solo-0) | Mode |
|-------|-----------|------|-------------------------------|------|
| resnet-graph | 0.9162 | cpu-async | - | - |

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
| cpu-async | 1.9908 | 0.7219 | 0.5640 | 0.5184 | 0.4810 | 0.4677 | 0.4412 | 0.4833 | 0.4787 | 0.4692 | 0.2145 | 0.1752 | 0.1601 | 0.1463 | 0.1395 | 0.0816 | 0.0736 | 0.0685 | 0.0621 | 0.0624 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | cpu-async | 0 | cuda:0 | 0.4041 | 2.8 |
| resnet-graph | cpu-async | 1 | cuda:1 | 0.3018 | 3.7 |
| resnet-graph | cpu-async | 2 | cuda:2 | 0.2940 | 3.8 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) | GPU2 Peak (MB) | GPU2 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|---------------|---------------|
| resnet-graph | cpu-async | 358 | 357 | 398 | 392 | 390 | 379 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| cpu-async | gpu1 | 622.8 | 1.9 | epoch-boundary(70) |
| cpu-async | gpu1 | 710.2 | 1.3 | epoch-boundary(80) |
| cpu-async | gpu0 | 0.3 | 1.2 | epoch-boundary(0) |
| cpu-async | gpu2 | 596.0 | 1.2 | epoch-boundary(67) |
| cpu-async | gpu2 | 710.2 | 1.1 | epoch-boundary(80) |
| cpu-async | gpu1 | 596.1 | 1.1 | epoch-boundary(67) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | cpu-async | gpu0 | 1.2s | 0.0s | 0.0s | 0.0s | 1.2s |
| resnet-graph | cpu-async | gpu1 | 4.2s | 0.0s | 0.0s | 0.0s | 4.9s |
| resnet-graph | cpu-async | gpu2 | 2.3s | 0.0s | 0.0s | 0.0s | 3.0s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | cpu-async | 342 | 0 | 728 | 75.1 | 1314/7956 | 728 | 75.1 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | cpu-async | 167.6 | 9.6% |

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
| resnet-graph | cpu-async | 193 | 728 | 0 | 3.62e-3 | -4.93e-4 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | cpu-async | 0 | 728 | 8.45e-2 | 6.13e-2 | 2.29e-3 | 4.08e-1 | 27.3 | -1.80e-4 | 1.09e-3 |
| resnet-graph | cpu-async | 1 | 728 | 8.50e-2 | 6.26e-2 | 2.24e-3 | 4.60e-1 | 34.2 | -2.09e-4 | 1.32e-3 |
| resnet-graph | cpu-async | 2 | 728 | 8.49e-2 | 6.20e-2 | 2.30e-3 | 4.37e-1 | 38.5 | -2.08e-4 | 1.23e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | cpu-async | rank0 ↔ rank1 | +0.9934 |
| resnet-graph | cpu-async | rank0 ↔ rank2 | +0.9939 |
| resnet-graph | cpu-async | rank1 ↔ rank2 | +0.9925 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | cpu-async | 75 (1,2,4,5,7,8,9,10…143,149) | 0 (—) | — | 1,2,4,5,7,8,9,10…143,149 | — |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | cpu-async | 0e0 | 3 | 9 | 9 |
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
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 459 | +0.095 |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | 63 | +0.077 |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | 202 | +0.095 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | cpu-async | 726 | -0.022 | 192 | +0.400 | +0.591 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | cpu-async | 727 | 3.38e1–8.07e1 | 6.70e1 | 1.84e-3 | 3.25e-3 | 2.49e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 461 | 37–77659 | +4.093e-6 | 0.115 | +4.289e-6 | 0.137 | 94 | +9.891e-6 | 0.609 | 28–941 | +9.379e-4 | 0.658 |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 450 | 976–77659 | +5.219e-6 | 0.211 | +5.388e-6 | 0.244 | 93 | +1.037e-5 | 0.654 | 75–941 | +9.792e-4 | 0.854 |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | 64 | 78383–116942 | +7.856e-6 | 0.084 | +7.543e-6 | 0.078 | 49 | +6.269e-6 | 0.044 | 427–838 | -6.624e-4 | 0.044 |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | 203 | 117486–156051 | -2.224e-5 | 0.444 | -2.250e-5 | 0.457 | 50 | -2.722e-5 | 0.440 | 76–554 | +2.510e-3 | 0.689 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | +9.379e-4 | r0: +9.149e-4, r1: +9.536e-4, r2: +9.451e-4 | r0: 0.655, r1: 0.626, r2: 0.637 | 1.04× | ✓ |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +9.792e-4 | r0: +9.563e-4, r1: +9.953e-4, r2: +9.858e-4 | r0: 0.859, r1: 0.811, r2: 0.809 | 1.04× | ✓ |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | -6.624e-4 | r0: -6.860e-4, r1: -6.003e-4, r2: -6.990e-4 | r0: 0.049, r1: 0.035, r2: 0.048 | 1.16× | ✓ |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | +2.510e-3 | r0: +2.467e-3, r1: +2.564e-3, r2: +2.499e-3 | r0: 0.679, r1: 0.689, r2: 0.674 | 1.04× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | cpu-async | `█▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇███████▅▅▅▅▅▅▅▅▅▅▅▆▂▂▂▁▁▁▁▁▁▁▁▁▁` | `▁▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇███████▇▇██████████▆▇███▇▇▇▇▇▇▇▇` |

### Per-Epoch Detail: resnet-graph / cpu-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 11 | 1.12e-1 | 4.60e-1 | 2.05e-1 | 1.40e-1 | 45 | -3.14e-2 | +9.68e-3 | -1.13e-2 | -8.09e-3 |
| 1 | 3.00e-1 | 5 | 1.05e-1 | 1.62e-1 | 1.22e-1 | 1.11e-1 | 45 | -6.48e-3 | +1.65e-3 | -1.35e-3 | -4.55e-3 |
| 2 | 3.00e-1 | 6 | 1.17e-1 | 1.44e-1 | 1.27e-1 | 1.17e-1 | 47 | -1.96e-3 | +3.13e-3 | -1.54e-4 | -2.30e-3 |
| 3 | 3.00e-1 | 6 | 1.20e-1 | 1.59e-1 | 1.31e-1 | 1.32e-1 | 44 | -5.10e-3 | +3.38e-3 | -1.29e-4 | -1.17e-3 |
| 4 | 3.00e-1 | 6 | 1.18e-1 | 1.63e-1 | 1.32e-1 | 1.20e-1 | 39 | -4.07e-3 | +2.54e-3 | -8.30e-4 | -1.01e-3 |
| 5 | 3.00e-1 | 7 | 1.17e-1 | 1.53e-1 | 1.26e-1 | 1.18e-1 | 37 | -4.38e-3 | +3.56e-3 | -4.46e-4 | -7.21e-4 |
| 6 | 3.00e-1 | 7 | 1.14e-1 | 1.60e-1 | 1.23e-1 | 1.16e-1 | 37 | -7.75e-3 | +4.03e-3 | -6.80e-4 | -6.62e-4 |
| 7 | 3.00e-1 | 7 | 1.11e-1 | 1.53e-1 | 1.20e-1 | 1.14e-1 | 37 | -6.83e-3 | +3.88e-3 | -6.44e-4 | -6.14e-4 |
| 8 | 3.00e-1 | 11 | 9.80e-2 | 1.52e-1 | 1.12e-1 | 1.04e-1 | 33 | -7.17e-3 | +3.94e-3 | -7.40e-4 | -6.14e-4 |
| 9 | 3.00e-1 | 4 | 1.03e-1 | 1.51e-1 | 1.18e-1 | 1.03e-1 | 29 | -9.76e-3 | +4.96e-3 | -1.78e-3 | -1.05e-3 |
| 10 | 3.00e-1 | 7 | 1.08e-1 | 1.50e-1 | 1.21e-1 | 1.21e-1 | 46 | -8.61e-3 | +4.98e-3 | -3.64e-4 | -6.28e-4 |
| 11 | 3.00e-1 | 6 | 1.09e-1 | 1.62e-1 | 1.21e-1 | 1.10e-1 | 37 | -7.11e-3 | +3.23e-3 | -1.16e-3 | -8.40e-4 |
| 12 | 3.00e-1 | 6 | 1.12e-1 | 1.53e-1 | 1.23e-1 | 1.12e-1 | 35 | -5.62e-3 | +3.84e-3 | -7.30e-4 | -8.13e-4 |
| 13 | 3.00e-1 | 7 | 9.90e-2 | 1.49e-1 | 1.14e-1 | 1.17e-1 | 41 | -1.14e-2 | +4.20e-3 | -6.84e-4 | -6.03e-4 |
| 14 | 3.00e-1 | 7 | 1.09e-1 | 1.66e-1 | 1.21e-1 | 1.12e-1 | 40 | -7.20e-3 | +3.86e-3 | -8.04e-4 | -6.53e-4 |
| 15 | 3.00e-1 | 6 | 1.06e-1 | 1.50e-1 | 1.17e-1 | 1.06e-1 | 36 | -6.84e-3 | +3.61e-3 | -9.83e-4 | -8.02e-4 |
| 16 | 3.00e-1 | 6 | 1.20e-1 | 1.50e-1 | 1.28e-1 | 1.26e-1 | 48 | -4.25e-3 | +4.08e-3 | +7.30e-5 | -4.07e-4 |
| 17 | 3.00e-1 | 6 | 1.10e-1 | 1.49e-1 | 1.24e-1 | 1.10e-1 | 40 | -3.11e-3 | +2.22e-3 | -7.44e-4 | -6.03e-4 |
| 18 | 3.00e-1 | 6 | 1.14e-1 | 1.47e-1 | 1.23e-1 | 1.14e-1 | 40 | -5.02e-3 | +3.64e-3 | -4.75e-4 | -5.68e-4 |
| 19 | 3.00e-1 | 6 | 1.04e-1 | 1.47e-1 | 1.20e-1 | 1.28e-1 | 52 | -8.70e-3 | +3.46e-3 | -4.14e-4 | -3.95e-4 |
| 20 | 3.00e-1 | 7 | 1.17e-1 | 1.61e-1 | 1.33e-1 | 1.17e-1 | 43 | -2.96e-3 | +2.51e-3 | -5.35e-4 | -5.12e-4 |
| 21 | 3.00e-1 | 4 | 1.10e-1 | 1.48e-1 | 1.22e-1 | 1.14e-1 | 38 | -6.64e-3 | +2.95e-3 | -9.54e-4 | -6.61e-4 |
| 22 | 3.00e-1 | 6 | 1.13e-1 | 1.45e-1 | 1.21e-1 | 1.19e-1 | 42 | -5.96e-3 | +3.31e-3 | -3.59e-4 | -4.88e-4 |
| 23 | 3.00e-1 | 7 | 1.05e-1 | 1.55e-1 | 1.18e-1 | 1.09e-1 | 38 | -4.90e-3 | +3.18e-3 | -7.55e-4 | -5.84e-4 |
| 24 | 3.00e-1 | 9 | 9.79e-2 | 1.54e-1 | 1.10e-1 | 1.09e-1 | 38 | -1.06e-2 | +4.31e-3 | -8.28e-4 | -5.45e-4 |
| 25 | 3.00e-1 | 5 | 1.13e-1 | 1.48e-1 | 1.22e-1 | 1.13e-1 | 43 | -5.43e-3 | +4.02e-3 | -5.32e-4 | -5.60e-4 |
| 26 | 3.00e-1 | 9 | 1.10e-1 | 1.52e-1 | 1.18e-1 | 1.10e-1 | 36 | -5.73e-3 | +3.84e-3 | -4.99e-4 | -5.05e-4 |
| 27 | 3.00e-1 | 4 | 1.15e-1 | 1.53e-1 | 1.25e-1 | 1.16e-1 | 43 | -7.39e-3 | +4.00e-3 | -8.30e-4 | -6.30e-4 |
| 28 | 3.00e-1 | 6 | 1.06e-1 | 1.56e-1 | 1.19e-1 | 1.13e-1 | 39 | -6.54e-3 | +3.37e-3 | -8.08e-4 | -6.63e-4 |
| 29 | 3.00e-1 | 7 | 1.08e-1 | 1.58e-1 | 1.21e-1 | 1.21e-1 | 39 | -6.51e-3 | +3.80e-3 | -4.60e-4 | -4.62e-4 |
| 30 | 3.00e-1 | 7 | 9.95e-2 | 1.52e-1 | 1.15e-1 | 1.16e-1 | 49 | -6.49e-3 | +2.91e-3 | -6.98e-4 | -4.57e-4 |
| 31 | 3.00e-1 | 5 | 1.25e-1 | 1.60e-1 | 1.36e-1 | 1.25e-1 | 50 | -3.42e-3 | +3.50e-3 | -2.38e-4 | -4.04e-4 |
| 32 | 3.00e-1 | 7 | 1.15e-1 | 1.60e-1 | 1.30e-1 | 1.20e-1 | 41 | -3.28e-3 | +2.59e-3 | -4.08e-4 | -4.05e-4 |
| 33 | 3.00e-1 | 4 | 1.11e-1 | 1.59e-1 | 1.27e-1 | 1.14e-1 | 38 | -6.28e-3 | +3.18e-3 | -1.25e-3 | -7.08e-4 |
| 34 | 3.00e-1 | 6 | 1.04e-1 | 1.42e-1 | 1.15e-1 | 1.09e-1 | 36 | -8.01e-3 | +3.10e-3 | -9.56e-4 | -7.82e-4 |
| 35 | 3.00e-1 | 8 | 1.10e-1 | 1.49e-1 | 1.21e-1 | 1.10e-1 | 37 | -6.25e-3 | +4.22e-3 | -4.66e-4 | -6.50e-4 |
| 36 | 3.00e-1 | 5 | 1.16e-1 | 1.44e-1 | 1.23e-1 | 1.23e-1 | 44 | -5.48e-3 | +3.88e-3 | -9.23e-5 | -4.15e-4 |
| 37 | 3.00e-1 | 6 | 1.15e-1 | 1.54e-1 | 1.26e-1 | 1.15e-1 | 44 | -3.75e-3 | +2.98e-3 | -5.56e-4 | -4.96e-4 |
| 38 | 3.00e-1 | 6 | 1.10e-1 | 1.57e-1 | 1.24e-1 | 1.10e-1 | 36 | -5.27e-3 | +3.85e-3 | -7.66e-4 | -6.54e-4 |
| 39 | 3.00e-1 | 7 | 1.06e-1 | 1.56e-1 | 1.18e-1 | 1.18e-1 | 39 | -1.09e-2 | +4.13e-3 | -7.70e-4 | -5.98e-4 |
| 40 | 3.00e-1 | 6 | 1.18e-1 | 1.57e-1 | 1.25e-1 | 1.20e-1 | 43 | -6.75e-3 | +3.63e-3 | -5.38e-4 | -5.47e-4 |
| 41 | 3.00e-1 | 6 | 1.07e-1 | 1.62e-1 | 1.22e-1 | 1.07e-1 | 34 | -5.74e-3 | +3.47e-3 | -1.05e-3 | -7.85e-4 |
| 42 | 3.00e-1 | 7 | 1.09e-1 | 1.57e-1 | 1.18e-1 | 1.11e-1 | 36 | -9.41e-3 | +4.52e-3 | -7.93e-4 | -7.25e-4 |
| 43 | 3.00e-1 | 6 | 1.12e-1 | 1.59e-1 | 1.22e-1 | 1.16e-1 | 39 | -9.21e-3 | +4.28e-3 | -8.52e-4 | -7.36e-4 |
| 44 | 3.00e-1 | 7 | 1.08e-1 | 1.52e-1 | 1.17e-1 | 1.11e-1 | 40 | -7.37e-3 | +3.42e-3 | -7.21e-4 | -6.66e-4 |
| 45 | 3.00e-1 | 6 | 1.05e-1 | 1.56e-1 | 1.19e-1 | 1.12e-1 | 39 | -6.10e-3 | +4.34e-3 | -6.40e-4 | -6.27e-4 |
| 46 | 3.00e-1 | 8 | 9.81e-2 | 1.62e-1 | 1.11e-1 | 1.05e-1 | 30 | -8.98e-3 | +3.95e-3 | -1.09e-3 | -7.36e-4 |
| 47 | 3.00e-1 | 9 | 1.07e-1 | 1.49e-1 | 1.17e-1 | 1.07e-1 | 34 | -6.75e-3 | +4.64e-3 | -5.69e-4 | -6.36e-4 |
| 48 | 3.00e-1 | 5 | 1.08e-1 | 1.56e-1 | 1.19e-1 | 1.08e-1 | 34 | -8.97e-3 | +4.60e-3 | -1.35e-3 | -9.19e-4 |
| 49 | 3.00e-1 | 6 | 1.14e-1 | 1.46e-1 | 1.26e-1 | 1.23e-1 | 45 | -4.75e-3 | +4.29e-3 | -9.25e-5 | -5.52e-4 |
| 50 | 3.00e-1 | 7 | 1.05e-1 | 1.55e-1 | 1.19e-1 | 1.21e-1 | 40 | -8.54e-3 | +2.95e-3 | -7.59e-4 | -5.14e-4 |
| 51 | 3.00e-1 | 7 | 1.09e-1 | 1.64e-1 | 1.23e-1 | 1.19e-1 | 37 | -6.53e-3 | +3.60e-3 | -6.40e-4 | -4.99e-4 |
| 52 | 3.00e-1 | 6 | 1.09e-1 | 1.58e-1 | 1.26e-1 | 1.09e-1 | 36 | -5.11e-3 | +3.49e-3 | -9.68e-4 | -7.74e-4 |
| 53 | 3.00e-1 | 7 | 1.02e-1 | 1.53e-1 | 1.15e-1 | 1.18e-1 | 44 | -8.52e-3 | +4.17e-3 | -5.14e-4 | -5.16e-4 |
| 54 | 3.00e-1 | 6 | 1.10e-1 | 1.58e-1 | 1.24e-1 | 1.11e-1 | 38 | -3.62e-3 | +3.27e-3 | -7.53e-4 | -6.49e-4 |
| 55 | 3.00e-1 | 7 | 1.15e-1 | 1.59e-1 | 1.25e-1 | 1.15e-1 | 41 | -7.64e-3 | +4.49e-3 | -5.98e-4 | -6.25e-4 |
| 56 | 3.00e-1 | 6 | 1.12e-1 | 1.52e-1 | 1.22e-1 | 1.13e-1 | 38 | -4.38e-3 | +3.43e-3 | -7.52e-4 | -6.79e-4 |
| 57 | 3.00e-1 | 7 | 1.07e-1 | 1.56e-1 | 1.19e-1 | 1.12e-1 | 37 | -6.74e-3 | +4.06e-3 | -7.51e-4 | -6.66e-4 |
| 58 | 3.00e-1 | 7 | 1.02e-1 | 1.46e-1 | 1.16e-1 | 1.18e-1 | 39 | -6.26e-3 | +4.11e-3 | -3.57e-4 | -4.02e-4 |
| 59 | 3.00e-1 | 7 | 9.74e-2 | 1.65e-1 | 1.18e-1 | 1.22e-1 | 44 | -1.19e-2 | +4.48e-3 | -1.20e-3 | -6.25e-4 |
| 60 | 3.00e-1 | 5 | 1.33e-1 | 1.56e-1 | 1.39e-1 | 1.33e-1 | 58 | -3.01e-3 | +3.15e-3 | -1.60e-5 | -4.03e-4 |
| 61 | 3.00e-1 | 6 | 1.31e-1 | 1.73e-1 | 1.41e-1 | 1.31e-1 | 52 | -4.15e-3 | +2.58e-3 | -4.38e-4 | -4.24e-4 |
| 62 | 3.00e-1 | 4 | 1.14e-1 | 1.74e-1 | 1.34e-1 | 1.14e-1 | 41 | -5.41e-3 | +2.60e-3 | -1.66e-3 | -8.81e-4 |
| 63 | 3.00e-1 | 5 | 1.19e-1 | 1.46e-1 | 1.27e-1 | 1.22e-1 | 44 | -2.83e-3 | +3.23e-3 | -1.27e-4 | -5.86e-4 |
| 64 | 3.00e-1 | 6 | 1.19e-1 | 1.59e-1 | 1.29e-1 | 1.20e-1 | 42 | -4.61e-3 | +2.95e-3 | -5.91e-4 | -5.87e-4 |
| 65 | 3.00e-1 | 9 | 1.04e-1 | 1.61e-1 | 1.17e-1 | 1.09e-1 | 35 | -7.26e-3 | +3.41e-3 | -7.97e-4 | -6.11e-4 |
| 66 | 3.00e-1 | 3 | 1.25e-1 | 1.54e-1 | 1.36e-1 | 1.25e-1 | 44 | -3.81e-3 | +4.49e-3 | -9.65e-5 | -5.22e-4 |
| 67 | 3.00e-1 | 2 | 1.29e-1 | 2.26e-1 | 1.78e-1 | 2.26e-1 | 263 | +7.25e-4 | +2.13e-3 | +1.43e-3 | -1.44e-4 |
| 68 | 3.00e-1 | 1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 285 | +8.82e-5 | +8.82e-5 | +8.82e-5 | -1.21e-4 |
| 70 | 3.00e-1 | 2 | 2.27e-1 | 2.41e-1 | 2.34e-1 | 2.27e-1 | 263 | -2.18e-4 | +1.12e-4 | -5.32e-5 | -1.10e-4 |
| 72 | 3.00e-1 | 2 | 2.25e-1 | 2.31e-1 | 2.28e-1 | 2.25e-1 | 285 | -8.73e-5 | +5.21e-5 | -1.76e-5 | -9.28e-5 |
| 74 | 3.00e-1 | 1 | 2.36e-1 | 2.36e-1 | 2.36e-1 | 2.36e-1 | 329 | +1.43e-4 | +1.43e-4 | +1.43e-4 | -6.92e-5 |
| 75 | 3.00e-1 | 1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 301 | -1.82e-4 | -1.82e-4 | -1.82e-4 | -8.04e-5 |
| 76 | 3.00e-1 | 1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 322 | +4.18e-5 | +4.18e-5 | +4.18e-5 | -6.82e-5 |
| 77 | 3.00e-1 | 1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 296 | -2.69e-5 | -2.69e-5 | -2.69e-5 | -6.41e-5 |
| 78 | 3.00e-1 | 1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 293 | -2.26e-5 | -2.26e-5 | -2.26e-5 | -5.99e-5 |
| 79 | 3.00e-1 | 1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 316 | +9.06e-5 | +9.06e-5 | +9.06e-5 | -4.49e-5 |
| 80 | 3.00e-1 | 1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 285 | -1.01e-4 | -1.01e-4 | -1.01e-4 | -5.04e-5 |
| 82 | 3.00e-1 | 2 | 2.19e-1 | 2.35e-1 | 2.27e-1 | 2.19e-1 | 268 | -2.78e-4 | +1.50e-4 | -6.40e-5 | -5.52e-5 |
| 84 | 3.00e-1 | 2 | 2.23e-1 | 2.33e-1 | 2.28e-1 | 2.23e-1 | 268 | -1.74e-4 | +2.08e-4 | +1.70e-5 | -4.34e-5 |
| 86 | 3.00e-1 | 1 | 2.35e-1 | 2.35e-1 | 2.35e-1 | 2.35e-1 | 312 | +1.67e-4 | +1.67e-4 | +1.67e-4 | -2.24e-5 |
| 87 | 3.00e-1 | 1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 285 | -1.98e-4 | -1.98e-4 | -1.98e-4 | -3.99e-5 |
| 88 | 3.00e-1 | 1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 283 | +2.59e-5 | +2.59e-5 | +2.59e-5 | -3.33e-5 |
| 89 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 277 | -5.38e-5 | -5.38e-5 | -5.38e-5 | -3.54e-5 |
| 90 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 277 | -3.04e-5 | -3.04e-5 | -3.04e-5 | -3.49e-5 |
| 91 | 3.00e-1 | 1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 301 | +1.16e-4 | +1.16e-4 | +1.16e-4 | -1.98e-5 |
| 92 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 271 | -1.32e-4 | -1.32e-4 | -1.32e-4 | -3.11e-5 |
| 93 | 3.00e-1 | 1 | 2.28e-1 | 2.28e-1 | 2.28e-1 | 2.28e-1 | 293 | +1.58e-4 | +1.58e-4 | +1.58e-4 | -1.21e-5 |
| 94 | 3.00e-1 | 1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 271 | -1.23e-4 | -1.23e-4 | -1.23e-4 | -2.32e-5 |
| 95 | 3.00e-1 | 1 | 2.33e-1 | 2.33e-1 | 2.33e-1 | 2.33e-1 | 302 | +1.78e-4 | +1.78e-4 | +1.78e-4 | -3.11e-6 |
| 96 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 251 | -3.14e-4 | -3.14e-4 | -3.14e-4 | -3.42e-5 |
| 97 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 253 | -3.05e-5 | -3.05e-5 | -3.05e-5 | -3.38e-5 |
| 98 | 3.00e-1 | 1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 270 | +1.19e-4 | +1.19e-4 | +1.19e-4 | -1.85e-5 |
| 99 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 253 | -8.23e-5 | -8.23e-5 | -8.23e-5 | -2.49e-5 |
| 100 | 3.00e-2 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 261 | -1.53e-6 | -1.53e-6 | -1.53e-6 | -2.25e-5 |
| 101 | 3.00e-2 | 1 | 1.06e-1 | 1.06e-1 | 1.06e-1 | 1.06e-1 | 267 | -2.67e-3 | -2.67e-3 | -2.67e-3 | -2.87e-4 |
| 102 | 3.00e-2 | 2 | 3.53e-2 | 5.79e-2 | 4.66e-2 | 3.53e-2 | 237 | -2.10e-3 | -2.09e-3 | -2.10e-3 | -6.30e-4 |
| 104 | 3.00e-2 | 2 | 2.66e-2 | 3.04e-2 | 2.85e-2 | 2.66e-2 | 222 | -6.08e-4 | -4.93e-4 | -5.50e-4 | -6.16e-4 |
| 105 | 3.00e-2 | 1 | 2.73e-2 | 2.73e-2 | 2.73e-2 | 2.73e-2 | 251 | +1.07e-4 | +1.07e-4 | +1.07e-4 | -5.44e-4 |
| 106 | 3.00e-2 | 1 | 2.78e-2 | 2.78e-2 | 2.78e-2 | 2.78e-2 | 240 | +6.66e-5 | +6.66e-5 | +6.66e-5 | -4.83e-4 |
| 107 | 3.00e-2 | 1 | 2.94e-2 | 2.94e-2 | 2.94e-2 | 2.94e-2 | 260 | +2.27e-4 | +2.27e-4 | +2.27e-4 | -4.12e-4 |
| 108 | 3.00e-2 | 1 | 3.04e-2 | 3.04e-2 | 3.04e-2 | 3.04e-2 | 255 | +1.19e-4 | +1.19e-4 | +1.19e-4 | -3.59e-4 |
| 109 | 3.00e-2 | 1 | 3.04e-2 | 3.04e-2 | 3.04e-2 | 3.04e-2 | 260 | +9.11e-6 | +9.11e-6 | +9.11e-6 | -3.22e-4 |
| 110 | 3.00e-2 | 1 | 3.07e-2 | 3.07e-2 | 3.07e-2 | 3.07e-2 | 251 | +4.10e-5 | +4.10e-5 | +4.10e-5 | -2.85e-4 |
| 111 | 3.00e-2 | 1 | 3.18e-2 | 3.18e-2 | 3.18e-2 | 3.18e-2 | 255 | +1.27e-4 | +1.27e-4 | +1.27e-4 | -2.44e-4 |
| 112 | 3.00e-2 | 1 | 3.23e-2 | 3.23e-2 | 3.23e-2 | 3.23e-2 | 243 | +6.65e-5 | +6.65e-5 | +6.65e-5 | -2.13e-4 |
| 113 | 3.00e-2 | 2 | 3.27e-2 | 3.35e-2 | 3.31e-2 | 3.27e-2 | 227 | -1.03e-4 | +1.44e-4 | +2.05e-5 | -1.70e-4 |
| 114 | 3.00e-2 | 1 | 3.35e-2 | 3.35e-2 | 3.35e-2 | 3.35e-2 | 231 | +1.03e-4 | +1.03e-4 | +1.03e-4 | -1.43e-4 |
| 115 | 3.00e-2 | 1 | 3.46e-2 | 3.46e-2 | 3.46e-2 | 3.46e-2 | 237 | +1.31e-4 | +1.31e-4 | +1.31e-4 | -1.15e-4 |
| 116 | 3.00e-2 | 1 | 3.51e-2 | 3.51e-2 | 3.51e-2 | 3.51e-2 | 239 | +6.73e-5 | +6.73e-5 | +6.73e-5 | -9.71e-5 |
| 117 | 3.00e-2 | 1 | 3.74e-2 | 3.74e-2 | 3.74e-2 | 3.74e-2 | 265 | +2.40e-4 | +2.40e-4 | +2.40e-4 | -6.34e-5 |
| 118 | 3.00e-2 | 1 | 3.59e-2 | 3.59e-2 | 3.59e-2 | 3.59e-2 | 226 | -1.85e-4 | -1.85e-4 | -1.85e-4 | -7.55e-5 |
| 119 | 3.00e-2 | 1 | 3.62e-2 | 3.62e-2 | 3.62e-2 | 3.62e-2 | 221 | +3.26e-5 | +3.26e-5 | +3.26e-5 | -6.47e-5 |
| 120 | 3.00e-2 | 2 | 3.63e-2 | 3.71e-2 | 3.67e-2 | 3.63e-2 | 206 | -1.10e-4 | +1.15e-4 | +2.58e-6 | -5.30e-5 |
| 121 | 3.00e-2 | 1 | 3.76e-2 | 3.76e-2 | 3.76e-2 | 3.76e-2 | 226 | +1.58e-4 | +1.58e-4 | +1.58e-4 | -3.19e-5 |
| 122 | 3.00e-2 | 1 | 3.96e-2 | 3.96e-2 | 3.96e-2 | 3.96e-2 | 253 | +2.01e-4 | +2.01e-4 | +2.01e-4 | -8.59e-6 |
| 123 | 3.00e-2 | 1 | 3.96e-2 | 3.96e-2 | 3.96e-2 | 3.96e-2 | 239 | +2.20e-6 | +2.20e-6 | +2.20e-6 | -7.51e-6 |
| 124 | 3.00e-2 | 2 | 3.95e-2 | 4.03e-2 | 3.99e-2 | 4.03e-2 | 206 | -8.57e-6 | +9.66e-5 | +4.40e-5 | +2.80e-6 |
| 125 | 3.00e-2 | 1 | 4.30e-2 | 4.30e-2 | 4.30e-2 | 4.30e-2 | 247 | +2.62e-4 | +2.62e-4 | +2.62e-4 | +2.87e-5 |
| 126 | 3.00e-2 | 1 | 4.60e-2 | 4.60e-2 | 4.60e-2 | 4.60e-2 | 290 | +2.28e-4 | +2.28e-4 | +2.28e-4 | +4.87e-5 |
| 127 | 3.00e-2 | 1 | 4.64e-2 | 4.64e-2 | 4.64e-2 | 4.64e-2 | 265 | +3.21e-5 | +3.21e-5 | +3.21e-5 | +4.70e-5 |
| 128 | 3.00e-2 | 1 | 4.34e-2 | 4.34e-2 | 4.34e-2 | 4.34e-2 | 227 | -2.95e-4 | -2.95e-4 | -2.95e-4 | +1.29e-5 |
| 129 | 3.00e-2 | 2 | 4.12e-2 | 4.29e-2 | 4.20e-2 | 4.12e-2 | 180 | -2.23e-4 | -5.48e-5 | -1.39e-4 | -1.68e-5 |
| 130 | 3.00e-2 | 1 | 4.43e-2 | 4.43e-2 | 4.43e-2 | 4.43e-2 | 221 | +3.27e-4 | +3.27e-4 | +3.27e-4 | +1.76e-5 |
| 131 | 3.00e-2 | 1 | 4.56e-2 | 4.56e-2 | 4.56e-2 | 4.56e-2 | 219 | +1.34e-4 | +1.34e-4 | +1.34e-4 | +2.92e-5 |
| 132 | 3.00e-2 | 2 | 4.28e-2 | 4.46e-2 | 4.37e-2 | 4.28e-2 | 180 | -2.26e-4 | -1.16e-4 | -1.71e-4 | -9.38e-6 |
| 133 | 3.00e-2 | 1 | 4.87e-2 | 4.87e-2 | 4.87e-2 | 4.87e-2 | 234 | +5.53e-4 | +5.53e-4 | +5.53e-4 | +4.69e-5 |
| 134 | 3.00e-2 | 1 | 4.83e-2 | 4.83e-2 | 4.83e-2 | 4.83e-2 | 226 | -3.67e-5 | -3.67e-5 | -3.67e-5 | +3.85e-5 |
| 135 | 3.00e-2 | 2 | 4.37e-2 | 4.60e-2 | 4.48e-2 | 4.37e-2 | 166 | -3.15e-4 | -2.51e-4 | -2.83e-4 | -2.29e-5 |
| 136 | 3.00e-2 | 1 | 4.54e-2 | 4.54e-2 | 4.54e-2 | 4.54e-2 | 194 | +2.03e-4 | +2.03e-4 | +2.03e-4 | -2.43e-7 |
| 137 | 3.00e-2 | 2 | 4.57e-2 | 4.78e-2 | 4.68e-2 | 4.57e-2 | 166 | -2.74e-4 | +2.57e-4 | -8.23e-6 | -4.41e-6 |
| 138 | 3.00e-2 | 2 | 4.54e-2 | 4.59e-2 | 4.57e-2 | 4.54e-2 | 166 | -6.44e-5 | +2.62e-5 | -1.91e-5 | -7.65e-6 |
| 139 | 3.00e-2 | 1 | 5.16e-2 | 5.16e-2 | 5.16e-2 | 5.16e-2 | 216 | +5.91e-4 | +5.91e-4 | +5.91e-4 | +5.22e-5 |
| 140 | 3.00e-2 | 1 | 5.04e-2 | 5.04e-2 | 5.04e-2 | 5.04e-2 | 199 | -1.20e-4 | -1.20e-4 | -1.20e-4 | +3.50e-5 |
| 141 | 3.00e-2 | 2 | 4.72e-2 | 4.90e-2 | 4.81e-2 | 4.72e-2 | 166 | -2.24e-4 | -1.46e-4 | -1.85e-4 | -7.21e-6 |
| 142 | 3.00e-2 | 1 | 5.07e-2 | 5.07e-2 | 5.07e-2 | 5.07e-2 | 192 | +3.64e-4 | +3.64e-4 | +3.64e-4 | +2.99e-5 |
| 143 | 3.00e-2 | 2 | 4.95e-2 | 5.23e-2 | 5.09e-2 | 4.95e-2 | 175 | -3.16e-4 | +1.58e-4 | -7.90e-5 | +6.84e-6 |
| 144 | 3.00e-2 | 1 | 5.16e-2 | 5.16e-2 | 5.16e-2 | 5.16e-2 | 200 | +2.12e-4 | +2.12e-4 | +2.12e-4 | +2.73e-5 |
| 145 | 3.00e-2 | 2 | 5.11e-2 | 5.15e-2 | 5.13e-2 | 5.11e-2 | 166 | -4.17e-5 | -1.28e-5 | -2.72e-5 | +1.68e-5 |
| 146 | 3.00e-2 | 1 | 5.26e-2 | 5.26e-2 | 5.26e-2 | 5.26e-2 | 200 | +1.39e-4 | +1.39e-4 | +1.39e-4 | +2.90e-5 |
| 147 | 3.00e-2 | 2 | 5.10e-2 | 5.20e-2 | 5.15e-2 | 5.10e-2 | 166 | -1.11e-4 | -6.08e-5 | -8.61e-5 | +6.89e-6 |
| 148 | 3.00e-2 | 1 | 5.30e-2 | 5.30e-2 | 5.30e-2 | 5.30e-2 | 193 | +1.98e-4 | +1.98e-4 | +1.98e-4 | +2.60e-5 |
| 149 | 3.00e-2 | 2 | 5.21e-2 | 5.58e-2 | 5.40e-2 | 5.21e-2 | 166 | -4.03e-4 | +2.51e-4 | -7.59e-5 | +3.35e-6 |
| 150 | 3.00e-3 | 1 | 5.44e-2 | 5.44e-2 | 5.44e-2 | 5.44e-2 | 197 | +2.12e-4 | +2.12e-4 | +2.12e-4 | +2.42e-5 |
| 151 | 3.00e-3 | 3 | 8.15e-3 | 2.79e-2 | 1.66e-2 | 8.15e-3 | 142 | -4.68e-3 | -3.27e-3 | -3.86e-3 | -1.03e-3 |
| 152 | 3.00e-3 | 1 | 5.90e-3 | 5.90e-3 | 5.90e-3 | 5.90e-3 | 165 | -1.95e-3 | -1.95e-3 | -1.95e-3 | -1.12e-3 |
| 153 | 3.00e-3 | 2 | 4.60e-3 | 5.03e-3 | 4.81e-3 | 4.60e-3 | 125 | -1.13e-3 | -7.26e-4 | -9.27e-4 | -1.08e-3 |
| 154 | 3.00e-3 | 2 | 4.58e-3 | 5.03e-3 | 4.81e-3 | 4.58e-3 | 125 | -7.44e-4 | +5.84e-4 | -8.02e-5 | -9.00e-4 |
| 155 | 3.00e-3 | 2 | 4.79e-3 | 4.92e-3 | 4.85e-3 | 4.79e-3 | 133 | -2.01e-4 | +4.03e-4 | +1.01e-4 | -7.13e-4 |
| 156 | 3.00e-3 | 1 | 5.32e-3 | 5.32e-3 | 5.32e-3 | 5.32e-3 | 194 | +5.41e-4 | +5.41e-4 | +5.41e-4 | -5.88e-4 |
| 157 | 3.00e-3 | 3 | 4.45e-3 | 5.02e-3 | 4.71e-3 | 4.45e-3 | 121 | -6.03e-4 | -3.51e-4 | -4.44e-4 | -5.49e-4 |
| 158 | 3.00e-3 | 2 | 4.56e-3 | 4.75e-3 | 4.65e-3 | 4.56e-3 | 121 | -3.36e-4 | +4.50e-4 | +5.72e-5 | -4.38e-4 |
| 159 | 3.00e-3 | 2 | 4.79e-3 | 4.94e-3 | 4.87e-3 | 4.79e-3 | 129 | -2.39e-4 | +5.02e-4 | +1.31e-4 | -3.33e-4 |
| 160 | 3.00e-3 | 2 | 4.75e-3 | 4.94e-3 | 4.84e-3 | 4.75e-3 | 133 | -3.03e-4 | +2.02e-4 | -5.06e-5 | -2.82e-4 |
| 161 | 3.00e-3 | 2 | 4.81e-3 | 5.12e-3 | 4.97e-3 | 4.81e-3 | 129 | -4.86e-4 | +5.07e-4 | +1.06e-5 | -2.31e-4 |
| 162 | 3.00e-3 | 2 | 4.79e-3 | 5.11e-3 | 4.95e-3 | 4.79e-3 | 131 | -4.94e-4 | +3.97e-4 | -4.83e-5 | -2.01e-4 |
| 163 | 3.00e-3 | 2 | 4.86e-3 | 5.14e-3 | 5.00e-3 | 4.86e-3 | 122 | -4.50e-4 | +4.30e-4 | -1.04e-5 | -1.69e-4 |
| 164 | 3.00e-3 | 2 | 4.50e-3 | 4.97e-3 | 4.73e-3 | 4.50e-3 | 101 | -9.85e-4 | +1.53e-4 | -4.16e-4 | -2.22e-4 |
| 165 | 3.00e-3 | 3 | 4.15e-3 | 4.67e-3 | 4.38e-3 | 4.15e-3 | 101 | -7.54e-4 | +2.73e-4 | -2.95e-4 | -2.48e-4 |
| 166 | 3.00e-3 | 2 | 4.67e-3 | 4.70e-3 | 4.69e-3 | 4.67e-3 | 122 | -4.31e-5 | +9.39e-4 | +4.48e-4 | -1.21e-4 |
| 167 | 3.00e-3 | 3 | 4.36e-3 | 5.22e-3 | 4.66e-3 | 4.36e-3 | 97 | -1.80e-3 | +7.29e-4 | -3.75e-4 | -1.96e-4 |
| 168 | 3.00e-3 | 2 | 4.25e-3 | 4.60e-3 | 4.43e-3 | 4.25e-3 | 91 | -8.94e-4 | +4.15e-4 | -2.39e-4 | -2.11e-4 |
| 169 | 3.00e-3 | 2 | 4.24e-3 | 4.95e-3 | 4.60e-3 | 4.24e-3 | 91 | -1.68e-3 | +1.05e-3 | -3.18e-4 | -2.45e-4 |
| 170 | 3.00e-3 | 4 | 4.07e-3 | 4.83e-3 | 4.28e-3 | 4.09e-3 | 87 | -1.81e-3 | +1.03e-3 | -2.30e-4 | -2.44e-4 |
| 171 | 3.00e-3 | 2 | 4.02e-3 | 4.71e-3 | 4.37e-3 | 4.02e-3 | 76 | -2.06e-3 | +1.16e-3 | -4.53e-4 | -3.00e-4 |
| 172 | 3.00e-3 | 4 | 3.78e-3 | 4.75e-3 | 4.09e-3 | 3.78e-3 | 71 | -2.14e-3 | +1.28e-3 | -4.44e-4 | -3.60e-4 |
| 173 | 3.00e-3 | 4 | 3.49e-3 | 4.40e-3 | 3.80e-3 | 3.49e-3 | 68 | -2.28e-3 | +1.33e-3 | -4.87e-4 | -4.18e-4 |
| 174 | 3.00e-3 | 3 | 3.70e-3 | 4.77e-3 | 4.08e-3 | 3.70e-3 | 64 | -3.51e-3 | +2.65e-3 | -3.70e-4 | -4.31e-4 |
| 175 | 3.00e-3 | 4 | 3.37e-3 | 4.45e-3 | 3.84e-3 | 3.63e-3 | 62 | -2.85e-3 | +1.70e-3 | -4.14e-4 | -4.34e-4 |
| 176 | 3.00e-3 | 5 | 3.12e-3 | 4.37e-3 | 3.52e-3 | 3.12e-3 | 46 | -4.66e-3 | +1.86e-3 | -9.85e-4 | -6.75e-4 |
| 177 | 3.00e-3 | 6 | 3.09e-3 | 4.30e-3 | 3.43e-3 | 3.09e-3 | 51 | -4.12e-3 | +3.48e-3 | -4.94e-4 | -6.19e-4 |
| 178 | 3.00e-3 | 3 | 3.40e-3 | 4.31e-3 | 3.72e-3 | 3.40e-3 | 56 | -4.28e-3 | +3.43e-3 | -3.67e-4 | -5.83e-4 |
| 179 | 3.00e-3 | 6 | 2.82e-3 | 4.11e-3 | 3.19e-3 | 2.84e-3 | 45 | -4.48e-3 | +2.02e-3 | -1.01e-3 | -7.70e-4 |
| 180 | 3.00e-3 | 5 | 2.89e-3 | 3.83e-3 | 3.31e-3 | 2.89e-3 | 44 | -1.95e-3 | +3.67e-3 | -4.38e-4 | -7.15e-4 |
| 181 | 3.00e-3 | 6 | 2.81e-3 | 4.05e-3 | 3.21e-3 | 2.81e-3 | 34 | -4.60e-3 | +4.03e-3 | -8.22e-4 | -8.35e-4 |
| 182 | 3.00e-3 | 7 | 2.75e-3 | 3.50e-3 | 2.99e-3 | 2.84e-3 | 41 | -5.45e-3 | +2.69e-3 | -5.26e-4 | -6.67e-4 |
| 183 | 3.00e-3 | 5 | 2.83e-3 | 3.96e-3 | 3.20e-3 | 2.87e-3 | 40 | -3.84e-3 | +3.94e-3 | -7.57e-4 | -7.38e-4 |
| 184 | 3.00e-3 | 7 | 2.90e-3 | 3.80e-3 | 3.12e-3 | 3.14e-3 | 46 | -5.42e-3 | +3.80e-3 | -1.63e-4 | -3.95e-4 |
| 185 | 3.00e-3 | 5 | 3.00e-3 | 4.18e-3 | 3.30e-3 | 3.09e-3 | 43 | -6.26e-3 | +3.11e-3 | -7.74e-4 | -5.32e-4 |
| 186 | 3.00e-3 | 6 | 2.83e-3 | 4.08e-3 | 3.17e-3 | 2.83e-3 | 40 | -6.04e-3 | +3.18e-3 | -9.80e-4 | -7.42e-4 |
| 187 | 3.00e-3 | 7 | 2.85e-3 | 3.86e-3 | 3.07e-3 | 2.88e-3 | 40 | -5.03e-3 | +3.86e-3 | -4.96e-4 | -6.02e-4 |
| 188 | 3.00e-3 | 6 | 2.61e-3 | 3.62e-3 | 3.04e-3 | 2.72e-3 | 34 | -3.59e-3 | +3.06e-3 | -7.86e-4 | -7.29e-4 |
| 189 | 3.00e-3 | 8 | 2.36e-3 | 3.73e-3 | 2.77e-3 | 2.53e-3 | 29 | -6.96e-3 | +4.34e-3 | -1.02e-3 | -8.28e-4 |
| 190 | 3.00e-3 | 9 | 2.38e-3 | 3.37e-3 | 2.75e-3 | 3.37e-3 | 56 | -9.71e-3 | +4.06e-3 | +8.18e-5 | -2.22e-5 |
| 191 | 3.00e-3 | 4 | 3.19e-3 | 4.25e-3 | 3.57e-3 | 3.38e-3 | 54 | -4.30e-3 | +2.46e-3 | -6.35e-4 | -2.36e-4 |
| 192 | 3.00e-3 | 6 | 2.76e-3 | 4.06e-3 | 3.11e-3 | 2.92e-3 | 40 | -6.46e-3 | +2.30e-3 | -9.70e-4 | -5.16e-4 |
| 193 | 3.00e-3 | 6 | 2.97e-3 | 3.78e-3 | 3.15e-3 | 3.00e-3 | 42 | -4.51e-3 | +3.38e-3 | -3.63e-4 | -4.48e-4 |
| 194 | 3.00e-3 | 6 | 2.93e-3 | 4.09e-3 | 3.25e-3 | 3.02e-3 | 44 | -4.17e-3 | +3.87e-3 | -5.08e-4 | -4.80e-4 |
| 195 | 3.00e-3 | 5 | 2.94e-3 | 4.00e-3 | 3.29e-3 | 3.34e-3 | 54 | -6.10e-3 | +3.24e-3 | -3.95e-4 | -4.05e-4 |
| 196 | 3.00e-3 | 6 | 3.13e-3 | 4.32e-3 | 3.42e-3 | 3.13e-3 | 42 | -4.45e-3 | +2.74e-3 | -6.93e-4 | -5.35e-4 |
| 197 | 3.00e-3 | 6 | 3.10e-3 | 4.06e-3 | 3.40e-3 | 3.29e-3 | 44 | -3.80e-3 | +3.07e-3 | -3.50e-4 | -4.35e-4 |
| 198 | 3.00e-3 | 6 | 2.62e-3 | 4.25e-3 | 3.05e-3 | 2.89e-3 | 37 | -9.78e-3 | +3.00e-3 | -1.50e-3 | -8.07e-4 |
| 199 | 3.00e-3 | 5 | 3.11e-3 | 4.07e-3 | 3.49e-3 | 3.62e-3 | 60 | -7.45e-3 | +4.24e-3 | -9.19e-5 | -4.93e-4 |

