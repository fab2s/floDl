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
- resnet-graph/cpu-sync
- resnet-graph/cpu-cadence
- resnet-graph/cpu-async

## Per-Model Results

GPU columns = compute utilization % (not load). Idle = total time with <5% utilization.

### resnet-graph

> Published: CIFAR-10 91.25% acc, Graph builder ([He et al. 2015](https://arxiv.org/abs/1512.03385), Table 6)

| Mode | Loss | Eval | vs Ref | Total (s) | Syncs | Avg Sync (ms) | GPU0 | GPU1 | GPU2 | Idle (s) |
|------|------|------|--------|-----------|-------|--------------|------|------|------|----------|
| nccl-async | 0.064540 | 0.9194 | +0.0069 | 1843.6 | 248 | 39.1 | 100% | 100% | 100% | 5.1 |

## Best Mode per Model

| Model | Best Eval | Mode | Fastest (within 2% of solo-0) | Mode |
|-------|-----------|------|-------------------------------|------|
| resnet-graph | 0.9194 | nccl-async | - | - |

## Eval Quality (vs solo-0)

| Model | nccl-async |
|-------|------------|
| resnet-graph | - |

## Convergence Quality (loss ratio vs solo-0)

| Model | nccl-async |
|-------|------------|
| resnet-graph | - |

## Per-Epoch Loss Trajectory

### resnet-graph (sampled, 200 epochs)

| Mode | E0 | E10 | E21 | E31 | E42 | E52 | E63 | E73 | E84 | E94 | E105 | E115 | E126 | E136 | E147 | E157 | E168 | E178 | E189 | E199 |
|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| nccl-async | 1.9867 | 0.7538 | 0.5986 | 0.5608 | 0.5336 | 0.5006 | 0.4970 | 0.4852 | 0.4780 | 0.4698 | 0.2192 | 0.1800 | 0.1626 | 0.1537 | 0.1490 | 0.0863 | 0.0775 | 0.0747 | 0.0694 | 0.0645 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | nccl-async | 0 | cuda:0 | 0.4048 | 2.7 |
| resnet-graph | nccl-async | 1 | cuda:1 | 0.3027 | 3.6 |
| resnet-graph | nccl-async | 2 | cuda:2 | 0.2924 | 3.4 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) | GPU2 Peak (MB) | GPU2 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|---------------|---------------|
| resnet-graph | nccl-async | 356 | 355 | 386 | 382 | 380 | 379 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| nccl-async | gpu1 | 1842.5 | 1.0 | epoch-boundary(199) |
| nccl-async | gpu2 | 1842.7 | 0.8 | epoch-boundary(199) |
| nccl-async | gpu0 | 1842.5 | 0.6 | epoch-boundary(199) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | nccl-async | gpu0 | 0.6s | 0.0s | 0.0s | 0.0s | 1.6s |
| resnet-graph | nccl-async | gpu1 | 1.0s | 0.0s | 0.0s | 0.0s | 2.1s |
| resnet-graph | nccl-async | gpu2 | 0.8s | 0.0s | 0.0s | 0.0s | 1.4s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | nccl-async | 18 | 0 | 248 | 39.1 | 9825/11754 | 0 | 0.0 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | nccl-async | 155.5 | 8.4% |

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
| resnet-graph | nccl-async | 177 | 248 | 0 | 7.35e-3 | +5.15e-6 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | nccl-async | 0 | 248 | 1.09e-1 | 8.09e-2 | 0.00e0 | 4.62e-1 | 23.4 | -1.84e-4 | 1.86e-3 |
| resnet-graph | nccl-async | 1 | 248 | 1.10e-1 | 8.36e-2 | 0.00e0 | 5.20e-1 | 56.9 | -1.97e-4 | 2.17e-3 |
| resnet-graph | nccl-async | 2 | 248 | 1.10e-1 | 8.28e-2 | 0.00e0 | 4.20e-1 | 19.8 | -1.59e-4 | 2.05e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | nccl-async | rank0 ↔ rank1 | +0.9975 |
| resnet-graph | nccl-async | rank0 ↔ rank2 | +0.9905 |
| resnet-graph | nccl-async | rank1 ↔ rank2 | +0.9888 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | nccl-async | 51 (0,1,2,3,4,5,6,7…147,148) | 0 (—) | — | 0,1,2,3,4,5,6,7…147,148 | — |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | nccl-async | 0e0 | 3 | 22 | 22 |
| resnet-graph | nccl-async | 0e0 | 5 | 11 | 11 |
| resnet-graph | nccl-async | 0e0 | 10 | 5 | 5 |
| resnet-graph | nccl-async | 1e-4 | 3 | 0 | 0 |
| resnet-graph | nccl-async | 1e-4 | 5 | 0 | 0 |
| resnet-graph | nccl-async | 1e-4 | 10 | 0 | 0 |
| resnet-graph | nccl-async | 1e-3 | 3 | 0 | 0 |
| resnet-graph | nccl-async | 1e-3 | 5 | 0 | 0 |
| resnet-graph | nccl-async | 1e-3 | 10 | 0 | 0 |
| resnet-graph | nccl-async | 1e-2 | 3 | 0 | 0 |
| resnet-graph | nccl-async | 1e-2 | 5 | 0 | 0 |
| resnet-graph | nccl-async | 1e-2 | 10 | 0 | 0 |
| resnet-graph | nccl-async | 1e-1 | 3 | 0 | 0 |
| resnet-graph | nccl-async | 1e-1 | 5 | 0 | 0 |
| resnet-graph | nccl-async | 1e-1 | 10 | 0 | 0 |

### Predictive Value by LR Window (top scale)

Per-LR-window Pearson `r(λ_raw_t, ln(D_{t+1}))`. Pairs that straddle a window boundary are excluded so the LR-drop collapse cannot leak in as artefactual signal. Reading: a clean R1 (exponential growth at fixed LR) shows up as *non-zero* `r` within each window, with sign and magnitude that may differ between warmup, post-drop transient, and late-training phases.

| Model | Mode | LR | Epochs | n_pairs | r(λ → ln D_{t+1}) |
|-------|------|---:|--------|--------:|----------------:|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 156 | +0.147 |
| resnet-graph | nccl-async | 3.00e-2 | 100–148 | 43 | +0.239 |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | 44 | +0.064 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | nccl-async | 245 | +0.006 | 176 | +0.171 | +0.339 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | nccl-async | 246 | 3.41e1–7.95e1 | 6.34e1 | 4.79e-3 | 7.24e-3 | 8.08e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 158 | 67–77815 | +1.243e-5 | 0.495 | +1.299e-5 | 0.524 | 89 | +6.827e-6 | 0.357 | 34–1057 | +9.408e-4 | 0.717 |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 146 | 894–77815 | +1.218e-5 | 0.608 | +1.259e-5 | 0.613 | 88 | +6.436e-6 | 0.336 | 62–1057 | +9.413e-4 | 0.871 |
| resnet-graph | nccl-async | 3.00e-2 | 100–148 | 44 | 78661–116336 | +1.040e-5 | 0.072 | +1.031e-5 | 0.071 | 44 | +1.031e-5 | 0.071 | 787–1017 | -6.902e-4 | 0.007 |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | 45 | 117295–155773 | -8.975e-6 | 0.049 | -9.012e-6 | 0.049 | 44 | -9.435e-6 | 0.053 | 786–1002 | +8.161e-4 | 0.010 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | +9.408e-4 | r0: +9.306e-4, r1: +9.488e-4, r2: +9.474e-4 | r0: 0.736, r1: 0.712, r2: 0.700 | 1.02× | ✓ |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +9.413e-4 | r0: +9.288e-4, r1: +9.460e-4, r2: +9.515e-4 | r0: 0.882, r1: 0.855, r2: 0.866 | 1.02× | ✓ |
| resnet-graph | nccl-async | 3.00e-2 | 100–148 | -6.902e-4 | r0: -6.856e-4, r1: -6.838e-4, r2: -7.017e-4 | r0: 0.007, r1: 0.007, r2: 0.008 | 1.03× | ✓ |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | +8.161e-4 | r0: +8.578e-4, r1: +7.936e-4, r2: +7.985e-4 | r0: 0.011, r1: 0.009, r2: 0.009 | 1.08× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | nccl-async | `█▇▇▇██████████████████████████▅▄▄▄▄▅▅▅▅▅▅▅▅▅▆▁▁▁▁▁▁▁▁▁▁▁▁▁▁` | `▁█▇▇██████████████████████████▇▆▇▇▇██████████▆▇▇▇▇▇████████` |

### Per-Epoch Detail: resnet-graph / nccl-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 13 | 0.00e0 | 5.20e-1 | 1.32e-1 | 6.94e-2 | 21 | -4.42e-2 | +5.74e-3 | -1.06e-2 | -6.99e-3 |
| 1 | 3.00e-1 | 10 | 6.48e-2 | 1.04e-1 | 7.76e-2 | 7.59e-2 | 28 | -2.26e-2 | +1.83e-2 | +2.66e-4 | -1.80e-3 |
| 2 | 3.00e-1 | 10 | 7.65e-2 | 1.21e-1 | 9.02e-2 | 9.70e-2 | 38 | -1.56e-2 | +1.85e-2 | +1.00e-3 | -3.90e-6 |
| 3 | 3.00e-1 | 6 | 9.43e-2 | 1.46e-1 | 1.09e-1 | 9.43e-2 | 36 | -9.64e-3 | +8.91e-3 | -5.78e-5 | -1.90e-4 |
| 4 | 3.00e-1 | 7 | 9.63e-2 | 1.45e-1 | 1.11e-1 | 1.12e-1 | 41 | -1.23e-2 | +8.78e-3 | +3.31e-4 | +5.71e-5 |
| 5 | 3.00e-1 | 8 | 1.09e-1 | 1.42e-1 | 1.16e-1 | 1.12e-1 | 41 | -6.56e-3 | +5.52e-3 | -3.02e-5 | -2.22e-5 |
| 6 | 3.00e-1 | 5 | 1.06e-1 | 1.41e-1 | 1.17e-1 | 1.07e-1 | 40 | -7.17e-3 | +4.61e-3 | -3.39e-4 | -2.06e-4 |
| 7 | 3.00e-1 | 6 | 1.06e-1 | 1.46e-1 | 1.16e-1 | 1.12e-1 | 38 | -6.17e-3 | +6.93e-3 | +1.30e-4 | -8.25e-5 |
| 8 | 3.00e-1 | 7 | 1.03e-1 | 1.39e-1 | 1.10e-1 | 1.03e-1 | 39 | -7.53e-3 | +6.89e-3 | -2.46e-4 | -2.10e-4 |
| 9 | 3.00e-1 | 7 | 1.01e-1 | 1.42e-1 | 1.10e-1 | 1.06e-1 | 40 | -8.44e-3 | +8.29e-3 | +1.59e-4 | -7.40e-5 |
| 10 | 3.00e-1 | 1 | 1.02e-1 | 1.02e-1 | 1.02e-1 | 1.02e-1 | 40 | -8.83e-4 | -8.83e-4 | -8.83e-4 | -1.55e-4 |
| 11 | 3.00e-1 | 1 | 9.97e-2 | 9.97e-2 | 9.97e-2 | 9.97e-2 | 345 | -6.66e-5 | -6.66e-5 | -6.66e-5 | -1.46e-4 |
| 12 | 3.00e-1 | 1 | 2.35e-1 | 2.35e-1 | 2.35e-1 | 2.35e-1 | 323 | +2.65e-3 | +2.65e-3 | +2.65e-3 | +1.34e-4 |
| 13 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 323 | -2.49e-4 | -2.49e-4 | -2.49e-4 | +9.56e-5 |
| 14 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 369 | -3.28e-5 | -3.28e-5 | -3.28e-5 | +8.28e-5 |
| 16 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 386 | +3.26e-5 | +3.26e-5 | +3.26e-5 | +7.77e-5 |
| 17 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 304 | +4.40e-5 | +4.40e-5 | +4.40e-5 | +7.44e-5 |
| 18 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 320 | -2.25e-4 | -2.25e-4 | -2.25e-4 | +4.44e-5 |
| 19 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 337 | +7.09e-5 | +7.09e-5 | +7.09e-5 | +4.70e-5 |
| 21 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 390 | +5.22e-6 | +5.22e-6 | +5.22e-6 | +4.29e-5 |
| 22 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 320 | +8.65e-5 | +8.65e-5 | +8.65e-5 | +4.72e-5 |
| 23 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 289 | -1.61e-4 | -1.61e-4 | -1.61e-4 | +2.64e-5 |
| 24 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 270 | -1.06e-4 | -1.06e-4 | -1.06e-4 | +1.31e-5 |
| 25 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 259 | -4.48e-5 | -4.48e-5 | -4.48e-5 | +7.29e-6 |
| 26 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 275 | -3.05e-5 | -3.05e-5 | -3.05e-5 | +3.52e-6 |
| 27 | 3.00e-1 | 1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 304 | +2.03e-5 | +2.03e-5 | +2.03e-5 | +5.20e-6 |
| 28 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 337 | +9.33e-5 | +9.33e-5 | +9.33e-5 | +1.40e-5 |
| 29 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 335 | +1.50e-4 | +1.50e-4 | +1.50e-4 | +2.76e-5 |
| 30 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 324 | -4.09e-5 | -4.09e-5 | -4.09e-5 | +2.08e-5 |
| 32 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 369 | -8.81e-6 | -8.81e-6 | -8.81e-6 | +1.78e-5 |
| 33 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 313 | +1.17e-4 | +1.17e-4 | +1.17e-4 | +2.78e-5 |
| 34 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 272 | -1.93e-4 | -1.93e-4 | -1.93e-4 | +5.71e-6 |
| 35 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 292 | -1.04e-4 | -1.04e-4 | -1.04e-4 | -5.24e-6 |
| 36 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 304 | +5.56e-5 | +5.56e-5 | +5.56e-5 | +8.44e-7 |
| 37 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 293 | +1.73e-5 | +1.73e-5 | +1.73e-5 | +2.49e-6 |
| 38 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 291 | -5.13e-6 | -5.13e-6 | -5.13e-6 | +1.73e-6 |
| 39 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 292 | -5.60e-5 | -5.60e-5 | -5.60e-5 | -4.05e-6 |
| 40 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 267 | +7.94e-5 | +7.94e-5 | +7.94e-5 | +4.30e-6 |
| 41 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 282 | -1.47e-4 | -1.47e-4 | -1.47e-4 | -1.08e-5 |
| 42 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 313 | +1.11e-4 | +1.11e-4 | +1.11e-4 | +1.40e-6 |
| 43 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 323 | +3.83e-5 | +3.83e-5 | +3.83e-5 | +5.09e-6 |
| 44 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 289 | +6.65e-5 | +6.65e-5 | +6.65e-5 | +1.12e-5 |
| 46 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 349 | -8.34e-5 | -8.34e-5 | -8.34e-5 | +1.76e-6 |
| 47 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 323 | +1.66e-4 | +1.66e-4 | +1.66e-4 | +1.82e-5 |
| 48 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 316 | -6.27e-5 | -6.27e-5 | -6.27e-5 | +1.01e-5 |
| 49 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 321 | -3.64e-5 | -3.64e-5 | -3.64e-5 | +5.46e-6 |
| 50 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 330 | +5.79e-5 | +5.79e-5 | +5.79e-5 | +1.07e-5 |
| 52 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 375 | +7.24e-7 | +7.24e-7 | +7.24e-7 | +9.71e-6 |
| 53 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 288 | +9.72e-5 | +9.72e-5 | +9.72e-5 | +1.85e-5 |
| 54 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 305 | -2.12e-4 | -2.12e-4 | -2.12e-4 | -4.57e-6 |
| 55 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 307 | +7.55e-5 | +7.55e-5 | +7.55e-5 | +3.45e-6 |
| 56 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 356 | +2.69e-5 | +2.69e-5 | +2.69e-5 | +5.79e-6 |
| 57 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 305 | +3.45e-5 | +3.45e-5 | +3.45e-5 | +8.66e-6 |
| 58 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 311 | -9.67e-5 | -9.67e-5 | -9.67e-5 | -1.87e-6 |
| 59 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 328 | +5.09e-6 | +5.09e-6 | +5.09e-6 | -1.17e-6 |
| 61 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 331 | +5.33e-5 | +5.33e-5 | +5.33e-5 | +4.28e-6 |
| 62 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 290 | +1.02e-5 | +1.02e-5 | +1.02e-5 | +4.87e-6 |
| 63 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 279 | -1.35e-4 | -1.35e-4 | -1.35e-4 | -9.08e-6 |
| 64 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 268 | -1.90e-5 | -1.90e-5 | -1.90e-5 | -1.01e-5 |
| 65 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 327 | -2.85e-5 | -2.85e-5 | -2.85e-5 | -1.19e-5 |
| 66 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 310 | +1.49e-4 | +1.49e-4 | +1.49e-4 | +4.18e-6 |
| 67 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 320 | -4.21e-6 | -4.21e-6 | -4.21e-6 | +3.34e-6 |
| 68 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 341 | -1.97e-5 | -1.97e-5 | -1.97e-5 | +1.04e-6 |
| 69 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 347 | +1.21e-4 | +1.21e-4 | +1.21e-4 | +1.30e-5 |
| 71 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 360 | -1.58e-5 | -1.58e-5 | -1.58e-5 | +1.01e-5 |
| 72 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 300 | +5.47e-6 | +5.47e-6 | +5.47e-6 | +9.66e-6 |
| 73 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 290 | -9.64e-5 | -9.64e-5 | -9.64e-5 | -9.51e-7 |
| 74 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 315 | -8.37e-5 | -8.37e-5 | -8.37e-5 | -9.23e-6 |
| 75 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 323 | +8.37e-5 | +8.37e-5 | +8.37e-5 | +6.67e-8 |
| 76 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 356 | +1.09e-5 | +1.09e-5 | +1.09e-5 | +1.15e-6 |
| 78 | 3.00e-1 | 1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 380 | +1.17e-4 | +1.17e-4 | +1.17e-4 | +1.27e-5 |
| 79 | 3.00e-1 | 1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 317 | +2.67e-5 | +2.67e-5 | +2.67e-5 | +1.41e-5 |
| 80 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 297 | -2.14e-4 | -2.14e-4 | -2.14e-4 | -8.73e-6 |
| 81 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 308 | -3.93e-5 | -3.93e-5 | -3.93e-5 | -1.18e-5 |
| 82 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 305 | +5.76e-6 | +5.76e-6 | +5.76e-6 | -1.00e-5 |
| 83 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 319 | -5.67e-6 | -5.67e-6 | -5.67e-6 | -9.60e-6 |
| 85 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 359 | +6.64e-5 | +6.64e-5 | +6.64e-5 | -1.99e-6 |
| 86 | 3.00e-1 | 1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 359 | +1.32e-4 | +1.32e-4 | +1.32e-4 | +1.14e-5 |
| 87 | 3.00e-1 | 1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 274 | -5.26e-5 | -5.26e-5 | -5.26e-5 | +4.98e-6 |
| 88 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 284 | -2.24e-4 | -2.24e-4 | -2.24e-4 | -1.79e-5 |
| 89 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 305 | +1.11e-5 | +1.11e-5 | +1.11e-5 | -1.50e-5 |
| 90 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 293 | +5.22e-5 | +5.22e-5 | +5.22e-5 | -8.31e-6 |
| 91 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 288 | -8.68e-6 | -8.68e-6 | -8.68e-6 | -8.35e-6 |
| 92 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 308 | +2.19e-5 | +2.19e-5 | +2.19e-5 | -5.32e-6 |
| 94 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 356 | +1.85e-5 | +1.85e-5 | +1.85e-5 | -2.94e-6 |
| 95 | 3.00e-1 | 2 | 2.08e-1 | 2.21e-1 | 2.14e-1 | 2.08e-1 | 271 | -2.18e-4 | +1.07e-4 | -5.58e-5 | -1.46e-5 |
| 97 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 333 | -4.70e-5 | -4.70e-5 | -4.70e-5 | -1.79e-5 |
| 98 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 310 | +1.98e-4 | +1.98e-4 | +1.98e-4 | +3.70e-6 |
| 99 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 294 | -9.99e-5 | -9.99e-5 | -9.99e-5 | -6.66e-6 |
| 100 | 3.00e-2 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 309 | +3.14e-5 | +3.14e-5 | +3.14e-5 | -2.85e-6 |
| 101 | 3.00e-2 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 290 | -1.10e-4 | -1.10e-4 | -1.10e-4 | -1.36e-5 |
| 102 | 3.00e-2 | 1 | 2.26e-2 | 2.26e-2 | 2.26e-2 | 2.26e-2 | 275 | -8.05e-3 | -8.05e-3 | -8.05e-3 | -8.17e-4 |
| 103 | 3.00e-2 | 1 | 2.35e-2 | 2.35e-2 | 2.35e-2 | 2.35e-2 | 290 | +1.29e-4 | +1.29e-4 | +1.29e-4 | -7.23e-4 |
| 104 | 3.00e-2 | 1 | 2.52e-2 | 2.52e-2 | 2.52e-2 | 2.52e-2 | 290 | +2.44e-4 | +2.44e-4 | +2.44e-4 | -6.26e-4 |
| 106 | 3.00e-2 | 1 | 2.61e-2 | 2.61e-2 | 2.61e-2 | 2.61e-2 | 370 | +9.95e-5 | +9.95e-5 | +9.95e-5 | -5.54e-4 |
| 107 | 3.00e-2 | 1 | 2.98e-2 | 2.98e-2 | 2.98e-2 | 2.98e-2 | 360 | +3.64e-4 | +3.64e-4 | +3.64e-4 | -4.62e-4 |
| 108 | 3.00e-2 | 1 | 3.02e-2 | 3.02e-2 | 3.02e-2 | 3.02e-2 | 356 | +4.43e-5 | +4.43e-5 | +4.43e-5 | -4.11e-4 |
| 109 | 3.00e-2 | 1 | 3.13e-2 | 3.13e-2 | 3.13e-2 | 3.13e-2 | 270 | +1.26e-4 | +1.26e-4 | +1.26e-4 | -3.57e-4 |
| 110 | 3.00e-2 | 1 | 2.96e-2 | 2.96e-2 | 2.96e-2 | 2.96e-2 | 322 | -1.77e-4 | -1.77e-4 | -1.77e-4 | -3.39e-4 |
| 112 | 3.00e-2 | 1 | 3.24e-2 | 3.24e-2 | 3.24e-2 | 3.24e-2 | 327 | +2.84e-4 | +2.84e-4 | +2.84e-4 | -2.77e-4 |
| 113 | 3.00e-2 | 1 | 3.32e-2 | 3.32e-2 | 3.32e-2 | 3.32e-2 | 313 | +7.56e-5 | +7.56e-5 | +7.56e-5 | -2.42e-4 |
| 114 | 3.00e-2 | 1 | 3.33e-2 | 3.33e-2 | 3.33e-2 | 3.33e-2 | 320 | +6.92e-6 | +6.92e-6 | +6.92e-6 | -2.17e-4 |
| 115 | 3.00e-2 | 1 | 3.40e-2 | 3.40e-2 | 3.40e-2 | 3.40e-2 | 308 | +7.16e-5 | +7.16e-5 | +7.16e-5 | -1.88e-4 |
| 116 | 3.00e-2 | 1 | 3.46e-2 | 3.46e-2 | 3.46e-2 | 3.46e-2 | 264 | +5.89e-5 | +5.89e-5 | +5.89e-5 | -1.63e-4 |
| 117 | 3.00e-2 | 1 | 3.37e-2 | 3.37e-2 | 3.37e-2 | 3.37e-2 | 268 | -9.75e-5 | -9.75e-5 | -9.75e-5 | -1.57e-4 |
| 118 | 3.00e-2 | 1 | 3.52e-2 | 3.52e-2 | 3.52e-2 | 3.52e-2 | 300 | +1.51e-4 | +1.51e-4 | +1.51e-4 | -1.26e-4 |
| 119 | 3.00e-2 | 1 | 3.67e-2 | 3.67e-2 | 3.67e-2 | 3.67e-2 | 286 | +1.43e-4 | +1.43e-4 | +1.43e-4 | -9.91e-5 |
| 120 | 3.00e-2 | 1 | 3.73e-2 | 3.73e-2 | 3.73e-2 | 3.73e-2 | 333 | +4.50e-5 | +4.50e-5 | +4.50e-5 | -8.47e-5 |
| 121 | 3.00e-2 | 1 | 4.07e-2 | 4.07e-2 | 4.07e-2 | 4.07e-2 | 295 | +3.01e-4 | +3.01e-4 | +3.01e-4 | -4.61e-5 |
| 122 | 3.00e-2 | 1 | 3.99e-2 | 3.99e-2 | 3.99e-2 | 3.99e-2 | 278 | -7.45e-5 | -7.45e-5 | -7.45e-5 | -4.90e-5 |
| 124 | 3.00e-2 | 1 | 3.94e-2 | 3.94e-2 | 3.94e-2 | 3.94e-2 | 366 | -2.98e-5 | -2.98e-5 | -2.98e-5 | -4.71e-5 |
| 125 | 3.00e-2 | 1 | 4.40e-2 | 4.40e-2 | 4.40e-2 | 4.40e-2 | 311 | +3.49e-4 | +3.49e-4 | +3.49e-4 | -7.49e-6 |
| 126 | 3.00e-2 | 1 | 4.30e-2 | 4.30e-2 | 4.30e-2 | 4.30e-2 | 290 | -7.44e-5 | -7.44e-5 | -7.44e-5 | -1.42e-5 |
| 127 | 3.00e-2 | 1 | 4.27e-2 | 4.27e-2 | 4.27e-2 | 4.27e-2 | 282 | -3.03e-5 | -3.03e-5 | -3.03e-5 | -1.58e-5 |
| 128 | 3.00e-2 | 1 | 4.25e-2 | 4.25e-2 | 4.25e-2 | 4.25e-2 | 291 | -1.45e-5 | -1.45e-5 | -1.45e-5 | -1.57e-5 |
| 129 | 3.00e-2 | 1 | 4.48e-2 | 4.48e-2 | 4.48e-2 | 4.48e-2 | 312 | +1.74e-4 | +1.74e-4 | +1.74e-4 | +3.25e-6 |
| 130 | 3.00e-2 | 1 | 4.60e-2 | 4.60e-2 | 4.60e-2 | 4.60e-2 | 298 | +8.62e-5 | +8.62e-5 | +8.62e-5 | +1.15e-5 |
| 131 | 3.00e-2 | 1 | 4.60e-2 | 4.60e-2 | 4.60e-2 | 4.60e-2 | 305 | +2.77e-6 | +2.77e-6 | +2.77e-6 | +1.07e-5 |
| 133 | 3.00e-2 | 1 | 4.72e-2 | 4.72e-2 | 4.72e-2 | 4.72e-2 | 356 | +6.99e-5 | +6.99e-5 | +6.99e-5 | +1.66e-5 |
| 134 | 3.00e-2 | 1 | 5.07e-2 | 5.07e-2 | 5.07e-2 | 5.07e-2 | 318 | +2.27e-4 | +2.27e-4 | +2.27e-4 | +3.76e-5 |
| 135 | 3.00e-2 | 1 | 5.04e-2 | 5.04e-2 | 5.04e-2 | 5.04e-2 | 324 | -2.30e-5 | -2.30e-5 | -2.30e-5 | +3.16e-5 |
| 136 | 3.00e-2 | 1 | 5.14e-2 | 5.14e-2 | 5.14e-2 | 5.14e-2 | 298 | +7.08e-5 | +7.08e-5 | +7.08e-5 | +3.55e-5 |
| 137 | 3.00e-2 | 1 | 5.13e-2 | 5.13e-2 | 5.13e-2 | 5.13e-2 | 278 | -1.24e-5 | -1.24e-5 | -1.24e-5 | +3.07e-5 |
| 138 | 3.00e-2 | 1 | 5.08e-2 | 5.08e-2 | 5.08e-2 | 5.08e-2 | 313 | -3.19e-5 | -3.19e-5 | -3.19e-5 | +2.45e-5 |
| 139 | 3.00e-2 | 1 | 5.30e-2 | 5.30e-2 | 5.30e-2 | 5.30e-2 | 325 | +1.34e-4 | +1.34e-4 | +1.34e-4 | +3.54e-5 |
| 141 | 3.00e-2 | 1 | 5.40e-2 | 5.40e-2 | 5.40e-2 | 5.40e-2 | 371 | +5.01e-5 | +5.01e-5 | +5.01e-5 | +3.69e-5 |
| 142 | 3.00e-2 | 1 | 5.92e-2 | 5.92e-2 | 5.92e-2 | 5.92e-2 | 299 | +3.07e-4 | +3.07e-4 | +3.07e-4 | +6.39e-5 |
| 143 | 3.00e-2 | 1 | 5.48e-2 | 5.48e-2 | 5.48e-2 | 5.48e-2 | 299 | -2.58e-4 | -2.58e-4 | -2.58e-4 | +3.17e-5 |
| 144 | 3.00e-2 | 1 | 5.50e-2 | 5.50e-2 | 5.50e-2 | 5.50e-2 | 284 | +1.22e-5 | +1.22e-5 | +1.22e-5 | +2.97e-5 |
| 145 | 3.00e-2 | 1 | 5.49e-2 | 5.49e-2 | 5.49e-2 | 5.49e-2 | 285 | -4.98e-6 | -4.98e-6 | -4.98e-6 | +2.63e-5 |
| 146 | 3.00e-2 | 1 | 5.70e-2 | 5.70e-2 | 5.70e-2 | 5.70e-2 | 318 | +1.19e-4 | +1.19e-4 | +1.19e-4 | +3.55e-5 |
| 147 | 3.00e-2 | 1 | 5.88e-2 | 5.88e-2 | 5.88e-2 | 5.88e-2 | 314 | +9.48e-5 | +9.48e-5 | +9.48e-5 | +4.14e-5 |
| 148 | 3.00e-2 | 1 | 6.10e-2 | 6.10e-2 | 6.10e-2 | 6.10e-2 | 299 | +1.26e-4 | +1.26e-4 | +1.26e-4 | +4.98e-5 |
| 150 | 3.00e-3 | 1 | 5.95e-2 | 5.95e-2 | 5.95e-2 | 5.95e-2 | 346 | -7.10e-5 | -7.10e-5 | -7.10e-5 | +3.78e-5 |
| 151 | 3.00e-3 | 1 | 6.22e-2 | 6.22e-2 | 6.22e-2 | 6.22e-2 | 278 | +1.58e-4 | +1.58e-4 | +1.58e-4 | +4.98e-5 |
| 152 | 3.00e-3 | 1 | 6.01e-3 | 6.01e-3 | 6.01e-3 | 6.01e-3 | 282 | -8.29e-3 | -8.29e-3 | -8.29e-3 | -7.84e-4 |
| 153 | 3.00e-3 | 1 | 5.95e-3 | 5.95e-3 | 5.95e-3 | 5.95e-3 | 278 | -3.08e-5 | -3.08e-5 | -3.08e-5 | -7.09e-4 |
| 154 | 3.00e-3 | 1 | 5.73e-3 | 5.73e-3 | 5.73e-3 | 5.73e-3 | 309 | -1.25e-4 | -1.25e-4 | -1.25e-4 | -6.50e-4 |
| 155 | 3.00e-3 | 1 | 6.10e-3 | 6.10e-3 | 6.10e-3 | 6.10e-3 | 298 | +2.08e-4 | +2.08e-4 | +2.08e-4 | -5.65e-4 |
| 156 | 3.00e-3 | 1 | 5.97e-3 | 5.97e-3 | 5.97e-3 | 5.97e-3 | 318 | -6.47e-5 | -6.47e-5 | -6.47e-5 | -5.15e-4 |
| 157 | 3.00e-3 | 1 | 6.20e-3 | 6.20e-3 | 6.20e-3 | 6.20e-3 | 278 | +1.35e-4 | +1.35e-4 | +1.35e-4 | -4.50e-4 |
| 158 | 3.00e-3 | 1 | 5.96e-3 | 5.96e-3 | 5.96e-3 | 5.96e-3 | 298 | -1.31e-4 | -1.31e-4 | -1.31e-4 | -4.18e-4 |
| 160 | 3.00e-3 | 1 | 6.29e-3 | 6.29e-3 | 6.29e-3 | 6.29e-3 | 366 | +1.48e-4 | +1.48e-4 | +1.48e-4 | -3.61e-4 |
| 161 | 3.00e-3 | 2 | 6.31e-3 | 6.66e-3 | 6.48e-3 | 6.31e-3 | 278 | -1.93e-4 | +1.89e-4 | -2.39e-6 | -2.95e-4 |
| 163 | 3.00e-3 | 1 | 6.02e-3 | 6.02e-3 | 6.02e-3 | 6.02e-3 | 347 | -1.34e-4 | -1.34e-4 | -1.34e-4 | -2.79e-4 |
| 164 | 3.00e-3 | 1 | 6.72e-3 | 6.72e-3 | 6.72e-3 | 6.72e-3 | 299 | +3.66e-4 | +3.66e-4 | +3.66e-4 | -2.14e-4 |
| 165 | 3.00e-3 | 1 | 6.35e-3 | 6.35e-3 | 6.35e-3 | 6.35e-3 | 309 | -1.83e-4 | -1.83e-4 | -1.83e-4 | -2.11e-4 |
| 166 | 3.00e-3 | 1 | 6.59e-3 | 6.59e-3 | 6.59e-3 | 6.59e-3 | 325 | +1.16e-4 | +1.16e-4 | +1.16e-4 | -1.79e-4 |
| 168 | 3.00e-3 | 1 | 6.62e-3 | 6.62e-3 | 6.62e-3 | 6.62e-3 | 366 | +1.15e-5 | +1.15e-5 | +1.15e-5 | -1.60e-4 |
| 169 | 3.00e-3 | 1 | 7.03e-3 | 7.03e-3 | 7.03e-3 | 7.03e-3 | 300 | +2.02e-4 | +2.02e-4 | +2.02e-4 | -1.23e-4 |
| 170 | 3.00e-3 | 1 | 6.49e-3 | 6.49e-3 | 6.49e-3 | 6.49e-3 | 271 | -2.96e-4 | -2.96e-4 | -2.96e-4 | -1.41e-4 |
| 171 | 3.00e-3 | 1 | 6.52e-3 | 6.52e-3 | 6.52e-3 | 6.52e-3 | 300 | +1.78e-5 | +1.78e-5 | +1.78e-5 | -1.25e-4 |
| 172 | 3.00e-3 | 1 | 6.52e-3 | 6.52e-3 | 6.52e-3 | 6.52e-3 | 283 | -7.94e-7 | -7.94e-7 | -7.94e-7 | -1.12e-4 |
| 173 | 3.00e-3 | 1 | 6.45e-3 | 6.45e-3 | 6.45e-3 | 6.45e-3 | 287 | -3.78e-5 | -3.78e-5 | -3.78e-5 | -1.05e-4 |
| 174 | 3.00e-3 | 1 | 6.49e-3 | 6.49e-3 | 6.49e-3 | 6.49e-3 | 289 | +1.79e-5 | +1.79e-5 | +1.79e-5 | -9.27e-5 |
| 175 | 3.00e-3 | 1 | 6.48e-3 | 6.48e-3 | 6.48e-3 | 6.48e-3 | 303 | -4.05e-6 | -4.05e-6 | -4.05e-6 | -8.38e-5 |
| 176 | 3.00e-3 | 1 | 6.79e-3 | 6.79e-3 | 6.79e-3 | 6.79e-3 | 333 | +1.40e-4 | +1.40e-4 | +1.40e-4 | -6.14e-5 |
| 177 | 3.00e-3 | 1 | 7.04e-3 | 7.04e-3 | 7.04e-3 | 7.04e-3 | 302 | +1.19e-4 | +1.19e-4 | +1.19e-4 | -4.34e-5 |
| 179 | 3.00e-3 | 1 | 6.76e-3 | 6.76e-3 | 6.76e-3 | 6.76e-3 | 347 | -1.15e-4 | -1.15e-4 | -1.15e-4 | -5.06e-5 |
| 180 | 3.00e-3 | 1 | 7.36e-3 | 7.36e-3 | 7.36e-3 | 7.36e-3 | 342 | +2.49e-4 | +2.49e-4 | +2.49e-4 | -2.07e-5 |
| 181 | 3.00e-3 | 1 | 7.34e-3 | 7.34e-3 | 7.34e-3 | 7.34e-3 | 297 | -7.36e-6 | -7.36e-6 | -7.36e-6 | -1.93e-5 |
| 182 | 3.00e-3 | 1 | 6.89e-3 | 6.89e-3 | 6.89e-3 | 6.89e-3 | 304 | -2.11e-4 | -2.11e-4 | -2.11e-4 | -3.85e-5 |
| 183 | 3.00e-3 | 1 | 6.77e-3 | 6.77e-3 | 6.77e-3 | 6.77e-3 | 294 | -5.95e-5 | -5.95e-5 | -5.95e-5 | -4.06e-5 |
| 184 | 3.00e-3 | 1 | 7.01e-3 | 7.01e-3 | 7.01e-3 | 7.01e-3 | 284 | +1.24e-4 | +1.24e-4 | +1.24e-4 | -2.42e-5 |
| 185 | 3.00e-3 | 1 | 6.86e-3 | 6.86e-3 | 6.86e-3 | 6.86e-3 | 303 | -7.26e-5 | -7.26e-5 | -7.26e-5 | -2.90e-5 |
| 186 | 3.00e-3 | 1 | 6.78e-3 | 6.78e-3 | 6.78e-3 | 6.78e-3 | 289 | -3.78e-5 | -3.78e-5 | -3.78e-5 | -2.99e-5 |
| 187 | 3.00e-3 | 1 | 6.82e-3 | 6.82e-3 | 6.82e-3 | 6.82e-3 | 328 | +1.64e-5 | +1.64e-5 | +1.64e-5 | -2.53e-5 |
| 188 | 3.00e-3 | 1 | 7.29e-3 | 7.29e-3 | 7.29e-3 | 7.29e-3 | 300 | +2.21e-4 | +2.21e-4 | +2.21e-4 | -6.52e-7 |
| 190 | 3.00e-3 | 1 | 7.08e-3 | 7.08e-3 | 7.08e-3 | 7.08e-3 | 342 | -8.35e-5 | -8.35e-5 | -8.35e-5 | -8.94e-6 |
| 191 | 3.00e-3 | 1 | 7.34e-3 | 7.34e-3 | 7.34e-3 | 7.34e-3 | 291 | +1.25e-4 | +1.25e-4 | +1.25e-4 | +4.50e-6 |
| 192 | 3.00e-3 | 1 | 7.21e-3 | 7.21e-3 | 7.21e-3 | 7.21e-3 | 287 | -6.27e-5 | -6.27e-5 | -6.27e-5 | -2.22e-6 |
| 193 | 3.00e-3 | 1 | 7.18e-3 | 7.18e-3 | 7.18e-3 | 7.18e-3 | 349 | -1.34e-5 | -1.34e-5 | -1.34e-5 | -3.33e-6 |
| 194 | 3.00e-3 | 1 | 7.31e-3 | 7.31e-3 | 7.31e-3 | 7.31e-3 | 324 | +5.59e-5 | +5.59e-5 | +5.59e-5 | +2.59e-6 |
| 196 | 3.00e-3 | 1 | 7.34e-3 | 7.34e-3 | 7.34e-3 | 7.34e-3 | 379 | +1.12e-5 | +1.12e-5 | +1.12e-5 | +3.45e-6 |
| 197 | 3.00e-3 | 1 | 7.95e-3 | 7.95e-3 | 7.95e-3 | 7.95e-3 | 288 | +2.76e-4 | +2.76e-4 | +2.76e-4 | +3.07e-5 |
| 198 | 3.00e-3 | 1 | 7.07e-3 | 7.07e-3 | 7.07e-3 | 7.07e-3 | 320 | -3.65e-4 | -3.65e-4 | -3.65e-4 | -8.87e-6 |
| 199 | 3.00e-3 | 1 | 7.35e-3 | 7.35e-3 | 7.35e-3 | 7.35e-3 | 290 | +1.31e-4 | +1.31e-4 | +1.31e-4 | +5.15e-6 |

