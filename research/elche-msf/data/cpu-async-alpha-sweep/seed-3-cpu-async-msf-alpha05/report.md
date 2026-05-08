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
| cpu-async | 0.061460 | 0.9179 | +0.0054 | 1816.9 | 503 | 83.7 | 100% | 100% | 100% | 15.6 |

## Best Mode per Model

| Model | Best Eval | Mode | Fastest (within 2% of solo-0) | Mode |
|-------|-----------|------|-------------------------------|------|
| resnet-graph | 0.9179 | cpu-async | - | - |

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
| cpu-async | 2.0707 | 0.8338 | 0.6374 | 0.5685 | 0.5329 | 0.5194 | 0.4982 | 0.4807 | 0.4702 | 0.4673 | 0.2131 | 0.1706 | 0.1517 | 0.1372 | 0.1290 | 0.0755 | 0.0704 | 0.0679 | 0.0624 | 0.0615 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | cpu-async | 0 | cuda:0 | 0.4014 | 2.7 |
| resnet-graph | cpu-async | 1 | cuda:1 | 0.3025 | 3.6 |
| resnet-graph | cpu-async | 2 | cuda:2 | 0.2962 | 3.6 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) | GPU2 Peak (MB) | GPU2 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|---------------|---------------|
| resnet-graph | cpu-async | 358 | 357 | 406 | 391 | 396 | 385 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| cpu-async | gpu2 | 1435.7 | 2.8 | epoch-boundary(157) |
| cpu-async | gpu1 | 498.5 | 2.1 | epoch-boundary(55) |
| cpu-async | gpu2 | 1670.0 | 0.9 | epoch-boundary(183) |
| cpu-async | gpu0 | 662.3 | 0.8 | unexplained |
| cpu-async | gpu0 | 1427.6 | 0.7 | cpu-avg |
| cpu-async | gpu1 | 1670.1 | 0.7 | epoch-boundary(183) |
| cpu-async | gpu1 | 662.3 | 0.6 | unexplained |
| cpu-async | gpu2 | 662.3 | 0.6 | unexplained |
| cpu-async | gpu1 | 1427.6 | 0.6 | unexplained |
| cpu-async | gpu2 | 1427.6 | 0.6 | unexplained |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | cpu-async | gpu0 | 0.6s | 0.0s | 1.4s | 0.8s | 2.7s |
| resnet-graph | cpu-async | gpu1 | 4.5s | 0.0s | 0.0s | 1.2s | 6.3s |
| resnet-graph | cpu-async | gpu2 | 4.8s | 0.0s | 0.0s | 1.2s | 6.6s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | cpu-async | 302 | 0 | 503 | 83.7 | 1758/9678 | 503 | 83.7 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | cpu-async | 205.9 | 11.3% |

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
| resnet-graph | cpu-async | 185 | 503 | 0 | 7.53e-3 | -3.84e-5 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | cpu-async | 0 | 503 | 7.14e-2 | 8.01e-2 | 2.27e-3 | 4.89e-1 | 28.0 | -2.28e-4 | 1.34e-3 |
| resnet-graph | cpu-async | 1 | 503 | 7.17e-2 | 7.98e-2 | 2.33e-3 | 4.34e-1 | 35.2 | -2.35e-4 | 1.31e-3 |
| resnet-graph | cpu-async | 2 | 503 | 7.22e-2 | 8.09e-2 | 2.40e-3 | 4.33e-1 | 36.8 | -2.39e-4 | 1.42e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | cpu-async | rank0 ↔ rank1 | +0.9964 |
| resnet-graph | cpu-async | rank0 ↔ rank2 | +0.9962 |
| resnet-graph | cpu-async | rank1 ↔ rank2 | +0.9958 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | cpu-async | 48 (1,2,3,8,15,19,21,43…148,149) | 0 (—) | — | 1,2,3,8,15,19,21,43…148,149 | — |

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
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 161 | +0.137 |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | 255 | +0.086 |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | 83 | +0.120 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | cpu-async | 501 | +0.011 | 184 | +0.235 | +0.392 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | cpu-async | 502 | 3.37e1–7.88e1 | 6.50e1 | 2.39e-3 | 5.31e-3 | 6.44e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 163 | 32–77895 | +4.202e-6 | 0.180 | +4.408e-6 | 0.199 | 88 | +1.573e-6 | 0.052 | 28–947 | +7.052e-4 | 0.522 |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 152 | 927–77895 | +5.047e-6 | 0.289 | +5.155e-6 | 0.304 | 87 | +1.592e-6 | 0.051 | 93–947 | +8.801e-4 | 0.850 |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | 256 | 78303–117099 | +8.422e-6 | 0.124 | +8.506e-6 | 0.134 | 50 | +6.396e-6 | 0.087 | 82–408 | +7.198e-4 | 0.046 |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | 84 | 117307–155591 | +2.765e-5 | 0.375 | +2.839e-5 | 0.396 | 47 | +1.932e-5 | 0.432 | 92–1095 | +1.230e-3 | 0.595 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | +7.052e-4 | r0: +6.939e-4, r1: +7.055e-4, r2: +7.205e-4 | r0: 0.491, r1: 0.522, r2: 0.525 | 1.04× | ✓ |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +8.801e-4 | r0: +8.697e-4, r1: +8.813e-4, r2: +8.923e-4 | r0: 0.842, r1: 0.820, r2: 0.837 | 1.03× | ✓ |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | +7.198e-4 | r0: +5.956e-4, r1: +7.766e-4, r2: +7.797e-4 | r0: 0.033, r1: 0.051, r2: 0.051 | 1.31× | ✓ |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | +1.230e-3 | r0: +1.222e-3, r1: +1.230e-3, r2: +1.239e-3 | r0: 0.600, r1: 0.579, r2: 0.604 | 1.01× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | cpu-async | `█▇████████████████████▅▄▄▄▄▄▄▅▅▅▅▅▄▁▂▂▂▂▂▂▂▂▂▂▂` | `▁▆▇▇██████████████████▆▇▇▇▇▇▇▇▇▇▇▇▅▆▇██████████` |

### Per-Epoch Detail: resnet-graph / cpu-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 11 | 1.15e-1 | 4.89e-1 | 2.20e-1 | 1.56e-1 | 39 | -4.73e-2 | +9.56e-3 | -1.19e-2 | -8.49e-3 |
| 1 | 3.00e-1 | 7 | 9.42e-2 | 2.02e-1 | 1.20e-1 | 9.42e-2 | 44 | -1.14e-2 | +3.15e-3 | -2.38e-3 | -4.52e-3 |
| 2 | 3.00e-1 | 5 | 1.09e-1 | 1.32e-1 | 1.15e-1 | 1.13e-1 | 50 | -4.11e-3 | +4.30e-3 | +2.02e-4 | -2.40e-3 |
| 3 | 3.00e-1 | 8 | 1.12e-1 | 1.41e-1 | 1.18e-1 | 1.13e-1 | 37 | -3.80e-3 | +2.56e-3 | -3.36e-4 | -1.16e-3 |
| 4 | 3.00e-1 | 4 | 1.08e-1 | 1.46e-1 | 1.23e-1 | 1.08e-1 | 35 | -4.30e-3 | +3.67e-3 | -1.00e-3 | -1.17e-3 |
| 5 | 3.00e-1 | 6 | 1.21e-1 | 1.62e-1 | 1.33e-1 | 1.21e-1 | 39 | -4.98e-3 | +4.52e-3 | -3.71e-4 | -8.42e-4 |
| 6 | 3.00e-1 | 6 | 1.10e-1 | 1.65e-1 | 1.25e-1 | 1.10e-1 | 34 | -6.61e-3 | +3.53e-3 | -1.17e-3 | -1.00e-3 |
| 7 | 3.00e-1 | 1 | 1.12e-1 | 1.12e-1 | 1.12e-1 | 1.12e-1 | 35 | +3.76e-4 | +3.76e-4 | +3.76e-4 | -8.64e-4 |
| 8 | 3.00e-1 | 2 | 2.11e-1 | 2.21e-1 | 2.16e-1 | 2.11e-1 | 245 | -1.88e-4 | +2.06e-3 | +9.36e-4 | -5.32e-4 |
| 10 | 3.00e-1 | 1 | 2.36e-1 | 2.36e-1 | 2.36e-1 | 2.36e-1 | 349 | +3.17e-4 | +3.17e-4 | +3.17e-4 | -4.46e-4 |
| 11 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 264 | -3.77e-4 | -3.77e-4 | -3.77e-4 | -4.39e-4 |
| 12 | 3.00e-1 | 1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 332 | +1.19e-4 | +1.19e-4 | +1.19e-4 | -3.83e-4 |
| 13 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 295 | -1.09e-4 | -1.09e-4 | -1.09e-4 | -3.56e-4 |
| 14 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 313 | +4.96e-5 | +4.96e-5 | +4.96e-5 | -3.15e-4 |
| 15 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 320 | +4.14e-6 | +4.14e-6 | +4.14e-6 | -2.83e-4 |
| 16 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 311 | -5.12e-5 | -5.12e-5 | -5.12e-5 | -2.60e-4 |
| 17 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 282 | -1.20e-4 | -1.20e-4 | -1.20e-4 | -2.46e-4 |
| 18 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 284 | +6.07e-5 | +6.07e-5 | +6.07e-5 | -2.15e-4 |
| 19 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 313 | +3.02e-5 | +3.02e-5 | +3.02e-5 | -1.90e-4 |
| 21 | 3.00e-1 | 2 | 2.09e-1 | 2.24e-1 | 2.16e-1 | 2.09e-1 | 264 | -2.60e-4 | +1.29e-4 | -6.54e-5 | -1.69e-4 |
| 23 | 3.00e-1 | 2 | 2.05e-1 | 2.25e-1 | 2.15e-1 | 2.05e-1 | 248 | -3.85e-4 | +2.34e-4 | -7.53e-5 | -1.54e-4 |
| 25 | 3.00e-1 | 2 | 2.06e-1 | 2.20e-1 | 2.13e-1 | 2.06e-1 | 248 | -2.53e-4 | +2.34e-4 | -9.59e-6 | -1.29e-4 |
| 27 | 3.00e-1 | 2 | 2.09e-1 | 2.25e-1 | 2.17e-1 | 2.09e-1 | 248 | -3.00e-4 | +2.67e-4 | -1.62e-5 | -1.10e-4 |
| 29 | 3.00e-1 | 2 | 2.09e-1 | 2.16e-1 | 2.13e-1 | 2.09e-1 | 248 | -1.31e-4 | +1.20e-4 | -5.66e-6 | -9.17e-5 |
| 31 | 3.00e-1 | 2 | 2.09e-1 | 2.21e-1 | 2.15e-1 | 2.09e-1 | 248 | -2.20e-4 | +1.66e-4 | -2.70e-5 | -8.13e-5 |
| 33 | 3.00e-1 | 2 | 2.13e-1 | 2.24e-1 | 2.18e-1 | 2.13e-1 | 250 | -2.02e-4 | +2.15e-4 | +6.29e-6 | -6.68e-5 |
| 35 | 3.00e-1 | 2 | 2.10e-1 | 2.31e-1 | 2.20e-1 | 2.10e-1 | 248 | -3.82e-4 | +2.44e-4 | -6.89e-5 | -7.03e-5 |
| 37 | 3.00e-1 | 2 | 2.07e-1 | 2.24e-1 | 2.15e-1 | 2.07e-1 | 248 | -3.24e-4 | +2.18e-4 | -5.32e-5 | -6.98e-5 |
| 39 | 3.00e-1 | 2 | 2.02e-1 | 2.24e-1 | 2.13e-1 | 2.02e-1 | 230 | -4.50e-4 | +2.78e-4 | -8.60e-5 | -7.65e-5 |
| 41 | 3.00e-1 | 2 | 2.07e-1 | 2.24e-1 | 2.15e-1 | 2.07e-1 | 230 | -3.55e-4 | +3.42e-4 | -6.29e-6 | -6.66e-5 |
| 42 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 251 | +4.26e-6 | +4.26e-6 | +4.26e-6 | -5.95e-5 |
| 43 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 274 | +1.22e-4 | +1.22e-4 | +1.22e-4 | -4.14e-5 |
| 44 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 272 | +1.62e-5 | +1.62e-5 | +1.62e-5 | -3.56e-5 |
| 45 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 290 | +3.56e-5 | +3.56e-5 | +3.56e-5 | -2.85e-5 |
| 46 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 292 | +3.03e-5 | +3.03e-5 | +3.03e-5 | -2.26e-5 |
| 47 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 284 | -6.66e-5 | -6.66e-5 | -6.66e-5 | -2.70e-5 |
| 48 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 286 | +3.59e-5 | +3.59e-5 | +3.59e-5 | -2.07e-5 |
| 49 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 262 | -2.91e-5 | -2.91e-5 | -2.91e-5 | -2.16e-5 |
| 50 | 3.00e-1 | 1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 285 | +1.16e-4 | +1.16e-4 | +1.16e-4 | -7.86e-6 |
| 51 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 225 | -3.69e-4 | -3.69e-4 | -3.69e-4 | -4.40e-5 |
| 52 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 248 | +9.42e-5 | +9.42e-5 | +9.42e-5 | -3.02e-5 |
| 53 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 254 | +5.87e-5 | +5.87e-5 | +5.87e-5 | -2.13e-5 |
| 54 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 245 | -1.15e-4 | -1.15e-4 | -1.15e-4 | -3.07e-5 |
| 55 | 3.00e-1 | 2 | 2.12e-1 | 2.18e-1 | 2.15e-1 | 2.12e-1 | 229 | -1.08e-4 | +1.92e-4 | +4.16e-5 | -1.85e-5 |
| 56 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 253 | +3.11e-5 | +3.11e-5 | +3.11e-5 | -1.35e-5 |
| 57 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 260 | +4.57e-5 | +4.57e-5 | +4.57e-5 | -7.58e-6 |
| 58 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 223 | -1.88e-4 | -1.88e-4 | -1.88e-4 | -2.56e-5 |
| 59 | 3.00e-1 | 2 | 1.91e-1 | 2.05e-1 | 1.98e-1 | 1.91e-1 | 177 | -4.00e-4 | -6.95e-5 | -2.35e-4 | -6.70e-5 |
| 60 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 227 | +3.58e-4 | +3.58e-4 | +3.58e-4 | -2.45e-5 |
| 61 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 220 | -1.14e-4 | -1.14e-4 | -1.14e-4 | -3.34e-5 |
| 62 | 3.00e-1 | 2 | 1.95e-1 | 2.03e-1 | 1.99e-1 | 1.95e-1 | 192 | -1.92e-4 | +1.94e-5 | -8.63e-5 | -4.45e-5 |
| 63 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 217 | +1.41e-4 | +1.41e-4 | +1.41e-4 | -2.60e-5 |
| 64 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 203 | -3.51e-5 | -3.51e-5 | -3.51e-5 | -2.69e-5 |
| 65 | 3.00e-1 | 1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 270 | +3.78e-4 | +3.78e-4 | +3.78e-4 | +1.36e-5 |
| 66 | 3.00e-1 | 2 | 2.03e-1 | 2.16e-1 | 2.09e-1 | 2.03e-1 | 203 | -3.14e-4 | -1.00e-4 | -2.07e-4 | -2.93e-5 |
| 67 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 216 | +5.46e-5 | +5.46e-5 | +5.46e-5 | -2.10e-5 |
| 68 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 226 | +7.08e-5 | +7.08e-5 | +7.08e-5 | -1.18e-5 |
| 69 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 230 | +2.92e-5 | +2.92e-5 | +2.92e-5 | -7.68e-6 |
| 70 | 3.00e-1 | 2 | 1.94e-1 | 2.14e-1 | 2.04e-1 | 1.94e-1 | 179 | -5.67e-4 | +8.78e-5 | -2.40e-4 | -5.50e-5 |
| 71 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 205 | +2.04e-4 | +2.04e-4 | +2.04e-4 | -2.92e-5 |
| 72 | 3.00e-1 | 2 | 1.93e-1 | 2.09e-1 | 2.01e-1 | 1.93e-1 | 169 | -4.75e-4 | +1.55e-4 | -1.60e-4 | -5.72e-5 |
| 73 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 186 | +8.74e-5 | +8.74e-5 | +8.74e-5 | -4.27e-5 |
| 74 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 226 | +2.93e-4 | +2.93e-4 | +2.93e-4 | -9.16e-6 |
| 75 | 3.00e-1 | 2 | 1.87e-1 | 2.03e-1 | 1.95e-1 | 1.87e-1 | 169 | -4.86e-4 | -1.54e-4 | -3.20e-4 | -6.99e-5 |
| 76 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 212 | +4.47e-4 | +4.47e-4 | +4.47e-4 | -1.82e-5 |
| 77 | 3.00e-1 | 2 | 1.92e-1 | 1.99e-1 | 1.96e-1 | 1.92e-1 | 169 | -1.91e-4 | -1.77e-4 | -1.84e-4 | -4.99e-5 |
| 78 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 199 | +2.79e-4 | +2.79e-4 | +2.79e-4 | -1.70e-5 |
| 79 | 3.00e-1 | 2 | 1.95e-1 | 2.02e-1 | 1.99e-1 | 1.95e-1 | 185 | -1.79e-4 | -3.69e-5 | -1.08e-4 | -3.50e-5 |
| 80 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 219 | +2.18e-4 | +2.18e-4 | +2.18e-4 | -9.72e-6 |
| 81 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 220 | +4.78e-5 | +4.78e-5 | +4.78e-5 | -3.97e-6 |
| 82 | 3.00e-1 | 2 | 1.90e-1 | 2.12e-1 | 2.01e-1 | 1.90e-1 | 159 | -7.14e-4 | +1.16e-4 | -2.99e-4 | -6.41e-5 |
| 83 | 3.00e-1 | 1 | 1.91e-1 | 1.91e-1 | 1.91e-1 | 1.91e-1 | 173 | +4.48e-5 | +4.48e-5 | +4.48e-5 | -5.32e-5 |
| 84 | 3.00e-1 | 3 | 1.86e-1 | 1.98e-1 | 1.90e-1 | 1.86e-1 | 146 | -4.40e-4 | +3.41e-4 | -9.99e-5 | -6.84e-5 |
| 85 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 179 | +3.08e-4 | +3.08e-4 | +3.08e-4 | -3.08e-5 |
| 86 | 3.00e-1 | 2 | 1.85e-1 | 2.04e-1 | 1.95e-1 | 1.85e-1 | 154 | -6.43e-4 | +2.14e-4 | -2.15e-4 | -7.00e-5 |
| 87 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 186 | +4.68e-4 | +4.68e-4 | +4.68e-4 | -1.62e-5 |
| 88 | 3.00e-1 | 2 | 1.82e-1 | 1.99e-1 | 1.90e-1 | 1.82e-1 | 146 | -6.11e-4 | -7.96e-5 | -3.45e-4 | -8.14e-5 |
| 89 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 199 | +6.00e-4 | +6.00e-4 | +6.00e-4 | -1.33e-5 |
| 90 | 3.00e-1 | 2 | 1.81e-1 | 1.90e-1 | 1.86e-1 | 1.81e-1 | 137 | -4.55e-4 | -3.60e-4 | -4.07e-4 | -8.76e-5 |
| 91 | 3.00e-1 | 2 | 1.80e-1 | 1.98e-1 | 1.89e-1 | 1.80e-1 | 137 | -6.96e-4 | +4.79e-4 | -1.08e-4 | -9.75e-5 |
| 92 | 3.00e-1 | 2 | 1.86e-1 | 1.92e-1 | 1.89e-1 | 1.86e-1 | 137 | -2.41e-4 | +3.55e-4 | +5.72e-5 | -7.10e-5 |
| 93 | 3.00e-1 | 2 | 1.81e-1 | 1.90e-1 | 1.85e-1 | 1.81e-1 | 144 | -3.63e-4 | +1.34e-4 | -1.14e-4 | -8.18e-5 |
| 94 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 192 | +6.18e-4 | +6.18e-4 | +6.18e-4 | -1.18e-5 |
| 95 | 3.00e-1 | 2 | 1.80e-1 | 2.01e-1 | 1.90e-1 | 1.80e-1 | 136 | -8.12e-4 | -6.79e-5 | -4.40e-4 | -9.69e-5 |
| 96 | 3.00e-1 | 3 | 1.73e-1 | 1.94e-1 | 1.81e-1 | 1.73e-1 | 123 | -7.04e-4 | +4.57e-4 | -1.45e-4 | -1.16e-4 |
| 97 | 3.00e-1 | 1 | 1.86e-1 | 1.86e-1 | 1.86e-1 | 1.86e-1 | 158 | +4.66e-4 | +4.66e-4 | +4.66e-4 | -5.76e-5 |
| 98 | 3.00e-1 | 2 | 1.77e-1 | 1.92e-1 | 1.85e-1 | 1.77e-1 | 126 | -6.62e-4 | +2.05e-4 | -2.28e-4 | -9.43e-5 |
| 99 | 3.00e-1 | 2 | 1.69e-1 | 1.93e-1 | 1.81e-1 | 1.69e-1 | 111 | -1.16e-3 | +4.97e-4 | -3.31e-4 | -1.48e-4 |
| 100 | 3.00e-2 | 3 | 4.69e-2 | 1.91e-1 | 1.10e-1 | 4.69e-2 | 100 | -6.96e-3 | +7.82e-4 | -4.24e-3 | -1.32e-3 |
| 101 | 3.00e-2 | 2 | 2.12e-2 | 2.86e-2 | 2.49e-2 | 2.12e-2 | 112 | -3.76e-3 | -2.65e-3 | -3.21e-3 | -1.68e-3 |
| 102 | 3.00e-2 | 3 | 1.86e-2 | 2.03e-2 | 1.93e-2 | 1.90e-2 | 108 | -7.89e-4 | +2.03e-4 | -3.10e-4 | -1.30e-3 |
| 103 | 3.00e-2 | 1 | 2.16e-2 | 2.16e-2 | 2.16e-2 | 2.16e-2 | 145 | +8.73e-4 | +8.73e-4 | +8.73e-4 | -1.08e-3 |
| 104 | 3.00e-2 | 4 | 1.89e-2 | 2.20e-2 | 1.99e-2 | 1.91e-2 | 96 | -1.12e-3 | +1.23e-4 | -3.13e-4 | -8.14e-4 |
| 105 | 3.00e-2 | 2 | 2.03e-2 | 2.20e-2 | 2.12e-2 | 2.03e-2 | 96 | -8.77e-4 | +1.08e-3 | +1.01e-4 | -6.50e-4 |
| 106 | 3.00e-2 | 2 | 2.04e-2 | 2.26e-2 | 2.15e-2 | 2.04e-2 | 96 | -1.05e-3 | +8.49e-4 | -9.98e-5 | -5.55e-4 |
| 107 | 3.00e-2 | 4 | 2.02e-2 | 2.37e-2 | 2.15e-2 | 2.02e-2 | 88 | -9.70e-4 | +1.04e-3 | -1.60e-4 | -4.34e-4 |
| 108 | 3.00e-2 | 2 | 2.02e-2 | 2.32e-2 | 2.17e-2 | 2.02e-2 | 82 | -1.67e-3 | +1.05e-3 | -3.11e-4 | -4.24e-4 |
| 109 | 3.00e-2 | 2 | 2.10e-2 | 2.30e-2 | 2.20e-2 | 2.10e-2 | 83 | -1.12e-3 | +1.08e-3 | -2.22e-5 | -3.59e-4 |
| 110 | 3.00e-2 | 4 | 2.00e-2 | 2.46e-2 | 2.14e-2 | 2.06e-2 | 77 | -2.36e-3 | +1.23e-3 | -2.63e-4 | -3.26e-4 |
| 111 | 3.00e-2 | 3 | 2.10e-2 | 2.55e-2 | 2.27e-2 | 2.16e-2 | 77 | -2.20e-3 | +1.53e-3 | -1.51e-4 | -3.02e-4 |
| 112 | 3.00e-2 | 3 | 2.06e-2 | 2.43e-2 | 2.19e-2 | 2.06e-2 | 73 | -2.15e-3 | +1.06e-3 | -3.94e-4 | -3.37e-4 |
| 113 | 3.00e-2 | 4 | 2.13e-2 | 2.55e-2 | 2.25e-2 | 2.13e-2 | 69 | -2.25e-3 | +1.77e-3 | -2.04e-4 | -3.05e-4 |
| 114 | 3.00e-2 | 4 | 1.93e-2 | 2.48e-2 | 2.13e-2 | 1.93e-2 | 56 | -2.84e-3 | +1.49e-3 | -7.31e-4 | -4.67e-4 |
| 115 | 3.00e-2 | 4 | 1.98e-2 | 2.60e-2 | 2.14e-2 | 2.02e-2 | 59 | -5.13e-3 | +2.59e-3 | -5.47e-4 | -4.98e-4 |
| 116 | 3.00e-2 | 6 | 1.80e-2 | 2.74e-2 | 2.04e-2 | 1.80e-2 | 48 | -4.82e-3 | +2.50e-3 | -9.02e-4 | -6.82e-4 |
| 117 | 3.00e-2 | 4 | 1.90e-2 | 2.28e-2 | 2.02e-2 | 1.94e-2 | 50 | -3.13e-3 | +2.88e-3 | -1.47e-4 | -5.18e-4 |
| 118 | 3.00e-2 | 5 | 1.87e-2 | 2.47e-2 | 2.04e-2 | 1.87e-2 | 47 | -4.30e-3 | +2.69e-3 | -6.47e-4 | -5.85e-4 |
| 119 | 3.00e-2 | 6 | 1.95e-2 | 2.38e-2 | 2.07e-2 | 1.95e-2 | 47 | -2.75e-3 | +2.75e-3 | -2.45e-4 | -4.51e-4 |
| 120 | 3.00e-2 | 7 | 1.92e-2 | 2.54e-2 | 2.11e-2 | 1.98e-2 | 47 | -3.26e-3 | +3.16e-3 | -2.96e-4 | -3.80e-4 |
| 121 | 3.00e-2 | 4 | 1.92e-2 | 2.63e-2 | 2.16e-2 | 1.92e-2 | 40 | -5.30e-3 | +3.48e-3 | -9.72e-4 | -6.24e-4 |
| 122 | 3.00e-2 | 8 | 1.83e-2 | 2.59e-2 | 2.00e-2 | 1.84e-2 | 37 | -6.00e-3 | +3.59e-3 | -6.35e-4 | -6.12e-4 |
| 123 | 3.00e-2 | 5 | 1.86e-2 | 2.43e-2 | 2.07e-2 | 2.03e-2 | 40 | -4.66e-3 | +3.59e-3 | -1.02e-4 | -4.32e-4 |
| 124 | 3.00e-2 | 9 | 1.77e-2 | 2.61e-2 | 1.95e-2 | 1.78e-2 | 35 | -6.09e-3 | +3.25e-3 | -7.92e-4 | -5.88e-4 |
| 125 | 3.00e-2 | 5 | 1.91e-2 | 2.67e-2 | 2.13e-2 | 1.96e-2 | 35 | -6.66e-3 | +5.30e-3 | -7.06e-4 | -6.56e-4 |
| 126 | 3.00e-2 | 7 | 1.88e-2 | 2.66e-2 | 2.14e-2 | 2.12e-2 | 37 | -6.60e-3 | +4.13e-3 | -3.67e-4 | -4.50e-4 |
| 127 | 3.00e-2 | 7 | 1.81e-2 | 2.71e-2 | 2.20e-2 | 2.39e-2 | 52 | -8.39e-3 | +4.14e-3 | -5.56e-4 | -3.58e-4 |
| 128 | 3.00e-2 | 5 | 2.49e-2 | 3.05e-2 | 2.64e-2 | 2.49e-2 | 48 | -3.14e-3 | +2.95e-3 | -2.07e-4 | -3.24e-4 |
| 129 | 3.00e-2 | 5 | 2.27e-2 | 3.30e-2 | 2.55e-2 | 2.31e-2 | 43 | -6.36e-3 | +2.81e-3 | -1.06e-3 | -6.16e-4 |
| 130 | 3.00e-2 | 7 | 2.09e-2 | 3.06e-2 | 2.30e-2 | 2.12e-2 | 34 | -8.24e-3 | +3.16e-3 | -9.93e-4 | -7.47e-4 |
| 131 | 3.00e-2 | 7 | 2.03e-2 | 2.92e-2 | 2.22e-2 | 2.10e-2 | 32 | -7.95e-3 | +4.32e-3 | -7.67e-4 | -7.00e-4 |
| 132 | 3.00e-2 | 6 | 2.08e-2 | 3.04e-2 | 2.49e-2 | 2.76e-2 | 54 | -8.37e-3 | +4.97e-3 | -1.80e-4 | -3.64e-4 |
| 133 | 3.00e-2 | 6 | 2.60e-2 | 3.54e-2 | 2.84e-2 | 2.60e-2 | 51 | -3.85e-3 | +2.53e-3 | -5.55e-4 | -4.58e-4 |
| 134 | 3.00e-2 | 4 | 2.70e-2 | 3.60e-2 | 3.01e-2 | 2.70e-2 | 47 | -4.00e-3 | +3.40e-3 | -5.47e-4 | -5.28e-4 |
| 135 | 3.00e-2 | 6 | 2.51e-2 | 3.14e-2 | 2.67e-2 | 2.56e-2 | 48 | -3.44e-3 | +1.98e-3 | -4.08e-4 | -4.59e-4 |
| 136 | 3.00e-2 | 5 | 2.61e-2 | 3.51e-2 | 2.89e-2 | 2.61e-2 | 41 | -4.02e-3 | +3.63e-3 | -5.36e-4 | -5.28e-4 |
| 137 | 3.00e-2 | 7 | 2.26e-2 | 3.53e-2 | 2.59e-2 | 2.26e-2 | 32 | -6.71e-3 | +3.68e-3 | -1.16e-3 | -8.53e-4 |
| 138 | 3.00e-2 | 9 | 2.15e-2 | 3.27e-2 | 2.42e-2 | 2.31e-2 | 34 | -7.42e-3 | +4.78e-3 | -6.84e-4 | -6.46e-4 |
| 139 | 3.00e-2 | 4 | 2.84e-2 | 3.56e-2 | 3.10e-2 | 2.84e-2 | 44 | -4.70e-3 | +4.83e-3 | -1.71e-4 | -5.46e-4 |
| 140 | 3.00e-2 | 7 | 2.49e-2 | 3.64e-2 | 2.75e-2 | 2.65e-2 | 41 | -8.35e-3 | +2.86e-3 | -8.56e-4 | -6.07e-4 |
| 141 | 3.00e-2 | 6 | 2.70e-2 | 3.62e-2 | 2.90e-2 | 2.76e-2 | 41 | -5.94e-3 | +3.68e-3 | -4.89e-4 | -5.37e-4 |
| 142 | 3.00e-2 | 8 | 2.59e-2 | 3.73e-2 | 2.90e-2 | 2.69e-2 | 40 | -3.51e-3 | +3.51e-3 | -4.53e-4 | -4.71e-4 |
| 143 | 3.00e-2 | 4 | 2.64e-2 | 3.62e-2 | 3.04e-2 | 2.64e-2 | 36 | -3.80e-3 | +3.79e-3 | -9.71e-4 | -7.12e-4 |
| 144 | 3.00e-2 | 7 | 2.49e-2 | 3.50e-2 | 2.81e-2 | 2.81e-2 | 42 | -7.91e-3 | +4.06e-3 | -4.60e-4 | -5.11e-4 |
| 145 | 3.00e-2 | 6 | 2.56e-2 | 3.80e-2 | 2.91e-2 | 2.67e-2 | 35 | -6.65e-3 | +3.70e-3 | -1.01e-3 | -7.13e-4 |
| 146 | 3.00e-2 | 10 | 2.44e-2 | 3.65e-2 | 2.76e-2 | 2.44e-2 | 35 | -4.47e-3 | +4.08e-3 | -6.95e-4 | -7.04e-4 |
| 147 | 3.00e-2 | 5 | 2.72e-2 | 3.84e-2 | 3.02e-2 | 2.72e-2 | 35 | -7.82e-3 | +5.80e-3 | -8.07e-4 | -7.83e-4 |
| 148 | 3.00e-2 | 7 | 2.49e-2 | 3.79e-2 | 2.92e-2 | 3.14e-2 | 50 | -8.34e-3 | +4.47e-3 | -4.13e-4 | -4.43e-4 |
| 149 | 3.00e-2 | 5 | 3.20e-2 | 4.04e-2 | 3.52e-2 | 3.20e-2 | 45 | -2.51e-3 | +2.87e-3 | -3.96e-4 | -4.70e-4 |
| 150 | 3.00e-3 | 7 | 2.77e-3 | 3.80e-2 | 1.13e-2 | 2.77e-3 | 44 | -1.88e-2 | +1.92e-3 | -8.56e-3 | -4.45e-3 |
| 151 | 3.00e-3 | 5 | 2.88e-3 | 3.73e-3 | 3.14e-3 | 2.96e-3 | 44 | -3.76e-3 | +3.91e-3 | -2.66e-4 | -2.76e-3 |
| 152 | 3.00e-3 | 6 | 2.66e-3 | 4.09e-3 | 3.05e-3 | 2.86e-3 | 40 | -7.24e-3 | +3.79e-3 | -8.12e-4 | -1.80e-3 |
| 153 | 3.00e-3 | 7 | 2.55e-3 | 3.66e-3 | 2.80e-3 | 2.55e-3 | 36 | -5.57e-3 | +3.46e-3 | -8.70e-4 | -1.29e-3 |
| 154 | 3.00e-3 | 7 | 2.42e-3 | 3.65e-3 | 2.78e-3 | 2.47e-3 | 35 | -4.63e-3 | +4.40e-3 | -9.72e-4 | -1.12e-3 |
| 155 | 3.00e-3 | 6 | 3.03e-3 | 3.86e-3 | 3.21e-3 | 3.07e-3 | 49 | -4.03e-3 | +4.77e-3 | +1.14e-5 | -6.28e-4 |
| 156 | 3.00e-3 | 4 | 3.03e-3 | 3.86e-3 | 3.29e-3 | 3.08e-3 | 47 | -4.15e-3 | +2.96e-3 | -4.80e-4 | -5.95e-4 |
| 157 | 3.00e-3 | 2 | 2.99e-3 | 6.63e-3 | 4.81e-3 | 6.63e-3 | 308 | -6.20e-4 | +2.58e-3 | +9.80e-4 | -2.80e-4 |
| 159 | 3.00e-3 | 1 | 7.84e-3 | 7.84e-3 | 7.84e-3 | 7.84e-3 | 364 | +4.61e-4 | +4.61e-4 | +4.61e-4 | -2.06e-4 |
| 160 | 3.00e-3 | 1 | 7.59e-3 | 7.59e-3 | 7.59e-3 | 7.59e-3 | 321 | -9.99e-5 | -9.99e-5 | -9.99e-5 | -1.95e-4 |
| 161 | 3.00e-3 | 1 | 7.76e-3 | 7.76e-3 | 7.76e-3 | 7.76e-3 | 306 | +7.06e-5 | +7.06e-5 | +7.06e-5 | -1.68e-4 |
| 163 | 3.00e-3 | 2 | 7.72e-3 | 8.67e-3 | 8.20e-3 | 7.72e-3 | 310 | -3.73e-4 | +2.80e-4 | -4.65e-5 | -1.49e-4 |
| 165 | 3.00e-3 | 1 | 8.23e-3 | 8.23e-3 | 8.23e-3 | 8.23e-3 | 342 | +1.85e-4 | +1.85e-4 | +1.85e-4 | -1.15e-4 |
| 166 | 3.00e-3 | 1 | 7.92e-3 | 7.92e-3 | 7.92e-3 | 7.92e-3 | 302 | -1.25e-4 | -1.25e-4 | -1.25e-4 | -1.16e-4 |
| 167 | 3.00e-3 | 1 | 7.45e-3 | 7.45e-3 | 7.45e-3 | 7.45e-3 | 268 | -2.30e-4 | -2.30e-4 | -2.30e-4 | -1.28e-4 |
| 168 | 3.00e-3 | 1 | 7.45e-3 | 7.45e-3 | 7.45e-3 | 7.45e-3 | 275 | +2.10e-6 | +2.10e-6 | +2.10e-6 | -1.15e-4 |
| 169 | 3.00e-3 | 1 | 7.67e-3 | 7.67e-3 | 7.67e-3 | 7.67e-3 | 273 | +1.02e-4 | +1.02e-4 | +1.02e-4 | -9.29e-5 |
| 170 | 3.00e-3 | 1 | 7.27e-3 | 7.27e-3 | 7.27e-3 | 7.27e-3 | 252 | -2.11e-4 | -2.11e-4 | -2.11e-4 | -1.05e-4 |
| 171 | 3.00e-3 | 1 | 7.39e-3 | 7.39e-3 | 7.39e-3 | 7.39e-3 | 263 | +6.33e-5 | +6.33e-5 | +6.33e-5 | -8.79e-5 |
| 172 | 3.00e-3 | 1 | 7.76e-3 | 7.76e-3 | 7.76e-3 | 7.76e-3 | 298 | +1.65e-4 | +1.65e-4 | +1.65e-4 | -6.26e-5 |
| 173 | 3.00e-3 | 1 | 7.60e-3 | 7.60e-3 | 7.60e-3 | 7.60e-3 | 281 | -7.50e-5 | -7.50e-5 | -7.50e-5 | -6.39e-5 |
| 174 | 3.00e-3 | 1 | 7.54e-3 | 7.54e-3 | 7.54e-3 | 7.54e-3 | 281 | -2.88e-5 | -2.88e-5 | -2.88e-5 | -6.03e-5 |
| 175 | 3.00e-3 | 1 | 7.74e-3 | 7.74e-3 | 7.74e-3 | 7.74e-3 | 297 | +8.88e-5 | +8.88e-5 | +8.88e-5 | -4.54e-5 |
| 176 | 3.00e-3 | 1 | 8.07e-3 | 8.07e-3 | 8.07e-3 | 8.07e-3 | 301 | +1.38e-4 | +1.38e-4 | +1.38e-4 | -2.70e-5 |
| 177 | 3.00e-3 | 1 | 7.70e-3 | 7.70e-3 | 7.70e-3 | 7.70e-3 | 263 | -1.80e-4 | -1.80e-4 | -1.80e-4 | -4.23e-5 |
| 178 | 3.00e-3 | 1 | 7.94e-3 | 7.94e-3 | 7.94e-3 | 7.94e-3 | 289 | +1.07e-4 | +1.07e-4 | +1.07e-4 | -2.74e-5 |
| 179 | 3.00e-3 | 1 | 8.04e-3 | 8.04e-3 | 8.04e-3 | 8.04e-3 | 294 | +4.32e-5 | +4.32e-5 | +4.32e-5 | -2.03e-5 |
| 180 | 3.00e-3 | 1 | 8.02e-3 | 8.02e-3 | 8.02e-3 | 8.02e-3 | 289 | -1.02e-5 | -1.02e-5 | -1.02e-5 | -1.93e-5 |
| 181 | 3.00e-3 | 1 | 7.96e-3 | 7.96e-3 | 7.96e-3 | 7.96e-3 | 275 | -2.68e-5 | -2.68e-5 | -2.68e-5 | -2.01e-5 |
| 182 | 3.00e-3 | 1 | 7.95e-3 | 7.95e-3 | 7.95e-3 | 7.95e-3 | 275 | -2.80e-6 | -2.80e-6 | -2.80e-6 | -1.83e-5 |
| 183 | 3.00e-3 | 1 | 7.89e-3 | 7.89e-3 | 7.89e-3 | 7.89e-3 | 267 | -2.94e-5 | -2.94e-5 | -2.94e-5 | -1.95e-5 |
| 184 | 3.00e-3 | 1 | 7.94e-3 | 7.94e-3 | 7.94e-3 | 7.94e-3 | 275 | +2.11e-5 | +2.11e-5 | +2.11e-5 | -1.54e-5 |
| 185 | 3.00e-3 | 1 | 7.75e-3 | 7.75e-3 | 7.75e-3 | 7.75e-3 | 254 | -9.42e-5 | -9.42e-5 | -9.42e-5 | -2.33e-5 |
| 186 | 3.00e-3 | 1 | 7.95e-3 | 7.95e-3 | 7.95e-3 | 7.95e-3 | 265 | +9.46e-5 | +9.46e-5 | +9.46e-5 | -1.15e-5 |
| 187 | 3.00e-3 | 1 | 8.06e-3 | 8.06e-3 | 8.06e-3 | 8.06e-3 | 273 | +5.03e-5 | +5.03e-5 | +5.03e-5 | -5.30e-6 |
| 188 | 3.00e-3 | 1 | 7.94e-3 | 7.94e-3 | 7.94e-3 | 7.94e-3 | 296 | -5.02e-5 | -5.02e-5 | -5.02e-5 | -9.79e-6 |
| 189 | 3.00e-3 | 1 | 7.90e-3 | 7.90e-3 | 7.90e-3 | 7.90e-3 | 276 | -1.45e-5 | -1.45e-5 | -1.45e-5 | -1.03e-5 |
| 190 | 3.00e-3 | 1 | 8.04e-3 | 8.04e-3 | 8.04e-3 | 8.04e-3 | 282 | +6.16e-5 | +6.16e-5 | +6.16e-5 | -3.07e-6 |
| 191 | 3.00e-3 | 1 | 8.67e-3 | 8.67e-3 | 8.67e-3 | 8.67e-3 | 312 | +2.40e-4 | +2.40e-4 | +2.40e-4 | +2.12e-5 |
| 192 | 3.00e-3 | 1 | 7.99e-3 | 7.99e-3 | 7.99e-3 | 7.99e-3 | 263 | -3.08e-4 | -3.08e-4 | -3.08e-4 | -1.18e-5 |
| 193 | 3.00e-3 | 1 | 8.53e-3 | 8.53e-3 | 8.53e-3 | 8.53e-3 | 291 | +2.23e-4 | +2.23e-4 | +2.23e-4 | +1.17e-5 |
| 194 | 3.00e-3 | 1 | 7.76e-3 | 7.76e-3 | 7.76e-3 | 7.76e-3 | 249 | -3.77e-4 | -3.77e-4 | -3.77e-4 | -2.71e-5 |
| 195 | 3.00e-3 | 1 | 7.98e-3 | 7.98e-3 | 7.98e-3 | 7.98e-3 | 255 | +1.07e-4 | +1.07e-4 | +1.07e-4 | -1.37e-5 |
| 196 | 3.00e-3 | 1 | 8.27e-3 | 8.27e-3 | 8.27e-3 | 8.27e-3 | 268 | +1.35e-4 | +1.35e-4 | +1.35e-4 | +1.09e-6 |
| 197 | 3.00e-3 | 1 | 7.98e-3 | 7.98e-3 | 7.98e-3 | 7.98e-3 | 260 | -1.37e-4 | -1.37e-4 | -1.37e-4 | -1.27e-5 |
| 198 | 3.00e-3 | 1 | 7.97e-3 | 7.97e-3 | 7.97e-3 | 7.97e-3 | 244 | -4.29e-6 | -4.29e-6 | -4.29e-6 | -1.19e-5 |
| 199 | 3.00e-3 | 1 | 7.53e-3 | 7.53e-3 | 7.53e-3 | 7.53e-3 | 206 | -2.77e-4 | -2.77e-4 | -2.77e-4 | -3.84e-5 |

