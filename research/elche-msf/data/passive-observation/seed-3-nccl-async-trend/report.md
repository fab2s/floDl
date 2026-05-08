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

GPU0/GPU1 = compute utilization % (not load). Idle = total time with <5% utilization.

### resnet-graph

> Published: CIFAR-10 91.25% acc, Graph builder ([He et al. 2015](https://arxiv.org/abs/1512.03385), Table 6)

| Mode | Loss | Eval | vs Ref | Total (s) | Syncs | Avg Sync (ms) | GPU0 | GPU1 | Idle (s) |
|------|------|------|--------|-----------|-------|--------------|------|------|----------|
| nccl-async | 0.047587 | 0.9212 | +0.0087 | 1935.6 | 757 | 39.7 | 100% | 100% | 8.9 |

## Best Mode per Model

| Model | Best Eval | Mode | Fastest (within 2% of solo-0) | Mode |
|-------|-----------|------|-------------------------------|------|
| resnet-graph | 0.9212 | nccl-async | - | - |

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
| nccl-async | 1.9582 | 0.6898 | 0.5287 | 0.4666 | 0.5081 | 0.5030 | 0.4646 | 0.4612 | 0.4608 | 0.4359 | 0.1821 | 0.1408 | 0.1353 | 0.1268 | 0.1227 | 0.0633 | 0.0561 | 0.0537 | 0.0481 | 0.0476 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | nccl-async | 0 | cuda:0 | 0.4007 | 2.6 |
| resnet-graph | nccl-async | 1 | cuda:1 | 0.3030 | 3.3 |
| resnet-graph | nccl-async | 2 | cuda:2 | 0.2963 | 3.3 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|
| resnet-graph | nccl-async | 356 | 355 | 400 | 395 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| nccl-async | gpu1 | 1933.0 | 2.6 | epoch-boundary(199) |
| nccl-async | gpu2 | 1933.1 | 2.5 | epoch-boundary(199) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | nccl-async | gpu0 | 0.0s | 0.0s | 0.0s | 0.0s | 1.0s |
| resnet-graph | nccl-async | gpu1 | 2.6s | 0.0s | 0.0s | 0.0s | 4.4s |
| resnet-graph | nccl-async | gpu2 | 2.5s | 0.0s | 0.0s | 0.0s | 3.5s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | nccl-async | 354 | 0 | 757 | 39.7 | 650/9115 | 0 | 0.0 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | nccl-async | 173.8 | 9.0% |

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
| resnet-graph | nccl-async | 196 | 757 | 0 | 5.62e-3 | -6.28e-6 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | nccl-async | 0 | 757 | 7.45e-2 | 5.01e-2 | 0.00e0 | 4.03e-1 | 52.3 | -1.38e-4 | 3.55e-3 |
| resnet-graph | nccl-async | 1 | 757 | 7.47e-2 | 5.30e-2 | 0.00e0 | 4.30e-1 | 30.3 | -1.59e-4 | 5.44e-3 |
| resnet-graph | nccl-async | 2 | 757 | 7.36e-2 | 5.26e-2 | 0.00e0 | 4.33e-1 | 17.4 | -1.57e-4 | 5.46e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | nccl-async | rank0 ↔ rank1 | +0.9897 |
| resnet-graph | nccl-async | rank0 ↔ rank2 | +0.9897 |
| resnet-graph | nccl-async | rank1 ↔ rank2 | +0.9979 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | nccl-async | 64 (0,1,2,3,4,5,6,7…145,146) | 0 (—) | — | 0,1,2,3,4,5,6,7…145,146 | — |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | nccl-async | 0e0 | 3 | 39 | 39 |
| resnet-graph | nccl-async | 0e0 | 5 | 14 | 14 |
| resnet-graph | nccl-async | 0e0 | 10 | 5 | 5 |
| resnet-graph | nccl-async | 1e-4 | 3 | 19 | 19 |
| resnet-graph | nccl-async | 1e-4 | 5 | 7 | 7 |
| resnet-graph | nccl-async | 1e-4 | 10 | 3 | 3 |
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
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 598 | +0.028 |
| resnet-graph | nccl-async | 3.00e-2 | 100–148 | 95 | +0.100 |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | 59 | +0.008 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | nccl-async | 754 | +0.010 | 195 | +0.179 | +0.103 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | nccl-async | 755 | 3.38e1–8.04e1 | 6.60e1 | 1.74e-3 | 2.86e-3 | 4.02e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 600 | 62–77961 | +1.663e-5 | 0.522 | +1.709e-5 | 0.532 | 100 | +1.690e-5 | 0.596 | 28–821 | +1.873e-3 | 0.710 |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 582 | 884–77961 | +1.685e-5 | 0.555 | +1.730e-5 | 0.564 | 99 | +1.683e-5 | 0.587 | 32–821 | +1.870e-3 | 0.755 |
| resnet-graph | nccl-async | 3.00e-2 | 100–148 | 96 | 78313–116492 | +3.950e-5 | 0.415 | +3.987e-5 | 0.416 | 46 | +4.112e-5 | 0.600 | 68–974 | +1.593e-3 | 0.612 |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | 60 | 117393–155703 | -1.005e-5 | 0.066 | -1.019e-5 | 0.069 | 50 | -5.291e-6 | 0.030 | 464–901 | +1.402e-3 | 0.065 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | +1.873e-3 | r0: +1.815e-3, r1: +1.892e-3, r2: +1.917e-3 | r0: 0.763, r1: 0.675, r2: 0.684 | 1.06× | ✓ |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +1.870e-3 | r0: +1.813e-3, r1: +1.888e-3, r2: +1.915e-3 | r0: 0.818, r1: 0.716, r2: 0.725 | 1.06× | ✓ |
| resnet-graph | nccl-async | 3.00e-2 | 100–148 | +1.593e-3 | r0: +1.587e-3, r1: +1.613e-3, r2: +1.583e-3 | r0: 0.626, r1: 0.605, r2: 0.601 | 1.02× | ✓ |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | +1.402e-3 | r0: +1.455e-3, r1: +1.399e-3, r2: +1.353e-3 | r0: 0.070, r1: 0.064, r2: 0.060 | 1.08× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | nccl-async | `▇▇▇▇▇▇▇▇▇████████████████▄▄▄▄▃▅▅▅▅▅▆▅▁▁▁▁▁▁▁▁▁▁▁▁` | `▃▅▅▆▅▆▆▅▇█▇▇▇▆▆▆▆▆▆▆▆▆▆▆▆▁▅▆▅▅▆▆▆▆▆▆▆▄▅▅▆▆▆▆▆▆▆▆▆` |

### Per-Epoch Detail: resnet-graph / nccl-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 19 | 0.00e0 | 4.33e-1 | 8.15e-2 | 5.44e-2 | 16 | -7.99e-2 | +1.80e-2 | -1.11e-2 | -4.63e-3 |
| 1 | 3.00e-1 | 14 | 4.68e-2 | 1.12e-1 | 6.47e-2 | 6.53e-2 | 17 | -5.45e-2 | +5.38e-2 | +1.31e-3 | -3.28e-4 |
| 2 | 3.00e-1 | 15 | 5.78e-2 | 1.15e-1 | 6.73e-2 | 7.30e-2 | 18 | -4.57e-2 | +4.25e-2 | +4.23e-4 | +4.47e-4 |
| 3 | 3.00e-1 | 15 | 6.40e-2 | 1.24e-1 | 7.20e-2 | 6.49e-2 | 16 | -3.56e-2 | +3.48e-2 | -3.00e-4 | -3.78e-4 |
| 4 | 3.00e-1 | 15 | 6.03e-2 | 1.20e-1 | 7.31e-2 | 7.39e-2 | 17 | -4.57e-2 | +4.26e-2 | +5.45e-4 | +2.48e-4 |
| 5 | 3.00e-1 | 18 | 6.47e-2 | 1.30e-1 | 7.37e-2 | 6.65e-2 | 17 | -3.61e-2 | +3.85e-2 | +7.69e-5 | -3.01e-4 |
| 6 | 3.00e-1 | 14 | 6.06e-2 | 1.32e-1 | 7.09e-2 | 6.39e-2 | 15 | -4.85e-2 | +4.05e-2 | -5.16e-4 | -5.57e-4 |
| 7 | 3.00e-1 | 14 | 6.60e-2 | 1.38e-1 | 7.83e-2 | 6.67e-2 | 17 | -3.49e-2 | +4.09e-2 | +9.84e-6 | -7.16e-4 |
| 8 | 3.00e-1 | 15 | 5.86e-2 | 1.31e-1 | 7.73e-2 | 9.61e-2 | 31 | -5.14e-2 | +4.26e-2 | +3.88e-4 | +3.96e-4 |
| 9 | 3.00e-1 | 9 | 7.54e-2 | 1.38e-1 | 8.85e-2 | 7.54e-2 | 19 | -1.94e-2 | +1.45e-2 | -1.34e-3 | -7.42e-4 |
| 10 | 3.00e-1 | 17 | 6.49e-2 | 1.35e-1 | 7.48e-2 | 6.49e-2 | 19 | -3.32e-2 | +3.21e-2 | -3.39e-4 | -6.94e-4 |
| 11 | 3.00e-1 | 11 | 5.87e-2 | 1.36e-1 | 7.34e-2 | 7.16e-2 | 18 | -3.78e-2 | +3.59e-2 | +2.57e-4 | -1.59e-5 |
| 12 | 3.00e-1 | 15 | 5.95e-2 | 1.28e-1 | 7.37e-2 | 8.33e-2 | 23 | -4.32e-2 | +2.95e-2 | -1.88e-4 | +4.31e-4 |
| 13 | 3.00e-1 | 14 | 6.43e-2 | 1.37e-1 | 7.59e-2 | 7.13e-2 | 17 | -3.90e-2 | +2.83e-2 | -5.76e-4 | -1.72e-4 |
| 14 | 3.00e-1 | 19 | 5.50e-2 | 1.38e-1 | 7.43e-2 | 7.60e-2 | 21 | -4.22e-2 | +4.20e-2 | +1.06e-4 | +1.63e-5 |
| 15 | 3.00e-1 | 11 | 5.59e-2 | 1.30e-1 | 7.11e-2 | 6.78e-2 | 18 | -6.01e-2 | +3.50e-2 | -1.11e-3 | -4.69e-4 |
| 16 | 3.00e-1 | 22 | 5.14e-2 | 1.36e-1 | 6.74e-2 | 6.60e-2 | 16 | -5.20e-2 | +4.13e-2 | -5.69e-4 | -2.53e-4 |
| 17 | 3.00e-1 | 10 | 5.72e-2 | 1.37e-1 | 7.20e-2 | 7.13e-2 | 18 | -6.65e-2 | +4.84e-2 | -4.24e-4 | -1.81e-4 |
| 18 | 3.00e-1 | 15 | 5.96e-2 | 1.38e-1 | 7.11e-2 | 6.69e-2 | 16 | -4.45e-2 | +3.88e-2 | -4.51e-4 | -2.56e-4 |
| 19 | 3.00e-1 | 15 | 5.94e-2 | 1.37e-1 | 7.15e-2 | 6.28e-2 | 18 | -2.86e-2 | +3.58e-2 | -1.62e-4 | -4.08e-4 |
| 20 | 3.00e-1 | 14 | 5.16e-2 | 1.40e-1 | 7.36e-2 | 7.65e-2 | 21 | -6.12e-2 | +3.80e-2 | -6.73e-4 | -1.47e-4 |
| 21 | 3.00e-1 | 12 | 6.87e-2 | 1.42e-1 | 8.10e-2 | 8.05e-2 | 23 | -3.64e-2 | +2.90e-2 | +1.17e-4 | +1.14e-4 |
| 22 | 3.00e-1 | 12 | 6.65e-2 | 1.34e-1 | 8.12e-2 | 7.73e-2 | 18 | -3.11e-2 | +2.75e-2 | -9.84e-5 | -1.12e-4 |
| 23 | 3.00e-1 | 18 | 6.51e-2 | 1.38e-1 | 7.46e-2 | 7.30e-2 | 18 | -3.02e-2 | +4.18e-2 | +7.68e-4 | +2.66e-4 |
| 24 | 3.00e-1 | 11 | 5.91e-2 | 1.39e-1 | 7.24e-2 | 6.64e-2 | 16 | -5.34e-2 | +3.91e-2 | -6.56e-4 | -3.46e-4 |
| 25 | 3.00e-1 | 15 | 5.57e-2 | 1.43e-1 | 7.21e-2 | 7.26e-2 | 19 | -5.36e-2 | +4.91e-2 | +6.94e-5 | +9.94e-5 |
| 26 | 3.00e-1 | 14 | 6.56e-2 | 1.46e-1 | 7.68e-2 | 7.68e-2 | 20 | -4.14e-2 | +3.65e-2 | +8.30e-5 | +1.89e-4 |
| 27 | 3.00e-1 | 14 | 6.03e-2 | 1.35e-1 | 7.32e-2 | 6.89e-2 | 17 | -4.74e-2 | +3.58e-2 | -3.43e-4 | -2.36e-4 |
| 28 | 3.00e-1 | 18 | 5.43e-2 | 1.35e-1 | 6.63e-2 | 6.23e-2 | 16 | -5.52e-2 | +4.32e-2 | -3.87e-4 | -3.03e-4 |
| 29 | 3.00e-1 | 17 | 5.63e-2 | 1.35e-1 | 6.82e-2 | 5.87e-2 | 14 | -4.39e-2 | +4.33e-2 | -5.38e-4 | -8.88e-4 |
| 30 | 3.00e-1 | 11 | 6.05e-2 | 1.41e-1 | 8.39e-2 | 9.20e-2 | 25 | -3.65e-2 | +4.22e-2 | +1.00e-3 | +2.93e-4 |
| 31 | 3.00e-1 | 13 | 6.56e-2 | 1.42e-1 | 8.17e-2 | 6.56e-2 | 16 | -2.04e-2 | +2.07e-2 | -1.17e-3 | -1.05e-3 |
| 32 | 3.00e-1 | 15 | 5.41e-2 | 1.44e-1 | 7.00e-2 | 6.70e-2 | 19 | -6.52e-2 | +5.89e-2 | +3.82e-4 | -1.28e-4 |
| 33 | 3.00e-1 | 14 | 6.06e-2 | 1.46e-1 | 7.66e-2 | 6.94e-2 | 21 | -4.71e-2 | +3.93e-2 | +1.63e-4 | -1.09e-4 |
| 34 | 3.00e-1 | 13 | 6.52e-2 | 1.48e-1 | 7.81e-2 | 8.46e-2 | 17 | -4.05e-2 | +3.46e-2 | +7.32e-4 | +9.12e-4 |
| 35 | 3.00e-1 | 3 | 6.96e-2 | 7.70e-2 | 7.29e-2 | 6.96e-2 | 203 | -8.48e-3 | +3.55e-3 | -1.81e-3 | +2.47e-4 |
| 36 | 3.00e-1 | 1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 242 | +4.84e-3 | +4.84e-3 | +4.84e-3 | +7.06e-4 |
| 37 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 238 | -1.11e-4 | -1.11e-4 | -1.11e-4 | +6.24e-4 |
| 38 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 247 | -1.30e-4 | -1.30e-4 | -1.30e-4 | +5.49e-4 |
| 39 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 278 | -8.58e-5 | -8.58e-5 | -8.58e-5 | +4.85e-4 |
| 40 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 289 | +9.37e-5 | +9.37e-5 | +9.37e-5 | +4.46e-4 |
| 41 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 280 | +2.63e-5 | +2.63e-5 | +2.63e-5 | +4.04e-4 |
| 42 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 260 | -1.02e-4 | -1.02e-4 | -1.02e-4 | +3.54e-4 |
| 43 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 276 | -1.16e-5 | -1.16e-5 | -1.16e-5 | +3.17e-4 |
| 44 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 262 | -2.14e-5 | -2.14e-5 | -2.14e-5 | +2.83e-4 |
| 45 | 3.00e-1 | 2 | 1.97e-1 | 2.07e-1 | 2.02e-1 | 1.97e-1 | 215 | -2.36e-4 | +1.33e-5 | -1.11e-4 | +2.07e-4 |
| 46 | 3.00e-1 | 1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 234 | -8.91e-5 | -8.91e-5 | -8.91e-5 | +1.77e-4 |
| 47 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 240 | +1.84e-4 | +1.84e-4 | +1.84e-4 | +1.78e-4 |
| 48 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 240 | +2.58e-5 | +2.58e-5 | +2.58e-5 | +1.63e-4 |
| 49 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 275 | -5.98e-7 | -5.98e-7 | -5.98e-7 | +1.47e-4 |
| 50 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 252 | +1.64e-4 | +1.64e-4 | +1.64e-4 | +1.48e-4 |
| 51 | 3.00e-1 | 2 | 2.03e-1 | 2.04e-1 | 2.03e-1 | 2.04e-1 | 215 | -1.63e-4 | +1.03e-5 | -7.65e-5 | +1.06e-4 |
| 52 | 3.00e-1 | 1 | 1.91e-1 | 1.91e-1 | 1.91e-1 | 1.91e-1 | 246 | -2.68e-4 | -2.68e-4 | -2.68e-4 | +6.90e-5 |
| 53 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 238 | +3.24e-4 | +3.24e-4 | +3.24e-4 | +9.45e-5 |
| 54 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 237 | -1.34e-4 | -1.34e-4 | -1.34e-4 | +7.16e-5 |
| 55 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 253 | -1.00e-5 | -1.00e-5 | -1.00e-5 | +6.34e-5 |
| 56 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 251 | +1.67e-4 | +1.67e-4 | +1.67e-4 | +7.38e-5 |
| 57 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 235 | -8.92e-5 | -8.92e-5 | -8.92e-5 | +5.75e-5 |
| 58 | 3.00e-1 | 2 | 2.02e-1 | 2.06e-1 | 2.04e-1 | 2.06e-1 | 197 | -2.01e-5 | +9.41e-5 | +3.70e-5 | +5.42e-5 |
| 59 | 3.00e-1 | 1 | 1.91e-1 | 1.91e-1 | 1.91e-1 | 1.91e-1 | 218 | -3.48e-4 | -3.48e-4 | -3.48e-4 | +1.39e-5 |
| 60 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 227 | +1.16e-4 | +1.16e-4 | +1.16e-4 | +2.41e-5 |
| 61 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 225 | +1.23e-4 | +1.23e-4 | +1.23e-4 | +3.40e-5 |
| 62 | 3.00e-1 | 2 | 1.99e-1 | 2.01e-1 | 2.00e-1 | 1.99e-1 | 197 | -3.29e-5 | -1.83e-5 | -2.56e-5 | +2.26e-5 |
| 63 | 3.00e-1 | 1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 220 | -2.16e-4 | -2.16e-4 | -2.16e-4 | -1.25e-6 |
| 64 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 216 | +2.83e-4 | +2.83e-4 | +2.83e-4 | +2.72e-5 |
| 65 | 3.00e-1 | 2 | 1.97e-1 | 2.00e-1 | 1.98e-1 | 1.97e-1 | 197 | -8.05e-5 | -4.80e-5 | -6.43e-5 | +9.68e-6 |
| 66 | 3.00e-1 | 1 | 1.89e-1 | 1.89e-1 | 1.89e-1 | 1.89e-1 | 229 | -1.87e-4 | -1.87e-4 | -1.87e-4 | -9.99e-6 |
| 67 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 223 | +3.05e-4 | +3.05e-4 | +3.05e-4 | +2.15e-5 |
| 68 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 226 | -7.71e-5 | -7.71e-5 | -7.71e-5 | +1.16e-5 |
| 69 | 3.00e-1 | 2 | 1.98e-1 | 2.05e-1 | 2.02e-1 | 1.98e-1 | 197 | -1.71e-4 | +1.52e-4 | -9.71e-6 | +5.96e-6 |
| 70 | 3.00e-1 | 1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 223 | -1.23e-4 | -1.23e-4 | -1.23e-4 | -6.94e-6 |
| 71 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 260 | +1.35e-4 | +1.35e-4 | +1.35e-4 | +7.22e-6 |
| 72 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 243 | +1.84e-4 | +1.84e-4 | +1.84e-4 | +2.49e-5 |
| 73 | 3.00e-1 | 2 | 2.01e-1 | 2.07e-1 | 2.04e-1 | 2.07e-1 | 179 | -1.55e-4 | +1.55e-4 | +1.28e-7 | +2.18e-5 |
| 74 | 3.00e-1 | 1 | 1.85e-1 | 1.85e-1 | 1.85e-1 | 1.85e-1 | 210 | -5.25e-4 | -5.25e-4 | -5.25e-4 | -3.29e-5 |
| 75 | 3.00e-1 | 1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 217 | +2.41e-4 | +2.41e-4 | +2.41e-4 | -5.59e-6 |
| 76 | 3.00e-1 | 2 | 1.94e-1 | 1.95e-1 | 1.95e-1 | 1.94e-1 | 156 | -3.61e-5 | +6.90e-7 | -1.77e-5 | -8.08e-6 |
| 77 | 3.00e-1 | 2 | 1.75e-1 | 1.94e-1 | 1.84e-1 | 1.94e-1 | 156 | -5.59e-4 | +6.89e-4 | +6.53e-5 | +1.21e-5 |
| 78 | 3.00e-1 | 1 | 1.77e-1 | 1.77e-1 | 1.77e-1 | 1.77e-1 | 185 | -5.11e-4 | -5.11e-4 | -5.11e-4 | -4.02e-5 |
| 79 | 3.00e-1 | 2 | 1.91e-1 | 2.01e-1 | 1.96e-1 | 2.01e-1 | 165 | +3.06e-4 | +3.66e-4 | +3.36e-4 | +3.10e-5 |
| 80 | 3.00e-1 | 1 | 1.83e-1 | 1.83e-1 | 1.83e-1 | 1.83e-1 | 190 | -4.97e-4 | -4.97e-4 | -4.97e-4 | -2.18e-5 |
| 81 | 3.00e-1 | 2 | 1.92e-1 | 1.96e-1 | 1.94e-1 | 1.96e-1 | 156 | +1.37e-4 | +2.28e-4 | +1.83e-4 | +1.66e-5 |
| 82 | 3.00e-1 | 1 | 1.75e-1 | 1.75e-1 | 1.75e-1 | 1.75e-1 | 183 | -6.43e-4 | -6.43e-4 | -6.43e-4 | -4.93e-5 |
| 83 | 3.00e-1 | 2 | 1.88e-1 | 1.96e-1 | 1.92e-1 | 1.96e-1 | 156 | +2.89e-4 | +3.71e-4 | +3.30e-4 | +2.23e-5 |
| 84 | 3.00e-1 | 1 | 1.79e-1 | 1.79e-1 | 1.79e-1 | 1.79e-1 | 206 | -4.57e-4 | -4.57e-4 | -4.57e-4 | -2.56e-5 |
| 85 | 3.00e-1 | 2 | 1.94e-1 | 2.01e-1 | 1.97e-1 | 2.01e-1 | 156 | +2.13e-4 | +4.03e-4 | +3.08e-4 | +3.68e-5 |
| 86 | 3.00e-1 | 2 | 1.81e-1 | 1.92e-1 | 1.87e-1 | 1.92e-1 | 147 | -5.53e-4 | +4.11e-4 | -7.12e-5 | +2.11e-5 |
| 87 | 3.00e-1 | 1 | 1.75e-1 | 1.75e-1 | 1.75e-1 | 1.75e-1 | 177 | -5.34e-4 | -5.34e-4 | -5.34e-4 | -3.44e-5 |
| 88 | 3.00e-1 | 2 | 1.83e-1 | 1.89e-1 | 1.86e-1 | 1.83e-1 | 137 | -2.26e-4 | +4.34e-4 | +1.04e-4 | -1.14e-5 |
| 89 | 3.00e-1 | 2 | 1.68e-1 | 1.80e-1 | 1.74e-1 | 1.80e-1 | 137 | -5.32e-4 | +4.81e-4 | -2.54e-5 | -8.97e-6 |
| 90 | 3.00e-1 | 2 | 1.69e-1 | 1.91e-1 | 1.80e-1 | 1.91e-1 | 137 | -3.40e-4 | +8.86e-4 | +2.73e-4 | +5.07e-5 |
| 91 | 3.00e-1 | 1 | 1.71e-1 | 1.71e-1 | 1.71e-1 | 1.71e-1 | 181 | -6.19e-4 | -6.19e-4 | -6.19e-4 | -1.63e-5 |
| 92 | 3.00e-1 | 2 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 130 | -3.43e-6 | +6.12e-4 | +3.04e-4 | +4.15e-5 |
| 93 | 3.00e-1 | 2 | 1.70e-1 | 1.85e-1 | 1.77e-1 | 1.85e-1 | 123 | -7.90e-4 | +7.11e-4 | -3.95e-5 | +3.36e-5 |
| 94 | 3.00e-1 | 3 | 1.65e-1 | 1.86e-1 | 1.73e-1 | 1.67e-1 | 126 | -8.33e-4 | +9.60e-4 | -1.75e-4 | -2.52e-5 |
| 95 | 3.00e-1 | 1 | 1.63e-1 | 1.63e-1 | 1.63e-1 | 1.63e-1 | 165 | -1.50e-4 | -1.50e-4 | -1.50e-4 | -3.77e-5 |
| 96 | 3.00e-1 | 3 | 1.57e-1 | 1.82e-1 | 1.71e-1 | 1.57e-1 | 114 | -9.46e-4 | +7.56e-4 | -1.98e-4 | -9.71e-5 |
| 97 | 3.00e-1 | 1 | 1.62e-1 | 1.62e-1 | 1.62e-1 | 1.62e-1 | 170 | +1.87e-4 | +1.87e-4 | +1.87e-4 | -6.87e-5 |
| 98 | 3.00e-1 | 3 | 1.60e-1 | 1.89e-1 | 1.79e-1 | 1.60e-1 | 112 | -1.42e-3 | +8.71e-4 | -2.01e-4 | -1.26e-4 |
| 99 | 3.00e-1 | 2 | 1.59e-1 | 1.74e-1 | 1.67e-1 | 1.74e-1 | 99 | -4.66e-5 | +9.17e-4 | +4.35e-4 | -1.49e-5 |
| 100 | 3.00e-2 | 3 | 1.45e-2 | 1.69e-1 | 1.13e-1 | 1.45e-2 | 95 | -2.59e-2 | +9.13e-4 | -8.62e-3 | -2.59e-3 |
| 101 | 3.00e-2 | 2 | 1.39e-2 | 1.57e-2 | 1.48e-2 | 1.57e-2 | 102 | -3.29e-4 | +1.18e-3 | +4.26e-4 | -2.01e-3 |
| 102 | 3.00e-2 | 3 | 1.48e-2 | 1.76e-2 | 1.61e-2 | 1.59e-2 | 120 | -8.26e-4 | +1.66e-3 | +1.42e-4 | -1.43e-3 |
| 103 | 3.00e-2 | 2 | 1.78e-2 | 1.95e-2 | 1.86e-2 | 1.95e-2 | 93 | +7.41e-4 | +9.81e-4 | +8.61e-4 | -9.93e-4 |
| 104 | 3.00e-2 | 2 | 1.58e-2 | 2.04e-2 | 1.81e-2 | 2.04e-2 | 95 | -1.35e-3 | +2.67e-3 | +6.60e-4 | -6.59e-4 |
| 105 | 3.00e-2 | 4 | 1.54e-2 | 1.83e-2 | 1.65e-2 | 1.54e-2 | 91 | -2.03e-3 | +1.03e-3 | -6.96e-4 | -6.63e-4 |
| 106 | 3.00e-2 | 2 | 1.73e-2 | 1.95e-2 | 1.84e-2 | 1.95e-2 | 90 | +9.82e-4 | +1.29e-3 | +1.14e-3 | -3.19e-4 |
| 107 | 3.00e-2 | 3 | 1.73e-2 | 2.00e-2 | 1.84e-2 | 1.80e-2 | 90 | -1.15e-3 | +1.61e-3 | -1.73e-4 | -2.82e-4 |
| 108 | 3.00e-2 | 4 | 1.75e-2 | 2.09e-2 | 1.85e-2 | 1.81e-2 | 80 | -2.15e-3 | +2.06e-3 | +1.39e-5 | -1.91e-4 |
| 109 | 3.00e-2 | 3 | 1.71e-2 | 2.05e-2 | 1.84e-2 | 1.71e-2 | 71 | -2.59e-3 | +2.15e-3 | -2.25e-4 | -2.24e-4 |
| 110 | 3.00e-2 | 4 | 1.65e-2 | 2.07e-2 | 1.80e-2 | 1.66e-2 | 72 | -1.85e-3 | +2.95e-3 | -1.11e-4 | -2.20e-4 |
| 111 | 3.00e-2 | 3 | 1.71e-2 | 2.18e-2 | 1.88e-2 | 1.71e-2 | 62 | -3.91e-3 | +3.25e-3 | -8.55e-5 | -2.26e-4 |
| 112 | 3.00e-2 | 4 | 1.52e-2 | 2.16e-2 | 1.72e-2 | 1.52e-2 | 50 | -6.84e-3 | +5.15e-3 | -5.25e-4 | -3.79e-4 |
| 113 | 3.00e-2 | 5 | 1.52e-2 | 1.87e-2 | 1.65e-2 | 1.70e-2 | 54 | -3.91e-3 | +3.37e-3 | +3.74e-4 | -7.65e-5 |
| 114 | 3.00e-2 | 7 | 1.42e-2 | 2.14e-2 | 1.64e-2 | 1.42e-2 | 42 | -5.90e-3 | +5.82e-3 | -4.63e-4 | -3.63e-4 |
| 115 | 3.00e-2 | 5 | 1.27e-2 | 1.94e-2 | 1.48e-2 | 1.27e-2 | 33 | -9.21e-3 | +7.23e-3 | -8.55e-4 | -6.65e-4 |
| 116 | 3.00e-2 | 9 | 1.12e-2 | 1.97e-2 | 1.30e-2 | 1.19e-2 | 26 | -1.25e-2 | +1.29e-2 | -3.98e-4 | -5.33e-4 |
| 117 | 3.00e-2 | 1 | 1.16e-2 | 1.16e-2 | 1.16e-2 | 1.16e-2 | 26 | -7.89e-4 | -7.89e-4 | -7.89e-4 | -5.59e-4 |
| 118 | 3.00e-2 | 1 | 1.18e-2 | 1.18e-2 | 1.18e-2 | 1.18e-2 | 297 | +3.96e-5 | +3.96e-5 | +3.96e-5 | -4.99e-4 |
| 119 | 3.00e-2 | 1 | 4.02e-2 | 4.02e-2 | 4.02e-2 | 4.02e-2 | 316 | +3.89e-3 | +3.89e-3 | +3.89e-3 | -6.02e-5 |
| 120 | 3.00e-2 | 1 | 4.18e-2 | 4.18e-2 | 4.18e-2 | 4.18e-2 | 301 | +1.24e-4 | +1.24e-4 | +1.24e-4 | -4.18e-5 |
| 121 | 3.00e-2 | 1 | 4.14e-2 | 4.14e-2 | 4.14e-2 | 4.14e-2 | 313 | -3.07e-5 | -3.07e-5 | -3.07e-5 | -4.07e-5 |
| 122 | 3.00e-2 | 1 | 4.19e-2 | 4.19e-2 | 4.19e-2 | 4.19e-2 | 301 | +4.24e-5 | +4.24e-5 | +4.24e-5 | -3.24e-5 |
| 123 | 3.00e-2 | 1 | 4.32e-2 | 4.32e-2 | 4.32e-2 | 4.32e-2 | 353 | +8.95e-5 | +8.95e-5 | +8.95e-5 | -2.02e-5 |
| 125 | 3.00e-2 | 1 | 4.61e-2 | 4.61e-2 | 4.61e-2 | 4.61e-2 | 361 | +1.79e-4 | +1.79e-4 | +1.79e-4 | -3.20e-7 |
| 126 | 3.00e-2 | 1 | 4.77e-2 | 4.77e-2 | 4.77e-2 | 4.77e-2 | 313 | +1.08e-4 | +1.08e-4 | +1.08e-4 | +1.05e-5 |
| 127 | 3.00e-2 | 1 | 4.63e-2 | 4.63e-2 | 4.63e-2 | 4.63e-2 | 350 | -8.33e-5 | -8.33e-5 | -8.33e-5 | +1.13e-6 |
| 128 | 3.00e-2 | 1 | 4.95e-2 | 4.95e-2 | 4.95e-2 | 4.95e-2 | 343 | +1.90e-4 | +1.90e-4 | +1.90e-4 | +2.00e-5 |
| 129 | 3.00e-2 | 1 | 5.01e-2 | 5.01e-2 | 5.01e-2 | 5.01e-2 | 282 | +4.82e-5 | +4.82e-5 | +4.82e-5 | +2.29e-5 |
| 130 | 3.00e-2 | 1 | 4.75e-2 | 4.75e-2 | 4.75e-2 | 4.75e-2 | 266 | -2.00e-4 | -2.00e-4 | -2.00e-4 | +5.71e-7 |
| 131 | 3.00e-2 | 1 | 4.73e-2 | 4.73e-2 | 4.73e-2 | 4.73e-2 | 302 | -1.58e-5 | -1.58e-5 | -1.58e-5 | -1.07e-6 |
| 132 | 3.00e-2 | 1 | 4.99e-2 | 4.99e-2 | 4.99e-2 | 4.99e-2 | 300 | +1.80e-4 | +1.80e-4 | +1.80e-4 | +1.71e-5 |
| 133 | 3.00e-2 | 1 | 5.07e-2 | 5.07e-2 | 5.07e-2 | 5.07e-2 | 316 | +4.83e-5 | +4.83e-5 | +4.83e-5 | +2.02e-5 |
| 134 | 3.00e-2 | 1 | 5.29e-2 | 5.29e-2 | 5.29e-2 | 5.29e-2 | 323 | +1.32e-4 | +1.32e-4 | +1.32e-4 | +3.13e-5 |
| 136 | 3.00e-2 | 1 | 5.45e-2 | 5.45e-2 | 5.45e-2 | 5.45e-2 | 338 | +8.85e-5 | +8.85e-5 | +8.85e-5 | +3.70e-5 |
| 137 | 3.00e-2 | 1 | 5.69e-2 | 5.69e-2 | 5.69e-2 | 5.69e-2 | 275 | +1.53e-4 | +1.53e-4 | +1.53e-4 | +4.87e-5 |
| 138 | 3.00e-2 | 1 | 5.27e-2 | 5.27e-2 | 5.27e-2 | 5.27e-2 | 292 | -2.62e-4 | -2.62e-4 | -2.62e-4 | +1.76e-5 |
| 139 | 3.00e-2 | 1 | 5.62e-2 | 5.62e-2 | 5.62e-2 | 5.62e-2 | 270 | +2.38e-4 | +2.38e-4 | +2.38e-4 | +3.96e-5 |
| 140 | 3.00e-2 | 1 | 5.37e-2 | 5.37e-2 | 5.37e-2 | 5.37e-2 | 305 | -1.47e-4 | -1.47e-4 | -1.47e-4 | +2.09e-5 |
| 141 | 3.00e-2 | 1 | 5.74e-2 | 5.74e-2 | 5.74e-2 | 5.74e-2 | 254 | +2.59e-4 | +2.59e-4 | +2.59e-4 | +4.47e-5 |
| 142 | 3.00e-2 | 1 | 5.35e-2 | 5.35e-2 | 5.35e-2 | 5.35e-2 | 254 | -2.75e-4 | -2.75e-4 | -2.75e-4 | +1.28e-5 |
| 143 | 3.00e-2 | 2 | 5.44e-2 | 5.51e-2 | 5.47e-2 | 5.51e-2 | 244 | +5.37e-5 | +6.56e-5 | +5.97e-5 | +2.16e-5 |
| 145 | 3.00e-2 | 1 | 5.57e-2 | 5.57e-2 | 5.57e-2 | 5.57e-2 | 319 | +3.77e-5 | +3.77e-5 | +3.77e-5 | +2.32e-5 |
| 146 | 3.00e-2 | 1 | 6.18e-2 | 6.18e-2 | 6.18e-2 | 6.18e-2 | 280 | +3.70e-4 | +3.70e-4 | +3.70e-4 | +5.79e-5 |
| 147 | 3.00e-2 | 1 | 5.95e-2 | 5.95e-2 | 5.95e-2 | 5.95e-2 | 309 | -1.22e-4 | -1.22e-4 | -1.22e-4 | +3.99e-5 |
| 148 | 3.00e-2 | 2 | 5.92e-2 | 6.19e-2 | 6.06e-2 | 5.92e-2 | 223 | -1.96e-4 | +1.46e-4 | -2.52e-5 | +2.58e-5 |
| 150 | 3.00e-3 | 2 | 5.56e-2 | 6.82e-2 | 6.19e-2 | 6.82e-2 | 211 | -1.78e-4 | +9.69e-4 | +3.95e-4 | +1.02e-4 |
| 151 | 3.00e-3 | 1 | 5.14e-3 | 5.14e-3 | 5.14e-3 | 5.14e-3 | 255 | -1.01e-2 | -1.01e-2 | -1.01e-2 | -9.22e-4 |
| 152 | 3.00e-3 | 1 | 5.38e-3 | 5.38e-3 | 5.38e-3 | 5.38e-3 | 241 | +1.91e-4 | +1.91e-4 | +1.91e-4 | -8.11e-4 |
| 153 | 3.00e-3 | 1 | 5.38e-3 | 5.38e-3 | 5.38e-3 | 5.38e-3 | 240 | -4.08e-6 | -4.08e-6 | -4.08e-6 | -7.30e-4 |
| 154 | 3.00e-3 | 1 | 5.18e-3 | 5.18e-3 | 5.18e-3 | 5.18e-3 | 236 | -1.61e-4 | -1.61e-4 | -1.61e-4 | -6.73e-4 |
| 155 | 3.00e-3 | 1 | 5.42e-3 | 5.42e-3 | 5.42e-3 | 5.42e-3 | 267 | +1.72e-4 | +1.72e-4 | +1.72e-4 | -5.89e-4 |
| 156 | 3.00e-3 | 1 | 5.45e-3 | 5.45e-3 | 5.45e-3 | 5.45e-3 | 245 | +1.86e-5 | +1.86e-5 | +1.86e-5 | -5.28e-4 |
| 157 | 3.00e-3 | 1 | 5.22e-3 | 5.22e-3 | 5.22e-3 | 5.22e-3 | 237 | -1.81e-4 | -1.81e-4 | -1.81e-4 | -4.93e-4 |
| 158 | 3.00e-3 | 2 | 5.31e-3 | 5.35e-3 | 5.33e-3 | 5.31e-3 | 228 | -3.65e-5 | +1.04e-4 | +3.35e-5 | -3.94e-4 |
| 159 | 3.00e-3 | 1 | 5.03e-3 | 5.03e-3 | 5.03e-3 | 5.03e-3 | 258 | -2.07e-4 | -2.07e-4 | -2.07e-4 | -3.75e-4 |
| 160 | 3.00e-3 | 1 | 5.54e-3 | 5.54e-3 | 5.54e-3 | 5.54e-3 | 255 | +3.75e-4 | +3.75e-4 | +3.75e-4 | -3.00e-4 |
| 161 | 3.00e-3 | 1 | 5.55e-3 | 5.55e-3 | 5.55e-3 | 5.55e-3 | 266 | +9.49e-6 | +9.49e-6 | +9.49e-6 | -2.69e-4 |
| 162 | 3.00e-3 | 1 | 5.55e-3 | 5.55e-3 | 5.55e-3 | 5.55e-3 | 267 | -2.86e-6 | -2.86e-6 | -2.86e-6 | -2.43e-4 |
| 163 | 3.00e-3 | 1 | 5.61e-3 | 5.61e-3 | 5.61e-3 | 5.61e-3 | 279 | +4.20e-5 | +4.20e-5 | +4.20e-5 | -2.14e-4 |
| 164 | 3.00e-3 | 1 | 5.87e-3 | 5.87e-3 | 5.87e-3 | 5.87e-3 | 253 | +1.74e-4 | +1.74e-4 | +1.74e-4 | -1.75e-4 |
| 165 | 3.00e-3 | 1 | 5.58e-3 | 5.58e-3 | 5.58e-3 | 5.58e-3 | 256 | -1.99e-4 | -1.99e-4 | -1.99e-4 | -1.78e-4 |
| 166 | 3.00e-3 | 1 | 5.27e-3 | 5.27e-3 | 5.27e-3 | 5.27e-3 | 236 | -2.35e-4 | -2.35e-4 | -2.35e-4 | -1.83e-4 |
| 167 | 3.00e-3 | 2 | 5.29e-3 | 5.56e-3 | 5.43e-3 | 5.29e-3 | 194 | -2.51e-4 | +2.36e-4 | -7.78e-6 | -1.52e-4 |
| 168 | 3.00e-3 | 1 | 4.80e-3 | 4.80e-3 | 4.80e-3 | 4.80e-3 | 241 | -4.09e-4 | -4.09e-4 | -4.09e-4 | -1.78e-4 |
| 169 | 3.00e-3 | 1 | 5.51e-3 | 5.51e-3 | 5.51e-3 | 5.51e-3 | 248 | +5.59e-4 | +5.59e-4 | +5.59e-4 | -1.04e-4 |
| 170 | 3.00e-3 | 1 | 5.36e-3 | 5.36e-3 | 5.36e-3 | 5.36e-3 | 235 | -1.15e-4 | -1.15e-4 | -1.15e-4 | -1.05e-4 |
| 171 | 3.00e-3 | 2 | 5.46e-3 | 5.57e-3 | 5.51e-3 | 5.46e-3 | 200 | -9.99e-5 | +1.62e-4 | +3.10e-5 | -8.08e-5 |
| 172 | 3.00e-3 | 1 | 5.05e-3 | 5.05e-3 | 5.05e-3 | 5.05e-3 | 222 | -3.46e-4 | -3.46e-4 | -3.46e-4 | -1.07e-4 |
| 173 | 3.00e-3 | 1 | 5.39e-3 | 5.39e-3 | 5.39e-3 | 5.39e-3 | 250 | +2.54e-4 | +2.54e-4 | +2.54e-4 | -7.13e-5 |
| 174 | 3.00e-3 | 1 | 5.72e-3 | 5.72e-3 | 5.72e-3 | 5.72e-3 | 216 | +2.76e-4 | +2.76e-4 | +2.76e-4 | -3.65e-5 |
| 175 | 3.00e-3 | 2 | 5.21e-3 | 5.79e-3 | 5.50e-3 | 5.79e-3 | 184 | -3.66e-4 | +5.73e-4 | +1.04e-4 | -5.17e-6 |
| 176 | 3.00e-3 | 1 | 5.06e-3 | 5.06e-3 | 5.06e-3 | 5.06e-3 | 213 | -6.27e-4 | -6.27e-4 | -6.27e-4 | -6.74e-5 |
| 177 | 3.00e-3 | 1 | 5.39e-3 | 5.39e-3 | 5.39e-3 | 5.39e-3 | 199 | +3.09e-4 | +3.09e-4 | +3.09e-4 | -2.98e-5 |
| 178 | 3.00e-3 | 2 | 4.77e-3 | 5.09e-3 | 4.93e-3 | 4.77e-3 | 160 | -4.05e-4 | -3.32e-4 | -3.69e-4 | -9.45e-5 |
| 179 | 3.00e-3 | 1 | 4.70e-3 | 4.70e-3 | 4.70e-3 | 4.70e-3 | 240 | -6.56e-5 | -6.56e-5 | -6.56e-5 | -9.16e-5 |
| 180 | 3.00e-3 | 1 | 5.98e-3 | 5.98e-3 | 5.98e-3 | 5.98e-3 | 263 | +9.14e-4 | +9.14e-4 | +9.14e-4 | +8.92e-6 |
| 181 | 3.00e-3 | 1 | 5.85e-3 | 5.85e-3 | 5.85e-3 | 5.85e-3 | 254 | -8.36e-5 | -8.36e-5 | -8.36e-5 | -3.30e-7 |
| 182 | 3.00e-3 | 2 | 5.62e-3 | 5.91e-3 | 5.77e-3 | 5.62e-3 | 203 | -2.42e-4 | +4.37e-5 | -9.92e-5 | -2.05e-5 |
| 183 | 3.00e-3 | 1 | 5.26e-3 | 5.26e-3 | 5.26e-3 | 5.26e-3 | 229 | -2.90e-4 | -2.90e-4 | -2.90e-4 | -4.75e-5 |
| 184 | 3.00e-3 | 1 | 5.86e-3 | 5.86e-3 | 5.86e-3 | 5.86e-3 | 225 | +4.78e-4 | +4.78e-4 | +4.78e-4 | +5.05e-6 |
| 185 | 3.00e-3 | 1 | 5.40e-3 | 5.40e-3 | 5.40e-3 | 5.40e-3 | 244 | -3.35e-4 | -3.35e-4 | -3.35e-4 | -2.90e-5 |
| 186 | 3.00e-3 | 1 | 5.65e-3 | 5.65e-3 | 5.65e-3 | 5.65e-3 | 265 | +1.71e-4 | +1.71e-4 | +1.71e-4 | -8.97e-6 |
| 187 | 3.00e-3 | 1 | 6.05e-3 | 6.05e-3 | 6.05e-3 | 6.05e-3 | 292 | +2.33e-4 | +2.33e-4 | +2.33e-4 | +1.53e-5 |
| 188 | 3.00e-3 | 1 | 6.32e-3 | 6.32e-3 | 6.32e-3 | 6.32e-3 | 263 | +1.66e-4 | +1.66e-4 | +1.66e-4 | +3.03e-5 |
| 189 | 3.00e-3 | 2 | 5.77e-3 | 6.16e-3 | 5.97e-3 | 5.77e-3 | 214 | -3.04e-4 | -1.07e-4 | -2.06e-4 | -1.55e-5 |
| 190 | 3.00e-3 | 1 | 5.56e-3 | 5.56e-3 | 5.56e-3 | 5.56e-3 | 250 | -1.48e-4 | -1.48e-4 | -1.48e-4 | -2.88e-5 |
| 191 | 3.00e-3 | 1 | 5.77e-3 | 5.77e-3 | 5.77e-3 | 5.77e-3 | 254 | +1.41e-4 | +1.41e-4 | +1.41e-4 | -1.18e-5 |
| 192 | 3.00e-3 | 1 | 5.96e-3 | 5.96e-3 | 5.96e-3 | 5.96e-3 | 232 | +1.45e-4 | +1.45e-4 | +1.45e-4 | +3.94e-6 |
| 193 | 3.00e-3 | 2 | 5.68e-3 | 5.83e-3 | 5.76e-3 | 5.68e-3 | 178 | -1.48e-4 | -1.01e-4 | -1.25e-4 | -2.07e-5 |
| 194 | 3.00e-3 | 1 | 5.16e-3 | 5.16e-3 | 5.16e-3 | 5.16e-3 | 255 | -3.79e-4 | -3.79e-4 | -3.79e-4 | -5.66e-5 |
| 195 | 3.00e-3 | 1 | 6.16e-3 | 6.16e-3 | 6.16e-3 | 6.16e-3 | 224 | +7.93e-4 | +7.93e-4 | +7.93e-4 | +2.84e-5 |
| 196 | 3.00e-3 | 2 | 5.77e-3 | 6.03e-3 | 5.90e-3 | 6.03e-3 | 188 | -2.65e-4 | +2.26e-4 | -1.93e-5 | +2.18e-5 |
| 197 | 3.00e-3 | 1 | 5.28e-3 | 5.28e-3 | 5.28e-3 | 5.28e-3 | 214 | -6.21e-4 | -6.21e-4 | -6.21e-4 | -4.25e-5 |
| 198 | 3.00e-3 | 1 | 5.66e-3 | 5.66e-3 | 5.66e-3 | 5.66e-3 | 203 | +3.45e-4 | +3.45e-4 | +3.45e-4 | -3.73e-6 |
| 199 | 3.00e-3 | 1 | 5.62e-3 | 5.62e-3 | 5.62e-3 | 5.62e-3 | 221 | -2.92e-5 | -2.92e-5 | -2.92e-5 | -6.28e-6 |

