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
| cpu-async | 0.053181 | 0.9165 | +0.0040 | 1786.5 | 493 | 78.7 | 100% | 100% | 100% | 7.2 |

## Best Mode per Model

| Model | Best Eval | Mode | Fastest (within 2% of solo-0) | Mode |
|-------|-----------|------|-------------------------------|------|
| resnet-graph | 0.9165 | cpu-async | - | - |

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
| cpu-async | 2.0408 | 0.7700 | 0.5702 | 0.5184 | 0.5254 | 0.5050 | 0.4978 | 0.4824 | 0.4784 | 0.4644 | 0.2119 | 0.1702 | 0.1468 | 0.1361 | 0.1165 | 0.0717 | 0.0636 | 0.0588 | 0.0561 | 0.0532 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | cpu-async | 0 | cuda:0 | 0.4017 | 2.7 |
| resnet-graph | cpu-async | 1 | cuda:1 | 0.3032 | 3.6 |
| resnet-graph | cpu-async | 2 | cuda:2 | 0.2951 | 3.7 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) | GPU2 Peak (MB) | GPU2 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|---------------|---------------|
| resnet-graph | cpu-async | 358 | 357 | 396 | 392 | 390 | 386 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| cpu-async | gpu1 | 823.5 | 1.0 | epoch-boundary(91) |
| cpu-async | gpu2 | 476.2 | 0.9 | epoch-boundary(52) |
| cpu-async | gpu1 | 476.1 | 0.9 | epoch-boundary(52) |
| cpu-async | gpu1 | 1411.4 | 0.7 | epoch-boundary(157) |
| cpu-async | gpu1 | 698.6 | 0.6 | epoch-boundary(77) |
| cpu-async | gpu2 | 698.6 | 0.6 | epoch-boundary(77) |
| cpu-async | gpu2 | 1411.4 | 0.5 | epoch-boundary(157) |
| cpu-async | gpu1 | 1786.0 | 0.5 | epoch-boundary(199) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | cpu-async | gpu1 | 3.7s | 0.0s | 0.0s | 0.0s | 4.4s |
| resnet-graph | cpu-async | gpu2 | 2.1s | 0.0s | 0.0s | 0.0s | 2.8s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | cpu-async | 293 | 0 | 493 | 78.7 | 1953/9637 | 493 | 78.7 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | cpu-async | 196.4 | 11.0% |

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
| resnet-graph | cpu-async | 186 | 493 | 0 | 7.56e-3 | -4.82e-6 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | cpu-async | 0 | 493 | 9.20e-2 | 6.86e-2 | 2.31e-3 | 4.17e-1 | 28.8 | -2.12e-4 | 1.20e-3 |
| resnet-graph | cpu-async | 1 | 493 | 9.26e-2 | 6.95e-2 | 2.39e-3 | 4.17e-1 | 31.6 | -2.21e-4 | 1.39e-3 |
| resnet-graph | cpu-async | 2 | 493 | 9.40e-2 | 7.30e-2 | 2.34e-3 | 5.16e-1 | 39.6 | -2.44e-4 | 1.39e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | cpu-async | rank0 ↔ rank1 | +0.9939 |
| resnet-graph | cpu-async | rank0 ↔ rank2 | +0.9895 |
| resnet-graph | cpu-async | rank1 ↔ rank2 | +0.9890 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | cpu-async | 54 (1,2,3,4,5,6,8,9…148,149) | 0 (—) | — | 1,2,3,4,5,6,8,9…148,149 | — |

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
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 291 | +0.103 |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | 127 | +0.091 |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | 71 | +0.262 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | cpu-async | 491 | +0.031 | 185 | +0.282 | +0.446 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | cpu-async | 492 | 3.40e1–8.02e1 | 6.56e1 | 2.50e-3 | 4.25e-3 | 5.69e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 293 | 33–77927 | +9.911e-6 | 0.427 | +1.026e-5 | 0.478 | 92 | +1.007e-5 | 0.620 | 33–1003 | +9.574e-4 | 0.734 |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 281 | 899–77927 | +1.118e-5 | 0.618 | +1.140e-5 | 0.650 | 91 | +1.037e-5 | 0.638 | 72–1003 | +1.013e-3 | 0.933 |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | 128 | 78594–117145 | -2.693e-6 | 0.016 | -2.717e-6 | 0.018 | 50 | -3.912e-6 | 0.019 | 98–667 | +3.759e-4 | 0.066 |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | 72 | 117362–155934 | +2.371e-5 | 0.388 | +2.441e-5 | 0.420 | 44 | +1.331e-5 | 0.305 | 86–1040 | +1.210e-3 | 0.699 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | +9.574e-4 | r0: +9.466e-4, r1: +9.616e-4, r2: +9.666e-4 | r0: 0.749, r1: 0.736, r2: 0.689 | 1.02× | ✓ |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +1.013e-3 | r0: +9.956e-4, r1: +1.018e-3, r2: +1.027e-3 | r0: 0.926, r1: 0.914, r2: 0.916 | 1.03× | ✓ |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | +3.759e-4 | r0: +3.133e-4, r1: +4.133e-4, r2: +4.019e-4 | r0: 0.048, r1: 0.080, r2: 0.067 | 1.32× | ✓ |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | +1.210e-3 | r0: +1.206e-3, r1: +1.222e-3, r2: +1.205e-3 | r0: 0.719, r1: 0.673, r2: 0.705 | 1.01× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | cpu-async | `█▇▇▇▇▇▇▇▇██████████████▆▄▄▅▅▅▅▅▅▅▅▅▄▁▂▂▂▂▂▂▂▂▂▂` | `▁▇▇▇▇▇▇▇▇██████████████▇▇▇███████▇▇▄▆▇█████████` |

### Per-Epoch Detail: resnet-graph / cpu-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 12 | 1.02e-1 | 5.16e-1 | 2.18e-1 | 1.12e-1 | 27 | -3.42e-2 | +7.39e-3 | -9.48e-3 | -7.34e-3 |
| 1 | 3.00e-1 | 7 | 1.06e-1 | 1.68e-1 | 1.17e-1 | 1.12e-1 | 48 | -1.72e-2 | +5.15e-3 | -1.54e-3 | -3.59e-3 |
| 2 | 3.00e-1 | 5 | 9.86e-2 | 1.48e-1 | 1.17e-1 | 9.86e-2 | 34 | -6.37e-3 | +3.01e-3 | -1.52e-3 | -2.72e-3 |
| 3 | 3.00e-1 | 6 | 1.08e-1 | 1.40e-1 | 1.19e-1 | 1.23e-1 | 44 | -6.97e-3 | +4.92e-3 | +5.08e-5 | -1.32e-3 |
| 4 | 3.00e-1 | 9 | 1.03e-1 | 1.59e-1 | 1.17e-1 | 1.15e-1 | 45 | -7.59e-3 | +2.79e-3 | -7.57e-4 | -8.11e-4 |
| 5 | 3.00e-1 | 4 | 1.21e-1 | 1.45e-1 | 1.28e-1 | 1.23e-1 | 48 | -3.17e-3 | +3.15e-3 | -5.43e-5 | -5.71e-4 |
| 6 | 3.00e-1 | 8 | 1.10e-1 | 1.58e-1 | 1.21e-1 | 1.10e-1 | 37 | -5.65e-3 | +2.74e-3 | -7.24e-4 | -6.34e-4 |
| 7 | 3.00e-1 | 4 | 1.07e-1 | 1.50e-1 | 1.21e-1 | 1.12e-1 | 38 | -8.04e-3 | +3.86e-3 | -1.12e-3 | -8.02e-4 |
| 8 | 3.00e-1 | 6 | 1.08e-1 | 1.53e-1 | 1.18e-1 | 1.17e-1 | 43 | -9.26e-3 | +3.73e-3 | -7.68e-4 | -7.02e-4 |
| 9 | 3.00e-1 | 6 | 1.14e-1 | 1.55e-1 | 1.23e-1 | 1.17e-1 | 45 | -5.46e-3 | +3.39e-3 | -4.51e-4 | -5.68e-4 |
| 10 | 3.00e-1 | 6 | 1.14e-1 | 1.54e-1 | 1.23e-1 | 1.14e-1 | 41 | -6.29e-3 | +3.03e-3 | -7.09e-4 | -6.23e-4 |
| 11 | 3.00e-1 | 5 | 1.13e-1 | 1.50e-1 | 1.26e-1 | 1.20e-1 | 49 | -7.28e-3 | +3.31e-3 | -5.97e-4 | -6.10e-4 |
| 12 | 3.00e-1 | 5 | 1.17e-1 | 1.62e-1 | 1.32e-1 | 1.17e-1 | 46 | -4.43e-3 | +2.84e-3 | -7.82e-4 | -7.14e-4 |
| 13 | 3.00e-1 | 6 | 1.12e-1 | 1.53e-1 | 1.22e-1 | 1.12e-1 | 40 | -4.68e-3 | +2.97e-3 | -6.57e-4 | -6.95e-4 |
| 14 | 3.00e-1 | 8 | 1.07e-1 | 1.45e-1 | 1.16e-1 | 1.19e-1 | 48 | -8.75e-3 | +3.40e-3 | -3.75e-4 | -4.11e-4 |
| 15 | 3.00e-1 | 5 | 9.83e-2 | 1.51e-1 | 1.14e-1 | 9.83e-2 | 31 | -8.06e-3 | +2.70e-3 | -1.95e-3 | -1.04e-3 |
| 16 | 3.00e-1 | 8 | 1.05e-1 | 1.38e-1 | 1.13e-1 | 1.11e-1 | 40 | -8.04e-3 | +4.63e-3 | -2.55e-4 | -5.59e-4 |
| 17 | 3.00e-1 | 5 | 1.05e-1 | 1.45e-1 | 1.17e-1 | 1.09e-1 | 42 | -5.21e-3 | +3.40e-3 | -7.86e-4 | -6.52e-4 |
| 18 | 3.00e-1 | 7 | 1.06e-1 | 1.59e-1 | 1.17e-1 | 1.06e-1 | 34 | -8.45e-3 | +4.08e-3 | -1.06e-3 | -8.42e-4 |
| 19 | 3.00e-1 | 7 | 1.05e-1 | 1.45e-1 | 1.20e-1 | 1.25e-1 | 48 | -7.30e-3 | +4.24e-3 | -2.09e-4 | -4.41e-4 |
| 20 | 3.00e-1 | 5 | 1.14e-1 | 1.44e-1 | 1.22e-1 | 1.14e-1 | 41 | -3.54e-3 | +1.93e-3 | -7.13e-4 | -5.55e-4 |
| 21 | 3.00e-1 | 9 | 1.02e-1 | 1.48e-1 | 1.13e-1 | 1.09e-1 | 37 | -5.18e-3 | +3.62e-3 | -4.43e-4 | -4.04e-4 |
| 22 | 3.00e-1 | 5 | 1.02e-1 | 1.51e-1 | 1.15e-1 | 1.02e-1 | 33 | -7.16e-3 | +4.17e-3 | -1.33e-3 | -7.92e-4 |
| 23 | 3.00e-1 | 10 | 1.02e-1 | 1.47e-1 | 1.09e-1 | 1.03e-1 | 33 | -9.01e-3 | +5.29e-3 | -5.47e-4 | -5.64e-4 |
| 24 | 3.00e-1 | 5 | 1.08e-1 | 1.52e-1 | 1.20e-1 | 1.12e-1 | 38 | -8.07e-3 | +4.86e-3 | -7.43e-4 | -6.40e-4 |
| 25 | 3.00e-1 | 8 | 9.53e-2 | 1.53e-1 | 1.08e-1 | 1.06e-1 | 31 | -9.43e-3 | +3.88e-3 | -1.00e-3 | -6.56e-4 |
| 26 | 3.00e-1 | 10 | 9.96e-2 | 1.42e-1 | 1.08e-1 | 9.96e-2 | 32 | -9.40e-3 | +4.31e-3 | -7.55e-4 | -6.69e-4 |
| 27 | 3.00e-1 | 5 | 1.13e-1 | 1.47e-1 | 1.21e-1 | 1.17e-1 | 41 | -7.28e-3 | +5.59e-3 | -2.29e-4 | -5.02e-4 |
| 28 | 3.00e-1 | 6 | 1.05e-1 | 1.55e-1 | 1.20e-1 | 1.09e-1 | 34 | -4.24e-3 | +3.44e-3 | -8.80e-4 | -6.81e-4 |
| 29 | 3.00e-1 | 6 | 1.05e-1 | 1.54e-1 | 1.26e-1 | 1.26e-1 | 51 | -4.51e-3 | +4.24e-3 | -1.92e-4 | -4.52e-4 |
| 30 | 3.00e-1 | 6 | 1.25e-1 | 1.62e-1 | 1.34e-1 | 1.27e-1 | 51 | -3.63e-3 | +2.49e-3 | -3.91e-4 | -4.17e-4 |
| 31 | 3.00e-1 | 4 | 1.18e-1 | 1.57e-1 | 1.31e-1 | 1.18e-1 | 45 | -4.45e-3 | +2.42e-3 | -9.20e-4 | -6.09e-4 |
| 32 | 3.00e-1 | 6 | 1.15e-1 | 1.51e-1 | 1.26e-1 | 1.15e-1 | 45 | -3.83e-3 | +2.94e-3 | -5.21e-4 | -5.89e-4 |
| 33 | 3.00e-1 | 8 | 1.08e-1 | 1.58e-1 | 1.17e-1 | 1.08e-1 | 37 | -6.82e-3 | +3.64e-3 | -7.58e-4 | -6.40e-4 |
| 34 | 3.00e-1 | 1 | 1.12e-1 | 1.12e-1 | 1.12e-1 | 1.12e-1 | 38 | +1.08e-3 | +1.08e-3 | +1.08e-3 | -4.68e-4 |
| 35 | 3.00e-1 | 1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 359 | +1.98e-3 | +1.98e-3 | +1.98e-3 | -2.23e-4 |
| 36 | 3.00e-1 | 1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 314 | +6.93e-6 | +6.93e-6 | +6.93e-6 | -2.00e-4 |
| 37 | 3.00e-1 | 1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 325 | +1.71e-5 | +1.71e-5 | +1.71e-5 | -1.79e-4 |
| 39 | 3.00e-1 | 2 | 2.20e-1 | 2.38e-1 | 2.29e-1 | 2.20e-1 | 291 | -2.61e-4 | +9.03e-5 | -8.54e-5 | -1.63e-4 |
| 41 | 3.00e-1 | 1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 341 | +1.25e-4 | +1.25e-4 | +1.25e-4 | -1.34e-4 |
| 42 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 269 | -2.94e-4 | -2.94e-4 | -2.94e-4 | -1.50e-4 |
| 43 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 259 | -5.73e-6 | -5.73e-6 | -5.73e-6 | -1.35e-4 |
| 44 | 3.00e-1 | 1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 316 | +1.94e-4 | +1.94e-4 | +1.94e-4 | -1.02e-4 |
| 45 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 297 | -8.63e-5 | -8.63e-5 | -8.63e-5 | -1.01e-4 |
| 46 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 283 | -3.10e-5 | -3.10e-5 | -3.10e-5 | -9.39e-5 |
| 47 | 3.00e-1 | 1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 326 | +1.19e-4 | +1.19e-4 | +1.19e-4 | -7.26e-5 |
| 48 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 294 | -1.15e-4 | -1.15e-4 | -1.15e-4 | -7.68e-5 |
| 49 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 266 | -7.13e-5 | -7.13e-5 | -7.13e-5 | -7.63e-5 |
| 50 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 263 | -6.64e-5 | -6.64e-5 | -6.64e-5 | -7.53e-5 |
| 51 | 3.00e-1 | 1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 308 | +2.01e-4 | +2.01e-4 | +2.01e-4 | -4.76e-5 |
| 52 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 273 | -1.27e-4 | -1.27e-4 | -1.27e-4 | -5.56e-5 |
| 54 | 3.00e-1 | 2 | 2.18e-1 | 2.34e-1 | 2.26e-1 | 2.18e-1 | 273 | -2.69e-4 | +2.17e-4 | -2.59e-5 | -5.24e-5 |
| 56 | 3.00e-1 | 2 | 2.13e-1 | 2.37e-1 | 2.25e-1 | 2.13e-1 | 248 | -4.34e-4 | +2.44e-4 | -9.52e-5 | -6.39e-5 |
| 58 | 3.00e-1 | 2 | 2.13e-1 | 2.30e-1 | 2.21e-1 | 2.13e-1 | 248 | -3.26e-4 | +2.45e-4 | -4.03e-5 | -6.23e-5 |
| 60 | 3.00e-1 | 2 | 2.11e-1 | 2.27e-1 | 2.19e-1 | 2.11e-1 | 248 | -2.95e-4 | +2.14e-4 | -4.05e-5 | -6.07e-5 |
| 62 | 3.00e-1 | 2 | 2.12e-1 | 2.24e-1 | 2.18e-1 | 2.12e-1 | 237 | -2.29e-4 | +1.96e-4 | -1.63e-5 | -5.44e-5 |
| 64 | 3.00e-1 | 2 | 2.12e-1 | 2.27e-1 | 2.20e-1 | 2.12e-1 | 234 | -2.88e-4 | +2.13e-4 | -3.72e-5 | -5.36e-5 |
| 65 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 260 | +5.04e-5 | +5.04e-5 | +5.04e-5 | -4.32e-5 |
| 66 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 250 | -1.89e-5 | -1.89e-5 | -1.89e-5 | -4.08e-5 |
| 67 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 247 | -4.57e-5 | -4.57e-5 | -4.57e-5 | -4.13e-5 |
| 68 | 3.00e-1 | 1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 275 | +1.72e-4 | +1.72e-4 | +1.72e-4 | -1.99e-5 |
| 69 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 255 | -1.38e-4 | -1.38e-4 | -1.38e-4 | -3.17e-5 |
| 70 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 252 | -1.40e-5 | -1.40e-5 | -1.40e-5 | -2.99e-5 |
| 71 | 3.00e-1 | 1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 300 | +1.89e-4 | +1.89e-4 | +1.89e-4 | -7.98e-6 |
| 72 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 266 | -1.25e-4 | -1.25e-4 | -1.25e-4 | -1.97e-5 |
| 73 | 3.00e-1 | 1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 264 | +6.24e-5 | +6.24e-5 | +6.24e-5 | -1.15e-5 |
| 74 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 250 | -1.22e-4 | -1.22e-4 | -1.22e-4 | -2.25e-5 |
| 75 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 230 | -9.63e-5 | -9.63e-5 | -9.63e-5 | -2.99e-5 |
| 76 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 254 | +1.38e-4 | +1.38e-4 | +1.38e-4 | -1.31e-5 |
| 77 | 3.00e-1 | 2 | 2.01e-1 | 2.16e-1 | 2.09e-1 | 2.01e-1 | 206 | -3.36e-4 | -4.77e-5 | -1.92e-4 | -4.85e-5 |
| 78 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 255 | +2.94e-4 | +2.94e-4 | +2.94e-4 | -1.42e-5 |
| 79 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 234 | -1.18e-4 | -1.18e-4 | -1.18e-4 | -2.47e-5 |
| 80 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 244 | +4.93e-5 | +4.93e-5 | +4.93e-5 | -1.73e-5 |
| 81 | 3.00e-1 | 2 | 2.05e-1 | 2.15e-1 | 2.10e-1 | 2.05e-1 | 215 | -2.27e-4 | +3.31e-5 | -9.68e-5 | -3.37e-5 |
| 82 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 239 | +1.58e-4 | +1.58e-4 | +1.58e-4 | -1.45e-5 |
| 83 | 3.00e-1 | 1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 275 | +1.87e-4 | +1.87e-4 | +1.87e-4 | +5.68e-6 |
| 84 | 3.00e-1 | 1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 271 | -1.21e-5 | -1.21e-5 | -1.21e-5 | +3.90e-6 |
| 85 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 220 | -3.15e-4 | -3.15e-4 | -3.15e-4 | -2.79e-5 |
| 86 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 242 | +1.71e-4 | +1.71e-4 | +1.71e-4 | -8.09e-6 |
| 87 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 236 | -1.29e-4 | -1.29e-4 | -1.29e-4 | -2.02e-5 |
| 88 | 3.00e-1 | 2 | 2.00e-1 | 2.08e-1 | 2.04e-1 | 2.00e-1 | 195 | -2.17e-4 | -5.61e-5 | -1.37e-4 | -4.31e-5 |
| 89 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 207 | +9.28e-5 | +9.28e-5 | +9.28e-5 | -2.95e-5 |
| 90 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 227 | +1.03e-4 | +1.03e-4 | +1.03e-4 | -1.62e-5 |
| 91 | 3.00e-1 | 2 | 2.07e-1 | 2.13e-1 | 2.10e-1 | 2.07e-1 | 205 | -1.56e-4 | +9.24e-5 | -3.17e-5 | -2.04e-5 |
| 92 | 3.00e-1 | 1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 250 | +2.74e-4 | +2.74e-4 | +2.74e-4 | +9.00e-6 |
| 93 | 3.00e-1 | 1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 270 | +8.50e-5 | +8.50e-5 | +8.50e-5 | +1.66e-5 |
| 94 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 227 | -3.10e-4 | -3.10e-4 | -3.10e-4 | -1.60e-5 |
| 95 | 3.00e-1 | 2 | 1.99e-1 | 2.03e-1 | 2.01e-1 | 1.99e-1 | 188 | -1.80e-4 | -1.13e-4 | -1.47e-4 | -4.06e-5 |
| 96 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 238 | +2.90e-4 | +2.90e-4 | +2.90e-4 | -7.54e-6 |
| 97 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 204 | -1.75e-4 | -1.75e-4 | -1.75e-4 | -2.43e-5 |
| 98 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 220 | +4.89e-5 | +4.89e-5 | +4.89e-5 | -1.69e-5 |
| 99 | 3.00e-1 | 2 | 2.05e-1 | 2.10e-1 | 2.08e-1 | 2.05e-1 | 188 | -1.26e-4 | +4.19e-5 | -4.20e-5 | -2.25e-5 |
| 100 | 3.00e-2 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 246 | +9.81e-5 | +9.81e-5 | +9.81e-5 | -1.05e-5 |
| 101 | 3.00e-2 | 1 | 1.02e-1 | 1.02e-1 | 1.02e-1 | 1.02e-1 | 234 | -3.09e-3 | -3.09e-3 | -3.09e-3 | -3.18e-4 |
| 102 | 3.00e-2 | 2 | 3.17e-2 | 5.31e-2 | 4.24e-2 | 3.17e-2 | 181 | -2.98e-3 | -2.85e-3 | -2.92e-3 | -8.11e-4 |
| 103 | 3.00e-2 | 1 | 2.59e-2 | 2.59e-2 | 2.59e-2 | 2.59e-2 | 210 | -9.54e-4 | -9.54e-4 | -9.54e-4 | -8.26e-4 |
| 104 | 3.00e-2 | 2 | 2.33e-2 | 2.41e-2 | 2.37e-2 | 2.33e-2 | 164 | -3.56e-4 | -2.13e-4 | -2.85e-4 | -7.22e-4 |
| 105 | 3.00e-2 | 1 | 2.52e-2 | 2.52e-2 | 2.52e-2 | 2.52e-2 | 217 | +3.49e-4 | +3.49e-4 | +3.49e-4 | -6.15e-4 |
| 106 | 3.00e-2 | 1 | 2.55e-2 | 2.55e-2 | 2.55e-2 | 2.55e-2 | 216 | +6.19e-5 | +6.19e-5 | +6.19e-5 | -5.47e-4 |
| 107 | 3.00e-2 | 2 | 2.66e-2 | 2.68e-2 | 2.67e-2 | 2.66e-2 | 182 | -2.36e-5 | +2.37e-4 | +1.07e-4 | -4.24e-4 |
| 108 | 3.00e-2 | 1 | 2.84e-2 | 2.84e-2 | 2.84e-2 | 2.84e-2 | 227 | +2.86e-4 | +2.86e-4 | +2.86e-4 | -3.53e-4 |
| 109 | 3.00e-2 | 2 | 2.60e-2 | 2.76e-2 | 2.68e-2 | 2.60e-2 | 159 | -3.67e-4 | -1.54e-4 | -2.61e-4 | -3.37e-4 |
| 110 | 3.00e-2 | 2 | 2.71e-2 | 2.79e-2 | 2.75e-2 | 2.79e-2 | 159 | +1.94e-4 | +2.17e-4 | +2.06e-4 | -2.34e-4 |
| 111 | 3.00e-2 | 1 | 2.94e-2 | 2.94e-2 | 2.94e-2 | 2.94e-2 | 187 | +2.77e-4 | +2.77e-4 | +2.77e-4 | -1.83e-4 |
| 112 | 3.00e-2 | 2 | 2.96e-2 | 3.16e-2 | 3.06e-2 | 2.96e-2 | 176 | -3.69e-4 | +3.25e-4 | -2.19e-5 | -1.56e-4 |
| 113 | 3.00e-2 | 1 | 3.23e-2 | 3.23e-2 | 3.23e-2 | 3.23e-2 | 219 | +3.99e-4 | +3.99e-4 | +3.99e-4 | -1.00e-4 |
| 114 | 3.00e-2 | 2 | 2.87e-2 | 3.15e-2 | 3.01e-2 | 2.87e-2 | 136 | -6.77e-4 | -1.36e-4 | -4.06e-4 | -1.61e-4 |
| 115 | 3.00e-2 | 1 | 3.02e-2 | 3.02e-2 | 3.02e-2 | 3.02e-2 | 167 | +2.97e-4 | +2.97e-4 | +2.97e-4 | -1.15e-4 |
| 116 | 3.00e-2 | 2 | 2.95e-2 | 3.18e-2 | 3.07e-2 | 2.95e-2 | 136 | -5.50e-4 | +2.87e-4 | -1.32e-4 | -1.22e-4 |
| 117 | 3.00e-2 | 2 | 3.05e-2 | 3.22e-2 | 3.14e-2 | 3.05e-2 | 136 | -3.86e-4 | +4.81e-4 | +4.74e-5 | -9.45e-5 |
| 118 | 3.00e-2 | 1 | 3.35e-2 | 3.35e-2 | 3.35e-2 | 3.35e-2 | 179 | +5.14e-4 | +5.14e-4 | +5.14e-4 | -3.37e-5 |
| 119 | 3.00e-2 | 2 | 3.01e-2 | 3.24e-2 | 3.13e-2 | 3.01e-2 | 129 | -5.62e-4 | -2.00e-4 | -3.81e-4 | -1.01e-4 |
| 120 | 3.00e-2 | 2 | 3.00e-2 | 3.19e-2 | 3.10e-2 | 3.00e-2 | 128 | -4.61e-4 | +3.65e-4 | -4.79e-5 | -9.54e-5 |
| 121 | 3.00e-2 | 3 | 3.19e-2 | 3.39e-2 | 3.28e-2 | 3.26e-2 | 140 | -4.36e-4 | +7.61e-4 | +1.58e-4 | -3.23e-5 |
| 122 | 3.00e-2 | 1 | 3.75e-2 | 3.75e-2 | 3.75e-2 | 3.75e-2 | 179 | +7.80e-4 | +7.80e-4 | +7.80e-4 | +4.89e-5 |
| 123 | 3.00e-2 | 2 | 3.37e-2 | 3.85e-2 | 3.61e-2 | 3.37e-2 | 130 | -1.03e-3 | +1.41e-4 | -4.43e-4 | -5.05e-5 |
| 124 | 3.00e-2 | 2 | 3.18e-2 | 3.43e-2 | 3.30e-2 | 3.18e-2 | 111 | -7.03e-4 | +1.23e-4 | -2.90e-4 | -1.00e-4 |
| 125 | 3.00e-2 | 2 | 3.27e-2 | 3.33e-2 | 3.30e-2 | 3.27e-2 | 123 | -1.46e-4 | +3.39e-4 | +9.63e-5 | -6.52e-5 |
| 126 | 3.00e-2 | 2 | 3.50e-2 | 3.72e-2 | 3.61e-2 | 3.50e-2 | 128 | -4.80e-4 | +7.69e-4 | +1.44e-4 | -3.16e-5 |
| 127 | 3.00e-2 | 3 | 3.41e-2 | 3.89e-2 | 3.65e-2 | 3.41e-2 | 122 | -5.93e-4 | +6.10e-4 | -1.52e-4 | -7.55e-5 |
| 128 | 3.00e-2 | 1 | 3.64e-2 | 3.64e-2 | 3.64e-2 | 3.64e-2 | 142 | +4.70e-4 | +4.70e-4 | +4.70e-4 | -2.09e-5 |
| 129 | 3.00e-2 | 3 | 3.50e-2 | 3.91e-2 | 3.65e-2 | 3.50e-2 | 111 | -8.02e-4 | +4.47e-4 | -1.66e-4 | -6.56e-5 |
| 130 | 3.00e-2 | 1 | 3.69e-2 | 3.69e-2 | 3.69e-2 | 3.69e-2 | 144 | +3.80e-4 | +3.80e-4 | +3.80e-4 | -2.10e-5 |
| 131 | 3.00e-2 | 4 | 3.48e-2 | 3.86e-2 | 3.59e-2 | 3.54e-2 | 110 | -9.24e-4 | +3.11e-4 | -1.23e-4 | -5.33e-5 |
| 132 | 3.00e-2 | 1 | 3.89e-2 | 3.89e-2 | 3.89e-2 | 3.89e-2 | 140 | +6.64e-4 | +6.64e-4 | +6.64e-4 | +1.84e-5 |
| 133 | 3.00e-2 | 3 | 3.34e-2 | 4.08e-2 | 3.64e-2 | 3.34e-2 | 82 | -1.63e-3 | +3.36e-4 | -6.03e-4 | -1.58e-4 |
| 134 | 3.00e-2 | 3 | 3.37e-2 | 3.52e-2 | 3.43e-2 | 3.37e-2 | 88 | -3.67e-4 | +4.81e-4 | -4.69e-6 | -1.22e-4 |
| 135 | 3.00e-2 | 3 | 3.58e-2 | 3.68e-2 | 3.63e-2 | 3.64e-2 | 98 | -2.68e-4 | +7.70e-4 | +2.17e-4 | -3.55e-5 |
| 136 | 3.00e-2 | 2 | 3.81e-2 | 4.30e-2 | 4.05e-2 | 3.81e-2 | 103 | -1.16e-3 | +1.07e-3 | -4.50e-5 | -4.84e-5 |
| 137 | 3.00e-2 | 3 | 3.46e-2 | 4.02e-2 | 3.68e-2 | 3.46e-2 | 79 | -1.43e-3 | +4.07e-4 | -4.63e-4 | -1.68e-4 |
| 138 | 3.00e-2 | 4 | 3.38e-2 | 4.00e-2 | 3.57e-2 | 3.39e-2 | 82 | -1.70e-3 | +1.24e-3 | -2.05e-4 | -1.89e-4 |
| 139 | 3.00e-2 | 2 | 3.40e-2 | 4.13e-2 | 3.76e-2 | 3.40e-2 | 72 | -2.71e-3 | +1.57e-3 | -5.70e-4 | -2.83e-4 |
| 140 | 3.00e-2 | 4 | 3.02e-2 | 4.13e-2 | 3.41e-2 | 3.02e-2 | 61 | -3.23e-3 | +1.72e-3 | -8.31e-4 | -4.90e-4 |
| 141 | 3.00e-2 | 4 | 3.27e-2 | 3.65e-2 | 3.38e-2 | 3.27e-2 | 66 | -1.60e-3 | +2.05e-3 | +9.78e-5 | -3.08e-4 |
| 142 | 3.00e-2 | 5 | 3.26e-2 | 3.93e-2 | 3.53e-2 | 3.33e-2 | 65 | -1.15e-3 | +1.80e-3 | -1.27e-4 | -2.56e-4 |
| 143 | 3.00e-2 | 3 | 3.10e-2 | 3.81e-2 | 3.37e-2 | 3.10e-2 | 55 | -3.36e-3 | +1.43e-3 | -8.43e-4 | -4.33e-4 |
| 144 | 3.00e-2 | 5 | 2.78e-2 | 3.90e-2 | 3.18e-2 | 2.78e-2 | 43 | -4.99e-3 | +2.55e-3 | -9.81e-4 | -6.94e-4 |
| 145 | 3.00e-2 | 6 | 2.75e-2 | 3.68e-2 | 3.02e-2 | 3.05e-2 | 49 | -5.52e-3 | +3.20e-3 | -2.34e-4 | -4.35e-4 |
| 146 | 3.00e-2 | 5 | 2.80e-2 | 3.86e-2 | 3.19e-2 | 2.80e-2 | 43 | -3.95e-3 | +2.91e-3 | -7.95e-4 | -6.29e-4 |
| 147 | 3.00e-2 | 6 | 2.78e-2 | 3.66e-2 | 3.00e-2 | 2.93e-2 | 44 | -4.86e-3 | +3.27e-3 | -3.24e-4 | -4.63e-4 |
| 148 | 3.00e-2 | 6 | 2.63e-2 | 4.17e-2 | 3.08e-2 | 2.63e-2 | 37 | -5.98e-3 | +3.55e-3 | -1.30e-3 | -8.67e-4 |
| 149 | 3.00e-2 | 7 | 2.63e-2 | 3.61e-2 | 2.98e-2 | 2.89e-2 | 40 | -5.48e-3 | +4.14e-3 | -2.53e-4 | -5.47e-4 |
| 150 | 3.00e-3 | 6 | 2.48e-3 | 2.23e-2 | 7.52e-3 | 2.48e-3 | 34 | -2.26e-2 | -9.28e-4 | -1.10e-2 | -5.15e-3 |
| 151 | 3.00e-3 | 6 | 2.49e-3 | 3.66e-3 | 2.83e-3 | 2.76e-3 | 45 | -7.44e-3 | +4.32e-3 | -6.50e-4 | -3.00e-3 |
| 152 | 3.00e-3 | 6 | 2.61e-3 | 3.64e-3 | 2.92e-3 | 2.87e-3 | 45 | -5.26e-3 | +3.17e-3 | -5.08e-4 | -1.78e-3 |
| 153 | 3.00e-3 | 8 | 2.53e-3 | 3.65e-3 | 2.80e-3 | 2.69e-3 | 42 | -5.33e-3 | +3.01e-3 | -5.08e-4 | -9.88e-4 |
| 154 | 3.00e-3 | 1 | 2.79e-3 | 2.79e-3 | 2.79e-3 | 2.79e-3 | 42 | +8.74e-4 | +8.74e-4 | +8.74e-4 | -8.02e-4 |
| 155 | 3.00e-3 | 1 | 6.85e-3 | 6.85e-3 | 6.85e-3 | 6.85e-3 | 375 | +2.39e-3 | +2.39e-3 | +2.39e-3 | -4.83e-4 |
| 156 | 3.00e-3 | 1 | 6.98e-3 | 6.98e-3 | 6.98e-3 | 6.98e-3 | 329 | +5.67e-5 | +5.67e-5 | +5.67e-5 | -4.29e-4 |
| 157 | 3.00e-3 | 1 | 6.99e-3 | 6.99e-3 | 6.99e-3 | 6.99e-3 | 295 | +6.62e-6 | +6.62e-6 | +6.62e-6 | -3.85e-4 |
| 159 | 3.00e-3 | 2 | 7.18e-3 | 7.47e-3 | 7.33e-3 | 7.18e-3 | 295 | -1.33e-4 | +1.90e-4 | +2.83e-5 | -3.08e-4 |
| 161 | 3.00e-3 | 1 | 7.98e-3 | 7.98e-3 | 7.98e-3 | 7.98e-3 | 384 | +2.75e-4 | +2.75e-4 | +2.75e-4 | -2.50e-4 |
| 162 | 3.00e-3 | 1 | 7.26e-3 | 7.26e-3 | 7.26e-3 | 7.26e-3 | 296 | -3.21e-4 | -3.21e-4 | -3.21e-4 | -2.57e-4 |
| 163 | 3.00e-3 | 1 | 7.33e-3 | 7.33e-3 | 7.33e-3 | 7.33e-3 | 314 | +3.00e-5 | +3.00e-5 | +3.00e-5 | -2.28e-4 |
| 164 | 3.00e-3 | 1 | 7.37e-3 | 7.37e-3 | 7.37e-3 | 7.37e-3 | 291 | +1.87e-5 | +1.87e-5 | +1.87e-5 | -2.04e-4 |
| 165 | 3.00e-3 | 1 | 7.11e-3 | 7.11e-3 | 7.11e-3 | 7.11e-3 | 273 | -1.31e-4 | -1.31e-4 | -1.31e-4 | -1.96e-4 |
| 166 | 3.00e-3 | 1 | 6.93e-3 | 6.93e-3 | 6.93e-3 | 6.93e-3 | 266 | -9.78e-5 | -9.78e-5 | -9.78e-5 | -1.87e-4 |
| 167 | 3.00e-3 | 1 | 7.02e-3 | 7.02e-3 | 7.02e-3 | 7.02e-3 | 285 | +4.54e-5 | +4.54e-5 | +4.54e-5 | -1.63e-4 |
| 168 | 3.00e-3 | 1 | 7.42e-3 | 7.42e-3 | 7.42e-3 | 7.42e-3 | 312 | +1.79e-4 | +1.79e-4 | +1.79e-4 | -1.29e-4 |
| 169 | 3.00e-3 | 1 | 7.07e-3 | 7.07e-3 | 7.07e-3 | 7.07e-3 | 267 | -1.81e-4 | -1.81e-4 | -1.81e-4 | -1.34e-4 |
| 170 | 3.00e-3 | 1 | 7.15e-3 | 7.15e-3 | 7.15e-3 | 7.15e-3 | 269 | +4.48e-5 | +4.48e-5 | +4.48e-5 | -1.16e-4 |
| 172 | 3.00e-3 | 2 | 7.17e-3 | 8.00e-3 | 7.59e-3 | 7.17e-3 | 266 | -4.09e-4 | +3.06e-4 | -5.17e-5 | -1.08e-4 |
| 174 | 3.00e-3 | 2 | 7.11e-3 | 8.00e-3 | 7.55e-3 | 7.11e-3 | 262 | -4.51e-4 | +3.18e-4 | -6.65e-5 | -1.04e-4 |
| 176 | 3.00e-3 | 1 | 7.81e-3 | 7.81e-3 | 7.81e-3 | 7.81e-3 | 334 | +2.80e-4 | +2.80e-4 | +2.80e-4 | -6.53e-5 |
| 177 | 3.00e-3 | 1 | 7.54e-3 | 7.54e-3 | 7.54e-3 | 7.54e-3 | 289 | -1.21e-4 | -1.21e-4 | -1.21e-4 | -7.09e-5 |
| 178 | 3.00e-3 | 1 | 6.90e-3 | 6.90e-3 | 6.90e-3 | 6.90e-3 | 246 | -3.60e-4 | -3.60e-4 | -3.60e-4 | -9.98e-5 |
| 179 | 3.00e-3 | 2 | 6.75e-3 | 6.90e-3 | 6.83e-3 | 6.75e-3 | 223 | -1.04e-4 | +2.97e-6 | -5.07e-5 | -9.10e-5 |
| 181 | 3.00e-3 | 2 | 7.01e-3 | 8.08e-3 | 7.54e-3 | 7.01e-3 | 237 | -5.98e-4 | +5.29e-4 | -3.45e-5 | -8.59e-5 |
| 182 | 3.00e-3 | 1 | 7.01e-3 | 7.01e-3 | 7.01e-3 | 7.01e-3 | 253 | +2.14e-6 | +2.14e-6 | +2.14e-6 | -7.71e-5 |
| 183 | 3.00e-3 | 1 | 7.15e-3 | 7.15e-3 | 7.15e-3 | 7.15e-3 | 239 | +8.19e-5 | +8.19e-5 | +8.19e-5 | -6.12e-5 |
| 184 | 3.00e-3 | 1 | 7.55e-3 | 7.55e-3 | 7.55e-3 | 7.55e-3 | 295 | +1.84e-4 | +1.84e-4 | +1.84e-4 | -3.67e-5 |
| 185 | 3.00e-3 | 1 | 7.40e-3 | 7.40e-3 | 7.40e-3 | 7.40e-3 | 271 | -7.25e-5 | -7.25e-5 | -7.25e-5 | -4.03e-5 |
| 186 | 3.00e-3 | 1 | 7.20e-3 | 7.20e-3 | 7.20e-3 | 7.20e-3 | 256 | -1.07e-4 | -1.07e-4 | -1.07e-4 | -4.70e-5 |
| 187 | 3.00e-3 | 1 | 7.12e-3 | 7.12e-3 | 7.12e-3 | 7.12e-3 | 238 | -4.60e-5 | -4.60e-5 | -4.60e-5 | -4.69e-5 |
| 188 | 3.00e-3 | 1 | 7.35e-3 | 7.35e-3 | 7.35e-3 | 7.35e-3 | 262 | +1.18e-4 | +1.18e-4 | +1.18e-4 | -3.04e-5 |
| 189 | 3.00e-3 | 1 | 7.81e-3 | 7.81e-3 | 7.81e-3 | 7.81e-3 | 292 | +2.08e-4 | +2.08e-4 | +2.08e-4 | -6.52e-6 |
| 190 | 3.00e-3 | 1 | 7.70e-3 | 7.70e-3 | 7.70e-3 | 7.70e-3 | 296 | -4.54e-5 | -4.54e-5 | -4.54e-5 | -1.04e-5 |
| 191 | 3.00e-3 | 1 | 7.74e-3 | 7.74e-3 | 7.74e-3 | 7.74e-3 | 284 | +1.81e-5 | +1.81e-5 | +1.81e-5 | -7.56e-6 |
| 192 | 3.00e-3 | 1 | 7.21e-3 | 7.21e-3 | 7.21e-3 | 7.21e-3 | 242 | -2.93e-4 | -2.93e-4 | -2.93e-4 | -3.61e-5 |
| 193 | 3.00e-3 | 1 | 7.21e-3 | 7.21e-3 | 7.21e-3 | 7.21e-3 | 262 | -6.13e-7 | -6.13e-7 | -6.13e-7 | -3.26e-5 |
| 194 | 3.00e-3 | 1 | 7.27e-3 | 7.27e-3 | 7.27e-3 | 7.27e-3 | 238 | +3.57e-5 | +3.57e-5 | +3.57e-5 | -2.57e-5 |
| 195 | 3.00e-3 | 2 | 6.66e-3 | 7.20e-3 | 6.93e-3 | 6.66e-3 | 205 | -3.82e-4 | -4.10e-5 | -2.11e-4 | -6.27e-5 |
| 196 | 3.00e-3 | 1 | 6.94e-3 | 6.94e-3 | 6.94e-3 | 6.94e-3 | 237 | +1.75e-4 | +1.75e-4 | +1.75e-4 | -3.89e-5 |
| 197 | 3.00e-3 | 1 | 7.50e-3 | 7.50e-3 | 7.50e-3 | 7.50e-3 | 255 | +3.01e-4 | +3.01e-4 | +3.01e-4 | -4.93e-6 |
| 198 | 3.00e-3 | 1 | 7.95e-3 | 7.95e-3 | 7.95e-3 | 7.95e-3 | 295 | +1.98e-4 | +1.98e-4 | +1.98e-4 | +1.54e-5 |
| 199 | 3.00e-3 | 1 | 7.56e-3 | 7.56e-3 | 7.56e-3 | 7.56e-3 | 270 | -1.87e-4 | -1.87e-4 | -1.87e-4 | -4.82e-6 |

