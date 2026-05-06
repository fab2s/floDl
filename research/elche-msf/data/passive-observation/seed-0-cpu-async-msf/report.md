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
| cpu-async | 0.055268 | 0.9191 | +0.0066 | 1938.5 | 408 | 88.9 | 100% | 100% | 15.0 |

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
| cpu-async | 2.0265 | 0.7627 | 0.5824 | 0.5103 | 0.5343 | 0.5097 | 0.4917 | 0.4820 | 0.4681 | 0.4659 | 0.2026 | 0.1653 | 0.1491 | 0.1361 | 0.1322 | 0.0729 | 0.0646 | 0.0614 | 0.0558 | 0.0553 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | cpu-async | 0 | cuda:0 | 0.3979 | 2.5 |
| resnet-graph | cpu-async | 1 | cuda:1 | 0.3059 | 3.5 |
| resnet-graph | cpu-async | 2 | cuda:2 | 0.2962 | 3.3 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|
| resnet-graph | cpu-async | 358 | 357 | 396 | 392 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| cpu-async | gpu1 | 1200.7 | 2.6 | epoch-boundary(123) |
| cpu-async | gpu2 | 1200.8 | 2.6 | epoch-boundary(123) |
| cpu-async | gpu2 | 397.9 | 1.5 | epoch-boundary(40) |
| cpu-async | gpu2 | 1655.8 | 1.4 | epoch-boundary(170) |
| cpu-async | gpu1 | 398.0 | 1.3 | epoch-boundary(40) |
| cpu-async | gpu1 | 428.0 | 1.1 | epoch-boundary(43) |
| cpu-async | gpu1 | 931.4 | 0.8 | epoch-boundary(95) |
| cpu-async | gpu0 | 0.4 | 0.7 | cpu-avg |
| cpu-async | gpu2 | 1829.7 | 0.6 | epoch-boundary(188) |
| cpu-async | gpu2 | 1714.8 | 0.5 | epoch-boundary(176) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | cpu-async | gpu0 | 0.0s | 0.0s | 0.7s | 0.0s | 0.7s |
| resnet-graph | cpu-async | gpu1 | 5.7s | 0.0s | 0.0s | 0.0s | 6.6s |
| resnet-graph | cpu-async | gpu2 | 7.1s | 0.0s | 0.0s | 0.0s | 7.7s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | cpu-async | 262 | 0 | 408 | 88.9 | 2885/10598 | 408 | 88.9 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | cpu-async | 221.0 | 11.4% |

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
| resnet-graph | cpu-async | 184 | 408 | 0 | 7.45e-3 | -1.65e-5 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | cpu-async | 0 | 408 | 1.07e-1 | 6.97e-2 | 5.25e-3 | 5.44e-1 | 27.0 | -1.99e-4 | 1.32e-3 |
| resnet-graph | cpu-async | 1 | 408 | 1.09e-1 | 7.28e-2 | 5.71e-3 | 5.38e-1 | 36.5 | -2.31e-4 | 1.54e-3 |
| resnet-graph | cpu-async | 2 | 408 | 1.09e-1 | 7.08e-2 | 5.34e-3 | 4.53e-1 | 36.5 | -2.15e-4 | 1.45e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | cpu-async | rank0 ↔ rank1 | +0.9903 |
| resnet-graph | cpu-async | rank0 ↔ rank2 | +0.9890 |
| resnet-graph | cpu-async | rank1 ↔ rank2 | +0.9908 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | cpu-async | 52 (0,1,3,4,5,7,8,9…147,150) | 0 (—) | — | 0,1,3,4,5,7,8,9…147,150 | — |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | cpu-async | 0e0 | 3 | 9 | 9 |
| resnet-graph | cpu-async | 0e0 | 5 | 5 | 5 |
| resnet-graph | cpu-async | 0e0 | 10 | 2 | 2 |
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
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 284 | +0.101 |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | 62 | +0.061 |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | 58 | -0.395 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | cpu-async | 406 | -0.056 | 183 | +0.460 | +0.659 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | cpu-async | 407 | 3.46e1–8.04e1 | 6.63e1 | 2.99e-3 | 4.26e-3 | 4.36e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 286 | 35–77899 | +1.044e-5 | 0.478 | +1.065e-5 | 0.519 | 94 | +1.015e-5 | 0.628 | 32–1004 | +9.732e-4 | 0.743 |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 274 | 918–77899 | +1.155e-5 | 0.658 | +1.169e-5 | 0.691 | 93 | +1.042e-5 | 0.642 | 75–1004 | +1.028e-3 | 0.942 |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | 63 | 78485–116843 | +2.066e-5 | 0.291 | +2.105e-5 | 0.304 | 44 | +1.990e-5 | 0.283 | 362–1024 | +1.460e-3 | 0.451 |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | 59 | 117564–155723 | -1.383e-5 | 0.159 | -1.420e-5 | 0.169 | 46 | -1.707e-5 | 0.203 | 416–904 | +1.323e-3 | 0.105 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | +9.732e-4 | r0: +9.583e-4, r1: +9.874e-4, r2: +9.756e-4 | r0: 0.752, r1: 0.720, r2: 0.731 | 1.03× | ✓ |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +1.028e-3 | r0: +1.008e-3, r1: +1.047e-3, r2: +1.030e-3 | r0: 0.940, r1: 0.924, r2: 0.922 | 1.04× | ✓ |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | +1.460e-3 | r0: +1.434e-3, r1: +1.478e-3, r2: +1.469e-3 | r0: 0.445, r1: 0.456, r2: 0.444 | 1.03× | ✓ |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | +1.323e-3 | r0: +1.276e-3, r1: +1.304e-3, r2: +1.388e-3 | r0: 0.097, r1: 0.100, r2: 0.116 | 1.09× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | cpu-async | `█▇▇▇▇▇▇▇▇██████████████▇▄▄▄▄▄▅▅▅▅▅▅▂▁▁▁▁▁▁▁▁▁▁` | `▁▇▇▇▇▇▇▇▇███████████████▇▇█████████▇▇█████████` |

### Per-Epoch Detail: resnet-graph / cpu-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 12 | 9.86e-2 | 5.44e-1 | 2.10e-1 | 1.19e-1 | 29 | -5.01e-2 | +4.42e-3 | -1.17e-2 | -8.34e-3 |
| 1 | 3.00e-1 | 7 | 9.04e-2 | 1.76e-1 | 1.15e-1 | 1.07e-1 | 41 | -1.28e-2 | +4.92e-3 | -1.69e-3 | -4.07e-3 |
| 2 | 3.00e-1 | 5 | 1.12e-1 | 1.49e-1 | 1.24e-1 | 1.21e-1 | 45 | -4.61e-3 | +3.64e-3 | -1.95e-4 | -2.32e-3 |
| 3 | 3.00e-1 | 5 | 1.15e-1 | 1.50e-1 | 1.31e-1 | 1.15e-1 | 41 | -2.34e-3 | +2.35e-3 | -6.39e-4 | -1.66e-3 |
| 4 | 3.00e-1 | 7 | 1.11e-1 | 1.48e-1 | 1.19e-1 | 1.15e-1 | 40 | -6.80e-3 | +3.19e-3 | -5.01e-4 | -9.98e-4 |
| 5 | 3.00e-1 | 7 | 1.06e-1 | 1.45e-1 | 1.14e-1 | 1.11e-1 | 38 | -8.44e-3 | +2.96e-3 | -7.42e-4 | -7.86e-4 |
| 6 | 3.00e-1 | 7 | 1.05e-1 | 1.51e-1 | 1.14e-1 | 1.05e-1 | 31 | -8.15e-3 | +3.91e-3 | -9.52e-4 | -8.39e-4 |
| 7 | 3.00e-1 | 7 | 1.09e-1 | 1.43e-1 | 1.16e-1 | 1.11e-1 | 37 | -8.66e-3 | +4.16e-3 | -6.11e-4 | -6.95e-4 |
| 8 | 3.00e-1 | 6 | 1.06e-1 | 1.53e-1 | 1.18e-1 | 1.11e-1 | 38 | -9.26e-3 | +3.94e-3 | -1.10e-3 | -8.35e-4 |
| 9 | 3.00e-1 | 7 | 1.04e-1 | 1.58e-1 | 1.18e-1 | 1.17e-1 | 41 | -1.04e-2 | +3.77e-3 | -8.38e-4 | -7.09e-4 |
| 10 | 3.00e-1 | 6 | 1.16e-1 | 1.50e-1 | 1.24e-1 | 1.18e-1 | 45 | -5.46e-3 | +2.93e-3 | -4.91e-4 | -5.97e-4 |
| 11 | 3.00e-1 | 8 | 1.10e-1 | 1.57e-1 | 1.21e-1 | 1.23e-1 | 49 | -6.67e-3 | +3.14e-3 | -3.90e-4 | -3.69e-4 |
| 12 | 3.00e-1 | 4 | 1.04e-1 | 1.56e-1 | 1.23e-1 | 1.04e-1 | 32 | -6.03e-3 | +2.52e-3 | -1.90e-3 | -9.38e-4 |
| 13 | 3.00e-1 | 7 | 1.03e-1 | 1.36e-1 | 1.13e-1 | 1.14e-1 | 42 | -9.78e-3 | +3.96e-3 | -4.29e-4 | -6.00e-4 |
| 14 | 3.00e-1 | 8 | 1.10e-1 | 1.48e-1 | 1.20e-1 | 1.10e-1 | 36 | -4.11e-3 | +3.29e-3 | -4.94e-4 | -5.52e-4 |
| 15 | 3.00e-1 | 5 | 1.07e-1 | 1.45e-1 | 1.17e-1 | 1.10e-1 | 37 | -6.93e-3 | +3.62e-3 | -8.40e-4 | -6.56e-4 |
| 16 | 3.00e-1 | 7 | 1.02e-1 | 1.52e-1 | 1.14e-1 | 1.02e-1 | 32 | -8.07e-3 | +3.54e-3 | -1.18e-3 | -9.07e-4 |
| 17 | 3.00e-1 | 6 | 1.06e-1 | 1.55e-1 | 1.17e-1 | 1.09e-1 | 36 | -7.94e-3 | +4.72e-3 | -8.42e-4 | -8.48e-4 |
| 18 | 3.00e-1 | 9 | 1.04e-1 | 1.45e-1 | 1.14e-1 | 1.08e-1 | 36 | -6.12e-3 | +3.97e-3 | -4.85e-4 | -5.88e-4 |
| 19 | 3.00e-1 | 5 | 1.11e-1 | 1.45e-1 | 1.23e-1 | 1.24e-1 | 46 | -6.19e-3 | +4.22e-3 | -1.05e-4 | -3.70e-4 |
| 20 | 3.00e-1 | 8 | 1.13e-1 | 1.46e-1 | 1.19e-1 | 1.13e-1 | 40 | -4.32e-3 | +2.16e-3 | -4.88e-4 | -4.03e-4 |
| 21 | 3.00e-1 | 4 | 1.16e-1 | 1.52e-1 | 1.30e-1 | 1.30e-1 | 52 | -7.20e-3 | +3.32e-3 | -4.26e-4 | -3.96e-4 |
| 22 | 3.00e-1 | 7 | 1.20e-1 | 1.67e-1 | 1.30e-1 | 1.20e-1 | 44 | -5.15e-3 | +2.49e-3 | -6.38e-4 | -5.04e-4 |
| 23 | 3.00e-1 | 4 | 1.08e-1 | 1.62e-1 | 1.26e-1 | 1.08e-1 | 36 | -8.44e-3 | +2.99e-3 | -2.06e-3 | -1.06e-3 |
| 24 | 3.00e-1 | 6 | 1.12e-1 | 1.53e-1 | 1.22e-1 | 1.18e-1 | 41 | -6.97e-3 | +4.05e-3 | -4.42e-4 | -7.32e-4 |
| 25 | 3.00e-1 | 6 | 1.05e-1 | 1.46e-1 | 1.18e-1 | 1.13e-1 | 41 | -5.71e-3 | +2.87e-3 | -7.13e-4 | -6.87e-4 |
| 26 | 3.00e-1 | 8 | 1.11e-1 | 1.48e-1 | 1.23e-1 | 1.29e-1 | 46 | -7.69e-3 | +3.63e-3 | -9.48e-5 | -2.70e-4 |
| 27 | 3.00e-1 | 4 | 1.19e-1 | 1.57e-1 | 1.31e-1 | 1.20e-1 | 43 | -4.68e-3 | +2.37e-3 | -8.80e-4 | -4.92e-4 |
| 28 | 3.00e-1 | 9 | 1.08e-1 | 1.64e-1 | 1.19e-1 | 1.08e-1 | 34 | -8.11e-3 | +3.15e-3 | -9.20e-4 | -6.82e-4 |
| 29 | 3.00e-1 | 4 | 1.02e-1 | 1.62e-1 | 1.22e-1 | 1.02e-1 | 32 | -9.32e-3 | +4.44e-3 | -2.16e-3 | -1.23e-3 |
| 30 | 3.00e-1 | 6 | 1.04e-1 | 1.46e-1 | 1.20e-1 | 1.24e-1 | 45 | -8.34e-3 | +4.65e-3 | -1.79e-4 | -6.75e-4 |
| 31 | 3.00e-1 | 6 | 1.14e-1 | 1.64e-1 | 1.24e-1 | 1.17e-1 | 40 | -8.01e-3 | +2.91e-3 | -9.11e-4 | -7.33e-4 |
| 32 | 3.00e-1 | 6 | 1.18e-1 | 1.54e-1 | 1.25e-1 | 1.18e-1 | 43 | -6.27e-3 | +3.66e-3 | -4.91e-4 | -6.17e-4 |
| 33 | 3.00e-1 | 1 | 1.17e-1 | 1.17e-1 | 1.17e-1 | 1.17e-1 | 43 | -1.95e-4 | -1.95e-4 | -1.95e-4 | -5.75e-4 |
| 34 | 3.00e-1 | 1 | 2.38e-1 | 2.38e-1 | 2.38e-1 | 2.38e-1 | 360 | +1.97e-3 | +1.97e-3 | +1.97e-3 | -3.20e-4 |
| 35 | 3.00e-1 | 1 | 2.40e-1 | 2.40e-1 | 2.40e-1 | 2.40e-1 | 342 | +2.61e-5 | +2.61e-5 | +2.61e-5 | -2.85e-4 |
| 36 | 3.00e-1 | 1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 308 | -1.32e-4 | -1.32e-4 | -1.32e-4 | -2.70e-4 |
| 37 | 3.00e-1 | 1 | 2.36e-1 | 2.36e-1 | 2.36e-1 | 2.36e-1 | 322 | +7.43e-5 | +7.43e-5 | +7.43e-5 | -2.36e-4 |
| 38 | 3.00e-1 | 1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 317 | -8.65e-5 | -8.65e-5 | -8.65e-5 | -2.21e-4 |
| 39 | 3.00e-1 | 1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 291 | -6.44e-5 | -6.44e-5 | -6.44e-5 | -2.05e-4 |
| 40 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 255 | -1.68e-4 | -1.68e-4 | -1.68e-4 | -2.01e-4 |
| 41 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 261 | +9.79e-6 | +9.79e-6 | +9.79e-6 | -1.80e-4 |
| 43 | 3.00e-1 | 2 | 2.14e-1 | 2.42e-1 | 2.28e-1 | 2.14e-1 | 250 | -5.04e-4 | +3.23e-4 | -9.01e-5 | -1.67e-4 |
| 44 | 3.00e-1 | 1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 285 | +1.37e-4 | +1.37e-4 | +1.37e-4 | -1.37e-4 |
| 46 | 3.00e-1 | 2 | 2.17e-1 | 2.33e-1 | 2.25e-1 | 2.17e-1 | 250 | -2.86e-4 | +1.45e-4 | -7.07e-5 | -1.26e-4 |
| 48 | 3.00e-1 | 2 | 2.16e-1 | 2.32e-1 | 2.24e-1 | 2.16e-1 | 250 | -2.88e-4 | +2.13e-4 | -3.76e-5 | -1.12e-4 |
| 50 | 3.00e-1 | 2 | 2.17e-1 | 2.37e-1 | 2.27e-1 | 2.17e-1 | 250 | -3.53e-4 | +2.68e-4 | -4.22e-5 | -1.02e-4 |
| 52 | 3.00e-1 | 2 | 2.23e-1 | 2.37e-1 | 2.30e-1 | 2.23e-1 | 255 | -2.42e-4 | +2.85e-4 | +2.16e-5 | -8.11e-5 |
| 54 | 3.00e-1 | 1 | 2.40e-1 | 2.40e-1 | 2.40e-1 | 2.40e-1 | 348 | +2.12e-4 | +2.12e-4 | +2.12e-4 | -5.18e-5 |
| 55 | 3.00e-1 | 1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 290 | -2.04e-4 | -2.04e-4 | -2.04e-4 | -6.70e-5 |
| 56 | 3.00e-1 | 1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 288 | -2.98e-6 | -2.98e-6 | -2.98e-6 | -6.06e-5 |
| 57 | 3.00e-1 | 1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 270 | -5.20e-5 | -5.20e-5 | -5.20e-5 | -5.97e-5 |
| 58 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 271 | -5.35e-5 | -5.35e-5 | -5.35e-5 | -5.91e-5 |
| 59 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 266 | +1.37e-5 | +1.37e-5 | +1.37e-5 | -5.18e-5 |
| 60 | 3.00e-1 | 1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 286 | +1.40e-4 | +1.40e-4 | +1.40e-4 | -3.27e-5 |
| 61 | 3.00e-1 | 1 | 2.38e-1 | 2.38e-1 | 2.38e-1 | 2.38e-1 | 322 | +1.11e-4 | +1.11e-4 | +1.11e-4 | -1.83e-5 |
| 62 | 3.00e-1 | 1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 313 | -8.04e-5 | -8.04e-5 | -8.04e-5 | -2.45e-5 |
| 63 | 3.00e-1 | 1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 282 | -1.38e-4 | -1.38e-4 | -1.38e-4 | -3.59e-5 |
| 64 | 3.00e-1 | 1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 271 | +2.98e-5 | +2.98e-5 | +2.98e-5 | -2.93e-5 |
| 65 | 3.00e-1 | 1 | 2.28e-1 | 2.28e-1 | 2.28e-1 | 2.28e-1 | 286 | +4.86e-5 | +4.86e-5 | +4.86e-5 | -2.15e-5 |
| 66 | 3.00e-1 | 1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 270 | -8.40e-5 | -8.40e-5 | -8.40e-5 | -2.78e-5 |
| 67 | 3.00e-1 | 2 | 2.06e-1 | 2.21e-1 | 2.13e-1 | 2.06e-1 | 209 | -3.32e-4 | -3.33e-5 | -1.83e-4 | -5.87e-5 |
| 68 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 255 | +1.83e-4 | +1.83e-4 | +1.83e-4 | -3.46e-5 |
| 69 | 3.00e-1 | 1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 256 | +1.67e-4 | +1.67e-4 | +1.67e-4 | -1.44e-5 |
| 70 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 253 | -1.20e-4 | -1.20e-4 | -1.20e-4 | -2.50e-5 |
| 71 | 3.00e-1 | 1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 267 | +1.68e-4 | +1.68e-4 | +1.68e-4 | -5.72e-6 |
| 72 | 3.00e-1 | 1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 281 | +1.77e-5 | +1.77e-5 | +1.77e-5 | -3.38e-6 |
| 73 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 247 | -2.15e-4 | -2.15e-4 | -2.15e-4 | -2.45e-5 |
| 74 | 3.00e-1 | 2 | 2.08e-1 | 2.12e-1 | 2.10e-1 | 2.08e-1 | 206 | -1.13e-4 | -9.05e-5 | -1.02e-4 | -3.91e-5 |
| 75 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 228 | +6.16e-5 | +6.16e-5 | +6.16e-5 | -2.90e-5 |
| 76 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 247 | +1.41e-4 | +1.41e-4 | +1.41e-4 | -1.20e-5 |
| 77 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 250 | -1.25e-5 | -1.25e-5 | -1.25e-5 | -1.21e-5 |
| 78 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 238 | -3.08e-5 | -3.08e-5 | -3.08e-5 | -1.40e-5 |
| 79 | 3.00e-1 | 1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 266 | +1.85e-4 | +1.85e-4 | +1.85e-4 | +5.99e-6 |
| 80 | 3.00e-1 | 1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 271 | -3.94e-5 | -3.94e-5 | -3.94e-5 | +1.45e-6 |
| 81 | 3.00e-1 | 1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 288 | +1.15e-4 | +1.15e-4 | +1.15e-4 | +1.28e-5 |
| 82 | 3.00e-1 | 1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 250 | -1.67e-4 | -1.67e-4 | -1.67e-4 | -5.18e-6 |
| 83 | 3.00e-1 | 2 | 2.10e-1 | 2.11e-1 | 2.10e-1 | 2.10e-1 | 207 | -2.40e-4 | -3.68e-5 | -1.39e-4 | -2.95e-5 |
| 84 | 3.00e-1 | 1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 257 | +2.40e-4 | +2.40e-4 | +2.40e-4 | -2.55e-6 |
| 85 | 3.00e-1 | 1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 261 | +1.05e-4 | +1.05e-4 | +1.05e-4 | +8.26e-6 |
| 86 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 204 | -3.92e-4 | -3.92e-4 | -3.92e-4 | -3.18e-5 |
| 87 | 3.00e-1 | 2 | 2.09e-1 | 2.20e-1 | 2.14e-1 | 2.09e-1 | 196 | -2.66e-4 | +1.59e-4 | -5.36e-5 | -3.81e-5 |
| 88 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 221 | +1.78e-4 | +1.78e-4 | +1.78e-4 | -1.64e-5 |
| 89 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 226 | -1.03e-5 | -1.03e-5 | -1.03e-5 | -1.58e-5 |
| 90 | 3.00e-1 | 2 | 2.02e-1 | 2.09e-1 | 2.05e-1 | 2.02e-1 | 196 | -1.76e-4 | -1.68e-4 | -1.72e-4 | -4.56e-5 |
| 91 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 222 | +2.14e-4 | +2.14e-4 | +2.14e-4 | -1.96e-5 |
| 92 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 227 | +4.29e-5 | +4.29e-5 | +4.29e-5 | -1.34e-5 |
| 93 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 234 | +6.01e-5 | +6.01e-5 | +6.01e-5 | -6.02e-6 |
| 94 | 3.00e-1 | 2 | 2.10e-1 | 2.18e-1 | 2.14e-1 | 2.10e-1 | 198 | -1.99e-4 | +3.31e-5 | -8.28e-5 | -2.18e-5 |
| 95 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 223 | +1.09e-4 | +1.09e-4 | +1.09e-4 | -8.65e-6 |
| 96 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 223 | +8.20e-5 | +8.20e-5 | +8.20e-5 | +4.11e-7 |
| 97 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 206 | -2.54e-4 | -2.54e-4 | -2.54e-4 | -2.50e-5 |
| 98 | 3.00e-1 | 2 | 2.04e-1 | 2.20e-1 | 2.12e-1 | 2.04e-1 | 171 | -4.49e-4 | +2.39e-4 | -1.05e-4 | -4.37e-5 |
| 99 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 217 | +1.99e-4 | +1.99e-4 | +1.99e-4 | -1.94e-5 |
| 100 | 3.00e-2 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 223 | -3.21e-5 | -3.21e-5 | -3.21e-5 | -2.06e-5 |
| 101 | 3.00e-2 | 2 | 5.38e-2 | 1.06e-1 | 8.00e-2 | 5.38e-2 | 182 | -3.75e-3 | -3.09e-3 | -3.42e-3 | -6.70e-4 |
| 102 | 3.00e-2 | 1 | 3.41e-2 | 3.41e-2 | 3.41e-2 | 3.41e-2 | 224 | -2.03e-3 | -2.03e-3 | -2.03e-3 | -8.05e-4 |
| 103 | 3.00e-2 | 1 | 2.67e-2 | 2.67e-2 | 2.67e-2 | 2.67e-2 | 218 | -1.13e-3 | -1.13e-3 | -1.13e-3 | -8.38e-4 |
| 104 | 3.00e-2 | 2 | 2.29e-2 | 2.59e-2 | 2.44e-2 | 2.29e-2 | 162 | -7.59e-4 | -1.27e-4 | -4.43e-4 | -7.66e-4 |
| 105 | 3.00e-2 | 1 | 2.50e-2 | 2.50e-2 | 2.50e-2 | 2.50e-2 | 202 | +4.26e-4 | +4.26e-4 | +4.26e-4 | -6.47e-4 |
| 106 | 3.00e-2 | 2 | 2.49e-2 | 2.63e-2 | 2.56e-2 | 2.49e-2 | 176 | -3.07e-4 | +2.40e-4 | -3.35e-5 | -5.33e-4 |
| 107 | 3.00e-2 | 2 | 2.42e-2 | 2.69e-2 | 2.56e-2 | 2.42e-2 | 143 | -7.40e-4 | +3.94e-4 | -1.73e-4 | -4.70e-4 |
| 108 | 3.00e-2 | 1 | 2.58e-2 | 2.58e-2 | 2.58e-2 | 2.58e-2 | 165 | +3.85e-4 | +3.85e-4 | +3.85e-4 | -3.85e-4 |
| 109 | 3.00e-2 | 2 | 2.61e-2 | 2.77e-2 | 2.69e-2 | 2.61e-2 | 149 | -3.81e-4 | +3.72e-4 | -4.39e-6 | -3.16e-4 |
| 110 | 3.00e-2 | 1 | 2.92e-2 | 2.92e-2 | 2.92e-2 | 2.92e-2 | 210 | +5.26e-4 | +5.26e-4 | +5.26e-4 | -2.32e-4 |
| 111 | 3.00e-2 | 2 | 2.68e-2 | 2.97e-2 | 2.82e-2 | 2.68e-2 | 143 | -7.30e-4 | +8.55e-5 | -3.22e-4 | -2.53e-4 |
| 112 | 3.00e-2 | 1 | 2.82e-2 | 2.82e-2 | 2.82e-2 | 2.82e-2 | 182 | +2.94e-4 | +2.94e-4 | +2.94e-4 | -1.99e-4 |
| 113 | 3.00e-2 | 3 | 2.73e-2 | 2.95e-2 | 2.82e-2 | 2.73e-2 | 143 | -4.10e-4 | +2.54e-4 | -9.26e-5 | -1.73e-4 |
| 114 | 3.00e-2 | 1 | 2.99e-2 | 2.99e-2 | 2.99e-2 | 2.99e-2 | 168 | +5.41e-4 | +5.41e-4 | +5.41e-4 | -1.02e-4 |
| 115 | 3.00e-2 | 2 | 2.96e-2 | 3.10e-2 | 3.03e-2 | 2.96e-2 | 144 | -3.26e-4 | +2.07e-4 | -5.94e-5 | -9.64e-5 |
| 116 | 3.00e-2 | 2 | 3.04e-2 | 3.05e-2 | 3.04e-2 | 3.05e-2 | 143 | +2.26e-5 | +1.49e-4 | +8.57e-5 | -6.25e-5 |
| 117 | 3.00e-2 | 1 | 3.45e-2 | 3.45e-2 | 3.45e-2 | 3.45e-2 | 212 | +5.79e-4 | +5.79e-4 | +5.79e-4 | +1.70e-6 |
| 118 | 3.00e-2 | 2 | 3.01e-2 | 3.24e-2 | 3.12e-2 | 3.01e-2 | 134 | -5.36e-4 | -3.78e-4 | -4.57e-4 | -8.62e-5 |
| 119 | 3.00e-2 | 2 | 3.09e-2 | 3.15e-2 | 3.12e-2 | 3.09e-2 | 134 | -1.34e-4 | +2.84e-4 | +7.48e-5 | -5.77e-5 |
| 120 | 3.00e-2 | 2 | 3.12e-2 | 3.36e-2 | 3.24e-2 | 3.12e-2 | 134 | -5.58e-4 | +5.19e-4 | -1.93e-5 | -5.58e-5 |
| 121 | 3.00e-2 | 2 | 3.24e-2 | 3.51e-2 | 3.37e-2 | 3.24e-2 | 134 | -5.88e-4 | +6.75e-4 | +4.32e-5 | -4.33e-5 |
| 122 | 3.00e-2 | 1 | 3.49e-2 | 3.49e-2 | 3.49e-2 | 3.49e-2 | 156 | +4.72e-4 | +4.72e-4 | +4.72e-4 | +8.23e-6 |
| 123 | 3.00e-2 | 2 | 3.30e-2 | 4.33e-2 | 3.81e-2 | 4.33e-2 | 289 | -4.17e-4 | +9.40e-4 | +2.62e-4 | +6.32e-5 |
| 125 | 3.00e-2 | 2 | 4.68e-2 | 5.10e-2 | 4.89e-2 | 4.68e-2 | 264 | -3.28e-4 | +4.46e-4 | +5.93e-5 | +5.86e-5 |
| 127 | 3.00e-2 | 1 | 5.40e-2 | 5.40e-2 | 5.40e-2 | 5.40e-2 | 381 | +3.74e-4 | +3.74e-4 | +3.74e-4 | +9.01e-5 |
| 128 | 3.00e-2 | 1 | 4.85e-2 | 4.85e-2 | 4.85e-2 | 4.85e-2 | 264 | -4.04e-4 | -4.04e-4 | -4.04e-4 | +4.07e-5 |
| 129 | 3.00e-2 | 1 | 4.87e-2 | 4.87e-2 | 4.87e-2 | 4.87e-2 | 270 | +1.65e-5 | +1.65e-5 | +1.65e-5 | +3.83e-5 |
| 130 | 3.00e-2 | 1 | 5.14e-2 | 5.14e-2 | 5.14e-2 | 5.14e-2 | 315 | +1.69e-4 | +1.69e-4 | +1.69e-4 | +5.14e-5 |
| 131 | 3.00e-2 | 1 | 5.12e-2 | 5.12e-2 | 5.12e-2 | 5.12e-2 | 279 | -1.19e-5 | -1.19e-5 | -1.19e-5 | +4.50e-5 |
| 132 | 3.00e-2 | 1 | 5.41e-2 | 5.41e-2 | 5.41e-2 | 5.41e-2 | 301 | +1.84e-4 | +1.84e-4 | +1.84e-4 | +5.90e-5 |
| 133 | 3.00e-2 | 1 | 5.33e-2 | 5.33e-2 | 5.33e-2 | 5.33e-2 | 301 | -5.17e-5 | -5.17e-5 | -5.17e-5 | +4.79e-5 |
| 134 | 3.00e-2 | 1 | 5.62e-2 | 5.62e-2 | 5.62e-2 | 5.62e-2 | 327 | +1.61e-4 | +1.61e-4 | +1.61e-4 | +5.92e-5 |
| 135 | 3.00e-2 | 1 | 5.59e-2 | 5.59e-2 | 5.59e-2 | 5.59e-2 | 310 | -1.78e-5 | -1.78e-5 | -1.78e-5 | +5.15e-5 |
| 137 | 3.00e-2 | 2 | 5.52e-2 | 5.97e-2 | 5.74e-2 | 5.52e-2 | 255 | -3.05e-4 | +1.87e-4 | -5.93e-5 | +2.80e-5 |
| 139 | 3.00e-2 | 2 | 5.66e-2 | 5.83e-2 | 5.75e-2 | 5.66e-2 | 255 | -1.16e-4 | +1.83e-4 | +3.36e-5 | +2.76e-5 |
| 141 | 3.00e-2 | 2 | 5.90e-2 | 6.37e-2 | 6.14e-2 | 5.90e-2 | 270 | -2.86e-4 | +3.47e-4 | +3.02e-5 | +2.49e-5 |
| 143 | 3.00e-2 | 1 | 6.40e-2 | 6.40e-2 | 6.40e-2 | 6.40e-2 | 346 | +2.38e-4 | +2.38e-4 | +2.38e-4 | +4.62e-5 |
| 144 | 3.00e-2 | 1 | 6.42e-2 | 6.42e-2 | 6.42e-2 | 6.42e-2 | 327 | +4.97e-6 | +4.97e-6 | +4.97e-6 | +4.21e-5 |
| 145 | 3.00e-2 | 1 | 6.21e-2 | 6.21e-2 | 6.21e-2 | 6.21e-2 | 292 | -1.09e-4 | -1.09e-4 | -1.09e-4 | +2.70e-5 |
| 146 | 3.00e-2 | 1 | 6.45e-2 | 6.45e-2 | 6.45e-2 | 6.45e-2 | 300 | +1.25e-4 | +1.25e-4 | +1.25e-4 | +3.68e-5 |
| 147 | 3.00e-2 | 1 | 6.45e-2 | 6.45e-2 | 6.45e-2 | 6.45e-2 | 317 | +1.21e-6 | +1.21e-6 | +1.21e-6 | +3.33e-5 |
| 148 | 3.00e-2 | 1 | 6.18e-2 | 6.18e-2 | 6.18e-2 | 6.18e-2 | 278 | -1.54e-4 | -1.54e-4 | -1.54e-4 | +1.45e-5 |
| 149 | 3.00e-2 | 1 | 6.31e-2 | 6.31e-2 | 6.31e-2 | 6.31e-2 | 252 | +7.83e-5 | +7.83e-5 | +7.83e-5 | +2.09e-5 |
| 150 | 3.00e-3 | 1 | 6.32e-2 | 6.32e-2 | 6.32e-2 | 6.32e-2 | 256 | +1.02e-5 | +1.02e-5 | +1.02e-5 | +1.99e-5 |
| 151 | 3.00e-3 | 1 | 3.26e-2 | 3.26e-2 | 3.26e-2 | 3.26e-2 | 282 | -2.35e-3 | -2.35e-3 | -2.35e-3 | -2.17e-4 |
| 152 | 3.00e-3 | 1 | 1.67e-2 | 1.67e-2 | 1.67e-2 | 1.67e-2 | 271 | -2.48e-3 | -2.48e-3 | -2.48e-3 | -4.43e-4 |
| 153 | 3.00e-3 | 1 | 1.01e-2 | 1.01e-2 | 1.01e-2 | 1.01e-2 | 296 | -1.69e-3 | -1.69e-3 | -1.69e-3 | -5.67e-4 |
| 154 | 3.00e-3 | 1 | 7.31e-3 | 7.31e-3 | 7.31e-3 | 7.31e-3 | 287 | -1.14e-3 | -1.14e-3 | -1.14e-3 | -6.24e-4 |
| 155 | 3.00e-3 | 1 | 6.24e-3 | 6.24e-3 | 6.24e-3 | 6.24e-3 | 258 | -6.09e-4 | -6.09e-4 | -6.09e-4 | -6.23e-4 |
| 156 | 3.00e-3 | 1 | 6.05e-3 | 6.05e-3 | 6.05e-3 | 6.05e-3 | 245 | -1.30e-4 | -1.30e-4 | -1.30e-4 | -5.73e-4 |
| 157 | 3.00e-3 | 2 | 5.83e-3 | 5.88e-3 | 5.85e-3 | 5.83e-3 | 223 | -1.16e-4 | -4.43e-5 | -8.02e-5 | -4.79e-4 |
| 159 | 3.00e-3 | 2 | 6.33e-3 | 6.71e-3 | 6.52e-3 | 6.33e-3 | 247 | -2.37e-4 | +4.38e-4 | +1.00e-4 | -3.72e-4 |
| 161 | 3.00e-3 | 2 | 5.95e-3 | 6.13e-3 | 6.04e-3 | 6.13e-3 | 243 | -2.55e-4 | +1.18e-4 | -6.85e-5 | -3.13e-4 |
| 162 | 3.00e-3 | 1 | 6.25e-3 | 6.25e-3 | 6.25e-3 | 6.25e-3 | 249 | +8.20e-5 | +8.20e-5 | +8.20e-5 | -2.73e-4 |
| 163 | 3.00e-3 | 1 | 6.39e-3 | 6.39e-3 | 6.39e-3 | 6.39e-3 | 268 | +8.17e-5 | +8.17e-5 | +8.17e-5 | -2.38e-4 |
| 164 | 3.00e-3 | 1 | 6.09e-3 | 6.09e-3 | 6.09e-3 | 6.09e-3 | 233 | -2.08e-4 | -2.08e-4 | -2.08e-4 | -2.35e-4 |
| 165 | 3.00e-3 | 1 | 6.15e-3 | 6.15e-3 | 6.15e-3 | 6.15e-3 | 262 | +3.92e-5 | +3.92e-5 | +3.92e-5 | -2.07e-4 |
| 166 | 3.00e-3 | 1 | 6.33e-3 | 6.33e-3 | 6.33e-3 | 6.33e-3 | 267 | +1.06e-4 | +1.06e-4 | +1.06e-4 | -1.76e-4 |
| 167 | 3.00e-3 | 1 | 6.15e-3 | 6.15e-3 | 6.15e-3 | 6.15e-3 | 244 | -1.13e-4 | -1.13e-4 | -1.13e-4 | -1.70e-4 |
| 168 | 3.00e-3 | 2 | 6.15e-3 | 6.53e-3 | 6.34e-3 | 6.15e-3 | 225 | -2.70e-4 | +2.33e-4 | -1.83e-5 | -1.44e-4 |
| 170 | 3.00e-3 | 2 | 6.30e-3 | 6.85e-3 | 6.57e-3 | 6.30e-3 | 225 | -3.72e-4 | +3.51e-4 | -1.05e-5 | -1.22e-4 |
| 171 | 3.00e-3 | 1 | 6.42e-3 | 6.42e-3 | 6.42e-3 | 6.42e-3 | 251 | +7.72e-5 | +7.72e-5 | +7.72e-5 | -1.02e-4 |
| 172 | 3.00e-3 | 1 | 6.41e-3 | 6.41e-3 | 6.41e-3 | 6.41e-3 | 240 | -6.42e-6 | -6.42e-6 | -6.42e-6 | -9.24e-5 |
| 173 | 3.00e-3 | 1 | 6.12e-3 | 6.12e-3 | 6.12e-3 | 6.12e-3 | 230 | -1.99e-4 | -1.99e-4 | -1.99e-4 | -1.03e-4 |
| 174 | 3.00e-3 | 1 | 6.19e-3 | 6.19e-3 | 6.19e-3 | 6.19e-3 | 250 | +4.02e-5 | +4.02e-5 | +4.02e-5 | -8.88e-5 |
| 175 | 3.00e-3 | 2 | 6.13e-3 | 6.33e-3 | 6.23e-3 | 6.13e-3 | 214 | -1.51e-4 | +1.00e-4 | -2.52e-5 | -7.79e-5 |
| 176 | 3.00e-3 | 1 | 6.50e-3 | 6.50e-3 | 6.50e-3 | 6.50e-3 | 266 | +2.22e-4 | +2.22e-4 | +2.22e-4 | -4.80e-5 |
| 177 | 3.00e-3 | 1 | 6.49e-3 | 6.49e-3 | 6.49e-3 | 6.49e-3 | 259 | -3.92e-6 | -3.92e-6 | -3.92e-6 | -4.36e-5 |
| 178 | 3.00e-3 | 1 | 6.19e-3 | 6.19e-3 | 6.19e-3 | 6.19e-3 | 211 | -2.25e-4 | -2.25e-4 | -2.25e-4 | -6.17e-5 |
| 179 | 3.00e-3 | 1 | 6.32e-3 | 6.32e-3 | 6.32e-3 | 6.32e-3 | 233 | +9.20e-5 | +9.20e-5 | +9.20e-5 | -4.64e-5 |
| 180 | 3.00e-3 | 1 | 6.53e-3 | 6.53e-3 | 6.53e-3 | 6.53e-3 | 239 | +1.34e-4 | +1.34e-4 | +1.34e-4 | -2.83e-5 |
| 181 | 3.00e-3 | 2 | 6.07e-3 | 6.47e-3 | 6.27e-3 | 6.07e-3 | 199 | -3.22e-4 | -4.06e-5 | -1.81e-4 | -5.88e-5 |
| 182 | 3.00e-3 | 1 | 6.56e-3 | 6.56e-3 | 6.56e-3 | 6.56e-3 | 236 | +3.29e-4 | +3.29e-4 | +3.29e-4 | -2.00e-5 |
| 183 | 3.00e-3 | 2 | 5.96e-3 | 6.61e-3 | 6.28e-3 | 5.96e-3 | 189 | -5.48e-4 | +3.17e-5 | -2.58e-4 | -6.82e-5 |
| 185 | 3.00e-3 | 2 | 6.05e-3 | 6.68e-3 | 6.37e-3 | 6.05e-3 | 184 | -5.43e-4 | +4.24e-4 | -5.93e-5 | -7.13e-5 |
| 186 | 3.00e-3 | 1 | 6.32e-3 | 6.32e-3 | 6.32e-3 | 6.32e-3 | 225 | +1.96e-4 | +1.96e-4 | +1.96e-4 | -4.46e-5 |
| 187 | 3.00e-3 | 1 | 6.26e-3 | 6.26e-3 | 6.26e-3 | 6.26e-3 | 211 | -4.36e-5 | -4.36e-5 | -4.36e-5 | -4.45e-5 |
| 188 | 3.00e-3 | 2 | 5.99e-3 | 6.58e-3 | 6.29e-3 | 5.99e-3 | 184 | -5.15e-4 | +2.16e-4 | -1.50e-4 | -6.82e-5 |
| 189 | 3.00e-3 | 1 | 6.30e-3 | 6.30e-3 | 6.30e-3 | 6.30e-3 | 227 | +2.20e-4 | +2.20e-4 | +2.20e-4 | -3.94e-5 |
| 190 | 3.00e-3 | 1 | 6.34e-3 | 6.34e-3 | 6.34e-3 | 6.34e-3 | 230 | +3.23e-5 | +3.23e-5 | +3.23e-5 | -3.22e-5 |
| 191 | 3.00e-3 | 2 | 6.03e-3 | 6.22e-3 | 6.12e-3 | 6.03e-3 | 184 | -1.61e-4 | -9.77e-5 | -1.29e-4 | -5.09e-5 |
| 192 | 3.00e-3 | 1 | 6.18e-3 | 6.18e-3 | 6.18e-3 | 6.18e-3 | 212 | +1.16e-4 | +1.16e-4 | +1.16e-4 | -3.43e-5 |
| 193 | 3.00e-3 | 1 | 6.60e-3 | 6.60e-3 | 6.60e-3 | 6.60e-3 | 231 | +2.83e-4 | +2.83e-4 | +2.83e-4 | -2.57e-6 |
| 194 | 3.00e-3 | 2 | 6.08e-3 | 6.70e-3 | 6.39e-3 | 6.08e-3 | 183 | -5.26e-4 | +6.47e-5 | -2.30e-4 | -4.88e-5 |
| 195 | 3.00e-3 | 1 | 6.61e-3 | 6.61e-3 | 6.61e-3 | 6.61e-3 | 243 | +3.39e-4 | +3.39e-4 | +3.39e-4 | -1.00e-5 |
| 196 | 3.00e-3 | 2 | 5.89e-3 | 6.45e-3 | 6.17e-3 | 5.89e-3 | 161 | -5.62e-4 | -1.17e-4 | -3.40e-4 | -7.49e-5 |
| 197 | 3.00e-3 | 1 | 6.50e-3 | 6.50e-3 | 6.50e-3 | 6.50e-3 | 208 | +4.75e-4 | +4.75e-4 | +4.75e-4 | -1.99e-5 |
| 198 | 3.00e-3 | 1 | 5.73e-3 | 5.73e-3 | 5.73e-3 | 5.73e-3 | 146 | -8.62e-4 | -8.62e-4 | -8.62e-4 | -1.04e-4 |
| 199 | 3.00e-3 | 1 | 7.45e-3 | 7.45e-3 | 7.45e-3 | 7.45e-3 | 340 | +7.72e-4 | +7.72e-4 | +7.72e-4 | -1.65e-5 |

