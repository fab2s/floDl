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
| cpu-async | 0.059894 | 0.9186 | +0.0061 | 1822.5 | 307 | 87.2 | 100% | 100% | 100% | 12.8 |

## Best Mode per Model

| Model | Best Eval | Mode | Fastest (within 2% of solo-0) | Mode |
|-------|-----------|------|-------------------------------|------|
| resnet-graph | 0.9186 | cpu-async | - | - |

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
| cpu-async | 2.0616 | 0.8074 | 0.6343 | 0.5751 | 0.5364 | 0.5193 | 0.5035 | 0.4858 | 0.4882 | 0.4840 | 0.2123 | 0.1715 | 0.1544 | 0.1357 | 0.1340 | 0.0753 | 0.0693 | 0.0643 | 0.0605 | 0.0599 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | cpu-async | 0 | cuda:0 | 0.3968 | 2.7 |
| resnet-graph | cpu-async | 1 | cuda:1 | 0.3058 | 3.6 |
| resnet-graph | cpu-async | 2 | cuda:2 | 0.2973 | 3.6 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) | GPU2 Peak (MB) | GPU2 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|---------------|---------------|
| resnet-graph | cpu-async | 358 | 357 | 396 | 393 | 386 | 378 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| cpu-async | gpu1 | 1702.0 | 2.5 | epoch-boundary(186) |
| cpu-async | gpu2 | 1702.2 | 2.3 | epoch-boundary(186) |
| cpu-async | gpu2 | 1722.0 | 0.8 | epoch-boundary(188) |
| cpu-async | gpu1 | 1040.0 | 0.7 | epoch-boundary(113) |
| cpu-async | gpu1 | 1721.9 | 0.7 | epoch-boundary(188) |
| cpu-async | gpu1 | 1013.2 | 0.7 | epoch-boundary(110) |
| cpu-async | gpu2 | 1013.1 | 0.6 | epoch-boundary(110) |
| cpu-async | gpu2 | 1040.1 | 0.6 | epoch-boundary(113) |
| cpu-async | gpu1 | 394.4 | 0.6 | epoch-boundary(42) |
| cpu-async | gpu2 | 394.4 | 0.6 | epoch-boundary(42) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | cpu-async | gpu0 | 0.0s | 0.0s | 0.6s | 0.0s | 0.6s |
| resnet-graph | cpu-async | gpu1 | 5.3s | 0.0s | 0.0s | 0.0s | 6.1s |
| resnet-graph | cpu-async | gpu2 | 5.5s | 0.0s | 0.0s | 0.0s | 6.1s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | cpu-async | 263 | 0 | 307 | 87.2 | 6550/10047 | 307 | 87.2 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | cpu-async | 179.0 | 9.8% |

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
| resnet-graph | cpu-async | 189 | 307 | 0 | 7.62e-3 | -9.28e-6 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | cpu-async | 0 | 307 | 9.94e-2 | 8.83e-2 | 3.21e-3 | 4.81e-1 | 20.8 | -2.32e-4 | 1.78e-3 |
| resnet-graph | cpu-async | 1 | 307 | 1.01e-1 | 9.17e-2 | 3.18e-3 | 5.26e-1 | 34.5 | -2.58e-4 | 2.17e-3 |
| resnet-graph | cpu-async | 2 | 307 | 1.01e-1 | 9.06e-2 | 3.23e-3 | 5.72e-1 | 44.6 | -2.72e-4 | 2.17e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | cpu-async | rank0 ↔ rank1 | +0.9963 |
| resnet-graph | cpu-async | rank0 ↔ rank2 | +0.9958 |
| resnet-graph | cpu-async | rank1 ↔ rank2 | +0.9938 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | cpu-async | 44 (0,2,3,4,5,6,7,8…145,148) | 0 (—) | — | 0,2,3,4,5,6,7,8…145,148 | — |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | cpu-async | 0e0 | 3 | 8 | 8 |
| resnet-graph | cpu-async | 0e0 | 5 | 3 | 3 |
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
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 158 | +0.272 |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | 57 | +0.244 |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | 88 | +0.054 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | cpu-async | 305 | -0.028 | 188 | +0.362 | +0.525 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | cpu-async | 306 | 3.51e1–8.11e1 | 6.46e1 | 3.90e-3 | 6.56e-3 | 7.65e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 160 | 39–78015 | +9.926e-6 | 0.431 | +1.019e-5 | 0.452 | 95 | +5.177e-6 | 0.339 | 32–1049 | +9.751e-4 | 0.677 |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 150 | 886–78015 | +1.011e-5 | 0.573 | +1.016e-5 | 0.581 | 94 | +4.942e-6 | 0.319 | 90–1049 | +1.024e-3 | 0.922 |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | 58 | 78777–117002 | +1.349e-5 | 0.312 | +1.337e-5 | 0.310 | 46 | +1.210e-5 | 0.231 | 448–962 | -9.520e-4 | 0.126 |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | 89 | 117538–156030 | -3.822e-6 | 0.012 | -3.664e-6 | 0.011 | 48 | -2.832e-7 | 0.000 | 183–937 | +1.140e-3 | 0.402 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | +9.751e-4 | r0: +9.567e-4, r1: +9.805e-4, r2: +9.918e-4 | r0: 0.689, r1: 0.650, r2: 0.681 | 1.04× | ✓ |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +1.024e-3 | r0: +1.010e-3, r1: +1.037e-3, r2: +1.024e-3 | r0: 0.916, r1: 0.915, r2: 0.919 | 1.03× | ✓ |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | -9.520e-4 | r0: -1.014e-3, r1: -9.189e-4, r2: -9.248e-4 | r0: 0.143, r1: 0.121, r2: 0.113 | 1.10× | ✓ |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | +1.140e-3 | r0: +1.128e-3, r1: +1.129e-3, r2: +1.163e-3 | r0: 0.400, r1: 0.404, r2: 0.396 | 1.03× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | cpu-async | `█▇██████████████████████▅▄▅▅▅▅▅▅▅▅▅▄▁▁▁▁▁▁▁▁▁▂▂▂` | `▁▇▇▇████████████████████▇██████████▇▇▇█████▇████` |

### Per-Epoch Detail: resnet-graph / cpu-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 10 | 6.83e-2 | 5.72e-1 | 1.90e-1 | 8.85e-2 | 53 | -7.26e-2 | +4.78e-3 | -1.81e-2 | -1.36e-2 |
| 1 | 3.00e-1 | 6 | 8.36e-2 | 1.33e-1 | 9.98e-2 | 8.36e-2 | 35 | -3.38e-3 | +4.24e-3 | -9.21e-4 | -6.20e-3 |
| 2 | 3.00e-1 | 7 | 9.18e-2 | 1.30e-1 | 1.02e-1 | 1.03e-1 | 35 | -9.20e-3 | +5.39e-3 | -1.09e-4 | -2.61e-3 |
| 3 | 3.00e-1 | 7 | 9.82e-2 | 1.46e-1 | 1.10e-1 | 1.09e-1 | 38 | -1.21e-2 | +4.08e-3 | -8.05e-4 | -1.50e-3 |
| 4 | 3.00e-1 | 6 | 1.09e-1 | 1.51e-1 | 1.21e-1 | 1.20e-1 | 38 | -7.01e-3 | +4.25e-3 | -3.01e-4 | -8.82e-4 |
| 5 | 3.00e-1 | 7 | 1.08e-1 | 1.54e-1 | 1.20e-1 | 1.23e-1 | 44 | -8.27e-3 | +3.45e-3 | -3.87e-4 | -5.15e-4 |
| 6 | 3.00e-1 | 9 | 1.07e-1 | 1.61e-1 | 1.16e-1 | 1.07e-1 | 35 | -1.03e-2 | +3.21e-3 | -8.83e-4 | -6.60e-4 |
| 7 | 3.00e-1 | 4 | 1.03e-1 | 1.58e-1 | 1.17e-1 | 1.06e-1 | 35 | -1.30e-2 | +4.51e-3 | -1.89e-3 | -1.07e-3 |
| 8 | 3.00e-1 | 6 | 1.05e-1 | 1.47e-1 | 1.17e-1 | 1.09e-1 | 38 | -9.56e-3 | +4.29e-3 | -7.58e-4 | -9.19e-4 |
| 9 | 3.00e-1 | 2 | 1.00e-1 | 2.11e-1 | 1.56e-1 | 2.11e-1 | 236 | -2.29e-3 | +3.15e-3 | +4.26e-4 | -6.36e-4 |
| 10 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 239 | -1.52e-5 | -1.52e-5 | -1.52e-5 | -5.74e-4 |
| 11 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 231 | -1.28e-4 | -1.28e-4 | -1.28e-4 | -5.29e-4 |
| 12 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 258 | +6.67e-5 | +6.67e-5 | +6.67e-5 | -4.70e-4 |
| 13 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 287 | +4.19e-5 | +4.19e-5 | +4.19e-5 | -4.18e-4 |
| 14 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 317 | +1.41e-4 | +1.41e-4 | +1.41e-4 | -3.62e-4 |
| 15 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 297 | -8.08e-5 | -8.08e-5 | -8.08e-5 | -3.34e-4 |
| 17 | 3.00e-1 | 2 | 2.08e-1 | 2.37e-1 | 2.22e-1 | 2.08e-1 | 274 | -4.71e-4 | +2.57e-4 | -1.07e-4 | -2.95e-4 |
| 19 | 3.00e-1 | 1 | 2.31e-1 | 2.31e-1 | 2.31e-1 | 2.31e-1 | 368 | +2.87e-4 | +2.87e-4 | +2.87e-4 | -2.36e-4 |
| 20 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 309 | -2.56e-4 | -2.56e-4 | -2.56e-4 | -2.38e-4 |
| 21 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 273 | -9.72e-5 | -9.72e-5 | -9.72e-5 | -2.24e-4 |
| 22 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 267 | -1.17e-5 | -1.17e-5 | -1.17e-5 | -2.03e-4 |
| 23 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 282 | +9.50e-5 | +9.50e-5 | +9.50e-5 | -1.73e-4 |
| 24 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 292 | +7.26e-5 | +7.26e-5 | +7.26e-5 | -1.49e-4 |
| 25 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 286 | -3.35e-5 | -3.35e-5 | -3.35e-5 | -1.37e-4 |
| 26 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 273 | -7.20e-5 | -7.20e-5 | -7.20e-5 | -1.31e-4 |
| 27 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 274 | -8.08e-7 | -8.08e-7 | -8.08e-7 | -1.18e-4 |
| 28 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 302 | +1.18e-4 | +1.18e-4 | +1.18e-4 | -9.40e-5 |
| 29 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 257 | -2.55e-4 | -2.55e-4 | -2.55e-4 | -1.10e-4 |
| 30 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 245 | -1.81e-5 | -1.81e-5 | -1.81e-5 | -1.01e-4 |
| 31 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 274 | +1.92e-4 | +1.92e-4 | +1.92e-4 | -7.16e-5 |
| 32 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 252 | -1.34e-4 | -1.34e-4 | -1.34e-4 | -7.78e-5 |
| 33 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 261 | +5.96e-5 | +5.96e-5 | +5.96e-5 | -6.41e-5 |
| 34 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 278 | +9.87e-5 | +9.87e-5 | +9.87e-5 | -4.78e-5 |
| 35 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 284 | +4.64e-5 | +4.64e-5 | +4.64e-5 | -3.84e-5 |
| 36 | 3.00e-1 | 1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 285 | +2.91e-5 | +2.91e-5 | +2.91e-5 | -3.17e-5 |
| 37 | 3.00e-1 | 1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 295 | +4.13e-5 | +4.13e-5 | +4.13e-5 | -2.44e-5 |
| 38 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 242 | -3.76e-4 | -3.76e-4 | -3.76e-4 | -5.95e-5 |
| 39 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 244 | -2.23e-5 | -2.23e-5 | -2.23e-5 | -5.58e-5 |
| 40 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 235 | +6.21e-5 | +6.21e-5 | +6.21e-5 | -4.40e-5 |
| 41 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 247 | +5.54e-5 | +5.54e-5 | +5.54e-5 | -3.41e-5 |
| 42 | 3.00e-1 | 2 | 2.08e-1 | 2.11e-1 | 2.10e-1 | 2.08e-1 | 243 | -5.02e-5 | +2.19e-5 | -1.42e-5 | -3.06e-5 |
| 43 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 264 | +1.94e-4 | +1.94e-4 | +1.94e-4 | -8.20e-6 |
| 44 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 251 | -1.45e-4 | -1.45e-4 | -1.45e-4 | -2.18e-5 |
| 45 | 3.00e-1 | 1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 273 | +1.79e-4 | +1.79e-4 | +1.79e-4 | -1.80e-6 |
| 46 | 3.00e-1 | 1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 295 | +5.65e-5 | +5.65e-5 | +5.65e-5 | +4.03e-6 |
| 47 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 255 | -2.24e-4 | -2.24e-4 | -2.24e-4 | -1.88e-5 |
| 48 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 252 | -6.33e-5 | -6.33e-5 | -6.33e-5 | -2.32e-5 |
| 49 | 3.00e-1 | 1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 291 | +2.57e-4 | +2.57e-4 | +2.57e-4 | +4.75e-6 |
| 50 | 3.00e-1 | 1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 267 | -6.26e-5 | -6.26e-5 | -6.26e-5 | -1.99e-6 |
| 51 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 246 | -1.84e-4 | -1.84e-4 | -1.84e-4 | -2.01e-5 |
| 52 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 251 | -4.46e-5 | -4.46e-5 | -4.46e-5 | -2.26e-5 |
| 53 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 251 | +1.01e-4 | +1.01e-4 | +1.01e-4 | -1.03e-5 |
| 54 | 3.00e-1 | 2 | 2.05e-1 | 2.10e-1 | 2.08e-1 | 2.05e-1 | 207 | -1.21e-4 | -1.04e-4 | -1.12e-4 | -2.98e-5 |
| 55 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 253 | +1.31e-4 | +1.31e-4 | +1.31e-4 | -1.37e-5 |
| 56 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 243 | -3.84e-6 | -3.84e-6 | -3.84e-6 | -1.27e-5 |
| 57 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 277 | +1.39e-4 | +1.39e-4 | +1.39e-4 | +2.46e-6 |
| 58 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 256 | -6.63e-5 | -6.63e-5 | -6.63e-5 | -4.42e-6 |
| 59 | 3.00e-1 | 2 | 2.04e-1 | 2.23e-1 | 2.14e-1 | 2.04e-1 | 207 | -4.24e-4 | +1.14e-4 | -1.55e-4 | -3.58e-5 |
| 61 | 3.00e-1 | 2 | 1.95e-1 | 2.39e-1 | 2.17e-1 | 1.95e-1 | 187 | -1.10e-3 | +4.76e-4 | -3.11e-4 | -9.60e-5 |
| 62 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 237 | +3.58e-4 | +3.58e-4 | +3.58e-4 | -5.06e-5 |
| 63 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 231 | -1.81e-5 | -1.81e-5 | -1.81e-5 | -4.74e-5 |
| 64 | 3.00e-1 | 2 | 1.95e-1 | 2.10e-1 | 2.02e-1 | 1.95e-1 | 187 | -3.98e-4 | -3.00e-5 | -2.14e-4 | -8.09e-5 |
| 65 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 198 | +1.18e-4 | +1.18e-4 | +1.18e-4 | -6.10e-5 |
| 66 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 216 | +2.24e-4 | +2.24e-4 | +2.24e-4 | -3.25e-5 |
| 67 | 3.00e-1 | 2 | 1.99e-1 | 2.01e-1 | 2.00e-1 | 2.01e-1 | 198 | -2.45e-4 | +5.77e-5 | -9.38e-5 | -4.26e-5 |
| 68 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 233 | +2.64e-4 | +2.64e-4 | +2.64e-4 | -1.19e-5 |
| 69 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 219 | -1.77e-4 | -1.77e-4 | -1.77e-4 | -2.84e-5 |
| 70 | 3.00e-1 | 2 | 2.03e-1 | 2.13e-1 | 2.08e-1 | 2.03e-1 | 198 | -2.31e-4 | +1.36e-4 | -4.76e-5 | -3.39e-5 |
| 71 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 213 | +2.29e-5 | +2.29e-5 | +2.29e-5 | -2.82e-5 |
| 72 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 237 | +3.17e-4 | +3.17e-4 | +3.17e-4 | +6.29e-6 |
| 73 | 3.00e-1 | 2 | 1.94e-1 | 2.13e-1 | 2.04e-1 | 1.94e-1 | 185 | -5.13e-4 | -1.39e-4 | -3.26e-4 | -5.87e-5 |
| 74 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 250 | +4.55e-4 | +4.55e-4 | +4.55e-4 | -7.38e-6 |
| 75 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 230 | -2.68e-5 | -2.68e-5 | -2.68e-5 | -9.32e-6 |
| 76 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 227 | -1.20e-4 | -1.20e-4 | -1.20e-4 | -2.04e-5 |
| 77 | 3.00e-1 | 2 | 1.98e-1 | 2.13e-1 | 2.06e-1 | 1.98e-1 | 198 | -3.77e-4 | +6.97e-5 | -1.54e-4 | -4.80e-5 |
| 78 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 205 | +1.73e-4 | +1.73e-4 | +1.73e-4 | -2.59e-5 |
| 79 | 3.00e-1 | 1 | 1.87e-1 | 1.87e-1 | 1.87e-1 | 1.87e-1 | 152 | -6.06e-4 | -6.06e-4 | -6.06e-4 | -8.39e-5 |
| 80 | 3.00e-1 | 1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 303 | +5.96e-4 | +5.96e-4 | +5.96e-4 | -1.59e-5 |
| 81 | 3.00e-1 | 1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 262 | +3.01e-5 | +3.01e-5 | +3.01e-5 | -1.13e-5 |
| 82 | 3.00e-1 | 1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 268 | -4.74e-5 | -4.74e-5 | -4.74e-5 | -1.49e-5 |
| 83 | 3.00e-1 | 1 | 2.37e-1 | 2.37e-1 | 2.37e-1 | 2.37e-1 | 316 | +1.94e-4 | +1.94e-4 | +1.94e-4 | +5.95e-6 |
| 84 | 3.00e-1 | 1 | 2.28e-1 | 2.28e-1 | 2.28e-1 | 2.28e-1 | 298 | -1.35e-4 | -1.35e-4 | -1.35e-4 | -8.13e-6 |
| 85 | 3.00e-1 | 1 | 2.44e-1 | 2.44e-1 | 2.44e-1 | 2.44e-1 | 366 | +1.92e-4 | +1.92e-4 | +1.92e-4 | +1.19e-5 |
| 87 | 3.00e-1 | 2 | 2.23e-1 | 2.45e-1 | 2.34e-1 | 2.23e-1 | 266 | -3.48e-4 | +5.34e-6 | -1.71e-4 | -2.47e-5 |
| 89 | 3.00e-1 | 1 | 2.41e-1 | 2.41e-1 | 2.41e-1 | 2.41e-1 | 343 | +2.20e-4 | +2.20e-4 | +2.20e-4 | -2.44e-7 |
| 90 | 3.00e-1 | 1 | 2.31e-1 | 2.31e-1 | 2.31e-1 | 2.31e-1 | 320 | -1.24e-4 | -1.24e-4 | -1.24e-4 | -1.26e-5 |
| 91 | 3.00e-1 | 1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 309 | -1.71e-5 | -1.71e-5 | -1.71e-5 | -1.30e-5 |
| 92 | 3.00e-1 | 1 | 2.28e-1 | 2.28e-1 | 2.28e-1 | 2.28e-1 | 291 | -2.68e-5 | -2.68e-5 | -2.68e-5 | -1.44e-5 |
| 93 | 3.00e-1 | 1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 286 | -1.45e-5 | -1.45e-5 | -1.45e-5 | -1.44e-5 |
| 94 | 3.00e-1 | 1 | 2.28e-1 | 2.28e-1 | 2.28e-1 | 2.28e-1 | 294 | +1.52e-5 | +1.52e-5 | +1.52e-5 | -1.15e-5 |
| 95 | 3.00e-1 | 1 | 2.33e-1 | 2.33e-1 | 2.33e-1 | 2.33e-1 | 302 | +6.61e-5 | +6.61e-5 | +6.61e-5 | -3.71e-6 |
| 96 | 3.00e-1 | 1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 287 | -8.98e-5 | -8.98e-5 | -8.98e-5 | -1.23e-5 |
| 97 | 3.00e-1 | 1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 293 | -7.96e-6 | -7.96e-6 | -7.96e-6 | -1.19e-5 |
| 98 | 3.00e-1 | 1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 294 | -4.50e-5 | -4.50e-5 | -4.50e-5 | -1.52e-5 |
| 99 | 3.00e-1 | 1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 261 | -3.24e-6 | -3.24e-6 | -3.24e-6 | -1.40e-5 |
| 100 | 3.00e-2 | 1 | 1.45e-1 | 1.45e-1 | 1.45e-1 | 1.45e-1 | 260 | -1.65e-3 | -1.65e-3 | -1.65e-3 | -1.78e-4 |
| 102 | 3.00e-2 | 2 | 2.81e-2 | 5.04e-2 | 3.92e-2 | 2.81e-2 | 272 | -3.05e-3 | -2.14e-3 | -2.60e-3 | -6.33e-4 |
| 104 | 3.00e-2 | 2 | 2.65e-2 | 2.94e-2 | 2.80e-2 | 2.65e-2 | 253 | -4.05e-4 | +1.33e-4 | -1.36e-4 | -5.42e-4 |
| 106 | 3.00e-2 | 2 | 2.70e-2 | 2.98e-2 | 2.84e-2 | 2.70e-2 | 237 | -4.22e-4 | +3.75e-4 | -2.36e-5 | -4.47e-4 |
| 108 | 3.00e-2 | 2 | 2.90e-2 | 3.18e-2 | 3.04e-2 | 2.90e-2 | 237 | -3.88e-4 | +5.00e-4 | +5.56e-5 | -3.56e-4 |
| 109 | 3.00e-2 | 1 | 3.07e-2 | 3.07e-2 | 3.07e-2 | 3.07e-2 | 256 | +2.28e-4 | +2.28e-4 | +2.28e-4 | -2.98e-4 |
| 110 | 3.00e-2 | 1 | 3.12e-2 | 3.12e-2 | 3.12e-2 | 3.12e-2 | 249 | +6.60e-5 | +6.60e-5 | +6.60e-5 | -2.61e-4 |
| 111 | 3.00e-2 | 1 | 3.26e-2 | 3.26e-2 | 3.26e-2 | 3.26e-2 | 265 | +1.57e-4 | +1.57e-4 | +1.57e-4 | -2.19e-4 |
| 112 | 3.00e-2 | 1 | 3.35e-2 | 3.35e-2 | 3.35e-2 | 3.35e-2 | 277 | +1.01e-4 | +1.01e-4 | +1.01e-4 | -1.87e-4 |
| 113 | 3.00e-2 | 1 | 3.39e-2 | 3.39e-2 | 3.39e-2 | 3.39e-2 | 261 | +4.87e-5 | +4.87e-5 | +4.87e-5 | -1.64e-4 |
| 114 | 3.00e-2 | 1 | 3.38e-2 | 3.38e-2 | 3.38e-2 | 3.38e-2 | 251 | -1.23e-5 | -1.23e-5 | -1.23e-5 | -1.49e-4 |
| 115 | 3.00e-2 | 1 | 3.50e-2 | 3.50e-2 | 3.50e-2 | 3.50e-2 | 258 | +1.37e-4 | +1.37e-4 | +1.37e-4 | -1.20e-4 |
| 116 | 3.00e-2 | 1 | 3.54e-2 | 3.54e-2 | 3.54e-2 | 3.54e-2 | 255 | +4.16e-5 | +4.16e-5 | +4.16e-5 | -1.04e-4 |
| 117 | 3.00e-2 | 1 | 3.81e-2 | 3.81e-2 | 3.81e-2 | 3.81e-2 | 288 | +2.51e-4 | +2.51e-4 | +2.51e-4 | -6.84e-5 |
| 118 | 3.00e-2 | 1 | 3.95e-2 | 3.95e-2 | 3.95e-2 | 3.95e-2 | 291 | +1.25e-4 | +1.25e-4 | +1.25e-4 | -4.90e-5 |
| 119 | 3.00e-2 | 1 | 3.97e-2 | 3.97e-2 | 3.97e-2 | 3.97e-2 | 285 | +1.84e-5 | +1.84e-5 | +1.84e-5 | -4.23e-5 |
| 120 | 3.00e-2 | 1 | 3.93e-2 | 3.93e-2 | 3.93e-2 | 3.93e-2 | 267 | -4.04e-5 | -4.04e-5 | -4.04e-5 | -4.21e-5 |
| 121 | 3.00e-2 | 1 | 3.91e-2 | 3.91e-2 | 3.91e-2 | 3.91e-2 | 252 | -1.73e-5 | -1.73e-5 | -1.73e-5 | -3.96e-5 |
| 122 | 3.00e-2 | 1 | 3.81e-2 | 3.81e-2 | 3.81e-2 | 3.81e-2 | 219 | -1.14e-4 | -1.14e-4 | -1.14e-4 | -4.70e-5 |
| 123 | 3.00e-2 | 2 | 3.90e-2 | 3.91e-2 | 3.91e-2 | 3.91e-2 | 224 | +1.01e-5 | +9.93e-5 | +5.47e-5 | -2.81e-5 |
| 124 | 3.00e-2 | 1 | 4.32e-2 | 4.32e-2 | 4.32e-2 | 4.32e-2 | 253 | +3.91e-4 | +3.91e-4 | +3.91e-4 | +1.38e-5 |
| 125 | 3.00e-2 | 1 | 4.20e-2 | 4.20e-2 | 4.20e-2 | 4.20e-2 | 240 | -1.13e-4 | -1.13e-4 | -1.13e-4 | +1.08e-6 |
| 126 | 3.00e-2 | 1 | 4.55e-2 | 4.55e-2 | 4.55e-2 | 4.55e-2 | 262 | +3.07e-4 | +3.07e-4 | +3.07e-4 | +3.16e-5 |
| 127 | 3.00e-2 | 1 | 4.43e-2 | 4.43e-2 | 4.43e-2 | 4.43e-2 | 248 | -1.10e-4 | -1.10e-4 | -1.10e-4 | +1.74e-5 |
| 128 | 3.00e-2 | 1 | 4.45e-2 | 4.45e-2 | 4.45e-2 | 4.45e-2 | 241 | +1.69e-5 | +1.69e-5 | +1.69e-5 | +1.74e-5 |
| 129 | 3.00e-2 | 2 | 4.14e-2 | 4.62e-2 | 4.38e-2 | 4.14e-2 | 183 | -5.90e-4 | +1.48e-4 | -2.21e-4 | -3.17e-5 |
| 130 | 3.00e-2 | 1 | 4.33e-2 | 4.33e-2 | 4.33e-2 | 4.33e-2 | 219 | +1.98e-4 | +1.98e-4 | +1.98e-4 | -8.75e-6 |
| 131 | 3.00e-2 | 1 | 4.76e-2 | 4.76e-2 | 4.76e-2 | 4.76e-2 | 241 | +3.92e-4 | +3.92e-4 | +3.92e-4 | +3.14e-5 |
| 132 | 3.00e-2 | 1 | 4.86e-2 | 4.86e-2 | 4.86e-2 | 4.86e-2 | 262 | +8.57e-5 | +8.57e-5 | +8.57e-5 | +3.68e-5 |
| 133 | 3.00e-2 | 2 | 4.16e-2 | 4.63e-2 | 4.39e-2 | 4.16e-2 | 173 | -6.25e-4 | -2.22e-4 | -4.23e-4 | -5.26e-5 |
| 134 | 3.00e-2 | 1 | 4.53e-2 | 4.53e-2 | 4.53e-2 | 4.53e-2 | 215 | +4.04e-4 | +4.04e-4 | +4.04e-4 | -6.92e-6 |
| 135 | 3.00e-2 | 2 | 4.45e-2 | 4.87e-2 | 4.66e-2 | 4.45e-2 | 178 | -5.03e-4 | +3.06e-4 | -9.82e-5 | -2.83e-5 |
| 136 | 3.00e-2 | 1 | 4.82e-2 | 4.82e-2 | 4.82e-2 | 4.82e-2 | 218 | +3.64e-4 | +3.64e-4 | +3.64e-4 | +1.09e-5 |
| 137 | 3.00e-2 | 1 | 4.87e-2 | 4.87e-2 | 4.87e-2 | 4.87e-2 | 212 | +4.26e-5 | +4.26e-5 | +4.26e-5 | +1.41e-5 |
| 138 | 3.00e-2 | 1 | 5.26e-2 | 5.26e-2 | 5.26e-2 | 5.26e-2 | 253 | +3.10e-4 | +3.10e-4 | +3.10e-4 | +4.37e-5 |
| 139 | 3.00e-2 | 2 | 4.95e-2 | 5.10e-2 | 5.02e-2 | 4.95e-2 | 205 | -1.46e-4 | -1.41e-4 | -1.44e-4 | +8.09e-6 |
| 140 | 3.00e-2 | 1 | 5.26e-2 | 5.26e-2 | 5.26e-2 | 5.26e-2 | 233 | +2.63e-4 | +2.63e-4 | +2.63e-4 | +3.36e-5 |
| 141 | 3.00e-2 | 1 | 5.31e-2 | 5.31e-2 | 5.31e-2 | 5.31e-2 | 240 | +4.15e-5 | +4.15e-5 | +4.15e-5 | +3.44e-5 |
| 142 | 3.00e-2 | 1 | 5.55e-2 | 5.55e-2 | 5.55e-2 | 5.55e-2 | 239 | +1.87e-4 | +1.87e-4 | +1.87e-4 | +4.97e-5 |
| 143 | 3.00e-2 | 2 | 4.88e-2 | 5.43e-2 | 5.15e-2 | 4.88e-2 | 191 | -5.51e-4 | -1.05e-4 | -3.28e-4 | -2.43e-5 |
| 144 | 3.00e-2 | 1 | 5.36e-2 | 5.36e-2 | 5.36e-2 | 5.36e-2 | 216 | +4.27e-4 | +4.27e-4 | +4.27e-4 | +2.08e-5 |
| 145 | 3.00e-2 | 1 | 5.48e-2 | 5.48e-2 | 5.48e-2 | 5.48e-2 | 224 | +9.97e-5 | +9.97e-5 | +9.97e-5 | +2.87e-5 |
| 146 | 3.00e-2 | 2 | 5.29e-2 | 5.33e-2 | 5.31e-2 | 5.29e-2 | 192 | -1.36e-4 | -3.90e-5 | -8.74e-5 | +7.16e-6 |
| 147 | 3.00e-2 | 1 | 5.79e-2 | 5.79e-2 | 5.79e-2 | 5.79e-2 | 221 | +4.09e-4 | +4.09e-4 | +4.09e-4 | +4.73e-5 |
| 148 | 3.00e-2 | 1 | 6.09e-2 | 6.09e-2 | 6.09e-2 | 6.09e-2 | 241 | +2.12e-4 | +2.12e-4 | +2.12e-4 | +6.38e-5 |
| 149 | 3.00e-2 | 2 | 4.94e-2 | 5.68e-2 | 5.31e-2 | 4.94e-2 | 155 | -9.07e-4 | -3.24e-4 | -6.16e-4 | -6.82e-5 |
| 150 | 3.00e-3 | 1 | 5.15e-2 | 5.15e-2 | 5.15e-2 | 5.15e-2 | 191 | +2.14e-4 | +2.14e-4 | +2.14e-4 | -4.00e-5 |
| 151 | 3.00e-3 | 2 | 6.33e-3 | 1.61e-2 | 1.12e-2 | 6.33e-3 | 147 | -6.36e-3 | -5.98e-3 | -6.17e-3 | -1.21e-3 |
| 152 | 3.00e-3 | 2 | 4.81e-3 | 5.39e-3 | 5.10e-3 | 4.81e-3 | 149 | -8.43e-4 | -7.59e-4 | -8.01e-4 | -1.13e-3 |
| 153 | 3.00e-3 | 1 | 4.94e-3 | 4.94e-3 | 4.94e-3 | 4.94e-3 | 182 | +1.48e-4 | +1.48e-4 | +1.48e-4 | -1.00e-3 |
| 154 | 3.00e-3 | 2 | 4.48e-3 | 5.16e-3 | 4.82e-3 | 4.48e-3 | 147 | -9.62e-4 | +2.31e-4 | -3.65e-4 | -8.87e-4 |
| 155 | 3.00e-3 | 1 | 5.06e-3 | 5.06e-3 | 5.06e-3 | 5.06e-3 | 198 | +6.09e-4 | +6.09e-4 | +6.09e-4 | -7.37e-4 |
| 156 | 3.00e-3 | 2 | 4.85e-3 | 5.35e-3 | 5.10e-3 | 4.85e-3 | 176 | -5.60e-4 | +2.82e-4 | -1.39e-4 | -6.28e-4 |
| 157 | 3.00e-3 | 2 | 4.59e-3 | 4.87e-3 | 4.73e-3 | 4.59e-3 | 151 | -3.89e-4 | +1.77e-5 | -1.86e-4 | -5.46e-4 |
| 158 | 3.00e-3 | 1 | 4.82e-3 | 4.82e-3 | 4.82e-3 | 4.82e-3 | 158 | +3.13e-4 | +3.13e-4 | +3.13e-4 | -4.60e-4 |
| 159 | 3.00e-3 | 2 | 4.84e-3 | 5.09e-3 | 4.97e-3 | 4.84e-3 | 147 | -3.36e-4 | +3.10e-4 | -1.30e-5 | -3.78e-4 |
| 160 | 3.00e-3 | 2 | 4.80e-3 | 5.15e-3 | 4.97e-3 | 4.80e-3 | 162 | -4.35e-4 | +3.09e-4 | -6.27e-5 | -3.22e-4 |
| 161 | 3.00e-3 | 1 | 5.36e-3 | 5.36e-3 | 5.36e-3 | 5.36e-3 | 193 | +5.77e-4 | +5.77e-4 | +5.77e-4 | -2.32e-4 |
| 162 | 3.00e-3 | 2 | 4.93e-3 | 5.35e-3 | 5.14e-3 | 4.93e-3 | 162 | -5.05e-4 | -1.72e-5 | -2.61e-4 | -2.40e-4 |
| 163 | 3.00e-3 | 1 | 5.52e-3 | 5.52e-3 | 5.52e-3 | 5.52e-3 | 200 | +5.71e-4 | +5.71e-4 | +5.71e-4 | -1.59e-4 |
| 164 | 3.00e-3 | 2 | 4.88e-3 | 5.50e-3 | 5.19e-3 | 4.88e-3 | 151 | -7.96e-4 | -1.80e-5 | -4.07e-4 | -2.10e-4 |
| 165 | 3.00e-3 | 2 | 4.56e-3 | 5.33e-3 | 4.95e-3 | 4.56e-3 | 132 | -1.19e-3 | +4.57e-4 | -3.65e-4 | -2.48e-4 |
| 166 | 3.00e-3 | 2 | 4.41e-3 | 5.15e-3 | 4.78e-3 | 4.41e-3 | 122 | -1.26e-3 | +7.07e-4 | -2.78e-4 | -2.63e-4 |
| 167 | 3.00e-3 | 2 | 4.29e-3 | 4.87e-3 | 4.58e-3 | 4.29e-3 | 115 | -1.10e-3 | +6.55e-4 | -2.21e-4 | -2.64e-4 |
| 168 | 3.00e-3 | 2 | 4.42e-3 | 4.75e-3 | 4.58e-3 | 4.42e-3 | 123 | -5.74e-4 | +6.52e-4 | +3.90e-5 | -2.12e-4 |
| 169 | 3.00e-3 | 2 | 4.57e-3 | 4.94e-3 | 4.75e-3 | 4.57e-3 | 122 | -6.40e-4 | +7.19e-4 | +3.94e-5 | -1.71e-4 |
| 170 | 3.00e-3 | 2 | 4.37e-3 | 4.85e-3 | 4.61e-3 | 4.37e-3 | 115 | -9.13e-4 | +4.05e-4 | -2.54e-4 | -1.94e-4 |
| 171 | 3.00e-3 | 2 | 4.25e-3 | 5.17e-3 | 4.71e-3 | 5.17e-3 | 172 | -2.46e-4 | +1.14e-3 | +4.47e-4 | -6.50e-5 |
| 172 | 3.00e-3 | 3 | 4.38e-3 | 5.10e-3 | 4.69e-3 | 4.38e-3 | 115 | -9.20e-4 | -8.20e-5 | -4.72e-4 | -1.78e-4 |
| 173 | 3.00e-3 | 1 | 5.01e-3 | 5.01e-3 | 5.01e-3 | 5.01e-3 | 163 | +8.25e-4 | +8.25e-4 | +8.25e-4 | -7.80e-5 |
| 174 | 3.00e-3 | 2 | 4.31e-3 | 4.84e-3 | 4.57e-3 | 4.31e-3 | 108 | -1.06e-3 | -2.11e-4 | -6.36e-4 | -1.88e-4 |
| 175 | 3.00e-3 | 3 | 4.14e-3 | 4.99e-3 | 4.48e-3 | 4.14e-3 | 98 | -1.37e-3 | +1.00e-3 | -2.55e-4 | -2.19e-4 |
| 176 | 3.00e-3 | 2 | 4.22e-3 | 4.46e-3 | 4.34e-3 | 4.22e-3 | 100 | -5.52e-4 | +5.60e-4 | +4.31e-6 | -1.82e-4 |
| 177 | 3.00e-3 | 3 | 4.14e-3 | 4.74e-3 | 4.36e-3 | 4.14e-3 | 98 | -1.23e-3 | +8.46e-4 | -1.77e-4 | -1.90e-4 |
| 178 | 3.00e-3 | 2 | 4.04e-3 | 4.85e-3 | 4.44e-3 | 4.04e-3 | 94 | -1.94e-3 | +1.04e-3 | -4.54e-4 | -2.55e-4 |
| 179 | 3.00e-3 | 4 | 3.68e-3 | 4.67e-3 | 4.07e-3 | 3.68e-3 | 85 | -1.23e-3 | +1.12e-3 | -3.78e-4 | -3.14e-4 |
| 180 | 3.00e-3 | 2 | 4.07e-3 | 4.45e-3 | 4.26e-3 | 4.07e-3 | 92 | -9.78e-4 | +1.44e-3 | +2.32e-4 | -2.23e-4 |
| 181 | 3.00e-3 | 2 | 3.98e-3 | 4.79e-3 | 4.39e-3 | 3.98e-3 | 89 | -2.07e-3 | +1.12e-3 | -4.73e-4 | -2.86e-4 |
| 182 | 3.00e-3 | 3 | 3.76e-3 | 4.91e-3 | 4.15e-3 | 3.77e-3 | 76 | -3.53e-3 | +1.46e-3 | -6.84e-4 | -4.06e-4 |
| 183 | 3.00e-3 | 4 | 3.42e-3 | 4.57e-3 | 3.82e-3 | 3.47e-3 | 70 | -2.32e-3 | +1.55e-3 | -5.37e-4 | -4.63e-4 |
| 184 | 3.00e-3 | 3 | 3.47e-3 | 4.17e-3 | 3.71e-3 | 3.47e-3 | 70 | -2.56e-3 | +1.79e-3 | -2.86e-4 | -4.32e-4 |
| 185 | 3.00e-3 | 3 | 3.41e-3 | 4.45e-3 | 3.80e-3 | 3.41e-3 | 70 | -3.26e-3 | +2.12e-3 | -5.64e-4 | -4.92e-4 |
| 186 | 3.00e-3 | 2 | 3.63e-3 | 6.39e-3 | 5.01e-3 | 6.39e-3 | 277 | +8.40e-4 | +2.04e-3 | +1.44e-3 | -1.18e-4 |
| 187 | 3.00e-3 | 1 | 6.89e-3 | 6.89e-3 | 6.89e-3 | 6.89e-3 | 279 | +2.71e-4 | +2.71e-4 | +2.71e-4 | -7.94e-5 |
| 188 | 3.00e-3 | 1 | 6.78e-3 | 6.78e-3 | 6.78e-3 | 6.78e-3 | 262 | -6.59e-5 | -6.59e-5 | -6.59e-5 | -7.81e-5 |
| 190 | 3.00e-3 | 2 | 6.98e-3 | 7.57e-3 | 7.27e-3 | 6.98e-3 | 262 | -3.14e-4 | +3.11e-4 | -1.20e-6 | -6.66e-5 |
| 192 | 3.00e-3 | 1 | 7.54e-3 | 7.54e-3 | 7.54e-3 | 7.54e-3 | 328 | +2.36e-4 | +2.36e-4 | +2.36e-4 | -3.63e-5 |
| 193 | 3.00e-3 | 1 | 7.36e-3 | 7.36e-3 | 7.36e-3 | 7.36e-3 | 321 | -7.48e-5 | -7.48e-5 | -7.48e-5 | -4.01e-5 |
| 194 | 3.00e-3 | 1 | 7.13e-3 | 7.13e-3 | 7.13e-3 | 7.13e-3 | 300 | -1.05e-4 | -1.05e-4 | -1.05e-4 | -4.67e-5 |
| 195 | 3.00e-3 | 1 | 7.25e-3 | 7.25e-3 | 7.25e-3 | 7.25e-3 | 312 | +5.45e-5 | +5.45e-5 | +5.45e-5 | -3.65e-5 |
| 196 | 3.00e-3 | 1 | 7.45e-3 | 7.45e-3 | 7.45e-3 | 7.45e-3 | 307 | +8.59e-5 | +8.59e-5 | +8.59e-5 | -2.43e-5 |
| 197 | 3.00e-3 | 1 | 7.14e-3 | 7.14e-3 | 7.14e-3 | 7.14e-3 | 277 | -1.50e-4 | -1.50e-4 | -1.50e-4 | -3.68e-5 |
| 198 | 3.00e-3 | 1 | 7.34e-3 | 7.34e-3 | 7.34e-3 | 7.34e-3 | 284 | +9.53e-5 | +9.53e-5 | +9.53e-5 | -2.36e-5 |
| 199 | 3.00e-3 | 1 | 7.62e-3 | 7.62e-3 | 7.62e-3 | 7.62e-3 | 313 | +1.20e-4 | +1.20e-4 | +1.20e-4 | -9.28e-6 |

