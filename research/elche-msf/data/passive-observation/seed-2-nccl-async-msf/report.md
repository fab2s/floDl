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
| nccl-async | 0.060433 | 0.9132 | +0.0007 | 1927.4 | 508 | 43.5 | 100% | 100% | 100% | 12.8 |

## Best Mode per Model

| Model | Best Eval | Mode | Fastest (within 2% of solo-0) | Mode |
|-------|-----------|------|-------------------------------|------|
| resnet-graph | 0.9132 | nccl-async | - | - |

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
| nccl-async | 1.9004 | 0.7945 | 0.6267 | 0.5609 | 0.5332 | 0.5129 | 0.4891 | 0.4913 | 0.4776 | 0.4701 | 0.2146 | 0.1800 | 0.1593 | 0.1462 | 0.1463 | 0.0808 | 0.0726 | 0.0697 | 0.0611 | 0.0604 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | nccl-async | 0 | cuda:0 | 0.3983 | 2.6 |
| resnet-graph | nccl-async | 1 | cuda:1 | 0.3042 | 3.5 |
| resnet-graph | nccl-async | 2 | cuda:2 | 0.2974 | 3.2 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) | GPU2 Peak (MB) | GPU2 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|---------------|---------------|
| resnet-graph | nccl-async | 356 | 355 | 398 | 384 | 384 | 382 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| nccl-async | gpu1 | 1923.7 | 3.7 | epoch-boundary(199) |
| nccl-async | gpu2 | 1923.7 | 3.7 | epoch-boundary(199) |
| nccl-async | gpu0 | 1926.4 | 0.6 | epoch-boundary(199) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | nccl-async | gpu0 | 0.6s | 0.0s | 0.0s | 0.0s | 2.3s |
| resnet-graph | nccl-async | gpu1 | 3.7s | 0.0s | 0.0s | 0.0s | 5.5s |
| resnet-graph | nccl-async | gpu2 | 3.7s | 0.0s | 0.0s | 0.0s | 5.0s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | nccl-async | 342 | 0 | 508 | 43.5 | 1994/9392 | 0 | 0.0 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | nccl-async | 199.7 | 10.4% |

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
| resnet-graph | nccl-async | 195 | 508 | 0 | 1.77e-3 | -3.98e-4 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | nccl-async | 0 | 508 | 5.31e-2 | 7.55e-2 | 0.00e0 | 3.25e-1 | 35.8 | -1.41e-4 | 2.99e-3 |
| resnet-graph | nccl-async | 1 | 508 | 5.40e-2 | 7.72e-2 | 0.00e0 | 3.78e-1 | 41.9 | -1.53e-4 | 4.73e-3 |
| resnet-graph | nccl-async | 2 | 508 | 5.36e-2 | 7.65e-2 | 0.00e0 | 3.46e-1 | 22.2 | -1.52e-4 | 4.65e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | nccl-async | rank0 ↔ rank1 | +0.9991 |
| resnet-graph | nccl-async | rank0 ↔ rank2 | +0.9995 |
| resnet-graph | nccl-async | rank1 ↔ rank2 | +0.9996 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | nccl-async | 32 (0,1,2,5,25,37,43,44…140,142) | 2 (194,195) | — | 0,1,2,5,25,37,43,44…140,142 | 194,195 |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | nccl-async | 0e0 | 3 | 33 | 33 |
| resnet-graph | nccl-async | 0e0 | 5 | 18 | 18 |
| resnet-graph | nccl-async | 0e0 | 10 | 6 | 6 |
| resnet-graph | nccl-async | 1e-4 | 3 | 12 | 12 |
| resnet-graph | nccl-async | 1e-4 | 5 | 6 | 6 |
| resnet-graph | nccl-async | 1e-4 | 10 | 2 | 2 |
| resnet-graph | nccl-async | 1e-3 | 3 | 2 | 2 |
| resnet-graph | nccl-async | 1e-3 | 5 | 1 | 1 |
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
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 149 | +0.116 |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | 63 | +0.206 |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | 291 | -0.014 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | nccl-async | 505 | -0.015 | 194 | +0.009 | -0.121 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | nccl-async | 506 | 3.32e1–7.89e1 | 6.16e1 | 2.40e-3 | 6.89e-3 | 8.03e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 151 | 61–77648 | +1.214e-5 | 0.428 | +1.249e-5 | 0.433 | 96 | +2.776e-6 | 0.096 | 29–988 | +1.432e-3 | 0.778 |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 133 | 948–77648 | +9.825e-6 | 0.391 | +1.002e-5 | 0.393 | 95 | +2.044e-6 | 0.067 | 44–988 | +1.396e-3 | 0.855 |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | 64 | 78236–116979 | +1.190e-5 | 0.132 | +1.178e-5 | 0.129 | 49 | +1.532e-5 | 0.256 | 407–840 | -1.722e-3 | 0.168 |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | 292 | 117441–156039 | -4.492e-5 | 0.625 | -4.629e-5 | 0.636 | 50 | -4.256e-5 | 0.661 | 33–545 | +3.378e-3 | 0.657 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | +1.432e-3 | r0: +1.417e-3, r1: +1.445e-3, r2: +1.439e-3 | r0: 0.788, r1: 0.775, r2: 0.768 | 1.02× | ✓ |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +1.396e-3 | r0: +1.400e-3, r1: +1.388e-3, r2: +1.401e-3 | r0: 0.867, r1: 0.851, r2: 0.842 | 1.01× | ✓ |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | -1.722e-3 | r0: -1.661e-3, r1: -1.755e-3, r2: -1.750e-3 | r0: 0.158, r1: 0.174, r2: 0.172 | 1.06× | ✓ |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | +3.378e-3 | r0: +3.263e-3, r1: +3.439e-3, r2: +3.450e-3 | r0: 0.688, r1: 0.632, r2: 0.640 | 1.06× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | nccl-async | `▇███████████████████████▅▅▅▅▅▅▅▅▅▆▆▆▄▂▂▂▂▂▂▁▁▁▁▁▁` | `▁█▇▆▆▆▆▆▆▆▆▆▅▆▅▅▆▆▅▆▅▅▅▅▂▃▄▅▅▅▆▆▆▆▆▆▂▂▄▄▅▅▅▄▃▄▃█▂` |

### Per-Epoch Detail: resnet-graph / nccl-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 19 | 0.00e0 | 3.78e-1 | 8.43e-2 | 6.62e-2 | 18 | -4.53e-2 | +1.60e-2 | -8.44e-3 | -3.04e-3 |
| 1 | 3.00e-1 | 12 | 6.12e-2 | 1.21e-1 | 7.41e-2 | 7.64e-2 | 15 | -2.94e-2 | +3.39e-2 | +1.09e-3 | -7.99e-5 |
| 2 | 3.00e-1 | 12 | 6.26e-2 | 1.18e-1 | 7.54e-2 | 6.71e-2 | 20 | -3.31e-2 | +3.74e-2 | -3.99e-6 | -4.40e-4 |
| 3 | 3.00e-1 | 3 | 7.42e-2 | 7.71e-2 | 7.57e-2 | 7.42e-2 | 217 | -7.34e-4 | +6.89e-3 | +2.02e-3 | +1.66e-4 |
| 4 | 3.00e-1 | 1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 258 | +3.74e-3 | +3.74e-3 | +3.74e-3 | +5.27e-4 |
| 5 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 254 | +7.60e-5 | +7.60e-5 | +7.60e-5 | +4.82e-4 |
| 6 | 3.00e-1 | 1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 268 | -1.24e-4 | -1.24e-4 | -1.24e-4 | +4.21e-4 |
| 7 | 3.00e-1 | 1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 300 | -2.79e-5 | -2.79e-5 | -2.79e-5 | +3.76e-4 |
| 8 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 282 | +1.87e-4 | +1.87e-4 | +1.87e-4 | +3.57e-4 |
| 9 | 3.00e-1 | 1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 311 | -1.07e-4 | -1.07e-4 | -1.07e-4 | +3.10e-4 |
| 10 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 300 | +9.39e-5 | +9.39e-5 | +9.39e-5 | +2.88e-4 |
| 11 | 3.00e-1 | 1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 287 | -4.88e-5 | -4.88e-5 | -4.88e-5 | +2.54e-4 |
| 13 | 3.00e-1 | 2 | 1.96e-1 | 2.08e-1 | 2.02e-1 | 2.08e-1 | 264 | -7.53e-6 | +2.23e-4 | +1.08e-4 | +2.28e-4 |
| 15 | 3.00e-1 | 1 | 1.88e-1 | 1.88e-1 | 1.88e-1 | 1.88e-1 | 327 | -3.16e-4 | -3.16e-4 | -3.16e-4 | +1.73e-4 |
| 16 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 294 | +2.34e-4 | +2.34e-4 | +2.34e-4 | +1.79e-4 |
| 17 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 305 | -3.74e-5 | -3.74e-5 | -3.74e-5 | +1.57e-4 |
| 18 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 288 | -5.17e-5 | -5.17e-5 | -5.17e-5 | +1.37e-4 |
| 19 | 3.00e-1 | 1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 264 | -1.18e-5 | -1.18e-5 | -1.18e-5 | +1.22e-4 |
| 20 | 3.00e-1 | 1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 290 | -1.06e-4 | -1.06e-4 | -1.06e-4 | +9.88e-5 |
| 21 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 276 | +1.74e-4 | +1.74e-4 | +1.74e-4 | +1.06e-4 |
| 22 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 265 | -5.88e-5 | -5.88e-5 | -5.88e-5 | +8.98e-5 |
| 23 | 3.00e-1 | 1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 272 | -2.53e-5 | -2.53e-5 | -2.53e-5 | +7.83e-5 |
| 24 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 322 | +2.27e-5 | +2.27e-5 | +2.27e-5 | +7.27e-5 |
| 25 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 330 | +1.26e-4 | +1.26e-4 | +1.26e-4 | +7.81e-5 |
| 26 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 295 | -2.14e-5 | -2.14e-5 | -2.14e-5 | +6.81e-5 |
| 27 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 280 | -6.51e-5 | -6.51e-5 | -6.51e-5 | +5.48e-5 |
| 28 | 3.00e-1 | 1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 278 | -3.45e-5 | -3.45e-5 | -3.45e-5 | +4.59e-5 |
| 29 | 3.00e-1 | 1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 267 | -9.79e-5 | -9.79e-5 | -9.79e-5 | +3.15e-5 |
| 30 | 3.00e-1 | 1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 267 | +5.35e-5 | +5.35e-5 | +5.35e-5 | +3.37e-5 |
| 31 | 3.00e-1 | 1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 279 | -3.43e-5 | -3.43e-5 | -3.43e-5 | +2.69e-5 |
| 32 | 3.00e-1 | 1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 273 | +7.92e-5 | +7.92e-5 | +7.92e-5 | +3.21e-5 |
| 33 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 275 | -1.96e-5 | -1.96e-5 | -1.96e-5 | +2.69e-5 |
| 34 | 3.00e-1 | 1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 248 | -2.72e-5 | -2.72e-5 | -2.72e-5 | +2.15e-5 |
| 35 | 3.00e-1 | 1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 240 | -1.09e-4 | -1.09e-4 | -1.09e-4 | +8.51e-6 |
| 36 | 3.00e-1 | 1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 260 | +3.98e-5 | +3.98e-5 | +3.98e-5 | +1.16e-5 |
| 37 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 239 | +1.60e-4 | +1.60e-4 | +1.60e-4 | +2.65e-5 |
| 38 | 3.00e-1 | 2 | 1.93e-1 | 1.95e-1 | 1.94e-1 | 1.93e-1 | 226 | -9.47e-5 | -4.70e-5 | -7.08e-5 | +8.20e-6 |
| 39 | 3.00e-1 | 1 | 1.85e-1 | 1.85e-1 | 1.85e-1 | 1.85e-1 | 248 | -1.58e-4 | -1.58e-4 | -1.58e-4 | -8.45e-6 |
| 40 | 3.00e-1 | 1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 237 | +1.66e-4 | +1.66e-4 | +1.66e-4 | +9.01e-6 |
| 41 | 3.00e-1 | 1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 262 | -1.85e-5 | -1.85e-5 | -1.85e-5 | +6.26e-6 |
| 42 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 261 | +8.76e-5 | +8.76e-5 | +8.76e-5 | +1.44e-5 |
| 43 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 274 | +4.39e-5 | +4.39e-5 | +4.39e-5 | +1.73e-5 |
| 44 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 241 | +2.12e-5 | +2.12e-5 | +2.12e-5 | +1.77e-5 |
| 45 | 3.00e-1 | 1 | 1.88e-1 | 1.88e-1 | 1.88e-1 | 1.88e-1 | 246 | -2.40e-4 | -2.40e-4 | -2.40e-4 | -8.08e-6 |
| 46 | 3.00e-1 | 1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 275 | +8.78e-5 | +8.78e-5 | +8.78e-5 | +1.50e-6 |
| 47 | 3.00e-1 | 1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 265 | +7.40e-5 | +7.40e-5 | +7.40e-5 | +8.75e-6 |
| 48 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 257 | +1.93e-5 | +1.93e-5 | +1.93e-5 | +9.81e-6 |
| 49 | 3.00e-1 | 2 | 1.91e-1 | 2.01e-1 | 1.96e-1 | 2.01e-1 | 213 | -1.23e-4 | +2.21e-4 | +4.90e-5 | +1.90e-5 |
| 50 | 3.00e-1 | 1 | 1.84e-1 | 1.84e-1 | 1.84e-1 | 1.84e-1 | 226 | -3.88e-4 | -3.88e-4 | -3.88e-4 | -2.17e-5 |
| 51 | 3.00e-1 | 1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 231 | +1.46e-4 | +1.46e-4 | +1.46e-4 | -4.90e-6 |
| 52 | 3.00e-1 | 1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 233 | +5.95e-5 | +5.95e-5 | +5.95e-5 | +1.54e-6 |
| 53 | 3.00e-1 | 1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 247 | +4.38e-5 | +4.38e-5 | +4.38e-5 | +5.76e-6 |
| 54 | 3.00e-1 | 2 | 1.93e-1 | 1.94e-1 | 1.94e-1 | 1.93e-1 | 213 | -2.21e-5 | -8.23e-6 | -1.52e-5 | +1.86e-6 |
| 55 | 3.00e-1 | 1 | 1.87e-1 | 1.87e-1 | 1.87e-1 | 1.87e-1 | 260 | -1.27e-4 | -1.27e-4 | -1.27e-4 | -1.10e-5 |
| 56 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 246 | +3.17e-4 | +3.17e-4 | +3.17e-4 | +2.18e-5 |
| 57 | 3.00e-1 | 1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 252 | -1.87e-4 | -1.87e-4 | -1.87e-4 | +9.21e-7 |
| 58 | 3.00e-1 | 1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 231 | +4.47e-5 | +4.47e-5 | +4.47e-5 | +5.30e-6 |
| 59 | 3.00e-1 | 2 | 1.90e-1 | 1.91e-1 | 1.90e-1 | 1.91e-1 | 198 | -1.25e-4 | +3.42e-5 | -4.52e-5 | -3.50e-6 |
| 60 | 3.00e-1 | 1 | 1.79e-1 | 1.79e-1 | 1.79e-1 | 1.79e-1 | 239 | -2.74e-4 | -2.74e-4 | -2.74e-4 | -3.05e-5 |
| 61 | 3.00e-1 | 1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 240 | +3.13e-4 | +3.13e-4 | +3.13e-4 | +3.84e-6 |
| 62 | 3.00e-1 | 1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 214 | +5.08e-5 | +5.08e-5 | +5.08e-5 | +8.54e-6 |
| 63 | 3.00e-1 | 2 | 1.78e-1 | 1.87e-1 | 1.82e-1 | 1.78e-1 | 183 | -2.60e-4 | -2.34e-4 | -2.47e-4 | -4.01e-5 |
| 64 | 3.00e-1 | 1 | 1.77e-1 | 1.77e-1 | 1.77e-1 | 1.77e-1 | 256 | -1.34e-5 | -1.34e-5 | -1.34e-5 | -3.74e-5 |
| 65 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 245 | +5.79e-4 | +5.79e-4 | +5.79e-4 | +2.42e-5 |
| 66 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 237 | -1.24e-4 | -1.24e-4 | -1.24e-4 | +9.45e-6 |
| 67 | 3.00e-1 | 1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 277 | -5.82e-5 | -5.82e-5 | -5.82e-5 | +2.68e-6 |
| 68 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 261 | +1.51e-4 | +1.51e-4 | +1.51e-4 | +1.75e-5 |
| 69 | 3.00e-1 | 2 | 1.91e-1 | 1.97e-1 | 1.94e-1 | 1.91e-1 | 218 | -1.63e-4 | -1.26e-4 | -1.44e-4 | -1.34e-5 |
| 71 | 3.00e-1 | 2 | 1.86e-1 | 2.08e-1 | 1.97e-1 | 2.08e-1 | 218 | -8.70e-5 | +5.06e-4 | +2.09e-4 | +3.19e-5 |
| 72 | 3.00e-1 | 1 | 1.87e-1 | 1.87e-1 | 1.87e-1 | 1.87e-1 | 264 | -3.86e-4 | -3.86e-4 | -3.86e-4 | -9.95e-6 |
| 73 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 279 | +2.60e-4 | +2.60e-4 | +2.60e-4 | +1.71e-5 |
| 74 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 250 | -2.43e-6 | -2.43e-6 | -2.43e-6 | +1.51e-5 |
| 75 | 3.00e-1 | 1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 234 | -1.84e-4 | -1.84e-4 | -1.84e-4 | -4.79e-6 |
| 76 | 3.00e-1 | 1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 233 | +3.25e-6 | +3.25e-6 | +3.25e-6 | -3.99e-6 |
| 77 | 3.00e-1 | 2 | 1.91e-1 | 1.97e-1 | 1.94e-1 | 1.97e-1 | 207 | -3.44e-5 | +1.33e-4 | +4.91e-5 | +6.93e-6 |
| 78 | 3.00e-1 | 1 | 1.86e-1 | 1.86e-1 | 1.86e-1 | 1.86e-1 | 240 | -2.40e-4 | -2.40e-4 | -2.40e-4 | -1.78e-5 |
| 79 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 261 | +2.05e-4 | +2.05e-4 | +2.05e-4 | +4.47e-6 |
| 80 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 264 | +1.12e-4 | +1.12e-4 | +1.12e-4 | +1.52e-5 |
| 81 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 253 | +1.85e-5 | +1.85e-5 | +1.85e-5 | +1.55e-5 |
| 82 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 209 | -6.87e-5 | -6.87e-5 | -6.87e-5 | +7.09e-6 |
| 83 | 3.00e-1 | 1 | 1.85e-1 | 1.85e-1 | 1.85e-1 | 1.85e-1 | 217 | -3.43e-4 | -3.43e-4 | -3.43e-4 | -2.79e-5 |
| 84 | 3.00e-1 | 1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 238 | +1.61e-4 | +1.61e-4 | +1.61e-4 | -9.05e-6 |
| 85 | 3.00e-1 | 2 | 1.94e-1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 209 | +2.86e-5 | +3.67e-5 | +3.27e-5 | -1.08e-6 |
| 86 | 3.00e-1 | 1 | 1.88e-1 | 1.88e-1 | 1.88e-1 | 1.88e-1 | 256 | -1.51e-4 | -1.51e-4 | -1.51e-4 | -1.61e-5 |
| 87 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 233 | +2.39e-4 | +2.39e-4 | +2.39e-4 | +9.40e-6 |
| 88 | 3.00e-1 | 2 | 1.90e-1 | 1.93e-1 | 1.91e-1 | 1.93e-1 | 209 | -1.83e-4 | +6.27e-5 | -6.00e-5 | -2.56e-6 |
| 90 | 3.00e-1 | 2 | 1.85e-1 | 2.01e-1 | 1.93e-1 | 2.01e-1 | 209 | -1.56e-4 | +3.89e-4 | +1.16e-4 | +2.28e-5 |
| 91 | 3.00e-1 | 1 | 1.83e-1 | 1.83e-1 | 1.83e-1 | 1.83e-1 | 233 | -3.93e-4 | -3.93e-4 | -3.93e-4 | -1.89e-5 |
| 92 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 251 | +2.73e-4 | +2.73e-4 | +2.73e-4 | +1.03e-5 |
| 93 | 3.00e-1 | 1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 220 | +2.99e-5 | +2.99e-5 | +2.99e-5 | +1.23e-5 |
| 94 | 3.00e-1 | 2 | 1.88e-1 | 1.91e-1 | 1.89e-1 | 1.91e-1 | 207 | -2.23e-4 | +9.04e-5 | -6.61e-5 | -1.06e-6 |
| 95 | 3.00e-1 | 1 | 1.84e-1 | 1.84e-1 | 1.84e-1 | 1.84e-1 | 249 | -1.41e-4 | -1.41e-4 | -1.41e-4 | -1.50e-5 |
| 96 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 222 | +3.78e-4 | +3.78e-4 | +3.78e-4 | +2.42e-5 |
| 97 | 3.00e-1 | 2 | 1.88e-1 | 1.91e-1 | 1.89e-1 | 1.88e-1 | 181 | -2.32e-4 | -7.73e-5 | -1.55e-4 | -9.01e-6 |
| 98 | 3.00e-1 | 1 | 1.77e-1 | 1.77e-1 | 1.77e-1 | 1.77e-1 | 198 | -3.04e-4 | -3.04e-4 | -3.04e-4 | -3.85e-5 |
| 99 | 3.00e-1 | 1 | 1.88e-1 | 1.88e-1 | 1.88e-1 | 1.88e-1 | 200 | +3.11e-4 | +3.11e-4 | +3.11e-4 | -3.55e-6 |
| 100 | 3.00e-2 | 2 | 1.86e-1 | 1.89e-1 | 1.87e-1 | 1.86e-1 | 193 | -6.22e-5 | +9.35e-6 | -2.64e-5 | -8.25e-6 |
| 101 | 3.00e-2 | 1 | 1.86e-2 | 1.86e-2 | 1.86e-2 | 1.86e-2 | 228 | -1.01e-2 | -1.01e-2 | -1.01e-2 | -1.02e-3 |
| 102 | 3.00e-2 | 2 | 2.01e-2 | 2.08e-2 | 2.05e-2 | 2.01e-2 | 179 | -1.85e-4 | +6.19e-4 | +2.17e-4 | -7.87e-4 |
| 103 | 3.00e-2 | 1 | 2.03e-2 | 2.03e-2 | 2.03e-2 | 2.03e-2 | 263 | +2.80e-5 | +2.80e-5 | +2.80e-5 | -7.06e-4 |
| 104 | 3.00e-2 | 1 | 2.50e-2 | 2.50e-2 | 2.50e-2 | 2.50e-2 | 260 | +8.09e-4 | +8.09e-4 | +8.09e-4 | -5.54e-4 |
| 105 | 3.00e-2 | 1 | 2.56e-2 | 2.56e-2 | 2.56e-2 | 2.56e-2 | 250 | +9.89e-5 | +9.89e-5 | +9.89e-5 | -4.89e-4 |
| 106 | 3.00e-2 | 1 | 2.57e-2 | 2.57e-2 | 2.57e-2 | 2.57e-2 | 295 | +5.28e-6 | +5.28e-6 | +5.28e-6 | -4.40e-4 |
| 107 | 3.00e-2 | 1 | 2.84e-2 | 2.84e-2 | 2.84e-2 | 2.84e-2 | 250 | +4.01e-4 | +4.01e-4 | +4.01e-4 | -3.56e-4 |
| 108 | 3.00e-2 | 1 | 2.76e-2 | 2.76e-2 | 2.76e-2 | 2.76e-2 | 245 | -1.10e-4 | -1.10e-4 | -1.10e-4 | -3.31e-4 |
| 109 | 3.00e-2 | 1 | 2.80e-2 | 2.80e-2 | 2.80e-2 | 2.80e-2 | 259 | +5.89e-5 | +5.89e-5 | +5.89e-5 | -2.92e-4 |
| 110 | 3.00e-2 | 1 | 2.93e-2 | 2.93e-2 | 2.93e-2 | 2.93e-2 | 278 | +1.55e-4 | +1.55e-4 | +1.55e-4 | -2.47e-4 |
| 111 | 3.00e-2 | 1 | 3.11e-2 | 3.11e-2 | 3.11e-2 | 3.11e-2 | 249 | +2.36e-4 | +2.36e-4 | +2.36e-4 | -1.99e-4 |
| 112 | 3.00e-2 | 2 | 2.98e-2 | 3.03e-2 | 3.00e-2 | 3.03e-2 | 219 | -1.71e-4 | +7.93e-5 | -4.59e-5 | -1.69e-4 |
| 114 | 3.00e-2 | 2 | 2.95e-2 | 3.55e-2 | 3.25e-2 | 3.55e-2 | 207 | -8.52e-5 | +8.92e-4 | +4.04e-4 | -5.50e-5 |
| 115 | 3.00e-2 | 1 | 3.01e-2 | 3.01e-2 | 3.01e-2 | 3.01e-2 | 228 | -7.13e-4 | -7.13e-4 | -7.13e-4 | -1.21e-4 |
| 116 | 3.00e-2 | 1 | 3.26e-2 | 3.26e-2 | 3.26e-2 | 3.26e-2 | 225 | +3.56e-4 | +3.56e-4 | +3.56e-4 | -7.31e-5 |
| 117 | 3.00e-2 | 1 | 3.33e-2 | 3.33e-2 | 3.33e-2 | 3.33e-2 | 247 | +7.42e-5 | +7.42e-5 | +7.42e-5 | -5.84e-5 |
| 118 | 3.00e-2 | 1 | 3.45e-2 | 3.45e-2 | 3.45e-2 | 3.45e-2 | 236 | +1.57e-4 | +1.57e-4 | +1.57e-4 | -3.69e-5 |
| 119 | 3.00e-2 | 1 | 3.44e-2 | 3.44e-2 | 3.44e-2 | 3.44e-2 | 229 | -1.65e-5 | -1.65e-5 | -1.65e-5 | -3.48e-5 |
| 120 | 3.00e-2 | 2 | 3.48e-2 | 3.78e-2 | 3.63e-2 | 3.78e-2 | 207 | +4.63e-5 | +3.92e-4 | +2.19e-4 | +1.51e-5 |
| 121 | 3.00e-2 | 1 | 3.47e-2 | 3.47e-2 | 3.47e-2 | 3.47e-2 | 206 | -4.14e-4 | -4.14e-4 | -4.14e-4 | -2.78e-5 |
| 122 | 3.00e-2 | 1 | 3.46e-2 | 3.46e-2 | 3.46e-2 | 3.46e-2 | 220 | -7.23e-6 | -7.23e-6 | -7.23e-6 | -2.57e-5 |
| 123 | 3.00e-2 | 1 | 3.61e-2 | 3.61e-2 | 3.61e-2 | 3.61e-2 | 243 | +1.76e-4 | +1.76e-4 | +1.76e-4 | -5.58e-6 |
| 124 | 3.00e-2 | 2 | 3.77e-2 | 3.88e-2 | 3.83e-2 | 3.88e-2 | 196 | +1.45e-4 | +1.82e-4 | +1.64e-4 | +2.64e-5 |
| 125 | 3.00e-2 | 1 | 3.61e-2 | 3.61e-2 | 3.61e-2 | 3.61e-2 | 247 | -2.93e-4 | -2.93e-4 | -2.93e-4 | -5.55e-6 |
| 126 | 3.00e-2 | 1 | 4.06e-2 | 4.06e-2 | 4.06e-2 | 4.06e-2 | 243 | +4.81e-4 | +4.81e-4 | +4.81e-4 | +4.31e-5 |
| 127 | 3.00e-2 | 1 | 4.05e-2 | 4.05e-2 | 4.05e-2 | 4.05e-2 | 214 | -5.89e-6 | -5.89e-6 | -5.89e-6 | +3.82e-5 |
| 128 | 3.00e-2 | 2 | 3.94e-2 | 4.17e-2 | 4.05e-2 | 4.17e-2 | 193 | -1.22e-4 | +2.99e-4 | +8.82e-5 | +4.98e-5 |
| 129 | 3.00e-2 | 1 | 3.90e-2 | 3.90e-2 | 3.90e-2 | 3.90e-2 | 200 | -3.34e-4 | -3.34e-4 | -3.34e-4 | +1.15e-5 |
| 130 | 3.00e-2 | 1 | 4.04e-2 | 4.04e-2 | 4.04e-2 | 4.04e-2 | 210 | +1.65e-4 | +1.65e-4 | +1.65e-4 | +2.68e-5 |
| 131 | 3.00e-2 | 2 | 4.00e-2 | 4.06e-2 | 4.03e-2 | 4.06e-2 | 182 | -4.70e-5 | +8.17e-5 | +1.73e-5 | +2.57e-5 |
| 132 | 3.00e-2 | 1 | 3.95e-2 | 3.95e-2 | 3.95e-2 | 3.95e-2 | 242 | -1.15e-4 | -1.15e-4 | -1.15e-4 | +1.16e-5 |
| 133 | 3.00e-2 | 1 | 4.51e-2 | 4.51e-2 | 4.51e-2 | 4.51e-2 | 226 | +5.88e-4 | +5.88e-4 | +5.88e-4 | +6.92e-5 |
| 134 | 3.00e-2 | 1 | 4.41e-2 | 4.41e-2 | 4.41e-2 | 4.41e-2 | 224 | -1.03e-4 | -1.03e-4 | -1.03e-4 | +5.20e-5 |
| 135 | 3.00e-2 | 2 | 4.47e-2 | 4.67e-2 | 4.57e-2 | 4.67e-2 | 197 | +6.26e-5 | +2.24e-4 | +1.43e-4 | +7.02e-5 |
| 136 | 3.00e-2 | 1 | 4.39e-2 | 4.39e-2 | 4.39e-2 | 4.39e-2 | 233 | -2.65e-4 | -2.65e-4 | -2.65e-4 | +3.67e-5 |
| 137 | 3.00e-2 | 2 | 4.55e-2 | 4.75e-2 | 4.65e-2 | 4.75e-2 | 199 | +1.53e-4 | +2.10e-4 | +1.82e-4 | +6.45e-5 |
| 138 | 3.00e-2 | 1 | 4.46e-2 | 4.46e-2 | 4.46e-2 | 4.46e-2 | 224 | -2.84e-4 | -2.84e-4 | -2.84e-4 | +2.97e-5 |
| 139 | 3.00e-2 | 1 | 4.67e-2 | 4.67e-2 | 4.67e-2 | 4.67e-2 | 237 | +1.95e-4 | +1.95e-4 | +1.95e-4 | +4.62e-5 |
| 140 | 3.00e-2 | 2 | 4.46e-2 | 4.83e-2 | 4.65e-2 | 4.46e-2 | 153 | -5.20e-4 | +1.84e-4 | -1.68e-4 | +2.05e-6 |
| 141 | 3.00e-2 | 1 | 4.18e-2 | 4.18e-2 | 4.18e-2 | 4.18e-2 | 176 | -3.70e-4 | -3.70e-4 | -3.70e-4 | -3.51e-5 |
| 142 | 3.00e-2 | 2 | 4.45e-2 | 4.72e-2 | 4.58e-2 | 4.72e-2 | 153 | +3.21e-4 | +3.93e-4 | +3.57e-4 | +3.97e-5 |
| 143 | 3.00e-2 | 2 | 4.17e-2 | 4.73e-2 | 4.45e-2 | 4.73e-2 | 171 | -6.21e-4 | +7.41e-4 | +6.01e-5 | +5.04e-5 |
| 144 | 3.00e-2 | 1 | 4.65e-2 | 4.65e-2 | 4.65e-2 | 4.65e-2 | 202 | -9.29e-5 | -9.29e-5 | -9.29e-5 | +3.61e-5 |
| 145 | 3.00e-2 | 1 | 4.93e-2 | 4.93e-2 | 4.93e-2 | 4.93e-2 | 197 | +3.00e-4 | +3.00e-4 | +3.00e-4 | +6.25e-5 |
| 146 | 3.00e-2 | 2 | 4.82e-2 | 5.24e-2 | 5.03e-2 | 5.24e-2 | 189 | -1.00e-4 | +4.49e-4 | +1.74e-4 | +8.65e-5 |
| 147 | 3.00e-2 | 1 | 4.97e-2 | 4.97e-2 | 4.97e-2 | 4.97e-2 | 235 | -2.28e-4 | -2.28e-4 | -2.28e-4 | +5.51e-5 |
| 148 | 3.00e-2 | 1 | 5.37e-2 | 5.37e-2 | 5.37e-2 | 5.37e-2 | 219 | +3.54e-4 | +3.54e-4 | +3.54e-4 | +8.50e-5 |
| 149 | 3.00e-2 | 2 | 4.75e-2 | 5.37e-2 | 5.06e-2 | 4.75e-2 | 145 | -8.50e-4 | -4.98e-6 | -4.27e-4 | -1.66e-5 |
| 150 | 3.00e-3 | 2 | 4.42e-2 | 4.76e-2 | 4.59e-2 | 4.76e-2 | 138 | -3.94e-4 | +5.36e-4 | +7.13e-5 | +4.74e-6 |
| 151 | 3.00e-3 | 2 | 4.29e-3 | 4.67e-3 | 4.48e-3 | 4.67e-3 | 148 | -1.42e-2 | +5.85e-4 | -6.79e-3 | -1.21e-3 |
| 152 | 3.00e-3 | 1 | 4.39e-3 | 4.39e-3 | 4.39e-3 | 4.39e-3 | 183 | -3.42e-4 | -3.42e-4 | -3.42e-4 | -1.13e-3 |
| 153 | 3.00e-3 | 2 | 4.76e-3 | 5.10e-3 | 4.93e-3 | 5.10e-3 | 154 | +4.16e-4 | +4.44e-4 | +4.30e-4 | -8.30e-4 |
| 154 | 3.00e-3 | 1 | 4.54e-3 | 4.54e-3 | 4.54e-3 | 4.54e-3 | 183 | -6.32e-4 | -6.32e-4 | -6.32e-4 | -8.10e-4 |
| 155 | 3.00e-3 | 2 | 4.88e-3 | 4.98e-3 | 4.93e-3 | 4.98e-3 | 161 | +1.24e-4 | +3.72e-4 | +2.48e-4 | -6.10e-4 |
| 156 | 3.00e-3 | 2 | 4.57e-3 | 4.65e-3 | 4.61e-3 | 4.65e-3 | 137 | -4.70e-4 | +1.34e-4 | -1.68e-4 | -5.23e-4 |
| 157 | 3.00e-3 | 2 | 4.23e-3 | 4.87e-3 | 4.55e-3 | 4.87e-3 | 137 | -5.13e-4 | +1.02e-3 | +2.56e-4 | -3.68e-4 |
| 158 | 3.00e-3 | 1 | 4.43e-3 | 4.43e-3 | 4.43e-3 | 4.43e-3 | 176 | -5.45e-4 | -5.45e-4 | -5.45e-4 | -3.85e-4 |
| 159 | 3.00e-3 | 3 | 4.32e-3 | 5.01e-3 | 4.77e-3 | 4.32e-3 | 130 | -1.14e-3 | +6.64e-4 | -1.36e-4 | -3.35e-4 |
| 160 | 3.00e-3 | 1 | 4.48e-3 | 4.48e-3 | 4.48e-3 | 4.48e-3 | 216 | +1.61e-4 | +1.61e-4 | +1.61e-4 | -2.86e-4 |
| 161 | 3.00e-3 | 2 | 4.63e-3 | 5.38e-3 | 5.00e-3 | 4.63e-3 | 120 | -1.24e-3 | +1.18e-3 | -3.24e-5 | -2.50e-4 |
| 162 | 3.00e-3 | 2 | 4.26e-3 | 4.46e-3 | 4.36e-3 | 4.46e-3 | 110 | -5.18e-4 | +4.23e-4 | -4.78e-5 | -2.06e-4 |
| 163 | 3.00e-3 | 3 | 4.18e-3 | 4.77e-3 | 4.46e-3 | 4.42e-3 | 110 | -6.79e-4 | +1.02e-3 | -1.89e-5 | -1.59e-4 |
| 164 | 3.00e-3 | 1 | 4.14e-3 | 4.14e-3 | 4.14e-3 | 4.14e-3 | 150 | -4.36e-4 | -4.36e-4 | -4.36e-4 | -1.87e-4 |
| 165 | 3.00e-3 | 3 | 3.91e-3 | 4.77e-3 | 4.48e-3 | 3.91e-3 | 97 | -2.05e-3 | +8.58e-4 | -3.92e-4 | -2.70e-4 |
| 166 | 3.00e-3 | 2 | 4.05e-3 | 4.58e-3 | 4.31e-3 | 4.58e-3 | 106 | +2.57e-4 | +1.16e-3 | +7.10e-4 | -7.92e-5 |
| 167 | 3.00e-3 | 3 | 4.01e-3 | 4.56e-3 | 4.27e-3 | 4.24e-3 | 110 | -9.08e-4 | +1.16e-3 | -1.39e-4 | -9.38e-5 |
| 168 | 3.00e-3 | 1 | 4.25e-3 | 4.25e-3 | 4.25e-3 | 4.25e-3 | 147 | +1.89e-5 | +1.89e-5 | +1.89e-5 | -8.25e-5 |
| 169 | 3.00e-3 | 3 | 4.19e-3 | 4.68e-3 | 4.50e-3 | 4.19e-3 | 104 | -1.04e-3 | +5.98e-4 | -1.21e-4 | -1.09e-4 |
| 170 | 3.00e-3 | 2 | 4.01e-3 | 4.84e-3 | 4.42e-3 | 4.84e-3 | 104 | -2.92e-4 | +1.80e-3 | +7.56e-4 | +6.60e-5 |
| 171 | 3.00e-3 | 3 | 4.17e-3 | 4.42e-3 | 4.29e-3 | 4.17e-3 | 108 | -9.05e-4 | +3.07e-4 | -3.83e-4 | -5.27e-5 |
| 172 | 3.00e-3 | 2 | 4.28e-3 | 4.44e-3 | 4.36e-3 | 4.44e-3 | 101 | +2.11e-4 | +3.63e-4 | +2.87e-4 | +1.26e-5 |
| 173 | 3.00e-3 | 3 | 3.54e-3 | 4.21e-3 | 3.95e-3 | 3.54e-3 | 79 | -2.19e-3 | +3.25e-4 | -8.52e-4 | -2.37e-4 |
| 174 | 3.00e-3 | 4 | 3.61e-3 | 4.38e-3 | 3.84e-3 | 3.61e-3 | 90 | -2.09e-3 | +2.19e-3 | +1.82e-5 | -1.76e-4 |
| 175 | 3.00e-3 | 2 | 3.87e-3 | 4.21e-3 | 4.04e-3 | 4.21e-3 | 75 | +6.31e-4 | +1.11e-3 | +8.72e-4 | +2.51e-5 |
| 176 | 3.00e-3 | 5 | 3.35e-3 | 4.24e-3 | 3.64e-3 | 3.53e-3 | 72 | -3.34e-3 | +2.24e-3 | -3.46e-4 | -1.14e-4 |
| 177 | 3.00e-3 | 3 | 3.30e-3 | 3.92e-3 | 3.55e-3 | 3.42e-3 | 69 | -1.98e-3 | +2.49e-3 | -6.18e-5 | -1.13e-4 |
| 178 | 3.00e-3 | 3 | 3.33e-3 | 4.29e-3 | 3.78e-3 | 3.74e-3 | 70 | -1.97e-3 | +3.34e-3 | +3.74e-4 | +8.88e-7 |
| 179 | 3.00e-3 | 4 | 3.08e-3 | 4.06e-3 | 3.51e-3 | 3.08e-3 | 59 | -4.12e-3 | +1.49e-3 | -8.07e-4 | -3.08e-4 |
| 180 | 3.00e-3 | 5 | 2.88e-3 | 3.87e-3 | 3.22e-3 | 2.88e-3 | 53 | -3.21e-3 | +3.98e-3 | -2.65e-4 | -3.53e-4 |
| 181 | 3.00e-3 | 5 | 2.69e-3 | 3.92e-3 | 3.05e-3 | 2.69e-3 | 48 | -6.40e-3 | +5.96e-3 | -9.84e-5 | -3.25e-4 |
| 182 | 3.00e-3 | 7 | 2.57e-3 | 3.66e-3 | 2.84e-3 | 2.65e-3 | 52 | -5.85e-3 | +5.38e-3 | -1.53e-4 | -2.76e-4 |
| 183 | 3.00e-3 | 4 | 2.61e-3 | 4.05e-3 | 3.06e-3 | 2.87e-3 | 38 | -8.92e-3 | +8.82e-3 | +6.49e-4 | -5.80e-6 |
| 184 | 3.00e-3 | 8 | 1.89e-3 | 3.73e-3 | 2.40e-3 | 1.89e-3 | 23 | -1.56e-2 | +1.15e-2 | -1.76e-3 | -1.12e-3 |
| 185 | 3.00e-3 | 11 | 1.53e-3 | 3.26e-3 | 1.87e-3 | 1.83e-3 | 20 | -3.21e-2 | +3.18e-2 | +8.38e-5 | -3.61e-4 |
| 186 | 3.00e-3 | 17 | 1.34e-3 | 3.06e-3 | 1.61e-3 | 1.56e-3 | 20 | -4.27e-2 | +3.77e-2 | -2.85e-4 | -1.05e-4 |
| 187 | 3.00e-3 | 12 | 1.20e-3 | 3.05e-3 | 1.58e-3 | 1.40e-3 | 16 | -4.38e-2 | +4.11e-2 | -7.10e-4 | -7.62e-4 |
| 188 | 3.00e-3 | 18 | 1.21e-3 | 2.93e-3 | 1.49e-3 | 1.53e-3 | 18 | -4.60e-2 | +4.75e-2 | +4.67e-4 | +3.48e-4 |
| 189 | 3.00e-3 | 11 | 1.28e-3 | 2.91e-3 | 1.54e-3 | 1.33e-3 | 22 | -4.24e-2 | +3.92e-2 | -5.73e-4 | -5.11e-4 |
| 190 | 3.00e-3 | 13 | 1.63e-3 | 3.36e-3 | 1.81e-3 | 1.63e-3 | 21 | -3.21e-2 | +2.96e-2 | -3.50e-5 | -3.73e-4 |
| 191 | 3.00e-3 | 13 | 1.40e-3 | 3.01e-3 | 1.69e-3 | 1.42e-3 | 17 | -3.27e-2 | +2.92e-2 | -7.30e-4 | -8.88e-4 |
| 192 | 3.00e-3 | 16 | 1.14e-3 | 3.05e-3 | 1.46e-3 | 1.36e-3 | 18 | -7.02e-2 | +5.74e-2 | -2.01e-4 | -4.09e-4 |
| 193 | 3.00e-3 | 16 | 1.10e-3 | 3.09e-3 | 1.49e-3 | 1.55e-3 | 17 | -4.38e-2 | +4.90e-2 | +5.20e-4 | +5.19e-4 |
| 194 | 3.00e-3 | 9 | 1.26e-3 | 3.19e-3 | 2.04e-3 | 2.82e-3 | 40 | -4.66e-2 | +4.67e-2 | +3.38e-3 | +2.18e-3 |
| 195 | 3.00e-3 | 8 | 2.00e-3 | 3.69e-3 | 2.48e-3 | 2.00e-3 | 27 | -1.03e-2 | +1.00e-2 | -1.18e-3 | +1.39e-4 |
| 196 | 3.00e-3 | 11 | 1.65e-3 | 3.58e-3 | 2.00e-3 | 1.67e-3 | 19 | -2.23e-2 | +1.53e-2 | -1.43e-3 | -9.31e-4 |
| 197 | 3.00e-3 | 15 | 1.24e-3 | 3.03e-3 | 1.57e-3 | 1.38e-3 | 16 | -3.32e-2 | +3.53e-2 | -1.15e-3 | -1.11e-3 |
| 198 | 3.00e-3 | 15 | 1.11e-3 | 3.08e-3 | 1.48e-3 | 1.30e-3 | 22 | -6.79e-2 | +5.30e-2 | -2.23e-4 | -5.76e-4 |
| 199 | 3.00e-3 | 8 | 1.64e-3 | 3.60e-3 | 2.13e-3 | 1.77e-3 | 21 | -1.99e-2 | +2.91e-2 | +5.81e-4 | -3.98e-4 |

