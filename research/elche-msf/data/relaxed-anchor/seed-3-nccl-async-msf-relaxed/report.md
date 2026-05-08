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
| nccl-async | 0.066516 | 0.9210 | +0.0085 | 1850.0 | 254 | 37.7 | 100% | 100% | 100% | 5.4 |

## Best Mode per Model

| Model | Best Eval | Mode | Fastest (within 2% of solo-0) | Mode |
|-------|-----------|------|-------------------------------|------|
| resnet-graph | 0.9210 | nccl-async | - | - |

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
| nccl-async | 2.0323 | 0.7847 | 0.6226 | 0.5597 | 0.5347 | 0.5158 | 0.4990 | 0.4799 | 0.4836 | 0.4649 | 0.2200 | 0.1788 | 0.1632 | 0.1559 | 0.1497 | 0.0844 | 0.0779 | 0.0740 | 0.0661 | 0.0665 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | nccl-async | 0 | cuda:0 | 0.4007 | 2.7 |
| resnet-graph | nccl-async | 1 | cuda:1 | 0.3049 | 3.5 |
| resnet-graph | nccl-async | 2 | cuda:2 | 0.2944 | 3.5 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) | GPU2 Peak (MB) | GPU2 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|---------------|---------------|
| resnet-graph | nccl-async | 356 | 355 | 388 | 382 | 380 | 378 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| nccl-async | gpu1 | 1848.8 | 1.1 | epoch-boundary(199) |
| nccl-async | gpu2 | 1849.0 | 0.9 | epoch-boundary(199) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | nccl-async | gpu0 | 0.0s | 0.0s | 0.0s | 0.0s | 1.2s |
| resnet-graph | nccl-async | gpu1 | 1.1s | 0.0s | 0.0s | 0.0s | 2.7s |
| resnet-graph | nccl-async | gpu2 | 0.9s | 0.0s | 0.0s | 0.0s | 1.6s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | nccl-async | 15 | 0 | 254 | 37.7 | 8397/9698 | 0 | 0.0 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | nccl-async | 189.1 | 10.2% |

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
| resnet-graph | nccl-async | 187 | 254 | 0 | 6.89e-3 | +9.75e-6 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | nccl-async | 0 | 254 | 1.04e-1 | 8.25e-2 | 0.00e0 | 3.93e-1 | 19.3 | -1.52e-4 | 1.67e-3 |
| resnet-graph | nccl-async | 1 | 254 | 1.06e-1 | 8.50e-2 | 0.00e0 | 4.03e-1 | 57.9 | -1.56e-4 | 1.80e-3 |
| resnet-graph | nccl-async | 2 | 254 | 1.05e-1 | 8.30e-2 | 0.00e0 | 3.71e-1 | 22.8 | -1.50e-4 | 1.68e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | nccl-async | rank0 ↔ rank1 | +0.9986 |
| resnet-graph | nccl-async | rank0 ↔ rank2 | +0.9983 |
| resnet-graph | nccl-async | rank1 ↔ rank2 | +0.9971 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | nccl-async | 46 (0,1,2,3,4,11,12,21…147,148) | 0 (—) | — | 0,1,2,3,4,11,12,21…147,148 | — |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | nccl-async | 0e0 | 3 | 36 | 36 |
| resnet-graph | nccl-async | 0e0 | 5 | 19 | 19 |
| resnet-graph | nccl-async | 0e0 | 10 | 8 | 8 |
| resnet-graph | nccl-async | 1e-4 | 3 | 6 | 6 |
| resnet-graph | nccl-async | 1e-4 | 5 | 3 | 3 |
| resnet-graph | nccl-async | 1e-4 | 10 | 1 | 1 |
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
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 143 | +0.246 |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | 53 | +0.232 |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | 53 | +0.116 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | nccl-async | 251 | +0.027 | 186 | +0.229 | +0.241 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | nccl-async | 252 | 3.35e1–7.90e1 | 6.43e1 | 4.75e-3 | 9.29e-3 | 8.64e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 145 | 69–77675 | +1.077e-5 | 0.390 | +1.108e-5 | 0.398 | 93 | +4.305e-6 | 0.242 | 37–908 | +1.276e-3 | 0.704 |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 133 | 921–77675 | +9.598e-6 | 0.448 | +9.798e-6 | 0.448 | 92 | +3.749e-6 | 0.216 | 60–908 | +1.285e-3 | 0.890 |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | 54 | 78439–116697 | +1.255e-5 | 0.117 | +1.244e-5 | 0.116 | 47 | +1.307e-5 | 0.164 | 582–854 | +3.890e-4 | 0.003 |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | 54 | 117367–155822 | -6.964e-6 | 0.035 | -6.798e-6 | 0.034 | 47 | -4.926e-6 | 0.025 | 599–860 | -1.158e-3 | 0.023 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | +1.276e-3 | r0: +1.247e-3, r1: +1.283e-3, r2: +1.300e-3 | r0: 0.707, r1: 0.685, r2: 0.716 | 1.04× | ✓ |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +1.285e-3 | r0: +1.260e-3, r1: +1.305e-3, r2: +1.293e-3 | r0: 0.905, r1: 0.879, r2: 0.879 | 1.04× | ✓ |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | +3.890e-4 | r0: +4.459e-4, r1: +3.634e-4, r2: +3.586e-4 | r0: 0.004, r1: 0.003, r2: 0.003 | 1.24× | ✓ |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | -1.158e-3 | r0: -1.097e-3, r1: -1.167e-3, r2: -1.213e-3 | r0: 0.021, r1: 0.023, r2: 0.025 | 1.11× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | nccl-async | `▇▇█████████████████████▇▄▄▄▅▅▅▅▅▅▅▅▃▁▁▁▁▁▁▁▁▁▁▁` | `▁███▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▆▅▆▇▇▇▇▇▇▇▇▇▅▅▆▆▇▇▇▇▇▇▇▇` |

### Per-Epoch Detail: resnet-graph / nccl-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 13 | 0.00e0 | 4.03e-1 | 1.14e-1 | 6.41e-2 | 33 | -4.98e-2 | +1.32e-2 | -9.12e-3 | -5.65e-3 |
| 1 | 3.00e-1 | 9 | 6.78e-2 | 1.00e-1 | 7.61e-2 | 7.98e-2 | 31 | -1.48e-2 | +1.14e-2 | +4.35e-4 | -1.40e-3 |
| 2 | 3.00e-1 | 9 | 6.98e-2 | 1.12e-1 | 8.09e-2 | 8.49e-2 | 31 | -1.12e-2 | +9.82e-3 | -2.86e-5 | -3.62e-4 |
| 3 | 3.00e-1 | 10 | 8.91e-2 | 1.32e-1 | 9.92e-2 | 1.04e-1 | 40 | -1.06e-2 | +1.18e-2 | +4.69e-4 | +1.54e-4 |
| 4 | 3.00e-1 | 2 | 1.09e-1 | 1.13e-1 | 1.11e-1 | 1.09e-1 | 36 | -9.55e-4 | +1.98e-3 | +5.11e-4 | +2.08e-4 |
| 5 | 3.00e-1 | 1 | 1.05e-1 | 1.05e-1 | 1.05e-1 | 1.05e-1 | 267 | -1.50e-4 | -1.50e-4 | -1.50e-4 | +1.72e-4 |
| 6 | 3.00e-1 | 2 | 1.95e-1 | 2.16e-1 | 2.06e-1 | 1.95e-1 | 206 | -4.81e-4 | +2.83e-3 | +1.17e-3 | +3.47e-4 |
| 8 | 3.00e-1 | 2 | 1.81e-1 | 2.01e-1 | 1.91e-1 | 2.01e-1 | 206 | -2.56e-4 | +5.05e-4 | +1.24e-4 | +3.08e-4 |
| 9 | 3.00e-1 | 1 | 1.82e-1 | 1.82e-1 | 1.82e-1 | 1.82e-1 | 216 | -4.66e-4 | -4.66e-4 | -4.66e-4 | +2.30e-4 |
| 10 | 3.00e-1 | 1 | 1.84e-1 | 1.84e-1 | 1.84e-1 | 1.84e-1 | 256 | +4.82e-5 | +4.82e-5 | +4.82e-5 | +2.12e-4 |
| 11 | 3.00e-1 | 1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 263 | +1.28e-4 | +1.28e-4 | +1.28e-4 | +2.04e-4 |
| 12 | 3.00e-1 | 1 | 1.91e-1 | 1.91e-1 | 1.91e-1 | 1.91e-1 | 266 | +6.84e-6 | +6.84e-6 | +6.84e-6 | +1.84e-4 |
| 13 | 3.00e-1 | 2 | 1.89e-1 | 1.93e-1 | 1.91e-1 | 1.93e-1 | 245 | -4.02e-5 | +9.01e-5 | +2.49e-5 | +1.54e-4 |
| 15 | 3.00e-1 | 2 | 1.89e-1 | 1.99e-1 | 1.94e-1 | 1.99e-1 | 257 | -5.78e-5 | +1.93e-4 | +6.79e-5 | +1.39e-4 |
| 17 | 3.00e-1 | 1 | 1.88e-1 | 1.88e-1 | 1.88e-1 | 1.88e-1 | 279 | -1.97e-4 | -1.97e-4 | -1.97e-4 | +1.05e-4 |
| 18 | 3.00e-1 | 2 | 1.89e-1 | 1.94e-1 | 1.92e-1 | 1.89e-1 | 224 | -1.15e-4 | +1.21e-4 | +3.32e-6 | +8.47e-5 |
| 19 | 3.00e-1 | 1 | 1.81e-1 | 1.81e-1 | 1.81e-1 | 1.81e-1 | 233 | -2.01e-4 | -2.01e-4 | -2.01e-4 | +5.61e-5 |
| 20 | 3.00e-1 | 1 | 1.85e-1 | 1.85e-1 | 1.85e-1 | 1.85e-1 | 238 | +8.63e-5 | +8.63e-5 | +8.63e-5 | +5.91e-5 |
| 21 | 3.00e-1 | 1 | 1.89e-1 | 1.89e-1 | 1.89e-1 | 1.89e-1 | 255 | +8.47e-5 | +8.47e-5 | +8.47e-5 | +6.17e-5 |
| 22 | 3.00e-1 | 1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 260 | +8.99e-5 | +8.99e-5 | +8.99e-5 | +6.45e-5 |
| 23 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 243 | +5.32e-5 | +5.32e-5 | +5.32e-5 | +6.34e-5 |
| 24 | 3.00e-1 | 1 | 1.88e-1 | 1.88e-1 | 1.88e-1 | 1.88e-1 | 266 | -1.53e-4 | -1.53e-4 | -1.53e-4 | +4.17e-5 |
| 25 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 277 | +1.84e-4 | +1.84e-4 | +1.84e-4 | +5.60e-5 |
| 26 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 257 | -3.37e-5 | -3.37e-5 | -3.37e-5 | +4.70e-5 |
| 27 | 3.00e-1 | 2 | 1.92e-1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 236 | -7.48e-5 | +2.12e-5 | -2.68e-5 | +3.34e-5 |
| 29 | 3.00e-1 | 2 | 1.87e-1 | 2.12e-1 | 2.00e-1 | 2.12e-1 | 240 | -8.95e-5 | +5.21e-4 | +2.16e-4 | +7.12e-5 |
| 31 | 3.00e-1 | 2 | 1.91e-1 | 1.98e-1 | 1.94e-1 | 1.98e-1 | 236 | -3.92e-4 | +1.51e-4 | -1.20e-4 | +3.75e-5 |
| 32 | 3.00e-1 | 1 | 1.89e-1 | 1.89e-1 | 1.89e-1 | 1.89e-1 | 243 | -1.82e-4 | -1.82e-4 | -1.82e-4 | +1.55e-5 |
| 33 | 3.00e-1 | 1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 269 | +1.04e-4 | +1.04e-4 | +1.04e-4 | +2.43e-5 |
| 34 | 3.00e-1 | 1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 255 | +5.65e-5 | +5.65e-5 | +5.65e-5 | +2.75e-5 |
| 35 | 3.00e-1 | 1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 279 | -3.63e-5 | -3.63e-5 | -3.63e-5 | +2.12e-5 |
| 36 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 269 | +1.02e-4 | +1.02e-4 | +1.02e-4 | +2.92e-5 |
| 37 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 267 | -5.98e-5 | -5.98e-5 | -5.98e-5 | +2.03e-5 |
| 38 | 3.00e-1 | 1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 252 | -2.07e-5 | -2.07e-5 | -2.07e-5 | +1.62e-5 |
| 39 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 267 | -9.83e-6 | -9.83e-6 | -9.83e-6 | +1.36e-5 |
| 40 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 265 | +9.26e-5 | +9.26e-5 | +9.26e-5 | +2.15e-5 |
| 41 | 3.00e-1 | 1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 306 | -6.89e-5 | -6.89e-5 | -6.89e-5 | +1.25e-5 |
| 42 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 290 | +1.64e-4 | +1.64e-4 | +1.64e-4 | +2.76e-5 |
| 43 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 253 | -6.52e-5 | -6.52e-5 | -6.52e-5 | +1.83e-5 |
| 44 | 3.00e-1 | 1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 224 | -2.03e-4 | -2.03e-4 | -2.03e-4 | -3.84e-6 |
| 45 | 3.00e-1 | 1 | 1.89e-1 | 1.89e-1 | 1.89e-1 | 1.89e-1 | 247 | -1.04e-4 | -1.04e-4 | -1.04e-4 | -1.38e-5 |
| 46 | 3.00e-1 | 2 | 1.91e-1 | 1.95e-1 | 1.93e-1 | 1.91e-1 | 208 | -1.12e-4 | +1.39e-4 | +1.34e-5 | -9.92e-6 |
| 47 | 3.00e-1 | 1 | 1.86e-1 | 1.86e-1 | 1.86e-1 | 1.86e-1 | 263 | -1.01e-4 | -1.01e-4 | -1.01e-4 | -1.91e-5 |
| 48 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 270 | +2.63e-4 | +2.63e-4 | +2.63e-4 | +9.18e-6 |
| 49 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 275 | +4.96e-5 | +4.96e-5 | +4.96e-5 | +1.32e-5 |
| 50 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 271 | -2.29e-6 | -2.29e-6 | -2.29e-6 | +1.17e-5 |
| 51 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 253 | +3.49e-5 | +3.49e-5 | +3.49e-5 | +1.40e-5 |
| 52 | 3.00e-1 | 1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 280 | -1.18e-4 | -1.18e-4 | -1.18e-4 | +7.93e-7 |
| 53 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 280 | +9.16e-5 | +9.16e-5 | +9.16e-5 | +9.88e-6 |
| 54 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 249 | +3.53e-5 | +3.53e-5 | +3.53e-5 | +1.24e-5 |
| 55 | 3.00e-1 | 2 | 1.94e-1 | 1.96e-1 | 1.95e-1 | 1.96e-1 | 213 | -1.98e-4 | +3.54e-5 | -8.15e-5 | -4.26e-6 |
| 57 | 3.00e-1 | 2 | 1.86e-1 | 2.02e-1 | 1.94e-1 | 2.02e-1 | 213 | -1.91e-4 | +3.94e-4 | +1.02e-4 | +1.88e-5 |
| 58 | 3.00e-1 | 1 | 1.87e-1 | 1.87e-1 | 1.87e-1 | 1.87e-1 | 247 | -3.10e-4 | -3.10e-4 | -3.10e-4 | -1.41e-5 |
| 59 | 3.00e-1 | 1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 244 | +1.97e-4 | +1.97e-4 | +1.97e-4 | +6.94e-6 |
| 60 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 248 | +3.26e-5 | +3.26e-5 | +3.26e-5 | +9.50e-6 |
| 61 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 247 | +6.89e-5 | +6.89e-5 | +6.89e-5 | +1.54e-5 |
| 62 | 3.00e-1 | 1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 280 | -1.43e-4 | -1.43e-4 | -1.43e-4 | -4.13e-7 |
| 63 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 291 | +1.72e-4 | +1.72e-4 | +1.72e-4 | +1.68e-5 |
| 64 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 275 | +7.04e-5 | +7.04e-5 | +7.04e-5 | +2.22e-5 |
| 65 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 280 | -4.82e-5 | -4.82e-5 | -4.82e-5 | +1.51e-5 |
| 66 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 250 | -3.68e-5 | -3.68e-5 | -3.68e-5 | +9.94e-6 |
| 67 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 295 | -5.08e-5 | -5.08e-5 | -5.08e-5 | +3.87e-6 |
| 68 | 3.00e-1 | 2 | 2.02e-1 | 2.05e-1 | 2.03e-1 | 2.02e-1 | 212 | -7.78e-5 | +8.75e-5 | +4.84e-6 | +3.22e-6 |
| 70 | 3.00e-1 | 2 | 1.86e-1 | 2.09e-1 | 1.97e-1 | 2.09e-1 | 212 | -2.69e-4 | +5.55e-4 | +1.43e-4 | +3.39e-5 |
| 71 | 3.00e-1 | 1 | 1.86e-1 | 1.86e-1 | 1.86e-1 | 1.86e-1 | 236 | -4.86e-4 | -4.86e-4 | -4.86e-4 | -1.80e-5 |
| 72 | 3.00e-1 | 1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 240 | +2.21e-4 | +2.21e-4 | +2.21e-4 | +5.89e-6 |
| 73 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 279 | +4.33e-5 | +4.33e-5 | +4.33e-5 | +9.63e-6 |
| 74 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 264 | +8.66e-5 | +8.66e-5 | +8.66e-5 | +1.73e-5 |
| 75 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 268 | -1.73e-5 | -1.73e-5 | -1.73e-5 | +1.39e-5 |
| 76 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 277 | +4.76e-5 | +4.76e-5 | +4.76e-5 | +1.72e-5 |
| 77 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 253 | -3.08e-5 | -3.08e-5 | -3.08e-5 | +1.24e-5 |
| 78 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 259 | -8.22e-5 | -8.22e-5 | -8.22e-5 | +2.97e-6 |
| 79 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 272 | +4.83e-5 | +4.83e-5 | +4.83e-5 | +7.50e-6 |
| 80 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 271 | +3.30e-7 | +3.30e-7 | +3.30e-7 | +6.78e-6 |
| 81 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 262 | +3.37e-5 | +3.37e-5 | +3.37e-5 | +9.47e-6 |
| 82 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 268 | -3.79e-5 | -3.79e-5 | -3.79e-5 | +4.74e-6 |
| 83 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 247 | +8.80e-5 | +8.80e-5 | +8.80e-5 | +1.31e-5 |
| 84 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 243 | -1.66e-4 | -1.66e-4 | -1.66e-4 | -4.84e-6 |
| 85 | 3.00e-1 | 2 | 1.99e-1 | 2.03e-1 | 2.01e-1 | 2.03e-1 | 226 | +2.31e-5 | +8.29e-5 | +5.30e-5 | +6.45e-6 |
| 86 | 3.00e-1 | 1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 283 | -2.01e-4 | -2.01e-4 | -2.01e-4 | -1.43e-5 |
| 87 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 309 | +2.18e-4 | +2.18e-4 | +2.18e-4 | +8.93e-6 |
| 88 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 262 | +8.17e-5 | +8.17e-5 | +8.17e-5 | +1.62e-5 |
| 89 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 254 | -1.76e-4 | -1.76e-4 | -1.76e-4 | -3.00e-6 |
| 90 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 232 | -1.08e-4 | -1.08e-4 | -1.08e-4 | -1.35e-5 |
| 91 | 3.00e-1 | 1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 238 | -4.16e-5 | -4.16e-5 | -4.16e-5 | -1.63e-5 |
| 92 | 3.00e-1 | 1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 252 | +8.95e-6 | +8.95e-6 | +8.95e-6 | -1.38e-5 |
| 93 | 3.00e-1 | 2 | 1.94e-1 | 1.99e-1 | 1.96e-1 | 1.94e-1 | 214 | -1.23e-4 | +1.05e-4 | -9.17e-6 | -1.40e-5 |
| 94 | 3.00e-1 | 1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 240 | -9.01e-5 | -9.01e-5 | -9.01e-5 | -2.17e-5 |
| 95 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 246 | +1.95e-4 | +1.95e-4 | +1.95e-4 | +1.14e-8 |
| 96 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 258 | +2.20e-5 | +2.20e-5 | +2.20e-5 | +2.21e-6 |
| 97 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 227 | +4.83e-5 | +4.83e-5 | +4.83e-5 | +6.82e-6 |
| 98 | 3.00e-1 | 1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 250 | -1.65e-4 | -1.65e-4 | -1.65e-4 | -1.04e-5 |
| 99 | 3.00e-1 | 1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 277 | +4.65e-5 | +4.65e-5 | +4.65e-5 | -4.66e-6 |
| 100 | 3.00e-2 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 283 | +1.22e-4 | +1.22e-4 | +1.22e-4 | +7.99e-6 |
| 101 | 3.00e-2 | 2 | 2.21e-2 | 2.05e-1 | 1.14e-1 | 2.21e-2 | 232 | -9.61e-3 | +3.25e-5 | -4.79e-3 | -9.52e-4 |
| 103 | 3.00e-2 | 1 | 2.13e-2 | 2.13e-2 | 2.13e-2 | 2.13e-2 | 318 | -1.17e-4 | -1.17e-4 | -1.17e-4 | -8.68e-4 |
| 104 | 3.00e-2 | 2 | 2.51e-2 | 2.53e-2 | 2.52e-2 | 2.51e-2 | 218 | -4.44e-5 | +6.42e-4 | +2.99e-4 | -6.50e-4 |
| 106 | 3.00e-2 | 1 | 2.33e-2 | 2.33e-2 | 2.33e-2 | 2.33e-2 | 312 | -2.33e-4 | -2.33e-4 | -2.33e-4 | -6.08e-4 |
| 107 | 3.00e-2 | 2 | 2.66e-2 | 2.78e-2 | 2.72e-2 | 2.66e-2 | 219 | -1.97e-4 | +6.73e-4 | +2.38e-4 | -4.52e-4 |
| 108 | 3.00e-2 | 1 | 2.60e-2 | 2.60e-2 | 2.60e-2 | 2.60e-2 | 227 | -1.07e-4 | -1.07e-4 | -1.07e-4 | -4.17e-4 |
| 109 | 3.00e-2 | 1 | 2.62e-2 | 2.62e-2 | 2.62e-2 | 2.62e-2 | 272 | +3.08e-5 | +3.08e-5 | +3.08e-5 | -3.73e-4 |
| 110 | 3.00e-2 | 1 | 2.86e-2 | 2.86e-2 | 2.86e-2 | 2.86e-2 | 256 | +3.41e-4 | +3.41e-4 | +3.41e-4 | -3.01e-4 |
| 111 | 3.00e-2 | 1 | 2.89e-2 | 2.89e-2 | 2.89e-2 | 2.89e-2 | 257 | +4.63e-5 | +4.63e-5 | +4.63e-5 | -2.66e-4 |
| 112 | 3.00e-2 | 1 | 2.98e-2 | 2.98e-2 | 2.98e-2 | 2.98e-2 | 259 | +1.18e-4 | +1.18e-4 | +1.18e-4 | -2.28e-4 |
| 113 | 3.00e-2 | 1 | 2.99e-2 | 2.99e-2 | 2.99e-2 | 2.99e-2 | 258 | +3.05e-6 | +3.05e-6 | +3.05e-6 | -2.05e-4 |
| 114 | 3.00e-2 | 1 | 3.10e-2 | 3.10e-2 | 3.10e-2 | 3.10e-2 | 250 | +1.50e-4 | +1.50e-4 | +1.50e-4 | -1.69e-4 |
| 115 | 3.00e-2 | 2 | 3.12e-2 | 3.12e-2 | 3.12e-2 | 3.12e-2 | 218 | -7.65e-7 | +2.67e-5 | +1.30e-5 | -1.35e-4 |
| 116 | 3.00e-2 | 1 | 3.06e-2 | 3.06e-2 | 3.06e-2 | 3.06e-2 | 249 | -8.15e-5 | -8.15e-5 | -8.15e-5 | -1.30e-4 |
| 117 | 3.00e-2 | 1 | 3.37e-2 | 3.37e-2 | 3.37e-2 | 3.37e-2 | 257 | +3.85e-4 | +3.85e-4 | +3.85e-4 | -7.81e-5 |
| 118 | 3.00e-2 | 1 | 3.41e-2 | 3.41e-2 | 3.41e-2 | 3.41e-2 | 275 | +3.35e-5 | +3.35e-5 | +3.35e-5 | -6.70e-5 |
| 119 | 3.00e-2 | 1 | 3.56e-2 | 3.56e-2 | 3.56e-2 | 3.56e-2 | 266 | +1.65e-4 | +1.65e-4 | +1.65e-4 | -4.38e-5 |
| 120 | 3.00e-2 | 1 | 3.64e-2 | 3.64e-2 | 3.64e-2 | 3.64e-2 | 256 | +8.82e-5 | +8.82e-5 | +8.82e-5 | -3.06e-5 |
| 121 | 3.00e-2 | 1 | 3.63e-2 | 3.63e-2 | 3.63e-2 | 3.63e-2 | 248 | -1.45e-5 | -1.45e-5 | -1.45e-5 | -2.90e-5 |
| 122 | 3.00e-2 | 1 | 3.66e-2 | 3.66e-2 | 3.66e-2 | 3.66e-2 | 276 | +3.07e-5 | +3.07e-5 | +3.07e-5 | -2.30e-5 |
| 123 | 3.00e-2 | 1 | 4.02e-2 | 4.02e-2 | 4.02e-2 | 4.02e-2 | 260 | +3.69e-4 | +3.69e-4 | +3.69e-4 | +1.61e-5 |
| 124 | 3.00e-2 | 1 | 3.93e-2 | 3.93e-2 | 3.93e-2 | 3.93e-2 | 297 | -7.83e-5 | -7.83e-5 | -7.83e-5 | +6.70e-6 |
| 125 | 3.00e-2 | 1 | 4.14e-2 | 4.14e-2 | 4.14e-2 | 4.14e-2 | 283 | +1.82e-4 | +1.82e-4 | +1.82e-4 | +2.43e-5 |
| 126 | 3.00e-2 | 1 | 4.17e-2 | 4.17e-2 | 4.17e-2 | 4.17e-2 | 257 | +2.83e-5 | +2.83e-5 | +2.83e-5 | +2.47e-5 |
| 127 | 3.00e-2 | 1 | 4.10e-2 | 4.10e-2 | 4.10e-2 | 4.10e-2 | 249 | -6.58e-5 | -6.58e-5 | -6.58e-5 | +1.56e-5 |
| 128 | 3.00e-2 | 1 | 4.11e-2 | 4.11e-2 | 4.11e-2 | 4.11e-2 | 235 | +4.89e-6 | +4.89e-6 | +4.89e-6 | +1.45e-5 |
| 129 | 3.00e-2 | 1 | 4.15e-2 | 4.15e-2 | 4.15e-2 | 4.15e-2 | 271 | +3.60e-5 | +3.60e-5 | +3.60e-5 | +1.67e-5 |
| 130 | 3.00e-2 | 1 | 4.43e-2 | 4.43e-2 | 4.43e-2 | 4.43e-2 | 285 | +2.34e-4 | +2.34e-4 | +2.34e-4 | +3.84e-5 |
| 131 | 3.00e-2 | 1 | 4.51e-2 | 4.51e-2 | 4.51e-2 | 4.51e-2 | 240 | +7.45e-5 | +7.45e-5 | +7.45e-5 | +4.20e-5 |
| 132 | 3.00e-2 | 2 | 4.38e-2 | 4.43e-2 | 4.41e-2 | 4.43e-2 | 216 | -1.18e-4 | +4.91e-5 | -3.47e-5 | +2.83e-5 |
| 133 | 3.00e-2 | 1 | 4.25e-2 | 4.25e-2 | 4.25e-2 | 4.25e-2 | 247 | -1.68e-4 | -1.68e-4 | -1.68e-4 | +8.71e-6 |
| 134 | 3.00e-2 | 1 | 4.60e-2 | 4.60e-2 | 4.60e-2 | 4.60e-2 | 237 | +3.36e-4 | +3.36e-4 | +3.36e-4 | +4.14e-5 |
| 135 | 3.00e-2 | 1 | 4.45e-2 | 4.45e-2 | 4.45e-2 | 4.45e-2 | 270 | -1.24e-4 | -1.24e-4 | -1.24e-4 | +2.49e-5 |
| 136 | 3.00e-2 | 1 | 4.77e-2 | 4.77e-2 | 4.77e-2 | 4.77e-2 | 293 | +2.39e-4 | +2.39e-4 | +2.39e-4 | +4.63e-5 |
| 137 | 3.00e-2 | 1 | 5.04e-2 | 5.04e-2 | 5.04e-2 | 5.04e-2 | 290 | +1.86e-4 | +1.86e-4 | +1.86e-4 | +6.02e-5 |
| 138 | 3.00e-2 | 1 | 5.18e-2 | 5.18e-2 | 5.18e-2 | 5.18e-2 | 265 | +1.05e-4 | +1.05e-4 | +1.05e-4 | +6.47e-5 |
| 139 | 3.00e-2 | 1 | 4.94e-2 | 4.94e-2 | 4.94e-2 | 4.94e-2 | 287 | -1.67e-4 | -1.67e-4 | -1.67e-4 | +4.15e-5 |
| 140 | 3.00e-2 | 1 | 5.13e-2 | 5.13e-2 | 5.13e-2 | 5.13e-2 | 242 | +1.61e-4 | +1.61e-4 | +1.61e-4 | +5.35e-5 |
| 141 | 3.00e-2 | 1 | 4.91e-2 | 4.91e-2 | 4.91e-2 | 4.91e-2 | 260 | -1.73e-4 | -1.73e-4 | -1.73e-4 | +3.08e-5 |
| 142 | 3.00e-2 | 2 | 5.02e-2 | 5.20e-2 | 5.11e-2 | 5.20e-2 | 211 | +9.19e-5 | +1.66e-4 | +1.29e-4 | +4.98e-5 |
| 144 | 3.00e-2 | 2 | 4.83e-2 | 5.44e-2 | 5.14e-2 | 5.44e-2 | 202 | -2.60e-4 | +5.96e-4 | +1.68e-4 | +7.65e-5 |
| 145 | 3.00e-2 | 1 | 4.93e-2 | 4.93e-2 | 4.93e-2 | 4.93e-2 | 240 | -4.12e-4 | -4.12e-4 | -4.12e-4 | +2.77e-5 |
| 146 | 3.00e-2 | 1 | 5.36e-2 | 5.36e-2 | 5.36e-2 | 5.36e-2 | 245 | +3.41e-4 | +3.41e-4 | +3.41e-4 | +5.91e-5 |
| 147 | 3.00e-2 | 1 | 5.43e-2 | 5.43e-2 | 5.43e-2 | 5.43e-2 | 270 | +4.83e-5 | +4.83e-5 | +4.83e-5 | +5.80e-5 |
| 148 | 3.00e-2 | 1 | 5.54e-2 | 5.54e-2 | 5.54e-2 | 5.54e-2 | 260 | +7.32e-5 | +7.32e-5 | +7.32e-5 | +5.95e-5 |
| 149 | 3.00e-2 | 1 | 5.53e-2 | 5.53e-2 | 5.53e-2 | 5.53e-2 | 261 | -2.19e-6 | -2.19e-6 | -2.19e-6 | +5.33e-5 |
| 150 | 3.00e-3 | 1 | 5.66e-2 | 5.66e-2 | 5.66e-2 | 5.66e-2 | 234 | +9.24e-5 | +9.24e-5 | +9.24e-5 | +5.72e-5 |
| 151 | 3.00e-3 | 2 | 5.62e-3 | 5.57e-2 | 3.07e-2 | 5.62e-3 | 213 | -1.08e-2 | -5.95e-5 | -5.42e-3 | -1.04e-3 |
| 152 | 3.00e-3 | 1 | 5.14e-3 | 5.14e-3 | 5.14e-3 | 5.14e-3 | 283 | -3.09e-4 | -3.09e-4 | -3.09e-4 | -9.64e-4 |
| 153 | 3.00e-3 | 1 | 5.84e-3 | 5.84e-3 | 5.84e-3 | 5.84e-3 | 260 | +4.87e-4 | +4.87e-4 | +4.87e-4 | -8.19e-4 |
| 154 | 3.00e-3 | 1 | 5.74e-3 | 5.74e-3 | 5.74e-3 | 5.74e-3 | 260 | -6.69e-5 | -6.69e-5 | -6.69e-5 | -7.44e-4 |
| 155 | 3.00e-3 | 1 | 5.70e-3 | 5.70e-3 | 5.70e-3 | 5.70e-3 | 251 | -2.55e-5 | -2.55e-5 | -2.55e-5 | -6.72e-4 |
| 156 | 3.00e-3 | 1 | 5.48e-3 | 5.48e-3 | 5.48e-3 | 5.48e-3 | 266 | -1.47e-4 | -1.47e-4 | -1.47e-4 | -6.19e-4 |
| 157 | 3.00e-3 | 1 | 5.64e-3 | 5.64e-3 | 5.64e-3 | 5.64e-3 | 272 | +1.03e-4 | +1.03e-4 | +1.03e-4 | -5.47e-4 |
| 158 | 3.00e-3 | 1 | 5.71e-3 | 5.71e-3 | 5.71e-3 | 5.71e-3 | 253 | +5.00e-5 | +5.00e-5 | +5.00e-5 | -4.87e-4 |
| 159 | 3.00e-3 | 1 | 5.79e-3 | 5.79e-3 | 5.79e-3 | 5.79e-3 | 280 | +5.26e-5 | +5.26e-5 | +5.26e-5 | -4.33e-4 |
| 160 | 3.00e-3 | 1 | 6.00e-3 | 6.00e-3 | 6.00e-3 | 6.00e-3 | 272 | +1.27e-4 | +1.27e-4 | +1.27e-4 | -3.77e-4 |
| 161 | 3.00e-3 | 2 | 6.11e-3 | 6.12e-3 | 6.12e-3 | 6.12e-3 | 224 | +1.20e-5 | +6.73e-5 | +3.97e-5 | -2.98e-4 |
| 163 | 3.00e-3 | 1 | 5.60e-3 | 5.60e-3 | 5.60e-3 | 5.60e-3 | 290 | -3.07e-4 | -3.07e-4 | -3.07e-4 | -2.99e-4 |
| 164 | 3.00e-3 | 2 | 6.05e-3 | 6.45e-3 | 6.25e-3 | 6.05e-3 | 224 | -2.92e-4 | +5.67e-4 | +1.38e-4 | -2.21e-4 |
| 165 | 3.00e-3 | 1 | 5.71e-3 | 5.71e-3 | 5.71e-3 | 5.71e-3 | 251 | -2.26e-4 | -2.26e-4 | -2.26e-4 | -2.21e-4 |
| 166 | 3.00e-3 | 1 | 6.06e-3 | 6.06e-3 | 6.06e-3 | 6.06e-3 | 263 | +2.27e-4 | +2.27e-4 | +2.27e-4 | -1.76e-4 |
| 167 | 3.00e-3 | 1 | 6.14e-3 | 6.14e-3 | 6.14e-3 | 6.14e-3 | 230 | +5.49e-5 | +5.49e-5 | +5.49e-5 | -1.53e-4 |
| 168 | 3.00e-3 | 1 | 5.81e-3 | 5.81e-3 | 5.81e-3 | 5.81e-3 | 283 | -1.94e-4 | -1.94e-4 | -1.94e-4 | -1.57e-4 |
| 169 | 3.00e-3 | 1 | 6.42e-3 | 6.42e-3 | 6.42e-3 | 6.42e-3 | 296 | +3.35e-4 | +3.35e-4 | +3.35e-4 | -1.08e-4 |
| 170 | 3.00e-3 | 1 | 6.31e-3 | 6.31e-3 | 6.31e-3 | 6.31e-3 | 283 | -5.89e-5 | -5.89e-5 | -5.89e-5 | -1.03e-4 |
| 171 | 3.00e-3 | 1 | 6.37e-3 | 6.37e-3 | 6.37e-3 | 6.37e-3 | 302 | +3.13e-5 | +3.13e-5 | +3.13e-5 | -8.97e-5 |
| 172 | 3.00e-3 | 1 | 6.64e-3 | 6.64e-3 | 6.64e-3 | 6.64e-3 | 243 | +1.70e-4 | +1.70e-4 | +1.70e-4 | -6.37e-5 |
| 173 | 3.00e-3 | 1 | 6.08e-3 | 6.08e-3 | 6.08e-3 | 6.08e-3 | 241 | -3.65e-4 | -3.65e-4 | -3.65e-4 | -9.39e-5 |
| 174 | 3.00e-3 | 1 | 5.93e-3 | 5.93e-3 | 5.93e-3 | 5.93e-3 | 236 | -1.04e-4 | -1.04e-4 | -1.04e-4 | -9.49e-5 |
| 175 | 3.00e-3 | 1 | 6.07e-3 | 6.07e-3 | 6.07e-3 | 6.07e-3 | 254 | +8.55e-5 | +8.55e-5 | +8.55e-5 | -7.68e-5 |
| 176 | 3.00e-3 | 1 | 6.30e-3 | 6.30e-3 | 6.30e-3 | 6.30e-3 | 260 | +1.47e-4 | +1.47e-4 | +1.47e-4 | -5.45e-5 |
| 177 | 3.00e-3 | 2 | 6.34e-3 | 6.36e-3 | 6.35e-3 | 6.34e-3 | 236 | -1.67e-5 | +3.40e-5 | +8.66e-6 | -4.27e-5 |
| 178 | 3.00e-3 | 1 | 6.23e-3 | 6.23e-3 | 6.23e-3 | 6.23e-3 | 290 | -5.83e-5 | -5.83e-5 | -5.83e-5 | -4.43e-5 |
| 179 | 3.00e-3 | 1 | 6.77e-3 | 6.77e-3 | 6.77e-3 | 6.77e-3 | 314 | +2.63e-4 | +2.63e-4 | +2.63e-4 | -1.35e-5 |
| 181 | 3.00e-3 | 2 | 6.97e-3 | 7.09e-3 | 7.03e-3 | 7.09e-3 | 235 | +7.08e-5 | +8.84e-5 | +7.96e-5 | +4.08e-6 |
| 182 | 3.00e-3 | 1 | 6.36e-3 | 6.36e-3 | 6.36e-3 | 6.36e-3 | 257 | -4.19e-4 | -4.19e-4 | -4.19e-4 | -3.82e-5 |
| 183 | 3.00e-3 | 1 | 6.37e-3 | 6.37e-3 | 6.37e-3 | 6.37e-3 | 239 | +8.17e-6 | +8.17e-6 | +8.17e-6 | -3.36e-5 |
| 184 | 3.00e-3 | 1 | 6.30e-3 | 6.30e-3 | 6.30e-3 | 6.30e-3 | 229 | -5.32e-5 | -5.32e-5 | -5.32e-5 | -3.55e-5 |
| 185 | 3.00e-3 | 1 | 6.20e-3 | 6.20e-3 | 6.20e-3 | 6.20e-3 | 225 | -7.02e-5 | -7.02e-5 | -7.02e-5 | -3.90e-5 |
| 186 | 3.00e-3 | 1 | 6.38e-3 | 6.38e-3 | 6.38e-3 | 6.38e-3 | 250 | +1.17e-4 | +1.17e-4 | +1.17e-4 | -2.34e-5 |
| 187 | 3.00e-3 | 1 | 6.39e-3 | 6.39e-3 | 6.39e-3 | 6.39e-3 | 281 | +2.38e-6 | +2.38e-6 | +2.38e-6 | -2.08e-5 |
| 188 | 3.00e-3 | 1 | 6.75e-3 | 6.75e-3 | 6.75e-3 | 6.75e-3 | 276 | +2.03e-4 | +2.03e-4 | +2.03e-4 | +1.53e-6 |
| 189 | 3.00e-3 | 2 | 6.57e-3 | 6.71e-3 | 6.64e-3 | 6.57e-3 | 232 | -9.25e-5 | -2.50e-5 | -5.88e-5 | -1.03e-5 |
| 191 | 3.00e-3 | 2 | 6.43e-3 | 7.05e-3 | 6.74e-3 | 7.05e-3 | 233 | -7.63e-5 | +3.95e-4 | +1.59e-4 | +2.43e-5 |
| 192 | 3.00e-3 | 1 | 6.48e-3 | 6.48e-3 | 6.48e-3 | 6.48e-3 | 265 | -3.16e-4 | -3.16e-4 | -3.16e-4 | -9.68e-6 |
| 193 | 3.00e-3 | 1 | 6.76e-3 | 6.76e-3 | 6.76e-3 | 6.76e-3 | 275 | +1.52e-4 | +1.52e-4 | +1.52e-4 | +6.45e-6 |
| 194 | 3.00e-3 | 1 | 6.97e-3 | 6.97e-3 | 6.97e-3 | 6.97e-3 | 272 | +1.17e-4 | +1.17e-4 | +1.17e-4 | +1.75e-5 |
| 195 | 3.00e-3 | 1 | 6.99e-3 | 6.99e-3 | 6.99e-3 | 6.99e-3 | 265 | +1.04e-5 | +1.04e-5 | +1.04e-5 | +1.68e-5 |
| 196 | 3.00e-3 | 1 | 6.66e-3 | 6.66e-3 | 6.66e-3 | 6.66e-3 | 246 | -2.01e-4 | -2.01e-4 | -2.01e-4 | -5.00e-6 |
| 197 | 3.00e-3 | 1 | 6.69e-3 | 6.69e-3 | 6.69e-3 | 6.69e-3 | 238 | +2.25e-5 | +2.25e-5 | +2.25e-5 | -2.25e-6 |
| 198 | 3.00e-3 | 1 | 6.66e-3 | 6.66e-3 | 6.66e-3 | 6.66e-3 | 255 | -1.79e-5 | -1.79e-5 | -1.79e-5 | -3.81e-6 |
| 199 | 3.00e-3 | 1 | 6.89e-3 | 6.89e-3 | 6.89e-3 | 6.89e-3 | 259 | +1.32e-4 | +1.32e-4 | +1.32e-4 | +9.75e-6 |

