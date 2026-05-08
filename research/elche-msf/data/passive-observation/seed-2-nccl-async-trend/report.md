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
| nccl-async | 0.048872 | 0.9156 | +0.0031 | 1986.1 | 802 | 42.6 | 100% | 100% | 100% | 6.5 |

## Best Mode per Model

| Model | Best Eval | Mode | Fastest (within 2% of solo-0) | Mode |
|-------|-----------|------|-------------------------------|------|
| resnet-graph | 0.9156 | nccl-async | - | - |

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
| nccl-async | 2.0147 | 0.8072 | 0.6293 | 0.5658 | 0.5204 | 0.5190 | 0.4902 | 0.4898 | 0.4839 | 0.4515 | 0.2014 | 0.1626 | 0.1300 | 0.1106 | 0.1025 | 0.0640 | 0.0578 | 0.0554 | 0.0504 | 0.0489 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | nccl-async | 0 | cuda:0 | 0.3965 | 2.5 |
| resnet-graph | nccl-async | 1 | cuda:1 | 0.3053 | 3.2 |
| resnet-graph | nccl-async | 2 | cuda:2 | 0.2982 | 3.2 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) | GPU2 Peak (MB) | GPU2 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|---------------|---------------|
| resnet-graph | nccl-async | 356 | 355 | 394 | 391 | 384 | 379 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| nccl-async | gpu1 | 1984.8 | 1.2 | epoch-boundary(199) |
| nccl-async | gpu2 | 1984.8 | 1.2 | epoch-boundary(199) |
| nccl-async | gpu0 | 1984.8 | 0.5 | epoch-boundary(199) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | nccl-async | gpu0 | 0.5s | 0.0s | 0.0s | 0.0s | 1.9s |
| resnet-graph | nccl-async | gpu1 | 1.2s | 0.0s | 0.0s | 0.0s | 2.6s |
| resnet-graph | nccl-async | gpu2 | 1.2s | 0.0s | 0.0s | 0.0s | 2.0s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | nccl-async | 399 | 0 | 802 | 42.6 | 701/9622 | 0 | 0.0 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | nccl-async | 191.5 | 9.6% |

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
| resnet-graph | nccl-async | 197 | 802 | 0 | 6.93e-3 | -1.68e-5 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | nccl-async | 0 | 802 | 4.56e-2 | 6.53e-2 | 0.00e0 | 4.33e-1 | 45.0 | -1.28e-4 | 3.95e-3 |
| resnet-graph | nccl-async | 1 | 802 | 4.62e-2 | 6.66e-2 | 0.00e0 | 4.15e-1 | 34.3 | -1.29e-4 | 5.72e-3 |
| resnet-graph | nccl-async | 2 | 802 | 4.58e-2 | 6.63e-2 | 0.00e0 | 4.28e-1 | 20.7 | -1.28e-4 | 5.72e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | nccl-async | rank0 ↔ rank1 | +0.9976 |
| resnet-graph | nccl-async | rank0 ↔ rank2 | +0.9980 |
| resnet-graph | nccl-async | rank1 ↔ rank2 | +0.9996 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | nccl-async | 59 (0,1,2,3,4,5,21,26…149,150) | 3 (128,137,138) | 128,137,138 | 0,1,2,3,4,5,21,26…149,150 | — |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | nccl-async | 0e0 | 3 | 38 | 38 |
| resnet-graph | nccl-async | 0e0 | 5 | 18 | 18 |
| resnet-graph | nccl-async | 0e0 | 10 | 6 | 6 |
| resnet-graph | nccl-async | 1e-4 | 3 | 23 | 23 |
| resnet-graph | nccl-async | 1e-4 | 5 | 14 | 14 |
| resnet-graph | nccl-async | 1e-4 | 10 | 4 | 4 |
| resnet-graph | nccl-async | 1e-3 | 3 | 3 | 3 |
| resnet-graph | nccl-async | 1e-3 | 5 | 2 | 2 |
| resnet-graph | nccl-async | 1e-3 | 10 | 1 | 1 |
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
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 197 | +0.149 |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | 516 | +0.009 |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | 84 | +0.134 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | nccl-async | 799 | +0.017 | 196 | +0.126 | +0.145 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | nccl-async | 800 | 3.36e1–7.97e1 | 6.45e1 | 1.56e-3 | 4.79e-3 | 8.20e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 199 | 74–78043 | +1.220e-5 | 0.447 | +1.252e-5 | 0.454 | 99 | +2.625e-6 | 0.056 | 31–912 | +1.468e-3 | 0.716 |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 183 | 901–78043 | +1.127e-5 | 0.460 | +1.148e-5 | 0.460 | 98 | +1.927e-6 | 0.034 | 38–912 | +1.417e-3 | 0.782 |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | 517 | 78404–117108 | +5.519e-6 | 0.024 | +4.854e-6 | 0.019 | 50 | -9.419e-6 | 0.074 | 35–404 | +2.100e-3 | 0.172 |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | 85 | 117251–156068 | +3.978e-5 | 0.438 | +4.080e-5 | 0.447 | 48 | +1.899e-5 | 0.295 | 44–989 | +1.753e-3 | 0.622 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | +1.468e-3 | r0: +1.433e-3, r1: +1.488e-3, r2: +1.488e-3 | r0: 0.713, r1: 0.718, r2: 0.712 | 1.04× | ✓ |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +1.417e-3 | r0: +1.392e-3, r1: +1.427e-3, r2: +1.436e-3 | r0: 0.797, r1: 0.776, r2: 0.770 | 1.03× | ✓ |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | +2.100e-3 | r0: +2.049e-3, r1: +2.103e-3, r2: +2.167e-3 | r0: 0.196, r1: 0.157, r2: 0.161 | 1.06× | ✓ |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | +1.753e-3 | r0: +1.725e-3, r1: +1.772e-3, r2: +1.765e-3 | r0: 0.635, r1: 0.615, r2: 0.614 | 1.03× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | nccl-async | `█▇███████████████████████▄▄▄▄▄▄▄▄▄▄▄▅▂▁▂▂▂▂▂▂▂▂▂▂▂` | `▂▆▇▇▇▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▇▆▆▄▆▆▅▅▆█▇▆█▇█▁▅▆▆▆▆▆▆▆▆▆▆▆` |

### Per-Epoch Detail: resnet-graph / nccl-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 17 | 0.00e0 | 4.33e-1 | 9.13e-2 | 5.96e-2 | 18 | -9.20e-2 | +1.61e-2 | -1.02e-2 | -4.82e-3 |
| 1 | 3.00e-1 | 15 | 5.29e-2 | 1.04e-1 | 6.18e-2 | 5.60e-2 | 15 | -3.77e-2 | +3.52e-2 | -2.19e-4 | -1.31e-3 |
| 2 | 3.00e-1 | 10 | 5.53e-2 | 1.25e-1 | 8.46e-2 | 8.04e-2 | 25 | -1.71e-2 | +3.01e-2 | +1.35e-3 | -1.65e-5 |
| 3 | 3.00e-1 | 9 | 7.88e-2 | 1.31e-1 | 9.17e-2 | 9.25e-2 | 23 | -2.06e-2 | +1.97e-2 | +5.07e-4 | +2.38e-4 |
| 4 | 3.00e-1 | 13 | 7.09e-2 | 1.39e-1 | 8.70e-2 | 8.78e-2 | 20 | -3.19e-2 | +2.23e-2 | -6.36e-4 | -9.20e-5 |
| 5 | 3.00e-1 | 13 | 6.85e-2 | 1.47e-1 | 8.48e-2 | 8.71e-2 | 17 | -2.47e-2 | +2.69e-2 | -1.09e-4 | +2.68e-4 |
| 6 | 3.00e-1 | 3 | 7.58e-2 | 7.90e-2 | 7.77e-2 | 7.58e-2 | 244 | -5.73e-3 | -1.29e-4 | -2.16e-3 | -3.38e-4 |
| 7 | 3.00e-1 | 1 | 2.33e-1 | 2.33e-1 | 2.33e-1 | 2.33e-1 | 256 | +4.39e-3 | +4.39e-3 | +4.39e-3 | +1.34e-4 |
| 8 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 283 | -2.52e-4 | -2.52e-4 | -2.52e-4 | +9.58e-5 |
| 9 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 277 | -7.78e-5 | -7.78e-5 | -7.78e-5 | +7.84e-5 |
| 10 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 300 | -9.58e-5 | -9.58e-5 | -9.58e-5 | +6.10e-5 |
| 11 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 267 | +9.63e-5 | +9.63e-5 | +9.63e-5 | +6.45e-5 |
| 13 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 341 | -2.03e-4 | -2.03e-4 | -2.03e-4 | +3.78e-5 |
| 14 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 302 | +2.40e-4 | +2.40e-4 | +2.40e-4 | +5.80e-5 |
| 15 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 300 | -1.26e-4 | -1.26e-4 | -1.26e-4 | +3.96e-5 |
| 16 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 304 | +8.96e-6 | +8.96e-6 | +8.96e-6 | +3.65e-5 |
| 17 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 308 | -3.73e-5 | -3.73e-5 | -3.73e-5 | +2.91e-5 |
| 18 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 292 | +5.25e-5 | +5.25e-5 | +5.25e-5 | +3.15e-5 |
| 19 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 299 | -2.16e-5 | -2.16e-5 | -2.16e-5 | +2.62e-5 |
| 20 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 317 | +4.12e-5 | +4.12e-5 | +4.12e-5 | +2.77e-5 |
| 21 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 299 | +7.08e-5 | +7.08e-5 | +7.08e-5 | +3.20e-5 |
| 22 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 281 | -6.68e-5 | -6.68e-5 | -6.68e-5 | +2.21e-5 |
| 23 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 268 | -7.33e-5 | -7.33e-5 | -7.33e-5 | +1.26e-5 |
| 24 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 289 | -8.22e-5 | -8.22e-5 | -8.22e-5 | +3.09e-6 |
| 25 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 294 | +8.96e-5 | +8.96e-5 | +8.96e-5 | +1.17e-5 |
| 26 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 260 | +7.55e-6 | +7.55e-6 | +7.55e-6 | +1.13e-5 |
| 27 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 266 | -5.16e-5 | -5.16e-5 | -5.16e-5 | +5.02e-6 |
| 28 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 315 | -6.54e-6 | -6.54e-6 | -6.54e-6 | +3.87e-6 |
| 29 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 283 | +1.45e-4 | +1.45e-4 | +1.45e-4 | +1.80e-5 |
| 30 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 285 | -5.77e-5 | -5.77e-5 | -5.77e-5 | +1.04e-5 |
| 31 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 265 | -2.80e-5 | -2.80e-5 | -2.80e-5 | +6.58e-6 |
| 32 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 251 | -7.76e-5 | -7.76e-5 | -7.76e-5 | -1.84e-6 |
| 33 | 3.00e-1 | 1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 247 | -1.30e-4 | -1.30e-4 | -1.30e-4 | -1.47e-5 |
| 34 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 244 | +8.10e-5 | +8.10e-5 | +8.10e-5 | -5.12e-6 |
| 35 | 3.00e-1 | 1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 243 | -9.26e-5 | -9.26e-5 | -9.26e-5 | -1.39e-5 |
| 36 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 300 | +8.16e-5 | +8.16e-5 | +8.16e-5 | -4.32e-6 |
| 37 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 270 | +2.24e-4 | +2.24e-4 | +2.24e-4 | +1.85e-5 |
| 38 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 293 | -1.10e-4 | -1.10e-4 | -1.10e-4 | +5.66e-6 |
| 39 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 278 | +9.27e-5 | +9.27e-5 | +9.27e-5 | +1.44e-5 |
| 40 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 225 | -8.94e-6 | -8.94e-6 | -8.94e-6 | +1.20e-5 |
| 41 | 3.00e-1 | 2 | 1.90e-1 | 1.96e-1 | 1.93e-1 | 1.96e-1 | 212 | -4.09e-4 | +1.48e-4 | -1.31e-4 | -1.23e-5 |
| 42 | 3.00e-1 | 1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 221 | -1.48e-4 | -1.48e-4 | -1.48e-4 | -2.59e-5 |
| 43 | 3.00e-1 | 1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 252 | +1.12e-4 | +1.12e-4 | +1.12e-4 | -1.21e-5 |
| 44 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 248 | +9.11e-5 | +9.11e-5 | +9.11e-5 | -1.77e-6 |
| 45 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 240 | +2.40e-6 | +2.40e-6 | +2.40e-6 | -1.35e-6 |
| 46 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 271 | -1.85e-5 | -1.85e-5 | -1.85e-5 | -3.07e-6 |
| 47 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 267 | +2.03e-4 | +2.03e-4 | +2.03e-4 | +1.76e-5 |
| 48 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 234 | -9.38e-5 | -9.38e-5 | -9.38e-5 | +6.43e-6 |
| 49 | 3.00e-1 | 2 | 1.98e-1 | 2.00e-1 | 1.99e-1 | 2.00e-1 | 202 | -1.45e-4 | +5.18e-5 | -4.64e-5 | -2.63e-6 |
| 50 | 3.00e-1 | 1 | 1.89e-1 | 1.89e-1 | 1.89e-1 | 1.89e-1 | 256 | -2.18e-4 | -2.18e-4 | -2.18e-4 | -2.41e-5 |
| 51 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 253 | +2.80e-4 | +2.80e-4 | +2.80e-4 | +6.27e-6 |
| 52 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 246 | +3.09e-5 | +3.09e-5 | +3.09e-5 | +8.73e-6 |
| 53 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 240 | -8.04e-5 | -8.04e-5 | -8.04e-5 | -1.86e-7 |
| 54 | 3.00e-1 | 2 | 1.99e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 202 | -3.19e-5 | +3.35e-5 | +7.90e-7 | +3.27e-7 |
| 55 | 3.00e-1 | 1 | 1.91e-1 | 1.91e-1 | 1.91e-1 | 1.91e-1 | 248 | -1.89e-4 | -1.89e-4 | -1.89e-4 | -1.86e-5 |
| 56 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 237 | +2.81e-4 | +2.81e-4 | +2.81e-4 | +1.14e-5 |
| 57 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 274 | -6.07e-5 | -6.07e-5 | -6.07e-5 | +4.21e-6 |
| 58 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 241 | +1.40e-4 | +1.40e-4 | +1.40e-4 | +1.78e-5 |
| 59 | 3.00e-1 | 2 | 1.97e-1 | 1.98e-1 | 1.97e-1 | 1.97e-1 | 183 | -2.26e-4 | -1.04e-5 | -1.18e-4 | -7.01e-6 |
| 60 | 3.00e-1 | 1 | 1.89e-1 | 1.89e-1 | 1.89e-1 | 1.89e-1 | 209 | -2.11e-4 | -2.11e-4 | -2.11e-4 | -2.74e-5 |
| 61 | 3.00e-1 | 1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 211 | +8.76e-5 | +8.76e-5 | +8.76e-5 | -1.59e-5 |
| 62 | 3.00e-1 | 2 | 1.93e-1 | 2.00e-1 | 1.97e-1 | 2.00e-1 | 194 | +1.22e-5 | +1.97e-4 | +1.05e-4 | +7.91e-6 |
| 63 | 3.00e-1 | 1 | 1.88e-1 | 1.88e-1 | 1.88e-1 | 1.88e-1 | 228 | -2.66e-4 | -2.66e-4 | -2.66e-4 | -1.95e-5 |
| 64 | 3.00e-1 | 1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 221 | +1.92e-4 | +1.92e-4 | +1.92e-4 | +1.68e-6 |
| 65 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 228 | -1.32e-5 | -1.32e-5 | -1.32e-5 | +1.97e-7 |
| 66 | 3.00e-1 | 2 | 2.00e-1 | 2.05e-1 | 2.03e-1 | 2.05e-1 | 189 | +8.20e-5 | +1.36e-4 | +1.09e-4 | +2.11e-5 |
| 67 | 3.00e-1 | 1 | 1.87e-1 | 1.87e-1 | 1.87e-1 | 1.87e-1 | 233 | -4.04e-4 | -4.04e-4 | -4.04e-4 | -2.14e-5 |
| 68 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 217 | +3.44e-4 | +3.44e-4 | +3.44e-4 | +1.51e-5 |
| 69 | 3.00e-1 | 2 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 174 | -2.11e-4 | +4.29e-6 | -1.04e-4 | -6.33e-6 |
| 70 | 3.00e-1 | 1 | 1.83e-1 | 1.83e-1 | 1.83e-1 | 1.83e-1 | 224 | -2.25e-4 | -2.25e-4 | -2.25e-4 | -2.82e-5 |
| 71 | 3.00e-1 | 2 | 1.97e-1 | 1.98e-1 | 1.98e-1 | 1.97e-1 | 165 | -4.67e-5 | +3.69e-4 | +1.61e-4 | +5.68e-6 |
| 72 | 3.00e-1 | 1 | 1.82e-1 | 1.82e-1 | 1.82e-1 | 1.82e-1 | 191 | -4.22e-4 | -4.22e-4 | -4.22e-4 | -3.71e-5 |
| 73 | 3.00e-1 | 2 | 1.89e-1 | 1.94e-1 | 1.92e-1 | 1.94e-1 | 165 | +1.46e-4 | +2.01e-4 | +1.74e-4 | +2.68e-6 |
| 74 | 3.00e-1 | 1 | 1.77e-1 | 1.77e-1 | 1.77e-1 | 1.77e-1 | 205 | -4.51e-4 | -4.51e-4 | -4.51e-4 | -4.27e-5 |
| 75 | 3.00e-1 | 2 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 168 | +1.77e-5 | +4.65e-4 | +2.41e-4 | +8.99e-6 |
| 76 | 3.00e-1 | 1 | 1.85e-1 | 1.85e-1 | 1.85e-1 | 1.85e-1 | 204 | -2.57e-4 | -2.57e-4 | -2.57e-4 | -1.76e-5 |
| 77 | 3.00e-1 | 1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 219 | +2.37e-4 | +2.37e-4 | +2.37e-4 | +7.85e-6 |
| 78 | 3.00e-1 | 2 | 1.93e-1 | 1.99e-1 | 1.96e-1 | 1.93e-1 | 146 | -2.11e-4 | +1.03e-4 | -5.40e-5 | -5.48e-6 |
| 79 | 3.00e-1 | 2 | 1.71e-1 | 1.92e-1 | 1.82e-1 | 1.92e-1 | 148 | -6.12e-4 | +7.73e-4 | +8.03e-5 | +1.78e-5 |
| 80 | 3.00e-1 | 1 | 1.77e-1 | 1.77e-1 | 1.77e-1 | 1.77e-1 | 210 | -3.90e-4 | -3.90e-4 | -3.90e-4 | -2.30e-5 |
| 81 | 3.00e-1 | 2 | 1.87e-1 | 1.95e-1 | 1.91e-1 | 1.87e-1 | 146 | -3.09e-4 | +5.19e-4 | +1.05e-4 | -2.77e-6 |
| 82 | 3.00e-1 | 1 | 1.77e-1 | 1.77e-1 | 1.77e-1 | 1.77e-1 | 186 | -2.78e-4 | -2.78e-4 | -2.78e-4 | -3.03e-5 |
| 83 | 3.00e-1 | 2 | 1.85e-1 | 1.86e-1 | 1.85e-1 | 1.85e-1 | 146 | -4.08e-5 | +2.72e-4 | +1.15e-4 | -4.20e-6 |
| 84 | 3.00e-1 | 1 | 1.76e-1 | 1.76e-1 | 1.76e-1 | 1.76e-1 | 192 | -2.56e-4 | -2.56e-4 | -2.56e-4 | -2.94e-5 |
| 85 | 3.00e-1 | 3 | 1.81e-1 | 1.99e-1 | 1.91e-1 | 1.81e-1 | 149 | -6.39e-4 | +4.57e-4 | +6.27e-6 | -3.02e-5 |
| 86 | 3.00e-1 | 1 | 1.77e-1 | 1.77e-1 | 1.77e-1 | 1.77e-1 | 178 | -1.05e-4 | -1.05e-4 | -1.05e-4 | -3.77e-5 |
| 87 | 3.00e-1 | 1 | 1.87e-1 | 1.87e-1 | 1.87e-1 | 1.87e-1 | 173 | +2.88e-4 | +2.88e-4 | +2.88e-4 | -5.05e-6 |
| 88 | 3.00e-1 | 2 | 1.82e-1 | 1.85e-1 | 1.83e-1 | 1.82e-1 | 137 | -1.39e-4 | -4.35e-5 | -9.14e-5 | -2.19e-5 |
| 89 | 3.00e-1 | 2 | 1.69e-1 | 1.92e-1 | 1.81e-1 | 1.92e-1 | 137 | -3.75e-4 | +9.46e-4 | +2.85e-4 | +4.31e-5 |
| 90 | 3.00e-1 | 2 | 1.70e-1 | 1.90e-1 | 1.80e-1 | 1.90e-1 | 147 | -7.05e-4 | +7.89e-4 | +4.19e-5 | +5.03e-5 |
| 91 | 3.00e-1 | 1 | 1.78e-1 | 1.78e-1 | 1.78e-1 | 1.78e-1 | 175 | -4.00e-4 | -4.00e-4 | -4.00e-4 | +5.25e-6 |
| 92 | 3.00e-1 | 2 | 1.87e-1 | 1.95e-1 | 1.91e-1 | 1.95e-1 | 126 | +2.49e-4 | +3.43e-4 | +2.96e-4 | +6.10e-5 |
| 93 | 3.00e-1 | 2 | 1.68e-1 | 1.79e-1 | 1.73e-1 | 1.79e-1 | 125 | -9.67e-4 | +5.02e-4 | -2.32e-4 | +1.26e-5 |
| 94 | 3.00e-1 | 3 | 1.64e-1 | 1.81e-1 | 1.70e-1 | 1.64e-1 | 125 | -7.73e-4 | +6.96e-4 | -1.87e-4 | -4.46e-5 |
| 95 | 3.00e-1 | 1 | 1.66e-1 | 1.66e-1 | 1.66e-1 | 1.66e-1 | 172 | +4.37e-5 | +4.37e-5 | +4.37e-5 | -3.58e-5 |
| 96 | 3.00e-1 | 3 | 1.60e-1 | 1.89e-1 | 1.79e-1 | 1.60e-1 | 113 | -1.51e-3 | +7.56e-4 | -2.50e-4 | -1.15e-4 |
| 97 | 3.00e-1 | 1 | 1.60e-1 | 1.60e-1 | 1.60e-1 | 1.60e-1 | 151 | +1.06e-5 | +1.06e-5 | +1.06e-5 | -1.03e-4 |
| 98 | 3.00e-1 | 2 | 1.80e-1 | 1.92e-1 | 1.86e-1 | 1.92e-1 | 113 | +5.69e-4 | +6.68e-4 | +6.18e-4 | +3.37e-5 |
| 99 | 3.00e-1 | 3 | 1.59e-1 | 1.76e-1 | 1.65e-1 | 1.59e-1 | 105 | -1.18e-3 | +8.22e-4 | -4.33e-4 | -9.12e-5 |
| 100 | 3.00e-2 | 2 | 1.18e-1 | 1.55e-1 | 1.37e-1 | 1.18e-1 | 111 | -2.45e-3 | -2.22e-4 | -1.34e-3 | -3.39e-4 |
| 101 | 3.00e-2 | 3 | 1.58e-2 | 1.80e-2 | 1.65e-2 | 1.59e-2 | 105 | -1.33e-2 | +1.26e-3 | -4.40e-3 | -1.33e-3 |
| 102 | 3.00e-2 | 2 | 1.65e-2 | 1.91e-2 | 1.78e-2 | 1.91e-2 | 97 | +2.89e-4 | +1.51e-3 | +8.98e-4 | -8.98e-4 |
| 103 | 3.00e-2 | 3 | 1.64e-2 | 1.95e-2 | 1.75e-2 | 1.65e-2 | 97 | -1.73e-3 | +1.78e-3 | -3.78e-4 | -7.64e-4 |
| 104 | 3.00e-2 | 2 | 1.71e-2 | 1.97e-2 | 1.84e-2 | 1.97e-2 | 97 | +2.93e-4 | +1.46e-3 | +8.76e-4 | -4.46e-4 |
| 105 | 3.00e-2 | 4 | 1.77e-2 | 2.07e-2 | 1.86e-2 | 1.83e-2 | 87 | -1.60e-3 | +1.73e-3 | -1.04e-4 | -3.29e-4 |
| 106 | 3.00e-2 | 2 | 1.73e-2 | 2.08e-2 | 1.91e-2 | 2.08e-2 | 96 | -4.08e-4 | +1.91e-3 | +7.50e-4 | -1.13e-4 |
| 107 | 3.00e-2 | 2 | 1.94e-2 | 2.19e-2 | 2.06e-2 | 2.19e-2 | 80 | -5.43e-4 | +1.54e-3 | +4.97e-4 | +1.36e-5 |
| 108 | 3.00e-2 | 4 | 1.77e-2 | 2.09e-2 | 1.85e-2 | 1.77e-2 | 80 | -2.06e-3 | +1.97e-3 | -4.51e-4 | -1.43e-4 |
| 109 | 3.00e-2 | 2 | 1.80e-2 | 2.26e-2 | 2.03e-2 | 2.26e-2 | 90 | +1.19e-4 | +2.57e-3 | +1.34e-3 | +1.52e-4 |
| 110 | 3.00e-2 | 5 | 1.69e-2 | 2.17e-2 | 1.86e-2 | 1.71e-2 | 64 | -3.67e-3 | +1.10e-3 | -6.88e-4 | -1.85e-4 |
| 111 | 3.00e-2 | 3 | 1.67e-2 | 2.22e-2 | 1.89e-2 | 1.77e-2 | 67 | -3.42e-3 | +4.37e-3 | +2.43e-4 | -1.01e-4 |
| 112 | 3.00e-2 | 4 | 1.55e-2 | 2.05e-2 | 1.78e-2 | 1.55e-2 | 52 | -4.38e-3 | +1.63e-3 | -7.85e-4 | -3.84e-4 |
| 113 | 3.00e-2 | 6 | 1.59e-2 | 2.05e-2 | 1.71e-2 | 1.61e-2 | 46 | -3.56e-3 | +4.62e-3 | +6.98e-5 | -2.35e-4 |
| 114 | 3.00e-2 | 5 | 1.44e-2 | 1.99e-2 | 1.58e-2 | 1.53e-2 | 46 | -6.60e-3 | +7.50e-3 | -2.28e-5 | -1.76e-4 |
| 115 | 3.00e-2 | 8 | 1.39e-2 | 2.23e-2 | 1.63e-2 | 1.50e-2 | 33 | -7.45e-3 | +7.68e-3 | -7.47e-5 | -1.56e-4 |
| 116 | 3.00e-2 | 6 | 1.20e-2 | 2.13e-2 | 1.41e-2 | 1.20e-2 | 26 | -1.72e-2 | +1.52e-2 | -1.02e-3 | -6.69e-4 |
| 117 | 3.00e-2 | 11 | 9.61e-3 | 1.87e-2 | 1.12e-2 | 9.86e-3 | 20 | -2.65e-2 | +2.09e-2 | -8.40e-4 | -8.46e-4 |
| 118 | 3.00e-2 | 14 | 8.02e-3 | 1.83e-2 | 1.03e-2 | 9.62e-3 | 20 | -3.32e-2 | +3.14e-2 | -2.85e-4 | -4.48e-4 |
| 119 | 3.00e-2 | 15 | 8.22e-3 | 2.05e-2 | 1.01e-2 | 9.28e-3 | 18 | -4.64e-2 | +4.69e-2 | +2.87e-5 | -2.02e-4 |
| 120 | 3.00e-2 | 13 | 8.43e-3 | 2.08e-2 | 1.07e-2 | 9.96e-3 | 19 | -5.39e-2 | +5.55e-2 | +4.73e-4 | -1.28e-4 |
| 121 | 3.00e-2 | 14 | 8.65e-3 | 2.07e-2 | 1.05e-2 | 9.69e-3 | 18 | -4.91e-2 | +3.89e-2 | -4.78e-4 | -3.38e-4 |
| 122 | 3.00e-2 | 14 | 8.37e-3 | 2.22e-2 | 1.11e-2 | 9.88e-3 | 18 | -6.09e-2 | +5.31e-2 | +8.92e-5 | -3.66e-4 |
| 123 | 3.00e-2 | 22 | 8.01e-3 | 2.01e-2 | 9.66e-3 | 9.91e-3 | 16 | -4.55e-2 | +4.28e-2 | -1.14e-4 | +3.11e-4 |
| 124 | 3.00e-2 | 11 | 8.92e-3 | 2.14e-2 | 1.09e-2 | 1.09e-2 | 19 | -4.92e-2 | +4.84e-2 | +3.58e-4 | +3.92e-4 |
| 125 | 3.00e-2 | 14 | 9.64e-3 | 2.05e-2 | 1.17e-2 | 1.09e-2 | 17 | -3.45e-2 | +3.73e-2 | +3.74e-4 | +2.09e-4 |
| 126 | 3.00e-2 | 16 | 8.58e-3 | 2.13e-2 | 1.09e-2 | 1.09e-2 | 13 | -4.50e-2 | +4.67e-2 | +7.06e-4 | +9.01e-4 |
| 127 | 3.00e-2 | 14 | 8.65e-3 | 2.12e-2 | 1.20e-2 | 1.41e-2 | 22 | -5.43e-2 | +5.99e-2 | +1.68e-3 | +1.44e-3 |
| 128 | 3.00e-2 | 15 | 9.07e-3 | 2.51e-2 | 1.16e-2 | 9.70e-3 | 16 | -5.48e-2 | +3.76e-2 | -1.29e-3 | -5.98e-4 |
| 129 | 3.00e-2 | 17 | 1.02e-2 | 2.53e-2 | 1.26e-2 | 1.49e-2 | 25 | -4.06e-2 | +4.16e-2 | +6.61e-4 | +7.76e-4 |
| 130 | 3.00e-2 | 8 | 1.30e-2 | 2.63e-2 | 1.55e-2 | 1.39e-2 | 23 | -2.74e-2 | +2.48e-2 | -3.81e-4 | +3.56e-5 |
| 131 | 3.00e-2 | 13 | 1.12e-2 | 2.79e-2 | 1.52e-2 | 1.73e-2 | 27 | -4.57e-2 | +3.62e-2 | +4.10e-4 | +5.66e-4 |
| 132 | 3.00e-2 | 9 | 1.30e-2 | 2.64e-2 | 1.61e-2 | 1.30e-2 | 20 | -2.27e-2 | +2.00e-2 | -1.42e-3 | -8.39e-4 |
| 133 | 3.00e-2 | 14 | 1.07e-2 | 2.52e-2 | 1.38e-2 | 1.20e-2 | 15 | -3.13e-2 | +2.77e-2 | -4.16e-4 | -6.54e-4 |
| 134 | 3.00e-2 | 15 | 1.09e-2 | 2.47e-2 | 1.35e-2 | 1.55e-2 | 20 | -4.83e-2 | +4.60e-2 | +1.17e-3 | +9.44e-4 |
| 135 | 3.00e-2 | 15 | 1.06e-2 | 2.63e-2 | 1.39e-2 | 1.30e-2 | 16 | -4.76e-2 | +3.60e-2 | -7.52e-4 | -3.60e-4 |
| 136 | 3.00e-2 | 20 | 1.03e-2 | 2.60e-2 | 1.30e-2 | 1.13e-2 | 13 | -5.01e-2 | +4.38e-2 | -1.01e-3 | -1.05e-3 |
| 137 | 3.00e-2 | 9 | 1.06e-2 | 2.70e-2 | 1.68e-2 | 1.89e-2 | 33 | -3.85e-2 | +6.66e-2 | +5.30e-3 | +2.10e-3 |
| 138 | 3.00e-2 | 13 | 1.28e-2 | 2.72e-2 | 1.56e-2 | 1.34e-2 | 18 | -2.62e-2 | +1.29e-2 | -1.88e-3 | -5.35e-4 |
| 139 | 3.00e-2 | 14 | 1.26e-2 | 3.05e-2 | 1.53e-2 | 1.34e-2 | 15 | -3.81e-2 | +4.45e-2 | +5.80e-5 | -3.90e-4 |
| 140 | 3.00e-2 | 13 | 1.09e-2 | 3.24e-2 | 1.61e-2 | 1.68e-2 | 21 | -3.84e-2 | +4.72e-2 | +1.08e-3 | +6.34e-4 |
| 141 | 3.00e-2 | 18 | 1.29e-2 | 3.08e-2 | 1.52e-2 | 1.40e-2 | 20 | -4.93e-2 | +3.28e-2 | -6.36e-4 | -1.98e-4 |
| 142 | 3.00e-2 | 11 | 1.42e-2 | 3.07e-2 | 1.65e-2 | 1.63e-2 | 17 | -4.24e-2 | +3.88e-2 | +2.42e-4 | +1.79e-4 |
| 143 | 3.00e-2 | 21 | 1.04e-2 | 2.86e-2 | 1.45e-2 | 1.65e-2 | 20 | -4.05e-2 | +4.19e-2 | +2.57e-4 | +7.99e-4 |
| 144 | 3.00e-2 | 9 | 1.47e-2 | 3.16e-2 | 1.77e-2 | 1.49e-2 | 18 | -3.04e-2 | +3.26e-2 | -6.19e-4 | -3.26e-4 |
| 145 | 3.00e-2 | 16 | 1.13e-2 | 2.95e-2 | 1.49e-2 | 1.60e-2 | 20 | -4.18e-2 | +4.03e-2 | -2.34e-4 | +2.79e-4 |
| 146 | 3.00e-2 | 12 | 1.53e-2 | 3.34e-2 | 1.86e-2 | 1.87e-2 | 20 | -3.25e-2 | +3.00e-2 | +2.53e-4 | +3.73e-4 |
| 147 | 3.00e-2 | 15 | 1.31e-2 | 3.29e-2 | 1.66e-2 | 1.77e-2 | 18 | -4.17e-2 | +4.08e-2 | +6.04e-4 | +8.06e-4 |
| 148 | 3.00e-2 | 11 | 1.55e-2 | 3.80e-2 | 2.30e-2 | 2.20e-2 | 27 | -1.62e-2 | +2.81e-2 | +6.49e-4 | +4.22e-4 |
| 149 | 3.00e-2 | 8 | 1.57e-2 | 3.67e-2 | 2.09e-2 | 1.90e-2 | 24 | -4.25e-2 | +2.56e-2 | -9.56e-4 | -3.20e-4 |
| 150 | 3.00e-3 | 13 | 1.49e-3 | 3.14e-2 | 5.53e-3 | 1.67e-3 | 19 | -1.37e-1 | +1.64e-2 | -9.87e-3 | -4.62e-3 |
| 151 | 3.00e-3 | 12 | 1.28e-3 | 2.86e-3 | 1.59e-3 | 1.56e-3 | 19 | -4.35e-2 | +3.95e-2 | +9.49e-5 | -1.23e-3 |
| 152 | 3.00e-3 | 11 | 1.46e-3 | 3.54e-3 | 2.00e-3 | 1.75e-3 | 20 | -1.93e-2 | +2.87e-2 | +1.71e-4 | -5.38e-4 |
| 153 | 3.00e-3 | 1 | 1.59e-3 | 1.59e-3 | 1.59e-3 | 1.59e-3 | 20 | -4.86e-3 | -4.86e-3 | -4.86e-3 | -9.70e-4 |
| 154 | 3.00e-3 | 2 | 1.35e-3 | 7.11e-3 | 4.23e-3 | 7.11e-3 | 277 | -5.41e-4 | +6.01e-3 | +2.73e-3 | -2.34e-4 |
| 156 | 3.00e-3 | 1 | 6.85e-3 | 6.85e-3 | 6.85e-3 | 6.85e-3 | 373 | -9.92e-5 | -9.92e-5 | -9.92e-5 | -2.20e-4 |
| 157 | 3.00e-3 | 1 | 7.50e-3 | 7.50e-3 | 7.50e-3 | 7.50e-3 | 279 | +3.23e-4 | +3.23e-4 | +3.23e-4 | -1.66e-4 |
| 158 | 3.00e-3 | 1 | 6.85e-3 | 6.85e-3 | 6.85e-3 | 6.85e-3 | 328 | -2.79e-4 | -2.79e-4 | -2.79e-4 | -1.77e-4 |
| 159 | 3.00e-3 | 1 | 7.29e-3 | 7.29e-3 | 7.29e-3 | 7.29e-3 | 267 | +2.37e-4 | +2.37e-4 | +2.37e-4 | -1.36e-4 |
| 160 | 3.00e-3 | 1 | 6.75e-3 | 6.75e-3 | 6.75e-3 | 6.75e-3 | 269 | -2.89e-4 | -2.89e-4 | -2.89e-4 | -1.51e-4 |
| 161 | 3.00e-3 | 1 | 6.65e-3 | 6.65e-3 | 6.65e-3 | 6.65e-3 | 349 | -4.11e-5 | -4.11e-5 | -4.11e-5 | -1.40e-4 |
| 162 | 3.00e-3 | 1 | 7.60e-3 | 7.60e-3 | 7.60e-3 | 7.60e-3 | 323 | +4.13e-4 | +4.13e-4 | +4.13e-4 | -8.48e-5 |
| 163 | 3.00e-3 | 1 | 7.10e-3 | 7.10e-3 | 7.10e-3 | 7.10e-3 | 280 | -2.43e-4 | -2.43e-4 | -2.43e-4 | -1.01e-4 |
| 164 | 3.00e-3 | 1 | 7.13e-3 | 7.13e-3 | 7.13e-3 | 7.13e-3 | 251 | +1.89e-5 | +1.89e-5 | +1.89e-5 | -8.87e-5 |
| 165 | 3.00e-3 | 1 | 6.71e-3 | 6.71e-3 | 6.71e-3 | 6.71e-3 | 250 | -2.45e-4 | -2.45e-4 | -2.45e-4 | -1.04e-4 |
| 166 | 3.00e-3 | 1 | 6.85e-3 | 6.85e-3 | 6.85e-3 | 6.85e-3 | 310 | +6.77e-5 | +6.77e-5 | +6.77e-5 | -8.72e-5 |
| 167 | 3.00e-3 | 1 | 7.27e-3 | 7.27e-3 | 7.27e-3 | 7.27e-3 | 271 | +2.17e-4 | +2.17e-4 | +2.17e-4 | -5.67e-5 |
| 168 | 3.00e-3 | 1 | 6.93e-3 | 6.93e-3 | 6.93e-3 | 6.93e-3 | 290 | -1.62e-4 | -1.62e-4 | -1.62e-4 | -6.72e-5 |
| 169 | 3.00e-3 | 1 | 7.02e-3 | 7.02e-3 | 7.02e-3 | 7.02e-3 | 290 | +4.14e-5 | +4.14e-5 | +4.14e-5 | -5.64e-5 |
| 170 | 3.00e-3 | 1 | 7.26e-3 | 7.26e-3 | 7.26e-3 | 7.26e-3 | 261 | +1.31e-4 | +1.31e-4 | +1.31e-4 | -3.77e-5 |
| 171 | 3.00e-3 | 1 | 6.76e-3 | 6.76e-3 | 6.76e-3 | 6.76e-3 | 277 | -2.58e-4 | -2.58e-4 | -2.58e-4 | -5.97e-5 |
| 172 | 3.00e-3 | 1 | 7.08e-3 | 7.08e-3 | 7.08e-3 | 7.08e-3 | 235 | +1.98e-4 | +1.98e-4 | +1.98e-4 | -3.39e-5 |
| 173 | 3.00e-3 | 1 | 6.49e-3 | 6.49e-3 | 6.49e-3 | 6.49e-3 | 257 | -3.39e-4 | -3.39e-4 | -3.39e-4 | -6.45e-5 |
| 174 | 3.00e-3 | 1 | 6.76e-3 | 6.76e-3 | 6.76e-3 | 6.76e-3 | 252 | +1.64e-4 | +1.64e-4 | +1.64e-4 | -4.16e-5 |
| 175 | 3.00e-3 | 1 | 6.94e-3 | 6.94e-3 | 6.94e-3 | 6.94e-3 | 258 | +9.74e-5 | +9.74e-5 | +9.74e-5 | -2.77e-5 |
| 176 | 3.00e-3 | 1 | 6.90e-3 | 6.90e-3 | 6.90e-3 | 6.90e-3 | 245 | -2.22e-5 | -2.22e-5 | -2.22e-5 | -2.72e-5 |
| 177 | 3.00e-3 | 1 | 6.43e-3 | 6.43e-3 | 6.43e-3 | 6.43e-3 | 289 | -2.44e-4 | -2.44e-4 | -2.44e-4 | -4.89e-5 |
| 178 | 3.00e-3 | 1 | 7.38e-3 | 7.38e-3 | 7.38e-3 | 7.38e-3 | 302 | +4.57e-4 | +4.57e-4 | +4.57e-4 | +1.65e-6 |
| 179 | 3.00e-3 | 1 | 7.32e-3 | 7.32e-3 | 7.32e-3 | 7.32e-3 | 297 | -2.67e-5 | -2.67e-5 | -2.67e-5 | -1.18e-6 |
| 180 | 3.00e-3 | 1 | 7.52e-3 | 7.52e-3 | 7.52e-3 | 7.52e-3 | 278 | +9.44e-5 | +9.44e-5 | +9.44e-5 | +8.38e-6 |
| 181 | 3.00e-3 | 1 | 6.82e-3 | 6.82e-3 | 6.82e-3 | 6.82e-3 | 267 | -3.61e-4 | -3.61e-4 | -3.61e-4 | -2.86e-5 |
| 182 | 3.00e-3 | 2 | 6.80e-3 | 6.98e-3 | 6.89e-3 | 6.80e-3 | 226 | -1.17e-4 | +9.39e-5 | -1.18e-5 | -2.64e-5 |
| 184 | 3.00e-3 | 2 | 6.66e-3 | 7.12e-3 | 6.89e-3 | 7.12e-3 | 217 | -7.45e-5 | +3.12e-4 | +1.19e-4 | +3.06e-6 |
| 185 | 3.00e-3 | 1 | 6.42e-3 | 6.42e-3 | 6.42e-3 | 6.42e-3 | 227 | -4.59e-4 | -4.59e-4 | -4.59e-4 | -4.31e-5 |
| 186 | 3.00e-3 | 1 | 6.66e-3 | 6.66e-3 | 6.66e-3 | 6.66e-3 | 245 | +1.48e-4 | +1.48e-4 | +1.48e-4 | -2.40e-5 |
| 187 | 3.00e-3 | 1 | 6.98e-3 | 6.98e-3 | 6.98e-3 | 6.98e-3 | 250 | +1.92e-4 | +1.92e-4 | +1.92e-4 | -2.38e-6 |
| 188 | 3.00e-3 | 1 | 6.93e-3 | 6.93e-3 | 6.93e-3 | 6.93e-3 | 270 | -2.78e-5 | -2.78e-5 | -2.78e-5 | -4.92e-6 |
| 189 | 3.00e-3 | 1 | 7.26e-3 | 7.26e-3 | 7.26e-3 | 7.26e-3 | 281 | +1.66e-4 | +1.66e-4 | +1.66e-4 | +1.22e-5 |
| 190 | 3.00e-3 | 1 | 7.20e-3 | 7.20e-3 | 7.20e-3 | 7.20e-3 | 266 | -3.37e-5 | -3.37e-5 | -3.37e-5 | +7.62e-6 |
| 191 | 3.00e-3 | 1 | 7.26e-3 | 7.26e-3 | 7.26e-3 | 7.26e-3 | 262 | +3.39e-5 | +3.39e-5 | +3.39e-5 | +1.03e-5 |
| 192 | 3.00e-3 | 1 | 7.17e-3 | 7.17e-3 | 7.17e-3 | 7.17e-3 | 230 | -5.79e-5 | -5.79e-5 | -5.79e-5 | +3.43e-6 |
| 193 | 3.00e-3 | 2 | 6.77e-3 | 6.85e-3 | 6.81e-3 | 6.85e-3 | 236 | -2.40e-4 | +5.27e-5 | -9.38e-5 | -1.36e-5 |
| 194 | 3.00e-3 | 1 | 6.79e-3 | 6.79e-3 | 6.79e-3 | 6.79e-3 | 236 | -3.55e-5 | -3.55e-5 | -3.55e-5 | -1.58e-5 |
| 195 | 3.00e-3 | 1 | 6.76e-3 | 6.76e-3 | 6.76e-3 | 6.76e-3 | 283 | -1.85e-5 | -1.85e-5 | -1.85e-5 | -1.60e-5 |
| 196 | 3.00e-3 | 1 | 7.60e-3 | 7.60e-3 | 7.60e-3 | 7.60e-3 | 275 | +4.26e-4 | +4.26e-4 | +4.26e-4 | +2.82e-5 |
| 197 | 3.00e-3 | 1 | 7.70e-3 | 7.70e-3 | 7.70e-3 | 7.70e-3 | 282 | +4.57e-5 | +4.57e-5 | +4.57e-5 | +3.00e-5 |
| 198 | 3.00e-3 | 1 | 7.55e-3 | 7.55e-3 | 7.55e-3 | 7.55e-3 | 251 | -7.68e-5 | -7.68e-5 | -7.68e-5 | +1.93e-5 |
| 199 | 3.00e-3 | 1 | 6.93e-3 | 6.93e-3 | 6.93e-3 | 6.93e-3 | 253 | -3.42e-4 | -3.42e-4 | -3.42e-4 | -1.68e-5 |

