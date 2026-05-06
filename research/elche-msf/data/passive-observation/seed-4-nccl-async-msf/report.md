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
| nccl-async | 0.053311 | 0.9184 | +0.0059 | 2020.2 | 768 | 42.0 | 100% | 100% | 6.1 |

## Best Mode per Model

| Model | Best Eval | Mode | Fastest (within 2% of solo-0) | Mode |
|-------|-----------|------|-------------------------------|------|
| resnet-graph | - | - | - | - |

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
| nccl-async | 1.9483 | 0.6995 | 0.6308 | 0.5501 | 0.5192 | 0.5089 | 0.4964 | 0.4811 | 0.4772 | 0.4598 | 0.2099 | 0.1715 | 0.1476 | 0.1174 | 0.1167 | 0.0698 | 0.0622 | 0.0581 | 0.0546 | 0.0533 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | nccl-async | 0 | cuda:0 | 0.4017 | 2.5 |
| resnet-graph | nccl-async | 1 | cuda:1 | 0.3044 | 3.3 |
| resnet-graph | nccl-async | 2 | cuda:2 | 0.2939 | 3.1 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|
| resnet-graph | nccl-async | 356 | 355 | 390 | 384 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| nccl-async | gpu1 | 2018.9 | 1.2 | epoch-boundary(199) |
| nccl-async | gpu2 | 2019.1 | 1.1 | epoch-boundary(199) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | nccl-async | gpu0 | 0.0s | 0.0s | 0.0s | 0.0s | 2.0s |
| resnet-graph | nccl-async | gpu1 | 1.2s | 0.0s | 0.0s | 0.0s | 1.9s |
| resnet-graph | nccl-async | gpu2 | 1.1s | 0.0s | 0.0s | 0.0s | 2.2s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | nccl-async | 401 | 0 | 768 | 42.0 | 762/9607 | 0 | 0.0 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | nccl-async | 184.8 | 9.1% |

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
| resnet-graph | nccl-async | 196 | 768 | 0 | 5.97e-3 | +3.12e-5 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | nccl-async | 0 | 768 | 5.73e-2 | 5.94e-2 | 0.00e0 | 4.26e-1 | 46.6 | -1.85e-4 | 3.70e-3 |
| resnet-graph | nccl-async | 1 | 768 | 5.81e-2 | 6.14e-2 | 0.00e0 | 4.42e-1 | 35.7 | -2.05e-4 | 5.39e-3 |
| resnet-graph | nccl-async | 2 | 768 | 5.76e-2 | 6.31e-2 | 0.00e0 | 6.16e-1 | 17.7 | -2.37e-4 | 5.48e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | nccl-async | rank0 ↔ rank1 | +0.9959 |
| resnet-graph | nccl-async | rank0 ↔ rank2 | +0.9898 |
| resnet-graph | nccl-async | rank1 ↔ rank2 | +0.9934 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | nccl-async | 59 (0,1,2,3,4,5,6,8…146,150) | 0 (—) | — | 0,1,2,3,4,5,6,8…146,150 | — |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | nccl-async | 0e0 | 3 | 35 | 35 |
| resnet-graph | nccl-async | 0e0 | 5 | 15 | 15 |
| resnet-graph | nccl-async | 0e0 | 10 | 2 | 2 |
| resnet-graph | nccl-async | 1e-4 | 3 | 13 | 13 |
| resnet-graph | nccl-async | 1e-4 | 5 | 2 | 2 |
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
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 375 | +0.052 |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | 286 | +0.042 |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | 102 | +0.069 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | nccl-async | 765 | +0.020 | 195 | +0.168 | +0.121 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | nccl-async | 766 | 3.53e1–7.98e1 | 6.29e1 | 1.56e-3 | 4.13e-3 | 8.80e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 377 | 62–77905 | +1.861e-5 | 0.591 | +1.913e-5 | 0.610 | 97 | +1.247e-5 | 0.422 | 31–944 | +1.588e-3 | 0.736 |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 358 | 866–77905 | +1.823e-5 | 0.654 | +1.871e-5 | 0.663 | 96 | +1.198e-5 | 0.404 | 34–944 | +1.558e-3 | 0.818 |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | 287 | 78399–117115 | -1.844e-5 | 0.217 | -1.963e-5 | 0.237 | 50 | -1.624e-5 | 0.222 | 32–531 | +1.688e-3 | 0.343 |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | 103 | 117265–156035 | +3.497e-5 | 0.422 | +3.550e-5 | 0.427 | 49 | +1.398e-5 | 0.180 | 45–814 | +2.240e-3 | 0.653 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | +1.588e-3 | r0: +1.559e-3, r1: +1.599e-3, r2: +1.611e-3 | r0: 0.769, r1: 0.721, r2: 0.712 | 1.03× | ✓ |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +1.558e-3 | r0: +1.525e-3, r1: +1.566e-3, r2: +1.587e-3 | r0: 0.855, r1: 0.794, r2: 0.797 | 1.04× | ✓ |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | +1.688e-3 | r0: +1.625e-3, r1: +1.702e-3, r2: +1.751e-3 | r0: 0.374, r1: 0.320, r2: 0.330 | 1.08× | ✓ |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | +2.240e-3 | r0: +2.234e-3, r1: +2.230e-3, r2: +2.261e-3 | r0: 0.672, r1: 0.645, r2: 0.640 | 1.01× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | nccl-async | `█▇▇▇▇▇██████████████████▆▄▄▅▅▅▅▅▅▅▅▅▅▁▂▂▂▂▂▂▂▂▂▂▂` | `▅▆█▆▄▅▆▇▇▇▇▇▇▇▇▇▇███▇▇▇█▅▅▇▇██▇▇▅▅█▇▁▇█████▇▇▇█▇█` |

### Per-Epoch Detail: resnet-graph / nccl-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 20 | 0.00e0 | 6.16e-1 | 8.43e-2 | 5.29e-2 | 13 | -6.38e-2 | +2.52e-2 | -1.00e-2 | -1.82e-3 |
| 1 | 3.00e-1 | 16 | 3.91e-2 | 9.89e-2 | 5.61e-2 | 5.56e-2 | 17 | -6.73e-2 | +7.74e-2 | +8.41e-4 | -1.45e-4 |
| 2 | 3.00e-1 | 15 | 4.93e-2 | 1.06e-1 | 6.37e-2 | 6.16e-2 | 17 | -2.80e-2 | +3.44e-2 | -1.64e-4 | -3.35e-4 |
| 3 | 3.00e-1 | 15 | 6.04e-2 | 1.20e-1 | 7.25e-2 | 7.18e-2 | 17 | -4.92e-2 | +4.15e-2 | +2.05e-4 | +6.13e-6 |
| 4 | 3.00e-1 | 16 | 6.27e-2 | 1.35e-1 | 7.72e-2 | 6.27e-2 | 16 | -4.21e-2 | +3.98e-2 | -7.70e-4 | -1.50e-3 |
| 5 | 3.00e-1 | 14 | 5.96e-2 | 1.37e-1 | 7.97e-2 | 8.36e-2 | 20 | -6.39e-2 | +5.70e-2 | +1.06e-3 | +2.45e-4 |
| 6 | 3.00e-1 | 20 | 5.99e-2 | 1.42e-1 | 7.73e-2 | 7.79e-2 | 15 | -2.43e-2 | +2.78e-2 | -3.13e-4 | +2.09e-4 |
| 7 | 3.00e-1 | 10 | 6.55e-2 | 1.25e-1 | 7.52e-2 | 7.16e-2 | 18 | -4.32e-2 | +4.23e-2 | +1.26e-4 | +1.68e-5 |
| 8 | 3.00e-1 | 15 | 6.02e-2 | 1.48e-1 | 7.40e-2 | 8.21e-2 | 17 | -4.87e-2 | +4.33e-2 | +1.72e-6 | +5.26e-4 |
| 9 | 3.00e-1 | 17 | 6.26e-2 | 1.50e-1 | 7.76e-2 | 8.78e-2 | 26 | -3.06e-2 | +3.27e-2 | -8.45e-5 | +6.45e-4 |
| 10 | 3.00e-1 | 9 | 6.91e-2 | 1.36e-1 | 8.53e-2 | 7.58e-2 | 21 | -3.77e-2 | +2.05e-2 | -1.22e-3 | -5.02e-4 |
| 11 | 3.00e-1 | 14 | 6.05e-2 | 1.39e-1 | 7.39e-2 | 7.03e-2 | 20 | -4.22e-2 | +2.83e-2 | -1.17e-3 | -5.98e-4 |
| 12 | 3.00e-1 | 15 | 4.86e-2 | 1.37e-1 | 7.37e-2 | 6.37e-2 | 16 | -6.91e-2 | +3.78e-2 | -2.25e-3 | -1.65e-3 |
| 13 | 3.00e-1 | 15 | 5.66e-2 | 1.41e-1 | 6.97e-2 | 7.17e-2 | 21 | -5.02e-2 | +5.42e-2 | +1.06e-3 | +3.71e-4 |
| 14 | 3.00e-1 | 13 | 6.34e-2 | 1.44e-1 | 7.90e-2 | 8.14e-2 | 22 | -4.12e-2 | +3.15e-2 | -1.77e-4 | +2.32e-4 |
| 15 | 3.00e-1 | 13 | 6.23e-2 | 1.28e-1 | 7.52e-2 | 6.45e-2 | 17 | -2.81e-2 | +2.23e-2 | -1.06e-3 | -7.87e-4 |
| 16 | 3.00e-1 | 15 | 6.51e-2 | 1.55e-1 | 8.54e-2 | 6.51e-2 | 20 | -1.65e-2 | +2.52e-2 | -7.06e-4 | -1.28e-3 |
| 17 | 3.00e-1 | 8 | 6.87e-2 | 1.34e-1 | 8.30e-2 | 9.70e-2 | 30 | -2.98e-2 | +2.82e-2 | +1.03e-3 | +1.28e-4 |
| 18 | 3.00e-1 | 10 | 7.88e-2 | 1.40e-1 | 9.06e-2 | 7.94e-2 | 21 | -1.86e-2 | +1.36e-2 | -9.98e-4 | -6.51e-4 |
| 19 | 3.00e-1 | 15 | 5.91e-2 | 1.34e-1 | 7.19e-2 | 5.96e-2 | 16 | -4.68e-2 | +3.26e-2 | -1.18e-3 | -1.32e-3 |
| 20 | 3.00e-1 | 1 | 6.33e-2 | 6.33e-2 | 6.33e-2 | 6.33e-2 | 46 | +1.32e-3 | +1.32e-3 | +1.32e-3 | -1.05e-3 |
| 21 | 3.00e-1 | 1 | 1.23e-1 | 1.23e-1 | 1.23e-1 | 1.23e-1 | 308 | +2.16e-3 | +2.16e-3 | +2.16e-3 | -7.33e-4 |
| 22 | 3.00e-1 | 1 | 2.35e-1 | 2.35e-1 | 2.35e-1 | 2.35e-1 | 311 | +2.08e-3 | +2.08e-3 | +2.08e-3 | -4.51e-4 |
| 23 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 311 | -2.24e-4 | -2.24e-4 | -2.24e-4 | -4.29e-4 |
| 24 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 296 | -7.60e-5 | -7.60e-5 | -7.60e-5 | -3.93e-4 |
| 25 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 290 | -5.50e-5 | -5.50e-5 | -5.50e-5 | -3.60e-4 |
| 26 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 336 | -5.07e-5 | -5.07e-5 | -5.07e-5 | -3.29e-4 |
| 28 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 329 | +9.65e-5 | +9.65e-5 | +9.65e-5 | -2.86e-4 |
| 29 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 317 | -3.12e-5 | -3.12e-5 | -3.12e-5 | -2.61e-4 |
| 30 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 315 | -3.65e-5 | -3.65e-5 | -3.65e-5 | -2.38e-4 |
| 31 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 322 | -2.21e-5 | -2.21e-5 | -2.21e-5 | -2.17e-4 |
| 32 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 302 | +1.61e-5 | +1.61e-5 | +1.61e-5 | -1.93e-4 |
| 33 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 297 | -6.31e-5 | -6.31e-5 | -6.31e-5 | -1.80e-4 |
| 34 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 260 | +2.37e-5 | +2.37e-5 | +2.37e-5 | -1.60e-4 |
| 35 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 275 | -1.34e-4 | -1.34e-4 | -1.34e-4 | -1.57e-4 |
| 36 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 275 | +6.95e-5 | +6.95e-5 | +6.95e-5 | -1.35e-4 |
| 37 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 309 | +1.27e-5 | +1.27e-5 | +1.27e-5 | -1.20e-4 |
| 38 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 289 | +7.73e-5 | +7.73e-5 | +7.73e-5 | -1.00e-4 |
| 39 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 276 | -3.74e-5 | -3.74e-5 | -3.74e-5 | -9.39e-5 |
| 40 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 284 | -7.67e-5 | -7.67e-5 | -7.67e-5 | -9.22e-5 |
| 41 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 272 | +6.87e-5 | +6.87e-5 | +6.87e-5 | -7.61e-5 |
| 42 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 265 | -9.92e-5 | -9.92e-5 | -9.92e-5 | -7.84e-5 |
| 43 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 263 | -4.33e-6 | -4.33e-6 | -4.33e-6 | -7.10e-5 |
| 44 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 276 | +2.11e-5 | +2.11e-5 | +2.11e-5 | -6.18e-5 |
| 45 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 270 | +5.25e-5 | +5.25e-5 | +5.25e-5 | -5.04e-5 |
| 46 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 259 | -2.35e-5 | -2.35e-5 | -2.35e-5 | -4.77e-5 |
| 47 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 270 | -1.42e-5 | -1.42e-5 | -1.42e-5 | -4.43e-5 |
| 48 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 276 | +8.31e-5 | +8.31e-5 | +8.31e-5 | -3.16e-5 |
| 49 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 285 | -6.14e-5 | -6.14e-5 | -6.14e-5 | -3.46e-5 |
| 50 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 261 | +1.39e-4 | +1.39e-4 | +1.39e-4 | -1.72e-5 |
| 51 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 304 | -1.39e-4 | -1.39e-4 | -1.39e-4 | -2.93e-5 |
| 52 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 243 | +1.09e-4 | +1.09e-4 | +1.09e-4 | -1.55e-5 |
| 53 | 3.00e-1 | 1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 250 | -2.65e-4 | -2.65e-4 | -2.65e-4 | -4.05e-5 |
| 54 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 256 | +1.07e-4 | +1.07e-4 | +1.07e-4 | -2.57e-5 |
| 55 | 3.00e-1 | 2 | 1.94e-1 | 2.00e-1 | 1.97e-1 | 1.94e-1 | 214 | -1.58e-4 | +1.38e-5 | -7.20e-5 | -3.54e-5 |
| 56 | 3.00e-1 | 1 | 1.89e-1 | 1.89e-1 | 1.89e-1 | 1.89e-1 | 265 | -9.75e-5 | -9.75e-5 | -9.75e-5 | -4.16e-5 |
| 57 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 276 | +2.94e-4 | +2.94e-4 | +2.94e-4 | -8.06e-6 |
| 58 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 286 | +4.48e-5 | +4.48e-5 | +4.48e-5 | -2.78e-6 |
| 59 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 252 | +3.60e-5 | +3.60e-5 | +3.60e-5 | +1.09e-6 |
| 60 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 259 | -1.82e-4 | -1.82e-4 | -1.82e-4 | -1.72e-5 |
| 61 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 237 | +3.85e-6 | +3.85e-6 | +3.85e-6 | -1.51e-5 |
| 62 | 3.00e-1 | 1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 250 | -6.04e-5 | -6.04e-5 | -6.04e-5 | -1.96e-5 |
| 63 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 250 | +8.90e-5 | +8.90e-5 | +8.90e-5 | -8.74e-6 |
| 64 | 3.00e-1 | 2 | 1.98e-1 | 2.03e-1 | 2.00e-1 | 1.98e-1 | 220 | -9.91e-5 | +2.88e-5 | -3.52e-5 | -1.44e-5 |
| 65 | 3.00e-1 | 1 | 1.91e-1 | 1.91e-1 | 1.91e-1 | 1.91e-1 | 238 | -1.61e-4 | -1.61e-4 | -1.61e-4 | -2.91e-5 |
| 66 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 222 | +1.29e-4 | +1.29e-4 | +1.29e-4 | -1.33e-5 |
| 67 | 3.00e-1 | 1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 231 | +6.36e-6 | +6.36e-6 | +6.36e-6 | -1.13e-5 |
| 68 | 3.00e-1 | 1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 230 | +2.00e-5 | +2.00e-5 | +2.00e-5 | -8.20e-6 |
| 69 | 3.00e-1 | 1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 237 | -2.01e-5 | -2.01e-5 | -2.01e-5 | -9.39e-6 |
| 70 | 3.00e-1 | 2 | 1.99e-1 | 2.07e-1 | 2.03e-1 | 2.07e-1 | 224 | +5.68e-5 | +1.55e-4 | +1.06e-4 | +1.30e-5 |
| 72 | 3.00e-1 | 2 | 1.94e-1 | 2.10e-1 | 2.02e-1 | 2.10e-1 | 192 | -2.13e-4 | +4.10e-4 | +9.85e-5 | +3.24e-5 |
| 73 | 3.00e-1 | 1 | 1.85e-1 | 1.85e-1 | 1.85e-1 | 1.85e-1 | 242 | -5.28e-4 | -5.28e-4 | -5.28e-4 | -2.36e-5 |
| 74 | 3.00e-1 | 2 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 199 | -3.48e-6 | +3.02e-4 | +1.49e-4 | +7.64e-6 |
| 76 | 3.00e-1 | 2 | 1.88e-1 | 2.03e-1 | 1.96e-1 | 2.03e-1 | 181 | -1.92e-4 | +4.30e-4 | +1.19e-4 | +3.19e-5 |
| 77 | 3.00e-1 | 1 | 1.84e-1 | 1.84e-1 | 1.84e-1 | 1.84e-1 | 215 | -4.74e-4 | -4.74e-4 | -4.74e-4 | -1.87e-5 |
| 78 | 3.00e-1 | 2 | 1.87e-1 | 1.92e-1 | 1.90e-1 | 1.87e-1 | 173 | -1.37e-4 | +2.24e-4 | +4.36e-5 | -8.65e-6 |
| 79 | 3.00e-1 | 1 | 1.78e-1 | 1.78e-1 | 1.78e-1 | 1.78e-1 | 245 | -2.20e-4 | -2.20e-4 | -2.20e-4 | -2.98e-5 |
| 80 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 218 | +6.55e-4 | +6.55e-4 | +6.55e-4 | +3.87e-5 |
| 81 | 3.00e-1 | 2 | 1.94e-1 | 2.02e-1 | 1.98e-1 | 2.02e-1 | 196 | -2.35e-4 | +2.14e-4 | -1.07e-5 | +3.16e-5 |
| 82 | 3.00e-1 | 1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 211 | -2.97e-4 | -2.97e-4 | -2.97e-4 | -1.28e-6 |
| 83 | 3.00e-1 | 1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 227 | +1.73e-5 | +1.73e-5 | +1.73e-5 | +5.87e-7 |
| 84 | 3.00e-1 | 1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 223 | +1.62e-4 | +1.62e-4 | +1.62e-4 | +1.67e-5 |
| 85 | 3.00e-1 | 2 | 1.88e-1 | 1.94e-1 | 1.91e-1 | 1.88e-1 | 161 | -1.99e-4 | -9.55e-5 | -1.47e-4 | -1.50e-5 |
| 86 | 3.00e-1 | 1 | 1.70e-1 | 1.70e-1 | 1.70e-1 | 1.70e-1 | 202 | -4.98e-4 | -4.98e-4 | -4.98e-4 | -6.33e-5 |
| 87 | 3.00e-1 | 2 | 1.91e-1 | 1.99e-1 | 1.95e-1 | 1.99e-1 | 172 | +2.31e-4 | +4.99e-4 | +3.65e-4 | +1.67e-5 |
| 88 | 3.00e-1 | 1 | 1.79e-1 | 1.79e-1 | 1.79e-1 | 1.79e-1 | 206 | -5.16e-4 | -5.16e-4 | -5.16e-4 | -3.65e-5 |
| 89 | 3.00e-1 | 2 | 1.91e-1 | 1.94e-1 | 1.93e-1 | 1.91e-1 | 177 | -8.32e-5 | +3.93e-4 | +1.55e-4 | -2.54e-6 |
| 90 | 3.00e-1 | 1 | 1.78e-1 | 1.78e-1 | 1.78e-1 | 1.78e-1 | 202 | -3.41e-4 | -3.41e-4 | -3.41e-4 | -3.64e-5 |
| 91 | 3.00e-1 | 1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 222 | +2.74e-4 | +2.74e-4 | +2.74e-4 | -5.33e-6 |
| 92 | 3.00e-1 | 2 | 1.85e-1 | 1.99e-1 | 1.92e-1 | 1.85e-1 | 146 | -4.95e-4 | +2.53e-4 | -1.21e-4 | -3.10e-5 |
| 93 | 3.00e-1 | 2 | 1.69e-1 | 1.90e-1 | 1.79e-1 | 1.90e-1 | 150 | -4.69e-4 | +7.74e-4 | +1.53e-4 | +1.01e-5 |
| 94 | 3.00e-1 | 1 | 1.70e-1 | 1.70e-1 | 1.70e-1 | 1.70e-1 | 184 | -5.86e-4 | -5.86e-4 | -5.86e-4 | -4.96e-5 |
| 95 | 3.00e-1 | 2 | 1.84e-1 | 1.86e-1 | 1.85e-1 | 1.84e-1 | 146 | -8.07e-5 | +4.76e-4 | +1.98e-4 | -5.39e-6 |
| 96 | 3.00e-1 | 2 | 1.71e-1 | 1.92e-1 | 1.82e-1 | 1.92e-1 | 166 | -3.43e-4 | +7.14e-4 | +1.86e-4 | +3.62e-5 |
| 97 | 3.00e-1 | 1 | 1.79e-1 | 1.79e-1 | 1.79e-1 | 1.79e-1 | 206 | -3.45e-4 | -3.45e-4 | -3.45e-4 | -1.88e-6 |
| 98 | 3.00e-1 | 1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 210 | +3.80e-4 | +3.80e-4 | +3.80e-4 | +3.63e-5 |
| 99 | 3.00e-1 | 2 | 1.88e-1 | 1.94e-1 | 1.91e-1 | 1.88e-1 | 166 | -2.03e-4 | +7.95e-6 | -9.76e-5 | +9.78e-6 |
| 100 | 3.00e-2 | 2 | 1.83e-1 | 1.83e-1 | 1.83e-1 | 1.83e-1 | 145 | -1.43e-4 | +5.62e-6 | -6.87e-5 | -4.39e-6 |
| 101 | 3.00e-2 | 2 | 1.71e-2 | 1.83e-2 | 1.77e-2 | 1.83e-2 | 145 | -1.35e-2 | +4.48e-4 | -6.51e-3 | -1.17e-3 |
| 102 | 3.00e-2 | 1 | 1.80e-2 | 1.80e-2 | 1.80e-2 | 1.80e-2 | 174 | -8.48e-5 | -8.48e-5 | -8.48e-5 | -1.06e-3 |
| 103 | 3.00e-2 | 2 | 1.97e-2 | 2.01e-2 | 1.99e-2 | 2.01e-2 | 145 | +1.44e-4 | +5.28e-4 | +3.36e-4 | -7.99e-4 |
| 104 | 3.00e-2 | 2 | 1.88e-2 | 2.06e-2 | 1.97e-2 | 2.06e-2 | 145 | -4.19e-4 | +6.34e-4 | +1.07e-4 | -6.21e-4 |
| 105 | 3.00e-2 | 1 | 1.99e-2 | 1.99e-2 | 1.99e-2 | 1.99e-2 | 173 | -2.09e-4 | -2.09e-4 | -2.09e-4 | -5.80e-4 |
| 106 | 3.00e-2 | 2 | 2.15e-2 | 2.33e-2 | 2.24e-2 | 2.33e-2 | 138 | +4.23e-4 | +5.75e-4 | +4.99e-4 | -3.74e-4 |
| 107 | 3.00e-2 | 2 | 2.11e-2 | 2.30e-2 | 2.20e-2 | 2.30e-2 | 137 | -5.52e-4 | +6.13e-4 | +3.04e-5 | -2.92e-4 |
| 108 | 3.00e-2 | 2 | 2.09e-2 | 2.38e-2 | 2.24e-2 | 2.38e-2 | 138 | -5.33e-4 | +9.69e-4 | +2.18e-4 | -1.87e-4 |
| 109 | 3.00e-2 | 1 | 2.26e-2 | 2.26e-2 | 2.26e-2 | 2.26e-2 | 189 | -2.76e-4 | -2.76e-4 | -2.76e-4 | -1.96e-4 |
| 110 | 3.00e-2 | 2 | 2.41e-2 | 2.48e-2 | 2.44e-2 | 2.41e-2 | 130 | -2.06e-4 | +5.40e-4 | +1.67e-4 | -1.31e-4 |
| 111 | 3.00e-2 | 2 | 2.21e-2 | 2.57e-2 | 2.39e-2 | 2.57e-2 | 130 | -4.89e-4 | +1.16e-3 | +3.34e-4 | -3.43e-5 |
| 112 | 3.00e-2 | 3 | 2.29e-2 | 2.60e-2 | 2.41e-2 | 2.34e-2 | 123 | -8.48e-4 | +9.66e-4 | -1.84e-4 | -7.71e-5 |
| 113 | 3.00e-2 | 1 | 2.33e-2 | 2.33e-2 | 2.33e-2 | 2.33e-2 | 156 | -3.35e-5 | -3.35e-5 | -3.35e-5 | -7.27e-5 |
| 114 | 3.00e-2 | 2 | 2.56e-2 | 2.57e-2 | 2.57e-2 | 2.56e-2 | 123 | -1.74e-5 | +6.30e-4 | +3.06e-4 | -3.98e-6 |
| 115 | 3.00e-2 | 2 | 2.42e-2 | 2.64e-2 | 2.53e-2 | 2.64e-2 | 123 | -3.74e-4 | +7.09e-4 | +1.68e-4 | +3.40e-5 |
| 116 | 3.00e-2 | 2 | 2.43e-2 | 2.81e-2 | 2.62e-2 | 2.81e-2 | 125 | -5.12e-4 | +1.19e-3 | +3.37e-4 | +1.00e-4 |
| 117 | 3.00e-2 | 3 | 2.42e-2 | 2.80e-2 | 2.59e-2 | 2.42e-2 | 113 | -1.29e-3 | +7.65e-4 | -3.78e-4 | -3.66e-5 |
| 118 | 3.00e-2 | 2 | 2.45e-2 | 2.77e-2 | 2.61e-2 | 2.77e-2 | 106 | +9.81e-5 | +1.14e-3 | +6.19e-4 | +9.31e-5 |
| 119 | 3.00e-2 | 2 | 2.42e-2 | 2.84e-2 | 2.63e-2 | 2.84e-2 | 99 | -8.72e-4 | +1.61e-3 | +3.70e-4 | +1.58e-4 |
| 120 | 3.00e-2 | 3 | 2.43e-2 | 2.89e-2 | 2.64e-2 | 2.59e-2 | 97 | -1.13e-3 | +1.69e-3 | -1.64e-4 | +6.93e-5 |
| 121 | 3.00e-2 | 3 | 2.32e-2 | 2.56e-2 | 2.48e-2 | 2.32e-2 | 86 | -1.13e-3 | +1.06e-5 | -4.14e-4 | -7.14e-5 |
| 122 | 3.00e-2 | 2 | 2.36e-2 | 2.87e-2 | 2.61e-2 | 2.87e-2 | 86 | +1.30e-4 | +2.25e-3 | +1.19e-3 | +1.79e-4 |
| 123 | 3.00e-2 | 3 | 2.30e-2 | 3.03e-2 | 2.60e-2 | 2.30e-2 | 79 | -3.45e-3 | +2.61e-3 | -6.32e-4 | -6.55e-5 |
| 124 | 3.00e-2 | 4 | 2.39e-2 | 2.65e-2 | 2.49e-2 | 2.47e-2 | 77 | -1.26e-3 | +1.15e-3 | +2.01e-4 | +1.41e-5 |
| 125 | 3.00e-2 | 3 | 2.31e-2 | 2.74e-2 | 2.46e-2 | 2.31e-2 | 70 | -2.49e-3 | +2.26e-3 | -2.46e-4 | -7.63e-5 |
| 126 | 3.00e-2 | 4 | 2.38e-2 | 2.88e-2 | 2.56e-2 | 2.58e-2 | 72 | -2.52e-3 | +2.66e-3 | +3.86e-4 | +7.00e-5 |
| 127 | 3.00e-2 | 3 | 2.44e-2 | 3.01e-2 | 2.64e-2 | 2.44e-2 | 70 | -3.01e-3 | +2.75e-3 | -1.88e-4 | -2.72e-5 |
| 128 | 3.00e-2 | 4 | 2.29e-2 | 2.98e-2 | 2.52e-2 | 2.34e-2 | 61 | -4.28e-3 | +2.83e-3 | -2.44e-4 | -1.30e-4 |
| 129 | 3.00e-2 | 4 | 2.20e-2 | 3.01e-2 | 2.46e-2 | 2.20e-2 | 57 | -4.28e-3 | +4.49e-3 | -3.11e-4 | -2.46e-4 |
| 130 | 3.00e-2 | 5 | 1.99e-2 | 2.88e-2 | 2.29e-2 | 1.99e-2 | 51 | -5.46e-3 | +3.53e-3 | -5.89e-4 | -4.45e-4 |
| 131 | 3.00e-2 | 5 | 2.08e-2 | 2.76e-2 | 2.29e-2 | 2.08e-2 | 44 | -4.49e-3 | +4.26e-3 | -1.23e-4 | -3.86e-4 |
| 132 | 3.00e-2 | 10 | 1.66e-2 | 2.80e-2 | 1.95e-2 | 1.73e-2 | 26 | -9.57e-3 | +6.66e-3 | -5.34e-4 | -5.01e-4 |
| 133 | 3.00e-2 | 7 | 1.43e-2 | 2.60e-2 | 1.64e-2 | 1.50e-2 | 28 | -2.48e-2 | +2.07e-2 | -5.42e-4 | -5.68e-4 |
| 134 | 3.00e-2 | 13 | 1.17e-2 | 2.60e-2 | 1.73e-2 | 1.71e-2 | 28 | -2.99e-2 | +2.15e-2 | -6.66e-4 | -6.19e-4 |
| 135 | 3.00e-2 | 7 | 1.38e-2 | 2.75e-2 | 1.67e-2 | 1.38e-2 | 22 | -2.43e-2 | +2.04e-2 | -1.72e-3 | -1.30e-3 |
| 136 | 3.00e-2 | 10 | 1.25e-2 | 2.86e-2 | 1.57e-2 | 1.72e-2 | 20 | -3.18e-2 | +3.21e-2 | +4.32e-4 | +5.25e-5 |
| 137 | 3.00e-2 | 13 | 1.16e-2 | 2.54e-2 | 1.41e-2 | 1.45e-2 | 23 | -4.47e-2 | +3.33e-2 | -4.27e-4 | +2.72e-5 |
| 138 | 3.00e-2 | 16 | 1.19e-2 | 2.61e-2 | 1.45e-2 | 1.19e-2 | 15 | -2.54e-2 | +2.50e-2 | -8.71e-4 | -1.11e-3 |
| 139 | 3.00e-2 | 13 | 1.04e-2 | 2.47e-2 | 1.28e-2 | 1.33e-2 | 18 | -5.47e-2 | +4.99e-2 | +2.64e-4 | +9.49e-5 |
| 140 | 3.00e-2 | 17 | 1.16e-2 | 2.58e-2 | 1.44e-2 | 1.46e-2 | 19 | -3.55e-2 | +3.51e-2 | -1.73e-5 | +1.57e-4 |
| 141 | 3.00e-2 | 11 | 1.27e-2 | 2.64e-2 | 1.55e-2 | 1.42e-2 | 17 | -3.69e-2 | +2.99e-2 | -2.37e-4 | -2.82e-4 |
| 142 | 3.00e-2 | 15 | 1.16e-2 | 2.64e-2 | 1.44e-2 | 1.44e-2 | 17 | -4.82e-2 | +4.04e-2 | +3.14e-4 | +2.56e-4 |
| 143 | 3.00e-2 | 15 | 1.23e-2 | 2.83e-2 | 1.46e-2 | 1.46e-2 | 18 | -4.36e-2 | +4.29e-2 | +1.13e-4 | +2.47e-4 |
| 144 | 3.00e-2 | 16 | 1.00e-2 | 2.98e-2 | 1.42e-2 | 1.24e-2 | 17 | -6.36e-2 | +4.64e-2 | -1.26e-3 | -8.95e-4 |
| 145 | 3.00e-2 | 11 | 1.10e-2 | 3.04e-2 | 1.57e-2 | 1.52e-2 | 45 | -5.13e-2 | +5.32e-2 | +4.46e-4 | -2.55e-4 |
| 146 | 3.00e-2 | 8 | 2.42e-2 | 3.56e-2 | 2.70e-2 | 2.54e-2 | 35 | -9.54e-3 | +9.03e-3 | +6.61e-4 | +6.94e-5 |
| 147 | 3.00e-2 | 6 | 1.95e-2 | 3.51e-2 | 2.34e-2 | 1.95e-2 | 28 | -1.78e-2 | +1.19e-2 | -1.58e-3 | -7.74e-4 |
| 148 | 3.00e-2 | 12 | 1.95e-2 | 3.79e-2 | 2.33e-2 | 2.13e-2 | 24 | -1.21e-2 | +1.75e-2 | +1.31e-4 | -2.52e-4 |
| 149 | 3.00e-2 | 6 | 1.63e-2 | 3.46e-2 | 2.13e-2 | 1.63e-2 | 22 | -2.23e-2 | +2.42e-2 | -9.14e-4 | -8.73e-4 |
| 150 | 3.00e-3 | 13 | 1.35e-3 | 2.41e-2 | 4.50e-3 | 1.43e-3 | 18 | -1.19e-1 | +1.35e-2 | -9.28e-3 | -4.64e-3 |
| 151 | 3.00e-3 | 16 | 1.25e-3 | 3.22e-3 | 1.55e-3 | 1.56e-3 | 20 | -4.56e-2 | +5.42e-2 | +1.11e-3 | -3.00e-4 |
| 152 | 3.00e-3 | 9 | 1.33e-3 | 3.06e-3 | 1.70e-3 | 1.54e-3 | 18 | -3.12e-2 | +3.77e-2 | +4.92e-4 | -7.48e-5 |
| 153 | 3.00e-3 | 2 | 1.26e-3 | 1.39e-3 | 1.32e-3 | 1.26e-3 | 198 | -5.40e-3 | -5.14e-4 | -2.96e-3 | -5.98e-4 |
| 154 | 3.00e-3 | 1 | 5.64e-3 | 5.64e-3 | 5.64e-3 | 5.64e-3 | 220 | +6.82e-3 | +6.82e-3 | +6.82e-3 | +1.44e-4 |
| 155 | 3.00e-3 | 2 | 5.83e-3 | 5.83e-3 | 5.83e-3 | 5.83e-3 | 211 | -3.03e-6 | +1.54e-4 | +7.55e-5 | +1.30e-4 |
| 157 | 3.00e-3 | 2 | 5.76e-3 | 6.27e-3 | 6.02e-3 | 6.27e-3 | 228 | -4.06e-5 | +3.70e-4 | +1.65e-4 | +1.39e-4 |
| 158 | 3.00e-3 | 1 | 5.99e-3 | 5.99e-3 | 5.99e-3 | 5.99e-3 | 259 | -1.79e-4 | -1.79e-4 | -1.79e-4 | +1.07e-4 |
| 159 | 3.00e-3 | 1 | 6.41e-3 | 6.41e-3 | 6.41e-3 | 6.41e-3 | 237 | +2.90e-4 | +2.90e-4 | +2.90e-4 | +1.25e-4 |
| 160 | 3.00e-3 | 1 | 6.01e-3 | 6.01e-3 | 6.01e-3 | 6.01e-3 | 304 | -2.14e-4 | -2.14e-4 | -2.14e-4 | +9.13e-5 |
| 161 | 3.00e-3 | 1 | 6.79e-3 | 6.79e-3 | 6.79e-3 | 6.79e-3 | 290 | +4.21e-4 | +4.21e-4 | +4.21e-4 | +1.24e-4 |
| 162 | 3.00e-3 | 1 | 6.74e-3 | 6.74e-3 | 6.74e-3 | 6.74e-3 | 263 | -2.90e-5 | -2.90e-5 | -2.90e-5 | +1.09e-4 |
| 163 | 3.00e-3 | 1 | 6.47e-3 | 6.47e-3 | 6.47e-3 | 6.47e-3 | 260 | -1.55e-4 | -1.55e-4 | -1.55e-4 | +8.25e-5 |
| 164 | 3.00e-3 | 2 | 5.77e-3 | 6.53e-3 | 6.15e-3 | 5.77e-3 | 191 | -6.48e-4 | +3.94e-5 | -3.04e-4 | +5.55e-6 |
| 165 | 3.00e-3 | 1 | 5.72e-3 | 5.72e-3 | 5.72e-3 | 5.72e-3 | 218 | -4.20e-5 | -4.20e-5 | -4.20e-5 | +8.00e-7 |
| 166 | 3.00e-3 | 1 | 6.19e-3 | 6.19e-3 | 6.19e-3 | 6.19e-3 | 232 | +3.43e-4 | +3.43e-4 | +3.43e-4 | +3.51e-5 |
| 167 | 3.00e-3 | 1 | 6.27e-3 | 6.27e-3 | 6.27e-3 | 6.27e-3 | 217 | +5.77e-5 | +5.77e-5 | +5.77e-5 | +3.73e-5 |
| 168 | 3.00e-3 | 2 | 5.95e-3 | 6.06e-3 | 6.01e-3 | 6.06e-3 | 183 | -2.36e-4 | +9.94e-5 | -6.84e-5 | +1.89e-5 |
| 169 | 3.00e-3 | 1 | 5.46e-3 | 5.46e-3 | 5.46e-3 | 5.46e-3 | 235 | -4.44e-4 | -4.44e-4 | -4.44e-4 | -2.73e-5 |
| 170 | 3.00e-3 | 1 | 6.27e-3 | 6.27e-3 | 6.27e-3 | 6.27e-3 | 245 | +5.63e-4 | +5.63e-4 | +5.63e-4 | +3.17e-5 |
| 171 | 3.00e-3 | 2 | 5.78e-3 | 6.43e-3 | 6.10e-3 | 5.78e-3 | 194 | -5.48e-4 | +1.26e-4 | -2.11e-4 | -1.78e-5 |
| 172 | 3.00e-3 | 1 | 5.61e-3 | 5.61e-3 | 5.61e-3 | 5.61e-3 | 221 | -1.35e-4 | -1.35e-4 | -1.35e-4 | -2.95e-5 |
| 173 | 3.00e-3 | 1 | 6.36e-3 | 6.36e-3 | 6.36e-3 | 6.36e-3 | 224 | +5.64e-4 | +5.64e-4 | +5.64e-4 | +2.98e-5 |
| 174 | 3.00e-3 | 1 | 6.30e-3 | 6.30e-3 | 6.30e-3 | 6.30e-3 | 222 | -4.32e-5 | -4.32e-5 | -4.32e-5 | +2.25e-5 |
| 175 | 3.00e-3 | 2 | 6.29e-3 | 6.39e-3 | 6.34e-3 | 6.39e-3 | 194 | -1.16e-5 | +8.24e-5 | +3.54e-5 | +2.54e-5 |
| 176 | 3.00e-3 | 1 | 6.10e-3 | 6.10e-3 | 6.10e-3 | 6.10e-3 | 235 | -1.94e-4 | -1.94e-4 | -1.94e-4 | +3.49e-6 |
| 177 | 3.00e-3 | 1 | 6.36e-3 | 6.36e-3 | 6.36e-3 | 6.36e-3 | 216 | +1.90e-4 | +1.90e-4 | +1.90e-4 | +2.21e-5 |
| 178 | 3.00e-3 | 2 | 5.90e-3 | 6.10e-3 | 6.00e-3 | 5.90e-3 | 184 | -2.00e-4 | -1.83e-4 | -1.91e-4 | -1.84e-5 |
| 179 | 3.00e-3 | 1 | 5.78e-3 | 5.78e-3 | 5.78e-3 | 5.78e-3 | 247 | -7.98e-5 | -7.98e-5 | -7.98e-5 | -2.45e-5 |
| 180 | 3.00e-3 | 1 | 6.38e-3 | 6.38e-3 | 6.38e-3 | 6.38e-3 | 231 | +4.29e-4 | +4.29e-4 | +4.29e-4 | +2.08e-5 |
| 181 | 3.00e-3 | 2 | 6.11e-3 | 6.33e-3 | 6.22e-3 | 6.11e-3 | 174 | -2.01e-4 | -4.05e-5 | -1.21e-4 | -6.90e-6 |
| 182 | 3.00e-3 | 1 | 5.61e-3 | 5.61e-3 | 5.61e-3 | 5.61e-3 | 199 | -4.29e-4 | -4.29e-4 | -4.29e-4 | -4.91e-5 |
| 183 | 3.00e-3 | 1 | 6.12e-3 | 6.12e-3 | 6.12e-3 | 6.12e-3 | 199 | +4.36e-4 | +4.36e-4 | +4.36e-4 | -5.60e-7 |
| 184 | 3.00e-3 | 2 | 6.10e-3 | 6.50e-3 | 6.30e-3 | 6.50e-3 | 188 | -1.78e-5 | +3.39e-4 | +1.60e-4 | +3.18e-5 |
| 185 | 3.00e-3 | 1 | 5.89e-3 | 5.89e-3 | 5.89e-3 | 5.89e-3 | 228 | -4.29e-4 | -4.29e-4 | -4.29e-4 | -1.42e-5 |
| 186 | 3.00e-3 | 2 | 5.85e-3 | 6.52e-3 | 6.18e-3 | 5.85e-3 | 152 | -7.14e-4 | +5.26e-4 | -9.39e-5 | -3.56e-5 |
| 187 | 3.00e-3 | 1 | 5.31e-3 | 5.31e-3 | 5.31e-3 | 5.31e-3 | 180 | -5.39e-4 | -5.39e-4 | -5.39e-4 | -8.59e-5 |
| 188 | 3.00e-3 | 2 | 6.04e-3 | 6.30e-3 | 6.17e-3 | 6.30e-3 | 149 | +2.82e-4 | +6.14e-4 | +4.48e-4 | +1.38e-5 |
| 189 | 3.00e-3 | 2 | 5.06e-3 | 6.11e-3 | 5.59e-3 | 6.11e-3 | 156 | -1.11e-3 | +1.21e-3 | +4.94e-5 | +3.22e-5 |
| 190 | 3.00e-3 | 1 | 5.39e-3 | 5.39e-3 | 5.39e-3 | 5.39e-3 | 183 | -6.92e-4 | -6.92e-4 | -6.92e-4 | -4.02e-5 |
| 191 | 3.00e-3 | 2 | 5.80e-3 | 5.83e-3 | 5.82e-3 | 5.80e-3 | 147 | -3.65e-5 | +4.54e-4 | +2.09e-4 | +4.62e-6 |
| 192 | 3.00e-3 | 1 | 5.29e-3 | 5.29e-3 | 5.29e-3 | 5.29e-3 | 183 | -5.05e-4 | -5.05e-4 | -5.05e-4 | -4.64e-5 |
| 193 | 3.00e-3 | 2 | 5.72e-3 | 5.72e-3 | 5.72e-3 | 5.72e-3 | 147 | +3.60e-6 | +4.31e-4 | +2.18e-4 | +1.62e-6 |
| 194 | 3.00e-3 | 2 | 5.35e-3 | 5.90e-3 | 5.62e-3 | 5.90e-3 | 156 | -3.72e-4 | +6.25e-4 | +1.26e-4 | +3.03e-5 |
| 195 | 3.00e-3 | 1 | 5.48e-3 | 5.48e-3 | 5.48e-3 | 5.48e-3 | 177 | -4.19e-4 | -4.19e-4 | -4.19e-4 | -1.46e-5 |
| 196 | 3.00e-3 | 2 | 5.93e-3 | 5.96e-3 | 5.95e-3 | 5.93e-3 | 159 | -3.48e-5 | +4.57e-4 | +2.11e-4 | +2.58e-5 |
| 197 | 3.00e-3 | 1 | 5.94e-3 | 5.94e-3 | 5.94e-3 | 5.94e-3 | 207 | +9.24e-6 | +9.24e-6 | +9.24e-6 | +2.42e-5 |
| 198 | 3.00e-3 | 2 | 6.31e-3 | 6.37e-3 | 6.34e-3 | 6.37e-3 | 144 | +6.35e-5 | +2.92e-4 | +1.78e-4 | +5.22e-5 |
| 199 | 3.00e-3 | 2 | 5.49e-3 | 5.97e-3 | 5.73e-3 | 5.97e-3 | 131 | -8.35e-4 | +6.40e-4 | -9.75e-5 | +3.12e-5 |

