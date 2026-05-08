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
| nccl-async | 0.066496 | 0.9172 | +0.0047 | 1850.8 | 275 | 43.5 | 100% | 100% | 6.1 |

## Best Mode per Model

| Model | Best Eval | Mode | Fastest (within 2% of solo-0) | Mode |
|-------|-----------|------|-------------------------------|------|
| resnet-graph | 0.9172 | nccl-async | - | - |

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
| nccl-async | 1.9663 | 0.7783 | 0.6020 | 0.5466 | 0.5283 | 0.4995 | 0.4945 | 0.4862 | 0.4749 | 0.4687 | 0.2179 | 0.1838 | 0.1658 | 0.1550 | 0.1459 | 0.0863 | 0.0791 | 0.0749 | 0.0698 | 0.0665 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | nccl-async | 0 | cuda:0 | 0.4005 | 2.6 |
| resnet-graph | nccl-async | 1 | cuda:1 | 0.3065 | 3.4 |
| resnet-graph | nccl-async | 2 | cuda:2 | 0.2930 | 3.5 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|
| resnet-graph | nccl-async | 356 | 355 | 400 | 398 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| nccl-async | gpu2 | 1849.7 | 1.2 | epoch-boundary(199) |
| nccl-async | gpu1 | 1849.8 | 1.0 | epoch-boundary(199) |
| nccl-async | gpu0 | 1849.8 | 0.5 | epoch-boundary(199) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | nccl-async | gpu0 | 0.5s | 0.0s | 0.0s | 0.0s | 1.8s |
| resnet-graph | nccl-async | gpu1 | 1.0s | 0.0s | 0.0s | 0.0s | 1.8s |
| resnet-graph | nccl-async | gpu2 | 1.2s | 0.0s | 0.0s | 0.0s | 2.5s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | nccl-async | 81 | 0 | 275 | 43.5 | 8722/10595 | 0 | 0.0 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | nccl-async | 185.7 | 10.0% |

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
| resnet-graph | nccl-async | 186 | 275 | 0 | 6.52e-3 | +3.04e-7 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | nccl-async | 0 | 275 | 9.99e-2 | 7.81e-2 | 0.00e0 | 4.12e-1 | 22.9 | -1.64e-4 | 1.82e-3 |
| resnet-graph | nccl-async | 1 | 275 | 1.01e-1 | 8.02e-2 | 0.00e0 | 4.62e-1 | 57.1 | -1.82e-4 | 2.33e-3 |
| resnet-graph | nccl-async | 2 | 275 | 1.00e-1 | 7.93e-2 | 0.00e0 | 4.27e-1 | 20.0 | -1.76e-4 | 2.33e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | nccl-async | rank0 ↔ rank1 | +0.9967 |
| resnet-graph | nccl-async | rank0 ↔ rank2 | +0.9968 |
| resnet-graph | nccl-async | rank1 ↔ rank2 | +0.9993 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | nccl-async | 45 (0,1,2,3,4,5,6,7…148,149) | 0 (—) | — | 0,1,2,3,4,5,6,7…148,149 | — |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | nccl-async | 0e0 | 3 | 28 | 28 |
| resnet-graph | nccl-async | 0e0 | 5 | 14 | 14 |
| resnet-graph | nccl-async | 0e0 | 10 | 5 | 5 |
| resnet-graph | nccl-async | 1e-4 | 3 | 1 | 1 |
| resnet-graph | nccl-async | 1e-4 | 5 | 1 | 1 |
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
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 167 | +0.111 |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | 50 | +0.191 |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | 53 | +0.072 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | nccl-async | 272 | -0.003 | 185 | +0.220 | +0.331 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | nccl-async | 273 | 3.38e1–7.94e1 | 6.22e1 | 4.32e-3 | 5.68e-3 | 5.78e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 169 | 73–77825 | +1.363e-5 | 0.557 | +1.410e-5 | 0.572 | 90 | +6.902e-6 | 0.358 | 33–981 | +1.084e-3 | 0.791 |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 157 | 886–77825 | +1.330e-5 | 0.625 | +1.368e-5 | 0.629 | 89 | +6.454e-6 | 0.337 | 62–981 | +1.082e-3 | 0.897 |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | 51 | 78593–116785 | +1.089e-5 | 0.089 | +1.078e-5 | 0.087 | 49 | +1.079e-5 | 0.087 | 624–903 | -1.923e-3 | 0.064 |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | 54 | 117502–155745 | -7.435e-6 | 0.039 | -7.138e-6 | 0.036 | 47 | -6.171e-6 | 0.037 | 577–858 | -3.917e-4 | 0.002 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | +1.084e-3 | r0: +1.057e-3, r1: +1.094e-3, r2: +1.105e-3 | r0: 0.792, r1: 0.786, r2: 0.788 | 1.05× | ✓ |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +1.082e-3 | r0: +1.060e-3, r1: +1.091e-3, r2: +1.096e-3 | r0: 0.916, r1: 0.886, r2: 0.883 | 1.03× | ✓ |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | -1.923e-3 | r0: -1.919e-3, r1: -1.926e-3, r2: -1.924e-3 | r0: 0.064, r1: 0.064, r2: 0.065 | 1.00× | ✓ |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | -3.917e-4 | r0: -3.295e-4, r1: -4.531e-4, r2: -3.915e-4 | r0: 0.002, r1: 0.003, r2: 0.003 | 1.38× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | nccl-async | `█▇▇████████████████████▄▄▄▄▅▅▅▅▅▅▅▅▂▁▁▁▁▁▁▁▁▁▁▁` | `▁██████████████████████▆▇▇▇████████▆▆▇▇▇███████` |

### Per-Epoch Detail: resnet-graph / nccl-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 13 | 0.00e0 | 4.62e-1 | 1.08e-1 | 6.19e-2 | 26 | -4.60e-2 | +1.03e-2 | -1.20e-2 | -7.59e-3 |
| 1 | 3.00e-1 | 10 | 6.51e-2 | 1.03e-1 | 7.68e-2 | 8.32e-2 | 27 | -1.64e-2 | +1.77e-2 | +9.36e-4 | -1.43e-3 |
| 2 | 3.00e-1 | 10 | 7.14e-2 | 1.18e-1 | 8.34e-2 | 8.16e-2 | 22 | -2.08e-2 | +1.60e-2 | -1.50e-4 | -5.57e-4 |
| 3 | 3.00e-1 | 11 | 8.12e-2 | 1.31e-1 | 9.35e-2 | 8.60e-2 | 27 | -2.00e-2 | +2.19e-2 | +2.02e-4 | -3.27e-4 |
| 4 | 3.00e-1 | 6 | 9.35e-2 | 1.32e-1 | 1.02e-1 | 9.97e-2 | 29 | -1.05e-2 | +1.09e-2 | +6.87e-4 | +7.76e-5 |
| 5 | 3.00e-1 | 10 | 8.60e-2 | 1.28e-1 | 9.82e-2 | 1.00e-1 | 29 | -1.37e-2 | +1.07e-2 | +1.45e-5 | +9.25e-5 |
| 6 | 3.00e-1 | 6 | 9.20e-2 | 1.32e-1 | 1.04e-1 | 1.06e-1 | 40 | -1.06e-2 | +1.16e-2 | +5.97e-4 | +3.05e-4 |
| 7 | 3.00e-1 | 11 | 8.66e-2 | 1.35e-1 | 9.78e-2 | 9.87e-2 | 30 | -1.48e-2 | +6.92e-3 | -4.38e-4 | -2.53e-5 |
| 8 | 3.00e-1 | 6 | 8.31e-2 | 1.27e-1 | 9.85e-2 | 9.82e-2 | 30 | -1.41e-2 | +1.07e-2 | +1.74e-4 | +6.33e-5 |
| 9 | 3.00e-1 | 2 | 8.83e-2 | 9.20e-2 | 9.02e-2 | 8.83e-2 | 29 | -2.31e-3 | -1.43e-3 | -1.87e-3 | -2.99e-4 |
| 10 | 3.00e-1 | 2 | 9.34e-2 | 2.22e-1 | 1.58e-1 | 2.22e-1 | 270 | +1.80e-4 | +3.20e-3 | +1.69e-3 | +9.38e-5 |
| 12 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 338 | -3.20e-4 | -3.20e-4 | -3.20e-4 | +5.25e-5 |
| 13 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 287 | +1.67e-4 | +1.67e-4 | +1.67e-4 | +6.39e-5 |
| 14 | 3.00e-1 | 1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 281 | -2.13e-4 | -2.13e-4 | -2.13e-4 | +3.62e-5 |
| 15 | 3.00e-1 | 1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 273 | +2.35e-6 | +2.35e-6 | +2.35e-6 | +3.28e-5 |
| 16 | 3.00e-1 | 1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 324 | -4.00e-5 | -4.00e-5 | -4.00e-5 | +2.55e-5 |
| 17 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 328 | +1.37e-4 | +1.37e-4 | +1.37e-4 | +3.66e-5 |
| 18 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 310 | -1.92e-6 | -1.92e-6 | -1.92e-6 | +3.28e-5 |
| 19 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 290 | -6.33e-5 | -6.33e-5 | -6.33e-5 | +2.32e-5 |
| 21 | 3.00e-1 | 2 | 1.99e-1 | 2.05e-1 | 2.02e-1 | 2.05e-1 | 267 | -8.23e-6 | +1.13e-4 | +5.26e-5 | +2.94e-5 |
| 23 | 3.00e-1 | 1 | 1.88e-1 | 1.88e-1 | 1.88e-1 | 1.88e-1 | 360 | -2.40e-4 | -2.40e-4 | -2.40e-4 | +2.42e-6 |
| 24 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 284 | +3.70e-4 | +3.70e-4 | +3.70e-4 | +3.92e-5 |
| 25 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 283 | -1.84e-4 | -1.84e-4 | -1.84e-4 | +1.69e-5 |
| 26 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 306 | +1.72e-5 | +1.72e-5 | +1.72e-5 | +1.69e-5 |
| 27 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 287 | +1.24e-5 | +1.24e-5 | +1.24e-5 | +1.65e-5 |
| 28 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 318 | -7.87e-6 | -7.87e-6 | -7.87e-6 | +1.40e-5 |
| 29 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 313 | +1.26e-4 | +1.26e-4 | +1.26e-4 | +2.52e-5 |
| 30 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 305 | -4.00e-5 | -4.00e-5 | -4.00e-5 | +1.87e-5 |
| 31 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 281 | -1.50e-5 | -1.50e-5 | -1.50e-5 | +1.53e-5 |
| 32 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 266 | -1.20e-4 | -1.20e-4 | -1.20e-4 | +1.77e-6 |
| 33 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 293 | +3.71e-6 | +3.71e-6 | +3.71e-6 | +1.96e-6 |
| 34 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 287 | +9.04e-5 | +9.04e-5 | +9.04e-5 | +1.08e-5 |
| 35 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 279 | -5.02e-5 | -5.02e-5 | -5.02e-5 | +4.70e-6 |
| 36 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 268 | +2.76e-7 | +2.76e-7 | +2.76e-7 | +4.26e-6 |
| 38 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 357 | -1.86e-6 | -1.86e-6 | -1.86e-6 | +3.65e-6 |
| 39 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 351 | +2.25e-4 | +2.25e-4 | +2.25e-4 | +2.58e-5 |
| 40 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 322 | -2.34e-5 | -2.34e-5 | -2.34e-5 | +2.09e-5 |
| 41 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 326 | -1.13e-4 | -1.13e-4 | -1.13e-4 | +7.50e-6 |
| 42 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 298 | +2.88e-5 | +2.88e-5 | +2.88e-5 | +9.63e-6 |
| 43 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 283 | -3.92e-5 | -3.92e-5 | -3.92e-5 | +4.75e-6 |
| 44 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 308 | -7.07e-5 | -7.07e-5 | -7.07e-5 | -2.80e-6 |
| 45 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 259 | +3.35e-5 | +3.35e-5 | +3.35e-5 | +8.25e-7 |
| 46 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 296 | -1.05e-4 | -1.05e-4 | -1.05e-4 | -9.80e-6 |
| 48 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 356 | +1.02e-4 | +1.02e-4 | +1.02e-4 | +1.41e-6 |
| 49 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 322 | +1.53e-4 | +1.53e-4 | +1.53e-4 | +1.65e-5 |
| 50 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 302 | -8.17e-5 | -8.17e-5 | -8.17e-5 | +6.72e-6 |
| 51 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 326 | -5.07e-5 | -5.07e-5 | -5.07e-5 | +9.82e-7 |
| 52 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 249 | +4.36e-5 | +4.36e-5 | +4.36e-5 | +5.24e-6 |
| 53 | 3.00e-1 | 1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 280 | -2.44e-4 | -2.44e-4 | -2.44e-4 | -1.97e-5 |
| 54 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 260 | +1.31e-4 | +1.31e-4 | +1.31e-4 | -4.61e-6 |
| 55 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 265 | -6.92e-5 | -6.92e-5 | -6.92e-5 | -1.11e-5 |
| 56 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 258 | +7.73e-5 | +7.73e-5 | +7.73e-5 | -2.23e-6 |
| 57 | 3.00e-1 | 1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 290 | -9.31e-5 | -9.31e-5 | -9.31e-5 | -1.13e-5 |
| 58 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 276 | +1.63e-4 | +1.63e-4 | +1.63e-4 | +6.08e-6 |
| 59 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 260 | -4.39e-5 | -4.39e-5 | -4.39e-5 | +1.08e-6 |
| 60 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 295 | -5.52e-5 | -5.52e-5 | -5.52e-5 | -4.55e-6 |
| 61 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 295 | +6.45e-5 | +6.45e-5 | +6.45e-5 | +2.36e-6 |
| 62 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 305 | +3.98e-5 | +3.98e-5 | +3.98e-5 | +6.10e-6 |
| 63 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 292 | +6.43e-5 | +6.43e-5 | +6.43e-5 | +1.19e-5 |
| 64 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 288 | -5.62e-5 | -5.62e-5 | -5.62e-5 | +5.12e-6 |
| 65 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 298 | +7.90e-6 | +7.90e-6 | +7.90e-6 | +5.39e-6 |
| 66 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 290 | +3.59e-5 | +3.59e-5 | +3.59e-5 | +8.44e-6 |
| 68 | 3.00e-1 | 2 | 2.08e-1 | 2.12e-1 | 2.10e-1 | 2.12e-1 | 269 | -2.14e-5 | +6.18e-5 | +2.02e-5 | +1.11e-5 |
| 70 | 3.00e-1 | 2 | 2.02e-1 | 2.18e-1 | 2.10e-1 | 2.18e-1 | 269 | -1.42e-4 | +2.79e-4 | +6.83e-5 | +2.41e-5 |
| 72 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 353 | -1.76e-4 | -1.76e-4 | -1.76e-4 | +4.08e-6 |
| 73 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 304 | +2.33e-4 | +2.33e-4 | +2.33e-4 | +2.70e-5 |
| 74 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 270 | -1.73e-4 | -1.73e-4 | -1.73e-4 | +6.98e-6 |
| 75 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 285 | -8.66e-5 | -8.66e-5 | -8.66e-5 | -2.37e-6 |
| 76 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 317 | +5.27e-5 | +5.27e-5 | +5.27e-5 | +3.13e-6 |
| 77 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 330 | +5.96e-5 | +5.96e-5 | +5.96e-5 | +8.78e-6 |
| 78 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 291 | +7.49e-5 | +7.49e-5 | +7.49e-5 | +1.54e-5 |
| 79 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 272 | -1.37e-4 | -1.37e-4 | -1.37e-4 | +1.20e-7 |
| 80 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 292 | -4.42e-5 | -4.42e-5 | -4.42e-5 | -4.32e-6 |
| 82 | 3.00e-1 | 2 | 2.08e-1 | 2.14e-1 | 2.11e-1 | 2.14e-1 | 254 | +3.34e-5 | +1.03e-4 | +6.84e-5 | +9.86e-6 |
| 84 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 333 | -2.27e-4 | -2.27e-4 | -2.27e-4 | -1.38e-5 |
| 85 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 293 | +2.94e-4 | +2.94e-4 | +2.94e-4 | +1.69e-5 |
| 86 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 272 | -1.50e-4 | -1.50e-4 | -1.50e-4 | +1.65e-7 |
| 87 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 300 | -5.18e-5 | -5.18e-5 | -5.18e-5 | -5.03e-6 |
| 88 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 297 | +7.88e-5 | +7.88e-5 | +7.88e-5 | +3.35e-6 |
| 89 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 304 | +1.97e-5 | +1.97e-5 | +1.97e-5 | +4.98e-6 |
| 90 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 290 | +6.69e-5 | +6.69e-5 | +6.69e-5 | +1.12e-5 |
| 91 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 255 | -1.71e-4 | -1.71e-4 | -1.71e-4 | -7.02e-6 |
| 92 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 254 | -1.15e-4 | -1.15e-4 | -1.15e-4 | -1.78e-5 |
| 93 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 259 | -1.34e-5 | -1.34e-5 | -1.34e-5 | -1.73e-5 |
| 94 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 290 | +4.22e-5 | +4.22e-5 | +4.22e-5 | -1.14e-5 |
| 95 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 325 | +1.26e-4 | +1.26e-4 | +1.26e-4 | +2.38e-6 |
| 96 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 302 | +1.10e-4 | +1.10e-4 | +1.10e-4 | +1.31e-5 |
| 97 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 270 | -9.94e-5 | -9.94e-5 | -9.94e-5 | +1.87e-6 |
| 98 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 283 | -9.67e-5 | -9.67e-5 | -9.67e-5 | -7.99e-6 |
| 99 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 278 | +4.34e-5 | +4.34e-5 | +4.34e-5 | -2.85e-6 |
| 100 | 3.00e-2 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 268 | -4.93e-5 | -4.93e-5 | -4.93e-5 | -7.50e-6 |
| 101 | 3.00e-2 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 258 | -4.37e-5 | -4.37e-5 | -4.37e-5 | -1.11e-5 |
| 102 | 3.00e-2 | 1 | 2.17e-2 | 2.17e-2 | 2.17e-2 | 2.17e-2 | 288 | -7.76e-3 | -7.76e-3 | -7.76e-3 | -7.86e-4 |
| 103 | 3.00e-2 | 1 | 2.38e-2 | 2.38e-2 | 2.38e-2 | 2.38e-2 | 334 | +2.87e-4 | +2.87e-4 | +2.87e-4 | -6.79e-4 |
| 104 | 3.00e-2 | 1 | 2.62e-2 | 2.62e-2 | 2.62e-2 | 2.62e-2 | 317 | +3.04e-4 | +3.04e-4 | +3.04e-4 | -5.81e-4 |
| 105 | 3.00e-2 | 1 | 2.74e-2 | 2.74e-2 | 2.74e-2 | 2.74e-2 | 274 | +1.56e-4 | +1.56e-4 | +1.56e-4 | -5.07e-4 |
| 106 | 3.00e-2 | 1 | 2.64e-2 | 2.64e-2 | 2.64e-2 | 2.64e-2 | 306 | -1.16e-4 | -1.16e-4 | -1.16e-4 | -4.68e-4 |
| 107 | 3.00e-2 | 1 | 2.78e-2 | 2.78e-2 | 2.78e-2 | 2.78e-2 | 276 | +1.78e-4 | +1.78e-4 | +1.78e-4 | -4.03e-4 |
| 108 | 3.00e-2 | 1 | 2.83e-2 | 2.83e-2 | 2.83e-2 | 2.83e-2 | 263 | +7.88e-5 | +7.88e-5 | +7.88e-5 | -3.55e-4 |
| 109 | 3.00e-2 | 1 | 2.81e-2 | 2.81e-2 | 2.81e-2 | 2.81e-2 | 259 | -2.95e-5 | -2.95e-5 | -2.95e-5 | -3.23e-4 |
| 110 | 3.00e-2 | 1 | 2.90e-2 | 2.90e-2 | 2.90e-2 | 2.90e-2 | 270 | +1.15e-4 | +1.15e-4 | +1.15e-4 | -2.79e-4 |
| 111 | 3.00e-2 | 1 | 3.00e-2 | 3.00e-2 | 3.00e-2 | 3.00e-2 | 272 | +1.25e-4 | +1.25e-4 | +1.25e-4 | -2.38e-4 |
| 112 | 3.00e-2 | 1 | 3.07e-2 | 3.07e-2 | 3.07e-2 | 3.07e-2 | 255 | +8.39e-5 | +8.39e-5 | +8.39e-5 | -2.06e-4 |
| 113 | 3.00e-2 | 1 | 3.01e-2 | 3.01e-2 | 3.01e-2 | 3.01e-2 | 275 | -6.49e-5 | -6.49e-5 | -6.49e-5 | -1.92e-4 |
| 114 | 3.00e-2 | 1 | 3.23e-2 | 3.23e-2 | 3.23e-2 | 3.23e-2 | 273 | +2.52e-4 | +2.52e-4 | +2.52e-4 | -1.48e-4 |
| 115 | 3.00e-2 | 1 | 3.27e-2 | 3.27e-2 | 3.27e-2 | 3.27e-2 | 281 | +4.16e-5 | +4.16e-5 | +4.16e-5 | -1.29e-4 |
| 116 | 3.00e-2 | 1 | 3.35e-2 | 3.35e-2 | 3.35e-2 | 3.35e-2 | 281 | +9.09e-5 | +9.09e-5 | +9.09e-5 | -1.07e-4 |
| 117 | 3.00e-2 | 1 | 3.43e-2 | 3.43e-2 | 3.43e-2 | 3.43e-2 | 265 | +9.04e-5 | +9.04e-5 | +9.04e-5 | -8.70e-5 |
| 118 | 3.00e-2 | 1 | 3.47e-2 | 3.47e-2 | 3.47e-2 | 3.47e-2 | 285 | +4.50e-5 | +4.50e-5 | +4.50e-5 | -7.38e-5 |
| 119 | 3.00e-2 | 1 | 3.61e-2 | 3.61e-2 | 3.61e-2 | 3.61e-2 | 267 | +1.44e-4 | +1.44e-4 | +1.44e-4 | -5.21e-5 |
| 120 | 3.00e-2 | 1 | 3.59e-2 | 3.59e-2 | 3.59e-2 | 3.59e-2 | 303 | -1.46e-5 | -1.46e-5 | -1.46e-5 | -4.84e-5 |
| 121 | 3.00e-2 | 1 | 3.90e-2 | 3.90e-2 | 3.90e-2 | 3.90e-2 | 241 | +3.40e-4 | +3.40e-4 | +3.40e-4 | -9.56e-6 |
| 122 | 3.00e-2 | 1 | 3.59e-2 | 3.59e-2 | 3.59e-2 | 3.59e-2 | 259 | -3.23e-4 | -3.23e-4 | -3.23e-4 | -4.09e-5 |
| 123 | 3.00e-2 | 1 | 3.72e-2 | 3.72e-2 | 3.72e-2 | 3.72e-2 | 248 | +1.50e-4 | +1.50e-4 | +1.50e-4 | -2.19e-5 |
| 124 | 3.00e-2 | 1 | 3.80e-2 | 3.80e-2 | 3.80e-2 | 3.80e-2 | 270 | +7.96e-5 | +7.96e-5 | +7.96e-5 | -1.17e-5 |
| 125 | 3.00e-2 | 1 | 3.97e-2 | 3.97e-2 | 3.97e-2 | 3.97e-2 | 246 | +1.70e-4 | +1.70e-4 | +1.70e-4 | +6.50e-6 |
| 126 | 3.00e-2 | 1 | 3.92e-2 | 3.92e-2 | 3.92e-2 | 3.92e-2 | 308 | -4.23e-5 | -4.23e-5 | -4.23e-5 | +1.62e-6 |
| 127 | 3.00e-2 | 1 | 4.38e-2 | 4.38e-2 | 4.38e-2 | 4.38e-2 | 252 | +4.44e-4 | +4.44e-4 | +4.44e-4 | +4.59e-5 |
| 128 | 3.00e-2 | 2 | 4.08e-2 | 4.20e-2 | 4.14e-2 | 4.20e-2 | 232 | -2.65e-4 | +1.23e-4 | -7.13e-5 | +2.55e-5 |
| 130 | 3.00e-2 | 2 | 4.07e-2 | 4.59e-2 | 4.33e-2 | 4.59e-2 | 227 | -1.00e-4 | +5.27e-4 | +2.13e-4 | +6.44e-5 |
| 131 | 3.00e-2 | 1 | 4.07e-2 | 4.07e-2 | 4.07e-2 | 4.07e-2 | 252 | -4.80e-4 | -4.80e-4 | -4.80e-4 | +9.95e-6 |
| 132 | 3.00e-2 | 1 | 4.40e-2 | 4.40e-2 | 4.40e-2 | 4.40e-2 | 260 | +3.05e-4 | +3.05e-4 | +3.05e-4 | +3.95e-5 |
| 133 | 3.00e-2 | 1 | 4.51e-2 | 4.51e-2 | 4.51e-2 | 4.51e-2 | 267 | +8.51e-5 | +8.51e-5 | +8.51e-5 | +4.40e-5 |
| 134 | 3.00e-2 | 1 | 4.62e-2 | 4.62e-2 | 4.62e-2 | 4.62e-2 | 286 | +8.86e-5 | +8.86e-5 | +8.86e-5 | +4.85e-5 |
| 135 | 3.00e-2 | 1 | 4.69e-2 | 4.69e-2 | 4.69e-2 | 4.69e-2 | 296 | +4.73e-5 | +4.73e-5 | +4.73e-5 | +4.83e-5 |
| 136 | 3.00e-2 | 1 | 4.94e-2 | 4.94e-2 | 4.94e-2 | 4.94e-2 | 313 | +1.71e-4 | +1.71e-4 | +1.71e-4 | +6.06e-5 |
| 137 | 3.00e-2 | 1 | 5.16e-2 | 5.16e-2 | 5.16e-2 | 5.16e-2 | 278 | +1.52e-4 | +1.52e-4 | +1.52e-4 | +6.98e-5 |
| 138 | 3.00e-2 | 1 | 4.95e-2 | 4.95e-2 | 4.95e-2 | 4.95e-2 | 254 | -1.64e-4 | -1.64e-4 | -1.64e-4 | +4.63e-5 |
| 139 | 3.00e-2 | 1 | 4.78e-2 | 4.78e-2 | 4.78e-2 | 4.78e-2 | 246 | -1.41e-4 | -1.41e-4 | -1.41e-4 | +2.76e-5 |
| 140 | 3.00e-2 | 1 | 4.95e-2 | 4.95e-2 | 4.95e-2 | 4.95e-2 | 272 | +1.31e-4 | +1.31e-4 | +1.31e-4 | +3.80e-5 |
| 141 | 3.00e-2 | 1 | 5.15e-2 | 5.15e-2 | 5.15e-2 | 5.15e-2 | 284 | +1.36e-4 | +1.36e-4 | +1.36e-4 | +4.78e-5 |
| 142 | 3.00e-2 | 1 | 5.29e-2 | 5.29e-2 | 5.29e-2 | 5.29e-2 | 245 | +1.15e-4 | +1.15e-4 | +1.15e-4 | +5.45e-5 |
| 143 | 3.00e-2 | 1 | 5.08e-2 | 5.08e-2 | 5.08e-2 | 5.08e-2 | 289 | -1.41e-4 | -1.41e-4 | -1.41e-4 | +3.49e-5 |
| 144 | 3.00e-2 | 1 | 5.47e-2 | 5.47e-2 | 5.47e-2 | 5.47e-2 | 275 | +2.68e-4 | +2.68e-4 | +2.68e-4 | +5.83e-5 |
| 145 | 3.00e-2 | 1 | 5.38e-2 | 5.38e-2 | 5.38e-2 | 5.38e-2 | 283 | -6.10e-5 | -6.10e-5 | -6.10e-5 | +4.63e-5 |
| 146 | 3.00e-2 | 1 | 5.53e-2 | 5.53e-2 | 5.53e-2 | 5.53e-2 | 282 | +9.67e-5 | +9.67e-5 | +9.67e-5 | +5.14e-5 |
| 147 | 3.00e-2 | 1 | 5.53e-2 | 5.53e-2 | 5.53e-2 | 5.53e-2 | 295 | +1.05e-6 | +1.05e-6 | +1.05e-6 | +4.63e-5 |
| 148 | 3.00e-2 | 1 | 5.57e-2 | 5.57e-2 | 5.57e-2 | 5.57e-2 | 300 | +2.79e-5 | +2.79e-5 | +2.79e-5 | +4.45e-5 |
| 149 | 3.00e-2 | 1 | 5.71e-2 | 5.71e-2 | 5.71e-2 | 5.71e-2 | 283 | +8.20e-5 | +8.20e-5 | +8.20e-5 | +4.82e-5 |
| 150 | 3.00e-3 | 1 | 5.61e-2 | 5.61e-2 | 5.61e-2 | 5.61e-2 | 254 | -6.45e-5 | -6.45e-5 | -6.45e-5 | +3.70e-5 |
| 151 | 3.00e-3 | 2 | 5.57e-3 | 5.64e-2 | 3.10e-2 | 5.57e-3 | 223 | -1.04e-2 | +2.29e-5 | -5.18e-3 | -1.01e-3 |
| 153 | 3.00e-3 | 2 | 5.28e-3 | 6.06e-3 | 5.67e-3 | 6.06e-3 | 223 | -1.78e-4 | +6.23e-4 | +2.22e-4 | -7.69e-4 |
| 154 | 3.00e-3 | 1 | 5.22e-3 | 5.22e-3 | 5.22e-3 | 5.22e-3 | 250 | -5.97e-4 | -5.97e-4 | -5.97e-4 | -7.52e-4 |
| 155 | 3.00e-3 | 1 | 5.62e-3 | 5.62e-3 | 5.62e-3 | 5.62e-3 | 283 | +2.62e-4 | +2.62e-4 | +2.62e-4 | -6.50e-4 |
| 156 | 3.00e-3 | 1 | 5.94e-3 | 5.94e-3 | 5.94e-3 | 5.94e-3 | 262 | +2.07e-4 | +2.07e-4 | +2.07e-4 | -5.65e-4 |
| 157 | 3.00e-3 | 1 | 5.59e-3 | 5.59e-3 | 5.59e-3 | 5.59e-3 | 245 | -2.47e-4 | -2.47e-4 | -2.47e-4 | -5.33e-4 |
| 158 | 3.00e-3 | 1 | 5.65e-3 | 5.65e-3 | 5.65e-3 | 5.65e-3 | 295 | +4.10e-5 | +4.10e-5 | +4.10e-5 | -4.76e-4 |
| 159 | 3.00e-3 | 1 | 5.97e-3 | 5.97e-3 | 5.97e-3 | 5.97e-3 | 257 | +2.10e-4 | +2.10e-4 | +2.10e-4 | -4.07e-4 |
| 160 | 3.00e-3 | 1 | 5.84e-3 | 5.84e-3 | 5.84e-3 | 5.84e-3 | 263 | -8.42e-5 | -8.42e-5 | -8.42e-5 | -3.75e-4 |
| 161 | 3.00e-3 | 1 | 5.82e-3 | 5.82e-3 | 5.82e-3 | 5.82e-3 | 248 | -1.46e-5 | -1.46e-5 | -1.46e-5 | -3.39e-4 |
| 162 | 3.00e-3 | 2 | 5.75e-3 | 6.04e-3 | 5.89e-3 | 6.04e-3 | 206 | -4.38e-5 | +2.36e-4 | +9.63e-5 | -2.55e-4 |
| 164 | 3.00e-3 | 2 | 5.21e-3 | 6.20e-3 | 5.70e-3 | 6.20e-3 | 217 | -5.25e-4 | +8.06e-4 | +1.41e-4 | -1.73e-4 |
| 165 | 3.00e-3 | 1 | 5.52e-3 | 5.52e-3 | 5.52e-3 | 5.52e-3 | 252 | -4.64e-4 | -4.64e-4 | -4.64e-4 | -2.02e-4 |
| 166 | 3.00e-3 | 1 | 5.99e-3 | 5.99e-3 | 5.99e-3 | 5.99e-3 | 240 | +3.39e-4 | +3.39e-4 | +3.39e-4 | -1.48e-4 |
| 167 | 3.00e-3 | 1 | 5.95e-3 | 5.95e-3 | 5.95e-3 | 5.95e-3 | 280 | -2.24e-5 | -2.24e-5 | -2.24e-5 | -1.35e-4 |
| 168 | 3.00e-3 | 1 | 6.31e-3 | 6.31e-3 | 6.31e-3 | 6.31e-3 | 303 | +1.96e-4 | +1.96e-4 | +1.96e-4 | -1.02e-4 |
| 169 | 3.00e-3 | 1 | 6.55e-3 | 6.55e-3 | 6.55e-3 | 6.55e-3 | 266 | +1.39e-4 | +1.39e-4 | +1.39e-4 | -7.80e-5 |
| 170 | 3.00e-3 | 1 | 6.13e-3 | 6.13e-3 | 6.13e-3 | 6.13e-3 | 276 | -2.41e-4 | -2.41e-4 | -2.41e-4 | -9.44e-5 |
| 171 | 3.00e-3 | 1 | 6.32e-3 | 6.32e-3 | 6.32e-3 | 6.32e-3 | 279 | +1.12e-4 | +1.12e-4 | +1.12e-4 | -7.37e-5 |
| 172 | 3.00e-3 | 1 | 6.40e-3 | 6.40e-3 | 6.40e-3 | 6.40e-3 | 258 | +4.92e-5 | +4.92e-5 | +4.92e-5 | -6.14e-5 |
| 173 | 3.00e-3 | 1 | 6.34e-3 | 6.34e-3 | 6.34e-3 | 6.34e-3 | 246 | -4.06e-5 | -4.06e-5 | -4.06e-5 | -5.94e-5 |
| 174 | 3.00e-3 | 1 | 6.07e-3 | 6.07e-3 | 6.07e-3 | 6.07e-3 | 257 | -1.68e-4 | -1.68e-4 | -1.68e-4 | -7.02e-5 |
| 175 | 3.00e-3 | 1 | 6.30e-3 | 6.30e-3 | 6.30e-3 | 6.30e-3 | 238 | +1.56e-4 | +1.56e-4 | +1.56e-4 | -4.76e-5 |
| 176 | 3.00e-3 | 2 | 6.23e-3 | 6.26e-3 | 6.24e-3 | 6.26e-3 | 221 | -4.93e-5 | +2.17e-5 | -1.38e-5 | -4.09e-5 |
| 177 | 3.00e-3 | 1 | 5.77e-3 | 5.77e-3 | 5.77e-3 | 5.77e-3 | 229 | -3.53e-4 | -3.53e-4 | -3.53e-4 | -7.21e-5 |
| 178 | 3.00e-3 | 1 | 6.17e-3 | 6.17e-3 | 6.17e-3 | 6.17e-3 | 258 | +2.58e-4 | +2.58e-4 | +2.58e-4 | -3.91e-5 |
| 179 | 3.00e-3 | 1 | 6.40e-3 | 6.40e-3 | 6.40e-3 | 6.40e-3 | 277 | +1.34e-4 | +1.34e-4 | +1.34e-4 | -2.18e-5 |
| 180 | 3.00e-3 | 1 | 6.51e-3 | 6.51e-3 | 6.51e-3 | 6.51e-3 | 243 | +7.02e-5 | +7.02e-5 | +7.02e-5 | -1.26e-5 |
| 181 | 3.00e-3 | 1 | 6.24e-3 | 6.24e-3 | 6.24e-3 | 6.24e-3 | 258 | -1.66e-4 | -1.66e-4 | -1.66e-4 | -2.80e-5 |
| 182 | 3.00e-3 | 1 | 6.52e-3 | 6.52e-3 | 6.52e-3 | 6.52e-3 | 259 | +1.70e-4 | +1.70e-4 | +1.70e-4 | -8.22e-6 |
| 183 | 3.00e-3 | 1 | 6.29e-3 | 6.29e-3 | 6.29e-3 | 6.29e-3 | 256 | -1.36e-4 | -1.36e-4 | -1.36e-4 | -2.10e-5 |
| 184 | 3.00e-3 | 1 | 6.47e-3 | 6.47e-3 | 6.47e-3 | 6.47e-3 | 269 | +1.04e-4 | +1.04e-4 | +1.04e-4 | -8.54e-6 |
| 185 | 3.00e-3 | 1 | 6.53e-3 | 6.53e-3 | 6.53e-3 | 6.53e-3 | 262 | +3.42e-5 | +3.42e-5 | +3.42e-5 | -4.27e-6 |
| 186 | 3.00e-3 | 1 | 6.56e-3 | 6.56e-3 | 6.56e-3 | 6.56e-3 | 267 | +1.64e-5 | +1.64e-5 | +1.64e-5 | -2.20e-6 |
| 187 | 3.00e-3 | 1 | 6.87e-3 | 6.87e-3 | 6.87e-3 | 6.87e-3 | 269 | +1.76e-4 | +1.76e-4 | +1.76e-4 | +1.56e-5 |
| 188 | 3.00e-3 | 1 | 6.64e-3 | 6.64e-3 | 6.64e-3 | 6.64e-3 | 276 | -1.25e-4 | -1.25e-4 | -1.25e-4 | +1.50e-6 |
| 189 | 3.00e-3 | 1 | 6.54e-3 | 6.54e-3 | 6.54e-3 | 6.54e-3 | 272 | -5.85e-5 | -5.85e-5 | -5.85e-5 | -4.50e-6 |
| 190 | 3.00e-3 | 2 | 6.32e-3 | 6.68e-3 | 6.50e-3 | 6.32e-3 | 221 | -2.52e-4 | +8.90e-5 | -8.14e-5 | -2.08e-5 |
| 192 | 3.00e-3 | 2 | 6.26e-3 | 7.23e-3 | 6.74e-3 | 7.23e-3 | 221 | -2.93e-5 | +6.55e-4 | +3.13e-4 | +4.60e-5 |
| 193 | 3.00e-3 | 1 | 6.27e-3 | 6.27e-3 | 6.27e-3 | 6.27e-3 | 270 | -5.27e-4 | -5.27e-4 | -5.27e-4 | -1.13e-5 |
| 194 | 3.00e-3 | 1 | 6.68e-3 | 6.68e-3 | 6.68e-3 | 6.68e-3 | 267 | +2.33e-4 | +2.33e-4 | +2.33e-4 | +1.32e-5 |
| 195 | 3.00e-3 | 1 | 6.68e-3 | 6.68e-3 | 6.68e-3 | 6.68e-3 | 268 | +1.14e-6 | +1.14e-6 | +1.14e-6 | +1.20e-5 |
| 196 | 3.00e-3 | 1 | 6.55e-3 | 6.55e-3 | 6.55e-3 | 6.55e-3 | 261 | -7.28e-5 | -7.28e-5 | -7.28e-5 | +3.48e-6 |
| 197 | 3.00e-3 | 1 | 6.77e-3 | 6.77e-3 | 6.77e-3 | 6.77e-3 | 230 | +1.40e-4 | +1.40e-4 | +1.40e-4 | +1.71e-5 |
| 198 | 3.00e-3 | 1 | 6.38e-3 | 6.38e-3 | 6.38e-3 | 6.38e-3 | 237 | -2.51e-4 | -2.51e-4 | -2.51e-4 | -9.67e-6 |
| 199 | 3.00e-3 | 1 | 6.52e-3 | 6.52e-3 | 6.52e-3 | 6.52e-3 | 251 | +9.01e-5 | +9.01e-5 | +9.01e-5 | +3.04e-7 |

