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
| cpu-async | 0.070984 | 0.9182 | +0.0057 | 1811.5 | 731 | 77.0 | 100% | 100% | 100% | 6.7 |

## Best Mode per Model

| Model | Best Eval | Mode | Fastest (within 2% of solo-0) | Mode |
|-------|-----------|------|-------------------------------|------|
| resnet-graph | 0.9182 | cpu-async | - | - |

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
| cpu-async | 2.0584 | 0.7677 | 0.5911 | 0.5318 | 0.5124 | 0.4866 | 0.4768 | 0.4920 | 0.4905 | 0.4817 | 0.2284 | 0.1883 | 0.1718 | 0.1635 | 0.1508 | 0.0884 | 0.0821 | 0.0779 | 0.0748 | 0.0710 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | cpu-async | 0 | cuda:0 | 0.4013 | 2.6 |
| resnet-graph | cpu-async | 1 | cuda:1 | 0.3032 | 3.6 |
| resnet-graph | cpu-async | 2 | cuda:2 | 0.2955 | 3.7 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) | GPU2 Peak (MB) | GPU2 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|---------------|---------------|
| resnet-graph | cpu-async | 358 | 357 | 394 | 390 | 394 | 390 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| cpu-async | gpu2 | 1809.1 | 2.4 | epoch-boundary(199) |
| cpu-async | gpu1 | 1809.2 | 2.3 | epoch-boundary(199) |
| cpu-async | gpu0 | 1110.8 | 0.7 | epoch-boundary(123) |
| cpu-async | gpu1 | 1110.8 | 0.7 | epoch-boundary(123) |
| cpu-async | gpu2 | 1110.8 | 0.6 | epoch-boundary(123) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | cpu-async | gpu0 | 0.7s | 0.0s | 0.0s | 0.0s | 0.7s |
| resnet-graph | cpu-async | gpu1 | 3.0s | 0.0s | 0.0s | 0.0s | 3.0s |
| resnet-graph | cpu-async | gpu2 | 3.0s | 0.0s | 0.0s | 0.0s | 3.0s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | cpu-async | 347 | 0 | 731 | 77.0 | 1368/8356 | 731 | 77.0 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | cpu-async | 198.7 | 11.0% |

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
| resnet-graph | cpu-async | 199 | 731 | 0 | 4.00e-3 | -6.39e-4 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | cpu-async | 0 | 731 | 9.75e-2 | 7.01e-2 | 3.01e-3 | 4.84e-1 | 32.7 | -1.23e-4 | 7.49e-4 |
| resnet-graph | cpu-async | 1 | 731 | 9.75e-2 | 6.97e-2 | 3.12e-3 | 4.29e-1 | 32.8 | -1.46e-4 | 8.50e-4 |
| resnet-graph | cpu-async | 2 | 731 | 9.80e-2 | 7.04e-2 | 3.05e-3 | 4.41e-1 | 34.5 | -1.66e-4 | 9.91e-4 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | cpu-async | rank0 ↔ rank1 | +0.9924 |
| resnet-graph | cpu-async | rank0 ↔ rank2 | +0.9894 |
| resnet-graph | cpu-async | rank1 ↔ rank2 | +0.9898 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | cpu-async | 60 (0,1,2,4,5,7,8,9…148,149) | 0 (—) | — | 0,1,2,4,5,7,8,9…148,149 | — |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | cpu-async | 0e0 | 3 | 5 | 5 |
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
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 465 | +0.046 |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | 65 | -0.270 |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | 197 | +0.019 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | cpu-async | 729 | -0.026 | 198 | +0.381 | +0.592 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | cpu-async | 730 | 3.38e1–8.01e1 | 6.67e1 | 1.80e-3 | 3.43e-3 | 4.12e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 467 | 34–77903 | +2.262e-6 | 0.035 | +2.542e-6 | 0.048 | 100 | +8.791e-6 | 0.516 | 29–873 | +8.323e-4 | 0.501 |
| resnet-graph | cpu-async | 3.00e-1 | 2–99 (post-transient, skipped 2) | 449 | 1708–77903 | +4.430e-6 | 0.173 | +4.690e-6 | 0.210 | 98 | +9.846e-6 | 0.628 | 70–873 | +9.030e-4 | 0.833 |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | 66 | 78602–117063 | +1.275e-6 | 0.002 | +9.434e-7 | 0.001 | 49 | -2.431e-6 | 0.006 | 405–854 | -2.357e-4 | 0.007 |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | 198 | 117588–156005 | -2.171e-5 | 0.403 | -2.176e-5 | 0.416 | 50 | -2.905e-5 | 0.444 | 85–572 | +2.492e-3 | 0.648 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | +8.323e-4 | r0: +8.050e-4, r1: +8.432e-4, r2: +8.496e-4 | r0: 0.476, r1: 0.494, r2: 0.470 | 1.06× | ✓ |
| resnet-graph | cpu-async | 3.00e-1 | 2–99 (post-transient, skipped 2) | +9.030e-4 | r0: +8.759e-4, r1: +9.125e-4, r2: +9.214e-4 | r0: 0.814, r1: 0.793, r2: 0.753 | 1.05× | ✓ |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | -2.357e-4 | r0: -3.369e-4, r1: -1.755e-4, r2: -1.924e-4 | r0: 0.014, r1: 0.004, r2: 0.005 | 1.92× | ✓ |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | +2.492e-3 | r0: +2.436e-3, r1: +2.534e-3, r2: +2.503e-3 | r0: 0.620, r1: 0.648, r2: 0.639 | 1.04× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | cpu-async | `█▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇████████▇▅▅▅▅▅▅▅▅▅▅▅▅▂▂▁▁▁▁▁▁▁▁▁▁` | `▁▅▆▇▇▆▆▇▇▆▇▇▆▆▇▇▇▇███████▇▇▇▇████████▇▆▇▇▇▇▇▆▆▇▆▇▆` |

### Per-Epoch Detail: resnet-graph / cpu-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 11 | 1.86e-1 | 4.84e-1 | 2.83e-1 | 1.99e-1 | 33 | -2.14e-2 | +3.66e-3 | -6.17e-3 | -4.90e-3 |
| 1 | 3.00e-1 | 7 | 1.83e-1 | 2.68e-1 | 2.05e-1 | 2.03e-1 | 40 | -6.89e-3 | +3.38e-3 | -8.14e-4 | -2.19e-3 |
| 2 | 3.00e-1 | 9 | 1.52e-1 | 2.39e-1 | 1.91e-1 | 1.52e-1 | 33 | -3.44e-3 | +1.83e-3 | -1.23e-3 | -1.66e-3 |
| 3 | 3.00e-1 | 4 | 1.49e-1 | 1.85e-1 | 1.62e-1 | 1.49e-1 | 42 | -4.31e-3 | +2.73e-3 | -7.13e-4 | -1.36e-3 |
| 4 | 3.00e-1 | 6 | 1.39e-1 | 1.79e-1 | 1.49e-1 | 1.40e-1 | 46 | -3.45e-3 | +2.01e-3 | -5.87e-4 | -9.76e-4 |
| 5 | 3.00e-1 | 5 | 1.40e-1 | 1.77e-1 | 1.51e-1 | 1.40e-1 | 42 | -3.15e-3 | +2.52e-3 | -5.09e-4 | -8.00e-4 |
| 6 | 3.00e-1 | 9 | 1.21e-1 | 1.68e-1 | 1.35e-1 | 1.25e-1 | 32 | -3.77e-3 | +2.28e-3 | -5.58e-4 | -6.20e-4 |
| 7 | 3.00e-1 | 5 | 1.15e-1 | 1.70e-1 | 1.33e-1 | 1.15e-1 | 28 | -7.59e-3 | +3.57e-3 | -1.86e-3 | -1.16e-3 |
| 8 | 3.00e-1 | 8 | 1.13e-1 | 1.58e-1 | 1.27e-1 | 1.28e-1 | 36 | -7.56e-3 | +4.40e-3 | -4.76e-4 | -6.75e-4 |
| 9 | 3.00e-1 | 7 | 1.32e-1 | 1.68e-1 | 1.41e-1 | 1.39e-1 | 51 | -3.71e-3 | +3.52e-3 | -1.98e-4 | -4.11e-4 |
| 10 | 3.00e-1 | 6 | 1.26e-1 | 1.73e-1 | 1.39e-1 | 1.29e-1 | 42 | -5.09e-3 | +2.64e-3 | -8.71e-4 | -5.98e-4 |
| 11 | 3.00e-1 | 9 | 1.20e-1 | 1.77e-1 | 1.33e-1 | 1.25e-1 | 36 | -5.23e-3 | +3.41e-3 | -6.17e-4 | -5.37e-4 |
| 12 | 3.00e-1 | 3 | 1.46e-1 | 1.63e-1 | 1.51e-1 | 1.46e-1 | 50 | -2.17e-3 | +3.26e-3 | +3.43e-4 | -3.29e-4 |
| 13 | 3.00e-1 | 7 | 1.34e-1 | 1.72e-1 | 1.43e-1 | 1.34e-1 | 44 | -3.65e-3 | +1.81e-3 | -5.20e-4 | -4.17e-4 |
| 14 | 3.00e-1 | 4 | 1.33e-1 | 1.61e-1 | 1.42e-1 | 1.33e-1 | 44 | -2.97e-3 | +2.33e-3 | -4.65e-4 | -4.57e-4 |
| 15 | 3.00e-1 | 6 | 1.28e-1 | 1.62e-1 | 1.38e-1 | 1.35e-1 | 48 | -3.79e-3 | +2.36e-3 | -3.88e-4 | -4.02e-4 |
| 16 | 3.00e-1 | 7 | 1.22e-1 | 1.63e-1 | 1.32e-1 | 1.26e-1 | 38 | -5.32e-3 | +2.45e-3 | -7.51e-4 | -5.16e-4 |
| 17 | 3.00e-1 | 8 | 1.20e-1 | 1.62e-1 | 1.30e-1 | 1.28e-1 | 41 | -5.97e-3 | +3.12e-3 | -4.73e-4 | -4.16e-4 |
| 18 | 3.00e-1 | 5 | 1.27e-1 | 1.66e-1 | 1.41e-1 | 1.27e-1 | 38 | -4.06e-3 | +3.31e-3 | -6.65e-4 | -5.66e-4 |
| 19 | 3.00e-1 | 7 | 1.23e-1 | 1.60e-1 | 1.32e-1 | 1.23e-1 | 35 | -4.62e-3 | +3.14e-3 | -5.31e-4 | -5.53e-4 |
| 20 | 3.00e-1 | 7 | 1.21e-1 | 1.64e-1 | 1.32e-1 | 1.29e-1 | 38 | -6.22e-3 | +3.50e-3 | -5.31e-4 | -4.77e-4 |
| 21 | 3.00e-1 | 7 | 1.21e-1 | 1.62e-1 | 1.33e-1 | 1.27e-1 | 38 | -3.72e-3 | +2.97e-3 | -5.06e-4 | -4.60e-4 |
| 22 | 3.00e-1 | 7 | 1.21e-1 | 1.61e-1 | 1.30e-1 | 1.22e-1 | 37 | -5.60e-3 | +3.08e-3 | -7.66e-4 | -5.86e-4 |
| 23 | 3.00e-1 | 7 | 1.16e-1 | 1.63e-1 | 1.28e-1 | 1.24e-1 | 42 | -7.25e-3 | +3.29e-3 | -8.46e-4 | -6.43e-4 |
| 24 | 3.00e-1 | 6 | 1.33e-1 | 1.65e-1 | 1.43e-1 | 1.37e-1 | 50 | -3.18e-3 | +3.27e-3 | -1.38e-4 | -4.26e-4 |
| 25 | 3.00e-1 | 5 | 1.32e-1 | 1.78e-1 | 1.47e-1 | 1.32e-1 | 42 | -3.85e-3 | +2.67e-3 | -7.88e-4 | -6.00e-4 |
| 26 | 3.00e-1 | 6 | 1.27e-1 | 1.71e-1 | 1.39e-1 | 1.31e-1 | 39 | -4.00e-3 | +3.03e-3 | -5.64e-4 | -5.67e-4 |
| 27 | 3.00e-1 | 6 | 1.27e-1 | 1.68e-1 | 1.38e-1 | 1.31e-1 | 46 | -4.46e-3 | +2.90e-3 | -5.29e-4 | -5.36e-4 |
| 28 | 3.00e-1 | 6 | 1.34e-1 | 1.66e-1 | 1.42e-1 | 1.34e-1 | 46 | -4.02e-3 | +2.94e-3 | -3.35e-4 | -4.57e-4 |
| 29 | 3.00e-1 | 5 | 1.37e-1 | 1.64e-1 | 1.44e-1 | 1.39e-1 | 47 | -2.99e-3 | +2.59e-3 | -2.59e-4 | -3.87e-4 |
| 30 | 3.00e-1 | 9 | 1.19e-1 | 1.68e-1 | 1.34e-1 | 1.24e-1 | 38 | -2.82e-3 | +2.26e-3 | -5.56e-4 | -4.55e-4 |
| 31 | 3.00e-1 | 4 | 1.32e-1 | 1.58e-1 | 1.42e-1 | 1.32e-1 | 42 | -2.22e-3 | +3.31e-3 | -3.13e-4 | -4.51e-4 |
| 32 | 3.00e-1 | 9 | 1.21e-1 | 1.69e-1 | 1.33e-1 | 1.32e-1 | 42 | -6.61e-3 | +3.14e-3 | -5.08e-4 | -3.74e-4 |
| 33 | 3.00e-1 | 5 | 1.26e-1 | 1.69e-1 | 1.38e-1 | 1.26e-1 | 37 | -5.03e-3 | +3.13e-3 | -9.51e-4 | -6.23e-4 |
| 34 | 3.00e-1 | 9 | 1.17e-1 | 1.65e-1 | 1.29e-1 | 1.21e-1 | 36 | -5.49e-3 | +3.53e-3 | -6.06e-4 | -5.53e-4 |
| 35 | 3.00e-1 | 5 | 1.29e-1 | 1.60e-1 | 1.38e-1 | 1.32e-1 | 40 | -4.42e-3 | +3.59e-3 | -3.37e-4 | -4.76e-4 |
| 36 | 3.00e-1 | 6 | 1.24e-1 | 1.64e-1 | 1.37e-1 | 1.24e-1 | 34 | -4.38e-3 | +2.71e-3 | -8.77e-4 | -6.83e-4 |
| 37 | 3.00e-1 | 8 | 1.18e-1 | 1.64e-1 | 1.27e-1 | 1.20e-1 | 35 | -7.19e-3 | +3.78e-3 | -7.73e-4 | -6.64e-4 |
| 38 | 3.00e-1 | 10 | 1.20e-1 | 1.63e-1 | 1.29e-1 | 1.27e-1 | 33 | -4.94e-3 | +4.26e-3 | -3.24e-4 | -3.65e-4 |
| 39 | 3.00e-1 | 5 | 1.14e-1 | 1.64e-1 | 1.29e-1 | 1.14e-1 | 30 | -7.36e-3 | +3.10e-3 | -1.76e-3 | -9.48e-4 |
| 40 | 3.00e-1 | 9 | 1.12e-1 | 1.57e-1 | 1.24e-1 | 1.32e-1 | 41 | -8.37e-3 | +4.64e-3 | -5.00e-4 | -4.84e-4 |
| 41 | 3.00e-1 | 5 | 1.42e-1 | 1.69e-1 | 1.52e-1 | 1.49e-1 | 50 | -1.81e-3 | +2.94e-3 | +6.33e-5 | -2.79e-4 |
| 42 | 3.00e-1 | 5 | 1.37e-1 | 1.79e-1 | 1.51e-1 | 1.37e-1 | 46 | -3.26e-3 | +2.00e-3 | -7.69e-4 | -4.98e-4 |
| 43 | 3.00e-1 | 6 | 1.33e-1 | 1.72e-1 | 1.44e-1 | 1.33e-1 | 40 | -2.95e-3 | +2.70e-3 | -5.01e-4 | -5.16e-4 |
| 44 | 3.00e-1 | 7 | 1.25e-1 | 1.68e-1 | 1.36e-1 | 1.32e-1 | 41 | -5.40e-3 | +2.96e-3 | -5.83e-4 | -4.87e-4 |
| 45 | 3.00e-1 | 6 | 1.27e-1 | 1.66e-1 | 1.37e-1 | 1.27e-1 | 38 | -4.89e-3 | +3.03e-3 | -6.66e-4 | -5.65e-4 |
| 46 | 3.00e-1 | 7 | 1.26e-1 | 1.66e-1 | 1.36e-1 | 1.33e-1 | 42 | -4.63e-3 | +3.47e-3 | -3.53e-4 | -4.16e-4 |
| 47 | 3.00e-1 | 10 | 1.21e-1 | 1.73e-1 | 1.32e-1 | 1.23e-1 | 36 | -5.07e-3 | +2.99e-3 | -6.62e-4 | -4.87e-4 |
| 48 | 3.00e-1 | 4 | 1.33e-1 | 1.69e-1 | 1.46e-1 | 1.33e-1 | 40 | -3.28e-3 | +3.52e-3 | -5.83e-4 | -5.63e-4 |
| 49 | 3.00e-1 | 6 | 1.25e-1 | 1.78e-1 | 1.38e-1 | 1.29e-1 | 37 | -7.14e-3 | +3.41e-3 | -1.05e-3 | -7.38e-4 |
| 50 | 3.00e-1 | 8 | 1.25e-1 | 1.67e-1 | 1.36e-1 | 1.26e-1 | 34 | -4.15e-3 | +3.14e-3 | -5.18e-4 | -6.10e-4 |
| 51 | 3.00e-1 | 5 | 1.27e-1 | 1.60e-1 | 1.36e-1 | 1.30e-1 | 37 | -4.33e-3 | +3.34e-3 | -5.19e-4 | -5.74e-4 |
| 52 | 3.00e-1 | 8 | 1.29e-1 | 1.71e-1 | 1.40e-1 | 1.29e-1 | 36 | -4.11e-3 | +3.30e-3 | -4.69e-4 | -5.25e-4 |
| 53 | 3.00e-1 | 5 | 1.25e-1 | 1.64e-1 | 1.37e-1 | 1.25e-1 | 35 | -5.47e-3 | +3.27e-3 | -9.53e-4 | -7.17e-4 |
| 54 | 3.00e-1 | 6 | 1.33e-1 | 1.65e-1 | 1.41e-1 | 1.36e-1 | 39 | -4.08e-3 | +3.52e-3 | -2.87e-4 | -5.25e-4 |
| 55 | 3.00e-1 | 8 | 1.18e-1 | 1.75e-1 | 1.32e-1 | 1.22e-1 | 34 | -8.23e-3 | +3.04e-3 | -1.19e-3 | -8.35e-4 |
| 56 | 3.00e-1 | 9 | 1.21e-1 | 1.63e-1 | 1.31e-1 | 1.34e-1 | 42 | -6.47e-3 | +3.80e-3 | -3.54e-4 | -4.23e-4 |
| 57 | 3.00e-1 | 4 | 1.34e-1 | 1.70e-1 | 1.47e-1 | 1.34e-1 | 40 | -4.47e-3 | +2.79e-3 | -8.41e-4 | -5.98e-4 |
| 58 | 3.00e-1 | 5 | 1.42e-1 | 1.72e-1 | 1.51e-1 | 1.47e-1 | 55 | -3.07e-3 | +3.02e-3 | -8.07e-5 | -3.97e-4 |
| 59 | 3.00e-1 | 5 | 1.41e-1 | 1.84e-1 | 1.55e-1 | 1.41e-1 | 46 | -3.68e-3 | +2.16e-3 | -6.91e-4 | -5.36e-4 |
| 60 | 3.00e-1 | 5 | 1.37e-1 | 1.72e-1 | 1.50e-1 | 1.51e-1 | 60 | -3.39e-3 | +2.42e-3 | -2.18e-4 | -3.93e-4 |
| 61 | 3.00e-1 | 9 | 1.23e-1 | 1.72e-1 | 1.33e-1 | 1.23e-1 | 34 | -6.26e-3 | +3.03e-3 | -9.60e-4 | -6.55e-4 |
| 62 | 3.00e-1 | 5 | 1.32e-1 | 1.63e-1 | 1.40e-1 | 1.37e-1 | 40 | -5.35e-3 | +3.73e-3 | -2.98e-4 | -5.07e-4 |
| 63 | 3.00e-1 | 8 | 1.34e-1 | 1.73e-1 | 1.41e-1 | 1.35e-1 | 42 | -5.00e-3 | +2.84e-3 | -4.31e-4 | -4.28e-4 |
| 64 | 3.00e-1 | 5 | 1.31e-1 | 1.66e-1 | 1.42e-1 | 1.33e-1 | 44 | -3.79e-3 | +2.72e-3 | -6.73e-4 | -5.36e-4 |
| 65 | 3.00e-1 | 8 | 1.30e-1 | 1.77e-1 | 1.40e-1 | 1.30e-1 | 36 | -3.92e-3 | +3.35e-3 | -5.04e-4 | -5.07e-4 |
| 66 | 3.00e-1 | 1 | 1.27e-1 | 1.27e-1 | 1.27e-1 | 1.27e-1 | 35 | -6.17e-4 | -6.17e-4 | -6.17e-4 | -5.18e-4 |
| 67 | 3.00e-1 | 1 | 2.33e-1 | 2.33e-1 | 2.33e-1 | 2.33e-1 | 318 | +1.90e-3 | +1.90e-3 | +1.90e-3 | -2.76e-4 |
| 68 | 3.00e-1 | 1 | 2.52e-1 | 2.52e-1 | 2.52e-1 | 2.52e-1 | 319 | +2.46e-4 | +2.46e-4 | +2.46e-4 | -2.24e-4 |
| 69 | 3.00e-1 | 1 | 2.48e-1 | 2.48e-1 | 2.48e-1 | 2.48e-1 | 283 | -6.05e-5 | -6.05e-5 | -6.05e-5 | -2.08e-4 |
| 70 | 3.00e-1 | 1 | 2.41e-1 | 2.41e-1 | 2.41e-1 | 2.41e-1 | 264 | -1.02e-4 | -1.02e-4 | -1.02e-4 | -1.97e-4 |
| 71 | 3.00e-1 | 1 | 2.43e-1 | 2.43e-1 | 2.43e-1 | 2.43e-1 | 282 | +3.49e-5 | +3.49e-5 | +3.49e-5 | -1.74e-4 |
| 72 | 3.00e-1 | 1 | 2.37e-1 | 2.37e-1 | 2.37e-1 | 2.37e-1 | 255 | -1.03e-4 | -1.03e-4 | -1.03e-4 | -1.67e-4 |
| 73 | 3.00e-1 | 1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 232 | -9.59e-5 | -9.59e-5 | -9.59e-5 | -1.60e-4 |
| 74 | 3.00e-1 | 1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 245 | -4.30e-5 | -4.30e-5 | -4.30e-5 | -1.48e-4 |
| 75 | 3.00e-1 | 1 | 2.33e-1 | 2.33e-1 | 2.33e-1 | 2.33e-1 | 253 | +6.52e-5 | +6.52e-5 | +6.52e-5 | -1.27e-4 |
| 76 | 3.00e-1 | 1 | 2.41e-1 | 2.41e-1 | 2.41e-1 | 2.41e-1 | 304 | +1.07e-4 | +1.07e-4 | +1.07e-4 | -1.03e-4 |
| 77 | 3.00e-1 | 1 | 2.37e-1 | 2.37e-1 | 2.37e-1 | 2.37e-1 | 284 | -5.32e-5 | -5.32e-5 | -5.32e-5 | -9.84e-5 |
| 78 | 3.00e-1 | 1 | 2.39e-1 | 2.39e-1 | 2.39e-1 | 2.39e-1 | 274 | +3.14e-5 | +3.14e-5 | +3.14e-5 | -8.54e-5 |
| 79 | 3.00e-1 | 1 | 2.37e-1 | 2.37e-1 | 2.37e-1 | 2.37e-1 | 289 | -2.76e-5 | -2.76e-5 | -2.76e-5 | -7.96e-5 |
| 80 | 3.00e-1 | 1 | 2.41e-1 | 2.41e-1 | 2.41e-1 | 2.41e-1 | 292 | +5.14e-5 | +5.14e-5 | +5.14e-5 | -6.65e-5 |
| 81 | 3.00e-1 | 1 | 2.40e-1 | 2.40e-1 | 2.40e-1 | 2.40e-1 | 278 | -9.16e-6 | -9.16e-6 | -9.16e-6 | -6.08e-5 |
| 82 | 3.00e-1 | 1 | 2.40e-1 | 2.40e-1 | 2.40e-1 | 2.40e-1 | 284 | -1.62e-6 | -1.62e-6 | -1.62e-6 | -5.49e-5 |
| 83 | 3.00e-1 | 1 | 2.40e-1 | 2.40e-1 | 2.40e-1 | 2.40e-1 | 283 | -9.56e-6 | -9.56e-6 | -9.56e-6 | -5.03e-5 |
| 84 | 3.00e-1 | 1 | 2.35e-1 | 2.35e-1 | 2.35e-1 | 2.35e-1 | 275 | -7.31e-5 | -7.31e-5 | -7.31e-5 | -5.26e-5 |
| 85 | 3.00e-1 | 1 | 2.38e-1 | 2.38e-1 | 2.38e-1 | 2.38e-1 | 279 | +5.44e-5 | +5.44e-5 | +5.44e-5 | -4.19e-5 |
| 86 | 3.00e-1 | 1 | 2.38e-1 | 2.38e-1 | 2.38e-1 | 2.38e-1 | 275 | -1.23e-5 | -1.23e-5 | -1.23e-5 | -3.90e-5 |
| 87 | 3.00e-1 | 1 | 2.42e-1 | 2.42e-1 | 2.42e-1 | 2.42e-1 | 297 | +5.76e-5 | +5.76e-5 | +5.76e-5 | -2.93e-5 |
| 88 | 3.00e-1 | 1 | 2.45e-1 | 2.45e-1 | 2.45e-1 | 2.45e-1 | 311 | +4.08e-5 | +4.08e-5 | +4.08e-5 | -2.23e-5 |
| 89 | 3.00e-1 | 1 | 2.42e-1 | 2.42e-1 | 2.42e-1 | 2.42e-1 | 273 | -4.26e-5 | -4.26e-5 | -4.26e-5 | -2.43e-5 |
| 90 | 3.00e-1 | 1 | 2.40e-1 | 2.40e-1 | 2.40e-1 | 2.40e-1 | 271 | -3.62e-5 | -3.62e-5 | -3.62e-5 | -2.55e-5 |
| 91 | 3.00e-1 | 1 | 2.36e-1 | 2.36e-1 | 2.36e-1 | 2.36e-1 | 267 | -5.25e-5 | -5.25e-5 | -5.25e-5 | -2.82e-5 |
| 92 | 3.00e-1 | 1 | 2.39e-1 | 2.39e-1 | 2.39e-1 | 2.39e-1 | 275 | +3.72e-5 | +3.72e-5 | +3.72e-5 | -2.17e-5 |
| 93 | 3.00e-1 | 1 | 2.34e-1 | 2.34e-1 | 2.34e-1 | 2.34e-1 | 249 | -7.60e-5 | -7.60e-5 | -7.60e-5 | -2.71e-5 |
| 94 | 3.00e-1 | 1 | 2.37e-1 | 2.37e-1 | 2.37e-1 | 2.37e-1 | 273 | +4.25e-5 | +4.25e-5 | +4.25e-5 | -2.01e-5 |
| 95 | 3.00e-1 | 1 | 2.49e-1 | 2.49e-1 | 2.49e-1 | 2.49e-1 | 325 | +1.46e-4 | +1.46e-4 | +1.46e-4 | -3.50e-6 |
| 96 | 3.00e-1 | 1 | 2.38e-1 | 2.38e-1 | 2.38e-1 | 2.38e-1 | 283 | -1.48e-4 | -1.48e-4 | -1.48e-4 | -1.79e-5 |
| 97 | 3.00e-1 | 1 | 2.47e-1 | 2.47e-1 | 2.47e-1 | 2.47e-1 | 302 | +1.25e-4 | +1.25e-4 | +1.25e-4 | -3.65e-6 |
| 98 | 3.00e-1 | 1 | 2.46e-1 | 2.46e-1 | 2.46e-1 | 2.46e-1 | 289 | -2.12e-5 | -2.12e-5 | -2.12e-5 | -5.41e-6 |
| 99 | 3.00e-1 | 1 | 2.36e-1 | 2.36e-1 | 2.36e-1 | 2.36e-1 | 253 | -1.58e-4 | -1.58e-4 | -1.58e-4 | -2.06e-5 |
| 100 | 3.00e-2 | 1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 238 | -2.89e-4 | -2.89e-4 | -2.89e-4 | -4.75e-5 |
| 101 | 3.00e-2 | 1 | 1.53e-1 | 1.53e-1 | 1.53e-1 | 1.53e-1 | 253 | -1.44e-3 | -1.44e-3 | -1.44e-3 | -1.87e-4 |
| 102 | 3.00e-2 | 1 | 1.08e-1 | 1.08e-1 | 1.08e-1 | 1.08e-1 | 260 | -1.35e-3 | -1.35e-3 | -1.35e-3 | -3.03e-4 |
| 103 | 3.00e-2 | 1 | 7.93e-2 | 7.93e-2 | 7.93e-2 | 7.93e-2 | 273 | -1.13e-3 | -1.13e-3 | -1.13e-3 | -3.86e-4 |
| 104 | 3.00e-2 | 1 | 5.81e-2 | 5.81e-2 | 5.81e-2 | 5.81e-2 | 268 | -1.16e-3 | -1.16e-3 | -1.16e-3 | -4.63e-4 |
| 105 | 3.00e-2 | 1 | 4.57e-2 | 4.57e-2 | 4.57e-2 | 4.57e-2 | 258 | -9.33e-4 | -9.33e-4 | -9.33e-4 | -5.10e-4 |
| 106 | 3.00e-2 | 2 | 3.58e-2 | 3.90e-2 | 3.74e-2 | 3.58e-2 | 228 | -6.10e-4 | -3.71e-4 | -4.90e-4 | -5.05e-4 |
| 108 | 3.00e-2 | 2 | 3.45e-2 | 3.76e-2 | 3.61e-2 | 3.45e-2 | 216 | -4.08e-4 | +1.54e-4 | -1.27e-4 | -4.36e-4 |
| 109 | 3.00e-2 | 1 | 3.36e-2 | 3.36e-2 | 3.36e-2 | 3.36e-2 | 236 | -1.12e-4 | -1.12e-4 | -1.12e-4 | -4.04e-4 |
| 110 | 3.00e-2 | 1 | 3.47e-2 | 3.47e-2 | 3.47e-2 | 3.47e-2 | 246 | +1.30e-4 | +1.30e-4 | +1.30e-4 | -3.50e-4 |
| 111 | 3.00e-2 | 1 | 3.59e-2 | 3.59e-2 | 3.59e-2 | 3.59e-2 | 257 | +1.39e-4 | +1.39e-4 | +1.39e-4 | -3.01e-4 |
| 112 | 3.00e-2 | 1 | 3.86e-2 | 3.86e-2 | 3.86e-2 | 3.86e-2 | 297 | +2.41e-4 | +2.41e-4 | +2.41e-4 | -2.47e-4 |
| 113 | 3.00e-2 | 1 | 3.82e-2 | 3.82e-2 | 3.82e-2 | 3.82e-2 | 254 | -3.91e-5 | -3.91e-5 | -3.91e-5 | -2.26e-4 |
| 114 | 3.00e-2 | 1 | 3.86e-2 | 3.86e-2 | 3.86e-2 | 3.86e-2 | 269 | +3.84e-5 | +3.84e-5 | +3.84e-5 | -2.00e-4 |
| 115 | 3.00e-2 | 1 | 3.95e-2 | 3.95e-2 | 3.95e-2 | 3.95e-2 | 256 | +8.73e-5 | +8.73e-5 | +8.73e-5 | -1.71e-4 |
| 116 | 3.00e-2 | 2 | 3.88e-2 | 3.96e-2 | 3.92e-2 | 3.88e-2 | 209 | -1.04e-4 | +1.58e-5 | -4.42e-5 | -1.48e-4 |
| 117 | 3.00e-2 | 1 | 4.14e-2 | 4.14e-2 | 4.14e-2 | 4.14e-2 | 250 | +2.62e-4 | +2.62e-4 | +2.62e-4 | -1.07e-4 |
| 118 | 3.00e-2 | 1 | 4.28e-2 | 4.28e-2 | 4.28e-2 | 4.28e-2 | 266 | +1.27e-4 | +1.27e-4 | +1.27e-4 | -8.32e-5 |
| 119 | 3.00e-2 | 1 | 4.12e-2 | 4.12e-2 | 4.12e-2 | 4.12e-2 | 218 | -1.82e-4 | -1.82e-4 | -1.82e-4 | -9.31e-5 |
| 120 | 3.00e-2 | 1 | 4.29e-2 | 4.29e-2 | 4.29e-2 | 4.29e-2 | 244 | +1.68e-4 | +1.68e-4 | +1.68e-4 | -6.70e-5 |
| 121 | 3.00e-2 | 2 | 4.15e-2 | 4.39e-2 | 4.27e-2 | 4.15e-2 | 183 | -3.05e-4 | +9.42e-5 | -1.06e-4 | -7.63e-5 |
| 122 | 3.00e-2 | 1 | 4.20e-2 | 4.20e-2 | 4.20e-2 | 4.20e-2 | 196 | +6.11e-5 | +6.11e-5 | +6.11e-5 | -6.26e-5 |
| 123 | 3.00e-2 | 1 | 4.34e-2 | 4.34e-2 | 4.34e-2 | 4.34e-2 | 223 | +1.49e-4 | +1.49e-4 | +1.49e-4 | -4.14e-5 |
| 124 | 3.00e-2 | 2 | 4.42e-2 | 4.48e-2 | 4.45e-2 | 4.48e-2 | 204 | +6.12e-5 | +7.87e-5 | +6.99e-5 | -2.03e-5 |
| 125 | 3.00e-2 | 1 | 4.95e-2 | 4.95e-2 | 4.95e-2 | 4.95e-2 | 260 | +3.83e-4 | +3.83e-4 | +3.83e-4 | +2.01e-5 |
| 126 | 3.00e-2 | 1 | 4.83e-2 | 4.83e-2 | 4.83e-2 | 4.83e-2 | 236 | -9.99e-5 | -9.99e-5 | -9.99e-5 | +8.06e-6 |
| 127 | 3.00e-2 | 1 | 4.73e-2 | 4.73e-2 | 4.73e-2 | 4.73e-2 | 215 | -9.85e-5 | -9.85e-5 | -9.85e-5 | -2.59e-6 |
| 128 | 3.00e-2 | 2 | 4.74e-2 | 4.94e-2 | 4.84e-2 | 4.74e-2 | 172 | -2.40e-4 | +1.83e-4 | -2.81e-5 | -9.56e-6 |
| 129 | 3.00e-2 | 1 | 4.74e-2 | 4.74e-2 | 4.74e-2 | 4.74e-2 | 198 | -8.17e-6 | -8.17e-6 | -8.17e-6 | -9.42e-6 |
| 130 | 3.00e-2 | 2 | 4.83e-2 | 4.94e-2 | 4.88e-2 | 4.83e-2 | 176 | -1.31e-4 | +2.01e-4 | +3.50e-5 | -2.63e-6 |
| 131 | 3.00e-2 | 1 | 4.85e-2 | 4.85e-2 | 4.85e-2 | 4.85e-2 | 206 | +2.39e-5 | +2.39e-5 | +2.39e-5 | +2.03e-8 |
| 132 | 3.00e-2 | 1 | 4.93e-2 | 4.93e-2 | 4.93e-2 | 4.93e-2 | 207 | +8.03e-5 | +8.03e-5 | +8.03e-5 | +8.05e-6 |
| 133 | 3.00e-2 | 1 | 5.23e-2 | 5.23e-2 | 5.23e-2 | 5.23e-2 | 225 | +2.61e-4 | +2.61e-4 | +2.61e-4 | +3.33e-5 |
| 134 | 3.00e-2 | 2 | 5.01e-2 | 5.23e-2 | 5.12e-2 | 5.01e-2 | 163 | -2.66e-4 | -7.45e-7 | -1.33e-4 | +3.29e-7 |
| 135 | 3.00e-2 | 2 | 4.95e-2 | 5.03e-2 | 4.99e-2 | 4.95e-2 | 163 | -9.40e-5 | +2.01e-5 | -3.70e-5 | -7.33e-6 |
| 136 | 3.00e-2 | 1 | 5.09e-2 | 5.09e-2 | 5.09e-2 | 5.09e-2 | 187 | +1.43e-4 | +1.43e-4 | +1.43e-4 | +7.65e-6 |
| 137 | 3.00e-2 | 1 | 5.61e-2 | 5.61e-2 | 5.61e-2 | 5.61e-2 | 220 | +4.47e-4 | +4.47e-4 | +4.47e-4 | +5.16e-5 |
| 138 | 3.00e-2 | 2 | 5.30e-2 | 5.58e-2 | 5.44e-2 | 5.30e-2 | 175 | -2.91e-4 | -2.87e-5 | -1.60e-4 | +1.01e-5 |
| 139 | 3.00e-2 | 1 | 5.56e-2 | 5.56e-2 | 5.56e-2 | 5.56e-2 | 197 | +2.45e-4 | +2.45e-4 | +2.45e-4 | +3.36e-5 |
| 140 | 3.00e-2 | 2 | 5.34e-2 | 5.55e-2 | 5.44e-2 | 5.34e-2 | 151 | -2.65e-4 | -6.74e-6 | -1.36e-4 | +1.04e-7 |
| 141 | 3.00e-2 | 2 | 5.32e-2 | 5.64e-2 | 5.48e-2 | 5.32e-2 | 158 | -3.76e-4 | +2.75e-4 | -5.05e-5 | -1.28e-5 |
| 142 | 3.00e-2 | 1 | 5.81e-2 | 5.81e-2 | 5.81e-2 | 5.81e-2 | 203 | +4.35e-4 | +4.35e-4 | +4.35e-4 | +3.20e-5 |
| 143 | 3.00e-2 | 2 | 5.47e-2 | 5.79e-2 | 5.63e-2 | 5.47e-2 | 143 | -3.96e-4 | -1.51e-5 | -2.06e-4 | -1.50e-5 |
| 144 | 3.00e-2 | 2 | 5.30e-2 | 5.75e-2 | 5.52e-2 | 5.30e-2 | 143 | -5.68e-4 | +2.62e-4 | -1.53e-4 | -4.54e-5 |
| 145 | 3.00e-2 | 1 | 5.69e-2 | 5.69e-2 | 5.69e-2 | 5.69e-2 | 189 | +3.75e-4 | +3.75e-4 | +3.75e-4 | -3.37e-6 |
| 146 | 3.00e-2 | 2 | 5.62e-2 | 5.68e-2 | 5.65e-2 | 5.62e-2 | 157 | -7.06e-5 | -8.34e-6 | -3.95e-5 | -1.05e-5 |
| 147 | 3.00e-2 | 1 | 5.73e-2 | 5.73e-2 | 5.73e-2 | 5.73e-2 | 177 | +1.13e-4 | +1.13e-4 | +1.13e-4 | +1.82e-6 |
| 148 | 3.00e-2 | 2 | 5.89e-2 | 5.90e-2 | 5.90e-2 | 5.90e-2 | 157 | +1.17e-5 | +1.57e-4 | +8.41e-5 | +1.67e-5 |
| 149 | 3.00e-2 | 2 | 5.85e-2 | 5.91e-2 | 5.88e-2 | 5.85e-2 | 157 | -6.98e-5 | +1.20e-5 | -2.89e-5 | +7.65e-6 |
| 150 | 3.00e-3 | 1 | 5.26e-2 | 5.26e-2 | 5.26e-2 | 5.26e-2 | 203 | -5.20e-4 | -5.20e-4 | -5.20e-4 | -4.51e-5 |
| 151 | 3.00e-3 | 2 | 2.49e-2 | 3.58e-2 | 3.04e-2 | 2.49e-2 | 157 | -2.30e-3 | -2.07e-3 | -2.18e-3 | -4.52e-4 |
| 152 | 3.00e-3 | 1 | 1.86e-2 | 1.86e-2 | 1.86e-2 | 1.86e-2 | 213 | -1.38e-3 | -1.38e-3 | -1.38e-3 | -5.44e-4 |
| 153 | 3.00e-3 | 3 | 8.04e-3 | 1.35e-2 | 1.04e-2 | 8.04e-3 | 133 | -2.40e-3 | -1.48e-3 | -1.87e-3 | -9.01e-4 |
| 154 | 3.00e-3 | 1 | 6.97e-3 | 6.97e-3 | 6.97e-3 | 6.97e-3 | 158 | -9.02e-4 | -9.02e-4 | -9.02e-4 | -9.01e-4 |
| 155 | 3.00e-3 | 3 | 5.65e-3 | 6.44e-3 | 6.02e-3 | 5.65e-3 | 133 | -5.57e-4 | -4.35e-4 | -4.91e-4 | -7.89e-4 |
| 156 | 3.00e-3 | 1 | 6.11e-3 | 6.11e-3 | 6.11e-3 | 6.11e-3 | 166 | +4.75e-4 | +4.75e-4 | +4.75e-4 | -6.63e-4 |
| 157 | 3.00e-3 | 2 | 5.72e-3 | 6.21e-3 | 5.96e-3 | 5.72e-3 | 125 | -6.52e-4 | +9.24e-5 | -2.80e-4 | -5.94e-4 |
| 158 | 3.00e-3 | 3 | 5.43e-3 | 6.13e-3 | 5.68e-3 | 5.48e-3 | 115 | -1.06e-3 | +4.11e-4 | -1.87e-4 | -4.86e-4 |
| 159 | 3.00e-3 | 1 | 5.73e-3 | 5.73e-3 | 5.73e-3 | 5.73e-3 | 147 | +3.00e-4 | +3.00e-4 | +3.00e-4 | -4.08e-4 |
| 160 | 3.00e-3 | 3 | 5.48e-3 | 6.24e-3 | 5.82e-3 | 5.48e-3 | 111 | -6.75e-4 | +4.70e-4 | -2.12e-4 | -3.63e-4 |
| 161 | 3.00e-3 | 1 | 5.60e-3 | 5.60e-3 | 5.60e-3 | 5.60e-3 | 129 | +1.71e-4 | +1.71e-4 | +1.71e-4 | -3.10e-4 |
| 162 | 3.00e-3 | 3 | 5.35e-3 | 5.94e-3 | 5.55e-3 | 5.35e-3 | 97 | -1.04e-3 | +3.97e-4 | -2.24e-4 | -2.90e-4 |
| 163 | 3.00e-3 | 2 | 5.37e-3 | 5.67e-3 | 5.52e-3 | 5.37e-3 | 105 | -5.09e-4 | +4.14e-4 | -4.79e-5 | -2.49e-4 |
| 164 | 3.00e-3 | 2 | 5.43e-3 | 5.82e-3 | 5.63e-3 | 5.43e-3 | 107 | -6.38e-4 | +5.46e-4 | -4.57e-5 | -2.16e-4 |
| 165 | 3.00e-3 | 3 | 5.14e-3 | 6.01e-3 | 5.56e-3 | 5.14e-3 | 98 | -8.54e-4 | +6.87e-4 | -3.05e-4 | -2.53e-4 |
| 166 | 3.00e-3 | 2 | 5.35e-3 | 5.68e-3 | 5.51e-3 | 5.35e-3 | 112 | -5.24e-4 | +7.20e-4 | +9.81e-5 | -1.93e-4 |
| 167 | 3.00e-3 | 4 | 5.29e-3 | 6.03e-3 | 5.60e-3 | 5.29e-3 | 91 | -6.81e-4 | +7.68e-4 | -1.37e-4 | -1.86e-4 |
| 168 | 3.00e-3 | 2 | 5.34e-3 | 5.59e-3 | 5.47e-3 | 5.34e-3 | 91 | -4.92e-4 | +4.06e-4 | -4.26e-5 | -1.64e-4 |
| 169 | 3.00e-3 | 2 | 5.31e-3 | 5.60e-3 | 5.46e-3 | 5.31e-3 | 97 | -5.41e-4 | +3.45e-4 | -9.81e-5 | -1.56e-4 |
| 170 | 3.00e-3 | 3 | 5.14e-3 | 5.69e-3 | 5.39e-3 | 5.14e-3 | 88 | -5.94e-4 | +5.30e-4 | -1.69e-4 | -1.68e-4 |
| 171 | 3.00e-3 | 3 | 4.89e-3 | 5.57e-3 | 5.19e-3 | 4.89e-3 | 77 | -1.11e-3 | +6.61e-4 | -3.48e-4 | -2.28e-4 |
| 172 | 3.00e-3 | 3 | 4.76e-3 | 5.99e-3 | 5.28e-3 | 4.76e-3 | 70 | -2.29e-3 | +1.48e-3 | -5.99e-4 | -3.51e-4 |
| 173 | 3.00e-3 | 4 | 4.56e-3 | 5.37e-3 | 4.84e-3 | 4.56e-3 | 70 | -1.53e-3 | +1.02e-3 | -3.78e-4 | -3.71e-4 |
| 174 | 3.00e-3 | 5 | 4.36e-3 | 5.51e-3 | 4.79e-3 | 4.36e-3 | 56 | -1.89e-3 | +1.65e-3 | -4.27e-4 | -4.22e-4 |
| 175 | 3.00e-3 | 3 | 4.21e-3 | 5.31e-3 | 4.74e-3 | 4.21e-3 | 50 | -2.18e-3 | +1.75e-3 | -8.10e-4 | -5.64e-4 |
| 176 | 3.00e-3 | 5 | 3.82e-3 | 5.11e-3 | 4.25e-3 | 3.82e-3 | 47 | -3.25e-3 | +1.93e-3 | -8.12e-4 | -6.81e-4 |
| 177 | 3.00e-3 | 5 | 3.82e-3 | 4.73e-3 | 4.11e-3 | 3.85e-3 | 45 | -3.38e-3 | +2.29e-3 | -4.33e-4 | -5.98e-4 |
| 178 | 3.00e-3 | 7 | 3.64e-3 | 4.71e-3 | 3.92e-3 | 3.72e-3 | 46 | -3.51e-3 | +2.23e-3 | -3.61e-4 | -4.63e-4 |
| 179 | 3.00e-3 | 4 | 3.70e-3 | 5.15e-3 | 4.25e-3 | 3.70e-3 | 39 | -4.21e-3 | +3.01e-3 | -1.17e-3 | -7.51e-4 |
| 180 | 3.00e-3 | 5 | 3.70e-3 | 4.44e-3 | 3.97e-3 | 4.02e-3 | 48 | -3.66e-3 | +2.24e-3 | -1.30e-4 | -4.77e-4 |
| 181 | 3.00e-3 | 6 | 3.40e-3 | 5.14e-3 | 4.02e-3 | 3.57e-3 | 42 | -3.73e-3 | +2.50e-3 | -9.58e-4 | -7.10e-4 |
| 182 | 3.00e-3 | 8 | 3.60e-3 | 4.39e-3 | 3.81e-3 | 3.60e-3 | 38 | -3.24e-3 | +2.74e-3 | -2.73e-4 | -4.87e-4 |
| 183 | 3.00e-3 | 4 | 3.67e-3 | 4.73e-3 | 4.00e-3 | 3.67e-3 | 44 | -6.11e-3 | +3.31e-3 | -9.58e-4 | -6.71e-4 |
| 184 | 3.00e-3 | 6 | 3.99e-3 | 4.86e-3 | 4.21e-3 | 3.99e-3 | 50 | -2.83e-3 | +3.46e-3 | -1.05e-4 | -4.39e-4 |
| 185 | 3.00e-3 | 7 | 3.76e-3 | 4.87e-3 | 4.06e-3 | 3.91e-3 | 43 | -2.81e-3 | +2.17e-3 | -3.61e-4 | -3.71e-4 |
| 186 | 3.00e-3 | 5 | 3.63e-3 | 4.52e-3 | 3.98e-3 | 3.63e-3 | 37 | -2.70e-3 | +1.75e-3 | -6.98e-4 | -5.37e-4 |
| 187 | 3.00e-3 | 7 | 3.16e-3 | 4.52e-3 | 3.54e-3 | 3.23e-3 | 35 | -6.11e-3 | +3.05e-3 | -1.11e-3 | -7.99e-4 |
| 188 | 3.00e-3 | 7 | 3.57e-3 | 4.22e-3 | 3.77e-3 | 3.79e-3 | 44 | -3.40e-3 | +3.70e-3 | +7.67e-5 | -3.47e-4 |
| 189 | 3.00e-3 | 5 | 3.70e-3 | 4.81e-3 | 4.07e-3 | 3.75e-3 | 40 | -2.86e-3 | +3.02e-3 | -5.56e-4 | -4.58e-4 |
| 190 | 3.00e-3 | 8 | 3.46e-3 | 4.70e-3 | 3.84e-3 | 3.90e-3 | 50 | -6.70e-3 | +2.99e-3 | -5.08e-4 | -3.89e-4 |
| 191 | 3.00e-3 | 4 | 3.77e-3 | 5.08e-3 | 4.21e-3 | 3.77e-3 | 34 | -5.03e-3 | +2.68e-3 | -1.19e-3 | -6.94e-4 |
| 192 | 3.00e-3 | 7 | 3.63e-3 | 4.80e-3 | 3.94e-3 | 3.70e-3 | 36 | -4.91e-3 | +3.01e-3 | -6.03e-4 | -6.33e-4 |
| 193 | 3.00e-3 | 5 | 3.85e-3 | 4.63e-3 | 4.25e-3 | 4.49e-3 | 56 | -5.56e-3 | +2.82e-3 | -3.32e-6 | -3.47e-4 |
| 194 | 3.00e-3 | 4 | 4.39e-3 | 5.34e-3 | 4.74e-3 | 4.39e-3 | 59 | -2.05e-3 | +1.71e-3 | -4.60e-4 | -4.08e-4 |
| 195 | 3.00e-3 | 5 | 4.42e-3 | 5.41e-3 | 4.71e-3 | 4.43e-3 | 54 | -2.25e-3 | +2.02e-3 | -3.06e-4 | -3.81e-4 |
| 196 | 3.00e-3 | 5 | 4.00e-3 | 5.52e-3 | 4.49e-3 | 4.00e-3 | 46 | -3.65e-3 | +2.16e-3 | -9.02e-4 | -6.11e-4 |
| 197 | 3.00e-3 | 7 | 3.89e-3 | 5.21e-3 | 4.26e-3 | 3.98e-3 | 43 | -3.14e-3 | +2.92e-3 | -4.26e-4 | -5.13e-4 |
| 198 | 3.00e-3 | 4 | 3.82e-3 | 5.25e-3 | 4.37e-3 | 3.82e-3 | 40 | -4.31e-3 | +3.15e-3 | -1.13e-3 | -7.72e-4 |
| 199 | 3.00e-3 | 5 | 4.00e-3 | 4.93e-3 | 4.26e-3 | 4.00e-3 | 41 | -3.61e-3 | +3.16e-3 | -3.93e-4 | -6.39e-4 |

