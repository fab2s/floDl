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
| cpu-async | 0.068892 | 0.9160 | +0.0035 | 1787.6 | 335 | 84.4 | 100% | 99% | 100% | 14.0 |

## Best Mode per Model

| Model | Best Eval | Mode | Fastest (within 2% of solo-0) | Mode |
|-------|-----------|------|-------------------------------|------|
| resnet-graph | 0.9160 | cpu-async | - | - |

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
| cpu-async | 2.0630 | 0.7892 | 0.6285 | 0.5634 | 0.5429 | 0.5274 | 0.5072 | 0.4941 | 0.4918 | 0.4754 | 0.2251 | 0.1892 | 0.1700 | 0.1625 | 0.1510 | 0.0871 | 0.0815 | 0.0782 | 0.0717 | 0.0689 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | cpu-async | 0 | cuda:0 | 0.4040 | 2.7 |
| resnet-graph | cpu-async | 1 | cuda:1 | 0.3005 | 3.7 |
| resnet-graph | cpu-async | 2 | cuda:2 | 0.2956 | 3.7 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) | GPU2 Peak (MB) | GPU2 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|---------------|---------------|
| resnet-graph | cpu-async | 358 | 357 | 388 | 386 | 386 | 380 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| cpu-async | gpu1 | 1785.3 | 2.2 | epoch-boundary(199) |
| cpu-async | gpu2 | 1785.3 | 2.2 | epoch-boundary(199) |
| cpu-async | gpu1 | 1472.2 | 2.0 | epoch-boundary(164) |
| cpu-async | gpu2 | 1472.2 | 1.9 | epoch-boundary(164) |
| cpu-async | gpu1 | 1275.7 | 1.1 | epoch-boundary(142) |
| cpu-async | gpu1 | 251.0 | 0.8 | epoch-boundary(27) |
| cpu-async | gpu2 | 251.0 | 0.8 | epoch-boundary(27) |
| cpu-async | gpu2 | 197.5 | 0.6 | epoch-boundary(21) |
| cpu-async | gpu1 | 517.7 | 0.6 | epoch-boundary(57) |
| cpu-async | gpu1 | 197.4 | 0.6 | epoch-boundary(21) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | cpu-async | gpu1 | 7.4s | 0.0s | 0.0s | 0.0s | 7.9s |
| resnet-graph | cpu-async | gpu2 | 5.6s | 0.0s | 0.0s | 0.0s | 6.1s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | cpu-async | 239 | 0 | 335 | 84.4 | 6607/9878 | 335 | 84.4 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | cpu-async | 194.2 | 10.9% |

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
| resnet-graph | cpu-async | 188 | 335 | 0 | 8.49e-3 | -1.18e-5 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | cpu-async | 0 | 335 | 1.29e-1 | 7.91e-2 | 6.51e-3 | 4.27e-1 | 27.2 | -1.18e-4 | 8.56e-4 |
| resnet-graph | cpu-async | 1 | 335 | 1.30e-1 | 8.17e-2 | 6.26e-3 | 5.06e-1 | 35.5 | -1.30e-4 | 1.17e-3 |
| resnet-graph | cpu-async | 2 | 335 | 1.31e-1 | 8.20e-2 | 6.10e-3 | 5.24e-1 | 37.3 | -1.58e-4 | 1.22e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | cpu-async | rank0 ↔ rank1 | +0.9883 |
| resnet-graph | cpu-async | rank0 ↔ rank2 | +0.9879 |
| resnet-graph | cpu-async | rank1 ↔ rank2 | +0.9868 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | cpu-async | 47 (3,4,6,7,8,13,14,15…143,146) | 0 (—) | — | 3,4,6,7,8,13,14,15…143,146 | — |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | cpu-async | 0e0 | 3 | 8 | 8 |
| resnet-graph | cpu-async | 0e0 | 5 | 4 | 4 |
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
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 231 | +0.110 |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | 48 | -0.174 |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | 52 | -0.535 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | cpu-async | 333 | -0.051 | 187 | +0.430 | +0.617 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | cpu-async | 334 | 3.36e1–7.95e1 | 6.42e1 | 3.59e-3 | 4.63e-3 | 5.05e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 233 | 33–77947 | +7.318e-6 | 0.344 | +7.776e-6 | 0.391 | 96 | +5.917e-6 | 0.378 | 33–1011 | +7.964e-4 | 0.606 |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 220 | 874–77947 | +8.747e-6 | 0.576 | +9.120e-6 | 0.617 | 95 | +6.205e-6 | 0.404 | 77–1011 | +9.145e-4 | 0.921 |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | 49 | 78346–116600 | +1.446e-6 | 0.003 | +1.712e-6 | 0.004 | 47 | +1.168e-6 | 0.002 | 399–1073 | -1.910e-3 | 0.453 |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | 53 | 117250–155790 | -1.500e-5 | 0.155 | -1.448e-5 | 0.144 | 45 | -1.204e-5 | 0.131 | 540–982 | -8.067e-4 | 0.034 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | +7.964e-4 | r0: +7.890e-4, r1: +8.194e-4, r2: +7.859e-4 | r0: 0.612, r1: 0.599, r2: 0.572 | 1.04× | ✓ |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +9.145e-4 | r0: +9.043e-4, r1: +9.398e-4, r2: +9.031e-4 | r0: 0.910, r1: 0.899, r2: 0.893 | 1.04× | ✓ |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | -1.910e-3 | r0: -1.937e-3, r1: -1.928e-3, r2: -1.865e-3 | r0: 0.461, r1: 0.445, r2: 0.445 | 1.04× | ✓ |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | -8.067e-4 | r0: -8.680e-4, r1: -8.122e-4, r2: -7.392e-4 | r0: 0.039, r1: 0.035, r2: 0.029 | 1.17× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | cpu-async | `█▇▇▇▇███████████████████▇▅▄▅▅▅▅▅▅▅▅▅▃▁▁▁▁▁▁▁▁▁▁` | `▁▆▆▆▆▇██████████████████▇▇▇▇████████▆▆▇▇███████` |

### Per-Epoch Detail: resnet-graph / cpu-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 13 | 1.63e-1 | 5.24e-1 | 2.70e-1 | 1.63e-1 | 31 | -2.17e-2 | +1.40e-2 | -4.70e-3 | -3.93e-3 |
| 1 | 3.00e-1 | 7 | 1.62e-1 | 2.14e-1 | 1.81e-1 | 1.62e-1 | 40 | -3.31e-3 | +3.83e-3 | -6.93e-4 | -2.03e-3 |
| 2 | 3.00e-1 | 7 | 1.21e-1 | 1.83e-1 | 1.40e-1 | 1.23e-1 | 37 | -4.32e-3 | +1.37e-3 | -1.43e-3 | -1.63e-3 |
| 3 | 3.00e-1 | 11 | 1.16e-1 | 1.62e-1 | 1.25e-1 | 1.18e-1 | 35 | -6.76e-3 | +3.81e-3 | -5.37e-4 | -7.97e-4 |
| 4 | 3.00e-1 | 3 | 1.33e-1 | 1.63e-1 | 1.44e-1 | 1.33e-1 | 38 | -4.84e-3 | +4.03e-3 | -4.34e-4 | -7.39e-4 |
| 5 | 3.00e-1 | 7 | 1.31e-1 | 1.66e-1 | 1.39e-1 | 1.31e-1 | 44 | -4.24e-3 | +2.84e-3 | -4.12e-4 | -5.67e-4 |
| 6 | 3.00e-1 | 7 | 1.29e-1 | 1.74e-1 | 1.41e-1 | 1.43e-1 | 56 | -5.44e-3 | +3.04e-3 | -3.77e-4 | -4.03e-4 |
| 7 | 3.00e-1 | 3 | 1.49e-1 | 1.71e-1 | 1.56e-1 | 1.50e-1 | 56 | -2.50e-3 | +1.82e-3 | -1.75e-4 | -3.55e-4 |
| 8 | 3.00e-1 | 5 | 1.37e-1 | 1.72e-1 | 1.50e-1 | 1.37e-1 | 45 | -2.98e-3 | +1.51e-3 | -6.20e-4 | -4.82e-4 |
| 9 | 3.00e-1 | 7 | 1.23e-1 | 1.65e-1 | 1.34e-1 | 1.23e-1 | 41 | -3.89e-3 | +2.50e-3 | -7.87e-4 | -6.34e-4 |
| 10 | 3.00e-1 | 6 | 1.29e-1 | 1.61e-1 | 1.39e-1 | 1.29e-1 | 42 | -3.11e-3 | +3.05e-3 | -3.35e-4 | -5.26e-4 |
| 11 | 3.00e-1 | 9 | 1.18e-1 | 1.64e-1 | 1.32e-1 | 1.18e-1 | 31 | -4.86e-3 | +2.98e-3 | -7.50e-4 | -7.07e-4 |
| 12 | 3.00e-1 | 4 | 1.21e-1 | 1.52e-1 | 1.31e-1 | 1.23e-1 | 39 | -5.20e-3 | +3.69e-3 | -6.12e-4 | -6.95e-4 |
| 13 | 3.00e-1 | 7 | 1.20e-1 | 1.66e-1 | 1.32e-1 | 1.20e-1 | 32 | -5.32e-3 | +3.70e-3 | -7.43e-4 | -7.20e-4 |
| 14 | 3.00e-1 | 8 | 1.16e-1 | 1.64e-1 | 1.26e-1 | 1.16e-1 | 31 | -7.45e-3 | +3.79e-3 | -9.65e-4 | -8.05e-4 |
| 15 | 3.00e-1 | 7 | 1.13e-1 | 1.59e-1 | 1.29e-1 | 1.30e-1 | 40 | -8.86e-3 | +4.06e-3 | -7.13e-4 | -6.48e-4 |
| 16 | 3.00e-1 | 6 | 1.30e-1 | 1.65e-1 | 1.38e-1 | 1.30e-1 | 40 | -4.07e-3 | +2.89e-3 | -4.55e-4 | -5.62e-4 |
| 17 | 3.00e-1 | 7 | 1.19e-1 | 1.66e-1 | 1.32e-1 | 1.21e-1 | 36 | -5.55e-3 | +2.96e-3 | -8.07e-4 | -6.75e-4 |
| 18 | 3.00e-1 | 6 | 1.23e-1 | 1.59e-1 | 1.33e-1 | 1.27e-1 | 38 | -5.44e-3 | +3.46e-3 | -5.31e-4 | -5.97e-4 |
| 19 | 3.00e-1 | 6 | 1.25e-1 | 1.69e-1 | 1.41e-1 | 1.25e-1 | 35 | -3.67e-3 | +3.03e-3 | -7.25e-4 | -7.16e-4 |
| 20 | 3.00e-1 | 6 | 1.25e-1 | 1.52e-1 | 1.32e-1 | 1.33e-1 | 42 | -4.96e-3 | +2.89e-3 | -1.97e-4 | -4.41e-4 |
| 21 | 3.00e-1 | 2 | 1.33e-1 | 2.16e-1 | 1.74e-1 | 2.16e-1 | 247 | +1.18e-6 | +1.97e-3 | +9.84e-4 | -1.61e-4 |
| 22 | 3.00e-1 | 1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 242 | +1.19e-4 | +1.19e-4 | +1.19e-4 | -1.33e-4 |
| 23 | 3.00e-1 | 1 | 2.35e-1 | 2.35e-1 | 2.35e-1 | 2.35e-1 | 288 | +1.90e-4 | +1.90e-4 | +1.90e-4 | -1.00e-4 |
| 24 | 3.00e-1 | 1 | 2.35e-1 | 2.35e-1 | 2.35e-1 | 2.35e-1 | 280 | -4.39e-6 | -4.39e-6 | -4.39e-6 | -9.08e-5 |
| 25 | 3.00e-1 | 1 | 2.33e-1 | 2.33e-1 | 2.33e-1 | 2.33e-1 | 275 | -1.80e-5 | -1.80e-5 | -1.80e-5 | -8.35e-5 |
| 26 | 3.00e-1 | 1 | 2.36e-1 | 2.36e-1 | 2.36e-1 | 2.36e-1 | 299 | +3.23e-5 | +3.23e-5 | +3.23e-5 | -7.20e-5 |
| 27 | 3.00e-1 | 1 | 2.39e-1 | 2.39e-1 | 2.39e-1 | 2.39e-1 | 272 | +4.78e-5 | +4.78e-5 | +4.78e-5 | -6.00e-5 |
| 29 | 3.00e-1 | 2 | 2.38e-1 | 2.56e-1 | 2.47e-1 | 2.38e-1 | 272 | -2.66e-4 | +1.80e-4 | -4.33e-5 | -5.90e-5 |
| 31 | 3.00e-1 | 1 | 2.49e-1 | 2.49e-1 | 2.49e-1 | 2.49e-1 | 363 | +1.22e-4 | +1.22e-4 | +1.22e-4 | -4.10e-5 |
| 32 | 3.00e-1 | 1 | 2.38e-1 | 2.38e-1 | 2.38e-1 | 2.38e-1 | 296 | -1.59e-4 | -1.59e-4 | -1.59e-4 | -5.28e-5 |
| 33 | 3.00e-1 | 1 | 2.35e-1 | 2.35e-1 | 2.35e-1 | 2.35e-1 | 281 | -3.57e-5 | -3.57e-5 | -3.57e-5 | -5.11e-5 |
| 34 | 3.00e-1 | 1 | 2.37e-1 | 2.37e-1 | 2.37e-1 | 2.37e-1 | 302 | +2.12e-5 | +2.12e-5 | +2.12e-5 | -4.39e-5 |
| 35 | 3.00e-1 | 1 | 2.36e-1 | 2.36e-1 | 2.36e-1 | 2.36e-1 | 307 | -7.29e-6 | -7.29e-6 | -7.29e-6 | -4.02e-5 |
| 36 | 3.00e-1 | 1 | 2.31e-1 | 2.31e-1 | 2.31e-1 | 2.31e-1 | 281 | -7.53e-5 | -7.53e-5 | -7.53e-5 | -4.37e-5 |
| 37 | 3.00e-1 | 1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 268 | +1.45e-5 | +1.45e-5 | +1.45e-5 | -3.79e-5 |
| 38 | 3.00e-1 | 1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 252 | -1.16e-4 | -1.16e-4 | -1.16e-4 | -4.57e-5 |
| 39 | 3.00e-1 | 1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 284 | +1.08e-4 | +1.08e-4 | +1.08e-4 | -3.04e-5 |
| 40 | 3.00e-1 | 1 | 2.28e-1 | 2.28e-1 | 2.28e-1 | 2.28e-1 | 271 | -6.85e-5 | -6.85e-5 | -6.85e-5 | -3.42e-5 |
| 41 | 3.00e-1 | 1 | 2.36e-1 | 2.36e-1 | 2.36e-1 | 2.36e-1 | 296 | +1.16e-4 | +1.16e-4 | +1.16e-4 | -1.92e-5 |
| 42 | 3.00e-1 | 1 | 2.41e-1 | 2.41e-1 | 2.41e-1 | 2.41e-1 | 305 | +6.95e-5 | +6.95e-5 | +6.95e-5 | -1.03e-5 |
| 43 | 3.00e-1 | 1 | 2.33e-1 | 2.33e-1 | 2.33e-1 | 2.33e-1 | 261 | -1.41e-4 | -1.41e-4 | -1.41e-4 | -2.34e-5 |
| 44 | 3.00e-1 | 1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 274 | -1.06e-5 | -1.06e-5 | -1.06e-5 | -2.21e-5 |
| 45 | 3.00e-1 | 1 | 2.38e-1 | 2.38e-1 | 2.38e-1 | 2.38e-1 | 287 | +8.50e-5 | +8.50e-5 | +8.50e-5 | -1.14e-5 |
| 46 | 3.00e-1 | 1 | 2.31e-1 | 2.31e-1 | 2.31e-1 | 2.31e-1 | 264 | -1.03e-4 | -1.03e-4 | -1.03e-4 | -2.06e-5 |
| 47 | 3.00e-1 | 1 | 2.31e-1 | 2.31e-1 | 2.31e-1 | 2.31e-1 | 262 | +3.62e-6 | +3.62e-6 | +3.62e-6 | -1.82e-5 |
| 48 | 3.00e-1 | 1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 256 | -3.87e-5 | -3.87e-5 | -3.87e-5 | -2.02e-5 |
| 49 | 3.00e-1 | 1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 236 | -8.50e-5 | -8.50e-5 | -8.50e-5 | -2.67e-5 |
| 50 | 3.00e-1 | 1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 258 | +5.02e-5 | +5.02e-5 | +5.02e-5 | -1.90e-5 |
| 51 | 3.00e-1 | 1 | 2.31e-1 | 2.31e-1 | 2.31e-1 | 2.31e-1 | 270 | +5.99e-5 | +5.99e-5 | +5.99e-5 | -1.11e-5 |
| 52 | 3.00e-1 | 1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 248 | -8.98e-5 | -8.98e-5 | -8.98e-5 | -1.90e-5 |
| 53 | 3.00e-1 | 2 | 2.27e-1 | 2.31e-1 | 2.29e-1 | 2.27e-1 | 236 | -6.62e-5 | +7.47e-5 | +4.25e-6 | -1.53e-5 |
| 55 | 3.00e-1 | 2 | 2.36e-1 | 2.43e-1 | 2.40e-1 | 2.36e-1 | 236 | -1.16e-4 | +2.29e-4 | +5.67e-5 | -3.33e-6 |
| 57 | 3.00e-1 | 2 | 2.33e-1 | 2.47e-1 | 2.40e-1 | 2.33e-1 | 236 | -2.40e-4 | +1.36e-4 | -5.22e-5 | -1.45e-5 |
| 58 | 3.00e-1 | 1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 265 | -2.11e-5 | -2.11e-5 | -2.11e-5 | -1.52e-5 |
| 59 | 3.00e-1 | 1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 258 | -5.41e-5 | -5.41e-5 | -5.41e-5 | -1.90e-5 |
| 60 | 3.00e-1 | 1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 267 | +4.73e-5 | +4.73e-5 | +4.73e-5 | -1.24e-5 |
| 61 | 3.00e-1 | 1 | 2.42e-1 | 2.42e-1 | 2.42e-1 | 2.42e-1 | 306 | +1.39e-4 | +1.39e-4 | +1.39e-4 | +2.71e-6 |
| 62 | 3.00e-1 | 1 | 2.34e-1 | 2.34e-1 | 2.34e-1 | 2.34e-1 | 276 | -1.22e-4 | -1.22e-4 | -1.22e-4 | -9.80e-6 |
| 63 | 3.00e-1 | 1 | 2.33e-1 | 2.33e-1 | 2.33e-1 | 2.33e-1 | 262 | -8.82e-6 | -8.82e-6 | -8.82e-6 | -9.70e-6 |
| 64 | 3.00e-1 | 1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 265 | -6.17e-5 | -6.17e-5 | -6.17e-5 | -1.49e-5 |
| 65 | 3.00e-1 | 1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 243 | -3.78e-5 | -3.78e-5 | -3.78e-5 | -1.72e-5 |
| 66 | 3.00e-1 | 1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 237 | -1.54e-5 | -1.54e-5 | -1.54e-5 | -1.70e-5 |
| 67 | 3.00e-1 | 2 | 2.26e-1 | 2.31e-1 | 2.29e-1 | 2.26e-1 | 219 | -1.13e-4 | +8.80e-5 | -1.27e-5 | -1.72e-5 |
| 68 | 3.00e-1 | 1 | 2.28e-1 | 2.28e-1 | 2.28e-1 | 2.28e-1 | 242 | +3.49e-5 | +3.49e-5 | +3.49e-5 | -1.20e-5 |
| 69 | 3.00e-1 | 1 | 2.35e-1 | 2.35e-1 | 2.35e-1 | 2.35e-1 | 261 | +1.19e-4 | +1.19e-4 | +1.19e-4 | +1.11e-6 |
| 70 | 3.00e-1 | 1 | 2.31e-1 | 2.31e-1 | 2.31e-1 | 2.31e-1 | 243 | -6.55e-5 | -6.55e-5 | -6.55e-5 | -5.55e-6 |
| 71 | 3.00e-1 | 1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 240 | -1.64e-5 | -1.64e-5 | -1.64e-5 | -6.63e-6 |
| 72 | 3.00e-1 | 1 | 2.34e-1 | 2.34e-1 | 2.34e-1 | 2.34e-1 | 246 | +7.06e-5 | +7.06e-5 | +7.06e-5 | +1.08e-6 |
| 73 | 3.00e-1 | 1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 244 | -1.00e-4 | -1.00e-4 | -1.00e-4 | -9.07e-6 |
| 74 | 3.00e-1 | 2 | 2.16e-1 | 2.23e-1 | 2.19e-1 | 2.16e-1 | 190 | -1.81e-4 | -1.10e-4 | -1.45e-4 | -3.53e-5 |
| 75 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 198 | +2.92e-5 | +2.92e-5 | +2.92e-5 | -2.88e-5 |
| 76 | 3.00e-1 | 1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 224 | +1.04e-4 | +1.04e-4 | +1.04e-4 | -1.55e-5 |
| 77 | 3.00e-1 | 2 | 2.26e-1 | 2.28e-1 | 2.27e-1 | 2.26e-1 | 201 | -3.29e-5 | +1.10e-4 | +3.85e-5 | -5.98e-6 |
| 78 | 3.00e-1 | 1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 241 | +9.48e-5 | +9.48e-5 | +9.48e-5 | +4.09e-6 |
| 79 | 3.00e-1 | 1 | 2.35e-1 | 2.35e-1 | 2.35e-1 | 2.35e-1 | 255 | +5.80e-5 | +5.80e-5 | +5.80e-5 | +9.48e-6 |
| 80 | 3.00e-1 | 1 | 2.31e-1 | 2.31e-1 | 2.31e-1 | 2.31e-1 | 238 | -8.06e-5 | -8.06e-5 | -8.06e-5 | +4.73e-7 |
| 81 | 3.00e-1 | 2 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 185 | -2.31e-4 | +4.18e-6 | -1.14e-4 | -2.00e-5 |
| 82 | 3.00e-1 | 1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 228 | +1.08e-4 | +1.08e-4 | +1.08e-4 | -7.24e-6 |
| 83 | 3.00e-1 | 1 | 2.31e-1 | 2.31e-1 | 2.31e-1 | 2.31e-1 | 243 | +9.51e-5 | +9.51e-5 | +9.51e-5 | +2.99e-6 |
| 84 | 3.00e-1 | 2 | 2.16e-1 | 2.19e-1 | 2.17e-1 | 2.16e-1 | 185 | -2.53e-4 | -9.24e-5 | -1.73e-4 | -2.96e-5 |
| 85 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 189 | -1.96e-6 | -1.96e-6 | -1.96e-6 | -2.68e-5 |
| 86 | 3.00e-1 | 1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 216 | +1.69e-4 | +1.69e-4 | +1.69e-4 | -7.21e-6 |
| 87 | 3.00e-1 | 2 | 2.17e-1 | 2.23e-1 | 2.20e-1 | 2.17e-1 | 185 | -1.69e-4 | -6.08e-7 | -8.47e-5 | -2.28e-5 |
| 88 | 3.00e-1 | 1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 216 | +1.47e-4 | +1.47e-4 | +1.47e-4 | -5.81e-6 |
| 89 | 3.00e-1 | 2 | 2.15e-1 | 2.35e-1 | 2.25e-1 | 2.15e-1 | 173 | -5.17e-4 | +2.16e-4 | -1.51e-4 | -3.70e-5 |
| 90 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 203 | +8.33e-5 | +8.33e-5 | +8.33e-5 | -2.49e-5 |
| 91 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 208 | +2.15e-5 | +2.15e-5 | +2.15e-5 | -2.03e-5 |
| 92 | 3.00e-1 | 2 | 2.11e-1 | 2.19e-1 | 2.15e-1 | 2.11e-1 | 173 | -2.25e-4 | -1.28e-5 | -1.19e-4 | -4.01e-5 |
| 93 | 3.00e-1 | 2 | 2.11e-1 | 2.17e-1 | 2.14e-1 | 2.11e-1 | 173 | -1.55e-4 | +1.43e-4 | -5.68e-6 | -3.50e-5 |
| 94 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 193 | +1.69e-4 | +1.69e-4 | +1.69e-4 | -1.47e-5 |
| 95 | 3.00e-1 | 2 | 2.15e-1 | 2.24e-1 | 2.19e-1 | 2.15e-1 | 181 | -2.43e-4 | +1.31e-4 | -5.63e-5 | -2.45e-5 |
| 96 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 199 | +5.24e-5 | +5.24e-5 | +5.24e-5 | -1.68e-5 |
| 97 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 204 | +7.84e-5 | +7.84e-5 | +7.84e-5 | -7.25e-6 |
| 98 | 3.00e-1 | 1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 221 | +2.29e-4 | +2.29e-4 | +2.29e-4 | +1.63e-5 |
| 99 | 3.00e-1 | 2 | 2.09e-1 | 2.25e-1 | 2.17e-1 | 2.09e-1 | 174 | -4.27e-4 | -1.47e-4 | -2.87e-4 | -4.27e-5 |
| 100 | 3.00e-2 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 143 | +5.62e-5 | +5.62e-5 | +5.62e-5 | -3.28e-5 |
| 101 | 3.00e-2 | 1 | 1.41e-1 | 1.41e-1 | 1.41e-1 | 1.41e-1 | 298 | -1.35e-3 | -1.35e-3 | -1.35e-3 | -1.65e-4 |
| 102 | 3.00e-2 | 1 | 9.96e-2 | 9.96e-2 | 9.96e-2 | 9.96e-2 | 292 | -1.18e-3 | -1.18e-3 | -1.18e-3 | -2.67e-4 |
| 103 | 3.00e-2 | 1 | 7.27e-2 | 7.27e-2 | 7.27e-2 | 7.27e-2 | 300 | -1.05e-3 | -1.05e-3 | -1.05e-3 | -3.45e-4 |
| 104 | 3.00e-2 | 1 | 5.48e-2 | 5.48e-2 | 5.48e-2 | 5.48e-2 | 311 | -9.06e-4 | -9.06e-4 | -9.06e-4 | -4.01e-4 |
| 105 | 3.00e-2 | 1 | 4.59e-2 | 4.59e-2 | 4.59e-2 | 4.59e-2 | 331 | -5.37e-4 | -5.37e-4 | -5.37e-4 | -4.15e-4 |
| 106 | 3.00e-2 | 1 | 4.18e-2 | 4.18e-2 | 4.18e-2 | 4.18e-2 | 347 | -2.69e-4 | -2.69e-4 | -2.69e-4 | -4.00e-4 |
| 108 | 3.00e-2 | 1 | 4.24e-2 | 4.24e-2 | 4.24e-2 | 4.24e-2 | 408 | +3.33e-5 | +3.33e-5 | +3.33e-5 | -3.57e-4 |
| 109 | 3.00e-2 | 1 | 4.00e-2 | 4.00e-2 | 4.00e-2 | 4.00e-2 | 313 | -1.86e-4 | -1.86e-4 | -1.86e-4 | -3.40e-4 |
| 110 | 3.00e-2 | 1 | 3.97e-2 | 3.97e-2 | 3.97e-2 | 3.97e-2 | 308 | -2.38e-5 | -2.38e-5 | -2.38e-5 | -3.08e-4 |
| 111 | 3.00e-2 | 1 | 4.15e-2 | 4.15e-2 | 4.15e-2 | 4.15e-2 | 342 | +1.32e-4 | +1.32e-4 | +1.32e-4 | -2.64e-4 |
| 112 | 3.00e-2 | 1 | 4.16e-2 | 4.16e-2 | 4.16e-2 | 4.16e-2 | 312 | +9.03e-6 | +9.03e-6 | +9.03e-6 | -2.37e-4 |
| 114 | 3.00e-2 | 2 | 4.23e-2 | 4.35e-2 | 4.29e-2 | 4.23e-2 | 276 | -9.68e-5 | +1.28e-4 | +1.55e-5 | -1.90e-4 |
| 116 | 3.00e-2 | 1 | 4.61e-2 | 4.61e-2 | 4.61e-2 | 4.61e-2 | 350 | +2.46e-4 | +2.46e-4 | +2.46e-4 | -1.46e-4 |
| 117 | 3.00e-2 | 1 | 4.39e-2 | 4.39e-2 | 4.39e-2 | 4.39e-2 | 280 | -1.79e-4 | -1.79e-4 | -1.79e-4 | -1.50e-4 |
| 118 | 3.00e-2 | 1 | 4.45e-2 | 4.45e-2 | 4.45e-2 | 4.45e-2 | 271 | +5.81e-5 | +5.81e-5 | +5.81e-5 | -1.29e-4 |
| 119 | 3.00e-2 | 1 | 4.57e-2 | 4.57e-2 | 4.57e-2 | 4.57e-2 | 278 | +8.97e-5 | +8.97e-5 | +8.97e-5 | -1.07e-4 |
| 120 | 3.00e-2 | 1 | 4.58e-2 | 4.58e-2 | 4.58e-2 | 4.58e-2 | 260 | +1.16e-5 | +1.16e-5 | +1.16e-5 | -9.52e-5 |
| 121 | 3.00e-2 | 1 | 4.71e-2 | 4.71e-2 | 4.71e-2 | 4.71e-2 | 283 | +1.01e-4 | +1.01e-4 | +1.01e-4 | -7.55e-5 |
| 122 | 3.00e-2 | 1 | 5.02e-2 | 5.02e-2 | 5.02e-2 | 5.02e-2 | 302 | +2.06e-4 | +2.06e-4 | +2.06e-4 | -4.74e-5 |
| 123 | 3.00e-2 | 1 | 4.98e-2 | 4.98e-2 | 4.98e-2 | 4.98e-2 | 284 | -2.80e-5 | -2.80e-5 | -2.80e-5 | -4.55e-5 |
| 124 | 3.00e-2 | 1 | 5.10e-2 | 5.10e-2 | 5.10e-2 | 5.10e-2 | 288 | +8.73e-5 | +8.73e-5 | +8.73e-5 | -3.22e-5 |
| 125 | 3.00e-2 | 1 | 5.13e-2 | 5.13e-2 | 5.13e-2 | 5.13e-2 | 279 | +2.02e-5 | +2.02e-5 | +2.02e-5 | -2.69e-5 |
| 126 | 3.00e-2 | 1 | 5.28e-2 | 5.28e-2 | 5.28e-2 | 5.28e-2 | 289 | +9.80e-5 | +9.80e-5 | +9.80e-5 | -1.45e-5 |
| 127 | 3.00e-2 | 1 | 5.42e-2 | 5.42e-2 | 5.42e-2 | 5.42e-2 | 300 | +8.46e-5 | +8.46e-5 | +8.46e-5 | -4.55e-6 |
| 128 | 3.00e-2 | 1 | 5.35e-2 | 5.35e-2 | 5.35e-2 | 5.35e-2 | 278 | -4.70e-5 | -4.70e-5 | -4.70e-5 | -8.80e-6 |
| 129 | 3.00e-2 | 1 | 5.57e-2 | 5.57e-2 | 5.57e-2 | 5.57e-2 | 286 | +1.44e-4 | +1.44e-4 | +1.44e-4 | +6.44e-6 |
| 130 | 3.00e-2 | 1 | 5.62e-2 | 5.62e-2 | 5.62e-2 | 5.62e-2 | 281 | +2.96e-5 | +2.96e-5 | +2.96e-5 | +8.75e-6 |
| 131 | 3.00e-2 | 1 | 5.57e-2 | 5.57e-2 | 5.57e-2 | 5.57e-2 | 268 | -3.12e-5 | -3.12e-5 | -3.12e-5 | +4.76e-6 |
| 132 | 3.00e-2 | 1 | 5.75e-2 | 5.75e-2 | 5.75e-2 | 5.75e-2 | 251 | +1.24e-4 | +1.24e-4 | +1.24e-4 | +1.67e-5 |
| 133 | 3.00e-2 | 1 | 5.93e-2 | 5.93e-2 | 5.93e-2 | 5.93e-2 | 291 | +1.09e-4 | +1.09e-4 | +1.09e-4 | +2.59e-5 |
| 134 | 3.00e-2 | 1 | 5.91e-2 | 5.91e-2 | 5.91e-2 | 5.91e-2 | 281 | -9.50e-6 | -9.50e-6 | -9.50e-6 | +2.24e-5 |
| 135 | 3.00e-2 | 1 | 5.94e-2 | 5.94e-2 | 5.94e-2 | 5.94e-2 | 262 | +1.72e-5 | +1.72e-5 | +1.72e-5 | +2.19e-5 |
| 136 | 3.00e-2 | 1 | 6.06e-2 | 6.06e-2 | 6.06e-2 | 6.06e-2 | 271 | +7.08e-5 | +7.08e-5 | +7.08e-5 | +2.68e-5 |
| 137 | 3.00e-2 | 1 | 6.26e-2 | 6.26e-2 | 6.26e-2 | 6.26e-2 | 280 | +1.21e-4 | +1.21e-4 | +1.21e-4 | +3.62e-5 |
| 138 | 3.00e-2 | 1 | 6.28e-2 | 6.28e-2 | 6.28e-2 | 6.28e-2 | 260 | +8.07e-6 | +8.07e-6 | +8.07e-6 | +3.34e-5 |
| 139 | 3.00e-2 | 1 | 6.04e-2 | 6.04e-2 | 6.04e-2 | 6.04e-2 | 238 | -1.65e-4 | -1.65e-4 | -1.65e-4 | +1.35e-5 |
| 140 | 3.00e-2 | 1 | 6.27e-2 | 6.27e-2 | 6.27e-2 | 6.27e-2 | 267 | +1.41e-4 | +1.41e-4 | +1.41e-4 | +2.63e-5 |
| 141 | 3.00e-2 | 1 | 6.13e-2 | 6.13e-2 | 6.13e-2 | 6.13e-2 | 242 | -8.99e-5 | -8.99e-5 | -8.99e-5 | +1.46e-5 |
| 142 | 3.00e-2 | 2 | 6.18e-2 | 6.34e-2 | 6.26e-2 | 6.34e-2 | 224 | +3.34e-5 | +1.15e-4 | +7.40e-5 | +2.63e-5 |
| 143 | 3.00e-2 | 1 | 6.69e-2 | 6.69e-2 | 6.69e-2 | 6.69e-2 | 273 | +1.97e-4 | +1.97e-4 | +1.97e-4 | +4.34e-5 |
| 144 | 3.00e-2 | 1 | 6.59e-2 | 6.59e-2 | 6.59e-2 | 6.59e-2 | 258 | -6.02e-5 | -6.02e-5 | -6.02e-5 | +3.30e-5 |
| 145 | 3.00e-2 | 1 | 6.74e-2 | 6.74e-2 | 6.74e-2 | 6.74e-2 | 271 | +8.53e-5 | +8.53e-5 | +8.53e-5 | +3.82e-5 |
| 146 | 3.00e-2 | 1 | 6.77e-2 | 6.77e-2 | 6.77e-2 | 6.77e-2 | 262 | +1.36e-5 | +1.36e-5 | +1.36e-5 | +3.58e-5 |
| 147 | 3.00e-2 | 1 | 6.65e-2 | 6.65e-2 | 6.65e-2 | 6.65e-2 | 233 | -7.67e-5 | -7.67e-5 | -7.67e-5 | +2.45e-5 |
| 148 | 3.00e-2 | 1 | 6.55e-2 | 6.55e-2 | 6.55e-2 | 6.55e-2 | 234 | -6.41e-5 | -6.41e-5 | -6.41e-5 | +1.57e-5 |
| 149 | 3.00e-2 | 1 | 6.71e-2 | 6.71e-2 | 6.71e-2 | 6.71e-2 | 229 | +1.05e-4 | +1.05e-4 | +1.05e-4 | +2.46e-5 |
| 150 | 3.00e-3 | 2 | 4.71e-2 | 6.67e-2 | 5.69e-2 | 4.71e-2 | 211 | -1.65e-3 | -2.31e-5 | -8.37e-4 | -1.47e-4 |
| 151 | 3.00e-3 | 1 | 3.22e-2 | 3.22e-2 | 3.22e-2 | 3.22e-2 | 227 | -1.68e-3 | -1.68e-3 | -1.68e-3 | -3.00e-4 |
| 152 | 3.00e-3 | 1 | 2.31e-2 | 2.31e-2 | 2.31e-2 | 2.31e-2 | 251 | -1.31e-3 | -1.31e-3 | -1.31e-3 | -4.02e-4 |
| 153 | 3.00e-3 | 1 | 1.70e-2 | 1.70e-2 | 1.70e-2 | 1.70e-2 | 268 | -1.14e-3 | -1.14e-3 | -1.14e-3 | -4.76e-4 |
| 154 | 3.00e-3 | 1 | 1.28e-2 | 1.28e-2 | 1.28e-2 | 1.28e-2 | 244 | -1.16e-3 | -1.16e-3 | -1.16e-3 | -5.44e-4 |
| 155 | 3.00e-3 | 2 | 8.43e-3 | 9.84e-3 | 9.14e-3 | 8.43e-3 | 197 | -1.26e-3 | -7.86e-4 | -1.02e-3 | -6.33e-4 |
| 156 | 3.00e-3 | 1 | 7.65e-3 | 7.65e-3 | 7.65e-3 | 7.65e-3 | 237 | -4.12e-4 | -4.12e-4 | -4.12e-4 | -6.11e-4 |
| 157 | 3.00e-3 | 1 | 7.02e-3 | 7.02e-3 | 7.02e-3 | 7.02e-3 | 222 | -3.86e-4 | -3.86e-4 | -3.86e-4 | -5.88e-4 |
| 158 | 3.00e-3 | 1 | 7.02e-3 | 7.02e-3 | 7.02e-3 | 7.02e-3 | 207 | +2.59e-6 | +2.59e-6 | +2.59e-6 | -5.29e-4 |
| 159 | 3.00e-3 | 2 | 7.09e-3 | 7.39e-3 | 7.24e-3 | 7.09e-3 | 207 | -2.02e-4 | +1.87e-4 | -7.43e-6 | -4.32e-4 |
| 160 | 3.00e-3 | 1 | 6.79e-3 | 6.79e-3 | 6.79e-3 | 6.79e-3 | 218 | -2.01e-4 | -2.01e-4 | -2.01e-4 | -4.09e-4 |
| 161 | 3.00e-3 | 1 | 7.17e-3 | 7.17e-3 | 7.17e-3 | 7.17e-3 | 238 | +2.31e-4 | +2.31e-4 | +2.31e-4 | -3.45e-4 |
| 162 | 3.00e-3 | 1 | 7.13e-3 | 7.13e-3 | 7.13e-3 | 7.13e-3 | 231 | -2.36e-5 | -2.36e-5 | -2.36e-5 | -3.13e-4 |
| 163 | 3.00e-3 | 1 | 7.26e-3 | 7.26e-3 | 7.26e-3 | 7.26e-3 | 240 | +7.25e-5 | +7.25e-5 | +7.25e-5 | -2.74e-4 |
| 164 | 3.00e-3 | 2 | 7.06e-3 | 7.62e-3 | 7.34e-3 | 7.62e-3 | 274 | -1.32e-4 | +2.76e-4 | +7.19e-5 | -2.06e-4 |
| 166 | 3.00e-3 | 2 | 7.97e-3 | 8.08e-3 | 8.02e-3 | 7.97e-3 | 274 | -5.07e-5 | +1.76e-4 | +6.25e-5 | -1.56e-4 |
| 168 | 3.00e-3 | 1 | 8.48e-3 | 8.48e-3 | 8.48e-3 | 8.48e-3 | 332 | +1.88e-4 | +1.88e-4 | +1.88e-4 | -1.22e-4 |
| 169 | 3.00e-3 | 1 | 8.32e-3 | 8.32e-3 | 8.32e-3 | 8.32e-3 | 298 | -6.35e-5 | -6.35e-5 | -6.35e-5 | -1.16e-4 |
| 170 | 3.00e-3 | 1 | 8.10e-3 | 8.10e-3 | 8.10e-3 | 8.10e-3 | 269 | -1.01e-4 | -1.01e-4 | -1.01e-4 | -1.15e-4 |
| 171 | 3.00e-3 | 1 | 8.40e-3 | 8.40e-3 | 8.40e-3 | 8.40e-3 | 302 | +1.22e-4 | +1.22e-4 | +1.22e-4 | -9.10e-5 |
| 172 | 3.00e-3 | 1 | 8.21e-3 | 8.21e-3 | 8.21e-3 | 8.21e-3 | 296 | -7.83e-5 | -7.83e-5 | -7.83e-5 | -8.97e-5 |
| 173 | 3.00e-3 | 1 | 8.09e-3 | 8.09e-3 | 8.09e-3 | 8.09e-3 | 277 | -4.99e-5 | -4.99e-5 | -4.99e-5 | -8.57e-5 |
| 174 | 3.00e-3 | 1 | 8.30e-3 | 8.30e-3 | 8.30e-3 | 8.30e-3 | 281 | +9.14e-5 | +9.14e-5 | +9.14e-5 | -6.80e-5 |
| 175 | 3.00e-3 | 1 | 8.38e-3 | 8.38e-3 | 8.38e-3 | 8.38e-3 | 304 | +2.94e-5 | +2.94e-5 | +2.94e-5 | -5.83e-5 |
| 176 | 3.00e-3 | 1 | 8.38e-3 | 8.38e-3 | 8.38e-3 | 8.38e-3 | 291 | +2.88e-7 | +2.88e-7 | +2.88e-7 | -5.24e-5 |
| 177 | 3.00e-3 | 1 | 8.81e-3 | 8.81e-3 | 8.81e-3 | 8.81e-3 | 348 | +1.44e-4 | +1.44e-4 | +1.44e-4 | -3.28e-5 |
| 179 | 3.00e-3 | 2 | 8.61e-3 | 9.23e-3 | 8.92e-3 | 8.61e-3 | 259 | -2.67e-4 | +1.26e-4 | -7.01e-5 | -4.19e-5 |
| 181 | 3.00e-3 | 2 | 8.27e-3 | 9.12e-3 | 8.69e-3 | 8.27e-3 | 252 | -3.88e-4 | +1.68e-4 | -1.10e-4 | -5.76e-5 |
| 183 | 3.00e-3 | 1 | 8.73e-3 | 8.73e-3 | 8.73e-3 | 8.73e-3 | 298 | +1.85e-4 | +1.85e-4 | +1.85e-4 | -3.34e-5 |
| 184 | 3.00e-3 | 2 | 8.17e-3 | 8.33e-3 | 8.25e-3 | 8.17e-3 | 230 | -1.97e-4 | -8.23e-5 | -1.40e-4 | -5.30e-5 |
| 185 | 3.00e-3 | 1 | 8.11e-3 | 8.11e-3 | 8.11e-3 | 8.11e-3 | 246 | -2.93e-5 | -2.93e-5 | -2.93e-5 | -5.06e-5 |
| 186 | 3.00e-3 | 1 | 8.45e-3 | 8.45e-3 | 8.45e-3 | 8.45e-3 | 273 | +1.50e-4 | +1.50e-4 | +1.50e-4 | -3.06e-5 |
| 187 | 3.00e-3 | 1 | 8.61e-3 | 8.61e-3 | 8.61e-3 | 8.61e-3 | 292 | +6.34e-5 | +6.34e-5 | +6.34e-5 | -2.12e-5 |
| 188 | 3.00e-3 | 1 | 8.40e-3 | 8.40e-3 | 8.40e-3 | 8.40e-3 | 246 | -9.92e-5 | -9.92e-5 | -9.92e-5 | -2.90e-5 |
| 189 | 3.00e-3 | 1 | 8.45e-3 | 8.45e-3 | 8.45e-3 | 8.45e-3 | 254 | +2.27e-5 | +2.27e-5 | +2.27e-5 | -2.38e-5 |
| 190 | 3.00e-3 | 1 | 8.91e-3 | 8.91e-3 | 8.91e-3 | 8.91e-3 | 309 | +1.71e-4 | +1.71e-4 | +1.71e-4 | -4.35e-6 |
| 191 | 3.00e-3 | 1 | 8.56e-3 | 8.56e-3 | 8.56e-3 | 8.56e-3 | 252 | -1.57e-4 | -1.57e-4 | -1.57e-4 | -1.97e-5 |
| 192 | 3.00e-3 | 1 | 8.63e-3 | 8.63e-3 | 8.63e-3 | 8.63e-3 | 272 | +2.93e-5 | +2.93e-5 | +2.93e-5 | -1.48e-5 |
| 193 | 3.00e-3 | 1 | 8.62e-3 | 8.62e-3 | 8.62e-3 | 8.62e-3 | 270 | -3.80e-6 | -3.80e-6 | -3.80e-6 | -1.37e-5 |
| 194 | 3.00e-3 | 1 | 8.37e-3 | 8.37e-3 | 8.37e-3 | 8.37e-3 | 240 | -1.24e-4 | -1.24e-4 | -1.24e-4 | -2.47e-5 |
| 195 | 3.00e-3 | 1 | 8.38e-3 | 8.38e-3 | 8.38e-3 | 8.38e-3 | 253 | +6.45e-6 | +6.45e-6 | +6.45e-6 | -2.16e-5 |
| 196 | 3.00e-3 | 1 | 8.40e-3 | 8.40e-3 | 8.40e-3 | 8.40e-3 | 255 | +1.09e-5 | +1.09e-5 | +1.09e-5 | -1.83e-5 |
| 197 | 3.00e-3 | 1 | 8.68e-3 | 8.68e-3 | 8.68e-3 | 8.68e-3 | 277 | +1.18e-4 | +1.18e-4 | +1.18e-4 | -4.67e-6 |
| 198 | 3.00e-3 | 1 | 8.51e-3 | 8.51e-3 | 8.51e-3 | 8.51e-3 | 258 | -7.78e-5 | -7.78e-5 | -7.78e-5 | -1.20e-5 |
| 199 | 3.00e-3 | 1 | 8.49e-3 | 8.49e-3 | 8.49e-3 | 8.49e-3 | 252 | -1.01e-5 | -1.01e-5 | -1.01e-5 | -1.18e-5 |

