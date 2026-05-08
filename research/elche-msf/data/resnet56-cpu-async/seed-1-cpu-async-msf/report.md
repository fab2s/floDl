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
| cpu-async | 0.016327 | 0.9304 | +0.0179 | 4856.6 | 359 | 232.2 | 100% | 100% | 100% | 10.5 |

## Best Mode per Model

| Model | Best Eval | Mode | Fastest (within 2% of solo-0) | Mode |
|-------|-----------|------|-------------------------------|------|
| resnet-graph | 0.9304 | cpu-async | - | - |

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
| cpu-async | 2.4264 | 1.2908 | 0.8062 | 0.6423 | 0.5728 | 0.5389 | 0.4887 | 0.4698 | 0.4462 | 0.4353 | 0.1721 | 0.1291 | 0.1123 | 0.1012 | 0.0969 | 0.0306 | 0.0259 | 0.0204 | 0.0183 | 0.0163 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | cpu-async | 0 | cuda:0 | 0.3572 | 1.1 |
| resnet-graph | cpu-async | 1 | cuda:1 | 0.3247 | 1.3 |
| resnet-graph | cpu-async | 2 | cuda:2 | 0.3181 | 1.3 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) | GPU2 Peak (MB) | GPU2 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|---------------|---------------|
| resnet-graph | cpu-async | 514 | 513 | 580 | 576 | 574 | 571 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| cpu-async | gpu1 | 466.9 | 2.2 | epoch-boundary(18) |
| cpu-async | gpu2 | 466.9 | 2.2 | epoch-boundary(18) |
| cpu-async | gpu1 | 4855.3 | 1.2 | epoch-boundary(199) |
| cpu-async | gpu2 | 4855.3 | 1.2 | epoch-boundary(199) |
| cpu-async | gpu1 | 3203.2 | 0.9 | epoch-boundary(131) |
| cpu-async | gpu1 | 1823.0 | 0.8 | epoch-boundary(74) |
| cpu-async | gpu2 | 3203.3 | 0.8 | epoch-boundary(131) |
| cpu-async | gpu2 | 1823.0 | 0.6 | epoch-boundary(74) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | cpu-async | gpu1 | 5.1s | 0.0s | 0.0s | 0.0s | 5.1s |
| resnet-graph | cpu-async | gpu2 | 4.8s | 0.0s | 0.0s | 0.0s | 5.4s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | cpu-async | 286 | 0 | 359 | 232.2 | 15088/23583 | 359 | 232.2 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | cpu-async | 318.6 | 6.6% |

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
| resnet-graph | cpu-async | 198 | 359 | 0 | 5.76e-3 | -1.03e-5 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | cpu-async | 0 | 359 | 9.46e-2 | 1.08e-1 | 4.86e-3 | 1.20e0 | 19.2 | -2.90e-4 | 2.04e-3 |
| resnet-graph | cpu-async | 1 | 359 | 9.54e-2 | 1.08e-1 | 4.77e-3 | 1.09e0 | 35.1 | -2.84e-4 | 2.04e-3 |
| resnet-graph | cpu-async | 2 | 359 | 9.64e-2 | 1.13e-1 | 4.93e-3 | 1.41e0 | 45.7 | -3.22e-4 | 2.08e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | cpu-async | rank0 ↔ rank1 | +0.9824 |
| resnet-graph | cpu-async | rank0 ↔ rank2 | +0.9902 |
| resnet-graph | cpu-async | rank1 ↔ rank2 | +0.9770 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | cpu-async | 52 (0,1,2,3,4,5,6,8…146,147) | 0 (—) | — | 0,1,2,3,4,5,6,8…146,147 | — |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | cpu-async | 0e0 | 3 | 18 | 18 |
| resnet-graph | cpu-async | 0e0 | 5 | 10 | 10 |
| resnet-graph | cpu-async | 0e0 | 10 | 4 | 4 |
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
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 231 | +0.075 |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | 67 | +0.245 |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | 57 | -0.457 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | cpu-async | 357 | -0.016 | 197 | +0.315 | +0.474 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | cpu-async | 358 | 6.10e1–1.99e2 | 8.93e1 | 5.26e-3 | 2.11e-2 | 3.88e-1 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 233 | 35–78013 | +2.545e-5 | 0.495 | +2.594e-5 | 0.509 | 98 | +1.889e-5 | 0.433 | 35–899 | +2.991e-3 | 0.515 |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 221 | 921–78013 | +2.725e-5 | 0.636 | +2.747e-5 | 0.636 | 97 | +1.899e-5 | 0.428 | 97–899 | +3.239e-3 | 0.665 |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | 68 | 78358–117012 | +3.062e-5 | 0.518 | +3.117e-5 | 0.540 | 50 | +2.843e-5 | 0.578 | 191–810 | +1.731e-3 | 0.495 |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | 58 | 117710–155855 | -1.390e-5 | 0.158 | -1.398e-5 | 0.163 | 50 | -1.522e-5 | 0.167 | 526–788 | +8.045e-4 | 0.015 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | +2.991e-3 | r0: +2.986e-3, r1: +2.995e-3, r2: +3.003e-3 | r0: 0.507, r1: 0.514, r2: 0.524 | 1.01× | ✓ |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +3.239e-3 | r0: +3.226e-3, r1: +3.247e-3, r2: +3.246e-3 | r0: 0.660, r1: 0.664, r2: 0.667 | 1.01× | ✓ |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | +1.731e-3 | r0: +1.728e-3, r1: +1.766e-3, r2: +1.702e-3 | r0: 0.497, r1: 0.513, r2: 0.473 | 1.04× | ✓ |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | +8.045e-4 | r0: +7.374e-4, r1: +7.909e-4, r2: +8.874e-4 | r0: 0.013, r1: 0.014, r2: 0.019 | 1.20× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | cpu-async | `▆▅▅▆▇███████████████████▇▄▄▅▅▅▅▅▆▆▆▆▆▄▁▁▁▁▁▁▁▁▁▁▁▁` | `▁▇██████████████████████▇▇████████████▇███████████` |

### Per-Epoch Detail: resnet-graph / cpu-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 12 | 1.18e-2 | 1.41e0 | 3.23e-1 | 1.80e-2 | 36 | -4.95e-2 | +7.64e-3 | -2.40e-2 | -1.95e-2 |
| 1 | 3.00e-1 | 6 | 1.55e-2 | 2.39e-2 | 1.89e-2 | 1.55e-2 | 36 | -8.06e-3 | +3.73e-3 | -1.38e-3 | -9.43e-3 |
| 2 | 3.00e-1 | 7 | 1.41e-2 | 2.56e-2 | 1.65e-2 | 1.46e-2 | 37 | -1.24e-2 | +6.05e-3 | -1.31e-3 | -4.75e-3 |
| 3 | 3.00e-1 | 6 | 1.61e-2 | 2.34e-2 | 1.82e-2 | 1.84e-2 | 34 | -8.63e-3 | +5.80e-3 | -1.44e-4 | -2.42e-3 |
| 4 | 3.00e-1 | 6 | 2.12e-2 | 2.80e-2 | 2.39e-2 | 2.25e-2 | 43 | -6.45e-3 | +4.59e-3 | -1.05e-4 | -1.36e-3 |
| 5 | 3.00e-1 | 5 | 2.41e-2 | 3.40e-2 | 2.68e-2 | 2.48e-2 | 40 | -8.18e-3 | +4.44e-3 | -6.71e-4 | -1.09e-3 |
| 6 | 3.00e-1 | 7 | 2.79e-2 | 3.81e-2 | 3.03e-2 | 2.94e-2 | 41 | -6.72e-3 | +4.59e-3 | -2.17e-4 | -6.22e-4 |
| 7 | 3.00e-1 | 5 | 3.20e-2 | 3.85e-2 | 3.45e-2 | 3.42e-2 | 40 | -3.14e-3 | +3.62e-3 | +1.50e-4 | -3.16e-4 |
| 8 | 3.00e-1 | 8 | 3.60e-2 | 4.34e-2 | 3.82e-2 | 3.91e-2 | 40 | -4.65e-3 | +3.31e-3 | +8.23e-5 | -6.38e-5 |
| 9 | 3.00e-1 | 4 | 4.35e-2 | 5.42e-2 | 4.69e-2 | 4.62e-2 | 44 | -4.71e-3 | +3.78e-3 | +9.21e-5 | -1.71e-5 |
| 10 | 3.00e-1 | 7 | 4.65e-2 | 5.79e-2 | 4.91e-2 | 4.65e-2 | 39 | -4.69e-3 | +2.78e-3 | -3.99e-4 | -2.20e-4 |
| 11 | 3.00e-1 | 7 | 4.96e-2 | 6.81e-2 | 5.28e-2 | 5.17e-2 | 35 | -9.13e-3 | +4.39e-3 | -5.41e-4 | -3.15e-4 |
| 12 | 3.00e-1 | 4 | 5.86e-2 | 7.76e-2 | 6.52e-2 | 5.86e-2 | 44 | -4.26e-3 | +4.41e-3 | -3.18e-4 | -3.72e-4 |
| 13 | 3.00e-1 | 6 | 6.29e-2 | 7.79e-2 | 6.77e-2 | 6.40e-2 | 40 | -4.64e-3 | +3.13e-3 | -2.45e-4 | -3.50e-4 |
| 14 | 3.00e-1 | 6 | 6.45e-2 | 8.83e-2 | 7.16e-2 | 7.36e-2 | 48 | -7.20e-3 | +3.78e-3 | -2.11e-4 | -2.27e-4 |
| 15 | 3.00e-1 | 5 | 7.46e-2 | 9.62e-2 | 8.08e-2 | 8.00e-2 | 47 | -5.61e-3 | +3.16e-3 | -2.26e-4 | -2.10e-4 |
| 16 | 3.00e-1 | 5 | 7.68e-2 | 9.98e-2 | 8.38e-2 | 8.21e-2 | 44 | -5.16e-3 | +2.69e-3 | -3.75e-4 | -2.58e-4 |
| 17 | 3.00e-1 | 5 | 7.89e-2 | 1.06e-1 | 8.75e-2 | 8.75e-2 | 51 | -5.72e-3 | +3.11e-3 | -4.32e-4 | -2.97e-4 |
| 18 | 3.00e-1 | 2 | 9.49e-2 | 1.54e-1 | 1.24e-1 | 1.54e-1 | 251 | +1.21e-3 | +1.92e-3 | +1.57e-3 | +6.08e-5 |
| 19 | 3.00e-1 | 1 | 1.71e-1 | 1.71e-1 | 1.71e-1 | 1.71e-1 | 271 | +3.86e-4 | +3.86e-4 | +3.86e-4 | +9.33e-5 |
| 20 | 3.00e-1 | 1 | 1.79e-1 | 1.79e-1 | 1.79e-1 | 1.79e-1 | 305 | +1.55e-4 | +1.55e-4 | +1.55e-4 | +9.95e-5 |
| 22 | 3.00e-1 | 2 | 1.72e-1 | 1.87e-1 | 1.79e-1 | 1.72e-1 | 225 | -3.80e-4 | +1.30e-4 | -1.25e-4 | +5.42e-5 |
| 24 | 3.00e-1 | 2 | 1.75e-1 | 2.00e-1 | 1.87e-1 | 1.75e-1 | 213 | -6.08e-4 | +4.47e-4 | -8.05e-5 | +2.34e-5 |
| 25 | 3.00e-1 | 1 | 1.79e-1 | 1.79e-1 | 1.79e-1 | 1.79e-1 | 229 | +8.58e-5 | +8.58e-5 | +8.58e-5 | +2.96e-5 |
| 26 | 3.00e-1 | 1 | 1.81e-1 | 1.81e-1 | 1.81e-1 | 1.81e-1 | 235 | +4.10e-5 | +4.10e-5 | +4.10e-5 | +3.07e-5 |
| 27 | 3.00e-1 | 1 | 1.89e-1 | 1.89e-1 | 1.89e-1 | 1.89e-1 | 255 | +1.73e-4 | +1.73e-4 | +1.73e-4 | +4.49e-5 |
| 28 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 274 | +1.36e-4 | +1.36e-4 | +1.36e-4 | +5.41e-5 |
| 29 | 3.00e-1 | 1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 256 | -3.48e-5 | -3.48e-5 | -3.48e-5 | +4.52e-5 |
| 30 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 264 | +9.03e-5 | +9.03e-5 | +9.03e-5 | +4.97e-5 |
| 31 | 3.00e-1 | 1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 254 | -2.64e-5 | -2.64e-5 | -2.64e-5 | +4.21e-5 |
| 32 | 3.00e-1 | 2 | 1.89e-1 | 2.00e-1 | 1.94e-1 | 1.89e-1 | 213 | -2.84e-4 | +5.74e-5 | -1.13e-4 | +1.08e-5 |
| 33 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 245 | +2.23e-4 | +2.23e-4 | +2.23e-4 | +3.20e-5 |
| 34 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 260 | +1.32e-4 | +1.32e-4 | +1.32e-4 | +4.20e-5 |
| 35 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 267 | +3.57e-5 | +3.57e-5 | +3.57e-5 | +4.14e-5 |
| 36 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 227 | -1.73e-4 | -1.73e-4 | -1.73e-4 | +2.00e-5 |
| 37 | 3.00e-1 | 2 | 1.93e-1 | 1.98e-1 | 1.96e-1 | 1.93e-1 | 202 | -1.45e-4 | -3.60e-5 | -9.05e-5 | -1.56e-6 |
| 38 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 244 | +2.59e-4 | +2.59e-4 | +2.59e-4 | +2.45e-5 |
| 39 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 250 | +1.28e-5 | +1.28e-5 | +1.28e-5 | +2.33e-5 |
| 40 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 239 | -2.32e-5 | -2.32e-5 | -2.32e-5 | +1.86e-5 |
| 41 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 253 | +1.35e-4 | +1.35e-4 | +1.35e-4 | +3.03e-5 |
| 42 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 229 | -2.07e-4 | -2.07e-4 | -2.07e-4 | +6.53e-6 |
| 43 | 3.00e-1 | 2 | 1.94e-1 | 2.17e-1 | 2.05e-1 | 1.94e-1 | 192 | -6.04e-4 | +2.86e-4 | -1.59e-4 | -2.93e-5 |
| 44 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 221 | +2.07e-4 | +2.07e-4 | +2.07e-4 | -5.71e-6 |
| 45 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 237 | +1.02e-4 | +1.02e-4 | +1.02e-4 | +5.02e-6 |
| 46 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 209 | -1.72e-4 | -1.72e-4 | -1.72e-4 | -1.26e-5 |
| 47 | 3.00e-1 | 2 | 2.00e-1 | 2.01e-1 | 2.01e-1 | 2.00e-1 | 196 | -3.12e-5 | +2.30e-5 | -4.10e-6 | -1.13e-5 |
| 48 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 219 | +1.26e-4 | +1.26e-4 | +1.26e-4 | +2.42e-6 |
| 49 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 208 | -1.34e-4 | -1.34e-4 | -1.34e-4 | -1.13e-5 |
| 50 | 3.00e-1 | 2 | 1.98e-1 | 2.14e-1 | 2.06e-1 | 1.98e-1 | 190 | -4.07e-4 | +2.90e-4 | -5.87e-5 | -2.38e-5 |
| 51 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 228 | +3.02e-4 | +3.02e-4 | +3.02e-4 | +8.77e-6 |
| 52 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 241 | +7.05e-5 | +7.05e-5 | +7.05e-5 | +1.49e-5 |
| 53 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 254 | +3.73e-5 | +3.73e-5 | +3.73e-5 | +1.72e-5 |
| 54 | 3.00e-1 | 2 | 1.98e-1 | 2.03e-1 | 2.00e-1 | 1.98e-1 | 190 | -3.38e-4 | -1.36e-4 | -2.37e-4 | -3.02e-5 |
| 55 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 235 | +3.88e-4 | +3.88e-4 | +3.88e-4 | +1.16e-5 |
| 56 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 224 | -5.87e-5 | -5.87e-5 | -5.87e-5 | +4.60e-6 |
| 57 | 3.00e-1 | 2 | 1.98e-1 | 2.05e-1 | 2.02e-1 | 1.98e-1 | 176 | -2.00e-4 | -1.99e-4 | -1.99e-4 | -3.42e-5 |
| 58 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 221 | +3.05e-4 | +3.05e-4 | +3.05e-4 | -2.26e-7 |
| 59 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 190 | -2.58e-4 | -2.58e-4 | -2.58e-4 | -2.60e-5 |
| 60 | 3.00e-1 | 2 | 1.99e-1 | 2.05e-1 | 2.02e-1 | 1.99e-1 | 187 | -1.48e-4 | +7.95e-5 | -3.41e-5 | -2.87e-5 |
| 61 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 214 | +1.76e-4 | +1.76e-4 | +1.76e-4 | -8.22e-6 |
| 62 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 225 | +9.45e-5 | +9.45e-5 | +9.45e-5 | +2.05e-6 |
| 63 | 3.00e-1 | 2 | 1.97e-1 | 2.08e-1 | 2.03e-1 | 1.97e-1 | 176 | -3.07e-4 | -7.18e-5 | -1.89e-4 | -3.55e-5 |
| 64 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 201 | +2.66e-4 | +2.66e-4 | +2.66e-4 | -5.39e-6 |
| 65 | 3.00e-1 | 2 | 1.96e-1 | 2.06e-1 | 2.01e-1 | 1.96e-1 | 167 | -2.90e-4 | -5.95e-5 | -1.75e-4 | -3.88e-5 |
| 66 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 209 | +4.03e-4 | +4.03e-4 | +4.03e-4 | +5.40e-6 |
| 67 | 3.00e-1 | 2 | 1.97e-1 | 2.15e-1 | 2.06e-1 | 1.97e-1 | 170 | -5.35e-4 | +4.69e-5 | -2.44e-4 | -4.49e-5 |
| 68 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 187 | +2.79e-4 | +2.79e-4 | +2.79e-4 | -1.25e-5 |
| 69 | 3.00e-1 | 2 | 1.93e-1 | 2.15e-1 | 2.04e-1 | 1.93e-1 | 154 | -7.00e-4 | +1.73e-4 | -2.64e-4 | -6.46e-5 |
| 70 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 188 | +3.50e-4 | +3.50e-4 | +3.50e-4 | -2.31e-5 |
| 71 | 3.00e-1 | 2 | 1.88e-1 | 2.05e-1 | 1.96e-1 | 1.88e-1 | 148 | -5.88e-4 | -4.56e-5 | -3.17e-4 | -8.17e-5 |
| 72 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 212 | +7.12e-4 | +7.12e-4 | +7.12e-4 | -2.29e-6 |
| 73 | 3.00e-1 | 2 | 1.90e-1 | 2.06e-1 | 1.98e-1 | 1.90e-1 | 150 | -5.57e-4 | -3.04e-4 | -4.31e-4 | -8.50e-5 |
| 74 | 3.00e-1 | 2 | 1.92e-1 | 2.03e-1 | 1.98e-1 | 1.92e-1 | 148 | -3.55e-4 | +3.70e-4 | +7.34e-6 | -7.10e-5 |
| 75 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 179 | +3.53e-4 | +3.53e-4 | +3.53e-4 | -2.87e-5 |
| 76 | 3.00e-1 | 2 | 1.90e-1 | 2.06e-1 | 1.98e-1 | 1.90e-1 | 138 | -5.72e-4 | +2.78e-5 | -2.72e-4 | -7.80e-5 |
| 77 | 3.00e-1 | 2 | 1.88e-1 | 1.98e-1 | 1.93e-1 | 1.88e-1 | 141 | -3.58e-4 | +2.29e-4 | -6.48e-5 | -7.84e-5 |
| 78 | 3.00e-1 | 2 | 1.89e-1 | 1.97e-1 | 1.93e-1 | 1.89e-1 | 146 | -3.13e-4 | +2.99e-4 | -7.38e-6 | -6.80e-5 |
| 79 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 177 | +5.63e-4 | +5.63e-4 | +5.63e-4 | -4.83e-6 |
| 80 | 3.00e-1 | 2 | 1.93e-1 | 2.10e-1 | 2.02e-1 | 1.93e-1 | 147 | -5.82e-4 | +4.58e-5 | -2.68e-4 | -5.80e-5 |
| 81 | 3.00e-1 | 2 | 1.92e-1 | 2.03e-1 | 1.97e-1 | 1.92e-1 | 138 | -3.96e-4 | +2.75e-4 | -6.06e-5 | -6.19e-5 |
| 82 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 192 | +5.01e-4 | +5.01e-4 | +5.01e-4 | -5.58e-6 |
| 83 | 3.00e-1 | 2 | 1.90e-1 | 2.08e-1 | 1.99e-1 | 1.90e-1 | 138 | -6.76e-4 | -7.70e-5 | -3.77e-4 | -7.91e-5 |
| 84 | 3.00e-1 | 2 | 1.84e-1 | 2.02e-1 | 1.93e-1 | 1.84e-1 | 130 | -6.91e-4 | +3.83e-4 | -1.54e-4 | -9.87e-5 |
| 85 | 3.00e-1 | 2 | 1.88e-1 | 2.08e-1 | 1.98e-1 | 1.88e-1 | 130 | -7.99e-4 | +6.67e-4 | -6.58e-5 | -9.98e-5 |
| 86 | 3.00e-1 | 2 | 1.88e-1 | 2.01e-1 | 1.94e-1 | 1.88e-1 | 131 | -4.94e-4 | +4.00e-4 | -4.71e-5 | -9.42e-5 |
| 87 | 3.00e-1 | 2 | 1.81e-1 | 2.01e-1 | 1.91e-1 | 1.81e-1 | 123 | -8.64e-4 | +3.93e-4 | -2.36e-4 | -1.27e-4 |
| 88 | 3.00e-1 | 2 | 1.83e-1 | 2.08e-1 | 1.96e-1 | 1.83e-1 | 122 | -1.06e-3 | +8.01e-4 | -1.29e-4 | -1.37e-4 |
| 89 | 3.00e-1 | 2 | 1.86e-1 | 2.01e-1 | 1.94e-1 | 1.86e-1 | 121 | -6.20e-4 | +5.72e-4 | -2.40e-5 | -1.21e-4 |
| 90 | 3.00e-1 | 2 | 1.83e-1 | 2.04e-1 | 1.93e-1 | 1.83e-1 | 121 | -8.92e-4 | +5.43e-4 | -1.75e-4 | -1.39e-4 |
| 91 | 3.00e-1 | 2 | 1.82e-1 | 1.92e-1 | 1.87e-1 | 1.82e-1 | 120 | -4.28e-4 | +3.23e-4 | -5.27e-5 | -1.26e-4 |
| 92 | 3.00e-1 | 2 | 1.81e-1 | 1.96e-1 | 1.88e-1 | 1.81e-1 | 115 | -6.98e-4 | +4.84e-4 | -1.07e-4 | -1.28e-4 |
| 93 | 3.00e-1 | 2 | 1.80e-1 | 2.14e-1 | 1.97e-1 | 1.80e-1 | 115 | -1.53e-3 | +9.59e-4 | -2.83e-4 | -1.70e-4 |
| 94 | 3.00e-1 | 2 | 1.76e-1 | 1.96e-1 | 1.86e-1 | 1.76e-1 | 107 | -1.02e-3 | +6.10e-4 | -2.07e-4 | -1.85e-4 |
| 95 | 3.00e-1 | 2 | 1.75e-1 | 1.92e-1 | 1.84e-1 | 1.75e-1 | 111 | -8.33e-4 | +6.23e-4 | -1.05e-4 | -1.77e-4 |
| 96 | 3.00e-1 | 3 | 1.76e-1 | 1.97e-1 | 1.84e-1 | 1.78e-1 | 107 | -1.07e-3 | +8.36e-4 | -4.84e-5 | -1.49e-4 |
| 97 | 3.00e-1 | 3 | 1.73e-1 | 1.98e-1 | 1.84e-1 | 1.73e-1 | 107 | -8.84e-4 | +7.55e-4 | -1.67e-4 | -1.64e-4 |
| 98 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 141 | +9.86e-4 | +9.86e-4 | +9.86e-4 | -4.94e-5 |
| 99 | 3.00e-1 | 3 | 1.73e-1 | 2.00e-1 | 1.83e-1 | 1.73e-1 | 97 | -1.48e-3 | +8.26e-4 | -6.28e-4 | -2.09e-4 |
| 100 | 3.00e-2 | 2 | 8.35e-2 | 1.80e-1 | 1.32e-1 | 8.35e-2 | 100 | -7.70e-3 | +3.12e-4 | -3.69e-3 | -9.11e-4 |
| 101 | 3.00e-2 | 4 | 1.73e-2 | 4.66e-2 | 2.70e-2 | 1.73e-2 | 94 | -6.80e-3 | -8.28e-4 | -3.83e-3 | -1.85e-3 |
| 102 | 3.00e-2 | 2 | 1.76e-2 | 1.98e-2 | 1.87e-2 | 1.76e-2 | 86 | -1.37e-3 | +9.25e-4 | -2.23e-4 | -1.55e-3 |
| 103 | 3.00e-2 | 3 | 1.71e-2 | 2.05e-2 | 1.85e-2 | 1.71e-2 | 86 | -1.64e-3 | +1.18e-3 | -3.13e-4 | -1.23e-3 |
| 104 | 3.00e-2 | 3 | 1.86e-2 | 2.00e-2 | 1.91e-2 | 1.88e-2 | 87 | -9.19e-4 | +1.32e-3 | +1.79e-4 | -8.58e-4 |
| 105 | 3.00e-2 | 2 | 1.90e-2 | 2.25e-2 | 2.08e-2 | 1.90e-2 | 78 | -2.14e-3 | +1.31e-3 | -4.16e-4 | -7.91e-4 |
| 106 | 3.00e-2 | 4 | 1.79e-2 | 2.24e-2 | 1.92e-2 | 1.79e-2 | 73 | -3.00e-3 | +1.70e-3 | -5.78e-4 | -7.33e-4 |
| 107 | 3.00e-2 | 1 | 1.71e-2 | 1.71e-2 | 1.71e-2 | 1.71e-2 | 64 | -7.03e-4 | -7.03e-4 | -7.03e-4 | -7.30e-4 |
| 108 | 3.00e-2 | 2 | 3.00e-2 | 3.14e-2 | 3.07e-2 | 3.14e-2 | 223 | +1.98e-4 | +1.95e-3 | +1.07e-3 | -3.96e-4 |
| 109 | 3.00e-2 | 1 | 3.36e-2 | 3.36e-2 | 3.36e-2 | 3.36e-2 | 254 | +2.70e-4 | +2.70e-4 | +2.70e-4 | -3.30e-4 |
| 110 | 3.00e-2 | 1 | 3.54e-2 | 3.54e-2 | 3.54e-2 | 3.54e-2 | 253 | +2.02e-4 | +2.02e-4 | +2.02e-4 | -2.76e-4 |
| 111 | 3.00e-2 | 1 | 3.78e-2 | 3.78e-2 | 3.78e-2 | 3.78e-2 | 271 | +2.47e-4 | +2.47e-4 | +2.47e-4 | -2.24e-4 |
| 112 | 3.00e-2 | 1 | 3.84e-2 | 3.84e-2 | 3.84e-2 | 3.84e-2 | 285 | +5.11e-5 | +5.11e-5 | +5.11e-5 | -1.97e-4 |
| 113 | 3.00e-2 | 1 | 3.90e-2 | 3.90e-2 | 3.90e-2 | 3.90e-2 | 260 | +5.93e-5 | +5.93e-5 | +5.93e-5 | -1.71e-4 |
| 114 | 3.00e-2 | 1 | 4.04e-2 | 4.04e-2 | 4.04e-2 | 4.04e-2 | 268 | +1.30e-4 | +1.30e-4 | +1.30e-4 | -1.41e-4 |
| 115 | 3.00e-2 | 1 | 3.93e-2 | 3.93e-2 | 3.93e-2 | 3.93e-2 | 247 | -1.09e-4 | -1.09e-4 | -1.09e-4 | -1.38e-4 |
| 116 | 3.00e-2 | 1 | 4.28e-2 | 4.28e-2 | 4.28e-2 | 4.28e-2 | 273 | +3.10e-4 | +3.10e-4 | +3.10e-4 | -9.29e-5 |
| 117 | 3.00e-2 | 1 | 4.26e-2 | 4.26e-2 | 4.26e-2 | 4.26e-2 | 254 | -1.59e-5 | -1.59e-5 | -1.59e-5 | -8.52e-5 |
| 118 | 3.00e-2 | 1 | 4.52e-2 | 4.52e-2 | 4.52e-2 | 4.52e-2 | 276 | +2.12e-4 | +2.12e-4 | +2.12e-4 | -5.55e-5 |
| 119 | 3.00e-2 | 1 | 4.73e-2 | 4.73e-2 | 4.73e-2 | 4.73e-2 | 292 | +1.58e-4 | +1.58e-4 | +1.58e-4 | -3.41e-5 |
| 120 | 3.00e-2 | 1 | 4.55e-2 | 4.55e-2 | 4.55e-2 | 4.55e-2 | 258 | -1.48e-4 | -1.48e-4 | -1.48e-4 | -4.56e-5 |
| 121 | 3.00e-2 | 1 | 4.40e-2 | 4.40e-2 | 4.40e-2 | 4.40e-2 | 222 | -1.52e-4 | -1.52e-4 | -1.52e-4 | -5.62e-5 |
| 122 | 3.00e-2 | 2 | 4.41e-2 | 4.51e-2 | 4.46e-2 | 4.41e-2 | 210 | -1.07e-4 | +1.04e-4 | -1.68e-6 | -4.69e-5 |
| 123 | 3.00e-2 | 1 | 4.90e-2 | 4.90e-2 | 4.90e-2 | 4.90e-2 | 243 | +4.35e-4 | +4.35e-4 | +4.35e-4 | +1.31e-6 |
| 124 | 3.00e-2 | 1 | 5.08e-2 | 5.08e-2 | 5.08e-2 | 5.08e-2 | 265 | +1.36e-4 | +1.36e-4 | +1.36e-4 | +1.47e-5 |
| 125 | 3.00e-2 | 1 | 4.90e-2 | 4.90e-2 | 4.90e-2 | 4.90e-2 | 239 | -1.48e-4 | -1.48e-4 | -1.48e-4 | -1.50e-6 |
| 126 | 3.00e-2 | 1 | 5.06e-2 | 5.06e-2 | 5.06e-2 | 5.06e-2 | 244 | +1.28e-4 | +1.28e-4 | +1.28e-4 | +1.14e-5 |
| 127 | 3.00e-2 | 1 | 5.15e-2 | 5.15e-2 | 5.15e-2 | 5.15e-2 | 238 | +7.99e-5 | +7.99e-5 | +7.99e-5 | +1.83e-5 |
| 128 | 3.00e-2 | 2 | 5.06e-2 | 5.31e-2 | 5.19e-2 | 5.31e-2 | 262 | -8.43e-5 | +1.83e-4 | +4.95e-5 | +2.55e-5 |
| 129 | 3.00e-2 | 1 | 5.77e-2 | 5.77e-2 | 5.77e-2 | 5.77e-2 | 283 | +2.93e-4 | +2.93e-4 | +2.93e-4 | +5.23e-5 |
| 130 | 3.00e-2 | 1 | 5.61e-2 | 5.61e-2 | 5.61e-2 | 5.61e-2 | 266 | -1.09e-4 | -1.09e-4 | -1.09e-4 | +3.62e-5 |
| 131 | 3.00e-2 | 1 | 5.78e-2 | 5.78e-2 | 5.78e-2 | 5.78e-2 | 271 | +1.15e-4 | +1.15e-4 | +1.15e-4 | +4.40e-5 |
| 132 | 3.00e-2 | 1 | 6.22e-2 | 6.22e-2 | 6.22e-2 | 6.22e-2 | 292 | +2.48e-4 | +2.48e-4 | +2.48e-4 | +6.45e-5 |
| 133 | 3.00e-2 | 1 | 5.98e-2 | 5.98e-2 | 5.98e-2 | 5.98e-2 | 252 | -1.56e-4 | -1.56e-4 | -1.56e-4 | +4.25e-5 |
| 134 | 3.00e-2 | 1 | 5.96e-2 | 5.96e-2 | 5.96e-2 | 5.96e-2 | 255 | -1.51e-5 | -1.51e-5 | -1.51e-5 | +3.67e-5 |
| 135 | 3.00e-2 | 1 | 6.03e-2 | 6.03e-2 | 6.03e-2 | 6.03e-2 | 252 | +4.74e-5 | +4.74e-5 | +4.74e-5 | +3.78e-5 |
| 136 | 3.00e-2 | 1 | 6.01e-2 | 6.01e-2 | 6.01e-2 | 6.01e-2 | 251 | -1.39e-5 | -1.39e-5 | -1.39e-5 | +3.26e-5 |
| 137 | 3.00e-2 | 1 | 6.07e-2 | 6.07e-2 | 6.07e-2 | 6.07e-2 | 243 | +4.14e-5 | +4.14e-5 | +4.14e-5 | +3.35e-5 |
| 138 | 3.00e-2 | 1 | 6.55e-2 | 6.55e-2 | 6.55e-2 | 6.55e-2 | 281 | +2.72e-4 | +2.72e-4 | +2.72e-4 | +5.74e-5 |
| 139 | 3.00e-2 | 1 | 6.68e-2 | 6.68e-2 | 6.68e-2 | 6.68e-2 | 275 | +7.41e-5 | +7.41e-5 | +7.41e-5 | +5.90e-5 |
| 140 | 3.00e-2 | 1 | 6.38e-2 | 6.38e-2 | 6.38e-2 | 6.38e-2 | 257 | -1.79e-4 | -1.79e-4 | -1.79e-4 | +3.52e-5 |
| 141 | 3.00e-2 | 1 | 6.78e-2 | 6.78e-2 | 6.78e-2 | 6.78e-2 | 290 | +2.10e-4 | +2.10e-4 | +2.10e-4 | +5.26e-5 |
| 142 | 3.00e-2 | 1 | 6.49e-2 | 6.49e-2 | 6.49e-2 | 6.49e-2 | 236 | -1.88e-4 | -1.88e-4 | -1.88e-4 | +2.86e-5 |
| 143 | 3.00e-2 | 2 | 6.22e-2 | 6.41e-2 | 6.31e-2 | 6.22e-2 | 207 | -1.44e-4 | -5.61e-5 | -9.99e-5 | +3.74e-6 |
| 144 | 3.00e-2 | 1 | 6.19e-2 | 6.19e-2 | 6.19e-2 | 6.19e-2 | 223 | -2.09e-5 | -2.09e-5 | -2.09e-5 | +1.28e-6 |
| 145 | 3.00e-2 | 1 | 6.73e-2 | 6.73e-2 | 6.73e-2 | 6.73e-2 | 257 | +3.22e-4 | +3.22e-4 | +3.22e-4 | +3.34e-5 |
| 146 | 3.00e-2 | 1 | 6.73e-2 | 6.73e-2 | 6.73e-2 | 6.73e-2 | 233 | +1.75e-7 | +1.75e-7 | +1.75e-7 | +3.00e-5 |
| 147 | 3.00e-2 | 1 | 7.02e-2 | 7.02e-2 | 7.02e-2 | 7.02e-2 | 241 | +1.77e-4 | +1.77e-4 | +1.77e-4 | +4.47e-5 |
| 148 | 3.00e-2 | 2 | 6.44e-2 | 6.74e-2 | 6.59e-2 | 6.44e-2 | 216 | -2.10e-4 | -1.80e-4 | -1.95e-4 | -9.36e-7 |
| 149 | 3.00e-2 | 1 | 6.75e-2 | 6.75e-2 | 6.75e-2 | 6.75e-2 | 230 | +2.02e-4 | +2.02e-4 | +2.02e-4 | +1.94e-5 |
| 150 | 3.00e-3 | 1 | 6.02e-2 | 6.02e-2 | 6.02e-2 | 6.02e-2 | 255 | -4.47e-4 | -4.47e-4 | -4.47e-4 | -2.72e-5 |
| 151 | 3.00e-3 | 1 | 2.94e-2 | 2.94e-2 | 2.94e-2 | 2.94e-2 | 239 | -3.00e-3 | -3.00e-3 | -3.00e-3 | -3.24e-4 |
| 152 | 3.00e-3 | 1 | 1.54e-2 | 1.54e-2 | 1.54e-2 | 1.54e-2 | 255 | -2.53e-3 | -2.53e-3 | -2.53e-3 | -5.45e-4 |
| 153 | 3.00e-3 | 2 | 6.16e-3 | 8.83e-3 | 7.49e-3 | 6.16e-3 | 203 | -2.30e-3 | -1.77e-3 | -2.04e-3 | -8.25e-4 |
| 154 | 3.00e-3 | 1 | 5.79e-3 | 5.79e-3 | 5.79e-3 | 5.79e-3 | 252 | -2.43e-4 | -2.43e-4 | -2.43e-4 | -7.67e-4 |
| 155 | 3.00e-3 | 1 | 5.65e-3 | 5.65e-3 | 5.65e-3 | 5.65e-3 | 265 | -9.18e-5 | -9.18e-5 | -9.18e-5 | -7.00e-4 |
| 156 | 3.00e-3 | 1 | 5.29e-3 | 5.29e-3 | 5.29e-3 | 5.29e-3 | 216 | -3.06e-4 | -3.06e-4 | -3.06e-4 | -6.60e-4 |
| 157 | 3.00e-3 | 1 | 5.15e-3 | 5.15e-3 | 5.15e-3 | 5.15e-3 | 249 | -1.07e-4 | -1.07e-4 | -1.07e-4 | -6.05e-4 |
| 158 | 3.00e-3 | 1 | 5.36e-3 | 5.36e-3 | 5.36e-3 | 5.36e-3 | 240 | +1.66e-4 | +1.66e-4 | +1.66e-4 | -5.28e-4 |
| 159 | 3.00e-3 | 1 | 5.14e-3 | 5.14e-3 | 5.14e-3 | 5.14e-3 | 200 | -2.08e-4 | -2.08e-4 | -2.08e-4 | -4.96e-4 |
| 160 | 3.00e-3 | 2 | 5.32e-3 | 5.36e-3 | 5.34e-3 | 5.32e-3 | 225 | -3.47e-5 | +1.47e-4 | +5.61e-5 | -3.92e-4 |
| 161 | 3.00e-3 | 1 | 5.70e-3 | 5.70e-3 | 5.70e-3 | 5.70e-3 | 271 | +2.54e-4 | +2.54e-4 | +2.54e-4 | -3.27e-4 |
| 162 | 3.00e-3 | 1 | 5.98e-3 | 5.98e-3 | 5.98e-3 | 5.98e-3 | 276 | +1.76e-4 | +1.76e-4 | +1.76e-4 | -2.77e-4 |
| 163 | 3.00e-3 | 1 | 6.09e-3 | 6.09e-3 | 6.09e-3 | 6.09e-3 | 275 | +6.29e-5 | +6.29e-5 | +6.29e-5 | -2.43e-4 |
| 164 | 3.00e-3 | 1 | 6.06e-3 | 6.06e-3 | 6.06e-3 | 6.06e-3 | 279 | -1.51e-5 | -1.51e-5 | -1.51e-5 | -2.20e-4 |
| 165 | 3.00e-3 | 1 | 5.66e-3 | 5.66e-3 | 5.66e-3 | 5.66e-3 | 257 | -2.68e-4 | -2.68e-4 | -2.68e-4 | -2.25e-4 |
| 166 | 3.00e-3 | 1 | 5.80e-3 | 5.80e-3 | 5.80e-3 | 5.80e-3 | 273 | +9.02e-5 | +9.02e-5 | +9.02e-5 | -1.93e-4 |
| 167 | 3.00e-3 | 1 | 5.46e-3 | 5.46e-3 | 5.46e-3 | 5.46e-3 | 239 | -2.53e-4 | -2.53e-4 | -2.53e-4 | -1.99e-4 |
| 168 | 3.00e-3 | 1 | 5.82e-3 | 5.82e-3 | 5.82e-3 | 5.82e-3 | 254 | +2.55e-4 | +2.55e-4 | +2.55e-4 | -1.54e-4 |
| 169 | 3.00e-3 | 1 | 5.88e-3 | 5.88e-3 | 5.88e-3 | 5.88e-3 | 258 | +3.65e-5 | +3.65e-5 | +3.65e-5 | -1.35e-4 |
| 170 | 3.00e-3 | 1 | 6.03e-3 | 6.03e-3 | 6.03e-3 | 6.03e-3 | 276 | +8.94e-5 | +8.94e-5 | +8.94e-5 | -1.13e-4 |
| 171 | 3.00e-3 | 1 | 5.92e-3 | 5.92e-3 | 5.92e-3 | 5.92e-3 | 279 | -6.36e-5 | -6.36e-5 | -6.36e-5 | -1.08e-4 |
| 172 | 3.00e-3 | 1 | 5.73e-3 | 5.73e-3 | 5.73e-3 | 5.73e-3 | 257 | -1.24e-4 | -1.24e-4 | -1.24e-4 | -1.09e-4 |
| 173 | 3.00e-3 | 2 | 5.61e-3 | 5.97e-3 | 5.79e-3 | 5.61e-3 | 219 | -2.90e-4 | +1.62e-4 | -6.39e-5 | -1.03e-4 |
| 174 | 3.00e-3 | 1 | 5.50e-3 | 5.50e-3 | 5.50e-3 | 5.50e-3 | 224 | -8.56e-5 | -8.56e-5 | -8.56e-5 | -1.01e-4 |
| 175 | 3.00e-3 | 1 | 5.58e-3 | 5.58e-3 | 5.58e-3 | 5.58e-3 | 226 | +6.15e-5 | +6.15e-5 | +6.15e-5 | -8.49e-5 |
| 176 | 3.00e-3 | 1 | 5.64e-3 | 5.64e-3 | 5.64e-3 | 5.64e-3 | 240 | +4.81e-5 | +4.81e-5 | +4.81e-5 | -7.16e-5 |
| 177 | 3.00e-3 | 1 | 5.53e-3 | 5.53e-3 | 5.53e-3 | 5.53e-3 | 248 | -7.73e-5 | -7.73e-5 | -7.73e-5 | -7.22e-5 |
| 178 | 3.00e-3 | 1 | 5.75e-3 | 5.75e-3 | 5.75e-3 | 5.75e-3 | 246 | +1.55e-4 | +1.55e-4 | +1.55e-4 | -4.94e-5 |
| 179 | 3.00e-3 | 1 | 5.71e-3 | 5.71e-3 | 5.71e-3 | 5.71e-3 | 255 | -2.89e-5 | -2.89e-5 | -2.89e-5 | -4.74e-5 |
| 180 | 3.00e-3 | 2 | 5.53e-3 | 6.07e-3 | 5.80e-3 | 5.53e-3 | 212 | -4.41e-4 | +2.54e-4 | -9.40e-5 | -5.97e-5 |
| 181 | 3.00e-3 | 1 | 5.77e-3 | 5.77e-3 | 5.77e-3 | 5.77e-3 | 253 | +1.68e-4 | +1.68e-4 | +1.68e-4 | -3.69e-5 |
| 182 | 3.00e-3 | 1 | 5.69e-3 | 5.69e-3 | 5.69e-3 | 5.69e-3 | 236 | -5.48e-5 | -5.48e-5 | -5.48e-5 | -3.87e-5 |
| 183 | 3.00e-3 | 1 | 5.86e-3 | 5.86e-3 | 5.86e-3 | 5.86e-3 | 256 | +1.14e-4 | +1.14e-4 | +1.14e-4 | -2.35e-5 |
| 184 | 3.00e-3 | 1 | 5.70e-3 | 5.70e-3 | 5.70e-3 | 5.70e-3 | 237 | -1.21e-4 | -1.21e-4 | -1.21e-4 | -3.32e-5 |
| 185 | 3.00e-3 | 1 | 5.78e-3 | 5.78e-3 | 5.78e-3 | 5.78e-3 | 229 | +6.21e-5 | +6.21e-5 | +6.21e-5 | -2.37e-5 |
| 186 | 3.00e-3 | 2 | 5.21e-3 | 5.48e-3 | 5.34e-3 | 5.48e-3 | 229 | -5.15e-4 | +2.19e-4 | -1.48e-4 | -4.36e-5 |
| 187 | 3.00e-3 | 1 | 5.75e-3 | 5.75e-3 | 5.75e-3 | 5.75e-3 | 246 | +1.95e-4 | +1.95e-4 | +1.95e-4 | -1.97e-5 |
| 188 | 3.00e-3 | 1 | 5.50e-3 | 5.50e-3 | 5.50e-3 | 5.50e-3 | 211 | -2.06e-4 | -2.06e-4 | -2.06e-4 | -3.84e-5 |
| 189 | 3.00e-3 | 1 | 5.92e-3 | 5.92e-3 | 5.92e-3 | 5.92e-3 | 251 | +2.89e-4 | +2.89e-4 | +2.89e-4 | -5.67e-6 |
| 190 | 3.00e-3 | 2 | 5.29e-3 | 5.79e-3 | 5.54e-3 | 5.29e-3 | 202 | -4.43e-4 | -9.19e-5 | -2.67e-4 | -5.72e-5 |
| 191 | 3.00e-3 | 1 | 5.77e-3 | 5.77e-3 | 5.77e-3 | 5.77e-3 | 240 | +3.56e-4 | +3.56e-4 | +3.56e-4 | -1.59e-5 |
| 192 | 3.00e-3 | 1 | 5.41e-3 | 5.41e-3 | 5.41e-3 | 5.41e-3 | 217 | -2.97e-4 | -2.97e-4 | -2.97e-4 | -4.39e-5 |
| 193 | 3.00e-3 | 2 | 5.31e-3 | 5.63e-3 | 5.47e-3 | 5.31e-3 | 188 | -3.16e-4 | +1.73e-4 | -7.13e-5 | -5.16e-5 |
| 194 | 3.00e-3 | 1 | 5.16e-3 | 5.16e-3 | 5.16e-3 | 5.16e-3 | 207 | -1.38e-4 | -1.38e-4 | -1.38e-4 | -6.02e-5 |
| 195 | 3.00e-3 | 1 | 5.65e-3 | 5.65e-3 | 5.65e-3 | 5.65e-3 | 239 | +3.83e-4 | +3.83e-4 | +3.83e-4 | -1.59e-5 |
| 196 | 3.00e-3 | 1 | 5.94e-3 | 5.94e-3 | 5.94e-3 | 5.94e-3 | 231 | +2.13e-4 | +2.13e-4 | +2.13e-4 | +6.99e-6 |
| 197 | 3.00e-3 | 2 | 5.49e-3 | 5.91e-3 | 5.70e-3 | 5.49e-3 | 188 | -3.92e-4 | -1.99e-5 | -2.06e-4 | -3.53e-5 |
| 198 | 3.00e-3 | 1 | 5.68e-3 | 5.68e-3 | 5.68e-3 | 5.68e-3 | 247 | +1.39e-4 | +1.39e-4 | +1.39e-4 | -1.79e-5 |
| 199 | 3.00e-3 | 1 | 5.76e-3 | 5.76e-3 | 5.76e-3 | 5.76e-3 | 220 | +5.80e-5 | +5.80e-5 | +5.80e-5 | -1.03e-5 |

