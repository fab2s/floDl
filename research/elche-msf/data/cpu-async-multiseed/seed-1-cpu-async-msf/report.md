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
| cpu-async | 0.050330 | 0.9140 | +0.0015 | 1782.7 | 734 | 77.3 | 100% | 100% | 100% | 12.4 |

## Best Mode per Model

| Model | Best Eval | Mode | Fastest (within 2% of solo-0) | Mode |
|-------|-----------|------|-------------------------------|------|
| resnet-graph | 0.9140 | cpu-async | - | - |

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
| cpu-async | 2.0258 | 0.7695 | 0.6287 | 0.5648 | 0.5406 | 0.5210 | 0.5021 | 0.4920 | 0.4751 | 0.4620 | 0.2107 | 0.1691 | 0.1450 | 0.1263 | 0.1214 | 0.0698 | 0.0623 | 0.0563 | 0.0528 | 0.0503 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | cpu-async | 0 | cuda:0 | 0.4009 | 2.7 |
| resnet-graph | cpu-async | 1 | cuda:1 | 0.3034 | 3.8 |
| resnet-graph | cpu-async | 2 | cuda:2 | 0.2957 | 3.6 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) | GPU2 Peak (MB) | GPU2 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|---------------|---------------|
| resnet-graph | cpu-async | 358 | 357 | 400 | 381 | 388 | 386 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| cpu-async | gpu1 | 323.4 | 1.5 | epoch-boundary(35) |
| cpu-async | gpu1 | 165.7 | 1.4 | epoch-boundary(17) |
| cpu-async | gpu2 | 323.4 | 1.4 | epoch-boundary(35) |
| cpu-async | gpu2 | 165.8 | 1.3 | epoch-boundary(17) |
| cpu-async | gpu1 | 305.6 | 1.1 | epoch-boundary(33) |
| cpu-async | gpu2 | 305.6 | 0.9 | epoch-boundary(33) |
| cpu-async | gpu0 | 0.4 | 0.9 | cpu-avg |
| cpu-async | gpu1 | 616.1 | 0.6 | epoch-boundary(68) |
| cpu-async | gpu2 | 616.1 | 0.6 | epoch-boundary(68) |
| cpu-async | gpu1 | 534.4 | 0.6 | epoch-boundary(59) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | cpu-async | gpu0 | 0.0s | 0.0s | 0.9s | 0.0s | 0.9s |
| resnet-graph | cpu-async | gpu1 | 5.7s | 0.0s | 0.0s | 0.0s | 6.5s |
| resnet-graph | cpu-async | gpu2 | 4.2s | 0.0s | 0.0s | 0.0s | 5.0s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | cpu-async | 328 | 0 | 734 | 77.3 | 1293/8355 | 734 | 77.3 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | cpu-async | 180.8 | 10.1% |

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
| resnet-graph | cpu-async | 192 | 734 | 0 | 7.54e-3 | -2.04e-4 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | cpu-async | 0 | 734 | 5.48e-2 | 7.43e-2 | 2.26e-3 | 4.21e-1 | 30.1 | -2.05e-4 | 1.18e-3 |
| resnet-graph | cpu-async | 1 | 734 | 5.54e-2 | 7.56e-2 | 2.10e-3 | 4.07e-1 | 32.7 | -2.28e-4 | 1.33e-3 |
| resnet-graph | cpu-async | 2 | 734 | 5.56e-2 | 7.60e-2 | 2.10e-3 | 4.38e-1 | 37.2 | -2.36e-4 | 1.38e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | cpu-async | rank0 ↔ rank1 | +0.9967 |
| resnet-graph | cpu-async | rank0 ↔ rank2 | +0.9958 |
| resnet-graph | cpu-async | rank1 ↔ rank2 | +0.9973 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | cpu-async | 46 (2,3,4,5,7,9,10,11…147,149) | 0 (—) | — | 2,3,4,5,7,9,10,11…147,149 | — |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | cpu-async | 0e0 | 3 | 0 | 0 |
| resnet-graph | cpu-async | 0e0 | 5 | 0 | 0 |
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
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 204 | +0.175 |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | 205 | +0.141 |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | 321 | +0.091 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | cpu-async | 732 | +0.025 | 191 | +0.113 | +0.245 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | cpu-async | 733 | 3.35e1–8.03e1 | 6.32e1 | 1.68e-3 | 3.57e-3 | 4.76e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 206 | 31–77921 | +8.505e-6 | 0.425 | +8.670e-6 | 0.453 | 92 | +5.722e-6 | 0.320 | 31–1025 | +8.950e-4 | 0.697 |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 195 | 940–77921 | +9.559e-6 | 0.574 | +9.664e-6 | 0.594 | 91 | +5.898e-6 | 0.329 | 87–1025 | +9.933e-4 | 0.904 |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | 206 | 78439–117017 | -1.537e-6 | 0.004 | -1.507e-6 | 0.005 | 50 | -2.552e-6 | 0.012 | 67–518 | +6.354e-4 | 0.105 |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | 322 | 117147–155825 | -1.919e-6 | 0.008 | -1.982e-6 | 0.010 | 50 | +9.885e-7 | 0.004 | 76–941 | +2.014e-3 | 0.266 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | +8.950e-4 | r0: +8.853e-4, r1: +9.025e-4, r2: +8.995e-4 | r0: 0.703, r1: 0.695, r2: 0.673 | 1.02× | ✓ |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +9.933e-4 | r0: +9.770e-4, r1: +1.002e-3, r2: +1.002e-3 | r0: 0.893, r1: 0.902, r2: 0.890 | 1.03× | ✓ |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | +6.354e-4 | r0: +5.485e-4, r1: +6.549e-4, r2: +7.002e-4 | r0: 0.083, r1: 0.106, r2: 0.119 | 1.28× | ✓ |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | +2.014e-3 | r0: +1.894e-3, r1: +2.044e-3, r2: +2.094e-3 | r0: 0.249, r1: 0.263, r2: 0.262 | 1.11× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | cpu-async | `█▇▇▇███████████████████▅▄▄▄▅▅▅▅▅▅▅▅▄▁▁▁▁▁▁▁▁▁▁▁▁` | `▁▇▇▇▇▇█████████████████▆▇▇███▇▇▇▇▆▆▄▆▇▇▇▇▇▇▇▇▇▇▇` |

### Per-Epoch Detail: resnet-graph / cpu-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 11 | 1.18e-1 | 4.38e-1 | 2.21e-1 | 1.30e-1 | 34 | -3.07e-2 | +1.55e-2 | -8.52e-3 | -6.90e-3 |
| 1 | 3.00e-1 | 7 | 8.78e-2 | 2.04e-1 | 1.15e-1 | 9.24e-2 | 36 | -1.32e-2 | +5.26e-3 | -2.41e-3 | -3.93e-3 |
| 2 | 3.00e-1 | 7 | 9.01e-2 | 1.38e-1 | 1.04e-1 | 1.04e-1 | 43 | -1.10e-2 | +5.06e-3 | -5.77e-4 | -1.91e-3 |
| 3 | 3.00e-1 | 6 | 1.06e-1 | 1.52e-1 | 1.17e-1 | 1.09e-1 | 36 | -7.17e-3 | +4.10e-3 | -7.06e-4 | -1.30e-3 |
| 4 | 3.00e-1 | 6 | 1.16e-1 | 1.55e-1 | 1.25e-1 | 1.19e-1 | 41 | -7.07e-3 | +4.49e-3 | -4.52e-4 | -8.86e-4 |
| 5 | 3.00e-1 | 9 | 1.14e-1 | 1.52e-1 | 1.21e-1 | 1.17e-1 | 39 | -8.74e-3 | +3.14e-3 | -5.37e-4 | -5.80e-4 |
| 6 | 3.00e-1 | 4 | 1.17e-1 | 1.64e-1 | 1.31e-1 | 1.17e-1 | 37 | -7.42e-3 | +3.94e-3 | -1.26e-3 | -8.44e-4 |
| 7 | 3.00e-1 | 6 | 1.17e-1 | 1.54e-1 | 1.29e-1 | 1.23e-1 | 43 | -8.02e-3 | +3.64e-3 | -5.81e-4 | -7.13e-4 |
| 8 | 3.00e-1 | 5 | 1.23e-1 | 1.63e-1 | 1.35e-1 | 1.23e-1 | 46 | -4.74e-3 | +3.26e-3 | -5.62e-4 | -6.75e-4 |
| 9 | 3.00e-1 | 6 | 1.20e-1 | 1.65e-1 | 1.29e-1 | 1.21e-1 | 43 | -6.16e-3 | +3.21e-3 | -6.73e-4 | -6.48e-4 |
| 10 | 3.00e-1 | 8 | 1.15e-1 | 1.54e-1 | 1.24e-1 | 1.24e-1 | 46 | -4.96e-3 | +2.82e-3 | -2.88e-4 | -3.66e-4 |
| 11 | 3.00e-1 | 4 | 1.08e-1 | 1.60e-1 | 1.24e-1 | 1.08e-1 | 37 | -8.63e-3 | +2.72e-3 | -2.21e-3 | -1.00e-3 |
| 12 | 3.00e-1 | 7 | 1.09e-1 | 1.52e-1 | 1.21e-1 | 1.34e-1 | 56 | -7.12e-3 | +4.25e-3 | -2.01e-4 | -4.64e-4 |
| 13 | 3.00e-1 | 7 | 1.04e-1 | 1.55e-1 | 1.15e-1 | 1.13e-1 | 38 | -8.37e-3 | +1.97e-3 | -1.11e-3 | -6.39e-4 |
| 14 | 3.00e-1 | 6 | 1.06e-1 | 1.56e-1 | 1.18e-1 | 1.14e-1 | 38 | -8.45e-3 | +3.64e-3 | -1.03e-3 | -7.30e-4 |
| 15 | 3.00e-1 | 8 | 1.15e-1 | 1.63e-1 | 1.26e-1 | 1.23e-1 | 41 | -5.36e-3 | +3.81e-3 | -3.55e-4 | -4.63e-4 |
| 16 | 3.00e-1 | 4 | 1.11e-1 | 1.60e-1 | 1.27e-1 | 1.11e-1 | 41 | -7.69e-3 | +3.20e-3 | -1.63e-3 | -8.84e-4 |
| 17 | 3.00e-1 | 2 | 1.11e-1 | 2.22e-1 | 1.67e-1 | 2.22e-1 | 274 | +1.77e-5 | +2.52e-3 | +1.27e-3 | -4.63e-4 |
| 19 | 3.00e-1 | 1 | 2.44e-1 | 2.44e-1 | 2.44e-1 | 2.44e-1 | 368 | +2.52e-4 | +2.52e-4 | +2.52e-4 | -3.91e-4 |
| 20 | 3.00e-1 | 1 | 2.34e-1 | 2.34e-1 | 2.34e-1 | 2.34e-1 | 352 | -1.20e-4 | -1.20e-4 | -1.20e-4 | -3.64e-4 |
| 21 | 3.00e-1 | 1 | 2.35e-1 | 2.35e-1 | 2.35e-1 | 2.35e-1 | 357 | +2.21e-5 | +2.21e-5 | +2.21e-5 | -3.26e-4 |
| 22 | 3.00e-1 | 1 | 2.31e-1 | 2.31e-1 | 2.31e-1 | 2.31e-1 | 335 | -5.68e-5 | -5.68e-5 | -5.68e-5 | -2.99e-4 |
| 24 | 3.00e-1 | 2 | 2.18e-1 | 2.32e-1 | 2.25e-1 | 2.18e-1 | 282 | -2.21e-4 | +9.08e-6 | -1.06e-4 | -2.63e-4 |
| 26 | 3.00e-1 | 1 | 2.31e-1 | 2.31e-1 | 2.31e-1 | 2.31e-1 | 333 | +1.83e-4 | +1.83e-4 | +1.83e-4 | -2.19e-4 |
| 27 | 3.00e-1 | 1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 303 | -1.35e-4 | -1.35e-4 | -1.35e-4 | -2.10e-4 |
| 28 | 3.00e-1 | 1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 314 | +4.28e-5 | +4.28e-5 | +4.28e-5 | -1.85e-4 |
| 29 | 3.00e-1 | 1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 305 | -3.29e-5 | -3.29e-5 | -3.29e-5 | -1.70e-4 |
| 30 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 293 | -3.65e-5 | -3.65e-5 | -3.65e-5 | -1.56e-4 |
| 31 | 3.00e-1 | 1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 302 | +3.83e-5 | +3.83e-5 | +3.83e-5 | -1.37e-4 |
| 32 | 3.00e-1 | 1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 310 | +4.23e-5 | +4.23e-5 | +4.23e-5 | -1.19e-4 |
| 33 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 282 | -1.17e-4 | -1.17e-4 | -1.17e-4 | -1.19e-4 |
| 35 | 3.00e-1 | 2 | 2.22e-1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 282 | +2.88e-5 | +4.73e-5 | +3.81e-5 | -8.91e-5 |
| 37 | 3.00e-1 | 2 | 2.30e-1 | 2.42e-1 | 2.36e-1 | 2.30e-1 | 282 | -1.84e-4 | +2.11e-4 | +1.37e-5 | -7.16e-5 |
| 39 | 3.00e-1 | 1 | 2.40e-1 | 2.40e-1 | 2.40e-1 | 2.40e-1 | 351 | +1.24e-4 | +1.24e-4 | +1.24e-4 | -5.20e-5 |
| 40 | 3.00e-1 | 1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 291 | -2.53e-4 | -2.53e-4 | -2.53e-4 | -7.21e-5 |
| 41 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 268 | -6.55e-5 | -6.55e-5 | -6.55e-5 | -7.14e-5 |
| 42 | 3.00e-1 | 1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 276 | +4.06e-5 | +4.06e-5 | +4.06e-5 | -6.02e-5 |
| 43 | 3.00e-1 | 1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 279 | -1.66e-6 | -1.66e-6 | -1.66e-6 | -5.43e-5 |
| 44 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 273 | -2.14e-5 | -2.14e-5 | -2.14e-5 | -5.10e-5 |
| 45 | 3.00e-1 | 1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 270 | +2.98e-5 | +2.98e-5 | +2.98e-5 | -4.30e-5 |
| 46 | 3.00e-1 | 1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 285 | +7.24e-5 | +7.24e-5 | +7.24e-5 | -3.14e-5 |
| 47 | 3.00e-1 | 1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 261 | -7.54e-5 | -7.54e-5 | -7.54e-5 | -3.58e-5 |
| 48 | 3.00e-1 | 1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 287 | +1.24e-4 | +1.24e-4 | +1.24e-4 | -1.99e-5 |
| 49 | 3.00e-1 | 1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 281 | -6.38e-5 | -6.38e-5 | -6.38e-5 | -2.42e-5 |
| 50 | 3.00e-1 | 1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 271 | -3.20e-5 | -3.20e-5 | -3.20e-5 | -2.50e-5 |
| 51 | 3.00e-1 | 1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 292 | +8.99e-5 | +8.99e-5 | +8.99e-5 | -1.35e-5 |
| 52 | 3.00e-1 | 1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 272 | -1.10e-4 | -1.10e-4 | -1.10e-4 | -2.32e-5 |
| 53 | 3.00e-1 | 1 | 2.34e-1 | 2.34e-1 | 2.34e-1 | 2.34e-1 | 294 | +1.59e-4 | +1.59e-4 | +1.59e-4 | -5.00e-6 |
| 54 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 239 | -3.65e-4 | -3.65e-4 | -3.65e-4 | -4.10e-5 |
| 55 | 3.00e-1 | 1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 256 | +1.22e-4 | +1.22e-4 | +1.22e-4 | -2.46e-5 |
| 56 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 233 | -1.33e-4 | -1.33e-4 | -1.33e-4 | -3.54e-5 |
| 57 | 3.00e-1 | 2 | 2.14e-1 | 2.19e-1 | 2.17e-1 | 2.14e-1 | 219 | -1.09e-4 | +9.00e-5 | -9.41e-6 | -3.15e-5 |
| 59 | 3.00e-1 | 2 | 2.17e-1 | 2.24e-1 | 2.20e-1 | 2.17e-1 | 242 | -1.33e-4 | +1.73e-4 | +1.99e-5 | -2.32e-5 |
| 60 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 250 | +2.70e-5 | +2.70e-5 | +2.70e-5 | -1.82e-5 |
| 61 | 3.00e-1 | 1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 278 | +9.92e-5 | +9.92e-5 | +9.92e-5 | -6.48e-6 |
| 62 | 3.00e-1 | 1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 255 | -5.36e-5 | -5.36e-5 | -5.36e-5 | -1.12e-5 |
| 63 | 3.00e-1 | 1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 256 | +4.25e-5 | +4.25e-5 | +4.25e-5 | -5.82e-6 |
| 64 | 3.00e-1 | 1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 250 | -5.34e-5 | -5.34e-5 | -5.34e-5 | -1.06e-5 |
| 65 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 246 | -2.86e-5 | -2.86e-5 | -2.86e-5 | -1.24e-5 |
| 66 | 3.00e-1 | 2 | 2.14e-1 | 2.17e-1 | 2.15e-1 | 2.14e-1 | 219 | -7.08e-5 | -4.02e-5 | -5.55e-5 | -2.07e-5 |
| 68 | 3.00e-1 | 2 | 2.15e-1 | 2.23e-1 | 2.19e-1 | 2.15e-1 | 233 | -1.71e-4 | +1.72e-4 | +5.10e-7 | -1.84e-5 |
| 69 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 227 | -1.00e-4 | -1.00e-4 | -1.00e-4 | -2.66e-5 |
| 70 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 222 | +6.20e-5 | +6.20e-5 | +6.20e-5 | -1.77e-5 |
| 71 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 238 | +8.13e-5 | +8.13e-5 | +8.13e-5 | -7.84e-6 |
| 72 | 3.00e-1 | 2 | 2.06e-1 | 2.15e-1 | 2.10e-1 | 2.06e-1 | 201 | -2.16e-4 | -3.93e-5 | -1.27e-4 | -3.14e-5 |
| 73 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 236 | +2.81e-4 | +2.81e-4 | +2.81e-4 | -1.81e-7 |
| 74 | 3.00e-1 | 1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 241 | +4.00e-5 | +4.00e-5 | +4.00e-5 | +3.83e-6 |
| 75 | 3.00e-1 | 1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 246 | -3.40e-6 | -3.40e-6 | -3.40e-6 | +3.11e-6 |
| 76 | 3.00e-1 | 2 | 2.09e-1 | 2.19e-1 | 2.14e-1 | 2.09e-1 | 188 | -2.30e-4 | -6.47e-5 | -1.47e-4 | -2.63e-5 |
| 77 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 219 | +2.14e-5 | +2.14e-5 | +2.14e-5 | -2.16e-5 |
| 78 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 243 | +1.51e-4 | +1.51e-4 | +1.51e-4 | -4.31e-6 |
| 79 | 3.00e-1 | 2 | 2.04e-1 | 2.14e-1 | 2.09e-1 | 2.04e-1 | 188 | -2.57e-4 | -9.02e-5 | -1.74e-4 | -3.73e-5 |
| 80 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 224 | +2.89e-4 | +2.89e-4 | +2.89e-4 | -4.66e-6 |
| 81 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 206 | -1.75e-4 | -1.75e-4 | -1.75e-4 | -2.17e-5 |
| 82 | 3.00e-1 | 1 | 2.28e-1 | 2.28e-1 | 2.28e-1 | 2.28e-1 | 257 | +3.28e-4 | +3.28e-4 | +3.28e-4 | +1.33e-5 |
| 83 | 3.00e-1 | 2 | 2.02e-1 | 2.21e-1 | 2.12e-1 | 2.02e-1 | 176 | -4.91e-4 | -1.41e-4 | -3.16e-4 | -5.11e-5 |
| 84 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 212 | +3.11e-4 | +3.11e-4 | +3.11e-4 | -1.49e-5 |
| 85 | 3.00e-1 | 2 | 1.93e-1 | 2.16e-1 | 2.04e-1 | 1.93e-1 | 162 | -6.81e-4 | -1.32e-5 | -3.47e-4 | -8.14e-5 |
| 86 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 214 | +4.32e-4 | +4.32e-4 | +4.32e-4 | -3.01e-5 |
| 87 | 3.00e-1 | 2 | 1.96e-1 | 2.17e-1 | 2.06e-1 | 1.96e-1 | 162 | -6.41e-4 | +1.11e-4 | -2.65e-4 | -7.85e-5 |
| 88 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 192 | +2.86e-4 | +2.86e-4 | +2.86e-4 | -4.21e-5 |
| 89 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 209 | +1.65e-4 | +1.65e-4 | +1.65e-4 | -2.14e-5 |
| 90 | 3.00e-1 | 2 | 1.96e-1 | 2.11e-1 | 2.03e-1 | 1.96e-1 | 162 | -4.56e-4 | -7.81e-5 | -2.67e-4 | -6.99e-5 |
| 91 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 192 | +2.39e-4 | +2.39e-4 | +2.39e-4 | -3.91e-5 |
| 92 | 3.00e-1 | 2 | 2.00e-1 | 2.08e-1 | 2.04e-1 | 2.08e-1 | 186 | -1.59e-4 | +2.13e-4 | +2.71e-5 | -2.46e-5 |
| 93 | 3.00e-1 | 2 | 1.96e-1 | 2.12e-1 | 2.04e-1 | 1.96e-1 | 162 | -4.70e-4 | +9.31e-5 | -1.88e-4 | -5.86e-5 |
| 94 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 205 | +4.21e-4 | +4.21e-4 | +4.21e-4 | -1.07e-5 |
| 95 | 3.00e-1 | 2 | 1.95e-1 | 2.06e-1 | 2.01e-1 | 1.95e-1 | 153 | -3.74e-4 | -1.89e-4 | -2.81e-4 | -6.30e-5 |
| 96 | 3.00e-1 | 2 | 1.92e-1 | 2.09e-1 | 2.00e-1 | 1.92e-1 | 153 | -5.68e-4 | +3.61e-4 | -1.04e-4 | -7.54e-5 |
| 97 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 194 | +5.41e-4 | +5.41e-4 | +5.41e-4 | -1.37e-5 |
| 98 | 3.00e-1 | 2 | 1.95e-1 | 2.10e-1 | 2.02e-1 | 1.95e-1 | 157 | -4.74e-4 | -7.80e-5 | -2.76e-4 | -6.56e-5 |
| 99 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 190 | +3.83e-4 | +3.83e-4 | +3.83e-4 | -2.07e-5 |
| 100 | 3.00e-2 | 2 | 1.03e-1 | 2.08e-1 | 1.55e-1 | 1.03e-1 | 133 | -5.30e-3 | -3.95e-5 | -2.67e-3 | -5.50e-4 |
| 101 | 3.00e-2 | 2 | 3.09e-2 | 5.34e-2 | 4.21e-2 | 3.09e-2 | 133 | -4.11e-3 | -3.74e-3 | -3.93e-3 | -1.19e-3 |
| 102 | 3.00e-2 | 1 | 2.43e-2 | 2.43e-2 | 2.43e-2 | 2.43e-2 | 186 | -1.28e-3 | -1.28e-3 | -1.28e-3 | -1.20e-3 |
| 103 | 3.00e-2 | 2 | 2.12e-2 | 2.31e-2 | 2.22e-2 | 2.12e-2 | 133 | -6.36e-4 | -2.81e-4 | -4.58e-4 | -1.06e-3 |
| 104 | 3.00e-2 | 2 | 2.17e-2 | 2.36e-2 | 2.26e-2 | 2.17e-2 | 133 | -6.34e-4 | +5.84e-4 | -2.51e-5 | -8.72e-4 |
| 105 | 3.00e-2 | 2 | 2.22e-2 | 2.40e-2 | 2.31e-2 | 2.22e-2 | 125 | -6.24e-4 | +5.80e-4 | -2.21e-5 | -7.16e-4 |
| 106 | 3.00e-2 | 2 | 2.27e-2 | 2.44e-2 | 2.36e-2 | 2.27e-2 | 125 | -5.78e-4 | +5.57e-4 | -1.08e-5 | -5.88e-4 |
| 107 | 3.00e-2 | 2 | 2.29e-2 | 2.38e-2 | 2.33e-2 | 2.29e-2 | 125 | -3.15e-4 | +3.09e-4 | -3.26e-6 | -4.80e-4 |
| 108 | 3.00e-2 | 2 | 2.40e-2 | 2.52e-2 | 2.46e-2 | 2.40e-2 | 132 | -3.54e-4 | +5.91e-4 | +1.19e-4 | -3.71e-4 |
| 109 | 3.00e-2 | 1 | 2.68e-2 | 2.68e-2 | 2.68e-2 | 2.68e-2 | 165 | +6.56e-4 | +6.56e-4 | +6.56e-4 | -2.68e-4 |
| 110 | 3.00e-2 | 3 | 2.43e-2 | 2.83e-2 | 2.59e-2 | 2.43e-2 | 121 | -9.63e-4 | +3.19e-4 | -3.23e-4 | -2.89e-4 |
| 111 | 3.00e-2 | 2 | 2.49e-2 | 2.64e-2 | 2.56e-2 | 2.49e-2 | 121 | -4.83e-4 | +5.25e-4 | +2.08e-5 | -2.35e-4 |
| 112 | 3.00e-2 | 2 | 2.62e-2 | 2.77e-2 | 2.70e-2 | 2.62e-2 | 121 | -4.48e-4 | +6.87e-4 | +1.19e-4 | -1.74e-4 |
| 113 | 3.00e-2 | 2 | 2.68e-2 | 2.93e-2 | 2.81e-2 | 2.68e-2 | 108 | -8.31e-4 | +6.53e-4 | -8.91e-5 | -1.65e-4 |
| 114 | 3.00e-2 | 2 | 2.58e-2 | 2.98e-2 | 2.78e-2 | 2.58e-2 | 101 | -1.43e-3 | +6.79e-4 | -3.74e-4 | -2.15e-4 |
| 115 | 3.00e-2 | 2 | 2.62e-2 | 2.84e-2 | 2.73e-2 | 2.62e-2 | 101 | -7.96e-4 | +6.76e-4 | -6.01e-5 | -1.93e-4 |
| 116 | 3.00e-2 | 3 | 2.67e-2 | 2.90e-2 | 2.75e-2 | 2.69e-2 | 103 | -7.80e-4 | +7.30e-4 | +5.29e-6 | -1.45e-4 |
| 117 | 3.00e-2 | 2 | 2.69e-2 | 2.95e-2 | 2.82e-2 | 2.69e-2 | 101 | -9.14e-4 | +6.66e-4 | -1.24e-4 | -1.49e-4 |
| 118 | 3.00e-2 | 3 | 2.64e-2 | 3.05e-2 | 2.81e-2 | 2.64e-2 | 95 | -1.09e-3 | +9.09e-4 | -2.06e-4 | -1.77e-4 |
| 119 | 3.00e-2 | 3 | 2.64e-2 | 2.89e-2 | 2.75e-2 | 2.64e-2 | 95 | -6.10e-4 | +6.91e-4 | -8.78e-5 | -1.62e-4 |
| 120 | 3.00e-2 | 2 | 2.81e-2 | 3.03e-2 | 2.92e-2 | 2.81e-2 | 95 | -7.93e-4 | +1.14e-3 | +1.75e-4 | -1.08e-4 |
| 121 | 3.00e-2 | 3 | 2.70e-2 | 3.05e-2 | 2.82e-2 | 2.72e-2 | 90 | -1.37e-3 | +7.08e-4 | -1.80e-4 | -1.32e-4 |
| 122 | 3.00e-2 | 3 | 2.77e-2 | 3.13e-2 | 2.93e-2 | 2.77e-2 | 90 | -8.94e-4 | +1.02e-3 | -1.17e-4 | -1.42e-4 |
| 123 | 3.00e-2 | 3 | 2.71e-2 | 3.41e-2 | 3.01e-2 | 2.71e-2 | 84 | -1.94e-3 | +1.45e-3 | -4.29e-4 | -2.40e-4 |
| 124 | 3.00e-2 | 3 | 2.58e-2 | 3.14e-2 | 2.80e-2 | 2.58e-2 | 72 | -2.18e-3 | +1.17e-3 | -5.07e-4 | -3.28e-4 |
| 125 | 3.00e-2 | 3 | 2.68e-2 | 3.21e-2 | 2.87e-2 | 2.72e-2 | 72 | -2.68e-3 | +1.96e-3 | -1.81e-4 | -3.04e-4 |
| 126 | 3.00e-2 | 6 | 2.36e-2 | 3.28e-2 | 2.66e-2 | 2.36e-2 | 56 | -3.17e-3 | +1.57e-3 | -6.53e-4 | -4.78e-4 |
| 127 | 3.00e-2 | 3 | 2.57e-2 | 3.09e-2 | 2.76e-2 | 2.57e-2 | 52 | -3.00e-3 | +2.80e-3 | -1.83e-4 | -4.26e-4 |
| 128 | 3.00e-2 | 5 | 2.31e-2 | 3.12e-2 | 2.55e-2 | 2.33e-2 | 47 | -3.26e-3 | +2.06e-3 | -7.09e-4 | -5.47e-4 |
| 129 | 3.00e-2 | 7 | 2.22e-2 | 3.08e-2 | 2.41e-2 | 2.34e-2 | 44 | -4.60e-3 | +2.92e-3 | -4.56e-4 | -4.41e-4 |
| 130 | 3.00e-2 | 4 | 2.30e-2 | 3.07e-2 | 2.54e-2 | 2.32e-2 | 44 | -4.45e-3 | +3.32e-3 | -6.57e-4 | -5.41e-4 |
| 131 | 3.00e-2 | 6 | 2.15e-2 | 3.06e-2 | 2.45e-2 | 2.15e-2 | 36 | -4.27e-3 | +3.17e-3 | -8.00e-4 | -7.06e-4 |
| 132 | 3.00e-2 | 7 | 2.10e-2 | 3.01e-2 | 2.30e-2 | 2.22e-2 | 35 | -7.85e-3 | +4.04e-3 | -6.76e-4 | -6.12e-4 |
| 133 | 3.00e-2 | 6 | 2.22e-2 | 2.95e-2 | 2.46e-2 | 2.28e-2 | 35 | -6.08e-3 | +3.79e-3 | -6.88e-4 | -6.77e-4 |
| 134 | 3.00e-2 | 6 | 2.38e-2 | 3.07e-2 | 2.63e-2 | 2.71e-2 | 48 | -7.13e-3 | +4.05e-3 | -7.03e-5 | -3.59e-4 |
| 135 | 3.00e-2 | 9 | 2.08e-2 | 3.31e-2 | 2.45e-2 | 2.16e-2 | 33 | -5.61e-3 | +2.41e-3 | -1.08e-3 | -7.46e-4 |
| 136 | 3.00e-2 | 4 | 2.28e-2 | 3.10e-2 | 2.55e-2 | 2.39e-2 | 39 | -7.07e-3 | +5.23e-3 | -6.15e-4 | -7.25e-4 |
| 137 | 3.00e-2 | 7 | 2.35e-2 | 3.25e-2 | 2.57e-2 | 2.53e-2 | 39 | -9.37e-3 | +3.91e-3 | -5.56e-4 | -5.61e-4 |
| 138 | 3.00e-2 | 6 | 2.48e-2 | 3.36e-2 | 2.72e-2 | 2.48e-2 | 39 | -5.07e-3 | +3.36e-3 | -7.35e-4 | -6.52e-4 |
| 139 | 3.00e-2 | 7 | 2.36e-2 | 3.58e-2 | 2.71e-2 | 2.60e-2 | 38 | -6.06e-3 | +3.94e-3 | -7.12e-4 | -6.05e-4 |
| 140 | 3.00e-2 | 9 | 2.32e-2 | 3.63e-2 | 2.59e-2 | 2.40e-2 | 33 | -7.15e-3 | +3.95e-3 | -8.74e-4 | -6.53e-4 |
| 141 | 3.00e-2 | 5 | 2.42e-2 | 3.56e-2 | 2.72e-2 | 2.42e-2 | 33 | -8.31e-3 | +5.01e-3 | -1.32e-3 | -9.41e-4 |
| 142 | 3.00e-2 | 7 | 2.41e-2 | 3.49e-2 | 2.68e-2 | 2.45e-2 | 33 | -5.37e-3 | +4.71e-3 | -8.14e-4 | -8.58e-4 |
| 143 | 3.00e-2 | 11 | 2.18e-2 | 3.62e-2 | 2.48e-2 | 2.25e-2 | 29 | -9.99e-3 | +4.89e-3 | -9.77e-4 | -8.26e-4 |
| 144 | 3.00e-2 | 5 | 2.05e-2 | 3.56e-2 | 2.59e-2 | 2.05e-2 | 24 | -7.11e-3 | +6.10e-3 | -2.60e-3 | -1.66e-3 |
| 145 | 3.00e-2 | 5 | 2.89e-2 | 3.76e-2 | 3.40e-2 | 3.58e-2 | 58 | -1.87e-3 | +6.95e-3 | +1.83e-3 | -3.54e-4 |
| 146 | 3.00e-2 | 5 | 3.10e-2 | 4.26e-2 | 3.44e-2 | 3.22e-2 | 44 | -5.25e-3 | +1.81e-3 | -8.30e-4 | -5.27e-4 |
| 147 | 3.00e-2 | 7 | 2.80e-2 | 4.23e-2 | 3.20e-2 | 2.80e-2 | 41 | -5.32e-3 | +2.92e-3 | -9.05e-4 | -7.31e-4 |
| 148 | 3.00e-2 | 5 | 3.01e-2 | 3.98e-2 | 3.26e-2 | 3.02e-2 | 41 | -4.89e-3 | +4.07e-3 | -5.28e-4 | -6.70e-4 |
| 149 | 3.00e-2 | 5 | 2.86e-2 | 4.07e-2 | 3.26e-2 | 2.86e-2 | 41 | -5.24e-3 | +3.51e-3 | -9.98e-4 | -8.38e-4 |
| 150 | 3.00e-3 | 8 | 2.84e-3 | 3.19e-2 | 1.01e-2 | 2.84e-3 | 44 | -1.99e-2 | +2.29e-3 | -6.21e-3 | -3.66e-3 |
| 151 | 3.00e-3 | 5 | 2.55e-3 | 3.86e-3 | 2.90e-3 | 2.61e-3 | 36 | -7.92e-3 | +3.31e-3 | -1.50e-3 | -2.75e-3 |
| 152 | 3.00e-3 | 6 | 2.45e-3 | 3.43e-3 | 2.84e-3 | 2.94e-3 | 46 | -5.14e-3 | +3.54e-3 | -2.56e-4 | -1.51e-3 |
| 153 | 3.00e-3 | 6 | 2.70e-3 | 3.69e-3 | 2.97e-3 | 2.70e-3 | 40 | -6.03e-3 | +2.98e-3 | -7.61e-4 | -1.17e-3 |
| 154 | 3.00e-3 | 8 | 2.57e-3 | 3.60e-3 | 2.87e-3 | 2.74e-3 | 41 | -4.64e-3 | +3.36e-3 | -4.41e-4 | -7.22e-4 |
| 155 | 3.00e-3 | 4 | 2.55e-3 | 3.58e-3 | 2.90e-3 | 2.72e-3 | 46 | -6.50e-3 | +3.66e-3 | -8.71e-4 | -7.77e-4 |
| 156 | 3.00e-3 | 6 | 2.52e-3 | 3.49e-3 | 2.83e-3 | 2.90e-3 | 49 | -1.05e-2 | +3.54e-3 | -6.36e-4 | -6.22e-4 |
| 157 | 3.00e-3 | 6 | 2.94e-3 | 3.69e-3 | 3.19e-3 | 2.94e-3 | 46 | -2.58e-3 | +3.06e-3 | -2.78e-4 | -5.05e-4 |
| 158 | 3.00e-3 | 6 | 2.64e-3 | 3.77e-3 | 2.92e-3 | 2.64e-3 | 35 | -6.23e-3 | +2.92e-3 | -1.05e-3 | -7.48e-4 |
| 159 | 3.00e-3 | 9 | 2.34e-3 | 3.61e-3 | 2.78e-3 | 3.02e-3 | 45 | -8.70e-3 | +4.18e-3 | -3.31e-4 | -2.98e-4 |
| 160 | 3.00e-3 | 4 | 2.78e-3 | 3.59e-3 | 3.10e-3 | 2.78e-3 | 42 | -3.13e-3 | +2.22e-3 | -8.76e-4 | -5.31e-4 |
| 161 | 3.00e-3 | 7 | 2.56e-3 | 3.85e-3 | 2.96e-3 | 2.56e-3 | 37 | -3.92e-3 | +3.69e-3 | -8.53e-4 | -7.33e-4 |
| 162 | 3.00e-3 | 6 | 2.63e-3 | 3.47e-3 | 2.92e-3 | 2.72e-3 | 36 | -4.39e-3 | +4.10e-3 | -4.42e-4 | -6.43e-4 |
| 163 | 3.00e-3 | 7 | 2.45e-3 | 3.65e-3 | 2.78e-3 | 2.65e-3 | 36 | -7.45e-3 | +3.96e-3 | -8.56e-4 | -6.77e-4 |
| 164 | 3.00e-3 | 8 | 2.75e-3 | 3.68e-3 | 2.97e-3 | 2.76e-3 | 43 | -5.53e-3 | +4.09e-3 | -3.94e-4 | -5.34e-4 |
| 165 | 3.00e-3 | 4 | 2.79e-3 | 3.63e-3 | 3.13e-3 | 2.90e-3 | 41 | -3.43e-3 | +3.35e-3 | -5.10e-4 | -5.56e-4 |
| 166 | 3.00e-3 | 9 | 2.55e-3 | 3.87e-3 | 2.83e-3 | 2.55e-3 | 36 | -7.92e-3 | +3.34e-3 | -9.69e-4 | -7.37e-4 |
| 167 | 3.00e-3 | 5 | 2.64e-3 | 3.33e-3 | 2.84e-3 | 2.64e-3 | 33 | -5.53e-3 | +3.71e-3 | -6.63e-4 | -7.40e-4 |
| 168 | 3.00e-3 | 10 | 2.36e-3 | 3.73e-3 | 2.64e-3 | 2.46e-3 | 32 | -8.71e-3 | +4.59e-3 | -8.64e-4 | -6.80e-4 |
| 169 | 3.00e-3 | 5 | 2.50e-3 | 3.37e-3 | 2.78e-3 | 2.54e-3 | 34 | -5.06e-3 | +4.70e-3 | -8.20e-4 | -7.71e-4 |
| 170 | 3.00e-3 | 7 | 2.47e-3 | 3.79e-3 | 2.80e-3 | 2.69e-3 | 40 | -9.22e-3 | +5.02e-3 | -8.83e-4 | -7.42e-4 |
| 171 | 3.00e-3 | 10 | 2.35e-3 | 3.57e-3 | 2.68e-3 | 2.49e-3 | 33 | -6.32e-3 | +3.71e-3 | -6.59e-4 | -6.10e-4 |
| 172 | 3.00e-3 | 5 | 2.50e-3 | 3.46e-3 | 2.83e-3 | 3.02e-3 | 43 | -8.53e-3 | +4.58e-3 | -1.88e-4 | -3.63e-4 |
| 173 | 3.00e-3 | 7 | 2.77e-3 | 3.72e-3 | 3.12e-3 | 2.77e-3 | 41 | -4.15e-3 | +2.45e-3 | -6.24e-4 | -5.45e-4 |
| 174 | 3.00e-3 | 4 | 2.65e-3 | 3.89e-3 | 3.06e-3 | 2.66e-3 | 40 | -6.66e-3 | +4.18e-3 | -1.47e-3 | -8.98e-4 |
| 175 | 3.00e-3 | 7 | 2.82e-3 | 3.96e-3 | 3.09e-3 | 2.82e-3 | 40 | -5.94e-3 | +4.77e-3 | -5.26e-4 | -7.23e-4 |
| 176 | 3.00e-3 | 6 | 2.83e-3 | 4.07e-3 | 3.22e-3 | 2.83e-3 | 39 | -4.04e-3 | +4.47e-3 | -6.64e-4 | -7.50e-4 |
| 177 | 3.00e-3 | 6 | 2.84e-3 | 3.91e-3 | 3.10e-3 | 3.03e-3 | 39 | -9.70e-3 | +3.72e-3 | -7.15e-4 | -6.69e-4 |
| 178 | 3.00e-3 | 6 | 2.85e-3 | 3.76e-3 | 3.10e-3 | 3.00e-3 | 43 | -5.22e-3 | +2.65e-3 | -5.17e-4 | -5.66e-4 |
| 179 | 3.00e-3 | 6 | 2.96e-3 | 3.82e-3 | 3.20e-3 | 2.96e-3 | 43 | -3.79e-3 | +3.07e-3 | -4.70e-4 | -5.40e-4 |
| 180 | 3.00e-3 | 6 | 2.92e-3 | 4.10e-3 | 3.22e-3 | 2.92e-3 | 38 | -5.36e-3 | +3.77e-3 | -6.84e-4 | -6.24e-4 |
| 181 | 3.00e-3 | 6 | 2.85e-3 | 3.88e-3 | 3.14e-3 | 3.07e-3 | 45 | -5.64e-3 | +3.56e-3 | -4.60e-4 | -5.11e-4 |
| 182 | 3.00e-3 | 8 | 2.75e-3 | 3.71e-3 | 3.11e-3 | 3.34e-3 | 41 | -6.38e-3 | +2.49e-3 | -2.44e-4 | -2.16e-4 |
| 183 | 3.00e-3 | 5 | 2.75e-3 | 4.09e-3 | 3.17e-3 | 2.91e-3 | 44 | -6.93e-3 | +2.38e-3 | -1.45e-3 | -6.86e-4 |
| 184 | 3.00e-3 | 6 | 2.89e-3 | 4.31e-3 | 3.37e-3 | 2.89e-3 | 32 | -3.52e-3 | +4.27e-3 | -8.54e-4 | -8.40e-4 |
| 185 | 3.00e-3 | 8 | 2.60e-3 | 3.72e-3 | 2.91e-3 | 2.92e-3 | 37 | -7.57e-3 | +3.52e-3 | -5.67e-4 | -5.58e-4 |
| 186 | 3.00e-3 | 6 | 2.56e-3 | 3.93e-3 | 2.98e-3 | 2.70e-3 | 38 | -5.76e-3 | +3.64e-3 | -1.01e-3 | -7.50e-4 |
| 187 | 3.00e-3 | 8 | 2.88e-3 | 3.72e-3 | 3.13e-3 | 3.08e-3 | 43 | -5.57e-3 | +4.41e-3 | -1.18e-4 | -3.96e-4 |
| 188 | 3.00e-3 | 6 | 2.67e-3 | 3.88e-3 | 2.97e-3 | 2.91e-3 | 35 | -8.21e-3 | +3.17e-3 | -9.97e-4 | -5.69e-4 |
| 189 | 3.00e-3 | 7 | 2.63e-3 | 3.81e-3 | 2.96e-3 | 2.82e-3 | 35 | -6.32e-3 | +3.33e-3 | -8.50e-4 | -6.54e-4 |
| 190 | 3.00e-3 | 6 | 2.81e-3 | 3.88e-3 | 3.11e-3 | 2.86e-3 | 36 | -6.16e-3 | +3.66e-3 | -8.36e-4 | -7.39e-4 |
| 191 | 3.00e-3 | 7 | 2.70e-3 | 4.06e-3 | 3.04e-3 | 2.84e-3 | 40 | -8.65e-3 | +4.26e-3 | -8.91e-4 | -7.64e-4 |
| 192 | 3.00e-3 | 8 | 2.88e-3 | 4.05e-3 | 3.15e-3 | 2.90e-3 | 37 | -5.90e-3 | +4.34e-3 | -4.95e-4 | -6.09e-4 |
| 193 | 3.00e-3 | 6 | 2.64e-3 | 3.74e-3 | 2.97e-3 | 2.68e-3 | 33 | -6.40e-3 | +3.35e-3 | -1.06e-3 | -8.20e-4 |
| 194 | 3.00e-3 | 10 | 2.45e-3 | 3.78e-3 | 2.72e-3 | 2.45e-3 | 28 | -9.39e-3 | +4.98e-3 | -9.77e-4 | -8.79e-4 |
| 195 | 3.00e-3 | 6 | 2.56e-3 | 3.70e-3 | 2.84e-3 | 2.85e-3 | 37 | -1.06e-2 | +6.45e-3 | -4.69e-4 | -6.16e-4 |
| 196 | 3.00e-3 | 8 | 2.47e-3 | 3.97e-3 | 2.83e-3 | 2.74e-3 | 34 | -9.32e-3 | +4.66e-3 | -1.06e-3 | -6.94e-4 |
| 197 | 3.00e-3 | 10 | 2.56e-3 | 3.72e-3 | 2.83e-3 | 2.85e-3 | 36 | -8.08e-3 | +4.27e-3 | -4.14e-4 | -3.55e-4 |
| 198 | 3.00e-3 | 1 | 2.66e-3 | 2.66e-3 | 2.66e-3 | 2.66e-3 | 28 | -2.42e-3 | -2.42e-3 | -2.42e-3 | -5.61e-4 |
| 199 | 3.00e-3 | 1 | 7.54e-3 | 7.54e-3 | 7.54e-3 | 7.54e-3 | 345 | +3.02e-3 | +3.02e-3 | +3.02e-3 | -2.04e-4 |

