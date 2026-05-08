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
| cpu-async | 0.051794 | 0.9219 | +0.0094 | 1802.6 | 484 | 79.5 | 100% | 99% | 100% | 19.4 |

## Best Mode per Model

| Model | Best Eval | Mode | Fastest (within 2% of solo-0) | Mode |
|-------|-----------|------|-------------------------------|------|
| resnet-graph | 0.9219 | cpu-async | - | - |

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
| cpu-async | 2.0232 | 0.7436 | 0.6251 | 0.5618 | 0.5380 | 0.5158 | 0.4986 | 0.4845 | 0.4705 | 0.4712 | 0.2078 | 0.1652 | 0.1440 | 0.1253 | 0.1185 | 0.0690 | 0.0637 | 0.0597 | 0.0544 | 0.0518 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | cpu-async | 0 | cuda:0 | 0.4051 | 2.8 |
| resnet-graph | cpu-async | 1 | cuda:1 | 0.2991 | 3.7 |
| resnet-graph | cpu-async | 2 | cuda:2 | 0.2958 | 3.5 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) | GPU2 Peak (MB) | GPU2 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|---------------|---------------|
| resnet-graph | cpu-async | 358 | 357 | 394 | 390 | 392 | 382 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| cpu-async | gpu2 | 1800.6 | 2.1 | epoch-boundary(199) |
| cpu-async | gpu1 | 1800.7 | 2.0 | epoch-boundary(199) |
| cpu-async | gpu1 | 1555.6 | 1.4 | epoch-boundary(172) |
| cpu-async | gpu2 | 1555.6 | 1.4 | epoch-boundary(172) |
| cpu-async | gpu1 | 287.6 | 1.1 | epoch-boundary(31) |
| cpu-async | gpu2 | 287.7 | 0.9 | epoch-boundary(31) |
| cpu-async | gpu0 | 1165.5 | 0.8 | epoch-boundary(129) |
| cpu-async | gpu1 | 1610.8 | 0.8 | epoch-boundary(178) |
| cpu-async | gpu1 | 160.5 | 0.8 | epoch-boundary(17) |
| cpu-async | gpu2 | 1610.7 | 0.7 | epoch-boundary(178) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | cpu-async | gpu0 | 1.5s | 0.0s | 0.5s | 0.0s | 2.0s |
| resnet-graph | cpu-async | gpu1 | 8.5s | 0.0s | 0.0s | 0.0s | 9.1s |
| resnet-graph | cpu-async | gpu2 | 7.7s | 0.0s | 0.0s | 0.0s | 8.3s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | cpu-async | 299 | 0 | 484 | 79.5 | 2281/9438 | 484 | 79.5 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | cpu-async | 192.7 | 10.7% |

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
| resnet-graph | cpu-async | 190 | 484 | 0 | 8.14e-3 | +1.29e-5 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | cpu-async | 0 | 484 | 8.58e-2 | 7.76e-2 | 2.46e-3 | 4.08e-1 | 26.0 | -2.04e-4 | 1.29e-3 |
| resnet-graph | cpu-async | 1 | 484 | 8.65e-2 | 7.83e-2 | 2.42e-3 | 3.88e-1 | 32.2 | -2.28e-4 | 1.33e-3 |
| resnet-graph | cpu-async | 2 | 484 | 8.74e-2 | 7.93e-2 | 2.31e-3 | 4.03e-1 | 41.7 | -2.37e-4 | 1.30e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | cpu-async | rank0 ↔ rank1 | +0.9968 |
| resnet-graph | cpu-async | rank0 ↔ rank2 | +0.9955 |
| resnet-graph | cpu-async | rank1 ↔ rank2 | +0.9956 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | cpu-async | 55 (2,3,4,5,6,7,8,9…148,149) | 0 (—) | — | 2,3,4,5,6,7,8,9…148,149 | — |

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
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 223 | +0.135 |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | 189 | +0.086 |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | 68 | +0.250 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | cpu-async | 482 | +0.032 | 189 | +0.203 | +0.266 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | cpu-async | 483 | 3.33e1–8.03e1 | 6.48e1 | 2.55e-3 | 3.85e-3 | 3.91e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 225 | 30–77854 | +8.890e-6 | 0.458 | +9.235e-6 | 0.491 | 96 | +6.358e-6 | 0.355 | 30–973 | +9.330e-4 | 0.732 |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 213 | 936–77854 | +9.973e-6 | 0.608 | +1.023e-5 | 0.631 | 95 | +6.528e-6 | 0.363 | 74–973 | +1.025e-3 | 0.927 |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | 190 | 78344–117136 | +8.220e-8 | 0.000 | -1.184e-7 | 0.000 | 50 | -1.689e-6 | 0.005 | 84–553 | +4.325e-4 | 0.052 |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | 69 | 117336–155647 | +1.988e-5 | 0.309 | +2.014e-5 | 0.323 | 44 | +1.169e-5 | 0.265 | 91–971 | +1.073e-3 | 0.571 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | +9.330e-4 | r0: +9.297e-4, r1: +9.373e-4, r2: +9.353e-4 | r0: 0.731, r1: 0.728, r2: 0.709 | 1.01× | ✓ |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +1.025e-3 | r0: +1.019e-3, r1: +1.026e-3, r2: +1.033e-3 | r0: 0.921, r1: 0.918, r2: 0.907 | 1.01× | ✓ |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | +4.325e-4 | r0: +3.843e-4, r1: +4.408e-4, r2: +4.715e-4 | r0: 0.041, r1: 0.051, r2: 0.060 | 1.23× | ✓ |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | +1.073e-3 | r0: +1.074e-3, r1: +1.062e-3, r2: +1.085e-3 | r0: 0.591, r1: 0.538, r2: 0.582 | 1.02× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | cpu-async | `█▇▇▇▇███████████████████▅▄▄▄▄▅▅▅▄▅▅▅▄▁▂▂▂▂▂▂▂▂▂▂` | `▁▆▇▇▇███████████████████▆▇▇████▇▆▇▇▇▃▅▇▇▇███████` |

### Per-Epoch Detail: resnet-graph / cpu-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 12 | 1.15e-1 | 4.08e-1 | 2.14e-1 | 1.51e-1 | 33 | -3.23e-2 | +1.33e-2 | -8.69e-3 | -6.29e-3 |
| 1 | 3.00e-1 | 7 | 1.01e-1 | 1.85e-1 | 1.28e-1 | 1.01e-1 | 33 | -6.50e-3 | +2.38e-3 | -2.25e-3 | -3.78e-3 |
| 2 | 3.00e-1 | 10 | 9.89e-2 | 1.40e-1 | 1.09e-1 | 1.11e-1 | 31 | -8.36e-3 | +4.64e-3 | -3.51e-4 | -1.25e-3 |
| 3 | 3.00e-1 | 6 | 1.05e-1 | 1.56e-1 | 1.17e-1 | 1.15e-1 | 33 | -1.36e-2 | +4.55e-3 | -1.13e-3 | -1.07e-3 |
| 4 | 3.00e-1 | 7 | 1.12e-1 | 1.55e-1 | 1.21e-1 | 1.14e-1 | 35 | -5.33e-3 | +3.38e-3 | -6.63e-4 | -8.14e-4 |
| 5 | 3.00e-1 | 9 | 1.08e-1 | 1.55e-1 | 1.18e-1 | 1.17e-1 | 35 | -8.91e-3 | +3.90e-3 | -5.72e-4 | -5.37e-4 |
| 6 | 3.00e-1 | 5 | 1.13e-1 | 1.59e-1 | 1.24e-1 | 1.13e-1 | 38 | -1.10e-2 | +3.94e-3 | -1.49e-3 | -9.12e-4 |
| 7 | 3.00e-1 | 7 | 1.14e-1 | 1.61e-1 | 1.25e-1 | 1.16e-1 | 39 | -7.13e-3 | +4.09e-3 | -6.92e-4 | -7.83e-4 |
| 8 | 3.00e-1 | 5 | 1.24e-1 | 1.57e-1 | 1.32e-1 | 1.32e-1 | 54 | -5.54e-3 | +2.83e-3 | +9.26e-6 | -4.84e-4 |
| 9 | 3.00e-1 | 6 | 1.16e-1 | 1.60e-1 | 1.28e-1 | 1.20e-1 | 43 | -4.77e-3 | +2.27e-3 | -7.32e-4 | -5.68e-4 |
| 10 | 3.00e-1 | 6 | 1.08e-1 | 1.62e-1 | 1.25e-1 | 1.08e-1 | 33 | -5.76e-3 | +3.38e-3 | -1.13e-3 | -8.64e-4 |
| 11 | 3.00e-1 | 8 | 1.07e-1 | 1.57e-1 | 1.17e-1 | 1.13e-1 | 39 | -9.20e-3 | +4.49e-3 | -7.01e-4 | -6.74e-4 |
| 12 | 3.00e-1 | 5 | 1.16e-1 | 1.56e-1 | 1.28e-1 | 1.19e-1 | 42 | -4.63e-3 | +3.74e-3 | -3.11e-4 | -5.84e-4 |
| 13 | 3.00e-1 | 7 | 1.10e-1 | 1.64e-1 | 1.21e-1 | 1.10e-1 | 32 | -6.60e-3 | +3.30e-3 | -1.01e-3 | -7.61e-4 |
| 14 | 3.00e-1 | 7 | 1.04e-1 | 1.54e-1 | 1.14e-1 | 1.07e-1 | 32 | -8.75e-3 | +4.18e-3 | -9.01e-4 | -7.55e-4 |
| 15 | 3.00e-1 | 6 | 1.03e-1 | 1.55e-1 | 1.21e-1 | 1.26e-1 | 46 | -9.32e-3 | +4.45e-3 | -5.34e-4 | -5.61e-4 |
| 16 | 3.00e-1 | 7 | 1.19e-1 | 1.66e-1 | 1.29e-1 | 1.24e-1 | 51 | -5.98e-3 | +3.00e-3 | -5.44e-4 | -5.00e-4 |
| 17 | 3.00e-1 | 4 | 1.28e-1 | 1.55e-1 | 1.36e-1 | 1.30e-1 | 49 | -3.46e-3 | +2.63e-3 | -2.44e-4 | -4.27e-4 |
| 18 | 3.00e-1 | 8 | 1.11e-1 | 1.60e-1 | 1.20e-1 | 1.17e-1 | 39 | -8.16e-3 | +2.43e-3 | -7.70e-4 | -5.03e-4 |
| 19 | 3.00e-1 | 1 | 1.19e-1 | 1.19e-1 | 1.19e-1 | 1.19e-1 | 41 | +4.52e-4 | +4.52e-4 | +4.52e-4 | -4.07e-4 |
| 20 | 3.00e-1 | 1 | 2.33e-1 | 2.33e-1 | 2.33e-1 | 2.33e-1 | 366 | +1.84e-3 | +1.84e-3 | +1.84e-3 | -1.83e-4 |
| 21 | 3.00e-1 | 1 | 2.39e-1 | 2.39e-1 | 2.39e-1 | 2.39e-1 | 312 | +7.73e-5 | +7.73e-5 | +7.73e-5 | -1.57e-4 |
| 22 | 3.00e-1 | 1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 291 | -1.27e-4 | -1.27e-4 | -1.27e-4 | -1.54e-4 |
| 23 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 272 | -1.79e-4 | -1.79e-4 | -1.79e-4 | -1.56e-4 |
| 24 | 3.00e-1 | 1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 285 | +3.35e-5 | +3.35e-5 | +3.35e-5 | -1.37e-4 |
| 25 | 3.00e-1 | 1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 283 | -1.68e-6 | -1.68e-6 | -1.68e-6 | -1.24e-4 |
| 26 | 3.00e-1 | 1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 275 | +7.06e-6 | +7.06e-6 | +7.06e-6 | -1.11e-4 |
| 27 | 3.00e-1 | 1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 298 | +6.16e-5 | +6.16e-5 | +6.16e-5 | -9.35e-5 |
| 28 | 3.00e-1 | 1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 310 | -3.29e-6 | -3.29e-6 | -3.29e-6 | -8.45e-5 |
| 29 | 3.00e-1 | 1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 298 | +3.93e-6 | +3.93e-6 | +3.93e-6 | -7.56e-5 |
| 30 | 3.00e-1 | 1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 292 | -3.85e-5 | -3.85e-5 | -3.85e-5 | -7.19e-5 |
| 31 | 3.00e-1 | 1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 281 | -1.46e-5 | -1.46e-5 | -1.46e-5 | -6.62e-5 |
| 32 | 3.00e-1 | 1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 281 | -2.38e-5 | -2.38e-5 | -2.38e-5 | -6.20e-5 |
| 34 | 3.00e-1 | 2 | 2.21e-1 | 2.31e-1 | 2.26e-1 | 2.21e-1 | 281 | -1.62e-4 | +1.41e-4 | -1.05e-5 | -5.37e-5 |
| 36 | 3.00e-1 | 1 | 2.37e-1 | 2.37e-1 | 2.37e-1 | 2.37e-1 | 345 | +2.02e-4 | +2.02e-4 | +2.02e-4 | -2.81e-5 |
| 37 | 3.00e-1 | 1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 281 | -1.58e-4 | -1.58e-4 | -1.58e-4 | -4.11e-5 |
| 38 | 3.00e-1 | 1 | 2.34e-1 | 2.34e-1 | 2.34e-1 | 2.34e-1 | 312 | +1.00e-4 | +1.00e-4 | +1.00e-4 | -2.70e-5 |
| 39 | 3.00e-1 | 1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 307 | -2.13e-5 | -2.13e-5 | -2.13e-5 | -2.64e-5 |
| 40 | 3.00e-1 | 1 | 2.38e-1 | 2.38e-1 | 2.38e-1 | 2.38e-1 | 318 | +8.21e-5 | +8.21e-5 | +8.21e-5 | -1.56e-5 |
| 41 | 3.00e-1 | 1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 273 | -2.00e-4 | -2.00e-4 | -2.00e-4 | -3.40e-5 |
| 42 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 245 | -1.73e-4 | -1.73e-4 | -1.73e-4 | -4.80e-5 |
| 43 | 3.00e-1 | 1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 294 | +8.91e-5 | +8.91e-5 | +8.91e-5 | -3.42e-5 |
| 44 | 3.00e-1 | 1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 289 | +4.21e-5 | +4.21e-5 | +4.21e-5 | -2.66e-5 |
| 45 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 278 | -7.24e-5 | -7.24e-5 | -7.24e-5 | -3.12e-5 |
| 46 | 3.00e-1 | 1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 272 | +6.05e-5 | +6.05e-5 | +6.05e-5 | -2.20e-5 |
| 47 | 3.00e-1 | 1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 277 | +3.60e-5 | +3.60e-5 | +3.60e-5 | -1.62e-5 |
| 48 | 3.00e-1 | 1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 286 | +4.35e-5 | +4.35e-5 | +4.35e-5 | -1.02e-5 |
| 49 | 3.00e-1 | 1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 268 | -8.59e-5 | -8.59e-5 | -8.59e-5 | -1.78e-5 |
| 50 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 256 | -7.13e-5 | -7.13e-5 | -7.13e-5 | -2.32e-5 |
| 51 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 262 | -1.55e-5 | -1.55e-5 | -1.55e-5 | -2.24e-5 |
| 52 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 247 | +2.82e-5 | +2.82e-5 | +2.82e-5 | -1.73e-5 |
| 53 | 3.00e-1 | 1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 259 | +3.58e-5 | +3.58e-5 | +3.58e-5 | -1.20e-5 |
| 54 | 3.00e-1 | 1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 263 | -1.29e-5 | -1.29e-5 | -1.29e-5 | -1.21e-5 |
| 55 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 250 | -2.57e-5 | -2.57e-5 | -2.57e-5 | -1.35e-5 |
| 56 | 3.00e-1 | 2 | 2.12e-1 | 2.23e-1 | 2.18e-1 | 2.12e-1 | 228 | -2.32e-4 | +5.82e-5 | -8.68e-5 | -2.89e-5 |
| 58 | 3.00e-1 | 2 | 2.18e-1 | 2.29e-1 | 2.23e-1 | 2.18e-1 | 228 | -2.19e-4 | +2.79e-4 | +2.95e-5 | -2.02e-5 |
| 59 | 3.00e-1 | 1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 291 | +2.21e-4 | +2.21e-4 | +2.21e-4 | +3.91e-6 |
| 60 | 3.00e-1 | 1 | 2.31e-1 | 2.31e-1 | 2.31e-1 | 2.31e-1 | 267 | -2.13e-5 | -2.13e-5 | -2.13e-5 | +1.39e-6 |
| 61 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 233 | -2.76e-4 | -2.76e-4 | -2.76e-4 | -2.63e-5 |
| 62 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 239 | +2.49e-5 | +2.49e-5 | +2.49e-5 | -2.12e-5 |
| 63 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 241 | +4.70e-6 | +4.70e-6 | +4.70e-6 | -1.86e-5 |
| 64 | 3.00e-1 | 1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 263 | +9.01e-5 | +9.01e-5 | +9.01e-5 | -7.73e-6 |
| 65 | 3.00e-1 | 1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 262 | +5.59e-5 | +5.59e-5 | +5.59e-5 | -1.37e-6 |
| 66 | 3.00e-1 | 1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 266 | +3.87e-5 | +3.87e-5 | +3.87e-5 | +2.64e-6 |
| 67 | 3.00e-1 | 2 | 2.06e-1 | 2.12e-1 | 2.09e-1 | 2.06e-1 | 203 | -3.70e-4 | -1.42e-4 | -2.56e-4 | -4.54e-5 |
| 68 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 224 | +1.26e-4 | +1.26e-4 | +1.26e-4 | -2.83e-5 |
| 69 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 247 | +1.31e-4 | +1.31e-4 | +1.31e-4 | -1.24e-5 |
| 70 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 257 | +3.57e-5 | +3.57e-5 | +3.57e-5 | -7.58e-6 |
| 71 | 3.00e-1 | 1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 281 | +1.44e-4 | +1.44e-4 | +1.44e-4 | +7.56e-6 |
| 72 | 3.00e-1 | 2 | 2.12e-1 | 2.28e-1 | 2.20e-1 | 2.12e-1 | 209 | -3.61e-4 | -2.05e-5 | -1.91e-4 | -3.18e-5 |
| 73 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 220 | +3.77e-6 | +3.77e-6 | +3.77e-6 | -2.83e-5 |
| 74 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 228 | +1.31e-4 | +1.31e-4 | +1.31e-4 | -1.24e-5 |
| 75 | 3.00e-1 | 2 | 2.06e-1 | 2.15e-1 | 2.11e-1 | 2.06e-1 | 194 | -2.04e-4 | -7.57e-5 | -1.40e-4 | -3.72e-5 |
| 76 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 220 | +1.07e-4 | +1.07e-4 | +1.07e-4 | -2.28e-5 |
| 77 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 229 | +1.64e-4 | +1.64e-4 | +1.64e-4 | -4.07e-6 |
| 78 | 3.00e-1 | 1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 248 | +1.36e-4 | +1.36e-4 | +1.36e-4 | +9.98e-6 |
| 79 | 3.00e-1 | 2 | 2.08e-1 | 2.16e-1 | 2.12e-1 | 2.08e-1 | 194 | -2.36e-4 | -1.93e-4 | -2.14e-4 | -3.24e-5 |
| 80 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 207 | +4.92e-5 | +4.92e-5 | +4.92e-5 | -2.43e-5 |
| 81 | 3.00e-1 | 1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 235 | +2.85e-4 | +2.85e-4 | +2.85e-4 | +6.61e-6 |
| 82 | 3.00e-1 | 2 | 2.06e-1 | 2.14e-1 | 2.10e-1 | 2.06e-1 | 194 | -2.23e-4 | -1.96e-4 | -2.10e-4 | -3.44e-5 |
| 84 | 3.00e-1 | 2 | 2.07e-1 | 2.33e-1 | 2.20e-1 | 2.07e-1 | 194 | -6.10e-4 | +4.52e-4 | -7.91e-5 | -4.82e-5 |
| 85 | 3.00e-1 | 1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 245 | +3.85e-4 | +3.85e-4 | +3.85e-4 | -4.81e-6 |
| 86 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 212 | -3.07e-4 | -3.07e-4 | -3.07e-4 | -3.50e-5 |
| 87 | 3.00e-1 | 2 | 1.96e-1 | 2.05e-1 | 2.00e-1 | 1.96e-1 | 166 | -2.68e-4 | -2.14e-4 | -2.41e-4 | -7.44e-5 |
| 88 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 182 | +1.59e-4 | +1.59e-4 | +1.59e-4 | -5.11e-5 |
| 89 | 3.00e-1 | 2 | 2.04e-1 | 2.07e-1 | 2.06e-1 | 2.04e-1 | 177 | -9.29e-5 | +1.59e-4 | +3.30e-5 | -3.64e-5 |
| 90 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 219 | +2.30e-4 | +2.30e-4 | +2.30e-4 | -9.72e-6 |
| 91 | 3.00e-1 | 2 | 2.00e-1 | 2.06e-1 | 2.03e-1 | 2.00e-1 | 167 | -2.13e-4 | -1.76e-4 | -1.95e-4 | -4.47e-5 |
| 92 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 199 | +2.65e-4 | +2.65e-4 | +2.65e-4 | -1.37e-5 |
| 93 | 3.00e-1 | 2 | 1.99e-1 | 2.14e-1 | 2.06e-1 | 1.99e-1 | 167 | -4.41e-4 | +6.55e-5 | -1.88e-4 | -4.93e-5 |
| 94 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 187 | +1.79e-4 | +1.79e-4 | +1.79e-4 | -2.64e-5 |
| 95 | 3.00e-1 | 2 | 2.01e-1 | 2.12e-1 | 2.06e-1 | 2.01e-1 | 167 | -3.28e-4 | +1.53e-4 | -8.77e-5 | -4.05e-5 |
| 96 | 3.00e-1 | 2 | 1.99e-1 | 2.14e-1 | 2.06e-1 | 1.99e-1 | 157 | -4.79e-4 | +3.08e-4 | -8.55e-5 | -5.30e-5 |
| 97 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 182 | +6.65e-5 | +6.65e-5 | +6.65e-5 | -4.10e-5 |
| 98 | 3.00e-1 | 2 | 1.96e-1 | 2.11e-1 | 2.04e-1 | 1.96e-1 | 157 | -4.45e-4 | +2.41e-4 | -1.02e-4 | -5.61e-5 |
| 99 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 184 | +2.52e-4 | +2.52e-4 | +2.52e-4 | -2.53e-5 |
| 100 | 3.00e-2 | 2 | 1.02e-1 | 2.03e-1 | 1.52e-1 | 1.02e-1 | 157 | -4.38e-3 | -8.85e-5 | -2.23e-3 | -4.66e-4 |
| 101 | 3.00e-2 | 1 | 5.37e-2 | 5.37e-2 | 5.37e-2 | 5.37e-2 | 187 | -3.43e-3 | -3.43e-3 | -3.43e-3 | -7.62e-4 |
| 102 | 3.00e-2 | 3 | 2.21e-2 | 3.17e-2 | 2.59e-2 | 2.21e-2 | 146 | -3.09e-3 | -5.05e-4 | -1.86e-3 | -1.04e-3 |
| 103 | 3.00e-2 | 1 | 2.27e-2 | 2.27e-2 | 2.27e-2 | 2.27e-2 | 184 | +1.60e-4 | +1.60e-4 | +1.60e-4 | -9.16e-4 |
| 104 | 3.00e-2 | 2 | 2.33e-2 | 2.36e-2 | 2.35e-2 | 2.36e-2 | 162 | +7.46e-5 | +1.37e-4 | +1.06e-4 | -7.22e-4 |
| 105 | 3.00e-2 | 1 | 2.60e-2 | 2.60e-2 | 2.60e-2 | 2.60e-2 | 197 | +4.90e-4 | +4.90e-4 | +4.90e-4 | -6.01e-4 |
| 106 | 3.00e-2 | 2 | 2.33e-2 | 2.54e-2 | 2.44e-2 | 2.33e-2 | 140 | -6.08e-4 | -1.22e-4 | -3.65e-4 | -5.58e-4 |
| 107 | 3.00e-2 | 2 | 2.38e-2 | 2.64e-2 | 2.51e-2 | 2.38e-2 | 136 | -7.79e-4 | +6.97e-4 | -4.07e-5 | -4.67e-4 |
| 108 | 3.00e-2 | 2 | 2.36e-2 | 2.43e-2 | 2.39e-2 | 2.36e-2 | 127 | -2.06e-4 | +1.34e-4 | -3.60e-5 | -3.87e-4 |
| 109 | 3.00e-2 | 2 | 2.43e-2 | 2.52e-2 | 2.47e-2 | 2.43e-2 | 120 | -2.93e-4 | +4.00e-4 | +5.34e-5 | -3.07e-4 |
| 110 | 3.00e-2 | 2 | 2.50e-2 | 2.57e-2 | 2.54e-2 | 2.50e-2 | 129 | -2.08e-4 | +3.84e-4 | +8.83e-5 | -2.35e-4 |
| 111 | 3.00e-2 | 2 | 2.58e-2 | 2.76e-2 | 2.67e-2 | 2.58e-2 | 129 | -5.10e-4 | +5.83e-4 | +3.66e-5 | -1.89e-4 |
| 112 | 3.00e-2 | 2 | 2.55e-2 | 2.75e-2 | 2.65e-2 | 2.55e-2 | 122 | -6.22e-4 | +3.90e-4 | -1.16e-4 | -1.80e-4 |
| 113 | 3.00e-2 | 2 | 2.68e-2 | 2.86e-2 | 2.77e-2 | 2.68e-2 | 116 | -5.78e-4 | +7.37e-4 | +7.94e-5 | -1.37e-4 |
| 114 | 3.00e-2 | 2 | 2.61e-2 | 3.00e-2 | 2.81e-2 | 2.61e-2 | 107 | -1.33e-3 | +6.96e-4 | -3.15e-4 | -1.81e-4 |
| 115 | 3.00e-2 | 3 | 2.54e-2 | 2.87e-2 | 2.67e-2 | 2.59e-2 | 102 | -1.29e-3 | +6.77e-4 | -1.31e-4 | -1.71e-4 |
| 116 | 3.00e-2 | 2 | 2.67e-2 | 2.89e-2 | 2.78e-2 | 2.67e-2 | 98 | -7.85e-4 | +7.64e-4 | -1.08e-5 | -1.48e-4 |
| 117 | 3.00e-2 | 3 | 2.64e-2 | 2.83e-2 | 2.70e-2 | 2.64e-2 | 96 | -7.33e-4 | +4.20e-4 | -1.04e-4 | -1.40e-4 |
| 118 | 3.00e-2 | 2 | 2.73e-2 | 3.01e-2 | 2.87e-2 | 2.73e-2 | 96 | -1.04e-3 | +1.04e-3 | -7.72e-7 | -1.24e-4 |
| 119 | 3.00e-2 | 2 | 2.76e-2 | 2.97e-2 | 2.86e-2 | 2.76e-2 | 103 | -7.23e-4 | +6.57e-4 | -3.30e-5 | -1.14e-4 |
| 120 | 3.00e-2 | 4 | 2.86e-2 | 3.15e-2 | 2.95e-2 | 2.89e-2 | 105 | -7.86e-4 | +9.06e-4 | +1.38e-5 | -7.71e-5 |
| 121 | 3.00e-2 | 2 | 2.80e-2 | 3.23e-2 | 3.02e-2 | 2.80e-2 | 79 | -1.84e-3 | +8.19e-4 | -5.12e-4 | -1.73e-4 |
| 122 | 3.00e-2 | 3 | 2.62e-2 | 3.00e-2 | 2.76e-2 | 2.62e-2 | 75 | -1.46e-3 | +6.16e-4 | -3.71e-4 | -2.35e-4 |
| 123 | 3.00e-2 | 3 | 2.65e-2 | 3.17e-2 | 2.84e-2 | 2.65e-2 | 75 | -2.07e-3 | +1.52e-3 | -3.01e-4 | -2.70e-4 |
| 124 | 3.00e-2 | 3 | 2.68e-2 | 3.07e-2 | 2.81e-2 | 2.68e-2 | 75 | -1.73e-3 | +1.34e-3 | -1.64e-4 | -2.54e-4 |
| 125 | 3.00e-2 | 4 | 2.73e-2 | 3.07e-2 | 2.84e-2 | 2.73e-2 | 79 | -1.26e-3 | +1.28e-3 | -5.58e-5 | -1.99e-4 |
| 126 | 3.00e-2 | 3 | 2.92e-2 | 3.23e-2 | 3.04e-2 | 2.92e-2 | 75 | -1.12e-3 | +1.51e-3 | +7.60e-5 | -1.40e-4 |
| 127 | 3.00e-2 | 5 | 2.60e-2 | 3.27e-2 | 2.89e-2 | 2.60e-2 | 71 | -1.46e-3 | +9.90e-4 | -4.25e-4 | -2.80e-4 |
| 128 | 3.00e-2 | 2 | 2.89e-2 | 3.11e-2 | 3.00e-2 | 2.89e-2 | 70 | -1.04e-3 | +1.83e-3 | +3.92e-4 | -1.67e-4 |
| 129 | 3.00e-2 | 5 | 2.54e-2 | 3.21e-2 | 2.74e-2 | 2.56e-2 | 56 | -2.60e-3 | +1.14e-3 | -5.42e-4 | -3.17e-4 |
| 130 | 3.00e-2 | 5 | 2.48e-2 | 3.10e-2 | 2.69e-2 | 2.48e-2 | 49 | -2.97e-3 | +2.30e-3 | -3.72e-4 | -3.72e-4 |
| 131 | 3.00e-2 | 5 | 2.28e-2 | 3.23e-2 | 2.54e-2 | 2.36e-2 | 43 | -6.00e-3 | +2.80e-3 | -8.97e-4 | -5.69e-4 |
| 132 | 3.00e-2 | 8 | 2.18e-2 | 3.10e-2 | 2.43e-2 | 2.18e-2 | 37 | -4.98e-3 | +3.36e-3 | -6.67e-4 | -6.48e-4 |
| 133 | 3.00e-2 | 5 | 2.23e-2 | 2.98e-2 | 2.45e-2 | 2.23e-2 | 37 | -5.67e-3 | +4.16e-3 | -7.34e-4 | -7.16e-4 |
| 134 | 3.00e-2 | 6 | 2.16e-2 | 3.04e-2 | 2.51e-2 | 2.16e-2 | 31 | -5.34e-3 | +3.75e-3 | -1.02e-3 | -9.55e-4 |
| 135 | 3.00e-2 | 6 | 2.29e-2 | 3.19e-2 | 2.55e-2 | 2.62e-2 | 46 | -1.13e-2 | +4.75e-3 | -5.88e-4 | -6.97e-4 |
| 136 | 3.00e-2 | 6 | 2.69e-2 | 3.32e-2 | 2.97e-2 | 3.09e-2 | 59 | -3.80e-3 | +2.89e-3 | +1.21e-4 | -2.97e-4 |
| 137 | 3.00e-2 | 4 | 2.88e-2 | 3.85e-2 | 3.23e-2 | 2.88e-2 | 52 | -3.81e-3 | +2.21e-3 | -8.42e-4 | -5.11e-4 |
| 138 | 3.00e-2 | 6 | 2.80e-2 | 3.49e-2 | 2.98e-2 | 2.83e-2 | 52 | -2.83e-3 | +2.32e-3 | -2.83e-4 | -4.10e-4 |
| 139 | 3.00e-2 | 4 | 2.81e-2 | 3.66e-2 | 3.06e-2 | 2.85e-2 | 45 | -5.13e-3 | +3.12e-3 | -6.14e-4 | -4.94e-4 |
| 140 | 3.00e-2 | 6 | 2.72e-2 | 3.71e-2 | 2.94e-2 | 2.72e-2 | 45 | -5.61e-3 | +3.05e-3 | -6.38e-4 | -5.55e-4 |
| 141 | 3.00e-2 | 6 | 2.52e-2 | 3.45e-2 | 2.81e-2 | 2.70e-2 | 41 | -4.29e-3 | +2.90e-3 | -5.79e-4 | -5.29e-4 |
| 142 | 3.00e-2 | 8 | 2.48e-2 | 3.87e-2 | 2.87e-2 | 2.48e-2 | 35 | -6.46e-3 | +4.28e-3 | -8.75e-4 | -7.42e-4 |
| 143 | 3.00e-2 | 6 | 2.48e-2 | 3.53e-2 | 2.73e-2 | 2.50e-2 | 34 | -7.71e-3 | +5.01e-3 | -9.26e-4 | -8.23e-4 |
| 144 | 3.00e-2 | 7 | 2.53e-2 | 3.59e-2 | 2.76e-2 | 2.61e-2 | 36 | -7.67e-3 | +4.87e-3 | -6.50e-4 | -6.97e-4 |
| 145 | 3.00e-2 | 9 | 2.46e-2 | 3.94e-2 | 2.89e-2 | 3.31e-2 | 52 | -8.08e-3 | +4.49e-3 | -4.72e-4 | -3.15e-4 |
| 146 | 3.00e-2 | 2 | 3.65e-2 | 4.62e-2 | 4.13e-2 | 3.65e-2 | 58 | -4.07e-3 | +3.20e-3 | -4.34e-4 | -3.74e-4 |
| 147 | 3.00e-2 | 5 | 3.07e-2 | 4.20e-2 | 3.49e-2 | 3.14e-2 | 45 | -2.18e-3 | +1.62e-3 | -7.66e-4 | -5.48e-4 |
| 148 | 3.00e-2 | 6 | 3.00e-2 | 4.16e-2 | 3.36e-2 | 3.31e-2 | 47 | -5.57e-3 | +3.27e-3 | -3.49e-4 | -4.20e-4 |
| 149 | 3.00e-2 | 6 | 2.98e-2 | 4.37e-2 | 3.34e-2 | 3.03e-2 | 40 | -5.50e-3 | +3.07e-3 | -9.21e-4 | -6.37e-4 |
| 150 | 3.00e-3 | 6 | 2.50e-3 | 2.62e-2 | 8.74e-3 | 2.50e-3 | 33 | -2.31e-2 | -1.75e-3 | -1.11e-2 | -5.35e-3 |
| 151 | 3.00e-3 | 6 | 3.00e-3 | 3.73e-3 | 3.26e-3 | 3.33e-3 | 57 | -3.53e-3 | +4.48e-3 | +3.23e-4 | -2.71e-3 |
| 152 | 3.00e-3 | 4 | 2.96e-3 | 3.84e-3 | 3.31e-3 | 2.96e-3 | 51 | -2.57e-3 | +1.69e-3 | -8.22e-4 | -2.08e-3 |
| 153 | 3.00e-3 | 7 | 2.56e-3 | 3.93e-3 | 2.90e-3 | 2.62e-3 | 40 | -5.91e-3 | +2.93e-3 | -9.62e-4 | -1.45e-3 |
| 154 | 3.00e-3 | 1 | 2.82e-3 | 2.82e-3 | 2.82e-3 | 2.82e-3 | 44 | +1.63e-3 | +1.63e-3 | +1.63e-3 | -1.14e-3 |
| 155 | 3.00e-3 | 1 | 6.65e-3 | 6.65e-3 | 6.65e-3 | 6.65e-3 | 344 | +2.50e-3 | +2.50e-3 | +2.50e-3 | -7.79e-4 |
| 156 | 3.00e-3 | 1 | 7.10e-3 | 7.10e-3 | 7.10e-3 | 7.10e-3 | 300 | +2.21e-4 | +2.21e-4 | +2.21e-4 | -6.79e-4 |
| 157 | 3.00e-3 | 1 | 7.32e-3 | 7.32e-3 | 7.32e-3 | 7.32e-3 | 304 | +9.97e-5 | +9.97e-5 | +9.97e-5 | -6.01e-4 |
| 158 | 3.00e-3 | 1 | 7.54e-3 | 7.54e-3 | 7.54e-3 | 7.54e-3 | 338 | +8.68e-5 | +8.68e-5 | +8.68e-5 | -5.32e-4 |
| 160 | 3.00e-3 | 2 | 7.20e-3 | 7.61e-3 | 7.41e-3 | 7.20e-3 | 285 | -1.94e-4 | +2.57e-5 | -8.42e-5 | -4.48e-4 |
| 162 | 3.00e-3 | 1 | 7.58e-3 | 7.58e-3 | 7.58e-3 | 7.58e-3 | 335 | +1.55e-4 | +1.55e-4 | +1.55e-4 | -3.88e-4 |
| 163 | 3.00e-3 | 1 | 7.43e-3 | 7.43e-3 | 7.43e-3 | 7.43e-3 | 287 | -7.31e-5 | -7.31e-5 | -7.31e-5 | -3.56e-4 |
| 164 | 3.00e-3 | 1 | 7.27e-3 | 7.27e-3 | 7.27e-3 | 7.27e-3 | 286 | -7.53e-5 | -7.53e-5 | -7.53e-5 | -3.28e-4 |
| 165 | 3.00e-3 | 1 | 7.23e-3 | 7.23e-3 | 7.23e-3 | 7.23e-3 | 277 | -2.11e-5 | -2.11e-5 | -2.11e-5 | -2.97e-4 |
| 166 | 3.00e-3 | 1 | 7.17e-3 | 7.17e-3 | 7.17e-3 | 7.17e-3 | 275 | -2.66e-5 | -2.66e-5 | -2.66e-5 | -2.70e-4 |
| 167 | 3.00e-3 | 1 | 7.19e-3 | 7.19e-3 | 7.19e-3 | 7.19e-3 | 282 | +6.73e-6 | +6.73e-6 | +6.73e-6 | -2.43e-4 |
| 168 | 3.00e-3 | 1 | 7.30e-3 | 7.30e-3 | 7.30e-3 | 7.30e-3 | 283 | +5.60e-5 | +5.60e-5 | +5.60e-5 | -2.13e-4 |
| 169 | 3.00e-3 | 1 | 7.38e-3 | 7.38e-3 | 7.38e-3 | 7.38e-3 | 277 | +3.91e-5 | +3.91e-5 | +3.91e-5 | -1.88e-4 |
| 170 | 3.00e-3 | 1 | 7.51e-3 | 7.51e-3 | 7.51e-3 | 7.51e-3 | 291 | +5.77e-5 | +5.77e-5 | +5.77e-5 | -1.63e-4 |
| 171 | 3.00e-3 | 1 | 7.43e-3 | 7.43e-3 | 7.43e-3 | 7.43e-3 | 302 | -3.53e-5 | -3.53e-5 | -3.53e-5 | -1.50e-4 |
| 172 | 3.00e-3 | 1 | 7.12e-3 | 7.12e-3 | 7.12e-3 | 7.12e-3 | 267 | -1.57e-4 | -1.57e-4 | -1.57e-4 | -1.51e-4 |
| 173 | 3.00e-3 | 1 | 6.98e-3 | 6.98e-3 | 6.98e-3 | 6.98e-3 | 253 | -7.87e-5 | -7.87e-5 | -7.87e-5 | -1.44e-4 |
| 174 | 3.00e-3 | 1 | 7.24e-3 | 7.24e-3 | 7.24e-3 | 7.24e-3 | 259 | +1.40e-4 | +1.40e-4 | +1.40e-4 | -1.15e-4 |
| 175 | 3.00e-3 | 1 | 7.39e-3 | 7.39e-3 | 7.39e-3 | 7.39e-3 | 281 | +7.36e-5 | +7.36e-5 | +7.36e-5 | -9.65e-5 |
| 176 | 3.00e-3 | 1 | 7.18e-3 | 7.18e-3 | 7.18e-3 | 7.18e-3 | 262 | -1.09e-4 | -1.09e-4 | -1.09e-4 | -9.78e-5 |
| 177 | 3.00e-3 | 1 | 7.67e-3 | 7.67e-3 | 7.67e-3 | 7.67e-3 | 306 | +2.16e-4 | +2.16e-4 | +2.16e-4 | -6.64e-5 |
| 178 | 3.00e-3 | 1 | 7.42e-3 | 7.42e-3 | 7.42e-3 | 7.42e-3 | 268 | -1.23e-4 | -1.23e-4 | -1.23e-4 | -7.20e-5 |
| 180 | 3.00e-3 | 2 | 7.55e-3 | 8.18e-3 | 7.87e-3 | 7.55e-3 | 250 | -3.20e-4 | +2.79e-4 | -2.08e-5 | -6.53e-5 |
| 182 | 3.00e-3 | 2 | 7.27e-3 | 7.81e-3 | 7.54e-3 | 7.27e-3 | 252 | -2.89e-4 | +1.02e-4 | -9.33e-5 | -7.26e-5 |
| 184 | 3.00e-3 | 2 | 7.20e-3 | 8.19e-3 | 7.69e-3 | 7.20e-3 | 225 | -5.75e-4 | +3.83e-4 | -9.56e-5 | -8.17e-5 |
| 185 | 3.00e-3 | 1 | 7.24e-3 | 7.24e-3 | 7.24e-3 | 7.24e-3 | 250 | +2.26e-5 | +2.26e-5 | +2.26e-5 | -7.13e-5 |
| 186 | 3.00e-3 | 1 | 7.42e-3 | 7.42e-3 | 7.42e-3 | 7.42e-3 | 258 | +9.72e-5 | +9.72e-5 | +9.72e-5 | -5.44e-5 |
| 187 | 3.00e-3 | 1 | 7.09e-3 | 7.09e-3 | 7.09e-3 | 7.09e-3 | 235 | -1.94e-4 | -1.94e-4 | -1.94e-4 | -6.84e-5 |
| 188 | 3.00e-3 | 1 | 7.21e-3 | 7.21e-3 | 7.21e-3 | 7.21e-3 | 254 | +6.67e-5 | +6.67e-5 | +6.67e-5 | -5.49e-5 |
| 189 | 3.00e-3 | 1 | 7.43e-3 | 7.43e-3 | 7.43e-3 | 7.43e-3 | 259 | +1.15e-4 | +1.15e-4 | +1.15e-4 | -3.79e-5 |
| 190 | 3.00e-3 | 1 | 7.12e-3 | 7.12e-3 | 7.12e-3 | 7.12e-3 | 242 | -1.74e-4 | -1.74e-4 | -1.74e-4 | -5.15e-5 |
| 191 | 3.00e-3 | 2 | 6.73e-3 | 7.04e-3 | 6.88e-3 | 6.73e-3 | 214 | -2.10e-4 | -4.98e-5 | -1.30e-4 | -6.72e-5 |
| 192 | 3.00e-3 | 1 | 7.20e-3 | 7.20e-3 | 7.20e-3 | 7.20e-3 | 243 | +2.77e-4 | +2.77e-4 | +2.77e-4 | -3.27e-5 |
| 193 | 3.00e-3 | 1 | 7.17e-3 | 7.17e-3 | 7.17e-3 | 7.17e-3 | 226 | -1.72e-5 | -1.72e-5 | -1.72e-5 | -3.12e-5 |
| 194 | 3.00e-3 | 1 | 7.28e-3 | 7.28e-3 | 7.28e-3 | 7.28e-3 | 232 | +6.45e-5 | +6.45e-5 | +6.45e-5 | -2.16e-5 |
| 195 | 3.00e-3 | 1 | 7.43e-3 | 7.43e-3 | 7.43e-3 | 7.43e-3 | 260 | +8.03e-5 | +8.03e-5 | +8.03e-5 | -1.14e-5 |
| 196 | 3.00e-3 | 1 | 8.01e-3 | 8.01e-3 | 8.01e-3 | 8.01e-3 | 299 | +2.50e-4 | +2.50e-4 | +2.50e-4 | +1.47e-5 |
| 197 | 3.00e-3 | 2 | 7.07e-3 | 7.45e-3 | 7.26e-3 | 7.07e-3 | 211 | -2.84e-4 | -2.47e-4 | -2.66e-4 | -3.84e-5 |
| 199 | 3.00e-3 | 1 | 8.14e-3 | 8.14e-3 | 8.14e-3 | 8.14e-3 | 295 | +4.74e-4 | +4.74e-4 | +4.74e-4 | +1.29e-5 |

