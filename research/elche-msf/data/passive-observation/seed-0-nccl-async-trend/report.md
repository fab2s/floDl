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
| nccl-async | 0.058156 | 0.9189 | +0.0064 | 1745.5 | 631 | 37.5 | 100% | 100% | 100% | 4.9 |

## Best Mode per Model

| Model | Best Eval | Mode | Fastest (within 2% of solo-0) | Mode |
|-------|-----------|------|-------------------------------|------|
| resnet-graph | 0.9189 | nccl-async | - | - |

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
| nccl-async | 1.9935 | 0.6864 | 0.6115 | 0.5621 | 0.5363 | 0.4969 | 0.4971 | 0.4684 | 0.4619 | 0.4467 | 0.1980 | 0.1545 | 0.1504 | 0.1418 | 0.1324 | 0.0728 | 0.0670 | 0.0616 | 0.0588 | 0.0582 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | nccl-async | 0 | cuda:0 | 0.4030 | 2.8 |
| resnet-graph | nccl-async | 1 | cuda:1 | 0.2997 | 3.8 |
| resnet-graph | nccl-async | 2 | cuda:2 | 0.2973 | 3.7 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) | GPU2 Peak (MB) | GPU2 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|---------------|---------------|
| resnet-graph | nccl-async | 356 | 355 | 388 | 382 | 386 | 381 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| nccl-async | gpu1 | 1744.6 | 0.9 | epoch-boundary(199) |
| nccl-async | gpu2 | 1744.6 | 0.9 | epoch-boundary(199) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | nccl-async | gpu0 | 0.0s | 0.0s | 0.0s | 0.0s | 0.9s |
| resnet-graph | nccl-async | gpu1 | 0.9s | 0.0s | 0.0s | 0.0s | 2.2s |
| resnet-graph | nccl-async | gpu2 | 0.9s | 0.0s | 0.0s | 0.0s | 1.8s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | nccl-async | 388 | 0 | 631 | 37.5 | 1662/7654 | 0 | 0.0 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | nccl-async | 166.7 | 9.6% |

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
| resnet-graph | nccl-async | 196 | 631 | 0 | 5.49e-3 | -2.63e-5 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | nccl-async | 0 | 631 | 6.25e-2 | 6.82e-2 | 0.00e0 | 5.29e-1 | 44.2 | -1.50e-4 | 3.45e-3 |
| resnet-graph | nccl-async | 1 | 631 | 6.31e-2 | 6.95e-2 | 0.00e0 | 4.99e-1 | 33.8 | -1.64e-4 | 4.98e-3 |
| resnet-graph | nccl-async | 2 | 631 | 6.28e-2 | 6.87e-2 | 0.00e0 | 4.55e-1 | 22.0 | -1.49e-4 | 4.89e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | nccl-async | rank0 ↔ rank1 | +0.9972 |
| resnet-graph | nccl-async | rank0 ↔ rank2 | +0.9964 |
| resnet-graph | nccl-async | rank1 ↔ rank2 | +0.9991 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | nccl-async | 43 (0,1,2,3,4,5,6,7…137,139) | 0 (—) | — | 0,1,2,3,4,5,6,7…137,139 | — |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | nccl-async | 0e0 | 3 | 51 | 51 |
| resnet-graph | nccl-async | 0e0 | 5 | 24 | 24 |
| resnet-graph | nccl-async | 0e0 | 10 | 8 | 8 |
| resnet-graph | nccl-async | 1e-4 | 3 | 21 | 21 |
| resnet-graph | nccl-async | 1e-4 | 5 | 11 | 11 |
| resnet-graph | nccl-async | 1e-4 | 10 | 4 | 4 |
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
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 288 | +0.058 |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | 242 | +0.015 |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | 96 | +0.008 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | nccl-async | 628 | -0.001 | 195 | +0.172 | +0.190 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | nccl-async | 629 | 3.44e1–7.93e1 | 6.53e1 | 1.92e-3 | 3.53e-3 | 4.97e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 290 | 69–77944 | +1.352e-5 | 0.506 | +1.384e-5 | 0.513 | 96 | +5.364e-6 | 0.143 | 30–935 | +1.647e-3 | 0.737 |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 275 | 869–77944 | +1.344e-5 | 0.535 | +1.372e-5 | 0.539 | 95 | +4.857e-6 | 0.122 | 32–935 | +1.644e-3 | 0.791 |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | 243 | 78247–116959 | +5.489e-5 | 0.447 | +5.571e-5 | 0.448 | 50 | +4.790e-5 | 0.545 | 31–655 | +3.441e-3 | 0.780 |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | 97 | 117369–155879 | +7.333e-6 | 0.037 | +7.635e-6 | 0.039 | 50 | +6.692e-6 | 0.038 | 148–701 | +1.222e-3 | 0.235 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | +1.647e-3 | r0: +1.618e-3, r1: +1.669e-3, r2: +1.659e-3 | r0: 0.758, r1: 0.724, r2: 0.723 | 1.03× | ✓ |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +1.644e-3 | r0: +1.619e-3, r1: +1.662e-3, r2: +1.655e-3 | r0: 0.818, r1: 0.777, r2: 0.773 | 1.03× | ✓ |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | +3.441e-3 | r0: +3.385e-3, r1: +3.480e-3, r2: +3.468e-3 | r0: 0.803, r1: 0.765, r2: 0.767 | 1.03× | ✓ |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | +1.222e-3 | r0: +1.236e-3, r1: +1.211e-3, r2: +1.222e-3 | r0: 0.245, r1: 0.228, r2: 0.232 | 1.02× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | nccl-async | `█▇▇█████████████████████▅▄▃▃▃▅▅▅▅▅▅▅▄▁▁▁▁▁▂▂▂▂▂▂▂` | `▂▅▅█▇▇▇▇▇▆▆▆▆▆▆▆▆▆▆▆▇▇▇▆▁▅▅▇▇▇▇▇▇▇▇▇▅▄▅▆▆▆▇▆▇▆▆▆▆` |

### Per-Epoch Detail: resnet-graph / nccl-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 16 | 0.00e0 | 5.29e-1 | 9.54e-2 | 5.10e-2 | 20 | -8.58e-2 | +1.08e-2 | -1.29e-2 | -7.27e-3 |
| 1 | 3.00e-1 | 16 | 5.10e-2 | 1.12e-1 | 6.01e-2 | 6.73e-2 | 20 | -4.26e-2 | +3.82e-2 | +5.44e-4 | -1.55e-4 |
| 2 | 3.00e-1 | 14 | 5.56e-2 | 1.25e-1 | 7.34e-2 | 7.85e-2 | 16 | -5.78e-2 | +4.24e-2 | +3.55e-4 | +3.46e-4 |
| 3 | 3.00e-1 | 13 | 6.70e-2 | 1.37e-1 | 8.34e-2 | 9.10e-2 | 18 | -4.34e-2 | +4.75e-2 | +1.29e-3 | +9.64e-4 |
| 4 | 3.00e-1 | 15 | 6.65e-2 | 1.37e-1 | 7.81e-2 | 7.57e-2 | 18 | -4.51e-2 | +3.46e-2 | -2.72e-4 | +9.92e-5 |
| 5 | 3.00e-1 | 11 | 6.73e-2 | 1.35e-1 | 8.94e-2 | 9.24e-2 | 23 | -4.99e-2 | +3.66e-2 | -5.67e-5 | -1.38e-7 |
| 6 | 3.00e-1 | 16 | 7.05e-2 | 1.36e-1 | 8.25e-2 | 7.05e-2 | 19 | -2.67e-2 | +2.07e-2 | -7.81e-4 | -7.80e-4 |
| 7 | 3.00e-1 | 12 | 6.24e-2 | 1.34e-1 | 7.33e-2 | 6.45e-2 | 16 | -4.28e-2 | +3.41e-2 | -8.14e-4 | -8.74e-4 |
| 8 | 3.00e-1 | 16 | 5.75e-2 | 1.37e-1 | 7.05e-2 | 6.31e-2 | 16 | -5.08e-2 | +4.15e-2 | -9.08e-4 | -8.90e-4 |
| 9 | 3.00e-1 | 16 | 5.89e-2 | 1.36e-1 | 7.12e-2 | 6.76e-2 | 15 | -4.70e-2 | +4.72e-2 | +2.13e-4 | -1.07e-4 |
| 10 | 3.00e-1 | 14 | 5.03e-2 | 1.33e-1 | 7.02e-2 | 6.78e-2 | 30 | -6.68e-2 | +5.47e-2 | -6.42e-4 | -4.38e-4 |
| 11 | 3.00e-1 | 3 | 8.21e-2 | 1.07e-1 | 9.11e-2 | 8.21e-2 | 204 | -1.11e-2 | +1.97e-2 | +2.82e-3 | +2.63e-4 |
| 12 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 225 | +4.21e-3 | +4.21e-3 | +4.21e-3 | +6.58e-4 |
| 13 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 212 | -1.96e-4 | -1.96e-4 | -1.96e-4 | +5.72e-4 |
| 14 | 3.00e-1 | 1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 222 | -1.79e-4 | -1.79e-4 | -1.79e-4 | +4.97e-4 |
| 15 | 3.00e-1 | 2 | 1.91e-1 | 1.93e-1 | 1.92e-1 | 1.91e-1 | 235 | -4.78e-5 | -3.35e-5 | -4.07e-5 | +3.95e-4 |
| 16 | 3.00e-1 | 1 | 1.89e-1 | 1.89e-1 | 1.89e-1 | 1.89e-1 | 311 | -3.42e-5 | -3.42e-5 | -3.42e-5 | +3.52e-4 |
| 18 | 3.00e-1 | 2 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 268 | -2.99e-6 | +3.10e-4 | +1.53e-4 | +3.13e-4 |
| 20 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 338 | -1.59e-4 | -1.59e-4 | -1.59e-4 | +2.66e-4 |
| 21 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 325 | +1.61e-4 | +1.61e-4 | +1.61e-4 | +2.55e-4 |
| 22 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 265 | +2.09e-5 | +2.09e-5 | +2.09e-5 | +2.32e-4 |
| 23 | 3.00e-1 | 1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 259 | -2.51e-4 | -2.51e-4 | -2.51e-4 | +1.83e-4 |
| 24 | 3.00e-1 | 1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 274 | -2.74e-5 | -2.74e-5 | -2.74e-5 | +1.62e-4 |
| 25 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 265 | +8.21e-5 | +8.21e-5 | +8.21e-5 | +1.54e-4 |
| 26 | 3.00e-1 | 1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 275 | -5.43e-5 | -5.43e-5 | -5.43e-5 | +1.33e-4 |
| 27 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 270 | +6.37e-5 | +6.37e-5 | +6.37e-5 | +1.27e-4 |
| 28 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 266 | +2.41e-5 | +2.41e-5 | +2.41e-5 | +1.16e-4 |
| 29 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 253 | -2.06e-5 | -2.06e-5 | -2.06e-5 | +1.03e-4 |
| 30 | 3.00e-1 | 2 | 1.95e-1 | 2.02e-1 | 1.99e-1 | 1.95e-1 | 228 | -1.47e-4 | +3.14e-5 | -5.76e-5 | +7.13e-5 |
| 32 | 3.00e-1 | 1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 323 | -5.75e-5 | -5.75e-5 | -5.75e-5 | +5.84e-5 |
| 33 | 3.00e-1 | 2 | 2.01e-1 | 2.13e-1 | 2.07e-1 | 2.01e-1 | 228 | -2.46e-4 | +3.80e-4 | +6.71e-5 | +5.69e-5 |
| 34 | 3.00e-1 | 1 | 1.88e-1 | 1.88e-1 | 1.88e-1 | 1.88e-1 | 250 | -2.68e-4 | -2.68e-4 | -2.68e-4 | +2.44e-5 |
| 35 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 257 | +2.38e-4 | +2.38e-4 | +2.38e-4 | +4.58e-5 |
| 36 | 3.00e-1 | 1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 279 | -4.55e-5 | -4.55e-5 | -4.55e-5 | +3.67e-5 |
| 37 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 287 | +8.69e-5 | +8.69e-5 | +8.69e-5 | +4.17e-5 |
| 38 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 286 | +1.23e-4 | +1.23e-4 | +1.23e-4 | +4.98e-5 |
| 39 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 259 | -1.27e-4 | -1.27e-4 | -1.27e-4 | +3.21e-5 |
| 40 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 250 | -6.33e-5 | -6.33e-5 | -6.33e-5 | +2.26e-5 |
| 41 | 3.00e-1 | 2 | 1.98e-1 | 2.03e-1 | 2.00e-1 | 2.03e-1 | 214 | -3.53e-5 | +1.18e-4 | +4.11e-5 | +2.68e-5 |
| 43 | 3.00e-1 | 2 | 1.84e-1 | 2.07e-1 | 1.95e-1 | 2.07e-1 | 214 | -3.47e-4 | +5.55e-4 | +1.04e-4 | +4.60e-5 |
| 44 | 3.00e-1 | 1 | 1.88e-1 | 1.88e-1 | 1.88e-1 | 1.88e-1 | 240 | -4.03e-4 | -4.03e-4 | -4.03e-4 | +1.09e-6 |
| 45 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 248 | +1.81e-4 | +1.81e-4 | +1.81e-4 | +1.91e-5 |
| 46 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 235 | +3.50e-5 | +3.50e-5 | +3.50e-5 | +2.07e-5 |
| 47 | 3.00e-1 | 1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 252 | -6.08e-5 | -6.08e-5 | -6.08e-5 | +1.26e-5 |
| 48 | 3.00e-1 | 2 | 1.90e-1 | 1.98e-1 | 1.94e-1 | 1.90e-1 | 199 | -2.20e-4 | +7.31e-5 | -7.36e-5 | -5.28e-6 |
| 49 | 3.00e-1 | 1 | 1.84e-1 | 1.84e-1 | 1.84e-1 | 1.84e-1 | 250 | -1.13e-4 | -1.13e-4 | -1.13e-4 | -1.61e-5 |
| 50 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 233 | +3.69e-4 | +3.69e-4 | +3.69e-4 | +2.24e-5 |
| 51 | 3.00e-1 | 2 | 1.87e-1 | 1.94e-1 | 1.91e-1 | 1.87e-1 | 187 | -2.21e-4 | -1.68e-4 | -1.94e-4 | -1.91e-5 |
| 52 | 3.00e-1 | 1 | 1.81e-1 | 1.81e-1 | 1.81e-1 | 1.81e-1 | 212 | -1.37e-4 | -1.37e-4 | -1.37e-4 | -3.09e-5 |
| 53 | 3.00e-1 | 1 | 1.91e-1 | 1.91e-1 | 1.91e-1 | 1.91e-1 | 210 | +2.56e-4 | +2.56e-4 | +2.56e-4 | -2.17e-6 |
| 54 | 3.00e-1 | 1 | 1.91e-1 | 1.91e-1 | 1.91e-1 | 1.91e-1 | 235 | -1.09e-5 | -1.09e-5 | -1.09e-5 | -3.04e-6 |
| 55 | 3.00e-1 | 2 | 1.93e-1 | 1.97e-1 | 1.95e-1 | 1.93e-1 | 198 | -1.01e-4 | +1.49e-4 | +2.37e-5 | +7.79e-7 |
| 56 | 3.00e-1 | 1 | 1.89e-1 | 1.89e-1 | 1.89e-1 | 1.89e-1 | 210 | -1.10e-4 | -1.10e-4 | -1.10e-4 | -1.03e-5 |
| 57 | 3.00e-1 | 1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 253 | +8.40e-5 | +8.40e-5 | +8.40e-5 | -8.63e-7 |
| 58 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 207 | +1.98e-4 | +1.98e-4 | +1.98e-4 | +1.90e-5 |
| 59 | 3.00e-1 | 2 | 1.88e-1 | 1.93e-1 | 1.90e-1 | 1.93e-1 | 185 | -3.04e-4 | +1.31e-4 | -8.64e-5 | +1.19e-6 |
| 60 | 3.00e-1 | 1 | 1.80e-1 | 1.80e-1 | 1.80e-1 | 1.80e-1 | 208 | -3.21e-4 | -3.21e-4 | -3.21e-4 | -3.10e-5 |
| 61 | 3.00e-1 | 2 | 1.88e-1 | 1.91e-1 | 1.90e-1 | 1.88e-1 | 175 | -8.16e-5 | +2.84e-4 | +1.01e-4 | -7.68e-6 |
| 62 | 3.00e-1 | 1 | 1.79e-1 | 1.79e-1 | 1.79e-1 | 1.79e-1 | 211 | -2.49e-4 | -2.49e-4 | -2.49e-4 | -3.19e-5 |
| 63 | 3.00e-1 | 1 | 1.91e-1 | 1.91e-1 | 1.91e-1 | 1.91e-1 | 210 | +3.15e-4 | +3.15e-4 | +3.15e-4 | +2.78e-6 |
| 64 | 3.00e-1 | 2 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 175 | -2.23e-6 | +6.59e-5 | +3.18e-5 | +7.96e-6 |
| 65 | 3.00e-1 | 2 | 1.81e-1 | 1.98e-1 | 1.90e-1 | 1.98e-1 | 175 | -2.80e-4 | +4.89e-4 | +1.05e-4 | +3.02e-5 |
| 66 | 3.00e-1 | 1 | 1.80e-1 | 1.80e-1 | 1.80e-1 | 1.80e-1 | 216 | -4.31e-4 | -4.31e-4 | -4.31e-4 | -1.59e-5 |
| 67 | 3.00e-1 | 1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 204 | +3.59e-4 | +3.59e-4 | +3.59e-4 | +2.16e-5 |
| 68 | 3.00e-1 | 2 | 1.90e-1 | 1.93e-1 | 1.91e-1 | 1.90e-1 | 163 | -1.12e-4 | -1.97e-5 | -6.56e-5 | +4.58e-6 |
| 69 | 3.00e-1 | 1 | 1.75e-1 | 1.75e-1 | 1.75e-1 | 1.75e-1 | 178 | -4.53e-4 | -4.53e-4 | -4.53e-4 | -4.12e-5 |
| 70 | 3.00e-1 | 2 | 1.85e-1 | 1.85e-1 | 1.85e-1 | 1.85e-1 | 163 | +8.74e-6 | +3.08e-4 | +1.58e-4 | -4.81e-6 |
| 71 | 3.00e-1 | 1 | 1.74e-1 | 1.74e-1 | 1.74e-1 | 1.74e-1 | 195 | -2.95e-4 | -2.95e-4 | -2.95e-4 | -3.39e-5 |
| 72 | 3.00e-1 | 1 | 1.87e-1 | 1.87e-1 | 1.87e-1 | 1.87e-1 | 212 | +3.26e-4 | +3.26e-4 | +3.26e-4 | +2.09e-6 |
| 73 | 3.00e-1 | 2 | 1.92e-1 | 1.93e-1 | 1.93e-1 | 1.92e-1 | 162 | -1.69e-5 | +1.43e-4 | +6.31e-5 | +1.29e-5 |
| 74 | 3.00e-1 | 2 | 1.78e-1 | 1.79e-1 | 1.79e-1 | 1.79e-1 | 150 | -4.44e-4 | +3.78e-5 | -2.03e-4 | -2.57e-5 |
| 75 | 3.00e-1 | 1 | 1.69e-1 | 1.69e-1 | 1.69e-1 | 1.69e-1 | 180 | -3.41e-4 | -3.41e-4 | -3.41e-4 | -5.73e-5 |
| 76 | 3.00e-1 | 2 | 1.84e-1 | 2.00e-1 | 1.92e-1 | 2.00e-1 | 158 | +3.79e-4 | +5.20e-4 | +4.50e-4 | +3.98e-5 |
| 77 | 3.00e-1 | 2 | 1.76e-1 | 1.84e-1 | 1.80e-1 | 1.84e-1 | 149 | -7.22e-4 | +3.25e-4 | -1.98e-4 | -2.38e-7 |
| 78 | 3.00e-1 | 1 | 1.70e-1 | 1.70e-1 | 1.70e-1 | 1.70e-1 | 191 | -4.20e-4 | -4.20e-4 | -4.20e-4 | -4.22e-5 |
| 79 | 3.00e-1 | 2 | 1.80e-1 | 1.89e-1 | 1.85e-1 | 1.80e-1 | 136 | -3.30e-4 | +5.91e-4 | +1.31e-4 | -1.40e-5 |
| 80 | 3.00e-1 | 2 | 1.67e-1 | 1.73e-1 | 1.70e-1 | 1.73e-1 | 134 | -5.08e-4 | +2.76e-4 | -1.16e-4 | -2.95e-5 |
| 81 | 3.00e-1 | 2 | 1.63e-1 | 1.85e-1 | 1.74e-1 | 1.85e-1 | 140 | -3.35e-4 | +9.15e-4 | +2.90e-4 | +3.74e-5 |
| 82 | 3.00e-1 | 1 | 1.67e-1 | 1.67e-1 | 1.67e-1 | 1.67e-1 | 175 | -6.02e-4 | -6.02e-4 | -6.02e-4 | -2.65e-5 |
| 83 | 3.00e-1 | 2 | 1.84e-1 | 1.88e-1 | 1.86e-1 | 1.88e-1 | 130 | +1.69e-4 | +5.00e-4 | +3.35e-4 | +4.05e-5 |
| 84 | 3.00e-1 | 2 | 1.64e-1 | 1.75e-1 | 1.70e-1 | 1.75e-1 | 122 | -8.77e-4 | +5.58e-4 | -1.60e-4 | +9.67e-6 |
| 85 | 3.00e-1 | 2 | 1.63e-1 | 1.79e-1 | 1.71e-1 | 1.79e-1 | 137 | -4.42e-4 | +6.52e-4 | +1.05e-4 | +3.33e-5 |
| 86 | 3.00e-1 | 2 | 1.64e-1 | 1.87e-1 | 1.76e-1 | 1.87e-1 | 133 | -4.62e-4 | +9.86e-4 | +2.62e-4 | +8.40e-5 |
| 87 | 3.00e-1 | 2 | 1.63e-1 | 1.78e-1 | 1.71e-1 | 1.78e-1 | 133 | -8.64e-4 | +6.71e-4 | -9.64e-5 | +5.74e-5 |
| 88 | 3.00e-1 | 2 | 1.63e-1 | 1.78e-1 | 1.71e-1 | 1.78e-1 | 122 | -5.64e-4 | +6.95e-4 | +6.54e-5 | +6.52e-5 |
| 89 | 3.00e-1 | 2 | 1.54e-1 | 1.85e-1 | 1.70e-1 | 1.85e-1 | 135 | -8.23e-4 | +1.38e-3 | +2.76e-4 | +1.16e-4 |
| 90 | 3.00e-1 | 2 | 1.69e-1 | 1.74e-1 | 1.72e-1 | 1.74e-1 | 117 | -6.00e-4 | +2.37e-4 | -1.82e-4 | +6.39e-5 |
| 91 | 3.00e-1 | 2 | 1.61e-1 | 1.76e-1 | 1.68e-1 | 1.76e-1 | 100 | -5.35e-4 | +8.92e-4 | +1.78e-4 | +9.28e-5 |
| 92 | 3.00e-1 | 2 | 1.49e-1 | 1.74e-1 | 1.62e-1 | 1.74e-1 | 100 | -1.10e-3 | +1.54e-3 | +2.24e-4 | +1.31e-4 |
| 93 | 3.00e-1 | 3 | 1.45e-1 | 1.72e-1 | 1.57e-1 | 1.52e-1 | 100 | -1.29e-3 | +1.69e-3 | -2.81e-4 | +1.87e-5 |
| 94 | 3.00e-1 | 3 | 1.49e-1 | 1.71e-1 | 1.57e-1 | 1.49e-1 | 100 | -1.39e-3 | +1.17e-3 | -7.46e-5 | -2.05e-5 |
| 95 | 3.00e-1 | 2 | 1.54e-1 | 1.69e-1 | 1.61e-1 | 1.69e-1 | 106 | +2.63e-4 | +8.56e-4 | +5.59e-4 | +9.26e-5 |
| 96 | 3.00e-1 | 3 | 1.44e-1 | 1.68e-1 | 1.57e-1 | 1.44e-1 | 87 | -1.78e-3 | +7.07e-4 | -5.24e-4 | -8.73e-5 |
| 97 | 3.00e-1 | 3 | 1.36e-1 | 1.66e-1 | 1.49e-1 | 1.36e-1 | 80 | -2.56e-3 | +1.83e-3 | -2.37e-4 | -1.53e-4 |
| 98 | 3.00e-1 | 3 | 1.31e-1 | 1.54e-1 | 1.40e-1 | 1.31e-1 | 73 | -2.20e-3 | +1.63e-3 | -1.69e-4 | -1.80e-4 |
| 99 | 3.00e-1 | 3 | 1.35e-1 | 1.58e-1 | 1.44e-1 | 1.41e-1 | 80 | -1.47e-3 | +1.98e-3 | +2.60e-4 | -7.80e-5 |
| 100 | 3.00e-2 | 3 | 1.34e-2 | 1.65e-1 | 1.05e-1 | 1.34e-2 | 80 | -3.14e-2 | +2.24e-3 | -9.77e-3 | -3.01e-3 |
| 101 | 3.00e-2 | 4 | 1.35e-2 | 1.52e-2 | 1.41e-2 | 1.42e-2 | 80 | -1.49e-3 | +1.41e-3 | +1.82e-4 | -1.92e-3 |
| 102 | 3.00e-2 | 4 | 1.36e-2 | 1.55e-2 | 1.45e-2 | 1.44e-2 | 72 | -1.80e-3 | +9.56e-4 | +2.65e-5 | -1.25e-3 |
| 103 | 3.00e-2 | 3 | 1.42e-2 | 1.72e-2 | 1.54e-2 | 1.47e-2 | 66 | -2.43e-3 | +2.69e-3 | +3.84e-5 | -9.25e-4 |
| 104 | 3.00e-2 | 4 | 1.35e-2 | 1.81e-2 | 1.49e-2 | 1.38e-2 | 56 | -5.19e-3 | +4.28e-3 | -2.42e-4 | -7.22e-4 |
| 105 | 3.00e-2 | 5 | 1.29e-2 | 1.67e-2 | 1.41e-2 | 1.45e-2 | 49 | -4.97e-3 | +3.92e-3 | +2.41e-4 | -3.15e-4 |
| 106 | 3.00e-2 | 8 | 1.08e-2 | 1.74e-2 | 1.26e-2 | 1.09e-2 | 37 | -7.30e-3 | +4.48e-3 | -9.95e-4 | -6.92e-4 |
| 107 | 3.00e-2 | 5 | 1.09e-2 | 1.62e-2 | 1.22e-2 | 1.09e-2 | 34 | -1.09e-2 | +9.82e-3 | -2.04e-4 | -5.99e-4 |
| 108 | 3.00e-2 | 10 | 7.86e-3 | 1.61e-2 | 1.01e-2 | 7.86e-3 | 19 | -1.80e-2 | +1.37e-2 | -1.51e-3 | -1.49e-3 |
| 109 | 3.00e-2 | 18 | 6.14e-3 | 1.50e-2 | 7.84e-3 | 8.39e-3 | 23 | -4.87e-2 | +4.33e-2 | +2.00e-4 | +3.48e-5 |
| 110 | 3.00e-2 | 10 | 7.18e-3 | 1.56e-2 | 8.70e-3 | 8.03e-3 | 19 | -3.18e-2 | +2.61e-2 | -4.31e-4 | -2.07e-4 |
| 111 | 3.00e-2 | 16 | 6.51e-3 | 1.52e-2 | 7.92e-3 | 7.52e-3 | 16 | -5.29e-2 | +4.02e-2 | -2.00e-4 | -1.47e-4 |
| 112 | 3.00e-2 | 17 | 6.17e-3 | 1.48e-2 | 7.45e-3 | 6.99e-3 | 15 | -6.03e-2 | +5.02e-2 | -2.45e-4 | -3.22e-4 |
| 113 | 3.00e-2 | 23 | 5.39e-3 | 1.66e-2 | 7.42e-3 | 7.73e-3 | 15 | -6.65e-2 | +5.64e-2 | -1.69e-4 | +2.37e-4 |
| 114 | 3.00e-2 | 10 | 6.66e-3 | 1.66e-2 | 8.57e-3 | 8.05e-3 | 17 | -6.45e-2 | +5.80e-2 | -2.11e-4 | -2.05e-4 |
| 115 | 3.00e-2 | 13 | 8.09e-3 | 1.69e-2 | 9.49e-3 | 9.71e-3 | 20 | -3.97e-2 | +4.33e-2 | +8.12e-4 | +4.37e-4 |
| 116 | 3.00e-2 | 21 | 7.74e-3 | 1.85e-2 | 9.33e-3 | 9.70e-3 | 20 | -4.17e-2 | +3.74e-2 | -1.58e-4 | +2.88e-4 |
| 117 | 3.00e-2 | 7 | 9.61e-3 | 1.77e-2 | 1.12e-2 | 1.08e-2 | 20 | -2.93e-2 | +2.66e-2 | +4.11e-4 | +2.72e-4 |
| 118 | 3.00e-2 | 14 | 8.16e-3 | 1.90e-2 | 1.01e-2 | 1.07e-2 | 20 | -5.63e-2 | +4.16e-2 | -9.25e-5 | +3.03e-4 |
| 119 | 3.00e-2 | 2 | 9.37e-3 | 1.00e-2 | 9.69e-3 | 1.00e-2 | 200 | -6.52e-3 | +3.32e-4 | -3.09e-3 | -3.08e-4 |
| 120 | 3.00e-2 | 1 | 3.67e-2 | 3.67e-2 | 3.67e-2 | 3.67e-2 | 228 | +5.70e-3 | +5.70e-3 | +5.70e-3 | +2.92e-4 |
| 121 | 3.00e-2 | 2 | 3.81e-2 | 3.88e-2 | 3.84e-2 | 3.88e-2 | 181 | +1.02e-4 | +1.66e-4 | +1.34e-4 | +2.62e-4 |
| 122 | 3.00e-2 | 1 | 3.48e-2 | 3.48e-2 | 3.48e-2 | 3.48e-2 | 240 | -4.56e-4 | -4.56e-4 | -4.56e-4 | +1.90e-4 |
| 123 | 3.00e-2 | 1 | 4.03e-2 | 4.03e-2 | 4.03e-2 | 4.03e-2 | 212 | +6.93e-4 | +6.93e-4 | +6.93e-4 | +2.40e-4 |
| 124 | 3.00e-2 | 2 | 3.82e-2 | 3.82e-2 | 3.82e-2 | 3.82e-2 | 168 | -2.60e-4 | -5.55e-6 | -1.33e-4 | +1.71e-4 |
| 125 | 3.00e-2 | 1 | 3.73e-2 | 3.73e-2 | 3.73e-2 | 3.73e-2 | 213 | -1.15e-4 | -1.15e-4 | -1.15e-4 | +1.42e-4 |
| 126 | 3.00e-2 | 2 | 3.86e-2 | 3.98e-2 | 3.92e-2 | 3.86e-2 | 168 | -1.92e-4 | +3.60e-4 | +8.38e-5 | +1.28e-4 |
| 127 | 3.00e-2 | 1 | 3.71e-2 | 3.71e-2 | 3.71e-2 | 3.71e-2 | 203 | -1.90e-4 | -1.90e-4 | -1.90e-4 | +9.65e-5 |
| 128 | 3.00e-2 | 1 | 4.16e-2 | 4.16e-2 | 4.16e-2 | 4.16e-2 | 196 | +5.78e-4 | +5.78e-4 | +5.78e-4 | +1.45e-4 |
| 129 | 3.00e-2 | 2 | 4.01e-2 | 4.20e-2 | 4.10e-2 | 4.20e-2 | 184 | -1.85e-4 | +2.53e-4 | +3.40e-5 | +1.26e-4 |
| 130 | 3.00e-2 | 1 | 4.14e-2 | 4.14e-2 | 4.14e-2 | 4.14e-2 | 210 | -6.55e-5 | -6.55e-5 | -6.55e-5 | +1.07e-4 |
| 131 | 3.00e-2 | 2 | 4.09e-2 | 4.24e-2 | 4.16e-2 | 4.09e-2 | 170 | -2.15e-4 | +1.32e-4 | -4.12e-5 | +7.68e-5 |
| 132 | 3.00e-2 | 1 | 3.95e-2 | 3.95e-2 | 3.95e-2 | 3.95e-2 | 197 | -1.80e-4 | -1.80e-4 | -1.80e-4 | +5.11e-5 |
| 133 | 3.00e-2 | 2 | 4.32e-2 | 4.59e-2 | 4.46e-2 | 4.59e-2 | 170 | +3.64e-4 | +4.15e-4 | +3.89e-4 | +1.15e-4 |
| 134 | 3.00e-2 | 1 | 4.20e-2 | 4.20e-2 | 4.20e-2 | 4.20e-2 | 192 | -4.69e-4 | -4.69e-4 | -4.69e-4 | +5.68e-5 |
| 135 | 3.00e-2 | 2 | 4.37e-2 | 4.60e-2 | 4.49e-2 | 4.60e-2 | 170 | +2.06e-4 | +2.91e-4 | +2.49e-4 | +9.37e-5 |
| 136 | 3.00e-2 | 1 | 4.26e-2 | 4.26e-2 | 4.26e-2 | 4.26e-2 | 193 | -3.96e-4 | -3.96e-4 | -3.96e-4 | +4.47e-5 |
| 137 | 3.00e-2 | 2 | 4.53e-2 | 4.69e-2 | 4.61e-2 | 4.69e-2 | 160 | +2.14e-4 | +3.13e-4 | +2.63e-4 | +8.57e-5 |
| 138 | 3.00e-2 | 1 | 4.30e-2 | 4.30e-2 | 4.30e-2 | 4.30e-2 | 189 | -4.55e-4 | -4.55e-4 | -4.55e-4 | +3.16e-5 |
| 139 | 3.00e-2 | 2 | 4.56e-2 | 4.57e-2 | 4.56e-2 | 4.57e-2 | 160 | +1.11e-5 | +3.20e-4 | +1.66e-4 | +5.55e-5 |
| 140 | 3.00e-2 | 1 | 4.37e-2 | 4.37e-2 | 4.37e-2 | 4.37e-2 | 206 | -2.19e-4 | -2.19e-4 | -2.19e-4 | +2.81e-5 |
| 141 | 3.00e-2 | 2 | 4.77e-2 | 5.06e-2 | 4.91e-2 | 4.77e-2 | 160 | -3.67e-4 | +7.63e-4 | +1.98e-4 | +5.48e-5 |
| 142 | 3.00e-2 | 2 | 4.62e-2 | 4.94e-2 | 4.78e-2 | 4.94e-2 | 148 | -1.71e-4 | +4.51e-4 | +1.40e-4 | +7.41e-5 |
| 143 | 3.00e-2 | 1 | 4.34e-2 | 4.34e-2 | 4.34e-2 | 4.34e-2 | 171 | -7.57e-4 | -7.57e-4 | -7.57e-4 | -8.96e-6 |
| 144 | 3.00e-2 | 2 | 4.65e-2 | 4.79e-2 | 4.72e-2 | 4.65e-2 | 137 | -2.20e-4 | +5.78e-4 | +1.79e-4 | +2.27e-5 |
| 145 | 3.00e-2 | 2 | 4.33e-2 | 4.87e-2 | 4.60e-2 | 4.87e-2 | 146 | -3.78e-4 | +8.16e-4 | +2.19e-4 | +6.60e-5 |
| 146 | 3.00e-2 | 1 | 4.64e-2 | 4.64e-2 | 4.64e-2 | 4.64e-2 | 202 | -2.44e-4 | -2.44e-4 | -2.44e-4 | +3.50e-5 |
| 147 | 3.00e-2 | 2 | 5.16e-2 | 5.24e-2 | 5.20e-2 | 5.16e-2 | 135 | -1.10e-4 | +6.10e-4 | +2.50e-4 | +7.22e-5 |
| 148 | 3.00e-2 | 2 | 4.46e-2 | 5.10e-2 | 4.78e-2 | 5.10e-2 | 139 | -7.54e-4 | +9.75e-4 | +1.11e-4 | +8.82e-5 |
| 149 | 3.00e-2 | 2 | 4.63e-2 | 4.67e-2 | 4.65e-2 | 4.63e-2 | 115 | -6.13e-4 | -7.67e-5 | -3.45e-4 | +8.59e-6 |
| 150 | 3.00e-3 | 2 | 4.24e-2 | 4.92e-2 | 4.58e-2 | 4.92e-2 | 115 | -5.95e-4 | +1.31e-3 | +3.57e-4 | +8.43e-5 |
| 151 | 3.00e-3 | 3 | 3.98e-3 | 4.55e-3 | 4.30e-3 | 3.98e-3 | 129 | -1.67e-2 | +3.27e-4 | -5.80e-3 | -1.37e-3 |
| 152 | 3.00e-3 | 1 | 4.11e-3 | 4.11e-3 | 4.11e-3 | 4.11e-3 | 170 | +1.88e-4 | +1.88e-4 | +1.88e-4 | -1.21e-3 |
| 153 | 3.00e-3 | 3 | 4.03e-3 | 4.87e-3 | 4.58e-3 | 4.03e-3 | 118 | -1.60e-3 | +1.02e-3 | -1.73e-4 | -9.54e-4 |
| 154 | 3.00e-3 | 1 | 4.03e-3 | 4.03e-3 | 4.03e-3 | 4.03e-3 | 163 | +5.52e-6 | +5.52e-6 | +5.52e-6 | -8.58e-4 |
| 155 | 3.00e-3 | 3 | 4.01e-3 | 4.76e-3 | 4.41e-3 | 4.01e-3 | 111 | -9.60e-4 | +1.12e-3 | -1.44e-4 | -6.84e-4 |
| 156 | 3.00e-3 | 2 | 3.91e-3 | 4.37e-3 | 4.14e-3 | 4.37e-3 | 111 | -1.66e-4 | +1.01e-3 | +4.23e-4 | -4.68e-4 |
| 157 | 3.00e-3 | 2 | 3.90e-3 | 4.58e-3 | 4.24e-3 | 4.58e-3 | 118 | -6.98e-4 | +1.37e-3 | +3.33e-4 | -3.05e-4 |
| 158 | 3.00e-3 | 3 | 4.10e-3 | 4.23e-3 | 4.17e-3 | 4.10e-3 | 105 | -6.50e-4 | +7.59e-5 | -2.89e-4 | -2.98e-4 |
| 159 | 3.00e-3 | 2 | 3.81e-3 | 4.42e-3 | 4.12e-3 | 4.42e-3 | 100 | -5.31e-4 | +1.47e-3 | +4.68e-4 | -1.42e-4 |
| 160 | 3.00e-3 | 2 | 3.93e-3 | 4.23e-3 | 4.08e-3 | 4.23e-3 | 90 | -9.34e-4 | +8.31e-4 | -5.19e-5 | -1.16e-4 |
| 161 | 3.00e-3 | 3 | 3.49e-3 | 4.23e-3 | 3.82e-3 | 3.73e-3 | 85 | -1.49e-3 | +2.10e-3 | -2.86e-4 | -1.64e-4 |
| 162 | 3.00e-3 | 3 | 3.51e-3 | 4.50e-3 | 3.84e-3 | 3.51e-3 | 73 | -3.40e-3 | +2.85e-3 | -3.02e-4 | -2.32e-4 |
| 163 | 3.00e-3 | 4 | 3.30e-3 | 3.92e-3 | 3.53e-3 | 3.48e-3 | 73 | -1.62e-3 | +2.25e-3 | +7.08e-5 | -1.35e-4 |
| 164 | 3.00e-3 | 2 | 3.24e-3 | 4.00e-3 | 3.62e-3 | 4.00e-3 | 82 | -6.03e-4 | +2.57e-3 | +9.82e-4 | +9.32e-5 |
| 165 | 3.00e-3 | 4 | 3.02e-3 | 4.21e-3 | 3.46e-3 | 3.14e-3 | 69 | -4.82e-3 | +2.71e-3 | -6.61e-4 | -1.76e-4 |
| 166 | 3.00e-3 | 3 | 3.33e-3 | 4.10e-3 | 3.67e-3 | 3.57e-3 | 77 | -1.77e-3 | +2.66e-3 | +4.67e-4 | -2.45e-5 |
| 167 | 3.00e-3 | 4 | 3.04e-3 | 3.91e-3 | 3.46e-3 | 3.04e-3 | 60 | -2.76e-3 | +1.23e-3 | -7.03e-4 | -2.95e-4 |
| 168 | 3.00e-3 | 5 | 2.84e-3 | 3.95e-3 | 3.17e-3 | 2.91e-3 | 60 | -5.50e-3 | +3.73e-3 | -1.99e-4 | -2.91e-4 |
| 169 | 3.00e-3 | 3 | 2.99e-3 | 3.69e-3 | 3.27e-3 | 3.14e-3 | 60 | -2.68e-3 | +3.28e-3 | +2.91e-4 | -1.63e-4 |
| 170 | 3.00e-3 | 3 | 2.71e-3 | 3.08e-3 | 2.91e-3 | 2.71e-3 | 210 | -1.12e-3 | +9.42e-4 | -2.62e-4 | -1.85e-4 |
| 171 | 3.00e-3 | 1 | 5.82e-3 | 5.82e-3 | 5.82e-3 | 5.82e-3 | 243 | +3.14e-3 | +3.14e-3 | +3.14e-3 | +1.47e-4 |
| 172 | 3.00e-3 | 1 | 6.01e-3 | 6.01e-3 | 6.01e-3 | 6.01e-3 | 234 | +1.41e-4 | +1.41e-4 | +1.41e-4 | +1.46e-4 |
| 173 | 3.00e-3 | 1 | 5.98e-3 | 5.98e-3 | 5.98e-3 | 5.98e-3 | 224 | -2.44e-5 | -2.44e-5 | -2.44e-5 | +1.29e-4 |
| 174 | 3.00e-3 | 1 | 5.83e-3 | 5.83e-3 | 5.83e-3 | 5.83e-3 | 240 | -1.09e-4 | -1.09e-4 | -1.09e-4 | +1.05e-4 |
| 175 | 3.00e-3 | 2 | 5.96e-3 | 6.03e-3 | 6.00e-3 | 5.96e-3 | 183 | -5.98e-5 | +1.46e-4 | +4.29e-5 | +9.25e-5 |
| 176 | 3.00e-3 | 1 | 5.60e-3 | 5.60e-3 | 5.60e-3 | 5.60e-3 | 209 | -3.01e-4 | -3.01e-4 | -3.01e-4 | +5.31e-5 |
| 177 | 3.00e-3 | 2 | 5.57e-3 | 5.70e-3 | 5.63e-3 | 5.57e-3 | 183 | -1.33e-4 | +8.88e-5 | -2.20e-5 | +3.77e-5 |
| 178 | 3.00e-3 | 1 | 5.38e-3 | 5.38e-3 | 5.38e-3 | 5.38e-3 | 206 | -1.65e-4 | -1.65e-4 | -1.65e-4 | +1.74e-5 |
| 179 | 3.00e-3 | 1 | 5.65e-3 | 5.65e-3 | 5.65e-3 | 5.65e-3 | 239 | +2.04e-4 | +2.04e-4 | +2.04e-4 | +3.61e-5 |
| 180 | 3.00e-3 | 1 | 6.14e-3 | 6.14e-3 | 6.14e-3 | 6.14e-3 | 250 | +3.37e-4 | +3.37e-4 | +3.37e-4 | +6.62e-5 |
| 181 | 3.00e-3 | 1 | 6.28e-3 | 6.28e-3 | 6.28e-3 | 6.28e-3 | 254 | +8.35e-5 | +8.35e-5 | +8.35e-5 | +6.79e-5 |
| 182 | 3.00e-3 | 2 | 6.06e-3 | 6.40e-3 | 6.23e-3 | 6.06e-3 | 200 | -2.79e-4 | +9.37e-5 | -9.27e-5 | +3.55e-5 |
| 183 | 3.00e-3 | 1 | 5.61e-3 | 5.61e-3 | 5.61e-3 | 5.61e-3 | 228 | -3.32e-4 | -3.32e-4 | -3.32e-4 | -1.28e-6 |
| 184 | 3.00e-3 | 1 | 5.96e-3 | 5.96e-3 | 5.96e-3 | 5.96e-3 | 209 | +2.88e-4 | +2.88e-4 | +2.88e-4 | +2.76e-5 |
| 185 | 3.00e-3 | 2 | 5.69e-3 | 5.71e-3 | 5.70e-3 | 5.69e-3 | 186 | -2.20e-4 | -2.39e-5 | -1.22e-4 | +2.23e-7 |
| 186 | 3.00e-3 | 1 | 5.61e-3 | 5.61e-3 | 5.61e-3 | 5.61e-3 | 219 | -5.86e-5 | -5.86e-5 | -5.86e-5 | -5.66e-6 |
| 187 | 3.00e-3 | 2 | 5.88e-3 | 6.01e-3 | 5.95e-3 | 5.88e-3 | 186 | -1.17e-4 | +3.28e-4 | +1.05e-4 | +1.32e-5 |
| 188 | 3.00e-3 | 1 | 5.85e-3 | 5.85e-3 | 5.85e-3 | 5.85e-3 | 206 | -2.29e-5 | -2.29e-5 | -2.29e-5 | +9.55e-6 |
| 189 | 3.00e-3 | 1 | 5.97e-3 | 5.97e-3 | 5.97e-3 | 5.97e-3 | 236 | +8.42e-5 | +8.42e-5 | +8.42e-5 | +1.70e-5 |
| 190 | 3.00e-3 | 1 | 6.15e-3 | 6.15e-3 | 6.15e-3 | 6.15e-3 | 253 | +1.18e-4 | +1.18e-4 | +1.18e-4 | +2.71e-5 |
| 191 | 3.00e-3 | 2 | 6.07e-3 | 6.51e-3 | 6.29e-3 | 6.07e-3 | 184 | -3.81e-4 | +2.57e-4 | -6.22e-5 | +6.98e-6 |
| 192 | 3.00e-3 | 1 | 5.95e-3 | 5.95e-3 | 5.95e-3 | 5.95e-3 | 178 | -1.06e-4 | -1.06e-4 | -1.06e-4 | -4.32e-6 |
| 193 | 3.00e-3 | 2 | 5.50e-3 | 5.93e-3 | 5.72e-3 | 5.93e-3 | 163 | -4.22e-4 | +4.58e-4 | +1.77e-5 | +4.27e-6 |
| 194 | 3.00e-3 | 1 | 5.29e-3 | 5.29e-3 | 5.29e-3 | 5.29e-3 | 194 | -5.92e-4 | -5.92e-4 | -5.92e-4 | -5.54e-5 |
| 195 | 3.00e-3 | 2 | 5.90e-3 | 5.96e-3 | 5.93e-3 | 5.96e-3 | 163 | +6.28e-5 | +5.45e-4 | +3.04e-4 | +1.05e-5 |
| 196 | 3.00e-3 | 1 | 5.34e-3 | 5.34e-3 | 5.34e-3 | 5.34e-3 | 190 | -5.78e-4 | -5.78e-4 | -5.78e-4 | -4.84e-5 |
| 197 | 3.00e-3 | 2 | 5.64e-3 | 5.88e-3 | 5.76e-3 | 5.64e-3 | 152 | -2.68e-4 | +5.22e-4 | +1.27e-4 | -1.90e-5 |
| 198 | 3.00e-3 | 2 | 5.42e-3 | 5.54e-3 | 5.48e-3 | 5.54e-3 | 165 | -2.48e-4 | +1.30e-4 | -5.93e-5 | -2.47e-5 |
| 199 | 3.00e-3 | 1 | 5.49e-3 | 5.49e-3 | 5.49e-3 | 5.49e-3 | 206 | -4.01e-5 | -4.01e-5 | -4.01e-5 | -2.63e-5 |

