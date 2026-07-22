# Research Questions

Open questions and hypotheses that emerge from flodl development and benchmarking.
Not claims - observations worth investigating.

## Parameter averaging vs gradient averaging at scale

**Observation (v0.4.0 DDP benchmark):** CPU-sync (parameter averaging) consistently
achieves 0.15-0.25pp better test accuracy than NCCL-sync (gradient averaging) on
ResNet-20 CIFAR-10 across 200 epochs, despite identical training data and hyperparameters.

- ResNet-20 Graph: cpu-sync 92.53% vs nccl-sync 92.38%
- ResNet-20 manual: cpu-sync 92.44% vs nccl-sync 92.19%

**Hypothesis:** The CPU backend averages *parameters* after each GPU steps independently.
NCCL averages *gradients* before stepping. Parameter averaging is equivalent to continuous
Stochastic Weight Averaging (SWA), which is known to find wider minima with better
generalization ([Izmailov et al. 2018](https://arxiv.org/abs/1803.05407)).

**Implication for frontier scale:** If parameter averaging produces better generalization
than gradient AllReduce, a fire-and-forget parameter shipping architecture (no AllReduce
barrier, no GPU stall) could outperform synchronized gradient exchange at thousand-GPU
scale. The bottleneck shifts from network-wide AllReduce coordination to staleness
management - which El Che's anchor/cadence mechanism already handles for heterogeneous
local GPUs.

**What would confirm/refute this:**
- Longer training runs (500-1000 epochs) to check if the gap persists or closes
- Larger models where the parameter averaging cost is higher relative to compute
- Multi-node experiments where network latency amplifies the AllReduce penalty
- Controlled comparison with explicit SWA on the NCCL path to isolate the averaging effect

## El Che calibration time vs training length

**Observation:** On 5-epoch runs, El Che cadence/async modes show slightly lower eval
than sync modes. On 200-epoch runs, they match or exceed sync.

**Hypothesis:** El Che needs N epochs to calibrate the optimal sync interval via EMA
throughput tracking. On short runs, the calibration overhead dominates. The crossover
point depends on model size and hardware heterogeneity.

**What would help:** Measuring the exact calibration epoch (when anchor stabilizes)
across different speed ratios and model sizes.

## Heterogeneous GPU utilization floor

**Observation:** With a 2.5x speed gap (RTX 5060 Ti vs GTX 1060), async/cadence modes
achieve 100% utilization on both GPUs while sync modes drop to 99% on the fast GPU.

**Question:** Is there a speed ratio beyond which async mode's parameter staleness
degrades convergence faster than the throughput gain? The current 2.5x gap works
well. What about 5x? 10x?

## Graph DDP vs builder DDP convergence

**Observation:** ResNet-20 Graph (Trainer::setup) and ResNet-20 manual (Trainer::builder)
produce nearly identical convergence on the same architecture, but through very
different code paths. The Graph path scatters input across replicas inside forward().
The builder path gives each GPU its own data shard and training loop.

**Question:** Do these paths remain equivalent on models with stateful layers
(BatchNorm running statistics, dropout masks)? The Graph replicates buffers
differently from the builder's per-worker copies.

## DDP as a dynamical system: chaos theory analogies

**Observation:** El Che's divergence metric - `||pre - post|| / ||post||` measured
between sync points - behaves like a discrete-time Lyapunov exponent. It measures
how fast replica trajectories diverge from the last synchronization point.

Empirical data from ResNet-20 CIFAR-10 (200 epochs, nccl-cadence):
- Epoch 0-1: divergence 0.12 -> 0.04 (chaotic regime, high LR)
- Epoch 2-99: 0.025-0.04 (dynamic equilibrium, steady state)
- Epoch 100+ (LR/10): 0.0002 (stable attractor, replicas converge)
- Epoch 130+: 0.00002 (replicas essentially identical)

**The strange attractor interpretation:** Each GPU replica follows a different
trajectory through the loss landscape, perturbed by different data batches. Between
syncs they explore independently - chaotic trajectories diverging from slightly
different initial conditions (the last sync point). Averaging collapses them back
to a centroid.

That centroid is unreachable by any single trajectory. It's the average of multiple
chaotic explorations - a point in weight space no individual SGD path would visit.
This is the "strange attractor" of the averaging dynamics: a basin the system orbits
around but never lands on through any single path.

**The convergence guard as regime detector:** El Che's guard watches for 3 consecutive
rising divergence values above threshold. In dynamical systems terms, this detects a
positive Lyapunov exponent - the system transitioning from stable to chaotic. The
response (tighten cadence = sync more often) bounds the divergence, keeping replicas
within the basin of the attractor.

The LR schedule creates phase transitions visible in the divergence curve:
- **High LR phase:** chaotic regime, high divergence, replicas explore widely.
  Averaging provides maximum implicit regularization here.
- **Post-LR-drop:** rapid collapse to stable attractor. Replicas converge.
  Averaging overhead becomes pure cost with diminishing benefit.

**Connection to parameter averaging advantage:** If the centroid (parameter average)
lives in a wider minimum than any individual trajectory would find, this explains
why cpu-sync (parameter averaging) generalizes better than nccl-sync (gradient
averaging). Gradient averaging keeps replicas on the same trajectory. Parameter
averaging lets them diverge and finds the centroid - which by construction sits
in a wider basin.

This connects to the Stochastic Weight Averaging literature
([Izmailov et al. 2018](https://arxiv.org/abs/1803.05407)), but with a key
difference: SWA averages checkpoints along a single trajectory. DDP parameter
averaging combines simultaneous trajectories through different data. The
exploration is parallel, not serial.

**Research direction:** If the `||pre - post|| / ||post||` metric can be
formalized as a discrete-time Lyapunov exponent for Local SGD, the connection
between periodic averaging cadence and dynamical systems stability criteria
could be a publishable result. The empirical phase transitions in the divergence
curve (chaotic -> equilibrium -> stable attractor) map directly to the LR
schedule, providing a dynamical systems interpretation of learning rate annealing
in distributed training.

## Implicit regularization from trajectory diversity

**Observation (v0.3.0):** Async DDP achieved 87% accuracy in 5 epochs on the same
architecture where sync DDP reached only 4.2%. The diverged-then-averaged Local SGD
trajectories produced implicit regularization that accelerated convergence.

**Hypothesis:** The mechanism is analogous to ensemble methods. Each replica is a
"weak learner" - individually wrong, but wrong in different ways because they see
different data. Averaging their parameters cancels uncorrelated errors and preserves
the shared signal. More diverse trajectories (longer cadence intervals, async timing)
produce stronger regularization, up to the point where staleness degrades the signal.

**The cadence sweet spot:** Too frequent syncing (sync mode) keeps replicas identical,
eliminating diversity. Too infrequent syncing lets replicas diverge beyond the basin
of useful averaging. El Che's auto-tuned cadence searches for the boundary - maximum
diversity that still averages constructively.
