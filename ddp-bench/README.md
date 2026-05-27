# ddp-bench

DDP validation and benchmark suite for flodl. Reproduces published training setups to create scientifically valid solo baselines, then measures DDP/ElChe convergence quality against them.

## Models

### Logistic Regression -- MNIST

Flatten(784) -> Linear(784, 10). Cross-entropy loss.

- **Optimizer**: Adam, lr=1e-3
- **Schedule**: flat
- **Epochs**: 5
- **Expected**: ~92% accuracy
- **Reference**: standard ML baseline

### MLP -- MNIST

Flatten(784) -> Linear(784, 256) -> ReLU -> Linear(256, 10). Cross-entropy loss.

- **Optimizer**: Adam, lr=1e-3
- **Schedule**: flat
- **Epochs**: 5
- **Expected**: ~97-98% accuracy
- **Reference**: PyTorch MNIST tutorial

### LeNet-5 -- MNIST

Conv2d(1,6,5) -> BN -> ReLU -> MaxPool(2) -> Conv2d(6,16,5) -> BN -> ReLU -> MaxPool(2) -> Linear(256,120) -> ReLU -> Linear(120,84) -> ReLU -> Linear(84,10). Cross-entropy loss.

Modern variant with BatchNorm and ReLU (original used sigmoid/tanh).

- **Optimizer**: Adam, lr=1e-3
- **Schedule**: flat
- **Epochs**: 5
- **Expected**: ~99% accuracy
- **Reference**: LeCun et al., 1998. "Gradient-Based Learning Applied to Document Recognition." *Proceedings of the IEEE*, 86(11), 2278-2324.

### ResNet-20 -- CIFAR-10

conv1(3,16,3) -> BN -> ReLU -> 3x BasicBlock(16) -> 3x BasicBlock(32, stride=2) -> 3x BasicBlock(64, stride=2) -> AdaptiveAvgPool(1,1) -> Linear(64, 10). Cross-entropy loss.

BasicBlock: Conv(3x3) -> BN -> ReLU -> Conv(3x3) -> BN + skip connection -> ReLU.

- **Optimizer**: SGD, momentum=0.9, weight_decay=1e-4, lr=0.1
- **Schedule**: MultiStepLR at 50% and 75% of training, gamma=0.1
- **Epochs**: 200
- **Expected**: ~91-92% test accuracy (paper Table 6: 91.25% on CIFAR-10)
- **Reference**: He et al., 2015. "Deep Residual Learning for Image Recognition." *arXiv:1512.03385*. Table 1 (ResNet-20), Table 6 (CIFAR-10 results).

### Char-RNN -- Shakespeare

Embedding(65, 128) -> LSTM(128, 256, 2 layers) -> Dropout(0.2) -> Linear(256, 65). Cross-entropy loss on next-character prediction.

- **Optimizer**: RMSprop, lr=2e-3
- **Schedule**: flat
- **Epochs**: 50
- **Expected**: CE loss ~1.5 at convergence
- **Reference**: Karpathy, 2015. "The Unreasonable Effectiveness of Recurrent Neural Networks." Blog post. Code: github.com/karpathy/char-rnn.

### GPT-nano -- Shakespeare

4 pre-norm transformer layers, 4 heads, d_model=128, d_ff=512. Token + positional embeddings, causal mask, LayerNorm -> MultiheadAttention -> residual -> LayerNorm -> FFN(GELU) -> residual. Cross-entropy loss on next-character prediction.

- **Optimizer**: Adam, lr=3e-4
- **Schedule**: warmup 20% of training, then cosine decay to 1e-5
- **Epochs**: 50
- **Expected**: CE loss ~1.5-1.6 at convergence
- **Reference**: Karpathy, 2022. nanoGPT. github.com/karpathy/nanoGPT. Architecture follows Vaswani et al., 2017. "Attention Is All You Need." *NeurIPS*.

### Conv Autoencoder -- MNIST

Encoder: Conv2d(1,16,3,s=2,p=1) -> ReLU -> Conv2d(16,32,3,s=2,p=1) -> ReLU.
Decoder: ConvTranspose2d(32,16,3,s=2,p=1,op=1) -> ReLU -> ConvTranspose2d(16,1,3,s=2,p=1,op=1) -> Sigmoid. MSE loss (target = input).

- **Optimizer**: Adam, lr=1e-3
- **Schedule**: flat
- **Epochs**: 5
- **Expected**: MSE monotonically decreasing
- **Reference**: standard PyTorch autoencoder tutorial

## DDP Modes

Each mode maps 1:1 to an `ElCheMode` variant in `flodl`. The framework
default is `nccl-async` (`ElCheConfig::default()`). See
[DDP Reference: ElCheMode](../docs/ddp.md#elchemode--cadence--backend-in-one-name)
for full semantics.

| Mode | `ElCheMode` | Description |
|------|---|---|
| `solo-0` | -- | Single GPU (device 0), no DDP. Baseline. |
| `solo-1` | -- | Single GPU (device 1), no DDP. |
| `solo-2` | -- | Single GPU (device 2), no DDP. |
| `nccl-sync` | `NcclSync` | NCCL AllReduce every batch. Traditional DDP / strict-sync baseline. |
| `nccl-cadence` | `NcclCadence` | ElChe-driven anchor; cross-epoch lockstep. |
| `nccl-async` | `NcclAsync` | **Default.** Same in-epoch loop as `nccl-cadence` + cross-epoch lookahead — fast rank gets head start on next epoch. |
| `cpu-sync` | `CpuSync` | CPU-mediated parameter averaging, every batch. |
| `cpu-cadence` | `CpuCadence` | CPU averaging with ElChe-driven anchor. |
| `cpu-async` | `CpuAsync` | **Best-in-class** on the reference rig — fastest wall-time, best convergence. Opt into EASGD with `ElCheConfig::cpu_async().easgd_alpha(α)` in code; the bench harness exercises it without EASGD by default. |

## Usage

```bash
# Solo baseline
fdl ddp-bench --model gpt-nano --mode solo-0 --epochs 50

# ElChe cadence with LR scaling for 2 GPUs
fdl ddp-bench --model gpt-nano --mode nccl-cadence --epochs 50 --lr-scale 2

# All models, solo baseline
fdl ddp-bench --model all --mode solo-0

# List available models and modes
fdl ddp-bench --list

# Generate report from saved runs
fdl ddp-bench --report
```

## LR Scaling for Multi-GPU

With multiple GPUs, the per-batch LR schedule advances faster (global_step counts all GPUs' batches). Use `--lr-scale <N>` to compensate:

- `--lr-scale 2` for 2 GPUs (linear scaling rule, Goyal et al. 2017)
- The framework also supports `lr_scale_ratio` (default 1.0) which auto-computes the factor from world_size

## Output

Each run saves to `runs/<model>/<mode>/`:
- `training.log` -- epoch-level loss and metrics
- `timeline.json` / `timeline.csv` / `timeline.html` -- high-frequency profiling data
