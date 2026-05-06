//! ResNet-{20,32,44,56,110} on CIFAR-10 -- Graph version.
//!
//! He et al. 2015 CIFAR family: depth = 6n+2, where each of the 3 stages has
//! `n` BasicBlocks. Default `n=3` reproduces ResNet-20. Override per run with
//! the bench `--depth-n N` flag (sets `DEPTH_N` before the harness builds
//! the graph). Published Table 6 evals: n=3 91.25% (R-20), n=5 92.49% (R-32),
//! n=7 92.83% (R-44), n=9 93.03% (R-56), n=18 93.39% (R-110).
//!
//! Same architecture as `resnet.rs` but expressed as a FlowBuilder graph.
//! Demonstrates `also()` / `also_with()` for residual connections.
//! Gets `named_parameters`, `record_scalar`, observation, and `set_training`
//! for free from the Graph infrastructure.

use std::sync::atomic::{AtomicUsize, Ordering};

use flodl::autograd::Variable;
use flodl::graph::Graph;
use flodl::nn::{
    AdaptiveAvgPool2d, BatchNorm2d, Conv2d, Flatten, Linear, Module,
    MultiStepLR, ReLU, SGD,
};
use flodl::tensor::{DType, Device, Result, Tensor};
use flodl::FlowBuilder;

use super::ModelDef;
use crate::config::ModelDefaults;

/// Per-stage block count. `6 * DEPTH_N + 2` total layers. Default 3 = ResNet-20.
/// Set once from the bench `--depth-n` CLI flag before any model build.
static DEPTH_N: AtomicUsize = AtomicUsize::new(3);

/// Override the per-stage block count. n=3 → ResNet-20, n=9 → ResNet-56, etc.
pub fn set_depth_n(n: usize) {
    assert!(n >= 1, "ResNet depth-n must be >= 1");
    DEPTH_N.store(n, Ordering::SeqCst);
}

/// Current per-stage block count.
pub fn depth_n() -> usize {
    DEPTH_N.load(Ordering::SeqCst)
}

pub fn def() -> ModelDef {
    // Depth-dependent metadata. He et al. 2015 Table 6 published evals;
    // unrecognized depths fall back to ResNet-20 numbers (description tags
    // the actual layer count via 6n+2).
    let n = depth_n();
    let (description, reference, published) = match n {
        3 => (
            "ResNet-20 on CIFAR-10 (Graph builder, ~91% acc)",
            "CIFAR-10 91.25% acc, Graph builder ([He et al. 2015](https://arxiv.org/abs/1512.03385), Table 6)",
            Some(0.9125),
        ),
        5 => (
            "ResNet-32 on CIFAR-10 (Graph builder, ~92.5% acc)",
            "CIFAR-10 92.49% acc, Graph builder ([He et al. 2015](https://arxiv.org/abs/1512.03385), Table 6)",
            Some(0.9249),
        ),
        7 => (
            "ResNet-44 on CIFAR-10 (Graph builder, ~92.8% acc)",
            "CIFAR-10 92.83% acc, Graph builder ([He et al. 2015](https://arxiv.org/abs/1512.03385), Table 6)",
            Some(0.9283),
        ),
        9 => (
            "ResNet-56 on CIFAR-10 (Graph builder, ~93% acc)",
            "CIFAR-10 93.03% acc, Graph builder ([He et al. 2015](https://arxiv.org/abs/1512.03385), Table 6)",
            Some(0.9303),
        ),
        18 => (
            "ResNet-110 on CIFAR-10 (Graph builder, ~93.4% acc)",
            "CIFAR-10 93.39% acc, Graph builder ([He et al. 2015](https://arxiv.org/abs/1512.03385), Table 6)",
            Some(0.9339),
        ),
        _ => (
            "ResNet-{6n+2} on CIFAR-10 (Graph builder, depth-n custom)",
            "CIFAR-10 He et al. 2015 family, depth not in published Table 6",
            None,
        ),
    };
    ModelDef {
        name: "resnet-graph",
        description,
        build: build_model,
        dataset: super::resnet::make_dataset,
        train_fn: train_step,
        eval_fn: Some(super::resnet::eval_accuracy),
        test_dataset: Some(super::resnet::make_test_dataset),
        augment_fn: Some(super::resnet::augment_cifar10),
        optimizer: |p, lr| Box::new(SGD::new(p, lr, 0.9).weight_decay(1e-4)),
        scheduler: Some(|lr, total, _world_size| {
            Box::new(MultiStepLR::new(lr, &[total / 2, total * 3 / 4], 0.1))
        }),
        reference,
        eval_higher_is_better: true,
        published_eval: published,
        needs_baseline_eval: true,
        defaults: ModelDefaults {
            epochs: 200,
            batches_per_epoch: 0,
            batch_size: 64,
            lr: 0.1,
        },
    }
}

// ---------------------------------------------------------------------------
// Training step with accuracy tracking
// ---------------------------------------------------------------------------

fn train_step(model: &dyn Module, batch: &[Tensor]) -> Result<Variable> {
    let input = Variable::new(batch[0].clone(), false);
    let target = Variable::new(batch[1].to_dtype(DType::Int64)?, false);
    let pred = model.forward(&input)?;

    // Record per-batch training accuracy (aggregated across DDP ranks).
    let predicted = pred.data().argmax(-1, false)?;
    let correct: f64 = predicted.eq_tensor(&target.data())?.sum()?.item()?;
    let total = target.data().shape()[0] as f64;
    flodl::record_scalar("train_acc", correct / total);

    flodl::cross_entropy_loss(&pred, &target)
}

// ---------------------------------------------------------------------------
// Helpers: sub-graphs for residual blocks
// ---------------------------------------------------------------------------

fn conv3x3(in_ch: i64, out_ch: i64, stride: i64, d: Device) -> Result<Conv2d> {
    Conv2d::configure(in_ch, out_ch, 3)
        .with_stride(stride)
        .with_padding(1)
        .without_bias()
        .on_device(d)
        .done()
}

fn conv1x1(in_ch: i64, out_ch: i64, stride: i64, d: Device) -> Result<Conv2d> {
    Conv2d::configure(in_ch, out_ch, 1)
        .with_stride(stride)
        .without_bias()
        .on_device(d)
        .done()
}

/// Main path of a BasicBlock: conv3x3 -> BN -> ReLU -> conv3x3 -> BN.
fn res_main(in_ch: i64, out_ch: i64, stride: i64, d: Device) -> Result<Graph> {
    FlowBuilder::from(conv3x3(in_ch, out_ch, stride, d)?)
        .through(BatchNorm2d::on_device(out_ch, d)?)
        .through(ReLU)
        .through(conv3x3(out_ch, out_ch, 1, d)?)
        .through(BatchNorm2d::on_device(out_ch, d)?)
        .build()
}

/// Downsample skip path: 1x1 conv -> BN (matches spatial/channel dims).
fn downsample(in_ch: i64, out_ch: i64, stride: i64, d: Device) -> Result<Graph> {
    FlowBuilder::from(conv1x1(in_ch, out_ch, stride, d)?)
        .through(BatchNorm2d::on_device(out_ch, d)?)
        .build()
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

fn build_model(device: Device) -> Result<Box<dyn Module>> {
    let d = device;
    let n = depth_n();

    // Stem: conv3x3(3->16) + BN + ReLU
    let mut fb = FlowBuilder::from(conv3x3(3, 16, 1, d)?)
        .through(BatchNorm2d::on_device(16, d)?)
        .through(ReLU);

    // Stage 1: n blocks, 16 channels, no downsample (in_ch == out_ch, stride=1).
    for _ in 0..n {
        fb = fb.also(res_main(16, 16, 1, d)?).through(ReLU);
    }

    // Stage 2: n blocks, 32 channels. First block downsamples (16->32, stride=2).
    fb = fb
        .also_with(downsample(16, 32, 2, d)?, res_main(16, 32, 2, d)?)
        .through(ReLU);
    for _ in 1..n {
        fb = fb.also(res_main(32, 32, 1, d)?).through(ReLU);
    }

    // Stage 3: n blocks, 64 channels. First block downsamples (32->64, stride=2).
    fb = fb
        .also_with(downsample(32, 64, 2, d)?, res_main(32, 64, 2, d)?)
        .through(ReLU);
    for _ in 1..n {
        fb = fb.also(res_main(64, 64, 1, d)?).through(ReLU);
    }

    // Head: global avg pool -> flatten -> FC
    let model = fb
        .through(AdaptiveAvgPool2d::new([1, 1]))
        .through(Flatten::default())
        .through(Linear::on_device(64, 10, d)?)
        .tag("logits")
        .build()?;
    Ok(Box::new(model))
}
