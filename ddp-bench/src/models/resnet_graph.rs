//! ResNet-20 on CIFAR-10 -- Graph version.
//!
//! Same architecture as `resnet.rs` but expressed as a FlowBuilder graph.
//! Demonstrates `also()` / `also_with()` for residual connections.
//! Gets `named_parameters`, `record_scalar`, observation, and `set_training`
//! for free from the Graph infrastructure.

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

pub fn def() -> ModelDef {
    ModelDef {
        name: "resnet-graph",
        description: "ResNet-20 on CIFAR-10 (Graph builder, ~91% acc)",
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
        reference: "CIFAR-10 91.25% acc, Graph builder ([He et al. 2015](https://arxiv.org/abs/1512.03385), Table 6)",
        eval_higher_is_better: true,
        published_eval: Some(0.9125),
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
    let model = FlowBuilder::from(conv3x3(3, 16, 1, d)?)
        .through(BatchNorm2d::on_device(16, d)?)
        .through(ReLU)
        // Layer 1: 3 blocks, 16 channels, no downsample
        .also(res_main(16, 16, 1, d)?)
        .through(ReLU)
        .also(res_main(16, 16, 1, d)?)
        .through(ReLU)
        .also(res_main(16, 16, 1, d)?)
        .through(ReLU)
        // Layer 2: 3 blocks, 32 channels, first block downsamples
        .also_with(downsample(16, 32, 2, d)?, res_main(16, 32, 2, d)?)
        .through(ReLU)
        .also(res_main(32, 32, 1, d)?)
        .through(ReLU)
        .also(res_main(32, 32, 1, d)?)
        .through(ReLU)
        // Layer 3: 3 blocks, 64 channels, first block downsamples
        .also_with(downsample(32, 64, 2, d)?, res_main(32, 64, 2, d)?)
        .through(ReLU)
        .also(res_main(64, 64, 1, d)?)
        .through(ReLU)
        .also(res_main(64, 64, 1, d)?)
        .through(ReLU)
        // Head: global avg pool -> flatten -> FC
        .through(AdaptiveAvgPool2d::new([1, 1]))
        .through(Flatten::default())
        .through(Linear::on_device(64, 10, d)?)
        .tag("logits")
        .build()?;
    Ok(Box::new(model))
}
