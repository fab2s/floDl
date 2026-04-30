//! LeNet-5 on MNIST (modern variant with BN + ReLU).
//!
//! Ref: LeCun 1998 "Gradient-Based Learning Applied to Document Recognition".
//! Expected: ~99% accuracy @ 5 epochs.

use std::sync::Arc;

use flodl::autograd::Variable;
use flodl::data::BatchDataSet;
use flodl::nn::Module;
use flodl::tensor::{Device, DType, Result, Tensor};
use flodl::*;

use super::{DatasetConfig, ModelDef};
use crate::config::ModelDefaults;
use flodl::nn::Adam;

pub fn def() -> ModelDef {
    ModelDef {
        name: "lenet",
        description: "LeNet-5 on MNIST (~99% acc, LeCun 1998)",
        build: build_model,
        dataset: make_dataset,
        train_fn: train_step,
        eval_fn: Some(eval_accuracy),
        test_dataset: Some(make_test_dataset),
        augment_fn: None,
        optimizer: |p, lr| Box::new(Adam::new(p, lr)),
        scheduler: None,
        reference: "MNIST ~99% acc ([LeCun 1998](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf))",
        eval_higher_is_better: true,
        published_eval: Some(0.99),
        needs_baseline_eval: false,
        defaults: ModelDefaults {
            epochs: 5,
            batches_per_epoch: 0, // full dataset
            batch_size: 64,
            lr: 0.001,
        },
    }
}

fn build_model(device: Device) -> Result<Box<dyn Module>> {
    // LeNet-5 modern variant: BN + ReLU instead of Tanh
    // Input: [B, 1, 28, 28]
    // Conv1: 1->6, 5x5 -> [B, 6, 24, 24] -> MaxPool -> [B, 6, 12, 12]
    // Conv2: 6->16, 5x5 -> [B, 16, 8, 8] -> MaxPool -> [B, 16, 4, 4]
    // Flatten: 16*4*4 = 256
    let model = FlowBuilder::from(
        Conv2d::configure(1, 6, 5).on_device(device).done()?,
    )
    .through(BatchNorm2d::on_device(6, device)?)
    .through(ReLU)
    .through(MaxPool2d::new(2))
    .through(Conv2d::configure(6, 16, 5).on_device(device).done()?)
    .through(BatchNorm2d::on_device(16, device)?)
    .through(ReLU)
    .through(MaxPool2d::new(2))
    .through(Flatten::default())
    .through(Linear::on_device(256, 120, device)?)
    .through(ReLU)
    .through(Linear::on_device(120, 84, device)?)
    .through(ReLU)
    .through(Linear::on_device(84, 10, device)?)
    .build()?;
    Ok(Box::new(model))
}

fn make_dataset(cfg: &DatasetConfig) -> Result<Arc<dyn BatchDataSet>> {
    let mnist = crate::download::ensure_mnist(&cfg.data_dir)?;
    Ok(Arc::new(mnist))
}

fn make_test_dataset(cfg: &DatasetConfig) -> Result<Arc<dyn BatchDataSet>> {
    let mnist = crate::download::ensure_mnist_test(&cfg.data_dir)?;
    Ok(Arc::new(mnist))
}

fn train_step(model: &dyn Module, batch: &[Tensor]) -> Result<Variable> {
    let input = Variable::new(batch[0].clone(), false);
    let target = Variable::new(batch[1].to_dtype(DType::Int64)?, false);
    let pred = model.forward(&input)?;
    cross_entropy_loss(&pred, &target)
}

fn eval_accuracy(model: &dyn Module, batch: &[Tensor]) -> Result<f64> {
    let input = Variable::new(batch[0].clone(), false);
    let pred = model.forward(&input)?;
    let predicted = pred.data().argmax(-1, false)?;
    let labels = batch[1].to_dtype(DType::Int64)?;
    let correct: f64 = predicted.eq_tensor(&labels)?.sum()?.item()?;
    let total = labels.shape()[0] as f64;
    Ok(correct / total)
}
