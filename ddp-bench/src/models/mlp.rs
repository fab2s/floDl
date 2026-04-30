//! 2-layer MLP on MNIST.
//!
//! Ref: every DL textbook, PyTorch MNIST tutorial.
//! Expected: ~97-98% accuracy @ 5 epochs.

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
        name: "mlp",
        description: "2-layer MLP on MNIST (~97% acc)",
        build: build_model,
        dataset: make_dataset,
        train_fn: train_step,
        eval_fn: Some(eval_accuracy),
        test_dataset: Some(make_test_dataset),
        augment_fn: None,
        optimizer: |p, lr| Box::new(Adam::new(p, lr)),
        scheduler: None,
        reference: "MNIST ~97-98% acc ([PyTorch tutorial](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html))",
        eval_higher_is_better: true,
        published_eval: Some(0.975),
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
    let model = FlowBuilder::from(Flatten::default())
        .through(Linear::on_device(784, 256, device)?)
        .through(ReLU)
        .through(Linear::on_device(256, 10, device)?)
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
