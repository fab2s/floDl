//! Convolutional autoencoder on MNIST.
//!
//! Ref: Standard PyTorch tutorials.
//! Expected: MSE monotonically decreasing.

use std::sync::Arc;

use flodl::autograd::Variable;
use flodl::data::BatchDataSet;
use flodl::nn::Module;
use flodl::tensor::{Device, Result, Tensor};
use flodl::*;

use super::{DatasetConfig, ModelDef};
use crate::config::ModelDefaults;
use flodl::nn::Adam;

pub fn def() -> ModelDef {
    ModelDef {
        name: "conv-ae",
        description: "Conv autoencoder on MNIST (PyTorch tutorial)",
        build: build_model,
        dataset: make_dataset,
        train_fn: train_step,
        eval_fn: Some(eval_mse),
        test_dataset: Some(make_test_dataset),
        augment_fn: None,
        optimizer: |p, lr| Box::new(Adam::new(p, lr)),
        scheduler: None,
        reference: "MNIST reconstruction, eval=MSE ([PyTorch AE tutorial](https://pytorch.org/tutorials/beginner/introyt/autoencoders_intro.html))",
        eval_higher_is_better: false,
        published_eval: None,
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
    // Encoder: (1,28,28) -> (16,14,14) -> (32,7,7)
    // Decoder: (32,7,7) -> (16,14,14) -> (1,28,28)
    let model = FlowBuilder::from(
        Conv2d::configure(1, 16, 3)
            .with_stride(2)
            .with_padding(1)
            .on_device(device)
            .done()?,
    )
    .through(ReLU)
    .through(
        Conv2d::configure(16, 32, 3)
            .with_stride(2)
            .with_padding(1)
            .on_device(device)
            .done()?,
    )
    .through(ReLU)
    // Decoder
    .through(
        ConvTranspose2d::configure(32, 16, 3)
            .with_stride(2)
            .with_padding(1)
            .with_output_padding(1)
            .on_device(device)
            .done()?,
    )
    .through(ReLU)
    .through(
        ConvTranspose2d::configure(16, 1, 3)
            .with_stride(2)
            .with_padding(1)
            .with_output_padding(1)
            .on_device(device)
            .done()?,
    )
    .through(Sigmoid)
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
    // Autoencoder: input = target = images
    let input = Variable::new(batch[0].clone(), false);
    let target = Variable::new(batch[0].clone(), false);
    let pred = model.forward(&input)?;
    mse_loss(&pred, &target)
}

/// Reconstruction MSE on a held-out batch (lower is better).
fn eval_mse(model: &dyn Module, batch: &[Tensor]) -> Result<f64> {
    let input = Variable::new(batch[0].clone(), false);
    let target = Variable::new(batch[0].clone(), false);
    let pred = model.forward(&input)?;
    mse_loss(&pred, &target)?.item()
}
