//! ResNet-20 on CIFAR-10.
//!
//! Ref: He et al. 2015 "Deep Residual Learning for Image Recognition" (Table 1).
//! Architecture: conv1 + 3 layers x 3 blocks (16/32/64 channels) + global avg pool + fc.
//! Expected: ~90%+ accuracy @ 5 epochs with Adam 1e-3.

use std::sync::Arc;

use flodl::autograd::{self, Variable};
use flodl::data::BatchDataSet;
use flodl::nn::{
    BatchNorm2d, Buffer, Conv2d, Linear, Module, MultiStepLR, Parameter,
    SGD,
};
use flodl::tensor::{DType, Device, Result, Tensor, TensorOptions};

use super::{DatasetConfig, ModelDef};
use crate::config::ModelDefaults;

// CIFAR-10 per-channel normalization constants (computed from training set).
const CIFAR10_MEAN: [f32; 3] = [0.4914, 0.4822, 0.4465];
const CIFAR10_STD: [f32; 3] = [0.2023, 0.1994, 0.2010];

pub fn def() -> ModelDef {
    ModelDef {
        name: "resnet",
        description: "ResNet-20 on CIFAR-10 (~91% acc, He et al. 2015)",
        build: build_model,
        dataset: make_dataset,
        train_fn: train_step,
        eval_fn: Some(eval_accuracy),
        test_dataset: Some(make_test_dataset),
        augment_fn: Some(augment_cifar10),
        // He et al. 2015: SGD, momentum=0.9, weight_decay=1e-4, LR=0.1
        // MultiStep at 50% and 75% of training, gamma=0.1 (published: [100,150]/200 epochs)
        optimizer: |p, lr| Box::new(SGD::new(p, lr, 0.9).weight_decay(1e-4)),
        scheduler: Some(|lr, total, _world_size| Box::new(MultiStepLR::new(lr, &[total / 2, total * 3 / 4], 0.1))),
        reference: "CIFAR-10 91.25% acc ([He et al. 2015](https://arxiv.org/abs/1512.03385), Table 6)",
        eval_higher_is_better: true,
        published_eval: Some(0.9125),
        needs_baseline_eval: true,
        defaults: ModelDefaults {
            epochs: 200,
            batches_per_epoch: 0, // full dataset
            batch_size: 64,
            lr: 0.1,
        },
    }
}

fn build_model(device: Device) -> Result<Box<dyn Module>> {
    Ok(Box::new(ResNet20::new(device)?))
}

pub fn make_dataset(cfg: &DatasetConfig) -> Result<Arc<dyn BatchDataSet>> {
    let cifar = crate::download::ensure_cifar10(&cfg.data_dir)?;
    Ok(Arc::new(cifar))
}

pub fn make_test_dataset(cfg: &DatasetConfig) -> Result<Arc<dyn BatchDataSet>> {
    let cifar = crate::download::ensure_cifar10_test(&cfg.data_dir)?;
    Ok(Arc::new(cifar))
}

pub fn train_step(model: &dyn Module, batch: &[Tensor]) -> Result<Variable> {
    let input = Variable::new(batch[0].clone(), false);
    let target = Variable::new(batch[1].to_dtype(DType::Int64)?, false);
    let pred = model.forward(&input)?;
    flodl::cross_entropy_loss(&pred, &target)
}

pub fn eval_accuracy(model: &dyn Module, batch: &[Tensor]) -> Result<f64> {
    // Normalize test images (same constants as training augmentation).
    let normed = cifar10_normalize(&batch[0])?;
    let input = Variable::new(normed, false);
    let pred = model.forward(&input)?;
    let predicted = pred.data().argmax(-1, false)?;
    let labels = batch[1].to_dtype(DType::Int64)?;
    let correct: f64 = predicted.eq_tensor(&labels)?.sum()?.item()?;
    let total = labels.shape()[0] as f64;
    Ok(correct / total)
}

// ---------------------------------------------------------------------------
// CIFAR-10 normalization and augmentation (He et al. 2015 Section 4.2)
// ---------------------------------------------------------------------------

/// Per-channel normalize: (x - mean) / std.
pub fn cifar10_normalize(images: &Tensor) -> Result<Tensor> {
    let device = images.device();
    let mean = Tensor::from_f32(&CIFAR10_MEAN, &[1, 3, 1, 1], device)?;
    let std = Tensor::from_f32(&CIFAR10_STD, &[1, 3, 1, 1], device)?;
    images.sub(&mean)?.div(&std)
}

/// Standard CIFAR-10 training augmentation:
/// 1. Per-channel normalize
/// 2. Pad 4px (zero = mean in normalized space)
/// 3. Random crop 32x32
/// 4. Random horizontal flip (p=0.5)
pub fn augment_cifar10(batch: &[Tensor]) -> Result<Vec<Tensor>> {
    let images = cifar10_normalize(&batch[0])?;

    // Pad 4px on each spatial side: [B,3,32,32] -> [B,3,40,40]
    let padded = images.pad(&[4, 4, 4, 4], 0.0)?;

    // Random crop offset in [0, 8] (same for whole batch -- standard GPU augmentation)
    let cpu = TensorOptions { dtype: DType::Int64, device: Device::CPU };
    let offsets = Tensor::randint(0, 9, &[3], cpu)?.to_i64_vec()?;
    let dy = offsets[0];
    let dx = offsets[1];
    let cropped = padded.narrow(2, dy, 32)?.narrow(3, dx, 32)?;

    // Random horizontal flip (p=0.5, whole batch)
    let result = if offsets[2] % 2 == 1 { cropped.flip(&[3])? } else { cropped };

    Ok(vec![result, batch[1].clone()])
}

// ---------------------------------------------------------------------------
// BasicBlock: Conv->BN->ReLU->Conv->BN + skip -> ReLU
// ---------------------------------------------------------------------------

struct BasicBlock {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    conv2: Conv2d,
    bn2: BatchNorm2d,
    downsample_conv: Option<Conv2d>,
    downsample_bn: Option<BatchNorm2d>,
}

impl BasicBlock {
    fn new(in_ch: i64, out_ch: i64, stride: i64, device: Device) -> Result<Self> {
        let conv1 = Conv2d::configure(in_ch, out_ch, 3)
            .with_stride(stride)
            .with_padding(1)
            .without_bias()
            .on_device(device)
            .done()?;
        let bn1 = BatchNorm2d::on_device(out_ch, device)?;
        let conv2 = Conv2d::configure(out_ch, out_ch, 3)
            .with_padding(1)
            .without_bias()
            .on_device(device)
            .done()?;
        let bn2 = BatchNorm2d::on_device(out_ch, device)?;

        let (downsample_conv, downsample_bn) = if stride != 1 || in_ch != out_ch {
            let dc = Conv2d::configure(in_ch, out_ch, 1)
                .with_stride(stride)
                .without_bias()
                .on_device(device)
                .done()?;
            let db = BatchNorm2d::on_device(out_ch, device)?;
            (Some(dc), Some(db))
        } else {
            (None, None)
        };

        Ok(BasicBlock { conv1, bn1, conv2, bn2, downsample_conv, downsample_bn })
    }

    fn forward(&self, x: &Variable) -> Result<Variable> {
        let identity = if let (Some(dc), Some(db)) = (&self.downsample_conv, &self.downsample_bn) {
            db.forward(&dc.forward(x)?)?
        } else {
            x.clone()
        };

        let out = self.conv1.forward(x)?;
        let out = self.bn1.forward(&out)?;
        let out = out.relu()?;
        let out = self.conv2.forward(&out)?;
        let out = self.bn2.forward(&out)?;
        let out = out.add(&identity)?;
        out.relu()
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters());
        params.extend(self.bn1.parameters());
        params.extend(self.conv2.parameters());
        params.extend(self.bn2.parameters());
        if let Some(ref dc) = self.downsample_conv {
            params.extend(dc.parameters());
        }
        if let Some(ref db) = self.downsample_bn {
            params.extend(db.parameters());
        }
        params
    }

    fn buffers(&self) -> Vec<Buffer> {
        let mut bufs = Vec::new();
        bufs.extend(self.bn1.buffers());
        bufs.extend(self.bn2.buffers());
        if let Some(ref db) = self.downsample_bn {
            bufs.extend(db.buffers());
        }
        bufs
    }

    fn set_training(&self, mode: bool) {
        self.bn1.set_training(mode);
        self.bn2.set_training(mode);
        if let Some(ref db) = self.downsample_bn {
            db.set_training(mode);
        }
    }
}

// ---------------------------------------------------------------------------
// ResNet-20
// ---------------------------------------------------------------------------

struct ResNet20 {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    layer1: Vec<BasicBlock>,
    layer2: Vec<BasicBlock>,
    layer3: Vec<BasicBlock>,
    fc: Linear,
}

impl ResNet20 {
    fn new(device: Device) -> Result<Self> {
        let conv1 = Conv2d::configure(3, 16, 3)
            .with_padding(1)
            .without_bias()
            .on_device(device)
            .done()?;
        let bn1 = BatchNorm2d::on_device(16, device)?;

        let layer1 = Self::make_layer(16, 16, 3, 1, device)?;
        let layer2 = Self::make_layer(16, 32, 3, 2, device)?;
        let layer3 = Self::make_layer(32, 64, 3, 2, device)?;

        let fc = Linear::on_device(64, 10, device)?;

        Ok(ResNet20 { conv1, bn1, layer1, layer2, layer3, fc })
    }

    fn make_layer(
        in_ch: i64, out_ch: i64, num_blocks: usize, stride: i64, device: Device,
    ) -> Result<Vec<BasicBlock>> {
        let mut blocks = Vec::with_capacity(num_blocks);
        blocks.push(BasicBlock::new(in_ch, out_ch, stride, device)?);
        for _ in 1..num_blocks {
            blocks.push(BasicBlock::new(out_ch, out_ch, 1, device)?);
        }
        Ok(blocks)
    }
}

impl Module for ResNet20 {
    fn name(&self) -> &str { "resnet20" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        let x = self.conv1.forward(input)?;
        let x = self.bn1.forward(&x)?;
        let mut x = x.relu()?;

        for block in &self.layer1 { x = block.forward(&x)?; }
        for block in &self.layer2 { x = block.forward(&x)?; }
        for block in &self.layer3 { x = block.forward(&x)?; }

        // Global average pool: [B, 64, H, W] -> [B, 64, 1, 1]
        let x = autograd::adaptive_avg_pool2d(&x, [1, 1])?;
        // Flatten: [B, 64, 1, 1] -> [B, 64]
        let x = x.reshape(&[x.shape()[0], -1])?;
        self.fc.forward(&x)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters());
        params.extend(self.bn1.parameters());
        for b in &self.layer1 { params.extend(b.parameters()); }
        for b in &self.layer2 { params.extend(b.parameters()); }
        for b in &self.layer3 { params.extend(b.parameters()); }
        params.extend(self.fc.parameters());
        params
    }

    fn buffers(&self) -> Vec<Buffer> {
        let mut bufs = Vec::new();
        bufs.extend(self.bn1.buffers());
        for b in &self.layer1 { bufs.extend(b.buffers()); }
        for b in &self.layer2 { bufs.extend(b.buffers()); }
        for b in &self.layer3 { bufs.extend(b.buffers()); }
        bufs
    }

    fn set_training(&self, mode: bool) {
        self.bn1.set_training(mode);
        for b in &self.layer1 { b.set_training(mode); }
        for b in &self.layer2 { b.set_training(mode); }
        for b in &self.layer3 { b.set_training(mode); }
    }
}
