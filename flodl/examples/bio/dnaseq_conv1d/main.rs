//! DNA-sequence Conv1D classifier: synthetic ChIP-seq-style motif detection.
//!
//! One-hot encodes DNA into `[4, L]` tensors, plants a consensus motif in half
//! the samples, and trains a `Conv1d` / `ReLU` / `MaxPool1d` / `Linear` stack
//! with binary cross-entropy to detect motif presence, a peak-annotation toy
//! for genome-annotation work. To train on real data, replace `make_example`
//! (which fabricates a sequence and a 0/1 label) with a loader that yields your
//! own `(sequence, label)` pairs (e.g. sequences from a FASTA and labels from
//! your peak annotations), then feed them through `one_hot_encode` and
//! `build_batches` unchanged.
//!
//! Adapted from an example contributed by Gaurav Sablok (@gsablok),
//! flodl issue #14, updated to the current flodl API.
//!
//! Run: `cargo run --release --example dnaseq_conv1d`

use flodl::monitor::Monitor;
use flodl::*;

const SEQ_LEN: usize = 200;
const MOTIF: &str = "TGACGTCA"; // stand-in consensus motif (e.g. a CREB site)
const NUM_TRAIN: usize = 4000;
const NUM_VAL: usize = 500;
const BATCH_SIZE: usize = 64;
const NUM_EPOCHS: usize = 15;

const BASES: [char; 4] = ['A', 'C', 'G', 'T'];

/// One-hot encode a DNA string into a `[4, L]` tensor (A, C, G, T channels).
fn one_hot_encode(seq: &str) -> Tensor {
    let mut data = vec![0f32; 4 * SEQ_LEN];
    for (i, base) in seq.chars().enumerate().take(SEQ_LEN) {
        let channel = match base {
            'A' => 0,
            'C' => 1,
            'G' => 2,
            'T' => 3,
            _ => continue, // 'N' or anything unknown -> all-zero column
        };
        data[channel * SEQ_LEN + i] = 1.0;
    }
    Tensor::from_f32(&data, &[4, SEQ_LEN as i64], Device::CPU).expect("failed to build one-hot tensor")
}

/// Generate a random DNA string of length `len`.
fn random_seq(len: usize, rng: &mut Rng) -> String {
    (0..len).map(|_| BASES[rng.usize(4)]).collect()
}

/// Build one labeled example: half the time plant the motif at a random
/// position (label 1), otherwise leave the sequence pure noise (label 0).
fn make_example(rng: &mut Rng) -> (Tensor, f32) {
    let mut seq: Vec<char> = random_seq(SEQ_LEN, rng).chars().collect();
    let has_motif = rng.bernoulli(0.5);

    if has_motif {
        let motif: Vec<char> = MOTIF.chars().collect();
        let max_start = SEQ_LEN - motif.len();
        let start = rng.usize(max_start + 1); // 0..=max_start
        seq[start..start + motif.len()].copy_from_slice(&motif);
    }

    let seq_str: String = seq.into_iter().collect();
    (one_hot_encode(&seq_str), if has_motif { 1.0 } else { 0.0 })
}

/// Build a full dataset of (input, label) tensors, batched.
fn build_batches(n: usize, rng: &mut Rng) -> Vec<(Tensor, Tensor)> {
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for _ in 0..n {
        let (x, y) = make_example(rng);
        xs.push(x);
        ys.push(y);
    }

    xs.chunks(BATCH_SIZE)
        .zip(ys.chunks(BATCH_SIZE))
        .map(|(xb, yb)| {
            // stack expects &[&Tensor]
            let refs: Vec<&Tensor> = xb.iter().collect();
            let x_batch = Tensor::stack(&refs, 0).expect("stack inputs"); // [B, 4, L]
            let y_batch =
                Tensor::from_f32(yb, &[yb.len() as i64, 1], Device::CPU).expect("build labels"); // [B, 1]
            (x_batch, y_batch)
        })
        .collect()
}

fn build_model() -> Result<Graph> {
    FlowBuilder::from(Conv1d::new(4, 32, 5)?) // [B,4,200] -> [B,32,196]
        .through(ReLU)
        .through(Conv1d::configure(32, 64, 5).with_padding(2).done()?) // -> [B,64,196]
        .through(ReLU)
        .through(MaxPool1d::new(2)) // -> [B,64,98]
        .through(Conv1d::configure(64, 128, 3).with_padding(1).done()?) // -> [B,128,98]
        .through(ReLU)
        .through(MaxPool1d::new(2)) // -> [B,128,49]
        .through(Flatten::new(1, -1)) // -> [B, 128*49]
        .through(Linear::new(128 * 49, 64)?)
        .through(ReLU)
        .through(Dropout::new(0.3))
        .through(Linear::new(64, 1)?)
        .through(Sigmoid)
        .build()
}

fn main() -> Result<()> {
    manual_seed(42);
    let mut rng = Rng::seed(42);

    println!("Generating synthetic DNAseq dataset...");
    println!("  motif: {MOTIF}  |  sequence length: {SEQ_LEN}");
    let train_batches = build_batches(NUM_TRAIN, &mut rng);
    let val_batches = build_batches(NUM_VAL, &mut rng);

    let model = build_model()?;
    let params = model.parameters();
    let mut optimizer = Adam::new(&params, 1e-3);

    let mut monitor = Monitor::new(NUM_EPOCHS);
    // Optional live dashboard: uncomment to watch training in the browser.
    // monitor.serve(3000)?;

    for epoch in 0..NUM_EPOCHS {
        let t0 = std::time::Instant::now();
        model.train();

        for (xb, yb) in &train_batches {
            let input = Variable::new(xb.clone(), true);
            let target = Variable::new(yb.clone(), false);

            let pred = model.forward(&input)?;
            let loss = bce_loss(&pred, &target)?;

            optimizer.zero_grad();
            loss.backward()?;
            clip_grad_norm(&params, 1.0)?;
            optimizer.step()?;

            model.record_scalar("loss", loss.item()?);
        }

        // Quick validation pass: accuracy at a 0.5 threshold.
        model.eval();
        let mut correct = 0usize;
        let mut total = 0usize;
        for (xb, yb) in &val_batches {
            let input = Variable::new(xb.clone(), false);
            let pred = model.forward(&input)?;
            let pred_vals: Vec<f32> = pred.data().to_f32_vec()?;
            let target_vals: Vec<f32> = yb.to_f32_vec()?;
            for (p, t) in pred_vals.iter().zip(target_vals.iter()) {
                let predicted_label = if *p > 0.5 { 1.0 } else { 0.0 };
                if (predicted_label - t).abs() < 1e-6 {
                    correct += 1;
                }
                total += 1;
            }
        }
        let val_acc = correct as f32 / total as f32;
        model.record_scalar("val_accuracy", val_acc as f64);

        model.flush(&[]);
        monitor.log(epoch, t0.elapsed(), &model);
        println!("  epoch {epoch:>3}: val_accuracy = {val_acc:.3}");

        if model.trend("loss").converged(5, 1e-5) {
            println!("Loss converged, stopping early.");
            break;
        }
    }
    monitor.finish();

    model.save_checkpoint("dnaseq_conv1d.fdl.gz")?;
    println!("Saved trained model to dnaseq_conv1d.fdl.gz");

    Ok(())
}
