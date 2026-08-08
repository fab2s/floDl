//! DNA-sequence convolutional autoencoder built on flodl.
//!
//! Pipeline:
//!   raw sequence (e.g. "ACGTAC...")
//!     -> one-hot tensor [batch, 4, seq_len]     (A=0, C=1, G=2, T=3)
//!     -> Encoder (Conv1d x2 + pooling -> Linear) -> latent [batch, latent_dim]
//!     -> Decoder (Linear -> ConvTranspose1d x2)  -> logits [batch, 4, seq_len]
//!     -> argmax over the base dimension          -> reconstructed sequence
//!
//! The encoder and decoder are each built with flodl's fluent `FlowBuilder`
//! and are ordinary `Graph`s, so they get `save_checkpoint` / `load_checkpoint`
//! for free. `DnaAutoencoder` is a small wrapper that owns both graphs and
//! exposes `new`, `forward`, `save`, `load`, `predict`, and `embed`.
//!
//! Adapted from an example contributed by Gaurav Sablok (@gsablok),
//! flodl issue #15, updated to the current flodl API.
//!
//! Run: `cargo run --release --example dna_autoencoder`

use flodl::*;
use std::path::Path;

const BASES: [char; 4] = ['A', 'C', 'G', 'T'];

// ---------------------------------------------------------------------
// One-hot encoding / decoding helpers
// ---------------------------------------------------------------------

fn base_to_idx(c: char) -> Result<i64> {
    match c.to_ascii_uppercase() {
        'A' => Ok(0),
        'C' => Ok(1),
        'G' => Ok(2),
        'T' => Ok(3),
        other => Err(TensorError::new(&format!(
            "unsupported base '{other}': only A/C/G/T are supported"
        ))),
    }
}

fn idx_to_base(i: i64) -> char {
    BASES[(i.clamp(0, 3)) as usize]
}

/// Encode a batch of equal-length DNA strings into:
///   - one-hot input tensor  [batch, 4, seq_len]  (Float32)
///   - target index tensor   [batch, seq_len]     (Int64, values 0..3)
fn encode_batch(seqs: &[String], seq_len: usize) -> Result<(Tensor, Tensor)> {
    let batch = seqs.len();
    let mut onehot = vec![0f32; batch * 4 * seq_len];
    let mut targets = vec![0i64; batch * seq_len];

    for (b, seq) in seqs.iter().enumerate() {
        let chars: Vec<char> = seq.chars().collect();
        if chars.len() != seq_len {
            return Err(TensorError::new(&format!(
                "sequence {b} has length {} but expected {seq_len}",
                chars.len()
            )));
        }
        for (pos, c) in chars.iter().enumerate() {
            let idx = base_to_idx(*c)?;
            // NCL layout: onehot[b, idx, pos]
            onehot[b * 4 * seq_len + (idx as usize) * seq_len + pos] = 1.0;
            targets[b * seq_len + pos] = idx;
        }
    }

    let input = Tensor::from_f32(&onehot, &[batch as i64, 4, seq_len as i64], Device::CPU)?;
    let target = Tensor::from_i64(&targets, &[batch as i64, seq_len as i64], Device::CPU)?;
    Ok((input, target))
}

/// Decode a flat [batch * seq_len] index vector back into DNA strings.
fn decode_indices(idx: &[i64], batch: usize, seq_len: usize) -> Vec<String> {
    (0..batch)
        .map(|b| {
            (0..seq_len)
                .map(|p| idx_to_base(idx[b * seq_len + p]))
                .collect::<String>()
        })
        .collect()
}

// ---------------------------------------------------------------------
// Reshape: a tiny zero-parameter Module for use inside FlowBuilder,
// used to go from the decoder's dense layer back to a [N, C, L] tensor.
// ---------------------------------------------------------------------

#[derive(Clone)]
struct Reshape {
    shape: Vec<i64>,
}

impl Reshape {
    fn new(shape: Vec<i64>) -> Self {
        Reshape { shape }
    }
}

impl Module for Reshape {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        input.reshape(&self.shape)
    }
    fn name(&self) -> &str {
        "reshape"
    }
}

// ---------------------------------------------------------------------
// DnaAutoencoder: owns an encoder Graph and a decoder Graph
// ---------------------------------------------------------------------

struct DnaAutoencoder {
    encoder: Graph,
    decoder: Graph,
    seq_len: usize,
    latent_dim: usize,
}

impl DnaAutoencoder {
    /// Build a fresh (randomly initialized) autoencoder.
    /// `seq_len` must be divisible by 4 (two stride-2 pool/deconv stages).
    fn new(seq_len: usize, latent_dim: usize) -> Result<Self> {
        assert!(
            seq_len.is_multiple_of(4),
            "seq_len must be divisible by 4 (got {seq_len})"
        );
        let pooled_len = seq_len / 4;
        let flat_dim = 64 * pooled_len;

        // Encoder: [B, 4, L] -> [B, 32, L/2] -> [B, 64, L/4] -> [B, latent_dim]
        let encoder = FlowBuilder::from(Conv1d::configure(4, 32, 5).with_padding(2).done()?)
            .through(ReLU)
            .through(MaxPool1d::new(2))
            .through(Conv1d::configure(32, 64, 5).with_padding(2).done()?)
            .through(ReLU)
            .through(MaxPool1d::new(2))
            .through(Flatten::new(1, -1))
            .through(Linear::new(flat_dim as i64, latent_dim as i64)?)
            .tag("latent")
            .build()?;

        // Decoder: [B, latent_dim] -> [B, 64*L/4] -> [B, 64, L/4]
        //          -> [B, 32, L/2] -> [B, 4, L] (logits, one channel per base)
        let decoder = FlowBuilder::from(Linear::new(latent_dim as i64, flat_dim as i64)?)
            .through(ReLU)
            .through(Reshape::new(vec![-1, 64, pooled_len as i64]))
            .through(ConvTranspose1d::configure(64, 32, 2).with_stride(2).done()?)
            .through(ReLU)
            .through(ConvTranspose1d::configure(32, 4, 2).with_stride(2).done()?)
            .tag("logits")
            .build()?;

        Ok(DnaAutoencoder {
            encoder,
            decoder,
            seq_len,
            latent_dim,
        })
    }

    fn encode(&self, x: &Variable) -> Result<Variable> {
        self.encoder.forward(x)
    }

    fn decode(&self, z: &Variable) -> Result<Variable> {
        self.decoder.forward(z)
    }

    /// Full forward pass: one-hot input -> reconstruction logits.
    fn forward(&self, x: &Variable) -> Result<Variable> {
        let z = self.encode(x)?;
        self.decode(&z)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut p = self.encoder.parameters();
        p.extend(self.decoder.parameters());
        p
    }

    fn set_training(&self, training: bool) {
        self.encoder.set_training(training);
        self.decoder.set_training(training);
    }

    fn end_step(&self) {
        self.encoder.end_step();
        self.decoder.end_step();
    }

    // -------------------------------------------------------------
    // save / load
    // -------------------------------------------------------------

    /// Save encoder and decoder checkpoints into `dir` as
    /// `encoder.fdl` and `decoder.fdl`.
    fn save<P: AsRef<Path>>(&self, dir: P) -> Result<()> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)
            .map_err(|e| TensorError::new(&format!("failed to create {dir:?}: {e}")))?;
        self.encoder
            .save_checkpoint(dir.join("encoder.fdl").to_string_lossy().as_ref())?;
        self.decoder
            .save_checkpoint(dir.join("decoder.fdl").to_string_lossy().as_ref())?;
        Ok(())
    }

    /// Load encoder and decoder checkpoints from `dir`. The architecture
    /// (seq_len / latent_dim) of `self` must match the one used to save.
    fn load<P: AsRef<Path>>(&self, dir: P) -> Result<()> {
        let dir = dir.as_ref();
        let enc_report = self
            .encoder
            .load_checkpoint(dir.join("encoder.fdl").to_string_lossy().as_ref())?;
        let dec_report = self
            .decoder
            .load_checkpoint(dir.join("decoder.fdl").to_string_lossy().as_ref())?;
        flodl::msg!(
            "loaded encoder: {} params, decoder: {} params",
            enc_report.loaded.len(),
            dec_report.loaded.len()
        );
        Ok(())
    }

    // -------------------------------------------------------------
    // predict
    // -------------------------------------------------------------

    /// Reconstruct a batch of equal-length DNA sequences.
    /// Runs in eval mode under `no_grad`: no gradients, no dropout/batchnorm
    /// training behavior.
    fn predict(&self, seqs: &[String]) -> Result<Vec<String>> {
        self.set_training(false);
        let (input_t, _target_t) = encode_batch(seqs, self.seq_len)?;
        let batch = seqs.len();

        let logits = no_grad(|| {
            let input = Variable::new(input_t.clone(), false);
            self.forward(&input)
        })?;

        // logits: [batch, 4, seq_len] -> argmax over the base dimension (dim 1)
        let idx_tensor = logits.data().argmax(1, false)?; // [batch, seq_len]
        let idx: Vec<i64> = idx_tensor.to_i64_vec()?;
        Ok(decode_indices(&idx, batch, self.seq_len))
    }

    /// Encode a batch of sequences straight to their latent vectors
    /// (no decoding). Useful for downstream embedding-based tasks.
    fn embed(&self, seqs: &[String]) -> Result<Vec<Vec<f32>>> {
        self.set_training(false);
        let (input_t, _) = encode_batch(seqs, self.seq_len)?;
        let batch = seqs.len();

        let z = no_grad(|| {
            let input = Variable::new(input_t.clone(), false);
            self.encode(&input)
        })?;

        let flat = z.data().to_f32_vec()?;
        Ok(flat
            .chunks(self.latent_dim)
            .map(|c| c.to_vec())
            .take(batch)
            .collect())
    }
}

// ---------------------------------------------------------------------
// Loss: per-position cross-entropy over the 4 bases
// ---------------------------------------------------------------------

fn reconstruction_loss(logits: &Variable, target_idx: &Variable) -> Result<Variable> {
    // logits: [B, 4, L] -> [B, L, 4] -> [B*L, 4]
    let (b, c, l) = {
        let shape = logits.data().shape();
        (shape[0], shape[1], shape[2])
    };
    let logits_flat = logits.transpose(1, 2)?.reshape(&[b * l, c])?;
    let target_flat = target_idx.reshape(&[b * l])?;
    cross_entropy_loss(&logits_flat, &target_flat)
}

// ---------------------------------------------------------------------
// Synthetic dataset (swap this out for real sequences, e.g. loaded from FASTA)
// ---------------------------------------------------------------------

fn make_synthetic_dataset(n: usize, seq_len: usize, rng: &mut Rng) -> Vec<String> {
    // A handful of repeated motifs mixed with random bases so the
    // autoencoder has structure worth learning (pure noise is also fine,
    // it just won't compress as well).
    let motifs = ["TATA", "GATTACA", "ACGT", "CCGCGC"];
    (0..n)
        .map(|_| {
            let mut s = String::with_capacity(seq_len);
            while s.len() < seq_len {
                if rng.bernoulli(0.3) {
                    let m = motifs[rng.usize(motifs.len())];
                    s.push_str(m);
                } else {
                    s.push(BASES[rng.usize(4)]);
                }
            }
            s.truncate(seq_len);
            s
        })
        .collect()
}

fn per_base_accuracy(originals: &[String], reconstructed: &[String]) -> f64 {
    let mut correct = 0usize;
    let mut total = 0usize;
    for (o, r) in originals.iter().zip(reconstructed.iter()) {
        for (a, b) in o.chars().zip(r.chars()) {
            total += 1;
            if a == b {
                correct += 1;
            }
        }
    }
    correct as f64 / total as f64
}

// ---------------------------------------------------------------------
// main: build -> train -> save -> load into a fresh model -> predict
// ---------------------------------------------------------------------

fn main() -> Result<()> {
    manual_seed(42);
    let mut rng = Rng::seed(42);

    let seq_len = 64;
    let latent_dim = 32;
    let checkpoint_dir = "checkpoints/dna_ae";

    // ---- Build ----
    let model = DnaAutoencoder::new(seq_len, latent_dim)?;
    if flodl::gpu_available() {
        model.encoder.move_to_device(Device::CUDA(0));
        model.decoder.move_to_device(Device::CUDA(0));
    }

    // ---- Data ----
    let train_seqs = make_synthetic_dataset(2000, seq_len, &mut rng);
    let val_seqs = make_synthetic_dataset(64, seq_len, &mut rng);
    let batch_size = 32;

    // ---- Train ----
    let params = model.parameters();
    let mut optimizer = Adam::new(&params, 1e-3);
    let num_epochs = 20;

    model.set_training(true);
    for epoch in 0..num_epochs {
        let mut indices: Vec<usize> = (0..train_seqs.len()).collect();
        rng.shuffle(&mut indices);

        let mut epoch_loss = 0.0f64;
        let mut num_batches = 0usize;

        for chunk in indices.chunks(batch_size) {
            let batch_seqs: Vec<String> = chunk.iter().map(|&i| train_seqs[i].clone()).collect();
            let (input_t, target_t) = encode_batch(&batch_seqs, seq_len)?;

            let input = Variable::new(input_t, false);
            let target = Variable::new(target_t, false);

            let logits = model.forward(&input)?;
            let loss = reconstruction_loss(&logits, &target)?;

            optimizer.zero_grad();
            loss.backward()?;
            clip_grad_norm(&params, 1.0)?;
            optimizer.step()?;
            model.end_step();

            epoch_loss += loss.item()?;
            num_batches += 1;
        }

        if (epoch + 1) % 5 == 0 || epoch == 0 {
            let recon = model.predict(&val_seqs)?;
            let acc = per_base_accuracy(&val_seqs, &recon);
            model.set_training(true); // predict() switched to eval mode
            println!(
                "epoch {:3}  loss={:.4}  val_per_base_acc={:.3}",
                epoch + 1,
                epoch_loss / num_batches as f64,
                acc
            );
        }

        if (epoch + 1) % 10 == 0 {
            model.save(checkpoint_dir)?;
            println!("  saved checkpoint to {checkpoint_dir}");
        }
    }

    // ---- Final save ----
    model.save(checkpoint_dir)?;
    println!("final model saved to {checkpoint_dir}");

    // ---- Load into a *fresh* model instance to prove round-tripping works ----
    let loaded_model = DnaAutoencoder::new(seq_len, latent_dim)?;
    loaded_model.load(checkpoint_dir)?;

    // ---- Predict with the reloaded model ----
    let sample = &val_seqs[0..5];
    let reconstructed = loaded_model.predict(sample)?;

    println!("\nSample reconstructions (loaded model):");
    for (orig, recon) in sample.iter().zip(reconstructed.iter()) {
        let acc = per_base_accuracy(std::slice::from_ref(orig), std::slice::from_ref(recon));
        println!("  orig: {orig}");
        println!("  rec : {recon}   (acc={acc:.2})\n");
    }

    let embeddings = loaded_model.embed(sample)?;
    println!(
        "latent dim: {}, first embedding (truncated): {:?}",
        latent_dim,
        &embeddings[0][..8.min(embeddings[0].len())]
    );

    Ok(())
}
