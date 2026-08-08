//! Transfer learning — freeze a pretrained encoder and train a new head.
//!
//! Demonstrates checkpoint save/load, partial loading into a different
//! architecture, parameter freezing, and optimizer parameter groups with
//! per-group learning rates.
//!
//! Run: `cargo run --example transfer_learning`

use flodl::*;

fn main() -> Result<()> {
    manual_seed(42);
    let opts = TensorOptions::default();

    // --- Phase 1: Train an encoder on a proxy task ---
    println!("=== Phase 1: Train encoder ===");

    let encoder = FlowBuilder::from(Linear::new(4, 16)?)
        .through(GELU)
        .through(LayerNorm::new(16)?)
        .tag("encoded")
        .through(Linear::new(16, 4)?)
        .build()?;

    let params = encoder.parameters();
    let mut opt = Adam::new(&params, 0.01);
    encoder.train();

    for epoch in 0..50 {
        let x = Tensor::randn(&[32, 4], opts)?;
        let y = Tensor::randn(&[32, 4], opts)?;
        let input = Variable::new(x, true);
        let target = Variable::new(y, false);

        opt.zero_grad();
        let pred = encoder.forward(&input)?;
        let loss = mse_loss(&pred, &target)?;
        loss.backward()?;
        clip_grad_norm(&params, 1.0)?;
        opt.step()?;

        if epoch % 10 == 0 {
            println!("  epoch {:>3}: loss={:.4}", epoch, loss.item()?);
        }
    }

    // Save the pretrained encoder.
    let ckpt_path = "pretrained_encoder.fdl";
    let named = encoder.named_parameters();
    let named_bufs = encoder.named_buffers();
    save_checkpoint_file(ckpt_path, &named, &named_bufs, None)?;
    println!("Encoder saved to {}", ckpt_path);

    // --- Phase 2: Build a new model and load encoder weights ---
    println!("\n=== Phase 2: Transfer to new architecture ===");

    // New architecture reuses the encoder layers but adds a different head.
    let model = FlowBuilder::from(Linear::new(4, 16)?)
        .through(GELU)
        .through(LayerNorm::new(16)?)
        .tag("encoded")
        .also(Linear::new(16, 16)?) // new: residual connection
        .through(Linear::new(16, 2)?) // new: different output dim
        .build()?;

    // Partial load: matching names get loaded, new layers keep random init.
    let named2 = model.named_parameters();
    let named_bufs2 = model.named_buffers();
    let report = load_checkpoint_file(ckpt_path, &named2, &named_bufs2, None)?;

    println!(
        "Loaded {} parameters, skipped {}, missing {}",
        report.loaded.len(),
        report.skipped.len(),
        report.missing.len()
    );

    // Freeze the encoder layers (first 3 modules: Linear, GELU, LayerNorm).
    let all_params = model.parameters();
    for (i, p) in all_params.iter().enumerate() {
        if i < 3 {
            p.freeze()?;
        }
    }
    println!("Encoder layers frozen");

    // Set up optimizer with parameter groups:
    // - Frozen params are excluded automatically (zero grad).
    // - New head gets higher LR.
    let trainable: Vec<Parameter> = all_params
        .iter()
        .filter(|p| !p.is_frozen())
        .cloned()
        .collect();

    let mut opt2 = Adam::new(&trainable, 0.005);
    model.train();

    // --- Phase 3: Fine-tune the new head ---
    println!("\n=== Phase 3: Fine-tune new head ===");

    for epoch in 0..50 {
        let x = Tensor::randn(&[32, 4], opts)?;
        let y = Tensor::randn(&[32, 2], opts)?;
        let input = Variable::new(x, true);
        let target = Variable::new(y, false);

        opt2.zero_grad();
        let pred = model.forward(&input)?;
        let loss = mse_loss(&pred, &target)?;
        loss.backward()?;
        clip_grad_norm(&trainable, 1.0)?;
        opt2.step()?;

        if epoch % 10 == 0 {
            println!("  epoch {:>3}: loss={:.4}", epoch, loss.item()?);
        }
    }

    // --- Phase 4: Unfreeze and fine-tune everything ---
    println!("\n=== Phase 4: Full fine-tune (unfrozen) ===");

    for p in &all_params {
        p.unfreeze()?;
    }

    let mut opt3 = Adam::new(&all_params, 0.001); // lower LR for full model
    for epoch in 0..30 {
        let x = Tensor::randn(&[32, 4], opts)?;
        let y = Tensor::randn(&[32, 2], opts)?;
        let input = Variable::new(x, true);
        let target = Variable::new(y, false);

        opt3.zero_grad();
        let pred = model.forward(&input)?;
        let loss = mse_loss(&pred, &target)?;
        loss.backward()?;
        clip_grad_norm(&all_params, 1.0)?;
        opt3.step()?;

        if epoch % 10 == 0 {
            println!("  epoch {:>3}: loss={:.4}", epoch, loss.item()?);
        }
    }

    println!("\nTransfer learning complete.");

    // Clean up.
    std::fs::remove_file(ckpt_path).ok();
    Ok(())
}
