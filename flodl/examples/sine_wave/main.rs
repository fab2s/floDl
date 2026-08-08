//! Sine wave regression — the "it really works" moment.
//!
//! Trains a small network to learn sin(x) on [-2pi, 2pi], prints a
//! prediction-vs-actual comparison table, and verifies checkpoint
//! save/load round-trips.
//!
//! Run: `cargo run --example sine_wave`

use flodl::Monitor;
use flodl::*;

fn main() -> Result<()> {
    manual_seed(42);

    // --- Data: sin(x) on [-2pi, 2pi] ---
    let opts = TensorOptions::default();
    let n_samples = 200i64;
    let tau = std::f64::consts::TAU;
    let x_all = Tensor::linspace(-tau, tau, n_samples, opts)?; // [-2pi, 2pi]
    let y_all = x_all.sin()?;

    // Reshape to [n, 1] for the network.
    let x_data = x_all.reshape(&[n_samples, 1])?;
    let y_data = y_all.reshape(&[n_samples, 1])?;

    // Split into batches of 50.
    let x_batches = x_data.batches(50)?;
    let y_batches = y_data.batches(50)?;
    let batches: Vec<_> = x_batches.into_iter().zip(y_batches).collect();

    // --- Model: Linear(1,32) -> GELU -> LayerNorm -> residual -> Linear(32,1) ---
    let model = FlowBuilder::from(Linear::new(1, 32)?)
        .through(GELU)
        .through(LayerNorm::new(32)?)
        .also(Linear::new(32, 32)?) // residual connection
        .through(Linear::new(32, 1)?)
        .build()?;

    let params = model.parameters();
    let mut optimizer = Adam::new(&params, 0.005);
    let scheduler = CosineScheduler::new(0.005, 1e-5, 200);
    model.train();

    // --- Training ---
    let num_epochs = 200usize;
    let mut monitor = Monitor::new(num_epochs);
    // monitor.serve(3000)?;  // uncomment for live dashboard at http://localhost:3000
    // monitor.watch(&model);

    for epoch in 0..num_epochs {
        let t = std::time::Instant::now();

        for (xb, yb) in &batches {
            let input = Variable::new(xb.clone(), true);
            let target = Variable::new(yb.clone(), false);

            optimizer.zero_grad();
            let pred = model.forward(&input)?;
            let loss = mse_loss(&pred, &target)?;
            loss.backward()?;
            clip_grad_norm(&params, 1.0)?;
            optimizer.step()?;

            model.record_scalar("loss", loss.item()?);
        }

        let lr = scheduler.lr(epoch);
        optimizer.set_lr(lr);
        model.record_scalar("lr", lr);
        model.flush(&[]);
        monitor.log(epoch, t.elapsed(), &model);
    }

    monitor.finish();

    // --- Evaluation ---
    model.eval();
    println!(
        "\n{:>8}  {:>10}  {:>10}  {:>8}",
        "x", "actual", "predicted", "error"
    );
    println!("{}", "-".repeat(42));

    // Test on 10 evenly spaced points.
    let test_x = Tensor::linspace(-tau, tau, 10, opts)?;
    let test_y = test_x.sin()?;
    let test_input = test_x.reshape(&[10, 1])?;

    let pred = no_grad(|| {
        let input = Variable::new(test_input.clone(), false);
        model.forward(&input)
    })?;

    let pred_data = pred.data().to_f32_vec()?;
    let actual_data = test_y.to_f32_vec()?;
    let x_data_vec = test_x.to_f32_vec()?;

    let mut max_err: f32 = 0.0;
    for i in 0..10 {
        let err = (pred_data[i] - actual_data[i]).abs();
        if err > max_err {
            max_err = err;
        }
        println!(
            "{:>8.3}  {:>10.4}  {:>10.4}  {:>8.4}",
            x_data_vec[i], actual_data[i], pred_data[i], err
        );
    }
    println!("\nMax error: {:.4}", max_err);

    // --- Checkpoint round-trip ---
    let path = "sine_model.fdl";
    let named = model.named_parameters();
    let named_bufs = model.named_buffers();
    save_checkpoint_file(path, &named, &named_bufs, Some(model.structural_hash()))?;
    println!("Checkpoint saved to {}", path);

    // Rebuild architecture and load weights.
    let model2 = FlowBuilder::from(Linear::new(1, 32)?)
        .through(GELU)
        .through(LayerNorm::new(32)?)
        .also(Linear::new(32, 32)?)
        .through(Linear::new(32, 1)?)
        .build()?;

    let named2 = model2.named_parameters();
    let named_bufs2 = model2.named_buffers();
    load_checkpoint_file(path, &named2, &named_bufs2, Some(model2.structural_hash()))?;
    model2.eval();

    // Verify loaded model produces the same output.
    let pred2 = no_grad(|| {
        let input = Variable::new(test_input.clone(), false);
        model2.forward(&input)
    })?;

    let pred2_data = pred2.data().to_f32_vec()?;
    let mut reload_diff: f32 = 0.0;
    for i in 0..10 {
        let d = (pred_data[i] - pred2_data[i]).abs();
        if d > reload_diff {
            reload_diff = d;
        }
    }
    println!("Checkpoint reload max diff: {:.6}", reload_diff);
    assert!(
        reload_diff < 1e-5,
        "Checkpoint round-trip mismatch: {}",
        reload_diff
    );
    println!("Checkpoint round-trip verified.");

    // Clean up.
    std::fs::remove_file(path).ok();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sine_wave_converges() -> Result<()> {
        let opts = TensorOptions::default();
        let n = 100i64;
        let tau = std::f64::consts::TAU;
        let x = Tensor::linspace(-tau, tau, n, opts)?;
        let y = x.sin()?;
        let x_data = x.reshape(&[n, 1])?;
        let y_data = y.reshape(&[n, 1])?;

        let model = FlowBuilder::from(Linear::new(1, 32)?)
            .through(GELU)
            .through(LayerNorm::new(32)?)
            .also(Linear::new(32, 32)?)
            .through(Linear::new(32, 1)?)
            .build()?;

        let params = model.parameters();
        let mut opt = Adam::new(&params, 0.005);
        model.train();

        let mut last_loss = f64::MAX;
        for _ in 0..150 {
            let input = Variable::new(x_data.clone(), true);
            let target = Variable::new(y_data.clone(), false);

            opt.zero_grad();
            let pred = model.forward(&input)?;
            let loss = mse_loss(&pred, &target)?;
            loss.backward()?;
            clip_grad_norm(&params, 1.0)?;
            opt.step()?;

            last_loss = loss.item()?;
        }

        assert!(
            last_loss < 0.05,
            "sine wave loss should converge below 0.05, got {}",
            last_loss
        );
        Ok(())
    }
}
