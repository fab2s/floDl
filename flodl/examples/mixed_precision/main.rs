//! Mixed precision training — float16 forward, float32 gradients.
//!
//! Demonstrates `cast_parameters` + `GradScaler` workflow for training
//! with reduced precision. The scaler dynamically adjusts loss scaling
//! to prevent gradient underflow in float16.
//!
//! Run: `cargo run --example mixed_precision`

use flodl::*;
use flodl::monitor::Monitor;

fn main() -> Result<()> {
    manual_seed(42);
    let opts = TensorOptions::default();

    // Random regression data.
    let batches: Vec<(Tensor, Tensor)> = (0..4)
        .map(|_| {
            let x = Tensor::randn(&[50, 8], opts).unwrap();
            let y = Tensor::randn(&[50, 4], opts).unwrap();
            (x, y)
        })
        .collect();

    // Build the model (starts in float32).
    let model = FlowBuilder::from(Linear::new(8, 32)?)
        .through(GELU)
        .through(LayerNorm::new(32)?)
        .also(Linear::new(32, 32)?)
        .through(Linear::new(32, 4)?)
        .build()?;

    let params = model.parameters();

    // Cast parameters to float16 for reduced memory and faster matmuls.
    cast_parameters(&params, DType::Float16)?;
    println!("Parameters cast to float16");

    let mut optimizer = Adam::new(&params, 0.001);
    let mut scaler = GradScaler::new();
    model.train();

    let num_epochs = 100usize;
    let mut monitor = Monitor::new(num_epochs);

    for epoch in 0..num_epochs {
        let t = std::time::Instant::now();
        let mut steps_taken = 0u32;

        for (xb, yb) in &batches {
            // Cast inputs to match parameter dtype.
            let input = Variable::new(xb.to_dtype(DType::Float16)?, true);
            let target = Variable::new(yb.to_dtype(DType::Float16)?, false);

            optimizer.zero_grad();
            let pred = model.forward(&input)?;
            let loss = mse_loss(&pred, &target)?;

            // Scale loss before backward to prevent gradient underflow.
            let scaled_loss = scaler.scale(&loss)?;
            scaled_loss.backward()?;

            // Unscale, check for inf/nan, step if clean.
            let stepped = scaler.step(&params, &mut || optimizer.step())?;
            scaler.update();

            if stepped {
                steps_taken += 1;
            }

            model.record_scalar("loss", loss.item()?);
        }

        model.record_scalar("scale", scaler.scale_factor());
        model.flush(&[]);
        monitor.log(epoch, t.elapsed(), &model);

        if steps_taken == 0 {
            println!("epoch {}: all steps skipped (inf grads), scale={:.0}", epoch, scaler.scale_factor());
        }
    }

    monitor.finish();

    // Cast back to float32 for inference or checkpointing.
    cast_parameters(&params, DType::Float32)?;
    println!("Parameters cast back to float32 for export");

    Ok(())
}
