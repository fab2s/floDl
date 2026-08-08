//! Quickstart — build, train, and monitor a model with residual connections.
//!
//! Builds a small graph with a residual connection, trains it on random
//! data using Adam, and logs progress with the training monitor.
//!
//! Run: `cargo run --example quickstart`

use flodl::Monitor;
use flodl::*;

fn main() -> Result<()> {
    manual_seed(42);

    // Build the model.
    let model = FlowBuilder::from(Linear::new(2, 16)?)
        .through(GELU)
        .through(LayerNorm::new(16)?)
        .also(Linear::new(16, 16)?)
        .through(Linear::new(16, 2)?)
        .build()?;

    // Set up training.
    let params = model.parameters();
    let mut optimizer = Adam::new(&params, 0.01);
    model.train();

    // Generate some random data (XOR-ish pattern).
    let opts = TensorOptions::default();
    let batches: Vec<(Tensor, Tensor)> = (0..32)
        .map(|_| {
            let x = Tensor::randn(&[16, 2], opts).unwrap();
            let y = Tensor::randn(&[16, 2], opts).unwrap();
            (x, y)
        })
        .collect();

    // Training loop with monitor.
    let num_epochs = 50;
    let mut monitor = Monitor::new(num_epochs);
    // monitor.serve(3000)?;   // uncomment for live dashboard
    // monitor.watch(&model);  // uncomment to show graph SVG in dashboard

    for epoch in 0..num_epochs {
        let t = std::time::Instant::now();

        for (input_t, target_t) in &batches {
            let input = Variable::new(input_t.clone(), true);
            let target = Variable::new(target_t.clone(), false);

            let pred = model.forward(&input)?;
            let loss = mse_loss(&pred, &target)?;

            optimizer.zero_grad();
            loss.backward()?;
            clip_grad_norm(&params, 1.0)?;
            optimizer.step()?;

            model.record_scalar("loss", loss.item()?);
        }

        model.flush(&[]);
        monitor.log(epoch, t.elapsed(), &model);
    }

    monitor.finish();
    Ok(())
}
