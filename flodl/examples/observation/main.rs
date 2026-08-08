//! Observation and early stopping — collect, flush, trend, stop.
//!
//! Demonstrates the full observation workflow: tag graph nodes, collect
//! per-batch metrics, flush to epoch history, query trends for early
//! stopping decisions, and export training curves.
//!
//! Run: `cargo run --example observation`

use flodl::Monitor;
use flodl::*;

fn main() -> Result<()> {
    manual_seed(42);
    let opts = TensorOptions::default();

    // Build a model with tagged intermediate nodes.
    let model = FlowBuilder::from(Linear::new(4, 16)?)
        .through(GELU)
        .tag("hidden")
        .through(LayerNorm::new(16)?)
        .also(Linear::new(16, 16)?)
        .tag("residual")
        .through(Linear::new(16, 4)?)
        .build()?;

    let params = model.parameters();
    let mut optimizer = Adam::new(&params, 0.01);
    let scheduler = CosineScheduler::new(0.01, 1e-4, 200);
    model.train();

    let num_epochs = 200usize;
    let mut monitor = Monitor::new(num_epochs);

    // Generate stable data so trends are meaningful.
    let x_data = Tensor::randn(&[128, 4], opts)?;
    let y_data = Tensor::randn(&[128, 4], opts)?;

    // Split into batches.
    let x_batches = x_data.batches(32)?;
    let y_batches = y_data.batches(32)?;
    let batches: Vec<_> = x_batches.into_iter().zip(y_batches).collect();

    println!(
        "{:>5}  {:>10}  {:>10}  {:>8}  {:>10}",
        "epoch", "loss", "slope", "stalled", "status"
    );
    println!("{}", "-".repeat(50));

    for epoch in 0..num_epochs {
        let t = std::time::Instant::now();

        for (xb, yb) in &batches {
            let input = Variable::new(xb.clone(), true);
            let target = Variable::new(yb.clone(), false);

            optimizer.zero_grad();
            let pred = model.forward(&input)?;

            // Collect tagged node outputs as metrics.
            model.collect(&["hidden", "residual"])?;

            let loss = mse_loss(&pred, &target)?;
            loss.backward()?;
            clip_grad_norm(&params, 1.0)?;
            optimizer.step()?;

            model.record_scalar("loss", loss.item()?);
        }

        // Flush batch metrics to epoch history.
        let lr = scheduler.lr(epoch);
        optimizer.set_lr(lr);
        model.record_scalar("lr", lr);
        model.flush(&["hidden", "residual", "loss", "lr"]);
        monitor.log(epoch, t.elapsed(), &model);

        // Query trends for early stopping (window=10 epochs).
        let loss_trend = model.trend("loss");
        let stalled = loss_trend.stalled(10, 1e-5);
        let converged = loss_trend.converged(10, 1e-5);

        if epoch % 20 == 0 || stalled || converged {
            let status = if converged {
                "CONVERGED"
            } else if stalled {
                "stalled"
            } else if loss_trend.improving(10) {
                "improving"
            } else {
                ""
            };

            println!(
                "{:>5}  {:>10.6}  {:>10.6}  {:>8}  {:>10}",
                epoch,
                loss_trend.latest(),
                loss_trend.slope(10),
                stalled,
                status
            );
        }

        // Early stop when converged.
        if converged && epoch > 20 {
            println!("\nEarly stop at epoch {} — loss converged.", epoch);
            break;
        }
    }

    monitor.finish();

    // Group trend queries (window=10 for all).
    let group = model.trends(&["hidden", "residual", "loss"]);
    println!("\nFinal trend summary:");
    println!("  All improving: {}", group.all_improving(10));
    println!("  Any stalled:   {}", group.any_stalled(10, 1e-5));
    println!("  Mean slope:    {:.6}", group.mean_slope(10));

    Ok(())
}
