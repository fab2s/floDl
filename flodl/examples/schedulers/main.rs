//! LR scheduler composition — warmup + cosine + plateau fallback.
//!
//! Demonstrates composable learning rate scheduling: linear warmup into
//! cosine annealing, with a plateau scheduler as fallback when the
//! primary schedule finishes.
//!
//! Run: `cargo run --example schedulers`

use flodl::*;

fn main() -> Result<()> {
    manual_seed(42);
    let opts = TensorOptions::default();

    // Build a small model.
    let model = FlowBuilder::from(Linear::new(4, 16)?)
        .through(GELU)
        .through(LayerNorm::new(16)?)
        .also(Linear::new(16, 16)?)
        .through(Linear::new(16, 4)?)
        .build()?;

    let params = model.parameters();
    let mut optimizer = Adam::new(&params, 0.001);
    model.train();

    let num_epochs = 100usize;

    // --- Compose schedulers ---

    // Warmup for 10 epochs, then cosine anneal for the remaining 90.
    let cosine = CosineScheduler::new(0.001, 1e-5, num_epochs - 10);
    let scheduler = WarmupScheduler::new(cosine, 0.001, 10);

    // Plateau scheduler as a secondary control: if loss stalls,
    // reduce LR further regardless of the primary schedule.
    let mut plateau = PlateauScheduler::new(0.001, 10, 0.5, 1e-6);

    println!(
        "{:>5}  {:>10}  {:>10}  {:>10}",
        "epoch", "loss", "sched_lr", "eff_lr"
    );
    println!("{}", "-".repeat(40));

    for epoch in 0..num_epochs {
        let x = Tensor::randn(&[64, 4], opts)?;
        let y = Tensor::randn(&[64, 4], opts)?;
        let input = Variable::new(x, true);
        let target = Variable::new(y, false);

        optimizer.zero_grad();
        let pred = model.forward(&input)?;
        let loss = mse_loss(&pred, &target)?;
        loss.backward()?;
        clip_grad_norm(&params, 1.0)?;
        optimizer.step()?;

        let loss_val = loss.item()?;

        // Primary schedule (warmup + cosine).
        let sched_lr = scheduler.lr(epoch);

        // Plateau feedback — takes the minimum of primary and plateau LR.
        let plateau_lr = plateau.observe(loss_val);
        let effective_lr = sched_lr.min(plateau_lr);

        optimizer.set_lr(effective_lr);

        if epoch % 10 == 0 || epoch == num_epochs - 1 {
            println!(
                "{:>5}  {:>10.6}  {:>10.6}  {:>10.6}",
                epoch, loss_val, sched_lr, effective_lr
            );
        }
    }

    println!("\nFinal LR: {:.6}", optimizer.lr());
    Ok(())
}
