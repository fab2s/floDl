//! Linear and logistic regression from first principles.
//!
//! Neither is a dedicated module. In the PyTorch idiom that flodl follows,
//! both are just a `Linear` layer paired with the right loss:
//!
//! * linear regression   = `Linear` + `mse_loss`
//! * logistic regression = `Linear` + `bce_with_logits_loss` (binary)
//!   or `Linear` + `cross_entropy_loss` (multiclass)
//!
//! This example fits both on synthetic data and checks that the learned
//! parameters recover the ground truth.
//!
//! Run: `cargo run --example regression`

use flodl::*;

fn main() -> Result<()> {
    manual_seed(42);
    linear_regression()?;
    println!();
    logistic_regression()?;
    Ok(())
}

/// Fit `y = 2x + 1` with a single `Linear(1, 1)` and mean squared error.
fn linear_regression() -> Result<()> {
    println!("== Linear regression: recovering y = 2x + 1 ==");
    let opts = TensorOptions::default();

    // Data: x in [-1, 1], y = 2x + 1 + small Gaussian noise.
    let (true_w, true_b) = (2.0f32, 1.0f32);
    let n = 128usize;
    let xs: Vec<f32> = (0..n)
        .map(|i| -1.0 + 2.0 * i as f32 / (n as f32 - 1.0))
        .collect();
    let noise = Tensor::randn(&[n as i64, 1], opts)?.to_f32_vec()?;
    let ys: Vec<f32> = xs
        .iter()
        .zip(&noise)
        .map(|(x, e)| true_w * x + true_b + 0.1 * e)
        .collect();
    let x_data = Tensor::from_f32(&xs, &[n as i64, 1], Device::CPU)?;
    let y_data = Tensor::from_f32(&ys, &[n as i64, 1], Device::CPU)?;

    // The model is just one linear layer. That IS the regression.
    let model = Linear::new(1, 1)?;
    let params = model.parameters();
    let mut optimizer = Adam::new(&params, 0.1);
    model.train();

    for _ in 0..200 {
        let input = Variable::new(x_data.clone(), true);
        let target = Variable::new(y_data.clone(), false);

        optimizer.zero_grad();
        let pred = model.forward(&input)?;
        let loss = mse_loss(&pred, &target)?;
        loss.backward()?;
        optimizer.step()?;
    }

    // Recover slope and intercept by probing the fitted line:
    // f(0) = bias, f(1) - f(0) = weight.
    model.eval();
    let probe = |x: f32| -> Result<f32> {
        let t = Tensor::from_f32(&[x], &[1, 1], Device::CPU)?;
        let out = no_grad(|| model.forward(&Variable::new(t.clone(), false)))?;
        Ok(out.data().to_f32_vec()?[0])
    };
    let learned_b = probe(0.0)?;
    let learned_w = probe(1.0)? - learned_b;

    println!("  true:    w = {true_w:.3}, b = {true_b:.3}");
    println!("  learned: w = {learned_w:.3}, b = {learned_b:.3}");
    Ok(())
}

/// Binary logistic regression: classify whether `x0 + x1 > 0` with a single
/// `Linear(2, 1)` and binary cross-entropy on the raw logits.
fn logistic_regression() -> Result<()> {
    println!("== Logistic regression: is x0 + x1 > 0? ==");
    let opts = TensorOptions::default();

    // Data: 2D points, label 1 when their coordinates sum positive.
    let m = 256usize;
    let feats = Tensor::randn(&[m as i64, 2], opts)?;
    let feats_vec = feats.to_f32_vec()?;
    let labels: Vec<f32> = (0..m)
        .map(|i| {
            if feats_vec[2 * i] + feats_vec[2 * i + 1] > 0.0 {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    let y_data = Tensor::from_f32(&labels, &[m as i64, 1], Device::CPU)?;

    // One linear layer producing a single logit per sample.
    let model = Linear::new(2, 1)?;
    let params = model.parameters();
    let mut optimizer = Adam::new(&params, 0.1);
    model.train();

    for _ in 0..200 {
        let input = Variable::new(feats.clone(), true);
        let target = Variable::new(y_data.clone(), false);

        optimizer.zero_grad();
        let logits = model.forward(&input)?;
        // bce_with_logits applies the sigmoid internally, which is
        // numerically safer than a separate sigmoid + bce_loss.
        let loss = bce_with_logits_loss(&logits, &target)?;
        loss.backward()?;
        optimizer.step()?;
    }

    // Inference: logits -> sigmoid -> probability -> threshold at 0.5.
    model.eval();
    let logits = no_grad(|| model.forward(&Variable::new(feats.clone(), false)))?;
    let probs = logits.sigmoid()?.data().to_f32_vec()?;
    let correct = labels
        .iter()
        .zip(&probs)
        .filter(|(label, p)| {
            let predicted = if **p > 0.5 { 1.0 } else { 0.0 };
            (predicted - **label).abs() < 0.5
        })
        .count();

    println!(
        "  accuracy: {:.1}% ({correct}/{m})",
        100.0 * correct as f32 / m as f32
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_regression_recovers_parameters() -> Result<()> {
        manual_seed(0);
        let opts = TensorOptions::default();
        let (true_w, true_b) = (2.0f32, 1.0f32);
        let n = 128usize;
        let xs: Vec<f32> = (0..n)
            .map(|i| -1.0 + 2.0 * i as f32 / (n as f32 - 1.0))
            .collect();
        let noise = Tensor::randn(&[n as i64, 1], opts)?.to_f32_vec()?;
        let ys: Vec<f32> = xs
            .iter()
            .zip(&noise)
            .map(|(x, e)| true_w * x + true_b + 0.1 * e)
            .collect();
        let x_data = Tensor::from_f32(&xs, &[n as i64, 1], Device::CPU)?;
        let y_data = Tensor::from_f32(&ys, &[n as i64, 1], Device::CPU)?;

        let model = Linear::new(1, 1)?;
        let params = model.parameters();
        let mut opt = Adam::new(&params, 0.1);
        model.train();
        for _ in 0..300 {
            let input = Variable::new(x_data.clone(), true);
            let target = Variable::new(y_data.clone(), false);
            opt.zero_grad();
            let pred = model.forward(&input)?;
            let loss = mse_loss(&pred, &target)?;
            loss.backward()?;
            opt.step()?;
        }

        model.eval();
        let probe = |x: f32| -> Result<f32> {
            let t = Tensor::from_f32(&[x], &[1, 1], Device::CPU)?;
            let out = no_grad(|| model.forward(&Variable::new(t.clone(), false)))?;
            Ok(out.data().to_f32_vec()?[0])
        };
        let learned_b = probe(0.0)?;
        let learned_w = probe(1.0)? - learned_b;

        assert!((learned_w - true_w).abs() < 0.1, "weight off: {learned_w}");
        assert!((learned_b - true_b).abs() < 0.1, "bias off: {learned_b}");
        Ok(())
    }

    #[test]
    fn logistic_regression_separates_classes() -> Result<()> {
        manual_seed(0);
        let opts = TensorOptions::default();
        let m = 256usize;
        let feats = Tensor::randn(&[m as i64, 2], opts)?;
        let feats_vec = feats.to_f32_vec()?;
        let labels: Vec<f32> = (0..m)
            .map(|i| {
                if feats_vec[2 * i] + feats_vec[2 * i + 1] > 0.0 {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        let y_data = Tensor::from_f32(&labels, &[m as i64, 1], Device::CPU)?;

        let model = Linear::new(2, 1)?;
        let params = model.parameters();
        let mut opt = Adam::new(&params, 0.1);
        model.train();
        for _ in 0..300 {
            let input = Variable::new(feats.clone(), true);
            let target = Variable::new(y_data.clone(), false);
            opt.zero_grad();
            let logits = model.forward(&input)?;
            let loss = bce_with_logits_loss(&logits, &target)?;
            loss.backward()?;
            opt.step()?;
        }

        model.eval();
        let logits = no_grad(|| model.forward(&Variable::new(feats.clone(), false)))?;
        let probs = logits.sigmoid()?.data().to_f32_vec()?;
        let correct = labels
            .iter()
            .zip(&probs)
            .filter(|(label, p)| {
                let predicted = if **p > 0.5 { 1.0 } else { 0.0 };
                (predicted - **label).abs() < 0.5
            })
            .count();

        assert!(
            correct as f32 / m as f32 > 0.9,
            "accuracy too low: {correct}/{m}"
        );
        Ok(())
    }
}
