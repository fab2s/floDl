# Linear and Logistic Regression

Neither is a dedicated module. In the PyTorch idiom that flodl follows, both
are just a `Linear` layer paired with the right loss:

- linear regression = `Linear` + `mse_loss`
- logistic regression (binary) = `Linear` + `bce_with_logits_loss`
- logistic regression (multiclass) = `Linear` + `cross_entropy_loss`

This example fits both on synthetic data and checks that the learned
parameters recover the ground truth (`y = 2x + 1`, then a linear decision
boundary).

```sh
cargo run --example regression
```

## What it covers

- `Linear` used standalone as a complete model
- `mse_loss` for regression, `bce_with_logits_loss` for binary classification
- `Adam` optimizer with a plain training loop
- `no_grad` + `sigmoid` for evaluation and probability thresholding
