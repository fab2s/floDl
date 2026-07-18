# DNA Sequence Conv1D Classifier

A synthetic ChIP-seq-style task: detect whether a consensus motif is present in
a DNA sequence. One-hot encodes DNA into `[4, L]` tensors, plants a motif in
half the samples, and trains a `Conv1d` stack to classify motif presence, a
toy for peak-annotation / genome-annotation work. To train on real data, replace
`make_example` (which fabricates a sequence and a 0/1 label) with a loader that
yields your own `(sequence, label)` pairs (e.g. sequences from a FASTA and
labels from your peak annotations), then feed them through `one_hot_encode` and
`build_batches` unchanged.

Adapted from an example contributed by Gaurav Sablok ([@gsablok](https://github.com/gsablok)), issue #14.

```sh
cargo run --release --example dnaseq_conv1d
```

## What it covers

- `FlowBuilder` with `Conv1d` / `ReLU` / `MaxPool1d` / `Flatten` / `Linear` / `Dropout` / `Sigmoid`
- One-hot DNA encoding into `[channels, length]` tensors
- `Adam` + `bce_loss` + `clip_grad_norm` training loop
- `record_scalar` / `flush` with `Monitor::log`, early stop on `trend().converged()`
- `save_checkpoint`
