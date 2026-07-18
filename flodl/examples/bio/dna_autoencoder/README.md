# DNA Sequence Convolutional Autoencoder

Compresses DNA sequences to a latent vector and reconstructs them: one-hot input
`[B, 4, L]` -> `Conv1d` encoder -> latent `[B, latent_dim]` -> `ConvTranspose1d`
decoder -> per-base logits, trained with per-position cross-entropy. The encoder
and decoder are separate `Graph`s, so each gets checkpointing for free.

Adapted from an example contributed by Gaurav Sablok ([@gsablok](https://github.com/gsablok)), issue #15.

```sh
cargo run --release --example dna_autoencoder
```

## What it covers

- Two `FlowBuilder` `Graph`s (encoder / decoder) plus a custom zero-parameter `Reshape` `Module`
- `Conv1d` + `MaxPool1d` encoder, `ConvTranspose1d` decoder
- Per-position `cross_entropy_loss` over the 4 bases (`transpose` + `reshape`)
- `Adam` + `clip_grad_norm`; `no_grad` inference; per-base accuracy
- `save_checkpoint` / `load_checkpoint` round-trip into a fresh model, then `embed`
