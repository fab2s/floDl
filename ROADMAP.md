# flodl Roadmap

flodl is developed one thing at a time. Whatever has `[started]` is the
current focus; everything else is a candidate for next. The pull
happens when the current item ships, with the priority chosen against
what's known *that day*, not what was written when the line was added.

For the historical record of shipped phases and individual changes, see
[CHANGELOG.md](CHANGELOG.md). For the project philosophy, see
[the trajectory thesis](https://flodl.dev/thesis).

---

## Shipped

- **Phase 1..11** complete: tensor + autograd + nn + graph + data + multi-GPU DDP
- **ResNet-20 200e** validated across all DDP modes on a heterogeneous
  RTX 5060 Ti + GTX 1060 box. NCCL cadence reaches 0.9242 in 44m10s,
  beating the solo-1 baseline (0.918 in 61m42s) on both axes.
- **AccumulateGrad stream fix**: three undocumented libtorch facts
  resolved, no more "stream does not match" warning on DDP backward.
- **NCCL init-on-main + split**: dead worker recovery without CUDA
  context corruption.
- **ElChe cadence battle-tested** across six architecture families
  (logistic, MLP, LeNet, char-RNN, GPT-nano, conv autoencoder) plus
  ResNet-20 with deep residuals.
- **fdl maturity pass**: zero-dep `FdlArgs` derive (flodl-cli +
  flodl-cli-macros), `--fdl-schema` contract, ddp-bench and flodl-cli
  migrated, multi-env overlay (`fdl.local.yml`, `fdl.ci.yml`, …) with
  `@env` selector routing, deep-merge, conflict detection, and
  `fdl config show`. See
  [docs/design/run-config.md](docs/design/run-config.md).
- **HuggingFace integration** sibling crate `flodl-hf`: six BERT-family
  architectures (BERT, RoBERTa, DistilBERT, ALBERT, XLM-RoBERTa,
  DeBERTa-v2 / v3) with four task heads each (sequence / token
  classification, extractive QA, masked language modeling),
  `AutoModel` dispatch from `config.json`'s `model_type`,
  `HfTokenizer` with `save` round-trip, strict-load safetensors with
  key-set validation and native-dtype preservation, `from_pretrained`
  Hub integration, PyTorch parity tests at `max_abs_diff <= 1e-5` on
  29 of 30 head cells (DeBERTa-v2 MLM gap documented in
  `flodl-hf/tests/deberta_v2_parity.rs`). Task heads `impl Module` +
  `HasGraph` make fine-tuning transparent across CPU / single GPU /
  multi-GPU through the cooperative (`Trainer::builder(...).into_worker()`)
  and managed (`.run()`) tiers, and `compute_loss(enc, labels)` mirrors
  HF Python's one-call loss shape. Round-trip back to the HF ecosystem with
  `fdl flodl-hf export` (Hub or local `.fdl` checkpoint) and
  `fdl flodl-hf verify-export` (auto-detect family/head, loadability
  + bit-exact forward parity). `fdl add flodl-hf` `--playground` /
  `--install` modes plus `fdl init --with-hf` close the discovery gap.

See [CHANGELOG.md](CHANGELOG.md) for the full per-version detail.

---

## In progress

*(empty)*

(Length-1 by design. The next item pulls from Possibilities when work
on it begins.)

---

## Possibilities

Unordered. Any line can become the next In Progress. Adding a line is
not a commitment; only moving one to In Progress is.

- **Phase 12 Cloud DDP**: `OuterOptimizer` trait, DiLoCo baseline,
  hierarchical ElChe (intra-host NCCL + inter-host DiLoCo). Real
  feature PR with design doc and cloud test budget. See
  [docs/design/cloud-ddp.md](docs/design/cloud-ddp.md).
- **Async stale-update correction (HeLoCo-style)**: direction-aware
  pseudo-gradient correction for fully-async and wide-window regimes,
  where worker drift exceeds ElChe's bounded-staleness envelope.
  Per-tensor-block cosine alignment against the outer-optimizer
  momentum (preserve aligned blocks, shrink conflicting components,
  reorient weakly-aligned ones at fixed norm) plus a momentum
  look-ahead initialization, layered on the DiLoCo `NesterovMomentum`
  outer optimizer already shipped. The complement to ElChe's
  cadence/window control: ElChe keeps drift small so utilization can
  reclaim barrier idle; this repairs drift when a deployment chooses
  to let it grow (geo-distributed, non-IID per-node shards). After
  HeLoCo (Asif et al., arXiv 2606.00271). See
  [docs/design/cloud-ddp.md](docs/design/cloud-ddp.md).
- **ddp-bench: next published model**. Class-level pull, repeats as
  models land. Top candidates: small transformer (GPT-2 tiny / BERT-
  small) for the attention family, ViT, UNet for multi-scale skip
  connections, MoE as a routing pressure test, PPO for the RL data
  loop.
- **Zero-dispatch training**: full train-step CUDA Graph capture
  (forward + backward + optimizer as one replay) plus double-buffered
  static I/O tensors. Resident and streaming DataLoader modes already
  ship; this is the Graph integration layer on top. See
  [docs/design/resident-training.md](docs/design/resident-training.md).
- **Graph serialization**: save/load graph topology, ONNX import/export.
- **Model parallelism**: tensor / pipeline parallelism for models that
  exceed single-GPU VRAM.
- **Higher-order gradients**: differentiate through backward.
- **2:4 semi-structured sparsity**: FFI through to
  `at::sparse_semi_structured`, sparse training and inference on
  Ampere+ Sparse Tensor Cores. Covers both the LTH-style
  "train dense, prune, retrain sparse" path and the recent
  train-from-scratch with periodic mask updates.
- **flodl-hf next**: ModernBERT (RoPE, GeGLU, alternating local/global
  attention), LLaMA (RoPE, GQA, SwiGLU, then the architecture), LoRA
  adapters, ViT. Then a flagship ElChe-driven fine-tuning benchmark on
  heterogeneous consumer GPUs to validate the transparent fine-tune
  plumbing end-to-end (task heads `impl Module` + `compute_loss`, driven
  through the cooperative / managed Trainer tiers; see Shipped). See
  [docs/design/cloud-ddp.md](docs/design/cloud-ddp.md) for the
  downstream ElChe tie-in.
- **flodl-manager CLI evolution**: keep maturing `fdl` toward a true
  DL package manager on top of cargo. Remaining slices: flodl-aware
  feature selection on `fdl add` (`fdl add hf --for bert|vit|offline`),
  argv forwarding on `fdl build` / `fdl clippy` (matching the `--`
  separator + `append:` pattern that shipped on `run:`-kind commands
  in 0.5.3), and `model-info` / `doctor` commands. Earlier slices
  already in Shipped: `fdl add flodl-hf` + `fdl init --with-hf`, the
  `--playground` / `--install` mode split, `run:`-kind argv
  forwarding, Docker-aware schema probing, and bare-project help
  fallthrough.
- **JEPA exploration**: two-tower EMA target-encoder infrastructure,
  latent predictive training, via BYOL as the stepping stone. The
  infra (EMA updates, stop-gradient composition, latent probes) is
  reusable for I-JEPA / V-JEPA and any FBRL-native objective that
  follows.

---

## Considered and set aside

(Ideas that were on the table and consciously deferred. Not blocked,
just not currently a priority. Easy to reactivate if context shifts.)

- *(empty for now)*

---

## How this list moves

- A Possibility moves to `In Progress` only when `In Progress` is empty.
  The decision is made fresh each time, with whatever is true that day.
- Shipped items get a one-line entry here when notable; routine work is
  in CHANGELOG only.
- Ideas that turn out to be the wrong priority move to `Considered and
  set aside` rather than being silently deleted. They can move back if
  the context changes.
- No dates. Ordinal commitments only. Slippage shows up as "took
  longer than expected" rather than "broken promise."
