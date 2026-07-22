---
title: "HuggingFace, both ways"
subtitle: "v0.5.3: bit-exact round-trip export across 30 head cells (BERT, RoBERTa, DistilBERT, ALBERT, XLM-RoBERTa, DeBERTa-v2/v3 with all four task heads), masked-LM heads everywhere, and a universal Trainer for transparent fine-tuning on CPU, single GPU, or heterogeneous multi-GPU"
date: 2026-04-28
description: "flodl 0.5.3 closes the round-trip on HuggingFace integration. Every supported family and task head exports back to the HF ecosystem with bit-exact verification across 30 cells (max abs diff = 0). Three more families join the roster: ALBERT (factorised embeddings + cross-layer sharing), XLM-RoBERTa (multilingual SentencePiece, structurally a RoBERTa sibling), and DeBERTa-v2/v3 (disentangled attention, mask-gated embeddings). Masked-LM heads land on all six families. The new universal Trainer takes a step closure and runs identically on CPU, single GPU, or heterogeneous multi-GPU."
---

[v0.5.2 made HuggingFace checkpoints load in Rust]({% post_url 2026-04-22-huggingface %}).
v0.5.3 closes the loop: every supported family-and-head combination
now round-trips back to the HF ecosystem too, with bit-exact
verification at the safetensors layer. Six families, four task heads
each, 30 cells of parity matrix, all PASS, all `max abs diff = 0`.
Plus a universal `Trainer` that fine-tunes transparently across CPU,
single GPU, and heterogeneous multi-GPU with the same code.

## Round-trip: bit-exact, both directions

`fdl flodl-hf export --hub <repo> --out <dir>` re-emits any flodl-hf
checkpoint as an HF-canonical staged directory: `model.safetensors` +
`config.json` + `tokenizer.json`, layout-identical to what HF Python's
`AutoModel.from_pretrained` reads natively. The companion command
`fdl flodl-hf verify-export <dir>` loads the staged dir back into HF
Python's `AutoModelFor*`, runs a fixed prompt through both
implementations, and asserts bit-exact agreement on every head output
(plus zero `missing_keys` / `unexpected_keys` on load).

```bash
fdl flodl-hf export --hub bert-base-uncased --out staged/
fdl flodl-hf verify-export staged/
# verify-export /workspace/staged
#   model_type=bert architecture=BertModel -> AutoModel
#   hub source: bert-base-uncased
#   outputs to compare: ('last_hidden_state',)
# PASS loadability (AutoModel <- /workspace/staged)
# PASS forward last_hidden_state (1, 8, 768) bit-exact (max abs diff = 0)
```

`fdl flodl-hf verify-matrix` is the heavyweight gate that runs export
then verify-export across the full **6 families x 5 head shapes**
matrix. The 0.5.3 release ships 30/30 PASS:

```
             base      seqcls    tokcls    qa        mlm
bert         PASS      PASS      PASS      PASS      PASS
roberta      PASS      PASS      PASS      PASS      PASS
distilbert   PASS      PASS      PASS      PASS      PASS
xlm-roberta  PASS      PASS      PASS      PASS      PASS
albert       PASS      PASS      PASS      PASS      PASS
deberta-v2   PASS      PASS      PASS      PASS      PASS

all 30 cells passed
```

Every cell verified bit-exact (`max abs diff = 0`) against HF Python
on a pinned reference checkpoint. Twelve of the cells were already
covered by parity tests in 0.5.2 (BERT, RoBERTa, DistilBERT × backbone
+ three task heads); the remaining eighteen close the round-trip gap
in this release.

The export path also supports `--checkpoint <path>` for re-emitting
local `.fdl` checkpoints (the format flodl uses internally for
training checkpoints), so a fine-tune run that ends with
`graph.save_checkpoint("my-finetune.fdl")?` becomes an
HF-shareable directory in one CLI command:

```bash
fdl flodl-hf export --checkpoint my-finetune.fdl --out my-finetune-hf/
fdl flodl-hf verify-export my-finetune-hf/
```

The `architectures` field in the exported `config.json` is
normalised against what was actually built, so HF Python's
`AutoModelFor*` dispatch on the staged dir reaches for the right
class (this was the subtle bug `verify-matrix` flagged on its first
run; the fix landed alongside the gate).

## Three more families: ALBERT, XLM-RoBERTa, DeBERTa-v2/v3

The roster doubles from three to six. The shared `TransformerLayer`
scaffold from 0.5.2 absorbs the new families without code
duplication; each addition carries only its architecture-specific
deltas.

**ALBERT** (`albert-base-v1`, `albert-base-v2`, fine-tunes on top):
factorised embeddings (token / position / type embeddings live in a
smaller `embedding_size` space, lifted into `hidden_size` via a
single projection) plus cross-layer parameter sharing (one
transformer block re-applied N times). The dedicated
`AlbertMLMHeadTransform` feeds back to vocabulary through the tied
embedding decoder. Both v1 and v2 ship `gelu_new` (the tanh
approximation), which is why `GELU::tanh()` joined the lib in
0.5.3; picking the wrong GELU form silently produces
~1e-2 max-abs diff, so this is load-bearing.

**XLM-RoBERTa** (`xlm-roberta-base` and the multilingual fine-tunes
built on top): structurally identical to RoBERTa. SentencePiece over
~250k multilingual tokens (vs RoBERTa's 50k BPE) is handled
transparently by `HfTokenizer::from_pretrained`. The `model_type:
"xlm-roberta"` signal stays typed through `AutoConfig`, so the
multilingual axis is explicit at the dispatch layer rather than
hidden under a RoBERTa alias.

**DeBERTa-v2 / DeBERTa-v3** (`microsoft/deberta-v3-{xsmall, small,
base, large}` and the SQuAD / NLI fine-tunes on top): three
load-bearing departures from BERT.

1. **Disentangled self-attention**: each layer computes
   content-to-content + content-to-position + position-to-content
   scores, scaled by `sqrt(head_dim * 3)`. Dedicated
   `deberta_transformer_layer` module; the math is fundamentally
   different from BERT's so it is not folded into the shared
   `TransformerLayer`.
2. **No absolute positional embedding**: position information lives
   in a `rel_embeddings` table threaded into every layer as a
   disentangled bias.
3. **Mask-gated embeddings**: post-LayerNorm, the embedding output is
   multiplied element-wise by the padding mask, zeroing pad positions
   before they enter the encoder.

`AutoConfig` and `AutoModelFor*` dispatch enums grew matching variants
for the three new families and gained `#[non_exhaustive]` so the next
family addition (ModernBERT, LLaMA, ViT, ...) is BC-clean by attribute.

## Masked language modeling, everywhere

A fourth task shape joins the roster: every family ships a
`*ForMaskedLM` wrapper that consumes raw text and returns top-k
fill-mask candidates with probabilities, mirroring HF Python's
`pipeline("fill-mask")`.

```rust
use flodl_hf::models::auto::AutoModelForMaskedLM;

let mlm = AutoModelForMaskedLM::from_pretrained("xlm-roberta-base")?;
let predictions = mlm.fill_mask("Paris is the capital of <mask>.", 5)?;
for (token, prob) in &predictions {
    println!("{} ({:.3})", token, prob);
}
// France (0.857)
// Spain (0.024)
// Germany (0.013)
// Italy (0.011)
// Belgium (0.009)
```

Five of six family MLM cells are bit-exact against HF Python on
pinned checkpoints (BERT, RoBERTa, DistilBERT, XLM-RoBERTa, ALBERT).
The DeBERTa-v2 MLM parity gap is fully scoped in
[`flodl-hf/tests/deberta_v2_parity.rs`](https://github.com/flodl-labs/flodl/blob/main/flodl-hf/tests/deberta_v2_parity.rs):
V3 RTD checkpoints have no real MLM weights (the MLM head is random-
init by design), and V2 xlarge MLM checkpoints use
`conv_kernel_size=3` which this port does not implement yet. The
backbone, sequence-classification, token-classification, and
question-answering cells for DeBERTa-v2 are all bit-exact.

## The universal Trainer

`Trainer` is the new default training entry. It works on any Module
(Graph or otherwise), CPU, single GPU, or heterogeneous multi-GPU,
with the same code. You describe one training step as a closure
(forward + loss); the framework runs the loop, the backward pass,
the optimizer step, and the gradient sync.

```rust
// Step closure: takes the replica's model and one batch, returns the
// loss Variable. The framework calls backward + optimizer step + sync.
fn train_step(model: &dyn Module, batch: &[Tensor]) -> Result<Variable> {
    let input = Variable::new(batch[0].clone(), false);
    let target = Variable::new(batch[1].to_dtype(DType::Int64)?, false);
    cross_entropy_loss(&model.forward(&input)?, &target)
}

let handle = Trainer::builder(
    |dev| build_model_on(dev),
    |params| Adam::new(params, 0.001),
    train_step,
)
    .dataset(dataset)
    .batch_size(32)
    .num_epochs(10)
    .run()?;

let state = handle.join()?;  // averaged params + buffers, ready for inference
```

Task-head wrappers like `flodl-hf`'s `BertForSequenceClassification`
`impl Module` directly, so they ride the exact same `Trainer` entry -
hand the head factory to `Trainer::builder(...)` and it distributes
the wrapper instead of the bare graph:

```rust
let head = DistilBertForSequenceClassification::from_pretrained(repo)?;
let config = head.config().clone();
let num_labels = head.labels().len() as i64;

Trainer::builder(
    move |dev| DistilBertForSequenceClassification::on_device(&config, num_labels, dev),
    |p| Adam::new(p, 5e-5),
    |head, batch| head.compute_loss(/* enc, labels from batch */),
)
.dataset(dataset)
.batch_size(16)
.num_epochs(3)
.run()?;
```

Same call shape on 1 or N GPUs. On heterogeneous multi-GPU, El Che cadence
auto-tunes how often the slow card synchronises with the fast one
(the original arc this whole project pulls from). The `flodl-hf`
crate ships a complete fine-tune walkthrough as
`fdl flodl-hf example distilbert-finetune`: SST-2 polarity on
`distilbert-base-uncased-finetuned-sst-2-english`, ten labelled
examples, five training steps, loss curve and probe printed in about
30 seconds after the one-time weight fetch.

## What flodl gained underneath

A few flodl-internal pieces had to land for this release to work
cleanly. They are not flodl-hf-specific; the broader framework
benefits.

- **Native-dtype safetensors round-trip** plus `Tensor::to_blob`:
  f16, bf16, and f64 checkpoints now round-trip bit-exact through
  `flodl-hf`'s safetensors I/O without going through f32. The
  `Tensor::to_blob` primitive on `flodl::Tensor` lets the save side
  write any libtorch dtype straight into the safetensors payload.
- **`GeluApprox` enum on `nn::GELU`**: BC-clean parametrisation of
  the GELU activation. `GELU::exact()` is the erf form (PyTorch's
  default, HF `hidden_act = "gelu"`); `GELU::tanh()` is the
  approximation required by ALBERT, GPT-2, and `hidden_act` in
  `{gelu_new, gelu_pytorch_tanh}`. The bare-name usage `.through(GELU)`
  keeps compiling: types and constants live in separate namespaces
  in Rust, so `pub const GELU: GELU = GELU::exact();` re-exports the
  default-constructed value under the type name.
- **`HfTokenizer::save`**: persist a loaded tokenizer back to disk in
  the form HF Python's `AutoTokenizer.from_pretrained` reads back.
  Required for the export round-trip; useful as a standalone
  checkpointing primitive at fine-tune save points.
- **`fdl run` argv forwarding**: `fdl test -- -p flodl-hf --test foo`
  splices the user args through to the underlying `cargo test`
  command. Same ergonomics for any `run:`-kind command in `fdl.yml`.
  Loud error on stray args (no silent drop).
- **`fdl add --playground` / `--install` mode split**: try-it-out
  (sandbox under `./flodl-hf/`) versus wire-it-in (root `Cargo.toml`
  dependency). Combinable. Idempotent.

## What's next

Six families, twenty-four heads, four task shapes, all round-trip
verified, fine-tunable across heterogeneous GPUs with one call. The
HF ecosystem is now bidirectional from Rust.

Next on the
[roadmap](https://github.com/flodl-labs/flodl/blob/main/ROADMAP.md):

- **ModernBERT** (RoPE, GeGLU, alternating local/global attention)
- **LLaMA** (RoPE, GQA, SwiGLU)
- **LoRA adapters** for parameter-efficient fine-tuning
- **ViT** for the vision branch
- A flagship **El Che fine-tune benchmark** on heterogeneous consumer
  GPUs, validating the task-head `Trainer` path end-to-end (heads
  `impl Module`; the plumbing landed in 0.5.3, the demo is the next
  thing on this axis)

## Where to go next

- **Read the tutorial**:
  [HuggingFace Integration](/guide/flodl-hf) walks through every
  family, both directions of the round-trip, and the fine-tune
  walkthrough.
- **Read the training tutorial**:
  [Training](/guide/training) covers the three Trainer tiers
  (`Trainer::run` / `Trainer::builder().run()` framework-managed,
  `Trainer::builder().into_worker()` cooperative, `Ddp::wrap` bypass)
  with the same code on CPU, single GPU, and multi-GPU.
- **Skim the crate README**:
  [flodl-hf on GitHub](https://github.com/flodl-labs/flodl/tree/main/flodl-hf)
  has the install matrix and a code block per task shape.
- **Read the changelog**: the
  [0.5.3 entry](https://github.com/flodl-labs/flodl/blob/main/CHANGELOG.md)
  is the per-feature record.
- **Try it**: `fdl init my-model --with-hf` drops a working
  AutoModel + AutoModelForSequenceClassification scaffold in under
  thirty seconds (assuming libtorch is already provisioned).

Phase one was loading. Phase two was fine-tuning. The fine-tune
plumbing now spans CPU, single GPU, and heterogeneous multi-GPU with
the same code; the next milestone is the demo on the hardware people
actually own.
