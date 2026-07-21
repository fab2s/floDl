# Tutorial 14: HuggingFace Integration

Load BERT, RoBERTa, DistilBERT, ALBERT, XLM-RoBERTa, or DeBERTa-v2
checkpoints from the HuggingFace Hub in a single line; run sequence
classification, NER, extractive QA, or masked-language modeling; fine-
tune the result on your own data; and round-trip the trained head back
out as an HF-compatible directory that loads into HF Python's
`AutoModelFor*`. All of this lives in the
[`flodl-hf`](https://crates.io/crates/flodl-hf) sibling crate; this
tutorial covers how to use it inside a floDl project.

> **Prerequisites**: [Tensors](01-tensors.md),
> [Modules](03-modules.md), [Graph Builder](05-graph-builder.md), and
> (for the fine-tune walkthrough) [Training](04-training.md) and
> [Multi-GPU DDP](11-multi-gpu.md). Familiarity with HuggingFace
> Transformers helps but is not required.

> **Time**: ~30 minutes.

## Quick start

Inside an existing flodl project (one scaffolded with `fdl init`), `fdl
add flodl-hf` exposes two modes that you can combine.

```bash
fdl add flodl-hf --playground   # try it: drops ./flodl-hf/ sandbox crate
fdl flodl-hf classify           # runs a real fine-tune via AutoModel

fdl add flodl-hf --install      # wire it: appends flodl-hf="=0.5.3" to Cargo.toml
fdl build                       # cargo pulls + compiles the new dep
```

`--playground` drops a standalone cargo crate under `./flodl-hf/` with
its own `Cargo.toml`, a one-file `AutoModel` example, an `fdl.yml` with
runnable commands, and a `flodl-hf:` entry in the root `fdl.yml` so
`fdl flodl-hf <cmd>` routes from project root. The host project's
`Cargo.toml` and `fdl.yml` are untouched.

`--install` appends `flodl-hf = "=0.5.3"` (default features = `hub` +
`tokenizer`) to the root `Cargo.toml` `[dependencies]` and stops there.
Idempotent. Edit the entry by hand to switch flavors (see
[Install](#install) below).

`fdl add flodl-hf` with no flag prompts interactively `[Y/n]`. Non-tty
(CI, piped input) errors loudly with the explicit-flag guidance.

Scaffolding a fresh project with HuggingFace included from day one:

```bash
fdl init my-model --with-hf
cd my-model && fdl flodl-hf classify
```

## Install

If you prefer to wire `flodl-hf` directly into an existing crate, three
feature profiles cover the common deployment shapes.

### Full HuggingFace experience (default)

```toml
flodl-hf = "=0.5.3"
```

Pulls `safetensors` + `hf-hub` + `tokenizers`. Everything needed to
load `bert-base-uncased` out of the box, including text tokenization
and Hub downloads.

### Vision-only (hub, no tokenizer)

For ViT, CLIP vision towers, or any image model that does not need
tokenization. Drops the `tokenizers` crate and its regex + unicode
surface.

```toml
flodl-hf = { version = "=0.5.3", default-features = false, features = ["hub"] }
```

### Offline / minimal (safetensors-only)

For air-gapped environments, embedded training, or pipelines that load
checkpoints from local disk. Drops Hub downloads and tokenizers. No
network, no async runtime, no TLS stack, no regex.

```toml
flodl-hf = { version = "=0.5.3", default-features = false }
```

### Feature matrix

| Feature     | Adds dependency         | Enables                          |
|-------------|-------------------------|----------------------------------|
| `hub`       | `hf-hub` (sync, rustls) | Download models from the Hub     |
| `tokenizer` | `tokenizers`            | Text tokenization for LLMs, BERT |
| `cuda`      | `flodl/cuda`            | GPU-accelerated tensor ops       |

`safetensors` is always included. The HTTP backend is `ureq` +
`rustls-tls`; no tokio, no openssl.

## `AutoModel`: family-agnostic loading

`AutoModel` inspects `config.json`'s `model_type` field and dispatches
to the right architecture without the caller knowing which family the
checkpoint belongs to. This mirrors HuggingFace Python's `AutoModel` /
`AutoModelFor*` entry points.

```rust
use flodl_hf::models::auto::AutoModelForSequenceClassification;

let clf = AutoModelForSequenceClassification::from_pretrained(
    "cardiffnlp/twitter-roberta-base-sentiment-latest",
)?;

let results = clf.predict(&["I love this framework"])?;
for (label, score) in &results[0] {
    println!("{} ({:.3})", label, score);
}
```

The same three-line caller works for `bert-base-uncased`,
`roberta-base`, `distilbert-base-uncased`, `albert-base-v2`,
`xlm-roberta-base`, `microsoft/deberta-v3-base`, or any fine-tune on
top of those. Swap the repo id and the family wiring happens under the
hood.

Five `AutoModel` entry points cover the supported task shapes:

| Entry point                             | Task                        | Output shape                          |
|-----------------------------------------|-----------------------------|---------------------------------------|
| `AutoModel`                             | Backbone (hidden states)    | `[batch, seq_len, hidden]`            |
| `AutoModelForSequenceClassification`    | Whole-text labels           | `Vec<Vec<(String, f32)>>`             |
| `AutoModelForTokenClassification`       | Per-token labels (NER)      | `Vec<Vec<TokenPrediction>>`           |
| `AutoModelForQuestionAnswering`         | Extractive answer span      | `Answer { text, start, end, score }`  |
| `AutoModelForMaskedLM`                  | Fill-mask candidates        | `Vec<(String, f32)>` (top-k by prob)  |

All five enums dispatch across the six supported families and are
marked `#[non_exhaustive]`, so future family additions
(ModernBERT, LLaMA, ViT, ...) do not require an exhaustive-match break
on adopters.

## Per-family entry points

When the family is known upfront, use the concrete type. Same API,
no dispatch layer, and a slightly richer surface (BERT keeps its
pooler; RoBERTa exposes `on_device` for checkpoints that ship one).

### Sequence classification

```rust
use flodl_hf::models::bert::BertForSequenceClassification;

let clf = BertForSequenceClassification::from_pretrained(
    "nateraw/bert-base-uncased-emotion",
)?;
let top = clf.predict(&["I love this framework"])?;
println!("{} ({:.3})", top[0][0].0, top[0][0].1);
```

`predict(&[&str])` returns a per-input `Vec<(String, f32)>` sorted by
probability, with label names parsed from the checkpoint's `id2label`
(or `LABEL_k` as a fallback). Each family ships its native head
structure unchanged from HF reference, so the call site stays identical
across families even though the internals differ.

### Token classification (NER)

```rust
use flodl_hf::models::bert::BertForTokenClassification;

let ner = BertForTokenClassification::from_pretrained("dslim/bert-base-NER")?;
for t in &ner.predict(&["fab2s lives in Latent"])?[0] {
    if t.attends && t.label != "O" {
        println!("{} -> {} ({:.3})", t.token, t.label, t.score);
    }
}
```

`TokenPrediction { token, label, score, attends }` covers each
sub-token. The `attends` flag mirrors the attention mask, so padding
drops cleanly from the result. Works across all six families
(BERT-NER, RoBERTa-NER, DistilBERT-NER, ALBERT-NER, XLM-R multilingual
NER, DeBERTa-v3 medical NER).

### Extractive question answering

```rust
use flodl_hf::models::bert::BertForQuestionAnswering;

let qa = BertForQuestionAnswering::from_pretrained(
    "csarron/bert-base-uncased-squad-v1",
)?;
let a = qa.answer(
    "Where does fab2s live?",
    "fab2s lives in Latent and writes Rust deep learning code.",
)?;
println!("{:?} (score {:.3})", a.text, a.score);
```

The span search is restricted to context tokens via the tokenizer's
`sequence_ids`, so the question region cannot answer itself. Works with
SQuAD-family fine-tunes across every family that ships them
(`csarron/bert-base-uncased-squad-v1`, `deepset/roberta-base-squad2`,
`distilbert/distilbert-base-cased-distilled-squad`,
`twmkn9/albert-base-v2-squad2`,
`deepset/xlm-roberta-large-squad2`,
`deepset/deberta-v3-base-squad2`).

`answer_batch(&[(question, context)])` runs a batch of pairs in one
forward.

### Masked language modeling

```rust
use flodl_hf::models::bert::BertForMaskedLM;

let mlm = BertForMaskedLM::from_pretrained("bert-base-uncased")?;
let candidates = mlm.fill_mask("The capital of France is [MASK].", /*top_k=*/5)?;
for (tok, score) in &candidates {
    println!("{} ({:.3})", tok, score);
}
```

`fill_mask(text, top_k)` resolves the mask token automatically:
`[MASK]` for BERT / DistilBERT / ALBERT, `<mask>` for RoBERTa /
XLM-RoBERTa / DeBERTa-v2. The forward runs once and returns the top-k
vocabulary candidates with probabilities for the masked position.
Per-family head shapes mirror the HF reference exactly (BERT's
transform-then-tied-decoder, RoBERTa's flat tied decoder with bias,
DistilBERT's `vocab_layer_norm + vocab_projector`, ALBERT's
`embedding_size`-factored decoder, DeBERTa-v2's V3 non-legacy layout).

### Embeddings

```rust
use flodl_hf::models::bert::BertModel;

let model = BertModel::from_pretrained("bert-base-uncased")?;
// model is a flodl::Graph; run it with forward_multi
```

BERT returns pooled output (CLS passed through tanh) by default;
RoBERTa and DistilBERT return `last_hidden_state` because their
reference checkpoints either lack a pooler or ship one randomly
initialised. `BertModel::on_device_without_pooler` matches HuggingFace
Python's `add_pooling_layer=False` if you want the hidden state
directly. ALBERT and XLM-RoBERTa follow the same pattern as BERT and
RoBERTa respectively; DeBERTa-v2 uses its own `ContextPooler` (linear
+ tanh on `[CLS]`) when a downstream head is wired on top.

The runnable `*_embed` examples (see below) wire the tokenizer to the
model end-to-end and print per-sentence vectors.

## Tokenizer

`HfTokenizer` is a thin wrapper over the `tokenizers` crate. One
wrapper serves every family: the loaded `tokenizer.json` carries the
model-specific pre-tokenizer and post-processor.

```rust
use flodl_hf::tokenizer::HfTokenizer;

let tok = HfTokenizer::from_pretrained("bert-base-uncased")?;
let batch = tok.encode(&["hello world", "another input"])?;
// batch.input_ids / attention_mask / token_type_ids / position_ids
// are i64 [B, S] Variables; sequence_ids carries the paired-segment
// tag (0 first / 1 second / -1 special).
```

`encode_pairs(&[(q, c)])` produces paired encodings with
`token_type_ids == 1` on the second segment, required for QA and
useful for NLI or sentence-pair classification.

`tok.save("./checkpoint/tokenizer.json")?` persists the wrapped
tokenizer back to disk in the form HF Python's
`AutoTokenizer.from_pretrained` reads. Useful for fine-tune save points
and required by the export round-trip walkthrough below.

Padding defaults to `BatchLongest` with `pad_id = [PAD]` when
`tokenizer.json` has no padding config of its own. There is no default
truncation: oversized texts error loudly at the model rather than
silently truncate. If you need truncation, configure it on the
underlying `Tokenizer` directly before encoding.

Task-head wrappers (`*ForSequenceClassification`, etc.) pull the
tokenizer from the same repo id at `from_pretrained` time, so
`predict(&[&str])` takes raw text without manual tokenization. Direct
`*Model` callers wire the tokenizer themselves.

## Loading from local disk

Every `from_pretrained` variant has a sibling that skips the Hub and
reads a local safetensors file. Useful for air-gapped deploys and for
users on the `default-features = false` profile.

```rust
use flodl_hf::models::bert::{BertConfig, BertModel};
use flodl_hf::safetensors_io::load_safetensors_file_into_graph;

let config = BertConfig::from_json_str(&std::fs::read_to_string("config.json")?)?;
let mut graph = BertModel::build(&config)?;
load_safetensors_file_into_graph(&mut graph, "model.safetensors")?;
```

The loader runs a strict key-set validation before touching any
parameter:

- Missing keys (expected by the graph but absent from the checkpoint)
- Unused keys (present in the checkpoint but not consumed)
- Shape mismatches

A disagreement in any bucket errors with up to 20 entries per bucket
and a `"... and N more"` truncation tail. The graph is either fully
loaded or fully untouched; no silent drift.

The loader preserves the checkpoint's native dtype: a `bf16` or `f16`
checkpoint loads at native precision, where prior releases force-cast
to `f32`. Combined with the dtype-preserving save side
(`save_safetensors_file_from_graph`), this means a non-`f32`
checkpoint round-trips bit-exact through flodl-hf.

Rename-aware variants handle legacy checkpoint conventions (for
example BERT's pre-2020 `LayerNorm.gamma` / `LayerNorm.beta` to
`weight` / `bias`). Allow-unused variants log and skip extra keys
instead of erroring, used under the hood by `*For*::from_pretrained`
when a base-model checkpoint carries task-specific heads flodl-hf does
not consume.

## Fine-tune, save, export, verify

The end-to-end recipe walks a pretrained head through a real
fine-tuning pass and back out as an HF-compatible directory. It runs
in two halves: the in-process Rust example does the fine-tune and
checkpoint dump; two host-side `fdl` commands then re-export the
checkpoint and verify the export round-trips into HF Python.

The full source is at
[`flodl-hf/examples/distilbert_finetune.rs`](https://github.com/flodl-labs/flodl/blob/main/flodl-hf/examples/distilbert_finetune.rs);
this section walks the load-bearing pieces. Run it as:

```bash
fdl flodl-hf example distilbert-finetune
```

It downloads
`distilbert-base-uncased-finetuned-sst-2-english` once (`~250 MB`
cached via `hf-hub`), fine-tunes for 5 steps on a hand-crafted
domain-specific dataset, prints the loss curve and a final eval probe,
saves the trained head as a flodl `.fdl` checkpoint, and prints the
two host-side commands to round-trip it through HF Python. Total
runtime after the cache is warm is about 30 seconds on CPU.

### 1. Load the pretrained head

```rust
use flodl_hf::models::distilbert::DistilBertForSequenceClassification;
use flodl_hf::tokenizer::HfTokenizer;

let model_repo = "distilbert-base-uncased-finetuned-sst-2-english";
// SST-2 ships only legacy vocab.txt; the fast tokenizer.json lives at
// the base repo (vocabulary is identical, SST-2 fine-tuning does not
// retrain it).
let tok_repo   = "distilbert-base-uncased";

let head = DistilBertForSequenceClassification::from_pretrained(model_repo)?;
let tok  = HfTokenizer::from_pretrained(tok_repo)?;
```

### 2. Wire the optimizer

Task heads `impl Module` directly, so on a single device you wire the
optimizer straight onto the head's graph and flip on training mode:

```rust
use flodl::Adam;

head.graph().set_optimizer(|p| Adam::new(p, 5e-5));
head.graph().set_training(true);
```

The training loop body below is byte-identical whether you run on CPU or
a single GPU. To fan the same head across N GPUs, hand it to the
`Trainer` tiers instead - because the head is a `Module`, you drive it
directly:

```rust
use flodl::{Adam, Trainer};

let replica_config = head.config().clone();
let num_labels     = head.labels().len() as i64;
// managed tier - framework owns the loop:
Trainer::builder(
    move |dev| DistilBertForSequenceClassification::on_device(
        &replica_config, num_labels, dev,
    ),
    |p| Adam::new(p, 5e-5),
    |head, batch| head.compute_loss(/* enc, labels from batch */),
)
.dataset(dataset)
.batch_size(16)
.num_epochs(3)
.run()?;
```

Swap `.run()` for `.into_worker()?` to keep the loop body yourself (the
cooperative tier) while the controller owns cadence, partition,
eval-election, and checkpointing. See
[Multi-GPU DDP](11-multi-gpu.md) for the multi-device story and
[`docs/design/trainer-execution-tiers.md`](../design/trainer-execution-tiers.md)
for the three tiers.

### 3. Train

```rust
use flodl::{clip_grad_norm, Device, Tensor, Variable};

let train: &[(&str, i64)] = &[
    ("This framework is a real joy to work with",          1),
    ("I absolutely love the clean API surface",            1),
    // ... 6 more positive / negative pairs ...
];

let params = head.graph().parameters();

for step in 0..5 {
    let texts: Vec<&str> = train.iter().map(|(t, _)| *t).collect();
    let enc = tok.encode(&texts)?;
    let label_ids: Vec<i64> = train.iter().map(|(_, l)| *l).collect();
    let labels = Variable::new(
        Tensor::from_i64(&label_ids, &[train.len() as i64], Device::CPU)?,
        false,
    );

    let loss = head.compute_loss(&enc, &labels)?;
    println!("step {} loss {:.4}", step, loss.item()?);

    loss.backward()?;
    clip_grad_norm(&params, 1.0)?;
    head.graph().step()?;
}
```

`compute_loss(enc, labels)` mirrors HF Python's
`model(..., labels=...).loss` one-call pattern: the head runs its
forward, picks the appropriate task-head loss
(`sequence_classification_loss` over `[batch, num_labels]` logits and
`[batch]` indices in this case), and returns a single `Variable` for
backward. The free functions `sequence_classification_loss`,
`token_classification_loss`, and `question_answering_loss` are
available in `flodl_hf::task_heads` if you need to compose the loss
yourself. `clip_grad_norm` is the standard fine-tune stabilizer.

### 4. Save

```rust
let scratch_dir = "target/distilbert_finetune";
std::fs::create_dir_all(scratch_dir)?;
let ckpt_path      = format!("{scratch_dir}/sst2_finetuned.fdl");
let tokenizer_path = format!("{scratch_dir}/tokenizer.json");

head.graph().save_checkpoint(&ckpt_path)?;
tok.save(&tokenizer_path)?;
```

`save_checkpoint` writes the `.fdl` blob plus an auto-emitted
`<stem>.config.json` sidecar (the graph's `source_config()` carries the
right `architectures: ["DistilBertForSequenceClassification"]` already,
because `from_pretrained` stamped it). The companion `tok.save` call
persists the tokenizer next to the checkpoint so the downstream export
step can pick it up via its auto-tokenizer-copy whitelist; without it,
`fdl flodl-hf export --checkpoint` warns about the missing tokenizer
and `verify-export` cannot run forward parity for lack of an
`AutoTokenizer`.

### 5. Re-export and verify (host-side)

The example exits after printing two `fdl` commands:

```bash
fdl flodl-hf export --checkpoint target/distilbert_finetune/sst2_finetuned.fdl \
                    --out        target/distilbert_finetune/sst2_export
fdl flodl-hf verify-export target/distilbert_finetune/sst2_export --no-hub-source
```

`fdl flodl-hf export --checkpoint <ckpt>` reads the `.fdl` blob and the
sidecar `<stem>.config.json`, rebuilds the head class via
`build_<family>_for_export`, and writes a HF-canonical
`model.safetensors` + `config.json` + `tokenizer.json` under `--out`.

`fdl flodl-hf verify-export <dir> --no-hub-source` loads the export
through HF Python's `AutoModelForSequenceClassification.from_pretrained`
and asserts zero `missing_keys` / `unexpected_keys`. The
`--no-hub-source` flag skips forward parity: a fine-tuned head has no
upstream Hub repo to compare logits against, so the loadability check
is the meaningful gate. A round-tripped Hub checkpoint (no fine-tune)
keeps `flodl_source_repo` in its config and gets full forward parity
automatically.

The recipe is split between in-process and host-side because the
example runs in the `dev` Docker service while `verify-export` runs in
the `hf-parity` service (HF Python + `transformers`); spawning the
verifier from inside the example would require docker-in-docker. Same
pattern as `fdl flodl-hf verify-matrix`.

## Round-trip export

`fdl flodl-hf export` and `fdl flodl-hf verify-export` together form
the HF-ecosystem round-trip gate. The walkthrough above uses the
`--checkpoint` mode for fine-tuned heads; the same tooling re-exports
any Hub checkpoint flodl-hf supports.

### Export

```bash
# Round-trip a Hub checkpoint
fdl flodl-hf export --hub bert-base-uncased --out /tmp/bert-export

# Force a specific head class instead of dispatching on the upstream
# `architectures[0]`. Useful for treating a pretraining checkpoint as a
# feature-extraction encoder:
fdl flodl-hf export --hub bert-base-uncased --head base --out /tmp/bert-base

# Re-export a local fine-tune
fdl flodl-hf export --checkpoint ./my-head.fdl --out /tmp/my-head-export
```

- `--hub <repo>` and `--checkpoint <path>` are mutually exclusive.
- `--head <auto|base|seqcls|tokcls|qa|mlm>` (Hub mode only): force a
  head class. `auto` (default) reads upstream `architectures[0]`;
  `base` re-exports the bare backbone even when the upstream advertises
  a head.
- `--out <dir>` is required. The output is `<out>/model.safetensors` +
  `<out>/config.json`, plus `<out>/tokenizer.json` when the source
  ships a fast tokenizer.
- `--force` overwrites existing files in `<out>`.
- `--preserve-source-config` (checkpoint mode only) also writes the
  loaded source config verbatim to `<out>/config.source.json`, for
  research provenance since `to_json_str` normalises some fields away.

`--hub` also stamps `flodl_source_repo: <repo>` into the exported
`config.json`, so `verify-export` can recover the source automatically
without an explicit `--hub-source` flag.

### Verify-export

```bash
# Auto-detect Hub source from the stamped config.json
fdl flodl-hf verify-export /tmp/bert-export

# Override the Hub source (hand-staged dirs, or comparing against a
# different upstream)
fdl flodl-hf verify-export /tmp/bert-export --hub-source bert-base-uncased

# Skip forward parity (for fine-tuned heads with no upstream)
fdl flodl-hf verify-export /tmp/my-head-export --no-hub-source
```

The verifier reads `<dir>/config.json`, dispatches on `(model_type,
architectures[0])` to the matching HF `AutoModelFor*`, then asserts
loadability (zero `missing_keys` / `unexpected_keys`) and, when a Hub
source is available, bit-exact forward parity on a fixed prompt. Six
per-family `verify-export-{bert,roberta,distilbert,xlm-roberta,albert,deberta-v2}`
commands are thin wrappers that bake in the matching `--hub-source`.

### Verify-matrix

```bash
fdl flodl-hf verify-matrix
fdl flodl-hf verify-matrix -- --families bert,albert --heads base,seqcls
```

`verify-matrix` runs `export` then `verify-export` across the full
30-cell head matrix (six families x `{base, seqcls, tokcls, qa, mlm}`)
and prints a PASS/FAIL grid at the end. Cell list lives in
`flodl-hf/tests/fixtures/head_matrix.json`; adding a cell is a one-line
JSON edit. The runner is fail-soft (a red cell does not abort the run),
heavyweight (`~10+ GB` of Hub weights on a cold cache), and documented
as a quarterly-manual pre-release gate, not a per-PR gate.

## Parity with PyTorch

Every architecture and task head has an `_live` integration test that
asserts bit-exact agreement against the HuggingFace Python reference
(`max_abs_diff <= 1e-5` by default) on a pinned checkpoint. Same
matrix that `verify-matrix` runs end-to-end:

| Family       | base | seqcls | tokcls       | qa  | mlm        |
|--------------|------|--------|--------------|-----|------------|
| BERT         | OK   | OK     | OK           | OK  | OK         |
| RoBERTa      | OK   | OK     | OK           | OK  | OK         |
| DistilBERT   | OK   | OK     | OK           | OK  | OK         |
| ALBERT       | OK   | OK     | OK           | OK  | OK         |
| XLM-RoBERTa  | OK   | OK     | OK (5e-5)    | OK  | OK         |
| DeBERTa-v2   | OK   | OK     | OK           | OK  | gap [^1]   |

`OK` means `max_abs_diff <= 1e-5` against HF Python. Two caveats:

- **XLM-RoBERTa tokencls at 5e-5**: the
  `Davlan/xlm-roberta-large-ner-hrl` checkpoint drifts ~1.7e-5 on the
  fixed prompt. Same flodl encoder runs at 6e-6 on
  `deepset/xlm-roberta-large-squad2` and 4e-6 on
  `roberta-large-ner-english`, so the drift is checkpoint-specific
  weight noise near the f32 precision floor, not depth or sequence
  length. Tolerance for this one cell sits at 5e-5 with a 3x margin
  over the empirical max.

[^1]: **DeBERTa-v2 MLM gap**: the wrapper compiles and runs, but no
public DeBERTa-v2 checkpoint produces meaningful MLM logits.
DeBERTa-v3 trains via Replaced-Token-Detection so the MLM head ships
random-init by design; DeBERTa-v2 xlarge has real MLM weights but uses
`conv_kernel_size=3` (ConvLayer not implemented in flodl-hf), and HF
Python's own `DebertaV2ForMaskedLM` has a key-naming mismatch with
Microsoft's V2 layout. Full investigation in the module-doc of
[`flodl-hf/tests/deberta_v2_parity.rs`](https://github.com/flodl-labs/flodl/blob/main/flodl-hf/tests/deberta_v2_parity.rs).
ConvLayer support would unblock V2 xlarge MLM parity.

Run the parity gates locally:

```bash
fdl test-live
```

This executes `cargo test live -- --nocapture --ignored`, picking up
any test with a `_live` suffix behind `#[ignore]`. The tests need
network access (for Hub downloads) and cache weights under
`./.hf-cache/` via the `HF_HOME` env var.

The parity fixtures themselves are regenerated through
`fdl flodl-hf parity <cell>` (29 per-head commands plus
`fdl flodl-hf parity all` to run the lot in sequence). These run a
Python Docker service (`hf-parity`, `python:3.12-slim` + torch 2.8.0
CPU + `transformers`) to produce reference outputs; flodl-hf then
consumes the resulting safetensors files at test time. Contributors
rerun these when bumping checkpoint shas; end users do not need to.

## Checkpoints with only `pytorch_model.bin`

Some older Hub uploads ship only the unsafe PyTorch pickle format. For
those, run the one-off converter:

```bash
fdl flodl-hf convert <repo_id>
```

This writes a `model.safetensors` into the local Hub cache, after
which `from_pretrained` picks it up automatically.

## Supported families and roadmap

Landed in 0.5.3:

- **Six BERT-family architectures**: BERT, RoBERTa, DistilBERT,
  ALBERT, XLM-RoBERTa, DeBERTa-v2 / DeBERTa-v3.
- **Four task heads per family**: sequence classification, token
  classification, extractive QA, masked language modeling.
  All five `AutoModelFor*` enums dispatch across the six families.
- **Loss wiring on every task head**: `compute_loss(enc, labels)`
  mirrors HF Python's `model(..., labels=...).loss`; free functions in
  `flodl_hf::task_heads` for hand-rolled compositions.
- **Task heads `impl Module`**: drive a head straight through
  `Trainer::run` / `Trainer::builder(...)` for transparent 1-or-N-GPU
  training; same loop code on CPU, single GPU, and multi-GPU.
- **Round-trip export to the HF ecosystem**: `fdl flodl-hf export`
  (Hub or local checkpoint), `verify-export` (auto-detect family +
  head from `architectures[0]`), `verify-matrix` (30-cell pre-release
  gate).
- **Native-dtype safetensors I/O**: `f16` / `bf16` / `f32` / `f64`
  round-trip bit-exact through `flodl-hf::safetensors_io`.
- **`HfTokenizer::save`**: persist a loaded tokenizer back to disk.
- **`fdl add flodl-hf` with `--playground` / `--install` modes**, plus
  the existing `fdl init --with-hf`.

On the roadmap:

- ModernBERT (RoPE, GeGLU, alternating local/global attention)
- LLaMA (RoPE, GQA, SwiGLU)
- LoRA adapters
- ViT
- DeBERTa-v2 ConvLayer (unblocks V2 xlarge MLM parity)

## Runnable examples

Fourteen examples ship with flodl-hf, covering every family x task plus
the `AutoModel` demo and the fine-tune walkthrough:

```bash
fdl flodl-hf example auto-classify                   # any family

fdl flodl-hf example bert-embed                      # also: bert-classify / -ner / -qa
fdl flodl-hf example roberta-embed                   # also: roberta-classify / -ner / -qa
fdl flodl-hf example distilbert-embed                # also: distilbert-classify / -ner / -qa

fdl flodl-hf example distilbert-finetune             # the fine-tune walkthrough above
```

Each example downloads a real fine-tune, runs a small pinned batch,
and prints top labels, entities, or extracted spans. ALBERT,
XLM-RoBERTa, and DeBERTa-v2 are exercised through `auto-classify` and
through the `_live` integration tests; per-family example binaries for
those three are on the next ergonomics pass.

## Further reading

- [`flodl-hf` crate README](https://github.com/flodl-labs/flodl/blob/main/flodl-hf/README.md)
- [flodl-hf examples](https://github.com/flodl-labs/flodl/tree/main/flodl-hf/examples)
- [The floDl CLI](../cli.md) (see `fdl add`, `fdl flodl-hf`, `fdl test-live`)
- [Multi-GPU DDP](11-multi-gpu.md) (the same `Trainer` entry point
  distributes the fine-tune walkthrough across N devices - heads
  `impl Module`)

---

Previous: [Data Loading](13-data-loading.md) |
Next: [DDP Reference](../ddp.md)
