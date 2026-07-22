# Inference and Export Roadmap

A sketch of how flodl delivers inference and export without building an
inference server. Eval needs inference. Trained models need to be usable.
Neither requires flodl to become a serving stack.

**Status:** roadmap sketch. No code deliverables in this doc; concrete work
tracks as separate items in the flodl-hf arc.

---

## Why this exists

Two pressures from the user side drive this:

1. **Eval during and after training.** Metrics past raw training loss
   require forward passes, and for generative tasks they require generation
   loops. flodl already runs eval forward passes; it lacks generation.
2. **Downstream use.** A model trained in flodl must be usable in the
   stacks teams already deploy: HuggingFace `transformers`, `vLLM`, `TGI`,
   `ollama`, `llama.cpp`. flodl does not need to be an inference server to
   answer this - it needs to emit formats those stacks already read.

The distinction matters because it keeps flodl's scope bounded. flodl
differentiates on training (DDP, ElChe, real libtorch autograd, parity-
tested HF loaders, task heads). Inference is where trained models leave
flodl's domain; export is the bridge.

---

## Scope

### In

- **Eval harness.** Loss, classification / tagging / QA metrics (already
  partially covered per family), generation-based metrics (perplexity,
  BLEU, ROUGE) once a generation loop exists.
- **External export.** Save formats and file layouts that drop into the HF
  ecosystem without manual glue.
- **In-loop inference for training-adjacent workflows** when concrete use
  cases arrive: large-scale held-out eval, rejection sampling, RL rollouts,
  synthetic data generation. These reuse flodl's existing NCCL + per-GPU
  worker machinery.

### Out

- **Serving infrastructure.** Continuous batching, paged KV-cache,
  OpenAI-compatible APIs, load balancing. vLLM, TGI, and friends do this
  well; flodl cannot win that fight and has no reason to try.
- **Inference-only engineering.** Speculative decoding engines,
  inference-specialised kernel rewrites, quantization as a first-class
  training target. These belong to the serving stacks; flodl integrates
  with them, not around them.

---

## What already works

- `Graph::eval()` / `Graph::train()` toggle module training state (dropout,
  BatchNorm running stats, ...). Forward passes in eval mode work for
  every family currently ported.
- **Encoder task-head inference methods:**
  - `ClassificationHead::classify` / `predict`: softmax + label mapping.
  - `TaggingHead::tag` / `predict`: per-token predictions with
    `sequence_ids` masking.
  - `QaHead::answer` / `answer_batch` / `extract`: best-span decoding over
    `[start, end]` logits.
  - `MaskedLmHead::fill_mask`: top-k predictions at `[MASK]` positions.
- **safetensors save + HF-canonical keys.** `Graph::save_checkpoint`
  writes params and buffers with structural-hash validation;
  `hf_key_from_flodl_key` in `flodl-hf` swaps the `/` separator for `.`
  so the result is a `transformers`-loadable state_dict.
- **Parity tests.** Per-family `*_parity.rs` (gated `#[ignore]` + `_live`)
  load real Hub checkpoints and assert forward parity against HF Python.
  The save/load path round-trips bit-identically on those keys.

## Gap analysis

### Eval

Encoder-task eval is covered: classification, token classification, and
extractive QA each have inference methods plus cross-entropy. What's
missing:

- **Generation-based metrics.** Perplexity, BLEU, ROUGE, text-completion
  evals all require a generation loop (see below).
- **Evaluator abstraction.** No shared `Evaluator` type takes a dataset +
  model + metric and returns a scalar. Every caller rolls their own loop.
  Worth extracting once the second or third duplicate appears.

### Text generation

No generation primitives exist. When the first decoder LLM lands, it needs:

- **KV cache.** Per-layer `past_key_values` with shape
  `[batch, n_heads, past_len, head_dim]`, incremental append per step,
  batch indexing for beam search.
- **Causal mask.** Expressible via the existing
  `build_extended_attention_mask` path, but generation wants a cached
  and shifted version matching the KV cache's growing length.
- **Sampling strategies.** Greedy, temperature, top-k, top-p (nucleus),
  min-p. Mechanically cheap: one `softmax` + one `multinomial` or
  `argmax` per step.
- **Stop-token handling.** Per-sequence termination tracking with early
  exit on batched generation.
- **Beam search** (lower priority): fixed-k beam maintenance with length
  normalization.

### Export

Safetensors save with HF-canonical keys is already HF-compatible for
encoder families. What's still needed:

- **Round-trip parity per family.** A new `*_roundtrip.rs` test variant
  that: builds a fresh head on device, saves to safetensors, shells out
  to Python to load via `transformers.AutoModel.*`, runs a forward pass
  in Python, asserts shape + numerical match against a flodl-side
  forward on the same inputs. This proves end-to-end usability from the
  HF ecosystem, not just key-layout alignment.
- **Export recipe subcommand.** `fdl flodl-hf export` wrapping existing
  Python conversion tools: safetensors -> GGUF via
  `llama.cpp/convert_hf_to_gguf.py`, safetensors -> ONNX via `optimum`,
  safetensors -> HF-directory layout (`config.json`, `tokenizer.json`,
  weight shards colocated). Thin wrappers; flodl does not duplicate
  conversion logic.
- **Native GGUF writer** (deferred). Rust implementation that drops the
  Python dependency for GGUF export. Only worth building if Python-in-loop
  becomes measurable friction.

### In-loop multi-GPU inference

Not required until decoder LLMs and RL / synthetic-data workflows land.
Building blocks already exist (NCCL, `NcclRankComm::split`, per-GPU worker
threads, CUDA streams). What would be new:

- **Tensor parallelism.** Row / column-parallel linear layers with
  all-gather / reduce-scatter in forward.
- **Pipeline parallelism.** Microbatch scheduling across stage-sharded
  copies. Comes after tensor parallel, if at all.
- **KV-cache sharding.** Per-GPU slice with all-gather on attention.

Decisions on this section wait for a concrete use case.

---

## Near-term plan

Committed work, ordered:

1. **DeBERTa-v3 port + Python parity fixture**, as a single slice. Closes
   the NLP encoder family ports.
2. **Round-trip parity tests per family.** One `*_roundtrip.rs` per
   encoder family (BERT, RoBERTa, DistilBERT, XLM-R, ALBERT, DeBERTa-v3).
   Each asserts save -> HF Python load -> forward parity. Validates the
   export promise end-to-end.
3. **`fdl flodl-hf export` recipe subcommand.** Python-wrapper for GGUF
   (via llama.cpp's converter) and HF-directory (config + tokenizer +
   safetensors). Ships the smallest thing that makes a trained flodl
   model usable in the ecosystem with one command.

## Beyond near-term

Sketched, not scheduled:

- Text generation primitives when the first decoder LLM is planned.
- Evaluator abstraction once per-family eval duplication justifies
  extracting it.
- In-loop multi-GPU inference when a concrete RL or synthetic-data
  workflow demands it.
- Native GGUF writer if the Python-wrapper path ever becomes friction.

Update this doc when a beyond-near-term item gets promoted to active.

---

## The shape of the bet

External inference stacks (vLLM, TGI, ollama) already do production
serving well. flodl does not compete there. flodl's value is that users
train in a framework with real multi-GPU training primitives, full HF
ecosystem compatibility, and a training-first design, then hand off
cleanly to whichever inference stack they already deploy. The near-term
plan is the minimum work that makes that handoff trustworthy.

The longer-term opportunity is the reverse direction: using the same
DDP / worker / NCCL machinery for inference-as-a-training-primitive -
RL rollouts, synthetic data, large-scale eval. That is where flodl can
add something candle and vLLM do not, because it lives inside the
training process rather than outside it. Not now, but the primitives
are already in place when it becomes the right move.
