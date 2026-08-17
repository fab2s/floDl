# Working in the flodl Source Checkout

The flodl repo's own `fdl.yml` ships the concrete command set used to
develop floDl itself. These are examples of the manifest system from the
previous section, not built-in commands.

## Development loop

```bash
fdl check              # type-check without building
fdl build              # debug build
fdl fmt                # format the workspace (rustfmt)
fdl fmt-check          # formatting check, byte-for-byte what CI runs
fdl clippy             # style gate: rustfmt check first, then clippy (tests + workspace + ddp-bench)
fdl test               # all CPU tests
fdl test-release       # tests in release mode
fdl test-live          # tests needing network / external resources (see below)
fdl doc                # rustdoc, strict (-D warnings)
fdl ci                 # the whole CPU job: fmt + build + test + clippy + doc
```

`fdl ci` is the closest local equivalent to what CI does to a PR; run it
before pushing. `fdl test` alone catches neither rustdoc nor formatting
warnings.

## Coverage

```bash
fdl coverage           # CPU only -- a FLOOR, it scores GPU code as missed
fdl coverage-all       # every suite this box can run, and names the ones it can't
```

A number to look at, never a threshold to clear. `coverage-all` prints a
`RAN` / `SKIPPED` / `FAILED` roster rather than one percentage, because a
hardware- or tool-gated suite returns early and reports `ok`: a single
number can be built entirely out of suites that executed nothing.

## Live tests

`fdl test-live` runs integration tests that depend on network access or
external resources (Hugging Face Hub downloads, cached safetensors
checkpoints, etc.). The canonical pattern:

- Test name ends in `_live`.
- Test is annotated `#[ignore = "live: requires network"]` (or similar
  reason) so `fdl test` skips it by default.
- `fdl test-live` delegates to `cargo test live` with
  `-- --nocapture --ignored` declared as `append:`, which picks them up.
  Pass cargo flags after `--` to scope (e.g.
  `fdl test-live -- -p flodl-hf --test xlm_roberta_parity`).

flodl-hf uses this for its PyTorch parity tests
(`bert_parity_vs_pytorch_live`, `bert_tokenizer_matches_parity_fixture_live`,
and the RoBERTa / DistilBERT / ALBERT / XLM-RoBERTa siblings), each
asserting `max_abs_diff <= 1e-5` on logits or hidden state against a
pinned HF Python reference. Weights cache under `.hf-cache/` via
`HF_HOME=/workspace/.hf-cache` in the Docker service.

Any project (not just flodl itself) can adopt the `_live` suffix +
`#[ignore]` convention; `fdl test-live` picks up any test matching
the pattern within its `cargo test` scope.

## GPU testing

```bash
fdl gpu-build            # build with the active variant's GPU feature
fdl gpu-clippy           # lint with the active variant's GPU feature
fdl gpu-test             # parallel GPU tests (excludes NCCL / Graph)
fdl gpu-test-nccl        # NCCL/DDP tests only (isolated processes)
fdl gpu-test-graph       # CUDA Graph tests (exclusive GPU, single-threaded)
fdl gpu-test-serial      # remaining serial tests
fdl gpu-test-all         # full suite: parallel + NCCL isolated + serial
```

## Benchmarks

`bench` is a `path:`-kind sub-command rooted at `./benchmarks/`. Presets
are defined in `benchmarks/fdl.yml`; options come from
`benchmarks/run.sh --fdl-schema` and are auto-cached on first use.

```bash
fdl bench                              # quick single-round run (CUDA)
fdl bench publish                      # publication run (10 interleaved rounds, 15s warmup)
fdl bench cpu                          # CPU-only quick run
fdl bench cpu-publish                  # CPU-only publication run

fdl bench --rounds 20 --output ...     # ad-hoc flags (listed by `fdl bench -h`)
```

## DDP validation suite

`ddp-bench/` is a `path:`-kind sub-command with its own `fdl.yml` and
preset commands. Example presets (from `ddp-bench/fdl.yml`):

```bash
fdl ddp-bench quick                   # fast smoke test (1 epoch, linear model)
fdl ddp-bench validate                # full DDP validation matrix
fdl ddp-bench validate --report out   # validation + write report to out/
fdl ddp-bench --help                  # list all presets + options
```

## HuggingFace (flodl-hf)

`flodl-hf/` is another `path:`-kind sub-command with its own
`fdl.yml`, enabled through the convention entry `flodl-hf:` in the
root manifest. Same shape as `ddp-bench/` and `benchmarks/`: the root
declares the sub-command, the child `fdl.yml` defines its tasks.

```bash
fdl flodl-hf                          # list sub-commands
fdl flodl-hf convert <repo_id>        # convert pytorch_model.bin -> model.safetensors

# Runnable examples (fourteen demos across the six BERT-family architectures)
fdl flodl-hf example                  # list example names
fdl flodl-hf example auto-classify    # family-agnostic via AutoModel
fdl flodl-hf example bert-embed       # + bert-classify / bert-ner / bert-qa
fdl flodl-hf example roberta-embed    # + roberta-classify / -ner / -qa
fdl flodl-hf example distilbert-embed # + distilbert-classify / -ner / -qa
fdl flodl-hf example distilbert-finetune  # fine-tune walkthrough (loss curve + export recipe)

# Round-trip export to the HF ecosystem (any supported family/head)
fdl flodl-hf export --hub google-bert/bert-base-uncased --out /tmp/bert-export
fdl flodl-hf export --checkpoint ./my.fdl  --out /tmp/my-export
fdl flodl-hf verify-export /tmp/bert-export             # auto-detects Hub source from stamped config
fdl flodl-hf verify-export /tmp/my-export --no-hub-source

# 30-cell pre-release gate (six families x base/seqcls/tokcls/qa/mlm)
fdl flodl-hf verify-matrix
fdl flodl-hf verify-matrix -- --families bert,albert --heads base,seqcls

# Parity-fixture regeneration (contributors; 29 per-head commands plus `parity all`)
fdl flodl-hf parity                       # list parity targets
fdl flodl-hf parity all                   # run every fixture in sequence (PASS/FAIL grid)
fdl flodl-hf parity bert                  # google-bert/bert-base-uncased backbone fixture
fdl flodl-hf parity bert-seqcls           # per-head fixtures
fdl flodl-hf parity albert-mlm            # ALBERT family masked-LM fixture
fdl flodl-hf parity deberta-v2-qa         # DeBERTa-v2 QA fixture
# (29 in total: bert/roberta/distilbert/albert/xlm-roberta + seqcls/tokencls/qa/mlm
#  per family, plus the bare-backbone targets; deberta-v2 has no -mlm fixture
#  due to a documented MLM gap in flodl-hf/tests/deberta_v2_parity.rs)
```

`hub`, `checkpoint`, and `parity` modes all run in a dedicated
`hf-parity` Docker service (`python:3.12-slim` + torch CPU wheel +
`transformers`) declared in `docker-compose.yml`.
`HF_HOME=/workspace/.hf-cache` keeps weights and tokenizers cached
between runs (gitignored). The `verify-export` and `verify-matrix`
runners route Python through the same service automatically.

See the
[HuggingFace Integration tutorial](../tutorials/14-flodl-hf.md) for
end-user usage of the crate itself (API walkthroughs, install
profiles, `AutoModel` dispatch, fine-tune + export round-trip
recipe, the 30-cell parity matrix).

## Interactive shells

```bash
fdl shell         # dev container (CPU)
fdl gpu-shell    # GPU container (cuda or rocm service)
```

## Re-building the CLI

After editing `flodl-cli/`:

```bash
fdl self-build    # rebuild fdl and replace the installed binary
```

This uses the currently-running `fdl` to rebuild itself, and swaps the
new binary into place atomically.

---

## Architecture notes

The CLI is built as a pure Rust binary with **zero external crate
dependencies** beyond serde. GPU detection uses `nvidia-smi`, downloads
use `curl`/`wget`, and zip extraction uses `unzip` (or PowerShell on
Windows). This means:

- **~750KB binary** - trivially distributable.
- **Compiles in under 1 second** - no C++ compilation, no libtorch
  linking.
- **Cross-platform** - Linux x86_64/aarch64, macOS arm64, Windows
  x86_64.
- **No runtime dependencies** - works on any machine; GPU features
  degrade gracefully when `nvidia-smi` is absent.

Pre-compiled binaries are published to GitHub Releases on every tagged
release. The `fdl` shell script is a thin bootstrap that downloads the
right binary, falling back to `cargo build` if no binary is available
for your platform.

<!-- nav: generated by site/build_guide.py — do not edit below -->

---

Previous: [The fdl.yml Manifest](05-manifest.md) | Next: [Distributed Architecture](../distributed/architecture.md)
