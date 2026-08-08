# Contributing to floDl

Thank you for your interest in floDl. Contributions are welcome and appreciated.

## Getting Started

floDl builds against libtorch via FFI, so all development happens inside
Docker. Everything is driven by the `fdl` CLI:

```bash
git clone https://github.com/flodl-labs/flodl.git
cd flodl
./fdl setup     # detect hardware, download libtorch, build dev container
./fdl shell     # interactive shell inside the container
./fdl test      # run all tests (CPU)
./fdl clippy    # lint (includes test code)
```

You do **not** need Rust or libtorch installed on the host machine.
`fdl --help` lists every command.

### Committed templates, local copies

Two files follow the same convention: the `*.example` is committed, your
working copy is gitignored, and you make it by copying.

| Committed | Your copy | Holds |
|-----------|-----------|-------|
| `fdl.yml.example` | `fdl.yml` | the command manifest `fdl` reads (`fdl` copies it for you on first run) |
| `.env.example` | `.env` | per-machine `docker-compose` values |

Nothing in `.env` is required on a stock single-user Linux box, so copy it only
when you need one of two things. Your own `UID`/`GID`, so files the container
writes into the bind-mounted workspace belong to you rather than root — compose
defaults to `1000:1000`, correct on most Linux boxes and wrong on macOS, where
it is `501:20`. Or `CARGO_BUILD_JOBS`, which Apple Silicon hosts need; see
[docs/mac-apple-silicon.md](docs/mac-apple-silicon.md).

```bash
cp .env.example .env    # then uncomment what applies
```

## Development Workflow

1. Fork the repository and create your branch from `main`.
2. Make your changes inside the dev container (`fdl shell`).
3. Run `fdl fmt` to format (CI checks this, so it fails the build if skipped).
4. Run `fdl test` to verify all tests pass.
5. Run `fdl clippy` to ensure zero warnings.
6. Open a pull request.

`fdl ci` runs all of the above in one go and is the closest local
equivalent to what CI will do to your PR.

## Code Style

- Standard Rust conventions: `rustfmt`, zero clippy warnings.
- Formatting is enforced (`cargo fmt --all --check` in CI). `fdl fmt`
  writes, `fdl fmt-check` is the exact check CI runs.
- The bulk reformat that adopted rustfmt is listed in
  `.git-blame-ignore-revs`. Run this once so `git blame` keeps pointing at
  real authors rather than at that commit:
  ```bash
  git config blame.ignoreRevsFile .git-blame-ignore-revs
  ```
- Keep the API consistent with existing patterns.
- Every fallible operation returns `Result<T>` — use `?` for propagation.
- Every differentiable operation needs a backward function and a numerical
  gradient check in the autograd tests.
- Public types and methods should have `///` doc comments.

## What We're Looking For

**High value contributions:**
- New NN modules (with forward, backward, parameter collection, and gradient checks)
- New autograd operations (with backward and numerical verification)
- Performance improvements to the FFI dispatch path
- Bug fixes with reproducing tests
- **Backend support**: Apple MPS, Intel XPU. If you have hardware we don't,
  this is a great way to contribute. See the
  [architecture section](README.md#architecture) for context.

  AMD ROCm already ships — `--features rocm`, with the vendor derived from
  the active libtorch variant. It is also the best guide to what a new
  backend actually costs, and it contradicts the obvious assumption: ROCm
  added **no `Device` variant at all**. `Device` is still `{ CPU, CUDA(u8) }`,
  because ROCm masquerades as CUDA all the way down to libtorch's
  dispatcher. The work went into `flodl-sys/gpu_compat.h` instead — read its
  header comment before starting.

  MPS and XPU are different: they are genuinely distinct device types, so
  those *will* need a `Device` variant plus FFI shim and resource-monitoring
  work. Don't assume the ROCm shape transfers.

**Also welcome:**
- Documentation improvements and examples
- Doc tests for public APIs
- CI improvements

**Please discuss first:**
- Changes to public API signatures
- New dependencies — and note that **`Cargo.lock` is deliberately not
  committed**. A library's consumers resolve their own tree, and letting
  security patches reach whoever compiles is worth more than a reproducible
  release build. Please don't add one. The trade-off is accepted, not
  overlooked: the supply-chain gate can go red on a morning nobody touched
  the code, and the MSRV floor tracks the ecosystem for the same reason.
  See the comments in `deny.toml` and on `rust-version` in `Cargo.toml`.
- Architecture changes

Open an issue to discuss before investing significant effort on these.

## Testing

Every PR should pass the existing test suite on **both CPU and GPU**:

```bash
fdl test            # CPU tests
fdl gpu-test        # GPU tests (parallel, excludes NCCL/Graph)
fdl gpu-test-all    # full GPU suite (parallel + NCCL isolated + serial)
```

The `gpu-*` commands take their cargo feature from the active libtorch
variant (`cuda` or `rocm`), so the same command line covers either vendor.

**Never trust a green summary on its own.** A test gated on hardware or on
a tool it cannot find returns early and is reported as `ok`, so a suite can
pass having executed nothing. If you add a gated test, prove it runs once
for real. `fdl coverage-all` exists partly for this: it runs every suite
this box can, and prints an explicit `RAN` / `SKIPPED` / `FAILED` roster
instead of one number that hides which suites never ran.

```bash
fdl coverage        # CPU only -- a FLOOR, it scores GPU code as missed
fdl coverage-all    # every suite this box can run, and names the ones it can't
```

Coverage is a number to look at, never a threshold to clear.

All tests use `test_device()` / `test_opts()` from `tensor.rs` so the same
test code runs on whichever device is available. When writing new tests:

- Use `test_device()` instead of `Device::CPU` for device selection
- Use `test_opts()` instead of `TensorOptions::default()` or `Default::default()`
- Use `on_device(..., test_device())` constructors instead of `::new()` for modules
- Tests that are inherently CPU-only (e.g. RSS-based leak checks) should guard
  with `if test_device() != Device::CPU { return; }` at the top

**Test template:**
```rust
#[test]
fn test_my_feature() {
    let dev = test_device();
    let opts = test_opts();

    let input = Tensor::randn(&[2, 4], opts).unwrap();
    let layer = Linear::on_device(4, 2, dev).unwrap();
    let x = Variable::new(input, true);
    let y = layer.forward(&x).unwrap();

    assert_eq!(y.data().shape(), vec![2, 2]);
}
```

If you add new functionality:

- **Tensor ops**: add tests in `tensor.rs`
- **Autograd ops**: add a numerical gradient check
- **NN modules**: add both a functional test and a gradient check
- **Graph features**: add a test in the graph module
- **Module constructors**: always provide an `on_device()` variant alongside `new()`

## Before Publishing to crates.io

The release gates live in `ci/release/` — one numbered script per
invariant, plus a `run-all.sh` that runs them in order:

```bash
sh ci/release/run-all.sh
```

Order matters for some of them. `11-semver-checks.sh` asks "is the
DECLARED version sufficient for the changes made", so it reads the version
out of `Cargo.toml` and must run *after* `02-version-sync.sh` — run it
before the bump and it correctly fails. Scripts that need a tool you don't
have SKIP loudly and print `UNVERIFIED` rather than passing quietly.

Everything below is also covered by that suite; it is spelled out because
these are the failures that cost the most when missed.

Always validate the docs.rs build locally before publishing. docs.rs uses nightly
Rust with `--cfg docsrs` and no libtorch — things that build fine in the dev
container can fail there.

```bash
make docs-rs    # simulates docs.rs build for every publishable crate
```

This covers every publishable crate in the workspace, in one disposable
container. It catches:
- Broken intra-doc links (`rustdoc::broken_intra_doc_links`)
- Dependencies that don't compile on nightly with `--cfg docsrs`
- Example scraping failures (examples need libtorch)
- Missing `#[cfg(docsrs)]` gates on FFI code

crates.io is immutable — a broken publish means bumping the version. Run this
before every `cargo publish`.

### Matching GitHub CI locally

`make docs-rs` mirrors docs.rs. For GitHub CI parity (which runs stable Rust
workspace-wide with `-D warnings` on rustdoc), use the `fdl` shortcuts:

```bash
fdl doc         # strict rustdoc pass (matches CI's "Doc" step)
fdl ci          # full CI CPU job: fmt + build + test + clippy + doc
```

`fdl doc` is fast and catches rustdoc regressions. `fdl ci` is the complete
CPU-job equivalent -- run it before pushing any PR that might fail CI.
`fdl test` alone will not catch rustdoc or formatting warnings.

CI runs two gates that need no libtorch and are not part of `fdl ci`:

- **`cargo deny check`** — advisories, licences and dependency sources. It
  runs as its own job so it never masks the test signal. Duplicate-version
  warnings are expected and deliberate (`multiple-versions = "warn"`);
  they are almost all transitive and not ours to fix.
- **MSRV** — the workspace is checked on exactly the `rust-version` in
  `Cargo.toml`, so a wrong value there is a failing build rather than a
  promise nobody tests.

Both can go red without anyone touching the code, when an advisory lands
or a dependency moves. That is the no-lockfile trade-off working as
intended, not a broken gate.

The Makefile still exposes a few targets that pre-date `fdl` (useful
when `fdl` itself is broken); treat it as a fallback, not the primary
workflow.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](./LICENSE).
