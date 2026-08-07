# flodl development -- legacy Makefile
#
# All development commands are available via fdl (see fdl.yml.example).
# This Makefile retains only host-side tasks that fdl can't handle.
#
# Quick start:
#   fdl setup             # detect hardware, download libtorch, build Docker image
#   fdl test              # run CPU tests
#   fdl gpu-test-all     # full CUDA suite

COMPOSE = docker compose

# docker-compose.yml sets each service's `hostname:` from HOSTNAME so the
# cluster launcher's `cluster.hosts[i].name == hostname()` match works from
# inside the container. bash keeps HOSTNAME as a shell variable but never
# exports it, so make doesn't inherit it and every COMPOSE target below
# would otherwise run with a blank one. `fdl` fills it the same way
# (flodl-cli/src/main.rs). `?=` leaves an already-exported value alone.
HOSTNAME ?= $(shell hostname)
export HOSTNAME

.PHONY: docs-rs site site-stop sync-skills test-init release-check clean

# --- docs.rs validation (host-side mkdir + nightly toolchain) ---
#
# `--cfg docsrs` is set as a rustdoc-arg ONLY, not as a rustc-arg.
# Setting it via `build.rustflags` would propagate to every dep compile,
# including serde's. serde 1.0.228's build script gates a path-rewrite
# on `cfg(all(docsrs, if_docsrs_then_no_serde_core))` that loads an
# incomplete `private` submodule, breaking every `#[derive(Serialize)]`
# site with "cannot find type Formatter in module _serde::__private228".
# Real docs.rs only applies the cfg at doc time, not compile time —
# match that exactly to keep this gate honest.
#
# `-D warnings` promotes rustdoc warnings (broken intra-doc-links,
# private-item leaks, etc.) to hard errors so this target catches
# anything CI would catch. Without it, broken doc-links emit silent
# warnings and CI fails on the same crate with `-D warnings` in
# RUSTDOCFLAGS — defeating the "gate locally first" purpose.
#
# Three pass layers, each with `-D warnings`:
#
# 1. **CI parity pass** — `cargo doc --no-deps --document-private-items`
#    with stable toolchain, no `--cfg docsrs`. Mirrors the CI workflow
#    (.github/workflows/ci.yml) verbatim. `--document-private-items`
#    catches warnings rustdoc skips by default (e.g. redundant explicit
#    links inside private-item docs). The CI gate fails if this fails;
#    we must catch it locally first.
#
# 2. **docs.rs hosting pass (flodl)** — nightly + `--cfg docsrs` +
#    `--no-default-features --features rng`, mirroring
#    `[package.metadata.docs.rs]`. This is what the published docs.rs
#    page would generate. Catches docsrs-specific breakage.
#
# 3. **docs.rs hosting pass (per-crate)** — flodl-hw, flodl-sys,
#    flodl-cli, flodl-cli-macros, flodl-hf each documented with `--cfg docsrs`
#    and their docs.rs feature set. Plus a flodl `--all-features` pass
#    to catch any cuda-gated breakage CI's default-feature build
#    wouldn't see (rustdoc parses without GPU; flodl-sys's libtorch
#    skip is gated on DOCS_RS=1, set in docker-compose).
docs-rs:
	@mkdir -p .cargo-cache-docsrs .cargo-git-docsrs .target-docsrs
	$(COMPOSE) run --rm docs-rs bash -c "\
		rustup install nightly 2>&1 | tail -1 && \
		RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps --document-private-items && \
		cargo +nightly rustdoc --lib -p flodl-hw \
			--config 'build.rustdocflags=[\"--cfg\", \"docsrs\", \"-D\", \"warnings\"]' && \
		cargo +nightly rustdoc --lib -p flodl-sys \
			--config 'build.rustdocflags=[\"--cfg\", \"docsrs\", \"-D\", \"warnings\"]' && \
		cargo +nightly rustdoc --lib -p flodl \
			--no-default-features --features rng \
			--config 'build.rustdocflags=[\"--cfg\", \"docsrs\", \"-D\", \"warnings\"]' && \
		cargo +nightly rustdoc --lib -p flodl \
			--all-features \
			--config 'build.rustdocflags=[\"--cfg\", \"docsrs\", \"-D\", \"warnings\"]' && \
		cargo +nightly rustdoc --lib -p flodl-cli \
			--config 'build.rustdocflags=[\"--cfg\", \"docsrs\", \"-D\", \"warnings\"]' && \
		cargo +nightly rustdoc --lib -p flodl-cli-macros \
			--config 'build.rustdocflags=[\"--cfg\", \"docsrs\", \"-D\", \"warnings\"]' && \
		cargo +nightly rustdoc --lib -p flodl-hf \
			--all-features \
			--config 'build.rustdocflags=[\"--cfg\", \"docsrs\", \"-D\", \"warnings\"]'"

# --- Site (host python + docker compose up/down) ---

site:
	@python3 site/build_guide.py
	$(COMPOSE) up jekyll

site-stop:
	$(COMPOSE) down jekyll

# --- Skill assets: sync the ai/ skill sources into the crate ---
#
# flodl-cli/assets/skills/ is the copy include_str!'d into the fdl binary
# (the out-of-repo fallback for `fdl skill install`). crates.io only
# packages the crate dir, so the copy must live under flodl-cli/. `make
# site` refreshes it automatically as a side effect; this is the explicit
# form. ci/release/09-skill-assets.sh fails release on drift.
sync-skills:
	@cp ai/skills/port/guide.md          flodl-cli/assets/skills/port-guide.md
	@cp ai/skills/port/instructions.md   flodl-cli/assets/skills/port-instructions.md
	@cp ai/adapters/claude/port-skill.md flodl-cli/assets/skills/claude-port.md
	@echo "synced flodl-cli/assets/skills/ from ai/"

# --- Smoke test: init.sh end-to-end ---
#
# Scaffolds a project with --docker (explicit to avoid the interactive
# prompt), then verifies the expected files landed and docker compose
# accepts the generated config. Uses $FDL_BIN to run the locally-built
# binary rather than the last-released one on GitHub.
#
# We do NOT run `./fdl build` here -- that downloads a release binary
# and pulls base images, which is too heavy for a smoke test. Build
# correctness is covered by `fdl test`.

test-init:
	@echo "=== Testing init.sh scaffold ==="
	@cargo build --release -p flodl-cli >/dev/null
	@cd /tmp && rm -rf flodl-init-test && \
		FDL_BIN=$(CURDIR)/target/release/fdl \
		sh $(CURDIR)/init.sh flodl-init-test --docker
	@for f in Cargo.toml src/main.rs fdl.yml.example fdl .gitignore \
	          Dockerfile.cpu Dockerfile.cuda docker-compose.yml; do \
		test -f /tmp/flodl-init-test/$$f || { echo "missing: $$f"; exit 1; }; \
	done
	@test -x /tmp/flodl-init-test/fdl || { echo "fdl bootstrap not executable"; exit 1; }
	@cd /tmp/flodl-init-test && docker compose config >/dev/null
	@rm -rf /tmp/flodl-init-test
	@echo "=== init.sh smoke test passed ==="

# --- Release readiness ---
#
# Runs every ci/release/NN-*.sh check and prints a pass/fail summary.
# See docs/release.md for what each script verifies and how to fix a
# failing check.

release-check:
	@sh ci/release/run-all.sh

# --- Cleanup ---

clean:
	$(COMPOSE) down -v --rmi local
