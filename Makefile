# flodl development -- legacy Makefile
#
# All development commands are available via fdl (see fdl.yml.example).
# This Makefile retains only host-side tasks that fdl can't handle.
#
# Quick start:
#   fdl setup             # detect hardware, download libtorch, build Docker image
#   fdl test              # run CPU tests
#   fdl cuda-test-all     # full CUDA suite

COMPOSE = docker compose

.PHONY: docs-rs site site-stop test-init release-check clean

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
docs-rs:
	@mkdir -p .cargo-cache-docsrs .cargo-git-docsrs .target-docsrs
	$(COMPOSE) run --rm docs-rs bash -c "\
		rustup install nightly 2>&1 | tail -1 && \
		cargo +nightly rustdoc --lib -p flodl \
			--no-default-features --features rng \
			--config 'build.rustdocflags=[\"--cfg\", \"docsrs\"]' && \
		cargo +nightly rustdoc --lib -p flodl-cli \
			--config 'build.rustdocflags=[\"--cfg\", \"docsrs\"]' && \
		cargo +nightly rustdoc --lib -p flodl-cli-macros \
			--config 'build.rustdocflags=[\"--cfg\", \"docsrs\"]' && \
		cargo +nightly rustdoc --lib -p flodl-hf \
			--all-features \
			--config 'build.rustdocflags=[\"--cfg\", \"docsrs\"]'"

# --- Site (host python + docker compose up/down) ---

site:
	@python3 site/build_guide.py
	$(COMPOSE) up jekyll

site-stop:
	$(COMPOSE) down jekyll

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
