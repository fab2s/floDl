#!/bin/sh
# `cargo publish --dry-run --workspace` for every publishable workspace
# crate, in dependency order (cargo computes it).
#
# --workspace (cargo >= 1.90) verifies non-leaf crates against a local
# overlay of the just-packaged workspace deps instead of crates.io, so
# a synchronized version bump dry-runs clean BEFORE anything is
# published. The old per-crate loop could never pass on a bump: flodl's
# verify step tried to resolve the new flodl-sys version from the
# registry, where it doesn't exist yet.
#
# Runs inside the `dev` docker service so libtorch is available to the
# build step: `flodl-sys` and `flodl` both link against libtorch, and a
# host without `LIBTORCH` exported would fail at `build.rs` or link
# time. Docker keeps the environment uniform and close to what
# crates.io / docs.rs sees.
#
# Catches: missing package metadata, stale `path = "../foo"` deps
# without `version = "..."` companions, oversized crates, uncommitted
# file rejection, link-time breakage.
#
# RELEASE_PREP=1 (or `run-all.sh --prep`) adds --allow-dirty so the
# dry-run can validate the uncommitted release prep (version bump +
# CHANGELOG). The strict run keeps cargo's dirty-tree rejection.
#
# This does NOT actually publish -- dry-run stops right before upload.

set -u
cd "$(git rev-parse --show-toplevel)"

# Mirror flodl-cli/src/run.rs::libtorch_env -- docker-compose.yml uses
# LIBTORCH_CPU_PATH (always) and LIBTORCH_HOST_PATH / CUDA_VERSION /
# CUDA_TAG (when an active CUDA variant exists) to pick mount points
# and image tags. Exporting them here gives docker-compose the same
# resolved state that `fdl build` would see.
ACTIVE=$(tr -d '[:space:]' < libtorch/.active 2>/dev/null || true)
export LIBTORCH_CPU_PATH="./libtorch/precompiled/cpu"
if [ -n "$ACTIVE" ]; then
    export LIBTORCH_HOST_PATH="./libtorch/$ACTIVE"
    ARCH_CUDA=$(grep '^cuda=' "./libtorch/$ACTIVE/.arch" 2>/dev/null | cut -d= -f2 || true)
    if [ -n "$ARCH_CUDA" ] && [ "$ARCH_CUDA" != "none" ]; then
        case "$ARCH_CUDA" in
            *.*.*) CUDA_VERSION="$ARCH_CUDA" ;;
            *)     CUDA_VERSION="$ARCH_CUDA.0" ;;
        esac
        CUDA_TAG=$(echo "$CUDA_VERSION" | cut -d. -f1,2)
        export CUDA_VERSION CUDA_TAG
    fi
fi

# Workspace members are exactly the published crates (benchmarks,
# ddp-bench, hf-ddp are workspace-excluded), so --workspace covers
# flodl-sys, flodl-cli-macros, flodl, flodl-cli, flodl-hf.
echo "=== cargo publish --dry-run --workspace (in docker dev) ==="
if ! docker compose run --rm -T dev cargo publish --dry-run --workspace ${RELEASE_PREP:+--allow-dirty}; then
    echo "FAIL: cargo publish --dry-run --workspace failed (see output above)"
    exit 1
fi

echo ""
echo "PASS: all published crates pass cargo publish --dry-run --workspace (docker dev)"
