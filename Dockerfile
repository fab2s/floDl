# flodl CPU image -- libtorch mounted at runtime.
#
# No libtorch is baked into this image. Mount the appropriate variant
# from libtorch/ via docker-compose volumes.

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    unzip \
    ca-certificates \
    git \
    gcc \
    g++ \
    pkg-config \
    graphviz \
    # rsync: the source-fetch resolver's mtime-preserving transport. Present
    # so its tests exercise the real tool rather than skipping.
    rsync \
    # python3: Ubuntu's rrsync (in the rsync package) is a python3 script,
    # and the publish rrsync-pairing test runs the real thing — without the
    # interpreter the test skips instead of covering.
    python3-minimal \
    && rm -rf /var/lib/apt/lists/*

# --- Rust ---
ENV CARGO_HOME="/usr/local/cargo"
ENV RUSTUP_HOME="/usr/local/rustup"
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable \
    && chmod -R a+rwx "$CARGO_HOME" "$RUSTUP_HOME"
ENV PATH="${CARGO_HOME}/bin:${PATH}"

# libtorch is bind-mounted at runtime to /usr/local/libtorch
ENV LIBTORCH_PATH="/usr/local/libtorch"
ENV LD_LIBRARY_PATH="${LIBTORCH_PATH}/lib"
ENV LIBRARY_PATH="${LIBTORCH_PATH}/lib"

WORKDIR /workspace
