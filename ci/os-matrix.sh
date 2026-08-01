#!/usr/bin/env bash
# Per-OS coverage for the parts of floDl that are OS-specific.
#
# WHY A SCRIPT AND NOT YAML STEPS
#
# This runs end to end on a developer box. Every CI bug this file replaced
# was a YAML bug that only a runner could reveal: `libtorch: yes` parsed as
# a boolean so `== 'yes'` never matched and every gated step silently
# skipped GREEN; a job-level `needs:` that skipped macOS because Windows
# was red. Logic that can be executed locally does not fail that way.
#
# It also follows the shape ci/release/*.sh already uses, and folds what
# were twenty conditional steps into one readable sequence:
#
#   test -> clippy -> build fdl -> probe(bare) -> URL checks -> refusal
#        -> install libtorch -> probe(configured) -> toolkit -> build+lint
#
# WHAT IT SIMULATES
#
# A person on a fresh machine, using `fdl` the documented way. It asks the
# tool what it sees, follows the tool's own advice when something is
# missing, and checks the advice worked. Nothing is hand-rolled that `fdl`
# is supposed to do -- if a step here needs a raw wget or an undocumented
# path, that is a gap in fdl, not a thing to work around.
#
# Local use:  bash ci/os-matrix.sh
# Env:        FDL_CI_VARIANT=cuda|rocm   force the Linux GPU vendor
#             FDL_CI_SKIP_INSTALL=1      skip the sudo apt steps
#             FDL_CI_SKIP_LIBTORCH=1     skip the download/install phase,
#                                        so a local run never mutates an
#                                        existing libtorch/.active

set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

# --- log helpers -----------------------------------------------------
# ::group:: collapses in the Actions UI and is inert in a local shell,
# so the same output is readable in both places.
IN_CI="${GITHUB_ACTIONS:-}"

# Actions renders ANSI, so the same codes serve the log and a terminal.
# Suppressed when NO_COLOR is set or stdout is not a tty and we are not
# in CI, so piping the output somewhere stays clean.
if [ -n "${NO_COLOR:-}" ] || { [ -z "$IN_CI" ] && [ ! -t 1 ]; }; then
    C_GREEN=""; C_RED=""; C_YELLOW=""; C_OFF=""
else
    C_GREEN=$'\033[32m'; C_RED=$'\033[31m'; C_YELLOW=$'\033[33m'; C_OFF=$'\033[0m'
fi

# GROUP_OPEN tracks whether a ::group:: fold is currently open. A failure
# printed inside one is HIDDEN behind the toggle, which is exactly what
# made the last red run hard to read -- so `fail` closes the fold first.
GROUP_OPEN=0
group()    { if [ -n "$IN_CI" ]; then echo "::group::$*"; GROUP_OPEN=1; else echo; echo "===== $* ====="; fi; }
endgroup() { if [ -n "$IN_CI" ] && [ "$GROUP_OPEN" = 1 ]; then echo "::endgroup::"; GROUP_OPEN=0; fi; }
pass()     { echo "${C_GREEN}PASS${C_OFF}: $*"; }
note()     { if [ -n "$IN_CI" ]; then echo "::notice::$*"; else echo "NOTE: $*"; fi; }

# `fdl probe` exits 1 whenever it reports ANY issue, and a CI host always
# has some (no GPU, no NCCL). Under `set -o pipefail` that makes
# `probe | grep -q` return non-zero EVEN WHEN GREP MATCHES -- which
# silently inverted every probe assertion in this script and turned four
# green hosts red. Capture the output first, then match against the text.
probe_json() { "$FDL" probe --json 2>/dev/null || true; }
probe_text() { "$FDL" probe 2>&1 || true; }
has()        { printf '%s' "$1" | grep -qF "$2"; }

SOFT_FAIL=0
# FAIL EARLY, and out in the open. Collecting every failure sounded
# useful but buried the real one: Actions folds each ::group::, so the
# first error scrolled away inside a closed toggle while the run ended on
# unrelated output. Stopping at the first hard failure makes it the LAST
# thing in the log, every time. Per-host coverage is unaffected --
# `fail-fast: false` on the matrix means the other hosts still run.
fail() {
    endgroup
    echo "${C_RED}FAIL${C_OFF}: $*"
    [ -n "$IN_CI" ] && echo "::error::$*"
    echo
    echo "${C_RED}RESULT: $HOST FAILED${C_OFF} (stopped at the first failure)"
    exit 1
}
# Advisory: report, keep going, do not fail the job. For steps running
# code no host has ever run, so that a red one does not gate the merge
# queue on day one. Promote to `fail` once green.
soft() {
    SOFT_FAIL=1
    echo "${C_YELLOW}ADVISORY${C_OFF}: $*"
    [ -n "$IN_CI" ] && echo "::warning::(advisory) $*"
    return 0
}

# --- host identification --------------------------------------------
# Derived, not passed in from the matrix: the script should see the host
# the way a user's machine does, and the workflow then carries nothing
# but a list of runner images.
KERNEL="$(uname -s)"
MACHINE="$(uname -m)"
case "$KERNEL" in
    Linux)                       HOST_OS=linux ;;
    Darwin)                      HOST_OS=macos ;;
    MINGW*|MSYS*|CYGWIN*)        HOST_OS=windows ;;
    *)                           echo "unsupported kernel: $KERNEL"; exit 1 ;;
esac
case "$MACHINE" in
    x86_64|amd64)                HOST_ARCH=x86_64 ;;
    arm64|aarch64)               HOST_ARCH=aarch64 ;;
    *)                           echo "unsupported machine: $MACHINE"; exit 1 ;;
esac
HOST="$HOST_OS-$HOST_ARCH"

# --- per-host plan ---------------------------------------------------
# LT_FLAG      what `fdl libtorch download` is asked for ("" = nothing
#              upstream publishes for this host)
# LT_DIR       the variant directory that install must produce
# LT_LIB       the vendor library that proves the archive really unpacked
# REFUSE_FLAG  a variant this host must REFUSE, with the message it must
#              refuse with. Keyed on the (host, variant) PAIR because
#              refusal is not a property of the host: download_url_for
#              has three distinct refusal arms and macOS resolves CPU
#              happily while still having to reject --cuda.
# COMPILE      the framework can be built here afterwards
# *_ADVISORY   this path has never run anywhere; report, do not block
LT_FLAG=""; LT_DIR=""; LT_LIB=""
REFUSE_FLAG=""; REFUSE_MSG=""
COMPILE=0; GPU=0; ALL_VARIANTS=0
INSTALL_ADVISORY=0; COMPILE_ADVISORY=0

case "$HOST" in
    linux-x86_64)
        # What a real Linux user has. The CPU build on Linux is ci.yml's
        # every-PR job already, so this leg spends itself on the vendor
        # path instead. Vendor alternates per run so steady-state cost
        # stays at one toolkit install; deterministic, so a re-run
        # repeats the same variant rather than testing something else.
        GPU=1; COMPILE=1; ALL_VARIANTS=1
        VARIANT="${FDL_CI_VARIANT:-}"
        if [ -z "$VARIANT" ] || [ "$VARIANT" = rotate ]; then
            case $(( ${GITHUB_RUN_NUMBER:-0} % 2 )) in
                0) VARIANT=cuda ;;
                1) VARIANT=rocm ;;
            esac
        fi
        case "$VARIANT" in
            cuda) LT_FLAG="--cuda 12.8"; LT_DIR=cu128  ;;
            rocm) LT_FLAG="--rocm 7.0";  LT_DIR=rocm70 ;;
            *)    echo "unknown FDL_CI_VARIANT: $VARIANT"; exit 1 ;;
        esac
        LT_LIB="lib/libtorch.so"
        ;;
    linux-aarch64)
        # Upstream publishes no archive for this host in any variant, so
        # the whole leg is the assertion that fdl says so rather than
        # building a URL that 404s.
        REFUSE_FLAG="--cpu"; REFUSE_MSG="Unsupported platform"
        ;;
    macos-aarch64)
        # CPU only, and not a gap to fill later: upstream publishes no
        # CUDA or ROCm libtorch for macOS at all, which the refusal
        # below asserts. flodl-sys's C++ shim has never been compiled by
        # clang-on-macOS, so the build is advisory until it is green.
        LT_FLAG="--cpu"; LT_DIR=cpu; LT_LIB="lib/libtorch.dylib"
        COMPILE=1; COMPILE_ADVISORY=1
        REFUSE_FLAG="--cuda 12.8"; REFUSE_MSG="macOS only supports CPU libtorch"
        ;;
    windows-x86_64)
        # Installs but does not compile. `flodl-sys/build.rs` has no
        # target_os arm at all -- it emits `rustc-link-lib=dylib=dl`
        # unconditionally and probes `lib/libtorch.so`, and ops_nn.cpp
        # includes <dlfcn.h>. Clippy would not dodge it either: build.rs
        # still runs and `cc` still compiles the shim.
        #
        # Installing is covered on purpose, and separately from
        # compiling: util/archive.rs reaches for `PowerShell
        # Expand-Archive` here against `unzip` everywhere else, and that
        # branch had never executed anywhere. `--dry-run` stops at the
        # URL without extracting, so a URL check cannot reach it.
        LT_FLAG="--cpu"; LT_DIR=cpu; LT_LIB="lib/torch.dll"
        INSTALL_ADVISORY=1
        REFUSE_FLAG="--rocm 7.0"; REFUSE_MSG="not available for Windows"
        ;;
    *)
        echo "no plan for host: $HOST"; exit 1 ;;
esac

note "host $HOST${LT_FLAG:+ -> installs '$LT_FLAG' ($LT_DIR)}"

# `fdl` is `fdl.exe` under Git Bash.
FDL=target/release/fdl
[ -f "$FDL" ] || FDL=target/release/fdl.exe

# =====================================================================
group "Test (flodl-cli + flodl-hw, no libtorch)"
# Both crates are libtorch-free by construction -- flodl-cli is zero-dep
# on flodl and flodl-hw is dependency-free -- so this needs no native
# stack at all. `--all-targets` is load-bearing: the release workflow
# builds these without it, so test code had never been compiled off
# Linux, and three POSIX-assuming fixtures were waiting there when it
# first was.
cargo test -p flodl-cli -p flodl-hw --all-targets || fail "$HOST: cli/hw tests"
endgroup

group "Clippy (flodl-cli + flodl-hw)"
cargo clippy -p flodl-cli -p flodl-hw --all-targets -- -W clippy::all \
    || fail "$HOST: cli/hw clippy"
endgroup

group "Build fdl"
cargo build --release -p flodl-cli || { fail "$HOST: fdl build"; exit 1; }
FDL=target/release/fdl
[ -f "$FDL" ] || FDL=target/release/fdl.exe
endgroup

# =====================================================================
group "fdl probe -- bare host"
# A user's first move. Before any libtorch exists `libtorch` is null and
# the issue list says what to run.
#
# Asserted on --json, never on the exit code: probe exits 1 for ANY
# issue, and a hosted runner always has some (no GPU, no NCCL). The
# JSON is emitted compact, so `"libtorch":null` has no spaces in it.
"$FDL" probe || true
# A CI checkout is always bare (libtorch/ is gitignored), but a developer
# box usually is not -- so this asserts only when the host really is
# unprovisioned rather than failing anyone who runs the script locally.
if [ -e libtorch/.active ]; then
    note "libtorch already provisioned here; skipping the bare-host assertions"
else
    BARE_JSON=$(probe_json)
    if has "$BARE_JSON" '"libtorch":null'; then
        pass "probe reports libtorch unconfigured on a bare host"
    else
        fail "$HOST: probe should report libtorch:null before any install"
    fi
    if has "$BARE_JSON" 'libtorch not configured'; then
        pass "probe names the fix"
    else
        fail "$HOST: probe should tell the user to run 'fdl libtorch download'"
    fi
fi
endgroup

# =====================================================================
if [ -n "$LT_FLAG" ]; then
group "libtorch URL resolves and is live"
# The URL grammar differs per OS (macOS has its own filename, Windows
# carries a `-win-` infix) and unit tests cover the grammar offline.
# This covers what they cannot: that the URL this host resolves is live
# upstream. `--dry-run` prints it and stops, so nothing downloads.
URL=$("$FDL" libtorch download $LT_FLAG --dry-run | sed -n 's/.*URL:[[:space:]]*//p' | head -1)
if [ -z "$URL" ]; then
    fail "$HOST: --dry-run printed no URL"
else
    echo "resolved: $URL"
    CODE=$(curl -sL -o /dev/null -w '%{http_code}' --max-time 60 \
                --retry 3 --retry-delay 5 -r 0-0 "$URL")
    case "$CODE" in
        200|206) pass "$HOST URL is live (HTTP $CODE)" ;;
        *)       fail "$HOST: $URL -> HTTP $CODE" ;;
    esac
fi
endgroup
fi

if [ "$ALL_VARIANTS" = 1 ]; then
group "Every variant URL is live"
# The vendor rotation installs one variant per run; this keeps the
# others from rotting silently upstream. Range requests only.
for FLAG in "--cpu" "--cuda 12.6" "--cuda 12.8" "--rocm 7.0"; do
    U=$("$FDL" libtorch download $FLAG --dry-run | sed -n 's/.*URL:[[:space:]]*//p' | head -1)
    if [ -z "$U" ]; then fail "$FLAG resolved no URL"; continue; fi
    C=$(curl -sL -o /dev/null -w '%{http_code}' --max-time 60 --retry 3 --retry-delay 5 -r 0-0 "$U")
    printf '%-14s HTTP %s  %s\n' "$FLAG" "$C" "$U"
    case "$C" in 200|206) ;; *) fail "$FLAG -> HTTP $C" ;; esac
done
endgroup
fi

# =====================================================================
if [ -n "$REFUSE_FLAG" ]; then
group "libtorch is correctly refused ($REFUSE_FLAG)"
# Where upstream publishes nothing, refusing is the correct behaviour
# and must stay loud. download_url_for's arms are unit-tested from any
# host; what this adds is that the real binary on the real host turns
# that Err into a non-zero exit and prints the message.
OUT=$("$FDL" libtorch download $REFUSE_FLAG --dry-run 2>&1) && RC=0 || RC=$?
echo "$OUT"
if [ "$RC" -eq 0 ]; then
    fail "$HOST accepted '$REFUSE_FLAG', expected a refusal"
elif echo "$OUT" | grep -qF "$REFUSE_MSG"; then
    pass "$HOST refuses '$REFUSE_FLAG' with a reason"
else
    fail "$HOST refused '$REFUSE_FLAG' but not with: $REFUSE_MSG"
fi
endgroup
fi

# =====================================================================
if [ -n "$LT_FLAG" ] && [ "${FDL_CI_SKIP_LIBTORCH:-}" = 1 ]; then
    note "FDL_CI_SKIP_LIBTORCH=1 -- not installing libtorch"
elif [ -n "$LT_FLAG" ]; then
group "Install libtorch via fdl ($LT_FLAG)"
# The step ci.yml cannot make: libtorch arrives through the tool's own
# resolve-download-extract-activate path, not a wget whose URL is
# duplicated in YAML.
#
# The assertions are the point on Windows. A zero exit from
# Expand-Archive is not evidence the tree landed -- assert what the rest
# of the toolchain reads.
INSTALL_OK=1
"$FDL" libtorch download $LT_FLAG || INSTALL_OK=0
"$FDL" libtorch list || true

LT="libtorch/precompiled/$LT_DIR"
[ -d "$LT" ]                                              || { INSTALL_OK=0; echo "missing dir $LT"; }
[ -f "$LT/$LT_LIB" ]                                      || { INSTALL_OK=0; echo "missing $LT_LIB"; }
[ -f "$LT/include/torch/csrc/api/include/torch/torch.h" ] || { INSTALL_OK=0; echo "missing headers"; }
# `.active` is read with CR stripped: fdl writes plain \n, but a
# CRLF-normalising checkout on Windows must not turn a healthy install
# into a confusing assertion failure.
if [ -f libtorch/.active ]; then
    ACTIVE=$(tr -d '\r\n' < libtorch/.active)
    [ "$ACTIVE" = "precompiled/$LT_DIR" ] || { INSTALL_OK=0; echo ".active is '$ACTIVE'"; }
else
    INSTALL_OK=0; echo "no libtorch/.active"
fi

# MAX_PATH canary, diagnostic only. PowerShell's Expand-Archive goes
# through .NET ZipFile, which fails past 260 chars, and libtorch's
# include/torch/csrc/api/include/... tree is deep. /usr/bin/find
# explicitly: a bare `find` in Git Bash can resolve to System32's
# find.exe, which does not speak `-type f`.
if [ -d "$LT" ]; then
    echo "deepest extracted path:"
    /usr/bin/find "$LT" -type f 2>/dev/null | awk '{print length, $0}' | sort -rn | head -1 \
        || echo "  (probe unavailable)"
fi

if [ "$INSTALL_OK" = 1 ]; then
    pass "$HOST installed libtorch via fdl"
elif [ "$INSTALL_ADVISORY" = 1 ]; then
    soft "$HOST: libtorch install incomplete (first run of this path)"
else
    fail "$HOST: libtorch install incomplete"
fi
endgroup

group "fdl probe -- configured host"
# The other half of the pair: same command, and now it reports a variant
# instead of telling you to install one.
"$FDL" probe || true
if has "$(probe_json)" "\"path\":\"precompiled/$LT_DIR\""; then
    pass "probe now reports precompiled/$LT_DIR"
elif [ "$INSTALL_ADVISORY" = 1 ]; then
    soft "$HOST: probe does not report the variant (install was advisory)"
else
    fail "$HOST: probe should report precompiled/$LT_DIR after install"
fi

# On a vendor variant the toolkit is still absent at this point, and
# probe is supposed to say so BEFORE a build fails. This is the earlier
# half of the pair build.rs guards at compile time.
if [ "$GPU" = 1 ] && ! has "$(probe_text)" 'toolkit headers are missing'; then
    if [ -e "${ROCM_PATH:-/opt/rocm}/include/hip/hip_runtime.h" ] \
       || [ -e "${CUDA_HOME:-/usr/local/cuda}/include/cuda_runtime.h" ]; then
        note "toolkit already installed; nothing for probe to warn about"
    else
        fail "$HOST: probe should warn that the $LT_DIR toolkit headers are missing"
    fi
elif [ "$GPU" = 1 ]; then
    pass "probe warns about the missing vendor toolkit before any build"
fi
endgroup
fi

# =====================================================================
# Point the toolchain at the variant this run installed, BEFORE anything
# compiles. Found by running the script locally: without it the toolkit
# phase below picked up whatever libtorch was already on the box and
# tripped the "needs a ROCm libtorch" guard instead of the toolkit-header
# one it is meant to exercise.
if [ -n "$LT_DIR" ] && [ -d "libtorch/precompiled/$LT_DIR" ]; then
    LT_ABS="$PWD/libtorch/precompiled/$LT_DIR"
    export LIBTORCH_PATH="$LT_ABS"
    export LD_LIBRARY_PATH="$LT_ABS/lib:${LD_LIBRARY_PATH:-}"
    export LIBRARY_PATH="$LT_ABS/lib:${LIBRARY_PATH:-}"
    export DYLD_LIBRARY_PATH="$LT_ABS/lib:${DYLD_LIBRARY_PATH:-}"
    note "LIBTORCH_PATH=$LT_ABS"
fi

if [ "$GPU" = 1 ]; then
group "Vendor toolkit: fdl says what is missing, then we do it"
# The three-phase test of the GUIDANCE, not merely of a message. Build
# once with nothing installed and expect build.rs to name the packages;
# install exactly those; build again and expect success.
#
# Native, no container. libtorch already ships every library the link
# needs -- libamdhip64 included -- so what a vendor install adds is
# HEADERS: hip/hip_runtime.h + rccl/rccl.h, or cuda_runtime.h + nccl.h.
# Package names were read out of the dev images with `dpkg -S` on the
# readlink -f'd path (/opt/rocm is a versioned symlink, so the naive
# lookup reports "not owned").
FEATURE=$([ "$LT_DIR" = rocm70 ] && echo rocm || echo cuda)

OUT=$(cargo build -p flodl-sys --features "$FEATURE" 2>&1) && RC=0 || RC=$?
if [ "$RC" -eq 0 ]; then
    note "toolkit already present; skipping the missing-case assertion"
elif echo "$OUT" | grep -q 'needs the .* toolkit headers'; then
    pass "build.rs named the missing toolkit instead of dying in the C++ compile"
    echo "$OUT" | grep -A9 'toolkit headers' | head -12
else
    fail "$HOST: missing toolkit should produce an actionable message, got:"
    echo "$OUT" | tail -20
fi

if [ "${FDL_CI_SKIP_INSTALL:-}" = 1 ]; then
    note "FDL_CI_SKIP_INSTALL=1 -- not installing the toolkit"
    # Nothing downstream can build without it. Skipping a prerequisite
    # on purpose is not a failure; only CI leaves this unset.
    [ "$RC" -ne 0 ] && COMPILE=0
elif [ "$RC" -ne 0 ]; then
    # Exactly the packages the message names. If these drift, the
    # message is wrong and this step is what says so.
    if [ "$FEATURE" = rocm ]; then
        sudo mkdir -p --mode=0755 /etc/apt/keyrings
        curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key \
            | sudo gpg --dearmor -o /etc/apt/keyrings/rocm.gpg
        echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/7.0 noble main" \
            | sudo tee /etc/apt/sources.list.d/rocm.list >/dev/null
        sudo apt-get update -qq
        sudo apt-get install -y --no-install-recommends hip-dev rccl-dev || fail "rocm toolkit install"
    else
        curl -fsSLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
        sudo dpkg -i cuda-keyring_1.1-1_all.deb
        sudo apt-get update -qq
        sudo apt-get install -y --no-install-recommends cuda-cudart-dev-12-8 cuda-crt-12-8 libnccl-dev \
            || fail "cuda toolkit install"
    fi
fi
endgroup
fi

# =====================================================================
if [ "$COMPILE" = 1 ]; then
group "Build and clippy flodl"
if [ "$GPU" = 1 ]; then
    # Unscoped on purpose. ci.yml's rocm job must scope to
    # -p flodl-sys -p flodl because its sparse extract cannot satisfy a
    # link; this installed the whole tree, so a workspace build links
    # for real -- which is where the libtorch_cuda/hip force-load and
    # gpu_compat.h's symbol mapping fail if they are wrong.
    FEATURE=$([ "$LT_DIR" = rocm70 ] && echo rocm || echo cuda)
    BUILD_CMD="cargo build --features $FEATURE"
    LINT_CMD="cargo clippy --features $FEATURE --all-targets -- -W clippy::all"
else
    BUILD_CMD="cargo build -p flodl-sys -p flodl"
    LINT_CMD="cargo clippy -p flodl-sys -p flodl --all-targets -- -W clippy::all"
fi

OK=1
$BUILD_CMD || OK=0
[ "$OK" = 1 ] && { $LINT_CMD || OK=0; }

if [ "$OK" = 1 ]; then
    pass "$HOST built and linted flodl"
elif [ "$COMPILE_ADVISORY" = 1 ]; then
    soft "$HOST: flodl build/clippy failed (never compiled on this host before)"
else
    fail "$HOST: flodl build/clippy"
fi
endgroup
fi

# =====================================================================
echo
if [ "$SOFT_FAIL" != 0 ]; then
    echo "${C_YELLOW}RESULT: $HOST ok, with advisories${C_OFF} (see the warnings above)"
    exit 0
fi
echo "${C_GREEN}RESULT: $HOST ok${C_OFF}"
