#!/usr/bin/env bash
# Per-OS coverage for the parts of floDl that are OS-specific.
#
# A script rather than YAML steps, so the logic can be run on a developer
# box before a runner sees it, following the shape ci/release/*.sh uses.
# One sequence per host:
#
#   test -> clippy -> build fdl -> probe(bare) -> URL checks -> refusal
#        -> install libtorch -> probe(configured) -> toolkit -> build+lint
#
# It exercises `fdl` the documented way: ask the tool what it sees,
# follow its advice when something is missing, verify the advice worked.
# Anything here that needs a raw wget or an undocumented path is a gap in
# fdl rather than something to work around.
#
# Local use:  bash ci/os-matrix.sh
# Env:        FDL_CI_VARIANT=cuda|rocm|cpu   force the Linux variant
#                            rotate-alt = rotate on the opposite phase,
#                            so a second Linux leg covers the other
#                            vendor in the same run
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

# GROUP_OPEN tracks whether a ::group:: fold is open. A failure printed
# inside one is hidden behind the toggle, so `fail` closes it first.
GROUP_OPEN=0
group()    { if [ -n "$IN_CI" ]; then echo "::group::$*"; GROUP_OPEN=1; else echo; echo "===== $* ====="; fi; }
endgroup() { if [ -n "$IN_CI" ] && [ "$GROUP_OPEN" = 1 ]; then echo "::endgroup::"; GROUP_OPEN=0; fi; }
pass()     { echo "${C_GREEN}PASS${C_OFF}: $*"; }
note()     { if [ -n "$IN_CI" ]; then echo "::notice::$*"; else echo "NOTE: $*"; fi; }

# `fdl probe` exits 1 whenever it reports any issue, and a CI host always
# has some (no GPU, no NCCL). Under `set -o pipefail`, `probe | grep -q`
# then returns non-zero even when grep matches, so capture the output
# first and match against the text.
probe_json() { "$FDL" probe --json 2>/dev/null || true; }
probe_text() { "$FDL" probe 2>&1 || true; }
has()        { printf '%s' "$1" | grep -qF "$2"; }

# `fdl` is `fdl.exe` under Git Bash.
find_fdl() {
    FDL=target/release/fdl
    [ -f "$FDL" ] || FDL=target/release/fdl.exe
}

# Resolve the URL a variant flag produces and assert it is live upstream
# (range request only, nothing downloads). Unit tests cover the URL
# grammar offline; this covers what they cannot -- that the URL this
# host resolves still exists.
url_live() {
    local flag="$1" url code rc err
    # shellcheck disable=SC2086
    url=$("$FDL" libtorch download $flag --dry-run | sed -n 's/.*URL:[[:space:]]*//p' | head -1)
    [ -n "$url" ] || fail "$flag resolved no URL"
    err=$(mktemp)
    # -sS keeps stdout to the write-out code while stderr carries the
    # reason, because a check that cannot say WHY it failed sends the
    # next reader guessing: this printed an empty code once and the
    # cause was unrecoverable from the log.
    code=$(curl -sSL -o /dev/null -w '%{http_code}' --max-time 120 \
                --retry 3 --retry-delay 5 --retry-all-errors -r 0-0 "$url" 2>"$err")
    rc=$?
    printf '%-14s HTTP %s  %s\n' "$flag" "${code:-none}" "$url"
    # An answer we did not like and no answer at all are different
    # findings. Upstream saying 404 means the bucket is gone, which is
    # the thing this check exists to catch, so it stays a hard failure.
    # curl never completing (killed, DNS, TLS, a CDN hiccup, code 000 or
    # empty) is OUR side of the wire failing: it is no evidence about
    # the URL, and a third-party CDN must not be able to redden a
    # portability gate on a bad minute.
    if [ "$rc" -ne 0 ] || [ -z "$code" ] || [ "$code" = "000" ]; then
        soft "$flag: no answer from the CDN, availability UNVERIFIED (curl exit $rc, code '${code:-none}'): $(tr '\n' ' ' <"$err" | head -c 200)"
        rm -f "$err"
        return 0
    fi
    rm -f "$err"
    case "$code" in 200|206) ;; *) fail "$flag -> HTTP $code" ;; esac
}

# Run a command with every libtorch-locating variable scrubbed, so what
# executes sees the machine the way a fresh user's shell does.
scrubbed() {
    env -u LIBTORCH_PATH -u LD_LIBRARY_PATH -u DYLD_LIBRARY_PATH -u LIBRARY_PATH "$@"
}

SOFT_FAIL=0
# Stop at the first hard failure so it is the last thing in the log:
# Actions folds each ::group::, so anything after an error scrolls it out
# of view. Per-host coverage is unaffected, since `fail-fast: false` on
# the matrix keeps the other hosts running.
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
# INSTALL_ADVISORY is live (windows installs but does not compile).
# COMPILE_ADVISORY is set by NO host now that macOS is green; it stays as
# the mechanism for the next host added before it has ever run, which is
# the only honest use for it. Anything left advisory once it passes stops
# being a report and becomes a warning nobody reads.
INSTALL_ADVISORY=0; COMPILE_ADVISORY=0

case "$HOST" in
    linux-x86_64)
        # What a real Linux user has. The CPU build on Linux is ci.yml's
        # every-PR job already, so this leg spends itself on the vendor
        # path instead. Vendor alternates per run so steady-state cost
        # stays at one toolkit install; deterministic, so a re-run
        # repeats the same variant rather than testing something else.
        #
        # Only the ubuntu hosts rotate. The two EL containers are pinned
        # (EL9 cuda, EL10 rocm) because each release can install exactly
        # one of the vendors, so between them the dnf spelling of the
        # toolkit advice gets exercised for BOTH on every run -- see the
        # repo-setup block below for which half blocks where. `rotate-alt`
        # stays supported for dispatch even though nothing schedules it.
        # `cpu` likewise: the cheapest full pass (no toolkit phase, and
        # with it no sudo -- what a root container needs), installing the
        # cpu variant through fdl.
        GPU=1; COMPILE=1; ALL_VARIANTS=1
        VARIANT="${FDL_CI_VARIANT:-}"
        if [ -z "$VARIANT" ] || [ "$VARIANT" = rotate ] || [ "$VARIANT" = rotate-alt ]; then
            PHASE=0; [ "$VARIANT" = rotate-alt ] && PHASE=1
            case $(( (${GITHUB_RUN_NUMBER:-0} + PHASE) % 2 )) in
                0) VARIANT=cuda ;;
                1) VARIANT=rocm ;;
            esac
        fi
        case "$VARIANT" in
            cuda) LT_FLAG="--cuda 12.8"; LT_DIR=cu128  ;;
            rocm) LT_FLAG="--rocm 7.0";  LT_DIR=rocm70 ;;
            cpu)  LT_FLAG="--cpu";       LT_DIR=cpu; GPU=0 ;;
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
        # below asserts.
        #
        # No longer advisory. It went green on 2026-08-07 -- clang builds
        # the shim, and a scaffolded project trains -- so it is a gate
        # like every other host from here. It was advisory while nothing
        # had ever run, and staying advisory after that is how the two
        # defects it was carrying stayed invisible: the binary could not
        # find libtorch (no rpath, and macOS ignores LD_LIBRARY_PATH),
        # and upstream's own dylibs asked for a bundled libomp by
        # absolute Homebrew path. Both were reported as a warning nobody
        # had to act on.
        LT_FLAG="--cpu"; LT_DIR=cpu; LT_LIB="lib/libtorch.dylib"
        COMPILE=1
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

# The cargo feature the vendor phases build with, derived from the
# variant directory ONCE -- LT_DIR is final above and two phases need
# this. Matched on the `rocm` PREFIX rather than a literal, so adding a
# rocm71 variant cannot silently build `--features cuda`.
#
# A plain case, deliberately not `FEATURE=$(case ...)`: bash 3.2 (what
# macOS ships, and the oldest shell in this matrix) cannot parse a case
# inside a command substitution -- the `)` of the first pattern closes
# the `$(`, and the leg dies with "syntax error near unexpected token
# `;;'" before running anything.
case "$LT_DIR" in
    rocm*) FEATURE=rocm ;;
    *)     FEATURE=cuda ;;
esac

find_fdl

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
find_fdl
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
if [ "$ALL_VARIANTS" = 1 ]; then
group "Every variant URL is live"
# The URL grammar differs per OS (macOS has its own filename, Windows
# carries a `-win-` infix). The vendor rotation installs one variant per
# run; checking the whole list here keeps the others from rotting
# silently upstream, and it covers this run's own flag too.
for FLAG in "--cpu" "--cuda 12.6" "--cuda 12.8" "--rocm 7.0" "--rocm 7.1"; do
    url_live "$FLAG"
done
pass "every variant URL is live on $HOST"
endgroup
elif [ -n "$LT_FLAG" ]; then
group "libtorch URL resolves and is live"
url_live "$LT_FLAG"
pass "$HOST URL is live"
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

OUT=$(cargo build -p flodl-sys --features "$FEATURE" 2>&1) && RC=0 || RC=$?
if [ "$RC" -eq 0 ]; then
    note "toolkit already present; skipping the missing-case assertion"
else
    # Assert the contract rather than the wording: the build script
    # identified itself and handed back a command. Matching a sentence
    # would turn any rewording of build.rs's message into a failure.
    echo "$OUT" | tail -25
    if has "$OUT" 'flodl-sys:' && has "$OUT" 'apt install'; then
        pass "build.rs named the missing toolkit instead of dying in the C++ compile"
    else
        # Printed before failing: `fail` exits, so anything after it
        # would be dead code.
        fail "$HOST: a missing toolkit must produce a message naming packages to install"
    fi
fi

if [ "${FDL_CI_SKIP_INSTALL:-}" = 1 ]; then
    note "FDL_CI_SKIP_INSTALL=1 -- not installing the toolkit"
    # Nothing downstream can build without it. Skipping a prerequisite
    # on purpose is not a failure; only CI leaves this unset.
    [ "$RC" -ne 0 ] && COMPILE=0
elif [ "$RC" -ne 0 ]; then
    # The package list is parsed from the message build.rs just printed,
    # not written here: the phase tests that the advice a user is given
    # works. It also keeps this from drifting, since the list already
    # exists in flodl-sys/build.rs and flodl-cli's util/requirements.rs.
    # build.rs prints one install line per distro family; parse the one
    # this host's package manager can execute.
    if command -v dnf >/dev/null 2>&1; then
        PKG_MGR=dnf
        PKGS=$(printf '%s\n' "$OUT" | sed -n 's/.*sudo dnf install //p' | head -1)
    else
        PKG_MGR=apt
        PKGS=$(printf '%s\n' "$OUT" | sed -n 's/.*sudo apt install //p' | head -1)
    fi
    # build.rs cannot know which CUDA release is wanted, so its message
    # carries <M>-<m> placeholders. The version this run installed is the
    # one thing the script legitimately knows and the tool does not.
    PKGS=$(printf '%s' "$PKGS" | sed 's/<M>-<m>/12-8/g')
    [ -n "$PKGS" ] || fail "$HOST: no $PKG_MGR package list in build.rs's message"
    note "installing exactly what fdl asked for ($PKG_MGR): $PKGS"
    # Root (the rocky container) has no sudo and needs none.
    SUDO=sudo; [ "$(id -u)" = 0 ] && SUDO=""
    if [ "$PKG_MGR" = dnf ]; then
        # RHEL-family repo setup, and the vendor repos do NOT track the
        # EL releases together -- each is pinned to what that release can
        # actually install, measured 2026-08-07:
        #
        #   EL9   cuda 12.8 from rhel9      rocm: NO (glibc 2.34 < 2.35)
        #   EL10  cuda: NO (see below)      rocm 7.0.2/7.1 from el10
        #
        # NVIDIA ships a .repo carrying its own gpgkey; AMD publishes the
        # key URL for the .repo to reference, so neither needs a separate
        # import step.
        EL_MAJOR=$(. /etc/os-release 2>/dev/null && echo "${VERSION_ID%%.*}")
        if [ "$FEATURE" = rocm ]; then
            # AMD publishes el10 only from 7.0.2 on: there is no
            # el10/7.0, so the EL10 pin is a patch newer than the
            # rocm70 archive it supplies headers for. That is fine --
            # this phase compiles the shim, and the libtorch archive
            # brings its own runtime libraries.
            case "$EL_MAJOR" in
                9|8) ROCM_REPO="rhel9/7.0" ;;
                *)   ROCM_REPO="el10/7.0.2" ;;
            esac
            $SUDO tee /etc/yum.repos.d/rocm.repo >/dev/null <<ROCMREPO
[ROCm]
name=ROCm
baseurl=https://repo.radeon.com/rocm/$ROCM_REPO/main
enabled=1
gpgcheck=1
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
ROCMREPO
        else
            # There is no CUDA 12.8 an EL10 box can install. NVIDIA's
            # rhel10 repo starts at CUDA 13 (`libcublas-devel-13-0` is
            # its oldest), and fdl offers no 13.x variant; pointing EL10
            # at the rhel9 repo instead dies in the GPG import, because
            # EL10's rpm uses rpm-sequoia and its policy rejects that
            # key outright ("No binding signature at time ...") where
            # EL9's legacy parser accepts it. Neither half heals on its
            # own, so say which one blocked rather than failing in a key
            # import 2.6 GB later.
            if [ "${EL_MAJOR:-9}" -ge 10 ] 2>/dev/null; then
                fail "$HOST: no CUDA $LT_DIR toolkit is installable on EL$EL_MAJOR (nvidia's rhel10 repo starts at CUDA 13; the rhel9 key is rejected by rpm-sequoia here). Give this leg the rocm variant."
            fi
            $SUDO dnf install -y -q dnf-plugins-core
            $SUDO dnf config-manager --add-repo \
                https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
        fi
        # Weak deps stay on for the same reason recommends do below: the
        # devel packages pull their runtimes, and dodging that would
        # validate a configuration no user has.
        # shellcheck disable=SC2086
        $SUDO dnf install -y $PKGS || fail "$HOST: toolkit install failed: $PKGS"
    else
        if [ "$FEATURE" = rocm ]; then
            $SUDO mkdir -p --mode=0755 /etc/apt/keyrings
            curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key \
                | $SUDO gpg --dearmor -o /etc/apt/keyrings/rocm.gpg
            echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/7.0 noble main" \
                | $SUDO tee /etc/apt/sources.list.d/rocm.list >/dev/null
            $SUDO apt-get update -qq
        else
            curl -fsSLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
            $SUDO dpkg -i cuda-keyring_1.1-1_all.deb
            $SUDO apt-get update -qq
        fi
        # No --no-install-recommends: these -dev packages hard-Depend on
        # their runtimes (hipblaslt alone is ~4 GB of Tensile kernels), and
        # dodging that would validate a configuration no user has.
        # shellcheck disable=SC2086
        $SUDO apt-get install -y $PKGS || fail "$HOST: toolkit install failed: $PKGS"
    fi
fi
endgroup
fi

# =====================================================================
if [ "$COMPILE" = 1 ]; then
group "Build and clippy flodl"
if [ "$GPU" = 1 ]; then
    # Unscoped on purpose: this leg installed the whole tree, so it can
    # build every member against the vendor libtorch it just fetched.
    #
    # LINK_CMD is separate, and it is the one that matters. Neither of the
    # other two reaches ld: the workspace's only `[[bin]]` is `fdl`, which
    # is zero-dep on flodl by policy, so BUILD_CMD emits exactly one
    # executable referencing no libtorch; clippy runs in check mode
    # (--emit=metadata) whatever `--all-targets` selects. This comment
    # used to claim the workspace build "links for real" -- measured, it
    # produces 1 executable and links no libtorch at all.
    #
    # One integration-test binary, not `--tests`: it must be a SEPARATE
    # test binary (flodl has no tests/ dir, so its lib test is the case
    # the force-load invariant says keeps passing), and one is enough for
    # a link check while 60-odd would add ~8 GB beside a libtorch that is
    # 11 GB on the rocm rotation. Linked, never run -- no GPU here.
    BUILD_CMD="cargo build --features $FEATURE"
    CLIPPY_CMD="cargo clippy --features $FEATURE --all-targets -- -W clippy::all"
    LINK_CMD="cargo build --features $FEATURE -p flodl-hf --test bert_cuda_smoke"
else
    BUILD_CMD="cargo build -p flodl-sys -p flodl"
    CLIPPY_CMD="cargo clippy -p flodl-sys -p flodl --all-targets -- -W clippy::all"
    # No separate link command on the CPU legs: the two commands above
    # stop at rlib and at metadata, and the OS-specific link + load +
    # run now happens in the scaffold smoke below, which builds a real
    # binary against libtorch and trains it. Windows stays the gap --
    # COMPILE=0 there, so neither this phase nor the smoke runs.
    LINK_CMD=":"
fi

OK=1
$BUILD_CMD || OK=0
[ "$OK" = 1 ] && { $CLIPPY_CMD || OK=0; }
[ "$OK" = 1 ] && { $LINK_CMD || OK=0; }

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
if [ "$COMPILE" = 1 ] && [ -n "$LT_DIR" ] && [ -d "libtorch/precompiled/$LT_DIR" ]; then
group "Scaffold smoke: fdl init --native -> build -> run"
# A user's real first session, end to end: scaffold a project, build it,
# train the template -- against THIS checkout's flodl, not the registry.
#
# Three regression guards live here and nowhere else:
#   - the dependency line must be a registry pin, never the floating git
#     fallback (the crates.io probe once sent no User-Agent, got policy-
#     rejected, and EVERY scaffold silently carried the git dep);
#   - the scaffold's printed next steps must be true without hand
#     exports: native fdl commands fill LIBTORCH_PATH / LD_LIBRARY_PATH
#     from the project's active variant, so the runs go through
#     `scrubbed` (a dev box leaks LIBTORCH_PATH from .bashrc, and a
#     build that links through the leak validates nothing);
#   - the template trains on CPU.
#
# The libtorch symlink stands in for the scaffold's own
# `./fdl libtorch download` -- same .active resolution through the
# project root, no second multi-GB download. The vendor variants carry
# the CPU libraries too, so the no-feature build links on any rotation.
FDL_ABS="$PWD/$FDL"
SCAF_ROOT=$(mktemp -d)
SCAF="$SCAF_ROOT/fdl-ci-scaffold"
SCAF_OK=1
# Which step failed, so the advisory can name the real cause. It said
# "flodl has never compiled on this host" for every failure, which was
# read off the macOS log as a compile gap while `PASS: built and linted
# flodl` sat 70 lines above it -- the binary compiles there and does not
# LOAD. An advisory naming the wrong cause is worse than one saying it
# does not know.
SCAF_STEP=""
(cd "$SCAF_ROOT" && "$FDL_ABS" init fdl-ci-scaffold --native < /dev/null) \
    || { SCAF_OK=0; SCAF_STEP="fdl init"; }

if [ "$SCAF_OK" = 1 ]; then
    if grep -Eq '^flodl = "[0-9]' "$SCAF/Cargo.toml"; then
        pass "scaffold dependency is a registry pin: $(grep '^flodl' "$SCAF/Cargo.toml")"
    else
        SCAF_OK=0; SCAF_STEP="registry-pin check"
        echo "dep line is not a registry pin: $(grep '^flodl' "$SCAF/Cargo.toml" || echo '<missing>')"
    fi
    if grep -q 'git *=' "$SCAF/Cargo.toml"; then
        SCAF_OK=0; SCAF_STEP="registry-pin check"; echo "scaffold fell back to a git dependency"
    fi
fi

if [ "$SCAF_OK" = 1 ]; then
    # sed -i.bak: the one -i spelling GNU and BSD sed agree on.
    sed -i.bak 's|^flodl = ".*"$|flodl = { path = "'"$PWD"'/flodl" }|' "$SCAF/Cargo.toml"
    ln -s "$PWD/libtorch" "$SCAF/libtorch"
    (cd "$SCAF" && scrubbed "$FDL_ABS" build) || { SCAF_OK=0; SCAF_STEP="fdl build"; }
fi
# A variant whose baseline C library is newer than this distribution's
# will build and link and then refuse to start. That is a real platform
# limit with no fix on our side (RHEL 9 ships glibc 2.34 and cannot go
# further, while the rocm7.0 archive wants 2.35), so the honest test is
# that fdl SAYS SO rather than that the binary runs.
UNMET=$("$FDL_ABS" probe --skip-mount 2>&1 | grep -c "cannot load on this host" || true)
if [ "$SCAF_OK" = 1 ] && [ "$UNMET" != 0 ]; then
    note "$HOST: the active libtorch cannot load here (older C library than the archive wants) — fdl reported it, so the run step is skipped rather than asserting the impossible"
    "$FDL_ABS" probe --skip-mount 2>&1 | grep -A 2 "cannot load on this host" | sed 's/^/    /'
elif [ "$SCAF_OK" = 1 ]; then
    (cd "$SCAF" && scrubbed "$FDL_ABS" run) || { SCAF_OK=0; SCAF_STEP="fdl run (it BUILT; this is a load/runtime failure)"; }
fi
rm -rf "$SCAF_ROOT"

if [ "$SCAF_OK" = 1 ]; then
    pass "$HOST scaffolded, built and trained a native project"
elif [ "$COMPILE_ADVISORY" = 1 ]; then
    soft "$HOST: scaffold smoke failed at ${SCAF_STEP:-an unrecorded step}; this host is advisory until it is green"
else
    fail "$HOST: scaffold smoke failed at ${SCAF_STEP:-an unrecorded step} (init -> build -> run)"
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
