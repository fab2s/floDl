#!/bin/bash
# Overshoot x EASGD measurement, over the WALK-IN topology (2026-08-14).
#
# NOT a benchmark and not a release gate. It answers one question:
# during the CPU-averaging window, does letting the fast rank run ahead
# (`--max-overshoot`) recover the idle it otherwise spends at the
# barrier, and does that recovered compute actually reach the model?
#
#   ./overshoot-sweep.sh                    # stage 1, free cadence (5 cells)
#   STAGE=2 ./overshoot-sweep.sh            # + the alpha=0.25 pair
#   PIN=320 STAGE=2 ./overshoot-sweep.sh    # stage 1b: cadence pinned (7 cells)
#   PIN=320 STAGE=3 ./overshoot-sweep.sh    # + the depth probe (N=200, 400)
#
# Env: SWEEP_OUT (defaults to runs/overshoot-easgd, or runs/overshoot-pinned
#      when PIN is set), STAGE (1, 2 or 3), PIN (fixed anchor, see below),
#      FARM (default cluster-join), OVS_WALKINS (REQUIRED, see below).
#
# ── WHY WALK-IN AND NOT `fdl @cluster` FAN-OUT ─────────────────────────
# The AMD rental topology is locked (2026-08-06): controller at home
# behind an opened port, droplet walks in. A roster fan-out is a
# different code path — the controller configures every host and pushes
# a binary — so measuring on it would measure something the rental will
# not run. This sweep therefore uses the same shape as the rental: a
# controller that opens a join window, boxes that dial in, `fdl start`
# to fire the freeze. `ci/rig-ladder.sh 4` is the pre-flight that proves
# the plumbing on this box; run it once before the first sweep, because
# a rung-4 failure and a sweep failure look identical from here.
#
# ── THE HAZARD THIS SCRIPT EXISTS TO ELIMINATE ─────────────────────────
# In a walk-in cohort the training args travel by TWO roads: the
# controller's own invocation, and every walk-in's `-- <args>` (rank
# children re-enter the binary with them, which is why `fdl join`'s help
# says they must match the run). An arm here differs ONLY in
# --max-overshoot / --easgd-alpha, so a cell whose walk-ins carry the
# previous arm's copy trains the previous arm's hyperparameters and
# reports a clean run. Five arms silently identical would read as
# "overshoot has no effect", which is the exact wrong conclusion and
# leaves no trace to find it by.
# Two defences, both structural:
#   1. the arm's arg string is built ONCE and appended to the controller
#      invocation AND to every dial command, so they cannot diverge;
#   2. the cell is REJECTED unless the harness's own effective-config
#      echo agrees with the arm's declared knobs.
# OVS_WALKINS entries therefore stop BEFORE the `--`; this script adds
# it. An entry that carries its own `--` is refused rather than merged.
#
# OVS_WALKINS example for this rig (one dial per box, newline-separated;
# exa's GPU over loopback, pascal through the join sshd):
#
#   export OVS_WALKINS="docker exec -w /workspace rdl-cuda-rank-1 fdl join 127.0.0.1:1337 --token \$TOK --host exa-cuda --devices 0 --bin /workspace/target/cluster/exa-cuda/precompiled-cu128/release/ddp-bench
#   ssh -o BatchMode=yes flodl-pascal 'cd /mnt/rdl && FDL_LIBTORCH_CASE=pascal fdl join 127.0.0.1:1337 --ssh ubuntu@192.168.122.1:2222 --identity ~/.ssh/flodl-join --token '\$TOK' --bin /mnt/rdl/target/cluster/flodl-pascal/builds-sm61-sm120/release/ddp-bench'"
#
# That pair is what `ci/rig-ladder.sh 4` was proven with on 2026-08-14,
# down to the port and the key: the door is :2222, and the key pascal needs
# is THIS farm's (~/.ssh/flodl-join, landed from .fdl/<farm>/keys/), not
# another farm's. A refused dial prints "Permission denied (publickey)";
# a WORKING one prints "This account is currently not available", which is
# nologin running as the forced command and means the guardrail did its
# job. pascal keeps cwd /mnt/rdl because FDL_LIBTORCH_CASE resolves the
# variant from that checkout's libtorch/.active.pascal.
#
# $TOK is exported from the farm overlay's `token:` before the dials run.
# For the AMD leg the droplet's entry uses --source instead of --bin (no
# shared mount), which changes nothing here: the args still append.
#
# ── WHY --output IS THE ONE ARG THAT DOES *NOT* TRAVEL ─────────────────
# It is per-side on purpose, which is the exact opposite of the rule above
# and for a reason worth stating. A relative --output resolves against
# wherever each dial left the agent, and the two boxes punish that
# differently: the exa walk-in goes through `docker exec` on cuda-rank,
# compose's ONE deliberate non-`user:` service (its sshd must start as
# root), so a relative path lands ROOT-OWNED inside the repo bind mount;
# pascal's /mnt/rdl is read-only, so the same path simply fails. One line,
# two hazards, both observed on 2026-08-14.
# Rung 4 also showed the divergence is harmless: its controller wrote
# `runs/` while its walk-ins wrote an absolute /tmp path, and the cohort
# finished clean. --output is an ARTIFACT path, not training semantics,
# and rank telemetry ships to the controller regardless. So the identity
# guarantee covers CORE_ARGS (model, mode, hyperparameters -- the arm) and
# deliberately stops short of the output dir.
#
# ── WHAT MOTIVATES THE ARMS ────────────────────────────────────────────
# Measured on runs/olmo-graph/cpu-async-diloco (the 2026-08-13 baseline,
# 24 reduces over 1830s): the sync window is 16.85% of wall at 12.85s a
# piece, and rank 0 sits at 43.0% mean GPU util inside it against 93.1%
# outside, hard-idle (<5%) for 32.7% of in-sync samples. The Pascals
# barely dip. The fast rank is the one that stalls, which is what the
# overshoot mechanism predicts.
#
# The budget was pinned at its ceiling and is an order of magnitude too
# small: steady k_max was 634 = rank 0's planned 619 plus exactly the
# `overshoot_ceiling` of 15. At rank 0's 8.06 batches/s those 15 buy
# 1.86s of cover against a 12.85s window, so ~104 is the derived full
# cover. Hence N in {0, 104}.
#
# EASGD is the other half, and it is why util alone cannot decide this.
# `load_averaged` blends W_local := (1-a)*W_local + a*W_avg, and the
# snapshot ships at RequestParams, so every overshoot batch lands in
# W_local AFTER it. The overshoot delta therefore survives at weight
# (1-a): at a=1.0 the blend degenerates to full overwrite and every
# overshoot batch is DISCARDED at writeback while util looks exactly as
# good. That makes a=1.0 the control that separates real throughput from
# util theater, and it is the reason quality, not util, is the verdict.
#
# ── TWO MECHANISM FACTS THAT FIX THE ARM SET ───────────────────────────
# 1. The guard floor keys on `easgd_alpha.is_some()`, NOT on its value
#    (`default_trend_threshold`: 0.3 when set, 0.05 when not). Sweeping
#    alpha VALUES is therefore confound-free, but an alpha=None arm would
#    move the guard 6x and confound everything it touches. There is no
#    such arm here on purpose; add one only with --divergence-threshold
#    pinned.
# 2. An explicit --max-overshoot N maps to `overshoot(n, n, false)`:
#    pinned from window 1, no ramp, and `overshoot_auto=false` disarms
#    the guard's NudgeDown reset, leaving only the anchor responsive. The
#    absent-flag default instead RAMPS 3 -> 15, so it is a trajectory
#    rather than a level. Arm A carries it as "what ships today"; it is
#    not a substitute for a pinned 15.
#
# --epoch-splits is deliberately FIXED at 20 across every arm. Changing
# it changes how often the cohort reduces, which is a different axis
# (less sync, rather than better-covered sync) and would confound both
# the util and the divergence readings.
#
# ── PRE-REGISTERED PREDICTIONS (recorded here so they can fail) ────────
#   - in-sync util and wall: MAIN EFFECT OF N ONLY. alpha touches
#     writeback, not dispatch. If util moves with alpha, the model of the
#     mechanism is wrong and the rest of the reading is void.
#   - eval CE and d_raw: N x alpha INTERACTION. CE improves with N at
#     a=0.5 and NOT at a=1.0; d_raw rises with N and with (1-a), and
#     compounds across windows only when a < 1.
#   - decisive contrast: (B1-B0) against (C1-C0). Equal throughput gain
#     in both with a quality gain only in the first is the signature that
#     overshoot buys cycles and EASGD converts them into progress.
#
# Headroom: the baseline peaks at d_raw 0.101 against the 0.3 guard
# floor, so there is ~3x of room before the guard arms. d_raw is a
# PRIMARY measured quantity here, not just a guardrail.
#
# CLAIM DISCIPLINE: util, wall and d_raw are well powered at one seed per
# arm (24 internal sync replicates each). Eval CE is NOT. Do not call a
# quality winner off this sweep; replicate the leading arm across seeds
# first.
#
# Resume-safe: a cell whose training.log carries both an `epoch ` line
# and the `# total:` footer is skipped. Same predicate as sweep.sh, and
# for the same reason: the footer alone is written at teardown even by a
# cell that trained nothing.
set -u
cd "$(dirname "$0")/.."

# ── PIN=<N>: FIXED CADENCE (stage 1b), the fix for the confound the first
# unpinned pass exposed ────────────────────────────────────────────────
# Raising overshoot does not merely lengthen each window, it MULTIPLIES the
# reduce COUNT: measured 2026-08-15, `--max-overshoot 104` took the run from
# 24/25 reduces to 64 (alpha 0.5) and 58 (alpha 1.0), because pinned
# overshoot leaves no idle pressure for ElChe's anchor growth to answer, so
# the anchor stalls and the overshoot becomes most of the window. Averaging
# FREQUENCY is itself a convergence variable, so every unpinned arm compares
# regimes rather than knobs, and **seeds cannot fix that** — it is
# structural, not noise. (The one unconfounded pair in the first pass was
# B0 vs C0: both at overshoot 0, so neither collapsed its window, 25 reduces
# and k_used 977 apiece, isolating alpha alone.)
# PIN fixes the window geometry so overshoot is the sole variable and eval
# CE becomes attributable. `--min-anchor N --max-anchor N --guard none` is
# the documented fixed-k probe: NudgeDown is the only path that bypasses
# min_anchor, so disabling the guard is what makes the pin hard. It costs
# nothing here — measured d_peak was 0.034 to 0.146 against a 0.3 floor, so
# the guard never fired in any arm — and it independently removes the growth
# gating, one of the two candidate causes of the anchor stall.
# 320 is arm A's converged anchor (it produced k_used 1022 over 24 reduces).
# Second thing the pin buys, and the one that decides the AMD leg: whether
# the 13.6% wall win SURVIVES at 24 reduces instead of 64, since the 2.7x
# wire traffic is the part a metered link may not afford.
PIN=${PIN:-}
if [ -n "$PIN" ]; then
  DEFAULT_OUT=runs/overshoot-pinned
else
  DEFAULT_OUT=runs/overshoot-easgd
fi

OUT=${SWEEP_OUT:-$DEFAULT_OUT}
ABS_OUT=ddp-bench/$OUT
STAGE=${STAGE:-1}
FARM=${FARM:-cluster-join}
LOGDIR=${OVS_LOGDIR:-target/overshoot-sweep}
mkdir -p "$ABS_OUT" "$LOGDIR"

MODEL=olmo-graph
# Every arm runs cpu-async + diloco, so the harness writes each cell to
# <arm>/olmo-graph/cpu-async-diloco/. The ARM is the outer directory
# because the model and mode are constant here; only N and alpha move.
CELL=$MODEL/cpu-async-diloco
# --save-dashboard is timing-safe and therefore safe here even though wall
# is a measured variable: the cluster path builds its dashboard sink
# unconditionally, so the flag adds only the run-scoped cards (built
# before the timer starts) and one file write at teardown. Same reasoning
# sweep.sh records for the published numbers. --reports-per-epoch stays
# OFF for the opposite reason: it emits on the hot path at the reduce
# boundary, which is precisely the window being measured.
FIXED="--model $MODEL --mode cpu-async --outer-optimizer diloco --train-tokens 20M --bf16-wire --epochs 1 --epoch-splits 20 --seed 42 --save-dashboard"
if [ -n "$PIN" ]; then
  FIXED="$FIXED --min-anchor $PIN --max-anchor $PIN --guard none"
fi

# label:max_overshoot:easgd_alpha — "auto" and "default" mean OMIT the
# flag, which is not the same as passing its default value (see fact 2
# above for max_overshoot; for alpha the omitted value is the CpuAsync
# mode default of 0.5, despite what --easgd-alpha's own help claims).
STAGE1_ARMS="A:auto:default B0:0:0.5 B1:104:0.5 C0:0:1.0 C1:104:1.0"
# Lower alpha retains more of the overshoot, so this pair is where the
# (1-a) scaling gets TESTED rather than assumed. Conditional on stage 1
# showing the interaction AND d_raw still having room under 0.3.
STAGE2_ARMS="D0:0:0.25 D1:104:0.25"

# DEPTH probe (stage 3): everything measured so far sits BELOW full cover,
# so "too deep is waste" is theoretically sound and empirically untouched.
# At pinned geometry rank0 runs ~107 ms/batch against a 13.8s sync, so full
# cover is ~128 batches and 104 was only 81% of it. 200 and 400 are ~1.6x
# and ~3.1x cover, bracketing the knee.
# Two DISTINCT bounds are being separated here, and they need not coincide:
#   HARDWARE  - past full cover there is no idle left to fill, so wall
#               should FLATTEN somewhere near 130 and buy nothing after.
#   STATISTICAL - the pseudo-gradient's usefulness decays with local steps,
#               because a trajectory from stale params follows curvature
#               that drifts from the consensus point. Nothing to do with
#               the sync window, and on a slow link it can bind FIRST.
# Run against the pinned dir so B0 (N=0) and B1 (N=104) are reused as the
# lower half of the curve rather than re-run.
# NOTE this probe runs unguarded like the rest of the pinned sweep, and that
# is deliberate: the point is to see the divergence curve with nothing
# braking it. d_peak was 0.108 at N=104, so N=400 could approach the 0.3
# floor. A cell that DIVERGES still exits 0 and still prints `done:`, so
# read eval_CE and d_peak, not the OK line.
DEPTH_ARMS="E1:200:0.5 E2:400:0.5"

arms_for_stage() {
  case "$1" in
    3) echo "$STAGE1_ARMS $STAGE2_ARMS $DEPTH_ARMS" ;;
    2) echo "$STAGE1_ARMS $STAGE2_ARMS" ;;
    *) echo "$STAGE1_ARMS" ;;
  esac
}

# Both knobs are omitted rather than defaulted when the arm says so; the
# distinction is the whole point of arm A.
arm_flags() {
  flags=""
  if [ "$1" != "auto" ]; then flags="$flags --max-overshoot $1"; fi
  if [ "$2" != "default" ]; then flags="$flags --easgd-alpha $2"; fi
  echo "$flags"
}

ts() { date '+%F %T'; }

cell_done() {
  log="$ABS_OUT/$1/$CELL/training.log"
  grep -q '^epoch ' "$log" 2>/dev/null && grep -q '# total:' "$log" 2>/dev/null
}

strays() { pgrep -af 'release/ddp-benc[h]' >/dev/null 2>&1; }

# libtorch `[W...] Warning:` lines do NOT fail a run, and a sweep whose
# green summary hides them is worse than no sweep.
warns_in() { grep -cE '\[W[0-9]* ' "$1" 2>/dev/null || true; }

# THE ARM-IDENTITY GATE. The harness echoes its EFFECTIVE elche config,
# so the run itself can be asked which knobs it used rather than trusted
# to have received them. This is what makes a stale walk-in loud instead
# of silent: a cell whose echo disagrees with its label is rejected and
# deleted, so a resume re-runs it. Missing echo also fails — an absent
# proof is not a passed check.
config_matches() {
  # $1 log, $2 max_overshoot, $3 easgd_alpha
  line=$(grep -m1 '  elche:' "$1" 2>/dev/null)
  if [ -z "$line" ]; then echo "no elche config echo in log"; return 1; fi
  # The harness prints `max_overshoot=auto` when the flag was omitted and
  # the number otherwise, so the arm's own encoding compares directly.
  # The trailing space is load-bearing: an unanchored substring test
  # matches `max_overshoot=104` against a wanted `10` and passes the
  # wrong cell. `meta_controller=` always follows, so the space is there.
  want_o=$2
  case "$line" in
    *"max_overshoot=$want_o "*) ;;
    *) echo "effective max_overshoot != $want_o | $line"; return 1 ;;
  esac
  want_a=$3
  if [ "$want_a" = "default" ]; then want_a="0.5"; fi
  case "$line" in
    *"easgd_alpha=Some($want_a)"*) ;;
    *) echo "effective easgd_alpha != $want_a | $line"; return 1 ;;
  esac
  return 0
}

# Written from the SUCCESS path only, so provenance exists if and only if
# the cell holds results. The arm's own N and alpha go in explicitly:
# they are the entire independent variable, and a run directory that
# cannot say which arm it is cannot be read six months from now.
stamp() {
  d="$ABS_OUT/$1/$CELL"; mkdir -p "$d"
  { echo "arm:            $1"
    echo "max_overshoot:  $2"
    echo "easgd_alpha:    $3"
    echo "topology:       walk-in (farm=$FARM), rental-parity"
    echo "invocation:     $4"
    echo "walkins:        $5 box(es)"
    echo "git_sha:        $(git rev-parse HEAD 2>/dev/null)"
    echo "git_dirty:      $(git status --porcelain 2>/dev/null | wc -l) file(s) modified"
    echo "utc:            $(date -u '+%F %T UTC')"
    echo "host:           $(hostname)"
    echo "libtorch:       $(cat libtorch/.active 2>/dev/null || echo unknown)"
  } > "$d/provenance.txt"
}

if [ ! -f "fdl.$FARM.yml" ]; then
  echo "$(ts) ABORT: farm overlay fdl.$FARM.yml not found (it is user-local; create it from its .example)"
  exit 1
fi
if [ -z "${OVS_WALKINS:-}" ]; then
  echo "$(ts) ABORT: set OVS_WALKINS to one dial-in command per box (newline-separated, WITHOUT the trailing '-- <args>')"
  exit 1
fi
# Exported so a dial template can reference it, matching rig-ladder's
# convention. Read from the overlay rather than asked for: the token is
# the farm's, and a second copy in the environment is a second thing to
# get wrong.
TOK=$(sed -n "s/^ *token: *//p" "fdl.$FARM.yml" | head -1 | tr -d "\"' \r")
export TOK
if [ -z "$TOK" ]; then
  echo "$(ts) NOTE: no token: in fdl.$FARM.yml — dials must carry their own credential"
fi

ARMS=$(arms_for_stage "$STAGE")
echo "$(ts) OVERSHOOT SWEEP BEGIN rev=$(git rev-parse --short HEAD) stage=$STAGE farm=$FARM out=$OUT"
echo "$(ts) fixed: $FIXED"
echo "$(ts) arms: $ARMS"

for arm in $ARMS; do
  label=${arm%%:*}
  rest=${arm#*:}
  n=${rest%%:*}
  alpha=${rest##*:}

  if cell_done "$label"; then echo "$(ts) SKIP $label (done)"; continue; fi
  echo "$(ts) START $label (max_overshoot=$n easgd_alpha=$alpha)"

  # ONE core arg string, used by the controller AND appended to every dial.
  # Divergence there is the hazard this whole script is shaped around, so
  # it is never written twice. --output is added per side (see the header):
  # the controller's is the sweep's cell, each walk-in's is absolute and
  # node-local so it can be neither root-owned repo pollution nor a write
  # into a read-only mount.
  CORE_ARGS="$FIXED$(arm_flags "$n" "$alpha")"
  WALKIN_OUT="/tmp/ovs-$label"
  clog="$LOGDIR/$label-controller.log"

  # shellcheck disable=SC2086
  ./fdl "@$FARM" ddp-bench $CORE_ARGS --output "$OUT/$label" > "$clog" 2>&1 &
  cpid=$!

  # The window has to be open before anyone dials, or the dial is refused
  # and the cell fails for a reason that is not the one under test.
  waited=0
  until grep -aq "join: window open" "$clog" 2>/dev/null; do
    sleep 2; waited=$((waited+2))
    if [ "$waited" -ge 120 ] || ! kill -0 "$cpid" 2>/dev/null; then
      echo "$(ts) FAIL $label: controller never opened a join window"
      tail -15 "$clog" | sed 's/^/    /'
      kill "$cpid" 2>/dev/null; wait "$cpid" 2>/dev/null
      continue 2
    fi
  done

  wpids=""; wlogs=""; nw=0
  while IFS= read -r cmd; do
    [ -n "$cmd" ] || continue
    case "$cmd" in
      *" -- "*)
        echo "$(ts) FAIL $label: an OVS_WALKINS entry carries its own '--'; args are appended by this script so the arm cannot drift"
        # shellcheck disable=SC2086
        kill "$cpid" $wpids 2>/dev/null; wait 2>/dev/null
        continue 2 ;;
    esac
    nw=$((nw+1))
    wlog="$LOGDIR/$label-walkin-$nw.log"
    sh -c "$cmd -- $CORE_ARGS --output $WALKIN_OUT" > "$wlog" 2>&1 &
    wpids="$wpids $!"
    wlogs="$wlogs $wlog"
  done <<EOF
$OVS_WALKINS
EOF
  echo "$(ts) $label: $nw box(es) dialing in"

  # Quorum lives in the overlay's min_rank_start and counts RANKS, not
  # boxes: on this rig 3 = exa's one GPU plus pascal's two, the same
  # 3-rank cohort the fan-out baseline measured.
  waited=0
  until ./fdl "@$FARM" status 2>/dev/null | grep -q "roster startable"; do
    sleep 3; waited=$((waited+3))
    if [ "$waited" -ge 300 ]; then
      echo "$(ts) FAIL $label: quorum never reached"
      ./fdl "@$FARM" status 2>&1 | sed 's/^/    /' | tail -8
      # shellcheck disable=SC2086
      kill "$cpid" $wpids 2>/dev/null; wait 2>/dev/null
      continue 2
    fi
  done
  ./fdl "@$FARM" start >> "$clog" 2>&1

  wait "$cpid"; rc=$?
  # shellcheck disable=SC2086
  wait $wpids 2>/dev/null

  degraded=$(grep -c "finished DEGRADED\|child exit(s) tolerated\|device-side assert" "$clog")
  # Warnings are counted across the AGENT logs as well as the controller's.
  # Under fan-out the remote rank output streams back to the controller, so
  # one log held everything; a walk-in agent instead owns its own stdout, so
  # a rank-side libtorch `[W...]` never reaches $clog. Counting only the
  # controller here would be a warning check that structurally cannot see
  # the ranks, which is the failure mode it exists to prevent.
  warns=$(warns_in "$clog")
  mkdir -p "$ABS_OUT/$label/$CELL"
  grep -m1 '  elche:' "$clog" > "$ABS_OUT/$label/$CELL/elche-config.txt" 2>/dev/null
  cp "$clog" "$ABS_OUT/$label/$CELL/controller.log" 2>/dev/null

  # Each agent's own exit matters as much as the controller's: one that
  # died still lets the controller finish with the ranks it had, and a
  # cohort that lost a rank is not the allocation being measured.
  agents_ok=1
  for wlog in $wlogs; do
    if ! grep -aq "finished cleanly" "$wlog"; then
      echo "$(ts) $label: walk-in agent did not finish cleanly ($wlog)"
      tail -12 "$wlog" | sed 's/^/    /'
      agents_ok=0
    fi
    wwarn=$(warns_in "$wlog")
    if [ "$wwarn" -gt 0 ]; then
      echo "$(ts) $label: $wwarn libtorch warning(s) in $wlog"
      grep -aE '\[W[0-9]* ' "$wlog" | head -5 | sed 's/^/    /'
    fi
    warns=$((warns + wwarn))
    # Agent logs are part of the record: a cell's warning count is only
    # auditable later if the logs it was computed from travel with it.
    cp "$wlog" "$ABS_OUT/$label/$CELL/$(basename "$wlog")" 2>/dev/null
  done

  if cfg_err=$(config_matches "$clog" "$n" "$alpha"); then cfg_ok=1; else cfg_ok=0; fi

  if [ $rc -eq 0 ] && [ "$degraded" -eq 0 ] && [ "$warns" -eq 0 ] \
     && [ "$agents_ok" -eq 1 ] && [ "$cfg_ok" -eq 1 ] \
     && grep -aq "done:" "$clog" && cell_done "$label"; then
    stamp "$label" "$n" "$alpha" "fdl @$FARM ddp-bench $CORE_ARGS --output $OUT/$label" "$nw"
    echo "$(ts) OK $label"
  else
    echo "$(ts) FAIL $label rc=$rc degraded=$degraded libtorch_warnings=$warns agents_ok=$agents_ok arm_identity=$cfg_ok"
    [ "$cfg_ok" -eq 1 ] || echo "$(ts)   arm identity: $cfg_err"
    rm -rf "${ABS_OUT:?}/$label"
    # The agent wrapper's kill-trap needs up to 10s to KILL its child
    # after a rank crash; probing sooner aborts on strays that clear
    # themselves. 15s separates those from a wedged launcher.
    sleep 15
    if strays; then
      echo "$(ts) ABORT-DIRTY: leftover ddp-bench processes after failed cell — manual cleanup needed:"
      pgrep -af 'release/ddp-benc[h]'
      exit 1
    fi
    echo "$(ts) CONTINUE (strays cleared; cell will retry on the next sweep pass)"
  fi
  # Settle window between cells: rank-0 SIGSEGV at NCCL/CUDA init was
  # observed on cells starting within ~1s of the previous teardown.
  sleep 10
done

echo "$(ts) ARMS DONE"

# ── Analysis ───────────────────────────────────────────────────────────
# The verdict needs in-sync vs outside GPU util per rank, which nothing
# else in the tree computes: it means intersecting each rank_samples
# entry against the sync_start/sync_end spans in the same timeline.json.
# Kept next to the recipe so the numbers stay reproducible from the
# artifacts alone.
if command -v python3 >/dev/null 2>&1; then
  python3 ddp-bench/scripts/overshoot_report.py "$ABS_OUT" | tee "$ABS_OUT/report.txt"
else
  echo "$(ts) NOTE: python3 not found; run scripts/overshoot_report.py by hand for the table"
fi

echo "$(ts) OVERSHOOT SWEEP COMPLETE"
