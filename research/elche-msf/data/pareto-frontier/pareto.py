"""Pareto frontier aggregator — cross-sweep eval-vs-syncs characterization.

The publishable result is the empirical eval-vs-cost Pareto frontier across
all 200-epoch configurations on disk. This script pulls cells from three
sweep dirs:

  - passive-observation/      (default-anchor; cpu-async = EASGD α=0.5)
  - relaxed-anchor/           (relaxed-anchor on nccl-async)
  - cliff-bracket/            (fixed-k cliff)

and produces:

  - per-config aggregate (mean ± sd over seeds)
  - frontier identification (non-dominated configs on (mean_syncs, mean_eval))
  - dominated configs with the dominator listed

Cost axis: syncs / 200ep (network-volume proxy at fixed model size). Eval axis:
held-out test accuracy.

Run from project root:
    python3 research/elche-msf/data/pareto-frontier/pareto.py
"""
import os
import re
from statistics import mean, stdev

PROJECT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
RUNS_DIR = os.path.join("research", "elche-msf", "data")


# ---------------------------------------------------------------------------
# Cell registry — every per-seed run dir we want on the Pareto plot.
# Format: (config_label, seed, report_path_relative_to_project_root)
# ---------------------------------------------------------------------------

CELLS = []

# default-anchor (passive-observation): 5 seeds × 2 modes × 2 guards
for seed in range(5):
    for mode in ("nccl-async", "cpu-async"):
        for guard in ("trend", "msf"):
            CELLS.append((
                f"{mode} default {guard}",
                seed,
                f"{RUNS_DIR}/passive-observation/seed-{seed}-{mode}-{guard}/report.md",
            ))

# relaxed-anchor: nccl-async × 5 seeds × 2 guards
for seed in range(5):
    for guard in ("trend", "msf"):
        CELLS.append((
            f"nccl-async relaxed {guard}",
            seed,
            f"{RUNS_DIR}/relaxed-anchor/seed-{seed}-nccl-async-{guard}-relaxed/report.md",
        ))

# fixed-k cliff (cliff-bracket): 3 seeds × 6 k values
for seed in range(3):
    for k in (3200, 6400, 12800, 16000, 25600, 51200):
        CELLS.append((
            f"fixed k={k}",
            seed,
            f"{RUNS_DIR}/cliff-bracket/seed-{seed}-fixed-k-{k}/report.md",
        ))


# ---------------------------------------------------------------------------
# Parser — pull eval, syncs, total_s from the Per-Model Results row
# ---------------------------------------------------------------------------

def parse_main(path):
    full = os.path.join(PROJECT_ROOT, path)
    if not os.path.exists(full):
        return None
    with open(full) as f:
        text = f.read()
    m = re.search(
        r"\| (\S+) \| ([\d.]+) \| ([\d.]+) \| [+-][\d.]+ \| ([\d.]+) \| (\d+) \|",
        text,
    )
    if not m:
        return None
    return {
        "mode": m.group(1),
        "loss": float(m.group(2)),
        "eval": float(m.group(3)),
        "total_s": float(m.group(4)),
        "syncs": int(m.group(5)),
    }


# ---------------------------------------------------------------------------
# Load all cells, group by config
# ---------------------------------------------------------------------------

def msd(xs):
    n = len(xs)
    if n == 0: return None, None
    m = sum(xs) / n
    if n == 1: return m, 0.0
    return m, (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5


configs = {}
missing = []
for label, seed, path in CELLS:
    rec = parse_main(path)
    if rec is None:
        missing.append((label, seed, path))
        continue
    configs.setdefault(label, []).append((seed, rec))

print("=" * 100)
print(" PARETO FRONTIER AGGREGATION — eval vs cost (syncs/200ep)")
print("=" * 100)
print(f" Loaded {sum(len(v) for v in configs.values())} cells across {len(configs)} configurations.")
if missing:
    print(f" {len(missing)} cells missing on disk:")
    for lbl, s, p in missing[:5]:
        print(f"   {lbl}  seed={s}  {p}")
    if len(missing) > 5:
        print(f"   ... +{len(missing) - 5} more")
print()


# ---------------------------------------------------------------------------
# Per-config aggregate
# ---------------------------------------------------------------------------

per_config = {}
for label, recs in configs.items():
    evals = [r["eval"] for _, r in recs]
    syncs = [r["syncs"] for _, r in recs]
    wall = [r["total_s"] for _, r in recs]
    em, esd = msd(evals)
    sm, ssd = msd(syncs)
    wm, wsd = msd(wall)
    per_config[label] = {
        "n": len(recs),
        "eval_mean": em, "eval_sd": esd,
        "syncs_mean": sm, "syncs_sd": ssd,
        "wall_mean": wm, "wall_sd": wsd,
    }


# ---------------------------------------------------------------------------
# Pareto frontier — config A is dominated if exists B with B.eval > A.eval
# AND B.syncs < A.syncs. Strict domination on both axes.
# ---------------------------------------------------------------------------

labels_sorted = sorted(
    per_config.keys(),
    key=lambda lab: per_config[lab]["syncs_mean"],
)

frontier = []
dominated = {}
EPS_EVAL = 1e-4   # eval ties under this are ignored for domination
EPS_SYNC = 0.5    # sync ties under 0.5 are ignored

for a in labels_sorted:
    A = per_config[a]
    dominator = None
    for b in labels_sorted:
        if a == b:
            continue
        B = per_config[b]
        if (B["eval_mean"] > A["eval_mean"] + EPS_EVAL
            and B["syncs_mean"] < A["syncs_mean"] - EPS_SYNC):
            dominator = b
            break
    if dominator is None:
        frontier.append(a)
    else:
        dominated[a] = dominator


# ---------------------------------------------------------------------------
# Output: full table sorted by sync count, frontier marked
# ---------------------------------------------------------------------------

print("-" * 100)
print(" Per-config aggregate (sorted by mean sync count, ascending)")
print("-" * 100)
print(f" {'config':<32} {'n':>3}  {'eval mean ± sd':>17}  {'syncs mean ± sd':>17}  {'wall (s) mean ± sd':>20}  status")
print(" " + "-" * 99)
for label in labels_sorted:
    c = per_config[label]
    n = c["n"]
    em = c["eval_mean"] * 100
    esd = c["eval_sd"] * 100
    sm = c["syncs_mean"]
    ssd = c["syncs_sd"]
    wm = c["wall_mean"]
    wsd = c["wall_sd"]
    if label in dominated:
        status = f"dominated by {dominated[label]}"
    else:
        status = "FRONTIER" if n >= 3 else "frontier (n<3)"
    print(
        f" {label:<32} {n:>3}  {em:>6.2f}% ± {esd:>4.2f}     {sm:>8.0f} ± {ssd:>5.0f}     {wm:>8.1f} ± {wsd:>5.1f}     {status}"
    )

print()
print("-" * 100)
print(" Pareto frontier (non-dominated configs, sorted by sync count)")
print("-" * 100)
for label in labels_sorted:
    if label not in frontier:
        continue
    c = per_config[label]
    n_marker = "" if c["n"] >= 3 else f"  [n={c['n']}]"
    print(
        f"  {label:<32}  eval = {c['eval_mean']*100:>5.2f}%  syncs = {c['syncs_mean']:>6.0f}{n_marker}"
    )

print()
print("-" * 100)
print(" ASCII Pareto scatter (eval vs syncs, x is log-spaced)")
print("-" * 100)

# Simple log-x scatter, 80 cols wide, plot eval ∈ [10, 92] mapped to ~20 rows
import math
def log10_safe(x):
    return math.log10(max(x, 1))

x_min = log10_safe(min(c["syncs_mean"] for c in per_config.values()))
x_max = log10_safe(max(c["syncs_mean"] for c in per_config.values()))
y_min = 10
y_max = 92
W = 70
H = 22

grid = [[" "] * W for _ in range(H)]
for label, c in per_config.items():
    xn = (log10_safe(c["syncs_mean"]) - x_min) / (x_max - x_min)
    yn = (c["eval_mean"] * 100 - y_min) / (y_max - y_min)
    col = max(0, min(W - 1, int(xn * (W - 1))))
    row = max(0, min(H - 1, H - 1 - int(yn * (H - 1))))
    marker = "F" if label in frontier else "x"
    if grid[row][col] in (" ", "x") or marker == "F":
        grid[row][col] = marker

print(f"  {'eval %':>7}")
for r, line in enumerate(grid):
    yval = y_max - (y_max - y_min) * r / (H - 1)
    label_y = f"{yval:>6.1f}" if r % 4 == 0 else "      "
    print(f"  {label_y} | {''.join(line)}")
print(f"          +{'-' * W}")
print(f"           1{' ' * (W // 4 - 4)}10{' ' * (W // 4 - 5)}100{' ' * (W // 4 - 6)}1000   syncs/200ep (log)")
print(f"  Legend: F = on Pareto frontier;  x = dominated configuration")

print()
print("-" * 100)
print(" Writing matplotlib plot")
print("-" * 100)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    fig, ax = plt.subplots(figsize=(11, 7), dpi=120)

    # Frontier line — sort frontier configs by sync count, draw step-function
    frontier_pts = sorted(
        [(per_config[lab]["syncs_mean"], per_config[lab]["eval_mean"] * 100, lab)
         for lab in frontier],
        key=lambda t: t[0],
    )
    # The Pareto frontier in (lower-cost, higher-eval) is the upper-left envelope.
    # We trace it as a piecewise-linear curve through the eval-monotone-increasing
    # subsequence (so it doesn't dip into collapse cells).
    upper_envelope = []
    for x, y, lab in frontier_pts:
        if not upper_envelope or y > upper_envelope[-1][1]:
            upper_envelope.append((x, y, lab))
    if len(upper_envelope) > 1:
        ax.plot(
            [p[0] for p in upper_envelope],
            [p[1] for p in upper_envelope],
            color="#3d5a80", linewidth=2, alpha=0.5, zorder=1,
            label="Pareto frontier (upper envelope)",
        )

    # Scatter all configs, color-code by family
    family_color = {
        "fixed":     "#ee6c4d",   # warm — fixed-k cliff axis
        "default":   "#3d5a80",   # blue — production default
        "relaxed":   "#98c1d9",   # light blue — relaxed-anchor
    }
    def family_of(label):
        if "fixed" in label: return "fixed"
        if "relaxed" in label: return "relaxed"
        return "default"

    for label, c in per_config.items():
        fam = family_of(label)
        on_front = label in frontier
        marker = "o" if on_front else "x"
        size = 110 if on_front else 70
        edgecolor = "black" if on_front else family_color[fam]
        ax.errorbar(
            c["syncs_mean"], c["eval_mean"] * 100,
            xerr=c["syncs_sd"] if c["n"] > 1 else None,
            yerr=c["eval_sd"] * 100 if c["n"] > 1 else None,
            fmt=marker, markersize=10 if on_front else 7,
            color=family_color[fam], ecolor=family_color[fam], elinewidth=1, capsize=3,
            markeredgecolor=edgecolor, markeredgewidth=1.2 if on_front else 0.8,
            zorder=3 if on_front else 2,
        )

    # Annotate noteworthy points
    annotation_overrides = {
        "nccl-async default msf":   ("ElChe default + msf\n(production, dominated)", (45, -35)),
        "cpu-async default trend":  ("cpu-async default + trend\n(EASGD α=0.5)", (-130, 35)),
        "nccl-async relaxed trend": ("nccl-async relaxed + trend\n(frontier, lowest-sync at parity)", (-235, 8)),
        "fixed k=3200":             ("k=3200 (frontier knee)", (-30, -35)),
        "fixed k=12800":            ("k=12800", (5, 12)),
        "fixed k=25600":            ("k=25600 (cliff edge, bimodal)", (-50, 22)),
        "fixed k=51200":            ("k=51200 (collapsed)", (-30, 25)),
    }
    for label, (text, offset) in annotation_overrides.items():
        if label not in per_config:
            continue
        c = per_config[label]
        ax.annotate(
            text,
            xy=(c["syncs_mean"], c["eval_mean"] * 100),
            xytext=offset, textcoords="offset points",
            fontsize=8.5, color="#293241",
            arrowprops=dict(arrowstyle="-", color="gray", linewidth=0.6, alpha=0.6),
        )

    # Family legend
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=family_color["fixed"],
               markersize=8, label="fixed-k cadence"),
        Line2D([0], [0], marker="o", linestyle="", color=family_color["default"],
               markersize=8, label="ElChe default-anchor"),
        Line2D([0], [0], marker="o", linestyle="", color=family_color["relaxed"],
               markersize=8, label="ElChe relaxed-anchor"),
        Line2D([0], [0], marker="o", linestyle="", color="white",
               markeredgecolor="black", markeredgewidth=1.2, markersize=10,
               label="on Pareto frontier (filled)"),
        Line2D([0], [0], marker="x", linestyle="", color="gray",
               markersize=8, label="dominated"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", framealpha=0.95, fontsize=9)

    ax.set_xscale("log")
    ax.set_xlabel("Syncs per 200-epoch run (log scale, network-volume proxy)", fontsize=11)
    ax.set_ylabel("Final eval accuracy (%)", fontsize=11)
    ax.set_title(
        "Eval-vs-cost Pareto frontier for heterogeneous DDP\n"
        "ResNet-20 / CIFAR-10 / 200 epochs / 3-GPU (1×RTX 5060 Ti + 2×GTX 1060)",
        fontsize=12, pad=14,
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.set_ylim(5, 95)

    out_path = os.path.join(
        PROJECT_ROOT, RUNS_DIR, "pareto-frontier", "pareto.png",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f" Wrote {out_path}")

    # Zoomed-in version on the safe regime (eval > 89%) for paper figure
    fig2, ax2 = plt.subplots(figsize=(11, 6.5), dpi=120)
    safe_configs = [
        (lab, c) for lab, c in per_config.items()
        if c["eval_mean"] * 100 >= 89
    ]
    for label, c in safe_configs:
        fam = family_of(label)
        on_front = label in frontier
        marker = "o" if on_front else "x"
        ax2.errorbar(
            c["syncs_mean"], c["eval_mean"] * 100,
            xerr=c["syncs_sd"] if c["n"] > 1 else None,
            yerr=c["eval_sd"] * 100 if c["n"] > 1 else None,
            fmt=marker, markersize=11 if on_front else 8,
            color=family_color[fam], ecolor=family_color[fam], elinewidth=1, capsize=3,
            markeredgecolor="black" if on_front else family_color[fam],
            markeredgewidth=1.4 if on_front else 0.8,
            zorder=3 if on_front else 2,
        )
    safe_envelope = [(x, y, lab) for x, y, lab in upper_envelope if y >= 89]
    if len(safe_envelope) > 1:
        ax2.plot(
            [p[0] for p in safe_envelope],
            [p[1] for p in safe_envelope],
            color="#3d5a80", linewidth=2, alpha=0.5, zorder=1,
            label="Pareto frontier (upper envelope)",
        )

    for label, (text, offset) in annotation_overrides.items():
        if label not in per_config:
            continue
        c = per_config[label]
        if c["eval_mean"] * 100 < 89:
            continue
        ax2.annotate(
            text,
            xy=(c["syncs_mean"], c["eval_mean"] * 100),
            xytext=offset, textcoords="offset points",
            fontsize=9, color="#293241",
            arrowprops=dict(arrowstyle="-", color="gray", linewidth=0.6, alpha=0.6),
        )

    ax2.legend(handles=legend_handles, loc="lower right", framealpha=0.95, fontsize=9)
    ax2.set_xscale("log")
    ax2.set_xlabel("Syncs per 200-epoch run (log scale, network-volume proxy)", fontsize=11)
    ax2.set_ylabel("Final eval accuracy (%)", fontsize=11)
    ax2.set_title(
        "Eval-vs-cost Pareto frontier — safe regime (eval ≥ 89%)\n"
        "ResNet-20 / CIFAR-10 / 200 epochs / 3-GPU heterogeneous",
        fontsize=12, pad=14,
    )
    ax2.grid(True, which="both", alpha=0.25)
    ax2.set_ylim(89.5, 92.5)

    out_path2 = os.path.join(
        PROJECT_ROOT, RUNS_DIR, "pareto-frontier", "pareto-safe-zoom.png",
    )
    plt.tight_layout()
    plt.savefig(out_path2, dpi=140, bbox_inches="tight")
    plt.close(fig2)
    print(f" Wrote {out_path2}")
except ImportError:
    print(" matplotlib not installed — install with `pip install --user --break-system-packages matplotlib`")

print()
print("=" * 100)
print(" SUMMARY")
print("=" * 100)
print(f" Configurations on frontier: {len(frontier)}")
print(f" Configurations dominated:   {len(dominated)}")
print(f" Total cells loaded:         {sum(len(v) for v in configs.values())}")
print(f" Cells missing on disk:      {len(missing)}")
print()
print(" Production default verdict (ElChe default + msf):")
default_label = "nccl-async default msf"
if default_label in per_config:
    c = per_config[default_label]
    if default_label in dominated:
        print(f"   eval = {c['eval_mean']*100:.2f}% ± {c['eval_sd']*100:.2f}, syncs = {c['syncs_mean']:.0f} ± {c['syncs_sd']:.0f}")
        print(f"   DOMINATED by: {dominated[default_label]}")
    else:
        print(f"   on frontier — eval = {c['eval_mean']*100:.2f}%, syncs = {c['syncs_mean']:.0f}")
print()
print(" cpu-async default candidacy check (EASGD α=0.5 cohort):")
for label in ("cpu-async default msf", "cpu-async default trend"):
    if label in per_config:
        c = per_config[label]
        marker = "" if c["n"] >= 3 else f" (n={c['n']}, needs multi-seed)"
        on_front = "FRONTIER" if label in frontier else f"dominated by {dominated.get(label, '?')}"
        print(f"   {label}: eval = {c['eval_mean']*100:.2f}%, syncs = {c['syncs_mean']:.0f}{marker}  →  {on_front}")
