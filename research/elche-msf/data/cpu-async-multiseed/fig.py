#!/usr/bin/env python3
"""Headline figure for the cpu-async multi-seed (Gate A) sweep.

One panel: per-seed eval + sync count for the 4-seed × 2-guard cohort,
with the seed-0 single-shot smoke value overlaid as a star and the
α=1.0 cohort baseline mean shown as a horizontal band. Highlights the
fact that the seed-0 sync-reduction was a tail outlier and the
multi-seed mean is essentially flat against the α=1.0 baseline.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE / "analysis"
ANALYSIS.mkdir(exist_ok=True)

# Pull (eval, syncs) from each cell's report.md main table row.
ROW_RE = re.compile(
    r"\| cpu-async \| ([\d.]+) \| ([\d.]+) \| [+\-\d.]+ \| ([\d.]+) \| (\d+) \|"
)


def parse_cell(cell_dir: Path):
    text = (cell_dir / "report.md").read_text()
    m = ROW_RE.search(text)
    if not m:
        raise ValueError(f"could not parse main table row in {cell_dir}/report.md")
    return float(m.group(2)) * 100, int(m.group(4))  # eval%, syncs


# Gate A α=0.5 cohort (4 seeds × 2 guards)
seeds = [1, 2, 3, 4]
data = {"msf": [], "trend": []}
for s in seeds:
    for g in ("msf", "trend"):
        eval_pct, syncs = parse_cell(HERE / f"seed-{s}-cpu-async-{g}")
        data[g].append((s, eval_pct, syncs))

# Reference values (from design doc Gate A spec, lines ~498-510 of
# docs/design/msf-cadence-control-v2.md).
SEED0_MSF_EVAL = 91.91          # seed-0 single-shot smoke
SEED0_MSF_SYNCS = 408
SEED0_TREND_EVAL = 91.39
SEED0_TREND_SYNCS = 726
ALPHA1_MSF_BASELINE_EVAL = 91.86  # α=1.0 cross-seed baseline (n=4)
ALPHA1_MSF_BASELINE_SD = 0.27
ALPHA1_TREND_BASELINE_EVAL = 91.96
ALPHA1_TREND_BASELINE_SD = 0.23

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
colors = {"msf": "#1f77b4", "trend": "#ff7f0e"}
markers = {"msf": "o", "trend": "s"}

# Left panel — eval per seed, with α=1.0 baseline band + seed-0 smoke star.
ax = axes[0]
for guard in ("msf", "trend"):
    s_arr = np.array([row[0] for row in data[guard]])
    e_arr = np.array([row[1] for row in data[guard]])
    ax.scatter(s_arr, e_arr, color=colors[guard], marker=markers[guard], s=80,
               label=f"α=0.5 {guard} (multi-seed)", zorder=3)
    ax.axhline(np.mean(e_arr), color=colors[guard], linestyle="--", alpha=0.5,
               label=f"α=0.5 {guard} mean = {np.mean(e_arr):.2f}%")

# α=1.0 baseline bands (msf + trend).
for guard, base, sd, c in (
    ("msf", ALPHA1_MSF_BASELINE_EVAL, ALPHA1_MSF_BASELINE_SD, colors["msf"]),
    ("trend", ALPHA1_TREND_BASELINE_EVAL, ALPHA1_TREND_BASELINE_SD, colors["trend"]),
):
    ax.axhspan(base - sd, base + sd, color=c, alpha=0.10,
               label=f"α=1.0 {guard} baseline {base:.2f}±{sd:.2f}%")

# Seed-0 single-shot smoke (the single-cell prediction-source).
ax.scatter([0], [SEED0_MSF_EVAL], color=colors["msf"], marker="*", s=250,
           edgecolors="black", linewidths=1.0, zorder=4,
           label=f"seed-0 smoke msf ({SEED0_MSF_EVAL:.2f}%)")
ax.scatter([0], [SEED0_TREND_EVAL], color=colors["trend"], marker="*", s=250,
           edgecolors="black", linewidths=1.0, zorder=4,
           label=f"seed-0 smoke trend ({SEED0_TREND_EVAL:.2f}%)")

ax.set_xticks([0, 1, 2, 3, 4])
ax.set_xticklabels(["s0\n(smoke)", "s1", "s2", "s3", "s4"])
ax.set_xlabel("seed")
ax.set_ylabel("final eval (%)")
ax.set_title("Eval per seed — Gate A (R-20 cpu-async α=0.5)")
ax.set_ylim(91.0, 92.5)
ax.grid(axis="y", alpha=0.3)
ax.legend(loc="lower center", fontsize=7, ncol=2, framealpha=0.9)

# Right panel — syncs per seed, same overlays.
ax = axes[1]
for guard in ("msf", "trend"):
    s_arr = np.array([row[0] for row in data[guard]])
    sync_arr = np.array([row[2] for row in data[guard]])
    ax.scatter(s_arr, sync_arr, color=colors[guard], marker=markers[guard],
               s=80, label=f"α=0.5 {guard} (multi-seed)", zorder=3)
    ax.axhline(np.mean(sync_arr), color=colors[guard], linestyle="--", alpha=0.5,
               label=f"α=0.5 {guard} mean = {np.mean(sync_arr):.0f}")

ax.scatter([0], [SEED0_MSF_SYNCS], color=colors["msf"], marker="*", s=250,
           edgecolors="black", linewidths=1.0, zorder=4,
           label=f"seed-0 smoke msf ({SEED0_MSF_SYNCS} syncs)")
ax.scatter([0], [SEED0_TREND_SYNCS], color=colors["trend"], marker="*", s=250,
           edgecolors="black", linewidths=1.0, zorder=4,
           label=f"seed-0 smoke trend ({SEED0_TREND_SYNCS} syncs)")

ax.set_xticks([0, 1, 2, 3, 4])
ax.set_xticklabels(["s0\n(smoke)", "s1", "s2", "s3", "s4"])
ax.set_xlabel("seed")
ax.set_ylabel("syncs / 200 ep")
ax.set_title("Sync count per seed — seed-0's 408 syncs (msf) was a tail outlier")
ax.grid(axis="y", alpha=0.3)
ax.legend(loc="upper right", fontsize=7, framealpha=0.9)

fig.suptitle("Gate A — multi-seed EASGD α=0.5 confirmation pass (ResNet-20)",
             fontsize=11, fontweight="bold")
fig.tight_layout()
out_path = ANALYSIS / "gate_a_alpha_predictions.png"
fig.savefig(out_path, dpi=110, bbox_inches="tight")
print(f"wrote {out_path}")
