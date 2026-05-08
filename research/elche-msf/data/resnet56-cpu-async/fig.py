#!/usr/bin/env python3
"""Headline figure for the ResNet-56 cpu-async (Gate D) sweep.

Two panels:
  (a) Eval per seed for the α=0.5 cohort (4 seeds) + α=1.0 single-seed
      datapoint, with the published ResNet-56 baseline (93.03%) drawn
      as a reference line.
  (b) Sync count per seed across both α-cohorts, to show the directional
      gap between α=0.5 (means ≈ 450 syncs) and α=1.0 (286 syncs at
      seed-1) without claiming a paired comparison.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE / "analysis"
ANALYSIS.mkdir(exist_ok=True)

ROW_RE = re.compile(
    r"\| cpu-async \| ([\d.]+) \| ([\d.]+) \| [+\-\d.]+ \| ([\d.]+) \| (\d+) \|"
)


def parse_cell(cell_dir: Path):
    text = (cell_dir / "report.md").read_text()
    m = ROW_RE.search(text)
    if not m:
        raise ValueError(f"could not parse main table row in {cell_dir}/report.md")
    return float(m.group(2)) * 100, int(m.group(4))


# α=0.5 cohort (n=4)
alpha05 = []
for s in (1, 2, 3, 4):
    eval_pct, syncs = parse_cell(HERE / f"seed-{s}-cpu-async-msf")
    alpha05.append((s, eval_pct, syncs))

# α=1.0 single-seed
eval_pct, syncs = parse_cell(HERE / "seed-1-cpu-async-msf-alpha10")
alpha10 = [(1, eval_pct, syncs)]

PUB_BASELINE = 93.03  # He et al. 2015, ResNet-56 CIFAR-10
R20_R56_PUB = 91.25   # ResNet-20 reference (for context)

c05 = "#2ca02c"  # α=0.5 (canonical, green)
c10 = "#d62728"  # α=1.0 (single-seed, red)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

# Left panel — eval per seed.
ax = axes[0]
seeds05 = np.array([row[0] for row in alpha05])
evals05 = np.array([row[1] for row in alpha05])
ax.scatter(seeds05, evals05, color=c05, marker="o", s=90,
           label=f"α=0.5 (n=4, mean={evals05.mean():.2f}%)", zorder=3)
ax.axhline(evals05.mean(), color=c05, linestyle="--", alpha=0.6)

seeds10 = np.array([row[0] for row in alpha10])
evals10 = np.array([row[1] for row in alpha10])
ax.scatter(seeds10, evals10, color=c10, marker="X", s=160,
           label=f"α=1.0 (n=1, single-seed = {evals10[0]:.2f}%)",
           edgecolors="black", linewidths=0.8, zorder=4)

ax.axhline(PUB_BASELINE, color="black", linestyle=":", linewidth=1.4, alpha=0.8,
           label=f"published ResNet-56 baseline ({PUB_BASELINE}%)")

ax.set_xticks([1, 2, 3, 4])
ax.set_xlabel("seed")
ax.set_ylabel("final eval (%)")
ax.set_title("ResNet-56 cpu-async msf — eval per seed")
ax.set_ylim(92.5, 93.5)
ax.grid(axis="y", alpha=0.3)
ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

# Right panel — sync count per seed.
ax = axes[1]
syncs05 = np.array([row[2] for row in alpha05])
syncs10 = np.array([row[2] for row in alpha10])
ax.scatter(seeds05, syncs05, color=c05, marker="o", s=90,
           label=f"α=0.5 (n=4, mean={syncs05.mean():.0f})", zorder=3)
ax.axhline(syncs05.mean(), color=c05, linestyle="--", alpha=0.6)

ax.scatter(seeds10, syncs10, color=c10, marker="X", s=160,
           label=f"α=1.0 (n=1, {syncs10[0]} syncs)",
           edgecolors="black", linewidths=0.8, zorder=4)

ax.set_xticks([1, 2, 3, 4])
ax.set_xlabel("seed")
ax.set_ylabel("syncs / 200 ep")
ax.set_title("Sync count per seed — α=1.0 single-seed limits comparison")
ax.grid(axis="y", alpha=0.3)
ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

fig.suptitle("Gate D — ResNet-56 bytes-axis confirmation (α=0.5 cohort hits "
             f"{evals05.mean():.2f}% vs published {PUB_BASELINE}%)",
             fontsize=11, fontweight="bold")
fig.tight_layout()
out_path = ANALYSIS / "gate_d_resnet56_alpha.png"
fig.savefig(out_path, dpi=110, bbox_inches="tight")
print(f"wrote {out_path}")
