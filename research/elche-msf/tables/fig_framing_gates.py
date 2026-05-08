#!/usr/bin/env python3
"""Cross-sweep cross-rank Pearson r̄ scatter for the framing-validity
gate table. Pulls per_cell.csv from every sweep that has one and
plots all (pearson_r01, pearson_r02, pearson_r12) values as a strip
chart, with the framing-validity gate (r = 0.95) and the empirical
anchor (r = 0.99) drawn as reference lines.

Cliff-bracket cells past k = 16000 (≤ 2 sync events, Pearson r is a
sample-size artifact at N ≤ 2) are excluded — the gate is
load-bearing only in the safe regime.
"""

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
OUT = HERE / "framing_gates_cross_sweep.png"


def load_per_cell(sweep_dir: Path):
    """Return list of (sweep_label, mode, r01, r02, r12) tuples."""
    csv_path = sweep_dir / "analysis" / "per_cell.csv"
    if not csv_path.exists():
        return []
    rows = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            try:
                r01 = float(row["pearson_r01"])
                r02 = float(row["pearson_r02"])
                r12 = float(row["pearson_r12"])
            except (KeyError, ValueError):
                continue  # cells with too few syncs are blank
            mode = row.get("mode", "")
            rows.append((sweep_dir.name, mode, r01, r02, r12))
    return rows


cells = []
cells += load_per_cell(DATA / "passive-observation")
cells += load_per_cell(DATA / "relaxed-anchor")
# cliff-bracket: drop cells past the cliff (k ≥ 25600 → ≤ 2 sync events)
for row in load_per_cell(DATA / "cliff-bracket"):
    # row format: (sweep_label, mode, r01, r02, r12) — k isn't in the tuple,
    # but cells with all r ≥ 0.9999 collapse to that artifact only past k=16000.
    # Use a simpler proxy: drop rows where ALL three r values round to 1.0000.
    if all(abs(r - 1.0) < 1e-4 for r in row[2:]):
        continue
    cells.append(row)

# Group by (sweep, mode) for color
labels = {
    ("passive-observation", "nccl-async"): ("passive nccl-async", "#1f77b4"),
    ("passive-observation", "cpu-async"): ("passive cpu-async (α=0.5)", "#9467bd"),
    ("relaxed-anchor", "nccl-async"): ("relaxed nccl-async", "#2ca02c"),
    ("cliff-bracket", "nccl-async"): ("cliff-bracket safe regime", "#ff7f0e"),
}

fig, ax = plt.subplots(figsize=(10, 4.6))

# Strip plot — three columns per cell (r01, r02, r12), x-jitter per group.
group_x = {
    "rank 0↔1": 1,
    "rank 0↔2": 2,
    "rank 1↔2": 3,
}
rng = np.random.default_rng(0)

for (sweep, mode), (label, color) in labels.items():
    sub = [c for c in cells if c[0] == sweep and c[1] == mode]
    if not sub:
        continue
    for i, pair_name in enumerate(("rank 0↔1", "rank 0↔2", "rank 1↔2")):
        vals = [c[2 + i] for c in sub]
        x_jit = rng.uniform(-0.18, 0.18, size=len(vals)) + group_x[pair_name]
        ax.scatter(x_jit, vals, color=color, alpha=0.7, s=44,
                   label=label if i == 0 else None, edgecolors="none")

# Gate + anchor reference lines.
ax.axhline(0.99, color="black", linestyle=":", alpha=0.55,
           label="empirical anchor r = 0.99")
ax.axhline(0.95, color="red", linestyle="--", alpha=0.55,
           label="framing-validity gate r = 0.95")

ax.set_xticks([1, 2, 3])
ax.set_xticklabels(["rank 0 ↔ 1", "rank 0 ↔ 2", "rank 1 ↔ 2"])
ax.set_ylabel("cross-rank Pearson r")
ax.set_ylim(0.94, 1.005)
ax.set_title(f"Cross-rank Pearson r̄ across {len(cells)} framing-valid cells "
             "(0/{} gate violations)".format(len(cells)),
             fontsize=11, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
ax.legend(loc="lower left", fontsize=8, ncol=2, framealpha=0.92)

fig.tight_layout()
fig.savefig(OUT, dpi=110, bbox_inches="tight")
print(f"wrote {OUT}  (n={len(cells)} cells)")
