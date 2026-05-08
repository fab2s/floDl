#!/usr/bin/env python3
"""Per-sweep focused analyzer for ResNet-56 cpu-async (Gate D).

Surfaces:
  1. Bytes-axis confirmation: 4-seed α=0.5 cohort vs published
     ResNet-56 baseline (93.03 %, He et al. 2015 Table 6) and the
     single-cell α=1.0 datapoint.
  2. Single-seed limitation flag for the α=1.0 cell.
  3. Per-rank heterogeneity profile at 3.1× the parameter count.

Reads from this directory (5 cells × report.md); writes to ./analysis/.

Usage:
    python3 analyze.py
"""

import argparse
import csv
import re
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "analysis"
OUT.mkdir(exist_ok=True)

GPU_LABELS = ["RTX 5060 Ti", "GTX 1060 (#1)", "GTX 1060 (#2)"]
PUB_BASELINE = 93.03   # He et al. 2015, ResNet-56 CIFAR-10
R20_PUB = 91.25        # ResNet-20 reference for context

# (cell_name, seed, alpha)
CELLS = [
    ("seed-1-cpu-async-msf",        1, 0.5),
    ("seed-2-cpu-async-msf",        2, 0.5),
    ("seed-3-cpu-async-msf",        3, 0.5),
    ("seed-4-cpu-async-msf",        4, 0.5),
    ("seed-1-cpu-async-msf-alpha10", 1, 1.0),
]

ROW_RE = re.compile(
    r"\| cpu-async \| ([\d.]+) \| ([\d.]+) \| [+\-\d.]+ \| ([\d.]+) \| (\d+) \| ([\d.]+) \|"
    r" (\d+)% \| (\d+)% \| (\d+)% \| ([\d.]+) \|"
)
VRAM_RE = re.compile(
    r"\| resnet-graph \| cpu-async \| (\d+) \| (\d+) \| (\d+) \| (\d+) \| (\d+) \| (\d+) \|"
)


def parse_cell(cell_dir: Path) -> dict:
    text = (cell_dir / "report.md").read_text()
    m = ROW_RE.search(text); v = VRAM_RE.search(text)
    if not m or not v:
        raise ValueError(f"could not parse {cell_dir}/report.md")
    return {
        "loss": float(m.group(1)), "eval_pct": float(m.group(2)) * 100,
        "total_s": float(m.group(3)), "syncs": int(m.group(4)),
        "avg_sync_ms": float(m.group(5)),
        "gpu0_pct": int(m.group(6)), "gpu1_pct": int(m.group(7)), "gpu2_pct": int(m.group(8)),
        "idle_s": float(m.group(9)),
        "g0_peak_mb": int(v.group(1)), "g0_mean_mb": int(v.group(2)),
        "g1_peak_mb": int(v.group(3)), "g1_mean_mb": int(v.group(4)),
        "g2_peak_mb": int(v.group(5)), "g2_mean_mb": int(v.group(6)),
    }


def load_cells():
    cells = []
    for name, seed, alpha in CELLS:
        row = parse_cell(HERE / name)
        row["cell"] = name; row["seed"] = seed; row["alpha"] = alpha
        cells.append(row)
    return cells


def write_per_cell_csv(cells, path):
    cols = ["alpha", "seed", "cell", "eval_pct", "loss", "syncs",
            "avg_sync_ms", "total_s", "idle_s",
            "gpu0_pct", "gpu1_pct", "gpu2_pct",
            "g0_peak_mb", "g0_mean_mb", "g1_peak_mb", "g1_mean_mb",
            "g2_peak_mb", "g2_mean_mb"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for c in cells:
            w.writerow({k: c[k] for k in cols})


def write_per_rank_csv(cells, path):
    cols = ["alpha", "seed", "cell", "rank", "gpu_label",
            "util_pct", "peak_vram_mb", "mean_vram_mb"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for c in cells:
            for rank in range(3):
                w.writerow({
                    "alpha": c["alpha"], "seed": c["seed"], "cell": c["cell"],
                    "rank": rank, "gpu_label": GPU_LABELS[rank],
                    "util_pct": c[f"gpu{rank}_pct"],
                    "peak_vram_mb": c[f"g{rank}_peak_mb"],
                    "mean_vram_mb": c[f"g{rank}_mean_mb"],
                })


def cohort_summary(cells):
    rows = []
    for alpha in (0.5, 1.0):
        cohort = [c for c in cells if c["alpha"] == alpha]
        evs = [c["eval_pct"] for c in cohort]
        sys = [c["syncs"] for c in cohort]
        row = {"alpha": alpha, "n": len(cohort),
               "eval_mean": statistics.mean(evs),
               "eval_min": min(evs), "eval_max": max(evs),
               "sync_mean": statistics.mean(sys)}
        row["eval_sd"] = statistics.stdev(evs) if len(evs) > 1 else None
        rows.append(row)
    return rows


def per_rank_summary(cells):
    rows = []
    for rank in range(3):
        util = [c[f"gpu{rank}_pct"] for c in cells]
        peak = [c[f"g{rank}_peak_mb"] for c in cells]
        mean_v = [c[f"g{rank}_mean_mb"] for c in cells]
        rows.append({
            "rank": rank, "gpu_label": GPU_LABELS[rank],
            "util_mean": statistics.mean(util),
            "util_sd": statistics.stdev(util) if len(util) > 1 else 0.0,
            "peak_mean": statistics.mean(peak), "mean_mean": statistics.mean(mean_v),
        })
    return rows


# ---------------- figure -----------------

def write_figure(cells, path):
    alpha05 = sorted([c for c in cells if c["alpha"] == 0.5], key=lambda r: r["seed"])
    alpha10 = sorted([c for c in cells if c["alpha"] == 1.0], key=lambda r: r["seed"])
    c05 = "#2ca02c"; c10 = "#d62728"

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    # Left — eval per seed.
    ax = axes[0]
    s05 = np.array([r["seed"] for r in alpha05])
    e05 = np.array([r["eval_pct"] for r in alpha05])
    ax.scatter(s05, e05, color=c05, marker="o", s=90,
               label=f"α=0.5 (n=4, mean={e05.mean():.2f}%)", zorder=3)
    ax.axhline(e05.mean(), color=c05, linestyle="--", alpha=0.6)

    s10 = np.array([r["seed"] for r in alpha10])
    e10 = np.array([r["eval_pct"] for r in alpha10])
    ax.scatter(s10, e10, color=c10, marker="X", s=160,
               label=f"α=1.0 (n=1, single-seed = {e10[0]:.2f}%)",
               edgecolors="black", linewidths=0.8, zorder=4)

    ax.axhline(PUB_BASELINE, color="black", linestyle=":", linewidth=1.4, alpha=0.8,
               label=f"published ResNet-56 baseline ({PUB_BASELINE}%)")

    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("seed"); ax.set_ylabel("final eval (%)")
    ax.set_title("ResNet-56 cpu-async msf — eval per seed")
    ax.set_ylim(92.5, 93.5)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    # Right — sync count per seed.
    ax = axes[1]
    sy05 = np.array([r["syncs"] for r in alpha05])
    sy10 = np.array([r["syncs"] for r in alpha10])
    ax.scatter(s05, sy05, color=c05, marker="o", s=90,
               label=f"α=0.5 (n=4, mean={sy05.mean():.0f})", zorder=3)
    ax.axhline(sy05.mean(), color=c05, linestyle="--", alpha=0.6)
    ax.scatter(s10, sy10, color=c10, marker="X", s=160,
               label=f"α=1.0 (n=1, {sy10[0]} syncs)",
               edgecolors="black", linewidths=0.8, zorder=4)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("seed"); ax.set_ylabel("syncs / 200 ep")
    ax.set_title("Sync count per seed — α=1.0 single-seed limits comparison")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    fig.suptitle("Gate D — ResNet-56 bytes-axis confirmation (α=0.5 cohort hits "
                 f"{e05.mean():.2f}% vs published {PUB_BASELINE}%)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="ResNet-56 cpu-async analyzer.")
    parser.parse_args()

    cells = load_cells()
    print(f"loaded {len(cells)} cells")

    write_per_cell_csv(cells, OUT / "per_cell.csv")
    write_per_rank_csv(cells, OUT / "per_rank.csv")
    print(f"wrote {OUT/'per_cell.csv'} ({len(cells)} rows)")
    print(f"wrote {OUT/'per_rank.csv'} ({len(cells)*3} rows)")

    summary = cohort_summary(cells)
    rank_summary = per_rank_summary(cells)

    print("\nCohort summary (Gate D — ResNet-56 cpu-async msf):")
    print(f'  {"α":>4} {"n":>3} {"eval mean":>10} {"eval sd":>8} {"range":>14} {"sync mean":>10}')
    for s in summary:
        sd_str = f'{s["eval_sd"]:.3f}' if s["eval_sd"] is not None else "  n/a"
        print(f'  {s["alpha"]:>4} {s["n"]:>3} {s["eval_mean"]:>9.2f}% {sd_str:>8}'
              f'  {s["eval_min"]:.2f}-{s["eval_max"]:.2f}  {s["sync_mean"]:>9.0f}')

    print("\nVs published baselines:")
    a05 = next(s for s in summary if s["alpha"] == 0.5)
    print(f'  α=0.5 mean {a05["eval_mean"]:.2f}% vs ResNet-56 published {PUB_BASELINE}%'
          f' (Δ = {a05["eval_mean"] - PUB_BASELINE:+.2f} pp)')

    print("\nPer-rank heterogeneity (across 5 cells):")
    print(f'  {"rank":>4} {"GPU":<14} {"util mean":>10} {"util sd":>8}'
          f'  {"peak VRAM":>10} {"mean VRAM":>10}')
    for r in rank_summary:
        print(f'  {r["rank"]:>4} {r["gpu_label"]:<14} {r["util_mean"]:>9.1f}%'
              f' {r["util_sd"]:>7.1f}  {r["peak_mean"]:>7.0f} MB {r["mean_mean"]:>7.0f} MB')

    fig_path = OUT / "gate_d_resnet56_alpha.png"
    write_figure(cells, fig_path)
    print(f"\nwrote {fig_path}")


if __name__ == "__main__":
    main()
