#!/usr/bin/env python3
"""Per-sweep focused analyzer for cpu-async multiseed (Gate A).

Surfaces:
  1. 4-seed × 2-guard α=0.5 cohort summary (the multi-seed Gate A
     confirmation pass).
  2. Reproducibility check vs the seed-0 single-shot smoke (the source
     of the design-doc Gate A predictions).
  3. Per-rank heterogeneity profile.

Reads from this directory (8 cells × report.md); writes to ./analysis/.

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

SEEDS = [1, 2, 3, 4]
GUARDS = ["msf", "trend"]
GPU_LABELS = ["RTX 5060 Ti", "GTX 1060 (#1)", "GTX 1060 (#2)"]

# Reference values from the design-doc Gate A spec. Single-shot smoke at
# seed-0 was the source of the multi-seed predictions; the α=1.0 baseline
# was originally read from passive-observation/seed-N-nccl-async-msf —
# mode-confounded (nccl-async, not cpu-async). The cpu-async α=1.0 R-20
# baseline was first cleanly measured by ../cpu-async-alpha-sweep:
# msf 91.77 ± 0.19 (n=4). The trend-guard cpu-async α=1.0 baseline has
# never been measured in-mode.
SEED0_SMOKE = {"msf": (91.91, 408), "trend": (91.39, 726)}
ALPHA1_BASELINE = {  # historical reference values (mode-confounded for the cited 91.86)
    "msf": (91.86, 0.27),
    "trend": (91.96, 0.23),
}

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
    for seed in SEEDS:
        for guard in GUARDS:
            d = HERE / f"seed-{seed}-cpu-async-{guard}"
            row = parse_cell(d)
            row["seed"] = seed
            row["guard"] = guard
            row["cell"] = d.name
            cells.append(row)
    return cells


def write_per_cell_csv(cells, path):
    cols = ["seed", "guard", "cell", "eval_pct", "loss", "syncs",
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
    cols = ["seed", "guard", "cell", "rank", "gpu_label",
            "util_pct", "peak_vram_mb", "mean_vram_mb"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for c in cells:
            for rank in range(3):
                w.writerow({
                    "seed": c["seed"], "guard": c["guard"], "cell": c["cell"],
                    "rank": rank, "gpu_label": GPU_LABELS[rank],
                    "util_pct": c[f"gpu{rank}_pct"],
                    "peak_vram_mb": c[f"g{rank}_peak_mb"],
                    "mean_vram_mb": c[f"g{rank}_mean_mb"],
                })


def cohort_summary(cells):
    rows = []
    for guard in GUARDS:
        cohort = [c for c in cells if c["guard"] == guard]
        evs = [c["eval_pct"] for c in cohort]
        sys = [c["syncs"] for c in cohort]
        rows.append({
            "guard": guard, "n": len(cohort),
            "eval_mean": statistics.mean(evs), "eval_sd": statistics.stdev(evs),
            "eval_min": min(evs), "eval_max": max(evs),
            "sync_mean": statistics.mean(sys),
            "sync_min": min(sys), "sync_max": max(sys),
        })
    return rows


def per_rank_summary(cells):
    rows = []
    for rank in range(3):
        util = [c[f"gpu{rank}_pct"] for c in cells]
        peak = [c[f"g{rank}_peak_mb"] for c in cells]
        mean_v = [c[f"g{rank}_mean_mb"] for c in cells]
        rows.append({
            "rank": rank, "gpu_label": GPU_LABELS[rank],
            "util_mean": statistics.mean(util), "util_sd": statistics.stdev(util),
            "peak_mean": statistics.mean(peak), "mean_mean": statistics.mean(mean_v),
        })
    return rows


# ---------------- figure -----------------

def write_figure(cells, path):
    by_guard = {g: [c for c in cells if c["guard"] == g] for g in GUARDS}
    colors = {"msf": "#1f77b4", "trend": "#ff7f0e"}
    markers = {"msf": "o", "trend": "s"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    # Left — eval per seed.
    ax = axes[0]
    for guard in GUARDS:
        rows = sorted(by_guard[guard], key=lambda r: r["seed"])
        s_arr = np.array([r["seed"] for r in rows])
        e_arr = np.array([r["eval_pct"] for r in rows])
        ax.scatter(s_arr, e_arr, color=colors[guard], marker=markers[guard], s=80,
                   label=f"α=0.5 {guard} (multi-seed)", zorder=3)
        ax.axhline(np.mean(e_arr), color=colors[guard], linestyle="--", alpha=0.5,
                   label=f"α=0.5 {guard} mean = {np.mean(e_arr):.2f}%")
    for guard, (base, sd) in ALPHA1_BASELINE.items():
        ax.axhspan(base - sd, base + sd, color=colors[guard], alpha=0.10,
                   label=f"α=1.0 {guard} baseline {base:.2f}±{sd:.2f}%")
    for guard, (eval_pct, _) in SEED0_SMOKE.items():
        ax.scatter([0], [eval_pct], color=colors[guard], marker="*", s=250,
                   edgecolors="black", linewidths=1.0, zorder=4,
                   label=f"seed-0 smoke {guard} ({eval_pct:.2f}%)")
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xticklabels(["s0\n(smoke)", "s1", "s2", "s3", "s4"])
    ax.set_xlabel("seed"); ax.set_ylabel("final eval (%)")
    ax.set_title("Eval per seed — Gate A (R-20 cpu-async α=0.5)")
    ax.set_ylim(91.0, 92.5); ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower center", fontsize=7, ncol=2, framealpha=0.9)

    # Right — sync count per seed.
    ax = axes[1]
    for guard in GUARDS:
        rows = sorted(by_guard[guard], key=lambda r: r["seed"])
        s_arr = np.array([r["seed"] for r in rows])
        sync_arr = np.array([r["syncs"] for r in rows])
        ax.scatter(s_arr, sync_arr, color=colors[guard], marker=markers[guard],
                   s=80, label=f"α=0.5 {guard} (multi-seed)", zorder=3)
        ax.axhline(np.mean(sync_arr), color=colors[guard], linestyle="--", alpha=0.5,
                   label=f"α=0.5 {guard} mean = {np.mean(sync_arr):.0f}")
    for guard, (_, syncs) in SEED0_SMOKE.items():
        ax.scatter([0], [syncs], color=colors[guard], marker="*", s=250,
                   edgecolors="black", linewidths=1.0, zorder=4,
                   label=f"seed-0 smoke {guard} ({syncs} syncs)")
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xticklabels(["s0\n(smoke)", "s1", "s2", "s3", "s4"])
    ax.set_xlabel("seed"); ax.set_ylabel("syncs / 200 ep")
    ax.set_title("Sync count per seed — seed-0's 408 syncs (msf) was a tail outlier")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)

    fig.suptitle("Gate A — multi-seed EASGD α=0.5 confirmation pass (ResNet-20)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="cpu-async multiseed analyzer.")
    parser.parse_args()

    cells = load_cells()
    print(f"loaded {len(cells)} cells")

    write_per_cell_csv(cells, OUT / "per_cell.csv")
    write_per_rank_csv(cells, OUT / "per_rank.csv")
    print(f"wrote {OUT/'per_cell.csv'} ({len(cells)} rows)")
    print(f"wrote {OUT/'per_rank.csv'} ({len(cells)*3} rows)")

    summary = cohort_summary(cells)
    rank_summary = per_rank_summary(cells)

    print("\nCohort summary (Gate A α=0.5, n=4 per guard):")
    print(f'  {"guard":>6} {"n":>3} {"eval mean":>10} {"eval sd":>8} {"range":>14} {"sync mean":>10}')
    for s in summary:
        print(f'  {s["guard"]:>6} {s["n"]:>3} {s["eval_mean"]:>9.2f}% {s["eval_sd"]:>7.3f}'
              f'  {s["eval_min"]:.2f}-{s["eval_max"]:.2f}  {s["sync_mean"]:>9.0f}')

    print("\nSeed-0 smoke vs multi-seed mean (msf only — load-bearing):")
    msf_summary = next(s for s in summary if s["guard"] == "msf")
    print(f'  smoke s0 = 91.91%  /  408 syncs')
    print(f'  multi-seed mean = {msf_summary["eval_mean"]:.2f}%  /  {msf_summary["sync_mean"]:.0f} syncs')

    print("\nPer-rank heterogeneity (across 8 cells):")
    print(f'  {"rank":>4} {"GPU":<14} {"util mean":>10} {"util sd":>8}'
          f'  {"peak VRAM":>10} {"mean VRAM":>10}')
    for r in rank_summary:
        print(f'  {r["rank"]:>4} {r["gpu_label"]:<14} {r["util_mean"]:>9.1f}%'
              f' {r["util_sd"]:>7.1f}  {r["peak_mean"]:>7.0f} MB {r["mean_mean"]:>7.0f} MB')

    fig_path = OUT / "gate_a_alpha_predictions.png"
    write_figure(cells, fig_path)
    print(f"\nwrote {fig_path}")


if __name__ == "__main__":
    main()
