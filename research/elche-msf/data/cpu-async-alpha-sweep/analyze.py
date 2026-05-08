#!/usr/bin/env python3
"""Per-sweep focused analyzer for cpu-async EASGD α-axis.

Surfaces:
  1. α-axis read at fixed cpu-async × msf × ResNet-20 / 3-GPU.
  2. Paired-seed contrasts vs α=1.0 (the load-bearing baseline).
  3. Per-rank heterogeneity profile (uniform across α, since same recipe).

Reads from this directory (16 cells × report.md); writes to ./analysis/.

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

ALPHAS = [("10", 1.0), ("07", 0.7), ("05", 0.5), ("03", 0.3)]
SEEDS = [1, 2, 3, 4]
GPU_LABELS = ["RTX 5060 Ti", "GTX 1060 (#1)", "GTX 1060 (#2)"]

# Main row: cpu-async | loss | eval | delta | total | syncs | avg_sync | g0% | g1% | g2% | idle
ROW_RE = re.compile(
    r"\| cpu-async \| ([\d.]+) \| ([\d.]+) \| [+\-\d.]+ \| ([\d.]+) \| (\d+) \| ([\d.]+) \|"
    r" (\d+)% \| (\d+)% \| (\d+)% \| ([\d.]+) \|"
)
# VRAM row: model | mode | g0_peak | g0_mean | g1_peak | g1_mean | g2_peak | g2_mean
VRAM_RE = re.compile(
    r"\| resnet-graph \| cpu-async \| (\d+) \| (\d+) \| (\d+) \| (\d+) \| (\d+) \| (\d+) \|"
)


def parse_cell(cell_dir: Path) -> dict:
    text = (cell_dir / "report.md").read_text()
    m = ROW_RE.search(text)
    v = VRAM_RE.search(text)
    if not m or not v:
        raise ValueError(f"could not parse {cell_dir}/report.md")
    return {
        "loss": float(m.group(1)),
        "eval_pct": float(m.group(2)) * 100,
        "total_s": float(m.group(3)),
        "syncs": int(m.group(4)),
        "avg_sync_ms": float(m.group(5)),
        "gpu0_pct": int(m.group(6)),
        "gpu1_pct": int(m.group(7)),
        "gpu2_pct": int(m.group(8)),
        "idle_s": float(m.group(9)),
        "g0_peak_mb": int(v.group(1)), "g0_mean_mb": int(v.group(2)),
        "g1_peak_mb": int(v.group(3)), "g1_mean_mb": int(v.group(4)),
        "g2_peak_mb": int(v.group(5)), "g2_mean_mb": int(v.group(6)),
    }


def load_cells():
    cells = []
    for suffix, alpha in ALPHAS:
        for seed in SEEDS:
            d = HERE / f"seed-{seed}-cpu-async-msf-alpha{suffix}"
            row = parse_cell(d)
            row["alpha"] = alpha
            row["seed"] = seed
            row["cell"] = d.name
            cells.append(row)
    return cells


def write_per_cell_csv(cells, path: Path):
    cols = ["alpha", "seed", "cell", "eval_pct", "loss", "syncs",
            "avg_sync_ms", "total_s", "idle_s",
            "gpu0_pct", "gpu1_pct", "gpu2_pct",
            "g0_peak_mb", "g0_mean_mb",
            "g1_peak_mb", "g1_mean_mb",
            "g2_peak_mb", "g2_mean_mb"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for c in cells:
            w.writerow({k: c[k] for k in cols})


def write_per_rank_csv(cells, path: Path):
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
    """Return list of dicts, one per α, with mean/sd/range/sync_mean."""
    rows = []
    for _, alpha in ALPHAS:
        cohort = [c for c in cells if c["alpha"] == alpha]
        evs = [c["eval_pct"] for c in cohort]
        sys = [c["syncs"] for c in cohort]
        rows.append({
            "alpha": alpha,
            "n": len(cohort),
            "eval_mean": statistics.mean(evs),
            "eval_sd": statistics.stdev(evs),
            "eval_min": min(evs),
            "eval_max": max(evs),
            "sync_mean": statistics.mean(sys),
            "sync_min": min(sys),
            "sync_max": max(sys),
        })
    return rows


def paired_contrasts(cells):
    """Paired contrasts vs α=1.0 baseline (per-seed differences)."""
    baseline = {c["seed"]: c["eval_pct"] for c in cells if c["alpha"] == 1.0}
    rows = []
    for alpha in (0.7, 0.5, 0.3):
        diffs = [c["eval_pct"] - baseline[c["seed"]]
                 for c in cells if c["alpha"] == alpha]
        mean = statistics.mean(diffs)
        sd = statistics.stdev(diffs)
        se = sd / np.sqrt(len(diffs))
        t = mean / se if se > 0 else float("nan")
        rows.append({"alpha": alpha, "delta": mean, "sd_diff": sd, "t": t})
    return rows


def per_rank_summary(cells):
    """Per-rank means/sd across all 16 cells."""
    rows = []
    for rank in range(3):
        util = [c[f"gpu{rank}_pct"] for c in cells]
        peak = [c[f"g{rank}_peak_mb"] for c in cells]
        mean_v = [c[f"g{rank}_mean_mb"] for c in cells]
        rows.append({
            "rank": rank, "gpu_label": GPU_LABELS[rank],
            "util_mean": statistics.mean(util), "util_sd": statistics.stdev(util),
            "peak_mean": statistics.mean(peak), "peak_sd": statistics.stdev(peak),
            "mean_mean": statistics.mean(mean_v), "mean_sd": statistics.stdev(mean_v),
        })
    return rows


# ---------------- figure -----------------

COLORS = {1.0: "#1f77b4", 0.7: "#2ca02c", 0.5: "#ff7f0e", 0.3: "#d62728"}
MARKERS = {1.0: "o", 0.7: "s", 0.5: "^", 0.3: "D"}
THRESHOLD_PP = 0.15
RECIPE_CEILING_LO = 91.6
RECIPE_CEILING_HI = 92.0


def write_figure(cells, summary, contrasts, path: Path):
    by_alpha = {}
    for alpha in (1.0, 0.7, 0.5, 0.3):
        rows = sorted([c for c in cells if c["alpha"] == alpha], key=lambda r: r["seed"])
        by_alpha[alpha] = rows

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    # (a) eval per seed per α
    ax = axes[0]
    ax.axhspan(RECIPE_CEILING_LO, RECIPE_CEILING_HI, color="gray", alpha=0.12,
               label="recipe-ceiling band (91.6–92.0%)")
    for s in summary:
        alpha = s["alpha"]
        seeds = np.array([r["seed"] for r in by_alpha[alpha]])
        evals = np.array([r["eval_pct"] for r in by_alpha[alpha]])
        ax.scatter(seeds, evals, color=COLORS[alpha], marker=MARKERS[alpha], s=80,
                   edgecolors="black", linewidths=0.5,
                   label=f"α={alpha} (mean={s['eval_mean']:.2f}%, sd={s['eval_sd']:.3f})", zorder=3)
        ax.axhline(s["eval_mean"], color=COLORS[alpha], linestyle="--", alpha=0.55, linewidth=1)
    ax.set_xticks(SEEDS)
    ax.set_xlabel("seed")
    ax.set_ylabel("final eval (%)")
    ax.set_title("(a) Eval per seed across α-axis (n=4 each)")
    ax.set_ylim(91.0, 92.4)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.92)

    # (b) paired-seed Δ vs α=1.0
    ax = axes[1]
    ax.axhline(0, color="black", linewidth=1)
    ax.axhline(THRESHOLD_PP, color="black", linestyle=":", linewidth=1,
               label=f"±{THRESHOLD_PP} pp pre-registered threshold")
    ax.axhline(-THRESHOLD_PP, color="black", linestyle=":", linewidth=1)
    baseline = {r["seed"]: r["eval_pct"] for r in by_alpha[1.0]}
    x_pos, x_lab = [], []
    for i, contrast in enumerate(contrasts):
        alpha = contrast["alpha"]
        rows = by_alpha[alpha]
        seeds = np.array([r["seed"] for r in rows])
        delta = np.array([r["eval_pct"] - baseline[r["seed"]] for r in rows])
        se = contrast["sd_diff"] / np.sqrt(len(delta))
        x_jit = i + (seeds - 2.5) * 0.06
        ax.scatter(x_jit, delta, color=COLORS[alpha], marker=MARKERS[alpha], s=80,
                   edgecolors="black", linewidths=0.5, zorder=3)
        ax.errorbar(i, contrast["delta"], yerr=se, fmt="_", color=COLORS[alpha], markersize=22,
                    capsize=8, elinewidth=2.2, zorder=4,
                    label=f"α={alpha}: Δ={contrast['delta']:+.3f}pp, t={contrast['t']:+.2f}")
        x_pos.append(i); x_lab.append(f"α={alpha}\n(vs 1.0)")
    ax.set_xticks(x_pos); ax.set_xticklabels(x_lab)
    ax.set_ylabel("Δ eval vs α=1.0 (paired by seed, pp)")
    ax.set_title("(b) Paired contrasts vs α=1.0 (n=4, df=3)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower left", fontsize=7.5, framealpha=0.92)

    # (c) sync count per seed per α
    ax = axes[2]
    for s in summary:
        alpha = s["alpha"]
        seeds = np.array([r["seed"] for r in by_alpha[alpha]])
        syncs = np.array([r["syncs"] for r in by_alpha[alpha]])
        ax.scatter(seeds, syncs, color=COLORS[alpha], marker=MARKERS[alpha], s=80,
                   edgecolors="black", linewidths=0.5,
                   label=f"α={alpha} (mean={s['sync_mean']:.0f})", zorder=3)
        ax.axhline(s["sync_mean"], color=COLORS[alpha], linestyle="--", alpha=0.55, linewidth=1)
    ax.set_xticks(SEEDS)
    ax.set_xlabel("seed")
    ax.set_ylabel("syncs / 200 ep")
    ax.set_title("(c) Sync count per seed — not monotone in α")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.92)

    means = {s["alpha"]: s["eval_mean"] for s in summary}
    fig.suptitle(
        f"P1 — EASGD α-axis sweep (cpu-async × msf × ResNet-20). "
        f"α=1.0 / 0.7 / 0.5 flat within seed noise; α=0.3 degrades by "
        f"{means[1.0] - means[0.3]:.2f} pp (paired t≈3.08, p<0.05).",
        fontsize=10.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------- main -----------------

def main() -> None:
    parser = argparse.ArgumentParser(description="cpu-async α-axis analyzer.")
    parser.parse_args()

    cells = load_cells()
    print(f"loaded {len(cells)} cells")

    write_per_cell_csv(cells, OUT / "per_cell.csv")
    write_per_rank_csv(cells, OUT / "per_rank.csv")
    print(f"wrote {OUT/'per_cell.csv'} ({len(cells)} rows)")
    print(f"wrote {OUT/'per_rank.csv'} ({len(cells)*3} rows)")

    summary = cohort_summary(cells)
    contrasts = paired_contrasts(cells)
    rank_summary = per_rank_summary(cells)

    print("\nCohort summary (α-axis):")
    print(f'  {"α":>4} {"n":>3} {"eval mean":>10} {"eval sd":>8} {"range":>14} {"sync mean":>10}')
    for s in summary:
        print(f'  {s["alpha"]:>4} {s["n"]:>3} {s["eval_mean"]:>9.2f}% {s["eval_sd"]:>7.3f}'
              f'  {s["eval_min"]:.2f}-{s["eval_max"]:.2f}  {s["sync_mean"]:>9.0f}')

    print("\nPaired contrasts vs α=1.0 (n=4, df=3):")
    print(f'  {"α":>4} {"Δ mean":>9} {"sd of diff":>11} {"paired t":>10}')
    for c in contrasts:
        print(f'  {c["alpha"]:>4} {c["delta"]:>+8.3f}pp {c["sd_diff"]:>10.3f}  {c["t"]:>+9.2f}')

    print("\nPer-rank heterogeneity (across all 16 cells):")
    print(f'  {"rank":>4} {"GPU":<14} {"util mean":>10} {"util sd":>8}'
          f'  {"peak VRAM":>10} {"mean VRAM":>10}')
    for r in rank_summary:
        print(f'  {r["rank"]:>4} {r["gpu_label"]:<14} {r["util_mean"]:>9.1f}%'
              f' {r["util_sd"]:>7.1f}  {r["peak_mean"]:>7.0f} MB {r["mean_mean"]:>7.0f} MB')

    fig_path = OUT / "p1_easgd_alpha_axis.png"
    write_figure(cells, summary, contrasts, fig_path)
    print(f"\nwrote {fig_path}")


if __name__ == "__main__":
    main()
