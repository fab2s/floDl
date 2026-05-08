#!/usr/bin/env python3
"""Six cross-sweep figures aggregating all extracted sweep data.

Outputs to tables/cross_sweep/. Each figure addresses one paper claim
that no single sweep aggregator can answer alone.

Figures
-------
fig1  r20_vs_r56                — bytes-axis comparison (4 panels)
fig2  wall_time_pareto          — eval vs wall-time (parallel to eval vs syncs)
fig3  r1_slope_invariance       — by-k slope per cohort (the bottom-scale physics)
fig4  eval_strip_plot           — every cell's eval, by cohort
fig5  pearson_vs_cadence        — r̄ vs effective cadence
fig6  cliff_distance            — cadence regime map

Reproduce:
    python3 research/elche-msf/tables/cross_sweep.py
"""

import csv
import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
OUT = HERE / "cross_sweep"
OUT.mkdir(exist_ok=True)

# Report.md main table row: | <mode> | <loss> | <eval> | <vs_ref> | <total_s> | <syncs> | <avg_sync_ms> | …
ROW_RE = re.compile(
    r"\| (cpu-async|nccl-async) \| ([\d.]+) \| ([\d.]+) \| [+\-]?[\d.]+ \| "
    r"([\d.]+) \| (\d+) \| ([\d.]+) \|"
)


def parse_report(cell_dir: Path):
    """Return dict(mode, eval_pct, wall_s, syncs, avg_sync_ms) or None."""
    p = cell_dir / "report.md"
    if not p.exists():
        return None
    m = ROW_RE.search(p.read_text())
    if not m:
        return None
    return {
        "mode": m.group(1),
        "eval_pct": float(m.group(3)) * 100,
        "wall_s": float(m.group(4)),
        "syncs": int(m.group(5)),
        "avg_sync_ms": float(m.group(6)),
    }


def load_per_cell_csv(sweep_dir: Path):
    p = sweep_dir / "analysis" / "per_cell.csv"
    if not p.exists():
        return []
    with p.open() as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Cohort definitions — one place to enumerate where each cell lives.
# ---------------------------------------------------------------------------

COHORTS = [
    # (label, color, list of cell dirs)
    ("nccl-async\ndefault msf", "#1f77b4", [
        DATA / "passive-observation" / f"seed-{s}-nccl-async-msf" for s in range(5)
    ]),
    ("nccl-async\ndefault trend", "#aec7e8", [
        DATA / "passive-observation" / f"seed-{s}-nccl-async-trend" for s in range(5)
    ]),
    ("nccl-async\nrelaxed msf", "#2ca02c", [
        DATA / "relaxed-anchor" / f"seed-{s}-nccl-async-msf-relaxed" for s in range(5)
    ]),
    ("nccl-async\nrelaxed trend", "#98df8a", [
        DATA / "relaxed-anchor" / f"seed-{s}-nccl-async-trend-relaxed" for s in range(5)
    ]),
    ("cpu-async α=0.5\nmsf (5-seed)", "#9467bd", [
        DATA / "passive-observation" / f"seed-{s}-cpu-async-msf" for s in range(5)
    ]),
    ("cpu-async α=0.5\ntrend (5-seed)", "#c5b0d5", [
        DATA / "passive-observation" / f"seed-{s}-cpu-async-trend" for s in range(5)
    ]),
    ("cpu-async α=0.5\nmsf (Gate A 4-seed)", "#8c564b", [
        DATA / "cpu-async-multiseed" / f"seed-{s}-cpu-async-msf" for s in (1, 2, 3, 4)
    ]),
    ("cpu-async α=0.5\ntrend (Gate A 4-seed)", "#c49c94", [
        DATA / "cpu-async-multiseed" / f"seed-{s}-cpu-async-trend" for s in (1, 2, 3, 4)
    ]),
]

CLIFF_COHORTS = [
    (f"fixed k={k}", c, [
        DATA / "cliff-bracket" / f"seed-{s}-fixed-k-{k}" for s in range(3)
    ])
    for k, c in (
        (3200, "#ffbb78"), (6400, "#ff9896"), (12800, "#f7b6d2"),
        (16000, "#dbdb8d"), (25600, "#c7c7c7"), (51200, "#9edae5"),
    )
]

R56_COHORTS = [
    ("R-56 cpu-async α=0.5\nmsf (4-seed)", "#2ca02c", [
        DATA / "resnet56-cpu-async" / f"seed-{s}-cpu-async-msf" for s in (1, 2, 3, 4)
    ]),
    ("R-56 cpu-async α=1.0\nmsf (1-seed)", "#d62728", [
        DATA / "resnet56-cpu-async" / "seed-1-cpu-async-msf-alpha10",
    ]),
]


def collect(cohorts):
    """For each cohort, parse all cells; return list of (label, color, [dicts])."""
    out = []
    for label, color, cells in cohorts:
        rows = [r for r in (parse_report(c) for c in cells) if r is not None]
        out.append((label, color, rows))
    return out


# ---------------------------------------------------------------------------
# Figure 1 — R-20 vs R-56 head-to-head (bytes-axis)
# ---------------------------------------------------------------------------

def fig1_r20_vs_r56():
    r20 = [r for r in (parse_report(DATA / "passive-observation"
                                    / f"seed-{s}-cpu-async-msf") for s in range(5))
           if r is not None]
    r56_05 = [r for r in (parse_report(DATA / "resnet56-cpu-async"
                                       / f"seed-{s}-cpu-async-msf") for s in (1, 2, 3, 4))
              if r is not None]
    r56_10 = [r for r in (parse_report(DATA / "resnet56-cpu-async"
                                       / "seed-1-cpu-async-msf-alpha10"),) if r is not None]

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.0))

    cohorts = [("R-20 α=0.5\n(n=5)", "#9467bd", r20),
               ("R-56 α=0.5\n(n=4)", "#2ca02c", r56_05),
               ("R-56 α=1.0\n(n=1)", "#d62728", r56_10)]

    metrics = [("eval_pct", "final eval (%)", "Eval per cohort"),
               ("syncs", "syncs / 200 ep", "Sync count per cohort"),
               ("wall_s", "wall time (s)", "Wall time per cohort"),
               (None, "AllReduce-cost fraction (%)", "AllReduce overhead")]

    for ax_i, (key, ylabel, title) in enumerate(metrics):
        ax = axes[ax_i]
        for i, (label, color, rows) in enumerate(cohorts):
            if key:
                vals = [r[key] for r in rows]
            else:
                # AllReduce overhead = syncs * avg_sync_ms / total_s / 10 (→ %)
                vals = [r["syncs"] * r["avg_sync_ms"] / r["wall_s"] / 10 for r in rows]
            x = np.full(len(vals), i, dtype=float) + np.random.default_rng(0).uniform(-0.10, 0.10, len(vals))
            ax.scatter(x, vals, color=color, s=80, edgecolors="black", linewidths=0.5, zorder=3)
            if vals:
                ax.hlines(np.mean(vals), i - 0.20, i + 0.20, color=color, linewidth=2.5, zorder=4)
        ax.set_xticks(range(len(cohorts)))
        ax.set_xticklabels([c[0] for c in cohorts], fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("R-20 ↔ R-56 head-to-head — cpu-async msf, "
                 "bytes-axis comparison",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    out = OUT / "r20_vs_r56.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Figure 2 — Wall-time Pareto (parallel to eval-vs-syncs Pareto)
# ---------------------------------------------------------------------------

def fig2_wall_time_pareto():
    cohorts = collect(COHORTS) + collect(CLIFF_COHORTS)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))

    for ax_i, (cost_key, cost_label, title) in enumerate([
        ("syncs", "syncs / 200 ep (log)", "Eval vs sync count (existing Pareto axis)"),
        ("wall_s", "wall time (s)", "Eval vs wall time (parallel axis)"),
    ]):
        ax = axes[ax_i]
        for label, color, rows in cohorts:
            if not rows:
                continue
            xs = [r[cost_key] for r in rows]
            ys = [r["eval_pct"] for r in rows]
            mx, my = np.mean(xs), np.mean(ys)
            ax.scatter(xs, ys, color=color, alpha=0.4, s=30, edgecolors="none")
            ax.scatter([mx], [my], color=color, s=120, edgecolors="black",
                       linewidths=0.8, zorder=4, label=label.replace("\n", " "))
        if cost_key == "syncs":
            ax.set_xscale("log")
        ax.set_xlabel(cost_label)
        ax.set_ylabel("final eval (%)")
        ax.set_ylim(20, 95)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=6.5, ncol=2, framealpha=0.92)

    fig.suptitle("Pareto frontier on two cost axes (R-20, all configs)",
                 fontsize=11, fontweight="bold")
    fig.text(0.5, 0.005,
             "Wall-time spread is ≤ 215 s across all R-20 auto-tuned configs (~11 % of 1900 s) — "
             "validates the manuscript's choice of sync count as the load-bearing cost axis at this scale.",
             ha="center", fontsize=8.5, style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out = OUT / "wall_time_pareto.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Figure 3 — R1' by-k slope invariance
# ---------------------------------------------------------------------------

def fig3_r1_slope_invariance():
    """Hardcoded from existing aggregate.txt outputs — passive-observation +
    relaxed-anchor (LR=0.3 warmup window). cliff-bracket k=3200 added as
    marginal-observability annotation (only 6 events at LR=0.3, R²≈0.24)."""
    rows = [
        ("nccl-async\nmsf default",   1.558e-3, 3.03e-4, 0.779, "#1f77b4", 5),
        ("nccl-async\ntrend default", 1.626e-3, 1.63e-4, 0.777, "#aec7e8", 5),
        ("nccl-async\nmsf relaxed",   1.047e-3, 1.38e-4, 0.703, "#2ca02c", 5),
        ("nccl-async\ntrend relaxed", 1.121e-3, 1.43e-4, 0.807, "#98df8a", 5),
        # Cliff-bracket k=3200: very different — 6 events, R²≈0.24, slope sign reversed.
        ("cliff k=3200\n(boundary)", -0.198e-3, 1.43e-4, 0.235, "#ffbb78", 3),
    ]

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    xs = np.arange(len(rows))
    means = [r[1] for r in rows]
    sds = [r[2] for r in rows]
    colors = [r[4] for r in rows]
    ax.bar(xs, means, yerr=sds, color=colors, edgecolor="black", linewidth=0.6,
           capsize=4, alpha=0.85)

    # Annotate R² above (positive bars) or below (negative bar).
    for i, r in enumerate(rows):
        if r[1] > 0:
            y = r[1] + r[2] + 1.0e-4
            va = "bottom"
        else:
            y = r[1] - r[2] - 0.5e-4
            va = "top"
        ax.text(i, y, f"R²={r[3]:.2f}\n(N={r[5]})", ha="center", va=va, fontsize=8.5)

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_ylim(-7e-4, 2.4e-3)
    ax.set_ylabel(r"meta-D$_{\rm mean}$ by-k slope (ln(D)/step)")
    ax.set_title("R1' — within-cycle Lyapunov slope at LR=0.3 warmup\n"
                 "Bottom-scale physics is invariant of guard / anchor choice "
                 "(slope ≈ +1.0–1.6 × 10⁻³ across nccl-async cohorts)",
                 fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig.text(0.5, 0.005,
             "cliff k=3200 cell is at the OLS observability boundary "
             "(only 6 events per LR window, R²≈0.24); slope sign is noise-dominated, not a regime change.",
             ha="center", fontsize=8.0, style="italic")

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out = OUT / "r1_slope_invariance.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Figure 4 — Per-cell eval strip plot (every framing-valid cell)
# ---------------------------------------------------------------------------

def fig4_eval_strip_plot():
    cohorts = collect(COHORTS) + collect(CLIFF_COHORTS) + collect(R56_COHORTS)
    cohorts = [c for c in cohorts if c[2]]  # drop empty

    fig, ax = plt.subplots(figsize=(15, 5.4))
    rng = np.random.default_rng(1)
    for i, (label, color, rows) in enumerate(cohorts):
        evals = [r["eval_pct"] for r in rows]
        x = np.full(len(evals), i, dtype=float) + rng.uniform(-0.18, 0.18, len(evals))
        ax.scatter(x, evals, color=color, s=42, alpha=0.75, edgecolors="black", linewidths=0.4, zorder=3)
        if evals:
            ax.hlines(np.mean(evals), i - 0.28, i + 0.28, color=color, linewidth=2.5, zorder=4)

    ax.set_xticks(range(len(cohorts)))
    ax.set_xticklabels([c[0] for c in cohorts], fontsize=7.5, rotation=30, ha="right")
    ax.set_ylabel("final eval (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"Per-cell eval distribution across {sum(len(c[2]) for c in cohorts)} cells "
                 f"({len(cohorts)} cohorts) — horizontal bars are cohort means",
                 fontsize=10.5, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.axhspan(91.0, 92.5, color="green", alpha=0.07, zorder=0,
               label="R-20 safe-regime band")
    ax.axhspan(92.5, 93.5, color="blue", alpha=0.07, zorder=0,
               label="R-56 published band")
    ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    out = OUT / "eval_strip_plot.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Figure 5 — Cross-rank Pearson r̄ vs effective cadence
# ---------------------------------------------------------------------------

# CIFAR-10 = 50 000 samples, batch_size 64, 200 epochs, 3-GPU heterogeneous.
# Effective per-rank batches per cycle = (200 × 50 000 / 64 / 3) / syncs
TOTAL_BATCHES_PER_RANK_200EP = 200 * 50000 / 64 / 3  # ≈ 52 083


def _mean_r(row):
    try:
        return (float(row["pearson_r01"]) + float(row["pearson_r02"]) + float(row["pearson_r12"])) / 3
    except (KeyError, ValueError):
        return None


def fig5_pearson_vs_cadence():
    items = []  # (sweep_label, color, k_eff, r̄)

    # Sweeps with per_cell.csv (Pearson + syncs both available).
    for sweep, color in (("passive-observation", "#1f77b4"),
                         ("relaxed-anchor", "#2ca02c"),
                         ("cliff-bracket", "#ff7f0e")):
        for row in load_per_cell_csv(DATA / sweep):
            try:
                syncs = int(row["syncs"])
            except (KeyError, ValueError):
                continue
            if syncs <= 0:
                continue
            r_bar = _mean_r(row)
            if r_bar is None:
                continue
            mode = row.get("mode", "")
            label = f"{sweep} ({mode})" if mode else sweep
            k_eff = TOTAL_BATCHES_PER_RANK_200EP / syncs
            items.append((label, color, k_eff, r_bar, syncs))

    fig, ax = plt.subplots(figsize=(11, 4.8))
    seen = set()
    for label, color, k_eff, r_bar, syncs in items:
        legend_label = label if label not in seen else None
        seen.add(label)
        ax.scatter(k_eff, r_bar, color=color, s=46, alpha=0.7, edgecolors="black",
                   linewidths=0.4, label=legend_label)

    # Reference lines + cliff zone.
    ax.axhline(0.99, color="black", linestyle=":", alpha=0.55, label="empirical anchor r=0.99")
    ax.axhline(0.95, color="red", linestyle="--", alpha=0.55, label="framing-validity gate r=0.95")
    ax.axvspan(16000, 25600, color="orange", alpha=0.15, label="cliff zone")
    ax.axvspan(25600, 60000, color="red", alpha=0.10, label="past cliff (artifact r̄≈1)")

    ax.set_xscale("log")
    ax.set_xlabel("effective cadence (per-rank batches between syncs)")
    ax.set_ylabel("cross-rank Pearson r̄ (3-pair mean)")
    ax.set_ylim(0.94, 1.005)
    ax.set_title("Cross-rank Pearson r̄ vs effective cadence — "
                 "auto-tuned cluster (~80–130 batches/sync) sits well inside framing-valid regime",
                 fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=7.5, ncol=2, framealpha=0.92)

    fig.tight_layout()
    out = OUT / "pearson_vs_cadence.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out} (n={len(items)})")


# ---------------------------------------------------------------------------
# Figure 6 — Cliff-distance (cadence regime map)
# ---------------------------------------------------------------------------

def fig6_cliff_distance():
    """One log-x cadence axis. Shade regime bands. Mark auto-tuned cluster
    and fixed-k cells on it. Y-axis: eval %, color-coded by sweep."""
    cohorts = collect(COHORTS) + collect(CLIFF_COHORTS)

    fig, ax = plt.subplots(figsize=(13, 5.4))

    # Regime bands.
    ax.axvspan(50, 200, color="green", alpha=0.10, zorder=0)
    ax.axvspan(200, 16000, color="green", alpha=0.05, zorder=0)
    ax.axvspan(16000, 25600, color="orange", alpha=0.15, zorder=0)
    ax.axvspan(25600, 80000, color="red", alpha=0.15, zorder=0)

    # Plot each cell.
    for label, color, rows in cohorts:
        if not rows:
            continue
        for r in rows:
            if r["syncs"] <= 0:
                continue
            k_eff = TOTAL_BATCHES_PER_RANK_200EP / r["syncs"]
            ax.scatter(k_eff, r["eval_pct"], color=color, s=46, alpha=0.75,
                       edgecolors="black", linewidths=0.4, zorder=3)

    # Cohort-mean markers (one per cohort).
    label_handles = []
    seen = set()
    for label, color, rows in cohorts:
        if not rows:
            continue
        clean_label = label.replace("\n", " ")
        if clean_label in seen:
            continue
        seen.add(clean_label)
        ks = [TOTAL_BATCHES_PER_RANK_200EP / r["syncs"] for r in rows if r["syncs"] > 0]
        es = [r["eval_pct"] for r in rows]
        if ks:
            ax.scatter([np.mean(ks)], [np.mean(es)], color=color, s=160,
                       edgecolors="black", linewidths=1.2, zorder=5, label=clean_label)

    # Annotations.
    ax.text(110, 50, "auto-tune\noperating\nregion", ha="center", fontsize=9, style="italic")
    ax.text(20000, 50, "cliff\nzone", ha="center", fontsize=9, style="italic", color="darkorange")
    ax.text(45000, 50, "past\ncliff", ha="center", fontsize=9, style="italic", color="darkred")
    ax.axvline(16000, color="darkorange", linestyle="--", alpha=0.5)
    ax.axvline(25600, color="darkred", linestyle="--", alpha=0.5)

    ax.set_xscale("log")
    ax.set_xlabel("effective cadence (per-rank batches between syncs)")
    ax.set_ylabel("final eval (%)")
    ax.set_ylim(0, 95)
    ax.set_title("Cadence regime map — auto-tuned cluster operates ~80–125× below the synchronization threshold",
                 fontsize=10.5, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=7, ncol=2, framealpha=0.92)

    fig.tight_layout()
    out = OUT / "cliff_distance.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    fig1_r20_vs_r56()
    fig2_wall_time_pareto()
    fig3_r1_slope_invariance()
    fig4_eval_strip_plot()
    fig5_pearson_vs_cadence()
    fig6_cliff_distance()


if __name__ == "__main__":
    main()
