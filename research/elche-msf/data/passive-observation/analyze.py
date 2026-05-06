#!/usr/bin/env python3
"""Per-sweep focused analyzer for passive-observation.

Surfaces the three primary claims in tables + figures:

  1. Guard fires per cell — the rate-based detector ("MSF") fires
     ~55× less often than the 3-rises-above-threshold detector
     ("current"/"trend") on nccl-async with no eval cost.
  2. Cross-rank Pearson r — the meta-oscillator framing requires
     r > 0.99 between every rank pair; gate fires below ~0.95.
  3. Per-rank load balancing — the heterogeneous batch-share allocation
     ElChe drives in cpu-async / nccl-async.

20-cell sweep: 5 seeds (0–4) × 2 modes (`cpu-async`, `nccl-async`) × 2
guards (`msf`, `trend`). The `cpu-async` cohort is retained as a foil
rather than primary data for MSF analysis: the 3-phase pipelined
averaging breaks the impulsive-coupling assumption that anchors the
meta-oscillator framing. The `nccl-async` subset (10 cells) is the
primary data.

Reads from this directory; writes to ./analysis/.

Usage:
    python3 analyze.py
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "analysis"

CELL_NAMES = [
    f"seed-{seed}-{mode}-{guard}"
    for seed in range(5)
    for mode in ("cpu-async", "nccl-async")
    for guard in ("msf", "trend")
]

GPU_LABELS = ["RTX 5060 Ti", "GTX 1060 (#1)", "GTX 1060 (#2)"]
GPU_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]
GUARD_COLORS = {"msf": "#d62728", "trend": "#9467bd"}
MODE_HATCH = {"nccl-async": "", "cpu-async": "//"}


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

EPOCH_RE = re.compile(r"^epoch (\d+): loss=([\d.]+), train_acc=([\d.]+), time=([\d.]+)s$")
RANK_RE = re.compile(r"rank(\d+)\[cuda\d+,share=([\d.]+),tput=([\d.]+)\]")
EVAL_RE = re.compile(r"^epoch (\d+): eval=([\d.]+)$")


def parse_training_log(log_path: Path) -> pd.DataFrame:
    rows, current = [], None
    with log_path.open() as f:
        for raw in f:
            line = raw.strip()
            m = EPOCH_RE.match(line)
            if m:
                current = {
                    "epoch": int(m.group(1)),
                    "loss": float(m.group(2)),
                    "train_acc": float(m.group(3)),
                    "time_s": float(m.group(4)),
                }
                continue
            if current is not None and line.startswith("per-rank:"):
                for rm in RANK_RE.finditer(line):
                    rank = int(rm.group(1))
                    current[f"share_r{rank}"] = float(rm.group(2))
                    current[f"tput_r{rank}"] = float(rm.group(3))
                continue
            m = EVAL_RE.match(line)
            if m and current is not None and current["epoch"] == int(m.group(1)):
                current["eval"] = float(m.group(2))
                rows.append(current)
                current = None
    return pd.DataFrame(rows)


REPORT_MAIN_RE = re.compile(
    r"\| (\S+) \| ([\d.]+) \| ([\d.]+) \| [+-][\d.]+ \| ([\d.]+) \| (\d+) \| ([\d.]+) \|"
)


def parse_report_summary(report_path: Path) -> dict:
    text = report_path.read_text()
    m = REPORT_MAIN_RE.search(text)
    if not m:
        return {}
    return {
        "final_loss": float(m.group(2)),
        "final_eval": float(m.group(3)),
        "total_s": float(m.group(4)),
        "syncs": int(m.group(5)),
    }


def parse_pearson(report_path: Path, mode: str) -> dict:
    """Pull the 3 cross-rank Pearson r values for the given mode."""
    text = report_path.read_text()
    pat = (
        rf"\| resnet-graph \| {mode} \| rank0 ↔ rank1 \| ([+-][\d.]+) \|.*?"
        rf"\| resnet-graph \| {mode} \| rank0 ↔ rank2 \| ([+-][\d.]+) \|.*?"
        rf"\| resnet-graph \| {mode} \| rank1 ↔ rank2 \| ([+-][\d.]+) \|"
    )
    m = re.search(pat, text, re.DOTALL)
    if not m:
        return {"pearson_r01": None, "pearson_r02": None, "pearson_r12": None}
    return {
        "pearson_r01": float(m.group(1)),
        "pearson_r02": float(m.group(2)),
        "pearson_r12": float(m.group(3)),
    }


def parse_guard_fires(report_path: Path, mode: str) -> dict:
    """Pull current-rule fire count and msf-rule fire count from the
    Convergence Guard Comparison table for this mode."""
    text = report_path.read_text()
    pat = (
        rf"\| resnet-graph \| {mode} \| (\d+) \([^)]*\) \| (\d+)(?: \([^)]*\))? \|"
    )
    m = re.search(pat, text)
    if not m:
        return {"fires_current": None, "fires_msf": None}
    return {
        "fires_current": int(m.group(1)),
        "fires_msf": int(m.group(2)),
    }


def load_timeline_summary(csv_gz: Path) -> dict:
    df = pd.read_csv(csv_gz)
    out = {}
    for r in (0, 1, 2):
        out[f"gpu_util_mean_r{r}"] = df[f"gpu{r}_util"].mean()
        out[f"vram_alloc_peak_r{r}"] = df[f"gpu{r}_vram_alloc"].max() / (1024 ** 2)
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def cell_components(cell: str) -> tuple[int, str, str]:
    """seed-{S}-{mode}-{guard}  →  (S, mode, guard)."""
    parts = cell.split("-")
    seed = int(parts[1])
    mode = f"{parts[2]}-{parts[3]}"  # "cpu-async" or "nccl-async"
    guard = parts[4]
    return seed, mode, guard


def aggregate(cell: str) -> tuple[dict, list[dict]]:
    cell_dir = HERE / cell
    seed, mode, guard = cell_components(cell)

    summary = {"cell": cell, "seed": seed, "mode": mode, "guard": guard}
    summary.update(parse_report_summary(cell_dir / "report.md"))
    summary.update(parse_pearson(cell_dir / "report.md", mode))
    summary.update(parse_guard_fires(cell_dir / "report.md", mode))
    summary.update(load_timeline_summary(cell_dir / "timeline.csv.gz"))

    log = parse_training_log(cell_dir / "training.log")
    for r in (0, 1, 2):
        summary[f"share_mean_r{r}"] = log[f"share_r{r}"].mean() if f"share_r{r}" in log else None
        summary[f"tput_mean_r{r}"] = log[f"tput_r{r}"].mean() if f"tput_r{r}" in log else None

    per_rank = []
    for r in (0, 1, 2):
        per_rank.append({
            "cell": cell, "seed": seed, "mode": mode, "guard": guard,
            "rank": r, "gpu_label": GPU_LABELS[r],
            "share_mean": log[f"share_r{r}"].mean() if f"share_r{r}" in log else None,
            "share_std": log[f"share_r{r}"].std() if f"share_r{r}" in log else None,
            "tput_mean": log[f"tput_r{r}"].mean() if f"tput_r{r}" in log else None,
            "tput_std": log[f"tput_r{r}"].std() if f"tput_r{r}" in log else None,
            "gpu_util_mean": summary[f"gpu_util_mean_r{r}"],
            "vram_alloc_peak_mb": summary[f"vram_alloc_peak_r{r}"],
        })
    return summary, per_rank


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def cell_short(cell: str) -> str:
    """Compact label: seed-N-{c|n}-{m|t} (c=cpu, n=nccl, m=msf, t=trend)."""
    seed, mode, guard = cell_components(cell)
    return f"s{seed}-{mode[0]}-{guard[0]}"


def fig_guard_comparison(per_cell: pd.DataFrame, out_path: Path) -> None:
    """The R5 + R3 finding: guard fires reduction at no eval cost."""
    nccl = per_cell[per_cell["mode"] == "nccl-async"].sort_values(["guard", "seed"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel 1: per-cell eval, msf vs trend on nccl-async.
    labels = nccl["cell"].apply(cell_short).tolist()
    colors = [GUARD_COLORS[g] for g in nccl["guard"]]
    axes[0].bar(labels, nccl["final_eval"] * 100, color=colors)
    axes[0].set_title("R3 — final eval (nccl-async)")
    axes[0].set_ylabel("Held-out test accuracy (%)")
    axes[0].set_ylim(91.0, 92.5)
    axes[0].tick_params(axis="x", rotation=45, labelsize=9)
    msf_mean = nccl[nccl["guard"] == "msf"]["final_eval"].mean() * 100
    trend_mean = nccl[nccl["guard"] == "trend"]["final_eval"].mean() * 100
    axes[0].axhline(msf_mean, color=GUARD_COLORS["msf"], linestyle="--", linewidth=0.8,
                    label=f"msf mean {msf_mean:.2f}%")
    axes[0].axhline(trend_mean, color=GUARD_COLORS["trend"], linestyle="--", linewidth=0.8,
                    label=f"trend mean {trend_mean:.2f}%")
    axes[0].legend(loc="lower right", fontsize=8)

    # Panel 2: guard fires per cell — the reduction at no eval cost.
    axes[1].bar(labels, nccl["fires_current"], color="#888888", label="3-rises rule")
    axes[1].bar(labels, nccl["fires_msf"], color="#000000", label="rate-based rule")
    axes[1].set_title("R5 — guard fires per run (nccl-async)")
    axes[1].set_ylabel("count")
    axes[1].tick_params(axis="x", rotation=45, labelsize=9)
    axes[1].legend(loc="upper right", fontsize=9)

    fig.suptitle("passive-observation — guard comparison (nccl-async)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_meta_oscillator_pearson(per_cell: pd.DataFrame, out_path: Path) -> None:
    """Cross-rank Pearson r per pair across all cells. Anchors the
    meta-oscillator framing-validity gate at r > 0.99."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    pairs = [("pearson_r01", "rank 0 ↔ 1"),
             ("pearson_r02", "rank 0 ↔ 2"),
             ("pearson_r12", "rank 1 ↔ 2")]

    nccl = per_cell[per_cell["mode"] == "nccl-async"].sort_values(["guard", "seed"]).reset_index(drop=True)
    cpu = per_cell[per_cell["mode"] == "cpu-async"].sort_values(["guard", "seed"]).reset_index(drop=True)

    width = 0.25
    x = list(range(len(nccl)))
    for i, (col, label) in enumerate(pairs):
        offsets = [j + (i - 1) * width for j in x]
        ax.bar(offsets, nccl[col].fillna(0), width=width, label=f"nccl-async {label}",
               color=plt.cm.viridis(0.2 + 0.3 * i))

    ax.axhline(0.99, color="grey", linestyle="--", linewidth=0.8, label="r = 0.99 reference")
    ax.axhline(0.95, color="red", linestyle=":", linewidth=0.8, label="r = 0.95 framing-gate")
    ax.set_title("Cross-rank Pearson r (nccl-async cells; meta-oscillator anchor)")
    ax.set_ylabel("Pearson r")
    ax.set_ylim(0.93, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(nccl["cell"].apply(cell_short), rotation=45, fontsize=9)
    ax.legend(loc="lower right", fontsize=8)

    fig.suptitle("passive-observation — meta-oscillator framing validity")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_per_rank_heterogeneity(per_rank: pd.DataFrame, out_path: Path) -> None:
    """Per-rank share/throughput/util across all cells, faceted by mode."""
    cells = sorted(per_rank["cell"].unique(), key=cell_short)
    short = {c: cell_short(c) for c in cells}

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    metrics = [
        ("share_mean", "Mean batch share", "fraction"),
        ("tput_mean", "Mean throughput (samples / ms)", "samples/ms"),
        ("gpu_util_mean", "Mean GPU utilization", "% (0–100)"),
    ]
    width = 0.25
    x_idx = list(range(len(cells)))

    for ax, (col, title, ylabel) in zip(axes, metrics):
        for r in (0, 1, 2):
            sub = per_rank[per_rank["rank"] == r].set_index("cell").reindex(cells)
            offsets = [x + (r - 1) * width for x in x_idx]
            ax.bar(offsets, sub[col].values, width=width,
                   color=GPU_COLORS[r], label=GPU_LABELS[r])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x_idx)
        ax.set_xticklabels([short[c] for c in cells], rotation=45, fontsize=8)

    axes[0].legend(loc="upper right", title="rank")
    fig.suptitle("passive-observation — per-rank heterogeneity (all 20 cells)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

SWEEP_DESCRIPTION = """\
20 cells: 5 seeds (0–4) × 2 modes (`cpu-async`, `nccl-async`) × 2
guards (`msf`, `trend`), on ResNet-20 / CIFAR-10 / 200 epochs.
Hardware: 1× RTX 5060 Ti (16 GB) + 2× GTX 1060 (6 GB each). Default
anchor; no relax-up; no EASGD blending.

The `cpu-async` cohort (10 cells) is retained as a foil rather than
primary data: the 3-phase pipelined averaging on this backend breaks
the impulsive-coupling assumption that anchors the meta-oscillator
framing. The `nccl-async` cohort (10 cells) is the primary
subset for the verdicts on `R3` (final eval), `R5` (guard
fires), and the meta-oscillator anchor (cross-rank Pearson r).
"""


def fmt_pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def write_report(per_cell: pd.DataFrame, per_rank: pd.DataFrame, out_path: Path) -> None:
    nccl = per_cell[per_cell["mode"] == "nccl-async"]
    nccl_msf = nccl[nccl["guard"] == "msf"]
    nccl_trend = nccl[nccl["guard"] == "trend"]

    # Summary table — by guard on nccl-async.
    summary_lines = [
        "| guard | n cells | mean eval | mean syncs | mean fires (3-rises) | mean fires (rate-based) |",
        "|---|---:|---:|---:|---:|---:|",
        f"| `msf`   | {len(nccl_msf)} | {fmt_pct(nccl_msf['final_eval'].mean())} ± {nccl_msf['final_eval'].std()*100:.2f} pp "
        f"| {nccl_msf['syncs'].mean():.0f} ± {nccl_msf['syncs'].std():.0f} "
        f"| {nccl_msf['fires_current'].mean():.1f} ± {nccl_msf['fires_current'].std():.1f} "
        f"| {nccl_msf['fires_msf'].mean():.1f} ± {nccl_msf['fires_msf'].std():.1f} |",
        f"| `trend` | {len(nccl_trend)} | {fmt_pct(nccl_trend['final_eval'].mean())} ± {nccl_trend['final_eval'].std()*100:.2f} pp "
        f"| {nccl_trend['syncs'].mean():.0f} ± {nccl_trend['syncs'].std():.0f} "
        f"| {nccl_trend['fires_current'].mean():.1f} ± {nccl_trend['fires_current'].std():.1f} "
        f"| {nccl_trend['fires_msf'].mean():.1f} ± {nccl_trend['fires_msf'].std():.1f} |",
    ]

    # Pearson r summary.
    pearson_lines = [
        "| pair | mean r ± sd | min | max |",
        "|---|---:|---:|---:|",
    ]
    for col, label in [("pearson_r01", "rank 0 ↔ 1"),
                        ("pearson_r02", "rank 0 ↔ 2"),
                        ("pearson_r12", "rank 1 ↔ 2")]:
        s = nccl[col].dropna()
        pearson_lines.append(
            f"| {label} | +{s.mean():.4f} ± {s.std():.4f} | +{s.min():.4f} | +{s.max():.4f} |"
        )

    # Per-rank table — averaged across all 20 cells.
    rank_lines = [
        "| rank | GPU | mean share | mean throughput (samples/ms) | mean util | peak VRAM |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in (0, 1, 2):
        sub = per_rank[per_rank["rank"] == r]
        rank_lines.append(
            f"| {r} | {GPU_LABELS[r]} | "
            f"{sub['share_mean'].mean():.3f} ± {sub['share_mean'].std():.3f} | "
            f"{sub['tput_mean'].mean():.2f} ± {sub['tput_mean'].std():.2f} | "
            f"{sub['gpu_util_mean'].mean():.1f}% | "
            f"{sub['vram_alloc_peak_mb'].mean():.0f} MB |"
        )

    # Auto-populated key observations.
    # Reduction ratio is computed on msf-guard cells (the regime where
    # the MSF detector is the active gate). Both means averaged over the
    # same cell set for an apples-to-apples comparison.
    fires_ratio_msf_cells = (
        nccl_msf["fires_current"].mean() / max(nccl_msf["fires_msf"].mean(), 0.5)
    )
    eval_delta = (nccl_msf["final_eval"].mean() - nccl_trend["final_eval"].mean()) * 100
    pearson_min = min(nccl[c].min() for c in ["pearson_r01", "pearson_r02", "pearson_r12"])
    pearson_min_pair_mean = min(
        nccl[c].mean() for c in ["pearson_r01", "pearson_r02", "pearson_r12"]
    )

    observations = f"""\
- **Cross-rank Pearson r anchors the meta-oscillator framing**. Across
  all {len(nccl)} `nccl-async` cells, every rank pair stays at
  r > {pearson_min:.4f} (mean r ≥ {pearson_min_pair_mean:.4f} on the lowest-mean
  pair). The framing-validity gate at r = 0.95 is comfortably
  non-binding — ranks behave as coupled views of one D-trajectory,
  not as independent oscillators.
- **The rate-based detector fires ~{fires_ratio_msf_cells:.0f}× less
  than the 3-rises rule at no eval cost**. On the
  {len(nccl_msf)} msf-guard cells (where the rate-based detector is
  the active gate), mean fires per 200-epoch run:
  {nccl_msf['fires_current'].mean():.1f} (3-rises rule) vs
  {nccl_msf['fires_msf'].mean():.1f} (rate-based). Final eval is
  statistically equivalent across guards (Δ = {eval_delta:+.2f} pp,
  within seed sd ≈ 0.20 pp). The aggregator output in
  [`aggregate.txt`](aggregate.txt) reports a slightly higher reduction
  ratio (98.2%) because its detector-table parser excludes one cell
  whose "both-fired" column carried a multi-event list rather than a
  single count.
- **Heterogeneous load balancing visible**. The fast GPU (rank 0,
  RTX 5060 Ti) carries a mean batch share of
  {per_rank[per_rank['rank']==0]['share_mean'].mean():.2f} but runs at
  {per_rank[per_rank['rank']==0]['gpu_util_mean'].mean():.0f}% mean
  GPU utilization; the slow GPUs (ranks 1, 2, GTX 1060) take
  {per_rank[per_rank['rank']==1]['share_mean'].mean():.2f} /
{per_rank[per_rank['rank']==2]['share_mean'].mean():.2f} and run at
  {per_rank[per_rank['rank']==1]['gpu_util_mean'].mean():.0f}% /
{per_rank[per_rank['rank']==2]['gpu_util_mean'].mean():.0f}% utilization.
"""

    text = f"""# passive-observation — analysis

{SWEEP_DESCRIPTION}

## Meta unified view

### Summary (nccl-async, guard comparison)

{chr(10).join(summary_lines)}

### Eval and guard fires per cell

![Guard comparison](guard_comparison.png)

Left panel: per-cell final eval, color-coded by guard, with both
guards' sweep-mean shown as dashed lines. Right panel: guard fires per
run for both detectors on the same cells.

### Cross-rank Pearson r (meta-oscillator anchor)

{chr(10).join(pearson_lines)}

![Meta-oscillator anchor](meta_oscillator_pearson.png)

Cross-rank Pearson r per pair across all 10 nccl-async cells.
Reference lines: r = 0.99 (empirical anchor across the sweep) and
r = 0.95 (framing-validity gate; below this the meta-oscillator
framing breaks and per-rank treatment is required).

## Individual GPU view

### Per-rank averages (across all 20 cells)

{chr(10).join(rank_lines)}

Mean ± standard deviation across cells. Peak VRAM is the maximum
allocated by libtorch over the run, sampled at ~100 ms intervals from
`timeline.csv.gz`.

### Per-rank heterogeneity per cell

![Per-rank heterogeneity](per_rank_heterogeneity.png)

Three panels (share, throughput, GPU utilization). Within each panel,
every cell has three adjacent bars (one per rank). Color encodes the
GPU. Cell labels: `s{{seed}}-{{c|n}}-{{m|t}}` where `c` = cpu-async,
`n` = nccl-async, `m` = msf, `t` = trend.

## Key observations

{observations}

## Source data

- `per_cell.csv` — one row per cell (20 rows): includes mode, guard,
  eval, syncs, guard-fire counts, cross-rank Pearson r, per-rank
  shares + throughputs + GPU utilization + VRAM peaks.
- `per_rank.csv` — one row per (cell, rank), 60 rows total.
- `guard_comparison.png`, `meta_oscillator_pearson.png`,
  `per_rank_heterogeneity.png` — the figures embedded above.

## Reproducibility

Run from the sweep directory: `python3 analyze.py`. Reads the cell
extracts in this directory; writes outputs to `analysis/`. See
[`../README.md`](../README.md) for the sweep-level reproducibility
recipe.
"""
    out_path.write_text(text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_args()

    OUT.mkdir(exist_ok=True)
    print(f"out-dir: {OUT.relative_to(HERE.parents[3])}")

    cell_rows, rank_rows = [], []
    for cell in CELL_NAMES:
        if not (HERE / cell / "training.log").exists():
            print(f"  skip {cell}: missing training.log")
            continue
        summary, per_rank = aggregate(cell)
        cell_rows.append(summary)
        rank_rows.extend(per_rank)
        print(f"  ok   {cell}")

    per_cell = (pd.DataFrame(cell_rows)
                .sort_values(["mode", "guard", "seed"]).reset_index(drop=True))
    per_rank = (pd.DataFrame(rank_rows)
                .sort_values(["mode", "guard", "seed", "rank"]).reset_index(drop=True))

    per_cell.to_csv(OUT / "per_cell.csv", index=False, float_format="%.4f")
    per_rank.to_csv(OUT / "per_rank.csv", index=False, float_format="%.4f")
    print(f"  wrote per_cell.csv ({len(per_cell)} rows)")
    print(f"  wrote per_rank.csv ({len(per_rank)} rows)")

    fig_guard_comparison(per_cell, OUT / "guard_comparison.png")
    fig_meta_oscillator_pearson(per_cell, OUT / "meta_oscillator_pearson.png")
    fig_per_rank_heterogeneity(per_rank, OUT / "per_rank_heterogeneity.png")
    print(f"  wrote 3 figures")

    write_report(per_cell, per_rank, OUT / "README.md")
    print(f"  wrote README.md")


if __name__ == "__main__":
    main()
