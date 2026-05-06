#!/usr/bin/env python3
"""Per-sweep focused analyzer for cliff-bracket (fixed-k probe).

Surfaces the cliff-bracket finding: the Pecora-Carroll
synchronization threshold sits between k=16000 (last fully safe, all
3 seeds within ~1.3 pp of safe-regime mean) and k=25600 (first
bimodal seed split, within-cell range 35.1 pp). Cadence is pinned at
exactly `k` batches per cycle via
`--min-anchor=k --max-anchor=k --guard none`.

Three deliverables:

  1. Cliff-bracket scatter — eval vs k with per-seed markers and
     per-k mean, log-scale x-axis. Bimodality at k=25600 is
     immediately visible.
  2. Adjacent-cell delta bar — Δeval between consecutive k values,
     localizing the cliff edge as the first transition with a soft
     drop (>1 pp) and the cliff itself as the first transition with
     >30 pp drop.
  3. Per-rank heterogeneity — share / throughput / GPU utilization
     across cells, faceted by k (lets the reader see whether the
     per-rank dynamics change in a regime-dependent way).

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

K_VALUES = (3200, 6400, 12800, 16000, 25600, 51200)
CELL_NAMES = [
    f"seed-{seed}-fixed-k-{k}"
    for seed in (0, 1, 2)
    for k in K_VALUES
]

GPU_LABELS = ["RTX 5060 Ti", "GTX 1060 (#1)", "GTX 1060 (#2)"]
GPU_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]
SEED_MARKERS = {0: "o", 1: "s", 2: "^"}
SEED_COLORS = {0: "#4c72b0", 1: "#dd8452", 2: "#55a868"}
MODE = "nccl-async"


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

EPOCH_RE = re.compile(r"^epoch (\d+): loss=([\d.]+), train_acc=([\d.]+), time=([\d.]+)s$")
RANK_RE = re.compile(r"rank(\d+)\[cuda\d+,share=([\d.]+),tput=([\d.]+)\]")
EVAL_RE = re.compile(r"^epoch (\d+): eval=([\d.]+)$")
REPORT_MAIN_RE = re.compile(
    r"\| (\S+) \| ([\d.]+) \| ([\d.]+) \| [+-][\d.]+ \| ([\d.]+) \| (\d+) \| ([\d.]+) \|"
)


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

def cell_components(cell: str) -> tuple[int, int]:
    """seed-{S}-fixed-k-{K} → (S, K)."""
    parts = cell.split("-")
    return int(parts[1]), int(parts[4])


def aggregate(cell: str) -> tuple[dict, list[dict]]:
    cell_dir = HERE / cell
    seed, k = cell_components(cell)

    summary = {"cell": cell, "seed": seed, "k": k, "mode": MODE}
    summary.update(parse_report_summary(cell_dir / "report.md"))
    summary.update(parse_pearson(cell_dir / "report.md", MODE))
    summary.update(load_timeline_summary(cell_dir / "timeline.csv.gz"))

    log = parse_training_log(cell_dir / "training.log")
    for r in (0, 1, 2):
        summary[f"share_mean_r{r}"] = log[f"share_r{r}"].mean() if f"share_r{r}" in log else None
        summary[f"tput_mean_r{r}"] = log[f"tput_r{r}"].mean() if f"tput_r{r}" in log else None

    per_rank = []
    for r in (0, 1, 2):
        per_rank.append({
            "cell": cell, "seed": seed, "k": k, "rank": r, "gpu_label": GPU_LABELS[r],
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

def fig_cliff_bracket(per_cell: pd.DataFrame, out_path: Path) -> None:
    """Eval vs k with per-seed markers + per-k mean line.
    Bimodality at k=25600 is immediately visible."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for seed in sorted(per_cell["seed"].unique()):
        sub = per_cell[per_cell["seed"] == seed].sort_values("k")
        ax.scatter(sub["k"], sub["final_eval"] * 100,
                   marker=SEED_MARKERS[seed], s=70, alpha=0.85,
                   color=SEED_COLORS[seed], label=f"seed {seed}",
                   edgecolor="black", linewidth=0.6, zorder=3)

    # Per-k mean as horizontal tick + range as a vertical line per k.
    means = per_cell.groupby("k")["final_eval"].agg(["mean", "min", "max"]).sort_index()
    for k, row in means.iterrows():
        ax.plot([k * 0.92, k * 1.08], [row["mean"] * 100] * 2,
                color="black", linewidth=1.2, zorder=2)
        ax.plot([k, k], [row["min"] * 100, row["max"] * 100],
                color="grey", linewidth=0.8, alpha=0.5, zorder=1)

    # Cliff annotation.
    ax.axvspan(16000, 25600, alpha=0.12, color="red", zorder=0,
               label="bracket (16000 ↔ 25600)")

    # Reference line at random-chance accuracy on CIFAR-10.
    ax.axhline(10.0, color="grey", linestyle=":", linewidth=0.7,
               label="random-chance (10%)")

    ax.set_xscale("log")
    ax.set_xticks(K_VALUES)
    ax.set_xticklabels([str(k) for k in K_VALUES])
    ax.set_xlabel("k (cadence, batches per cycle)")
    ax.set_ylabel("Final eval (%)")
    ax.set_title("Cliff bracket — eval vs fixed-k cadence (3 seeds per k)")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_per_k_summary(per_cell: pd.DataFrame, out_path: Path) -> None:
    """Per-k mean eval as bar with sd error bars + range as text.
    Direct visual companion to the per-k summary table."""
    means = per_cell.groupby("k")["final_eval"].mean().sort_index() * 100
    sds = per_cell.groupby("k")["final_eval"].std().sort_index() * 100
    ranges = (per_cell.groupby("k")["final_eval"].max() -
              per_cell.groupby("k")["final_eval"].min()).sort_index() * 100

    # Color by regime: safe / pre-cliff / cliff / collapsed.
    def regime_color(k: int) -> str:
        if k <= 12800:
            return "#4c956c"   # safe
        if k == 16000:
            return "#f4a261"   # pre-cliff (soft drop)
        if k == 25600:
            return "#e76f51"   # bimodal cliff edge
        return "#c44536"       # collapsed

    colors = [regime_color(k) for k in K_VALUES]

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [str(k) for k in K_VALUES]
    bars = ax.bar(labels, means.values, yerr=sds.values, color=colors,
                  edgecolor="black", linewidth=0.5, capsize=4)

    # Text annotation: range above each bar.
    for bar, k, rng in zip(bars, K_VALUES, ranges.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + sds[k] + 2,
                f"range: {rng:.1f} pp", ha="center", fontsize=8, color="black")

    # Random-chance reference.
    ax.axhline(10.0, color="grey", linestyle=":", linewidth=0.7, label="random-chance (10%)")

    ax.set_xlabel("k (cadence, batches per cycle)")
    ax.set_ylabel("Mean final eval (%) ± sd")
    ax.set_title("Per-k summary — mean ± sd + within-cell range\n"
                 "(green: safe / orange: pre-cliff / red: cliff)")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_adjacent_deltas(per_cell: pd.DataFrame, out_path: Path) -> None:
    """Δeval between consecutive k values. Localizes the cliff edge."""
    means = per_cell.groupby("k")["final_eval"].mean().sort_index() * 100
    transitions = []
    deltas = []
    colors = []
    for i in range(1, len(K_VALUES)):
        k_prev, k_now = K_VALUES[i - 1], K_VALUES[i]
        delta = means[k_now] - means[k_prev]
        transitions.append(f"{k_prev}\n→\n{k_now}")
        deltas.append(delta)
        if delta > -1:
            colors.append("#4c956c")  # flat
        elif delta > -10:
            colors.append("#f4a261")  # soft drop
        else:
            colors.append("#c44536")  # cliff edge

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(transitions, deltas, color=colors, edgecolor="black", linewidth=0.5)

    for i, d in enumerate(deltas):
        ax.text(i, d - 1 if d < 0 else d + 0.3,
                f"{d:+.1f} pp", ha="center", fontsize=9,
                va="top" if d < 0 else "bottom")

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Δ mean eval (pp)")
    ax.set_title("Adjacent-cell deltas — cliff edge localization\n"
                 "(green: flat / orange: soft drop / red: cliff)")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_per_rank_heterogeneity(per_rank: pd.DataFrame, out_path: Path) -> None:
    """Per-rank metrics by k (faceted), averaged across seeds."""
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    metrics = [
        ("share_mean", "Mean batch share", "fraction"),
        ("tput_mean", "Mean throughput (samples / ms)", "samples/ms"),
        ("gpu_util_mean", "Mean GPU utilization", "% (0–100)"),
    ]
    width = 0.25
    x_idx = list(range(len(K_VALUES)))

    for ax, (col, title, ylabel) in zip(axes, metrics):
        for r in (0, 1, 2):
            means = []
            for k in K_VALUES:
                sub = per_rank[(per_rank["rank"] == r) & (per_rank["k"] == k)]
                means.append(sub[col].mean())
            offsets = [x + (r - 1) * width for x in x_idx]
            ax.bar(offsets, means, width=width, color=GPU_COLORS[r],
                   label=GPU_LABELS[r])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x_idx)
        ax.set_xticklabels([str(k) for k in K_VALUES])

    axes[2].set_xlabel("k (cadence, batches per cycle)")
    axes[0].legend(loc="upper right", title="rank")
    fig.suptitle("cliff-bracket — per-rank heterogeneity by k (averaged across 3 seeds)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

SWEEP_DESCRIPTION = """\
18 cells = 3 seeds × 6 fixed-k values, on ResNet-20 / CIFAR-10 /
200 epochs / nccl-async / 3-GPU heterogeneous (1× RTX 5060 Ti +
2× GTX 1060). Cadence is pinned at exactly `k` batches per cycle via
`--min-anchor=k --max-anchor=k --guard none`. The convergence guard
is disabled for the duration of every cell, so the auto-tune cannot
steer cadence away from the pinned value.

The sweep walks `k` across an order of magnitude (3200 → 51200) to
locate the synchronization threshold empirically. ElChe's auto-tuned
cadence saturates near `k ≈ 200` in normal operation; this sweep
operates 16–250× above that setpoint.
"""


def fmt_pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def write_report(per_cell: pd.DataFrame, per_rank: pd.DataFrame, out_path: Path) -> None:
    # Per-k aggregate table.
    cell_lines = ["| k | seed 0 | seed 1 | seed 2 | mean ± sd | range | syncs (mean) |",
                  "|---:|---:|---:|---:|---:|---:|---:|"]
    for k in K_VALUES:
        sub = per_cell[per_cell["k"] == k].sort_values("seed")
        evals = {row["seed"]: row["final_eval"] for _, row in sub.iterrows()}
        per_seed = " | ".join(fmt_pct(evals.get(s, float("nan"))) for s in (0, 1, 2))
        eval_mean = sub["final_eval"].mean()
        eval_sd = sub["final_eval"].std()
        eval_range = (sub["final_eval"].max() - sub["final_eval"].min()) * 100
        sync_mean = sub["syncs"].mean()
        cell_lines.append(
            f"| {k} | {per_seed} | {fmt_pct(eval_mean)} ± {eval_sd*100:.2f} pp "
            f"| {eval_range:.1f} pp | {sync_mean:.0f} |"
        )

    # Adjacent-cell deltas.
    means = per_cell.groupby("k")["final_eval"].mean().sort_index() * 100
    delta_lines = ["| transition | Δ mean eval | verdict |",
                   "|---|---:|---|"]
    for i in range(1, len(K_VALUES)):
        kp, kn = K_VALUES[i - 1], K_VALUES[i]
        d = means[kn] - means[kp]
        if d > -1:
            verdict = "flat"
        elif d > -10:
            verdict = "soft drop (>1 pp)"
        elif d > -30:
            verdict = "steep drop"
        else:
            verdict = "cliff edge (>30 pp)"
        delta_lines.append(f"| k = {kp} → {kn} | {d:+.2f} pp | {verdict} |")

    # Per-rank table — averaged across all 18 cells.
    rank_lines = ["| rank | GPU | mean share | mean throughput (samples/ms) | mean util | peak VRAM |",
                  "|---|---|---:|---:|---:|---:|"]
    for r in (0, 1, 2):
        sub = per_rank[per_rank["rank"] == r]
        rank_lines.append(
            f"| {r} | {GPU_LABELS[r]} | "
            f"{sub['share_mean'].mean():.3f} ± {sub['share_mean'].std():.3f} | "
            f"{sub['tput_mean'].mean():.2f} ± {sub['tput_mean'].std():.2f} | "
            f"{sub['gpu_util_mean'].mean():.1f}% | "
            f"{sub['vram_alloc_peak_mb'].mean():.0f} MB |"
        )

    # Computed observations.
    safe_means = per_cell[per_cell["k"].isin([3200, 6400, 12800])].groupby("k")["final_eval"].mean()
    safe_monotone = all(safe_means.iloc[i] >= safe_means.iloc[i + 1]
                        for i in range(len(safe_means) - 1))
    bimodal_cell = per_cell[per_cell["k"] == 25600]
    bimodal_range = (bimodal_cell["final_eval"].max() - bimodal_cell["final_eval"].min()) * 100
    collapse_mean = per_cell[per_cell["k"] == 51200]["final_eval"].mean() * 100
    safe_to_pre_cliff = means[16000] - means[12800]
    pre_cliff_to_bimodal = means[25600] - means[16000]
    bimodal_to_collapse = means[51200] - means[25600]

    observations = f"""\
- **The synchronization threshold sits between k = 16000 and k = 25600**.
  At k = 16000 all 3 seeds finish within ~1.3 pp of the safe-regime
  mean (mean {fmt_pct(per_cell[per_cell['k']==16000]['final_eval'].mean())}).
  At k = 25600 the within-cell range jumps to {bimodal_range:.1f} pp:
  three independently-seeded runs landing at distinct evals is the
  basin-of-attraction signature of a noise-perturbed system at the
  threshold.
- **Hard collapse at k = 51200**.  The k = 51200 cell sees only 1
  AllReduce event in the 200-epoch run; mean eval falls to
  {collapse_mean:.2f}% with two of three seeds essentially at random
  chance (10%).
- **Adjacent-cell delta gradient localizes the cliff edge**. The first
  soft drop (>1 pp) happens at k = 12800 → 16000
  ({safe_to_pre_cliff:+.2f} pp); the first major drop sits at
  k = 16000 → 25600 ({pre_cliff_to_bimodal:+.2f} pp); the cliff edge
  itself (Δ > 30 pp) is k = 25600 → 51200
  ({bimodal_to_collapse:+.2f} pp).
- **No eval peak above the auto-tune setpoint in the safe regime**.
  Across k ∈ {{3200, 6400, 12800}}, eval is {"monotone non-increasing" if safe_monotone else "non-monotone"}
  ({fmt_pct(safe_means[3200])} → {fmt_pct(safe_means[6400])} →
  {fmt_pct(safe_means[12800])}). The "ride the limit" hypothesis —
  that eval has a peak somewhere between the auto-tune's natural
  setpoint and the cliff — is not supported. The safe controller story
  is "stay below the cliff", not "target a peak".
- **Cross-rank Pearson r becomes uninformative past the cliff**. Cells
  with ≤ 2 within-training sync events have only 2 data points to
  correlate, so r is mathematically ±1 by construction; reported
  values for k ≥ 25600 should be read as a sample-size artifact, not
  a framing-validity signal.
"""

    text = f"""# cliff-bracket — analysis

{SWEEP_DESCRIPTION}

## Meta unified view

### Per-k summary

{chr(10).join(cell_lines)}

Per-cell evals shown for each of the 3 seeds, plus mean ± sd, range,
and mean sync count over the 200-epoch run.

![Per-k summary](per_k_summary.png)

Bar height is the per-k mean eval; error bars are seed-to-seed
standard deviation; "range" annotation above each bar is the within-
cell max minus min. Color encodes the regime classification: green =
safe (k ≤ 12800), orange = pre-cliff soft-drop (k = 16000), red =
cliff edge or beyond (k ≥ 25600).

### Eval vs k (cliff bracket)

![Cliff bracket](cliff_bracket.png)

X-axis is log-scaled to span the order-of-magnitude k range. Each
seed has its own marker; per-k mean is the black tick, per-k range is
the grey vertical line. The shaded region marks the cliff bracket
(k = 16000 ↔ k = 25600) — the safe / unsafe boundary.

### Adjacent-cell deltas

{chr(10).join(delta_lines)}

![Adjacent-cell deltas](adjacent_deltas.png)

Δ between consecutive k values, color-coded by verdict. The cliff
edge is the first transition with Δ > 30 pp.

## Individual GPU view

### Per-rank averages (across all 18 cells)

{chr(10).join(rank_lines)}

Mean ± standard deviation across cells. Note that under fixed-k +
guard-none the cadence is constant by construction, so the
heterogeneity here is driven entirely by ElChe's progressive
batch-share dispatch (still active under fixed-k).

### Per-rank metrics by k

![Per-rank heterogeneity](per_rank_heterogeneity.png)

Three panels (share, throughput, GPU utilization). Within each panel,
each k has three adjacent bars (one per rank). Cells past the cliff
(k ≥ 25600) reach only ≤ 2 sync events in the 200-epoch run, which
shows up as a regime change in the per-rank dynamics.

## Key observations

{observations}

## Source data

- `per_cell.csv` — 18 rows (one per cell), with seed, k, eval,
  syncs, Pearson r, per-rank metrics.
- `per_rank.csv` — 54 rows = 18 cells × 3 ranks.
- `cliff_bracket.png`, `adjacent_deltas.png`,
  `per_rank_heterogeneity.png` — the figures embedded above.
- The full per-LR-window aggregator output (slopes, R², per-rank
  ratio, etc.) is in [`aggregate.txt`](aggregate.txt).

## Reproducibility

Run from this directory: `python3 analyze.py`. Reads cell extracts in
this directory; writes outputs to `analysis/`. See
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
                .sort_values(["k", "seed"]).reset_index(drop=True))
    per_rank = (pd.DataFrame(rank_rows)
                .sort_values(["k", "seed", "rank"]).reset_index(drop=True))

    per_cell.to_csv(OUT / "per_cell.csv", index=False, float_format="%.4f")
    per_rank.to_csv(OUT / "per_rank.csv", index=False, float_format="%.4f")
    print(f"  wrote per_cell.csv ({len(per_cell)} rows)")
    print(f"  wrote per_rank.csv ({len(per_rank)} rows)")

    fig_per_k_summary(per_cell, OUT / "per_k_summary.png")
    fig_cliff_bracket(per_cell, OUT / "cliff_bracket.png")
    fig_adjacent_deltas(per_cell, OUT / "adjacent_deltas.png")
    fig_per_rank_heterogeneity(per_rank, OUT / "per_rank_heterogeneity.png")
    print(f"  wrote 4 figures")

    write_report(per_cell, per_rank, OUT / "README.md")
    print(f"  wrote README.md")


if __name__ == "__main__":
    main()
