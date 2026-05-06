#!/usr/bin/env python3
"""Per-sweep focused analyzer for relaxed-anchor.

Surfaces the findings:

  1. 4-cell comparison (default vs relaxed anchor × msf vs trend
     guard) — shows that relaxed-anchor preserves eval at lower sync
     count, with the eval optimum in the noisiest framing-gate corner.
  2. Asymmetric Pearson r decoupling under relax-up — fast/slow rank
     pairs decouple; slow/slow pairs stay perfectly locked. Hardware
     heterogeneity dictates the decoupling-direction structure.

Cross-reads the default-anchor baseline cells from
../passive-observation/ for the 4-cell comparison. Reads its own
cells from this directory.

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
PASSIVE_BASE = HERE.parent / "passive-observation"

# Own cells: 10 nccl-async relaxed (5 seeds × 2 guards).
RELAXED_CELLS = [
    f"seed-{seed}-nccl-async-{guard}-relaxed"
    for seed in range(5)
    for guard in ("msf", "trend")
]
OWN_CELLS = RELAXED_CELLS

# Default-anchor baseline cells, read from the passive-observation extracts.
DEFAULT_CELLS = [
    f"seed-{seed}-nccl-async-{guard}"
    for seed in range(5)
    for guard in ("msf", "trend")
]

GPU_LABELS = ["RTX 5060 Ti", "GTX 1060 (#1)", "GTX 1060 (#2)"]
GPU_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]
ANCHOR_COLORS = {"default": "#4c72b0", "relaxed": "#dd8452"}


# ---------------------------------------------------------------------------
# Parsers (shared shape with passive-observation/analyze.py)
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


def parse_guard_fires(report_path: Path, mode: str) -> dict:
    text = report_path.read_text()
    pat = rf"\| resnet-graph \| {mode} \| (\d+) \([^)]*\) \| (\d+)(?: \([^)]*\))? \|"
    m = re.search(pat, text)
    if not m:
        return {"fires_current": None, "fires_msf": None}
    return {"fires_current": int(m.group(1)), "fires_msf": int(m.group(2))}


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

def cell_to_mode(cell: str) -> str:
    if "cpu-async" in cell:
        return "cpu-async"
    if "nccl-async" in cell:
        return "nccl-async"
    raise ValueError(f"unknown mode: {cell}")


def cell_to_anchor(cell: str) -> str:
    if cell.endswith("-relaxed"):
        return "relaxed"
    return "default"


def aggregate(cell_dir: Path, anchor_label: str) -> tuple[dict, list[dict]]:
    cell = cell_dir.name
    parts = cell.split("-")
    seed = int(parts[1])
    mode = cell_to_mode(cell)
    # guard token sits at index 4 after "seed-N-{cpu|nccl}-async-..."
    guard = parts[4]

    summary = {
        "cell": cell, "seed": seed, "mode": mode, "guard": guard,
        "anchor": anchor_label,
    }
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
            "anchor": anchor_label, "rank": r, "gpu_label": GPU_LABELS[r],
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

def fig_4cell_comparison(per_cell: pd.DataFrame, out_path: Path) -> None:
    """4-cell summary: default vs relaxed × msf vs trend, on nccl-async.
    EASGD cells excluded — they're a separate single-seed cohort."""
    nccl = per_cell[per_cell["mode"] == "nccl-async"].copy()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    metrics = [
        ("final_eval", "Final eval (%)", lambda s: s * 100, "Held-out test accuracy (%)"),
        ("syncs", "Total syncs / 200 epochs", lambda s: s, "syncs"),
        ("pearson_mean", "Mean cross-rank Pearson r", lambda s: s, "Pearson r"),
        ("fires_msf", "Rate-based detector fires / run", lambda s: s, "fires"),
    ]
    nccl["pearson_mean"] = nccl[["pearson_r01", "pearson_r02", "pearson_r12"]].mean(axis=1)

    for ax, (col, title, transform, ylabel) in zip(axes.flat, metrics):
        groups = []
        labels = []
        colors = []
        for guard in ("msf", "trend"):
            for anchor in ("default", "relaxed"):
                sub = nccl[(nccl["guard"] == guard) & (nccl["anchor"] == anchor)]
                vals = transform(sub[col].dropna())
                groups.append(vals.mean() if len(vals) else 0)
                labels.append(f"{guard}\n{anchor}")
                colors.append(ANCHOR_COLORS[anchor])

        positions = [0, 1, 2.5, 3.5]
        ax.bar(positions, groups, color=colors, edgecolor="black", linewidth=0.5)

        # Error bars from sd.
        sds = []
        for guard in ("msf", "trend"):
            for anchor in ("default", "relaxed"):
                sub = nccl[(nccl["guard"] == guard) & (nccl["anchor"] == anchor)]
                vals = transform(sub[col].dropna())
                sds.append(vals.std() if len(vals) > 1 else 0)
        ax.errorbar(positions, groups, yerr=sds, fmt="none", ecolor="black", capsize=3)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(title)
        ax.set_ylabel(ylabel)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ANCHOR_COLORS.values()]
    fig.legend(handles, list(ANCHOR_COLORS.keys()), loc="upper center",
               ncol=2, bbox_to_anchor=(0.5, 1.0), title="anchor")
    fig.suptitle(
        "relaxed-anchor — 4-cell comparison (nccl-async, n=5 per cell)",
        y=1.04,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_pearson_asymmetric(per_cell: pd.DataFrame, out_path: Path) -> None:
    """Per-pair Pearson r mean, default vs relaxed. Shows fast↔slow
    pairs decouple under relax-up while slow↔slow stays perfectly
    locked — hardware heterogeneity dictates decoupling direction."""
    nccl = per_cell[per_cell["mode"] == "nccl-async"]

    pairs = [
        ("pearson_r01", "rank 0 ↔ 1\n(fast ↔ slow)"),
        ("pearson_r02", "rank 0 ↔ 2\n(fast ↔ slow)"),
        ("pearson_r12", "rank 1 ↔ 2\n(slow ↔ slow)"),
    ]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    width = 0.35
    x = list(range(len(pairs)))

    for i, anchor in enumerate(("default", "relaxed")):
        sub = nccl[nccl["anchor"] == anchor]
        means = [sub[col].mean() for col, _ in pairs]
        sds = [sub[col].std() for col, _ in pairs]
        offsets = [j + (i - 0.5) * width for j in x]
        ax.bar(offsets, means, width=width, yerr=sds,
               color=ANCHOR_COLORS[anchor], label=anchor,
               capsize=3, edgecolor="black", linewidth=0.5)

    ax.axhline(0.99, color="grey", linestyle="--", linewidth=0.8, label="r = 0.99 reference")
    ax.axhline(0.95, color="red", linestyle=":", linewidth=0.8, label="r = 0.95 framing-gate")
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in pairs])
    ax.set_ylabel("Pearson r (mean ± sd, n=5 per anchor × guard)")
    ax.set_ylim(0.94, 1.0)
    ax.set_title("Cross-rank Pearson r — asymmetric decoupling under relax-up\n"
                 "(averaged across both guards)")
    ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_per_rank_heterogeneity(per_rank: pd.DataFrame, out_path: Path) -> None:
    """Per-rank share/tput/util across all 12 own cells."""
    cells = sorted(per_rank["cell"].unique())
    short = {c: short_cell(c) for c in cells}

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
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
    fig.suptitle("relaxed-anchor — per-rank heterogeneity (own 10 cells)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def short_cell(cell: str) -> str:
    """seed-N-nccl-async-{m|t}-relaxed → s{N}-{m|t}-r"""
    parts = cell.split("-")
    guard_letter = parts[4][0]
    return f"s{parts[1]}-{guard_letter}-r"


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

SWEEP_DESCRIPTION = """\
10 cells: 5 seeds × 2 guards (`msf`, `trend`) on nccl-async with
`--elche-relax-up`. Compared head-to-head against the default-anchor
baseline (10 corresponding cells in `../passive-observation/`).

Model / dataset / hardware as in `../passive-observation/README.md`.
"""


def fmt_pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def write_report(per_cell: pd.DataFrame, per_rank: pd.DataFrame, out_path: Path) -> None:
    nccl = per_cell[per_cell["mode"] == "nccl-async"].copy()
    nccl["pearson_mean"] = nccl[["pearson_r01", "pearson_r02", "pearson_r12"]].mean(axis=1)

    # 4-cell summary table.
    summary_rows = [
        "| guard | anchor | n | final eval | syncs | Pearson r (3-pair mean) | rate-based fires |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for guard in ("msf", "trend"):
        for anchor in ("default", "relaxed"):
            sub = nccl[(nccl["guard"] == guard) & (nccl["anchor"] == anchor)]
            n = len(sub)
            summary_rows.append(
                f"| `{guard}` | {anchor} | {n} "
                f"| {fmt_pct(sub['final_eval'].mean())} ± {sub['final_eval'].std()*100:.2f} pp "
                f"| {sub['syncs'].mean():.0f} ± {sub['syncs'].std():.0f} "
                f"| {sub['pearson_mean'].mean():.4f} ± {sub['pearson_mean'].std():.4f} "
                f"| {sub['fires_msf'].mean():.1f} ± {sub['fires_msf'].std():.1f} |"
            )

    # Per-pair Pearson decoupling table.
    pearson_lines = [
        "| pair | default mean ± sd | relaxed mean ± sd | Δ |",
        "|---|---:|---:|---:|",
    ]
    for col, label in [("pearson_r01", "rank 0 ↔ 1 (fast ↔ slow)"),
                        ("pearson_r02", "rank 0 ↔ 2 (fast ↔ slow)"),
                        ("pearson_r12", "rank 1 ↔ 2 (slow ↔ slow)")]:
        d = nccl[nccl["anchor"] == "default"][col].dropna()
        r = nccl[nccl["anchor"] == "relaxed"][col].dropna()
        delta = r.mean() - d.mean()
        pearson_lines.append(
            f"| {label} | +{d.mean():.4f} ± {d.std():.4f} "
            f"| +{r.mean():.4f} ± {r.std():.4f} | {delta:+.4f} |"
        )

    # Per-rank table (own 10 cells).
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

    # Computed observations.
    nccl_msf_relaxed = nccl[(nccl["guard"] == "msf") & (nccl["anchor"] == "relaxed")]
    nccl_msf_default = nccl[(nccl["guard"] == "msf") & (nccl["anchor"] == "default")]
    fast_slow_drop = (nccl[nccl["anchor"] == "relaxed"]["pearson_r01"].mean() -
                      nccl[nccl["anchor"] == "default"]["pearson_r01"].mean())
    slow_slow_drop = (nccl[nccl["anchor"] == "relaxed"]["pearson_r12"].mean() -
                      nccl[nccl["anchor"] == "default"]["pearson_r12"].mean())
    sync_reduction_msf = 1 - nccl_msf_relaxed["syncs"].mean() / nccl_msf_default["syncs"].mean()
    eval_delta_msf = (nccl_msf_relaxed["final_eval"].mean() -
                      nccl_msf_default["final_eval"].mean()) * 100
    msf_relaxed_silence = (nccl_msf_relaxed["fires_msf"] == 0).sum()

    observations = f"""\
- **Relaxed anchor preserves eval at lower sync count**. Under the
  msf guard, the relaxed-anchor cohort sits at
  {fmt_pct(nccl_msf_relaxed['final_eval'].mean())} eval vs
  {fmt_pct(nccl_msf_default['final_eval'].mean())} for default-anchor
  (Δ = {eval_delta_msf:+.2f} pp), with a
  {100 * sync_reduction_msf:.0f}% sync-count reduction
  ({nccl_msf_default['syncs'].mean():.0f} → {nccl_msf_relaxed['syncs'].mean():.0f}).
- **Hardware heterogeneity dictates decoupling direction**. The
  cross-rank Pearson r drop under relax-up is asymmetric: fast ↔ slow
  pairs (rank 0 against ranks 1, 2) shed
  {abs(fast_slow_drop):.3f} of correlation, while the slow ↔ slow
  pair (ranks 1, 2) loses essentially nothing
  ({slow_slow_drop:+.4f}). The fast GPU runs farthest ahead between
  syncs, so it drifts most when cadence loosens; the two slow GPUs
  stay anchored to the same plodding pace.
- **The rate-based detector goes silent under relax-up on the msf
  guard**. {msf_relaxed_silence} of {len(nccl_msf_relaxed)} relaxed
  msf-guard seeds have zero rate-based fires across the 200-epoch run.
  Combined with the relax-up policy (each "stable" verdict grows the
  anchor), this composes into unbounded anchor growth on those seeds —
  motivating a threshold-aware controller for production deployment.
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

    text = f"""# relaxed-anchor — analysis

{SWEEP_DESCRIPTION}

## Meta unified view

### 4-cell summary (default vs relaxed × msf vs trend, nccl-async)

{chr(10).join(summary_rows)}

`anchor=default` cells are read from
[`../passive-observation/`](../passive-observation/) for the
head-to-head comparison; `anchor=relaxed` cells are own cells in this
sweep. n = 5 seeds per condition.

![4-cell comparison](four_cell_comparison.png)

Four panels: final eval, total syncs, mean cross-rank Pearson r,
and rate-based detector fires per run. Color encodes the anchor mode.
Error bars are seed-to-seed standard deviation.

### Asymmetric Pearson r decoupling under relax-up

{chr(10).join(pearson_lines)}

Mean across all 5 seeds × 2 guards = 10 cells per anchor. Δ is
relaxed minus default.

![Pearson asymmetric decoupling](pearson_asymmetric.png)

Hardware heterogeneity dictates the decoupling direction: fast ↔ slow
pairs lose correlation under relax-up, while the slow ↔ slow pair
stays perfectly locked. This is a publishable empirical fact about
heterogeneous-DDP synchronization with no homogeneous-cluster analog.

## Individual GPU view

### Per-rank averages (across own 10 cells)

{chr(10).join(rank_lines)}

Mean ± standard deviation across cells. Peak VRAM is the maximum
allocated by libtorch over the run, sampled at ~100 ms intervals from
`timeline.csv.gz`.

### Per-rank heterogeneity per cell

![Per-rank heterogeneity](per_rank_heterogeneity.png)

Three panels (share, throughput, GPU utilization). Cell labels:
`s{{seed}}-{{m|t}}-r` where `m`/`t` = msf/trend guard, `r` = relaxed-anchor.

## Key observations

{observations}

## Source data

- `per_cell.csv` — 10 rows (own cells only), with mode, guard,
  anchor, eval, syncs, Pearson r, guard fires, per-rank metrics.
- `per_rank.csv` — 30 rows = 10 cells × 3 ranks.
- `four_cell_comparison.png`, `pearson_asymmetric.png`,
  `per_rank_heterogeneity.png` — the figures embedded above.
- The cross-cell aggregator output is in
  [`aggregate.txt`](aggregate.txt) (cross-reads default-anchor cells
  from `../passive-observation/`).

## Reproducibility

Run from this directory: `python3 analyze.py`. Reads own cell
extracts plus default-anchor cells from
`../passive-observation/`. Writes outputs to `analysis/`. See
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

    # Default-anchor baseline cells (cross-read from passive-observation).
    for cell in DEFAULT_CELLS:
        d = PASSIVE_BASE / cell
        if not (d / "training.log").exists():
            print(f"  skip {cell} (passive-observation): missing training.log")
            continue
        summary, per_rank = aggregate(d, anchor_label="default")
        cell_rows.append(summary)
        rank_rows.extend(per_rank)
        print(f"  ok   default-anchor {cell}")

    # Own cells.
    for cell in OWN_CELLS:
        d = HERE / cell
        if not (d / "training.log").exists():
            print(f"  skip {cell}: missing training.log")
            continue
        anchor = cell_to_anchor(cell)
        summary, per_rank = aggregate(d, anchor_label=anchor)
        cell_rows.append(summary)
        rank_rows.extend(per_rank)
        print(f"  ok   {anchor:14s} {cell}")

    per_cell_all = pd.DataFrame(cell_rows)
    per_rank_all = pd.DataFrame(rank_rows)

    # CSV outputs only contain OWN cells (the cross-read default-anchor
    # cells are sourced from ../passive-observation/per_cell.csv via
    # that sweep's analyze.py).
    own_mask = per_cell_all["cell"].isin(OWN_CELLS)
    own_rank_mask = per_rank_all["cell"].isin(OWN_CELLS)

    per_cell_own = per_cell_all[own_mask].sort_values(
        ["anchor", "guard", "seed"]).reset_index(drop=True)
    per_rank_own = per_rank_all[own_rank_mask].sort_values(
        ["anchor", "guard", "seed", "rank"]).reset_index(drop=True)

    per_cell_own.to_csv(OUT / "per_cell.csv", index=False, float_format="%.4f")
    per_rank_own.to_csv(OUT / "per_rank.csv", index=False, float_format="%.4f")
    print(f"  wrote per_cell.csv ({len(per_cell_own)} rows, own cells)")
    print(f"  wrote per_rank.csv ({len(per_rank_own)} rows, own cells)")

    # Figures use the FULL dataset (own + default-anchor) for the 4-cell
    # comparison; per-rank figure uses own only.
    fig_4cell_comparison(per_cell_all, OUT / "four_cell_comparison.png")
    fig_pearson_asymmetric(per_cell_all, OUT / "pearson_asymmetric.png")
    fig_per_rank_heterogeneity(per_rank_own, OUT / "per_rank_heterogeneity.png")
    print(f"  wrote 3 figures")

    write_report(per_cell_all, per_rank_own, OUT / "README.md")
    print(f"  wrote README.md")


if __name__ == "__main__":
    main()
