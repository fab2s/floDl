#!/usr/bin/env python3
"""Per-sweep focused analyzer for pareto-frontier (cross-sweep aggregation).

Wraps the existing `pareto.py` cross-sweep aggregator: regenerates
`pareto.txt` + `pareto.png` + `pareto-safe-zoom.png` in the parent
directory, then writes a curated `analysis/README.md` that ties the
frontier table to the upstream sweep analyses.

Unlike the other sweep analyzers, this one does not parse per-cell
artifacts directly — `pareto.py` already reads `report.md` from the
four upstream sweep dirs. This wrapper exists to keep the
`<sweep>/analyze.py + <sweep>/analysis/` shape consistent across the
research arc.

Reads from the parent directory (after running `pareto.py`); writes
to ./analysis/.

Usage:
    python3 analyze.py
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "analysis"
REPO_ROOT = HERE.parents[3]
PARETO_PY = HERE / "pareto.py"
PARETO_TXT = HERE / "pareto.txt"


# ---------------------------------------------------------------------------
# Parse pareto.txt
# ---------------------------------------------------------------------------

CONFIG_RE = re.compile(
    r"^\s*(\S.+?)\s+(\d+)\s+([\d.]+)% ± ([\d.]+)\s+(\d+) ± \s*(\d+)\s+"
    r"([\d.]+) ± \s*([\d.]+)\s+(FRONTIER|dominated.*)$"
)


def parse_pareto_txt(path: Path) -> tuple[list[dict], dict]:
    """Return (per-config rows, summary dict)."""
    text = path.read_text()

    rows = []
    in_table = False
    for line in text.splitlines():
        if "Per-config aggregate" in line:
            in_table = True
            continue
        if in_table:
            if line.startswith("---") and "Pareto frontier" in line:
                break
            m = CONFIG_RE.match(line)
            if not m:
                continue
            rows.append({
                "config": m.group(1).strip(),
                "n": int(m.group(2)),
                "eval_mean": float(m.group(3)),
                "eval_sd": float(m.group(4)),
                "syncs_mean": int(m.group(5)),
                "syncs_sd": int(m.group(6)),
                "wall_mean": float(m.group(7)),
                "wall_sd": float(m.group(8)),
                "status": m.group(9).strip(),
            })

    summary = {}
    m = re.search(r"Configurations on frontier:\s+(\d+)", text)
    if m:
        summary["frontier_count"] = int(m.group(1))
    m = re.search(r"Configurations dominated:\s+(\d+)", text)
    if m:
        summary["dominated_count"] = int(m.group(1))
    m = re.search(r"Total cells loaded:\s+(\d+)", text)
    if m:
        summary["total_cells"] = int(m.group(1))

    # Production-default verdict.
    m = re.search(
        r"Production default verdict[^\n]*\n"
        r"\s+eval = ([\d.]+)% ± [\d.]+, syncs = (\d+) ± \d+\s*\n"
        r"\s+(?:DOMINATED by:\s+([^\n]+?)\s*$|on FRONTIER)",
        text, re.MULTILINE,
    )
    if m:
        summary["prod_eval"] = float(m.group(1))
        summary["prod_syncs"] = int(m.group(2))
        summary["prod_dominator"] = m.group(3).strip() if m.group(3) else None

    return rows, summary


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def fig_frontier_ranked(rows: list[dict], out_path: Path) -> None:
    """Horizontal bar chart of all configs sorted by sync count (matches
    the table sort). Two panels: eval mean and sync count, with
    frontier vs dominated as color encoding."""
    rows_sorted = sorted(rows, key=lambda r: r["syncs_mean"])
    labels = [r["config"] for r in rows_sorted]
    evals = [r["eval_mean"] for r in rows_sorted]
    eval_sds = [r["eval_sd"] for r in rows_sorted]
    syncs = [r["syncs_mean"] for r in rows_sorted]
    sync_sds = [r["syncs_sd"] for r in rows_sorted]
    colors = ["#2a9d8f" if r["status"] == "FRONTIER" else "#bbbbbb"
              for r in rows_sorted]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    y = list(range(len(rows_sorted)))

    axes[0].barh(y, evals, xerr=eval_sds, color=colors,
                 edgecolor="black", linewidth=0.5, capsize=3)
    axes[0].set_xlabel("Final eval (%)")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=9)
    axes[0].set_xlim(0, 100)
    axes[0].invert_yaxis()  # top of chart = lowest sync count
    axes[0].grid(True, axis="x", alpha=0.3)
    axes[0].set_title("Eval (mean ± sd)")

    axes[1].barh(y, syncs, xerr=sync_sds, color=colors,
                 edgecolor="black", linewidth=0.5, capsize=3)
    axes[1].set_xlabel("Syncs / 200 epochs")
    axes[1].set_xscale("log")
    axes[1].grid(True, axis="x", alpha=0.3)
    axes[1].set_title("Sync count (mean ± sd, log-x)")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c)
               for c in ("#2a9d8f", "#bbbbbb")]
    fig.legend(handles, ["frontier", "dominated"],
               loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.02), title="status")
    fig.suptitle(
        "Pareto frontier — all 14 configurations ranked by sync count",
        y=1.05,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

NARRATIVE = """\
The Pareto frontier is the cross-sweep eval-vs-cost characterization.
Cost axis: AllReduce events per 200-epoch run (a network-volume proxy
that stays meaningful at fixed model size; rotates toward wall-time
as parameter count grows). Eval axis: held-out CIFAR-10 test
accuracy.

Configurations span three upstream sweeps:

- [`../passive-observation/`](../passive-observation/) — default-anchor
  cells (20 cells: 5 seeds × 2 modes × 2 guards on nccl-async +
  cpu-async; cpu-async cohort uses EASGD α=0.5 elastic blending).
- [`../relaxed-anchor/`](../relaxed-anchor/) — `--elche-relax-up`
  cells (10 cells: 5 seeds × 2 guards × nccl-async).
- [`../cliff-bracket/`](../cliff-bracket/) — fixed-k cliff probe
  (18 cells: 3 seeds × 6 k values).
"""


def fmt_table(rows: list[dict]) -> str:
    out = ["| config | n | eval (%) | syncs / 200 ep | status |",
           "|---|---:|---:|---:|---|"]
    for r in rows:
        eval_str = f"{r['eval_mean']:.2f} ± {r['eval_sd']:.2f}"
        sync_str = f"{r['syncs_mean']} ± {r['syncs_sd']}"
        if r["status"] == "FRONTIER":
            status = "**frontier**"
        else:
            status = r["status"]
        out.append(f"| `{r['config']}` | {r['n']} | {eval_str} | {sync_str} | {status} |")
    return "\n".join(out)


def write_report(rows: list[dict], summary: dict, out_path: Path) -> None:
    frontier = [r for r in rows if r["status"] == "FRONTIER"]
    dominated = [r for r in rows if r["status"] != "FRONTIER"]

    # Identify the high-sync end of the frontier (eval-maximum + lowest-sync at near-parity).
    high_sync_frontier = sorted(
        [r for r in frontier if r["syncs_mean"] >= 100],
        key=lambda r: r["syncs_mean"],
    )

    eval_max = max(rows, key=lambda r: r["eval_mean"])
    lowest_sync_near_parity = min(
        [r for r in frontier if r["eval_mean"] > 91.5 and r["syncs_mean"] >= 100],
        key=lambda r: r["syncs_mean"], default=None,
    )

    obs_lines = []
    obs_lines.append(
        f"- **The frontier is {summary.get('frontier_count', '?')} configurations of "
        f"{summary.get('frontier_count', 0) + summary.get('dominated_count', 0)} total** "
        f"({summary.get('total_cells', '?')} cells across the four upstream sweeps)."
    )
    if eval_max:
        obs_lines.append(
            f"- **Eval maximum sits at `{eval_max['config']}`** "
            f"({eval_max['eval_mean']:.2f}% ± {eval_max['eval_sd']:.2f}, "
            f"{eval_max['syncs_mean']} ± {eval_max['syncs_sd']} syncs)."
        )
    if lowest_sync_near_parity is not None:
        obs_lines.append(
            f"- **Lowest-sync near-parity point is `{lowest_sync_near_parity['config']}`** "
            f"({lowest_sync_near_parity['eval_mean']:.2f}%, "
            f"{lowest_sync_near_parity['syncs_mean']} syncs) — trades a small eval drop for "
            f"a sync-count reduction at the high-sync end."
        )
    if "prod_eval" in summary and summary.get("prod_dominator"):
        obs_lines.append(
            f"- **Production default `nccl-async default msf` "
            f"({summary['prod_eval']:.2f}%, {summary['prod_syncs']} syncs) "
            f"is dominated by `{summary['prod_dominator']}`** — the production-config "
            f"improvement is a backend swap, not a coupling-mechanism change."
        )
    obs_lines.append(
        "- **Fixed-k cells dominate the low-sync end** by construction: pinning the "
        "cadence at `k` produces ~`200/k` syncs per 200-epoch run, three orders of "
        "magnitude below the auto-tuned regime. The cliff-localization sweep at "
        "[`../cliff-bracket/`](../cliff-bracket/) shows this is also where the "
        "synchronization threshold lives."
    )

    text = f"""# pareto-frontier — analysis

{NARRATIVE}

## Cross-sweep frontier table

{fmt_table(rows)}

Sorted by mean sync count, ascending. Per-config aggregates are
mean ± standard deviation across seeds. Status is the Pareto-frontier
classification: a configuration is dominated when another has
strictly better (or equal) eval at strictly lower sync count.

![Frontier ranked](frontier_ranked.png)

Direct table-companion: each row of the table above corresponds to a
horizontal bar pair (eval on the left panel, syncs on the right,
log-x). Rows are sorted by sync count, matching the table order.
Frontier configurations in green; dominated in grey.

## Pareto figure (full + safe-regime zoom)

![Pareto frontier](../pareto.png)

Full Pareto plot. X-axis is log-scaled syncs / 200 epochs. The fixed-k
cells span the low-sync end (1–13 syncs); the auto-tuned cells cluster
in the high-sync band (~400–900 syncs). Past the cliff, fixed-k cells
collapse to ≪ random-chance accuracy.

![Safe-regime zoom](../pareto-safe-zoom.png)

Safe-regime zoom — the auto-tune cluster in the high-sync band, with
the frontier knee visible at the low-sync end of the safe regime.

## Key observations

{chr(10).join(obs_lines)}

## Source data

- [`../pareto.py`](../pareto.py) — the cross-sweep aggregator. Reads
  `report.md` from the four upstream sweep extracts.
- [`../pareto.txt`](../pareto.txt) — canonical text output.
- [`../pareto.png`](../pareto.png) and
  [`../pareto-safe-zoom.png`](../pareto-safe-zoom.png) — the figures
  embedded above.

## Reproducibility

Run from this directory: `python3 analyze.py`. The wrapper invokes
`pareto.py` from the project root (regenerating `pareto.txt` plus the
two PNGs in the parent directory) and then writes this curated
`README.md`. See [`../README.md`](../README.md) for the directory-level
recipe.
"""
    out_path.write_text(text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_args()

    OUT.mkdir(exist_ok=True)
    print(f"out-dir: {OUT.relative_to(REPO_ROOT)}")

    # Regenerate pareto outputs from the four upstream sweep extracts.
    print(f"  running pareto.py …")
    result = subprocess.run(
        [sys.executable, str(PARETO_PY)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise SystemExit(f"pareto.py failed: exit {result.returncode}")
    PARETO_TXT.write_text(result.stdout)
    print(f"  pareto.py OK")

    rows, summary = parse_pareto_txt(PARETO_TXT)
    print(f"  parsed {len(rows)} configurations from pareto.txt")

    fig_frontier_ranked(rows, OUT / "frontier_ranked.png")
    print(f"  wrote frontier_ranked.png")

    write_report(rows, summary, OUT / "README.md")
    print(f"  wrote README.md")


if __name__ == "__main__":
    main()
