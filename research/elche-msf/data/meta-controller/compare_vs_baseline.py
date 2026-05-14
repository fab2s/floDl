#!/usr/bin/env python3
"""Paired-seed comparison: meta-on (this dir) vs meta-off baseline
(../passive-observation/).

Validates the decision criteria locked in
.claude/projects/.../project_06_controller_arc.md (2026-05-09):

  1. meta-on preserves sync-count saving at parity eval
     (paired-seed Δ_eval < 0.15 pp).
  2. meta-on reduces seed sd vs meta-off (load-bearing claim — phase-aware
     reactive correction handles LR-decay variance).
  3. Convergence watcher does NOT fire spuriously on the msf-silent cohort
     (verified via "meta-nudge" Custom-event count; expected 0 fires across
     the msf cells).

Reads:
  ./seed-{1..4}-{cpu-async,nccl-async}-{msf,trend}-meta/report.md (this)
  ../passive-observation/seed-{1..4}-{cpu-async,nccl-async}-{msf,trend}/report.md
  ./seed-*/timeline.csv.gz (for meta-fire counting)

Writes ./analysis/comparison.md and ./analysis/per_cell.csv.

Usage:
    python3 compare_vs_baseline.py
"""

import csv
import gzip
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "analysis"
OUT.mkdir(exist_ok=True)
BASELINE = HERE.parent / "passive-observation"

SEEDS = [1, 2, 3, 4]
MODES = ["cpu-async", "nccl-async"]
GUARDS = ["msf", "trend"]

# Capture (loss, eval_frac, total_s, syncs) from report.md row.
ROW_RE = re.compile(
    r"\| (?:cpu-async|nccl-async) \| ([\d.]+) \| ([\d.]+) \| [+\-\d.]+ \| ([\d.]+) \| (\d+) \|"
)


def parse_cell(cell_dir: Path) -> dict:
    text = (cell_dir / "report.md").read_text()
    m = ROW_RE.search(text)
    if not m:
        raise ValueError(f"could not parse {cell_dir}/report.md")
    return {
        "loss": float(m.group(1)),
        "eval_pct": float(m.group(2)) * 100,
        "total_s": float(m.group(3)),
        "syncs": int(m.group(4)),
    }


def count_meta_fires(cell_dir: Path) -> int:
    """Count 'meta-nudge' Custom-event occurrences in the timeline."""
    timeline = cell_dir / "timeline.csv.gz"
    if not timeline.exists():
        return -1
    fires = 0
    with gzip.open(timeline, "rt") as f:
        for line in f:
            if "meta-nudge" in line:
                fires += 1
    return fires


def load_pair(seed: int, mode: str, guard: str) -> tuple[dict, dict]:
    off_dir = BASELINE / f"seed-{seed}-{mode}-{guard}"
    on_dir = HERE / f"seed-{seed}-{mode}-{guard}-meta"
    off = parse_cell(off_dir)
    on = parse_cell(on_dir)
    on["fires"] = count_meta_fires(on_dir)
    return off, on


def cohort_key(mode: str, guard: str) -> str:
    return f"{mode}/{guard}"


def main():
    rows = []
    by_cohort: dict[str, dict[str, list]] = {}

    for mode in MODES:
        for guard in GUARDS:
            key = cohort_key(mode, guard)
            by_cohort[key] = {
                "off_eval": [], "on_eval": [],
                "off_syncs": [], "on_syncs": [], "fires": [],
            }
            for seed in SEEDS:
                try:
                    off, on = load_pair(seed, mode, guard)
                except (FileNotFoundError, ValueError) as e:
                    print(f"skip seed={seed} mode={mode} guard={guard}: {e}")
                    continue
                d_eval = on["eval_pct"] - off["eval_pct"]
                sync_red_pct = (
                    100 * (off["syncs"] - on["syncs"]) / off["syncs"]
                    if off["syncs"] else 0.0
                )
                rows.append({
                    "mode": mode, "guard": guard, "seed": seed,
                    "off_eval": off["eval_pct"], "on_eval": on["eval_pct"],
                    "d_eval_pp": d_eval,
                    "off_syncs": off["syncs"], "on_syncs": on["syncs"],
                    "sync_reduction_pct": sync_red_pct,
                    "meta_fires": on["fires"],
                })
                d = by_cohort[key]
                d["off_eval"].append(off["eval_pct"])
                d["on_eval"].append(on["eval_pct"])
                d["off_syncs"].append(off["syncs"])
                d["on_syncs"].append(on["syncs"])
                d["fires"].append(on["fires"])

    csv_path = OUT / "per_cell.csv"
    with open(csv_path, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"wrote {csv_path}  ({len(rows)} rows)")

    md = OUT / "comparison.md"
    with open(md, "w") as f:
        f.write("# Meta-controller validation — paired-seed comparison\n\n")
        f.write(
            "Comparing this sweep (meta-on) against `../passive-observation/` "
            "(meta-off). Recipes are identical except `--meta-controller`.\n\n"
        )

        f.write("## Per-cohort summary\n\n")
        f.write(
            "| mode | guard | n | off eval mean ± sd | on eval mean ± sd | "
            "Δ eval | off syncs | on syncs | sync redux | mean fires |\n"
        )
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for mode in MODES:
            for guard in GUARDS:
                key = cohort_key(mode, guard)
                d = by_cohort[key]
                n = len(d["off_eval"])
                if n == 0:
                    continue
                off_m = statistics.mean(d["off_eval"])
                off_sd = statistics.stdev(d["off_eval"]) if n > 1 else 0.0
                on_m = statistics.mean(d["on_eval"])
                on_sd = statistics.stdev(d["on_eval"]) if n > 1 else 0.0
                d_eval = on_m - off_m
                off_syncs_m = statistics.mean(d["off_syncs"])
                on_syncs_m = statistics.mean(d["on_syncs"])
                redux = (
                    100 * (off_syncs_m - on_syncs_m) / off_syncs_m
                    if off_syncs_m else 0.0
                )
                fires = [x for x in d["fires"] if x >= 0]
                mean_fires = statistics.mean(fires) if fires else float("nan")
                f.write(
                    f"| {mode} | {guard} | {n} | "
                    f"{off_m:.2f} ± {off_sd:.2f} | "
                    f"{on_m:.2f} ± {on_sd:.2f} | "
                    f"{d_eval:+.2f} pp | "
                    f"{off_syncs_m:.0f} | {on_syncs_m:.0f} | {redux:+.1f}% | "
                    f"{mean_fires:.1f} |\n"
                )

        f.write("\n## Decision criteria (per cohort)\n\n")
        for mode in MODES:
            for guard in GUARDS:
                key = cohort_key(mode, guard)
                d = by_cohort[key]
                if not d["off_eval"]:
                    continue
                d_eval = statistics.mean(d["on_eval"]) - statistics.mean(d["off_eval"])
                on_sd = statistics.stdev(d["on_eval"]) if len(d["on_eval"]) > 1 else 0.0
                off_sd = statistics.stdev(d["off_eval"]) if len(d["off_eval"]) > 1 else 0.0
                fires = [x for x in d["fires"] if x >= 0]
                mean_fires = statistics.mean(fires) if fires else float("nan")
                crit1 = "✓" if abs(d_eval) < 0.15 else "✗"
                crit2 = "✓" if on_sd <= off_sd else "✗"
                if guard == "msf":
                    crit3 = (
                        "✓" if mean_fires == 0
                        else ("?" if mean_fires != mean_fires else "✗")
                    )
                    crit3_label = " | no spurious fires (msf-silent regression check)"
                else:
                    crit3 = ""
                    crit3_label = ""
                f.write(f"### {mode} / {guard}\n")
                f.write(f"- Parity eval (|Δ| < 0.15 pp): **{crit1}** "
                        f"(Δ={d_eval:+.3f} pp)\n")
                f.write(f"- Reduced seed sd: **{crit2}** "
                        f"(off sd {off_sd:.2f} → on sd {on_sd:.2f})\n")
                if guard == "msf":
                    f.write(f"- No spurious fires: **{crit3}** "
                            f"(mean fires={mean_fires:.1f})\n")
                f.write("\n")

        f.write("## Per-cell detail\n\nSee `per_cell.csv`.\n")

    print(f"wrote {md}")


if __name__ == "__main__":
    main()
