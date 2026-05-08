#!/usr/bin/env python3
"""Extract paper-evidence artifacts from the ResNet-56 cpu-async sweep.

Cell shape: bytes-axis confirmation at 3.1× ResNet-20's parameter count.
4 seeds x α=0.5 (canonical) plus 1 seed x α=1.0 (single-cell baseline;
sweep halted before seeds 2-4 α=1.0 ran).

Cell-name policy: drop the `-easgd05` suffix (α=0.5 is canonical
cpu-async; uniform across this cohort would not differentiate). Keep an
`-alpha10` suffix on the α=1.0 cell since that IS the differentiator
this sweep probes.

Extraction policy: drop json/html/stdout, gzip csv, keep
training.log + report.md verbatim.

Note: the 4 α=0.5 cells were originally missing report.md (sweep launcher
bash was killed before the bench finished writing reports). Reports were
regenerated post-hoc via `ddp-bench --report` analyze-mode using the saved
timeline.csv. seed-1 α=1.0's report was regenerated the same way per the
runlog note.
"""

import argparse
import gzip
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
DEFAULT_RAW = REPO_ROOT / "ddp-bench" / "runs" / "overnight-2026-05-06-resnet56-easgd"

# (raw_cell_name, extract_cell_name).
CELLS = [
    (f"seed-{seed}-cpu-async-msf-easgd05", f"seed-{seed}-cpu-async-msf")
    for seed in (1, 2, 3, 4)
] + [
    ("seed-1-cpu-async-msf-easgd10", "seed-1-cpu-async-msf-alpha10"),
]


def extract_cell(raw_root: Path, out_root: Path, raw_name: str, out_name: str) -> bool:
    raw_cell = raw_root / raw_name
    raw_inner = raw_cell / "resnet-graph" / "cpu-async"
    out_cell = out_root / out_name

    csv_in = raw_inner / "timeline.csv"
    log_in = raw_inner / "training.log"
    report_in = raw_cell / "report.md"

    missing = [p.relative_to(raw_root) for p in (csv_in, log_in, report_in) if not p.exists()]
    if missing:
        print(f"  skip {raw_name}: missing {missing}")
        return False

    out_cell.mkdir(parents=True, exist_ok=True)

    with csv_in.open("rb") as src, gzip.open(out_cell / "timeline.csv.gz",
                                             "wb", compresslevel=9) as dst:
        shutil.copyfileobj(src, dst)
    shutil.copyfile(log_in, out_cell / "training.log")
    shutil.copyfile(report_in, out_cell / "report.md")

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="ResNet-56 cpu-async sweep extractor.")
    parser.add_argument(
        "--raw-base",
        type=Path,
        default=DEFAULT_RAW,
        help=f"Raw sweep output dir (default: {DEFAULT_RAW.relative_to(REPO_ROOT)})",
    )
    args = parser.parse_args()
    raw_root = args.raw_base.resolve()

    if not raw_root.exists():
        raise SystemExit(f"raw-base not found: {raw_root}")

    print(f"raw-base: {raw_root}")
    print(f"out-dir:  {HERE}")
    print(f"cells:    {len(CELLS)}")

    n_ok = 0
    for raw_name, out_name in CELLS:
        if extract_cell(raw_root, HERE, raw_name, out_name):
            print(f"  ok   {raw_name}  ->  {out_name}")
            n_ok += 1

    print(f"\n{n_ok}/{len(CELLS)} cells extracted.")


if __name__ == "__main__":
    main()
