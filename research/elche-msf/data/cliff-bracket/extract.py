#!/usr/bin/env python3
"""Extract paper-evidence artifacts from the cliff-bracket sweep.

Cell shape: 18 cells = 3 seeds x 6 fixed-k values. Cadence is pinned at
`k` via `--min-anchor=k --max-anchor=k --guard none`. Mode is nccl-async
for all cells; cell name encodes seed and k (no mode in name).

See `../passive-observation/extract.py` for the same pattern.
"""

import argparse
import gzip
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
DEFAULT_RAW = REPO_ROOT / "ddp-bench" / "runs" / "overnight-2026-05-05-sweep-b2-cliff"

K_VALUES = (3200, 6400, 12800, 16000, 25600, 51200)
CELL_NAMES = [
    f"seed-{seed}-fixed-k-{k}"
    for seed in (0, 1, 2)
    for k in K_VALUES
]
MODE = "nccl-async"  # uniform across all cells


def extract_cell(raw_root: Path, out_root: Path, cell: str) -> bool:
    raw_cell = raw_root / cell
    raw_inner = raw_cell / "resnet-graph" / MODE
    out_cell = out_root / cell

    csv_in = raw_inner / "timeline.csv"
    log_in = raw_inner / "training.log"
    report_in = raw_cell / "report.md"

    missing = [p.relative_to(raw_root) for p in (csv_in, log_in, report_in) if not p.exists()]
    if missing:
        print(f"  skip {cell}: missing {missing}")
        return False

    out_cell.mkdir(parents=True, exist_ok=True)

    with csv_in.open("rb") as src, gzip.open(out_cell / "timeline.csv.gz",
                                             "wb", compresslevel=9) as dst:
        shutil.copyfileobj(src, dst)
    shutil.copyfile(log_in, out_cell / "training.log")
    shutil.copyfile(report_in, out_cell / "report.md")

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Cliff-bracket fixed-k sweep extractor.")
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
    print(f"cells:    {len(CELL_NAMES)}")

    n_ok = 0
    for cell in CELL_NAMES:
        if extract_cell(raw_root, HERE, cell):
            print(f"  ok   {cell}")
            n_ok += 1

    print(f"\n{n_ok}/{len(CELL_NAMES)} cells extracted.")


if __name__ == "__main__":
    main()
