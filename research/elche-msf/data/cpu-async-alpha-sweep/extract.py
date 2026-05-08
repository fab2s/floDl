#!/usr/bin/env python3
"""Extract paper-evidence artifacts from the cpu-async α-axis sweep.

Cell shape: walk the EASGD α axis at fixed cpu-async × msf × ResNet-20.
4 α-values {0.3, 0.5, 0.7, 1.0} × 4 seeds {1, 2, 3, 4} = 16 cells.

Cell-name policy: keep the `-alphaXX` suffix on every cell. α IS the axis
under test in this sweep, so the suffix is the differentiator (unlike
cpu-async-multiseed where α=0.5 is uniform and the suffix is dropped).

Extraction policy: drop json/html/stdout, gzip csv, keep
training.log + report.md verbatim.
"""

import argparse
import gzip
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
DEFAULT_RAW = REPO_ROOT / "ddp-bench" / "runs" / "overnight-2026-05-08-easgd-alpha-sweep"

# Cell name is the same in raw and extract (direct mapping).
CELLS = [
    f"seed-{seed}-cpu-async-msf-alpha{suffix}"
    for seed in (1, 2, 3, 4)
    for suffix in ("03", "05", "07", "10")
]


def extract_cell(raw_root: Path, out_root: Path, name: str) -> bool:
    raw_cell = raw_root / name
    raw_inner = raw_cell / "resnet-graph" / "cpu-async"
    out_cell = out_root / name

    csv_in = raw_inner / "timeline.csv"
    log_in = raw_inner / "training.log"
    report_in = raw_cell / "report.md"

    missing = [p.relative_to(raw_root) for p in (csv_in, log_in, report_in) if not p.exists()]
    if missing:
        print(f"  skip {name}: missing {missing}")
        return False

    out_cell.mkdir(parents=True, exist_ok=True)

    with csv_in.open("rb") as src, gzip.open(out_cell / "timeline.csv.gz",
                                             "wb", compresslevel=9) as dst:
        shutil.copyfileobj(src, dst)
    shutil.copyfile(log_in, out_cell / "training.log")
    shutil.copyfile(report_in, out_cell / "report.md")

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="cpu-async α-axis sweep extractor.")
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
    for name in CELLS:
        if extract_cell(raw_root, HERE, name):
            print(f"  ok   {name}")
            n_ok += 1

    print(f"\n{n_ok}/{len(CELLS)} cells extracted.")


if __name__ == "__main__":
    main()
