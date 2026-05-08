#!/usr/bin/env python3
"""Extract paper-evidence artifacts from the cpu-async multi-seed sweep.

Cell shape: 4 seeds (1-4) x 2 guards (msf, trend) x cpu-async = 8 cells.
Raw cell names carry an `-easgd05` suffix; the extract drops it because
EASGD α=0.5 is the canonical cpu-async semantics post-2026-05-04 and the
suffix is no longer load-bearing (it would be uniform across the cohort).

Extraction policy: drop json/html/stdout, gzip csv, keep
training.log + report.md verbatim.
"""

import argparse
import gzip
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
DEFAULT_RAW = REPO_ROOT / "ddp-bench" / "runs" / "overnight-2026-05-06-easgd-multiseed"

# (raw_cell_name, extract_cell_name) — raw carries -easgd05, extract drops it.
CELLS = [
    (f"seed-{seed}-cpu-async-{guard}-easgd05", f"seed-{seed}-cpu-async-{guard}")
    for seed in (1, 2, 3, 4)
    for guard in ("msf", "trend")
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
    parser = argparse.ArgumentParser(description="cpu-async multi-seed sweep extractor.")
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
