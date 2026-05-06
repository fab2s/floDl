# Pareto frontier — cross-sweep aggregator

## What this directory holds

A cross-sweep aggregator that reads `report.md` from three sweep dirs
(`passive-observation/`, `relaxed-anchor/`, `cliff-bracket/`) and
produces the eval-vs-syncs Pareto plot. No raw per-cell data lives
here; this directory is purely the analysis layer on top of the
others.

## Files

```
.
├── README.md
├── pareto.py             cross-sweep aggregator (matplotlib)
├── pareto.txt            canonical text output (frontier table +
│                         dominated configs + safe-zone summary)
├── pareto.png            full Pareto figure across all configs
└── pareto-safe-zoom.png  zoomed-in safe-regime detail
```

## Running

From the repo root:

```
python3 research/elche-msf/data/pareto-frontier/pareto.py
```

Or via the analysis wrapper (also writes `analysis/README.md` and
`analysis/frontier_ranked.png`):

```
python3 research/elche-msf/data/pareto-frontier/analyze.py
```

The script reads from each upstream sweep dir's per-cell `report.md`
and writes the .txt + 2 PNGs back into this directory. Re-running
overwrites the figures in place; the .txt is what gets cited.

## Provenance

Flag-set + invariant model documented in the upstream sweep dirs that
this aggregator reads. The bench binary version, recipe, and hardware
are noted per-sweep in `../<slug>/README.md`.
