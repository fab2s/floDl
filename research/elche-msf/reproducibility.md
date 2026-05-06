# Reproducibility map — elche-msf

Every empirical claim in [`manuscript.md`](manuscript.md) maps to a row
below. Each row names: the sweep that produced the data, the
aggregator script that generated the table or figure, and the
pre-launch commit hash for the binary state under test. Aggregators
are co-located with the sweep they read.

The full working spec, including phase-by-phase verdicts, lives in
[`docs/design/msf-cadence-control-v2.md`](../../docs/design/msf-cadence-control-v2.md);
this file is the cited-evidence index for the manuscript.

## Sweeps

Each row points at the committed paper-evidence extracts in
`data/<slug>/`. Raw run output is local-only (gitignored at
`ddp-bench/runs/`); the committed extracts (~16 MB total across all
landed sweeps) carry every numeric the aggregators read.

| Sweep | Extracts | Aggregator | Status |
|---|---|---|---|
| Passive observation (5 seeds × 2 modes × 2 guards = 20 cells; cpu-async = EASGD α=0.5) | [`data/passive-observation/`](data/passive-observation/) | `aggregate.py` | landed; mixed-tree caveat (see below) |
| Relaxed-anchor sweep (5 seeds × 2 guards × nccl-async × `--elche-relax-up`) | [`data/relaxed-anchor/`](data/relaxed-anchor/) | `aggregate.py` (reads `passive-observation/` + `./`) | landed |
| Cliff bracket (fixed-k probe) | [`data/cliff-bracket/`](data/cliff-bracket/) | `aggregate.py` | landed; byte-identical |
| ResNet-56 bytes-axis | (pending) | (pending) | landing on 2026-05-07 |
| Pareto frontier (cross-sweep) | [`data/pareto-frontier/`](data/pareto-frontier/) | `pareto.py` (reads 3 upstream sweeps) | landed; 12 configs after reorg (was 14, EASGD family folded into cpu-async default) |


## Claim ↔ artifact

[STUB — populate as manuscript sections move from `[STUB]` to `[DRAFT]`
to `[LOCKED]`. Each entry should name: claim text, manuscript section,
table/figure id, run dir, aggregator script, key numerical values.]

Example row format (to be filled):

```
- claim: "cliff localized between k=16000 and k=25600 for the standard
  200-epoch schedule"
  section: §5.2
  table: tables/phase3-cliff-bracket.md
  run dir: ddp-bench/runs/overnight-2026-05-05-sweep-b2-cliff/
  aggregator: ddp-bench/runs/overnight-2026-05-05-sweep-b2-cliff/aggregate.py
  key numbers:
    - k=16000: 90.05% ± 0.64 (1.2pp range, soft pre-cliff)
    - k=25600: 77.09% ± 18.71 (35.1pp range, bimodal cliff edge)
    - k=51200: 27.75% ± 30.72 (53.2pp range, past cliff)
```

## Hardware fingerprint

For the entire arc, identical rig:

- GPU0: NVIDIA RTX 5060 Ti 16GB (sm_120, 16311 MiB)
- GPU1: NVIDIA GTX 1060 6GB (sm_61, 6144 MiB)
- GPU2: NVIDIA GTX 1060 6GB (sm_61, 6144 MiB)
- libtorch: see `libtorch/.active` and `libtorch/<variant>/.arch` at
  the pre-launch commit for each sweep.

## Software fingerprint

Branch `ddp-scale` throughout. Working spec:
[`docs/design/msf-cadence-control-v2.md`](../../docs/design/msf-cadence-control-v2.md).
v1 spec preserved as
[`docs/design/msf-cadence-control.md`](../../docs/design/msf-cadence-control.md).
