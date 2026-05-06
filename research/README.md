# Research

Empirical research arcs conducted with [flodl](../README.md). Each
subdirectory is a self-contained research artifact: manuscript draft,
figures regenerable from co-located scripts, and reproducibility map
linking every empirical claim to a sweep dir + commit hash.

This is **not** documentation. User-facing material lives in
[`docs/`](../docs/). Working design notes live in
[`docs/design/`](../docs/design/). What lives here is **what we
discovered with flodl**, intended for coauthors, citations, and eventual
preprint upload.

## Arcs

- [elche-msf/](elche-msf/) — Pecora-Carroll synchronization framing for
  heterogeneous DDP. ResNet-20/CIFAR-10/3-GPU empirical Pareto frontier;
  cliff localization between k=16000 and k=25600; bytes-axis confirmation
  on ResNet-56 in progress.

## Conventions

- **Manuscripts**: markdown-first; port to LaTeX only at submission time.
- **Figures**: PNG/SVG checked in alongside the script that generates them.
  Hand-drawn diagrams are acceptable when no script applies, but flagged
  in the figure caption.
- **Tables**: CSV (data) + MD (paper-ready) co-located. Aggregator scripts
  must live next to the sweep dirs they read.
- **Reproducibility map**: every claim in the manuscript links to a row in
  `reproducibility.md` that names the run dir, aggregator script, and
  pre-launch commit hash.
- **References**: `references.bib` per arc, BibTeX format.
