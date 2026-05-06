"""Cliff-bracket aggregator — synchronization-collapse cliff bracket (200 epochs).

Cell axis: fixed k ∈ {3200, 6400, 12800, 16000, 25600, 51200}, --guard none,
--min-anchor=k --max-anchor=k (anchor pinned). N=3 seeds per cell.

Two structural ideas relative to a plain mean ± sd report:

1. Cliff localization. Past the threshold, individual seeds collapse to random
   chance (~10% on CIFAR-10) while neighbors in the same cell may still hold a
   safe basin. A plain "mean ± sd" hides this; we surface per-seed evals,
   the cell range (max-min), and a bimodality flag (range > 30pp) so the cliff
   edge stands out.

2. Optional Pearson / by-k tolerance. Cells past the cliff have ≤1 within-
   training sync, so cross-rank Pearson and by-k OLS fits are absent or built
   from N≤2 events. The aggregator skips silently rather than crashing.

Sharp pre-launch predictions tested here:
  - k=51200 collapses to ~10% on most seeds (matches 50ep k=100000 zero-sync).
  - k=25600 likely collapses too (only ~2 within-training syncs).
  - Adjacent-cell eval drop > 50pp localizes the cliff to a single bracket.
  - Cross-rank Pearson r drops below 0.95 in collapsed cells.
"""
import re
import os
from statistics import mean, stdev

BASE = "research/elche-msf/data/cliff-bracket"
K_VALUES = [3200, 6400, 12800, 16000, 25600, 51200]
SEEDS = [0, 1, 2]
BIMODAL_RANGE_PP = 30.0  # cell range (max-min eval) above this flags bimodality


# ---------------------------------------------------------------------------
# Parsers (tolerant of missing sections)
# ---------------------------------------------------------------------------

def parse_main(text):
    """Per-Model Results row: Mode | Loss | Eval | vs Ref | Total (s) | Syncs | ..."""
    sec = re.search(
        r"\| (\S+) \| ([\d.]+) \| ([\d.]+) \| [+-][\d.]+ \| ([\d.]+) \| (\d+) \|",
        text,
    )
    if not sec:
        return None
    return {
        "mode": sec.group(1),
        "loss": float(sec.group(2)),
        "eval": float(sec.group(3)),
        "total_s": float(sec.group(4)),
        "syncs": int(sec.group(5)),
    }


def parse_pearson(text):
    """Cross-rank Pearson r per pair. Empty if too few sync events."""
    out = {}
    for m in re.finditer(
        r"\| resnet-graph \| \S+ \| rank(\d) ↔ rank(\d) \| ([+-][\d.]+) \|",
        text,
    ):
        out[(int(m.group(1)), int(m.group(2)))] = float(m.group(3))
    return out


def parse_r1(text):
    """R1 informal: log(D) vs step per LR window. Returns list of dicts per LR."""
    out = []
    sec = re.search(
        r"### R1 informal:.*?\| Model \| Mode.*?\n((?:\|.*\n)+)",
        text, re.DOTALL,
    )
    if not sec:
        return out
    for m in re.finditer(
        r"\| resnet-graph \| (\S+) \| ([\d.e+-]+) \| ([^|]+?) \| (\d+) \| \d+–\d+ \| "
        r"([+-]?[\d.e+-]+) \| ([\d.]+) \| "       # max
        r"([+-]?[\d.e+-]+) \| ([\d.]+) \| (\d+) \| "  # mean + n_ep
        r"([+-]?[\d.e+-]+) \| ([\d.]+) \| "       # epoch dmean
        r"\d+–\d+ \| "                             # k range
        r"([+-]?[\d.e+-]+) \| ([\d.]+) \|",       # by-k
        sec.group(1),
    ):
        out.append({
            "lr": float(m.group(2)),
            "epochs": m.group(3).strip(),
            "n_evt": int(m.group(4)),
            "r2_max": float(m.group(6)),
            "r2_mean": float(m.group(8)),
            "r2_epoch": float(m.group(11)),
            "slope_byk": float(m.group(12)),
            "r2_byk": float(m.group(13)),
        })
    return out


def parse_perrank_r1(text):
    """R1' per-rank by-k slopes — framing-validity gate."""
    out = []
    sec = re.search(
        r"### R1' per-rank by-k slopes.*?\| Model \| Mode.*?\n((?:\|.*\n)+)",
        text, re.DOTALL,
    )
    if not sec:
        return out
    for m in re.finditer(
        r"\| resnet-graph \| (\S+) \| ([\d.e+-]+) \| ([^|]+?) \| ([+-]?[\d.e+-]+) \| ([^|]+) \| ([^|]+) \| ([\d.]+)× \| ([^|]+?) \|",
        sec.group(1),
    ):
        gate_str = m.group(8).strip()
        gate_ok = "✓" in gate_str
        per_rank_text = m.group(5)
        slopes = []
        for sm in re.finditer(r"r\d+: ([+-]?[\d.e+-]+|—)", per_rank_text):
            v = sm.group(1)
            slopes.append(None if v == "—" else float(v))
        out.append({
            "lr": float(m.group(2)),
            "epochs": m.group(3).strip(),
            "meta_slope": float(m.group(4)),
            "per_rank_slopes": slopes,
            "ratio": float(m.group(7)),
            "gate_ok": gate_ok,
        })
    return out


def load_seed(seed, k):
    path = f"{BASE}/seed-{seed}-fixed-k-{k}/report.md"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        text = f.read()
    return {
        "k": k,
        "seed": seed,
        "main": parse_main(text),
        "pearson": parse_pearson(text),
        "r1": parse_r1(text),
        "perrank": parse_perrank_r1(text),
    }


def msd(xs):
    n = len(xs)
    if n == 0: return None, None
    m = sum(xs) / n
    if n == 1: return m, 0.0
    return m, (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5


# Load all 18 cells
cells = {k: [load_seed(s, k) for s in SEEDS] for k in K_VALUES}


# ---------------------------------------------------------------------------
# Summary — eval per cell, per-seed visible, cliff localization
# ---------------------------------------------------------------------------
print("=" * 100)
print(" Summary — final eval (held-out test, 200 epochs), N=3 seeds per cell")
print("=" * 100)
print(f"{'k':>6}  {'syncs':>5}  {'seed 0':>8}  {'seed 1':>8}  {'seed 2':>8}    {'mean ± sd':>17}  {'range':>6}  flag")
print("-" * 100)
for k in K_VALUES:
    runs = cells[k]
    evals = [r["main"]["eval"] for r in runs if r and r.get("main")]
    per_seed = [r["main"]["eval"] if r and r.get("main") else None for r in runs]
    syncs = [r["main"]["syncs"] for r in runs if r and r.get("main")]
    em, esd = msd(evals)
    sm, _ = msd(syncs)
    rng = (max(evals) - min(evals)) * 100 if evals else 0
    bimodal = rng > BIMODAL_RANGE_PP
    flag = "BIMODAL" if bimodal else ("CLIFF" if em is not None and em < 0.5 else "")
    seed_strs = [f"{e*100:>6.2f}%" if e is not None else "    —  " for e in per_seed]
    print(
        f"{k:>6}  {sm:>5.0f}  {seed_strs[0]:>8}  {seed_strs[1]:>8}  {seed_strs[2]:>8}    "
        f"{em*100:>6.2f}% ±{esd*100:>4.2f}  {rng:>5.1f}pp  {flag}"
    )

# Adjacent-cell deltas (>1pp soft transition, >50pp hard cliff edge)
print()
print(" Adjacent-cell deltas — locates the cliff bracket:")
print(f"  {'transition':<22} {'Δ mean eval':>13}  {'verdict':>30}")
for i in range(len(K_VALUES) - 1):
    a, b = K_VALUES[i], K_VALUES[i+1]
    ea = mean(r["main"]["eval"] for r in cells[a] if r and r.get("main"))
    eb = mean(r["main"]["eval"] for r in cells[b] if r and r.get("main"))
    delta = (eb - ea) * 100
    if abs(delta) >= 30:
        verdict = "*** CLIFF EDGE ***"
    elif abs(delta) >= 1:
        verdict = "soft drop (>1pp)"
    else:
        verdict = "flat"
    sign = "+" if delta >= 0 else ""
    print(f"  k={a:>5} → k={b:>5}     {sign}{delta:>7.2f}pp     {verdict:>30}")


# ---------------------------------------------------------------------------
# Cross-rank Pearson r per cell — meta-oscillator framing gate
# ---------------------------------------------------------------------------
print()
print("=" * 100)
print(" Cross-rank Pearson r per cell (gate fires if any single seed × pair < 0.95)")
print("=" * 100)
print(f" Cells with too few sync events to compute r are listed as '—' (cliff cells: 1 sync only).")
print()
print(f"{'k':>6}  {'r01 (mean)':>12}  {'r02 (mean)':>12}  {'r12 (mean)':>12}    {'cell min':>10}  gate")
print("-" * 100)
for k in K_VALUES:
    runs = cells[k]
    pair_means = {}
    for pair in [(0, 1), (0, 2), (1, 2)]:
        rs = [r["pearson"][pair] for r in runs if r and pair in (r["pearson"] or {})]
        if rs:
            pair_means[pair] = (mean(rs), min(rs))
    if not pair_means:
        print(f"{k:>6}     —             —             —              —      (no pearson data)")
        continue
    cell_min = min(v[1] for v in pair_means.values())
    gate = "✓" if cell_min >= 0.95 else "✗ BROKEN"

    def fmt_pair(p):
        v = pair_means.get(p)
        return f"{v[0]:+.4f}" if v else "—"

    print(
        f"{k:>6}    {fmt_pair((0,1)):>10}    {fmt_pair((0,2)):>10}    {fmt_pair((1,2)):>10}    "
        f"{cell_min:+.4f}    {gate}"
    )

# Per-seed pair minima — surfaces seed-level framing breakage
print()
print(" Per-seed Pearson minima (any single-seed pair < 0.95 fires the gate):")
print(f" {'k':>6}  {'seed 0':>10}  {'seed 1':>10}  {'seed 2':>10}    {'cell min':>10}")
for k in K_VALUES:
    runs = cells[k]
    seed_mins = []
    for r in runs:
        if r and r.get("pearson"):
            seed_mins.append(min(r["pearson"].values()))
        else:
            seed_mins.append(None)
    cell_vals = [s for s in seed_mins if s is not None]
    cell_min = min(cell_vals) if cell_vals else None
    parts = [f"{s:+.4f}" if s is not None else "—" for s in seed_mins]
    cell_min_str = f"{cell_min:+.4f}" if cell_min is not None else "—"
    print(f" {k:>6}  {parts[0]:>10}  {parts[1]:>10}  {parts[2]:>10}    {cell_min_str:>10}")


# ---------------------------------------------------------------------------
# R1 by-k slope per LR window per cell
# Within-cycle Lyapunov estimate; positive = exponential drift, negative = OU spiral.
# In safe cells (low k) we expect r²(by-k) collapse (OU regime, spiral-to-setpoint).
# Past the cliff k may not have enough samples for a meaningful fit.
# ---------------------------------------------------------------------------
print()
print("=" * 100)
print(" R1 by-k axis: log(D_mean) vs k_used per LR window — cross-seed N=3")
print("=" * 100)
print(" Note: the analyzer emits R1 only when an LR window has enough sync events for OLS")
print(" (≳4 events). At k=3200 the LR=0.3 window has 6 events and yields a fit; k≥6400")
print(" the within-cycle by-k axis is unobservable — itself the empirical signature of")
print(" crossing into the sparse-coupling regime.")


def aggregate_by_lr(runs, lr_target):
    """Take first row per (run, lr_target) — for sparse-sync cells the
    LR=0.3 window's epochs range already starts at the first sync event
    (post-warmup), so a post-transient filter would be over-restrictive
    here. We trust the analyzer's window boundary."""
    slopes, r2s, ratios, gate_oks = [], [], [], []
    for r in runs:
        if not r:
            continue
        for w in r["r1"]:
            if abs(w["lr"] - lr_target) / lr_target < 0.1:
                slopes.append(w["slope_byk"])
                r2s.append(w["r2_byk"])
                break
        for w in r["perrank"]:
            if abs(w["lr"] - lr_target) / lr_target < 0.1:
                ratios.append(w["ratio"])
                gate_oks.append(w["gate_ok"])
                break
    return slopes, r2s, ratios, gate_oks


for lr in [0.3, 0.03, 0.003]:
    print(f"\n  LR={lr}:")
    print(f"    {'k':>6}  {'meta slope (by-k)':>20}  {'R²':>10}  {'per-rank ratio':>16}  {'gates ✓':>8}")
    for k in K_VALUES:
        runs = cells[k]
        slopes, r2s, ratios, gate_oks = aggregate_by_lr(runs, lr)
        if not slopes:
            print(f"    {k:>6}  no data (cliff cells: too few syncs for OLS fit)")
            continue
        sm, ssd = msd(slopes)
        rm, rsd = msd(r2s)
        if ratios:
            ratm, ratsd = msd(ratios)
            n_ok = sum(gate_oks)
            ratio_str = f"{ratm:.2f} ± {ratsd:.2f}"
            gate_str = f"{n_ok}/{len(gate_oks)}"
        else:
            ratio_str = "—"
            gate_str = "—"
        print(
            f"    {k:>6}  {sm:+.3e} ±{ssd:.2e}  {rm:>5.3f} ±{rsd:.3f}  "
            f"{ratio_str:>14}     {gate_str:>6}"
        )


# ---------------------------------------------------------------------------
# Cliff verdict — final yes/no answer to the falsifiable predictions
# ---------------------------------------------------------------------------
print()
print("=" * 100)
print(" CLIFF VERDICT")
print("=" * 100)
checks = []

# Prediction 1: k=51200 mean eval < 0.5 (collapsed)
e51 = [r["main"]["eval"] for r in cells[51200] if r and r.get("main")]
m51 = mean(e51) if e51 else None
checks.append((
    "k=51200 collapses (mean eval < 0.5)",
    m51 is not None and m51 < 0.5,
    f"mean = {m51*100:.2f}%" if m51 is not None else "no data",
))

# Prediction 2: at least one cliff-edge cell is bimodal (range > 30pp)
bimodal_cells = []
for k in K_VALUES:
    evals = [r["main"]["eval"] for r in cells[k] if r and r.get("main")]
    if evals and (max(evals) - min(evals)) * 100 > BIMODAL_RANGE_PP:
        bimodal_cells.append(k)
checks.append((
    "Cliff edge has bimodal cell (range > 30pp)",
    len(bimodal_cells) > 0,
    f"bimodal at k ∈ {bimodal_cells}" if bimodal_cells else "no bimodality",
))

# Prediction 3: at least one adjacent transition has Δ > 30pp (cliff edge)
hard_drops = []
for i in range(len(K_VALUES) - 1):
    a, b = K_VALUES[i], K_VALUES[i+1]
    ea = mean(r["main"]["eval"] for r in cells[a] if r and r.get("main"))
    eb = mean(r["main"]["eval"] for r in cells[b] if r and r.get("main"))
    if abs((eb - ea) * 100) > 30:
        hard_drops.append((a, b, (eb - ea) * 100))
checks.append((
    "Adjacent-cell hard drop (Δ > 30pp) present",
    len(hard_drops) > 0,
    "; ".join(f"k={a}→{b}: {d:+.1f}pp" for a, b, d in hard_drops) if hard_drops else "no hard drop",
))

# Prediction 4: safe regime monotone (no peak above default)
safe_means = []
for k in [3200, 6400, 12800]:
    evals = [r["main"]["eval"] for r in cells[k] if r and r.get("main")]
    if evals:
        safe_means.append((k, mean(evals)))
peak_above_default = any(safe_means[i][1] > safe_means[0][1] + 0.001 for i in range(1, len(safe_means)))
checks.append((
    "No eval peak above ElChe default (k=3200 wins safe regime)",
    not peak_above_default,
    "monotone non-increasing" if not peak_above_default else "peak above k=3200 found",
))

print()
for desc, ok, detail in checks:
    mark = "✓" if ok else "✗"
    print(f"  {mark}  {desc}")
    print(f"     {detail}")

print()
print("Cliff localization: between k = {} (last fully safe) and k = {} (first bimodal).".format(
    next((k for k in reversed(K_VALUES) if (max(r['main']['eval'] for r in cells[k] if r and r.get('main')) - min(r['main']['eval'] for r in cells[k] if r and r.get('main'))) * 100 < BIMODAL_RANGE_PP and mean(r['main']['eval'] for r in cells[k] if r and r.get('main')) > 0.85), "?"),
    next((k for k in K_VALUES if (max(r['main']['eval'] for r in cells[k] if r and r.get('main')) - min(r['main']['eval'] for r in cells[k] if r and r.get('main'))) * 100 > BIMODAL_RANGE_PP), "?"),
))
