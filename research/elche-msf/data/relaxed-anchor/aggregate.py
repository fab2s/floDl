"""Cross-seed aggregate: relaxed-anchor vs default-anchor on nccl-async.
Pulls R1 by-k slopes (meta + per-rank), R² across all 3 LR windows,
cross-rank Pearson r, guard fires, kill-criterion correlations.

The relaxed-anchor cohort lives in this directory; the default-anchor
baseline is cross-read from `passive-observation/` (5 seeds × 2 guards
× nccl-async, default anchor)."""
import re
import os
from statistics import mean, stdev

BASE_DEF = "research/elche-msf/data/passive-observation"
BASE_RLX = "research/elche-msf/data/relaxed-anchor"

def parse_r1(text):
    """Extract R1 main table rows (top-scale: max + mean + epoch_dmean + by-k)."""
    out = []
    sec = re.search(
        r"### R1 informal:.*?\| Model \| Mode.*?\n((?:\|.*\n)+)",
        text, re.DOTALL,
    )
    if not sec:
        return out
    for m in re.finditer(
        r"\| resnet-graph \| (\S+) \| ([\d.e+-]+) \| ([^|]+?) \| (\d+) \| \d+–\d+ \| "
        r"([+-]?[\d.e+-]+) \| ([\d.]+) \| "        # max
        r"([+-]?[\d.e+-]+) \| ([\d.]+) \| (\d+) \| "  # mean + n_ep
        r"([+-]?[\d.e+-]+) \| ([\d.]+) \| "        # epoch dmean
        r"\d+–\d+ \| "                              # k range
        r"([+-]?[\d.e+-]+) \| ([\d.]+) \|",        # by-k
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
    """Extract R1' per-rank by-k slopes section."""
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
            if v == "—":
                slopes.append(None)
            else:
                slopes.append(float(v))
        out.append({
            "lr": float(m.group(2)),
            "epochs": m.group(3).strip(),
            "meta_slope": float(m.group(4)),
            "per_rank_slopes": slopes,
            "ratio": float(m.group(7)),
            "gate_ok": gate_ok,
        })
    return out


def parse_pearson(text):
    out = {}
    for m in re.finditer(
        r"\| resnet-graph \| \S+ \| rank(\d) ↔ rank(\d) \| ([+-][\d.]+) \|",
        text,
    ):
        out[(int(m.group(1)), int(m.group(2)))] = float(m.group(3))
    return out


def parse_guard(text):
    sec = re.search(
        r"\| resnet-graph \| (\S+) \| (\d+) \([^)]*\) \| (\d+) \([^)]*\) \| (\d+|—) \|",
        text,
    )
    if not sec:
        return None
    return {"current": int(sec.group(2)), "msf": int(sec.group(3))}


def parse_kill(text):
    sec = re.search(
        r"### Predictive Value \(Phase-1 kill criterion\).*?"
        r"\| resnet-graph \| \S+ \| \d+ \| ([+-][\d.]+) \| \d+ \| ([+-][\d.]+) \| ([+-][\d.]+) \|",
        text, re.DOTALL,
    )
    if not sec:
        return None
    return {
        "r_lambda_to_lnD": float(sec.group(1)),
        "r_lambda_mean_to_eval": float(sec.group(2)),
        "r_lambda_ema_to_eval": float(sec.group(3)),
    }


def parse_main(text):
    sec = re.search(
        r"\| (\S+) \| ([\d.]+) \| ([\d.]+) \| [+-][\d.]+ \| ([\d.]+) \| (\d+) \|",
        text,
    )
    if not sec:
        return None
    return {
        "loss": float(sec.group(2)),
        "eval": float(sec.group(3)),
        "total_s": float(sec.group(4)),
        "syncs": int(sec.group(5)),
    }


def load_seed(base, dirname):
    path = f"{base}/{dirname}/report.md"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        text = f.read()
    return {
        "r1": parse_r1(text),
        "perrank": parse_perrank_r1(text),
        "pearson": parse_pearson(text),
        "guard": parse_guard(text),
        "kill": parse_kill(text),
        "main": parse_main(text),
    }


def msd(xs):
    n = len(xs)
    if n == 0: return None, None
    m = sum(xs)/n
    if n == 1: return m, 0.0
    return m, (sum((x-m)**2 for x in xs)/(n-1))**0.5


def fmt(m, sd, p=3):
    if m is None: return "—"
    return f"{m:+.{p}e} ± {sd:.{p}e}" if abs(m) < 0.01 else f"{m:.{p}f} ± {sd:.{p}f}"


# Load both conditions
def load_set(base, name_fmt):
    return [load_seed(base, name_fmt.format(s=s)) for s in range(5)]

default_msf   = load_set(BASE_DEF, "seed-{s}-nccl-async-msf")
default_trend = load_set(BASE_DEF, "seed-{s}-nccl-async-trend")
relaxed_msf   = load_set(BASE_RLX, "seed-{s}-nccl-async-msf-relaxed")
relaxed_trend = load_set(BASE_RLX, "seed-{s}-nccl-async-trend-relaxed")

# Summary: eval comparison
print("="*90)
print(" Summary — final eval (held-out test, 200 epochs)")
print("="*90)
print(f"{'condition':<35} {'mean':>9} {'sd':>7} {'min':>7} {'max':>7}")
for label, runs in [
    ("nccl-async msf   (default)",   default_msf),
    ("nccl-async msf   (relaxed)",   relaxed_msf),
    ("nccl-async trend (default)",   default_trend),
    ("nccl-async trend (relaxed)",   relaxed_trend),
]:
    evals = [r["main"]["eval"] for r in runs if r and r.get("main")]
    m, sd = msd(evals)
    print(f"  {label:<33} {m*100:>7.2f}% ±{sd*100:.2f}  {min(evals)*100:.2f}  {max(evals)*100:.2f}")

# Sync count comparison
print()
print("="*90)
print(" Sync counts per 200-epoch run")
print("="*90)
for label, runs in [
    ("nccl-async msf   (default)",   default_msf),
    ("nccl-async msf   (relaxed)",   relaxed_msf),
    ("nccl-async trend (default)",   default_trend),
    ("nccl-async trend (relaxed)",   relaxed_trend),
]:
    syncs = [r["main"]["syncs"] for r in runs if r and r.get("main")]
    m, sd = msd(syncs)
    print(f"  {label:<33} {m:>5.0f} ±{sd:.0f}  range [{min(syncs)}, {max(syncs)}]")

# Cross-rank Pearson r
print()
print("="*90)
print(" Cross-rank Pearson r (rank pair) — meta-oscillator coupling")
print("="*90)
for label, runs in [
    ("nccl-async msf (default)",   default_msf),
    ("nccl-async msf (relaxed)",   relaxed_msf),
]:
    print(f"\n  {label}:")
    for pair in [(0,1),(0,2),(1,2)]:
        rs = [r["pearson"][pair] for r in runs if r and pair in r["pearson"]]
        m, sd = msd(rs)
        print(f"    rank{pair[0]} ↔ rank{pair[1]}:  {m:+.4f} ± {sd:.4f}  range [{min(rs):+.4f}, {max(rs):+.4f}]")

# R1 by-k by LR window
print()
print("="*90)
print(" R1 by-k axis: log(D_mean) vs k_used per LR window — cross-seed N=5")
print("="*90)

def aggregate_by_lr(runs, lr_target):
    """Pull (meta_slope, r2_byk) for the post-transient row matching lr_target."""
    slopes, r2s, ratios, gate_oks = [], [], [], []
    for r in runs:
        if not r: continue
        # Prefer post-transient row when present
        for w in r["r1"]:
            if abs(w["lr"]-lr_target)/lr_target < 0.1:
                if "post-transient" in w["epochs"] or lr_target != 0.3:
                    slopes.append(w["slope_byk"])
                    r2s.append(w["r2_byk"])
                    break
        # And the per-rank ratio + gate
        for w in r["perrank"]:
            if abs(w["lr"]-lr_target)/lr_target < 0.1:
                if "post-transient" in w["epochs"] or lr_target != 0.3:
                    ratios.append(w["ratio"])
                    gate_oks.append(w["gate_ok"])
                    break
    return slopes, r2s, ratios, gate_oks


for lr in [0.3, 0.03, 0.003]:
    print(f"\n  LR={lr}:")
    print(f"    {'condition':<28} {'meta slope':>14} {'R²(by-k)':>14} {'per-rank ratio':>18} {'gates ✓':>10}")
    for label, runs in [
        ("msf default",   default_msf),
        ("msf relaxed",   relaxed_msf),
        ("trend default", default_trend),
        ("trend relaxed", relaxed_trend),
    ]:
        slopes, r2s, ratios, gate_oks = aggregate_by_lr(runs, lr)
        if not slopes:
            print(f"    {label:<28}  no data")
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
        print(f"    {label:<28} {sm:+.3e} ±{ssd:.2e}  {rm:>5.3f} ±{rsd:.3f}  {ratio_str:>14}     {gate_str:>6}")

# Guard fires
print()
print("="*90)
print(" Guard fire counts per 200-epoch run")
print("="*90)
for label, runs in [
    ("msf default",   default_msf),
    ("msf relaxed",   relaxed_msf),
]:
    cur = [r["guard"]["current"] for r in runs if r and r.get("guard")]
    msf = [r["guard"]["msf"]     for r in runs if r and r.get("guard")]
    if cur:
        cm, csd = msd(cur)
        mm, msd_ = msd(msf)
        print(f"  {label}:  trend-fires {cm:.1f}±{csd:.1f}  msf-fires {mm:.1f}±{msd_:.1f}")
