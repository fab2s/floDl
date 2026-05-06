"""Cross-seed MSF passive-observation verdict — nccl-async only, all three R1 bases,
post-transient subwindow included for warmup."""
import re
import os

BASE = "research/elche-msf/data/passive-observation"

def parse_r1_table(text):
    """Extract R1 OLS rows. Returns list of dicts.
    Format: | resnet-graph | <mode> | <lr> | <epochs label> | <n_evt> | <step range> |
            <slope_max> | <r2_max> | <slope_mean> | <r2_mean> |
            <n_ep> | <slope_epoch> | <r2_epoch> |
    The <epochs label> can be plain '0–99' or '1–99 (post-transient, skipped 1)'."""
    out = []
    section = re.search(
        r"### R1 informal: log\(D\) vs step per LR window.*?\| Model \| Mode.*?\n((?:\|.*\n)+)",
        text, re.DOTALL,
    )
    if not section:
        return out
    rows = section.group(1)
    # Match flexibly — epochs label can contain spaces, parens, dashes.
    for m in re.finditer(
        r"\| resnet-graph \| (\S+) \| ([\d.e+-]+) \| ([^|]+?) \| (\d+) \| \d+–\d+ \| "
        r"([+-]?[\d.e+-]+) \| ([\d.]+) \| "
        r"([+-]?[\d.e+-]+) \| ([\d.]+) \| (\d+) \| "
        r"([+-]?[\d.e+-]+) \| ([\d.]+) \|",
        rows,
    ):
        out.append({
            "mode": m.group(1),
            "lr": float(m.group(2)),
            "epochs": m.group(3).strip(),
            "n_evt": int(m.group(4)),
            "slope_max": float(m.group(5)), "r2_max": float(m.group(6)),
            "slope_mean": float(m.group(7)), "r2_mean": float(m.group(8)),
            "n_ep": int(m.group(9)),
            "slope_epoch": float(m.group(10)), "r2_epoch": float(m.group(11)),
        })
    return out


def parse_pearson(text):
    """Cross-rank Pearson r for D trajectories — meta-oscillator anchor."""
    out = {}
    for m in re.finditer(
        r"\| resnet-graph \| \S+ \| rank(\d) ↔ rank(\d) \| ([+-][\d.]+) \|",
        text,
    ):
        out[(int(m.group(1)), int(m.group(2)))] = float(m.group(3))
    return out


def parse_kill(text):
    sec = re.search(
        r"### Predictive Value \(Phase-1 kill criterion\).*?"
        r"\| resnet-graph \| \S+ \| (\d+) \| ([+-][\d.]+) \| (\d+) \| ([+-][\d.]+) \| ([+-][\d.]+) \|",
        text, re.DOTALL,
    )
    if not sec:
        return None
    return {
        "n_evt": int(sec.group(1)),
        "r_lambda_to_lnD": float(sec.group(2)),
        "n_ep": int(sec.group(3)),
        "r_lambda_mean_to_eval": float(sec.group(4)),
        "r_lambda_ema_to_eval": float(sec.group(5)),
    }


def parse_guard(text):
    sec = re.search(
        r"\| resnet-graph \| (\S+) \| (\d+) \([^)]*\) \| (\d+) \([^)]*\) \| (\d+|—) \|",
        text,
    )
    if not sec:
        return None
    return {
        "current": int(sec.group(2)),
        "msf": int(sec.group(3)),
    }


def parse_main(text):
    """Main results table — eval/loss/total/syncs."""
    sec = re.search(
        r"\| (\S+) \| ([\d.]+) \| ([\d.]+) \| [+-][\d.]+ \| ([\d.]+) \| (\d+) \| ([\d.]+) \|",
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


def load_seed(seed, mode, guard):
    path = f"{BASE}/seed-{seed}-{mode}-{guard}/report.md"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        text = f.read()
    return {
        "r1": parse_r1_table(text),
        "pearson": parse_pearson(text),
        "kill": parse_kill(text),
        "guard": parse_guard(text),
        "main": parse_main(text),
    }


def mean_sd(xs):
    n = len(xs)
    if n == 0:
        return None, None
    m = sum(xs) / n
    if n == 1:
        return m, 0.0
    v = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, v ** 0.5


# ===== nccl-async msf only =====
seeds_msf = [load_seed(s, "nccl-async", "msf") for s in range(5)]
seeds_trend = [load_seed(s, "nccl-async", "trend") for s in range(5)]

print("=" * 78)
print("MSF passive-observation — cross-seed verdict on nccl-async (N=5)")
print("=" * 78)

# Final eval comparison
print("\n--- R3: final eval (held-out test, 200 epochs)")
ev_msf = [s["main"]["eval"] for s in seeds_msf if s and s.get("main")]
ev_tr  = [s["main"]["eval"] for s in seeds_trend if s and s.get("main")]
m_msf, sd_msf = mean_sd(ev_msf)
m_tr, sd_tr = mean_sd(ev_tr)
print(f"  msf:   {m_msf*100:.2f}% ± {sd_msf*100:.2f}  (range [{min(ev_msf)*100:.2f}, {max(ev_msf)*100:.2f}])  N={len(ev_msf)}")
print(f"  trend: {m_tr*100:.2f}% ± {sd_tr*100:.2f}  (range [{min(ev_tr)*100:.2f}, {max(ev_tr)*100:.2f}])  N={len(ev_tr)}")
print(f"  Δ(msf−trend) = {(m_msf-m_tr)*100:+.2f}pp  (within seed range)")

# Meta-oscillator anchor
print("\n--- Meta-oscillator anchor: cross-rank Pearson r(D)")
for pair in [(0,1), (0,2), (1,2)]:
    rs = [s["pearson"][pair] for s in seeds_msf if s and pair in s["pearson"]]
    if rs:
        m, sd = mean_sd(rs)
        print(f"  rank{pair[0]} ↔ rank{pair[1]}:  {m:+.4f} ± {sd:.4f}  (range [{min(rs):+.4f}, {max(rs):+.4f}])")

# R1 — focus on per-epoch d_mean basis (meta-oscillator aware), with sub-windows
print("\n--- R1: log(D) vs step OLS per LR window")
print("    (3 bases: D_max per-event, D_mean per-event, per-epoch d_mean)")
print("    Per-epoch d_mean is the meta-oscillator-aware basis (averages SGD intra-epoch variance).")
print()

# Group rows across seeds by (lr, epochs label) — keeps post-transient subwindows distinct.
key_set = set()
for s in seeds_msf:
    if not s: continue
    for r in s["r1"]:
        key_set.add((r["lr"], r["epochs"]))

# Sort by LR descending, then by epochs label so post-transient comes after primary
def key_order(k):
    lr, ep = k
    has_pt = "post-transient" in ep
    return (-lr, has_pt, ep)

print(f"  {'LR':>6} {'window':>30} {'N_seeds':>8} {'R²(D_max)':>12} {'R²(D_mean)':>12} {'R²(epoch)':>12}")
for key in sorted(key_set, key=key_order):
    lr, ep_label = key
    rows = []
    for s in seeds_msf:
        if not s: continue
        for r in s["r1"]:
            if r["lr"] == lr and r["epochs"] == ep_label:
                rows.append(r)
                break
    if not rows:
        continue
    r2max = [r["r2_max"] for r in rows]
    r2mean = [r["r2_mean"] for r in rows]
    r2ep = [r["r2_epoch"] for r in rows]
    m_max, sd_max = mean_sd(r2max)
    m_mean, sd_mean = mean_sd(r2mean)
    m_ep, sd_ep = mean_sd(r2ep)
    print(f"  {lr:>6.3g} {ep_label:>30} {len(rows):>8} "
          f"{m_max:>5.3f}±{sd_max:.3f}  "
          f"{m_mean:>5.3f}±{sd_mean:.3f}  "
          f"{m_ep:>5.3f}±{sd_ep:.3f}")

# Slopes per LR window (epoch d_mean basis only — the meta-oscillator one)
print("\n--- R1 slopes (epoch d_mean basis, ln(D) per step)")
print(f"  {'LR':>6} {'window':>30} {'slope (mean ± sd)':>26}")
for key in sorted(key_set, key=key_order):
    lr, ep_label = key
    rows = []
    for s in seeds_msf:
        if not s: continue
        for r in s["r1"]:
            if r["lr"] == lr and r["epochs"] == ep_label:
                rows.append(r)
                break
    if not rows:
        continue
    sls = [r["slope_epoch"] for r in rows]
    m, sd = mean_sd(sls)
    pos = sum(1 for x in sls if x > 0)
    print(f"  {lr:>6.3g} {ep_label:>30} {m:+.3e} ± {sd:.2e}  ({pos}/{len(sls)} seeds positive)")

# R5 guard fires
print("\n--- R5: guard fires per 200-epoch run")
cur = [s["guard"]["current"] for s in seeds_msf if s and s.get("guard")]
msf = [s["guard"]["msf"] for s in seeds_msf if s and s.get("guard")]
m_cur, sd_cur = mean_sd(cur)
m_msf_g, sd_msf_g = mean_sd(msf)
print(f"  trend guard (3 rises in D): mean {m_cur:.1f} ± {sd_cur:.1f}  (range [{min(cur)}, {max(cur)}])  N={len(cur)}")
print(f"  MSF guard (λ_ema > 1e-3, sustain 3): mean {m_msf_g:.1f} ± {sd_msf_g:.1f}  (range [{min(msf)}, {max(msf)}])")
print(f"  Reduction:  {(1 - m_msf_g/m_cur)*100:.1f}% fewer false positives")

# Kill criterion
print("\n--- Kill criterion (predictive correlations)")
for key, label in [
    ("r_lambda_to_lnD",        "r(λ_raw → ln D_{t+1})  [tests across-event proxy validity]"),
    ("r_lambda_mean_to_eval",  "r(λ_mean per epoch → eval)  [tests passive-observation success metric]"),
    ("r_lambda_ema_to_eval",   "r(λ_ema end-of-epoch → eval)"),
]:
    vals = [s["kill"][key] for s in seeds_msf if s and s.get("kill")]
    if vals:
        m, sd = mean_sd(vals)
        print(f"  {label}")
        print(f"    mean={m:+.4f} ± {sd:.4f}  range=[{min(vals):+.4f}, {max(vals):+.4f}]")

