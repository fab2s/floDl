#!/usr/bin/env python3
"""Read an overshoot x EASGD sweep and print the table its predictions need.

Companion to ../overshoot-sweep.sh. Everything here comes out of artifacts
the run already writes; nothing needs a flag that does not exist.

The one non-obvious quantity is in-sync GPU utilisation. A rank's samples
live in timeline.json's `rank_samples` and the averaging windows live in
the same file's `sync_start` / `sync_end` events, so "was this rank busy
while the cohort was averaging" is an interval intersection nothing else
in the tree performs. That number is the whole point of the sweep: the
fast rank's idle DURING the reduce is what overshoot is supposed to
recover.

It is also, on its own, not the verdict. Overshoot raises utilisation
whether or not the work survives writeback (at easgd_alpha=1.0 the blend
is a full overwrite and every overshoot batch is discarded), so the table
carries eval CE and the divergence trajectory beside it.

Usage:  python3 overshoot_report.py <sweep-dir>
"""

import json
import os
import re
import sys

# `TimelineSummary` counts a GPU idle below 5% compute util; matching it
# here keeps this table comparable with the `idle=[...]` line the harness
# prints, rather than inventing a second threshold that means almost the
# same thing.
IDLE_PCT = 5

CELL = os.path.join("olmo-graph", "cpu-async-diloco")


def sync_spans(events):
    """(start, end) ms pairs for each averaging window."""
    spans, start = [], None
    for e in events:
        if e.get("k") == "sync_start":
            start = e["t"]
        elif e.get("k") == "sync_end" and start is not None:
            spans.append((start, e["t"]))
            start = None
    return spans


def in_any(t, spans):
    return any(a <= t <= b for a, b in spans)


def stats(values):
    if not values:
        return None
    n = len(values)
    return {
        "n": n,
        "mean": sum(values) / n,
        "idle": sum(1 for v in values if v < IDLE_PCT) / n * 100.0,
    }


def median(values):
    if not values:
        return None
    s = sorted(values)
    return s[len(s) // 2]


def read_provenance(path):
    """arm / max_overshoot / easgd_alpha as recorded by the sweep."""
    out = {}
    try:
        with open(path) as fh:
            for line in fh:
                if ":" in line:
                    k, v = line.split(":", 1)
                    out[k.strip()] = v.strip()
    except OSError:
        pass
    return out


def read_log(path):
    """Wall seconds, final train loss and final eval from training.log.

    training.log is the authoritative completion record: it is the
    binary's own per-write file logger, so it survives a killed launcher
    whose block-buffered stdout stopped mid-epoch.
    """
    out = {"wall": None, "loss": None, "eval": None, "epochs": 0}
    try:
        with open(path) as fh:
            for line in fh:
                m = re.match(r"^epoch (\d+): loss=([0-9.]+)", line)
                if m:
                    out["epochs"] += 1
                    out["loss"] = float(m.group(2))
                    continue
                m = re.match(r"^final eval=([0-9.]+)", line)
                if m:
                    out["eval"] = float(m.group(1))
                    continue
                m = re.match(r"^# total: ([0-9.]+)s", line)
                if m:
                    out["wall"] = float(m.group(1))
    except OSError:
        pass
    return out


def read_arm(arm_dir):
    tl_path = os.path.join(arm_dir, CELL, "timeline.json")
    if not os.path.exists(tl_path):
        return None
    # A killed run leaves a zero-length or half-written timeline. Reading
    # sweeps that were interrupted is the normal case for this tool, not the
    # exception, so a corrupt cell is skipped rather than taking the report
    # down with it: the arms that DID finish are still worth reading, and a
    # rig that crashed is exactly when you want to see them.
    try:
        with open(tl_path) as fh:
            tl = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  skipping {os.path.basename(arm_dir)}: unreadable timeline ({exc})")
        return None

    events = tl.get("events", [])
    spans = sync_spans(events)
    samples = tl.get("samples", [])
    wall_ms = samples[-1]["t"] if samples else 0

    row = {
        "prov": read_provenance(os.path.join(arm_dir, CELL, "provenance.txt")),
        "log": read_log(os.path.join(arm_dir, CELL, "training.log")),
        "syncs": len(spans),
        "sync_ms": [b - a for a, b in spans],
        "wall_ms": wall_ms,
        "ranks": {},
    }

    # Per-rank utilisation, split by whether the sample landed inside an
    # averaging window. Ranks are keyed by (rank, device) because a host
    # can carry more than one.
    buckets = {}
    for s in tl.get("rank_samples", []):
        inside = in_any(s["t"], spans)
        for g in s.get("gpus", []):
            key = (s["rank"], g["d"], s.get("host", "?"))
            b = buckets.setdefault(key, {"in": [], "out": []})
            b["in" if inside else "out"].append(g["u"])
    for key, b in buckets.items():
        row["ranks"][key] = {"in": stats(b["in"]), "out": stats(b["out"])}

    divs = [e for e in events if e.get("k") == "div"]
    row["d"] = [e["d"] for e in divs]
    # WORK CONSISTENCY. A deep overshoot lets a rank spend work from a span
    # it has not been re-credited for, so "did every arm execute exactly the
    # same total?" is the question that catches lost or double-counted
    # samples. Every arm in a sweep trains the same corpus, so cross-arm
    # equality of the total IS the check: any arm that differs is the bug.
    row["total_steps"] = sum(e["k_used"] for e in divs)
    row["k_series"] = [e["k_used"] for e in divs]
    row["epochs_aggregated"] = sum(1 for e in events if e.get("k") == "div_epoch")
    # Steady-state window geometry: the anchor ramps over the opening
    # cycles, so the back half is what the run actually settled at.
    tail = divs[len(divs) // 2:]
    row["k_max"] = median([e["k_max"] for e in tail])
    row["k_used"] = median([e["k_used"] for e in tail])
    return row


def fmt(v, spec="{:.1f}", dash="-"):
    return dash if v is None else spec.format(v)


def main(root):
    arms = sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d, CELL))
    )
    if not arms:
        print(f"no cells under {root}/*/{CELL}")
        return 1

    rows = {}
    for arm in arms:
        r = read_arm(os.path.join(root, arm))
        if r:
            rows[arm] = r

    print(f"overshoot x EASGD sweep: {root}\n")

    print("ARM CONFIG AND OUTCOME")
    hdr = "{:<5}{:>9}{:>8}{:>9}{:>7}{:>9}{:>9}{:>10}"
    print(hdr.format("arm", "overshoot", "alpha", "wall_s", "syncs",
                     "sync_s", "sync_%", "eval_CE"))
    for arm in arms:
        r = rows.get(arm)
        if not r:
            continue
        p, lg = r["prov"], r["log"]
        wall = lg["wall"] or (r["wall_ms"] / 1000.0)
        sync_s = sum(r["sync_ms"]) / 1000.0
        print(hdr.format(
            arm,
            p.get("max_overshoot", "?"),
            p.get("easgd_alpha", "?"),
            fmt(wall),
            r["syncs"],
            fmt(sync_s),
            fmt(sync_s / wall * 100.0 if wall else None),
            fmt(lg["eval"], "{:.4f}"),
        ))

    print("\nGPU UTIL: INSIDE vs OUTSIDE THE AVERAGING WINDOW")
    print("(prediction: moves with overshoot, NOT with alpha)")
    uhdr = "{:<5}{:<22}{:>10}{:>10}{:>11}{:>11}"
    print(uhdr.format("arm", "rank/dev/host", "in_mean", "out_mean",
                      "in_idle%", "out_idle%"))
    for arm in arms:
        r = rows.get(arm)
        if not r:
            continue
        for key in sorted(r["ranks"]):
            rank, dev, host = key
            u = r["ranks"][key]
            if not u["in"] or not u["out"]:
                continue
            print(uhdr.format(
                arm,
                f"r{rank} cuda{dev} {host}"[:21],
                fmt(u["in"]["mean"]),
                fmt(u["out"]["mean"]),
                fmt(u["in"]["idle"]),
                fmt(u["out"]["idle"]),
            ))

    print("\nWINDOW GEOMETRY AND DIVERGENCE")
    print("(prediction: d_raw rises with overshoot AND with (1-alpha); guard floor is 0.3)")
    dhdr = "{:<5}{:>9}{:>9}{:>9}{:>9}{:>9}{:>9}"
    print(dhdr.format("arm", "k_used", "k_max", "d_min", "d_med", "d_peak", "d_end"))
    for arm in arms:
        r = rows.get(arm)
        if not r or not r["d"]:
            continue
        d = r["d"]
        print(dhdr.format(
            arm,
            r["k_used"] if r["k_used"] is not None else "-",
            r["k_max"] if r["k_max"] is not None else "-",
            fmt(min(d), "{:.4f}"),
            fmt(median(d), "{:.4f}"),
            fmt(max(d), "{:.4f}"),
            fmt(d[-1], "{:.4f}"),
        ))

    print("\nWORK CONSISTENCY")
    print("(every arm trains the same corpus: totals MUST match, and the epoch")
    print(" line count is the trainer's own record of a completed pass)")
    whdr = "{:<5}{:>9}{:>8}{:>9}{:>9}{:>8}{:>9}"
    print(whdr.format("arm", "total", "epochs", "aggreg", "windows", "w_min", "w_typ"))
    totals = {}
    for arm in arms:
        r = rows.get(arm)
        if not r or not r["k_series"]:
            continue
        ks = r["k_series"]
        totals[arm] = r["total_steps"]
        # w_min is the run tail (last window is short by construction); the
        # typical window is the median, which is what the geometry claim uses.
        print(whdr.format(arm, r["total_steps"], r["log"]["epochs"],
                          r["epochs_aggregated"], len(ks), min(ks), median(ks)))
    if totals:
        uniq = set(totals.values())
        if len(uniq) == 1:
            print(f"  OK: all {len(totals)} arms executed exactly {uniq.pop()} steps")
        else:
            print(f"  *** MISMATCH: arms did NOT execute equal work: {totals}")
            print("  *** work was lost or double-counted; the eval column is void")

    # The sweep exists for these two differences. Equal throughput gain in
    # both with a quality gain only in the blended pair is the signature
    # that overshoot buys cycles and EASGD converts them into progress.
    print("\nDECISIVE CONTRASTS")
    for hi, lo, note in (("B1", "B0", "alpha=0.5, work retained at (1-a)"),
                         ("C1", "C0", "alpha=1.0, work discarded at writeback"),
                         ("D1", "D0", "alpha=0.25, highest retention")):
        a, b = rows.get(hi), rows.get(lo)
        if not a or not b:
            continue
        wa = a["log"]["wall"] or a["wall_ms"] / 1000.0
        wb = b["log"]["wall"] or b["wall_ms"] / 1000.0
        d_wall = (wa - wb) / wb * 100.0 if wb else None
        ea, eb = a["log"]["eval"], b["log"]["eval"]
        d_eval = (ea - eb) if (ea is not None and eb is not None) else None
        print(f"  {hi}-{lo}  ({note})")
        print(f"    wall    {fmt(d_wall, '{:+.1f}')}%   "
              f"({fmt(wb)}s -> {fmt(wa)}s)")
        print(f"    eval CE {fmt(d_eval, '{:+.4f}')}   "
              f"({fmt(eb, '{:.4f}')} -> {fmt(ea, '{:.4f}')})  "
              f"lower is better; single seed, NOT a quality verdict")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
