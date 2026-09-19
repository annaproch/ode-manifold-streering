"""Regenerate every toy table in experiment-plan.md §9.

Usage:  python toy/run_all.py [--worlds 3] [--seeds 2]
Writes toy/results/runs.jsonl (one row per run) and toy/results/tables.md.

Each (world, seed) pair is an independent replicate: a world is a fresh ring, noise, random
sketch, classifier and KDE; a seed is a fresh set of 32 noisy starts.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics
import time
from collections import defaultdict

from ring_toy import Cond, make_world, run

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")

# §9.1: the first-review comparison, short T.
FIRST = [
    Cond("A"), Cond("B"), Cond("C"),
    Cond("D", lam=1.0),
    Cond("E1", lam=0.3), Cond("E1", lam=1.0), Cond("E1", lam=3.0),
    Cond("E1", lam=1.0, perp=True), Cond("E1", lam=3.0, perp=True), Cond("E1", lam=10.0, perp=True),
    Cond("E2", retraction="ms", r=3),
]
# §9.3: the final method and its ablations over the full grid.
SWEEP = [
    Cond("B"), Cond("B", stepctl="stop"),
    Cond("E2", retraction="ms", r=3),
    Cond("E2", retraction="scms", r=3),
    Cond("E2", retraction="scms", r=3, stepctl="stop"),
    Cond("E2", retraction="scms", r=3, stepctl="halve"),
]
R_SWEEP = [Cond("E2", retraction=ret, r=r, stepctl=ctl)
           for (ret, ctl), r in itertools.product([("ms", "none"), ("scms", "none"), ("scms", "halve")], (1, 3, 10, 30))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rows, t0 = [], time.time()
    for wi in range(a.worlds):
        w = make_world(seed=wi)
        print(f"world {wi}: classifier train acc {w.train_acc:.3f}, KDE bandwidth {w.bw:.3f}", flush=True)
        jobs = [("first", c, s, T) for c, s, T in itertools.product(FIRST, (120, 180), (4, 8, 10))]
        jobs += [("sweep", c, s, T) for c, s, T in itertools.product(SWEEP, (60, 90, 120, 150, 180), (10, 20, 40))]
        jobs += [("r_sweep", c, s, T) for c, s, T in itertools.product(R_SWEEP, (120, 180), (20, 40))]
        for (block, c, s, T), seed in itertools.product(jobs, range(a.seeds)):
            res = run(w, c, s, T, seed=seed)
            rows.append(dict(block=block, cond=c.label, start=s, T=T, world=wi, seed=seed,
                             train_acc=w.train_acc, bw=w.bw, **res))
        print(f"  done, {len(rows)} runs, {time.time() - t0:.0f}s", flush=True)
    with open(os.path.join(OUT, "runs.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT, "tables.md"), "w") as f:
        f.write(tables(rows, a.worlds, a.seeds))
    print(open(os.path.join(OUT, "tables.md")).read())


def agg(rows, key):
    g = defaultdict(list)
    for r in rows:
        g[key(r)].append(r)
    return g


def mean_sd(vals):
    m = statistics.mean(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return m, sd


def tables(rows, n_worlds, n_seeds) -> str:
    n = n_worlds * n_seeds
    out = [f"# Toy results\n\nMean ± SD over {n} replicates ({n_worlds} worlds × {n_seeds} start seeds, 32 starts each).",
           "success = final hue within ±30° of red; min_r = lowest radius ÷ R (median over starts); "
           "swing = hue range over the last 5 of 40 steps.\n"]
    accs = sorted({(r['world'], round(r['train_acc'], 3), round(r['bw'], 3)) for r in rows})
    out.append("Worlds (classifier train accuracy, KDE bandwidth): " + ", ".join(f"{a} / {b}" for _, a, b in accs) + "\n")

    def cell(v, f):
        m, sd = mean_sd([x[f] for x in v])
        return f"{m:.2f}±{sd:.2f}"

    # §9.1
    out.append("## First review (block `first`)\n")
    out.append("| condition | start | T | success | min_r | end |hue| | logp_end | clf_pos | offplane |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    g = agg([r for r in rows if r["block"] == "first"], lambda r: (r["cond"], r["start"], r["T"]))
    for c in [x.label for x in FIRST]:
        for s in (120, 180):
            for T in (4, 8, 10):
                v = g[(c, s, T)]
                out.append(f"| {c} | {s} | {T} | {cell(v,'success')} | {cell(v,'min_r')} | {cell(v,'end_hue')} | "
                           f"{cell(v,'logp_end')} | {cell(v,'clf_pos')} | {cell(v,'offplane')} |")
    # §9.3 grid
    out.append("\n## Final method grid (block `sweep`), r = 3\n")
    out.append("Cells: success / min_r / swing.\n")
    out.append("| condition | start | T=10 | T=20 | T=40 |")
    out.append("|---|---|---|---|---|")
    g = agg([r for r in rows if r["block"] == "sweep"], lambda r: (r["cond"], r["start"], r["T"]))
    for c in [x.label for x in SWEEP]:
        for s in (60, 90, 120, 150, 180):
            cells = []
            for T in (10, 20, 40):
                v = g[(c, s, T)]
                cells.append(f"{cell(v,'success')} / {cell(v,'min_r')} / {cell(v,'swing')}")
            out.append(f"| {c} | {s} | " + " | ".join(cells) + " |")
    # r sweep
    out.append("\n## Retraction steps r (block `r_sweep`)\n")
    out.append("Cells: success / min_r.\n")
    out.append("| condition | start | T | r=1 | r=3 | r=10 | r=30 |")
    out.append("|---|---|---|---|---|---|---|")
    g = agg([r for r in rows if r["block"] == "r_sweep"], lambda r: (r["cond"].replace(" r=" + r["cond"].split(" r=")[1].split(" ")[0], ""), r["start"], r["T"], r["cond"]))
    base = sorted({k[0] for k in g}, key=lambda s: (("scms" in s), ("halve" in s)))
    for b in base:
        for s in (120, 180):
            for T in (20, 40):
                cells = []
                for r_ in (1, 3, 10, 30):
                    lab = [k for k in g if k[0] == b and k[1] == s and k[2] == T and f" r={r_}" in k[3]
                           and k[3].split(f" r={r_}")[1] in ("", " +halve")][0]
                    v = g[lab]
                    cells.append(f"{cell(v,'success')} / {cell(v,'min_r')}")
                out.append(f"| {b} | {s} | {T} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    main()
