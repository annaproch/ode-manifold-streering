"""Check 2: do the steering arms end up in different places at all?

Runs the real `steer()` on collected activations and measures how far apart the arms'
endpoints are, and how far each lands from the oracle (a real target donor's coordinates).
No model forwards: if E2 and C-end land closer together than the distance at which the
readout responds, they will tie by construction and the expensive sweep is pointless.

    python checks/arm_separation.py --run pilot-runs/mps-smoke-v1
    python checks/arm_separation.py --run <run> --strengths 0.5 1.0 1.5 --methods B C C-end E2
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

from _common import activations, band_of, chosen_fit, load_fit, load_run, target_index
from pilot.steering import steer

DEFAULT_METHODS = ["A", "R0", "B", "C", "C-end", "E2", "F", "S"]
DEFAULT_STRENGTHS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
PAIRS = [("E2", "C-end"), ("E2", "C"), ("E2", "F"), ("C", "C-end"), ("B", "C"), ("B", "S"), ("A", "B")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--fit", help="fit name; default is the one the causal gate selected")
    ap.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    ap.add_argument("--strengths", nargs="*", type=float, default=DEFAULT_STRENGTHS)
    ap.add_argument("--max-rows", type=int, default=40)
    args = ap.parse_args()
    run = Path(args.run)
    cfg, rows, probabilities, pieces = load_run(run)
    target = target_index(cfg)
    choice = chosen_fit(run)
    name = args.fit or choice["fits"][0]
    probe, geometry, meta = load_fit(run, name)
    layer = meta["layer"]
    x_all = activations(pieces, layer)
    scale = meta["strength_scales"][cfg["steering"]["strength_scale"]]

    selected = [i for i, r in enumerate(rows)
                if r["split"] == "validation" and r["primary"] and r["template"] == "main"
                and probabilities[i].argmax() != target]
    eligible = [i for i in selected if bool(geometry.eligible(x_all[i])[0])]
    used = eligible[:args.max_rows]
    if not used:
        raise SystemExit("No eligible validation starts")

    # Oracle endpoint: move only the concept coordinates to the nearest real target donor,
    # using the same residual-matching rule as the patch stage.
    donors = geometry.x[geometry.labels == target]
    source = x_all[used]
    residual = lambda v: v - geometry.coordinates(v) @ geometry.u.T
    nearest = np.argmin(((residual(source)[:, None] - residual(donors)[None]) ** 2).sum(2), axis=1)
    oracle = source + ((donors[nearest] - source) @ geometry.u) @ geometry.u.T

    print(f"run {run}   fit {name}   layer {layer}   k={meta['dimension']}")
    print(f"strength scale: {cfg['steering']['strength_scale']} = {scale:.2f} raw units per multiplier")
    print(f"rows: {len(selected)} non-target validation starts, {len(eligible)} eligible, {len(used)} used")
    bands = sorted({band_of(rows[i], cfg, target) for i in used})
    print(f"bands present: {bands}")
    print(f"oracle displacement: {np.linalg.norm(oracle - source, axis=1).mean():.2f} raw units\n")

    for multiplier in args.strengths:
        endpoints, frozen = {}, {}
        for method in args.methods:
            out, stuck = [], 0
            for position, index in enumerate(used):
                trajectory = steer(x_all[index], method, multiplier * scale, probe, geometry,
                                   cfg["steering"], target)
                out.append(trajectory.states[-1])
                stuck += trajectory.reason != "ok"
            endpoints[method] = np.asarray(out)
            frozen[method] = stuck
        print(f"T = {multiplier:g}  ({multiplier * scale:.1f} raw units)")
        print(f"  {'method':<7} {'displacement':>13} {'to oracle':>11} {'frozen':>7}")
        for method in args.methods:
            moved = np.linalg.norm(endpoints[method] - source, axis=1).mean()
            to_oracle = np.linalg.norm(endpoints[method] - oracle, axis=1).mean()
            print(f"  {method:<7} {moved:>13.2f} {to_oracle:>11.2f} {frozen[method]:>7d}")
        print(f"  {'pair':<12} {'separation':>11}  (mean distance between the two arms' endpoints)")
        for a, b in PAIRS:
            if a in endpoints and b in endpoints:
                separation = np.linalg.norm(endpoints[a] - endpoints[b], axis=1).mean()
                print(f"  {a + ' vs ' + b:<12} {separation:>11.2f}")
        print()
    print("Reading: an on-manifold displacement of about one oracle distance flips the answer,")
    print("so a pair separated by only a few percent of that cannot produce different outputs.")


if __name__ == "__main__":
    main()
