"""Check 1: what is the best any on-manifold method could do?

Scores the donor patches already saved by the `patch` stage in the same metric the
steering arms are judged by (Bhattacharyya distance to natural target outputs), so the
ceiling is fixed before any method is run. No model forwards.

    python checks/oracle_ceiling.py --run pilot-runs/mps-smoke-v1
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

from _common import (bhattacharyya_to_nearest, chosen_fit, load_run, target_index)
from pilot.storage import load_arrays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run = Path(args.run)
    cfg, rows, probabilities, _ = load_run(run)
    target = target_index(cfg)
    choice = chosen_fit(run)
    patch_file = run / "patches" / f"layer{choice['layer']}-k{choice['dimension']}.npz"
    if not patch_file.exists():
        raise SystemExit(f"Missing {patch_file}")

    def reference(split):
        keep = [i for i, r in enumerate(rows)
                if r["primary"] and r["template"] == "main" and (split is None or r["split"] == split)]
        p = probabilities[keep]
        return p[p.argmax(1) == target]

    held_out = reference("reference")
    any_split = reference(None)
    print(f"run              {run}")
    print(f"selected fit     layer {choice['layer']}, k={choice['dimension']}")
    print(f"target class     {cfg['fit']['target']}\n")
    print(f"natural target outputs: {len(held_out)} in the reference split, {len(any_split)} across all primary stimuli")
    if len(held_out) < 5:
        print("  WARNING: the reference split is too thin to be a stable reference; widen it in the next run.")
    saved = load_arrays(patch_file)
    for name, ref in (("reference split", held_out), ("all primary", any_split)):
        if len(ref) == 0:
            continue
        print(f"\nagainst the {name} reference (n={len(ref)}):")
        print(f"  {'condition':<10} {'success':>8} {'p(target)':>10} {'other':>8} {'endpoint BC':>12}")
        for condition in ("identity", "full", "subspace"):
            if condition not in saved:
                continue
            p = saved[condition]
            bc = bhattacharyya_to_nearest(p, ref)
            print(f"  {condition:<10} {np.mean(p.argmax(1) == target):>8.3f} {p[:, target].mean():>10.3f} "
                  f"{p[:, -1].mean():>8.3f} {bc.mean():>12.4f}")
        within = bhattacharyya_to_nearest(ref, ref if len(ref) > 1 else ref)
        print(f"  {'natural':<10} {'1.000':>8} {ref[:, target].mean():>10.3f} {ref[:, -1].mean():>8.3f} "
              f"{within.mean():>12.4f}  <- floor (reference scored against itself)")
    print("\nReading: the 'subspace' row is the ceiling for any on-manifold steering method at this site.")
    print("Predeclare it before running the comparison; arms that land near it cannot be separated.")


if __name__ == "__main__":
    main()
