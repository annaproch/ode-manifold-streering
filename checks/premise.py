"""Check 3: can the readout see off-manifold damage at all?

Displaces real activations by the same distance in three directions and decodes each one:

  on_manifold     toward a real target donor's concept coordinates (positive control)
  in_U_off_ring   inside the concept subspace but off the local curve
  off_U           orthogonal to the concept subspace entirely

If the answer distribution degrades no more off-manifold than on-manifold, then the
endpoint metric cannot register off-manifold damage, every arm will tie, and the readout
has to change before the comparison is worth running.

Needs the model. Roughly 200-400 forwards.

    python checks/premise.py --run pilot-runs/mps-smoke-v1 --batch-size 16
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _common import activations, chosen_fit, load_fit, load_run, target_index
from pilot.model import Runtime
from pilot.storage import read_json

DISTANCES = [0.0, 2.0, 5.0, 10.0, 20.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--fit", help="default: the fit the causal gate selected")
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--batch-size", type=int, help="override; a diagnostic, so it need not match the calibration")
    ap.add_argument("--distances", nargs="*", type=float, default=DISTANCES)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run = Path(args.run)
    cfg, rows, probabilities, pieces = load_run(run)
    if args.batch_size:
        cfg["model"]["batch_size"] = args.batch_size
    target = target_index(cfg)
    choice = chosen_fit(run)
    probe, geometry, meta = load_fit(run, args.fit or choice["fits"][0])
    layer = meta["layer"]
    x_all = activations(pieces, layer)

    picked = [i for i, r in enumerate(rows)
              if r["split"] == "validation" and r["primary"] and r["template"] == "main"][:args.rows]
    if not picked:
        raise SystemExit("No validation stimuli")
    source = x_all[picked]
    u = geometry.u
    rng = np.random.default_rng(args.seed)

    donors = geometry.x[geometry.labels == target]
    residual = lambda v: v - geometry.coordinates(v) @ u.T
    nearest = np.argmin(((residual(source)[:, None] - residual(donors)[None]) ** 2).sum(2), axis=1)
    toward = ((donors[nearest] - source) @ u) @ u.T

    def normalise(v):
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)

    random_full = rng.normal(size=source.shape)
    off_u = normalise(random_full - (random_full @ u) @ u.T)
    in_u = (rng.normal(size=(len(source), u.shape[1])) @ u.T)
    tangent = np.stack([geometry.basis(geometry.coordinates(x)) for x in source])
    in_u_z = np.einsum("nd,dk->nk", in_u, u)
    off_ring_z = in_u_z - np.einsum("nmk,nm->nk", tangent, np.einsum("nmk,nk->nm", tangent, in_u_z))
    directions = {"on_manifold": normalise(toward), "in_U_off_ring": normalise(off_ring_z @ u.T), "off_U": off_u}

    revision = read_json(run / "collection" / "model.json")["revision"]
    runtime = Runtime(cfg, revision)
    baseline = probabilities[picked]
    print(f"run {run}   layer {layer}   k={meta['dimension']}   rows={len(picked)}")
    print(f"oracle displacement for reference: {np.linalg.norm(toward, axis=1).mean():.1f} raw units\n")
    print(f"{'direction':<15} {'d':>5} {'p(start class)':>15} {'p(target)':>10} {'other':>8} {'argmax changed':>16}")
    for name, direction in directions.items():
        for d in args.distances:
            states = source + d * direction
            probability, _, _, _ = runtime.batched([rows[i] for i in picked], patch_layer=layer,
                                                   replacements=states)
            start = baseline.argmax(1)
            p_start = probability[np.arange(len(start)), start].mean()
            changed = np.mean(probability.argmax(1) != start)
            print(f"{name:<15} {d:>5.0f} {p_start:>15.3f} {probability[:, target].mean():>10.3f} "
                  f"{probability[:, -1].mean():>8.3f} {changed:>16.2f}")
            if d == 0:
                error = float(np.max(np.abs(probability - baseline)))
                print(f"{'':<15} {'':>5} no-op check against the collection: max error {error:.2e}")
                if error > cfg["model"]["noop_tolerance"]:
                    raise SystemExit("Identity patch does not reproduce the collection; stop and check device/precision")
        print()
    print("Reading: compare 'other' and 'p(start class)' at equal d. If off_U and in_U_off_ring")
    print("degrade the output no more than on_manifold does, the readout cannot see off-manifold")
    print("damage and the endpoint hypothesis is untestable with this readout.")


if __name__ == "__main__":
    main()
