"""Small validation-only decoded comparison; no fitting or test-set selection.

python checks/endpoint_comparison.py --run pilot-runs/mps-smoke-v1 --dry-run
python checks/endpoint_comparison.py --run pilot-runs/mps-smoke-v1 --batch-size 4
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from _common import activations, bhattacharyya_to_nearest, chosen_fit, load_fit, load_run, target_index
from pilot.model import Runtime
from pilot.steering import steer
from pilot.storage import atomic, read_json, save_arrays, write_json

METHODS = ("C", "C-end", "E2")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--strengths", nargs="+", type=float, default=[0.5, 1.0, 1.5, 2.0])
    parser.add_argument("--output", help="Default: RUN/checks/endpoint-comparison; use a fresh directory")
    parser.add_argument("--dry-run", action="store_true", help="Inspect sample, references and cost without loading a model")
    args = parser.parse_args()
    strengths = sorted(set(args.strengths))
    if args.rows < 1 or args.batch_size < 1 or any(not np.isfinite(t) or t <= 0 for t in strengths):
        parser.error("Use positive rows, batch size and finite positive strengths")
    run = Path(args.run)
    out = Path(args.output) if args.output else run / "checks" / "endpoint-comparison"
    cfg, rows, probabilities, pieces = load_run(run)
    fit_name = chosen_fit(run)["fits"][0]
    probe, geometry, meta = load_fit(run, fit_name)
    target, layer = target_index(cfg), meta["layer"]
    x = activations(pieces, layer)
    candidates = [i for i, row in enumerate(rows)
                  if row["split"] == "validation" and row["primary"] and row["template"] == "main"
                  and probabilities[i].argmax() != target and geometry.eligible(x[i])[0]]
    candidates.sort(key=lambda i: (rows[i]["hue"], rows[i]["id"]))
    if not candidates:
        raise ValueError("No eligible non-target primary validation starts")
    picked = [candidates[i] for i in np.linspace(0, len(candidates)-1, min(args.rows, len(candidates))).round().astype(int)]
    reference_ids = [i for i, row in enumerate(rows)
                     if row["split"] == "reference" and row["primary"] and row["template"] == "main"
                     and probabilities[i].argmax() == target]
    if not reference_ids:
        raise ValueError("No independent target-reference outputs")
    reference = probabilities[reference_ids]
    scale = meta["strength_scales"][cfg["steering"]["strength_scale"]]
    identity = read_json(run / "collection" / "model.json")
    plan = {"run": str(run), "fit": fit_name, "model": identity, "rows": len(picked),
            "row_ids": [rows[i]["id"] for i in picked], "hues": [rows[i]["hue"] for i in picked],
            "reference_ids": [rows[i]["id"] for i in reference_ids], "reference_count": len(reference),
            "methods": list(METHODS), "strengths": strengths, "scale": scale,
            "batch_size": args.batch_size, "steering": cfg["steering"],
            "success_level": cfg["evaluation"]["success_level"], "success_tolerance": cfg["evaluation"]["success_tolerance"],
            "decode_examples": len(picked)*(1+len(METHODS)*len(strengths)),
            "scope": "Exploratory validation comparison, one fitted seed, endpoints only. No held-out claim or bootstrap.",
            "warning": "Few target references: endpoint distances are provisional." if len(reference) < 8 else None}
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return
    if out.exists():
        raise ValueError(f"Output directory already exists: {out}. Choose a new --output; previous results are preserved.")
    cfg["model"]["batch_size"] = args.batch_size
    runtime = Runtime(cfg, identity["revision"])
    if runtime.identity != identity:
        raise ValueError("Runtime model/tokenizer/precision differs from the collected activations")
    prompts, source = [rows[i] for i in picked], x[picked]
    baseline, _, _, _ = runtime.batched(prompts, patch_layer=layer, replacements=source)
    error = float(np.max(np.abs(baseline-probabilities[picked])))
    if error > cfg["model"]["noop_tolerance"]:
        raise ValueError(f"No-op error {error:.6g} exceeds {cfg['model']['noop_tolerance']}; stop before steering")
    source_paths = [Path(__file__), Path(__file__).with_name("_common.py")]
    source_paths += [Path(__file__).resolve().parents[1]/"pilot"/name for name in ("steering.py", "geometry.py", "probe.py", "model.py")]
    plan["source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    write_json(out / "request.json", plan)
    save_arrays(out / "baseline.npz", ids=np.array(plan["row_ids"]), states=source.astype(np.float32),
                probabilities=baseline, collection_probabilities=probabilities[picked], reference_probabilities=reference)
    cells = []
    for method in METHODS:
        for multiplier in strengths:
            trajectories = [steer(v, method, multiplier*scale, probe, geometry, cfg["steering"], target) for v in source]
            states = np.stack([r.states[-1] for r in trajectories])
            p, _, _, seconds = runtime.batched(prompts, patch_layer=layer, replacements=states)
            bc = bhattacharyya_to_nearest(p, reference)
            success = p.argmax(1) == target
            filename = f"{method}-T{multiplier:g}.npz"
            save_arrays(out / filename, ids=np.array(plan["row_ids"]), states=states.astype(np.float32),
                        probabilities=p, endpoint_bc=bc, success=success)
            cell = {"method": method, "strength": multiplier, "raw_strength": multiplier*scale,
                    "success": float(success.mean()), "p_target": float(p[:, target].mean()),
                    "endpoint_bc": float(bc.mean()), "other": float(p[:, -1].mean()),
                    "correction_length": float(np.mean([r.correction_length for r in trajectories])),
                    "displacement": float(np.mean([r.displacement for r in trajectories])),
                    "reasons": {reason: sum(r.reason == reason for r in trajectories) for reason in sorted({r.reason for r in trajectories})},
                    "model_seconds": seconds, "arrays": filename}
            cells.append(cell)
            write_json(out / "progress.json", {"complete": False, "cells": cells})
            print(json.dumps(cell), flush=True)
    operating = {}
    for method in METHODS:
        passing = [c for c in cells if c["method"] == method and c["success"] >= plan["success_level"]]
        operating[method] = passing[0] if passing else None
    comparable = all(c is not None for c in operating.values())
    if comparable:
        rates = [c["success"] for c in operating.values()]
        comparable = max(rates)-min(rates) <= plan["success_tolerance"]
    result = {"complete": True, "plan": plan, "runtime": runtime.metadata, "identity_max_error": error,
              "cells": cells, "operating_points": operating, "success_comparable": comparable,
              "selection_rule": "Smallest sampled strength reaching the configured success level for each method.",
              "interpretation": "Descriptive validation evidence only. Missing operating points or unmatched success prevent a quality ranking."}
    write_json(out / "summary.json", result)
    write_json(out / "progress.json", {"complete": True, "cells": cells})
    lines = ["# Small decoded endpoint comparison", "", plan["scope"], "",
             f"Starts: {len(picked)}; target references: {len(reference)}; no-op error: {error:.6g}.", "",
             "| Method | T multiplier | Red success | Mean endpoint BC | p(red) | Other |",
             "|---|---:|---:|---:|---:|---:|"]
    for c in cells:
        lines.append(f"| {c['method']} | {c['strength']:g} | {c['success']:.3f} | {c['endpoint_bc']:.5f} | {c['p_target']:.3f} | {c['other']:.3f} |")
    lines += ["", f"Success comparable at selected points: **{comparable}**.", "",
              "Operating points and complete nine-bin probabilities are saved alongside this report.",
              result["interpretation"]]
    if plan["warning"]:
        lines += ["", plan["warning"]]
    atomic(out / "report.md", "\n".join(lines)+"\n")
    print(f"Saved {out / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
