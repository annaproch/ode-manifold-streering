"""Explicit CLI: each invocation executes only the stage named by the user."""
from __future__ import annotations

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description="CPU/CUDA/MPS colour-manifold pilot; no automatic stage chaining")
    parser.add_argument("--config", required=True, help="JSON overrides for the documented defaults")
    sub = parser.add_subparsers(dest="stage", required=True)
    for command in ("plan", "data", "readout-check", "collect", "calibrate-noop", "fit", "patch", "diagnostics", "select"):
        sub.add_parser(command)
    p = sub.add_parser("estimate")
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--paths", action="store_true")
    pre = sub.add_parser("preflight")
    pre.add_argument("--odesteer-src", help="Optional existing pinned ODESteer src/ for numerical parity")
    for command in ("validate", "validate-paths", "test"):
        p = sub.add_parser(command)
        p.add_argument("--template", choices=("main", "paraphrase"), default="main")
    p = sub.add_parser("report")
    p.add_argument("--split", choices=("validation", "validation-paths", "test"), default="test")
    p.add_argument("--template", choices=("main", "paraphrase"), default="main")
    args = parser.parse_args()
    from .config import load
    cfg = load(args.config)
    if args.stage == "plan":
        result = {"configuration": cfg, "sequence": ["data", "estimate", "preflight", "readout-check", "collect", "calibrate-noop", "fit", "diagnostics", "patch", "validate", "select", "validate-paths", "test", "report"],
                  "note": "Configuration display only. No model loaded, files created, or experiment run."}
    elif args.stage == "data":
        from .data import make
        result = make(cfg)
    elif args.stage == "preflight":
        from .preflight import run
        result = run(cfg, args.odesteer_src)
    elif args.stage == "diagnostics":
        from .diagnostics import run
        result = run(cfg)
    elif args.stage == "report":
        from .reporting import report
        result = report(cfg, args.split, args.template)
    else:
        from . import pipeline
        if args.stage == "estimate":
            result = pipeline.estimate_cost(cfg, args.split, paths=args.paths)
        elif args.stage in {"validate", "validate-paths", "test"}:
            if args.stage != "test" and args.template != "main":
                raise ValueError("The paraphrase is a held-out generalization test; evaluate it after freezing settings")
            result = pipeline.evaluation(cfg, "test" if args.stage == "test" else "validation", args.template, paths=args.stage == "validate-paths")
        else:
            functions = {"readout-check": pipeline.readout_check, "collect": pipeline.collect,
                         "fit": pipeline.fit, "patch": pipeline.patch, "select": pipeline.select_operating_points,
                         "calibrate-noop": pipeline.calibrate_noop}
            result = functions[args.stage](cfg)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, AssertionError) as exc:
        print(f"Stopped: {exc}", file=sys.stderr)
        sys.exit(2)
