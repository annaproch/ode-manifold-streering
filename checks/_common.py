"""Shared loading helpers for the pre-run checks. Read-only: nothing here writes to a run directory."""
from __future__ import annotations

import glob
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pilot.data import ANGLES, LABELS  # noqa: E402
from pilot.geometry import Geometry  # noqa: E402
from pilot.probe import Probe  # noqa: E402
from pilot.storage import load_arrays, read_json  # noqa: E402


def load_run(run):
    """Return (cfg, rows, probabilities, ids) for a finished run directory."""
    run = Path(run)
    plan = json.loads((run / "effective-plan.json").read_text())
    cfg = plan.get("configuration", plan)
    rows = [json.loads(line) for line in (run / "stimuli.jsonl").read_text().splitlines() if line.strip()]
    shards = sorted(glob.glob(str(run / "collection" / "shards" / "*.npz")))
    if not shards:
        raise SystemExit(f"No collection shards in {run}")
    pieces = [np.load(s) for s in shards]
    ids = np.concatenate([p["ids"] for p in pieces])
    if list(ids) != [r["id"] for r in rows]:
        raise SystemExit("Collection order does not match stimuli.jsonl")
    probabilities = np.concatenate([p["probabilities"] for p in pieces])
    return cfg, rows, probabilities, pieces


def activations(pieces, layer):
    key = f"layer_{layer}"
    if key not in pieces[0]:
        raise SystemExit(f"Layer {layer} was not captured in this collection")
    return np.concatenate([p[key] for p in pieces]).astype(np.float64)


def load_fit(run, name):
    folder = Path(run) / "fits" / name
    arrays = load_arrays(folder / "arrays.npz")
    meta = read_json(folder / "metadata.json")
    return Probe.restore(arrays), Geometry(arrays, meta["geometry"]), meta


def chosen_fit(run):
    """The fit the causal gate selected, or the only passing one."""
    selection = read_json(Path(run) / "selection.json")
    if not selection.get("passed"):
        raise SystemExit("This run's causal gate did not pass; nothing to check")
    return selection["choice"]


def bhattacharyya_to_nearest(p, reference):
    """-log of the largest Bhattacharyya overlap with any reference distribution."""
    p, reference = np.atleast_2d(p), np.atleast_2d(reference)
    overlap = np.sqrt(np.clip(p, 0, 1)) @ np.sqrt(np.clip(reference, 0, 1)).T
    return -np.log(np.clip(overlap.max(axis=1), 1e-12, 1))


def target_index(cfg):
    return LABELS.index(cfg["fit"]["target"])


def band_of(row, cfg, target):
    from pilot.data import circular_distance
    distance = float(circular_distance(row["hue"], ANGLES[target]))
    return min(cfg["evaluation"]["source_bands"], key=lambda b: abs(b - distance))
