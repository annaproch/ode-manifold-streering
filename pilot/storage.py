"""Portable, non-pickle artifacts, atomic writes, and experiment fingerprints."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile

import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    atomic(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def save_arrays(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez_compressed(f, **arrays)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def load_arrays(path):
    with np.load(path, allow_pickle=False) as f:
        return {key: f[key] for key in f.files}


STAGE_SOURCES = {
    "dataset-manifest.json": ["data.py"],
    "readout": ["data.py", "model.py", "pipeline.py"],
    "collection": ["data.py", "model.py"],
    "fits": ["probe.py", "geometry.py", "pipeline.py"],
    "patches": ["model.py", "pipeline.py"],
    "main": ["model.py", "steering.py", "geometry.py", "pipeline.py"],
    "paraphrase": ["model.py", "steering.py", "geometry.py", "pipeline.py"],
}


def manifest(path, identity):
    """Reject mixed settings instead of overwriting a previous experiment."""
    implementation = hashlib.sha256()
    path = Path(path)
    names = STAGE_SOURCES.get(path.name, STAGE_SOURCES.get(path.parent.name, []))
    for name in sorted(names):
        source = Path(__file__).parent / name
        implementation.update(source.name.encode())
        implementation.update(source.read_bytes())
    # Source hashes are provenance, not cache keys. Comment/report edits must not
    # invalidate model activations. Bump an explicit format version for a real
    # artifact-schema change, or choose a new output directory for algorithm changes.
    value = {"fingerprint": digest(identity), "identity": identity,
             "source_provenance": {"files": names, "sha256": implementation.hexdigest()}}
    if path.exists():
        if read_json(path)["fingerprint"] != value["fingerprint"]:
            raise ValueError(f"Artifact settings differ at {path}; use a new output directory")
    else:
        write_json(path, value)
    return value["fingerprint"]


def versions():
    out = {}
    for name in ("numpy", "scipy", "scikit-learn", "torch", "transformers"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = "not-installed"
    return out


def finite_mean(values):
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None
