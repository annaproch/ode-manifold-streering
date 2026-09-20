"""Explicit, resumable stages for the colour pilot. Nothing runs on import."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import numpy as np

from . import data
from .config import root
from .geometry import Geometry
from .model import Runtime
from .probe import Probe
from .steering import steer
from .storage import digest, load_arrays, manifest, read_json, save_arrays, write_json, finite_mean


def _ensure_identity(actual, expected):
    if actual != expected:
        raise ValueError("Model/tokenizer/precision identity differs from the cached run. Use the same pinned checkpoint and dtype, or a new output directory.")


def readout_check(cfg):
    rows = [r for r in data.read(cfg) if r["split"] in {"train", "validation"} and r["template"] == "main" and r["expected"]]
    rows = data.balanced_limit(rows, cfg["readout"]["maximum_prompts"], key="expected")
    if not rows:
        raise ValueError("No labeled readout validation stimuli")
    out = root(cfg) / "readout"
    manifest(out / "request.json", {"model": cfg["model"], "readout": cfg["readout"], "ids": [r["id"] for r in rows]})
    runtime = Runtime(cfg)
    report = {"model": runtime.identity, "metadata": runtime.metadata, "row_ids": [r["id"] for r in rows]}
    expected = np.array([data.LABELS.index(r["expected"]) for r in rows])
    chromatic = expected < 6
    controls = ~chromatic
    for extended in (False, True):
        p, _, top, seconds = runtime.batched(rows, extended=extended)
        per_class = {}
        for index in range(8):
            mask = expected == index
            per_class[data.LABELS[index]] = float(np.mean(p[mask].argmax(1) == index)) if mask.any() else None
        report["eight" if extended else "six"] = {
            "chromatic_accuracy": float(np.mean(p[chromatic].argmax(1) == expected[chromatic])) if chromatic.any() else None,
            "control_accuracy": float(np.mean(p[controls].argmax(1) == expected[controls])) if controls.any() else None,
            "mean_answer_mass": float(p[:, :8].sum(1).mean()) if extended else float(p[chromatic, :6].sum(1).mean()) if chromatic.any() else 0.0,
            "answer_mass_population": "all labeled stimuli" if extended else "chromatic stimuli only",
            "per_class_accuracy": per_class,
            "seconds": seconds,
        }
        save_arrays(out / ("eight.npz" if extended else "six.npz"), probabilities=p, top_tokens=top,
                    expected=expected, ids=np.array([r["id"] for r in rows]))
    chosen = report["eight" if cfg["readout"]["extended"] else "six"]
    r = cfg["readout"]
    passed = chosen["chromatic_accuracy"] is not None and chosen["chromatic_accuracy"] >= r["minimum_accuracy"] and chosen["mean_answer_mass"] >= r["minimum_answer_mass"]
    if r["extended"]:
        passed = passed and report["six"]["chromatic_accuracy"] is not None and report["eight"]["chromatic_accuracy"] >= r["extended_accuracy_ratio"]*report["six"]["chromatic_accuracy"]
        passed = passed and all(chosen["per_class_accuracy"][name] is not None and chosen["per_class_accuracy"][name] >= r["minimum_control_accuracy"] for name in ("gray", "brown"))
    report["passed"] = bool(passed)
    report["settings"] = cfg["readout"]
    write_json(out / "report.json", report)
    return report


def collect(cfg):
    rows = data.read(cfg)
    if not data.coverage(rows, cfg)["adequate"]:
        raise ValueError("Insufficient primary evaluation coverage; inspect coverage.json before collection")
    check_path = root(cfg) / "readout" / "report.json"
    if not check_path.exists() or not read_json(check_path)["passed"]:
        raise ValueError("Run and pass readout-check before activation collection")
    if read_json(check_path)["settings"] != cfg["readout"]:
        raise ValueError("Readout settings changed since validation")
    out = root(cfg) / "collection"
    identity = {"model": cfg["model"], "readout_extended": cfg["readout"]["extended"],
                "layers": cfg["fit"]["layers"], "rows": digest(rows), "format": 2}
    request = manifest(out / "request.json", identity)
    completed = out / "complete.json"
    if completed.exists():
        # Loading catches a missing shard rather than trusting the marker alone.
        load_collection(cfg)
        return read_json(completed)
    pinned = read_json(out / "model.json")["revision"] if (out / "model.json").exists() else None
    runtime = Runtime(cfg, pinned)
    if (out / "model.json").exists():
        _ensure_identity(runtime.identity, read_json(out / "model.json"))
    else:
        write_json(out / "model.json", runtime.identity)
        write_json(out / "runtime.json", runtime.metadata)
    if check_path.exists():
        _ensure_identity(runtime.identity, read_json(check_path)["model"])
    size = cfg["model"]["batch_size"]
    shards = []
    for start in range(0, len(rows), size):
        batch = rows[start:start+size]
        path = out / "shards" / f"{start:07d}.npz"
        shards.append(path.name)
        ids = np.array([r["id"] for r in batch])
        if path.exists():
            cached = load_arrays(path)
            if not np.array_equal(ids, cached["ids"]):
                raise ValueError(f"Collection shard IDs differ: {path}")
            continue
        control = np.array([r["kind"] == "unrelated" for r in batch])
        p, acts, top, seconds = runtime.forward(batch, capture=cfg["fit"]["layers"], capture_full=bool(control.any()))
        extra = {"full_control_ids": ids[control], "full_control_probabilities": runtime.last_full_probabilities[control]} if control.any() else {}
        save_arrays(path, ids=ids, probabilities=p, top_tokens=top,
                    seconds=np.array(seconds), **{f"layer_{l}": a for l, a in acts.items()}, **extra)
        print(f"collect {start+len(batch)}/{len(rows)}", flush=True)
    result = {"request": request, "shards": shards, "rows": len(rows), "model": runtime.identity}
    write_json(completed, result)
    return result


def load_collection(cfg):
    rows = data.read(cfg)
    out = root(cfg) / "collection"
    if not (out / "complete.json").exists():
        raise ValueError("Activation collection is incomplete; finish collect first")
    complete = read_json(out / "complete.json")
    request = read_json(out / "request.json")
    if request["identity"]["rows"] != digest(rows) or request["identity"]["readout_extended"] != cfg["readout"]["extended"]:
        raise ValueError("Dataset/readout differs from cached collection")
    model = complete["model"]
    if model["name"] != cfg["model"]["name"] or model["chat_template"] != cfg["model"]["chat_template"]:
        raise ValueError("Model or prompt convention differs from collection")
    pieces = [load_arrays(out / "shards" / name) for name in complete["shards"]]
    ids = np.concatenate([p["ids"] for p in pieces])
    if ids.tolist() != [r["id"] for r in rows]:
        raise ValueError("Cached row order or completeness mismatch")
    arrays = {"probabilities": np.concatenate([p["probabilities"] for p in pieces])}
    arrays["full_controls"] = {str(row_id): prob for p in pieces if "full_control_ids" in p
                               for row_id, prob in zip(p["full_control_ids"], p["full_control_probabilities"])}
    for layer in cfg["fit"]["layers"]:
        key = f"layer_{layer}"
        if key not in pieces[0]:
            raise ValueError(f"Layer {layer} is not in the collection")
        arrays[key] = np.concatenate([p[key] for p in pieces])
    return rows, arrays, complete


def fit(cfg):
    rows, cached, collection = load_collection(cfg)
    check = root(cfg) / "readout" / "report.json"
    if not check.exists() or not read_json(check)["passed"]:
        raise ValueError("Run and pass readout-check before fitting. If the task fails, revise the design in a new output directory.")
    _ensure_identity(collection["model"], read_json(check)["model"])
    if read_json(check)["settings"] != cfg["readout"]:
        raise ValueError("Readout criteria changed since validation")
    target = data.LABELS.index(cfg["fit"]["target"])
    train = np.array([r["split"] == "train" and r["primary"] and r["template"] == "main" for r in rows])
    val = np.array([r["split"] == "validation" and r["primary"] and r["template"] == "main" for r in rows])
    y_all = cached["probabilities"].argmax(1)
    enable_graph = "E3" in cfg["steering"]["methods"]
    out = root(cfg) / "fits"
    request = manifest(out / "request.json", {"collection": collection["request"], "fit": cfg["fit"], "graph": enable_graph})
    entries, skipped = [], []
    for layer in cfg["fit"]["layers"]:
        x = cached[f"layer_{layer}"]
        for seed in cfg["fit"]["seeds"]:
            probe = Probe(x.shape[1], cfg["fit"]["sketch_features"], seed, cfg["fit"]["gamma"], cfg["fit"]["coef0"])
            probe.fit(x[train], y_all[train] == target, cfg["fit"]["logistic_c"])
            diagnostics = {"train": probe.diagnostics(x[train], y_all[train] == target),
                           "validation": probe.diagnostics(x[val], y_all[val] == target)}
            graph_targets = None
            if enable_graph:
                mode = cfg["fit"]["graph_targets"]
                score = probe.log_dre(x[train])
                if mode == "labels":
                    graph_targets = y_all[train] == target
                elif mode == "density_ratio":
                    graph_targets = score > 0
                else:
                    graph_targets = np.zeros(len(score), dtype=bool)
                    graph_targets[np.argsort(score)[-max(1, round(len(score)*cfg["fit"]["graph_top_fraction"])):]] = True
            for dimension in cfg["fit"]["dimensions"]:
                name = f"layer{layer}-k{dimension}-seed{seed}"
                folder = out / name
                if (folder / "metadata.json").exists():
                    saved = read_json(folder / "metadata.json")
                    if saved["request"] != request:
                        raise ValueError("Fit fingerprint mismatch")
                    load_arrays(folder / "arrays.npz")
                    entries.append(name)
                    continue
                try:
                    g = Geometry.fit(x[train], y_all[train], [r["hue"] for r, keep in zip(rows, train) if keep],
                                     x[val], dimension, cfg["fit"], seed, graph_targets)
                except ValueError as exc:
                    skipped.append({"fit": name, "reason": str(exc)})
                    continue
                control_mask = np.array([r["split"] == "validation" and r["kind"] != "colour" and r["template"] == "main" for r in rows])
                score = probe.log_dre(x[train])
                order = np.argsort(g.hues)
                meta = {"request": request, "name": name, "layer": layer, "dimension": dimension, "seed": seed,
                        "target": target, "geometry": g.metadata, "probe": diagnostics,
                        "train_ids": [r["id"] for r, keep in zip(rows, train) if keep],
                        "validation_gate_coverage": float(g.eligible(x[val]).mean()),
                        "control_gate_acceptance": float(g.eligible(x[control_mask]).mean()) if control_mask.any() else None,
                        "spline_available": g.spline is not None,
                        "restricted_score": {"hues": g.hues[order].tolist(), "log_dre": score[order].tolist()},
                        "model": collection["model"]}
                target_center = g.x[g.labels == target].mean(0)
                distances = [float(np.linalg.norm(g.x[g.labels == label].mean(0)-target_center))
                             for label in range(6) if label != target and np.any(g.labels == label)]
                meta["strength_scales"] = {"raw": 1.0, "activation_norm": float(np.median(np.linalg.norm(g.x, axis=1))),
                                           "centroid_distance": float(np.median(distances)) if distances else 0.0}
                save_arrays(folder / "arrays.npz", **g.arrays(), **probe.arrays())
                write_json(folder / "metadata.json", meta)
                entries.append(name)
                print(f"fit {name}", flush=True)
    if not entries:
        write_json(out / "skipped.json", skipped)
        raise ValueError("No valid fit; inspect fits/skipped.json and data coverage")
    result = {"request": request, "entries": entries, "skipped": skipped}
    write_json(out / "index.json", result)
    return result


def load_fit(cfg, name):
    folder = root(cfg) / "fits" / name
    request = read_json(root(cfg) / "fits" / "request.json")
    if request["identity"]["fit"] != cfg["fit"]:
        raise ValueError("Fit settings differ from saved artifacts; use a new run directory")
    arrays = load_arrays(folder / "arrays.npz")
    meta = read_json(folder / "metadata.json")
    if meta["request"] != request["fingerprint"]:
        raise ValueError("Fit was produced under a different request")
    return Probe.restore(arrays), Geometry(arrays, meta["geometry"]), meta


def patch(cfg):
    rows, cached, collection = load_collection(cfg)
    index = read_json(root(cfg) / "fits" / "index.json")
    selected_rows = [r for r in rows if r["split"] == "validation" and r["primary"] and r["template"] == "main"]
    selected_rows = data.balanced_limit(selected_rows, cfg["patch"]["maximum_prompts"])
    row_index = {r["id"]: i for i, r in enumerate(rows)}
    target = data.LABELS.index(cfg["fit"]["target"])
    selected_rows = [r for r in selected_rows if cached["probabilities"][row_index[r["id"]]].argmax() != target]
    if not selected_rows:
        raise ValueError("No non-target primary validation prompts for patching")
    out = root(cfg) / "patches"
    request = manifest(out / "request.json", {"fits": index["request"], "settings": cfg["patch"],
                       "noop_tolerance": cfg["model"]["noop_tolerance"], "ids": [r["id"] for r in selected_rows]})
    runtime = Runtime(cfg, collection["model"]["revision"])
    _ensure_identity(runtime.identity, collection["model"])
    groups = defaultdict(list)
    for name in index["entries"]:
        meta = read_json(root(cfg) / "fits" / name / "metadata.json")
        groups[(meta["layer"], meta["dimension"])].append(name)
    results = []
    ids = [row_index[r["id"]] for r in selected_rows]
    for (layer, dimension), names in groups.items():
        path = out / f"layer{layer}-k{dimension}.json"
        if path.exists():
            saved = read_json(path)
            if saved["request"] != request or not path.with_suffix(".npz").exists():
                raise ValueError(f"Incomplete or mismatched patch artifact: {path}")
            results.append(saved)
            continue
        _, g, meta = load_fit(cfg, names[0])
        sources = cached[f"layer_{layer}"][ids].astype(np.float64)
        donors = g.x[g.labels == target]
        if not len(donors):
            raise ValueError("No target donors in primary training data")
        # Choose real target donors closest in the preserved residual, using training data only.
        source_residual = sources-g.coordinates(sources) @ g.u.T
        donor_residual = donors-g.coordinates(donors) @ g.u.T
        nearest = np.argmin(np.sum((source_residual[:, None]-donor_residual[None])**2, axis=2), axis=1)
        full = donors[nearest]
        subspace = sources+((full-sources) @ g.u) @ g.u.T
        outputs = {}
        raw = {}
        for condition, replacements in (("identity", sources), ("full", full), ("subspace", subspace)):
            p, _, _, elapsed = runtime.batched(selected_rows, patch_layer=layer, replacements=replacements)
            raw[condition] = p
            outputs[condition] = {"success": float(np.mean(p.argmax(1) == target)),
                                  "target_probability": float(p[:, target].mean()), "other": float(p[:, -1].mean()), "seconds": elapsed}
        mismatch = float(np.max(np.abs(raw["identity"]-cached["probabilities"][ids])))
        passed = outputs["subspace"]["success"] >= cfg["patch"]["minimum_success"] and outputs["full"]["success"] >= cfg["patch"]["minimum_success"] and mismatch <= cfg["model"]["noop_tolerance"]
        result = {"layer": layer, "dimension": dimension, "fits": names, "count": len(ids),
                  "conditions": outputs, "identity_max_error": mismatch, "passed": bool(passed), "request": request}
        save_arrays(path.with_suffix(".npz"), ids=np.array([r["id"] for r in selected_rows]), **raw)
        write_json(path, result)
        results.append(result)
    passing = [r for r in results if r["passed"] and len(r["fits"]) == len(cfg["fit"]["seeds"])]
    # Predeclared selection: smallest adequate subspace, then strongest patch effect.
    choice = min(passing, key=lambda r: (r["dimension"], -r["conditions"]["subspace"]["success"], r["layer"])) if passing else None
    selection = {"request": request, "passed": choice is not None, "choice": choice, "candidates": results}
    write_json(root(cfg) / "selection.json", selection)
    return selection


def nearest_bc(p, reference):
    overlap = np.sqrt(np.clip(p, 0, 1)) @ np.sqrt(np.clip(reference, 0, 1)).T
    return -np.log(np.clip(overlap.max(axis=1), 1e-12, 1))


def estimate_cost(cfg, split="validation", template="main", paths=False):
    rows = data.read(cfg)
    selected = data.evaluation_rows(rows, cfg, split, template)
    frozen = split == "test" or paths
    settings = sum(1 if frozen or m in {"none", "R0"} else len(cfg["steering"]["strengths"])
                   for m in cfg["steering"]["methods"])
    checkpoints = cfg["steering"]["checkpoints"] if frozen else 2
    groups = len(selected)*len(cfg["fit"]["seeds"])*settings
    return {"split": split, "paths": frozen, "prompts": len(selected), "settings_per_seed": settings,
            "checkpoints": checkpoints, "decode_examples_upper_bound": groups*checkpoints,
            "batched_forwards_upper_bound": groups*int(np.ceil(checkpoints/cfg["model"]["batch_size"])),
            "budget": cfg["evaluation"]["maximum_decode_examples"], "coverage": data.coverage(rows, cfg),
            "note": "Conservative count before identical-state caching; not a wall-time estimate."}


def calibrate_noop(cfg):
    rows, cached, collection = load_collection(cfg)
    indices = {r["id"]: i for i, r in enumerate(rows)}
    sample = data.balanced_limit([r for r in rows if r["split"] == "validation" and r["template"] == "main"], 24, key="kind")
    if not sample:
        raise ValueError("No validation prompts for no-op calibration")
    runtime = Runtime(cfg, collection["model"]["revision"])
    _ensure_identity(runtime.identity, collection["model"])
    errors = []
    # Test singletons and full repeated-prompt batches against mixed collection batches.
    for layer in cfg["fit"]["layers"]:
        for row in sample:
            idx = indices[row["id"]]
            for size in sorted({1, cfg["model"]["batch_size"]}):
                states = np.repeat(cached[f"layer_{layer}"][idx:idx+1], size, axis=0)
                p, _, _, _ = runtime.batched([row]*size, patch_layer=layer, replacements=states)
                errors.append({"id": row["id"], "layer": layer, "batch_size": size,
                               "error": float(np.max(np.abs(p-cached["probabilities"][idx])))})
    maximum = max(r["error"] for r in errors)
    result = {"model": runtime.identity, "device": runtime.metadata["device"],
              "batch_size": cfg["model"]["batch_size"], "tolerance": cfg["model"]["noop_tolerance"],
              "max_error": maximum, "passed": maximum <= cfg["model"]["noop_tolerance"], "rows": errors,
              "note": "Inspect measured error before proceeding. Tolerance is never automatically widened."}
    write_json(root(cfg) / "noop-calibration.json", result)
    return result


def decode_states(cfg, runtime, collection, row, layer, states):
    states = np.asarray(states, dtype=np.float32)
    identity = digest({"model": collection["model"], "device": runtime.metadata["device"],
                       "batch_size": cfg["model"]["batch_size"], "row": row,
                       "layer": layer, "extended": cfg["readout"]["extended"], "format": 1})[:24]
    folder = root(cfg) / "decoded-cache" / identity
    keys = [hashlib.sha256(state.tobytes()).hexdigest() for state in states]
    pending = {key: state for key, state in zip(keys, states) if not (folder / f"{key}.npz").exists()}
    seconds = 0.0
    full = row["kind"] == "unrelated"
    if pending:
        p, _, _, seconds = runtime.batched([row]*len(pending), patch_layer=layer,
                                           replacements=np.stack(list(pending.values())), capture_full=full)
        for index, key in enumerate(pending):
            extra = {"full_probabilities": runtime.last_full_probabilities[index]} if full else {}
            save_arrays(folder / f"{key}.npz", probabilities=p[index], **extra)
    saved = [load_arrays(folder / f"{key}.npz") for key in keys]
    return (np.stack([s["probabilities"] for s in saved]),
            np.stack([s["full_probabilities"] for s in saved]) if full else None, seconds, len(pending))


def evaluation(cfg, split="validation", template="main", paths=False):
    if split not in {"validation", "test"}:
        raise ValueError("Only validation and test evaluation are supported")
    rows, cached, collection = load_collection(cfg)
    coverage = data.coverage(rows, cfg)
    if not coverage["adequate"]:
        raise ValueError(f"Insufficient deduplicated evaluation coverage: {coverage['splits']}. Expand the primary data before expensive evaluation.")
    calibration = read_json(root(cfg) / "noop-calibration.json")
    if not calibration["passed"] or calibration["tolerance"] != cfg["model"]["noop_tolerance"]:
        raise ValueError("Run and pass calibrate-noop with the intended device/precision/tolerance first")
    selection = read_json(root(cfg) / "selection.json")
    patch_request = read_json(root(cfg) / "patches" / "request.json")
    if (patch_request["identity"]["settings"] != cfg["patch"]
            or patch_request["identity"]["noop_tolerance"] != cfg["model"]["noop_tolerance"]
            or selection["request"] != patch_request["fingerprint"]):
        raise ValueError("Causal-gate configuration changed; rerun selection in a new experiment")
    if not selection["passed"]:
        raise ValueError("Causal donor-patching gate has not passed")
    chosen = selection["choice"]
    if "F" in cfg["steering"]["methods"]:
        if not all(load_fit(cfg, name)[1].spline is not None for name in chosen["fits"]):
            raise ValueError("Spline reference unavailable: insufficient six-class training coverage")
    selected_rows = data.evaluation_rows(rows, cfg, split, template)
    if not selected_rows:
        raise ValueError("No evaluation prompts; increase data coverage or source bands")
    row_index = {r["id"]: i for i, r in enumerate(rows)}
    target = data.LABELS.index(cfg["fit"]["target"])
    # Natural reference examples are independent of fit, gate calibration and test.
    reference_indices = [i for i, r in enumerate(rows) if r["split"] == "reference" and r["primary"] and r["template"] == "main"]
    ref_p = cached["probabilities"][reference_indices]
    target_ref = ref_p[ref_p.argmax(1) == target]
    if len(target_ref) < cfg["evaluation"]["minimum_reference_points"]:
        raise ValueError("Too few natural target-reference examples; expand the data before evaluating")
    frozen = None
    if split == "test" or paths:
        frozen = read_json(root(cfg) / "operating-points.json")
        expected = frozen_identity(cfg, selection)
        if frozen["identity"] != expected:
            raise ValueError("Configuration changed after freezing operating points; do not retune on test")
    stage = "validation-paths" if paths and split == "validation" else split
    out = root(cfg) / "evaluations" / stage / template
    request = manifest(out / "request.json", {"selection": digest(selection), "steering": cfg["steering"],
                       "evaluation": cfg["evaluation"], "ids": [r["id"] for r in selected_rows],
                       "noop_calibration": digest(calibration),
                       "frozen": digest(frozen) if frozen else None, "template": template, "paths": split == "test" or paths})
    estimate = estimate_cost(cfg, split, template, paths)
    if estimate["decode_examples_upper_bound"] > cfg["evaluation"]["maximum_decode_examples"]:
        raise ValueError(f"Estimated {estimate['decode_examples_upper_bound']} decoded prompt states exceed maximum_decode_examples={cfg['evaluation']['maximum_decode_examples']}. Inspect estimate and explicitly budget the run.")
    runtime = Runtime(cfg, collection["model"]["revision"])
    _ensure_identity(runtime.identity, collection["model"])
    if runtime.metadata["device"] != calibration["device"]:
        raise ValueError("No-op calibration used a different device; recalibrate before evaluation")
    _ensure_identity(runtime.identity, calibration["model"])
    if cfg["model"]["batch_size"] != calibration["batch_size"]:
        raise ValueError("Batch size changed after no-op calibration; recalibrate before evaluation")
    result_paths = []
    for name in chosen["fits"]:
        probe, g, meta = load_fit(cfg, name)
        layer = meta["layer"]
        ref_x = cached[f"layer_{layer}"][reference_indices]
        for method in cfg["steering"]["methods"]:
            if split == "test" or paths:
                strengths = [frozen["methods"][method]["strength"]]
            else:
                strengths = [0.0] if method in {"none", "R0"} else cfg["steering"]["strengths"]
            for strength in strengths:
                for row in selected_rows:
                    key = digest([name, method, strength, row["id"]])[:24]
                    path = out / "rows" / f"{key}.json"
                    result_paths.append(path.name)
                    if path.exists():
                        old = read_json(path)
                        if old["request"] != request or not path.with_suffix(".npz").exists():
                            raise ValueError(f"Incomplete or mismatched evaluation result: {path}")
                        continue
                    idx = row_index[row["id"]]
                    original = cached[f"layer_{layer}"][idx]
                    scale = meta["strength_scales"][cfg["steering"]["strength_scale"]]
                    if scale <= 0:
                        raise ValueError("Strength calibration scale is zero; inspect target and source class centres")
                    raw_strength = strength*scale
                    trajectory = steer(original, method, raw_strength, probe, g, cfg["steering"], target)
                    decoded_states = trajectory.states if split == "test" or paths else trajectory.states[[0, -1]]
                    p, full_p, seconds, decoded_count = decode_states(cfg, runtime, collection, row, layer, decoded_states)
                    identity_error = float(np.max(np.abs(p[0]-cached["probabilities"][idx])))
                    if identity_error > cfg["model"]["noop_tolerance"]:
                        raise ValueError(f"No-op patch mismatch {identity_error:.4g}; check device/precision/prompt identity")
                    path_deviation = nearest_bc(p, ref_p)
                    end_deviation = float(nearest_bc(p[-1:], target_ref)[0])
                    projected, full = g.distances(trajectory.states)
                    independent_full = np.linalg.norm(trajectory.states[:, None]-ref_x[None], axis=-1).min(axis=1)
                    independent_z = np.linalg.norm(g.coordinates(trajectory.states)[:, None]-g.coordinates(ref_x)[None], axis=-1).min(axis=1)
                    relative_hue = float(data.circular_distance(row["hue"], data.ANGLES[target]))
                    band = min(cfg["evaluation"]["source_bands"], key=lambda b: abs(b-relative_hue))
                    concentration = np.abs(p[:, :6] @ np.exp(1j*np.deg2rad(data.ANGLES)))
                    unrelated_kl = None
                    if row["kind"] == "unrelated":
                        baseline_full = cached["full_controls"].get(row["id"])
                        if baseline_full is None or full_p is None:
                            raise ValueError("Missing full-vocabulary control reference; recollect with format 2")
                        q = np.clip(full_p[-1], 1e-30, 1)
                        base = np.clip(baseline_full, 1e-30, 1)
                        unrelated_kl = float(np.maximum(0, np.sum(base*(np.log(base)-np.log(q)))))
                    record = {"request": request, "id": row["id"], "hex": row["hex"], "kind": row["kind"],
                              "template": template, "primary": row["primary"], "split": split,
                              "hue": row["hue"], "band": band, "antipodal": band == 180,
                              "fit": name, "seed": meta["seed"], "layer": layer, "dimension": meta["dimension"],
                              "method": method, "strength": float(strength), "raw_strength": float(raw_strength),
                              "strength_scale": cfg["steering"]["strength_scale"], "scale_value": float(scale),
                              "eligible": trajectory.eligible, "entry_gate_applied": method not in {"none", "A", "B", "S", "R0"},
                              "reason": trajectory.reason, "baseline_target": bool(cached["probabilities"][idx].argmax() == target),
                              "success": bool(p[-1].argmax() == target), "target_probability": float(p[-1, target]),
                              "endpoint_bc": end_deviation, "path_bc": float(path_deviation.mean()) if split == "test" or paths else None,
                              "path_bc_sum": float(path_deviation.sum()) if split == "test" or paths else None,
                              "peak_path_bc": float(path_deviation.max()) if split == "test" or paths else None,
                              "unrelated_endpoint_kl": unrelated_kl, "decoded_examples": decoded_count,
                              "full_path_decoded": bool(split == "test" or paths),
                              "end_gray": float(p[-1, 6]), "end_brown": float(p[-1, 7]), "end_other": float(p[-1, 8]),
                              "peak_gray": float(p[:, 6].max()) if split == "test" or paths else None,
                              "peak_brown": float(p[:, 7].max()) if split == "test" or paths else None,
                              "peak_other": float(p[:, 8].max()) if split == "test" or paths else None,
                              "minimum_concentration": float(concentration.min()) if split == "test" or paths else None,
                              "relative_minimum_concentration": float(concentration.min()/max(concentration[0], 1e-8)) if split == "test" or paths else None,
                              "path_length": trajectory.path_length, "correction_length": trajectory.correction_length,
                              "displacement": trajectory.displacement, "norm_change": trajectory.norm_change,
                              "max_reference_full_distance": float(independent_full.max()/g.metadata["typical_full_distance"]),
                              "max_reference_projected_distance": float(independent_z.max()/g.metadata["typical_projected_distance"]),
                              "max_support_full_distance": float(full.max()), "max_support_projected_distance": float(projected.max()),
                              "geometry_seconds": trajectory.seconds, "model_seconds": seconds, "identity_error": identity_error}
                    save_arrays(path.with_suffix(".npz"), states=trajectory.states.astype(np.float32),
                                solver_states=trajectory.solver_states.astype(np.float32),
                                decoded_states=decoded_states.astype(np.float32), probabilities=p,
                                path_bc=path_deviation, concentration=concentration)
                    write_json(path, record)
                print(f"{split} {name} {method} T={strength}", flush=True)
    result = {"request": request, "files": result_paths, "count": len(result_paths), "runtime": runtime.metadata}
    write_json(out / "complete.json", result)
    return result


def frozen_identity(cfg, selection):
    return {"selection": digest(selection), "fit": cfg["fit"], "steering": cfg["steering"],
            "evaluation": cfg["evaluation"], "readout": cfg["readout"], "noop_tolerance": cfg["model"]["noop_tolerance"]}


def load_evaluations(cfg, split, template="main"):
    folder = root(cfg) / "evaluations" / split / template
    identity = read_json(folder / "request.json")["identity"]
    if identity["steering"] != cfg["steering"] or identity["evaluation"] != cfg["evaluation"]:
        raise ValueError("Evaluation settings differ from saved results")
    complete = read_json(folder / "complete.json")
    records = [read_json(folder / "rows" / name) for name in complete["files"]]
    if any(r["request"] != complete["request"] for r in records):
        raise ValueError("Mixed evaluation fingerprints")
    return records


def select_operating_points(cfg):
    rows = load_evaluations(cfg, "validation")
    selection = read_json(root(cfg) / "selection.json")
    identity = read_json(root(cfg) / "evaluations" / "validation" / "main" / "request.json")["identity"]
    if identity["selection"] != digest(selection):
        raise ValueError("Validation results use a different causal selection")
    methods = {}
    for method in cfg["steering"]["methods"]:
        group = [r for r in rows if r["method"] == method and r["primary"] and not r["antipodal"] and r["eligible"] and not r["baseline_target"]]
        if not group:
            raise ValueError(f"No eligible non-antipodal validation starts for {method}")
        for seed in cfg["fit"]["seeds"]:
            for band in cfg["evaluation"]["source_bands"]:
                if band == 180:
                    continue
                count = len({r["id"] for r in group if r["seed"] == seed and r["band"] == band})
                if count < cfg["evaluation"]["minimum_band_prompts"]:
                    raise ValueError(f"Too few eligible validation starts: {method}, seed {seed}, band {band}: {count}")
        curve = []
        for strength in sorted({r["strength"] for r in group}):
            cell = [r for r in group if r["strength"] == strength]
            per_seed = [np.mean([r["success"] for r in cell if r["seed"] == seed]) for seed in sorted({r["seed"] for r in cell})]
            curve.append({"strength": strength, "success": float(np.mean(per_seed)), "count": len(cell),
                          "endpoint_bc": finite_mean(r["endpoint_bc"] for r in cell), "path_bc": finite_mean(r["path_bc"] for r in cell)})
        passing = [c for c in curve if c["success"] >= cfg["evaluation"]["success_level"]]
        choice = passing[0] if passing else min(curve, key=lambda c: (-c["success"], c["strength"]))
        methods[method] = {**choice, "matched_on_validation": bool(passing), "curve": curve,
                           "status": "diagnostic" if method in {"none", "R0", "S"} else "matched" if passing else "threshold_not_reached"}
    identity = frozen_identity(cfg, selection)
    destination = root(cfg) / "operating-points.json"
    frozen = {"identity": identity, "methods": methods,
              "hypotheses": {"endpoint": cfg["evaluation"]["endpoint_practical_improvement"],
                             "path": cfg["evaluation"]["path_practical_improvement"]},
              "interpretation": "exploratory" if any(cfg["evaluation"][k] is None for k in ("endpoint_practical_improvement", "path_practical_improvement")) else "thresholds_frozen",
              "selection_rule": "smallest validation T reaching mean per-seed eligible non-antipodal success; otherwise best success then smallest T"}
    if destination.exists() and read_json(destination) != frozen:
        raise ValueError("Operating points already frozen differently; use a new output directory")
    write_json(destination, frozen)
    return frozen
