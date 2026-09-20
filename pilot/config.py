"""Explicit experiment settings; devices affect inference, not geometry fitting."""
from __future__ import annotations

import copy
import json
from pathlib import Path

DEFAULTS = {
    "output": "pilot-runs/colour",
    "model": {"name": "meta-llama/Llama-3.1-8B", "revision": "main",
              "device": "cpu", "dtype": "float32", "batch_size": 2,
              "local_files_only": True, "cache_dir": None,
              "chat_template": False, "max_length": 256, "threads": 8,
              "noop_tolerance": 0.02},
    "data": {"hue_step": 0.25, "context_hue_step": 6.0, "saturations": [0.65, 0.8, 0.95],
             "lightnesses": [0.35, 0.5, 0.65], "primary_saturation": 0.8,
             "primary_lightness": 0.5, "templates": ["main"], "seed": 2718,
             "split_fractions": {"train": 0.45, "reference": 0.1,
                                 "validation": 0.225, "test": 0.225}},
    "readout": {"extended": True, "minimum_accuracy": 0.7,
                "minimum_answer_mass": 0.5, "extended_accuracy_ratio": 0.9,
                "minimum_control_accuracy": 0.7, "maximum_prompts": 100},
    "fit": {"layers": [18, 24, 27], "dimensions": [8, 16, 64],
            "seeds": [0, 1, 2], "sketch_features": 8000, "gamma": 0.1,
            "coef0": 1.0, "logistic_c": 1.0, "neighbors": 20,
            "tangent_dim": 1, "gate_quantile": 0.99,
            "bandwidth_factors": [0.5, 1.0, 2.0],
            "target": "red", "graph_neighbors": 8,
            "graph_targets": "labels", "graph_top_fraction": 0.167},
    "patch": {"minimum_success": 0.8, "maximum_prompts": 100},
    "steering": {"methods": ["none", "A", "R0", "B", "C", "C-end", "E2", "F", "S"],
                 "strengths": [0.0, 0.25, 0.5, 1.0, 2.0, 4.0],
                 "strength_scale": "centroid_distance",
                 "steps": 10, "retractions": 3, "correction_ratio": 0.25,
                 "checkpoints": 11, "tiny_gradient": 1e-10,
                 "support_multiplier": 3.0, "additive_weight": 1.0},
    "evaluation": {"source_bands": [90, 120, 150, 180], "band_halfwidth": 10,
                   "maximum_prompts": 0, "success_level": 0.8,
                   "success_tolerance": 0.05,
                   "bootstrap_samples": 2000, "bootstrap_seed": 108,
                   "minimum_reference_points": 8,
                   "minimum_band_prompts": 20, "minimum_antipodal_prompts": 10,
                   "maximum_decode_examples": 100000,
                   "endpoint_practical_improvement": None,
                   "path_practical_improvement": None},
}


def merge(base, override):
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key not in out:
            raise ValueError(f"Unknown configuration key: {key}")
        out[key] = merge(out[key], value) if isinstance(value, dict) else value
    return out


def load(path):
    cfg = merge(DEFAULTS, json.loads(Path(path).read_text()))
    validate(cfg)
    return cfg


def validate(c):
    if c["model"]["device"] not in {"cpu", "cuda", "mps", "auto"} and not c["model"]["device"].startswith("cuda:"):
        raise ValueError("device must be cpu, cuda[:index], mps, or auto")
    if c["model"]["dtype"] not in {"float32", "float16", "bfloat16", "auto"}:
        raise ValueError("Unsupported model dtype")
    if c["model"]["batch_size"] < 1 or c["model"]["max_length"] < 1 or c["model"]["threads"] < 1:
        raise ValueError("batch_size and max_length must be positive")
    if not 0 < c["model"]["noop_tolerance"] < 1:
        raise ValueError("noop_tolerance must be in (0,1)")
    if not 0 < c["data"]["hue_step"] <= 30:
        raise ValueError("hue_step must be in (0, 30]")
    if not 0 < c["data"]["context_hue_step"] <= 30:
        raise ValueError("context_hue_step must be in (0,30]")
    fractions = c["data"]["split_fractions"]
    if set(fractions) != {"train", "reference", "validation", "test"} or abs(sum(fractions.values()) - 1) > 1e-8 or min(fractions.values()) <= 0:
        raise ValueError("Four positive split fractions must sum to one")
    if c["data"]["primary_saturation"] not in c["data"]["saturations"] or c["data"]["primary_lightness"] not in c["data"]["lightnesses"]:
        raise ValueError("Primary slice must be present in the generated grid")
    if not set(c["data"]["templates"]) <= {"main", "paraphrase"} or "main" not in c["data"]["templates"]:
        raise ValueError("Templates must include main, optionally paraphrase")
    for axis in ("saturations", "lightnesses"):
        if any(not 0 < v < 1 for v in c["data"][axis]):
            raise ValueError(f"{axis} must lie strictly between 0 and 1")
    if not c["fit"]["layers"] or min(c["fit"]["layers"]) < 0 or not c["fit"]["seeds"]:
        raise ValueError("Provide nonnegative zero-based layers and at least one seed")
    if c["fit"]["target"] not in {"red", "orange", "yellow", "green", "blue", "purple"}:
        raise ValueError("This pilot supports chromatic target classes")
    if len(set(c["fit"]["seeds"])) != len(c["fit"]["seeds"]):
        raise ValueError("Fitted seeds must be unique")
    if c["fit"]["sketch_features"] < 2 or c["fit"]["logistic_c"] <= 0 or min(c["fit"]["bandwidth_factors"]) <= 0:
        raise ValueError("Invalid sketch, regularization or bandwidth configuration")
    if not 0 < c["fit"]["gate_quantile"] < 1 or not 0 < c["fit"]["graph_top_fraction"] < 1:
        raise ValueError("Invalid gate quantile or graph target fraction")
    if c["fit"]["graph_targets"] not in {"labels", "density_ratio", "top"}:
        raise ValueError("Unknown graph target definition")
    if c["fit"]["neighbors"] <= c["fit"]["tangent_dim"] or c["fit"]["tangent_dim"] < 1:
        raise ValueError("neighbors must exceed tangent_dim >= 1")
    if not c["fit"]["dimensions"] or min(c["fit"]["dimensions"]) <= c["fit"]["tangent_dim"]:
        raise ValueError("PCA dimension must exceed tangent dimension")
    s = c["steering"]
    if s["strength_scale"] not in {"raw", "activation_norm", "centroid_distance"}:
        raise ValueError("Unknown strength calibration scale")
    allowed = {"none", "A", "R0", "B", "C", "C-end", "E2", "F", "S", "E3", "E2-ms", "E2-h", "E1", "D", "E2-full"}
    if not s["methods"] or not set(s["methods"]) <= allowed or len(s["methods"]) != len(set(s["methods"])):
        raise ValueError("Unknown or duplicate steering method")
    if not s["strengths"] or min(s["strengths"]) < 0 or not all(float(t) < float("inf") for t in s["strengths"]):
        raise ValueError("Strengths must be finite and nonnegative")
    if len(s["strengths"]) != len(set(s["strengths"])):
        raise ValueError("Strengths must be unique")
    if s["steps"] < 1 or s["retractions"] < 1 or s["checkpoints"] < 2 or s["correction_ratio"] < 0:
        raise ValueError("Invalid integration/correction settings")
    if s["support_multiplier"] < 1 or s["tiny_gradient"] <= 0 or s["additive_weight"] < 0:
        raise ValueError("Invalid support, gradient or additive setting")
    if c["evaluation"]["bootstrap_samples"] < 100 or not 0 < c["evaluation"]["success_level"] <= 1:
        raise ValueError("Use at least 100 bootstrap samples and a success level in (0,1]")
    for key in ("minimum_band_prompts", "minimum_antipodal_prompts", "minimum_reference_points", "maximum_decode_examples"):
        value = c["evaluation"][key]
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if c["evaluation"]["maximum_prompts"] < 0:
        raise ValueError("maximum_prompts must be nonnegative")
    if not c["evaluation"]["source_bands"] or any(not 0 <= b <= 180 for b in c["evaluation"]["source_bands"]):
        raise ValueError("Source bands must be nonempty angular distances in [0,180]")
    if not 0 <= c["evaluation"]["success_tolerance"] <= 1 or c["evaluation"]["band_halfwidth"] <= 0:
        raise ValueError("Invalid success tolerance or hue band width")
    for key in ("endpoint_practical_improvement", "path_practical_improvement"):
        val = c["evaluation"][key]
        if val is not None and not 0 < val < 1:
            raise ValueError(f"{key} must be null or a fraction in (0,1)")


def root(cfg):
    return Path(cfg["output"]).expanduser().resolve()
