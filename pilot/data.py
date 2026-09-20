"""Deterministic HSL stimuli with hex-level split isolation across templates."""
from __future__ import annotations

import colorsys
import hashlib
import json

import numpy as np

from .config import root
from .storage import atomic, digest, manifest

LABELS = ["red", "orange", "yellow", "green", "blue", "purple", "gray", "brown", "other"]
ANGLES = np.array([0, 30, 58, 120, 235, 285], dtype=float)


def circular_distance(a, b):
    return np.abs((np.asarray(a) - b + 180) % 360 - 180)


def prompt(row, extended=True):
    if row["kind"] == "unrelated":
        return row["text"]
    choices = ", ".join(LABELS[:8 if extended else 6])
    if row["template"] == "main":
        return f"Question: Which color name best describes the hex code {row['hex']}? Choose one of: {choices}.\nAnswer:"
    return f"Choose the best name for color {row['hex']} from: {choices}.\nColor name:"


def split_for(hex_code, settings):
    # Hex identity alone determines split, including every template and control.
    raw = hashlib.sha256(f"{settings['seed']}:{hex_code}".encode()).digest()
    u = int.from_bytes(raw[:8], "big") / 2**64
    total = 0.0
    for name, fraction in settings["split_fractions"].items():
        total += fraction
        if u < total:
            return name
    return "test"


def make(cfg):
    d = cfg["data"]
    manifest(root(cfg) / "dataset-manifest.json", {"data": d, "format": 1})
    by_hex = {}

    def add(h, s, l, kind, expected=None):
        rgb = colorsys.hls_to_rgb(h / 360, l, s)
        hex_code = "#" + "".join(f"{round(v * 255):02X}" for v in rgb)
        if hex_code in by_hex:
            return
        primary = kind == "colour" and s == d["primary_saturation"] and l == d["primary_lightness"]
        distances = circular_distance(ANGLES, h)
        nearest = int(np.argmin(distances))
        if expected is None and distances[nearest] <= 6:
            expected = LABELS[nearest]
        by_hex[hex_code] = dict(hex=hex_code, hue=float(h), saturation=float(s), lightness=float(l),
                               kind=kind, primary=primary, expected=expected,
                               split=split_for(hex_code, d))

    for s in d["saturations"]:
        for l in d["lightnesses"]:
            primary = s == d["primary_saturation"] and l == d["primary_lightness"]
            step = d["hue_step"] if primary else d["context_hue_step"]
            for h in np.arange(0, 360, step):
                add(h, s, l, "colour")
    for light in np.linspace(0.2, 0.8, 31):
        add(0, 0, float(light), "gray", "gray")
    for hue in (20, 25, 30, 35):
        for light in (0.13, 0.17, 0.21, 0.25, 0.29, 0.33):
            add(hue, 0.75, light, "brown", "brown")
    unrelated = [
        "Question: What is seven plus five?\nAnswer:",
        "Question: Which planet is closest to the Sun?\nAnswer:",
        "Question: What is the capital of France?\nAnswer:",
        "Question: Complete the sequence: 2, 4, 6,\nAnswer:",
        "Question: What animal says meow?\nAnswer:",
        "Question: What is the opposite of hot?\nAnswer:",
        "Question: Name a musical instrument.\nAnswer:",
        "Question: How many sides does a triangle have?\nAnswer:",
        "Question: What is water's chemical formula?\nAnswer:",
        "Question: Which day follows Monday?\nAnswer:",
        "Question: Complete this sentence: The child opened the\nAnswer:",
        "Question: Write the first word of a polite greeting.\nAnswer:",
    ]
    for index, text in enumerate(unrelated):
        # OOD controls never enter training, reference or model-selection labels.
        key = f"unrelated-{index:03d}"
        by_hex[key] = dict(hex=key, hue=0.0, saturation=0.0, lightness=0.0,
                           kind="unrelated", primary=False, expected=None, text=text,
                           split="validation" if index % 2 == 0 else "test")
    rows = []
    for row in by_hex.values():
        for template in d["templates"]:
            item = {**row, "template": template}
            item["id"] = digest([row["hex"], template])[:20]
            rows.append(item)
    rows.sort(key=lambda r: r["id"])
    atomic(root(cfg) / "stimuli.jsonl", "".join(json.dumps(row) + "\n" for row in rows))
    from .storage import write_json
    summary = coverage(rows, cfg)
    write_json(root(cfg) / "coverage.json", summary)
    return {"rows": len(rows), "hexes": sum(r["kind"] != "unrelated" for r in by_hex.values()),
            "splits": {s: sum(r["split"] == s for r in rows) for s in d["split_fractions"]}, "coverage": summary}


def read(cfg):
    path = root(cfg) / "stimuli.jsonl"
    if not path.exists():
        raise ValueError("Run the data stage first")
    manifest(root(cfg) / "dataset-manifest.json", {"data": cfg["data"], "format": 1})
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def balanced_limit(rows, maximum, key="kind"):
    if not maximum or len(rows) <= maximum:
        return rows
    groups = {}
    for row in rows:
        groups.setdefault(row[key], []).append(row)
    out = []
    for i in range(max(map(len, groups.values()))):
        for name in sorted(groups):
            if i < len(groups[name]):
                out.append(groups[name][i])
                if len(out) == maximum:
                    return out
    return out


def evaluation_rows(rows, cfg, split, include_template="main"):
    target_angle = ANGLES[LABELS.index(cfg["fit"]["target"])]
    bands = cfg["evaluation"]["source_bands"]
    half = cfg["evaluation"]["band_halfwidth"]
    out = []
    for row in rows:
        if row["split"] != split or row["template"] != include_template:
            continue
        if row["kind"] != "colour":
            out.append(row)
        elif row["primary"] and any(abs(float(circular_distance(row["hue"], target_angle)) - b) <= half for b in bands):
            out.append(row)
    return balanced_limit(out, cfg["evaluation"]["maximum_prompts"])


def coverage(rows, cfg):
    result = {"primary_unique_hexes": len({r["hex"] for r in rows if r["primary"]}), "splits": {}, "adequate": True}
    for split in ("validation", "test"):
        selected = evaluation_rows(rows, cfg, split)
        counts = {}
        target_angle = ANGLES[LABELS.index(cfg["fit"]["target"])]
        for band in cfg["evaluation"]["source_bands"]:
            n = sum(r["primary"] and abs(float(circular_distance(r["hue"], target_angle))-band) <= cfg["evaluation"]["band_halfwidth"] for r in selected)
            required = cfg["evaluation"]["minimum_antipodal_prompts"] if band == 180 else cfg["evaluation"]["minimum_band_prompts"]
            counts[str(band)] = {"count": n, "required": required, "adequate": n >= required}
            result["adequate"] = result["adequate"] and n >= required
        result["splits"][split] = counts
    result["note"] = "Counts follow hex deduplication and evaluation subsampling, before support gating. Repeated templates are not independent hex examples."
    return result
