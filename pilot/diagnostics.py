"""Saved geometry plots and zero/small-strength audits of real activations."""
from __future__ import annotations

import numpy as np

from .config import root
from .pipeline import load_collection, load_fit
from .steering import steer
from .storage import read_json, write_json


def run(cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, cache, _ = load_collection(cfg)
    index = read_json(root(cfg) / "fits" / "index.json")
    out = root(cfg) / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    audits = []
    used = set()
    for name in index["entries"]:
        probe, geometry, meta = load_fit(cfg, name)
        key = (meta["layer"], meta["dimension"])
        if key in used:
            continue
        used.add(key)
        x = cache[f"layer_{meta['layer']}"]
        colors = np.array([r["kind"] == "colour" and r["template"] == "main" and r["split"] in {"train", "validation"} for r in rows])
        z = geometry.coordinates(x[colors])
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
        for ax, attribute in zip(axes, ("hue", "saturation", "lightness")):
            values = [r[attribute] for r, keep in zip(rows, colors) if keep]
            scatter = ax.scatter(z[:, 0], z[:, 1], c=values, s=7, cmap="hsv" if attribute == "hue" else "viridis")
            ax.set(title=f"{name}: {attribute}", xlabel="PC1", ylabel="PC2")
            fig.colorbar(scatter, ax=ax)
        fig.savefig(out / f"{name}-geometry.png", dpi=150)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(7, 3), constrained_layout=True)
        score = meta["restricted_score"]
        ax.plot(score["hues"], score["log_dre"], ".-")
        ax.axhline(0, color="gray", linewidth=.7)
        ax.set(xlabel="Training input hue", ylabel="Prior-corrected log density ratio", title=name)
        fig.savefig(out / f"{name}-score.png", dpi=150)
        plt.close(fig)
        # Deterministic small sample from each support category; no test inspection.
        picks = []
        for kind in ("colour", "gray", "brown", "unrelated"):
            picks.extend([i for i, r in enumerate(rows) if r["split"] == "validation" and r["template"] == "main"
                          and r["kind"] == kind and (r["primary"] or kind != "colour")][:8])
        for i in picks:
            for method in ("R0", "E2", "C-end"):
                for strength in (0.0, 1e-6, 1e-4, .01, .1):
                    result = steer(x[i], method, strength, probe, geometry, cfg["steering"], meta["target"])
                    bound = (1+cfg["steering"]["correction_ratio"])*strength
                    valid = method == "R0" or result.displacement <= bound+1e-8
                    if not valid:
                        raise AssertionError("Real-activation correction budget exceeded")
                    audits.append({"fit": name, "id": rows[i]["id"], "kind": rows[i]["kind"],
                                   "method": method, "strength": strength, "eligible": result.eligible,
                                   "displacement": result.displacement, "reason": result.reason,
                                   "budget_check": bool(valid)})
    write_json(out / "small-strength.json", {"rows": audits, "note": "Validation-only activation audit; no model forwards"})
    return {"directory": str(out), "audit_rows": len(audits)}
