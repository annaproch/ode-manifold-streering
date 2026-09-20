"""Paired prompt bootstrap, seed summaries, validation frontiers and saved plots."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .config import root
from .pipeline import load_evaluations
from .storage import atomic, finite_mean, read_json, write_json


def cohort(row):
    if not row["primary"]:
        return row["kind"]
    return "antipodal" if row["antipodal"] else "main"


def paired_bootstrap(left, right, metric, samples, seed, alpha=.05):
    a = {(r["id"], r["seed"]): r[metric] for r in left}
    b = {(r["id"], r["seed"]): r[metric] for r in right}
    common = sorted(a.keys() & b.keys())
    if not common:
        return {"difference": None, "interval": None, "prompts": 0, "seeds": 0}
    by_prompt = defaultdict(list)
    by_seed = defaultdict(list)
    for key in common:
        delta = float(a[key])-float(b[key])
        by_prompt[key[0]].append(delta)
        by_seed[key[1]].append(delta)
    # Resample prompts, averaging fitted seeds within each prompt. Seed variation
    # is reported separately, not treated as independent prompt observations.
    values = np.array([np.mean(v) for v in by_prompt.values()])
    rng = np.random.default_rng(seed)
    means = np.empty(samples)
    for start in range(0, samples, 256):
        n = min(256, samples-start)
        means[start:start+n] = rng.choice(values, size=(n, len(values)), replace=True).mean(1)
    return {"difference": float(values.mean()),
            "interval": np.quantile(means, [alpha/2, 1-alpha/2]).tolist(),
            "prompts": len(values), "seeds": len(by_seed),
            "per_seed_difference": {str(k): float(np.mean(v)) for k, v in by_seed.items()},
            "nominal_interval_level": 1-alpha,
            "uncertainty": "percentile prompt bootstrap, seeds averaged within prompt; family-adjusted alpha"}


def report(cfg, split="test", template="main"):
    rows = load_evaluations(cfg, split, template)
    out = root(cfg) / "reports" / split / template
    frozen_path = root(cfg) / "operating-points.json"
    frozen = read_json(frozen_path) if frozen_path.exists() else None
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["method"], r["strength"], cohort(r))].append(r)
    cells = []
    for (method, strength, group), cell in sorted(grouped.items()):
        eligible = [r for r in cell if r["eligible"]]
        main_starts = [r for r in eligible if not r["baseline_target"]]
        entry = {"method": method, "strength": strength, "cohort": group,
                 "prompt_count": len({r["id"] for r in cell}), "fitted_seeds": len({r["seed"] for r in cell}),
                 "gate_coverage": float(np.mean([r["eligible"] for r in cell])),
                 "full_success": finite_mean(r["success"] for r in cell),
                 "eligible_success": finite_mean(r["success"] for r in eligible),
                 "eligible_endpoint_bc": finite_mean(r["endpoint_bc"] for r in eligible),
                 "eligible_path_bc": finite_mean(r["path_bc"] for r in eligible),
                 "full_end_other": finite_mean(r["end_other"] for r in cell),
                 "full_displacement": finite_mean(r["displacement"] for r in cell),
                 "full_unrelated_endpoint_kl": finite_mean(r.get("unrelated_endpoint_kl") for r in cell),
                 "non_target_eligible_by_band": {
                     str(band): {"prompts": len({r["id"] for r in main_starts if r["band"] == band}),
                                 "success": finite_mean(r["success"] for r in main_starts if r["band"] == band),
                                 "endpoint_bc": finite_mean(r["endpoint_bc"] for r in main_starts if r["band"] == band),
                                 "path_bc": finite_mean(r["path_bc"] for r in main_starts if r["band"] == band)}
                     for band in cfg["evaluation"]["source_bands"]} if group in {"main", "antipodal"} else {},
                 "eligible_counts_by_band_seed": {
                     str(seed): {str(band): len({r["id"] for r in eligible if r["seed"] == seed and r["band"] == band and not r["baseline_target"]})
                                 for band in cfg["evaluation"]["source_bands"]}
                     for seed in cfg["fit"]["seeds"]} if group in {"main", "antipodal"} else {},
                 "geometry_seconds": finite_mean(r["geometry_seconds"] for r in cell),
                 "model_seconds": finite_mean(r["model_seconds"] for r in cell),
                 "per_seed": {str(seed): {"success": finite_mean(r["success"] for r in eligible if r["seed"] == seed),
                                         "endpoint_bc": finite_mean(r["endpoint_bc"] for r in eligible if r["seed"] == seed),
                                         "path_bc": finite_mean(r["path_bc"] for r in eligible if r["seed"] == seed)} for seed in sorted({r["seed"] for r in cell})}}
        cells.append(entry)
    comparisons = []
    baselines = [b for b in ("B", "C", "C-end") if b in cfg["steering"]["methods"]]
    alpha = .05/max(1, len(baselines)*2)
    if split == "test" and frozen:
        main = [r for r in rows if r["eligible"] and r["primary"] and not r["antipodal"] and not r["baseline_target"]]
        ours = [r for r in main if r["method"] == "E2"]
        for baseline in baselines:
            other = [r for r in main if r["method"] == baseline]
            if not ours or not other:
                continue
            success_a = finite_mean(r["success"] for r in ours)
            success_b = finite_mean(r["success"] for r in other)
            adequate = all(len({r["id"] for r in population if r["seed"] == seed and r["band"] == band}) >= cfg["evaluation"]["minimum_band_prompts"]
                           for population in (ours, other) for seed in cfg["fit"]["seeds"]
                           for band in cfg["evaluation"]["source_bands"] if band != 180)
            comparable = (frozen["methods"]["E2"]["matched_on_validation"] and frozen["methods"][baseline]["matched_on_validation"]
                          and min(success_a, success_b) >= cfg["evaluation"]["success_level"]
                          and abs(success_a-success_b) <= cfg["evaluation"]["success_tolerance"])
            for metric, hypothesis in (("endpoint_bc", "endpoint"), ("path_bc", "path")):
                effect = paired_bootstrap(ours, other, metric, cfg["evaluation"]["bootstrap_samples"],
                                          cfg["evaluation"]["bootstrap_seed"], alpha)
                base = finite_mean(r[metric] for r in other)
                relative = -effect["difference"]/base if base and effect["difference"] is not None else None
                threshold = frozen["hypotheses"][hypothesis]
                same_direction = bool(effect.get("per_seed_difference")) and all(v < 0 for v in effect["per_seed_difference"].values())
                meets = bool(adequate and comparable and threshold is not None and relative is not None and relative >= threshold
                             and effect["interval"] and effect["interval"][1] < 0 and same_direction and effect["seeds"] >= 3)
                comparisons.append({"method": "E2", "baseline": baseline, "metric": metric, "hypothesis": hypothesis,
                                    "success_comparable": bool(comparable), "adequate_eligible_band_counts": adequate, "relative_improvement": relative,
                                    "practical_threshold": threshold, "meets_pilot_criterion": meets, **effect})
    result = {"split": split, "template": template, "cells": cells, "comparisons": comparisons,
              "operating_points": frozen,
              "interpretation": "Pilot evidence only; validate another target/template before expansion. A/B/S run ungated; eligible comparisons share the same entry cohort. R0 is an ungated diagnostic. S is a numerical norm control, not a GeoSteer reproduction. Validation sweeps decode endpoints only; path metrics require validate-paths or test. T is a calibrated multiplier; raw activation distances are saved per row."}
    write_json(out / "summary.json", result)
    lines = [f"# Colour pilot: {split}, {template}", "", result["interpretation"], "",
             "Means below average prompt/seed rows; fitted-seed breakdowns and paired intervals are in summary.json.", "",
             "| Method | T multiplier | Cohort | Prompts | Gate coverage | Full success | Eligible success | Endpoint BC | Path BC | Unrelated KL |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    fmt = lambda v: "—" if v is None else f"{v:.4f}"
    for c in cells:
        lines.append(f"| {c['method']} | {c['strength']:g} | {c['cohort']} | {c['prompt_count']} | {fmt(c['gate_coverage'])} | {fmt(c['full_success'])} | {fmt(c['eligible_success'])} | {fmt(c['eligible_endpoint_bc'])} | {fmt(c['eligible_path_bc'])} | {fmt(c['full_unrelated_endpoint_kl'])} |")
    if comparisons:
        lines += ["", "## Paired E2 differences (negative favors E2)", "",
                  "Intervals adjust the nominal alpha across both metrics and all listed baselines. These are approximate bootstrap intervals, not a proof of familywise coverage.", "",
                  "| Baseline | Metric | Mean difference | Interval | Matched success on validation and test? | Meets predeclared pilot criterion? |",
                  "|---|---|---:|---|---|---|"]
        for c in comparisons:
            interval = "—" if not c["interval"] else f"[{c['interval'][0]:.4g}, {c['interval'][1]:.4g}]"
            lines.append(f"| {c['baseline']} | {c['metric']} | {fmt(c['difference'])} | {interval} | {c['success_comparable']} | {c['meets_pilot_criterion']} |")
    atomic(out / "report.md", "\n".join(lines)+"\n")
    _plot(cells, out)
    return {"report": str(out / "report.md"), "summary": str(out / "summary.json"), "comparisons": comparisons}


def _plot(cells, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for method in sorted({c["method"] for c in cells}):
        selected = [c for c in cells if c["method"] == method and c["cohort"] == "main" and c["eligible_success"] is not None]
        selected.sort(key=lambda c: c["strength"])
        for ax, metric, title in zip(axes, ("eligible_endpoint_bc", "eligible_path_bc"), ("Endpoint deviation", "Path deviation")):
            measured = [c for c in selected if c[metric] is not None]
            ax.plot([c["eligible_success"] for c in measured], [c[metric] for c in measured], "o-", label=method)
            ax.set(xlabel="Eligible target success", ylabel="Mean Bhattacharyya distance", title=title)
            ax.grid(alpha=.2)
    axes[1].legend(fontsize=8)
    fig.savefig(out / "success-quality.png", dpi=160)
    plt.close(fig)
