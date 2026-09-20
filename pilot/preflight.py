"""Explicit CPU-only diagnostics. This module is never executed on import."""
from __future__ import annotations

import sys

import numpy as np

from .config import root
from .data import ANGLES, circular_distance
from .geometry import Geometry
from .probe import Probe
from .steering import steer
from .storage import atomic, write_json


def fixture(seed=0, features=512):
    rng = np.random.default_rng(seed)
    d, n = 32, 480
    basis = np.linalg.qr(rng.normal(size=(d, d)))[0]
    mu = 20*basis[:, 3]
    angles = rng.uniform(0, 2*np.pi, n)
    points = mu+3*np.cos(angles[:, None])*basis[:, 0]+3*np.sin(angles[:, None])*basis[:, 1]+.08*rng.normal(size=(n, d))
    hues = np.rad2deg(angles)
    labels = np.argmin(circular_distance(hues[:, None], ANGLES[None]), axis=1)
    labels[circular_distance(hues, 0) < 30] = 0
    train, validation = points[:360], points[360:]
    probe = Probe(d, features, seed).fit(train, labels[:360] == 0)
    settings = {"neighbors": 20, "tangent_dim": 1, "gate_quantile": .99,
                "bandwidth_factors": [.5, 1, 2], "graph_neighbors": 8}
    geometry = Geometry.fit(train, labels[:360], hues[:360], validation, 3, settings, seed, labels[:360] == 0)
    return probe, geometry, mu, basis, validation


def gradient_error(probe, x):
    numerical = np.empty_like(x)
    step = 1e-4
    for i in range(len(x)):
        offset = np.zeros_like(x)
        offset[i] = step
        numerical[i] = (probe.logits(x+offset)[0]-probe.logits(x-offset)[0])/(2*step)
    analytic = probe.grad(x)
    return float(np.max(np.abs(numerical-analytic))), float(np.dot(analytic, x))


def upstream_check(source):
    """Optional read-only import of a user's pinned upstream checkout, not vendored."""
    sys.path.insert(0, str(source))
    import torch
    from odesteer.utils.kernels import NormedPolyClassifier

    torch.manual_seed(19)
    x = torch.randn(160, 24)
    labels = x[:, 0] > .5
    upstream = NormedPolyClassifier(n_components=512, degree=2, gamma=.1, coef0=1).fit(x[labels], x[~labels])
    ours = Probe(24, 512)
    ours.indices = upstream.kernel.indexHash_.cpu().numpy().copy()
    ours.signs = upstream.kernel.bitHash_.cpu().numpy().copy()
    ours.coef = upstream.coef.detach().cpu().numpy().copy()
    ours.intercept = float(upstream.intercept)
    ours.prior = float(labels.float().mean())
    feature_error = float(np.max(np.abs(ours.transform(x.numpy())-upstream.kernel.transform(x).numpy())))
    # Positive prior scaling cancels in unit vector fields.
    before = x[:4].clone()
    reference = before.clone()
    candidate = before.numpy().astype(np.float64)
    for _ in range(10):
        g = upstream.grad(reference)
        reference += .1*g/(g.norm(dim=1, keepdim=True)+1e-10)
        candidate += .1*ours.ode_direction(candidate)
    endpoint_error = float(np.max(np.abs(candidate-reference.numpy())))
    if feature_error > 1e-4 or endpoint_error > 1e-3:
        raise AssertionError(f"Upstream parity failed: features={feature_error}, endpoint={endpoint_error}")
    return {"feature_max_error": feature_error, "ten_step_endpoint_max_error": endpoint_error,
            "note": "Same upstream hashes/weights, compared with its normalized Euler vector field"}


def run(cfg, upstream_source=None):
    probe, geometry, mu, basis, validation = fixture()
    error, tangent_dot = gradient_error(probe, validation[0])
    if error > 1e-5 or abs(tangent_dot) > 1e-8:
        raise AssertionError(f"Probe gradient check failed: {error}, radial component {tangent_dot}")
    settings = cfg["steering"]
    sources = {f"hue{h}": mu+3*np.cos(np.deg2rad(h))*basis[:, 0]+3*np.sin(np.deg2rad(h))*basis[:, 1] for h in (90, 120, 150, 180)}
    sources["gray"] = mu+.05*basis[:, 0]
    sources["unrelated"] = mu+15*basis[:, 8]
    methods = ["none", "A", "R0", "B", "C", "C-end", "E2", "F", "S", "E3"]
    records = []
    for name, source in sources.items():
        for method in methods:
            for strength in (0.0, 1e-6, 1e-4, .1, 4.0, 10.0):
                result = steer(source, method, strength, probe, geometry, settings)
                if strength == 0 and method != "R0":
                    assert np.array_equal(result.states[-1], source), (name, method, "not an exact identity")
                if method in {"E2", "E3", "C-end"}:
                    assert result.displacement <= (1+settings["correction_ratio"])*strength+1e-8
                if method == "S":
                    assert abs(result.norm_change) < 1e-8
                if method in {"C", "C-end", "E2", "E3", "F"}:
                    delta = result.states[-1]-source
                    assert np.linalg.norm(delta-(delta @ geometry.u) @ geometry.u.T) < 1e-8
                endpoint = result.states[-1]-mu
                hue = float(np.rad2deg(np.arctan2(endpoint @ basis[:, 1], endpoint @ basis[:, 0])))
                records.append({"source": name, "method": method, "T": strength, "eligible": result.eligible,
                                "reason": result.reason, "displacement": result.displacement,
                                "norm_change": result.norm_change, "hue": hue,
                                "success": abs(hue) < 30, "posterior_positive": bool(probe.logits(result.states[-1])[0] > 0),
                                "density_ratio_positive": bool(probe.log_dre(result.states[-1])[0] > 0)})
    result = {"passed": True, "gradient_max_error": error, "gradient_radial_dot": tangent_dot,
              "rows": records, "upstream": upstream_check(upstream_source) if upstream_source else None,
              "scope": "Synthetic CPU numerical/control checks, not LLM evidence or archived toy reproduction"}
    out = root(cfg) / "preflight"
    write_json(out / "report.json", result)
    lines = ["# Numerical preflight", "", "Passed synthetic identities, derivative, displacement-budget, residual and spherical-norm checks.", "",
             f"Gradient maximum absolute error: {error:.6g}.", "",
             "| Source | Method | T | Eligible | Displacement | Final hue | Reason |", "|---|---|---:|---|---:|---:|---|"]
    for r in records:
        lines.append(f"| {r['source']} | {r['method']} | {r['T']:g} | {r['eligible']} | {r['displacement']:.6g} | {r['hue']:.2f} | {r['reason']} |")
    atomic(out / "report.md", "\n".join(lines)+"\n")
    return {"passed": True, "report": str(out / "report.json"), "upstream": result["upstream"]}
