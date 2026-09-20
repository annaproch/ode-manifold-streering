"""Steering in activation-length units with explicit zero-strength semantics."""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np


def unit(v, epsilon=1e-10):
    norm = np.linalg.norm(v)
    return v/norm if norm > epsilon else np.zeros_like(v)


def cap(v, budget):
    norm = np.linalg.norm(v)
    return v * min(1.0, budget/max(norm, 1e-30))


def checkpoints(states, count):
    """Sample recorded states at common normalized solver time, without chords."""
    indices = np.rint(np.linspace(0, len(states)-1, count)).astype(int)
    return np.asarray(states)[indices]


@dataclass
class Trajectory:
    states: np.ndarray
    eligible: bool
    reason: str
    path_length: float
    correction_length: float
    displacement: float
    norm_change: float
    seconds: float
    solver_states: np.ndarray


def steer(x, method, strength, probe, geometry, settings, target=0, gate=True):
    if not np.isfinite(strength) or strength < 0:
        raise ValueError("Nonnegative finite strength required")
    original = np.asarray(x).copy()
    x = original.astype(np.float64)
    start = time.perf_counter()
    eligible = bool(geometry.eligible(x)[0])
    steps, repeats = settings["steps"], settings["retractions"]
    ratio, epsilon = settings["correction_ratio"], settings["tiny_gradient"]
    states = [x.copy()]
    length = correction_length = 0.0
    reason = "ok"

    def result():
        # Keep original solver time after early freezing, rather than stretching
        # a short prefix across the entire reported trajectory.
        padded = states + [x.copy() for _ in range(max(0, steps+1-len(states)))]
        sampled = checkpoints(padded, settings["checkpoints"])
        # Preserve exact no-op values, including the original dtype's values.
        return Trajectory(sampled, eligible, reason, float(length), float(correction_length),
                          float(np.linalg.norm(x-original)), float(np.linalg.norm(x)-np.linalg.norm(original)),
                          time.perf_counter()-start, np.asarray(padded))

    if method == "none" or (strength == 0 and method != "R0"):
        reason = "identity"
        return result()
    # R0 intentionally exposes ungated, strength-independent retraction effects.
    if gate and not eligible and method not in {"R0", "A", "B", "S"}:
        reason = "rejected_start"
        return result()
    u = geometry.u
    caa = unit(geometry.x[geometry.labels == target].mean(0)-geometry.x[geometry.labels != target].mean(0))
    step_size = strength/steps
    previous = None
    route = geometry.spline_route(geometry.coordinates(x), target) if method == "F" else None
    if route is not None:
        segment_lengths = np.linalg.norm(np.diff(route, axis=0), axis=1)
        route_distance = np.r_[0, np.cumsum(segment_lengths)]
        valid = np.r_[True, np.diff(route_distance) > 1e-12]
        route, route_distance = route[valid], route_distance[valid]
    accepted_methods = {"A", "R0", "B", "C", "C-end", "E2", "F", "S", "E3", "E2-ms", "E2-h", "E1", "D", "E2-full"}
    if method not in accepted_methods:
        raise ValueError(f"Unknown method {method}")
    for i in range(steps):
        before = x.copy()
        correction_before = correction_length
        z0 = geometry.coordinates(x)
        if method == "F":
            distance = min(strength*(i+1)/steps, route_distance[-1])
            z = np.array([np.interp(distance, route_distance, route[:, j]) for j in range(route.shape[1])])
            candidate = original + (z-geometry.coordinates(original)) @ u.T
        elif method == "R0":
            z = z0.copy()
            for _ in range(repeats):
                correction = geometry.correction(z)
                z += correction
                correction_length += np.linalg.norm(correction)
            candidate = x + (z-z0) @ u.T
        else:
            gradient = probe.grad(x)
            if method == "A":
                candidate = x+step_size*caa
            elif method == "B":
                candidate = x+step_size*probe.ode_direction(x)
            elif method in {"C", "C-end"}:
                candidate = x+step_size*(unit(gradient @ u, epsilon) @ u.T)
            elif method == "S":
                radius = np.linalg.norm(x)
                axis = x/max(radius, epsilon)
                tangent = unit(gradient-axis*np.dot(axis, gradient), epsilon)
                # Arc-length T, not GeoSteer's angular T: tune independently.
                angle = step_size/max(radius, epsilon)
                candidate = x.copy() if not np.any(tangent) else radius*(np.cos(angle)*axis+np.sin(angle)*tangent)
            elif method in {"E1", "D"}:
                density = unit(geometry.mean_shift(z0), epsilon)
                weight = settings["additive_weight"]
                direction = unit(unit(gradient, epsilon)+weight*density @ u.T, epsilon) if method == "D" else unit(unit(gradient @ u, epsilon)+weight*density, epsilon) @ u.T
                candidate = x+step_size*direction
            else:
                basis = geometry.basis(z0)
                direction = geometry.graph_direction(z0) if method == "E3" else geometry.project(gradient @ u, basis)
                direction = unit(direction, epsilon)
                if method == "E2-h" and previous is not None and np.dot(direction, previous) < 0:
                    step_size *= .5
                previous = direction.copy()
                z = z0 + step_size*direction
                # Total correction <= ratio*T for E2, independent of r/n.
                for _ in range(repeats):
                    correction = cap(geometry.correction(z, plain=method == "E2-ms"), ratio*step_size/repeats)
                    z += correction
                    correction_length += np.linalg.norm(correction)
                candidate = x+(z-z0) @ u.T
                if method == "E2-full":
                    delta = geometry.regression_x(z)-geometry.regression_x(z0)
                    residual = delta-(delta @ u) @ u.T
                    residual = cap(residual, ratio*step_size)
                    candidate += residual
                    correction_length += np.linalg.norm(residual)
        if not np.isfinite(candidate).all():
            correction_length = correction_before
            reason = "nonfinite_proposal"
            break
        # Eligibility defines the common comparison cohort even for ungated
        # baselines. Support monitoring is part of E2/E3's intervention itself.
        if method.startswith("E2") or method == "E3":
            if not geometry.eligible(candidate, settings["support_multiplier"])[0]:
                correction_length = correction_before
                reason = "support_exit_frozen"
                break
        x = candidate
        length += np.linalg.norm(x-before)
        states.append(x.copy())
    if method == "C-end":
        z0 = geometry.coordinates(x)
        z = z0.copy()
        for _ in range(repeats):
            correction = cap(geometry.correction(z), ratio*strength/repeats)
            z += correction
            correction_length += np.linalg.norm(correction)
        candidate = x+(z-z0) @ u.T
        length += np.linalg.norm(candidate-x)
        x = candidate
        # Endpoint correction occurs at t=1, not an extra time interval.
        states[-1] = x.copy()
    return result()
