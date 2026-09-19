"""Synthetic hue-ring toy for the manifold-constrained ODESteer plan (experiment-plan.md §9).

A ring of radius R (hue) plus a lightness axis lives in a 3-D subspace of a d-dim space,
with a large mean offset and isotropic noise. The target classifier is ODESteer's own
NormedPolyClassifier. Every steering condition in the plan is implemented here so the
tables in §9 can be regenerated from one place.

This is a check of the steering mechanism only; it says nothing about any LLM.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

_ODESTEER_SRC = os.environ.get(
    "ODESTEER_SRC", os.path.join(os.path.dirname(__file__), "third_party", "odesteer", "src")
)
sys.path.insert(0, _ODESTEER_SRC)
# Import the kernels module directly: odesteer.steer pulls in optional deps (POT) we don't need.
from odesteer.utils.kernels import NormedPolyClassifier  # noqa: E402


@dataclass
class World:
    """One sampled toy: data, subspace, classifier and KDE."""

    d: int
    R: float
    noise: float
    mu: torch.Tensor        # true ring centre (d,)
    e: torch.Tensor         # true basis, rows e1 (cos hue), e2 (sin hue), e3 (lightness): (3, d)
    X: torch.Tensor         # data (N, d)
    m: torch.Tensor         # PCA mean (d,)
    U: torch.Tensor         # PCA basis (d, 3)
    Z: torch.Tensor         # projected data (N, 3)
    bw: float               # KDE bandwidth (5-fold CV)
    kde: KernelDensity
    clf: NormedPolyClassifier
    caa: torch.Tensor       # mean(pos) - mean(neg), unit norm (d,)
    train_acc: float

    def embed(self, theta: torch.Tensor, light: torch.Tensor | None = None) -> torch.Tensor:
        light = torch.zeros_like(theta) if light is None else light
        c = torch.stack([self.R * torch.cos(theta), self.R * torch.sin(theta), light], dim=1)
        return self.mu + c @ self.e


def make_world(seed: int = 0, d: int = 256, N: int = 3000, R: float = 3.0, noise: float = 0.3,
               mu_norm: float = 20.0, light_sd: float = 0.8, target_halfwidth_deg: float = 30.0) -> World:
    torch.manual_seed(seed)
    np.random.seed(seed)
    Q = torch.linalg.qr(torch.randn(d, d))[0]
    e = Q[:, :3].T.contiguous()
    mu = Q[:, 3] * mu_norm
    hue = torch.rand(N) * 2 * math.pi
    light = torch.randn(N) * light_sd
    c = torch.stack([R * torch.cos(hue), R * torch.sin(hue), light], dim=1)
    X = mu + c @ e + noise * torch.randn(N, d)
    pos = wrap(hue).abs() < math.radians(target_halfwidth_deg)

    clf = NormedPolyClassifier(n_components=8000, degree=2, gamma=0.1, coef0=1.0).fit(X[pos], X[~pos])
    train_acc = clf.score(X, pos.float())
    caa = X[pos].mean(0) - X[~pos].mean(0)

    m = X.mean(0)
    U = torch.linalg.svd(X - m, full_matrices=False)[2][:3].T.contiguous()
    Z = (X - m) @ U
    grid = GridSearchCV(KernelDensity(), {"bandwidth": np.logspace(-1.5, 0.5, 15)}, cv=5).fit(Z.numpy())
    return World(d, R, noise, mu, e, X, m, U, Z, float(grid.best_params_["bandwidth"]),
                 grid.best_estimator_, clf, caa / caa.norm(), float(train_acc))


def wrap(a: torch.Tensor) -> torch.Tensor:
    return (a + math.pi) % (2 * math.pi) - math.pi


def unit(v: torch.Tensor) -> torch.Tensor:
    return v / (v.norm(dim=-1, keepdim=True) + 1e-10)


# ---------------------------------------------------------------- manifold pieces (in z = Uᵀ(x − m))

def grad_logp(w: World, z: torch.Tensor) -> torch.Tensor:
    """∇_z log p for the Gaussian KDE; bw² · this is one mean-shift step."""
    diff = w.Z[None] - z[:, None]
    wt = torch.softmax(-(diff ** 2).sum(-1) / (2 * w.bw ** 2), dim=1)
    return (wt[..., None] * diff).sum(1) / w.bw ** 2


def local_basis(w: World, z: torch.Tensor, m_dim: int = 2, knn: int = 60) -> torch.Tensor:
    """Top-m local principal directions of the kNN of each z: (B, m, 3)."""
    idx = torch.cdist(z, w.Z).topk(knn, largest=False).indices
    nb = w.Z[idx]
    return torch.linalg.svd(nb - nb.mean(1, keepdim=True), full_matrices=False)[2][:, :m_dim]


def project(V: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bkd,bk->bd", V, torch.einsum("bkd,bd->bk", V, v))


def logp(w: World, z: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(w.kde.score_samples(z.numpy()))


# ---------------------------------------------------------------- conditions

@dataclass
class Cond:
    """A steering condition from experiment-plan.md §5.3.

    kind: 'A' CAA, 'B' ODESteer, 'C' projection only, 'D' density only (full space),
          'E1' additive density term in U, 'E2' tangent step + retraction in U.
    """

    kind: str
    lam: float = 0.0            # D / E1: weight of the normalised density term
    perp: bool = False          # E1: keep only the density component at right angles to the target direction
    retraction: str = "scms"    # E2: 'scms' or 'ms' (plain mean shift)
    r: int = 3                  # E2: retraction steps per Euler step
    stepctl: str = "none"       # 'none', 'stop' (freeze on direction reversal), 'halve' (halve step on reversal)
    m_dim: int = 2
    knn: int = 60

    @property
    def label(self) -> str:
        s = self.kind
        if self.kind in ("D", "E1"):
            s += f" λ={self.lam:g}" + (" ⟂" if self.perp else "")
        if self.kind == "E2":
            s += f" {self.retraction} r={self.r}"
        if self.stepctl != "none":
            s += f" +{self.stepctl}"
        return s


def direction(w: World, c: Cond, x: torch.Tensor) -> torch.Tensor:
    """Unit step direction in the full space for conditions A–E1."""
    if c.kind == "A":
        return w.caa.expand_as(x)
    g = w.clf.grad(x)
    if c.kind == "B":
        return unit(g)
    if c.kind == "C":
        return unit((g @ w.U) @ w.U.T)
    gl = unit(grad_logp(w, (x - w.m) @ w.U))
    if c.kind == "D":
        return unit(unit(g) + c.lam * gl @ w.U.T)
    if c.kind == "E1":
        a = unit(g @ w.U)
        if c.perp:
            gl = gl - (gl * a).sum(-1, keepdim=True) * a
        return unit((a + c.lam * gl) @ w.U.T)
    raise ValueError(c.kind)


def run(w: World, c: Cond, start_deg: float, T: float, seed: int = 0, steps: int = 40, B: int = 32) -> dict:
    """Steer B noisy starts at hue start_deg for total length T; return path metrics.

    Metrics (medians over the B starts unless noted):
      success   fraction whose final hue is within ±30° of red (0°)
      min_r     lowest radius along the path, ÷ R (1 = stays on the ring, 0 = through the centre)
      end_hue   |final hue| in degrees
      swing     range of hue over the last 5 steps, degrees
      logp_end  mean KDE log-density at the end
      clf_pos   fraction the classifier calls positive at the end
      offplane  mean growth of the component outside the true 3-D subspace
      stopped   fraction frozen by the 'stop' rule
    """
    g = torch.Generator().manual_seed(1000 + seed)
    th = torch.full((B,), math.radians(start_deg)) + 0.1 * torch.randn(B, generator=g)
    x = w.embed(th) + w.noise * torch.randn(B, w.d, generator=g)
    h = torch.full((B,), T / steps)
    active = torch.ones(B, dtype=torch.bool)
    prev = None
    hues, radii, offs = [], [], []
    for i in range(steps + 1):
        cc = (x - w.mu) @ w.e.T
        hues.append(torch.rad2deg(torch.atan2(cc[:, 1], cc[:, 0])))
        radii.append(cc[:, :2].norm(dim=1) / w.R)
        offs.append(((x - w.mu) - cc @ w.e).norm(dim=1))
        if i == steps:
            break
        if c.kind == "E2":
            z = (x - w.m) @ w.U
            t = unit(project(local_basis(w, z, c.m_dim, c.knn), w.clf.grad(x) @ w.U))
        else:
            t = direction(w, c, x)
        if prev is not None and c.stepctl != "none":
            rev = (t * prev).sum(-1) < 0
            if c.stepctl == "stop":
                active &= ~rev
            else:
                h = torch.where(rev, h * 0.5, h)
        prev = t
        step = (h * active.float())[:, None]
        if c.kind != "E2":
            x = x + step * t
            continue
        z = (x - w.m) @ w.U
        z = z + step * t
        for _ in range(c.r):
            ms = w.bw ** 2 * grad_logp(w, z)
            if c.retraction == "scms":
                ms = ms - project(local_basis(w, z, c.m_dim, c.knn), ms)
            z = z + active.float()[:, None] * ms
        x = w.m + z @ w.U.T + (x - w.m - ((x - w.m) @ w.U) @ w.U.T)
    H, Rr = torch.stack(hues), torch.stack(radii)
    last = H[-5:]
    return dict(
        success=float((H[-1].abs() < 30).float().mean()),
        min_r=float(Rr.min(0).values.median()),
        end_hue=float(H[-1].abs().median()),
        swing=float((last.max(0).values - last.min(0).values).median()),
        logp_end=float(logp(w, (x - w.m) @ w.U).mean()),
        clf_pos=float(w.clf.predict(x).float().mean()),
        offplane=float((offs[-1] - offs[0]).mean()),
        stopped=float((~active).float().mean()),
    )
