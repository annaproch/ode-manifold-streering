"""Non-circular toy geometries: does manifold-constrained steering help where the path must bend?

Two 1-D curves in a 3-D subspace of a d-dim space (same noise / offset / classifier as ring_toy):
  horseshoe  an arc of radius R from −150° to +150°; start near the −150° tip, target near the +150° tip.
             A straight path crosses the empty gap (length ≈ R); the on-data path goes the long way (≈ 5.2 R).
  Y          three arms of length 2R from a junction at 0°, 120°, 240°; start at the tip of the 240° arm,
             target at the tip of the 120° arm. The on-data path goes through the junction.

Conditions: B (ODESteer), E2 (tangent + SCMS, m = 1), and E3 = E2 with the classifier replaced by a
graph score: s = −(shortest-path distance on the kNN graph of stored points to the nearest positive).
E3 is geodesic-aware by construction; it tests whether the target score, not the constraint, is the bottleneck.

Metrics (medians over starts):
  success   final point within 0.5 R of the target tip (in the true coordinates)
  max_gap   largest distance from the path to the nearest stored data point, ÷ R (≈ noise level = stays on data)
  progress  fraction of the on-data distance from start to target covered (1 = arrived);
            0 if the run ends off the data (final gap > 0.3 R)
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import statistics

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from ring_toy import NormedPolyClassifier, project, unit

HERE = os.path.dirname(os.path.abspath(__file__))


def sample_curve(kind: str, n: int, R: float, g: torch.Generator):
    """Return points (n, 2) in the curve plane and their arc-length coordinate along the start→target path."""
    if kind == "horseshoe":
        a = (torch.rand(n, generator=g) * 2 - 1) * math.radians(150)
        pts = torch.stack([R * torch.cos(a), R * torch.sin(a)], 1)
        s = (a + math.radians(150)) * R                      # 0 at the start tip, 5π/3·R at the target tip
        return pts, s, 2 * math.radians(150) * R
    if kind == "Y":
        arm = torch.randint(0, 3, (n,), generator=g)
        t = torch.rand(n, generator=g) * 2 * R
        ang = torch.tensor([0.0, 2 * math.pi / 3, 4 * math.pi / 3])[arm]
        pts = torch.stack([t * torch.cos(ang), t * torch.sin(ang)], 1)
        # start = tip of arm 2, target = tip of arm 1; path coordinate along arm2 → junction → arm1
        s = torch.where(arm == 2, 2 * R - t, torch.where(arm == 1, 2 * R + t, torch.full_like(t, float("nan"))))
        return pts, s, 4 * R
    raise ValueError(kind)


def endpoints(kind: str, R: float):
    if kind == "horseshoe":
        return (R * math.cos(-2.618), R * math.sin(-2.618)), (R * math.cos(2.618), R * math.sin(2.618))
    tip = lambda k: (2 * R * math.cos(2 * math.pi * k / 3), 2 * R * math.sin(2 * math.pi * k / 3))
    return tip(2), tip(1)


class CurveWorld:
    def __init__(self, kind: str, seed: int = 0, d: int = 256, N: int = 3000, R: float = 3.0,
                 noise: float = 0.3, mu_norm: float = 20.0, knn_graph: int = 10):
        g = torch.Generator().manual_seed(seed)
        torch.manual_seed(seed)
        self.kind, self.d, self.R, self.noise = kind, d, R, noise
        Q = torch.linalg.qr(torch.randn(d, d, generator=g))[0]
        self.e = Q[:, :3].T.contiguous()
        self.mu = Q[:, 3] * mu_norm
        pts, self.s, self.total = sample_curve(kind, N, R, g)
        self.pts = pts
        c = torch.cat([pts, torch.zeros(N, 1)], 1)
        self.X = self.mu + c @ self.e + noise * torch.randn(N, d, generator=g)
        self.start, self.target = endpoints(kind, R)
        tgt = torch.tensor(self.target)
        pos = (pts - tgt).norm(dim=1) < 0.5 * R
        self.clf = NormedPolyClassifier(n_components=8000, degree=2, gamma=0.1, coef0=1.0).fit(self.X[pos], self.X[~pos])
        self.train_acc = float(self.clf.score(self.X, pos.float()))
        self.m = self.X.mean(0)
        self.U = torch.linalg.svd(self.X - self.m, full_matrices=False)[2][:3].T.contiguous()
        self.Z = (self.X - self.m) @ self.U
        # KDE bandwidth: fixed at the noise scale (CV on a thin curve picks about this; kept simple here)
        self.bw = 0.25
        # graph score for E3: −shortest-path distance to the nearest positive point on a kNN graph
        D = torch.cdist(self.Z, self.Z)
        nn = D.topk(knn_graph + 1, largest=False)
        rows = np.repeat(np.arange(N), knn_graph)
        W = csr_matrix((nn.values[:, 1:].reshape(-1).numpy(), (rows, nn.indices[:, 1:].reshape(-1).numpy())), shape=(N, N))
        dist = dijkstra(W, directed=False, indices=np.where(pos.numpy())[0], min_only=True)
        self.graph_score = torch.as_tensor(-np.nan_to_num(dist, posinf=dist[np.isfinite(dist)].max() * 2), dtype=torch.float32)

    def coords(self, x):
        return ((x - self.mu) @ self.e.T)[:, :2]


def grad_logp(w, z):
    diff = w.Z[None] - z[:, None]
    wt = torch.softmax(-(diff ** 2).sum(-1) / (2 * w.bw ** 2), dim=1)
    return (wt[..., None] * diff).sum(1) / w.bw ** 2


def local_basis(w, z, m_dim=1, knn=30):
    idx = torch.cdist(z, w.Z).topk(knn, largest=False).indices
    nb = w.Z[idx]
    return torch.linalg.svd(nb - nb.mean(1, keepdim=True), full_matrices=False)[2][:, :m_dim]


def graph_dir(w, z, knn=30):
    """Local regression gradient of the graph score around z."""
    idx = torch.cdist(z, w.Z).topk(knn, largest=False).indices
    nb, sc = w.Z[idx], w.graph_score[idx]
    dz = nb - nb.mean(1, keepdim=True)
    ds = sc - sc.mean(1, keepdim=True)
    return (dz * ds[..., None]).sum(1)


def run(w, method, T, seed=0, steps=60, B=32, r=3):
    g = torch.Generator().manual_seed(2000 + seed)
    st = torch.tensor(w.start) + 0.05 * w.R * torch.randn(B, 2, generator=g)
    c = torch.cat([st, torch.zeros(B, 1)], 1)
    x = w.mu + c @ w.e + w.noise * torch.randn(B, w.d, generator=g)
    gaps = []
    for i in range(steps + 1):
        z = (x - w.m) @ w.U
        gaps.append(torch.cdist(z, w.Z).min(1).values / w.R)
        if i == steps:
            break
        h = T / steps
        if method == "B":
            x = x + h * unit(w.clf.grad(x))
            continue
        V = local_basis(w, z)
        a = w.clf.grad(x) @ w.U if method == "E2" else graph_dir(w, z)
        z = z + h * unit(project(V, a))
        for _ in range(r):
            ms = w.bw ** 2 * grad_logp(w, z)
            z = z + ms - project(local_basis(w, z), ms)
        x = w.m + z @ w.U.T + (x - w.m - ((x - w.m) @ w.U) @ w.U.T)
    end = w.coords(x)
    tgt = torch.tensor(w.target)
    # progress: arc coordinate of the nearest stored point to the end, ÷ total path length
    near = torch.cdist(end, w.pts).argmin(1)
    prog = w.s[near] / w.total
    # only count progress for runs that end on the data (else the nearest stored point is meaningless)
    on_data = gaps[-1] < 0.3
    prog = torch.where(torch.isnan(prog) | ~on_data, torch.zeros_like(prog), prog)
    return dict(success=float(((end - tgt).norm(dim=1) < 0.5 * w.R).float().mean()),
                max_gap=float(torch.stack(gaps).max(0).values.median()),
                progress=float(prog.median()))


TS = (2, 4, 6, 10, 20, 40)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=2)
    a = ap.parse_args()
    rows = []
    for kind, wi in itertools.product(("horseshoe", "Y"), range(a.worlds)):
        w = CurveWorld(kind, seed=wi)
        print(f"{kind} world {wi}: train acc {w.train_acc:.3f}", flush=True)
        for method, T, seed in itertools.product(("B", "E2", "E3"), TS, range(a.seeds)):
            rows.append(dict(geometry=kind, world=wi, method=method, T=T, seed=seed, train_acc=w.train_acc,
                             **run(w, method, T, seed)))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "geometries.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    out = [f"# Non-circular geometries\n\nMean over {a.worlds} worlds × {a.seeds} seeds. Cells: success / max_gap ÷ R / progress.\n",
           "| geometry | method | " + " | ".join(f"T={T}" for T in TS) + " |", "|---|---|" + "---|" * len(TS)]
    for kind, method in itertools.product(("horseshoe", "Y"), ("B", "E2", "E3")):
        cells = []
        for T in TS:
            v = [r for r in rows if r["geometry"] == kind and r["method"] == method and r["T"] == T]
            cells.append(" / ".join(f"{statistics.mean(r[k] for r in v):.2f}" for k in ("success", "max_gap", "progress")))
        out.append(f"| {kind} | {method} | " + " | ".join(cells) + " |")
    text = "\n".join(out) + "\n"
    open(os.path.join(HERE, "results", "geometries.md"), "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
