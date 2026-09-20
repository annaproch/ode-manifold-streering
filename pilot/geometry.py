"""Conditional local geometry fitted on training points; gates use validation."""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.special import logsumexp
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KernelDensity, NearestNeighbors

from .data import ANGLES


class Geometry:
    def __init__(self, arrays, metadata):
        self.mu = arrays["mu"]
        self.u = arrays["u"]
        self.x = arrays["train_x"]
        self.z = arrays["train_z"]
        self.labels = arrays["train_labels"].astype(int)
        self.hues = arrays["train_hues"]
        self.graph_score = arrays.get("graph_score")
        self.metadata = metadata
        self.k = metadata["neighbors"]
        self.m = metadata["tangent_dim"]
        self.bandwidth = metadata["bandwidth"]
        self.nn = NearestNeighbors(n_neighbors=self.k).fit(self.z)
        self.full_nn = NearestNeighbors(n_neighbors=1).fit(self.x)
        self.spline = None
        if all(np.any(self.labels == i) for i in range(6)):
            centers = np.stack([self.z[self.labels == i].mean(0) for i in range(6)])
            self.spline = CubicSpline(np.r_[ANGLES, 360.0], np.vstack([centers, centers[0]]), bc_type="periodic")

    @classmethod
    def fit(cls, x, labels, hues, validation_x, dimension, settings, seed, graph_targets=None):
        if min(len(x)-1, x.shape[1]) < dimension or len(x) <= settings["neighbors"]:
            raise ValueError(f"Not enough primary-slice training points ({len(x)}) for k={dimension}/neighbors={settings['neighbors']}")
        if len(validation_x) < 5:
            raise ValueError("Need at least five primary validation points for support calibration")
        pca = PCA(n_components=dimension, svd_solver="full").fit(x.astype(np.float64))
        z = pca.transform(x)
        scale = float(np.median(NearestNeighbors(n_neighbors=2).fit(z).kneighbors(z)[0][:, 1]))
        scale = max(scale, 1e-6)
        cv = KFold(n_splits=min(5, len(x)//10), shuffle=True, random_state=seed)
        search = GridSearchCV(KernelDensity(), {"bandwidth": [scale*v for v in settings["bandwidth_factors"]]}, cv=cv).fit(z)
        meta = {"neighbors": settings["neighbors"], "tangent_dim": settings["tangent_dim"],
                "bandwidth": float(search.best_params_["bandwidth"]),
                "variance_ratio": pca.explained_variance_ratio_.tolist(),
                "gate_quantile": settings["gate_quantile"]}
        arrays = {"mu": pca.mean_, "u": pca.components_.T, "train_x": x.astype(np.float64),
                  "train_z": z, "train_labels": np.asarray(labels), "train_hues": np.asarray(hues)}
        obj = cls(arrays, meta)
        projected, full = obj.distances(validation_x)
        meta["projected_limit"] = max(float(np.quantile(projected, settings["gate_quantile"])), 1e-8)
        meta["full_limit"] = max(float(np.quantile(full, settings["gate_quantile"])), 1e-8)
        meta["typical_projected_distance"] = max(float(np.median(projected)), 1e-8)
        meta["typical_full_distance"] = max(float(np.median(full)), 1e-8)
        if graph_targets is not None:
            if not np.any(graph_targets):
                raise ValueError("Graph target set is empty")
            count = min(settings["graph_neighbors"] + 1, len(z))
            distances, indices = NearestNeighbors(n_neighbors=count).fit(z).kneighbors(z)
            rows = np.repeat(np.arange(len(z)), count-1)
            # Explicit undirected union; positive floor preserves coincident edges.
            graph = csr_matrix((np.maximum(distances[:, 1:].ravel(), 1e-12),
                                (rows, indices[:, 1:].ravel())), shape=(len(z), len(z)))
            graph = graph.maximum(graph.T)
            distances = dijkstra(graph, directed=False, indices=np.flatnonzero(graph_targets), min_only=True)
            obj.graph_score = -distances
            meta["graph_unreachable_fraction"] = float(np.mean(~np.isfinite(distances)))
            meta["graph_target_count"] = int(np.sum(graph_targets))
        return obj

    def arrays(self):
        out = {"mu": self.mu, "u": self.u, "train_x": self.x, "train_z": self.z,
               "train_labels": self.labels, "train_hues": self.hues}
        if self.graph_score is not None:
            out["graph_score"] = self.graph_score
        return out

    def coordinates(self, x):
        return (np.asarray(x)-self.mu) @ self.u

    def distances(self, x):
        x = np.atleast_2d(x)
        projected = self.nn.kneighbors(self.coordinates(x), n_neighbors=1)[0][:, 0]
        full = self.full_nn.kneighbors(x, n_neighbors=1)[0][:, 0]
        return projected, full

    def eligible(self, x, multiplier=1.0):
        projected, full = self.distances(x)
        return (projected <= multiplier*self.metadata["projected_limit"]) & (full <= multiplier*self.metadata["full_limit"])

    def basis(self, z):
        indices = self.nn.kneighbors(np.atleast_2d(z), return_distance=False)[0]
        points = self.z[indices]
        _, _, vh = np.linalg.svd(points-points.mean(0), full_matrices=False)
        return vh[:self.m]

    @staticmethod
    def project(v, basis):
        return (v @ basis.T) @ basis

    def mean_shift(self, z):
        delta = self.z-z
        log_w = -np.sum(delta*delta, axis=1)/(2*self.bandwidth**2)
        weights = np.exp(log_w-logsumexp(log_w))
        return weights @ delta

    def correction(self, z, plain=False):
        ms = self.mean_shift(z)
        return ms if plain else ms-self.project(ms, self.basis(z))

    def regression_x(self, z):
        distance, index = self.nn.kneighbors(np.atleast_2d(z))
        log_w = -(distance[0]/self.bandwidth)**2 / 2
        w = np.exp(log_w-logsumexp(log_w))
        return w @ self.x[index[0]]

    def graph_direction(self, z):
        if self.graph_score is None:
            raise ValueError("E3 was not enabled when fitting this artifact")
        indices = self.nn.kneighbors(np.atleast_2d(z), return_distance=False)[0]
        if not np.isfinite(self.graph_score[indices]).all():
            return np.zeros_like(z)
        points, score = self.z[indices], self.graph_score[indices]
        basis = self.basis(z)
        local = (points-points.mean(0)) @ basis.T
        # Solve in the tangent coordinates, with a scale-relative ridge.
        ridge = max(float(np.trace(local.T @ local))*1e-6, 1e-10)
        slope = np.linalg.solve(local.T @ local + ridge*np.eye(self.m), local.T @ (score-score.mean()))
        return slope @ basis

    def spline_route(self, z, target):
        if self.spline is None:
            raise ValueError("Spline needs at least one training example in each of six chromatic classes")
        grid = np.linspace(0, 360, 721)[:-1]
        start = grid[np.argmin(np.linalg.norm(self.spline(grid)-z, axis=1))]
        end = ANGLES[target]
        delta = (end-start) % 360
        routes = []
        for change in (delta, delta-360):
            theta = np.linspace(start, start+change, 257)
            route = np.vstack([z, self.spline(theta % 360)])
            length = np.linalg.norm(np.diff(route, axis=0), axis=1).sum()
            routes.append((length, route))
        return min(routes, key=lambda item: item[0])[1]
