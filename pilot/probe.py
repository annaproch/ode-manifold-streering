"""Independent degree-two TensorSketch logistic probe with analytic gradients.

Same feature family and regularization as ODESteer, not copied source. Hashes use
NumPy's seeded generator, so fitted seeds are not bit-for-bit upstream seeds.
All features and derivatives run on CPU to avoid backend-specific FFT behavior.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


class Probe:
    def __init__(self, dimension, features=8000, seed=0, gamma=0.1, coef0=1.0):
        self.dimension, self.features = dimension, features
        self.gamma, self.coef0 = gamma, coef0
        if features < 2 or gamma <= 0 or coef0 < 0:
            raise ValueError("Invalid sketch parameters")
        rng = np.random.default_rng(seed)
        extended = dimension + int(coef0 > 0)
        self.indices = rng.integers(features, size=(2, extended))
        self.signs = rng.choice([-1.0, 1.0], size=(2, extended))
        self.coef = np.zeros(features)
        self.intercept, self.prior = 0.0, 0.5

    def _sketches(self, x):
        x = np.atleast_2d(x).astype(np.float64, copy=False)
        normalized = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        extended = np.sqrt(self.gamma) * normalized
        if self.coef0:
            extended = np.column_stack([extended, np.full(len(x), np.sqrt(self.coef0))])
        sketches = np.zeros((2, len(x), self.features))
        rows = np.arange(len(x))[:, None]
        for degree in range(2):
            np.add.at(sketches[degree], (rows, self.indices[degree][None]), extended * self.signs[degree])
        return sketches

    def transform(self, x):
        spectra = np.fft.rfft(self._sketches(x), axis=-1)
        return np.fft.irfft(spectra[0] * spectra[1], n=self.features, axis=-1)

    def fit(self, x, y, c=1.0):
        y = np.asarray(y, dtype=int)
        if len(np.unique(y)) != 2 or min(np.bincount(y)) < 2:
            raise ValueError("Probe needs at least two positive and two negative training examples")
        self.prior = float(np.mean(y))
        # Chunk the transform so its intermediate FFT arrays do not scale with N.
        phi = np.concatenate([self.transform(x[i:i+64]) for i in range(0, len(x), 64)])
        lr = LogisticRegression(C=c, max_iter=1000).fit(phi, y)
        self.coef = lr.coef_[0].astype(np.float64)
        self.intercept = float(lr.intercept_[0])
        return self

    def logits(self, x):
        return self.transform(x) @ self.coef + self.intercept

    def log_dre(self, x):
        return self.logits(x) + np.log((1-self.prior) / self.prior)

    def grad(self, x):
        original = np.asarray(x)
        x = np.atleast_2d(x).astype(np.float64, copy=False)
        sketches = self._sketches(x)
        spectra = np.fft.rfft(sketches, axis=-1)
        coefficients = np.fft.rfft(self.coef)
        g = np.zeros_like(x)
        for degree in range(2):
            # Adjoint of circular convolution: cross-correlate w with other sketch.
            adjoint = np.fft.irfft(coefficients[None] * spectra[1-degree].conj(),
                                    n=self.features, axis=-1)
            g += adjoint[:, self.indices[degree, :self.dimension]] * self.signs[degree, :self.dimension]
        g *= np.sqrt(self.gamma)
        norms = np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        u = x / norms
        g = (g - u * np.sum(u*g, axis=1, keepdims=True)) / norms
        return g[0] if original.ndim == 1 else g

    def ode_direction(self, x):
        """Match upstream's prior-scaled gradient and epsilon normalization."""
        gradient = self.grad(x) * ((1-self.prior)/self.prior)
        return gradient/(np.linalg.norm(gradient, axis=-1, keepdims=True)+1e-10)

    def diagnostics(self, x, y):
        logits = np.concatenate([self.logits(x[i:i+64]) for i in range(0, len(x), 64)])
        y = np.asarray(y, dtype=int)
        posterior = expit(logits)
        ratio_positive = logits + np.log((1-self.prior)/self.prior) > 0
        return {"count": len(y), "positive_fraction": float(y.mean()),
                "posterior_accuracy": float(np.mean((posterior > .5) == y)),
                "balanced_accuracy": float(balanced_accuracy_score(y, posterior > .5)) if len(np.unique(y)) == 2 else None,
                "roc_auc": float(roc_auc_score(y, logits)) if len(np.unique(y)) == 2 else None,
                "brier": float(np.mean((posterior-y)**2)),
                "posterior_positive_fraction": float(np.mean(posterior > .5)),
                "density_ratio_positive_fraction": float(ratio_positive.mean()),
                "density_ratio_recall": float(ratio_positive[y == 1].mean()) if np.any(y == 1) else None,
                "density_ratio_false_positive": float(ratio_positive[y == 0].mean()) if np.any(y == 0) else None}

    def arrays(self):
        return {"probe_indices": self.indices, "probe_signs": self.signs,
                "probe_coef": self.coef,
                "probe_scalars": np.array([self.dimension, self.features, self.gamma, self.coef0,
                                            self.intercept, self.prior])}

    @classmethod
    def restore(cls, arrays):
        d, f, gamma, coef0, intercept, prior = arrays["probe_scalars"]
        obj = cls(int(d), int(f), gamma=gamma, coef0=coef0)
        obj.indices, obj.signs = arrays["probe_indices"], arrays["probe_signs"]
        obj.coef, obj.intercept, obj.prior = arrays["probe_coef"], float(intercept), float(prior)
        return obj
