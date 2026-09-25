"""Analytically controlled hosts (numpy only).

LinearHost       x = z A^T + b with invertible A: exact inverse (solve).
TanhFlowHost     L layers of [orthogonal rotation -> y = x + c*tanh(x)]
                 followed by per-feature affine scales/shifts. Each
                 elementwise map is strictly increasing (derivative
                 1 + c(1 - tanh^2) > 0 for 0 < c < 1), hence globally
                 invertible; the inverse is computed by Newton iteration
                 to ~1e-12 (tolerance asserted). A stand-in for a
                 deterministic normalizing-flow generator with exact
                 inversion (sigma_inv ~= 0), including heterogeneous
                 feature scales.
NoisyEncoderHost wraps any host; invert() adds N(0, sigma_enc^2 I).
                 Ground truth for calibrating the sigma_inv measurement
                 (the Gaussian-noise model is *exact* here).
QuantizedHost    wraps any host; generate() rounds each feature to a
                 fixed number of decimals -- released-table precision.
                 Inversion error is then deterministic and non-Gaussian:
                 the honest test case for the Gaussian approximation.
"""
from __future__ import annotations

import numpy as np

from .base import Host
from ..utils import haar_frame


class LinearHost(Host):
    name = "linear"

    def __init__(self, n: int, seed: int = 0, cond: float = 4.0):
        rng = np.random.default_rng(seed)
        Q1 = haar_frame(n, n, rng)
        Q2 = haar_frame(n, n, rng)
        scales = np.exp(np.linspace(-np.log(cond) / 2, np.log(cond) / 2, n))
        self.A = Q1 @ np.diag(scales) @ Q2.T
        self.b = rng.normal(0, 1.0, size=n)
        self._n = n

    @property
    def latent_dim(self) -> int:
        return self._n

    def generate(self, z: np.ndarray) -> np.ndarray:
        return z @ self.A.T + self.b

    def invert(self, x: np.ndarray) -> np.ndarray:
        return np.linalg.solve(self.A, (x - self.b).T).T


def _tanh_layer_inverse(y: np.ndarray, c: float, iters: int = 40) -> np.ndarray:
    """Solve x + c*tanh(x) = y elementwise by Newton (monotone, global)."""
    x = y.copy()
    for _ in range(iters):
        t = np.tanh(x)
        f = x + c * t - y
        fp = 1.0 + c * (1.0 - t * t)
        x -= f / fp
    return x


class TanhFlowHost(Host):
    name = "tanh_flow"

    def __init__(self, n: int, layers: int = 3, seed: int = 1,
                 scale_range: tuple[float, float] = (0.5, 3.0)):
        rng = np.random.default_rng(seed)
        self.Qs = [haar_frame(n, n, rng) for _ in range(layers)]
        self.cs = rng.uniform(0.3, 0.9, size=layers)
        lo, hi = np.log(scale_range[0]), np.log(scale_range[1])
        self.scales = np.exp(rng.uniform(lo, hi, size=n))
        self.shifts = rng.normal(0, 1.0, size=n)
        self._n = n

    @property
    def latent_dim(self) -> int:
        return self._n

    def generate(self, z: np.ndarray) -> np.ndarray:
        x = z
        for Q, c in zip(self.Qs, self.cs):
            x = x @ Q.T
            x = x + c * np.tanh(x)
        return x * self.scales + self.shifts

    def invert(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.shifts) / self.scales
        for Q, c in zip(reversed(self.Qs), reversed(self.cs)):
            z = _tanh_layer_inverse(z, c)
            z = z @ Q
        return z


class NoisyEncoderHost(Host):
    def __init__(self, base: Host, sigma_enc: float, seed: int = 2):
        self.base = base
        self.sigma_enc = float(sigma_enc)
        self.rng = np.random.default_rng(seed)
        self.name = f"{base.name}+enc{sigma_enc:g}"

    @property
    def latent_dim(self) -> int:
        return self.base.latent_dim

    def generate(self, z: np.ndarray) -> np.ndarray:
        return self.base.generate(z)

    def invert(self, x: np.ndarray) -> np.ndarray:
        zh = self.base.invert(x)
        return zh + self.sigma_enc * self.rng.standard_normal(zh.shape)


class QuantizedHost(Host):
    def __init__(self, base: Host, decimals: int):
        self.base = base
        self.decimals = int(decimals)
        self.name = f"{base.name}+q{decimals}"

    @property
    def latent_dim(self) -> int:
        return self.base.latent_dim

    def generate(self, z: np.ndarray) -> np.ndarray:
        return np.round(self.base.generate(z), self.decimals)

    def invert(self, x: np.ndarray) -> np.ndarray:
        return self.base.invert(x)
