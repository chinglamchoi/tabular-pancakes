"""Exact pancake (hCLWE-style) sampling in the standard-normal convention.

Target 1-D marginal along a secret direction (phase delta in [0,1)):

    p(t) ∝ exp(-t^2/2) * sum_{m in Z} exp(-(m + delta - gamma*t)^2 / (2 beta^2))

Completing the square in t (exact algebra, no approximation): with
gamma' = sqrt(gamma^2 + beta^2),

    t^2 + (gamma*t - (m+delta))^2/beta^2
      = (gamma'^2/beta^2) * (t - (m+delta)*gamma/gamma'^2)^2 + (m+delta)^2/gamma'^2,

so p(t) is exactly the mixture

    M ~ P(M=m) ∝ exp(-(m+delta)^2 / (2 gamma'^2)),
    t | M=m ~ N( (m+delta)*gamma/gamma'^2 ,  (beta/gamma')^2 ).

We sample the discrete Gaussian M by enumerating m in [-L, L] with
L = ceil(TAIL_SIGMAS * gamma') + 1; the truncated mass is bounded by
2*exp(-TAIL_SIGMAS^2/2) / (normalizer), i.e. < 1e-13 at TAIL_SIGMAS = 8 --
exact to double precision for all practical purposes.

The full latent is z = sum_j t_j w_j + (I - W W^T) g with g ~ N(0, I_n):
exactly standard Gaussian on the orthogonal complement, independent across
the k secret directions (the "baguette" product construction).
"""
from __future__ import annotations

import numpy as np

from .keys import Key

TAIL_SIGMAS = 8.0


def _layer_distribution(gamma: float, beta: float, delta: float):
    """Support and probabilities of the discrete-Gaussian layer index M."""
    gp = float(np.hypot(gamma, beta))
    L = int(np.ceil(TAIL_SIGMAS * gp)) + 1
    m = np.arange(-L, L + 1)
    logits = -((m + delta) ** 2) / (2.0 * gp**2)
    p = np.exp(logits - logits.max())
    p /= p.sum()
    return m, p, gp


def sample_pancake_1d(
    size: int,
    gamma: float,
    beta: float,
    delta: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw `size` samples of the exact 1-D pancake marginal."""
    rng = rng or np.random.default_rng()
    m, p, gp = _layer_distribution(gamma, beta, delta)
    idx = rng.choice(m.size, size=size, p=p)
    centers = (m[idx] + delta) * gamma / gp**2
    return centers + (beta / gp) * rng.standard_normal(size)


def pancake_density_1d(
    t: np.ndarray, gamma: float, beta: float, delta: float = 0.0
) -> np.ndarray:
    """Normalized 1-D pancake density (mixture form; exact)."""
    m, p, gp = _layer_distribution(gamma, beta, delta)
    centers = (m + delta) * gamma / gp**2
    s = beta / gp
    t = np.asarray(t, dtype=float)[..., None]
    comps = np.exp(-((t - centers) ** 2) / (2 * s**2)) / (s * np.sqrt(2 * np.pi))
    return (comps * p).sum(axis=-1)


def sample_latents(
    num_rows: int,
    key: Key,
    rng: np.random.Generator | None = None,
    marked: bool = True,
) -> np.ndarray:
    """Sample `num_rows` latents in R^n.

    marked=True : pancake law along each secret direction (with its phase),
                  exact N(0, I) on the orthogonal complement.
    marked=False: exact N(0, I_n) (the null / unmarked prior).
    """
    rng = rng or np.random.default_rng()
    g = rng.standard_normal((num_rows, key.n))
    if not marked:
        return g
    W = key.W                                   # (n, k)
    # remove the Gaussian component along W, then add pancake coordinates
    coeff = g @ W                               # (rows, k)
    z = g - coeff @ W.T
    t = np.empty((num_rows, key.k))
    for j in range(key.k):
        t[:, j] = sample_pancake_1d(
            num_rows, key.gamma, key.beta, float(key.delta[j]), rng
        )
    return z + t @ W.T
