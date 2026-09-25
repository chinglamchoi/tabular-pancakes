"""Shared numerical utilities."""
from __future__ import annotations

import numpy as np


def rng_from_seed(seed) -> np.random.Generator:
    return np.random.default_rng(seed)


def haar_frame(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-distributed orthonormal k-frame in R^n, as an (n, k) matrix.

    QR of a Gaussian matrix with the sign of R's diagonal fixed to +1;
    this is the standard construction whose Q is exactly Haar on the
    Stiefel manifold V_k(R^n) (Mezzadri 2007).
    """
    G = rng.standard_normal((n, k))
    Q, R = np.linalg.qr(G)
    d = np.sign(np.diagonal(R))
    d[d == 0] = 1.0
    return Q * d[np.newaxis, :]
