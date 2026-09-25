"""Key generation for Tabular Pancakes.

A key is (W, gamma, beta, delta, kappa):
  W     : n x k orthonormal secret frame, Haar-distributed (QR with sign fix).
  gamma : inverse pancake spacing (comb frequency along each secret direction).
  beta  : relative layer thickness (must be >= 1/poly(n); noiseless CLWE is
          broken by LLL [Song-Zadik-Bruna 2021], so beta == 0 is forbidden).
  delta : per-direction phases in [0,1)^k (the payload channel).
  kappa : 32-byte PRF seed. PRF instantiated as HMAC-SHA256, a standard
          PRF assumption; used for deriving per-scope phases/blocks in P1+.

Serialization is plain JSON-able dicts (no pickle), so keys can be escrowed
and moved to a cluster without trust issues.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass, field

import numpy as np

from .utils import haar_frame, rng_from_seed


@dataclass(frozen=True)
class Key:
    W: np.ndarray            # (n, k) orthonormal columns
    gamma: float
    beta: float
    delta: np.ndarray        # (k,) phases in [0, 1)
    kappa: bytes             # PRF seed

    @property
    def n(self) -> int:
        return self.W.shape[0]

    @property
    def k(self) -> int:
        return self.W.shape[1]

    @property
    def gamma_prime(self) -> float:
        """gamma' = sqrt(gamma^2 + beta^2)."""
        return float(np.hypot(self.gamma, self.beta))

    @property
    def score_frequency(self) -> float:
        """f = gamma'^2 / gamma: the frequency at which f*<z,w> mod 1
        concentrates on delta (exact; see package docstring)."""
        return self.gamma_prime**2 / self.gamma

    def to_dict(self) -> dict:
        return {
            "W": self.W.tolist(),
            "gamma": self.gamma,
            "beta": self.beta,
            "delta": self.delta.tolist(),
            "kappa": self.kappa.hex(),
        }

    @staticmethod
    def from_dict(d: dict) -> "Key":
        return Key(
            W=np.asarray(d["W"], dtype=float),
            gamma=float(d["gamma"]),
            beta=float(d["beta"]),
            delta=np.asarray(d["delta"], dtype=float),
            kappa=bytes.fromhex(d["kappa"]),
        )


def keygen(
    n: int,
    k: int = 1,
    gamma: float = 2.0,
    beta: float = 0.05,
    delta: np.ndarray | None = None,
    seed: int | None = None,
) -> Key:
    """Generate a key with a Haar-random orthonormal k-frame in R^n.

    `seed` is for reproducible experiments only; a deployment draws W and
    kappa from OS entropy (seed=None).
    """
    if not (0 < k <= n):
        raise ValueError("need 0 < k <= n")
    if beta <= 0:
        raise ValueError("beta must be > 0 (beta=0 is LLL-breakable)")
    if gamma <= 0:
        raise ValueError("gamma must be > 0")
    if seed is None:
        kappa = secrets.token_bytes(32)
        rng = np.random.default_rng()
    else:
        kappa = hashlib.sha256(b"kappa|" + int(seed).to_bytes(8, "big")).digest()
        rng = rng_from_seed(seed)
    W = haar_frame(n, k, rng)
    if delta is None:
        delta = np.zeros(k)
    delta = np.mod(np.asarray(delta, dtype=float), 1.0)
    if delta.shape != (k,):
        raise ValueError("delta must have shape (k,)")
    return Key(W=W, gamma=float(gamma), beta=float(beta), delta=delta, kappa=kappa)


def prf_uniform(kappa: bytes, *labels: bytes | str | int) -> float:
    """PRF_kappa(labels) -> uniform float in [0, 1).

    HMAC-SHA256 truncated to 53 bits (exactly representable in a float).
    Under the standard PRF assumption on HMAC-SHA256, outputs across distinct
    label tuples are computationally indistinguishable from i.i.d. uniforms.
    """
    h = hmac.new(kappa, digestmod=hashlib.sha256)
    for lab in labels:
        if isinstance(lab, int):
            lab = lab.to_bytes(8, "big", signed=True)
        elif isinstance(lab, str):
            lab = lab.encode()
        h.update(len(lab).to_bytes(4, "big") + lab)
    digest = h.digest()
    x = int.from_bytes(digest[:8], "big") >> 11  # top 53 bits
    return x / float(1 << 53)
