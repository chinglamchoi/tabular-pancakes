"""Keyed categorical channel: derandomized Gumbel-max sampling synced by
pancake indices.

Sampling. For a categorical cell with model conditionals p_1..p_C, draw
  category = argmax_c [ log p_c + g_c ],   g_c = -log(-log u_c),
  u_c = PRF_kappa(scope, sync_row, c)  in [0,1).
Gumbel-max is a faithful sampler, so per-cell conditionals (hence all
joints the model defines) are unchanged; under the PRF assumption the
u's are computationally indistinguishable from fresh uniforms, so the
undetectability reduction covers this channel too.

Sync token. sync_row = the vector of pancake layer indices
round(f * <z_row, w_j>), j = 1..r  -- secret without W, recoverable by the
verifier after inversion, and stable under perturbations smaller than the
inter-layer gap (a locality-sensitive seed where a hash of raw values
would shatter).

Detection WITHOUT model probabilities. Let v = u_{c_observed} recomputed
from the PRF. If the table was produced independently of kappa (the null),
v is exactly U(0,1) under the PRF assumption -- regardless of the unknown
p's -- so the per-cell scores 2(v - 1/2) in [-1,1] admit the same
Hoeffding / MC certificates as the numeric channel. Under keyed sampling
v is stochastically large, with more signal in higher-entropy cells
(deterministic cells carry none, as they must).
"""
from __future__ import annotations

import numpy as np

from .keys import Key, prf_uniform
from .detect import phase_angles


def pancake_indices(z: np.ndarray, key: Key, r: int | None = None) -> np.ndarray:
    """(rows, r) integer layer indices round(f * <z, w_j>)."""
    r = r or key.k
    f = key.score_frequency
    proj = (z @ key.W[:, :r]) * f - key.delta[None, :r]
    return np.rint(proj).astype(np.int64)


def _u_matrix(sync: np.ndarray, num_cats: int, kappa: bytes,
              scope: str) -> np.ndarray:
    """(rows, C) PRF uniforms keyed by (scope, sync_row, c)."""
    m = sync.shape[0]
    U = np.empty((m, num_cats))
    for i in range(m):
        s = ",".join(map(str, sync[i]))
        for c in range(num_cats):
            U[i, c] = prf_uniform(kappa, scope, s, c)
    return U


def sample_categorical(probs: np.ndarray, z: np.ndarray, key: Key,
                       scope: str = "cat0", r: int | None = None) -> np.ndarray:
    """Keyed Gumbel-max draw per row. probs: (rows, C) conditionals."""
    probs = np.asarray(probs, dtype=float)
    sync = pancake_indices(z, key, r)
    U = _u_matrix(sync, probs.shape[1], key.kappa, scope)
    g = -np.log(-np.log(np.clip(U, 1e-15, 1 - 1e-15)))
    return np.argmax(np.log(np.clip(probs, 1e-300, None)) + g, axis=1)


def categorical_scores(categories: np.ndarray, z_hat: np.ndarray, key: Key,
                       num_cats: int, scope: str = "cat0",
                       r: int | None = None) -> np.ndarray:
    """Per-row scores in [-1, 1]; exactly mean-0 under the null (v ~ U(0,1))."""
    sync = pancake_indices(z_hat, key, r)
    U = _u_matrix(sync, num_cats, key.kappa, scope)
    v = U[np.arange(categories.size), np.asarray(categories, dtype=int)]
    return 2.0 * (v - 0.5)


def detect_categorical(categories: np.ndarray, z_hat: np.ndarray, key: Key,
                       num_cats: int, scope: str = "cat0",
                       r: int | None = None) -> dict:
    """Hoeffding certificate on the mean categorical score."""
    s = categorical_scores(categories, z_hat, key, num_cats, scope, r)
    S = float(s.mean())
    p = float(min(1.0, np.exp(-0.5 * s.size * max(S, 0.0) ** 2)))
    return {"stat": S, "pvalue": p, "num_cells": int(s.size),
            "row_scores": s}
