"""Phase-aware Fourier detection.

Score frequency. With the exact mixture of sampler.py and
f = gamma'^2/gamma, a marked coordinate t satisfies

    f * t = (M + delta) + (beta * gamma'/gamma) * eps,   eps ~ N(0,1),

so f*t mod 1 concentrates at delta with *Gaussian* phase jitter of std
sigma0 = beta*gamma'/gamma (cycle units) -- exactly, not asymptotically.
Consequently E[cos(2*pi*(f<z,w> - delta))] = exp(-2*pi^2*sigma0^2) =: mu.

Statistics.
  angles      theta_{ij} = 2*pi*frac(f <z_i, w_j>)
  row score   S_i = (1/k) sum_j cos(theta_{ij} - 2*pi*delta_j)   in [-1, 1]
  global      S   = (1/m) sum_i S_i
Row scores are i.i.d. across rows (rows are independent draws), which is
what the certified null (null.py) relies on; nothing is assumed about
independence across directions within a row.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .keys import Key
from .null import hoeffding_pvalue, mc_key_null_pvalue


def phase_angles(z: np.ndarray, key: Key) -> np.ndarray:
    """(rows, k) array of angles 2*pi*frac(f * <z_i, w_j>)."""
    f = key.score_frequency
    proj = z @ key.W                     # (rows, k)
    return 2.0 * np.pi * np.mod(f * proj, 1.0)


def row_scores(z: np.ndarray, key: Key) -> np.ndarray:
    """(rows,) per-row scores S_i in [-1, 1]."""
    theta = phase_angles(z, key)
    return np.cos(theta - 2.0 * np.pi * key.delta[None, :]).mean(axis=1)


def direction_stats(z: np.ndarray, key: Key) -> dict:
    """Per-direction complex resultants and aligned real parts."""
    theta = phase_angles(z, key)
    C = np.exp(1j * theta).mean(axis=0)                       # (k,)
    T = np.real(np.exp(-2j * np.pi * key.delta) * C)          # (k,)
    delta_hat = np.mod(np.angle(C) / (2.0 * np.pi), 1.0)      # (k,)
    return {"C": C, "T": T, "delta_hat": delta_hat}


def global_stat(z: np.ndarray, key: Key) -> float:
    return float(row_scores(z, key).mean())


@dataclass
class DetectionResult:
    stat: float
    pvalue: float
    method: str
    num_rows: int
    row_scores: np.ndarray
    per_direction: dict

    def reject(self, alpha: float) -> bool:
        return self.pvalue <= alpha


def detect(
    z: np.ndarray,
    key: Key,
    method: str = "mc",
    mc_draws: int = 999,
    rng: np.random.Generator | None = None,
) -> DetectionResult:
    """Test H0: `z` was produced independently of `key`.

    method="mc"        exact Monte-Carlo key-rerandomization p-value
                       (assumption-free given data independent of key).
    method="hoeffding" analytic bound exp(-m * S^2 / 2) on the row-score
                       mean; conservative up to the (exponentially small,
                       quantified) null-mean bias -- see null.py.
    """
    s = row_scores(z, key)
    S = float(s.mean())
    if method == "hoeffding":
        p = hoeffding_pvalue(S, num_rows=s.size)
    elif method == "mc":
        p = mc_key_null_pvalue(z, key, S_obs=S, draws=mc_draws, rng=rng)
    else:
        raise ValueError(f"unknown method {method!r}")
    return DetectionResult(
        stat=S,
        pvalue=p,
        method=method,
        num_rows=s.size,
        row_scores=s,
        per_direction=direction_stats(z, key),
    )
