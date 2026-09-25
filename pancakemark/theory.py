"""Closed-form theory used to overlay predictions on P0 experiments.

All formulas are exact under the sampler of sampler.py unless noted.
"""
from __future__ import annotations

import numpy as np


def gamma_prime(gamma: float, beta: float) -> float:
    return float(np.hypot(gamma, beta))


def score_frequency(gamma: float, beta: float) -> float:
    return gamma_prime(gamma, beta) ** 2 / gamma


def phase_jitter_std(gamma: float, beta: float, noise_std: float = 0.0) -> float:
    """Std (cycle units) of the score phase f*t mod 1 around delta.

    Intrinsic jitter beta*gamma'/gamma (exact) plus f*sigma for additive
    on-axis Gaussian noise of std `noise_std` (exact: Gaussians add).
    """
    gp = gamma_prime(gamma, beta)
    f = gp**2 / gamma
    return float(np.hypot(beta * gp / gamma, f * noise_std))


def mu(gamma: float, beta: float, noise_std: float = 0.0) -> float:
    """E[cos(2 pi (f<z,w> - delta))] for a marked coordinate.

    Exact: conditional on the layer, the phase is delta + N(0, v) modulo 1
    with v = phase_jitter_std^2, and E cos(2 pi N(0,v)) = exp(-2 pi^2 v)
    (Gaussian characteristic function; the wrap-around changes nothing
    because cos is 1-periodic in the phase).
    """
    v = phase_jitter_std(gamma, beta, noise_std) ** 2
    return float(np.exp(-2.0 * np.pi**2 * v))


def rows_needed(
    mu_val: float, alpha: float, power: float = 0.9, k: int = 1
) -> int:
    """Rows m sufficient for level-alpha detection at the given power,
    via the Hoeffding certificate on row scores (bounded in [-1,1]).

    Threshold s* = sqrt(2 ln(1/alpha) / m); a sub-Gaussian (bounded) row
    score with mean mu_k needs  m >= 2 (sqrt(ln 1/alpha) + sqrt(ln 1/eta))^2
    * (1/mu_k)^2 * ... we use the clean sufficient condition

        sqrt(m) * mu_k >= sqrt(2 ln 1/alpha) + sqrt(2 ln 1/(1-power)),

    where mu_k = mu_val (each row score averages k directions; its mean is
    mu_val regardless of k, while its variance shrinks with k -- so this
    bound is conservative in k, as a certificate should be).
    """
    eta = 1.0 - power
    m = 2.0 * (np.sqrt(np.log(1 / alpha)) + np.sqrt(np.log(1 / eta))) ** 2 / mu_val**2
    return int(np.ceil(m))


def phase_estimate_std(mu_val: float, num_rows: int) -> float:
    """Delta-method std (cycle units) of the resultant-angle phase estimate:
    std(delta_hat) ~ sqrt((1 - mu^2)/2m) / (2 pi mu). Approximation
    (first-order delta method), accurate once m * mu^2 >> 1."""
    return float(
        np.sqrt((1.0 - mu_val**2) / (2.0 * num_rows)) / (2.0 * np.pi * mu_val)
    )
