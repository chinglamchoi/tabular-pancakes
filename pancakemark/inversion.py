"""Inversion-error measurement and end-to-end host reports (P1).

Protocol
--------
1. sigma_inv on UNMARKED latents: z ~ N(0, I), x = G(z), zhat = E(x);
   residual r = zhat - z. Measuring on unmarked latents avoids circularity
   (the verifier's noise floor must not be defined through the mark), and
   by undetectability the marked residual law cannot differ detectably --
   we still *check* parity empirically (marked_vs_unmarked_ratio).
2. On-axis component: sigma_onaxis^2 = mean_j Var(r @ w_j). For an
   isotropic residual this equals the per-coordinate variance; reporting
   it separately catches anisotropic encoders.
3. Prediction vs. reality: mu_pred = theory.mu(gamma, beta, sigma_onaxis)
   models the residual as additive Gaussian on-axis noise (exact for
   NoisyEncoderHost; an approximation otherwise). mu_emp is the actual
   mean statistic of marked rows after a full generate->invert round trip
   and requires no model. Both are reported; power planning should use
   mu_emp when they disagree.
"""
from __future__ import annotations

import numpy as np

from .detect import row_scores
from .keys import Key
from .sampler import sample_latents
from . import theory


def inversion_report(
    host,
    key: Key,
    num_rows: int = 2000,
    rng: np.random.Generator | None = None,
) -> dict:
    rng = rng or np.random.default_rng()
    n = key.n
    assert host.latent_dim == n, "key/host latent dims differ"

    # --- unmarked residuals -------------------------------------------------
    z0 = rng.standard_normal((num_rows, n))
    r0 = host.invert(host.generate(z0)) - z0
    sigma_percoord = float(r0.std())
    onaxis = r0 @ key.W                                  # (rows, k)
    sigma_onaxis = float(np.sqrt(np.mean(onaxis**2)))

    # --- marked residual parity (undetectability sanity check) -------------
    z1 = sample_latents(num_rows, key, rng, marked=True)
    r1 = host.invert(host.generate(z1)) - z1
    parity = float(r1.std() / max(sigma_percoord, 1e-300))

    # --- end-to-end detection ----------------------------------------------
    zhat = host.invert(host.generate(z1))
    s = row_scores(zhat, key)
    mu_emp = float(s.mean())
    mu_clean = theory.mu(key.gamma, key.beta)
    mu_pred = theory.mu(key.gamma, key.beta, noise_std=sigma_onaxis)

    return {
        "host": host.name,
        "n": n, "k": key.k, "gamma": key.gamma, "beta": key.beta,
        "num_rows": num_rows,
        "sigma_inv_percoord": sigma_percoord,
        "sigma_onaxis": sigma_onaxis,
        "marked_vs_unmarked_residual_ratio": parity,
        "mu_clean": mu_clean,
        "mu_pred_gaussian_model": mu_pred,
        "mu_emp": mu_emp,
        "rows_needed_alpha1e6_power0.9": (
            theory.rows_needed(mu_emp, alpha=1e-6, power=0.9)
            if mu_emp > 0.02 else np.inf
        ),
    }
