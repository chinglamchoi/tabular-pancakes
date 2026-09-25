"""Null distributions and certified p-values.

Two routes, complementary by design:

1) Monte-Carlo key-rerandomization null (exact, assumption-free).
   If the data are independent of the key (the false-accusation scenario),
   then (S(W_obs), S(W_1), ..., S(W_B)) are exchangeable when W_1..W_B are
   fresh i.i.d. Haar frames with fresh phases, so

       p = (1 + #{b : S(W_b) >= S(W_obs)}) / (B + 1)

   is a valid p-value (P(p <= alpha) <= alpha exactly; Phipson & Smyth 2010).
   This is the certificate to use in an adjudication: it assumes nothing
   about the data distribution -- only that it does not depend on the key,
   which is precisely what is being tested.

2) Hoeffding bound (analytic, transparent, approximately valid).
   Row scores S_i are i.i.d. in [-1, 1] across rows, so IF E[S_i] <= b0
   under the null, Hoeffding gives
       P(S_bar >= s) <= exp(-m * (s - b0)^2 / 2)  for s > b0.
   The null mean is not exactly 0 for a *fixed* dataset: for a fixed row z
   and Haar-random direction w, <z,w> is ||z||/sqrt(n) times a coordinate of
   a random unit vector, and

       |E_w cos(2 pi f <z,w> - phi)| <= |E_w e^{2 pi i f <z,w>}| =: b(z),

   which for large n is ~ exp(-2 pi^2 f^2 ||z||^2 / n) -- astronomically
   small for f >~ 1 and ||z||^2 ~ n. We expose `null_bias_estimate` to
   compute this per dataset and fold it into the bound; the MC null needs
   no such estimate and is the one we certify with.
"""
from __future__ import annotations

import numpy as np

from .utils import haar_frame


def hoeffding_pvalue(S: float, num_rows: int, bias: float = 0.0) -> float:
    """exp(-m (S - bias)^2 / 2) for S > bias, else 1. Valid for i.i.d.
    row scores in [-1,1] with null mean <= bias."""
    excess = S - bias
    if excess <= 0:
        return 1.0
    return float(min(1.0, np.exp(-0.5 * num_rows * excess**2)))


def null_bias_estimate(z: np.ndarray, f: float) -> float:
    """Upper estimate of the per-row null-mean bias b(z), averaged over rows.

    Uses the large-n Gaussian approximation for <z,w> under Haar w:
    <z,w> ~ N(0, ||z||^2/n) (error O(1/n) in the characteristic function),
    so b(z) ~= exp(-2 pi^2 f^2 ||z||^2 / n). Reported for transparency;
    the MC null is exact and does not use this quantity.
    """
    n = z.shape[1]
    norms2 = np.einsum("ij,ij->i", z, z)
    return float(np.mean(np.exp(-2.0 * np.pi**2 * f**2 * norms2 / n)))


def mc_key_null_pvalue(
    z: np.ndarray,
    key,
    S_obs: float | None = None,
    draws: int = 999,
    rng: np.random.Generator | None = None,
) -> float:
    """Exact MC p-value by re-randomizing the key (frame + phases).

    Complexity: draws x (rows x n x k) matmuls; vectorized over draws in
    blocks to bound memory.
    """
    from .detect import row_scores  # local import to avoid cycle

    rng = rng or np.random.default_rng()
    if S_obs is None:
        S_obs = float(row_scores(z, key).mean())
    n, k, f = key.n, key.k, key.score_frequency
    count = 0
    block = max(1, int(2e7 // (z.shape[0] * k + n * k)))
    done = 0
    while done < draws:
        b = min(block, draws - done)
        for _ in range(b):
            Wb = haar_frame(n, k, rng)
            db = rng.uniform(0.0, 1.0, size=k)
            theta = 2.0 * np.pi * np.mod(f * (z @ Wb), 1.0)
            Sb = float(np.cos(theta - 2.0 * np.pi * db[None, :]).mean())
            if Sb >= S_obs:
                count += 1
        done += b
    return (1 + count) / (draws + 1)
