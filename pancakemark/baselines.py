"""Baseline watermark constructions (latent-space), ported from the TabWak
repo (watermark/sample.py, watermark/detection.py) for controlled
comparison: same host, same rows, different watermark.

Implemented (vectorized, faithful to their code):
  gaussian_shading   sign-fixed half-Gaussians per coordinate, fixed key
                     seed (their 'GS' with the fixed-nonce setting they
                     evaluate); detector = sign-match bit accuracy.
  tabwak             binarize-by-sign -> self-clone first half into second
                     -> fixed permutation -> sample half-Gaussian per bit
                     (their 'TabWak'); detector = median-binarize, invert
                     permutation, compare halves (bit accuracy).
  tabwak_star        4-level quantile variant ('TabWak*'); detector =
                     quartile-binarize (2 bits) and compare halves.

Every construction returns latents shaped (m, n) to feed any Host, plus a
detector operating on inverted latents. Note none of these are
distribution-preserving -- that is the point of the comparison: our C2ST
learnability audit should sit at ~0.5 for PancakeMark and >0.5 for these.

"""
from __future__ import annotations

import numpy as np
from scipy import stats


# ------------------------------------------------------------ Gaussian Shading
class GaussianShading:
    name = "gaussian_shading"

    def __init__(self, n: int, seed: int = 217):
        rng = np.random.default_rng(seed)
        self.bits = rng.integers(0, 2, size=n)          # fixed key bits
        self.n = n

    def sample_latents(self, m: int, rng) -> np.ndarray:
        z = np.abs(rng.standard_normal((m, self.n)))
        sign = np.where(self.bits[None, :] == 1, 1.0, -1.0)
        return z * sign

    def bit_accuracy(self, z_hat: np.ndarray) -> float:
        signs = (z_hat > 0).astype(int)
        return float((signs == self.bits[None, :]).mean())


# ------------------------------------------------------------------- TabWak
class TabWak:
    name = "tabwak"

    def __init__(self, n: int, seed: int = 217, levels: int = 2):
        assert n % 2 == 0
        rng = np.random.default_rng(seed)
        self.perm = rng.permutation(n)
        self.inv_perm = np.argsort(self.perm)
        self.n = n
        self.levels = levels                            # 2 = TabWak, 4 = TabWak*

    def _quantize(self, x: np.ndarray) -> np.ndarray:
        if self.levels == 2:
            return (x > 0).astype(int)
        q = np.zeros_like(x, dtype=int)
        q[x <= -0.67449] = 0
        q[x >= 0.67449] = 1
        q[(x > -0.67449) & (x < 0)] = 2
        q[(x > 0) & (x < 0.67449)] = 3
        return q

    def _sample_from_levels(self, lv: np.ndarray, rng) -> np.ndarray:
        u = rng.random(lv.shape)
        if self.levels == 2:
            return stats.norm.ppf(np.where(lv == 0, u * 0.5, u * 0.5 + 0.5))
        lo = np.select([lv == 0, lv == 1, lv == 2, lv == 3],
                       [0.0, 0.75, 0.25, 0.5])
        return stats.norm.ppf(u * 0.25 + lo)

    def sample_latents(self, m: int, rng) -> np.ndarray:
        base = rng.standard_normal((m, self.n))
        lv = self._quantize(base)
        half = self.n // 2
        lv[:, half:] = lv[:, :half]                     # self-clone
        lv = lv[:, self.perm]
        return self._sample_from_levels(lv, rng)

    def bit_accuracy(self, z_hat: np.ndarray) -> float:
        """Their detector: median (or quartile) binarize, undo permutation,
        compare halves."""
        if self.levels == 2:
            med = np.median(z_hat)
            b = (z_hat > med).astype(int)
        else:
            qs = np.quantile(z_hat, [0.25, 0.5, 0.75])
            b = np.digitize(z_hat, qs)
        b = b[:, self.inv_perm]
        half = self.n // 2
        return float((b[:, :half] == b[:, half:]).mean())


def tabwak_star(n: int, seed: int = 217) -> TabWak:
    tw = TabWak(n, seed=seed, levels=4)
    tw.name = "tabwak_star"
    return tw
