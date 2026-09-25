"""Localization: p-value validity, FDR control, rho estimate, group scan."""
import numpy as np

from pancakemark import keygen, sample_latents, theory
from pancakemark.localize import (
    row_pvalues, bh_flag, estimate_rho, group_scan, localize,
)

RNG = np.random.default_rng(60)
N, K = 64, 4


def _mixture(key, m, rho, rng):
    n_marked = int(round(rho * m))
    z = rng.standard_normal((m, N))
    zm = sample_latents(n_marked, key, rng)
    idx = rng.choice(m, n_marked, replace=False)
    z[idx] = zm
    y = np.zeros(m, bool); y[idx] = True
    return z, y


def test_row_pvalues_valid_on_null():
    key = keygen(N, K, gamma=2.0, beta=0.05, seed=61)
    z = RNG.standard_t(4, size=(400, N))
    p = row_pvalues(z, key, draws=25, rng=RNG)
    # super-uniformity at a few thresholds
    for a in (0.05, 0.2):
        assert p.__le__(a).mean() < a + 3 * np.sqrt(a / 400), (a, (p <= a).mean())


def test_fdr_controlled_and_power_present():
    key = keygen(N, K, gamma=2.0, beta=0.02, seed=62)
    z, y = _mixture(key, 1200, rho=0.4, rng=RNG)
    p = row_pvalues(z, key, draws=40, rng=RNG)
    flags = bh_flag(p, q=0.10)
    assert flags.sum() > 50                       # nontrivial power
    fdr = (~y[flags]).mean() if flags.any() else 0.0
    assert fdr <= 0.20, fdr                       # ~q up to sampling noise


def test_rho_estimate_accurate():
    key = keygen(N, K, gamma=2.0, beta=0.05, seed=63)
    mu_ref = theory.mu(key.gamma, key.beta)
    for rho in (0.2, 0.6):
        z, _ = _mixture(key, 4000, rho, RNG)
        est = estimate_rho(z, key, mu_ref)
        assert abs(est["rho_hat"] - rho) < 4 * est["se"] + 0.02, (rho, est)


def test_group_scan_finds_the_arm():
    key = keygen(N, K, gamma=2.0, beta=0.05, seed=64)
    m = 900
    labels = np.repeat(["armA", "armB", "armC"], m // 3)
    z = RNG.standard_normal((m, N))
    z[labels == "armB"] = sample_latents(m // 3, key, RNG)  # armB imputed
    rep = group_scan(z, key, labels)
    by = {r["group"]: r for r in rep}
    assert by["armB"]["p_bonferroni"] < 1e-6
    assert by["armA"]["p_hoeffding"] > 1e-3
    assert by["armC"]["p_hoeffding"] > 1e-3


def test_localize_end_to_end():
    key = keygen(N, K, gamma=2.0, beta=0.05, seed=65)
    mu_ref = theory.mu(key.gamma, key.beta)
    z, y = _mixture(key, 800, rho=0.5, rng=RNG)
    rep = localize(z, key, mu_ref=mu_ref, q=0.05, draws=30, rng=RNG)
    assert rep["num_flagged"] > 100
    assert abs(rep["rho_hat"] - 0.5) < 0.1
