"""Detection: statistic matches closed-form mean; nulls are valid; invariances."""
import numpy as np
import pytest
from scipy.integrate import quad

from pancakemark import (
    keygen, sample_latents, detect, row_scores, global_stat,
    hoeffding_pvalue, theory,
)
from pancakemark.detect import direction_stats

RNG = np.random.default_rng(10)


@pytest.mark.parametrize("gamma,beta", [(2.0, 0.05), (4.0, 0.1), (2.0, 0.2)])
def test_marked_mean_matches_theory(gamma, beta):
    key = keygen(n=64, k=4, gamma=gamma, beta=beta, seed=11)
    z = sample_latents(60_000, key, RNG)
    S = global_stat(z, key)
    mu = theory.mu(gamma, beta)
    se = np.sqrt((1 - mu**2) / (60_000))  # generous SE for the mean
    assert abs(S - mu) < 6 * se + 1e-3, f"S={S:.4f} vs mu={mu:.4f}"


def test_mu_formula_matches_numerical_integration():
    """mu = E[cos(2 pi (f t - delta))] under the exact 1-D density, by quadrature."""
    from pancakemark.sampler import pancake_density_1d
    gamma, beta, delta = 2.0, 0.15, 0.3
    f = theory.score_frequency(gamma, beta)
    val, err = quad(
        lambda t: pancake_density_1d(t, gamma, beta, delta)
        * np.cos(2 * np.pi * (f * t - delta)),
        -12, 12, limit=800,
    )
    assert abs(val - theory.mu(gamma, beta)) < 1e-6


def test_added_noise_degrades_as_theory():
    gamma, beta, sigma = 2.0, 0.05, 0.08
    key = keygen(n=64, k=4, gamma=gamma, beta=beta, seed=12)
    z = sample_latents(60_000, key, RNG)
    z_noisy = z + sigma * RNG.standard_normal(z.shape)
    S = global_stat(z_noisy, key)
    mu = theory.mu(gamma, beta, noise_std=sigma)
    assert abs(S - mu) < 0.02, f"S={S:.4f} vs mu={mu:.4f}"


def test_row_permutation_and_subset_invariance():
    key = keygen(n=32, k=2, gamma=2.0, beta=0.05, seed=13)
    z = sample_latents(5_000, key, RNG)
    perm = RNG.permutation(z.shape[0])
    assert np.isclose(global_stat(z, key), global_stat(z[perm], key))
    sub = z[:1000]
    assert row_scores(sub, key).shape == (1000,)


def test_hoeffding_fpr_controlled_on_gaussian_null():
    """Fraction of Hoeffding p-values <= alpha must be <= alpha (empirically)."""
    key0 = keygen(n=32, k=2, gamma=2.0, beta=0.05, seed=14)
    alpha, trials, hits = 0.05, 400, 0
    for i in range(trials):
        key = keygen(n=32, k=2, gamma=2.0, beta=0.05, seed=1000 + i)
        z = RNG.standard_normal((300, 32))
        S = global_stat(z, key)
        if hoeffding_pvalue(S, num_rows=300) <= alpha:
            hits += 1
    # binomial(400, 0.05) 99.9% quantile ~ 34; being a *bound* we expect far fewer
    assert hits <= 34, f"{hits}/{trials} rejections at alpha={alpha}"


def test_mc_null_pvalue_valid_on_heavy_tailed_null():
    """MC key-null p-values on data independent of the key: P(p<=alpha)<=alpha,
    even for heavy-tailed, non-Gaussian data (no distributional assumption)."""
    alpha, trials, hits = 0.1, 60, 0
    for i in range(trials):
        key = keygen(n=24, k=2, gamma=2.0, beta=0.05, seed=2000 + i)
        z = RNG.standard_t(df=3, size=(200, 24))  # heavy tails, not Gaussian
        res = detect(z, key, method="mc", mc_draws=99, rng=RNG)
        if res.pvalue <= alpha:
            hits += 1
    # binomial(60, 0.1): P(hits > 13) < 0.003
    assert hits <= 13, f"{hits}/{trials}"


def test_detect_rejects_on_marked_data():
    key = keygen(n=48, k=4, gamma=2.0, beta=0.05, seed=15)
    z = sample_latents(500, key, RNG)
    res_h = detect(z, key, method="hoeffding")
    res_m = detect(z, key, method="mc", mc_draws=199, rng=RNG)
    assert res_h.pvalue < 1e-6
    assert res_m.pvalue <= 1 / 100  # smallest achievable at 199 draws is 1/200


def test_phase_recovery():
    delta = np.array([0.1, 0.45, 0.8])
    key = keygen(n=64, k=3, gamma=2.0, beta=0.05, delta=delta, seed=16)
    z = sample_latents(20_000, key, RNG)
    d = direction_stats(z, key)["delta_hat"]
    err = np.abs(np.mod(d - delta + 0.5, 1.0) - 0.5)
    assert np.all(err < 0.01), err
