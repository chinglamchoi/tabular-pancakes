"""Sampler correctness: exact marginals, Gaussian complement, unit scale."""
import numpy as np
import pytest
from scipy import stats
from scipy.integrate import quad

from pancakemark import keygen, sample_latents, sample_pancake_1d, pancake_density_1d

RNG = np.random.default_rng(0)


@pytest.mark.parametrize("gamma,beta,delta", [(2.0, 0.05, 0.0), (4.0, 0.1, 0.3), (1.5, 0.2, 0.75)])
def test_density_integrates_to_one(gamma, beta, delta):
    val, err = quad(lambda t: pancake_density_1d(t, gamma, beta, delta), -10, 10, limit=400)
    assert abs(val - 1.0) < 1e-8


@pytest.mark.parametrize("gamma,beta,delta", [(2.0, 0.05, 0.0), (3.0, 0.1, 0.4)])
def test_1d_samples_match_density(gamma, beta, delta):
    """KS test of samples against the exact mixture CDF."""
    x = sample_pancake_1d(200_000, gamma, beta, delta, RNG)

    # CDF via the mixture (exact): sum_m p_m * Phi((t - c_m)/s)
    from pancakemark.sampler import _layer_distribution
    m, p, gp = _layer_distribution(gamma, beta, delta)
    centers = (m + delta) * gamma / gp**2
    s = beta / gp

    def cdf(t):
        t = np.asarray(t, dtype=float)[..., None]
        return (stats.norm.cdf((t - centers) / s) * p).sum(axis=-1)

    stat, pval = stats.kstest(x, cdf)
    assert pval > 1e-3, f"KS p={pval:.2e} (stat={stat:.4f})"


def test_1d_unnormalized_form_matches_mixture():
    """The mixture density equals the defining comb-times-envelope form."""
    gamma, beta, delta = 2.5, 0.08, 0.2
    t = np.linspace(-8, 8, 4001)  # wide enough that tail mass < 1e-14
    mix = pancake_density_1d(t, gamma, beta, delta)
    m = np.arange(-60, 61)
    raw = np.exp(-t[:, None] ** 2 / 2) * np.exp(
        -((m[None, :] + delta - gamma * t[:, None]) ** 2) / (2 * beta**2)
    )
    raw = raw.sum(axis=1)
    Z = np.trapezoid(raw, t)
    np.testing.assert_allclose(mix, raw / Z, rtol=1e-6, atol=1e-9)


def test_marginal_variance_near_one():
    x = sample_pancake_1d(400_000, 2.0, 0.05, 0.0, RNG)
    assert abs(np.var(x) - 1.0) < 0.02


def test_orthogonal_complement_is_standard_gaussian():
    key = keygen(n=48, k=4, gamma=2.0, beta=0.05, seed=1)
    z = sample_latents(50_000, key, RNG, marked=True)
    # a fixed direction orthogonal to all of W
    v = RNG.standard_normal(key.n)
    v -= key.W @ (key.W.T @ v)
    v /= np.linalg.norm(v)
    proj = z @ v
    stat, pval = stats.kstest(proj, "norm")
    assert pval > 1e-3
    assert abs(proj.mean()) < 0.02 and abs(proj.std() - 1) < 0.02


def test_unmarked_is_standard_gaussian():
    key = keygen(n=32, k=2, seed=2)
    z = sample_latents(20_000, key, RNG, marked=False)
    proj = z @ key.W[:, 0]
    stat, pval = stats.kstest(proj, "norm")
    assert pval > 1e-3


def test_secret_directions_have_pancake_marginal():
    key = keygen(n=64, k=2, gamma=3.0, beta=0.1, delta=np.array([0.0, 0.5]), seed=3)
    z = sample_latents(100_000, key, RNG, marked=True)
    from pancakemark.sampler import _layer_distribution
    for j in range(2):
        proj = z @ key.W[:, j]
        m, p, gp = _layer_distribution(key.gamma, key.beta, float(key.delta[j]))
        centers = (m + key.delta[j]) * key.gamma / gp**2
        s = key.beta / gp

        def cdf(t):
            t = np.asarray(t, dtype=float)[..., None]
            return (stats.norm.cdf((t - centers) / s) * p).sum(axis=-1)

        stat, pval = stats.kstest(proj, cdf)
        assert pval > 1e-3, f"direction {j}: KS p={pval:.2e}"
