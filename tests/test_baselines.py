"""Baseline constructions: faithfulness of ports + detectors + learnability."""
import numpy as np

from pancakemark.baselines import GaussianShading, TabWak, tabwak_star

RNG = np.random.default_rng(90)
N = 48


def test_gaussian_shading_detector():
    gs = GaussianShading(N)
    z = gs.sample_latents(2000, RNG)
    assert gs.bit_accuracy(z) == 1.0
    assert abs(gs.bit_accuracy(RNG.standard_normal((2000, N))) - 0.5) < 0.05


def test_gs_marginals_are_half_gaussians():
    gs = GaussianShading(N)
    z = gs.sample_latents(20000, RNG)
    # per-coordinate sign fixed by key bit
    signs = np.sign(z).mean(axis=0)
    assert np.all(np.abs(signs) > 0.99)
    # magnitude is |N(0,1)|: mean ~ sqrt(2/pi)
    assert abs(np.abs(z).mean() - np.sqrt(2 / np.pi)) < 0.02


def test_tabwak_detector_separates():
    tw = TabWak(N)
    z = tw.sample_latents(3000, RNG)
    acc_marked = tw.bit_accuracy(z)
    acc_null = tw.bit_accuracy(RNG.standard_normal((3000, N)))
    assert acc_marked > 0.95
    assert abs(acc_null - 0.5) < 0.05


def test_tabwak_star_detector_separates():
    tws = tabwak_star(N)
    z = tws.sample_latents(3000, RNG)
    assert tws.bit_accuracy(z) > 0.9
    assert abs(tws.bit_accuracy(RNG.standard_normal((3000, N))) - 0.25) < 0.06


def test_tabwak_marginals_are_standard_gaussian_per_coordinate():
    """Their construction preserves per-coordinate N(0,1) (half/quantile
    resampling) -- the *joint* self-clone dependence is what a C2ST or a
    dependence-aware test can learn."""
    tw = TabWak(N)
    z = tw.sample_latents(20000, RNG)
    from scipy import stats
    p = stats.kstest(z[:, 3], "norm").pvalue
    assert p > 1e-3
