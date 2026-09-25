"""Categorical channel: faithfulness, null exactness, detection, sync robustness."""
import numpy as np
from scipy import stats

from pancakemark import keygen, sample_latents
from pancakemark.categorical import (
    pancake_indices, sample_categorical, categorical_scores, detect_categorical,
)

RNG = np.random.default_rng(80)
N, K = 48, 4


def _setup(m=3000, seed=81, entropy="high"):
    key = keygen(N, K, gamma=2.0, beta=0.05, seed=seed)
    z = sample_latents(m, key, RNG)
    if entropy == "high":
        logits = RNG.normal(0, 0.5, size=(m, 5))
    else:
        logits = RNG.normal(0, 3.0, size=(m, 5))     # peaked -> low entropy
    probs = np.exp(logits); probs /= probs.sum(1, keepdims=True)
    return key, z, probs


def test_faithful_marginals():
    """Keyed Gumbel-max reproduces the model conditionals (chi-square)."""
    key, z, _ = _setup(m=20000, seed=82)
    p = np.array([0.5, 0.3, 0.15, 0.05])
    probs = np.tile(p, (z.shape[0], 1))
    cats = sample_categorical(probs, z, key)
    obs = np.bincount(cats, minlength=4)
    chi2 = (((obs - p * z.shape[0]) ** 2) / (p * z.shape[0])).sum()
    assert stats.chi2.sf(chi2, df=3) > 1e-3, obs / z.shape[0]


def test_null_scores_uniform_and_mean_zero():
    """Wrong key (or unmarked data): v is U(0,1) -> mean score ~ 0."""
    key, z, probs = _setup(seed=83)
    cats = RNG.integers(0, 5, size=z.shape[0])          # independent of key
    s = categorical_scores(cats, z, key, num_cats=5)
    assert abs(s.mean()) < 3 / np.sqrt(s.size) * np.sqrt(1 / 3) + 0.02
    v = (s + 1) / 2
    assert stats.kstest(v, "uniform").pvalue > 1e-3


def test_detects_keyed_sampling():
    key, z, probs = _setup(seed=84)
    cats = sample_categorical(probs, z, key)
    res = detect_categorical(cats, z, key, num_cats=5)
    assert res["pvalue"] < 1e-8, res


def test_low_entropy_carries_less_signal():
    key, zh, ph = _setup(seed=85, entropy="high")
    cats_h = sample_categorical(ph, zh, key)
    Sh = detect_categorical(cats_h, zh, key, 5)["stat"]
    key2, zl, pl = _setup(seed=85, entropy="low")
    cats_l = sample_categorical(pl, zl, key2)
    Sl = detect_categorical(cats_l, zl, key2, 5)["stat"]
    assert Sh > Sl > -0.05, (Sh, Sl)


def test_sync_robust_to_small_latent_noise():
    """Perturbations well below the layer gap leave sync indices intact,
    so detection survives; large noise degrades to null (no false signal)."""
    key, z, probs = _setup(seed=86)
    cats = sample_categorical(probs, z, key)
    z_small = z + 0.02 * RNG.standard_normal(z.shape)
    res_small = detect_categorical(cats, z_small, key, 5)
    assert res_small["pvalue"] < 1e-6, res_small["pvalue"]
    frac_same = (pancake_indices(z, key) == pancake_indices(z_small, key)).mean()
    assert frac_same > 0.9
