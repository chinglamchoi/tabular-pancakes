"""Keys and utilities: Haar frames, PRF determinism, serialization."""
import numpy as np
import pytest

from pancakemark import keygen
from pancakemark.keys import Key, prf_uniform
from pancakemark.utils import haar_frame


def test_frame_orthonormal():
    W = haar_frame(64, 8, np.random.default_rng(0))
    np.testing.assert_allclose(W.T @ W, np.eye(8), atol=1e-12)


def test_frame_rotation_invariance_moment():
    """Haar => E[w w^T] = I/n for the first column."""
    rng = np.random.default_rng(1)
    acc = np.zeros((16, 16))
    trials = 4000
    for _ in range(trials):
        w = haar_frame(16, 1, rng)[:, 0]
        acc += np.outer(w, w)
    acc /= trials
    np.testing.assert_allclose(acc, np.eye(16) / 16, atol=0.01)


def test_keygen_reproducible_and_serializable():
    k1 = keygen(32, 4, gamma=2.0, beta=0.05, seed=7)
    k2 = keygen(32, 4, gamma=2.0, beta=0.05, seed=7)
    np.testing.assert_array_equal(k1.W, k2.W)
    assert k1.kappa == k2.kappa
    k3 = Key.from_dict(k1.to_dict())
    np.testing.assert_allclose(k3.W, k1.W)
    assert k3.kappa == k1.kappa


def test_keygen_rejects_bad_params():
    with pytest.raises(ValueError):
        keygen(8, 0)
    with pytest.raises(ValueError):
        keygen(8, 2, beta=0.0)
    with pytest.raises(ValueError):
        keygen(8, 2, gamma=-1.0)


def test_prf_uniform_deterministic_and_labelled():
    kappa = b"\x01" * 32
    a = prf_uniform(kappa, "row", 5)
    b = prf_uniform(kappa, "row", 5)
    c = prf_uniform(kappa, "row", 6)
    assert a == b and a != c and 0.0 <= a < 1.0


def test_prf_marginal_uniformity():
    kappa = b"\x02" * 32
    xs = np.array([prf_uniform(kappa, i) for i in range(2000)])
    from scipy import stats
    _, p = stats.kstest(xs, "uniform")
    assert p > 1e-3
