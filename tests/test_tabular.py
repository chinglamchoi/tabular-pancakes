"""TableCodec: whitening correctness, affine invariance, null safety."""
import numpy as np

from pancakemark import keygen, detect
from pancakemark.tabular import TableCodec

RNG = np.random.default_rng(40)


def test_whitening_gives_identity_covariance():
    n = 12
    cov = 0.8 ** np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    X = RNG.standard_normal((20000, n)) @ np.linalg.cholesky(cov).T
    Z = TableCodec(shrinkage=0.0).fit_transform(X)
    C = np.cov(Z, rowvar=False)
    assert np.max(np.abs(C - np.eye(n))) < 0.06


def test_column_affine_invariance_selffit():
    """Unit changes / rescaling / shifts leave the self-fit codec output
    invariant up to numerical noise -- the canonicalization claim."""
    X = RNG.standard_normal((5000, 8)) @ RNG.standard_normal((8, 8))
    a = RNG.uniform(0.2, 12.0, size=8)
    b = RNG.normal(0, 40.0, size=8)
    Z1 = TableCodec().fit_transform(X)
    Z2 = TableCodec().fit_transform(X * a + b)
    np.testing.assert_allclose(Z1, Z2, atol=1e-8)


def test_codec_plus_detector_null_safe_on_real_like_data():
    X = np.exp(RNG.standard_normal((800, 10)) * 0.7)  # lognormal columns
    Z = TableCodec().fit_transform(X)
    worst = 1.0
    for tr in range(20):
        key = keygen(10, k=2, gamma=2.0, beta=0.05, seed=90_000 + tr)
        p = detect(Z, key, method="hoeffding").pvalue
        worst = min(worst, p)
    assert worst > 1e-3, worst


def test_constant_column_handled():
    X = RNG.standard_normal((500, 5))
    X[:, 2] = 3.14
    Z = TableCodec().fit_transform(X)
    assert np.all(np.isfinite(Z))
