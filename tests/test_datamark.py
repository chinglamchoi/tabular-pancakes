"""Data-space variant: marking, detection, invariances, moment preservation."""
import numpy as np

from pancakemark import keygen
from pancakemark.datamark import DataSpaceMark, detect_data_space
from pancakemark.tabular import TableCodec
from pancakemark import attacks as A

RNG = np.random.default_rng(77)


def _table(m=6000, d=10):
    # correlated, scaled, shifted numeric block
    L = RNG.standard_normal((d, d)) * 0.3 + np.eye(d)
    X = RNG.standard_normal((m, d)) @ L.T
    return X * (1 + 9 * RNG.random(d)) + 50 * RNG.standard_normal(d)


def _marked(m=6000, d=10, k=3):
    X = _table(m, d)
    codec = TableCodec().fit(X)
    key = keygen(d, k, gamma=2.0, beta=0.05, seed=770)
    Xm, rep = DataSpaceMark(key, codec).mark(X, RNG)
    return X, Xm, key, rep


def test_codec_inverse_roundtrip():
    X = _table(2000, 6)
    c = TableCodec().fit(X)
    assert np.allclose(c.inverse(c.transform(X)), X, atol=1e-8)


def test_mark_detect_selffit():
    X, Xm, key, rep = _marked()
    res = detect_data_space(Xm, key)          # verifier self-fits the codec
    assert res.pvalue < 1e-10
    # unmarked table: no signal
    res0 = detect_data_space(X, key)
    assert res0.pvalue > 1e-3


def test_second_moments_preserved():
    """The hCLWE identity Var(t') = (gamma^2+beta^2)/gamma'^2 = Var(t)
    means marking barely moves the whitened second moments, so a re-fit
    (self-fit) codec recovers essentially the same directions."""
    X, Xm, key, rep = _marked(m=20000)
    t0 = TableCodec().fit(X).transform(X) @ key.W
    t1 = TableCodec().fit(Xm).transform(Xm) @ key.W
    assert abs(t1.var() - t0.var()) < 0.03
    assert rep["mean_cell_rmse_stdunits"] < 0.25


def test_affine_invariance_exact():
    """Per-column rescale/shift of the released table leaves the detection
    statistic exactly unchanged (codec absorbs it)."""
    X, Xm, key, rep = _marked()
    Xa = A.affine_rescale(Xm, RNG)
    s1 = detect_data_space(Xm, key).stat
    s2 = detect_data_space(Xa, key).stat
    assert abs(s1 - s2) < 1e-8


def test_shuffle_subsample_invariance():
    X, Xm, key, rep = _marked()
    assert detect_data_space(A.shuffle_rows(Xm, RNG), key).pvalue < 1e-10
    assert detect_data_space(A.subsample_rows(Xm, 0.2, RNG), key).pvalue < 1e-6


def test_column_permutation_with_schema_restore_is_exact():
    """The protocol binds the key to the schema: the verifier reorders a
    column-permuted suspect back to schema order by name, after which
    detection is bit-exact."""
    X, Xm, key, rep = _marked()
    perm = np.random.default_rng(5).permutation(Xm.shape[1])
    X_perm = Xm[:, perm]                       # attacker permutes columns
    X_restored = X_perm[:, np.argsort(perm)]   # verifier restores by schema
    assert np.array_equal(X_restored, Xm)
    s1 = detect_data_space(Xm, key).stat
    s2 = detect_data_space(X_restored, key).stat
    assert abs(s1 - s2) < 1e-12    # identical up to BLAS layout jitter


def test_robust_to_cell_noise():
    X, Xm, key, rep = _marked()
    Xn = A.gaussian_cell_noise(Xm, 0.05, RNG)
    assert detect_data_space(Xn, key).pvalue < 1e-6


def test_column_selection_preserves_point_masses_and_integers():
    """Adult-like block: zero-inflated and low-cardinality columns must be
    excluded (their point masses survive marking EXACTLY -- the learnable
    tell on real data), integer columns stay integer, and detection with
    the published reference codec stays strong."""
    m = 6000
    cont = RNG.standard_normal((m, 3)) * [13.0, 2.0, 40.0] + [40, 10, 180]
    ints = np.round(RNG.standard_normal((m, 2)) * [12.0, 9.0] + [40, 35])
    zinf = np.where(RNG.random((m, 1)) < 0.9, 0.0,
                    RNG.exponential(7000, (m, 1)))          # capital.gain-like
    lowcard = RNG.integers(1, 5, (m, 1)).astype(float)      # 4 values
    spike = np.where(RNG.random((m, 1)) < 0.47, 40.0,       # hours-like:
                     np.round(RNG.normal(38, 12, (m, 1))))  # 47% at 40
    X = np.hstack([cont, ints, zinf, lowcard, spike])

    codec = TableCodec(select_columns=True).fit(X)
    for j in (5, 6, 7):                                     # zinf/lowcard/spike
        assert j not in codec.cols_
    key = keygen(codec.dim, 2, gamma=2.0, beta=0.05, seed=771)
    Xm, rep = DataSpaceMark(key, codec).mark(X, RNG)

    for j in (5, 6, 7):                                     # excluded cols
        assert np.array_equal(Xm[:, j], X[:, j])            # pass bit-exact
    for j in (3, 4):                                        # integers stay
        assert np.allclose(Xm[:, j], np.round(Xm[:, j]))
        assert Xm[:, j].min() >= X[:, j].min()
        assert Xm[:, j].max() <= X[:, j].max()

    res = detect_data_space(Xm, key, codec=codec)
    assert res.stat > 0.5 and res.pvalue < 1e-10
    assert detect_data_space(X, key, codec=codec).pvalue > 1e-3
