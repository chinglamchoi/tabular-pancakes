"""Attacks: sanity, invariances against the detection stack, AUC machinery."""
import numpy as np

from pancakemark import keygen, sample_latents, detect, theory
from pancakemark import attacks as A
from pancakemark.hosts import TanhFlowHost

RNG = np.random.default_rng(50)
N = 48


def _marked_table(host, key, m):
    return host.generate(sample_latents(m, key, RNG))


def test_row_attacks_leave_detection_valid():
    host = TanhFlowHost(N, seed=51)
    key = keygen(N, k=4, gamma=2.0, beta=0.05, seed=52)
    X = _marked_table(host, key, 1200)
    for Xa in (A.shuffle_rows(X, RNG), A.subsample_rows(X, 0.4, RNG)):
        res = detect(host.invert(Xa), key, method="hoeffding")
        assert res.pvalue < 1e-6


def test_cell_noise_degrades_gracefully():
    host = TanhFlowHost(N, seed=53)
    key = keygen(N, k=4, gamma=2.0, beta=0.05, seed=54)
    X = _marked_table(host, key, 4000)
    mus = []
    for frac in (0.0, 0.02, 0.05):
        Xa = A.gaussian_cell_noise(X, frac, RNG)
        s = detect(host.invert(Xa), key, method="hoeffding")
        mus.append(s.stat)
    assert mus[0] > mus[1] > mus[2] > 0.0


def test_row_edit_linear_degradation():
    """mu under rho-fraction row edits ~= (1-rho) * mu -- the linear claim."""
    host = TanhFlowHost(N, seed=55)
    key = keygen(N, k=4, gamma=2.0, beta=0.05, seed=56)
    X = _marked_table(host, key, 6000)
    base = detect(host.invert(X), key, method="hoeffding").stat
    for rho in (0.3, 0.6):
        Xa = A.edit_rows(X, rho, RNG)
        s = detect(host.invert(Xa), key, method="hoeffding").stat
        assert abs(s - (1 - rho) * base) < 0.05, (rho, s, base)


def test_few_cell_edits_negligible():
    host = TanhFlowHost(N, seed=57)
    key = keygen(N, k=4, gamma=2.0, beta=0.05, seed=58)
    X = _marked_table(host, key, 1500)
    Xa = A.few_cell_edits(X, num_cells=30, size_in_std=2.0, rng=RNG)
    res = detect(host.invert(Xa), key, method="hoeffding")
    assert res.pvalue < 1e-6  # no small edit removes the watermark


def test_keyed_vs_keyless_scrub_separation():
    """At equal per-row L2 distortion, keyed scrubbing kills far more
    signal than keyless -- the sqrt(n/k) separation, empirically."""
    key = keygen(64, k=4, gamma=2.0, beta=0.05, seed=59)
    z = sample_latents(6000, key, RNG)
    budget = 0.15 * np.sqrt(key.k)          # per-row L2
    zk = A.latent_keyed_scrub(z, key.W, budget / np.sqrt(key.k), RNG)
    zi = A.latent_isotropic_noise(z, budget / np.sqrt(key.n), RNG)
    mu_k = detect(zk, key, method="hoeffding").stat
    mu_i = detect(zi, key, method="hoeffding").stat
    assert mu_k < mu_i - 0.1, (mu_k, mu_i)


def test_covariance_attack_blind_at_gamma2_sees_gamma_half():
    from pancakemark.sampler import sample_latents as sl

    def mk(gamma):
        def f(rng):
            k = keygen(24, 1, gamma=gamma, beta=0.05,
                       seed=int(rng.integers(1 << 30)))
            return sl(800, k, rng)
        return f

    def null(rng):
        return rng.standard_normal((800, 24))

    # NB standard-normal convention: our gamma = BRST rho-convention gamma
    # divided by sqrt(2*pi); the variance-collapse regime is gamma <~ 0.4.
    auc2 = A.covariance_attack_auc(mk(2.0), null, trials=40, rng=RNG)
    auc_small = A.covariance_attack_auc(mk(0.35), null, trials=40, rng=RNG)
    assert abs(auc2 - 0.5) < 0.15, auc2          # blind in the safe regime
    assert auc_small > 0.9, auc_small            # attack works when it should


def test_mix_with_real_composition():
    Xm = RNG.standard_normal((500, 5)) + 10
    Xr = RNG.standard_normal((1000, 5))
    X, y = A.mix_with_real(Xm, Xr, rho=0.3, rng=RNG)
    assert X.shape[0] == 1000 and abs(y.mean() - 0.3) < 0.01
    assert X[y].mean() > 5  # marked rows really are the marked ones
