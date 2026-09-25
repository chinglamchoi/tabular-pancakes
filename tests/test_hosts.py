"""Hosts and inversion protocol: exactness, calibration, end-to-end detection."""
import numpy as np
import pytest

from pancakemark import keygen, sample_latents, detect
from pancakemark.hosts import LinearHost, TanhFlowHost, NoisyEncoderHost, QuantizedHost
from pancakemark.inversion import inversion_report

RNG = np.random.default_rng(30)
N = 48


def test_linear_host_exact_inverse():
    h = LinearHost(N, seed=0)
    z = RNG.standard_normal((500, N))
    np.testing.assert_allclose(h.invert(h.generate(z)), z, atol=1e-9)


def test_tanh_flow_exact_inverse():
    h = TanhFlowHost(N, layers=3, seed=1)
    z = RNG.standard_normal((500, N))
    np.testing.assert_allclose(h.invert(h.generate(z)), z, atol=1e-9)


def test_sigma_inv_measurement_calibrated():
    """On NoisyEncoderHost the Gaussian model is exact: measured sigma must
    match sigma_enc, and mu_pred must match mu_emp."""
    sigma = 0.12
    h = NoisyEncoderHost(TanhFlowHost(N, seed=2), sigma_enc=sigma, seed=3)
    key = keygen(N, k=4, gamma=2.0, beta=0.05, seed=31)
    rep = inversion_report(h, key, num_rows=4000, rng=RNG)
    assert abs(rep["sigma_onaxis"] - sigma) < 0.01, rep["sigma_onaxis"]
    assert abs(rep["mu_pred_gaussian_model"] - rep["mu_emp"]) < 0.03, rep
    assert 0.9 < rep["marked_vs_unmarked_residual_ratio"] < 1.1


def test_end_to_end_detection_through_flow():
    h = TanhFlowHost(N, layers=3, seed=4)
    key = keygen(N, k=4, gamma=2.0, beta=0.05, seed=32)
    z = sample_latents(500, key, RNG)
    zhat = h.invert(h.generate(z))
    res = detect(zhat, key, method="hoeffding")
    assert res.pvalue < 1e-6, res.pvalue


def test_end_to_end_null_safe_through_flow():
    h = TanhFlowHost(N, layers=3, seed=5)
    key = keygen(N, k=4, gamma=2.0, beta=0.05, seed=33)
    z = RNG.standard_normal((500, N))
    zhat = h.invert(h.generate(z))
    res = detect(zhat, key, method="hoeffding")
    assert res.pvalue > 0.01


def test_quantized_host_detection_survives_rounding():
    base = TanhFlowHost(N, layers=3, seed=6)
    key = keygen(N, k=4, gamma=2.0, beta=0.05, seed=34)
    z = sample_latents(800, key, RNG)
    for dec, expect_detect in [(3, True), (2, True)]:
        h = QuantizedHost(base, decimals=dec)
        zhat = h.invert(h.generate(z))
        res = detect(zhat, key, method="hoeffding")
        assert (res.pvalue < 1e-4) == expect_detect, (dec, res.pvalue)


def test_report_fields_present():
    h = LinearHost(N, seed=7)
    key = keygen(N, k=2, gamma=2.0, beta=0.05, seed=35)
    rep = inversion_report(h, key, num_rows=500, rng=RNG)
    for f in ("sigma_onaxis", "mu_emp", "mu_pred_gaussian_model",
              "rows_needed_alpha1e6_power0.9"):
        assert f in rep
    assert rep["sigma_onaxis"] < 1e-6  # exact inverse
