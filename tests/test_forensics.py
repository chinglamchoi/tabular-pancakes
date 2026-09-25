"""Forensics baselines: crude fabrication caught, honest data passes."""
import numpy as np

from pancakemark.forensics import (
    benford_first_digit, terminal_digit, grim_consistency,
    last_two_digit_corr, forensic_panel,
)

RNG = np.random.default_rng(70)


def test_benford_on_benford_like_data():
    # log-uniform over 4 orders of magnitude is Benford to high accuracy
    x = 10 ** RNG.uniform(0, 4, size=20000)
    r = benford_first_digit(x)
    assert r["applicable"] and r["pvalue"] > 1e-3


def test_benford_flags_uniform_fabrication():
    x = RNG.uniform(10, 9999, size=20000)  # linear-uniform fabrication, 3 orders span
    r = benford_first_digit(x)
    assert r["applicable"] and r["pvalue"] < 1e-6


def test_terminal_digit_flags_rounded_fabrication():
    honest = RNG.normal(50, 13, size=5000)
    faked = np.round(RNG.normal(50, 13, size=5000) * 2) / 2  # .0/.5 favored
    assert terminal_digit(honest, decimals=2)["pvalue"] > 1e-3
    assert terminal_digit(faked, decimals=1)["pvalue"] < 1e-6


def test_grim():
    ns = np.array([20, 25, 30, 40])
    true_means = np.array([np.round(RNG.integers(0, 8, n).mean(), 2) for n in ns])
    fake_means = true_means + 0.013  # off-grid
    assert grim_consistency(true_means, ns)["consistent_frac"] == 1.0
    assert grim_consistency(fake_means, ns)["consistent_frac"] < 0.5


def test_last_two_digit_corr_on_honest_data():
    x = RNG.normal(0, 37.3, size=20000)
    r = last_two_digit_corr(x, decimals=3)
    assert r["applicable"] and r["pvalue"] > 1e-3


def test_panel_runs():
    X = np.column_stack([10 ** RNG.uniform(0, 3, 2000),
                         RNG.normal(5, 2, 2000)])
    out = forensic_panel(X, decimals=3)
    assert "benford" in out and "terminal" in out
