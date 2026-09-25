"""P4 -- Classical forensics baseline (toy): who gets caught by what.

Three tables through the standard fraud-forensics panel:
  real-like      breast-cancer measurements (genuine empirical data)
  crude fake     'typed numbers': linear-uniform magnitudes, half-unit
                 rounding, digit preferences
  generator      marked TanhFlowHost output fitted to nothing in
                 particular -- smooth continuous columns

Reading the panel: absolute pass/fail is meaningless -- REAL measurement
data itself fails digit-uniformity tests (values are recorded at fixed
precision) and most columns are not Benford-applicable. The meaningful
comparison is the *profile relative to real data*: the crude fake deviates
by orders of magnitude beyond real data's own deviations, while smooth
generator output differs from real data in the OPPOSITE direction (it is
"too clean": passes terminal-digit uniformity that quantized real data
fails). Consequence for the paper: matching real data's forensic profile
requires the generator's preprocessing to reproduce recorded precision --
which TabSyn's quantile transforms do; the cluster version of this
experiment (TabSyn-on-Adult vs. real Adult) tests exactly that.

Usage: python p4_forensics.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import outdir, save_csv  # noqa: E402

from pancakemark import keygen, sample_latents
from pancakemark.hosts import TanhFlowHost
from pancakemark.forensics import forensic_panel


def crude_fake(m, d, rng):
    """Excel-style fabrication: uniform magnitudes, favored round values."""
    X = rng.uniform(10, 5000, size=(m, d))
    X = np.round(X * 2) / 2                     # .0/.5 endings favored
    mask = rng.random(X.shape) < 0.2            # suspiciously round values
    X[mask] = np.round(X[mask], -1)
    return X


def main():
    out = outdir("p4_forensics")
    rng = np.random.default_rng(11)
    from sklearn.datasets import load_breast_cancer
    real = load_breast_cancer().data

    host = TanhFlowHost(24, layers=3, seed=12)
    key = keygen(24, 4, gamma=2.0, beta=0.05, seed=13)
    gen = host.generate(sample_latents(1500, key, rng))
    # scale generator output into positive 'measurement-like' ranges
    gen = np.abs(gen) * 40 + 5

    fake = crude_fake(1500, 10, rng)

    rows = []
    for name, X, dec in (("real_breast_cancer", real, 3),
                         ("crude_fake", fake, 1),
                         ("marked_generator", gen, 3)):
        panel = forensic_panel(X, decimals=dec)
        flat = {"table": name}
        for test, res in panel.items():
            flat[f"{test}_cols"] = res.get("num_cols", 0)
            flat[f"{test}_min_p_bonf"] = res.get("min_p_bonferroni", float("nan"))
        rows.append(flat)
        print(flat)
    save_csv(out / "forensics_panel.csv", rows)

    # crude fake deviates ORDERS beyond real data's own Benford deviation
    fake_vs_real = rows[1]["benford_min_p_bonf"] < 1e-30 * rows[0]["benford_min_p_bonf"]
    too_clean = rows[2]["terminal_min_p_bonf"] > 1e-3 > rows[0]["terminal_min_p_bonf"]
    print(f"crude fake beyond real-data profile: {fake_vs_real}")
    print(f"raw generator output 'too clean' vs quantized real data: {too_clean}")
    assert fake_vs_real, "panel failed to separate crude fake from real profile"


if __name__ == "__main__":
    main()
