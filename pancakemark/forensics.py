"""Classical data-fraud forensics: the tests a generator defeats.

Implemented (numeric tables):
  benford_first_digit   chi-square against Benford's law on leading digits
                        (only meaningful for columns spanning >~2 orders of
                        magnitude; the test reports which columns qualify).
  terminal_digit        chi-square of last-digit uniformity at a stated
                        reported precision (Mosimann-style terminal-digit
                        analysis; fabricated numbers over/under-use digits).
  grim_consistency      GRIM: for integer-valued underlying data, reported
                        means of n items must lie on the 1/n grid
                        (Brown & Heathers 2017). Returns the fraction of
                        (mean, n) pairs that are GRIM-consistent.
  last_two_digit_corr   serial correlation of final two digits -- humans
                        typing numbers produce dependent digit pairs.

These are *baseline* forensics for the P4 claim ("a competent generator
passes what crude fabrication fails"), not contributions of this work.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

BENFORD_P = np.log10(1 + 1 / np.arange(1, 10))


def _leading_digits(x: np.ndarray) -> np.ndarray:
    x = np.abs(np.asarray(x, dtype=float))
    x = x[(x > 0) & np.isfinite(x)]
    mag = np.floor(np.log10(x))
    return (x / 10.0**mag).astype(int)


def benford_first_digit(col: np.ndarray) -> dict:
    """Chi-square vs Benford; 'applicable' if the column spans >= 2 orders
    of magnitude (otherwise Benford is not expected even for real data)."""
    x = np.abs(col[np.isfinite(col)])
    x = x[x > 0]
    if x.size < 50:
        return {"applicable": False, "reason": "too few nonzero values"}
    span = np.log10(x.max()) - np.log10(x.min())
    d = _leading_digits(x)
    obs = np.bincount(d, minlength=10)[1:10]
    exp = BENFORD_P * d.size
    chi2 = float(((obs - exp) ** 2 / exp).sum())
    p = float(stats.chi2.sf(chi2, df=8))
    return {"applicable": bool(span >= 2.0), "span_orders": float(span),
            "chi2": chi2, "pvalue": p}


def terminal_digit(col: np.ndarray, decimals: int = 0) -> dict:
    """Uniformity of the terminal digit at the stated precision."""
    x = col[np.isfinite(col)]
    scaled = np.round(x * 10.0**decimals).astype(np.int64)
    digits = np.abs(scaled) % 10
    if digits.size < 50:
        return {"applicable": False, "reason": "too few values"}
    # terminal-digit analysis assumes enough spread that the last digit is
    # mechanistically arbitrary
    obs = np.bincount(digits, minlength=10)
    exp = np.full(10, digits.size / 10)
    chi2 = float(((obs - exp) ** 2 / exp).sum())
    return {"applicable": True, "chi2": chi2,
            "pvalue": float(stats.chi2.sf(chi2, df=9))}


def grim_consistency(means: np.ndarray, ns: np.ndarray,
                     decimals: int = 2) -> dict:
    """Fraction of reported (mean, n) pairs consistent with integer data:
    round(mean*n) / n must re-round to the reported mean."""
    means = np.asarray(means, dtype=float)
    ns = np.asarray(ns, dtype=int)
    ok = np.zeros(means.size, bool)
    for i, (mu, n) in enumerate(zip(means, ns)):
        cand = np.round(mu * n) / n
        ok[i] = np.round(cand, decimals) == np.round(mu, decimals)
    return {"consistent_frac": float(ok.mean()), "num_pairs": int(means.size)}


def last_two_digit_corr(col: np.ndarray, decimals: int = 2) -> dict:
    """Independence of the last two decimal digits (chi-square on the
    10x10 contingency table)."""
    x = col[np.isfinite(col)]
    scaled = np.abs(np.round(x * 10.0**decimals)).astype(np.int64)
    d1, d2 = scaled % 10, (scaled // 10) % 10
    if x.size < 200:
        return {"applicable": False, "reason": "too few values"}
    table = np.zeros((10, 10))
    np.add.at(table, (d1, d2), 1)
    chi2, p, _, _ = stats.chi2_contingency(table + 1e-9)
    return {"applicable": True, "chi2": float(chi2), "pvalue": float(p)}


def forensic_panel(X: np.ndarray, decimals: int = 2) -> dict:
    """Run the panel over all columns; report per-test min p-values with
    Bonferroni correction across columns (the auditor's aggregate)."""
    res = {"benford": [], "terminal": [], "lasttwo": []}
    for j in range(X.shape[1]):
        b = benford_first_digit(X[:, j])
        if b.get("applicable"):
            res["benford"].append(b["pvalue"])
        t = terminal_digit(X[:, j], decimals)
        if t.get("applicable"):
            res["terminal"].append(t["pvalue"])
        l2 = last_two_digit_corr(X[:, j], decimals)
        if l2.get("applicable"):
            res["lasttwo"].append(l2["pvalue"])
    out = {}
    for name, ps in res.items():
        if ps:
            m = len(ps)
            out[name] = {"num_cols": m,
                         "min_p_bonferroni": float(min(1.0, min(ps) * m))}
        else:
            out[name] = {"num_cols": 0}
    return out
