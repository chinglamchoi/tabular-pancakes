"""Localization: which rows (or groups) of a suspect table are marked.

Row-level p-values via a fresh-key empirical null.
For a row z_i generated independently of the key, its score S_i is
exchangeable with scores computed under freshly drawn keys, so ranking
S_i against a pooled null sample {S under B fresh keys} yields valid
per-row p-values (Phipson-Smyth correction), with no distributional
assumption on the data. Benjamini-Hochberg on these p-values controls
FDR over the flagged set for the null rows (BH under positive dependence;
we also report the flagged set at BY thresholds on request).

Mixing-rate estimate.
E[S_bar] = rho * mu_marked under a rho-mixture, so rho_hat =
clip(S_bar / mu_ref, 0, 1), where mu_ref is the verifier's calibrated
marked-row mean for this generator/key configuration (from
inversion_report on held-out generated data). A delta-method CI is
attached.

Group scan.
Given group labels (trial arm, site, time block), per-group means with
Hoeffding bounds localize *structured* fraud -- the forensically
meaningful signal ("the imputed arm").
"""
from __future__ import annotations

import numpy as np

from .detect import row_scores
from .keys import Key
from .utils import haar_frame


def row_null_sample(z: np.ndarray, key: Key, draws: int = 30,
                    rng: np.random.Generator | None = None) -> np.ndarray:
    """Pooled null row-scores under `draws` fresh keys (flattened)."""
    rng = rng or np.random.default_rng()
    n, k, f = key.n, key.k, key.score_frequency
    out = []
    for _ in range(draws):
        Wb = haar_frame(n, k, rng)
        db = rng.uniform(0, 1, size=k)
        theta = 2 * np.pi * np.mod(f * (z @ Wb), 1.0)
        out.append(np.cos(theta - 2 * np.pi * db[None, :]).mean(axis=1))
    return np.concatenate(out)


def row_pvalues(z: np.ndarray, key: Key, draws: int = 30,
                rng: np.random.Generator | None = None,
                null_scores: np.ndarray | None = None) -> np.ndarray:
    """Per-row p-values for H0: 'row is genuine' (rank-based).

    Two empirical nulls:
    * null_scores given (REFERENCE null): scores of a disjoint
      claimed-genuine sample under the TRUE key. Genuine suspect rows are
      exchangeable with the reference rows by construction, so the
      p-values -- and BH downstream -- are calibrated CONDITIONALLY on
      the realized key. This is the D2 verifier's null (it holds the
      claimed-genuine source anyway, for mu_null).
    * otherwise (fresh-key null): pooled scores under `draws` fresh keys.
      Valid marginally over the key draw, but on integer-latticed data a
      realized key can align with a data lattice direction and lift many
      genuine rows together (default at rho=.05: per-seed FDR .05-.16,
      nominal .05). The no-reference fallback."""
    s = row_scores(z, key)
    if null_scores is not None:
        null = np.sort(np.asarray(null_scores, dtype=float))
    else:
        null = np.sort(row_null_sample(z, key, draws, rng))
    # p_i = (1 + #{null >= s_i}) / (N + 1)
    geq = null.size - np.searchsorted(null, s, side="left")
    return (1.0 + geq) / (null.size + 1.0)


def bh_flag(pvals: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg: boolean flags at FDR level q."""
    m = pvals.size
    order = np.argsort(pvals)
    thresh = q * (np.arange(1, m + 1) / m)
    passed = pvals[order] <= thresh
    flags = np.zeros(m, bool)
    if passed.any():
        kmax = np.max(np.nonzero(passed)[0])
        flags[order[: kmax + 1]] = True
    return flags


def estimate_rho(z: np.ndarray, key: Key, mu_ref: float,
                 mu_null: float = 0.0) -> dict:
    """Mixing-rate estimate with a delta-method standard error.

    E[S_bar] = rho mu_ref + (1 - rho) mu_null under a rho-mixture:
    genuine REAL rows are key-independent but carry a small nonzero
    baseline coherence mu_null under the true key (the data's empirical
    characteristic function at the score frequency; measured: shoppers
    rho_hat overshoot +0.16 at rho=0.05 with the mu_null=0 estimator).
    In the D2 setting the verifier holds the claimed-genuine source, so
    mu_null is calibrated on a disjoint genuine sample and the
    method-of-moments estimate uses both endpoints."""
    s = row_scores(z, key)
    gap = mu_ref - mu_null
    rho = float(np.clip((s.mean() - mu_null) / gap, 0.0, 1.0))
    se = float(s.std(ddof=1) / (np.sqrt(s.size) * abs(gap)))
    return {"rho_hat": rho, "se": se, "S_bar": float(s.mean()),
            "mu_ref": mu_ref, "mu_null": mu_null}


def group_scan(z: np.ndarray, key: Key, labels: np.ndarray) -> list[dict]:
    """Per-group mean scores with Hoeffding p-values (certified per group;
    Bonferroni across groups is the caller's choice and reported here)."""
    s = row_scores(z, key)
    labels = np.asarray(labels)
    out = []
    groups = np.unique(labels)
    for g in groups:
        sg = s[labels == g]
        Sg = float(sg.mean())
        p = float(min(1.0, np.exp(-0.5 * sg.size * max(Sg, 0.0) ** 2)))
        out.append({"group": g, "n_rows": int(sg.size), "S_bar": Sg,
                    "p_hoeffding": p,
                    "p_bonferroni": float(min(1.0, p * groups.size))})
    return out


def localize(z: np.ndarray, key: Key, mu_ref: float, q: float = 0.05,
             draws: int = 30, labels: np.ndarray | None = None,
             mu_null: float = 0.0,
             null_scores: np.ndarray | None = None,
             rng: np.random.Generator | None = None) -> dict:
    """One-call forensic report: flags, FDR level, rho_hat, group scan."""
    p = row_pvalues(z, key, draws, rng, null_scores=null_scores)
    flags = bh_flag(p, q)
    rep = {
        "row_pvalues": p,
        "flags": flags,
        "num_flagged": int(flags.sum()),
        "fdr_level": q,
        **estimate_rho(z, key, mu_ref, mu_null=mu_null),
    }
    if labels is not None:
        rep["groups"] = group_scan(z, key, labels)
    return rep
