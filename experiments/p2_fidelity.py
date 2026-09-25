"""P2.2 -- Fidelity / distortion-freeness suite: marked vs unmarked outputs.

Undetectability predicts every efficiently computable statistic matches.
We test exactly that, on generator outputs (TanhFlowHost round trip):
  - per-column shape: max/mean KS statistic, mean W1 (in column-std units)
  - dependence: max |corr(marked) - corr(unmarked)|
  - classifier two-sample test (C2ST): 5-fold AUC of logistic regression
    and gradient boosting on marked-vs-unmarked rows -> should be ~ 0.5.
Positive control: a Gaussian-Shading-style sign-fixed prior (half-Gaussian
on a fixed coordinate) MUST be caught by the same C2ST -- validating that
the test has teeth, so our ~0.5 is evidence, not blindness.

Paired seeds across replicates give CIs. Usage: python p2_fidelity.py [--quick]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from common import outdir, save_csv  # noqa: E402

from pancakemark import keygen, sample_latents
from pancakemark.hosts import TanhFlowHost


def c2st_auc(X0, X1, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    X = np.vstack([X0, X1])
    y = np.r_[np.zeros(len(X0)), np.ones(len(X1))]
    aucs = {}
    for name, clf in [
        ("logreg", make_pipeline(StandardScaler(),
                                 LogisticRegression(max_iter=2000))),
        ("hgb", HistGradientBoostingClassifier(random_state=seed)),
    ]:
        s = cross_val_score(clf, X, y, cv=5, scoring="roc_auc", n_jobs=-1)
        aucs[name] = float(np.mean(s))
    return aucs


def column_metrics(X0, X1):
    ks = [stats.ks_2samp(X0[:, j], X1[:, j]).statistic for j in range(X0.shape[1])]
    w1 = [stats.wasserstein_distance(X0[:, j], X1[:, j]) / (X0[:, j].std() + 1e-12)
          for j in range(X0.shape[1])]
    c0 = np.corrcoef(X0, rowvar=False)
    c1 = np.corrcoef(X1, rowvar=False)
    return {
        "ks_max": float(np.max(ks)), "ks_mean": float(np.mean(ks)),
        "w1_mean_stdunits": float(np.mean(w1)),
        "corr_maxabsdiff": float(np.max(np.abs(c0 - c1))),
    }


def signfixed_latents(m, n, rng):
    """Gaussian-Shading-style positive control: coordinate 0 folded positive."""
    z = rng.standard_normal((m, n))
    z[:, 0] = np.abs(z[:, 0])
    return z


def main(quick: bool = False):
    out = outdir("p2_fidelity")
    n, k, gamma, beta = 64, 4, 2.0, 0.05
    m = 5000 if quick else 20000
    reps = 3 if quick else 5
    host = TanhFlowHost(n, layers=3, seed=60)

    rows = []
    for rep in range(reps):
        rng = np.random.default_rng(70 + rep)
        key = keygen(n, k, gamma, beta, seed=80 + rep)
        X_marked = host.generate(sample_latents(m, key, rng, marked=True))
        X_plain = host.generate(rng.standard_normal((m, n)))
        X_plain2 = host.generate(rng.standard_normal((m, n)))   # null-vs-null baseline
        X_ctrl = host.generate(signfixed_latents(m, n, rng))    # positive control

        for label, XA, XB in [
            ("marked_vs_unmarked", X_marked, X_plain),
            ("unmarked_vs_unmarked", X_plain, X_plain2),
            ("signfixed_vs_unmarked(control)", X_ctrl, X_plain),
        ]:
            met = column_metrics(XA, XB)
            met.update({f"c2st_{kk}": vv for kk, vv in
                        c2st_auc(XA, XB, seed=rep).items()})
            met.update({"pair": label, "rep": rep, "m": m})
            rows.append(met)
            print(f"rep{rep} {label:>32s}  ks_max={met['ks_max']:.4f}  "
                  f"corr_maxdiff={met['corr_maxabsdiff']:.4f}  "
                  f"AUC(logreg)={met['c2st_logreg']:.3f}  "
                  f"AUC(hgb)={met['c2st_hgb']:.3f}")
    save_csv(out / "fidelity.csv", rows)

    # aggregate check: marked-vs-unmarked must be statistically identical to
    # the unmarked-vs-unmarked baseline; the control must separate.
    def agg(label, field):
        return np.array([r[field] for r in rows if r["pair"] == label])
    for field in ("c2st_logreg", "c2st_hgb"):
        a = agg("marked_vs_unmarked", field)
        b = agg("unmarked_vs_unmarked", field)
        c = agg("signfixed_vs_unmarked(control)", field)
        print(f"{field}: marked {a.mean():.3f}+-{a.std():.3f}  "
              f"baseline {b.mean():.3f}+-{b.std():.3f}  control {c.mean():.3f}")
        assert abs(a.mean() - 0.5) < 0.02, f"{field}: marked data detectable!"
        assert c.mean() > 0.55, f"{field}: control not detected -- test blind?"
    print("fidelity suite passed: mark invisible, control caught.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(**vars(ap.parse_args()))
