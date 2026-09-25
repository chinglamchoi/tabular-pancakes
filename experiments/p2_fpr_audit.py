"""P2.1 -- False-accusation audit on real datasets (model-free path).

For real tables that were generated independently of any key, run the
codec + detector with many fresh keys and verify the advertised FPR:
empirical P(p <= alpha) <= alpha for both the MC certificate and the
Hoeffding bound. This is the panel an integrity office cares about most.

Datasets: sklearn's bundled real datasets (no network needed) plus
synthetic stress families (heavy tails, strong correlation, bimodality).

Usage: python p2_fpr_audit.py [--quick]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import outdir, save_csv, save_fig  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from pancakemark import keygen, detect
from pancakemark.tabular import TableCodec


def real_datasets():
    from sklearn.datasets import load_breast_cancer, load_wine, load_diabetes
    out = {
        "breast_cancer": load_breast_cancer().data,       # 569 x 30
        "wine": load_wine().data,                          # 178 x 13
        "diabetes": load_diabetes(scaled=False).data,      # 442 x 10
    }
    rng = np.random.default_rng(0)
    n = 16
    cov = 0.7 ** np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    L = np.linalg.cholesky(cov)
    out["synthetic_lognormal"] = np.exp(rng.standard_normal((500, n)) * 0.8)
    out["synthetic_correlated_t3"] = rng.standard_t(3, size=(500, n)) @ L.T
    return out


def main(quick: bool = False):
    out = outdir("p2_fpr_audit")
    rng = np.random.default_rng(5)
    trials = 100 if quick else 400
    mc_draws = 199 if quick else 499
    gamma, beta = 2.0, 0.05

    datasets = real_datasets()
    pvals = {name: {"mc": [], "hoeffding": []} for name in datasets}
    for name, X in datasets.items():
        zhat = TableCodec().fit_transform(X)          # self-fit canonicalization
        n = zhat.shape[1]
        k = max(1, min(4, n // 4))
        for tr in range(trials):
            key = keygen(n, k, gamma=gamma, beta=beta, seed=50_000 + tr)
            r_mc = detect(zhat, key, method="mc", mc_draws=mc_draws, rng=rng)
            r_h = detect(zhat, key, method="hoeffding")
            pvals[name]["mc"].append(r_mc.pvalue)
            pvals[name]["hoeffding"].append(r_h.pvalue)
        print(f"{name:>24s}: min mc p = {min(pvals[name]['mc']):.4f}  "
              f"min hoeffding p = {min(pvals[name]['hoeffding']):.3g}")

    summary = []
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    grid = np.linspace(0, 1, 400)
    for name, d in pvals.items():
        for meth, ps in d.items():
            ps = np.sort(np.asarray(ps))
            for a in (1e-2, 1e-3):
                summary.append({"dataset": name, "method": meth, "alpha": a,
                                "fpr": float(np.mean(ps <= a)),
                                "trials": ps.size})
            if meth == "mc":
                ecdf = np.searchsorted(ps, grid, side="right") / ps.size
                ax.plot(grid, ecdf, lw=1, label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="uniform (validity bound)")
    ax.set_xlabel("p"); ax.set_ylabel("ECDF of MC p-values")
    ax.set_title(f"False-accusation audit: {trials} fresh keys per dataset")
    ax.legend(fontsize=6.5)
    save_fig(fig, out / "mc_pvalue_ecdf.png")
    save_csv(out / "fpr_table.csv", summary)

    bad = [s for s in summary
           if s["fpr"] > s["alpha"] + 3 * np.sqrt(s["alpha"] / s["trials"]) + 1e-9]
    assert not bad, f"anti-conservative cells: {bad}"
    print("audit passed: FPR <= alpha on every dataset/method cell.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(**vars(ap.parse_args()))
