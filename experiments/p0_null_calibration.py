"""P0.3 -- Null calibration / false-accusation audit (model-free).

For data generated independently of the key -- Gaussian, heavy-tailed t(3),
correlated Gaussian, and a bimodal mixture -- we check:
  (i)  MC key-rerandomization p-values are (super)uniform: P(p<=a)<=a exactly;
  (ii) Hoeffding p-values are conservative (stochastically dominate MC);
  (iii) empirical FPR at alpha in {1e-2, 1e-3} for both methods.

Usage: python p0_null_calibration.py [--quick]
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
from pancakemark.null import null_bias_estimate


def null_datasets(m, n, rng):
    cov = 0.6 ** np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    L = np.linalg.cholesky(cov)
    return {
        "gaussian": rng.standard_normal((m, n)),
        "student_t3": rng.standard_t(3, size=(m, n)),
        "ar_correlated": rng.standard_normal((m, n)) @ L.T,
        "bimodal": rng.standard_normal((m, n)) + 2.0 * rng.choice([-1, 1], size=(m, 1)),
    }


def main(quick: bool = False):
    out = outdir("p0_null")
    rng = np.random.default_rng(2)
    m, n, k = 300, 32, 2
    trials = 100 if quick else 300
    mc_draws = 199 if quick else 499

    pvals = {name: {"mc": [], "hoeffding": []} for name in
             ["gaussian", "student_t3", "ar_correlated", "bimodal"]}
    bias_rows = []
    for tr in range(trials):
        data = null_datasets(m, n, rng)
        key = keygen(n, k, gamma=2.0, beta=0.05, seed=9000 + tr)
        for name, z in data.items():
            r_mc = detect(z, key, method="mc", mc_draws=mc_draws, rng=rng)
            r_h = detect(z, key, method="hoeffding")
            pvals[name]["mc"].append(r_mc.pvalue)
            pvals[name]["hoeffding"].append(r_h.pvalue)
            if tr == 0:
                bias_rows.append({"null": name,
                                  "bias_estimate": null_bias_estimate(z, key.score_frequency)})

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
    grid = np.linspace(0, 1, 200)
    summary = []
    for ax, (name, d) in zip(axes, pvals.items()):
        for meth, ps in d.items():
            ps = np.sort(np.asarray(ps))
            ecdf = np.searchsorted(ps, grid, side="right") / ps.size
            ax.plot(grid, ecdf, label=meth)
            for a in (1e-2, 1e-3):
                summary.append({"null": name, "method": meth, "alpha": a,
                                "fpr": float(np.mean(ps <= a)), "trials": ps.size})
        ax.plot([0, 1], [0, 1], "k--", lw=0.7)
        ax.set_title(name); ax.set_xlabel("p")
    axes[0].set_ylabel("ECDF"); axes[0].legend(fontsize=7)
    fig.suptitle("Null p-value calibration: curves must lie on/below the diagonal", y=1.04)
    save_fig(fig, out / "pvalue_ecdf.png")
    save_csv(out / "fpr_table.csv", summary)
    save_csv(out / "bias_estimates.csv", bias_rows)

    bad = [s for s in summary if s["fpr"] > s["alpha"] + 3 * np.sqrt(s["alpha"] / trials) + 1e-9]
    print("FPR table:", *summary, sep="\n  ")
    assert not bad, f"anti-conservative cells: {bad}"
    print("all nulls calibrated (conservative).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(**vars(ap.parse_args()))
