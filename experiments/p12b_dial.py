"""P12b -- the (gamma, beta) dial on a narrow real numeric block.

Sweeps operating points on the REAL Adult numeric columns (the paper's
detection-vs-learnability tradeoff figure for narrow tables): certified
detection scale vs HGB C2ST AUC vs utility cost, mean +- std over seeds.

Usage: python p12b_dial.py --adult-csv <path to adult .csv (headerless,
       UCI column order)> [--seeds N] [--quick]
On the cluster, TabWak/data/adult/adult.data works if present, else any
full adult csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import outdir, save_csv, save_fig, aggregate  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from pancakemark import keygen, theory
from pancakemark.datamark import DataSpaceMark, detect_data_space
from pancakemark.tabular import TableCodec


def hgb_auc(X0, X1, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    X = np.vstack([X0, X1]); y = np.r_[np.zeros(len(X0)), np.ones(len(X1))]
    return float(np.mean(cross_val_score(
        HistGradientBoostingClassifier(random_state=seed), X, y, cv=5,
        scoring="roc_auc", n_jobs=-1)))


POINTS = [(2.0, 0.05), (1.5, 0.15), (1.25, 0.20), (1.0, 0.20),
          (1.0, 0.30), (0.8, 0.30)]


def main(args):
    import pandas as pd
    out = outdir("p12b_dial")
    df = pd.read_csv(args.adult_csv, header=None)
    num = df[[0, 2, 4, 10, 11, 12]].to_numpy(float)
    m = 2000 if args.quick else 6000
    seeds = min(args.seeds, 2) if args.quick else args.seeds

    rows = []
    for gamma, beta in POINTS:
        for seed in range(seeds):
            rng = np.random.default_rng(1200 + seed)
            X = num[rng.choice(len(num), m, replace=False)]
            X1 = num[rng.choice(len(num), m, replace=False)]
            codec = TableCodec(select_columns=True).fit(X)
            key = keygen(codec.dim, 1, gamma=gamma, beta=beta,
                         seed=7700 + seed)
            Xm, rep = DataSpaceMark(key, codec).mark(X, rng)
            det_h = detect_data_space(Xm, key, codec=codec)
            det_mc = detect_data_space(Xm, key, codec=codec, method="mc",
                                       mc_draws=9999)
            rows.append({
                "gamma": gamma, "beta": beta, "seed": seed, "m": m,
                "mu_theory": theory.mu(gamma, beta),
                "S": det_h.stat,
                "log10p_hoeffding": float(np.log10(max(det_h.pvalue, 1e-300))),
                "p_mc": det_mc.pvalue,
                "hgb_auc": hgb_auc(Xm, X1, seed),
                "cell_rmse": rep["mean_cell_rmse_stdunits"],
            })
            print(rows[-1])
    save_csv(out / "dial_perseed.csv", rows)
    agg = aggregate(rows, ["gamma", "beta"],
                    ["mu_theory", "S", "log10p_hoeffding", "p_mc",
                     "hgb_auc", "cell_rmse"])
    save_csv(out / "dial.csv", agg)

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    xs = [-r["log10p_hoeffding_mean"] for r in agg]
    ys = [r["hgb_auc_mean"] for r in agg]
    ax.errorbar(xs, ys, xerr=[r["log10p_hoeffding_std"] for r in agg],
                yerr=[r["hgb_auc_std"] for r in agg], fmt="o-")
    for r, x, y in zip(agg, xs, ys):
        ax.annotate(f"({r['gamma']},{r['beta']})", (x, y), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.axhline(0.5, color="k", ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel(r"detection evidence $-\log_{10} p$ (analytic scale, $m$=6000)")
    ax.set_ylabel("HGB C2ST AUC (learnability)")
    ax.set_title(f"The $(\\gamma,\\beta)$ dial on a 2-column block "
                 f"({seeds} seeds)")
    save_fig(fig, out / "dial_curve.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--adult-csv", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args())
