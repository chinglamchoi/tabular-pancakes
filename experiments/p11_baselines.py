"""P11 -- Baseline comparison: PancakeMark vs Gaussian Shading / TabWak /
TabWak* on the SAME host and rows.

For each scheme:
  detection      its own detector's separation (bit accuracy for the
                 baselines vs their unmarked chance level; certified
                 p-value for PancakeMark)
  learnability   C2ST AUC of a classifier told to find marked outputs
                 (XGB-style HGB + logreg, 5-fold) -- THE undetectability
                 audit: ~0.5 for PancakeMark, >0.5 for sign/quantile-fixing
                 schemes (this is the killer figure).

Replicated over --seeds independent keys / draws; the CSV and figure
report mean +/- std across seeds.

Usage: python p11_baselines.py --host tabsyn [--quick] [--seeds N]
(Local validation: --host flow.)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import outdir, save_csv, save_fig, aggregate  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from hostglue import get_host, add_host_args, effective_seeds, run_tag  # noqa: E402

from pancakemark import keygen, sample_latents, detect
from pancakemark.baselines import GaussianShading, TabWak, tabwak_star


def c2st_auc(X0, X1, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    X = np.vstack([X0, X1]); y = np.r_[np.zeros(len(X0)), np.ones(len(X1))]
    out = {}
    for name, clf in [("logreg", make_pipeline(StandardScaler(),
                                               LogisticRegression(max_iter=2000))),
                      ("hgb", HistGradientBoostingClassifier(random_state=seed))]:
        out[name] = float(np.mean(cross_val_score(clf, X, y, cv=5,
                                                  scoring="roc_auc", n_jobs=-1)))
    return out


def main(args):
    out = outdir(f"p11_baselines_{run_tag(args)}")
    host = get_host(args.host, tabwak_root=args.tabwak_root,
                    dataname=args.dataname, device=args.device, steps=args.steps,
                    refine_iters=args.refine_iters,
                    fp_iters=args.fp_iters)
    n = host.latent_dim
    m = 3000 if args.quick else 8000
    seeds = effective_seeds(args)

    rows = []
    for seed in range(seeds):
        key = keygen(n, k=8, gamma=2.0, beta=0.05, seed=7300 + seed)
        gs = GaussianShading(n, seed=217 + seed)
        tw = TabWak(n, seed=217 + seed) if n % 2 == 0 else None
        tws = tabwak_star(n, seed=217 + seed) if n % 2 == 0 else None
        schemes = {
            "pancakemark": lambda r: sample_latents(m, key, r),
            "gaussian_shading": lambda r: gs.sample_latents(m, r),
        }
        if tw:
            schemes["tabwak"] = lambda r: tw.sample_latents(m, r)
            schemes["tabwak_star"] = lambda r: tws.sample_latents(m, r)

        rng = np.random.default_rng(23 + seed)
        Xu = host.numeric_view(host.generate(rng.standard_normal((m, n))))
        for name, sampler in schemes.items():
            r = np.random.default_rng(500 + 10 * seed)
            z = sampler(r)
            table = host.generate(z)
            Xm = host.numeric_view(table)
            zhat = host.invert(table)

            # scheme-native detection (uniform fields across schemes)
            det = {"detect_stat": np.nan, "detect_pvalue": np.nan,
                   "detect_null_acc": np.nan}
            if name == "pancakemark":
                res = detect(zhat, key, method="hoeffding")
                det.update(detect_stat=res.stat, detect_pvalue=res.pvalue)
            else:
                obj = {"gaussian_shading": gs, "tabwak": tw,
                       "tabwak_star": tws}[name]
                zh0 = host.invert(host.generate(
                    np.random.default_rng(501 + 10 * seed)
                    .standard_normal((m, n))))
                det.update(detect_stat=obj.bit_accuracy(zhat),
                           detect_null_acc=obj.bit_accuracy(zh0))

            aucs = c2st_auc(Xm, Xu, seed=1 + seed)
            row = {"scheme": name, "seed": seed, **det,
                   "c2st_logreg": aucs["logreg"], "c2st_hgb": aucs["hgb"],
                   "m": m}
            rows.append(row)
            print(row)
    save_csv(out / "baselines_perseed.csv", rows)
    agg = aggregate(rows, ["scheme"],
                    ["detect_stat", "detect_null_acc",
                     "c2st_logreg", "c2st_hgb"])
    save_csv(out / "baselines.csv", agg)

    fig = plt.figure(figsize=(6.2, 3.6))
    names = [r["scheme"] for r in agg]
    x = np.arange(len(agg))
    plt.bar(x - 0.18, [r["c2st_logreg_mean"] for r in agg], 0.36,
            yerr=[r["c2st_logreg_std"] for r in agg], capsize=3,
            label="C2ST logreg")
    plt.bar(x + 0.18, [r["c2st_hgb_mean"] for r in agg], 0.36,
            yerr=[r["c2st_hgb_std"] for r in agg], capsize=3,
            label="C2ST HGB")
    plt.axhline(0.5, color="k", ls="--", lw=0.8, label="undetectable (0.5)")
    plt.xticks(x, names, rotation=10, fontsize=8)
    plt.ylabel("marked-vs-unmarked AUC")
    plt.title(f"Learnability audit ({args.host} host, m={m}, {seeds} seeds)")
    plt.legend(fontsize=8)
    save_fig(fig, out / "learnability.png")

    pm = [r for r in agg if r["scheme"] == "pancakemark"][0]
    assert abs(pm["c2st_logreg_mean"] - 0.5) < 0.03, "PancakeMark learnable?!"


if __name__ == "__main__":
    ap = add_host_args(argparse.ArgumentParser())
    main(ap.parse_args())
