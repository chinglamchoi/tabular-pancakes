"""P14 -- data-space PancakeMark across GENERATIVE MODELS.

The data-space variant needs only a model's sampled tables, so any
generator is a host. This evaluates one model's samples end to end:

  detection    certified MC + Hoeffding-scale p on marked vs unmarked
  learnability C2ST (logreg + HGB) marked vs an independent sample
  robustness   compact attack set incl. EXACT affine invariance
  D2 localize  marked rows mixed into REAL rows: AUC, rho_hat, FDR

Inputs are plain CSVs, so it works for TabDDPM/STaSy/CoDi/SMOTE/GOGGLE/
GREAT samples from the upstream tabsyn repo, for TabWak/TabSyn samples,
or for anything else with the dataset's schema.

Usage:
  python p14_models_matrix.py --label tabddpm \
      --pool-csv  tabsyn-upstream/synthetic/adult/tabddpm.csv \
      --real-csv  TabWak/data/adult/train.csv \
      --info-json TabWak/data/Info/adult.json [--seeds 5]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import outdir, save_csv, aggregate  # noqa: E402

from pancakemark import keygen
from pancakemark.datamark import DataSpaceMark, detect_data_space
from pancakemark.tabular import TableCodec
from pancakemark.localize import localize
from pancakemark.detect import row_scores
from pancakemark import attacks as A


def numeric_block(csv_path: str, num_col_idx: list[int]) -> np.ndarray:
    import pandas as pd
    df = pd.read_csv(csv_path)                     # sampled/processed: header row
    return df.iloc[:, num_col_idx].to_numpy(float)


def c2st(X0, X1, seed):
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


def log10p(p):
    return float(np.log10(max(p, 1e-300)))


def main(args):
    out = outdir(f"p14_models/{args.label}")
    info = json.load(open(args.info_json))
    num_idx = info["num_col_idx"]
    pool = numeric_block(args.pool_csv, num_idx)
    real = numeric_block(args.real_csv, num_idx)
    seeds = min(args.seeds, 2) if args.quick else args.seeds
    m = min(args.m if not args.quick else 2000, len(pool) // 2 - 50,
            len(real) - 50)
    if m < 1000:
        raise SystemExit(f"pool too small: pool={len(pool)} real={len(real)}")
    print(f"[{args.label}] pool={len(pool)} real={len(real)} m={m} "
          f"num_cols={len(num_idx)}")

    head_rows, att_rows, loc_rows = [], [], []
    for seed in range(seeds):
        rng = np.random.default_rng(1500 + seed)
        idx = rng.choice(len(pool), 2 * m, replace=False)
        X, X1 = pool[idx[:m]], pool[idx[m:]]
        codec = TableCodec(select_columns=True).fit(X)
        d = codec.dim
        k = max(1, min(4, d // 2))
        key = keygen(d, k, gamma=args.gamma, beta=args.beta, seed=7900 + seed)
        Xm, rep = DataSpaceMark(key, codec).mark(X, rng)

        det_mc = detect_data_space(Xm, key, codec=codec, method="mc",
                                   mc_draws=4999)
        det_h = detect_data_space(Xm, key, codec=codec)
        det0 = detect_data_space(X1, key, codec=codec, method="mc",
                                 mc_draws=4999)
        aucs = c2st(Xm, X1, seed)
        head = {"model": args.label, "seed": seed, "d_marked": d,
                "d_total": len(num_idx), "k": k, "m": m,
                "gamma": args.gamma, "beta": args.beta,
                "cell_rmse_stdunits": rep["mean_cell_rmse_stdunits"],
                "stat_marked": det_h.stat,
                "log10p_hoeffding": log10p(det_h.pvalue),
                "pmc_marked": det_mc.pvalue,
                "stat_unmarked": det0.stat, "pmc_unmarked": det0.pvalue,
                "c2st_logreg": aucs["logreg"], "c2st_hgb": aucs["hgb"]}
        head_rows.append(head); print(head)
        assert det_mc.pvalue < 0.01, "marked not detected (MC)"
        assert det0.pvalue > 1e-3, "false signal on unmarked (MC)"

        for name, Xa in [("none", Xm),
                         ("cell_noise.05", A.gaussian_cell_noise(Xm, 0.05, rng)),
                         ("round_sig2", A.round_significant(Xm, 2)),
                         ("affine", A.affine_rescale(Xm, rng)),
                         ("subsample.2", A.subsample_rows(Xm, 0.2, rng))]:
            r = detect_data_space(np.asarray(Xa, float), key, codec=codec)
            att_rows.append({"model": args.label, "attack": name,
                             "seed": seed, "stat": r.stat,
                             "log10p": log10p(r.pvalue)})

        # D2: marked model rows mixed into REAL rows (published codec).
        # rho_hat baseline mu_null is calibrated on a DISJOINT genuine
        # sample (real rows have nonzero coherence under the true key).
        from scipy.stats import rankdata
        u_ref = codec.transform(Xm)
        mu_ref = float(row_scores(u_ref, key).mean())
        for rho in (0.1, 0.3):
            ns = int(rho * m)
            perm_pool = rng.permutation(len(real))
            gen = real[perm_pool[: m - ns]]
            base = real[perm_pool[m - ns: (m - ns) +
                                  min(4 * m, len(real) - (m - ns))]]
            base_scores = row_scores(codec.transform(base), key)
            mu_null = float(base_scores.mean())
            mixed = np.vstack([gen, Xm[:ns]])
            truth = np.r_[np.zeros(m - ns), np.ones(ns)].astype(bool)
            p_ = rng.permutation(m); mixed, truth = mixed[p_], truth[p_]
            repL = localize(codec.transform(mixed), key, mu_ref=mu_ref,
                            q=0.05, mu_null=mu_null,
                            null_scores=base_scores, rng=rng)
            r_ = rankdata(-repL["row_pvalues"]); npos = int(truth.sum())
            auc = float((r_[truth].mean() - (npos + 1) / 2) / (m - npos))
            f = repL["flags"]
            loc_rows.append({"model": args.label, "rho": rho, "seed": seed,
                             "auc": auc,
                             "fdr": (f & ~truth).sum() / max(f.sum(), 1),
                             "power": (f & truth).sum() / max(npos, 1),
                             "rho_hat": repL["rho_hat"]})
            print(f"  s{seed} D2 rho={rho}: AUC={auc:.3f} "
                  f"rho_hat={repL['rho_hat']:.3f}")

    save_csv(out / "headline_perseed.csv", head_rows)
    save_csv(out / "headline.csv", aggregate(
        head_rows, ["model"],
        ["d_marked", "cell_rmse_stdunits", "stat_marked", "log10p_hoeffding",
         "stat_unmarked", "c2st_logreg", "c2st_hgb"]))
    save_csv(out / "attacks_perseed.csv", att_rows)
    save_csv(out / "attacks.csv", aggregate(att_rows, ["model", "attack"],
                                            ["stat", "log10p"]))
    save_csv(out / "localize_perseed.csv", loc_rows)
    save_csv(out / "localize.csv", aggregate(
        loc_rows, ["model", "rho"], ["auc", "fdr", "power", "rho_hat"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--pool-csv", required=True)
    ap.add_argument("--real-csv", required=True)
    ap.add_argument("--info-json", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--m", type=int, default=6000)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args())
