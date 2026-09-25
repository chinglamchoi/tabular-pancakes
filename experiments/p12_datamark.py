"""P12 -- Data-space PancakeMark on real host output (diffusion-VAE fallback).

Marks the numeric block of UNMARKED host tables post hoc (QIM projection
onto the keyed pancake conditional in whitened space) and measures, per
seed:
  utility      whitened on-axis rms + mean cell RMSE (column-std units)
  detection    certified p-value (self-fit codec) on marked vs unmarked
  learnability C2ST marked-vs-unmarked AUC (the undetectability audit)
  robustness   attack sweep incl. EXACT affine invariance

Usage: python p12_datamark.py --host tabsyn [--quick] [--seeds N]
(Local validation: --host flow.)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import outdir, save_csv, aggregate  # noqa: E402
from hostglue import get_host, add_host_args, effective_seeds, run_tag  # noqa: E402

from pancakemark import keygen
from pancakemark.datamark import DataSpaceMark, detect_data_space
from pancakemark.tabular import TableCodec
from pancakemark import attacks as A


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
    # outdir tagged by operating point so strong/shallow runs never
    # overwrite each other's CSVs
    out = outdir(f"p12_datamark_{run_tag(args)}_g{args.gamma}_b{args.beta}")
    host = get_host(args.host, tabwak_root=args.tabwak_root,
                    dataname=args.dataname, device=args.device,
                    steps=args.steps)
    n = host.latent_dim
    m = 2000 if args.quick else 6000
    seeds = effective_seeds(args)

    head_rows, att_rows = [], []
    for seed in range(seeds):
        rng = np.random.default_rng(1200 + seed)
        # UNMARKED host output: the mark is applied post hoc at release
        X = host.numeric_view(host.generate(rng.standard_normal((m, n))))
        X1 = host.numeric_view(host.generate(rng.standard_normal((m, n))))
        # published, key-independent reference codec; select_columns keeps
        # the mark off zero-inflated / heavily discrete columns
        codec = TableCodec(select_columns=True).fit(X)
        d = codec.dim
        k = max(1, min(4, d // 2))
        key = keygen(d, k, gamma=args.gamma, beta=args.beta,
                     seed=7700 + seed)
        Xm, rep = DataSpaceMark(key, codec).mark(X, rng)

        # certified verdicts: exact MC key-rerandomization (the Hoeffding
        # bound's zero-mean null degrades on real data at low gamma)
        det_m = detect_data_space(Xm, key, codec=codec, method="mc",
                                  mc_draws=9999)
        det_0 = detect_data_space(X1, key, codec=codec, method="mc",
                                  mc_draws=9999)
        det_m_h = detect_data_space(Xm, key, codec=codec)   # analytic, for scale
        # key-holder self-fit: refit standardization + whitening on the
        # key's own column list (re-running the public eligibility rule
        # on a marked table flips boundary columns -- default: 14 -> 9)
        det_sf = detect_data_space(Xm, key, columns=codec.cols_)
        aucs = c2st(Xm, X1, seed=seed)
        head = {"seed": seed, "d_marked": d,
                "d_total": rep["total_columns"], "k": k, "m": m,
                "gamma": args.gamma, "beta": args.beta,
                "cell_rmse_stdunits": rep["mean_cell_rmse_stdunits"],
                "rms_onaxis_whitened": rep["rms_onaxis_whitened"],
                "stat_marked": det_m.stat,
                "pmc_marked": det_m.pvalue,
                "log10p_hoeffding_marked": log10p(det_m_h.pvalue),
                "stat_marked_selffit": det_sf.stat,
                "stat_unmarked": det_0.stat,
                "pmc_unmarked": det_0.pvalue,
                "c2st_logreg": aucs["logreg"], "c2st_hgb": aucs["hgb"]}
        head_rows.append(head)
        print(head)
        # MC floor: 1/(draws+1)=1e-4 -- but in tiny d_marked a random key
        # direction occasionally aligns with the true one (adult d=2:
        # observed marked p_mc up to ~2e-3), which is the honest certified
        # limit of a 2-column mark, not a detection failure.
        assert det_m.pvalue < 0.01, "marked table not detected (MC)"
        assert det_0.pvalue > 1e-3, "false signal on unmarked table (MC)"

        def att(name, intensity, Xa):
            r = detect_data_space(np.asarray(Xa, dtype=float), key,
                                  codec=codec)
            att_rows.append({"attack": name, "intensity": intensity,
                             "seed": seed, "stat": r.stat,
                             "log10p": log10p(r.pvalue)})
            print(f"s{seed} {name:>14s} {intensity!s:>5}: S={r.stat:.3f} "
                  f"p={r.pvalue:.2e}")

        att("none", 0, Xm)
        for frac in (0.02, 0.05, 0.10, 0.20):
            att("cell_noise", frac, A.gaussian_cell_noise(Xm, frac, rng))
        for dig in (3, 2):
            att("round_sig", dig, A.round_significant(Xm, dig))
        att("affine_rescale", 1, A.affine_rescale(Xm, rng))    # exact
        att("winsorize", 0.01, A.winsorize(Xm, 0.01))
        for rho in (0.3, 0.6):
            att("edit_rows", rho, A.edit_rows(Xm, rho, rng))
        att("subsample", 0.2, A.subsample_rows(Xm, 0.2, rng))
        att("reimpute", 0.05, A.reimpute_cells(Xm, 0.05, rng))

    save_csv(out / "headline_perseed.csv", head_rows)
    save_csv(out / "headline.csv", aggregate(
        head_rows, ["d_marked"], ["cell_rmse_stdunits", "stat_marked",
                           "log10p_hoeffding_marked", "stat_marked_selffit",
                           "stat_unmarked", "c2st_logreg", "c2st_hgb"]))
    save_csv(out / "attacks_perseed.csv", att_rows)
    save_csv(out / "attacks.csv", aggregate(
        att_rows, ["attack", "intensity"], ["stat", "log10p"]))


if __name__ == "__main__":
    ap = add_host_args(argparse.ArgumentParser())
    # The (gamma, beta) dial trades detection rows against learnability on
    # LOW-DIMENSIONAL numeric blocks (adult: d_eff=2). Strong default for
    # wide tables; run also with --gamma 1.0 --beta 0.3 on narrow ones
    # (adult, m=6000: p ~ 1e-25, HGB C2ST 0.58 vs 0.95 at full strength).
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--beta", type=float, default=0.05)
    main(ap.parse_args())
