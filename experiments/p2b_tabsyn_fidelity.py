"""P2b -- Fidelity + forensics on a real host (TabSyn) or the flow control.

(i)  C2ST marked-vs-unmarked on the numeric block of generated tables
     (+ unmarked-vs-unmarked baseline), as in p2_fidelity.
(ii) Forensic-profile comparison: real training table vs marked generator
     output through the Benford/terminal-digit panel -- the honest P4
     upgrade (does TabSyn's preprocessing reproduce recorded precision?).

Usage: python p2b_tabsyn_fidelity.py --host tabsyn [--quick]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import outdir, save_csv  # noqa: E402
from hostglue import get_host, add_host_args, run_tag  # noqa: E402

from pancakemark import keygen, sample_latents
from pancakemark.forensics import forensic_panel


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


def main(args):
    out = outdir(f"p2b_fidelity_{run_tag(args)}")
    rng = np.random.default_rng(21)
    host = get_host(args.host, tabwak_root=args.tabwak_root,
                    dataname=args.dataname, device=args.device, steps=args.steps,
                    refine_iters=args.refine_iters,
                    fp_iters=args.fp_iters)
    n = host.latent_dim
    m = 3000 if args.quick else 10000
    reps = 2 if args.quick else 5
    key = keygen(n, k=8, gamma=2.0, beta=0.05, seed=7100)

    rows = []
    for rep in range(reps):
        r = np.random.default_rng(400 + rep)
        Xm = host.numeric_view(host.generate(sample_latents(m, key, r)))
        Xp = host.numeric_view(host.generate(r.standard_normal((m, n))))
        Xp2 = host.numeric_view(host.generate(r.standard_normal((m, n))))
        for label, A, B in [("marked_vs_unmarked", Xm, Xp),
                            ("unmarked_vs_unmarked", Xp, Xp2)]:
            met = c2st(A, B, seed=rep); met.update({"pair": label, "rep": rep, "m": m})
            rows.append(met)
            print(f"rep{rep} {label:>24s}  AUC(logreg)={met['logreg']:.3f}  "
                  f"AUC(hgb)={met['hgb']:.3f}")
    save_csv(out / "c2st.csv", rows)

    a = np.array([r["logreg"] for r in rows if r["pair"] == "marked_vs_unmarked"])
    b = np.array([r["logreg"] for r in rows if r["pair"] == "unmarked_vs_unmarked"])
    print(f"marked {a.mean():.3f}+-{a.std():.3f} vs baseline {b.mean():.3f}+-{b.std():.3f}")
    assert abs(a.mean() - 0.5) < 0.03, "marked output detectable by C2ST!"

    # forensic profile: real train table vs marked output (tabsyn only)
    if args.host == "tabsyn":
        import pandas as pd
        real = pd.read_csv(Path(args.tabwak_root) / "data" / args.dataname / "train.csv")
        Xr = real.select_dtypes("number").to_numpy(dtype=float)
        prof = [{"table": "real_train", **_flat(forensic_panel(Xr, decimals=2))},
                {"table": "marked_tabsyn", **_flat(forensic_panel(Xm, decimals=2))}]
        save_csv(out / "forensic_profile.csv", prof)
        for p in prof:
            print(p)


def _flat(panel):
    out = {}
    for test, res in panel.items():
        out[f"{test}_cols"] = res.get("num_cols", 0)
        out[f"{test}_min_p_bonf"] = res.get("min_p_bonferroni", float("nan"))
    return out


if __name__ == "__main__":
    ap = add_host_args(argparse.ArgumentParser())
    main(ap.parse_args())
