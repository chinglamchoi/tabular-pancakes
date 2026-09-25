"""P5 -- Localization under mixing (scenario D2).

rho-mixtures of real (Gaussian-null) and marked rows: dataset-level
detection threshold, row-level AUC and realized FDR at BH level q,
rho_hat accuracy; plus the "imputed arm" group-scan case study.

Usage: python p5_localization.py [--quick]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import outdir, save_csv, save_fig  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from pancakemark import keygen, sample_latents, theory
from pancakemark.detect import row_scores
from pancakemark.localize import row_pvalues, bh_flag, estimate_rho, group_scan


def auc(scores, y):
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, float); ranks[order] = np.arange(1, len(order) + 1)
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main(quick: bool = False):
    out = outdir("p5_localization")
    rng = np.random.default_rng(7)
    n, k, gamma, beta = 64, 4, 2.0, 0.05
    m = 1500 if quick else 4000
    reps = 5 if quick else 15
    q = 0.05
    mu_ref = theory.mu(gamma, beta)
    rhos = [0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8]

    rows = []
    for rho in rhos:
        for rep in range(reps):
            key = keygen(n, k, gamma, beta, seed=95_000 + 100 * int(rho * 100) + rep)
            z = rng.standard_normal((m, n))
            n_marked = int(round(rho * m))
            idx = rng.choice(m, n_marked, replace=False)
            z[idx] = sample_latents(n_marked, key, rng)
            y = np.zeros(m, bool); y[idx] = True

            s = row_scores(z, key)
            p = row_pvalues(z, key, draws=25, rng=rng)
            flags = bh_flag(p, q=q)
            est = estimate_rho(z, key, mu_ref)
            rows.append({
                "rho": rho, "rep": rep, "m": m,
                "row_auc": auc(s, y),
                "power": float(flags[y].mean()) if y.any() else 0.0,
                "fdr": float((~y[flags]).mean()) if flags.any() else 0.0,
                "num_flagged": int(flags.sum()),
                "rho_hat": est["rho_hat"], "rho_se": est["se"],
            })
        rr = [r for r in rows if r["rho"] == rho]
        print(f"rho={rho:>5}: AUC={np.mean([r['row_auc'] for r in rr]):.3f}  "
              f"power={np.mean([r['power'] for r in rr]):.3f}  "
              f"FDR={np.mean([r['fdr'] for r in rr]):.3f}  "
              f"rho_hat={np.mean([r['rho_hat'] for r in rr]):.3f}")
    save_csv(out / "mixture_localization.csv", rows)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.5))
    for field, label in (("row_auc", "row AUC"), ("power", f"power @ FDR {q}"),
                         ("fdr", "realized FDR")):
        means = [np.mean([r[field] for r in rows if r["rho"] == rho]) for rho in rhos]
        a1.plot(rhos, means, "o-", label=label)
    a1.axhline(q, ls=":", lw=0.8, color="k")
    a1.set_xscale("log"); a1.set_xlabel(r"mixing rate $\rho$"); a1.legend(fontsize=7)
    a1.set_title(f"Row-level localization (m={m})")
    mh = [np.mean([r["rho_hat"] for r in rows if r["rho"] == rho]) for rho in rhos]
    sd = [np.std([r["rho_hat"] for r in rows if r["rho"] == rho]) for rho in rhos]
    a2.errorbar(rhos, mh, yerr=sd, fmt="o-")
    a2.plot([0, 1], [0, 1], "k--", lw=0.8)
    a2.set_xlabel(r"true $\rho$"); a2.set_ylabel(r"$\hat\rho$")
    a2.set_title("Mixing-rate estimation")
    save_fig(fig, out / "localization.png")

    # ---- case study: one imputed arm ------------------------------------
    key = keygen(n, k, gamma, beta, seed=97_001)
    arm = np.repeat(["control", "treatment", "followup"], m // 3)
    z = rng.standard_normal((arm.size, n))
    sel = arm == "treatment"
    # 60% of the treatment arm imputed
    tidx = np.nonzero(sel)[0]
    imp = rng.choice(tidx, int(0.6 * tidx.size), replace=False)
    z[imp] = sample_latents(imp.size, key, rng)
    scan = group_scan(z, key, arm)
    save_csv(out / "arm_case_study.csv", scan)
    for r in scan:
        print(f"arm {r['group']:>10s}: S_bar={r['S_bar']:.3f}  "
              f"p_bonf={r['p_bonferroni']:.3g}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(**vars(ap.parse_args()))
