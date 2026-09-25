"""P3 -- Robustness suite: detection vs attack intensity, with theory overlays.

Attacks on the released table of a TanhFlowHost (exact-inverse generator):
  cell noise (% of column std), rounding (significant digits), winsorize,
  MICE re-imputation, row edits, few-cell edits; invariances (shuffle,
  subsample, affine rescale) verified exactly.

For value attacks we overlay the Gaussian-model prediction computed from
the *measured* on-axis residual, so the plot shows both the raw fact
(empirical mu) and the theory's account of it.

Usage: python p3_robustness.py [--quick]
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
from pancakemark.null import hoeffding_pvalue
from pancakemark.hosts import TanhFlowHost
from pancakemark import attacks as A


def mu_and_tpr(host, key, X, z_true, trials_rows=400, alpha=1e-6, reps=20,
               rng=None):
    zhat = host.invert(X)
    s_all = row_scores(zhat, key)
    mu = float(s_all.mean())
    # on-axis residual for the Gaussian-model overlay
    r = (zhat - z_true) @ key.W
    sig = float(np.sqrt(np.mean(r ** 2)))
    # TPR at alpha from row-subsampled replicates
    hits = 0
    for _ in range(reps):
        idx = rng.choice(s_all.size, trials_rows, replace=False)
        if hoeffding_pvalue(float(s_all[idx].mean()), trials_rows) <= alpha:
            hits += 1
    return mu, theory.mu(key.gamma, key.beta, noise_std=sig), hits / reps, sig


def main(quick: bool = False, seeds: int = 3):
    out = outdir("p3_robustness")
    n, k, gamma, beta = 64, 4, 2.0, 0.05
    m = 4000 if quick else 12000
    if quick:
        seeds = min(seeds, 2)
    host = TanhFlowHost(n, layers=3, seed=90)
    rows = []
    for seed in range(seeds):
        rng = np.random.default_rng(600 + seed)
        key = keygen(n, k, gamma, beta, seed=910 + seed)
        z = sample_latents(m, key, rng)
        X0 = host.generate(z)

        def record(attack, intensity, X, note=""):
            mu, mu_pred, tpr, sig = mu_and_tpr(host, key, X, z, rng=rng)
            rows.append({"attack": attack, "intensity": intensity, "seed": seed,
                         "mu_emp": mu, "mu_pred_from_residual": mu_pred,
                         "tpr@1e-6(400rows)": tpr, "sigma_onaxis": sig,
                         "note": note})
            print(f"s{seed} {attack:>16s} {intensity!s:>6}: mu={mu:.3f} "
                  f"(pred {mu_pred:.3f})  TPR={tpr:.2f}")

        record("none", 0, X0)
        for frac in (0.01, 0.02, 0.05, 0.10, 0.20):
            record("cell_noise", frac, A.gaussian_cell_noise(X0, frac, rng))
        for digits in (4, 3, 2):
            record("round_sig", digits, A.round_significant(X0, digits))
        for q in (0.01, 0.05):
            record("winsorize", q, A.winsorize(X0, q))
        for frac in (0.05, 0.15):
            record("reimpute", frac, A.reimpute_cells(X0, frac, rng))
        for rho in (0.2, 0.4, 0.6, 0.8):
            record("edit_rows", rho, A.edit_rows(X0, rho, rng))
        record("few_cells", 50, A.few_cell_edits(X0, 50, 2.0, rng))

    # exact invariance check on the last seed's table
    from pancakemark.detect import global_stat
    S0 = global_stat(host.invert(X0), key)
    Ssh = global_stat(host.invert(A.shuffle_rows(X0, rng)), key)
    assert abs(S0 - Ssh) < 1e-12, "shuffle invariance violated"
    save_csv(out / "robustness_perseed.csv", rows)
    from common import aggregate
    agg = aggregate(rows, ["attack", "intensity"],
                    ["mu_emp", "mu_pred_from_residual", "tpr@1e-6(400rows)",
                     "sigma_onaxis"])
    save_csv(out / "robustness.csv", agg)

    # degradation figure with error bars: cell noise + row edits
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.6))
    cn = sorted([r for r in agg if r["attack"] in ("none", "cell_noise")],
                key=lambda r: float(r["intensity"]))
    a1.errorbar([r["intensity"] for r in cn], [r["mu_emp_mean"] for r in cn],
                yerr=[r["mu_emp_std"] for r in cn], fmt="o-",
                label=r"$\mu$ empirical")
    a1.plot([r["intensity"] for r in cn],
            [r["mu_pred_from_residual_mean"] for r in cn], "s--",
            label="Gaussian model (measured residual)")
    a1.set_xlabel("cell noise (fraction of column std)"); a1.set_ylabel(r"$\mu$")
    a1.legend(); a1.set_title(f"Value tampering ({seeds} seeds)")
    re_ = sorted([r for r in agg if r["attack"] in ("none", "edit_rows")],
                 key=lambda r: float(r["intensity"]))
    base = re_[0]["mu_emp_mean"]
    a2.errorbar([r["intensity"] for r in re_], [r["mu_emp_mean"] for r in re_],
                yerr=[r["mu_emp_std"] for r in re_], fmt="o-",
                label=r"$\mu$ empirical")
    xs = np.linspace(0, 0.9, 50)
    a2.plot(xs, base * (1 - xs), "k--", lw=0.9, label=r"$(1-\rho)\,\mu$ (theory)")
    a2.set_xlabel(r"fraction of rows replaced $\rho$")
    a2.legend(); a2.set_title("Row tampering: linear, no cliff")
    save_fig(fig, out / "degradation_curves.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    main(**vars(ap.parse_args()))
