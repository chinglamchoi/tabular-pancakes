"""P0.2 -- Power curves: empirical mean statistic and TPR vs. theory.

(i)  mu_hat vs closed-form mu across (gamma, beta) and added noise sigma.
(ii) TPR at fixed certified FPR (Hoeffding) and MC-null FPR, vs. rows m.

Usage: python p0_power_curves.py [--quick]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import outdir, save_csv, save_fig  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from pancakemark import keygen, sample_latents, global_stat, theory
from pancakemark.detect import row_scores
from pancakemark.null import hoeffding_pvalue


def mu_vs_noise(out, quick, rng):
    n, k, m = 64, 4, (20_000 if quick else 80_000)
    sigmas = np.linspace(0, 0.25, 11)
    rows, fig = [], plt.figure(figsize=(6, 4))
    for gamma, beta in [(2.0, 0.05), (4.0, 0.05), (2.0, 0.15)]:
        key = keygen(n, k, gamma, beta, seed=100)
        z = sample_latents(m, key, rng)
        emp, th = [], []
        for s in sigmas:
            zn = z + s * rng.standard_normal(z.shape)
            emp.append(global_stat(zn, key))
            th.append(theory.mu(gamma, beta, noise_std=s))
            rows.append({"gamma": gamma, "beta": beta, "sigma": s,
                         "mu_hat": emp[-1], "mu_theory": th[-1], "m": m, "k": k})
        (line,) = plt.plot(sigmas, th, lw=1)
        plt.plot(sigmas, emp, "o", ms=4, color=line.get_color(),
                 label=rf"$\gamma$={gamma}, $\beta$={beta}")
    plt.xlabel(r"added on-axis noise std $\sigma$")
    plt.ylabel(r"$\hat\mu$ (dots) vs $e^{-2\pi^2 v}$ (lines)")
    plt.legend(); plt.title("Signal mean: empirical vs closed form")
    save_fig(fig, out / "mu_vs_noise.png")
    save_csv(out / "mu_vs_noise.csv", rows)
    err = max(abs(r["mu_hat"] - r["mu_theory"]) for r in rows)
    print(f"max |mu_hat - mu_theory| = {err:.4f}")


def tpr_vs_rows(out, quick, rng):
    n, k, gamma, beta = 64, 4, 2.0, 0.05
    alpha = 1e-6
    trials = 60 if quick else 200
    ms = [25, 50, 100, 200, 400, 800]
    sigma = 0.10  # operate at beta_eff via noise so power is non-trivial
    mu = theory.mu(gamma, beta, noise_std=sigma)
    rows = []
    for m in ms:
        hits = 0
        for tr in range(trials):
            key = keygen(n, k, gamma, beta, seed=5000 + tr)
            z = sample_latents(m, key, rng)
            z += sigma * rng.standard_normal(z.shape)
            S = float(row_scores(z, key).mean())
            if hoeffding_pvalue(S, num_rows=m) <= alpha:
                hits += 1
        rows.append({"m": m, "tpr": hits / trials, "alpha": alpha,
                     "mu": mu, "trials": trials})
        print(f"m={m:5d}  TPR@alpha={alpha:g}: {hits/trials:.3f}")
    fig = plt.figure(figsize=(6, 4))
    plt.plot([r["m"] for r in rows], [r["tpr"] for r in rows], "o-")
    m_star = theory.rows_needed(mu, alpha, power=0.9)
    plt.axvline(m_star, ls="--", lw=0.8,
                label=f"theory sufficient m={m_star} (power 0.9)")
    plt.xscale("log"); plt.xlabel("rows m"); plt.ylabel(f"TPR @ certified FPR={alpha:g}")
    plt.title(rf"$\gamma$={gamma}, $\beta$={beta}, $\sigma$={sigma} ($\mu$={mu:.2f})")
    plt.legend()
    save_fig(fig, out / "tpr_vs_rows.png")
    save_csv(out / "tpr_vs_rows.csv", rows)


def main(quick: bool = False):
    out = outdir("p0_power")
    rng = np.random.default_rng(1)
    mu_vs_noise(out, quick, rng)
    tpr_vs_rows(out, quick, rng)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(**vars(ap.parse_args()))
