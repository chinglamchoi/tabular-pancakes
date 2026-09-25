"""P8 -- Sample-constrained concrete security: the covariance attack.

The best known keyless attack on pancake structure is the covariance
distinguisher (max eigenvalue deviation after whitening). We map its ROC
AUC across (gamma, per-key sample budget), in our standard-normal
convention (our gamma ~ BRST rho-convention gamma / sqrt(2*pi)): small
gamma collapses the on-axis variance and is detectable; the operating
regime gamma >= ~1 is blind at any realistic per-release budget.

This is the gamma-dial evidence: per-key sample exposure is a protocol
parameter (key rotation), so the x-axis is what a fraudster actually gets.

Usage: python p8_covariance_attack.py [--quick]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import outdir, save_csv, save_fig  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from pancakemark import keygen
from pancakemark.sampler import sample_latents
from pancakemark.attacks import covariance_attack_auc


def main(quick: bool = False):
    out = outdir("p8_covariance")
    rng = np.random.default_rng(9)
    n = 32
    trials = 30 if quick else 100
    gammas = [0.25, 0.35, 0.5, 0.75, 1.0, 2.0]
    budgets = [200, 500, 1000, 3000] if quick else [200, 500, 1000, 3000, 10000]

    rows = []
    for gamma in gammas:
        for m in budgets:
            def mk(rng, gamma=gamma, m=m):
                key = keygen(n, 1, gamma=gamma, beta=0.05,
                             seed=int(rng.integers(1 << 30)))
                return sample_latents(m, key, rng)

            def null(rng, m=m):
                return rng.standard_normal((m, n))

            a = covariance_attack_auc(mk, null, trials=trials, rng=rng)
            rows.append({"gamma": gamma, "samples_per_key": m, "auc": a,
                         "n": n, "trials": trials})
            print(f"gamma={gamma:<5} m={m:<6} AUC={a:.3f}")
    save_csv(out / "covariance_auc.csv", rows)

    fig = plt.figure(figsize=(6.2, 4))
    for gamma in gammas:
        rr = [r for r in rows if r["gamma"] == gamma]
        plt.plot([r["samples_per_key"] for r in rr], [r["auc"] for r in rr],
                 "o-", label=rf"$\gamma$={gamma}")
    plt.axhline(0.5, color="k", ls="--", lw=0.8)
    plt.xscale("log")
    plt.xlabel("attacker's per-key sample budget")
    plt.ylabel("covariance-attack ROC AUC")
    plt.title(f"Best known keyless attack vs sample exposure (n={n})")
    plt.legend(fontsize=7)
    save_fig(fig, out / "covariance_auc.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(**vars(ap.parse_args()))
