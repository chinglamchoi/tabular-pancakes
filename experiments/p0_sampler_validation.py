"""P0.1 -- Sampler validation: samples vs. exact density, envelope, KS stats.

Usage: python p0_sampler_validation.py [--quick]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from common import outdir, save_csv, save_fig  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from pancakemark import sample_pancake_1d, pancake_density_1d, keygen, sample_latents
from pancakemark.sampler import _layer_distribution


def main(quick: bool = False):
    out = outdir("p0_sampler")
    N = 100_000 if quick else 500_000
    rng = np.random.default_rng(0)
    configs = [(2.0, 0.05, 0.0), (2.0, 0.2, 0.0), (4.0, 0.1, 0.5), (8.0, 0.05, 0.25)]

    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    rows = []
    for ax, (gamma, beta, delta) in zip(axes.ravel(), configs):
        x = sample_pancake_1d(N, gamma, beta, delta, rng)
        t = np.linspace(-3.5, 3.5, 3000)
        ax.hist(x, bins=600, density=True, alpha=0.55, label="samples")
        ax.plot(t, pancake_density_1d(t, gamma, beta, delta), lw=0.8, label="exact density")
        ax.plot(t, np.exp(-t**2 / 2) / np.sqrt(2 * np.pi), "--", lw=0.8, label="N(0,1) envelope")
        ax.set_title(rf"$\gamma$={gamma}, $\beta$={beta}, $\delta$={delta}")
        ax.set_xlim(-3.5, 3.5)

        m, p, gp = _layer_distribution(gamma, beta, delta)
        centers = (m + delta) * gamma / gp**2
        s = beta / gp
        cdf = lambda tt: (stats.norm.cdf((np.asarray(tt)[..., None] - centers) / s) * p).sum(-1)
        ks, pval = stats.kstest(x, cdf)
        rows.append({"gamma": gamma, "beta": beta, "delta": delta, "N": N,
                     "ks_stat": ks, "ks_pvalue": pval,
                     "sample_var": float(np.var(x))})
    axes[0, 0].legend(fontsize=7)
    save_fig(fig, out / "marginals.png")

    # high-dim: secret direction periodic, random direction Gaussian
    key = keygen(n=64, k=1, gamma=2.0, beta=0.05, seed=1)
    z = sample_latents(N // 5, key, rng)
    v = rng.standard_normal(64); v -= key.W[:, 0] * (v @ key.W[:, 0]); v /= np.linalg.norm(v)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3))
    a1.hist(z @ key.W[:, 0], bins=400, density=True); a1.set_title("projection on secret $w$")
    a2.hist(z @ v, bins=400, density=True); a2.set_title("projection on random $v \\perp w$")
    save_fig(fig, out / "projections.png")

    save_csv(out / "ks_table.csv", rows)
    worst = min(r["ks_pvalue"] for r in rows)
    print(f"worst KS p-value across configs: {worst:.3g}")
    assert worst > 1e-4, "sampler mismatch!"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(**vars(ap.parse_args()))
