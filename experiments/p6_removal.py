"""P6 -- Removal-cost frontier: keyed vs keyless scrubbing.

Both attackers spend the same per-row latent l2 budget; the keyed oracle
concentrates it along the k secret directions, the keyless attacker must
spend isotropically (Thm: any direction-finding keyless strategy breaks
decision-hCLWE). Curves of mu vs budget show the sqrt(n/k) horizontal
separation, with closed-form overlays:
  keyed   : mu(sigma_onaxis = budget / sqrt(k))
  keyless : mu(sigma_onaxis = budget / sqrt(n))

Also: the utility price -- per-cell distortion in the released table (via
the host) as a function of the same budget, for the frontier plot.

Usage: python p6_removal.py [--quick] [--seeds N]
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
from pancakemark.detect import global_stat
from pancakemark.hosts import TanhFlowHost
from pancakemark import attacks as A


def main(quick: bool = False, seeds: int = 3):
    out = outdir("p6_removal")
    n, k, gamma, beta = 64, 4, 2.0, 0.05
    m = 4000 if quick else 12000
    if quick:
        seeds = min(seeds, 2)
    host = TanhFlowHost(n, layers=3, seed=98_001)
    budgets = np.linspace(0.0, 1.6, 12)   # per-row latent l2
    rows = []
    for seed in range(seeds):
        rng = np.random.default_rng(800 + seed)
        key = keygen(n, k, gamma, beta, seed=98_000 + seed)
        z = sample_latents(m, key, rng)
        X0 = host.generate(z)
        colstd = X0.std(axis=0)
        for b in budgets:
            zk = A.latent_keyed_scrub(z, key.W, b / np.sqrt(k), rng) if b > 0 else z
            zi = A.latent_isotropic_noise(z, b / np.sqrt(n), rng) if b > 0 else z
            mu_k = global_stat(zk, key)
            mu_i = global_stat(zi, key)
            # utility damage of the keyless scrub, measured in the table
            Xi = host.generate(zi)
            cell_rmse = float(np.mean(np.sqrt(np.mean((Xi - X0) ** 2, axis=0)) / colstd))
            rows.append({
                "budget_l2": float(b), "seed": seed,
                "mu_keyed": mu_k,
                "mu_keyless": mu_i,
                "mu_keyed_theory": theory.mu(gamma, beta, noise_std=b / np.sqrt(k)),
                "mu_keyless_theory": theory.mu(gamma, beta, noise_std=b / np.sqrt(n)),
                "keyless_cell_rmse_stdunits": cell_rmse,
            })
            print(f"s{seed} budget={b:5.2f}: mu_keyed={mu_k:.3f}  "
                  f"mu_keyless={mu_i:.3f}  cellRMSE={cell_rmse:.3f}")
    save_csv(out / "removal_frontier_perseed.csv", rows)
    from common import aggregate
    agg = aggregate(rows, ["budget_l2"],
                    ["mu_keyed", "mu_keyless", "mu_keyed_theory",
                     "mu_keyless_theory", "keyless_cell_rmse_stdunits"])
    agg.sort(key=lambda r: r["budget_l2"])
    save_csv(out / "removal_frontier.csv", agg)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.8, 3.6))
    B = [r["budget_l2"] for r in agg]
    a1.errorbar(B, [r["mu_keyed_mean"] for r in agg],
                yerr=[r["mu_keyed_std"] for r in agg], fmt="o", color="#B23A2E",
                label="keyed oracle (has $W$)")
    a1.plot(B, [r["mu_keyed_theory_mean"] for r in agg], "-", color="#B23A2E", lw=1)
    a1.errorbar(B, [r["mu_keyless_mean"] for r in agg],
                yerr=[r["mu_keyless_std"] for r in agg], fmt="s", color="#0C7FA3",
                label="keyless (isotropic)")
    a1.plot(B, [r["mu_keyless_theory_mean"] for r in agg], "-", color="#0C7FA3", lw=1)
    a1.set_xlabel("per-row latent $\\ell_2$ distortion budget")
    a1.set_ylabel(r"$\mu$")
    a1.set_title(f"Same budget, {np.sqrt(n/k):.0f}x apart in effect "
                 f"($\\sqrt{{n/k}}$, n={n}, k={k}, {seeds} seeds)")
    a1.legend(fontsize=8)
    a2.errorbar([r["keyless_cell_rmse_stdunits_mean"] for r in agg],
                [r["mu_keyless_mean"] for r in agg],
                yerr=[r["mu_keyless_std"] for r in agg],
                fmt="s-", color="#0C7FA3")
    a2.set_xlabel("keyless utility damage: mean cell RMSE (column-std units)")
    a2.set_ylabel(r"$\mu$ remaining")
    a2.set_title("Removal-cost frontier (keyless)")
    save_fig(fig, out / "removal_frontier.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    main(**vars(ap.parse_args()))
