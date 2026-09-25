"""P1.1 -- Host round-trip study: sigma_inv, predicted vs empirical mu,
and detection through generators with controlled inversion error.

Hosts: exact linear, exact nonlinear flow, flow + encoder noise (two
levels), and a quantization sweep (released-table precision).

Usage: python p1_hosts_sigma_inv.py [--quick] [--seeds N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import outdir, save_csv, save_fig  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from pancakemark import keygen, theory
from pancakemark.hosts import LinearHost, TanhFlowHost, NoisyEncoderHost, QuantizedHost
from pancakemark.inversion import inversion_report


def main(quick: bool = False, seeds: int = 3):
    out = outdir("p1_hosts")
    n, k, gamma, beta = 64, 4, 2.0, 0.05
    m = 2000 if quick else 8000
    if quick:
        seeds = min(seeds, 2)

    base = TanhFlowHost(n, layers=3, seed=42)
    hosts = [
        LinearHost(n, seed=43),
        base,
        NoisyEncoderHost(base, sigma_enc=0.05, seed=44),
        NoisyEncoderHost(base, sigma_enc=0.15, seed=45),
    ]
    rows, qrows = [], []
    for seed in range(seeds):
        rng = np.random.default_rng(400 + seed)
        key = keygen(n, k, gamma, beta, seed=41 + seed)
        for h in hosts:
            r = inversion_report(h, key, num_rows=m, rng=rng)
            r["seed"] = seed
            rows.append(r)
            print(f"s{seed} {r['host']:>22s}  sigma_onaxis={r['sigma_onaxis']:.4f}  "
                  f"mu_pred={r['mu_pred_gaussian_model']:.3f}  "
                  f"mu_emp={r['mu_emp']:.3f}  "
                  f"rows@1e-6={r['rows_needed_alpha1e6_power0.9']}")
        # quantization sweep: released-table precision vs signal
        for dec in (0, 1, 2, 3, 4):
            rep = inversion_report(QuantizedHost(base, dec), key, num_rows=m,
                                   rng=rng)
            rep["decimals"] = dec
            rep["seed"] = seed
            qrows.append(rep)
            print(f"s{seed} quantize {dec} decimals: "
                  f"sigma_onaxis={rep['sigma_onaxis']:.4f} "
                  f"mu_emp={rep['mu_emp']:.3f}")

    from common import aggregate
    save_csv(out / "host_reports_perseed.csv", rows)
    host_order = [h.name for h in hosts]
    agg = aggregate(rows, ["host"],
                    ["sigma_onaxis", "mu_pred_gaussian_model", "mu_emp",
                     "rows_needed_alpha1e6_power0.9",
                     "marked_vs_unmarked_residual_ratio"])
    agg.sort(key=lambda r: host_order.index(r["host"]))
    save_csv(out / "host_reports.csv", agg)

    fig = plt.figure(figsize=(6.2, 3.6))
    names = [r["host"] for r in agg]
    xpos = np.arange(len(agg))
    plt.bar(xpos - 0.18, [r["mu_pred_gaussian_model_mean"] for r in agg], 0.36,
            yerr=[r["mu_pred_gaussian_model_std"] for r in agg], capsize=3,
            label=r"$\mu$ predicted (Gaussian model)")
    plt.bar(xpos + 0.18, [r["mu_emp_mean"] for r in agg], 0.36,
            yerr=[r["mu_emp_std"] for r in agg], capsize=3,
            label=r"$\mu$ empirical (round trip)")
    plt.axhline(theory.mu(gamma, beta), ls="--", lw=0.8, color="k",
                label=r"clean $\mu$ (no inversion error)")
    plt.xticks(xpos, names, rotation=12, fontsize=7)
    plt.ylabel(r"$\mu$"); plt.legend(fontsize=7)
    plt.title(f"Host round trips (n={n}, k={k}, $\\gamma$={gamma}, "
              f"$\\beta$={beta}, {seeds} seeds)")
    save_fig(fig, out / "mu_pred_vs_emp.png")

    save_csv(out / "quantization_sweep_perseed.csv", qrows)
    qagg = aggregate(qrows, ["decimals"],
                     ["sigma_onaxis", "mu_pred_gaussian_model", "mu_emp"])
    qagg.sort(key=lambda r: r["decimals"])
    save_csv(out / "quantization_sweep.csv", qagg)
    fig = plt.figure(figsize=(5.4, 3.4))
    plt.errorbar([r["decimals"] for r in qagg],
                 [r["mu_emp_mean"] for r in qagg],
                 yerr=[r["mu_emp_std"] for r in qagg], fmt="o-",
                 label=r"$\mu$ empirical")
    plt.plot([r["decimals"] for r in qagg],
             [r["mu_pred_gaussian_model_mean"] for r in qagg], "s--",
             label=r"$\mu$ predicted (Gaussian approx)")
    plt.xlabel("released decimals per feature"); plt.ylabel(r"$\mu$")
    plt.title("Rounding the released table: signal vs precision")
    plt.legend()
    save_fig(fig, out / "quantization_sweep.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    main(**vars(ap.parse_args()))
