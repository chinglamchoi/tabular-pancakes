"""P1b -- TabSyn operating-point grid: inversion_report over (k, beta).

Runs the inversion-feasibility measurement across k in {4,8,16} x beta in {.02,.05,.1}
at gamma=2 (the P8-safe regime), plus the ODE-only error decomposition,
and prints the recommended operating point: minimal rows_needed at
alpha=1e-6 subject to mu_emp >= 0.1.

Local validation: --host flow (exact inverse; recommendations should pick
the largest k / smallest beta). Cluster: --host tabsyn.

Replicated over --seeds independent keys/draws; the grid CSV carries
mean +/- std and the recommendation is made on the seed means.

Usage: python p1b_tabsyn_grid.py --host tabsyn [--quick] [--seeds N]
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
from pancakemark.inversion import inversion_report


def main(args):
    out = outdir(f"p1b_grid_{run_tag(args)}")
    rng = np.random.default_rng(20)
    host = get_host(args.host, tabwak_root=args.tabwak_root,
                    dataname=args.dataname, device=args.device,
                    steps=args.steps,
                    refine_iters=args.refine_iters,
                    fp_iters=args.fp_iters)
    n = host.latent_dim
    m = 1000 if args.quick else 2000
    gamma = 2.0
    seeds = effective_seeds(args)

    # ODE-only error (skips the VAE/table round trip), when the host has it
    if hasattr(host.host, "invert_embedding"):
        z = rng.standard_normal((min(m, 1000), n)).astype(np.float32)
        r = host.host.invert_embedding(host.host.generate_embedding(z)) - z
        print(f"sigma_ode_only = {float(np.std(r)):.5f}")

    rows = []
    for seed in range(seeds):
        rng_s = np.random.default_rng(200 + seed)
        for k in (4, 8, 16):
            for beta in (0.02, 0.05, 0.10):
                key = keygen(n, k, gamma=gamma, beta=beta,
                             seed=7000 + 13 * k + 1000 * seed)
                rep = inversion_report(host, key, num_rows=m, rng=rng_s)
                rep["seed"] = seed
                rows.append(rep)
                print(f"s{seed} k={k:<3} beta={beta:<5} "
                      f"sigma_onaxis={rep['sigma_onaxis']:.4f} "
                      f"mu_emp={rep['mu_emp']:.3f} "
                      f"rows@1e-6={rep['rows_needed_alpha1e6_power0.9']}")
    save_csv(out / "grid_perseed.csv", rows)
    agg = aggregate(rows, ["k", "beta"],
                    ["sigma_onaxis", "mu_emp",
                     "rows_needed_alpha1e6_power0.9"])
    save_csv(out / "grid.csv", agg)

    ok = [r for r in agg if r["mu_emp_mean"] >= 0.1]
    if ok:
        best = min(ok, key=lambda r: r["rows_needed_alpha1e6_power0.9_mean"])
        print(f"\nRECOMMENDED OPERATING POINT: gamma={gamma}, "
              f"beta={best['beta']}, k={best['k']} "
              f"(mu_emp={best['mu_emp_mean']:.3f}"
              f"+/-{best['mu_emp_std']:.3f}, "
              f"rows@1e-6={best['rows_needed_alpha1e6_power0.9_mean']:.0f}, "
              f"{best['nseeds']} seeds)")
    else:
        print("\nNO viable operating point at gamma=2: escalate per RUNBOOK "
              "(encoder path / whitened variant / gamma dial).")


if __name__ == "__main__":
    ap = add_host_args(argparse.ArgumentParser())
    main(ap.parse_args())
