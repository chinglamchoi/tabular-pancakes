"""P3b -- Robustness on a real host: table attacks on the numeric block.

Same attack menu as p3_robustness, applied to the numeric columns of the
released table (categoricals untouched), then the full verifier round
trip. Reports mu and TPR at certified FPR 1e-6 per attack/intensity,
mean +/- std over --seeds independent keys/tables.

Usage: python p3b_tabsyn_robustness.py --host tabsyn [--quick] [--seeds N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import outdir, save_csv, aggregate  # noqa: E402
from hostglue import get_host, add_host_args, effective_seeds, run_tag  # noqa: E402

from pancakemark import keygen, sample_latents
from pancakemark.detect import row_scores
from pancakemark.null import hoeffding_pvalue
from pancakemark import attacks as A


def main(args):
    out = outdir(f"p3b_robustness_{run_tag(args)}")
    host = get_host(args.host, tabwak_root=args.tabwak_root,
                    dataname=args.dataname, device=args.device, steps=args.steps,
                    refine_iters=args.refine_iters,
                    fp_iters=args.fp_iters)
    n = host.latent_dim
    m = 2000 if args.quick else 6000
    seeds = effective_seeds(args)

    rows = []
    for seed in range(seeds):
        rng = np.random.default_rng(220 + seed)
        key = keygen(n, k=8, gamma=2.0, beta=0.05, seed=7200 + seed)
        table0 = host.generate(sample_latents(m, key, rng))
        X0 = host.numeric_view(table0)

        def record(attack, intensity, Xa):
            ta = host.reassemble(table0, Xa)
            s = row_scores(host.invert(ta), key)
            mu = float(s.mean())
            hits = 0
            reps = 20
            for _ in range(reps):
                idx = rng.choice(s.size, min(400, s.size), replace=False)
                if hoeffding_pvalue(float(s[idx].mean()), idx.size) <= 1e-6:
                    hits += 1
            rows.append({"attack": attack, "intensity": intensity, "seed": seed,
                         "mu_emp": mu, "tpr@1e-6(400rows)": hits / reps})
            print(f"s{seed} {attack:>12s} {intensity!s:>5}: mu={mu:.3f}  "
                  f"TPR={hits/reps:.2f}")

        record("none", 0, X0)
        for frac in (0.02, 0.05, 0.10, 0.20):
            record("cell_noise", frac, A.gaussian_cell_noise(X0, frac, rng))
        for digits in (3, 2):
            record("round_sig", digits, A.round_significant(X0, digits))
        record("winsorize", 0.01, A.winsorize(X0, 0.01))
        for rho in (0.3, 0.6):
            record("edit_rows", rho, A.edit_rows(X0, rho, rng))
        record("few_cells", 50, A.few_cell_edits(X0, 50, 2.0, rng))
        if not args.quick:
            record("reimpute", 0.05, A.reimpute_cells(X0, 0.05, rng))
    save_csv(out / "robustness_perseed.csv", rows)
    agg = aggregate(rows, ["attack", "intensity"],
                    ["mu_emp", "tpr@1e-6(400rows)"])
    save_csv(out / "robustness.csv", agg)


if __name__ == "__main__":
    ap = add_host_args(argparse.ArgumentParser())
    main(ap.parse_args())
