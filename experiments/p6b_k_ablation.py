"""P6b -- sqrt(n/k) law ablation: the keyed/keyless separation across k.

For n=64 and k in {1, 4, 16}, measure the budget at which each attacker
suppresses mu below 0.05 and report the ratio keyless/keyed, predicted
to be sqrt(n/k) (Thm). 5 seeds.

Usage: python p6b_k_ablation.py [--quick] [--seeds N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import outdir, save_csv, aggregate  # noqa: E402

from pancakemark import keygen, sample_latents
from pancakemark.detect import global_stat
from pancakemark import attacks as A


def main(quick: bool = False, seeds: int = 5):
    out = outdir("p6b_k_ablation")
    n = 64
    m = 4000 if quick else 12000
    if quick:
        seeds = min(seeds, 2)
    rows = []
    for k in (1, 4, 16):
        for seed in range(seeds):
            rng = np.random.default_rng(6800 + seed)
            key = keygen(n, k, gamma=2.0, beta=0.05, seed=68_000 + 31 * k + seed)
            z = sample_latents(m, key, rng)

            def kill_budget(attack):
                lo, hi = 0.0, 8.0
                for _ in range(18):
                    mid = 0.5 * (lo + hi)
                    za = attack(mid)
                    if global_stat(za, key) < 0.05:
                        hi = mid
                    else:
                        lo = mid
                return 0.5 * (lo + hi)

            b_keyed = kill_budget(
                lambda b: A.latent_keyed_scrub(z, key.W, b / np.sqrt(k), rng))
            b_keyless = kill_budget(
                lambda b: A.latent_isotropic_noise(z, b / np.sqrt(n), rng))
            rows.append({"n": n, "k": k, "seed": seed,
                         "budget_keyed": b_keyed,
                         "budget_keyless": b_keyless,
                         "ratio": b_keyless / b_keyed,
                         "sqrt_n_over_k": float(np.sqrt(n / k))})
            print(f"k={k:<3} s{seed}: keyed={b_keyed:.3f} "
                  f"keyless={b_keyless:.3f} ratio={b_keyless/b_keyed:.2f} "
                  f"(pred {np.sqrt(n/k):.2f})")
    save_csv(out / "k_ablation_perseed.csv", rows)
    agg = aggregate(rows, ["k"], ["budget_keyed", "budget_keyless", "ratio",
                                  "sqrt_n_over_k"])
    save_csv(out / "k_ablation.csv", agg)
    for r in agg:
        print(f"k={r['k']}: ratio={r['ratio_mean']:.2f}+-{r['ratio_std']:.2f} "
              f"pred={r['sqrt_n_over_k_mean']:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=5)
    main(**vars(ap.parse_args()))
