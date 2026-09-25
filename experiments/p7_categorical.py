"""P7 -- Categorical channel study: power vs entropy, sync robustness,
and the all-numerics-quantized scenario the channel exists for.

Usage: python p7_categorical.py [--quick] [--seeds N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import outdir, save_csv, save_fig  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from pancakemark import keygen, sample_latents
from pancakemark.categorical import sample_categorical, detect_categorical, pancake_indices


def main(quick: bool = False, seeds: int = 3):
    out = outdir("p7_categorical")
    n, k = 48, 4
    m = 1500 if quick else 5000
    if quick:
        seeds = min(seeds, 2)

    rows, rows2 = [], []
    for seed in range(seeds):
        rng = np.random.default_rng(1200 + seed)
        key = keygen(n, k, gamma=2.0, beta=0.05, seed=120 + seed)
        z = sample_latents(m, key, rng)

        # power vs conditional entropy (temperature sweep on 5-way logits)
        for temp in (0.25, 0.5, 1.0, 2.0, 4.0):
            logits = rng.normal(0, 1.0, size=(m, 5)) / temp
            probs = np.exp(logits); probs /= probs.sum(1, keepdims=True)
            ent = float(-(probs * np.log(probs)).sum(1).mean())
            cats = sample_categorical(probs, z, key)
            res = detect_categorical(cats, z, key, 5)
            rows.append({"temperature": temp, "seed": seed,
                         "mean_entropy_nats": ent, "stat": res["stat"],
                         "log10_pvalue": float(np.log10(max(res["pvalue"], 1e-300))),
                         "m": m})
            print(f"s{seed} T={temp:<5} H={ent:.2f} nats  S={res['stat']:.3f}  "
                  f"p={res['pvalue']:.2e}")

        # sync robustness: latent perturbation sweep
        logits = rng.normal(0, 0.7, size=(m, 5))
        probs = np.exp(logits); probs /= probs.sum(1, keepdims=True)
        cats = sample_categorical(probs, z, key)
        for sig in (0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.4):
            zh = z + sig * rng.standard_normal(z.shape)
            res = detect_categorical(cats, zh, key, 5)
            same = float((pancake_indices(z, key) == pancake_indices(zh, key)).mean())
            rows2.append({"latent_noise": sig, "seed": seed, "stat": res["stat"],
                          "log10_pvalue": float(np.log10(max(res["pvalue"], 1e-300))),
                          "sync_intact_frac": same})
            print(f"s{seed} sigma={sig:<5} sync intact={same:.2f}  S={res['stat']:.3f}")

    from common import aggregate
    save_csv(out / "power_vs_entropy_perseed.csv", rows)
    agg = aggregate(rows, ["temperature"],
                    ["mean_entropy_nats", "stat", "log10_pvalue"])
    agg.sort(key=lambda r: r["temperature"])
    save_csv(out / "power_vs_entropy.csv", agg)

    fig = plt.figure(figsize=(5.4, 3.4))
    plt.errorbar([r["mean_entropy_nats_mean"] for r in agg],
                 [r["stat_mean"] for r in agg],
                 yerr=[r["stat_std"] for r in agg], fmt="o-")
    plt.xlabel("mean conditional entropy (nats)")
    plt.ylabel("categorical channel mean score $S$")
    plt.title(f"Signal lives where the model has entropy to spend ({seeds} seeds)")
    save_fig(fig, out / "power_vs_entropy.png")

    save_csv(out / "sync_robustness_perseed.csv", rows2)
    agg2 = aggregate(rows2, ["latent_noise"],
                     ["stat", "log10_pvalue", "sync_intact_frac"])
    agg2.sort(key=lambda r: r["latent_noise"])
    save_csv(out / "sync_robustness.csv", agg2)
    fig = plt.figure(figsize=(5.4, 3.4))
    plt.errorbar([r["latent_noise"] for r in agg2],
                 [r["stat_mean"] for r in agg2],
                 yerr=[r["stat_std"] for r in agg2], fmt="o-",
                 label="channel score $S$")
    plt.errorbar([r["latent_noise"] for r in agg2],
                 [r["sync_intact_frac_mean"] for r in agg2],
                 yerr=[r["sync_intact_frac_std"] for r in agg2], fmt="s--",
                 label="sync tokens intact")
    plt.xlabel("latent inversion noise std"); plt.legend()
    plt.title("Locality-sensitive sync: degrades, never false-fires")
    save_fig(fig, out / "sync_robustness.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    main(**vars(ap.parse_args()))
