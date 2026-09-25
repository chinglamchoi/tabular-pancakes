"""P0.4 -- Payload bit error rate vs. rows and constellation size.

BER for b in {1,2,3} bits/direction as a function of rows m, with the
delta-method phase-std prediction overlaid (Gaussian phase error through
the nearest-symbol decision boundary at 1/2^{b+1} cycles).

Usage: python p0_payload_ber.py [--quick]
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

from pancakemark import keygen, sample_latents, encode_payload, decode_payload, bit_error_rate, theory
from pancakemark.detect import direction_stats


def main(quick: bool = False):
    out = outdir("p0_payload")
    rng = np.random.default_rng(3)
    n, k, gamma, beta, sigma = 64, 8, 2.0, 0.05, 0.10
    mu = theory.mu(gamma, beta, noise_std=sigma)
    ms = [50, 100, 200, 400, 800, 1600]
    trials = 30 if quick else 120

    rows = []
    fig = plt.figure(figsize=(6, 4))
    for b in (1, 2, 3):
        empirical, predicted = [], []
        for m in ms:
            errs = 0,
            tot = 0
            nerr = 0
            for tr in range(trials):
                bits = rng.integers(0, 2, size=k * b)
                delta = encode_payload(bits, k, b)
                key = keygen(n, k, gamma, beta, delta=delta, seed=30_000 + 100 * b + tr)
                z = sample_latents(m, key, rng)
                z += sigma * rng.standard_normal(z.shape)
                d_hat = direction_stats(z, key)["delta_hat"]
                out_bits = decode_payload(d_hat, b)
                nerr += int(np.sum(bits != out_bits))
                tot += bits.size
            ber = nerr / tot
            # symbol error ~ P(|phase error| > 1/2^{b+1}); Gray => ~1 bit/symbol err
            ph_std = theory.phase_estimate_std(mu, m)
            ser = 2 * stats.norm.sf((1 / 2 ** (b + 1)) / ph_std)
            pred = ser / b
            empirical.append(ber); predicted.append(pred)
            rows.append({"b": b, "m": m, "ber": ber, "ber_pred": pred,
                         "phase_std_pred": ph_std, "mu": mu, "trials": trials})
        (line,) = plt.plot(ms, np.maximum(predicted, 1e-6), lw=1)
        plt.plot(ms, np.maximum(empirical, 1e-6), "o", ms=4,
                 color=line.get_color(), label=f"b={b} bits/direction")
        print(f"b={b}: BER {list(np.round(empirical, 4))}")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("rows m"); plt.ylabel("bit error rate (floor 1e-6)")
    plt.title(rf"Payload BER, k={k}, $\mu$={mu:.2f} (dots: empirical, lines: delta-method)")
    plt.legend()
    save_fig(fig, out / "ber_vs_rows.png")
    save_csv(out / "ber.csv", rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(**vars(ap.parse_args()))
