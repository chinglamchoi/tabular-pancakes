"""P9 -- Payload survival under attack: bit error rate after tampering.

64-bit-class payloads (k=8 directions x 2 bits) through the flow host,
then the P3 attack menu; BER + payload-decodable-at-all verdicts.
With --seeds N the payload, key, and attack draws are replicated, so the
reported BER floor is over N x k x b total bits.

Usage: python p9_payload_attacks.py [--quick] [--seeds N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import outdir, save_csv  # noqa: E402

from pancakemark import keygen, sample_latents, encode_payload, decode_payload, bit_error_rate
from pancakemark.detect import direction_stats
from pancakemark.hosts import TanhFlowHost
from pancakemark import attacks as A


def main(quick: bool = False, seeds: int = 3):
    out = outdir("p9_payload")
    n, k, b = 64, 8, 2
    m = 3000 if quick else 10000
    if quick:
        seeds = min(seeds, 2)
    host = TanhFlowHost(n, layers=3, seed=140)

    rows = []
    for seed in range(seeds):
        rng = np.random.default_rng(1400 + seed)
        bits = rng.integers(0, 2, size=k * b)
        delta = encode_payload(bits, k, b)
        key = keygen(n, k, gamma=2.0, beta=0.05, delta=delta, seed=141 + seed)
        z = sample_latents(m, key, rng)
        X0 = host.generate(z)

        def ber_of(X):
            d_hat = direction_stats(host.invert(X), key)["delta_hat"]
            return bit_error_rate(bits, decode_payload(d_hat, b))

        menu = [("none", 0, X0),
                ("cell_noise", 0.02, A.gaussian_cell_noise(X0, 0.02, rng)),
                ("cell_noise", 0.05, A.gaussian_cell_noise(X0, 0.05, rng)),
                ("cell_noise", 0.10, A.gaussian_cell_noise(X0, 0.10, rng)),
                ("round_sig", 2, A.round_significant(X0, 2)),
                ("edit_rows", 0.4, A.edit_rows(X0, 0.4, rng)),
                ("edit_rows", 0.7, A.edit_rows(X0, 0.7, rng)),
                ("subsample", 0.1, A.subsample_rows(X0, 0.1, rng)),
                ("reimpute", 0.10, A.reimpute_cells(X0, 0.10, rng))]
        for name, inten, X in menu:
            ber = ber_of(X)
            rows.append({"attack": name, "intensity": inten, "seed": seed,
                         "ber": ber, "payload_bits": k * b,
                         "m_rows": X.shape[0]})
            print(f"s{seed} {name:>12s} {inten!s:>5}: BER = {ber:.3f}")
    save_csv(out / "payload_under_attack_perseed.csv", rows)
    from common import aggregate
    agg = aggregate(rows, ["attack", "intensity"], ["ber"])
    for r in agg:
        r["total_bits"] = k * b * r["nseeds"]
    save_csv(out / "payload_under_attack.csv", agg)
    print(f"(BER floor per cell: 1/{k * b * seeds} = {1/(k*b*seeds):.4f} "
          f"over {seeds} independent payloads)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    main(**vars(ap.parse_args()))
