"""P10 -- Radioactivity toy (scenario D3): does the mark survive a student?

Launderer trains a Gaussian-copula resampler (classic SDV-style baseline:
empirical marginals + Gaussian correlation) on 100%-marked generator
output and releases the student's samples. We measure residual watermark
signal in the student output, alongside the honest positive/negative
controls:

  released marked table (no laundering)   -> full signal (control)
  copula student trained on marked data   -> the D3 question
  copula student trained on unmarked data -> null control (FPR side)

Expected (and honest) outcome: row-wise latent-comb signal does NOT
survive marginal-copula resampling -- laundering through a *retrained*
model is the regeneration limit our theory concedes. The experiment
quantifies it and verifies no false signal on the null side; the paper
reports this as the boundary of the guarantee, with the utility cost of
the student (fidelity drop) as the accompanying price.

Usage: python p10_radioactivity_toy.py [--quick] [--seeds N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import stats as st

sys.path.insert(0, str(Path(__file__).parent))
from common import outdir, save_csv  # noqa: E402

from pancakemark import keygen, sample_latents, detect
from pancakemark.hosts import TanhFlowHost


class GaussianCopulaStudent:
    """Fit: empirical marginals (quantiles) + Gaussian copula correlation."""

    def fit(self, X: np.ndarray) -> "GaussianCopulaStudent":
        self.X_sorted = np.sort(X, axis=0)
        # ranks -> normal scores
        U = (st.rankdata(X, axis=0) - 0.5) / X.shape[0]
        Zs = st.norm.ppf(U)
        self.C = np.corrcoef(Zs, rowvar=False)
        self.L = np.linalg.cholesky(self.C + 1e-9 * np.eye(X.shape[1]))
        return self

    def sample(self, m: int, rng) -> np.ndarray:
        Zs = rng.standard_normal((m, self.L.shape[0])) @ self.L.T
        U = st.norm.cdf(Zs)
        out = np.empty_like(Zs)
        n = self.X_sorted.shape[0]
        for j in range(out.shape[1]):
            idx = np.clip((U[:, j] * n).astype(int), 0, n - 1)
            out[:, j] = self.X_sorted[idx, j]
        return out


def main(quick: bool = False, seeds: int = 3):
    out = outdir("p10_radioactivity")
    n, k = 64, 4
    m_train = 8000 if quick else 30000
    m_test = 3000 if quick else 8000
    if quick:
        seeds = min(seeds, 2)
    host = TanhFlowHost(n, layers=3, seed=130)

    rows = []
    for seed in range(seeds):
        rng = np.random.default_rng(1300 + seed)
        key = keygen(n, k, gamma=2.0, beta=0.05, seed=131 + seed)

        X_marked = host.generate(sample_latents(m_train, key, rng))
        X_plain = host.generate(rng.standard_normal((m_train, n)))

        student_m = GaussianCopulaStudent().fit(X_marked)
        student_0 = GaussianCopulaStudent().fit(X_plain)

        cases = {
            "released_marked(control)": X_marked[:m_test],
            "student_on_marked(D3)": student_m.sample(m_test, rng),
            "student_on_unmarked(null)": student_0.sample(m_test, rng),
        }
        per = []
        for name, X in cases.items():
            res = detect(host.invert(X), key, method="hoeffding")
            # student fidelity vs its training data (mean col-wise KS)
            ks = np.mean([st.ks_2samp(X[:, j], X_marked[:m_test, j]).statistic
                          for j in range(0, n, 8)])
            per.append({"case": name, "seed": seed, "stat": res.stat,
                        "log10_pvalue": float(np.log10(max(res.pvalue, 1e-300))),
                        "pvalue": res.pvalue,
                        "meanKS_vs_marked_release": float(ks), "m": m_test})
            print(f"s{seed} {name:>28s}: S={res.stat:+.4f}  p={res.pvalue:.3g}  "
                  f"KS~{ks:.3f}")
        assert per[0]["pvalue"] < 1e-10          # control detects, every seed
        assert per[2]["pvalue"] > 1e-3           # null student: no false signal
        rows.extend(per)
    save_csv(out / "radioactivity_perseed.csv", rows)
    from common import aggregate
    agg = aggregate(rows, ["case"],
                    ["stat", "log10_pvalue", "meanKS_vs_marked_release"])
    save_csv(out / "radioactivity.csv", agg)
    print("D3 boundary quantified: see radioactivity.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    main(**vars(ap.parse_args()))
