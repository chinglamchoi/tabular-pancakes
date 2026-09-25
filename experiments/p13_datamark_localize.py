"""P13 -- D2 localization with the data-space mark on a real host.

The D2 scenario, end to end on real material: a fraction rho of rows in a
"submitted" table are marked generator output, the rest genuine rows.
The verifier canonicalizes the mixed table (published reference whitening
+ suspect-refit standardization), computes row p-values by key
rerandomization, and flags rows by Benjamini-Hochberg.

Per (rho, seed): row-level AUC, realized FDR at q=0.05, power, rho_hat.
Hosts: --host flow validates locally (genuine rows = unmarked host
output); --host tabsyn uses real Adult rows as the genuine side
(--real-csv, headerless or headered UCI order).

Usage:
  python p13_datamark_localize.py --host flow [--quick]
  python p13_datamark_localize.py --host tabsyn --real-csv TabWak/data/adult/adult.data --seeds 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import outdir, save_csv, aggregate  # noqa: E402
from hostglue import get_host, add_host_args, effective_seeds, run_tag  # noqa: E402

from pancakemark import keygen, theory
from pancakemark.datamark import DataSpaceMark
from pancakemark.tabular import TableCodec
from pancakemark.localize import localize
from pancakemark.detect import row_scores


ADULT_NUM = [0, 2, 4, 10, 11, 12]


def load_real_numeric(path: str, info_json: str | None = None) -> np.ndarray:
    """Real csv -> numeric block in schema order. With --info-json the
    dataset's num_col_idx is used (any TabWak/TabSyn dataset); without it,
    the Adult layout. Tolerant of a header row either way."""
    import json
    import pandas as pd
    num_idx = (json.load(open(info_json))["num_col_idx"] if info_json
               else ADULT_NUM)
    df = pd.read_csv(path, header=None)
    try:
        return df.iloc[:, num_idx].to_numpy(float)
    except (ValueError, TypeError):
        df = pd.read_csv(path)                      # had a header
        return df.iloc[:, num_idx].to_numpy(float)


def main(args):
    out = outdir(f"p13_localize_{run_tag(args)}")
    host = get_host(args.host, tabwak_root=args.tabwak_root,
                    dataname=args.dataname, device=args.device,
                    steps=args.steps)
    n = host.latent_dim
    m = 2000 if args.quick else 6000
    seeds = effective_seeds(args)
    real_pool = (load_real_numeric(args.real_csv, args.info_json)
                 if args.real_csv else None)

    rows = []
    for seed in range(seeds):
        rng = np.random.default_rng(1300 + seed)
        # provider side: marked synthetic release + published codec
        X_syn = host.numeric_view(host.generate(rng.standard_normal((m, n))))
        codec = TableCodec(select_columns=True).fit(X_syn)
        key = keygen(codec.dim, max(1, min(4, codec.dim // 2)),
                     gamma=args.gamma, beta=args.beta, seed=7700 + seed)
        Xm, _ = DataSpaceMark(key, codec).mark(X_syn, rng)
        u_ref = (codec.transform(Xm) if args.codec_mode == "published"
                 else codec.with_standardization_of(Xm).transform(Xm))
        mu_ref = float(row_scores(u_ref, key).mean())

        # genuine rows + a DISJOINT genuine sample for the rho_hat
        # baseline (real rows carry nonzero coherence mu_null under the
        # true key; the verifier holds the claimed-genuine source, so it
        # calibrates both endpoints of the mixture estimator)
        if real_pool is not None:
            if len(real_pool) < m + 500:
                raise SystemExit(f"--real-csv has only {len(real_pool)} rows "
                                 f"(< m+500={m + 500}); point it at the FULL "
                                 "dataset")
            perm_pool = rng.permutation(len(real_pool))
            genuine = real_pool[perm_pool[:m]]
            base = real_pool[perm_pool[m: m + min(4 * m,
                                                  len(real_pool) - m)]]
        else:
            genuine = host.numeric_view(
                host.generate(rng.standard_normal((m, n))))
            base = host.numeric_view(
                host.generate(rng.standard_normal((m, n))))
        u_base = (codec.transform(base) if args.codec_mode == "published"
                  else codec.with_standardization_of(base).transform(base))
        base_scores = row_scores(u_base, key)   # reference null (true key)
        mu_null = float(base_scores.mean())

        for rho in (0.05, 0.1, 0.3, 0.5):
            n_syn = int(rho * m)
            mixed = np.vstack([genuine[: m - n_syn], Xm[:n_syn]])
            truth = np.r_[np.zeros(m - n_syn), np.ones(n_syn)].astype(bool)
            perm = rng.permutation(m)
            mixed, truth = mixed[perm], truth[perm]

            # Verifier canonicalization of the MIXED table. "published"
            # (default) applies the provider-published codec verbatim --
            # correct whenever units/schema are unchanged, and immune to
            # the mixture shifting the refit quantiles (real-Adult vs
            # TabSyn marginals differ slightly; a refit on the mixture
            # decoheres marked-row phases in proportion to the real
            # fraction -- measured: rho_hat 0.22 at rho=0.5). "refit"
            # re-standardizes on the suspect (the affine-attack mode).
            if args.codec_mode == "published":
                u = codec.transform(mixed)
            else:
                u = codec.with_standardization_of(mixed).transform(mixed)
            rep = localize(u, key, mu_ref=mu_ref, q=0.05, draws=args.draws,
                           mu_null=mu_null, null_scores=base_scores, rng=rng)
            flags, p = rep["flags"], rep["row_pvalues"]
            tp = int((flags & truth).sum())
            fp = int((flags & ~truth).sum())
            power = tp / max(truth.sum(), 1)
            fdr = fp / max(flags.sum(), 1)
            # row-level AUC (Mann-Whitney): lower p = more synthetic
            from scipy.stats import rankdata
            r_ = rankdata(-p)
            n_pos = int(truth.sum())
            auc = float((r_[truth].mean() - (n_pos + 1) / 2) / (m - n_pos))
            rows.append({"rho": rho, "seed": seed, "m": m,
                         "auc": auc, "power@q.05": power, "fdr": fdr,
                         "rho_hat": rep.get("rho_hat", np.nan),
                         "mu_null": mu_null,
                         "num_flagged": rep["num_flagged"]})
            print(f"s{seed} rho={rho:.2f}: AUC={auc:.3f} power={power:.2f} "
                  f"FDR={fdr:.3f} rho_hat={rep.get('rho_hat', float('nan')):.3f}")
    save_csv(out / "localize_perseed.csv", rows)
    agg = aggregate(rows, ["rho"], ["auc", "power@q.05", "fdr", "rho_hat"])
    save_csv(out / "localize.csv", agg)
    # conformal-BH is FDR-controlling in expectation (E[FDR] ~ .045 in
    # simulation); the tolerance covers the 5-seed-mean sampling spread
    # (sim p95 ~ .06, max-of-80 ~ .065). Quick smoke runs average fewer
    # seeds, so their mean has more spread.
    tol = 0.03 if seeds >= 5 else 0.06
    for r in agg:
        assert r["fdr_mean"] <= 0.05 + tol, f"FDR broken at rho={r['rho']}"


if __name__ == "__main__":
    ap = add_host_args(argparse.ArgumentParser())
    ap.add_argument("--real-csv", default=None,
                    help="genuine rows source (real dataset csv); default: "
                         "unmarked host output (flow validation)")
    ap.add_argument("--info-json", default=None,
                    help="dataset Info json (num_col_idx); default: Adult")
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--codec-mode", default="published",
                    choices=["published", "refit"])
    ap.add_argument("--draws", type=int, default=999,
                    help="key redraws for row null (min row p = 1/(draws+1)). "
                         "At 200 the p floor 1/201 puts ~m/201 null rows in "
                         "ties with marked rows and BH flags the lot "
                         "(default at rho=.05: FDR .085); 999 puts ~m/1000 "
                         "there and restores calibration.")
    main(ap.parse_args())
