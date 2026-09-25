"""Data-space PancakeMark: the diffusion-VAE fallback.

Motivation (exact-inversion measurements on TabSyn--Adult): on a diffusion+VAE host (TabSyn) the
release map contracts onto the data manifold, so ANY z-space verifier must
invert an expanding map that amplifies input-side noise (solver floor, VAE
reconstruction, table rounding) by a step-count-independent factor of
~50-100x. Sign-level schemes (TabWak) survive; phase-level schemes cannot
-- not ours, not anybody's. The robust instantiation for such hosts plants
the pancakes where the verifier lives: in the WHITENED TABLE representation
(TableCodec), whose verifier map is near-isometric by construction and
exactly invariant to per-column affine changes.

Provider side (QIM-style projection at release time):
    u  = codec.transform(X)                        whitened rows
    t  = u @ W                                     secret-direction coords
    m  = round(f t - delta)                        nearest pancake layer
    t' = (m + delta) gamma/gamma'^2  + (beta/gamma') eps    conditional draw
    X' = codec.inverse(u + (t' - t) @ W.T)

Properties:
  * Variance-exact: Var(t') = (gamma^2 + beta^2)/gamma'^2 = 1, so the
    whitened second moments -- and hence a re-fit codec -- are unchanged;
    the snapped layer index of a ~N(0,1) coordinate is automatically the
    discrete Gaussian with std gamma' that hCLWE prescribes.
  * Verification: the SAME detector as the latent variant (row_scores on
    codec.transform(X')), self-fit on the suspect table: no model, no key
    at embed... (key needed at embed here -- this variant is keyed at
    embed time, like the latent variant).
  * Robustness: attack noise enters the statistic directly (no inversion
    amplification): mu = exp(-2 pi^2 (jitter^2 + (f sigma_attack)^2)),
    i.e. the flow-host P3 numbers.
  * Undetectability: no longer reducible to hCLWE against the true data
    distribution (the data is not exactly whitened-Gaussian); it is
    audited empirically by the C2ST (p12) and inherits the hCLWE hardness
    of recovering W from the whitened rows.
  * Cost: per-row distortion Unif(-1/2f, 1/2f)-scale along k directions
    only: whitened rms ~ k / (12 f^2) total, reported per cell by p12.
"""
from __future__ import annotations

import numpy as np

from . import theory
from .keys import Key
from .tabular import TableCodec


class DataSpaceMark:
    """QIM projection of whitened rows onto the keyed pancake conditional."""

    def __init__(self, key: Key, codec: TableCodec):
        if codec.Wwhiten_ is None:
            raise RuntimeError("codec must be fit before marking")
        if key.n != codec.dim:
            raise ValueError(f"key.n={key.n} != codec.dim={codec.dim} "
                             "(with column selection, key.n = number of "
                             "eligible columns)")
        self.key = key
        self.codec = codec
        self.f = theory.score_frequency(key.gamma, key.beta)
        gp = theory.gamma_prime(key.gamma, key.beta)
        self._mean_scale = key.gamma / gp ** 2
        self._cond_std = key.beta / gp

    def mark_whitened(self, U: np.ndarray, rng, jitter: bool = True):
        T = U @ self.key.W                                  # (m, k)
        m_layer = np.round(self.f * T - self.key.delta)
        T2 = (m_layer + self.key.delta) * self._mean_scale
        if jitter:
            T2 = T2 + self._cond_std * rng.standard_normal(T.shape)
        return U + (T2 - T) @ self.key.W.T, float(np.sqrt(np.mean((T2 - T) ** 2)))

    def mark(self, X: np.ndarray, rng, jitter: bool = True,
             preserve_format: bool = True):
        """Numeric block -> marked numeric block. Returns (X', report).

        preserve_format re-rounds integer-valued marked columns and clips
        marked columns to their observed ranges, so the release carries no
        formatting tell (a non-integer age, a negative hours count). The
        induced quantization is the round_sig/winsorize noise the P3/P12
        sweeps already show the detector survives."""
        X = np.asarray(X, dtype=float)
        U = self.codec.transform(X)
        U2, rms_onaxis = self.mark_whitened(U, rng, jitter=jitter)
        X2 = self.codec.inverse(U2, X_full=X)
        cols = (self.codec.cols_ if self.codec.cols_ is not None
                else np.arange(X.shape[1]))
        if preserve_format:
            for j in cols:
                col = X[:, j]
                if np.allclose(col, np.round(col)):        # integer column
                    X2[:, j] = np.round(X2[:, j])
                X2[:, j] = np.clip(X2[:, j], col.min(), col.max())
        colstd = X.std(axis=0)
        colstd[colstd == 0] = 1.0
        cell = float(np.mean(np.sqrt(np.mean((X2 - X) ** 2, axis=0)) / colstd))
        return X2, {
            "rms_onaxis_whitened": rms_onaxis,
            "mean_cell_rmse_stdunits": cell,
            "f": self.f,
            "k": self.key.k,
            "marked_columns": int(len(cols)),
            "total_columns": int(X.shape[1]),
        }


def detect_data_space(X: np.ndarray, key: Key,
                      codec: TableCodec | None = None,
                      method: str = "hoeffding",
                      select_columns: bool = False,
                      columns=None,
                      mc_draws: int = 999):
    """Verifier -> whitened rows -> the standard pancake detector.

    codec=None: full self-fit on the suspect (fine when m >> d).
    codec=None, columns=<indices>: key-holder self-fit -- the marked
    column list is part of the key material, so the verifier refits
    standardization + whitening on exactly those columns of the suspect.
    This is the STABLE no-published-codec mode: re-running the public
    eligibility rule on a MARKED table can flip boundary columns (the
    mark + integer re-round concentrates mass; measured on default:
    14 eligible pre-mark vs 9 post-mark), while the column list itself
    never changes.
    codec=reference: robust standardization is re-fit on the suspect
    (exact per-column affine invariance) while the reference's whitening
    matrix is kept (stable at any suspect size). The codec is
    key-independent, so the reference can be published.

    method: "hoeffding" is the analytic bound; on real (non-Gaussian,
    integer-latticed) data its zero-null-mean assumption degrades at LOW
    score frequency (adult at gamma=1: unmarked stat ~ +0.06), so the
    certified verdict on real data is method="mc" -- exact key
    rerandomization, valid because the codec is key-independent."""
    from .detect import detect as _detect
    X = np.asarray(X, dtype=float)
    if codec is None and columns is not None:
        cols = np.asarray(columns, dtype=int)
        if cols.size != key.n:
            raise ValueError(f"len(columns)={cols.size} != key.n={key.n}")
        sub = TableCodec(select_columns=False).fit(X[:, cols])
        return _detect(sub.transform(X[:, cols]), key, method=method,
                       mc_draws=mc_draws)
    codec = TableCodec(select_columns=select_columns).fit(X) if codec is None \
        else codec.with_standardization_of(X)
    return _detect(codec.transform(X), key, method=method, mc_draws=mc_draws)
