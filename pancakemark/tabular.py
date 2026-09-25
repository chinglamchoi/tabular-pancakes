"""Model-free tabular canonicalization: table -> whitened z-space.

This is the verifier's front end for (a) the P2 false-accusation audit on
arbitrary real tables and (b) the model-free watermark variant, where the
pancake prior lives in whitened numeric-feature space rather than in a
generator latent.

Pipeline (numeric block only at this stage; categorical channels are P3+):
  1. robust per-column standardization: (x - median) / (IQR / 1.349),
     which estimates the std consistently under Gaussianity and absorbs
     any per-column affine change (unit conversions, rescaling) exactly;
  2. whitening by the shrunk covariance C = (1-lam) S + lam I of the
     standardized columns, via symmetric eigendecomposition with an
     eigenvalue floor -- full-rank by construction, so the map is a fixed,
     invertible affine transform of the table.

Because the codec is an affine map fit either on the verifier's reference
data (canonical mode) or on the suspect table itself (self-fit mode), the
composite "codec + random-key detector" run on key-independent data remains
key-independent -- the MC key-rerandomization certificate applies unchanged.
"""
from __future__ import annotations

import numpy as np


class TableCodec:
    def __init__(self, shrinkage: float = 0.05, eig_floor: float = 1e-4,
                 select_columns: bool = False,
                 min_unique: int = 20, max_mass: float = 0.3):
        self.shrinkage = float(shrinkage)
        self.eig_floor = float(eig_floor)
        # select_columns=True restricts the codec to "continuous-enough"
        # columns: IQR > 0, >= min_unique distinct values, and no single
        # value carrying more than max_mass of the rows. Zero-inflated /
        # heavily discrete columns (adult's capital.gain: ~90% exact
        # zeros) must NOT carry the data-space mark: perturbing an exact
        # point mass is trivially learnable, and their IQR flips from 0
        # to nonzero under marking, breaking standardization refits.
        # max_mass=0.3 also catches adult's hours.per.week (46.7% at 40):
        # marking smears the spike, the refit IQR drifts, and the rescaled
        # projections lose phase coherence (measured on real adult:
        # S 0.90 -> 0.35 when it is included).
        # The rule is deterministic and key-independent (publishable).
        self.select_columns = bool(select_columns)
        self.min_unique = int(min_unique)
        self.max_mass = float(max_mass)
        self.cols_: np.ndarray | None = None
        self.median_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.Wwhiten_: np.ndarray | None = None

    # ------------------------------------------------------------------
    @staticmethod
    def _eligible(X: np.ndarray, min_unique: int, max_mass: float):
        q75, q25 = np.percentile(X, [75, 25], axis=0)
        ok = (q75 - q25) > 0
        for j in range(X.shape[1]):
            if not ok[j]:
                continue
            vals, counts = np.unique(X[:, j], return_counts=True)
            if vals.size < min_unique or counts.max() > max_mass * X.shape[0]:
                ok[j] = False
        return np.flatnonzero(ok)

    def fit(self, X: np.ndarray) -> "TableCodec":
        X = np.asarray(X, dtype=float)
        if self.select_columns:
            self.cols_ = self._eligible(X, self.min_unique, self.max_mass)
            if self.cols_.size < 2:
                raise ValueError("fewer than 2 markable columns")
            X = X[:, self.cols_]
        self.median_ = np.median(X, axis=0)
        q75, q25 = np.percentile(X, [75, 25], axis=0)
        iqr = q75 - q25
        # columns with no spread get scale 1 (they carry no signal either way)
        self.scale_ = np.where(iqr > 0, iqr / 1.349, 1.0)
        Z = (X - self.median_) / self.scale_
        S = np.cov(Z, rowvar=False)
        S = np.atleast_2d(S)
        C = (1 - self.shrinkage) * S + self.shrinkage * np.eye(S.shape[0])
        evals, evecs = np.linalg.eigh(C)
        evals = np.maximum(evals, self.eig_floor)
        self.Wwhiten_ = evecs @ np.diag(evals**-0.5) @ evecs.T   # symmetric ZCA
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.Wwhiten_ is None:
            raise RuntimeError("fit() first")
        X = np.asarray(X, dtype=float)
        if self.cols_ is not None:
            X = X[:, self.cols_]
        Z = (X - self.median_) / self.scale_
        return Z @ self.Wwhiten_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse(self, U: np.ndarray, X_full: np.ndarray | None = None) -> np.ndarray:
        """Whitened rows -> table space (the codec is invertible affine).
        With column selection active, pass the original full table X_full:
        selected columns are replaced, the rest pass through untouched."""
        if self.Wwhiten_ is None:
            raise RuntimeError("fit() first")
        Z = np.asarray(U, dtype=float) @ np.linalg.inv(self.Wwhiten_)
        Xsel = Z * self.scale_ + self.median_
        if self.cols_ is None:
            return Xsel
        if X_full is None:
            raise ValueError("column selection active: inverse needs X_full")
        out = np.asarray(X_full, dtype=float).copy()
        out[:, self.cols_] = Xsel
        return out

    def with_standardization_of(self, X: np.ndarray) -> "TableCodec":
        """Reference-whitening mode: re-fit ONLY the per-column robust
        standardization on X, keep this codec's whitening matrix.

        The standardization step alone absorbs any per-column affine map
        exactly, so affine invariance is preserved -- while the whitening
        directions stay pinned to the (published, key-independent)
        reference fit instead of wobbling with the suspect table's sample
        covariance (a self-fit needs m >> d; at m/d ~ 6 the rotation
        already scrambles phases, e.g. under row subsampling)."""
        if self.Wwhiten_ is None:
            raise RuntimeError("fit() first")
        X = np.asarray(X, dtype=float)
        c = TableCodec(self.shrinkage, self.eig_floor)
        c.cols_ = self.cols_
        if self.cols_ is not None:
            X = X[:, self.cols_]
        c.median_ = np.median(X, axis=0)
        q75, q25 = np.percentile(X, [75, 25], axis=0)
        iqr = q75 - q25
        c.scale_ = np.where(iqr > 0, iqr / 1.349, 1.0)
        c.Wwhiten_ = self.Wwhiten_
        return c

    @property
    def dim(self) -> int:
        return int(self.Wwhiten_.shape[0])
