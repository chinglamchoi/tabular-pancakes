"""Attack suite: the tampering operations a dishonest releaser might apply.

Two families:
  Table attacks  operate on the released numeric table X (rows x features),
                 using only quantities the attacker can compute from X.
  Latent attacks operate on latents/latent estimates directly (used for
                 controlled scrub studies where the attacker is granted
                 latent access -- an over-powered adversary, reported as such).

All attacks are keyless: none receives the secret frame W. The keyed scrub
oracle (for the removal-cost separation) is explicit and named as such.

Also here: the covariance distinguisher (CLUE-Mark Alg. 2 style), the best
known keyless detector for pancake structure at small gamma; used in the
P8 sample-constrained security study.
"""
from __future__ import annotations

import numpy as np


# ------------------------------------------------------------ table attacks
def shuffle_rows(X: np.ndarray, rng) -> np.ndarray:
    return X[rng.permutation(X.shape[0])]


def subsample_rows(X: np.ndarray, frac: float, rng) -> np.ndarray:
    m = max(1, int(round(frac * X.shape[0])))
    idx = rng.choice(X.shape[0], size=m, replace=False)
    return X[idx]


def gaussian_cell_noise(X: np.ndarray, frac_of_std: float, rng) -> np.ndarray:
    """Add N(0, (frac*std_j)^2) to every cell -- 'toggling numbers slightly'."""
    s = X.std(axis=0, keepdims=True)
    return X + frac_of_std * s * rng.standard_normal(X.shape)


def round_decimals(X: np.ndarray, decimals: int) -> np.ndarray:
    return np.round(X, decimals)


def round_significant(X: np.ndarray, digits: int) -> np.ndarray:
    """Round to `digits` significant figures (reported-precision attack)."""
    Xa = np.where(X == 0, 1e-300, X)
    mag = np.floor(np.log10(np.abs(Xa)))
    factor = 10.0 ** (digits - 1 - mag)
    return np.round(X * factor) / factor


def winsorize(X: np.ndarray, q: float) -> np.ndarray:
    """Top/bottom coding at quantiles q and 1-q (outlier clipping)."""
    lo = np.quantile(X, q, axis=0, keepdims=True)
    hi = np.quantile(X, 1 - q, axis=0, keepdims=True)
    return np.clip(X, lo, hi)


def affine_rescale(X: np.ndarray, rng) -> np.ndarray:
    """Per-column unit changes: x -> a*x + b (kg->lb etc.)."""
    a = rng.uniform(0.2, 8.0, size=(1, X.shape[1]))
    b = rng.normal(0, 20.0, size=(1, X.shape[1]))
    return X * a + b


def permute_columns(X: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
    perm = rng.permutation(X.shape[1])
    return X[:, perm], perm


def edit_rows(X: np.ndarray, frac_rows: float, rng,
              replacement: np.ndarray | None = None) -> np.ndarray:
    """Replace a rho-fraction of rows arbitrarily (here: resampled from the
    empirical column marginals, i.e. utility-preserving replacement)."""
    X = X.copy()
    m = X.shape[0]
    idx = rng.choice(m, size=int(round(frac_rows * m)), replace=False)
    if replacement is None:
        # independent bootstrap per column: matches marginals, breaks rows
        for j in range(X.shape[1]):
            X[idx, j] = rng.choice(X[:, j], size=idx.size, replace=True)
    else:
        X[idx] = replacement[rng.choice(replacement.shape[0], idx.size)]
    return X


def few_cell_edits(X: np.ndarray, num_cells: int, size_in_std: float, rng) -> np.ndarray:
    """Adversarial 'manual toggling' of a handful of salient cells."""
    X = X.copy()
    s = X.std(axis=0)
    ii = rng.integers(0, X.shape[0], size=num_cells)
    jj = rng.integers(0, X.shape[1], size=num_cells)
    X[ii, jj] += size_in_std * s[jj] * rng.choice([-1.0, 1.0], size=num_cells)
    return X


def reimpute_cells(X: np.ndarray, frac_cells: float, rng,
                   max_iter: int = 8) -> np.ndarray:
    """Delete a random frac of cells and re-impute with sklearn's
    IterativeImputer (MICE-style ridge chained regression)."""
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer

    X = X.copy().astype(float)
    mask = rng.random(X.shape) < frac_cells
    X[mask] = np.nan
    imp = IterativeImputer(max_iter=max_iter, random_state=int(rng.integers(1 << 31)),
                           sample_posterior=False)
    return imp.fit_transform(X)


def mix_with_real(X_marked: np.ndarray, X_real: np.ndarray, rho: float,
                  rng) -> tuple[np.ndarray, np.ndarray]:
    """rho-fraction marked rows embedded among real rows (scenario D2).
    Returns (table, is_marked mask), rows shuffled."""
    m = X_real.shape[0]
    n_marked = int(round(rho * m))
    keep_real = m - n_marked
    Xr = X_real[rng.choice(X_real.shape[0], keep_real, replace=False)]
    Xm = X_marked[rng.choice(X_marked.shape[0], n_marked, replace=False)]
    X = np.vstack([Xr, Xm])
    y = np.r_[np.zeros(keep_real, bool), np.ones(n_marked, bool)]
    perm = rng.permutation(m)
    return X[perm], y[perm]


# ----------------------------------------------------------- latent attacks
def latent_isotropic_noise(z: np.ndarray, sigma: float, rng) -> np.ndarray:
    """Keyless scrub: noise spent isotropically over all n coordinates."""
    return z + sigma * rng.standard_normal(z.shape)


def latent_keyed_scrub(z: np.ndarray, W: np.ndarray, sigma: float, rng) -> np.ndarray:
    """KEYED oracle scrub: noise spent only along the k secret directions.
    Requires W -- exists only to quantify the keyed/keyless separation."""
    eps = sigma * rng.standard_normal((z.shape[0], W.shape[1]))
    return z + eps @ W.T


# ------------------------------------- keyless covariance distinguisher
def covariance_score(z: np.ndarray) -> float:
    """max_i |eigval_i(cov(z)) - 1|: the best known keyless test for
    pancake structure (variance deviation along the hidden direction),
    after the attacker whitens with the *presumed* prior scale.
    CLUE-Mark Alg. 2, adapted to the standard-normal convention."""
    C = np.cov(z, rowvar=False)
    evals = np.linalg.eigvalsh(np.atleast_2d(C))
    return float(np.max(np.abs(evals - 1.0)))


def covariance_attack_auc(sample_marked, sample_null, trials: int,
                          rng) -> float:
    """AUC of covariance_score for distinguishing marked from null batches.
    `sample_marked`/`sample_null`: callables rng -> (m, n) batch."""
    s1 = np.array([covariance_score(sample_marked(rng)) for _ in range(trials)])
    s0 = np.array([covariance_score(sample_null(rng)) for _ in range(trials)])
    # exact Mann-Whitney AUC
    order = np.argsort(np.r_[s0, s1], kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, order.size + 1)
    r1 = ranks[len(s0):].sum()
    return float((r1 - trials * (trials + 1) / 2) / (trials * trials))
