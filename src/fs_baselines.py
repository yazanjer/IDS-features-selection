"""
Modern wrapper-baseline feature-selection methods.

Adds four standard wrapper / embedded baselines that slot into the same
`fs_method` switch as the BGWO variants, so they appear in the experiment
matrix, the Wilcoxon tests, and every plot without further plumbing:

    rfe       — Recursive Feature Elimination with a LightGBM estimator
    lasso     — L1-penalised multinomial logistic regression
    rf_imp    — Random-Forest impurity-importance ranking (top-k)
    boruta    — In-house Boruta (shadow-feature permutation test on RF
                importances). No external dependency.

Each function takes (X, y, seed, …) and returns a list of selected
column names. All are deterministic given the seed (Boruta is
stochastic — it averages over `n_iter` runs but every run is seeded).
"""
from __future__ import annotations
from typing import List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def _default_k(n_features: int, fraction: float = 0.5) -> int:
    """Default 'keep top half, but at least 5'."""
    return max(5, int(np.ceil(fraction * n_features)))


# ====================================================================== #
# 1. Recursive Feature Elimination (RFE) with LightGBM
# ====================================================================== #
def rfe_fs(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int = 0,
    n_keep: Optional[int] = None,
    step: float = 0.1,
) -> List[str]:
    """
    Wrap LightGBM in sklearn.feature_selection.RFE. RFE repeatedly fits
    the estimator and prunes the least-important `step` fraction of the
    remaining features until `n_keep` survive.
    """
    from sklearn.feature_selection import RFE
    import lightgbm as lgb

    k = n_keep if n_keep is not None else _default_k(X.shape[1])
    estimator = lgb.LGBMClassifier(
        random_state=seed, verbosity=-1, n_estimators=80
    )
    rfe = RFE(estimator=estimator, n_features_to_select=k, step=step)
    rfe.fit(X, y)
    return [X.columns[i] for i, keep in enumerate(rfe.support_) if keep]


# ====================================================================== #
# 2. LASSO logistic regression (L1)
# ====================================================================== #
def lasso_fs(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int = 0,
    C: float = 0.1,
    max_iter: int = 300,
) -> List[str]:
    """
    Multinomial L1 logistic regression with standardised inputs. Keeps
    every feature that ends up with a non-zero coefficient in at least
    one class. Smaller `C` => sparser solution.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    Xs = StandardScaler().fit_transform(X)
    lr = LogisticRegression(
        penalty="l1", solver="saga", C=C, max_iter=max_iter,
        random_state=seed, n_jobs=-1,
    )
    lr.fit(Xs, y)

    # coef_ shape: (n_classes, n_features). Keep features non-zero in *any* class.
    coefs = np.atleast_2d(lr.coef_)
    nonzero = np.abs(coefs).max(axis=0) > 1e-10
    chosen = [X.columns[i] for i in range(X.shape[1]) if nonzero[i]]

    # Safety net — never return an empty subset.
    if not chosen:
        order = np.argsort(np.abs(coefs).max(axis=0))[::-1]
        chosen = [X.columns[i] for i in order[: _default_k(X.shape[1])]]
    return chosen


# ====================================================================== #
# 3. Random-Forest impurity-importance ranking
# ====================================================================== #
def rf_importance_fs(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int = 0,
    n_keep: Optional[int] = None,
    n_estimators: int = 120,
) -> List[str]:
    """
    Fit a single RandomForest and keep the top-k features by Gini
    importance. The cheapest 'wrapper-ish' baseline — fast and a
    standard reference in IDS-FS literature.
    """
    from sklearn.ensemble import RandomForestClassifier

    k = n_keep if n_keep is not None else _default_k(X.shape[1])
    rf = RandomForestClassifier(
        n_estimators=n_estimators, n_jobs=-1, random_state=seed,
    )
    rf.fit(X, y)
    order = np.argsort(rf.feature_importances_)[::-1]
    return [X.columns[i] for i in order[:k]]


# ====================================================================== #
# 4. Boruta — in-house implementation
# ====================================================================== #
def boruta_fs(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int = 0,
    n_iter: int = 25,
    n_estimators: int = 100,
    perc: int = 100,
    max_features_fallback: Optional[int] = None,
) -> List[str]:
    """
    Minimal Boruta — no external dependency.

    Procedure (Kursa & Rudnicki, 2010):
      1. Duplicate every feature and shuffle each duplicate column
         independently — these are the *shadow* features.
      2. Fit a RandomForest on the (real ∥ shadow) matrix.
      3. Each real feature whose importance exceeds the maximum shadow
         importance scores a "hit" this iteration.
      4. Repeat `n_iter` times. A feature is *confirmed* if its hit-rate
         is at least `perc/100` × maximum possible.
      5. If no feature is confirmed (rare on small smoke configs),
         fall back to top-k by hit count.

    The `perc` parameter mirrors the original BorutaPy knob: 100 is the
    strict default; 80–90 relaxes it.
    """
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.RandomState(seed)
    X_arr = X.to_numpy()
    n_samples, n_features = X_arr.shape
    feat_names = list(X.columns)
    hits = np.zeros(n_features, dtype=int)

    for it in range(n_iter):
        # Shadow matrix — independently permute each column.
        shadow = np.column_stack([
            rng.permutation(X_arr[:, j]) for j in range(n_features)
        ])
        X_combined = np.hstack([X_arr, shadow])
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            n_jobs=-1,
            random_state=seed + it,
        )
        rf.fit(X_combined, y)
        imp = rf.feature_importances_
        real_imp = imp[:n_features]
        shadow_imp = imp[n_features:]
        # `perc` percentile of shadow importances becomes the bar to clear.
        threshold = np.percentile(shadow_imp, perc) if perc < 100 else shadow_imp.max()
        hits += (real_imp > threshold).astype(int)

    hit_rate = hits / n_iter
    confirmed = [feat_names[i] for i in range(n_features) if hit_rate[i] >= 0.5]

    if not confirmed:
        # Fallback: top-k by hit count.
        k = max_features_fallback or _default_k(n_features)
        order = np.argsort(hits)[::-1]
        confirmed = [feat_names[i] for i in order[:k]]
    return confirmed


# ====================================================================== #
# Dispatch helper — keeps evaluation.py's run_one() tidy.
# ====================================================================== #
WRAPPER_METHODS = {
    "rfe":    rfe_fs,
    "lasso":  lasso_fs,
    "rf_imp": rf_importance_fs,
    "boruta": boruta_fs,
}


def run_wrapper_method(name: str, X: pd.DataFrame, y: pd.Series, seed: int) -> List[str]:
    if name not in WRAPPER_METHODS:
        raise ValueError(f"Unknown wrapper method: {name!r}. "
                         f"Available: {list(WRAPPER_METHODS)}")
    return WRAPPER_METHODS[name](X, y, seed=seed)
