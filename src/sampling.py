from __future__ import annotations
from typing import Tuple, Optional
import math

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_sample(
    X: pd.DataFrame,
    y: pd.Series,
    n: int,
    seed: int = 0,
    min_per_class: int = 1,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Stratified down-sample preserving class proportions, with a floor so
    rare-attack classes don't vanish.
    """
    if n >= len(X):
        if verbose:
            print(f"[sample] requested {n} >= available {len(X)}, returning full set")
        return X.reset_index(drop=True), y.reset_index(drop=True)

    counts = y.value_counts()
    if verbose:
        print(f"[sample] before — total={len(X)}, classes={dict(counts)}")

    rng = np.random.RandomState(seed)
    picked_idx = []
    # First reserve the per-class floor.
    for cls, cnt in counts.items():
        floor = min(min_per_class, cnt)
        cls_idx = y.index[y == cls].to_numpy()
        chosen = rng.choice(cls_idx, size=floor, replace=False)
        picked_idx.append(chosen)
    picked = np.concatenate(picked_idx)
    remaining = n - len(picked)

    # Then proportionally fill the rest.
    if remaining > 0:
        remaining_pool = y.index.difference(pd.Index(picked))
        remaining_y = y.loc[remaining_pool]
        # Stratified pick from the remaining pool.
        try:
            _, idx_pick = train_test_split(
                remaining_pool.to_numpy(),
                test_size=min(remaining, len(remaining_pool) - 1) / len(remaining_pool),
                stratify=remaining_y,
                random_state=seed,
            )
            picked = np.concatenate([picked, idx_pick])
        except ValueError:
            # Some class has only 1 remaining sample — fall back to random.
            extra = rng.choice(remaining_pool.to_numpy(), size=remaining, replace=False)
            picked = np.concatenate([picked, extra])

    rng.shuffle(picked)
    Xs = X.loc[picked].reset_index(drop=True)
    ys = y.loc[picked].reset_index(drop=True)
    if verbose:
        c2 = ys.value_counts()
        print(f"[sample] after  — total={len(Xs)}, classes={dict(c2)}")
    return Xs, ys


def train_test_split_stratified(
    X: pd.DataFrame, y: pd.Series, test_size: float, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    # If a class has only 1 sample, stratified split fails — fall back.
    if y.value_counts().min() < 2:
        return train_test_split(X, y, test_size=test_size, random_state=seed)
    return train_test_split(X, y, test_size=test_size,
                            stratify=y, random_state=seed)


def apply_smote(
    X_train: pd.DataFrame, y_train: pd.Series, min_count: int, seed: int,
    max_per_class: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Bring every class up to at least `min_count` samples via SMOTE,
    capped at `max_per_class` to prevent rare-class explosion (e.g.
    CIC-IDS2017 Heartbleed has ~11 raw samples — without a cap, an
    aggressive min_count would synthesise tens of thousands).

    SMOTE requires k_neighbors < class size, so classes too small for
    that constraint are skipped (they stay at their native count).
    """
    from imblearn.over_sampling import SMOTE

    counts = y_train.value_counts()
    target = {}
    for cls, cnt in counts.items():
        if cnt < min_count and cnt >= 2:
            t = max(min_count, cnt)
            if max_per_class is not None:
                t = min(t, max_per_class)
            target[cls] = t
    if not target:
        return X_train, y_train
    # k_neighbors must be < smallest class size in the target.
    k = min(5, min(counts[c] for c in target) - 1)
    if k < 1:
        return X_train, y_train
    # imbalanced-learn ≥0.12 dropped `n_jobs` from SMOTE — handle both.
    try:
        sm = SMOTE(sampling_strategy=target, k_neighbors=k,
                   random_state=seed, n_jobs=-1)
    except TypeError:
        sm = SMOTE(sampling_strategy=target, k_neighbors=k, random_state=seed)
    Xr, yr = sm.fit_resample(X_train, y_train)
    return pd.DataFrame(Xr, columns=X_train.columns), pd.Series(yr, name=y_train.name)
