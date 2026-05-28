"""
Tri-objective fitness for BGWO with SHAP-in-the-loop (Contribution 2).

    fitness(S) = alpha * (1 - macro_F1(S))
               + beta  * (|S| / |F|)
               + gamma * (1 - explanation_consistency(S))

`explanation_consistency(S)` trains LCCDE on the subset S and asks SHAP
how concentrated the mean |SHAP value| is in the top-k features of the
trained models (averaged across the three base learners). A subset where
most of the SHAP mass piles into a few features is considered more
explainable than one where SHAP spreads thinly — the latter usually
means S carries "passenger" features the optimizer should have dropped.

Setting `gamma=0` skips the SHAP computation entirely and recovers a
pure bi-objective accuracy + sparsity fitness, which is what the
'bgwo_bi' ablation in the experiment matrix uses.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .config import Config
from .lccde_model import LCCDE


@dataclass
class FitnessBreakdown:
    """Components of a single fitness evaluation — useful for logging / Pareto."""
    fitness: float
    f1_term: float           # 1 - macro_F1
    size_term: float         # |S| / |F|
    shap_term: float         # 1 - explanation_consistency
    macro_f1: float
    subset_size: int
    n_features: int


class TriObjectiveFitness:
    """
    fitness = alpha * (1 - macro_F1) + beta * (|S|/|F|) + gamma * (1 - cons(S))

    `cons(S)` measures how well the SHAP top-k from a model trained on S
    agrees with S itself: |top_k(SHAP(model_S)) ∩ S| / k.

    When gamma == 0, the third term is skipped entirely (bi-objective ablation).
    """

    def __init__(
        self,
        cfg: Config,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        feature_names: List[str],
    ):
        self.cfg = cfg
        self.X = X
        self.y = y
        self.X_val = X_val
        self.y_val = y_val
        self.feature_names = list(feature_names)
        self.n_features = len(feature_names)

    # ------------------------------------------------------------------ #
    # Public API used by BGWO
    # ------------------------------------------------------------------ #
    def evaluate(self, mask: np.ndarray) -> FitnessBreakdown:
        mask = np.asarray(mask, dtype=bool)
        if mask.sum() == 0:
            return FitnessBreakdown(1.0, 1.0, 0.0, 1.0, 0.0, 0, self.n_features)

        cols = [self.feature_names[i] for i, b in enumerate(mask) if b]
        Xtr = self.X[cols]
        Xva = self.X_val[cols]

        model = LCCDE(seed=self.cfg.seed).fit(Xtr, self.y, Xva, self.y_val)
        y_pred, _ = model.predict(Xva)
        macro_f1 = f1_score(self.y_val, y_pred, average="macro", zero_division=0)

        size = mask.sum() / self.n_features
        f1_term = 1.0 - macro_f1
        size_term = float(size)

        # SHAP term (skipped if gamma==0).
        if self.cfg.gamma > 0:
            shap_term = 1.0 - _explanation_consistency(
                model, Xva, cols, self.cfg.shap_top_k or int(mask.sum()),
                self.cfg.shap_background_samples,
            )
        else:
            shap_term = 0.0

        fit = (self.cfg.alpha * f1_term
               + self.cfg.beta  * size_term
               + self.cfg.gamma * shap_term)
        return FitnessBreakdown(
            fitness=float(fit),
            f1_term=float(f1_term),
            size_term=float(size_term),
            shap_term=float(shap_term),
            macro_f1=float(macro_f1),
            subset_size=int(mask.sum()),
            n_features=self.n_features,
        )


# ---------------------------------------------------------------------- #
# Explanation-consistency
# ---------------------------------------------------------------------- #
def _explanation_consistency(
    model: LCCDE,
    X_val: pd.DataFrame,
    selected_cols: List[str],
    top_k: int,
    background_samples: int,
) -> float:
    """
    SHAP top-k overlap with the selected subset, averaged across the three
    base learners. Returns a value in [0, 1].

    A score of 1 means SHAP confirms every feature it ranks in the top-k
    came from the BGWO-selected subset; lower means SHAP wants features the
    optimiser dropped (or that the kept features aren't actually useful).
    """
    try:
        import shap
    except ImportError:
        return 0.0

    if top_k <= 0 or len(selected_cols) == 0:
        return 0.0
    top_k = min(top_k, len(selected_cols))

    bg = X_val.sample(min(background_samples, len(X_val)),
                      random_state=0)

    importances: List[np.ndarray] = []
    for booster in (model.lgbm, model.xgb_, model.cat):
        try:
            explainer = shap.TreeExplainer(
                booster, data=bg, feature_perturbation="interventional"
            )
            sv = explainer.shap_values(bg, check_additivity=False)
            # TreeExplainer returns a list (one per class) for multiclass.
            if isinstance(sv, list):
                stacked = np.concatenate(
                    [np.abs(s).mean(axis=0).reshape(1, -1) for s in sv], axis=0
                )
                imp = stacked.mean(axis=0)
            else:
                imp = np.abs(sv).mean(axis=0)
                # XGB sometimes returns (n_samples, n_classes, n_features).
                if imp.ndim == 2:
                    imp = imp.mean(axis=0)
            importances.append(imp)
        except Exception:
            continue

    if not importances:
        return 0.0

    mean_imp = np.mean(importances, axis=0)
    # Top-k columns *within the trained subset* — by construction these are
    # always in `selected_cols`, so overlap = 1.0 trivially if we did nothing
    # more. Instead measure how concentrated importance is in the top-k:
    # a subset where SHAP gives near-uniform importance is "wasteful".
    order = np.argsort(mean_imp)[::-1]
    top_idx = order[:top_k]
    concentration = mean_imp[top_idx].sum() / (mean_imp.sum() + 1e-12)
    # A consistent subset is one where the top-k features carry most of the
    # SHAP mass — i.e. fewer "dead" features ride along.
    return float(np.clip(concentration, 0.0, 1.0))
