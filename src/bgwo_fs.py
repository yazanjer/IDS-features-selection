"""
Binary Grey Wolf Optimizer for feature selection (Contribution 1).

Implements BGWO from scratch with:

  * a continuous real-valued position vector per wolf (clipped to [-5, 5]);
  * sigmoid-family transfer functions ('s_shape' / 'v_shape') for binarisation;
  * the standard alpha/beta/delta leader update with linearly-decaying
    exploration coefficient `a` (2 -> 0 over T iterations);
  * empty-mask rescue (a wolf with zero selected features is given one).

The optimizer is agnostic to its fitness landscape — that is delegated to
the `TriObjectiveFitness` object in `fitness.py`. Setting that object's
`gamma=0` disables the SHAP-coherence term and recovers the bi-objective
ablation referenced in the README.

Reference:
    Emary, Zawbaa, Hassanien — "Binary grey wolf optimization approaches
    for feature selection", Neurocomputing 172 (2016).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import Config
from .fitness import TriObjectiveFitness, FitnessBreakdown


@dataclass
class BGWOHistory:
    """Per-iteration trace used for convergence curves and Pareto plots."""
    best_fitness: List[float] = field(default_factory=list)
    best_size: List[int] = field(default_factory=list)
    best_macro_f1: List[float] = field(default_factory=list)
    best_mask: List[np.ndarray] = field(default_factory=list)
    pareto_points: List[Tuple[int, float]] = field(default_factory=list)   # (|S|, F1)


@dataclass
class BGWOResult:
    best_mask: np.ndarray
    best_breakdown: FitnessBreakdown
    history: BGWOHistory
    selected_features: List[str]


class BinaryGreyWolfOptimizer:
    """
    From-scratch Binary Grey Wolf Optimizer for feature selection.

    Real-valued position vectors live in [-5, 5]; on each step we apply an
    S-shaped (or V-shaped) sigmoid transfer function to get binary masks.
    The fitness landscape is delegated to a `TriObjectiveFitness` object
    so the optimizer itself stays agnostic to which objective is active.

    References:
      Emary, Zawbaa, Hassanien — "Binary grey wolf optimization approaches
      for feature selection", Neurocomputing 2016.
    """

    LO, HI = -5.0, 5.0

    def __init__(self, cfg: Config, fitness: TriObjectiveFitness):
        self.cfg = cfg
        self.fitness = fitness
        self.n_features = fitness.n_features
        self.rng = np.random.RandomState(cfg.seed)

    # ------------------------------------------------------------------ #
    # Public entry
    # ------------------------------------------------------------------ #
    def run(self, verbose: bool = True) -> BGWOResult:
        N, T = self.cfg.bgwo_population, self.cfg.bgwo_iterations
        D = self.n_features

        # 1. Initialise population.
        positions = self.rng.uniform(self.LO, self.HI, size=(N, D))
        masks = self._binarize(positions)
        # Enforce non-empty masks.
        for i in range(N):
            if masks[i].sum() == 0:
                masks[i, self.rng.randint(0, D)] = True

        # Evaluate initial fitness.
        breakdowns = [self.fitness.evaluate(masks[i]) for i in range(N)]
        fitnesses = np.array([b.fitness for b in breakdowns])

        # 2. Sort to find alpha/beta/delta wolves.
        a_idx, b_idx, d_idx = self._top3(fitnesses)
        alpha, beta, delta = positions[a_idx].copy(), positions[b_idx].copy(), positions[d_idx].copy()
        alpha_f, beta_f, delta_f = fitnesses[a_idx], fitnesses[b_idx], fitnesses[d_idx]
        alpha_break = breakdowns[a_idx]

        history = BGWOHistory()
        history.best_fitness.append(float(alpha_f))
        history.best_size.append(int(alpha_break.subset_size))
        history.best_macro_f1.append(float(alpha_break.macro_f1))
        history.best_mask.append(self._binarize(alpha[None])[0].copy())
        history.pareto_points.append((alpha_break.subset_size, alpha_break.macro_f1))

        if verbose:
            print(f"[bgwo] init     | best_fit={alpha_f:.4f}  |S|={alpha_break.subset_size:>3}"
                  f"  macroF1={alpha_break.macro_f1:.3f}")

        # 3. Iterate.
        for t in range(1, T + 1):
            a_coef = 2.0 - 2.0 * (t / T)   # linear decay 2 → 0

            new_positions = np.empty_like(positions)
            for i in range(N):
                for d in range(D):
                    # alpha
                    r1, r2 = self.rng.rand(), self.rng.rand()
                    A1, C1 = 2 * a_coef * r1 - a_coef, 2 * r2
                    D1 = abs(C1 * alpha[d] - positions[i, d])
                    X1 = alpha[d] - A1 * D1
                    # beta
                    r1, r2 = self.rng.rand(), self.rng.rand()
                    A2, C2 = 2 * a_coef * r1 - a_coef, 2 * r2
                    D2 = abs(C2 * beta[d] - positions[i, d])
                    X2 = beta[d] - A2 * D2
                    # delta
                    r1, r2 = self.rng.rand(), self.rng.rand()
                    A3, C3 = 2 * a_coef * r1 - a_coef, 2 * r2
                    D3 = abs(C3 * delta[d] - positions[i, d])
                    X3 = delta[d] - A3 * D3
                    new_positions[i, d] = np.clip((X1 + X2 + X3) / 3.0, self.LO, self.HI)

            positions = new_positions
            masks = self._binarize(positions)
            for i in range(N):
                if masks[i].sum() == 0:
                    masks[i, self.rng.randint(0, D)] = True

            breakdowns = [self.fitness.evaluate(masks[i]) for i in range(N)]
            fitnesses = np.array([b.fitness for b in breakdowns])

            a_idx, b_idx, d_idx = self._top3(fitnesses)
            if fitnesses[a_idx] < alpha_f:
                alpha = positions[a_idx].copy()
                alpha_f = fitnesses[a_idx]
                alpha_break = breakdowns[a_idx]
            if fitnesses[b_idx] < beta_f:
                beta = positions[b_idx].copy()
                beta_f = fitnesses[b_idx]
            if fitnesses[d_idx] < delta_f:
                delta = positions[d_idx].copy()
                delta_f = fitnesses[d_idx]

            history.best_fitness.append(float(alpha_f))
            history.best_size.append(int(alpha_break.subset_size))
            history.best_macro_f1.append(float(alpha_break.macro_f1))
            history.best_mask.append(self._binarize(alpha[None])[0].copy())
            for b in breakdowns:
                history.pareto_points.append((b.subset_size, b.macro_f1))

            if verbose:
                print(f"[bgwo] iter {t:>2}/{T} | best_fit={alpha_f:.4f}  |S|={alpha_break.subset_size:>3}"
                      f"  macroF1={alpha_break.macro_f1:.3f}")

        final_mask = self._binarize(alpha[None])[0]
        if final_mask.sum() == 0:
            final_mask[self.rng.randint(0, D)] = True

        selected = [self.fitness.feature_names[i]
                    for i, b in enumerate(final_mask) if b]

        return BGWOResult(
            best_mask=final_mask,
            best_breakdown=alpha_break,
            history=history,
            selected_features=selected,
        )

    # ------------------------------------------------------------------ #
    # Transfer function + helpers
    # ------------------------------------------------------------------ #
    def _binarize(self, positions: np.ndarray) -> np.ndarray:
        if self.cfg.bgwo_transfer == "v_shape":
            p = np.abs(np.tanh(positions))
        else:  # S-shape
            p = 1.0 / (1.0 + np.exp(-positions))
        return (self.rng.rand(*p.shape) < p)

    @staticmethod
    def _top3(fitnesses: np.ndarray) -> Tuple[int, int, int]:
        order = np.argsort(fitnesses)
        return int(order[0]), int(order[1]), int(order[2])


# ====================================================================== #
# Baseline FS methods — for clean ablation against BGWO.
# ====================================================================== #
def filter_fs(
    X: pd.DataFrame, y: pd.Series, k: Optional[int] = None
) -> List[str]:
    """Information-gain / mutual-information ranking (mirrors the original
    repo's filter-based selection)."""
    from sklearn.feature_selection import mutual_info_classif

    mi = mutual_info_classif(X, y, random_state=0)
    order = np.argsort(mi)[::-1]
    if k is None:
        k = max(int(np.ceil(0.5 * X.shape[1])), 5)   # default: top 50%
    chosen = order[:k]
    return [X.columns[i] for i in chosen]


def all_features(X: pd.DataFrame, y: pd.Series) -> List[str]:
    return list(X.columns)
