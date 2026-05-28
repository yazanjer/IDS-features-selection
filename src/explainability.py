"""
Explainability layer for the LCCDE ensemble.

Two roles in this project:

  1. Inside the BGWO loop, `fitness.py` calls into the SHAP routines here
     to compute an explanation-coherence term that becomes part of the
     fitness function (Contribution 2).
  2. After training, the same routines extract per-attack-class
     "minimal explainable signatures" — the top-k features SHAP says
     most drive each class — and a fidelity score for the final model.

Also provides:

  * Kuncheva index + Jaccard for ranking stability across seeds.
  * `variable_stability` — mean pairwise Kuncheva of top-k subsets.
  * `cross_dataset_overlap` — Jaccard of feature subsets from different
    datasets, used by `plots.plot_cross_dataset_overlap`.
  * LIME baseline (`lime_signatures`) for the SHAP-vs-LIME fidelity
    comparison referenced in the README.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .lccde_model import LCCDE


# ====================================================================== #
# Per-class SHAP signatures
# ====================================================================== #
@dataclass
class ShapSignatures:
    """SHAP-derived per-class feature importance for a fitted LCCDE."""
    per_class_importance: Dict[int, np.ndarray]   # cls_id -> array[D]
    feature_names: List[str]
    classes: List[int]
    leader_per_class: Dict[int, str]

    def top_k_for_class(self, cls: int, k: int = 5) -> List[Tuple[str, float]]:
        imp = self.per_class_importance[int(cls)]
        order = np.argsort(imp)[::-1][:k]
        return [(self.feature_names[i], float(imp[i])) for i in order]

    def signature_table(self, k: int = 5) -> pd.DataFrame:
        rows = []
        for cls in self.classes:
            for rank, (name, val) in enumerate(self.top_k_for_class(cls, k), 1):
                rows.append({"class": int(cls), "rank": rank,
                             "feature": name, "importance": val})
        return pd.DataFrame(rows)


def compute_shap_signatures(
    model: LCCDE,
    X_sample: pd.DataFrame,
    background_n: int = 100,
) -> ShapSignatures:
    """
    For each class, compute |SHAP| from that class's leader model and
    average across samples. Returns a `ShapSignatures` object.
    """
    import shap

    bg = X_sample.sample(min(background_n, len(X_sample)), random_state=0)
    n_features = X_sample.shape[1]
    feature_names = list(X_sample.columns)

    # Pre-explain each base learner once.
    explanations: Dict[str, np.ndarray] = {}
    for tag, booster in (("lgbm", model.lgbm), ("xgb", model.xgb_), ("cat", model.cat)):
        try:
            ex = shap.TreeExplainer(booster, data=bg,
                                    feature_perturbation="interventional")
            X_eval = np.asarray(bg) if tag == "xgb" else bg
            sv = ex.shap_values(X_eval, check_additivity=False)
            # Normalize to (n_classes, n_samples, n_features)
            if isinstance(sv, list):
                arr = np.stack([np.asarray(s) for s in sv], axis=0)
            else:
                arr = np.asarray(sv)
                if arr.ndim == 3:
                    # XGB returns (n_samples, n_classes, n_features) — transpose.
                    arr = arr.transpose(1, 0, 2)
                elif arr.ndim == 2:
                    arr = arr[None]   # binary case
            explanations[tag] = np.abs(arr).mean(axis=1)   # (n_classes, n_features)
        except Exception as e:
            print(f"[shap] skipping {tag}: {e}")

    classes = list(model.classes_)
    per_class: Dict[int, np.ndarray] = {}
    for ci, cls in enumerate(classes):
        leader = model.leader_per_class.get(int(cls), "lgbm")
        if leader in explanations and ci < explanations[leader].shape[0]:
            per_class[int(cls)] = explanations[leader][ci]
        elif explanations:
            # Fall back to the average across whichever explainers worked.
            stacks = []
            for v in explanations.values():
                if ci < v.shape[0]:
                    stacks.append(v[ci])
            per_class[int(cls)] = (
                np.mean(stacks, axis=0) if stacks else np.zeros(n_features)
            )
        else:
            per_class[int(cls)] = np.zeros(n_features)

    return ShapSignatures(
        per_class_importance=per_class,
        feature_names=feature_names,
        classes=[int(c) for c in classes],
        leader_per_class=dict(model.leader_per_class),
    )


# ====================================================================== #
# Stability metrics across seeds / datasets
# ====================================================================== #
def kuncheva_index(set_a: Sequence[str], set_b: Sequence[str], universe: int) -> float:
    """
    Kuncheva stability index ∈ [-1, 1]. 1 = identical subsets.
    """
    A, B = set(set_a), set(set_b)
    k = len(A)
    if k != len(B) or k == 0 or k == universe:
        # Fallback: Jaccard if cardinalities differ (Kuncheva is undefined).
        return jaccard(A, B)
    r = len(A & B)
    num = (r * universe) - (k * k)
    den = (k * (universe - k))
    if den == 0:
        return 1.0
    return num / den


def jaccard(a, b) -> float:
    A, B = set(a), set(b)
    if not A and not B:
        return 1.0
    return len(A & B) / len(A | B)


def variable_stability(
    rankings: List[List[str]], top_k: int, universe: int
) -> float:
    """Mean pairwise Kuncheva of top-k feature subsets across runs."""
    if len(rankings) < 2:
        return 1.0
    subs = [r[:top_k] for r in rankings]
    scores = []
    for i in range(len(subs)):
        for j in range(i + 1, len(subs)):
            scores.append(kuncheva_index(subs[i], subs[j], universe))
    return float(np.mean(scores)) if scores else 1.0


def cross_dataset_overlap(
    features_a: Sequence[str], features_b: Sequence[str]
) -> float:
    return jaccard(features_a, features_b)


# ====================================================================== #
# Fidelity
# ====================================================================== #
def explanation_fidelity(
    model: LCCDE,
    X_test: pd.DataFrame,
    sig: ShapSignatures,
    top_k: int = 10,
) -> float:
    """
    Faithfulness: mean drop in confidence on a sample's predicted class
    when its top-k SHAP features are replaced by their column median.
    Higher is better (in [0, 1] approximately).
    """
    sample = X_test.sample(min(200, len(X_test)), random_state=0).reset_index(drop=True)
    base_preds, _ = model.predict(sample)
    medians = X_test.median(axis=0)

    drops = []
    # Use the per-sample predicted class's signature.
    for i in range(len(sample)):
        cls = int(base_preds[i])
        if cls not in sig.per_class_importance:
            continue
        imp = sig.per_class_importance[cls]
        order = np.argsort(imp)[::-1][:top_k]

        leader = model.leader_per_class.get(cls, "lgbm")
        booster = {"lgbm": model.lgbm, "xgb": model.xgb_, "cat": model.cat}[leader]
        x0 = sample.iloc[[i]].copy()
        p0 = booster.predict_proba(np.asarray(x0) if leader == "xgb" else x0)
        c0 = float(p0[0, list(model.classes_).index(cls)])

        x1 = x0.astype(float).copy()
        for j in order:
            x1.iloc[0, j] = float(medians.iloc[j])
        p1 = booster.predict_proba(np.asarray(x1) if leader == "xgb" else x1)
        c1 = float(p1[0, list(model.classes_).index(cls)])
        drops.append(max(0.0, c0 - c1))
    return float(np.mean(drops)) if drops else 0.0


# ====================================================================== #
# LIME comparison
# ====================================================================== #
def lime_signatures(
    model: LCCDE,
    X_sample: pd.DataFrame,
    y_sample: pd.Series,
    n_samples_to_explain: int = 50,
    top_k: int = 10,
) -> Dict[int, List[str]]:
    """Per-class LIME top-k features for fidelity/stability comparison."""
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError:
        return {}

    explainer = LimeTabularExplainer(
        training_data=np.asarray(X_sample),
        feature_names=list(X_sample.columns),
        class_names=[str(c) for c in model.classes_],
        discretize_continuous=False,
        mode="classification",
        random_state=0,
    )

    leader_booster = model.lgbm  # cheapest; LIME is model-agnostic anyway

    def predict_fn(arr):
        return leader_booster.predict_proba(pd.DataFrame(arr, columns=X_sample.columns))

    per_class_counts: Dict[int, Dict[str, float]] = {}
    sample_idx = np.random.RandomState(0).choice(
        len(X_sample), size=min(n_samples_to_explain, len(X_sample)), replace=False
    )
    for i in sample_idx:
        x = np.asarray(X_sample.iloc[i])
        try:
            exp = explainer.explain_instance(
                x, predict_fn, num_features=top_k, num_samples=200,
            )
            cls = int(y_sample.iloc[i])
            per_class_counts.setdefault(cls, {})
            for feat_name, weight in exp.as_list():
                # LIME returns conditions like "feat <= 0.5"; keep just the feat name.
                fname = feat_name.split(" ")[0]
                per_class_counts[cls][fname] = (
                    per_class_counts[cls].get(fname, 0.0) + abs(weight)
                )
        except Exception:
            continue

    out: Dict[int, List[str]] = {}
    for cls, d in per_class_counts.items():
        out[cls] = [k for k, _ in sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:top_k]]
    return out
