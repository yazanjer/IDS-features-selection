from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import Config
from .explainability import ShapSignatures


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def _savefig(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p_png = out_dir / f"{name}.png"
    p_pdf = out_dir / f"{name}.pdf"
    fig.savefig(p_png, dpi=140, bbox_inches="tight")
    fig.savefig(p_pdf, bbox_inches="tight")
    plt.close(fig)
    return p_png


# ====================================================================== #
# Per-class F1 bar chart
# ====================================================================== #
def plot_per_class_f1(
    per_class_f1: Dict[int, float], class_names: Dict[int, str],
    out_dir: Path, name: str = "per_class_f1",
) -> Path:
    classes = sorted(per_class_f1)
    vals = [per_class_f1[c] for c in classes]
    labels = [class_names.get(c, str(c)) for c in classes]
    fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(classes)), 4))
    bars = ax.bar(labels, vals, color="steelblue")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 score")
    ax.set_title("Per-class F1")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01,
                f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    plt.xticks(rotation=45, ha="right")
    return _savefig(fig, out_dir, name)


# ====================================================================== #
# Confusion matrix
# ====================================================================== #
def plot_confusion(
    cm: np.ndarray, class_names: List[str],
    out_dir: Path, name: str = "confusion",
) -> Path:
    fig, ax = plt.subplots(figsize=(max(5, 0.45 * len(class_names)),
                                    max(4, 0.4 * len(class_names))))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    return _savefig(fig, out_dir, name)


# ====================================================================== #
# SHAP summary / per-class signature
# ====================================================================== #
def plot_shap_summary_bar(sig: ShapSignatures, out_dir: Path,
                          name: str = "shap_summary_bar", top: int = 15) -> Path:
    # Global mean |SHAP| across classes.
    mat = np.stack([sig.per_class_importance[c] for c in sig.classes], axis=0)
    g = mat.mean(axis=0)
    order = np.argsort(g)[::-1][:top]
    feats = [sig.feature_names[i] for i in order]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.3 * top)))
    ax.barh(feats[::-1], g[order][::-1], color="darkorange")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Global SHAP feature importance")
    return _savefig(fig, out_dir, name)


def plot_per_class_signatures(
    sig: ShapSignatures, class_names: Dict[int, str],
    out_dir: Path, name: str = "shap_per_class_signature", top: int = 5,
) -> Path:
    n_cls = len(sig.classes)
    rows = int(np.ceil(n_cls / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(11, 2.8 * rows))
    axes = np.array(axes).reshape(-1)
    for ax, cls in zip(axes, sig.classes):
        topk = sig.top_k_for_class(cls, top)
        names = [t[0] for t in topk][::-1]
        vals  = [t[1] for t in topk][::-1]
        ax.barh(names, vals, color="teal")
        ax.set_title(f"{class_names.get(cls, cls)} — top-{top} SHAP")
        ax.tick_params(axis="y", labelsize=8)
    for ax in axes[len(sig.classes):]:
        ax.axis("off")
    fig.tight_layout()
    return _savefig(fig, out_dir, name)


# ====================================================================== #
# BGWO convergence + Pareto front
# ====================================================================== #
def plot_bgwo_convergence(history_fits: List[float], out_dir: Path,
                          name: str = "bgwo_convergence") -> Path:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(range(len(history_fits)), history_fits,
            marker="o", color="purple")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best fitness (lower=better)")
    ax.set_title("BGWO convergence")
    return _savefig(fig, out_dir, name)


def plot_pareto(pareto_points: List, out_dir: Path,
                name: str = "pareto_size_vs_f1") -> Path:
    if not pareto_points:
        return Path()
    arr = np.array(pareto_points, dtype=float)
    sizes = arr[:, 0]
    f1s   = arr[:, 1]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(sizes, f1s, alpha=0.45, color="navy")
    ax.set_xlabel("Selected features |S|")
    ax.set_ylabel("Macro F1")
    ax.set_title("BGWO search — feature count vs macro F1")
    return _savefig(fig, out_dir, name)


# ====================================================================== #
# Cross-dataset feature overlap heatmap
# ====================================================================== #
def plot_cross_dataset_overlap(
    runs_by_dataset: Dict[str, List[List[str]]], out_dir: Path,
    name: str = "cross_dataset_overlap",
) -> Path:
    """`runs_by_dataset` maps dataset -> list of feature-subsets (one per seed)."""
    from .explainability import jaccard

    datasets = list(runs_by_dataset)
    n = len(datasets)
    mat = np.eye(n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            scores = []
            for a in runs_by_dataset[datasets[i]]:
                for b in runs_by_dataset[datasets[j]]:
                    scores.append(jaccard(a, b))
            mat[i, j] = float(np.mean(scores)) if scores else 0.0
    fig, ax = plt.subplots(figsize=(4 + 0.4 * n, 4 + 0.4 * n))
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="viridis",
                xticklabels=datasets, yticklabels=datasets, ax=ax)
    ax.set_title("Cross-dataset feature overlap (Jaccard)")
    return _savefig(fig, out_dir, name)


# ====================================================================== #
# Latency vs feature count
# ====================================================================== #
def plot_latency_vs_features(df: pd.DataFrame, out_dir: Path,
                             name: str = "latency_vs_features") -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in df.method.unique():
        sub = df[df.method == m]
        ax.scatter(sub.n_features_selected, sub.latency_ms_per_flow,
                   label=m, alpha=0.7)
    ax.set_xlabel("Selected features |S|")
    ax.set_ylabel("Latency (ms / flow)")
    ax.set_title("Inference latency vs subset size")
    ax.legend()
    return _savefig(fig, out_dir, name)


# ====================================================================== #
# Comparison table (CSV + a rendered figure)
# ====================================================================== #
def render_comparison_table(agg: pd.DataFrame, out_dir: Path,
                            name: str = "comparison_table") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_dir / f"{name}.csv", index=False)

    fig, ax = plt.subplots(figsize=(min(14, 1.2 + 1.4 * len(agg.columns)),
                                    0.7 + 0.4 * len(agg)))
    ax.axis("off")
    tbl = ax.table(
        cellText=agg.round(4).astype(str).values,
        colLabels=agg.columns,
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.2)
    ax.set_title("Method × metric comparison (mean ± std across seeds)",
                 fontsize=10, pad=10)
    return _savefig(fig, out_dir, name)
