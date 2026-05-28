"""
End-to-end experiment runner.

`run_one(cfg, method, seed)` executes one trial of the pipeline:

    load -> stratified sample -> stratified split
         -> feature selection (per `method`)
         -> SMOTE rebalance -> fit LCCDE -> evaluate
         -> SHAP signatures + fidelity

`run_experiment_matrix(...)` is the multi-seed × multi-method driver
used by the Colab notebook; `aggregate(...)` reports mean ± std per
method and `wilcoxon_vs_reference(...)` computes the Wilcoxon
signed-rank p-values against the `bgwo_shap` reference.

Per-run JSONs land in `cfg.results_dir/run_<dataset>_<method>_seed<N>.json`
and the flat CSV summary in `raw_results_<dataset>.csv`.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import gc
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score,
)
from scipy.stats import wilcoxon

from .config import Config, seed_everything
from .data_loader import load_dataset
from .sampling import stratified_sample, train_test_split_stratified, apply_smote
from .lccde_model import LCCDE
from .bgwo_fs import BinaryGreyWolfOptimizer, filter_fs, all_features, BGWOResult
from .fitness import TriObjectiveFitness
from .fs_baselines import WRAPPER_METHODS, run_wrapper_method
from .explainability import (
    compute_shap_signatures, ShapSignatures, explanation_fidelity,
    variable_stability, kuncheva_index, jaccard, cross_dataset_overlap,
)


# ====================================================================== #
# Single-run record
# ====================================================================== #
@dataclass
class RunResult:
    method: str
    dataset: str
    seed: int
    selected_features: List[str]
    n_features_selected: int
    n_features_total: int

    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    per_class_f1: Dict[int, float]
    roc_auc_ovr: Optional[float]
    pr_auc_macro: Optional[float]

    train_time_s: float
    infer_total_s: float
    latency_ms_per_flow: float

    confusion: List[List[int]]
    fitness_history: Optional[List[float]] = None
    pareto_points: Optional[List[Tuple[int, float]]] = None
    shap_signatures: Optional[Dict[int, List[Tuple[str, float]]]] = None
    fidelity: Optional[float] = None


# ====================================================================== #
# Pipeline: one (method, seed) trial
# ====================================================================== #
def run_one(cfg: Config, method: str, seed: int) -> RunResult:
    cfg = _override_cfg(cfg, seed=seed, fs_method=method)
    seed_everything(seed)

    X_full, y_full, feat_names, label_map = load_dataset(cfg)
    X_s, y_s = stratified_sample(X_full, y_full,
                                 n=cfg.sample_size, seed=seed,
                                 min_per_class=5, verbose=cfg.verbose)
    X_tr, X_te, y_tr, y_te = train_test_split_stratified(
        X_s, y_s, test_size=cfg.test_size, seed=seed
    )

    # ------------------------------------------------------------------ #
    # FEATURE SELECTION
    # ------------------------------------------------------------------ #
    bgwo_result: Optional[BGWOResult] = None
    if method == "none":
        selected = all_features(X_tr, y_tr)
    elif method == "filter":
        selected = filter_fs(X_tr, y_tr)
    elif method in WRAPPER_METHODS:
        # Modern wrapper baselines (RFE, LASSO, RF-importance, Boruta)
        # use the inner-loop subset for speed; matches what BGWO sees.
        X_fs_tr = X_tr.head(cfg.fs_train_rows)
        y_fs_tr = y_tr.head(cfg.fs_train_rows)
        selected = run_wrapper_method(method, X_fs_tr, y_fs_tr, seed)
        if not selected:
            selected = all_features(X_tr, y_tr)
    elif method in ("bgwo_bi", "bgwo_shap"):
        # Cheap inner-loop training subset for the FS search.
        X_fs_tr = X_tr.head(cfg.fs_train_rows)
        y_fs_tr = y_tr.head(cfg.fs_train_rows)
        X_fs_va = X_te.head(cfg.fs_test_rows)
        y_fs_va = y_te.head(cfg.fs_test_rows)

        fit_cfg = cfg
        if method == "bgwo_bi":
            fit_cfg = _override_cfg(cfg, gamma=0.0)

        fitness = TriObjectiveFitness(
            fit_cfg, X_fs_tr, y_fs_tr, X_fs_va, y_fs_va, list(X_fs_tr.columns)
        )
        bgwo = BinaryGreyWolfOptimizer(fit_cfg, fitness)
        bgwo_result = bgwo.run(verbose=cfg.verbose)
        selected = bgwo_result.selected_features
        if not selected:
            selected = all_features(X_tr, y_tr)
    else:
        raise ValueError(f"Unknown FS method: {method}")

    if cfg.verbose:
        print(f"[run] method={method} | selected {len(selected)}/{len(feat_names)} features")

    # ------------------------------------------------------------------ #
    # FINAL FIT on the selected subset, with SMOTE rebalancing
    # ------------------------------------------------------------------ #
    Xtr_sel = X_tr[selected]
    Xte_sel = X_te[selected]
    Xtr_sm, ytr_sm = apply_smote(
        Xtr_sel, y_tr, cfg.smote_min_count, seed,
        max_per_class=getattr(cfg, "smote_max_per_class", None),
    )

    model = LCCDE(seed=seed).fit(Xtr_sm, ytr_sm, Xte_sel, y_te)
    eval_t0 = time.time()
    y_pred, infer_total = model.predict(Xte_sel)
    eval_t = time.time() - eval_t0
    train_time = model.train_time

    metrics = _compute_metrics(y_te, y_pred, model, Xte_sel)

    # SHAP signatures + fidelity for explainability reporting.
    # Skipped by default — see Config.compute_shap_in_matrix.
    shap_sig = None
    fidelity = None
    if getattr(cfg, "compute_shap_in_matrix", False):
        try:
            sig = compute_shap_signatures(model, Xte_sel.sample(
                min(500, len(Xte_sel)), random_state=seed
            ), background_n=cfg.shap_background_samples)
            shap_sig = {int(c): sig.top_k_for_class(c, k=5) for c in sig.classes}
            fidelity = explanation_fidelity(model, Xte_sel, sig, top_k=10)
        except Exception as e:
            print(f"[run] SHAP signatures skipped: {e}")

    return RunResult(
        method=method, dataset=cfg.dataset, seed=seed,
        selected_features=selected,
        n_features_selected=len(selected),
        n_features_total=len(feat_names),
        train_time_s=train_time,
        infer_total_s=infer_total,
        latency_ms_per_flow=1000.0 * infer_total / max(1, len(y_pred)),
        confusion=confusion_matrix(y_te, y_pred).tolist(),
        fitness_history=(bgwo_result.history.best_fitness if bgwo_result else None),
        pareto_points=(bgwo_result.history.pareto_points if bgwo_result else None),
        shap_signatures=shap_sig,
        fidelity=fidelity,
        **metrics,
    )


# ====================================================================== #
# Metric assembly
# ====================================================================== #
def _compute_metrics(y_te, y_pred, model: LCCDE, Xte_sel) -> dict:
    labels = list(model.classes_)
    accuracy = float(accuracy_score(y_te, y_pred))
    macro_p = float(precision_score(y_te, y_pred, average="macro", zero_division=0))
    macro_r = float(recall_score(y_te, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_te, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_te, y_pred, average="weighted", zero_division=0))
    per_class = f1_score(y_te, y_pred, labels=labels, average=None, zero_division=0)
    per_class_f1 = {int(c): float(v) for c, v in zip(labels, per_class)}

    # ROC-AUC + PR-AUC (use LCCDE's leader model probabilities — approx).
    roc_auc, pr_auc = None, None
    try:
        proba = model.lgbm.predict_proba(Xte_sel)
        # Align proba columns to label order.
        roc_auc = float(roc_auc_score(y_te, proba, multi_class="ovr",
                                      average="macro", labels=labels))
        # average_precision_score needs one-vs-rest indicators.
        y_oh = np.zeros((len(y_te), len(labels)))
        for i, lbl in enumerate(labels):
            y_oh[:, i] = (np.asarray(y_te) == lbl).astype(int)
        pr_auc = float(average_precision_score(y_oh, proba, average="macro"))
    except Exception as e:
        print(f"[run] ROC/PR-AUC skipped: {e}")

    return dict(
        accuracy=accuracy,
        macro_precision=macro_p,
        macro_recall=macro_r,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        per_class_f1=per_class_f1,
        roc_auc_ovr=roc_auc,
        pr_auc_macro=pr_auc,
    )


# ====================================================================== #
# Multi-seed driver + Wilcoxon
# ====================================================================== #
DEFAULT_METHODS = (
    "none", "filter",
    "rfe", "lasso", "rf_imp", "boruta",   # modern wrapper baselines
    "bgwo_bi", "bgwo_shap",
)


def run_experiment_matrix(
    cfg: Config,
    methods: List[str] = DEFAULT_METHODS,
    seeds: Optional[List[int]] = None,
    fail_fast: bool = False,
) -> pd.DataFrame:
    """Run the (method × seed) experiment grid.

    fail_fast=True re-raises the first exception (useful for debugging in
    Colab); default behaviour keeps going and prints full tracebacks so
    silent multi-failure can't hide downstream errors like the empty-df
    KeyError that aggregate() throws."""
    import traceback as _tb

    seeds = seeds if seeds is not None else list(range(cfg.n_seeds))
    rows = []
    failures = []
    for m in methods:
        for s in seeds:
            print(f"\n========== {m} | seed={s} ==========", flush=True)
            try:
                r = run_one(cfg, method=m, seed=s)
                rows.append(_flatten(r))
                _save_run_json(cfg, r)
                # Forget the heavy RunResult once it's serialised — every
                # trial keeps ~200 MB of confusion / signature / mask
                # arrays that we don't need across trials.
                del r
            except Exception as e:
                failures.append((m, s, repr(e)))
                print(f"[matrix] {m} seed={s} FAILED with {type(e).__name__}: {e}",
                      flush=True)
                _tb.print_exc()
                if fail_fast:
                    raise
            finally:
                # Force a GC pass between trials so transient peaks don't
                # stack across the matrix. Cheap (~50 ms), keeps RAM flat.
                gc.collect()
    df = pd.DataFrame(rows)
    out = cfg.results_dir / f"raw_results_{cfg.dataset}.csv"
    df.to_csv(out, index=False)
    print(f"[matrix] raw results saved to {out}")
    if failures:
        print(f"\n[matrix] {len(failures)}/{len(methods)*len(seeds)} trials failed:")
        for m, s, err in failures:
            print(f"    - {m} seed={s}: {err}")
    if df.empty:
        raise RuntimeError(
            "run_experiment_matrix produced an empty DataFrame — every trial "
            "failed. See the [matrix] FAILED lines above for the root cause; "
            "re-run with fail_fast=True to surface the first traceback directly."
        )
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "method" not in df.columns:
        raise ValueError(
            "aggregate(df) was called on an empty / methodless DataFrame. "
            "This usually means every trial in run_experiment_matrix failed — "
            "scroll up for the [matrix] FAILED tracebacks, or re-run the "
            "matrix with fail_fast=True."
        )
    metric_cols = [
        "accuracy", "macro_f1", "weighted_f1",
        "macro_precision", "macro_recall",
        "roc_auc_ovr", "pr_auc_macro",
        "n_features_selected", "latency_ms_per_flow",
        "train_time_s", "fidelity",
    ]
    cols = [c for c in metric_cols if c in df.columns]
    grouped = df.groupby("method")[cols].agg(["mean", "std"])
    grouped.columns = [f"{a}_{b}" for a, b in grouped.columns]
    return grouped.reset_index()


def wilcoxon_vs_reference(
    df: pd.DataFrame, reference: str = "bgwo_shap",
    metric: str = "macro_f1"
) -> pd.DataFrame:
    rows = []
    ref_vals = df[df.method == reference][metric].to_numpy()
    for m in df.method.unique():
        if m == reference:
            continue
        other_vals = df[df.method == m][metric].to_numpy()
        n = min(len(ref_vals), len(other_vals))
        if n < 2 or np.allclose(ref_vals[:n], other_vals[:n]):
            stat, p = (np.nan, np.nan)
        else:
            try:
                stat, p = wilcoxon(ref_vals[:n], other_vals[:n])
            except ValueError:
                stat, p = (np.nan, np.nan)
        rows.append({"reference": reference, "vs_method": m, "metric": metric,
                     "n": n, "statistic": stat, "p_value": p})
    return pd.DataFrame(rows)


# ====================================================================== #
# Helpers
# ====================================================================== #
def _override_cfg(cfg: Config, **overrides) -> Config:
    """Return a shallow copy of cfg with selected fields replaced."""
    from dataclasses import replace
    return replace(cfg, **overrides)


def _flatten(r: RunResult) -> dict:
    d = asdict(r)
    # Drop heavy / nested fields from the CSV.
    d.pop("selected_features", None)
    d.pop("per_class_f1", None)
    d.pop("confusion", None)
    d.pop("fitness_history", None)
    d.pop("pareto_points", None)
    d.pop("shap_signatures", None)
    return d


def _save_run_json(cfg: Config, r: RunResult) -> None:
    cfg.ensure_dirs()
    fname = f"run_{r.dataset}_{r.method}_seed{r.seed}.json"
    out = cfg.results_dir / fname
    out.write_text(json.dumps(asdict(r), default=str, indent=2))
