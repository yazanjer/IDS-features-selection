"""
Leader-Class Confidence Decision Ensemble (LCCDE).

Faithful reimplementation of the LCCDE algorithm from
    Yang, Shami et al., "LCCDE: A Decision-Based Ensemble Framework
    for Intrusion Detection in The Internet of Vehicles" (GLOBECOM '22).

The reference notebook in the baseline repo uses a per-sample Python
loop over `river.stream.iter_pandas` for prediction. We preserve the
exact decision logic but vectorise the predict path so it stays usable
on >100K-flow test sets.

When CatBoost is unavailable in the runtime (sandboxes, slim envs),
LCCDE falls back to a two-booster (LightGBM + XGBoost) mode and prints
a warning. Colab / full installs use the regular three-booster path.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import gc
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

# Silence sklearn's per-row UserWarning "X does not have valid feature names,
# but LGBMClassifier was fitted with feature names". LCCDE.evaluate() does
# per-row leader predictions on numpy arrays (fit was on a DataFrame), so the
# warning fires once per test row. At paper-grade budgets that's 100K+ warning
# lines per BGWO seed — enough to overflow Colab's cell output buffer and get
# the cell silently killed. Suppressed by exact message match so other sklearn
# UserWarnings still surface.
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)


# ---------------------------------------------------------------------- #
# Cheap CUDA probe used to route boosters to GPU when one is visible.
# Doesn't import torch / pycuda — just checks env + nvidia-smi.
# Set FORCE_CPU=1 to disable the GPU path (useful for benchmarking).
# ---------------------------------------------------------------------- #
def _use_catboost() -> bool:
    """Return True iff CatBoost should be the third LCCDE base learner.

    Default: False. CatBoost is opt-in via ENABLE_CATBOOST=1 because
    shap.TreeExplainer has a known heap-corruption bug on multi-class
    CatBoost trees (malloc unaligned tcache abort), which kills the
    Colab kernel mid-BGWO loop with no Python traceback. Disabled by
    default; LCCDE then runs as a 2-booster LightGBM+XGBoost
    leader-confidence ensemble.

    Set ENABLE_CATBOOST=1 to opt back in (e.g. for paper-reproduction
    runs against Yang et al.'s original 3-booster LCCDE).
    """
    if not os.environ.get("ENABLE_CATBOOST"):
        return False
    try:
        import catboost  # noqa: F401
        return True
    except ImportError:
        return False


def _cuda_available() -> bool:
    import os, shutil, subprocess
    if os.environ.get("FORCE_CPU"):
        return False
    if "CUDA_VISIBLE_DEVICES" in os.environ and os.environ["CUDA_VISIBLE_DEVICES"] == "":
        return False
    if shutil.which("nvidia-smi"):
        try:
            r = subprocess.run(
                ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=3,
            )
            return r.returncode == 0 and "GPU" in r.stdout
        except Exception:
            return False
    return False


@dataclass
class LCCDEResult:
    yt: np.ndarray
    yp: np.ndarray
    per_class_f1: Dict[int, float]
    leader_per_class: Dict[int, str]   # class_id -> "lgbm"|"xgb"|"cat"
    base_f1: Dict[str, np.ndarray]
    train_time_s: float
    infer_time_s: float
    n_test: int


class LCCDE:
    """
    Leader-Class Confidence Decision Ensemble (Yang et al., GLOBECOM '22).

    Reimplementation faithful to the reference notebook's `LCCDE(...)`
    function, but vectorised so it stays usable on >100K-row test sets.

    The decision rule is:

      1. If all three base learners agree → take their shared prediction.
      2. If exactly two agree → return the majority class, but predicted
         by *that majority class's leader model*.
      3. If all three disagree → for each base learner whose prediction
         coincides with its own class-leader, collect (class, conf).
         If exactly one such pair exists, return its class.
         Otherwise return the prediction with the highest confidence.
    """

    def __init__(self, seed: int = 0):
        self.seed = seed
        # Base learners — kept as attributes so SHAP can introspect them.
        self.lgbm = None
        self.xgb_ = None
        self.cat = None
        self.leader_per_class: Dict[int, str] = {}
        self.base_f1: Dict[str, np.ndarray] = {}
        self.classes_: np.ndarray = np.array([])
        self._train_time = 0.0

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "LCCDE":
        import lightgbm as lgb
        import xgboost as xgb
        _have_cat = _use_catboost()
        if _have_cat:
            import catboost as cbt
        else:
            cbt = None

        if X_val is None or y_val is None:
            X_val, y_val = X_train, y_train

        self.classes_ = np.array(sorted(pd.Series(y_train).unique()))
        t0 = time.time()

        # ---- GPU detection: prefer CUDA if a device is visible. ----
        use_gpu = _cuda_available()
        device_tag = "GPU" if use_gpu else "CPU"
        print(f"[lccde] training on {device_tag} | {len(X_train):,} rows × "
              f"{X_train.shape[1]} cols × {len(self.classes_)} classes",
              flush=True)

        # LightGBM stays on CPU regardless of GPU availability — its GPU
        # path has a known abort() bug on multi-class with imbalanced rare
        # classes ("Check failed: best_split_info.left_count > 0" from
        # serial_tree_learner.cpp). The fatal kills the whole kernel and
        # is not catchable from Python. CPU LightGBM is fast enough at
        # this scale (~30 s on 400K × 78 × 15c) that losing the GPU path
        # costs little; XGBoost + CatBoost on GPU still drive the overall
        # 5–8× speedup.
        # Set ENABLE_LGBM_GPU=1 to override (e.g. for benchmarking on a
        # less-imbalanced dataset).
        lgbm_kwargs = dict(
            random_state=self.seed,
            verbosity=-1,
            # Robustness against tiny rare-class buckets after SMOTE.
            min_data_in_leaf=20,
            min_child_samples=20,
        )
        if use_gpu and os.environ.get("ENABLE_LGBM_GPU"):
            lgbm_kwargs["device"] = "gpu"
        self.lgbm = lgb.LGBMClassifier(**lgbm_kwargs)
        self.lgbm.fit(X_train, y_train)
        print(f"[lccde]   LightGBM done  ({time.time() - t0:.1f}s elapsed)", flush=True)
        t_lgb = time.time()

        xgb_kwargs = dict(
            random_state=self.seed, verbosity=0,
            tree_method="hist", eval_metric="mlogloss",
        )
        if use_gpu:
            xgb_kwargs["device"] = "cuda"
        self.xgb_ = xgb.XGBClassifier(**xgb_kwargs)
        # XGB needs np arrays to avoid feature-name warnings on the predict path.
        self.xgb_.fit(np.asarray(X_train), np.asarray(y_train))
        print(f"[lccde]   XGBoost  done  ({time.time() - t_lgb:.1f}s, "
              f"{time.time() - t0:.1f}s total)", flush=True)
        t_xgb = time.time()

        if _have_cat:
            cat_kwargs = dict(
                verbose=0,
                boosting_type="Plain",
                random_seed=self.seed,
                # Cap CPU memory so CatBoost can't grab the whole 80GB
                # and trigger a Colab OOM. 20 GB is plenty for our scale.
                used_ram_limit="20gb",
            )
            if use_gpu:
                cat_kwargs["task_type"] = "GPU"
                cat_kwargs["devices"] = "0"
                # Leave half the A100's 40 GB VRAM free for XGBoost and
                # whatever else is running in the kernel.
                cat_kwargs["gpu_ram_part"] = 0.5
            self.cat = cbt.CatBoostClassifier(**cat_kwargs)
            self.cat.fit(X_train, y_train)
            print(f"[lccde]   CatBoost done  ({time.time() - t_xgb:.1f}s, "
                  f"{time.time() - t0:.1f}s total)", flush=True)
        else:
            # First-class 2-booster mode (CatBoost disabled by default).
            # LCCDE decision rule in predict() handles self.cat is None as
            # a clean LightGBM+XGBoost leader-confidence ensemble — no
            # mirroring, no biased voting.
            self.cat = None
            print("[lccde] 2-booster mode (LightGBM + XGBoost). "
                  "Set ENABLE_CATBOOST=1 to enable the original 3-booster LCCDE.")

        # Release intermediate fit-time state and trigger collection.
        # Big multi-class boosters keep substantial scratch around until
        # GC runs; in a 450-iteration BGWO loop that adds up fast.
        gc.collect()

        self._train_time = time.time() - t0

        # Per-class F1 of each base learner — defines the leader table.
        lg_p = self.lgbm.predict(X_val)
        xg_p = self.xgb_.predict(np.asarray(X_val))
        labels = list(self.classes_)
        self.base_f1 = {
            "lgbm": f1_score(y_val, lg_p, labels=labels, average=None, zero_division=0),
            "xgb":  f1_score(y_val, xg_p, labels=labels, average=None, zero_division=0),
        }
        if _have_cat:
            cb_p = self.cat.predict(X_val).ravel().astype(int)
            self.base_f1["cat"] = f1_score(y_val, cb_p, labels=labels,
                                           average=None, zero_division=0)
        for i, cls in enumerate(labels):
            scores = {tag: self.base_f1[tag][i] for tag in self.base_f1}
            self.leader_per_class[int(cls)] = max(scores, key=scores.get)
        return self

    # ------------------------------------------------------------------ #
    # Inference — vectorised LCCDE decision rule
    # ------------------------------------------------------------------ #
    def predict(self, X_test) -> Tuple[np.ndarray, float]:
        t0 = time.time()
        Xnp = np.asarray(X_test)

        p1 = self.lgbm.predict_proba(X_test)         # (n, C)
        p2 = self.xgb_.predict_proba(Xnp)

        y1 = self.classes_[np.argmax(p1, axis=1)]
        y2 = self.classes_[np.argmax(p2, axis=1)]
        c1 = p1.max(axis=1)
        c2 = p2.max(axis=1)
        n = len(Xnp)
        y_pred = np.empty(n, dtype=int)

        # ----- 2-booster mode: LightGBM + XGBoost only ----- #
        if self.cat is None:
            agree = (y1 == y2)
            y_pred[agree] = y1[agree]
            if (~agree).any():
                for k in np.where(~agree)[0]:
                    # For each disagreed sample, defer to the per-class leader
                    # of whichever class has the higher leader score; fall back
                    # to higher-confidence model if neither class's leader is
                    # one of the two predicting models.
                    leader1 = self.leader_per_class.get(int(y1[k]))
                    leader2 = self.leader_per_class.get(int(y2[k]))
                    if leader1 == "lgbm" and leader2 != "xgb":
                        y_pred[k] = y1[k]
                    elif leader2 == "xgb" and leader1 != "lgbm":
                        y_pred[k] = y2[k]
                    else:
                        # Either both class leaders are the same model, or
                        # neither — break tie on raw confidence.
                        y_pred[k] = y1[k] if c1[k] >= c2[k] else y2[k]
            return y_pred, time.time() - t0

        # ----- 3-booster mode: LightGBM + XGBoost + CatBoost ----- #
        p3 = self.cat.predict_proba(X_test)
        y3 = self.classes_[np.argmax(p3, axis=1)]
        c3 = p3.max(axis=1)

        all_agree = (y1 == y2) & (y2 == y3)
        all_diff  = (y1 != y2) & (y2 != y3) & (y1 != y3)
        two_agree = ~all_agree & ~all_diff

        # Case 1: unanimous
        y_pred[all_agree] = y1[all_agree]

        # Case 2: majority of two
        if two_agree.any():
            idx = np.where(two_agree)[0]
            for k in idx:
                if y1[k] == y2[k]:
                    maj = y1[k]
                elif y1[k] == y3[k]:
                    maj = y1[k]
                else:
                    maj = y2[k]
                leader = self.leader_per_class.get(int(maj), "lgbm")
                y_pred[k] = self._leader_predict_one(Xnp[k], leader)

        # Case 3: all three disagree — leader+confidence arbitration
        if all_diff.any():
            for k in np.where(all_diff)[0]:
                pairs = []   # (cls, conf) where pred came from its class leader
                if self.leader_per_class.get(int(y1[k])) == "lgbm":
                    pairs.append((y1[k], c1[k]))
                if self.leader_per_class.get(int(y2[k])) == "xgb":
                    pairs.append((y2[k], c2[k]))
                if self.leader_per_class.get(int(y3[k])) == "cat":
                    pairs.append((y3[k], c3[k]))
                if len(pairs) == 1:
                    y_pred[k] = pairs[0][0]
                elif len(pairs) > 1:
                    y_pred[k] = max(pairs, key=lambda t: t[1])[0]
                else:
                    confs = [(y1[k], c1[k]), (y2[k], c2[k]), (y3[k], c3[k])]
                    y_pred[k] = max(confs, key=lambda t: t[1])[0]

        return y_pred, time.time() - t0

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _leader_predict_one(self, x_row: np.ndarray, leader: str) -> int:
        x = x_row.reshape(1, -1)
        if leader == "lgbm":
            return int(self.lgbm.predict(x)[0])
        if leader == "xgb":
            return int(self.xgb_.predict(x)[0])
        # leader == "cat" — fall back to lgbm if CatBoost is disabled
        if self.cat is None:
            return int(self.lgbm.predict(x)[0])
        out = np.asarray(self.cat.predict(x)).ravel()
        return int(out[0])

    @property
    def train_time(self) -> float:
        return self._train_time

    def evaluate(self, X_test, y_test) -> LCCDEResult:
        y_test_arr = np.asarray(y_test)
        y_pred, infer_s = self.predict(X_test)
        per_class_f1 = dict(zip(
            self.classes_,
            f1_score(y_test_arr, y_pred, labels=list(self.classes_),
                     average=None, zero_division=0),
        ))
        return LCCDEResult(
            yt=y_test_arr,
            yp=y_pred,
            per_class_f1={int(k): float(v) for k, v in per_class_f1.items()},
            leader_per_class=dict(self.leader_per_class),
            base_f1=self.base_f1,
            train_time_s=self._train_time,
            infer_time_s=infer_s,
            n_test=len(y_pred),
        )
