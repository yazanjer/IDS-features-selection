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
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


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
        try:
            import catboost as cbt
            _have_cat = True
        except ImportError:
            cbt = None
            _have_cat = False

        if X_val is None or y_val is None:
            X_val, y_val = X_train, y_train

        self.classes_ = np.array(sorted(pd.Series(y_train).unique()))
        t0 = time.time()

        self.lgbm = lgb.LGBMClassifier(random_state=self.seed, verbosity=-1)
        self.lgbm.fit(X_train, y_train)

        self.xgb_ = xgb.XGBClassifier(
            random_state=self.seed,
            verbosity=0,
            tree_method="hist",
            eval_metric="mlogloss",
        )
        # XGB needs np arrays to avoid feature-name warnings on the predict path.
        self.xgb_.fit(np.asarray(X_train), np.asarray(y_train))

        if _have_cat:
            self.cat = cbt.CatBoostClassifier(
                verbose=0, boosting_type="Plain", random_seed=self.seed
            )
            self.cat.fit(X_train, y_train)
        else:
            # Graceful 2-booster degenerate mode (sandbox / lightweight envs).
            # In production (Colab + full deps) this branch is never taken.
            self.cat = self.lgbm
            print("[lccde] catboost not installed — running 2-booster mode "
                  "(cat slot mirrors lgbm). Numbers will not match the paper.")

        self._train_time = time.time() - t0

        # Per-class F1 of each base learner — defines the leader table.
        lg_p = self.lgbm.predict(X_val)
        xg_p = self.xgb_.predict(np.asarray(X_val))
        if _have_cat:
            cb_p = self.cat.predict(X_val).ravel().astype(int)
        else:
            cb_p = lg_p   # mirrored

        labels = list(self.classes_)
        self.base_f1 = {
            "lgbm": f1_score(y_val, lg_p, labels=labels, average=None, zero_division=0),
            "xgb":  f1_score(y_val, xg_p, labels=labels, average=None, zero_division=0),
            "cat":  f1_score(y_val, cb_p, labels=labels, average=None, zero_division=0),
        }
        for i, cls in enumerate(labels):
            scores = {"lgbm": self.base_f1["lgbm"][i],
                      "xgb":  self.base_f1["xgb"][i],
                      "cat":  self.base_f1["cat"][i]}
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
        p3 = self.cat.predict_proba(X_test)

        # Align class index order — assume identical, as we trained on the
        # same y; if not, project onto self.classes_.
        y1 = self.classes_[np.argmax(p1, axis=1)]
        y2 = self.classes_[np.argmax(p2, axis=1)]
        y3 = self.classes_[np.argmax(p3, axis=1)]

        c1 = p1.max(axis=1)
        c2 = p2.max(axis=1)
        c3 = p3.max(axis=1)

        n = len(Xnp)
        y_pred = np.empty(n, dtype=int)

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
