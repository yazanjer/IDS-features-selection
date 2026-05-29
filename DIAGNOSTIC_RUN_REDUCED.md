# Diagnostic — `run_reduced.ipynb` execution (2026-05-29)

Comprehensive analysis of what went wrong, what worked, and what to do next. Based on the uploaded results zip + the uploaded `run_reduced-2.ipynb` + the live source tree at `ids-bgwo-shap/`.

## Verdict

The notebook ran to completion, but the two contributions (`bgwo_bi`, `bgwo_shap`) produced **zero data**. A single root cause in `src/evaluation.py` explains both that crash and the high seed-variance you see in RFE/Boruta. Two further issues are real but smaller. One is a critical security incident.

## Findings ranked by severity

### CRITICAL-1 · GitHub PAT exposed in uploaded notebook

Cell 17 of `run_reduced-2.ipynb` contains a hardcoded fine-grained PAT (`github_pat_11AA3ZFZY0kVt9...`). The notebook was uploaded to this conversation, which means the token is now in the chat transcript and in the upload directory.

**Action:** revoke at https://github.com/settings/tokens immediately, then generate a fresh one. Do not paste PATs into notebooks again — use `getpass.getpass(...)` at the very least, or push from your Mac terminal after pulling Colab results.

### CRITICAL-2 · BGWO methods crashed on every seed — XGBoost rejected non-contiguous class labels

**Symptom (from your Colab log):**

```
[lccde] training on GPU | 30,000 rows × 42 cols × 13 classes
[lccde]   LightGBM done  (1.4s elapsed)
[matrix] bgwo_bi seed=0 FAILED with ValueError:
  Invalid classes inferred from unique values of `y`.
  Expected: [ 0  1  2  3  4  5  6  7  8  9 10 11 12]
  got      : [ 0  1  2  3  4  5  6  7  8       10 11 12 14]
```

XGBoost 3.x requires class labels to be contiguous `0..k-1`. The BGWO inner training subset has 13 unique classes but they're labelled `[0..8, 10, 11, 12, 14]` — classes 9 and 13 are missing entirely, and 14 is present. XGBoost rejects this before training even starts. LightGBM accepts it (you can see it finished in 1.4s).

**Why classes drop:** the BGWO inner training data comes from `src/evaluation.py` lines 113–116:

```python
X_fs_tr = X_tr.head(cfg.fs_train_rows)   # 30,000 rows
y_fs_tr = y_tr.head(cfg.fs_train_rows)
X_fs_va = X_te.head(cfg.fs_test_rows)
y_fs_va = y_te.head(cfg.fs_test_rows)
```

`pd.DataFrame.head` is **not stratified**. After `train_test_split`, X_tr has ~400K rows shuffled. Class 13 only has ~6 of those 400K rows. Probability that the first 30K rows include even one class-13 sample is roughly `1 - (1 - 30/400)^6 ≈ 38%` per seed, and the same applies to class 9 (~9 samples). On average **every other seed** loses a rare class from the subsample.

**Blast radius:** every BGWO trial. 5 seeds × 2 methods = 10 silent failures. The matrix kept going per v3 constraint #8, the baselines (which train on the FULL 500K sample) succeeded, and the per-trial tracebacks went into the Colab cell output — not into the zip.

### CRITICAL-3 · The same `.head()` bug causes RFE / Boruta instability

Lines 106–107 use the identical `.head(fs_train_rows)` slice for the wrapper baselines (`rfe`, `lasso`, `rf_imp`, `boruta`). They didn't crash because LightGBM tolerates non-contiguous classes, but their feature rankings are computed on whichever ~13 classes happened to land in the first 30K rows. That seed-dependent class composition is exactly what produces the variance you see in the aggregate table:

| Method | macro_F1 std across seeds | Why |
|---|---:|---|
| `rfe` | **0.306** | Worst — class composition flips per seed |
| `boruta` | **0.239** | Same root cause |
| `rf_imp` | 0.122 | Less sensitive, still elevated |
| `lasso` | 0.071 | Stable; L1 regularization dominates |
| `none` | 0.080 | Train uses full 500K, so seed only changes split |
| `filter` | 0.079 | Information-gain top-K is largely deterministic |

Look at the per-trial zero-F1 classes I extracted from the JSONs:

```
rfe seed=1: 9 zero-F1 classes
rfe seed=3: 7 zero-F1 classes
rfe seed=2: 1 zero-F1 class
rfe seed=4: 0 zero-F1 classes  ← only seed where every class is detected
```

This is not noise — it's a deterministic consequence of the unstratified inner sample.

### HIGH-1 · Rare classes 9 and 13 are F1=0 in nearly every trial (separate issue from above)

Even the `none` method (no FS, full 500K, full 78 features) gets F1=0.0 for class 13 in **all 5 seeds** and F1=0 for class 9 in 2 of 5 seeds. This is a downstream-model problem, not an FS problem:

- Class 13 (Heartbleed) has 21 raw flows in CIC-IDS2017. After stratified sample to 500K with floor=5 and test_size=0.2, you end up with ~6 train / 2 test samples.
- SMOTE synthesises class 13 up to `smote_min_count=500` from those 6 samples — pure k-nearest-neighbour interpolation in feature space. The 500 synthetic samples occupy a near-degenerate manifold that doesn't generalise to the 2 real test samples.
- F1=0 on a 2-sample test set means the model never predicts class 13 (or always wrong) for those two flows.

Class 9 (Infiltration, 36 raw flows) has similar dynamics but slightly more room. Class 14 (Bot, 652 raw flows) is borderline — usually fine, sometimes F1=0.36.

**This is a paper problem, not a code bug.** Reviewers at any tier-1 venue will ask: "how can you claim contribution X when your downstream classifier doesn't detect Heartbleed at all, with or without your FS?" Three reasonable responses:

1. Reframe as **macro-F1 over the detectable classes** and exclude class 13 (and possibly class 9) from macro, with a clear footnote citing sample-size insufficiency. Weighted-F1 stays the same (it's 0.998).
2. Switch to **binary attack/benign** classification — eliminates the rare-class issue but loses multi-class granularity (and most of the v3 paper's framing).
3. Bump `smote_min_count` to 1000+ and use **borderline-SMOTE or ADASYN** instead of vanilla SMOTE. Likely makes class 13 detectable but doesn't change the fundamental "6-train, 2-test" sample-size problem.

### MEDIUM-1 · Wilcoxon CSV is all `n=0` — expected given findings above

```
bgwo_shap,none,macro_f1,0,,
bgwo_shap,filter,macro_f1,0,,
...
```

`wilcoxon_vs_reference(df, 'bgwo_shap', ...)` looks for `bgwo_shap` rows in `df`; there aren't any (Critical-2), so n=0 everywhere. Not a bug — a downstream consequence.

### MEDIUM-2 · `fidelity` column empty in every row — expected

`cfg.compute_shap_in_matrix=False` (paper-grade default, per v3 constraint #3 to avoid the CatBoost TreeExplainer heap bug). With CatBoost now gone the gate could be re-opened, but it stays off until we explicitly turn it on. Not a bug.

### LOW-1 · `raw_results_cicids2017.csv` is a duplicate of `reduced_runs.csv`

`diff` on the two CSVs returns identical content. Two copies of the same data — minor duplication in `evaluation.py` and/or the notebook, no functional issue.

## What actually worked

- **The smoke pipeline.** All 8 methods completed at smoke budget — the silent failure only appears at 500K-sample × 30K-inner.
- **Install + Colab numpy 2.x fix.** No ABI errors, no `_blas_supports_fpe`. Prep cell did its job.
- **2-booster LCCDE (LightGBM + XGBoost).** Trained cleanly on baselines; the CatBoost SHAP heap bug never fired. The pivot worked.
- **Output-spam fix.** No sklearn warning flood — the targeted `filterwarnings` in `lccde_model.py` is doing its job.
- **In-process dataset cache.** Logs show `[data] cache hit for cicids2017` between trials. v3 constraint #4 OK.
- **Per-trial JSON streaming.** Every completed baseline trial has its own JSON under `results/`. If a Colab disconnect had cut the run mid-baseline, those would still be there.

## What to fix and in what order

### Fix #1 (root cause) — stratify the BGWO + wrapper inner subsample

Replace lines 106–116 in `src/evaluation.py`:

```python
# OLD
X_fs_tr = X_tr.head(cfg.fs_train_rows)
y_fs_tr = y_tr.head(cfg.fs_train_rows)
X_fs_va = X_te.head(cfg.fs_test_rows)
y_fs_va = y_te.head(cfg.fs_test_rows)
```

with a stratified sample reusing the existing helper:

```python
from .sampling import stratified_sample
X_fs_tr, y_fs_tr = stratified_sample(
    X_tr, y_tr, n=cfg.fs_train_rows, seed=seed,
    min_per_class=1, verbose=False,
)
X_fs_va, y_fs_va = stratified_sample(
    X_te, y_te, n=cfg.fs_test_rows, seed=seed,
    min_per_class=1, verbose=False,
)
```

This single change should resolve Critical-2, Critical-3, and the high RFE/Boruta variance.

### Fix #2 (defence in depth) — wrap XGBoost in a LabelEncoder inside `LCCDE.fit`

Even after Fix #1, if a future user sets `fs_train_rows` below the number of classes, XGBoost will still reject non-contiguous labels. Encode/decode at the XGBoost boundary so LCCDE doesn't care:

```python
from sklearn.preprocessing import LabelEncoder

# in __init__:
self._xgb_le: Optional[LabelEncoder] = None

# in fit(), before xgb.fit:
self._xgb_le = LabelEncoder().fit(np.asarray(y_train))
y_train_xgb = self._xgb_le.transform(np.asarray(y_train))
self.xgb_.fit(np.asarray(X_train), y_train_xgb)

# replace `xg_p = self.xgb_.predict(np.asarray(X_val))` with:
xg_p_raw = self.xgb_.predict(np.asarray(X_val))
xg_p     = self._xgb_le.inverse_transform(xg_p_raw)

# in _leader_predict_one for "xgb":
raw = int(self.xgb_.predict(x)[0])
return int(self._xgb_le.inverse_transform([raw])[0])
```

`predict_proba` column order is unchanged: LabelEncoder sorts unique classes, which is also `self.classes_`'s order, so the existing `self.classes_[np.argmax(p2, axis=1)]` line keeps working.

### Fix #3 (paper-grade re-run) — bump seeds + run sensitivity sweep

After Fix #1+#2 land:

1. Re-run `run_reduced.ipynb` once to confirm BGWO produces rows.
2. Then run `run_colab.ipynb` (paper-grade: `n_seeds=10`, `bgwo_population=15`, `bgwo_iterations=30`, plus α/β/γ sweep, pop/iter sweep, UNSW-NB15). ETA on A100: ~11 hours per my earlier estimate.

### Fix #4 (paper writing) — address the rare-class story

Decide between excluding class 13 from macro, switching to binary, or bumping SMOTE — and document it in §3 of your paper before submission. Without this, a reviewer will catch the F1=0 in your tables.

## Re-run checklist (after Fix #1+#2 land)

- [ ] Stratified inner sampler in `evaluation.py`
- [ ] `LabelEncoder` for XGBoost in `lccde_model.py`
- [ ] Re-push to GitHub (do NOT paste PAT into a notebook — use `getpass`)
- [ ] Revoke the leaked PAT first
- [ ] Smoke notebook → confirm `Pipeline smoke PASS` and bgwo_shap row exists
- [ ] Reduced notebook → confirm 8 method rows in `reduced_runs.csv`, non-empty `reduced_wilcoxon.csv`
- [ ] Paper-grade notebook (`run_colab.ipynb`) → overnight A100 run

I'll implement Fix #1 and Fix #2 in code as the next step. Confirm when you've revoked the PAT and I'll push everything.
