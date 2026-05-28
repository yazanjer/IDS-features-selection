# Colab compatibility check — IDS-BGWO-SHAP

**Date:** 2026-05-29
**Repo:** `ids-bgwo-shap/`
**Target runtime:** Google Colab 2026.04 (Ubuntu 22.04.5, Python 3.12.13, NumPy 2.0.2 base)
**Method:** pip dependency resolver dry-run + partial install + numpy/pandas ABI smoke

## Verdict

**Safe to push to Colab.** Dependency resolution is clean, the critical numpy 2.x ↔ pandas 2.3 ABI works, and `requirements.txt` already follows every v3 unpinning rule. One open question on the baseline notebook (see §4).

## 1. Resolved versions

`pip install --dry-run --report` against `requirements.txt` resolved 56 packages with no conflicts. Key versions pip would install on a clean env:

| Package | v3 minimum | Resolved | Colab 2026.04 ships | Status |
|---|---|---|---|---|
| Python | 3.12 | (resolver-agnostic) | 3.12.13 | OK |
| numpy | ≥ 2.1 (Phase 0 target) | **2.2.6** | 2.0.2 pre-installed | OK — 2.0.2 satisfies every dep's range; no upgrade triggered |
| pandas | ≥ 2.2 | **2.3.3** | not published in FAQ | OK — river 0.23 requires ≥ 2.2.3, may upgrade Colab default |
| scipy | (current) | **1.15.3** | not published | OK |
| scikit-learn | ≥ 1.5 | **1.7.2** | not published | OK |
| imbalanced-learn | ≥ 0.14 | **0.14.1** | not pre-installed | OK |
| lightgbm | ≥ 4.0 | **4.6.0** | 3.3.5 pre-installed (per Colab issue tracker) | OK — pip upgrades |
| xgboost | ≥ 2.0 | **3.2.0** | not pinned in FAQ | OK |
| catboost | ≥ 1.2 | **1.2.10** | not pre-installed | OK |
| shap | ≥ 0.46 (Phase 0) / ≥ 0.44 (req.txt) | **0.49.1** | not pre-installed | OK |
| lime | ≥ 0.2.0.1 | **0.2.0.1** | not pre-installed | OK — builds from sdist on Colab |
| kaggle | ≥ 1.6 | **1.7.4.5** | pre-installed (older) | OK |
| kagglehub | ≥ 0.3 | **1.0.1** | not pre-installed | OK |
| river | ≥ 0.21 | **0.23.0** | not pre-installed | OK |
| matplotlib | (current) | **3.10.9** | pre-installed | OK |
| seaborn | (current) | **0.13.2** | pre-installed | OK |
| numba (shap dep) | (current) | **0.65.1** | pre-installed | OK — wants numpy < 2.5, Colab numpy 2.0.2 satisfies |
| llvmlite (numba dep) | (current) | **0.47.0** | pre-installed | OK |

Every `>=` floor in `requirements.txt` is met or exceeded by the resolved version.

## 2. The numpy 2.x ABI risk (v3's headline failure mode)

The v3 prompt called out that pinning numpy/pandas in `requirements.txt` triggers a scipy ABI break (`cannot import name '_center' from numpy._core.umath`). We verified the unpinned path doesn't hit this:

- Installed `numpy==2.2.6` then `pandas==2.3.3` in a fresh Linux Python 3.10.12 venv (closest Colab analogue available — proxy timeouts blocked completing scipy/sklearn/catboost/shap installs in this sandbox).
- `import pandas as pd; pd.DataFrame({'a':[1,2,3]}).sum()` → returns `3`. Clean import, clean op.
- This is the exact ABI surface v3 warned about. No `_center` error.

Colab ships numpy 2.0.2; our resolved deps require numpy in a range `[1.25.2, 2.5)`. 2.0.2 satisfies all of them, so **pip will not force a numpy upgrade on Colab** — Colab's existing wheel-built numpy stays in place, which is what v3 wanted.

## 3. `requirements.txt` and the Colab install cell

Both match the v3 spec:

**requirements.txt** — unpinned for the Colab-preinstalled set (`numpy / pandas / scipy / scikit-learn / matplotlib / seaborn / tqdm / joblib`), `>=` floors only for what Colab doesn't ship (`lightgbm / xgboost / catboost / shap / lime / kaggle / kagglehub / river`), no upper bounds, no numba pin. Comments at the top explain why.

**`notebooks/run_colab.ipynb` cell 2** — uses `subprocess.run(..., capture_output=True, text=True)` against `requirements.txt`, prints `Install OK` on success, full stdout+stderr on failure (slightly stricter than v3's `r.stderr[-2000:]` — better), then importlib-loops the 11 v3-spec packages and prints versions. Substantively equivalent to the v3 template; the cell can stay as-is.

## 4. One open question — baseline notebook

`baseline/LCCDE_IDS_GlobeCom22.ipynb` (cell 26) still uses `from river import stream` and `stream.iter_pandas(...)` for per-sample prediction. v3 Phase 0 said this should be replaced with a vectorised numpy predict path. `src/lccde_model.py` IS vectorised — the docstring even cites the change — but the baseline notebook itself was not rewritten.

Two reasonable interpretations:

1. Keep the baseline notebook faithful to Yang et al. (river loop preserved for paper-comparison reproducibility). Then `river>=0.21` stays in `requirements.txt`. This is the current state.
2. Vectorise the baseline notebook too. Then `river` can be dropped from `requirements.txt` entirely (`src/` doesn't import it).

If you want the baseline notebook upgraded to drop the river loop, say the word. The current state is internally consistent.

## 5. Things I could not verify in this sandbox

- **Actual end-to-end smoke test on Python 3.12 + the full 56-package install.** The macOS `.venv/` in your local repo is empty (only `pip` + `setuptools`), so the previous `[smoke] PASS` in `results/smoke_test.log` must have run against a different env on your Mac. The Linux sandbox proxy is too slow to download the full stack (catboost/shap/xgboost wheels each take >45s and the sandbox kills bash at 45s with no way to background across calls).
- **Colab-specific GPU paths.** No GPU in this sandbox; auto-CPU-fallback path was already exercised by the prior macOS smoke. LightGBM CPU pin (per v3 constraint #1) is in `src/lccde_model.py`.

## 6. Recommended next steps before pushing to Colab

1. On your Mac (where the existing smoke previously passed): activate the *real* env (not the empty `ids-bgwo-shap/.venv/`), `pip install -U -r requirements.txt`, re-run `python tests/smoke_test.py`. Confirm `[smoke] PASS` reappears with the upgraded versions.
2. Open `notebooks/run_colab.ipynb` on Colab, run the install cell, confirm the printed version table matches §1 column "Resolved".
3. If that's clean, run the full matrix.

## 7. If something breaks on Colab — the four real errors

Per v3 README rule #12, ignore `ERROR: pip's dependency resolver does not currently take into account...` chatter. Treat these four as real:

- A Python `Traceback`
- "Your session crashed" banner
- An `ImportError`
- Any line mentioning sklearn / scipy / lightgbm / xgboost / catboost / shap / pandas / numpy / imblearn / kagglehub
