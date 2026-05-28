# IDS-BGWO-SHAP

**Metaheuristic feature selection + SHAP-in-the-loop explainability for the LCCDE intrusion-detection ensemble.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USER/ids-bgwo-shap/blob/main/notebooks/run_colab.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Replace `USER` in the Colab badge with your GitHub username after pushing.

## What this project is

A reproducible research extension of the open-source LCCDE IDS framework
(Yang et al., *GLOBECOM '22*). It contributes two changes to the baseline
pipeline while keeping the downstream classifier fixed, so any change in
performance is attributable to the feature-selection stage alone.

| Stage                | Baseline (LCCDE paper)              | This project                              |
|----------------------|-------------------------------------|-------------------------------------------|
| Feature selection    | Information-gain / FCBF filter      | **Binary Grey Wolf Optimizer (BGWO)**     |
| FS objective         | Univariate relevance only           | **Tri-objective: F1 + sparsity + SHAP coherence** |
| Downstream model     | LCCDE (XGBoost + LightGBM + CatBoost) | LCCDE (unchanged)                       |
| Explainability       | None / post-hoc                     | SHAP **inside** the FS loop + per-class signatures |

## The two contributions

### 1. Binary Grey Wolf Optimizer feature selection — `src/bgwo_fs.py`

A from-scratch BGWO with a real-valued continuous population of wolves
mapped to binary feature masks via an S-shaped (or V-shaped) sigmoid
transfer function. The alpha/beta/delta wolves attract the pack; the
exploration coefficient `a` decays linearly from 2 to 0 across iterations.
Empty masks are rescued by flipping a random bit on. Both population
size and iteration count are configurable from `src/config.py`.

### 2. SHAP-in-the-loop explanation coherence — `src/fitness.py`

SHAP is normally applied post-hoc. We inject an explanation-coherence
term directly into the BGWO fitness function so the optimizer can
prefer subsets whose SHAP top-k signature concentrates importance
rather than spreading it across "passenger" features. The fitness is

```
fitness(S) = α · (1 − macro_F1(S))
           + β · (|S| / |F|)
           + γ · (1 − explanation_consistency(S))
```

with `explanation_consistency(S)` defined as the fraction of mean
|SHAP| mass carried by the top-k features inside the trained subset
(averaged across the three base learners). Setting `γ = 0` recovers
the bi-objective ablation, isolating exactly the contribution of the
SHAP term. α, β, γ are all configurable.

Per-class signatures (the minimal explainable feature set per attack
type) are extracted by `src/explainability.py::compute_shap_signatures`
and rendered by `src/plots.py::plot_per_class_signatures`.

## Repo layout

```
ids-bgwo-shap/
├── src/
│   ├── config.py          # global Config dataclass + smoke_config()
│   ├── data_loader.py     # CIC-IDS2017 + UNSW-NB15 loaders (Kaggle + local)
│   ├── sampling.py        # stratified down-sampling, SMOTE rebalance
│   ├── lccde_model.py     # leader-class confidence decision ensemble
│   ├── bgwo_fs.py         # BGWO + filter baseline
│   ├── fitness.py         # tri-objective fitness w/ SHAP coherence
│   ├── explainability.py  # SHAP signatures, LIME, Kuncheva stability, fidelity
│   ├── evaluation.py      # multi-seed runner + Wilcoxon
│   └── plots.py           # all figure helpers
├── notebooks/
│   └── run_colab.ipynb    # thin Colab launcher (interactive kaggle.json + PAT)
├── tests/
│   └── smoke_test.py      # end-to-end pipeline check on a tiny config
├── baseline/
│   └── LCCDE_IDS_GlobeCom22.ipynb  # original notebook, unmodified, for reference
├── results/               # generated CSVs, JSONs, PNG/PDF figures
├── requirements.txt
├── LICENSE                # MIT
└── README.md
```

## Quickstart — Colab (recommended)

1. Click the **Open in Colab** badge above (after updating the URL with
   your GitHub username).
2. Run the cells top-to-bottom. The notebook will:
   * `pip install -r requirements.txt`
   * Prompt for an interactive `kaggle.json` upload (the API token from
     your [Kaggle account page](https://www.kaggle.com/settings/account)
     → *Create New Token*).
   * Download CIC-IDS2017 (and optionally UNSW-NB15) into the runtime.
   * Stratified-sample down to ~200K flows preserving rare classes.
   * Run the experiment matrix: 4 methods × `n_seeds`, with Wilcoxon
     signed-rank tests reported against `bgwo_shap`.
   * Run the α/β/γ and BGWO pop/iter sensitivity sweeps.
   * Render and save all plots to `results/`.
3. *Optional:* paste a GitHub Personal Access Token at the end if you
   want the runtime to push the populated `results/` folder back to
   your fork. The PAT is requested via `getpass` and never written to disk.

## Quickstart — local

```bash
git clone https://github.com/USER/ids-bgwo-shap.git
cd ids-bgwo-shap

python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Kaggle credentials (only needed for full datasets — smoke uses local CSV):
mkdir -p ~/.kaggle && cp /path/to/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

# Sanity check (~30 s on a laptop):
python tests/smoke_test.py

# Real run on CIC-IDS2017:
python -c "from src.evaluation import run_experiment_matrix, aggregate; \
           from src.config import Config; \
           df = run_experiment_matrix(Config()); print(aggregate(df))"
```

## Configuration — `src/config.py`

Every knob lives in one dataclass. Key fields:

| Field                 | Default      | Meaning                                                  |
|-----------------------|--------------|----------------------------------------------------------|
| `dataset`             | `cicids2017` | `cicids2017` or `unsw_nb15`                              |
| `sample_size`         | `200_000`    | Stratified-sample target after load                      |
| `n_seeds`             | `10`         | Multi-seed runs for mean ± std + Wilcoxon                |
| `fs_method`           | `bgwo_shap`  | `none`, `filter`, `bgwo_bi`, `bgwo_shap`                 |
| `bgwo_population`     | `10`         | BGWO pack size                                           |
| `bgwo_iterations`     | `20`         | BGWO iteration budget                                    |
| `alpha`, `beta`, `gamma` | `0.85`, `0.10`, `0.05` | Fitness weights                                |
| `fs_train_rows`       | `15_000`     | Inner-loop training subset for the FS search             |
| `fs_test_rows`        | `5_000`      | Inner-loop validation subset for the FS search           |

`Config` also exposes a `smoke_config()` factory used by `tests/smoke_test.py`.

## Evaluation protocol — what gets reported

* **Baselines (LCCDE downstream fixed in all cases):**
  1. `none` — full 77-feature set, no FS
  2. `filter` — information-gain ranking (mirrors the original repo)
  3. `bgwo_bi` — BGWO with `γ = 0` (bi-objective, SHAP-term ablated)
  4. `bgwo_shap` — full tri-objective BGWO
* **Metrics per seed:** accuracy, macro/weighted precision/recall/F1,
  per-class P/R/F1 (rare classes flagged), ROC-AUC (OvR), PR-AUC,
  confusion matrix, number of selected features, **inference latency
  (ms/flow)**, training time.
* **Aggregation:** mean ± std across `n_seeds` seeds, with
  Wilcoxon signed-rank tests vs the `bgwo_shap` reference.
* **Sensitivity sweeps:** small grid over α/β/γ; BGWO population × iterations.
* **Explainability metrics:** SHAP-vs-LIME fidelity on the LCCDE leader
  ensemble, Kuncheva index for ranking stability across seeds and
  across the two datasets, variable-stability index, per-attack-class
  minimal signature size.
* **All figures saved as PNG + PDF:** per-class F1 bars, confusion
  heatmaps, SHAP summary/beeswarm + bar, per-attack-class signatures,
  BGWO convergence curves, |S|–F1 Pareto fronts, cross-dataset overlap
  heatmap, latency-vs-feature-count plot, and a method × metric
  comparison table (CSV + rendered figure).

## Smoke-test result — sandbox run, 2026-05-28

The smoke test exercises load → sample → LCCDE → BGWO bi-objective →
BGWO tri-objective → metrics → plots on a tiny configuration
(3K flows, BGWO pop=3, iter=2, 1 seed). Full log saved at
`results/smoke_test.log`. Summary:

| Method     | macro F1 | Accuracy | \|S\| / \|F\| | Latency (ms/flow) | Fidelity |
|------------|---------:|---------:|--------------:|------------------:|---------:|
| `none`     |   0.850  |   0.993  |       77 / 77 |             0.026 |    0.004 |
| `filter`   |   0.842  |   0.988  |       39 / 77 |             0.025 |    0.112 |
| `bgwo_bi`  |   0.854  |   0.997  |       43 / 77 |             0.024 |    0.015 |
| `bgwo_shap`|   0.854  |   0.997  |       43 / 77 |             0.023 |    0.015 |

These numbers are **not** the paper-grade headline numbers — the smoke
config trains on 1.2K rows with BGWO pop=3 / iter=2 and ran in a
two-booster (LightGBM + XGBoost) sandbox mode because CatBoost was not
installable inside the timeout budget. Production runs in Colab use
the full three-booster LCCDE and the defaults in the table above.

## Compute reality check

BGWO retrains LCCDE `population × iterations` times. With the defaults
(pop 10, iter 20) that is ~200 LCCDE fits on the inner-loop subset
(15K train / 5K val rows), or roughly 30–60 minutes on a Colab free-tier
T4 per `bgwo_shap` run per seed. Use sampled subsets for the FS search
and only evaluate the final mask on the full sample. The notebook
defaults are conservative; tighten or relax `fs_train_rows` /
`fs_test_rows` / `bgwo_population` / `bgwo_iterations` to taste.

## Citations

If you use this code, please cite both the baseline framework and the
specific paper introducing LCCDE.

```bibtex
@inproceedings{yang2022lccde,
  title     = {{LCCDE}: A Decision-Based Ensemble Framework for Intrusion
               Detection in The Internet of Vehicles},
  author    = {Yang, Li and Shami, Abdallah and Stevens, Gary and De Rusett, Stephen},
  booktitle = {Proc.\ IEEE GLOBECOM},
  year      = {2022},
}

@article{yang2022mthids,
  title   = {{MTH-IDS}: A Multitiered Hybrid Intrusion Detection System for
             Internet of Vehicles},
  author  = {Yang, Li and Moubayed, Abdallah and Shami, Abdallah},
  journal = {IEEE Internet of Things Journal},
  volume  = {9}, number = {1}, pages = {616--632},
  year    = {2022},
}

@inproceedings{yang2019treebased,
  title     = {A Tree-Based Stacking Ensemble Technique with Feature
               Selection for Network Intrusion Detection},
  author    = {Yang, Li and Moubayed, Abdallah and Hamieh, Issam and Shami, Abdallah},
  booktitle = {Proc.\ IEEE GLOBECOM},
  year      = {2019},
}

@article{yang2022idsml,
  title   = {{IDS-ML}: An Open Source Code for Intrusion Detection
             System Development Using Machine Learning},
  author  = {Yang, Li and Shami, Abdallah},
  journal = {Software Impacts},
  year    = {2022},
}
```

For the BGWO algorithm itself:

```bibtex
@article{emary2016binary,
  title   = {Binary grey wolf optimization approaches for feature selection},
  author  = {Emary, E. and Zawbaa, H.M. and Hassanien, A.E.},
  journal = {Neurocomputing}, volume = {172}, pages = {371--381},
  year    = {2016},
}
```

## License

MIT — see [LICENSE](LICENSE). The baseline repository
[Western-OC2-Lab/Intrusion-Detection-System-Using-Machine-Learning](https://github.com/Western-OC2-Lab/Intrusion-Detection-System-Using-Machine-Learning)
is also MIT-licensed; its LCCDE notebook is preserved unmodified under
`baseline/` for reproducibility reference.

## Acknowledgements

LCCDE, MTH-IDS, and the tree-based IDS notebooks by the Western OC2 Lab
(L. Yang, A. Shami, et al.) provided the baseline framework that this
project extends. BGWO is due to Emary, Zawbaa, and Hassanien (2016).
SHAP is due to Lundberg & Lee (2017).
