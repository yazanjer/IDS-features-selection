from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import os
import random
from typing import Optional, Tuple, List

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
DATASETS_DIR = REPO_ROOT / "datasets"


@dataclass
class Config:
    # ------------------------------------------------------------------ #
    # Dataset selection
    #
    # The defaults below are *paper-grade* — they are the values used to
    # produce the headline results in the README's comparison table. They
    # assume a Colab Pro+ A100 (or equivalent) and ~80 GB RAM. For a
    # quick smoke run on a laptop, use Config(sample_size=50_000,
    # n_seeds=2, bgwo_population=6, bgwo_iterations=10) or call
    # `smoke_config()` directly.
    # ------------------------------------------------------------------ #
    dataset: str = "cicids2017"            # {"cicids2017", "unsw_nb15"}
    sample_size: int = 500_000             # stratified sample after load
    test_size: float = 0.2

    # ------------------------------------------------------------------ #
    # Reproducibility
    # ------------------------------------------------------------------ #
    seed: int = 0
    n_seeds: int = 10                      # multi-seed mean ± std + Wilcoxon

    # ------------------------------------------------------------------ #
    # Feature selection method (controls the FS branch of the pipeline)
    # ------------------------------------------------------------------ #
    fs_method: str = "bgwo_shap"           # see evaluation.DEFAULT_METHODS for the full set

    # ------------------------------------------------------------------ #
    # BGWO hyperparameters
    # ------------------------------------------------------------------ #
    bgwo_population: int = 15              # pack size
    bgwo_iterations: int = 30              # convergence budget
    bgwo_transfer: str = "s_shape"         # {"s_shape", "v_shape"} sigmoid family
    bgwo_init_density: float = 0.5         # initial fraction of features turned on

    # ------------------------------------------------------------------ #
    # Tri-objective fitness weights
    #   fitness = alpha * (1 - macro_F1)
    #           + beta  * (|S| / |F|)
    #           + gamma * (1 - explanation_consistency)
    # Set gamma=0 to recover the bi-objective ablation.
    # ------------------------------------------------------------------ #
    alpha: float = 0.85
    beta: float = 0.10
    gamma: float = 0.05

    # SHAP top-k overlap window used inside explanation_consistency.
    shap_top_k: Optional[int] = None       # None -> use |S| (the subset size itself)
    shap_background_samples: int = 200     # for SHAP TreeExplainer background

    # ------------------------------------------------------------------ #
    # Inner-loop training budget for the FS search (must be cheap).
    # The *final* fit on the chosen subset uses full sample_size.
    # ------------------------------------------------------------------ #
    fs_train_rows: int = 30_000
    fs_test_rows: int = 10_000

    # ------------------------------------------------------------------ #
    # SMOTE behaviour (mirrors the LCCDE baseline)
    # ------------------------------------------------------------------ #
    smote_min_count: int = 1_000

    # ------------------------------------------------------------------ #
    # Paths
    # ------------------------------------------------------------------ #
    results_dir: Path = field(default_factory=lambda: RESULTS_DIR)
    datasets_dir: Path = field(default_factory=lambda: DATASETS_DIR)
    local_cicids_csv: Optional[Path] = None   # fallback for sandbox

    # ------------------------------------------------------------------ #
    # Misc
    # ------------------------------------------------------------------ #
    verbose: bool = True

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #
    def to_json(self) -> str:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        return json.dumps(d, indent=2, default=str)

    def ensure_dirs(self) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------- #
# Reproducibility helper.
# ---------------------------------------------------------------------- #
def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, hash randomization, and the major ML libs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    # Best-effort lib seeding — silently skipped if a lib isn't installed.
    try:
        import lightgbm                 # noqa: F401
    except Exception:
        pass
    try:
        import xgboost                  # noqa: F401
    except Exception:
        pass


# ---------------------------------------------------------------------- #
# A small "smoke" config used by the tests; the production defaults sit
# on the Config dataclass above.
# ---------------------------------------------------------------------- #
def smoke_config() -> Config:
    """Tiny config used by tests/smoke_test.py — proves the pipeline wires
    together end-to-end, not that the numbers are competitive."""
    return Config(
        dataset="cicids2017",
        sample_size=3_000,
        seed=0,
        n_seeds=1,
        fs_method="bgwo_shap",
        bgwo_population=3,
        bgwo_iterations=2,
        fs_train_rows=1_200,
        fs_test_rows=600,
        shap_background_samples=20,
        smote_min_count=200,
        verbose=True,
    )
