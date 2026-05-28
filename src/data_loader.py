"""
Dataset loaders for CIC-IDS2017 and UNSW-NB15.

Search order:
  1. `cfg.local_cicids_csv` (or UNSW equivalent) — used by the smoke test
     and any environment that already has the CSV on disk.
  2. `cfg.datasets_dir` cache — anything previously unzipped.
  3. Kaggle API download via the `kaggle` CLI (the Colab launcher uploads
     `kaggle.json` interactively and `install_kaggle_credentials_from_upload`
     drops it into `~/.kaggle/` with chmod 600 before this is called).

Cleaning steps applied uniformly: strip whitespace from column names,
drop all-null columns, coerce stringy numeric columns, drop ±inf / NaN,
drop duplicates, encode the `Label` column to integer ids.
"""
from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional
import os
import subprocess
import zipfile
import glob

import numpy as np
import pandas as pd

from .config import Config


# Canonical Kaggle dataset slugs.
KAGGLE_SLUGS = {
    "cicids2017": "cicdataset/cicids2017",
    "unsw_nb15":  "mrwellsdavid/unsw-nb15",
}


# ====================================================================== #
# Public entry point
# ====================================================================== #
def load_dataset(cfg: Config) -> Tuple[pd.DataFrame, pd.Series, list, dict]:
    """
    Returns (X, y, feature_names, label_mapping) for the configured dataset.
    Tries (in order):
      1. cfg.local_cicids_csv (or local_unsw_csv) if set and present
      2. cfg.datasets_dir cache
      3. Kaggle API download into cfg.datasets_dir
    """
    cfg.ensure_dirs()
    if cfg.dataset == "cicids2017":
        df = _load_cicids2017(cfg)
    elif cfg.dataset == "unsw_nb15":
        df = _load_unsw_nb15(cfg)
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")

    df = _clean(df)
    X, y, label_map = _split_xy(df)
    return X, y, list(X.columns), label_map


# ====================================================================== #
# CIC-IDS2017
# ====================================================================== #
def _load_cicids2017(cfg: Config) -> pd.DataFrame:
    # 1. explicit local file (used in the sandbox smoke test)
    if cfg.local_cicids_csv and Path(cfg.local_cicids_csv).exists():
        if cfg.verbose:
            print(f"[data] loading local CIC-IDS2017 CSV: {cfg.local_cicids_csv}")
        return pd.read_csv(cfg.local_cicids_csv)

    # 2. cached
    cache_dir = cfg.datasets_dir / "cicids2017"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = sorted(glob.glob(str(cache_dir / "**/*.csv"), recursive=True))
    if cached:
        if cfg.verbose:
            print(f"[data] reading {len(cached)} cached CIC-IDS2017 CSV(s)")
        return _concat_csvs(cached)

    # 3. Kaggle download
    _kaggle_download(KAGGLE_SLUGS["cicids2017"], cache_dir, cfg.verbose)
    cached = sorted(glob.glob(str(cache_dir / "**/*.csv"), recursive=True))
    if not cached:
        raise FileNotFoundError(
            "Could not locate CIC-IDS2017 CSVs after Kaggle download."
        )
    return _concat_csvs(cached)


def _load_unsw_nb15(cfg: Config) -> pd.DataFrame:
    cache_dir = cfg.datasets_dir / "unsw_nb15"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # UNSW-NB15 ships with a pre-split train/test on Kaggle.
    cached = sorted(glob.glob(str(cache_dir / "**/UNSW_NB15_*.csv"), recursive=True))
    if not cached:
        _kaggle_download(KAGGLE_SLUGS["unsw_nb15"], cache_dir, cfg.verbose)
        cached = sorted(glob.glob(str(cache_dir / "**/UNSW_NB15_*.csv"), recursive=True))

    # Prefer the canonical training_set + testing_set if present.
    train = [p for p in cached if "training-set" in p.lower() or "training_set" in p.lower()]
    test  = [p for p in cached if "testing-set"  in p.lower() or "testing_set"  in p.lower()]
    if train and test:
        df = pd.concat([pd.read_csv(train[0]), pd.read_csv(test[0])], ignore_index=True)
    else:
        df = _concat_csvs(cached)

    # UNSW uses 'attack_cat' for the multiclass label, with NaN meaning normal.
    if "attack_cat" in df.columns:
        df["attack_cat"] = df["attack_cat"].fillna("Normal").astype(str).str.strip()
        df = df.rename(columns={"attack_cat": "Label"})
        if "label" in df.columns:
            df = df.drop(columns=["label"])
        if "id" in df.columns:
            df = df.drop(columns=["id"])
    elif "Label" not in df.columns:
        raise ValueError("UNSW-NB15 CSV missing both 'attack_cat' and 'Label' columns.")
    return df


# ====================================================================== #
# Helpers
# ====================================================================== #
def _concat_csvs(paths) -> pd.DataFrame:
    frames = []
    for p in paths:
        try:
            frames.append(pd.read_csv(p, low_memory=False))
        except UnicodeDecodeError:
            frames.append(pd.read_csv(p, low_memory=False, encoding="latin-1"))
    return pd.concat(frames, ignore_index=True)


def _kaggle_download(slug: str, target_dir: Path, verbose: bool) -> None:
    """Run `kaggle datasets download -d <slug> -p target_dir --unzip`."""
    if verbose:
        print(f"[data] downloading {slug} from Kaggle into {target_dir} ...")
    cmd = ["kaggle", "datasets", "download", "-d", slug,
           "-p", str(target_dir), "--unzip", "--quiet"]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as e:
        raise RuntimeError(
            "Kaggle CLI not installed. Run `pip install kaggle` and place "
            "kaggle.json in ~/.kaggle/."
        ) from e
    except subprocess.CalledProcessError as e:
        # Some Kaggle datasets ship as a zip the CLI doesn't auto-extract.
        for zp in target_dir.glob("*.zip"):
            with zipfile.ZipFile(zp) as zf:
                zf.extractall(target_dir)
        if not list(target_dir.glob("**/*.csv")):
            raise RuntimeError(f"Kaggle download failed for {slug}: {e}") from e


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names, drop dupes, drop inf/NaN rows."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    # Drop columns that are entirely null (some CIC files have stray ones)
    df = df.dropna(axis=1, how="all")
    # Coerce object columns that should be numeric (CIC has " " values)
    for c in df.columns:
        if c == "Label":
            continue
        if df[c].dtype == "object":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def _split_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, dict]:
    if "Label" not in df.columns:
        raise ValueError("Cleaned dataframe missing 'Label' column.")
    y_raw = df["Label"].astype(str)
    X = df.drop(columns=["Label"])
    # Drop any remaining non-numeric columns (e.g. flow IDs, timestamps).
    non_num = [c for c in X.columns if not np.issubdtype(X[c].dtype, np.number)]
    if non_num:
        X = X.drop(columns=non_num)
    classes = sorted(y_raw.unique())
    label_map = {name: i for i, name in enumerate(classes)}
    y = y_raw.map(label_map).astype(int)
    return X.reset_index(drop=True), y.reset_index(drop=True), label_map


# ====================================================================== #
# Kaggle credentials helper (used by the Colab launcher; safe-noop locally)
# ====================================================================== #
def install_kaggle_credentials_from_upload(uploaded: dict) -> None:
    """
    Given the dict returned by google.colab.files.upload() containing a
    `kaggle.json` entry, install it to ~/.kaggle/kaggle.json with the right
    permissions.
    """
    if "kaggle.json" not in uploaded:
        raise ValueError("Expected to receive a kaggle.json file.")
    kdir = Path(os.path.expanduser("~/.kaggle"))
    kdir.mkdir(parents=True, exist_ok=True)
    target = kdir / "kaggle.json"
    target.write_bytes(uploaded["kaggle.json"])
    target.chmod(0o600)
    print(f"[kaggle] credentials installed at {target}")
