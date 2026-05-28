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


# Kaggle dataset slugs to try, in priority order. Kaggle datasets get
# renamed/removed periodically, so we try each in turn and use the first
# one that downloads at least one CSV. You can monkey-patch this list at
# runtime from a Colab cell: `data_loader.KAGGLE_SLUGS["cicids2017"] = [...]`
KAGGLE_SLUGS = {
    "cicids2017": [
        "dhoogla/distrinetcicids2017",          # ✓ verified working via kagglehub
        "chethuhn/network-intrusion-dataset",   # MachineLearningCVE 7-CSV bundle
        "dhoogla/cicids2017",                   # preprocessed variant
        "cicdataset/cicids2017",                # legacy (often returns 404)
    ],
    "unsw_nb15": [
        "alextamboli/unsw-nb15",                # ✓ verified working via kagglehub
        "mrwellsdavid/unsw-nb15",
        "dhoogla/unswnb15",
    ],
}

# Label-column names seen across the preprocessed Kaggle variants.
# Order matters — first match wins, so prefer canonical 'Label' over
# 'attack_cat' (UNSW) over single-word fallbacks.
LABEL_COLUMN_ALIASES = [
    "Label", "label", "LABEL",
    "attack_cat", "Attack", "attack", "attack_label", "attack_type",
    "Class", "class",
    "category", "Category",
]


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

    # 3. Kaggle download — try each candidate slug in turn.
    _kaggle_download_first_working(KAGGLE_SLUGS["cicids2017"], cache_dir, cfg.verbose)
    cached = sorted(glob.glob(str(cache_dir / "**/*.csv"), recursive=True))
    if not cached:
        raise FileNotFoundError(
            "Could not locate CIC-IDS2017 CSVs after Kaggle download. "
            "Verify access at https://www.kaggle.com/ — open each slug in "
            f"{KAGGLE_SLUGS['cicids2017']} and click Download once to accept "
            "any terms-of-use prompt."
        )
    return _concat_csvs(cached)


def _load_unsw_nb15(cfg: Config) -> pd.DataFrame:
    cache_dir = cfg.datasets_dir / "unsw_nb15"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # UNSW-NB15 ships with a pre-split train/test on Kaggle.
    cached = sorted(glob.glob(str(cache_dir / "**/UNSW_NB15_*.csv"), recursive=True))
    if not cached:
        _kaggle_download_first_working(KAGGLE_SLUGS["unsw_nb15"], cache_dir, cfg.verbose)
        cached = sorted(glob.glob(str(cache_dir / "**/UNSW_NB15_*.csv"), recursive=True))
        # Some UNSW slugs ship CSVs without the UNSW_NB15_ prefix.
        if not cached:
            cached = sorted(glob.glob(str(cache_dir / "**/*.csv"), recursive=True))

    # Prefer the canonical training_set + testing_set if present.
    train = [p for p in cached if "training-set" in p.lower() or "training_set" in p.lower()]
    test  = [p for p in cached if "testing-set"  in p.lower() or "testing_set"  in p.lower()]
    if train and test:
        df = pd.concat([pd.read_csv(train[0]), pd.read_csv(test[0])], ignore_index=True)
    else:
        df = _concat_csvs(cached)

    # Drop the row-id column if the dataset variant carries one — it's a
    # sample index, not a feature.
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    # Label-column normalisation (attack_cat → Label, NaN → 'Normal') is
    # now handled centrally in _clean()/_find_label_column().
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
    """Run `kaggle datasets download -d <slug> -p target_dir --unzip`.

    No --quiet: kaggle's stderr is the only way to learn that a slug was
    renamed, requires terms acceptance, or that the dataset is private.
    """
    if verbose:
        print(f"[data] downloading {slug} from Kaggle into {target_dir} ...")
    cmd = ["kaggle", "datasets", "download", "-d", slug,
           "-p", str(target_dir), "--unzip"]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if verbose and proc.stdout:
            print(proc.stdout)
    except FileNotFoundError as e:
        raise RuntimeError(
            "Kaggle CLI not installed. Run `pip install kaggle` and place "
            "kaggle.json in ~/.kaggle/."
        ) from e
    except subprocess.CalledProcessError as e:
        # Fall through: some datasets ship as a zip the CLI doesn't auto-extract.
        for zp in target_dir.glob("*.zip"):
            with zipfile.ZipFile(zp) as zf:
                zf.extractall(target_dir)
        if not list(target_dir.glob("**/*.csv")):
            err = (e.stderr or "").strip() or (e.stdout or "").strip() or str(e)
            raise RuntimeError(
                f"Kaggle download failed for {slug}.\n"
                f"  exit_code={e.returncode}\n"
                f"  kaggle CLI said: {err}"
            ) from e


def _kagglehub_download(slug: str, target_dir: Path, verbose: bool) -> bool:
    """
    Preferred download path: uses the `kagglehub` package, which handles
    auth automatically (no need for ~/.kaggle/kaggle.json in Colab) and
    caches to ~/.cache/kagglehub/. Returns True if at least one CSV
    landed in `target_dir`, False if kagglehub isn't installed or the
    dataset returned no CSVs.
    """
    try:
        import kagglehub
    except ImportError:
        return False
    if verbose:
        print(f"[data] kagglehub: downloading {slug} ...")
    try:
        cached_path = Path(kagglehub.dataset_download(slug))
    except Exception as e:
        if verbose:
            print(f"[data] kagglehub failed for {slug}: {type(e).__name__}: {e}")
        return False
    csvs = sorted(cached_path.glob("**/*.csv"))
    if not csvs:
        if verbose:
            print(f"[data] kagglehub: {slug} downloaded but contains no CSVs")
        return False
    # Mirror the kagglehub cache structure into target_dir via symlinks
    # (or copies if the filesystem doesn't allow symlinks).
    target_dir.mkdir(parents=True, exist_ok=True)
    for csv in csvs:
        rel = csv.relative_to(cached_path)
        dest = target_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            try:
                dest.symlink_to(csv)
            except OSError:
                import shutil
                shutil.copy2(csv, dest)
    if verbose:
        print(f"[data] kagglehub: {len(csvs)} CSV(s) staged into {target_dir}")
    return True


def _kaggle_download_first_working(
    slugs, target_dir: Path, verbose: bool
) -> str:
    """
    Try each slug in `slugs` (str or list). For each slug, attempt
    `kagglehub` first (no auth file needed, more reliable), then fall
    back to the `kaggle` CLI. Return the first slug that puts ≥1 CSV
    in target_dir. Aggregate every failure so the user sees them all.
    """
    if isinstance(slugs, str):
        slugs = [slugs]
    errors = []
    for slug in slugs:
        # 1. kagglehub
        if _kagglehub_download(slug, target_dir, verbose):
            if verbose:
                print(f"[data] success with slug: {slug} (kagglehub)")
            return slug
        # 2. kaggle CLI fallback
        try:
            _kaggle_download(slug, target_dir, verbose)
            if list(target_dir.glob("**/*.csv")):
                if verbose:
                    print(f"[data] success with slug: {slug} (kaggle CLI)")
                return slug
            errors.append(f"{slug}: kaggle CLI succeeded but no CSVs found")
        except RuntimeError as e:
            errors.append(str(e))
            continue
    raise RuntimeError(
        "All Kaggle slug candidates failed:\n  - "
        + "\n  - ".join(errors)
        + "\nFix options:\n"
          "  - Install kagglehub:   pip install kagglehub\n"
          "  - Or accept terms-of-use on the dataset's Kaggle page in a browser\n"
          "  - Or pin a working slug:\n"
          "      data_loader.KAGGLE_SLUGS['cicids2017'] = ['your/working-slug']"
    )


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names, normalise the label column,
    drop dupes, drop inf/NaN rows."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Normalise the label column to 'Label' regardless of which alias the
    # preprocessed dataset variant happened to use.
    label_col = _find_label_column(df)
    if label_col is None:
        raise ValueError(
            f"Could not find a label column in the dataset. "
            f"Looked for any of: {LABEL_COLUMN_ALIASES}. "
            f"Available columns: {list(df.columns)[:25]}..."
        )
    if label_col != "Label":
        # UNSW's attack_cat uses NaN to mean 'normal' — preserve that mapping.
        if label_col == "attack_cat":
            df[label_col] = df[label_col].fillna("Normal")
        # If both the alias and a separate 'label' col exist (UNSW), drop the
        # numeric binary 'label' to avoid double-keep.
        if "label" in df.columns and label_col != "label":
            df = df.drop(columns=["label"])
        df = df.rename(columns={label_col: "Label"})

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


def _find_label_column(df: pd.DataFrame) -> Optional[str]:
    """Return the first matching label-column alias actually present in df."""
    cols_lower = {c.lower(): c for c in df.columns}
    for alias in LABEL_COLUMN_ALIASES:
        if alias in df.columns:
            return alias
        if alias.lower() in cols_lower:
            return cols_lower[alias.lower()]
    return None


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
