"""Load and parse NASA C-MAPSS raw text files into pandas DataFrames.

The raw files are whitespace-delimited, have no header row, and (in the
common redistributions) an extra trailing pair of blank columns from a
trailing delimiter. This module handles both quirks.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _read_raw_txt(path: Path, columns: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    # Some redistributions of this dataset have 2 trailing all-NaN columns
    # from a stray delimiter at end-of-line; drop any columns beyond the
    # expected schema.
    df = df.iloc[:, : len(columns)]
    df.columns = columns
    return df


def load_train(dataset: str, cfg: dict) -> pd.DataFrame:
    """Load train_<dataset>.txt, e.g. dataset='FD001'."""
    raw_dir = Path(cfg["data"]["raw_dir"])
    path = raw_dir / f"train_{dataset}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the C-MAPSS dataset and place "
            f"train_{dataset}.txt in {raw_dir}/ (see README)."
        )
    return _read_raw_txt(path, cfg["data"]["columns"])


def load_test(dataset: str, cfg: dict) -> pd.DataFrame:
    """Load test_<dataset>.txt (truncated run-to-failure sequences)."""
    raw_dir = Path(cfg["data"]["raw_dir"])
    path = raw_dir / f"test_{dataset}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the C-MAPSS dataset and place "
            f"test_{dataset}.txt in {raw_dir}/ (see README)."
        )
    return _read_raw_txt(path, cfg["data"]["columns"])


def load_test_rul(dataset: str, cfg: dict) -> pd.Series:
    """Load RUL_<dataset>.txt: the true RUL at the truncation point of
    each test engine, one value per engine, in engine-number order."""
    raw_dir = Path(cfg["data"]["raw_dir"])
    path = raw_dir / f"RUL_{dataset}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the C-MAPSS dataset and place "
            f"RUL_{dataset}.txt in {raw_dir}/ (see README)."
        )
    vals = pd.read_csv(path, sep=r"\s+", header=None)[0]
    vals.index = range(1, len(vals) + 1)  # engine numbers are 1-indexed
    vals.name = "true_rul"
    return vals


# Offset added to unit_number when combining sub-datasets, so engine IDs
# stay unique across FD001-FD004 (each sub-dataset numbers its own
# engines starting at 1). Spacing of 1000 comfortably covers FD002's
# largest engine count (260).
DATASET_OFFSETS = {"FD001": 0, "FD002": 1000, "FD003": 2000, "FD004": 3000}


def load_combined_train(datasets: list[str], cfg: dict) -> pd.DataFrame:
    """Load and concatenate train_<X>.txt for each dataset in `datasets`,
    remapping unit_number so engines from different sub-datasets never
    collide, and tagging each row with which sub-dataset it came from."""
    frames = []
    for ds in datasets:
        df = load_train(ds, cfg)
        df = df.copy()
        df["unit_number"] = df["unit_number"] + DATASET_OFFSETS[ds]
        df["source_dataset"] = ds
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_combined_test(datasets: list[str], cfg: dict) -> pd.DataFrame:
    frames = []
    for ds in datasets:
        df = load_test(ds, cfg)
        df = df.copy()
        df["unit_number"] = df["unit_number"] + DATASET_OFFSETS[ds]
        df["source_dataset"] = ds
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_combined_test_rul(datasets: list[str], cfg: dict) -> pd.Series:
    """Same as load_test_rul, but combined and reindexed to match the
    offset unit_numbers used by load_combined_test."""
    parts = []
    for ds in datasets:
        rul = load_test_rul(ds, cfg)
        rul = rul.copy()
        rul.index = rul.index + DATASET_OFFSETS[ds]
        parts.append(rul)
    return pd.concat(parts)


def drop_low_variance_sensors(df: pd.DataFrame, sensor_cols: list[str],
                               tol: float = 1e-6) -> tuple[pd.DataFrame, list[str]]:
    """Drop sensor columns with ~zero variance (uninformative on this
    sub-dataset's operating conditions). Returns the trimmed df and the
    list of columns that were dropped."""
    variances = df[sensor_cols].var()
    dropped = variances[variances < tol].index.tolist()
    return df.drop(columns=dropped), dropped
