"""Operating-condition regime detection and per-regime normalization.

FD001 and FD003 run under one fixed flight condition (sea level). FD002
and FD004 run under six distinct combinations of altitude, Mach number,
and throttle resolver angle. A sensor's raw reading shifts with the
operating condition itself, on top of any change from wear -- e.g. a
temperature sensor reads differently at cruise altitude vs. sea level
even on a perfectly healthy engine.

Combining all four sub-datasets without accounting for this means the
model partly learns "which operating condition is this" instead of
"how degraded is this engine" -- it would genuinely hurt accuracy, not
just add noise. The standard fix (used throughout the published RUL
literature on this dataset) is to cluster the 3 operating-condition
columns into a small number of discrete regimes, then z-score
normalize each sensor within its own regime.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

OP_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]


def fit_regimes(df: pd.DataFrame, n_clusters: int = 6, random_state: int = 42) -> KMeans:
    """Fit regime clusters on TRAINING data only."""
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    km.fit(df[OP_COLS])
    return km


def assign_regime(df: pd.DataFrame, km: KMeans) -> np.ndarray:
    return km.predict(df[OP_COLS])


def fit_regime_stats(df: pd.DataFrame, sensor_cols: list[str],
                      regimes: np.ndarray) -> dict[int, dict[str, pd.Series]]:
    """Per-regime mean/std for each sensor, computed on training data.
    std of 0 is replaced with 1 to avoid divide-by-zero on a sensor
    that happens to be constant within one regime."""
    stats = {}
    tmp = df[sensor_cols].copy()
    tmp["_regime"] = regimes
    for regime, group in tmp.groupby("_regime"):
        stats[int(regime)] = {
            "mean": group[sensor_cols].mean(),
            "std": group[sensor_cols].std().replace(0, 1.0),
        }
    return stats


def apply_regime_normalization(df: pd.DataFrame, sensor_cols: list[str],
                                regimes: np.ndarray,
                                stats: dict[int, dict[str, pd.Series]]) -> pd.DataFrame:
    """Z-score normalize each sensor within its assigned regime, reusing
    the mean/std computed on training data (so test data is normalized
    consistently with how the model was trained, not against itself)."""
    df = df.copy()
    # Some sensor columns can be constant integers within a regime (e.g.
    # a demanded speed setting), which pandas infers as int64 -- cast to
    # float first so writing normalized (non-integer) values back doesn't
    # raise a dtype error.
    df[sensor_cols] = df[sensor_cols].astype(float)
    for regime in np.unique(regimes):
        regime = int(regime)
        mask = regimes == regime
        if regime not in stats:
            continue  # a regime that only appears at test time; leave as-is
        mean = stats[regime]["mean"]
        std = stats[regime]["std"]
        df.loc[mask, sensor_cols] = (df.loc[mask, sensor_cols] - mean) / std
    return df
