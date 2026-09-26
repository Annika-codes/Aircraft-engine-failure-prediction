"""Rolling-window sensor features and per-engine degradation slopes.

Raw sensor readings are noisy. Rather than feed the model single noisy
readings, we add rolling mean/std over a couple of window sizes, plus a
simple linear-trend ("slope") feature that approximates how fast each
sensor is drifting -- often more predictive of imminent failure than
the raw level.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rolling_slope(x: pd.Series, window: int) -> pd.Series:
    """Slope of a simple linear fit over the trailing `window` points."""
    idx = np.arange(window)
    idx_mean = idx.mean()
    denom = ((idx - idx_mean) ** 2).sum()

    def _slope(vals: np.ndarray) -> float:
        return float(((vals - vals.mean()) * (idx - idx_mean)).sum() / denom)

    return x.rolling(window, min_periods=window).apply(_slope, raw=True)


def add_rolling_features(df: pd.DataFrame, sensor_cols: list[str],
                          windows: list[int]) -> pd.DataFrame:
    df = df.sort_values(["unit_number", "time_in_cycles"]).copy()
    grouped = df.groupby("unit_number")

    new_cols = {}
    for w in windows:
        for col in sensor_cols:
            roll = grouped[col].rolling(w, min_periods=1)
            new_cols[f"{col}_rmean_{w}"] = roll.mean().reset_index(level=0, drop=True)
            new_cols[f"{col}_rstd_{w}"] = roll.std().reset_index(level=0, drop=True).fillna(0.0)

    # Slope only really means something once a full window has elapsed;
    # use the largest window as the "trend" signal.
    trend_window = max(windows)
    for col in sensor_cols:
        slope = grouped[col].apply(lambda s: _rolling_slope(s, trend_window))
        new_cols[f"{col}_slope_{trend_window}"] = slope.reset_index(level=0, drop=True)

    feat_df = pd.DataFrame(new_cols, index=df.index)
    out = pd.concat([df, feat_df], axis=1)
    # Slope has NaNs for the first `trend_window` rows of each engine;
    # backfill within engine so early rows aren't dropped.
    slope_cols = [c for c in out.columns if c.endswith(f"_slope_{trend_window}")]
    out[slope_cols] = out.groupby("unit_number")[slope_cols].bfill()
    return out


def get_feature_columns(df: pd.DataFrame, sensor_cols: list[str],
                         op_setting_cols: list[str]) -> list[str]:
    engineered = [c for c in df.columns if any(
        c.startswith(f"{s}_") for s in sensor_cols
    )]
    return op_setting_cols + sensor_cols + engineered
