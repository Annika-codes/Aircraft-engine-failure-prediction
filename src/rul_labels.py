"""Compute Remaining Useful Life (RUL) labels for train and test splits."""
from __future__ import annotations

import pandas as pd


def add_train_rul(df: pd.DataFrame, clip: int | None = 125) -> pd.DataFrame:
    """Training engines run to failure, so RUL at each row is simply
    (max cycle for that engine) - (current cycle).

    Early in an engine's life RUL isn't really inferable from sensors
    (it hasn't started degrading yet), so by convention this is clipped
    to a max value -- training on the unclipped ramp just adds noise
    without adding signal. Set clip=None to disable.
    """
    df = df.copy()
    max_cycle = df.groupby("unit_number")["time_in_cycles"].transform("max")
    rul = max_cycle - df["time_in_cycles"]
    if clip is not None:
        rul = rul.clip(upper=clip)
    df["rul"] = rul
    return df


def add_test_rul(df: pd.DataFrame, true_rul: pd.Series,
                  clip: int | None = 125) -> pd.DataFrame:
    """Test engines are truncated before failure. true_rul gives the
    actual remaining cycles at the truncation point (last row per
    engine); earlier rows get true_rul + (cycles remaining until the
    truncation point)."""
    df = df.copy()
    max_cycle = df.groupby("unit_number")["time_in_cycles"].transform("max")
    engine_true_rul = df["unit_number"].map(true_rul)
    rul = (max_cycle - df["time_in_cycles"]) + engine_true_rul
    if clip is not None:
        rul = rul.clip(upper=clip)
    df["rul"] = rul
    return df


def last_cycle_per_engine(df: pd.DataFrame) -> pd.DataFrame:
    """For test-time inference: NASA's benchmark task predicts RUL only
    at the final observed cycle of each (truncated) test engine."""
    idx = df.groupby("unit_number")["time_in_cycles"].idxmax()
    return df.loc[idx].reset_index(drop=True)
