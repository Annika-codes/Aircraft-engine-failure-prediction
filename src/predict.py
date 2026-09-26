"""Inference: apply a saved model artifact to the C-MAPSS test split
(or any freshly-loaded engine data), producing a point RUL estimate, an
80% interval, and a simple risk tier for each engine's latest cycle."""
from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from src.data_loading import load_config, load_test, load_test_rul
from src.features import add_rolling_features
from src.rul_labels import add_test_rul, last_cycle_per_engine


def load_artifact(dataset: str, models_dir: str = "models") -> dict:
    path = Path(models_dir) / f"{dataset}_models.pkl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run `python main.py train` first.")
    with open(path, "rb") as f:
        return pickle.load(f)


def build_test_table(dataset: str, cfg: dict, artifact: dict):
    df = load_test(dataset, cfg)
    true_rul = load_test_rul(dataset, cfg)
    df = add_test_rul(df, true_rul, clip=cfg["labels"]["rul_clip"])

    all_sensor_cols = [c for c in cfg["data"]["columns"] if c.startswith("s")]
    sensor_cols = [c for c in all_sensor_cols if c not in artifact["dropped_sensor_cols"]]
    df = df.drop(columns=[c for c in artifact["dropped_sensor_cols"] if c in df.columns])

    df = add_rolling_features(df, sensor_cols, cfg["features"]["rolling_windows"])
    return df


def risk_tier(rul_pred: float, cfg: dict) -> str:
    tiers = cfg["dashboard"]["risk_tiers"]
    if rul_pred < tiers["urgent_rul_below"]:
        return "Urgent"
    if rul_pred < tiers["watch_rul_below"]:
        return "Watch"
    return "Safe"


def predict_latest_cycle(dataset: str, cfg: dict, artifact: dict | None = None) -> pd.DataFrame:
    """Returns one row per test engine (its most recent observed
    cycle): point RUL estimate, 80% interval, true RUL, and risk tier."""
    artifact = artifact or load_artifact(dataset)
    df = build_test_table(dataset, cfg, artifact)
    latest = last_cycle_per_engine(df)

    feature_cols = artifact["feature_cols"]
    X = latest[feature_cols]

    latest = latest.copy()
    latest["rul_pred"] = artifact["point_model"].predict(X)
    latest["rul_lower"] = artifact["lower_model"].predict(X)
    latest["rul_upper"] = artifact["upper_model"].predict(X)
    # Quantile models can occasionally cross; enforce lower <= upper.
    swap = latest["rul_lower"] > latest["rul_upper"]
    latest.loc[swap, ["rul_lower", "rul_upper"]] = latest.loc[
        swap, ["rul_upper", "rul_lower"]
    ].values

    latest["risk_tier"] = latest["rul_pred"].apply(lambda v: risk_tier(v, cfg))

    cols = ["unit_number", "time_in_cycles", "rul", "rul_pred",
            "rul_lower", "rul_upper", "risk_tier"]
    return latest[cols].rename(columns={"rul": "true_rul", "unit_number": "engine_id"})
