"""Train the point (median RUL) model and the two quantile models
(lower/upper interval bounds) on a given C-MAPSS sub-dataset."""
from __future__ import annotations

import pickle
from pathlib import Path

import xgboost as xgb

from src.data_loading import drop_low_variance_sensors, load_config, load_train
from src.features import add_rolling_features, get_feature_columns
from src.rul_labels import add_train_rul


def build_training_table(dataset: str, cfg: dict):
    """Runs the full feature pipeline on the training split and returns
    (X, y, feature_columns, dropped_sensor_cols) for reuse at inference
    time (test data must be transformed identically)."""
    df = load_train(dataset, cfg)
    df = add_train_rul(df, clip=cfg["labels"]["rul_clip"])

    all_sensor_cols = [c for c in cfg["data"]["columns"] if c.startswith("s")]
    df, dropped = drop_low_variance_sensors(df, all_sensor_cols)
    sensor_cols = [c for c in all_sensor_cols if c not in dropped]

    df = add_rolling_features(df, sensor_cols, cfg["features"]["rolling_windows"])

    op_cols = [c for c in cfg["data"]["columns"] if c.startswith("op_setting")]
    feature_cols = get_feature_columns(df, sensor_cols, op_cols)

    X = df[feature_cols]
    y = df["rul"]
    return X, y, feature_cols, dropped


def train_point_model(X, y, cfg: dict) -> xgb.XGBRegressor:
    params = cfg["model"]["point"]
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        random_state=42,
    )
    model.fit(X, y)
    return model


def train_quantile_model(X, y, quantile: float, cfg: dict) -> xgb.XGBRegressor:
    params = cfg["model"]["quantile"]
    model = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=quantile,
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        random_state=42,
    )
    model.fit(X, y)
    return model


def run_training(dataset: str, cfg: dict) -> None:
    X, y, feature_cols, dropped = build_training_table(dataset, cfg)

    point_model = train_point_model(X, y, cfg)
    lower_model = train_quantile_model(X, y, cfg["model"]["quantile"]["lower"], cfg)
    upper_model = train_quantile_model(X, y, cfg["model"]["quantile"]["upper"], cfg)

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    artifact = {
        "point_model": point_model,
        "lower_model": lower_model,
        "upper_model": upper_model,
        "feature_cols": feature_cols,
        "dropped_sensor_cols": dropped,
        "dataset": dataset,
    }
    out_path = models_dir / f"{dataset}_models.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(artifact, f)
    print(f"Saved model artifact to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="FD001")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_training(args.dataset, cfg)
