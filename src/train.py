"""Train the point (median RUL) model, the two quantile interval
models, and the failure-risk classifier -- on a single C-MAPSS
sub-dataset, or on several combined (e.g. FD001+FD002+FD003+FD004 for
a much larger, more varied training set).
"""
from __future__ import annotations

import pickle
from pathlib import Path

import xgboost as xgb

from src.classification import make_failure_label, train_classifier
from src.data_loading import (drop_low_variance_sensors, load_combined_train,
                               load_config, load_train)
from src.features import add_rolling_features, get_feature_columns
from src.op_regimes import apply_regime_normalization, assign_regime, fit_regime_stats, fit_regimes
from src.rul_labels import add_train_rul

ALL_DATASETS = ["FD001", "FD002", "FD003", "FD004"]


def artifact_name(datasets: list[str]) -> str:
    if set(datasets) == set(ALL_DATASETS):
        return "ALL"
    return "-".join(datasets)


def build_training_table(datasets: list[str], cfg: dict):
    """Runs the full feature pipeline on the training split(s) and
    returns (X, y, feature_columns, dropped_sensor_cols, regime_model,
    regime_stats). regime_model/regime_stats are None for a single
    sub-dataset (no operating-condition normalization needed); for
    combined datasets they must be reused at inference time so test
    data is normalized the same way."""
    combined = len(datasets) > 1
    df = load_combined_train(datasets, cfg) if combined else load_train(datasets[0], cfg)
    df = add_train_rul(df, clip=cfg["labels"]["rul_clip"])

    all_sensor_cols = [c for c in cfg["data"]["columns"] if c.startswith("s")]
    df, dropped = drop_low_variance_sensors(df, all_sensor_cols)
    sensor_cols = [c for c in all_sensor_cols if c not in dropped]

    regime_model = None
    regime_stats = None
    if combined:
        # FD002/FD004 run under 6 distinct operating conditions; FD001/FD003
        # under 1. Normalize each sensor within its own operating-condition
        # regime before rolling-window features, or the model partly learns
        # "which condition is this" instead of "how worn is this engine."
        n_regimes = cfg["features"].get("n_regimes", 6)
        regime_model = fit_regimes(df, n_clusters=n_regimes)
        regimes = assign_regime(df, regime_model)
        regime_stats = fit_regime_stats(df, sensor_cols, regimes)
        df = apply_regime_normalization(df, sensor_cols, regimes, regime_stats)

    df = add_rolling_features(df, sensor_cols, cfg["features"]["rolling_windows"])

    op_cols = [c for c in cfg["data"]["columns"] if c.startswith("op_setting")]
    feature_cols = get_feature_columns(df, sensor_cols, op_cols)

    X = df[feature_cols]
    y = df["rul"]
    return X, y, feature_cols, dropped, regime_model, regime_stats


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


def run_training(datasets: list[str], cfg: dict) -> None:
    X, y, feature_cols, dropped, regime_model, regime_stats = build_training_table(datasets, cfg)

    point_model = train_point_model(X, y, cfg)
    lower_model = train_quantile_model(X, y, cfg["model"]["quantile"]["lower"], cfg)
    upper_model = train_quantile_model(X, y, cfg["model"]["quantile"]["upper"], cfg)

    failure_threshold = cfg["model"].get("failure_threshold", 30)
    y_binary = make_failure_label(y, threshold=failure_threshold)
    classifier_model = train_classifier(X, y_binary, cfg)

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    name = artifact_name(datasets)
    artifact = {
        "point_model": point_model,
        "lower_model": lower_model,
        "upper_model": upper_model,
        "classifier_model": classifier_model,
        "failure_threshold": failure_threshold,
        "feature_cols": feature_cols,
        "dropped_sensor_cols": dropped,
        "datasets": datasets,
        "regime_model": regime_model,
        "regime_stats": regime_stats,
    }
    out_path = models_dir / f"{name}_models.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(artifact, f)
    print(f"Trained on {', '.join(datasets)}  ({len(X)} rows)")
    print(f"Saved model artifact to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="FD001",
        help="One dataset (e.g. FD001), a comma-separated list (e.g. "
             "FD001,FD002,FD003,FD004), or ALL for all four combined.",
    )
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    datasets = ALL_DATASETS if args.dataset.upper() == "ALL" else args.dataset.split(",")

    cfg = load_config(args.config)
    run_training(datasets, cfg)
