"""Inference: apply a saved model artifact to the C-MAPSS test split(s)
(or any freshly-loaded engine data), producing a point RUL estimate, an
80% interval, a failure-risk probability, and a simple risk tier for
each engine's latest cycle. Works the same whether the artifact was
trained on one sub-dataset or several combined."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loading import load_combined_test, load_combined_test_rul, load_config, load_test, load_test_rul
from src.features import add_rolling_features
from src.op_regimes import apply_regime_normalization, assign_regime
from src.rul_labels import add_test_rul, last_cycle_per_engine
from src.sensor_names import SENSOR_INFO, category_of
from src.train import ALL_DATASETS, artifact_name


def resolve_datasets(dataset_arg: str) -> list[str]:
    if dataset_arg.upper() == "ALL":
        return list(ALL_DATASETS)
    return dataset_arg.split(",")


def load_artifact(dataset_arg: str, models_dir: str = "models") -> dict:
    """dataset_arg may be a single name (FD001), a comma-separated list
    (FD001,FD002), or ALL -- resolved to the same artifact filename
    run_training() would have saved it under."""
    datasets = resolve_datasets(dataset_arg)
    name = artifact_name(datasets)
    path = Path(models_dir) / f"{name}_models.pkl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run `python main.py train` first.")
    with open(path, "rb") as f:
        return pickle.load(f)


def build_test_table(datasets: list[str], cfg: dict, artifact: dict):
    combined = len(datasets) > 1
    if combined:
        df = load_combined_test(datasets, cfg)
        true_rul = load_combined_test_rul(datasets, cfg)
    else:
        df = load_test(datasets[0], cfg)
        true_rul = load_test_rul(datasets[0], cfg)
    df = add_test_rul(df, true_rul, clip=cfg["labels"]["rul_clip"])

    all_sensor_cols = [c for c in cfg["data"]["columns"] if c.startswith("s")]
    sensor_cols = [c for c in all_sensor_cols if c not in artifact["dropped_sensor_cols"]]
    df = df.drop(columns=[c for c in artifact["dropped_sensor_cols"] if c in df.columns])

    if artifact.get("regime_model") is not None:
        regimes = assign_regime(df, artifact["regime_model"])
        df = apply_regime_normalization(df, sensor_cols, regimes, artifact["regime_stats"])

    df = add_rolling_features(df, sensor_cols, cfg["features"]["rolling_windows"])
    return df


def risk_tier(rul_pred: float, cfg: dict) -> str:
    tiers = cfg["dashboard"]["risk_tiers"]
    if rul_pred < tiers["urgent_rul_below"]:
        return "Urgent"
    if rul_pred < tiers["watch_rul_below"]:
        return "Watch"
    return "Safe"


def predict_latest_cycle(dataset_arg: str, cfg: dict, artifact: dict | None = None) -> pd.DataFrame:
    """Returns one row per test engine (its most recent observed
    cycle): point RUL estimate, 80% interval, true RUL, risk tier,
    failure probability, and which sub-dataset it came from."""
    artifact = artifact or load_artifact(dataset_arg)
    datasets = artifact.get("datasets") or resolve_datasets(dataset_arg)
    df = build_test_table(datasets, cfg, artifact)
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

    # Failure-risk classifier: probability this engine is in the
    # "Failure" (high-risk) class, i.e. true RUL below the training
    # threshold (see config.yaml -> model.failure_threshold).
    failure_proba = artifact["classifier_model"].predict_proba(X)[:, 1]
    latest["failure_probability"] = failure_proba
    latest["predicted_status"] = np.where(
        failure_proba >= 0.5, "Likely Failure", "Likely Normal"
    )

    cols = ["unit_number", "time_in_cycles", "rul", "rul_pred",
            "rul_lower", "rul_upper", "risk_tier",
            "failure_probability", "predicted_status"]
    if "source_dataset" in latest.columns:
        cols.append("source_dataset")
    return latest[cols].rename(columns={"rul": "true_rul", "unit_number": "engine_id"})


def feature_importance_by_category(artifact: dict, top_n: int = 6) -> pd.DataFrame:
    """Aggregate the classifier's feature importances by sensor
    category (Temperature / Pressure / Speed / Fuel Flow / Other),
    since importance on individual engineered columns (e.g.
    's4_rmean_20') isn't meaningful to a general audience."""
    model = artifact["classifier_model"]
    feature_cols = artifact["feature_cols"]
    importances = model.feature_importances_

    cat_totals: dict[str, float] = {}
    for col, imp in zip(feature_cols, importances):
        cat = category_of(col)
        cat_totals[cat] = cat_totals.get(cat, 0.0) + float(imp)

    total = sum(cat_totals.values()) or 1.0
    rows = [{"category": cat, "importance": val / total}
            for cat, val in cat_totals.items()]
    df = pd.DataFrame(rows).sort_values("importance", ascending=False)
    return df.head(top_n).reset_index(drop=True)


def evaluate_classifier_on_test(dataset_arg: str, cfg: dict, artifact: dict | None = None):
    """Runs the classifier on the same held-out test engines used for
    regression evaluation, deriving ground-truth failure labels from
    the true RUL the same way training labels were derived."""
    from src.classification import evaluate_classifier, make_failure_label

    artifact = artifact or load_artifact(dataset_arg)
    preds = predict_latest_cycle(dataset_arg, cfg, artifact)
    threshold = artifact.get("failure_threshold", cfg["model"].get("failure_threshold", 30))
    y_true = make_failure_label(preds["true_rul"].values, threshold=threshold)
    y_pred = (preds["failure_probability"].values >= 0.5).astype(int)
    return evaluate_classifier(y_true, y_pred)
