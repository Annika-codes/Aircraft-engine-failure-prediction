"""Binary 'imminent failure' classifier, alongside the RUL regressor.

Regression gives a continuous RUL estimate; this adds a complementary
classification view -- "is this engine at high risk of failing soon?"
-- with the usual accuracy/precision/recall/F1 plus a confusion matrix,
which is what most people expect to see under "model performance."

The label is derived from RUL: an engine is 'Failure' (positive class)
if its true RUL is below a threshold (default 30 cycles), else 'Normal'.
This is a standard, simple way to turn this dataset's RUL labels into a
binary risk classification without needing separate annotations.
"""
from __future__ import annotations

import numpy as np
import xgboost as xgb
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              precision_score, recall_score)


def make_failure_label(rul, threshold: int = 30):
    """1 = 'Failure' (at risk, RUL below threshold), 0 = 'Normal'."""
    return (np.asarray(rul) < threshold).astype(int)


def train_classifier(X, y_binary, cfg: dict) -> xgb.XGBClassifier:
    params = cfg["model"]["point"]  # reuse point-model hyperparams; same complexity budget
    model = xgb.XGBClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X, y_binary)
    return model


def evaluate_classifier(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]),
    }
