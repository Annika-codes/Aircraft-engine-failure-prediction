"""Evaluation metrics, including the PHM08-style asymmetric RUL score.

Plain RMSE penalizes an early prediction and a late prediction by the
same amount for the same error magnitude. In practice, a *late*
prediction (model says "you have 30 cycles left" when really you had
10) risks an unplanned failure -- much more costly than an early one
(model says "10 left" when really you had 30, so maintenance happens
sooner than strictly needed). The PHM08 challenge scoring function
captures this by using an asymmetric exponential penalty.
"""
from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def phm08_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Lower is better. d = predicted - true.
    d < 0 (early prediction):  penalty = exp(-d/13) - 1
    d >= 0 (late prediction):  penalty = exp( d/10) - 1
    The late-side exponent (10) is smaller than the early-side (13),
    so the penalty grows faster for late predictions -- i.e. being
    late is punished more severely than being early by the same
    number of cycles.
    """
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    penalty = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(penalty))


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """Quantile ('pinball') loss -- used to check calibration of the
    upper/lower quantile models, separately from training."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    diff = y_true - y_pred
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1) * diff)))


def interval_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of true values that actually fall within [lower, upper].
    For a well-calibrated 10th/90th percentile interval this should be
    close to 0.80."""
    y_true = np.asarray(y_true)
    inside = (y_true >= np.asarray(lower)) & (y_true <= np.asarray(upper))
    return float(np.mean(inside))
