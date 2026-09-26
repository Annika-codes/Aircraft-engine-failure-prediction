import numpy as np

from src.scoring import interval_coverage, mae, phm08_score, pinball_loss, rmse


def test_rmse_zero_for_perfect_prediction():
    y = np.array([10.0, 20.0, 30.0])
    assert rmse(y, y) == 0.0


def test_mae_basic():
    assert mae(np.array([10.0]), np.array([12.0])) == 2.0


def test_phm08_penalizes_late_more_than_early():
    # Same magnitude of error (5 cycles), but late (positive d) should
    # be penalized more heavily than early (negative d).
    y_true = np.array([50.0])
    late_pred = np.array([55.0])   # d = +5
    early_pred = np.array([45.0])  # d = -5
    late_penalty = phm08_score(y_true, late_pred)
    early_penalty = phm08_score(y_true, early_pred)
    assert late_penalty > early_penalty


def test_phm08_zero_for_perfect_prediction():
    y = np.array([10.0, 50.0, 100.0])
    assert phm08_score(y, y) == 0.0


def test_pinball_loss_zero_for_perfect_prediction():
    y = np.array([10.0, 20.0])
    assert pinball_loss(y, y, quantile=0.1) == 0.0
    assert pinball_loss(y, y, quantile=0.9) == 0.0


def test_interval_coverage_all_inside():
    y_true = np.array([5.0, 10.0, 15.0])
    lower = np.array([0.0, 0.0, 0.0])
    upper = np.array([20.0, 20.0, 20.0])
    assert interval_coverage(y_true, lower, upper) == 1.0


def test_interval_coverage_none_inside():
    y_true = np.array([5.0, 10.0, 15.0])
    lower = np.array([100.0, 100.0, 100.0])
    upper = np.array([200.0, 200.0, 200.0])
    assert interval_coverage(y_true, lower, upper) == 0.0
