"""
Tests for Hand-Computed Evaluation Metrics.
"""
import numpy as np
import pytest
from src.evaluation import calc_mae, calc_mape, calc_mase, calc_rmse, diebold_mariano_test


def test_mae_hand_computed():
    """Verify MAE with known vector differences."""
    actual = np.array([10.0, 20.0, 30.0])
    pred = np.array([12.0, 18.0, 35.0])
    # Differences: |2|, |2|, |5| -> sum=9 -> mean=3.0
    assert calc_mae(actual, pred) == pytest.approx(3.0, abs=1e-5)


def test_rmse_hand_computed():
    """Verify RMSE with known square deviations."""
    actual = np.array([10.0, 20.0, 30.0])
    pred = np.array([13.0, 20.0, 26.0])
    # Differences: 3, 0, -4 -> squares: 9, 0, 16 -> mean=25/3 -> sqrt(8.3333)=2.88675
    assert calc_rmse(actual, pred) == pytest.approx(np.sqrt(25.0 / 3.0), abs=1e-5)


def test_mape_hand_computed():
    """Verify MAPE with zero-safe denominator."""
    actual = np.array([100.0, 200.0])
    pred = np.array([110.0, 180.0])
    # Absolute pct errors: 10/100 = 10%, 20/200 = 10% -> mean=10.0%
    assert calc_mape(actual, pred) == pytest.approx(10.0, abs=1e-5)


def test_mase_hand_computed():
    """Verify MASE scaled against in-sample seasonal naive."""
    # In-sample seasonal differences with s=2
    # in_sample: [10, 20, 12, 22, 14, 24]
    # diffs lag 2: |12-10|=2, |22-20|=2, |14-12|=2, |24-22|=2 -> mean scale = 2.0
    in_sample = np.array([10.0, 20.0, 12.0, 22.0, 14.0, 24.0])
    actual = np.array([16.0, 26.0])
    pred = np.array([18.0, 25.0])
    # Forecast errors: |18-16|=2, |25-26|=1 -> MAE = 1.5
    # MASE = 1.5 / 2.0 = 0.75
    mase = calc_mase(actual, pred, in_sample_train=in_sample, seasonality=2)
    assert mase == pytest.approx(0.75, abs=1e-5)


def test_diebold_mariano_identical_forecasts():
    """Identical forecast error sequences yield 0 test stat and p=1.0."""
    e1 = np.array([1.0, -1.0, 2.0, -2.0, 1.0])
    e2 = np.array([1.0, -1.0, 2.0, -2.0, 1.0])
    dm_stat, p_val = diebold_mariano_test(e1, e2, h=1)
    assert dm_stat == 0.0
    assert p_val == 1.0
