"""
Uncertainty Quantification Module.

Implements:
  1. Residual Bootstrap for VAR prediction intervals
  2. Moving Block Bootstrap (MBB) with Bagged ETS (STL decomposition -> MBB on residuals -> ETS -> ensemble)
  3. Empirical coverage evaluation for 80% and 95% prediction intervals
  4. Mean prediction interval width metrics
"""
from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL

from src.config import TABLES

logger = logging.getLogger(__name__)


def residual_bootstrap_var(
    var_results: Any,
    steps: int,
    n_bootstrap: int = 100,
    seed: int = 42,
    alpha_levels: Tuple[float, float] = (0.20, 0.05),
) -> Dict[str, np.ndarray]:
    """Generate bootstrapped prediction intervals for VAR by resampling fitted residuals.

    Parameters
    ----------
    var_results : statsmodels VARResults
        Fitted VAR model.
    steps : int
        Horizon.
    n_bootstrap : int
        Number of bootstrap replications.
    seed : int
        Random seed.
    alpha_levels : tuple of float
        Significance levels (e.g. 0.20 for 80% CI, 0.05 for 95% CI).

    Returns
    -------
    dict
        Point forecast, lower/upper bounds for 80% and 95% intervals across all variables.
    """
    rng = np.random.default_rng(seed)
    resid = var_results.resid.values  # Shape: (T - p, k)
    k_ar = var_results.k_ar
    coefs = var_results.coefs  # Shape: (p, k, k)
    intercept = var_results.intercept  # Shape: (k,)
    y_hist = var_results.endog[-k_ar:]  # Shape: (p, k)
    k_vars = var_results.k_ar_orig if hasattr(var_results, "k_ar_orig") else var_results.neqs

    bootstrap_forecasts = np.zeros((n_bootstrap, steps, k_vars))

    for b in range(n_bootstrap):
        # Sample residuals with replacement
        sample_idx = rng.integers(0, len(resid), size=steps)
        resampled_noise = resid[sample_idx]  # Shape: (steps, k)

        sim_y = list(y_hist)  # history buffer

        for h in range(steps):
            pred_h = intercept.copy()
            for lag in range(1, k_ar + 1):
                pred_h += coefs[lag - 1] @ sim_y[-lag]
            pred_h += resampled_noise[h]
            sim_y.append(pred_h)
            bootstrap_forecasts[b, h, :] = pred_h

    # Calculate percentiles
    p10 = np.percentile(bootstrap_forecasts, 10, axis=0)
    p90 = np.percentile(bootstrap_forecasts, 90, axis=0)
    p2_5 = np.percentile(bootstrap_forecasts, 2.5, axis=0)
    p97_5 = np.percentile(bootstrap_forecasts, 97.5, axis=0)
    median_pred = np.median(bootstrap_forecasts, axis=0)

    logger.info("Generated %d residual bootstrap replications for VAR (steps=%d).", n_bootstrap, steps)
    return {
        "median": median_pred,
        "lower_80": p10,
        "upper_80": p90,
        "lower_95": p2_5,
        "upper_95": p97_5,
        "all_replications": bootstrap_forecasts,
    }


def moving_block_bootstrap(
    series: np.ndarray,
    block_length: int = 14,
    n_samples: int = 100,
    seed: int = 42,
) -> np.ndarray:
    """Perform Moving Block Bootstrap on a 1D stationary remainder series.

    Parameters
    ----------
    series : 1D array
        Residual series to resample.
    block_length : int
        Length of continuous blocks.
    n_samples : int
        Number of bootstrapped series to generate.
    seed : int
        RNG seed.

    Returns
    -------
    np.ndarray
        Array of shape (n_samples, len(series)).
    """
    rng = np.random.default_rng(seed)
    n = len(series)
    num_blocks = int(np.ceil(n / block_length))

    bootstrapped = np.zeros((n_samples, n))
    max_start = n - block_length

    if max_start <= 0:
        # Fallback to simple i.i.d. bootstrap if series is shorter than block
        for b in range(n_samples):
            bootstrapped[b] = rng.choice(series, size=n, replace=True)
        return bootstrapped

    for b in range(n_samples):
        starts = rng.integers(0, max_start + 1, size=num_blocks)
        blocks = [series[s : s + block_length] for s in starts]
        concatenated = np.concatenate(blocks)[:n]
        bootstrapped[b] = concatenated

    return bootstrapped


def bagged_ets_forecast(
    train_series: pd.Series,
    horizon: int = 7,
    period: int = 7,
    n_bootstrap: int = 100,
    block_length: int = 14,
    seed: int = 42,
) -> Dict[str, Any]:
    """Bagged ETS with STL decomposition and Moving Block Bootstrap (Bergmeir et al. approach).

    1. Decompose series via STL: Y_t = Trend_t + Seasonal_t + Remainder_t
    2. Bootstrap Remainder using Moving Block Bootstrap
    3. Reconstruct synthetic series: Y*_t = Trend_t + Seasonal_t + Remainder*_t
    4. Fit ETS to each synthetic series and forecast
    5. Aggregate forecasts (mean/median) and obtain empirical prediction intervals

    Parameters
    ----------
    train_series : pd.Series
        Training series.
    horizon : int
        Forecast horizon.
    period : int
        Seasonal period.
    n_bootstrap : int
        Number of bootstrap replications.
    block_length : int
        MBB block size.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Point forecast, lower/upper 80% and 95% bounds.
    """
    clean_series = train_series.dropna()
    n = len(clean_series)

    # Step 1: STL decomposition
    try:
        stl = STL(clean_series, period=period, robust=True)
        res = stl.fit()
        trend = res.trend.values
        seasonal = res.seasonal.values
        remainder = res.resid.values
    except Exception as exc:
        logger.warning("STL in bagged ETS failed (%s). Falling back to direct Holt-Winters.", exc)
        from src.baselines import hw_ets_forecast
        point_fc, _ = hw_ets_forecast(clean_series, horizon, period)
        return {
            "mean": point_fc.values,
            "lower_80": point_fc.values * 0.85,
            "upper_80": point_fc.values * 1.15,
            "lower_95": point_fc.values * 0.70,
            "upper_95": point_fc.values * 1.30,
        }

    # Step 2: MBB on remainder
    boot_remainders = moving_block_bootstrap(remainder, block_length=block_length, n_samples=n_bootstrap, seed=seed)

    # Step 3 & 4: Reconstruct and fit ETS
    forecasts = np.zeros((n_bootstrap, horizon))

    for b in range(n_bootstrap):
        synth_y = trend + seasonal + boot_remainders[b]
        synth_series = pd.Series(synth_y, index=clean_series.index)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ExponentialSmoothing(
                    synth_series, trend="add", seasonal="add", seasonal_periods=period
                ).fit(optimized=True)
            forecasts[b] = model.forecast(horizon).values
        except Exception:
            # Fallback to last known value persistence
            forecasts[b] = np.repeat(clean_series.iloc[-1], horizon)

    # Step 5: Ensemble aggregation
    mean_forecast = np.mean(forecasts, axis=0)
    p10 = np.percentile(forecasts, 10, axis=0)
    p90 = np.percentile(forecasts, 90, axis=0)
    p2_5 = np.percentile(forecasts, 2.5, axis=0)
    p97_5 = np.percentile(forecasts, 97.5, axis=0)

    logger.info("Bagged ETS completed with %d bootstrap replications.", n_bootstrap)

    future_index = pd.date_range(clean_series.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    return {
        "mean": pd.Series(mean_forecast, index=future_index),
        "lower_80": pd.Series(p10, index=future_index),
        "upper_80": pd.Series(p90, index=future_index),
        "lower_95": pd.Series(p2_5, index=future_index),
        "upper_95": pd.Series(p97_5, index=future_index),
    }


def evaluate_interval_coverage(
    actuals: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> Tuple[float, float]:
    """Compute empirical coverage percentage and mean prediction interval width.

    Parameters
    ----------
    actuals : array-like
        True observed values.
    lower, upper : array-like
        Lower and upper prediction interval bounds.

    Returns
    -------
    (coverage_pct, mean_width) : tuple of floats
    """
    act = np.asarray(actuals)
    low = np.asarray(lower)
    up = np.asarray(upper)

    within = (act >= low) & (act <= up)
    coverage = float(np.mean(within) * 100.0)
    width = float(np.mean(up - low))

    return round(coverage, 2), round(width, 2)
