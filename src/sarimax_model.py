"""
SARIMAX Model – SARIMA with weather and event exogenous regressors.

For forecasting, exogenous variables are handled honestly:
  - Weather variables are lagged by 1 day (use yesterday's weather to predict today)
  - Event flags (diwali, monsoon, winter, weekend) are known in advance (calendar)
  - This avoids leaking future actuals into the forecast
"""
from __future__ import annotations

import logging
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.baselines import sarima_aic_grid_search
from src.config import MODELS, TABLES

logger = logging.getLogger(__name__)


def prepare_exog(
    df: pd.DataFrame,
    weather_cols: List[str],
    event_cols: List[str],
    lag_weather: int = 1,
) -> pd.DataFrame:
    """Prepare exogenous variables with appropriate lagging.

    Weather variables are lagged to avoid future leakage.
    Calendar/event flags are used as-is (known in advance).

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset.
    weather_cols : list of str
        Weather variable names.
    event_cols : list of str
        Event/calendar flag names.
    lag_weather : int
        Lag to apply to weather variables.

    Returns
    -------
    pd.DataFrame
        Exogenous regressor matrix.
    """
    exog_parts = []

    for col in weather_cols:
        if col in df.columns:
            lagged = df[col].shift(lag_weather)
            lagged.name = f"{col}_lag{lag_weather}"
            exog_parts.append(lagged)

    for col in event_cols:
        if col in df.columns:
            exog_parts.append(df[col])

    if not exog_parts:
        return pd.DataFrame(index=df.index)

    exog = pd.concat(exog_parts, axis=1)
    return exog


def fit_sarimax(
    train: pd.Series,
    exog_train: pd.DataFrame,
    cfg: Dict[str, Any],
    period: int = 7,
) -> Tuple[Optional[Any], Optional[Tuple], float]:
    """Fit SARIMAX with AIC grid search.

    Parameters
    ----------
    train : pd.Series
        Target training series.
    exog_train : pd.DataFrame
        Exogenous training data.
    cfg : dict
        Config with sarima_grid.
    period : int
        Seasonal period.

    Returns
    -------
    (model, order, aic) : tuple
    """
    grid = cfg.get("sarima_grid", {})

    # Clean exog – drop columns that are all NaN
    exog_clean = exog_train.dropna(axis=1, how="all")
    # Align and forward-fill any remaining NaN
    common_idx = train.dropna().index.intersection(exog_clean.dropna().index)
    if len(common_idx) < 30:
        logger.warning("Not enough aligned data for SARIMAX")
        return None, None, np.inf

    train_aligned = train.loc[common_idx]
    exog_aligned = exog_clean.loc[common_idx].ffill().bfill()

    best_aic = np.inf
    best_order = None
    best_model = None

    for p, d, q in [(1, 1, 1), (2, 1, 1), (1, 1, 2), (0, 1, 1), (1, 0, 1), (2, 1, 2)]:
        for P, D, Q in [(0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 0, 1), (1, 1, 0), (0, 1, 1)]:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = SARIMAX(
                        train_aligned,
                        exog=exog_aligned,
                        order=(p, d, q),
                        seasonal_order=(P, D, Q, period),
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit(disp=False, maxiter=200)
                if model.aic < best_aic:
                    best_aic = model.aic
                    best_order = ((p, d, q), (P, D, Q, period))
                    best_model = model
            except Exception:
                continue

    if best_model is not None:
        logger.info(
            "SARIMAX best: order=%s seasonal=%s AIC=%.1f",
            best_order[0], best_order[1], best_aic,
        )

    return best_model, best_order, best_aic


def forecast_sarimax(
    model: Any,
    horizon: int,
    exog_future: pd.DataFrame,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Generate forecasts and prediction intervals from fitted SARIMAX.

    Parameters
    ----------
    model : SARIMAX results
        Fitted model.
    horizon : int
        Forecast horizon.
    exog_future : pd.DataFrame
        Future exogenous data.

    Returns
    -------
    (forecast, conf_int) : tuple
        Point forecasts and confidence intervals.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Align exog columns
        exog = exog_future.iloc[:horizon]
        exog = exog.ffill().bfill()

        forecast_result = model.get_forecast(steps=horizon, exog=exog)
        mean = forecast_result.predicted_mean
        ci = forecast_result.conf_int(alpha=0.05)

    return mean, ci


def run_sarimax(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    target: str = "pm2_5",
    train_end: Optional[pd.Timestamp] = None,
    horizon: int = 7,
) -> Dict[str, Any]:
    """Run SARIMAX pipeline for a single target variable.

    Parameters
    ----------
    df : pd.DataFrame
        Full featured data.
    cfg : dict
        Config.
    target : str
        Target variable.
    train_end : pd.Timestamp, optional
        Training cutoff.
    horizon : int
        Forecast horizon.

    Returns
    -------
    dict
        Results including model, forecasts, residuals.
    """
    logger.info("=== SARIMAX for %s ===", target)

    if target not in df.columns:
        logger.warning("Target %s not in data", target)
        return {}

    period = cfg.get("seasonal_period_daily", 7)

    # Prepare exog
    weather_cols = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m",
                    "precipitation", "surface_pressure"]
    event_cols = ["diwali_window", "winter", "monsoon", "weekend"]
    exog = prepare_exog(df, weather_cols, event_cols, lag_weather=1)

    # Split
    if train_end is None:
        n = len(df)
        train_end = df.index[int(n * cfg.get("eval_initial_train_frac", 0.7))]

    train = df[target].loc[:train_end]
    exog_train = exog.loc[:train_end]

    # Fit
    model, order, aic = fit_sarimax(train, exog_train, cfg, period)

    if model is None:
        logger.warning("SARIMAX fitting failed for %s", target)
        return {}

    # Save model
    model_path = MODELS / f"sarimax_{target}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "order": order, "aic": aic}, f)
    logger.info("SARIMAX model saved to %s", model_path)

    # Forecast
    test_start = train_end + pd.Timedelta(days=1)
    exog_future = exog.loc[test_start:]
    if len(exog_future) < horizon:
        # Extend with last known values
        last_row = exog.iloc[-1:].copy()
        extra_idx = pd.date_range(exog.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
        extra = pd.DataFrame(np.tile(last_row.values, (len(extra_idx), 1)),
                            index=extra_idx, columns=exog.columns)
        exog_future = pd.concat([exog_future, extra]).iloc[:horizon]

    forecast, conf_int = forecast_sarimax(model, horizon, exog_future)

    # Residuals
    residuals = model.resid

    result = {
        "model": model,
        "order": order,
        "aic": aic,
        "forecast": forecast,
        "conf_int": conf_int,
        "residuals": residuals,
        "train_end": train_end,
    }

    return result
