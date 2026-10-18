"""
Baseline Models – Naive, Seasonal Naive, Holt-Winters ETS, ARIMA, SARIMA.

SARIMA uses a statsmodels-based AIC grid search (no pmdarima).
"""
from __future__ import annotations

import itertools
import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.config import MODELS, TABLES

logger = logging.getLogger(__name__)


def naive_forecast(
    train: pd.Series, horizon: int
) -> pd.Series:
    """Naive forecast: repeat the last observation.

    Parameters
    ----------
    train : pd.Series
        Training series.
    horizon : int
        Forecast horizon.

    Returns
    -------
    pd.Series
        Forecast series.
    """
    last_val = train.iloc[-1]
    idx = pd.date_range(
        train.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D"
    )
    return pd.Series(last_val, index=idx, name=train.name)


def seasonal_naive_forecast(
    train: pd.Series, horizon: int, period: int = 7
) -> pd.Series:
    """Seasonal naive: repeat the last season.

    Parameters
    ----------
    train : pd.Series
        Training series.
    horizon : int
        Forecast horizon.
    period : int
        Seasonal period.

    Returns
    -------
    pd.Series
        Forecast series.
    """
    last_season = train.iloc[-period:].values
    reps = (horizon // period) + 1
    forecast_vals = np.tile(last_season, reps)[:horizon]
    idx = pd.date_range(
        train.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D"
    )
    return pd.Series(forecast_vals, index=idx, name=train.name)


def hw_ets_forecast(
    train: pd.Series, horizon: int, period: int = 7
) -> Tuple[pd.Series, Optional[Any]]:
    """Holt-Winters Exponential Smoothing forecast.

    Parameters
    ----------
    train : pd.Series
        Training series.
    horizon : int
        Forecast horizon.
    period : int
        Seasonal period.

    Returns
    -------
    (forecast, model) : tuple
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(
                train,
                trend="add",
                seasonal="add",
                seasonal_periods=period,
                initialization_method="estimated",
            ).fit(optimized=True)
        fcast = model.forecast(horizon)
        return fcast, model
    except Exception as exc:
        logger.warning("Holt-Winters failed: %s. Falling back to additive trend only.", exc)
        try:
            model = ExponentialSmoothing(
                train, trend="add", seasonal=None, initialization_method="estimated"
            ).fit(optimized=True)
            return model.forecast(horizon), model
        except Exception as exc2:
            logger.error("Holt-Winters completely failed: %s", exc2)
            return naive_forecast(train, horizon), None


def sarima_aic_grid_search(
    train: pd.Series,
    p_range: List[int],
    d_range: List[int],
    q_range: List[int],
    P_range: List[int],
    D_range: List[int],
    Q_range: List[int],
    s: int = 7,
) -> Tuple[Tuple, float, Optional[Any]]:
    """Grid search over SARIMA orders by AIC.

    Parameters
    ----------
    train : pd.Series
        Training series.
    p_range, d_range, q_range : list of int
        Non-seasonal order ranges.
    P_range, D_range, Q_range : list of int
        Seasonal order ranges.
    s : int
        Seasonal period.

    Returns
    -------
    (best_order, best_aic, best_model) : tuple
    """
    best_aic = np.inf
    best_order = None
    best_model = None
    total = (
        len(p_range) * len(d_range) * len(q_range)
        * len(P_range) * len(D_range) * len(Q_range)
    )

    logger.info("SARIMA grid search: %d combinations (s=%d)", total, s)
    count = 0

    for p, d, q, P, D, Q in itertools.product(
        p_range, d_range, q_range, P_range, D_range, Q_range
    ):
        count += 1
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = SARIMAX(
                    train,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, s),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False, maxiter=200)

            if model.aic < best_aic:
                best_aic = model.aic
                best_order = ((p, d, q), (P, D, Q, s))
                best_model = model
        except Exception:
            continue

    if best_order:
        logger.info(
            "Best SARIMA: order=%s seasonal=%s AIC=%.1f (searched %d/%d)",
            best_order[0], best_order[1], best_aic, count, total,
        )
    else:
        logger.warning("No valid SARIMA model found in grid search")

    return best_order, best_aic, best_model


def run_baselines(
    df: pd.DataFrame,
    target_cols: List[str],
    cfg: Dict[str, Any],
    train_end: pd.Timestamp,
    horizons: List[int],
    is_synthetic: bool = False,
) -> pd.DataFrame:
    """Run all baseline models and collect forecasts/metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Featured data.
    target_cols : list of str
        Pollutant columns to forecast.
    cfg : dict
        Project config.
    train_end : pd.Timestamp
        End of training period.
    horizons : list of int
        Forecast horizons.
    is_synthetic : bool
        Synthetic data flag.

    Returns
    -------
    pd.DataFrame
        Baseline results summary.
    """
    logger.info("=== Running Baseline Models ===")
    period = cfg.get("seasonal_period_daily", 7)
    grid = cfg.get("sarima_grid", {})

    results = []
    for col in target_cols:
        if col not in df.columns:
            continue

        series = df[col].dropna()
        train = series.loc[:train_end]
        if len(train) < 30:
            logger.warning("Skipping %s: too few training points", col)
            continue

        for h in horizons:
            test_start = train_end + pd.Timedelta(days=1)
            test_end = test_start + pd.Timedelta(days=h - 1)
            test = series.loc[test_start:test_end]
            if len(test) < h:
                continue

            # Naive
            fcast_naive = naive_forecast(train, h)
            mae_naive = np.abs(fcast_naive.values[:len(test)] - test.values).mean()

            # Seasonal Naive
            fcast_sn = seasonal_naive_forecast(train, h, period)
            mae_sn = np.abs(fcast_sn.values[:len(test)] - test.values).mean()

            # HW ETS
            fcast_hw, _ = hw_ets_forecast(train, h, period)
            mae_hw = np.abs(fcast_hw.values[:len(test)] - test.values).mean()

            results.append({"variable": col, "horizon": h, "model": "Naive", "MAE": round(mae_naive, 3)})
            results.append({"variable": col, "horizon": h, "model": "SeasonalNaive", "MAE": round(mae_sn, 3)})
            results.append({"variable": col, "horizon": h, "model": "HW_ETS", "MAE": round(mae_hw, 3)})

            # SARIMA (only for h <= 7 and first few cols to save time)
            if h <= 7 and col in target_cols[:2]:
                order, aic, model = sarima_aic_grid_search(
                    train,
                    grid.get("p_range", [0, 1]),
                    grid.get("d_range", [0, 1]),
                    grid.get("q_range", [0, 1]),
                    grid.get("P_range", [0, 1]),
                    grid.get("D_range", [0, 1]),
                    grid.get("Q_range", [0, 1]),
                    s=period,
                )
                if model is not None:
                    fcast_sarima = model.forecast(h)
                    mae_sarima = np.abs(fcast_sarima.values[:len(test)] - test.values).mean()
                    results.append({
                        "variable": col, "horizon": h,
                        "model": f"SARIMA{order[0]}{order[1]}",
                        "MAE": round(mae_sarima, 3),
                    })

    results_df = pd.DataFrame(results)
    results_df.to_csv(TABLES / "baseline_results.csv", index=False)
    logger.info("Baseline results:\n%s", results_df.to_string(index=False))
    return results_df
