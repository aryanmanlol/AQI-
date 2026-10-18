"""
Prophet Forecasting Module (Optional).

Wraps Prophet imports in try/except to allow seamless execution when Prophet
is not installed. Integrates daily, weekly, and yearly seasonality, weather
regressors, and custom Diwali holiday windows.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import FIGURES, MODELS

logger = logging.getLogger(__name__)

# Try importing Prophet cleanly
PROPHET_AVAILABLE = False
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    try:
        from fbprophet import Prophet
        PROPHET_AVAILABLE = True
    except ImportError:
        logger.warning(
            "Prophet is not installed. Prophet modeling will be skipped cleanly. "
            "To enable, install via `pip install prophet`."
        )


def is_prophet_available() -> bool:
    """Return whether Prophet is installed in the current environment."""
    return PROPHET_AVAILABLE


def prepare_prophet_data(
    df: pd.DataFrame,
    target_col: str = "pm2_5",
    regressors: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Convert a time series DataFrame into Prophet's expected 'ds' and 'y' schema.

    Parameters
    ----------
    df : pd.DataFrame
        Data with DatetimeIndex.
    target_col : str
        Target pollutant column.
    regressors : list of str, optional
        Additional exogenous variables.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'ds', 'y', and regressors.
    """
    prophet_df = pd.DataFrame()
    prophet_df["ds"] = df.index
    prophet_df["y"] = df[target_col].values

    if regressors:
        for reg in regressors:
            if reg in df.columns:
                prophet_df[reg] = df[reg].values

    return prophet_df


def build_holidays_df(cfg: Dict[str, Any]) -> pd.DataFrame:
    """Construct Prophet holidays DataFrame from config events (Diwali, Holi).

    Parameters
    ----------
    cfg : dict
        Project config.

    Returns
    -------
    pd.DataFrame
        Holidays DataFrame with 'holiday', 'ds', 'lower_window', 'upper_window'.
    """
    events = cfg.get("events", {})
    records = []

    # Diwali window
    diwali_dates = events.get("diwali", [])
    before = events.get("diwali_window_before", 2)
    after = events.get("diwali_window_after", 5)
    for d in diwali_dates:
        records.append({
            "holiday": "diwali",
            "ds": pd.to_datetime(d),
            "lower_window": -before,
            "upper_window": after,
        })

    # Holi
    holi_dates = events.get("holi", [])
    for d in holi_dates:
        records.append({
            "holiday": "holi",
            "ds": pd.to_datetime(d),
            "lower_window": -1,
            "upper_window": 1,
        })

    return pd.DataFrame(records) if records else pd.DataFrame(columns=["holiday", "ds", "lower_window", "upper_window"])


def fit_prophet(
    train_df: pd.DataFrame,
    cfg: Dict[str, Any],
    regressors: Optional[List[str]] = None,
) -> Optional[Any]:
    """Initialize and fit a Prophet model.

    Parameters
    ----------
    train_df : pd.DataFrame
        Data prepared via prepare_prophet_data.
    cfg : dict
        Config.
    regressors : list of str, optional
        Exogenous regressors to register.

    Returns
    -------
    Fitted Prophet model or None if Prophet not available.
    """
    if not PROPHET_AVAILABLE:
        logger.warning("Prophet not available. Skipping model fit.")
        return None

    holidays = build_holidays_df(cfg)

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        holidays=holidays if not holidays.empty else None,
        interval_width=0.95,
    )

    if regressors:
        for reg in regressors:
            if reg in train_df.columns:
                model.add_regressor(reg)

    clean_train = train_df.dropna()
    model.fit(clean_train)
    logger.info("Fitted Prophet model with %d observations and %d regressors.",
                len(clean_train), len(regressors or []))
    return model


def forecast_prophet(
    model: Any,
    future_df: pd.DataFrame,
) -> pd.DataFrame:
    """Generate forecast from fitted Prophet model.

    Parameters
    ----------
    model : Prophet model
        Fitted model.
    future_df : pd.DataFrame
        Dataframe containing future 'ds' and regressor columns.

    Returns
    -------
    pd.DataFrame
        Prophet prediction dataframe with 'yhat', 'yhat_lower', 'yhat_upper'.
    """
    if model is None:
        return pd.DataFrame()

    forecast = model.predict(future_df)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]


def run_prophet_pipeline(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    target_col: str = "pm2_5",
    train_end: Optional[pd.Timestamp] = None,
    horizon: int = 7,
) -> Dict[str, Any]:
    """End-to-end Prophet workflow with clean skip if unavailable.

    Parameters
    ----------
    df : pd.DataFrame
        Full featured dataframe.
    cfg : dict
        Project configuration.
    target_col : str
        Target pollutant column.
    train_end : pd.Timestamp, optional
        Train split date.
    horizon : int
        Horizon days.

    Returns
    -------
    dict
        Model, forecasts, or empty dict if skipped.
    """
    if not PROPHET_AVAILABLE:
        logger.info("Prophet not installed: skipping Prophet pipeline cleanly.")
        return {"status": "skipped", "reason": "prophet_not_installed"}

    weather_regressors = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m"]
    available_regs = [r for r in weather_regressors if r in df.columns]

    prophet_df = prepare_prophet_data(df, target_col=target_col, regressors=available_regs)

    if train_end is None:
        n = len(df)
        train_end = df.index[int(n * cfg.get("eval_initial_train_frac", 0.7))]

    train_data = prophet_df[prophet_df["ds"] <= train_end]
    test_data = prophet_df[prophet_df["ds"] > train_end].iloc[:horizon]

    model = fit_prophet(train_data, cfg, regressors=available_regs)
    if model is None:
        return {"status": "failed"}

    # Build future df
    future_dates = pd.date_range(train_end + pd.Timedelta(days=1), periods=horizon, freq="D")
    future_df = pd.DataFrame({"ds": future_dates})

    # Forward fill regressors from test or train
    for reg in available_regs:
        if reg in test_data.columns and len(test_data) == horizon:
            future_df[reg] = test_data[reg].values
        else:
            future_df[reg] = train_data[reg].iloc[-1]

    pred = forecast_prophet(model, future_df)
    pred = pred.set_index("ds")

    return {
        "status": "success",
        "model": model,
        "forecast": pred["yhat"],
        "lower_ci": pred["yhat_lower"],
        "upper_ci": pred["yhat_upper"],
    }
