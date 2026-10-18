"""
Machine Learning Baseline Module (LightGBM).

Implements:
  - Lagged target and weather features + calendar features
  - Strict time-ordered splits (NO shuffling to prevent temporal data leakage)
  - Recursive multi-step forecasting strategy
"""
from __future__ import annotations

import logging
import pickle
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import MODELS, TABLES

logger = logging.getLogger(__name__)


def create_ml_feature_matrix(
    df: pd.DataFrame,
    target_col: str = "pm2_5",
    lags: Optional[List[int]] = None,
    weather_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Construct tabular feature matrix with lagged targets and exogenous weather/calendar regressors.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned, featured daily time series.
    target_col : str
        Target series name.
    lags : list of int, optional
        Lags to include. Defaults to [1, 2, 3, 7, 14].
    weather_cols : list of str, optional
        Weather columns.

    Returns
    -------
    (X, y) : tuple of DataFrame and Series
    """
    if lags is None:
        lags = [1, 2, 3, 7, 14]

    if weather_cols is None:
        weather_cols = [
            "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
            "precipitation", "surface_pressure",
        ]

    feat_df = pd.DataFrame(index=df.index)

    # 1. Target lags
    for lag in lags:
        feat_df[f"{target_col}_lag{lag}"] = df[target_col].shift(lag)

    # 2. Weather features (lagged by 1 day to prevent contemporaneous leakage)
    for col in weather_cols:
        if col in df.columns:
            feat_df[f"{col}_lag1"] = df[col].shift(1)

    # 3. Calendar & Event features (deterministic, known in advance)
    calendar_cols = ["diwali_window", "winter", "monsoon", "weekend", "day_of_week", "month"]
    for ccol in calendar_cols:
        if ccol in df.columns:
            feat_df[ccol] = df[ccol]

    # Target
    y = df[target_col]

    # Drop NaNs created by lagging
    valid_mask = ~feat_df.isna().any(axis=1) & ~y.isna()
    X_clean = feat_df.loc[valid_mask]
    y_clean = y.loc[valid_mask]

    return X_clean, y_clean


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    lgbm_params: Optional[Dict[str, Any]] = None,
) -> lgb.LGBMRegressor:
    """Train a LightGBM Regressor using strictly chronological data.

    Parameters
    ----------
    X_train : pd.DataFrame
        Features.
    y_train : pd.Series
        Target.
    lgbm_params : dict, optional
        Hyperparameters.

    Returns
    -------
    Fitted LGBMRegressor.
    """
    default_params = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 5,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 15,
        "random_state": 42,
        "verbose": -1,
    }
    if lgbm_params:
        default_params.update(lgbm_params)

    model = lgb.LGBMRegressor(**default_params)
    model.fit(X_train, y_train)
    logger.info("Fitted LightGBM model with %d samples and %d features.", len(X_train), X_train.shape[1])
    return model


def recursive_multistep_forecast(
    model: lgb.LGBMRegressor,
    history_df: pd.DataFrame,
    future_dates: pd.DatetimeIndex,
    target_col: str = "pm2_5",
    lags: Optional[List[int]] = None,
    weather_cols: Optional[List[str]] = None,
) -> pd.Series:
    """Generate recursive multi-step forecasts using LightGBM.

    At each step t+h, previous predictions are fed back into lag features.

    Parameters
    ----------
    model : LGBMRegressor
        Fitted model.
    history_df : pd.DataFrame
        Historical data up to t.
    future_dates : DatetimeIndex
        Dates to forecast.
    target_col : str
        Target series.
    lags : list of int, optional
        Target lags.
    weather_cols : list of str, optional
        Weather features.

    Returns
    -------
    pd.Series
        Forecasted values over future_dates.
    """
    if lags is None:
        lags = [1, 2, 3, 7, 14]

    sim_df = history_df.copy()
    predictions = []

    for step, f_date in enumerate(future_dates):
        # Build single row feature vector for date f_date
        row = {}
        for lag in lags:
            # Look backwards from f_date
            row[f"{target_col}_lag{lag}"] = sim_df[target_col].iloc[-lag]

        for wcol in (weather_cols or []):
            if wcol in sim_df.columns:
                row[f"{wcol}_lag1"] = sim_df[wcol].iloc[-1]

        # Calendar features
        row["diwali_window"] = int(sim_df["diwali_window"].iloc[-1]) if "diwali_window" in sim_df else 0
        row["winter"] = int(f_date.month in [11, 12, 1, 2])
        row["monsoon"] = int(f_date.month in [6, 7, 8, 9])
        row["weekend"] = int(f_date.dayofweek >= 5)
        row["day_of_week"] = f_date.dayofweek
        row["month"] = f_date.month

        # Convert to DataFrame matching model columns
        row_df = pd.DataFrame([row])
        # Reorder to match model feature_name_
        missing_cols = set(model.feature_name_) - set(row_df.columns)
        for mc in missing_cols:
            row_df[mc] = 0.0
        row_df = row_df[model.feature_name_]

        pred_val = float(model.predict(row_df)[0])
        pred_val = max(0.0, pred_val)  # AQI pollutants are non-negative
        predictions.append(pred_val)

        # Append prediction back into sim_df to serve as lag for subsequent steps
        new_row = sim_df.iloc[-1:].copy()
        new_row.index = [f_date]
        new_row[target_col] = pred_val
        sim_df = pd.concat([sim_df, new_row])

    return pd.Series(predictions, index=future_dates, name=f"{target_col}_lgbm_forecast")


def run_lightgbm_pipeline(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    target_col: str = "pm2_5",
    train_end: Optional[pd.Timestamp] = None,
    horizon: int = 7,
) -> Dict[str, Any]:
    """Execute complete LightGBM training and recursive forecasting pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset.
    cfg : dict
        Project config.
    target_col : str
        Target variable.
    train_end : pd.Timestamp, optional
        Train cutoff.
    horizon : int
        Steps.

    Returns
    -------
    dict
        Model, forecast series, feature importances.
    """
    logger.info("=== Running LightGBM Baseline Pipeline ===")
    lags = cfg.get("lgbm_lag_features", [1, 2, 3, 7, 14])
    lgbm_params = cfg.get("lgbm_params", {})

    X, y = create_ml_feature_matrix(df, target_col=target_col, lags=lags)

    if train_end is None:
        n = len(X)
        train_end = X.index[int(n * cfg.get("eval_initial_train_frac", 0.7))]

    train_mask = X.index <= train_end
    X_train, y_train = X.loc[train_mask], y.loc[train_mask]

    model = train_lightgbm(X_train, y_train, lgbm_params)

    # Feature Importance
    importance_df = pd.DataFrame({
        "feature": model.feature_name_,
        "importance_gain": model.booster_.feature_importance(importance_type="gain"),
        "importance_split": model.booster_.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False)
    importance_df.to_csv(TABLES / f"lgbm_feature_importance_{target_col}.csv", index=False)

    # Recursive forecast
    future_dates = pd.date_range(train_end + pd.Timedelta(days=1), periods=horizon, freq="D")
    forecast_series = recursive_multistep_forecast(
        model, df.loc[:train_end], future_dates, target_col=target_col, lags=lags
    )

    # Save model
    with open(MODELS / f"lgbm_{target_col}.pkl", "wb") as f:
        pickle.dump(model, f)

    return {
        "model": model,
        "forecast": forecast_series,
        "feature_importance": importance_df,
    }
