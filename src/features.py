"""
Feature Engineering Module – log transforms, differencing, lags, event flags.
"""
from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def add_log_transforms(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    eps: float = 1.0,
) -> pd.DataFrame:
    """Add log-transformed columns for specified variables.

    Uses log(x + eps) to handle zeros.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    cols : list of str, optional
        Columns to transform. Defaults to pollutant columns.
    eps : float
        Small constant added before log.

    Returns
    -------
    pd.DataFrame
        DataFrame with added log_ columns.
    """
    if cols is None:
        cols = ["pm2_5", "pm10", "nitrogen_dioxide", "ozone", "carbon_monoxide", "sulphur_dioxide"]
    cols = [c for c in cols if c in df.columns]

    for col in cols:
        log_col = f"log_{col.replace('nitrogen_dioxide', 'no2').replace('sulphur_dioxide', 'so2').replace('carbon_monoxide', 'co')}"
        # Simplify names
        name_map = {
            "log_pm2_5": "log_pm2_5",
            "log_pm10": "log_pm10",
            "log_no2": "log_no2",
            "log_so2": "log_so2",
            "log_co": "log_co",
            "log_ozone": "log_o3",
        }
        log_col = name_map.get(log_col, log_col)
        df[log_col] = np.log(df[col].clip(lower=eps) + eps)
        logger.info("Added %s = log(%s + %.1f)", log_col, col, eps)

    return df


def add_event_flags(
    df: pd.DataFrame, cfg: Dict[str, Any]
) -> pd.DataFrame:
    """Add event and calendar flags.

    Creates: diwali_window, holi_window, winter, monsoon, weekend columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with DatetimeIndex.
    cfg : dict
        Project config with events and season definitions.

    Returns
    -------
    pd.DataFrame
        DataFrame with added flag columns.
    """
    events = cfg.get("events", {})
    seasons = cfg.get("seasons", {})
    idx = df.index

    # Diwali window
    diwali_dates = [pd.Timestamp(d) for d in events.get("diwali", [])]
    before = events.get("diwali_window_before", 2)
    after = events.get("diwali_window_after", 5)
    diwali_mask = pd.Series(False, index=idx)
    for dd in diwali_dates:
        window_start = dd - timedelta(days=before)
        window_end = dd + timedelta(days=after)
        diwali_mask |= (idx >= window_start) & (idx <= window_end)
    df["diwali_window"] = diwali_mask.astype(int).values

    # Holi (single day flag)
    holi_dates = [pd.Timestamp(d) for d in events.get("holi", [])]
    holi_mask = pd.Series(False, index=idx)
    for hd in holi_dates:
        holi_mask |= (idx >= hd - timedelta(days=1)) & (idx <= hd + timedelta(days=1))
    df["holi_window"] = holi_mask.astype(int).values

    # Seasonal flags
    winter_months = seasons.get("winter", [11, 12, 1, 2])
    df["winter"] = idx.month.isin(winter_months).astype(int)

    monsoon_months = seasons.get("monsoon", [6, 7, 8, 9])
    df["monsoon"] = idx.month.isin(monsoon_months).astype(int)

    # Weekend
    df["weekend"] = (idx.dayofweek >= 5).astype(int)

    # Day of week (for ML features)
    df["day_of_week"] = idx.dayofweek
    df["month"] = idx.month
    df["day_of_year"] = idx.dayofyear

    logger.info(
        "Event flags added: diwali_window=%d, winter=%d, monsoon=%d, weekend=%d days",
        df["diwali_window"].sum(),
        df["winter"].sum(),
        df["monsoon"].sum(),
        df["weekend"].sum(),
    )
    return df


def add_lag_features(
    df: pd.DataFrame,
    cols: List[str],
    lags: List[int],
) -> pd.DataFrame:
    """Add lagged features for the specified columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    cols : list of str
        Columns to lag.
    lags : list of int
        Lag values.

    Returns
    -------
    pd.DataFrame
        DataFrame with added lag columns.
    """
    for col in cols:
        if col not in df.columns:
            continue
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    cols: List[str],
    windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Add rolling mean and std features.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    cols : list of str
        Columns to compute rolling stats for.
    windows : list of int, optional
        Window sizes. Defaults to [7, 14, 30].

    Returns
    -------
    pd.DataFrame
        DataFrame with added rolling features.
    """
    if windows is None:
        windows = [7, 14, 30]

    for col in cols:
        if col not in df.columns:
            continue
        for w in windows:
            df[f"{col}_rmean{w}"] = df[col].rolling(w, min_periods=1).mean()
            df[f"{col}_rstd{w}"] = df[col].rolling(w, min_periods=1).std()
    return df


def prepare_features(
    df: pd.DataFrame, cfg: Dict[str, Any]
) -> pd.DataFrame:
    """Full feature engineering pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned daily data.
    cfg : dict
        Project config.

    Returns
    -------
    pd.DataFrame
        Feature-rich DataFrame.
    """
    logger.info("=== Feature engineering ===")

    # Log transforms for pollutants
    pollutant_cols = [c for c in cfg["pollutant_vars"] if c in df.columns]
    df = add_log_transforms(df, cols=pollutant_cols)

    # Event and calendar flags
    df = add_event_flags(df, cfg)

    # Save featured data
    from src.config import DATA_PROCESSED
    city = cfg.get("default_city", "Delhi")
    df.to_parquet(DATA_PROCESSED / f"featured_{city}.parquet")
    logger.info("Featured data: %d rows × %d cols", *df.shape)

    return df
