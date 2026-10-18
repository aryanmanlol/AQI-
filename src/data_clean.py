"""
Data Cleaning Module – enforces regular index, handles missing data, caps outliers.

Steps:
  1. Enforce regular DatetimeIndex (daily or hourly)
  2. Align timezone to Asia/Kolkata
  3. Interpolate short gaps (limit from config)
  4. Flag long gaps (do not fill silently)
  5. Cap extreme outliers with IQR or z-score rule
  6. Save unmodified copy and cleaned version
  7. Sensitivity check: linear vs seasonal imputation for PM2.5
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.config import DATA_INTERIM, DATA_PROCESSED, TABLES

logger = logging.getLogger(__name__)


def enforce_regular_index(
    df: pd.DataFrame, freq: str = "D"
) -> pd.DataFrame:
    """Re-index DataFrame to a regular frequency, filling gaps with NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with DatetimeIndex.
    freq : str
        Pandas frequency string ('D' for daily, 'h' for hourly).

    Returns
    -------
    pd.DataFrame
        Re-indexed DataFrame.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")

    full_idx = pd.date_range(
        start=df.index.min(), end=df.index.max(), freq=freq, name="datetime"
    )
    result = df.reindex(full_idx)
    added = len(result) - len(df)
    if added > 0:
        logger.info("Added %d rows to enforce regular %s frequency", added, freq)
    return result


def align_timezone(
    df: pd.DataFrame, tz: str = "Asia/Kolkata"
) -> pd.DataFrame:
    """Localise or convert the index to the target timezone.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    tz : str
        Target timezone.

    Returns
    -------
    pd.DataFrame
        Timezone-aligned DataFrame (tz-naive after conversion).
    """
    if df.index.tz is None:
        # Assume UTC, convert to target
        df = df.tz_localize("UTC").tz_convert(tz).tz_localize(None)
        logger.info("Localized UTC → %s (then stripped tz info)", tz)
    elif str(df.index.tz) != tz:
        df = df.tz_convert(tz).tz_localize(None)
        logger.info("Converted %s → %s", df.index.tz, tz)
    else:
        df = df.tz_localize(None)
    return df


def interpolate_short_gaps(
    df: pd.DataFrame, limit: int = 3, method: str = "linear"
) -> pd.DataFrame:
    """Interpolate short gaps (up to `limit` consecutive NaNs).

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    limit : int
        Maximum consecutive NaNs to fill.
    method : str
        Interpolation method ('linear', 'time', etc.).

    Returns
    -------
    pd.DataFrame
        Interpolated DataFrame.
    """
    before = df.isna().sum().sum()
    result = df.interpolate(method=method, limit=limit)
    after = result.isna().sum().sum()
    filled = before - after
    logger.info(
        "Interpolated %d values (method=%s, limit=%d)", filled, method, limit
    )
    return result


def flag_long_gaps(
    df: pd.DataFrame, limit: int = 3
) -> pd.DataFrame:
    """Create a boolean mask DataFrame indicating long gaps (> limit).

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    limit : int
        Gap length threshold.

    Returns
    -------
    pd.DataFrame
        Boolean DataFrame: True where the value is in a long gap.
    """
    gap_flags = pd.DataFrame(False, index=df.index, columns=df.columns)
    for col in df.columns:
        is_na = df[col].isna()
        if not is_na.any():
            continue
        groups = (is_na != is_na.shift()).cumsum()
        gap_lengths = is_na.groupby(groups).transform("sum")
        gap_flags[col] = is_na & (gap_lengths > limit)

    total_flagged = gap_flags.sum().sum()
    if total_flagged > 0:
        logger.warning(
            "Flagged %d values in long gaps (> %d consecutive missing)",
            total_flagged,
            limit,
        )
    return gap_flags


def cap_outliers_iqr(
    df: pd.DataFrame, multiplier: float = 3.0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Cap outliers using the IQR method.

    Values below Q1 - multiplier*IQR or above Q3 + multiplier*IQR are clipped.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    multiplier : float
        IQR multiplier for defining outlier bounds.

    Returns
    -------
    (capped_df, outlier_report) : tuple
        Capped DataFrame and a report of outliers per column.
    """
    capped = df.copy()
    records = []
    for col in df.select_dtypes(include=[np.number]).columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr

        n_low = (df[col] < lower).sum()
        n_high = (df[col] > upper).sum()
        capped[col] = df[col].clip(lower=lower, upper=upper)

        if n_low + n_high > 0:
            records.append(
                {
                    "variable": col,
                    "lower_bound": round(lower, 2),
                    "upper_bound": round(upper, 2),
                    "n_capped_low": int(n_low),
                    "n_capped_high": int(n_high),
                }
            )
            logger.info(
                "  %-25s capped %d low, %d high (bounds: [%.1f, %.1f])",
                col,
                n_low,
                n_high,
                lower,
                upper,
            )

    report = pd.DataFrame(records) if records else pd.DataFrame()
    return capped, report


def cap_outliers_zscore(
    df: pd.DataFrame, threshold: float = 4.0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Cap outliers using the z-score method.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    threshold : float
        Z-score threshold.

    Returns
    -------
    (capped_df, outlier_report) : tuple
    """
    capped = df.copy()
    records = []
    for col in df.select_dtypes(include=[np.number]).columns:
        mean = df[col].mean()
        std = df[col].std()
        if std == 0 or np.isnan(std):
            continue
        z = np.abs((df[col] - mean) / std)
        n_outliers = (z > threshold).sum()
        lower = mean - threshold * std
        upper = mean + threshold * std
        capped[col] = df[col].clip(lower=lower, upper=upper)
        if n_outliers > 0:
            records.append(
                {
                    "variable": col,
                    "lower_bound": round(lower, 2),
                    "upper_bound": round(upper, 2),
                    "n_capped": int(n_outliers),
                }
            )
    report = pd.DataFrame(records) if records else pd.DataFrame()
    return capped, report


def sensitivity_imputation(
    df: pd.DataFrame, col: str = "pm2_5", n_points: int = 50, seed: int = 42
) -> pd.DataFrame:
    """Compare linear vs seasonal interpolation on a subset.

    Artificially removes n_points from the column, imputes with both methods,
    and reports MAE for each.

    Parameters
    ----------
    df : pd.DataFrame
        Input data (should have no NaN in the target column for a fair test).
    col : str
        Column to test.
    n_points : int
        Number of points to mask.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Comparison table.
    """
    if col not in df.columns:
        logger.warning("Column %s not found for imputation sensitivity check", col)
        return pd.DataFrame()

    clean = df[col].dropna()
    if len(clean) < n_points * 2:
        logger.warning("Not enough clean data for imputation sensitivity check")
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    mask_idx = rng.choice(clean.index, size=n_points, replace=False)
    true_vals = clean.loc[mask_idx].copy()

    masked = clean.copy()
    masked.loc[mask_idx] = np.nan

    # Linear
    linear_filled = masked.interpolate(method="linear")
    linear_mae = np.abs(linear_filled.loc[mask_idx] - true_vals).mean()

    # Seasonal (using spline as a proxy for periodic interpolation)
    seasonal_filled = masked.interpolate(method="spline", order=3)
    seasonal_mae = np.abs(seasonal_filled.loc[mask_idx] - true_vals).mean()

    result = pd.DataFrame(
        {
            "method": ["linear", "spline_order3"],
            "MAE": [round(linear_mae, 4), round(seasonal_mae, 4)],
        }
    )
    logger.info("Imputation sensitivity for %s:\n%s", col, result.to_string(index=False))
    return result


# ─── Main cleaning orchestrator ──────────────────────────────
def clean_data(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    city: str,
    freq: str = "D",
) -> pd.DataFrame:
    """Full cleaning pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data (hourly or daily).
    cfg : dict
        Project config.
    city : str
        City name.
    freq : str
        Target frequency ('D' or 'h').

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    logger.info("=== Cleaning data for %s (freq=%s) ===", city, freq)

    # Save unmodified copy
    raw_copy_path = DATA_INTERIM / f"unmodified_{city}_{freq}.parquet"
    df.to_parquet(raw_copy_path)
    logger.info("Unmodified copy saved to %s", raw_copy_path)

    # 1. Enforce regular index
    df = enforce_regular_index(df, freq=freq)

    # 2. Timezone alignment (already naive from fetch, but be safe)
    try:
        df = align_timezone(df)
    except Exception:
        pass  # Already tz-naive

    # 3. Interpolate short gaps
    limit = cfg.get("interpolation_limit_days", 3)
    if freq == "h":
        limit = limit * 24  # convert to hours
    df = interpolate_short_gaps(df, limit=limit)

    # 4. Flag long gaps
    gap_flags = flag_long_gaps(df, limit=limit)
    gap_flags.to_parquet(DATA_INTERIM / f"gap_flags_{city}_{freq}.parquet")

    # 5. Cap outliers
    method = cfg.get("outlier_method", "iqr")
    if method == "iqr":
        df, outlier_report = cap_outliers_iqr(
            df, multiplier=cfg.get("iqr_multiplier", 3.0)
        )
    else:
        df, outlier_report = cap_outliers_zscore(
            df, threshold=cfg.get("zscore_threshold", 4.0)
        )
    if not outlier_report.empty:
        outlier_report.to_csv(TABLES / f"outlier_report_{city}.csv", index=False)

    # 6. Imputation sensitivity check
    sensitivity = sensitivity_imputation(
        df, col="pm2_5", seed=cfg["project"]["random_seed"]
    )
    if not sensitivity.empty:
        sensitivity.to_csv(TABLES / f"imputation_sensitivity_{city}.csv", index=False)

    # 7. Save cleaned data
    interim_path = DATA_INTERIM / f"cleaned_{city}_{freq}.parquet"
    df.to_parquet(interim_path)
    logger.info("Cleaned data saved to %s", interim_path)

    processed_path = DATA_PROCESSED / f"processed_{city}_{freq}.parquet"
    df.to_parquet(processed_path)
    df.to_csv(DATA_PROCESSED / f"processed_{city}_{freq}.csv")
    logger.info("Processed data saved to %s", processed_path)

    return df
