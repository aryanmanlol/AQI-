"""
Exploratory Data Analysis – time plots, seasonal plots, ACF/PACF, STL, correlations.

All figures are saved to outputs/figures/.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL

from src.config import FIGURES

logger = logging.getLogger(__name__)

# Plotting defaults
plt.rcParams.update({
    "figure.figsize": (14, 5),
    "figure.dpi": 120,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})


def _save(fig: plt.Figure, name: str) -> None:
    """Save figure and close."""
    path = FIGURES / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    logger.info("Saved figure: %s", path)


def plot_time_series(
    df: pd.DataFrame,
    cols: List[str],
    title: str = "Time Series",
    prefix: str = "ts",
    is_synthetic: bool = False,
) -> None:
    """Plot time series for selected columns.

    Parameters
    ----------
    df : pd.DataFrame
        Data with DatetimeIndex.
    cols : list of str
        Columns to plot.
    title : str
        Plot title.
    prefix : str
        Filename prefix.
    is_synthetic : bool
        If True, stamp SYNTHETIC in the title.
    """
    actual_cols = [c for c in cols if c in df.columns]
    if not actual_cols:
        logger.warning("No columns found for time series plot")
        return

    n = len(actual_cols)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    t = f"[SYNTHETIC] {title}" if is_synthetic else title
    fig.suptitle(t, fontsize=14, fontweight="bold")

    for ax, col in zip(axes, actual_cols):
        ax.plot(df.index, df[col], linewidth=0.7)
        ax.set_ylabel(col)
        ax.set_title(col, fontsize=10)

    plt.tight_layout()
    _save(fig, f"{prefix}_time_series")


def plot_seasonal(
    df: pd.DataFrame,
    col: str,
    period: int = 7,
    is_synthetic: bool = False,
) -> None:
    """Seasonal subseries plot (box-plot by period unit).

    Parameters
    ----------
    df : pd.DataFrame
        Data with DatetimeIndex.
    col : str
        Column to plot.
    period : int
        Seasonal period (7=weekly for daily data).
    is_synthetic : bool
        Synthetic data flag.
    """
    if col not in df.columns:
        return

    series = df[col].dropna()
    if period == 7:
        group_key = series.index.dayofweek
        labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    elif period == 12:
        group_key = series.index.month
        labels = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ]
    else:
        group_key = np.arange(len(series)) % period
        labels = [str(i) for i in range(period)]

    grouped = series.groupby(group_key)
    data_list = [grouped.get_group(k).values for k in sorted(grouped.groups.keys())]

    fig, ax = plt.subplots(figsize=(10, 5))
    t = f"{'[SYNTHETIC] ' if is_synthetic else ''}Seasonal Subseries: {col}"
    ax.set_title(t, fontweight="bold")
    bp = ax.boxplot(data_list, patch_artist=True)
    ax.set_xticks(range(1, len(data_list) + 1))
    ax.set_xticklabels(labels[:len(data_list)])
    for patch in bp["boxes"]:
        patch.set_facecolor("#4C72B0")
        patch.set_alpha(0.6)
    ax.set_ylabel(col)
    _save(fig, f"seasonal_{col}")


def plot_monthly_seasonal(
    df: pd.DataFrame, col: str, is_synthetic: bool = False
) -> None:
    """Monthly box plot for annual seasonality.

    Parameters
    ----------
    df : pd.DataFrame
        Data with DatetimeIndex.
    col : str
        Column name.
    is_synthetic : bool
        Synthetic flag.
    """
    if col not in df.columns:
        return
    series = df[col].dropna()
    months = series.index.month
    data = [series[months == m].values for m in range(1, 13)]
    labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, ax = plt.subplots(figsize=(12, 5))
    t = f"{'[SYNTHETIC] ' if is_synthetic else ''}Monthly Seasonal: {col}"
    ax.set_title(t, fontweight="bold")
    bp = ax.boxplot(data, patch_artist=True)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    for patch in bp["boxes"]:
        patch.set_facecolor("#DD8452")
        patch.set_alpha(0.6)
    ax.set_ylabel(col)
    _save(fig, f"monthly_seasonal_{col}")


def plot_lag(
    df: pd.DataFrame, col: str, lags: int = 7, is_synthetic: bool = False
) -> None:
    """Lag plot (scatter of y(t) vs y(t-k)).

    Parameters
    ----------
    df : pd.DataFrame
        Data.
    col : str
        Column.
    lags : int
        Number of lag subplots.
    is_synthetic : bool
        Synthetic flag.
    """
    if col not in df.columns:
        return

    series = df[col].dropna()
    n_plots = min(lags, 6)
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    t = f"{'[SYNTHETIC] ' if is_synthetic else ''}Lag Plots: {col}"
    fig.suptitle(t, fontweight="bold")

    for i, ax in enumerate(axes, 1):
        ax.scatter(series.values[:-i], series.values[i:], alpha=0.3, s=5)
        ax.set_xlabel(f"y(t)")
        ax.set_ylabel(f"y(t-{i})")
        ax.set_title(f"Lag {i}")

    plt.tight_layout()
    _save(fig, f"lag_plot_{col}")


def plot_acf_pacf(
    df: pd.DataFrame, col: str, lags: int = 40, is_synthetic: bool = False
) -> None:
    """ACF and PACF plots.

    Parameters
    ----------
    df : pd.DataFrame
        Data.
    col : str
        Column.
    lags : int
        Number of lags.
    is_synthetic : bool
        Synthetic flag.
    """
    if col not in df.columns:
        return

    series = df[col].dropna()
    nlags = min(lags, len(series) // 2 - 1)
    if nlags < 5:
        logger.warning("Not enough data for ACF/PACF of %s", col)
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    t = f"{'[SYNTHETIC] ' if is_synthetic else ''}{col}"
    fig.suptitle(t, fontweight="bold")
    plot_acf(series, lags=nlags, ax=ax1, title=f"ACF – {col}")
    plot_pacf(series, lags=nlags, ax=ax2, title=f"PACF – {col}", method="ywm")
    plt.tight_layout()
    _save(fig, f"acf_pacf_{col}")


def plot_stl_decomposition(
    df: pd.DataFrame, col: str, period: int = 7, is_synthetic: bool = False
) -> Optional[Dict[str, float]]:
    """STL decomposition plot with trend and seasonal strength metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Data.
    col : str
        Column.
    period : int
        Seasonal period.
    is_synthetic : bool
        Synthetic flag.

    Returns
    -------
    dict or None
        Trend and seasonal strength metrics.
    """
    if col not in df.columns:
        return None

    series = df[col].dropna()
    if len(series) < 2 * period + 1:
        logger.warning("Not enough data for STL of %s", col)
        return None

    try:
        stl = STL(series, period=period, robust=True)
        result = stl.fit()
    except Exception as exc:
        logger.warning("STL failed for %s: %s", col, exc)
        return None

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    t = f"{'[SYNTHETIC] ' if is_synthetic else ''}STL Decomposition: {col}"
    fig.suptitle(t, fontweight="bold")

    axes[0].plot(series.index, series.values, linewidth=0.7)
    axes[0].set_ylabel("Observed")
    axes[1].plot(series.index, result.trend, linewidth=0.7, color="orange")
    axes[1].set_ylabel("Trend")
    axes[2].plot(series.index, result.seasonal, linewidth=0.7, color="green")
    axes[2].set_ylabel("Seasonal")
    axes[3].plot(series.index, result.resid, linewidth=0.7, color="red")
    axes[3].set_ylabel("Residual")

    plt.tight_layout()
    _save(fig, f"stl_{col}")

    # Strength metrics
    var_resid = np.nanvar(result.resid)
    var_detrended = np.nanvar(result.seasonal + result.resid)
    var_deseasoned = np.nanvar(result.trend + result.resid)

    trend_strength = max(0, 1 - var_resid / var_deseasoned) if var_deseasoned > 0 else 0
    seasonal_strength = max(0, 1 - var_resid / var_detrended) if var_detrended > 0 else 0

    logger.info(
        "  %s – trend strength: %.3f, seasonal strength: %.3f",
        col, trend_strength, seasonal_strength,
    )
    return {"trend_strength": trend_strength, "seasonal_strength": seasonal_strength}


def plot_correlation_heatmap(
    df: pd.DataFrame, cols: List[str], is_synthetic: bool = False
) -> None:
    """Correlation heatmap.

    Parameters
    ----------
    df : pd.DataFrame
        Data.
    cols : list of str
        Columns to include.
    is_synthetic : bool
        Synthetic flag.
    """
    actual_cols = [c for c in cols if c in df.columns]
    if len(actual_cols) < 2:
        return

    corr = df[actual_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    t = f"{'[SYNTHETIC] ' if is_synthetic else ''}Correlation Heatmap"
    ax.set_title(t, fontweight="bold")

    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(actual_cols)))
    ax.set_yticks(range(len(actual_cols)))
    ax.set_xticklabels(actual_cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(actual_cols, fontsize=8)
    fig.colorbar(im, ax=ax)

    # Annotate
    for i in range(len(actual_cols)):
        for j in range(len(actual_cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(corr.iloc[i, j]) > 0.5 else "black")

    plt.tight_layout()
    _save(fig, "correlation_heatmap")


def plot_cross_correlation(
    df: pd.DataFrame,
    target: str = "pm2_5",
    weather_cols: Optional[List[str]] = None,
    max_lag: int = 30,
    is_synthetic: bool = False,
) -> None:
    """Cross-correlation between target and weather variables.

    Parameters
    ----------
    df : pd.DataFrame
        Data.
    target : str
        Target variable.
    weather_cols : list of str, optional
        Weather variables.
    max_lag : int
        Maximum lag for cross-correlation.
    is_synthetic : bool
        Synthetic flag.
    """
    if weather_cols is None:
        weather_cols = ["temperature_2m", "relative_humidity_2m",
                        "wind_speed_10m", "precipitation", "surface_pressure"]
    weather_cols = [c for c in weather_cols if c in df.columns]

    if target not in df.columns or not weather_cols:
        return

    n = len(weather_cols)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    t = f"{'[SYNTHETIC] ' if is_synthetic else ''}Cross-Correlation: {target} vs Weather"
    fig.suptitle(t, fontweight="bold")

    target_series = df[target].dropna()
    for ax, wcol in zip(axes, weather_cols):
        weather_series = df[wcol].dropna()
        # Align
        common = target_series.index.intersection(weather_series.index)
        if len(common) < max_lag * 2:
            continue
        ts = target_series.loc[common].values
        ws = weather_series.loc[common].values
        ts = (ts - ts.mean()) / (ts.std() + 1e-10)
        ws = (ws - ws.mean()) / (ws.std() + 1e-10)

        ccf = np.correlate(ts, ws, mode="full")
        ccf = ccf / len(ts)
        lags_arr = np.arange(-len(ts) + 1, len(ts))
        mask = np.abs(lags_arr) <= max_lag
        ax.bar(lags_arr[mask], ccf[mask], width=0.8, alpha=0.7)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.axhline(1.96 / np.sqrt(len(ts)), color="red", linestyle="--", alpha=0.5)
        ax.axhline(-1.96 / np.sqrt(len(ts)), color="red", linestyle="--", alpha=0.5)
        ax.set_ylabel(wcol)
        ax.set_title(f"{target} × {wcol}")

    plt.tight_layout()
    _save(fig, f"cross_corr_{target}")


def run_eda(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    is_synthetic: bool = False,
) -> pd.DataFrame:
    """Run the full EDA pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned, featured daily data.
    cfg : dict
        Project config.
    is_synthetic : bool
        Synthetic data flag.

    Returns
    -------
    pd.DataFrame
        STL strength metrics.
    """
    logger.info("=== Exploratory Data Analysis ===")
    period = cfg.get("seasonal_period_daily", 7)

    # Pollutant columns present
    poll_cols = [c for c in cfg["pollutant_vars"] if c in df.columns]
    weather_cols = [c for c in cfg["weather_vars"] if c in df.columns]
    all_cols = poll_cols + weather_cols

    # 1. Time series plots
    plot_time_series(df, poll_cols, "Pollutant Time Series", "pollutants", is_synthetic)
    plot_time_series(df, weather_cols, "Weather Time Series", "weather", is_synthetic)

    # 2. Seasonal plots
    for col in poll_cols[:3]:  # Top 3 pollutants
        plot_seasonal(df, col, period=period, is_synthetic=is_synthetic)
        plot_monthly_seasonal(df, col, is_synthetic=is_synthetic)

    # 3. Lag plots
    for col in ["pm2_5", "pm10"]:
        plot_lag(df, col, lags=6, is_synthetic=is_synthetic)

    # 4. ACF/PACF
    for col in poll_cols[:4]:
        plot_acf_pacf(df, col, lags=40, is_synthetic=is_synthetic)

    # 5. STL decomposition
    stl_records = []
    for col in poll_cols + weather_cols[:3]:
        metrics = plot_stl_decomposition(df, col, period=period, is_synthetic=is_synthetic)
        if metrics:
            stl_records.append({"variable": col, **metrics})

    stl_df = pd.DataFrame(stl_records)
    if not stl_df.empty:
        from src.config import TABLES
        stl_df.to_csv(TABLES / "stl_strength.csv", index=False)

    # 6. Correlation heatmap
    plot_correlation_heatmap(df, all_cols, is_synthetic)

    # 7. Cross-correlation
    plot_cross_correlation(df, "pm2_5", weather_cols, is_synthetic=is_synthetic)

    logger.info("EDA complete – figures saved to %s", FIGURES)
    return stl_df
