"""
Shared Plotting Helpers Module.

Provides standard plotting utilities for:
  - Fan charts with 80% and 95% bootstrapped prediction intervals
  - Publication-grade styled figures with synthetic data watermarks
  - Interactive Plotly figures for Streamlit
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.config import FIGURES

logger = logging.getLogger(__name__)


def plot_fan_chart_matplotlib(
    history: pd.Series,
    forecast: pd.Series,
    lower_80: pd.Series,
    upper_80: pd.Series,
    lower_95: pd.Series,
    upper_95: pd.Series,
    target_name: str = "PM2.5",
    model_name: str = "VAR",
    is_synthetic: bool = False,
    n_history_days: int = 30,
) -> Path:
    """Generate and save a static fan chart using Matplotlib.

    Parameters
    ----------
    history : pd.Series
        Recent historical observations.
    forecast : pd.Series
        Point forecast.
    lower_80, upper_80 : pd.Series
        80% prediction interval bounds.
    lower_95, upper_95 : pd.Series
        95% prediction interval bounds.
    target_name : str
        Variable name.
    model_name : str
        Model identifier.
    is_synthetic : bool
        Synthetic data flag.
    n_history_days : int
        Number of historical days to show on plot.

    Returns
    -------
    Path
        Path to saved figure.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    hist_tail = history.iloc[-n_history_days:]

    # History line
    ax.plot(hist_tail.index, hist_tail.values, color="black", lw=1.5, label="Observed History")

    # Connect history to forecast
    bridge_dates = [hist_tail.index[-1], forecast.index[0]]
    bridge_vals = [hist_tail.values[-1], forecast.values[0]]
    ax.plot(bridge_dates, bridge_vals, color="#1f77b4", linestyle="--", lw=1.2)

    # Point forecast
    ax.plot(forecast.index, forecast.values, color="#1f77b4", lw=2.0, marker="o", markersize=4, label=f"{model_name} Forecast")

    # 95% Interval Fan
    ax.fill_between(
        forecast.index,
        lower_95.values,
        upper_95.values,
        color="#1f77b4",
        alpha=0.18,
        label="95% Prediction Interval",
    )

    # 80% Interval Fan
    ax.fill_between(
        forecast.index,
        lower_80.values,
        upper_80.values,
        color="#1f77b4",
        alpha=0.32,
        label="80% Prediction Interval",
    )

    t = f"{'[SYNTHETIC] ' if is_synthetic else ''}{model_name} Fan Chart: {target_name} Forecast"
    ax.set_title(t, fontsize=12, fontweight="bold")
    ax.set_ylabel(f"{target_name} Concentration")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    plt.tight_layout()
    fig_path = FIGURES / f"fan_chart_{model_name}_{target_name}.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    return fig_path


def create_interactive_fan_chart_plotly(
    history: pd.Series,
    forecast: pd.Series,
    lower_80: pd.Series,
    upper_80: pd.Series,
    lower_95: pd.Series,
    upper_95: pd.Series,
    target_name: str = "PM2.5",
    model_name: str = "VAR",
    is_synthetic: bool = False,
    n_history_days: int = 45,
) -> go.Figure:
    """Construct an interactive Plotly fan chart for the Streamlit dashboard.

    Parameters
    ----------
    history : pd.Series
        Historical series.
    forecast : pd.Series
        Forecast.
    lower_80, upper_80 : pd.Series
        80% interval.
    lower_95, upper_95 : pd.Series
        95% interval.
    target_name : str
        Target variable.
    model_name : str
        Model name.
    is_synthetic : bool
        Synthetic flag.
    n_history_days : int
        Days of history.

    Returns
    -------
    go.Figure
        Interactive Plotly figure object.
    """
    hist_tail = history.iloc[-n_history_days:]
    fig = go.Figure()

    # Observed History
    fig.add_trace(go.Scatter(
        x=hist_tail.index,
        y=hist_tail.values,
        mode="lines",
        name="Observed History",
        line=dict(color="#2c3e50", width=2),
    ))

    # 95% Confidence Interval Band (fill down to lower_95)
    fig.add_trace(go.Scatter(
        x=forecast.index,
        y=upper_95.values,
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        name="Upper 95%",
    ))
    fig.add_trace(go.Scatter(
        x=forecast.index,
        y=lower_95.values,
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(41, 128, 185, 0.20)",
        name="95% Interval",
    ))

    # 80% Confidence Interval Band (fill down to lower_80)
    fig.add_trace(go.Scatter(
        x=forecast.index,
        y=upper_80.values,
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        name="Upper 80%",
    ))
    fig.add_trace(go.Scatter(
        x=forecast.index,
        y=lower_80.values,
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(41, 128, 185, 0.35)",
        name="80% Interval",
    ))

    # Point Forecast
    fig.add_trace(go.Scatter(
        x=forecast.index,
        y=forecast.values,
        mode="lines+markers",
        name=f"{model_name} Point Forecast",
        line=dict(color="#2980b9", width=2.5),
        marker=dict(size=6),
    ))

    title_text = f"{'[SYNTHETIC] ' if is_synthetic else ''}{model_name} Fan Chart: {target_name} Forecast with Uncertainty Bands"
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=15, weight="bold")),
        xaxis_title="Date",
        yaxis_title=f"{target_name} Level",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )

    return fig
