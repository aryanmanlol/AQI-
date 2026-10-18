"""
Regime-Specific Analysis Module.

Analyzes atmospheric regimes:
  - Winter (high stagnation, severe pollution peaks)
  - Monsoon (wet scavenging, low pollutant concentrations)
  - Diwali Window (extreme episodic emission spikes)
Compares model forecasting errors and structural responses across regimes.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

from src.config import FIGURES, TABLES

logger = logging.getLogger(__name__)


def segment_regimes(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Partition dataset into Winter, Monsoon, and Diwali window subsets.

    Parameters
    ----------
    df : pd.DataFrame
        Featured dataset with 'winter', 'monsoon', 'diwali_window' flags.

    Returns
    -------
    dict of str to pd.DataFrame
    """
    regimes = {}
    if "winter" in df.columns:
        regimes["Winter"] = df[df["winter"] == 1]
    if "monsoon" in df.columns:
        regimes["Monsoon"] = df[df["monsoon"] == 1]
    if "diwali_window" in df.columns:
        regimes["Diwali_Window"] = df[df["diwali_window"] == 1]

    # Remainder / Normal
    other_mask = (df.get("winter", 0) == 0) & (df.get("monsoon", 0) == 0) & (df.get("diwali_window", 0) == 0)
    if other_mask.any():
        regimes["Transition_Season"] = df[other_mask]

    for name, r_df in regimes.items():
        logger.info("Regime '%s': %d observations", name, len(r_df))

    return regimes


def evaluate_regime_statistics(
    regimes: Dict[str, pd.DataFrame],
    target_var: str = "pm2_5",
) -> pd.DataFrame:
    """Compute summary statistics of pollutant levels and variability across atmospheric regimes.

    Parameters
    ----------
    regimes : dict of DataFrame
        Partitioned regimes.
    target_var : str
        Target pollutant.

    Returns
    -------
    pd.DataFrame
        Table summarizing mean, median, IQR, max, and std across regimes.
    """
    records = []
    for r_name, r_df in regimes.items():
        if target_var not in r_df.columns or len(r_df) == 0:
            continue
        vals = r_df[target_var].dropna()
        records.append({
            "Regime": r_name,
            "N_Obs": len(vals),
            "Mean": round(float(vals.mean()), 2),
            "Median": round(float(vals.median()), 2),
            "Std_Dev": round(float(vals.std()), 2),
            "IQR": round(float(vals.quantile(0.75) - vals.quantile(0.25)), 2),
            "Max": round(float(vals.max()), 2),
        })

    summary_df = pd.DataFrame(records)
    summary_df.to_csv(TABLES / "regime_comparison.csv", index=False)
    logger.info("Saved regime comparison table to %s", TABLES / "regime_comparison.csv")
    return summary_df


def plot_regime_boxplots(
    regimes: Dict[str, pd.DataFrame],
    target_var: str = "pm2_5",
    is_synthetic: bool = False,
) -> None:
    """Generate comparative boxplots for pollutants across environmental regimes.

    Parameters
    ----------
    regimes : dict of DataFrame
        Partitioned regimes.
    target_var : str
        Target pollutant.
    is_synthetic : bool
        Synthetic data flag.
    """
    data = []
    labels = []
    for r_name, r_df in regimes.items():
        if target_var in r_df.columns and len(r_df) > 0:
            data.append(r_df[target_var].dropna().values)
            labels.append(r_name)

    if not data:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    t = f"{'[SYNTHETIC] ' if is_synthetic else ''}Pollutant Distribution by Atmospheric Regime ({target_var})"
    ax.set_title(t, fontweight="bold")
    bp = ax.boxplot(data, patch_artist=True)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i % len(colors)])
        patch.set_alpha(0.7)

    ax.set_ylabel(f"{target_var} (µg/m³)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = FIGURES / f"regimes_boxplot_{target_var}.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    logger.info("Saved regime boxplot to %s", fig_path)


def run_regimes_analysis(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    target_var: str = "pm2_5",
    is_synthetic: bool = False,
) -> pd.DataFrame:
    """Run full atmospheric regime analysis workflow.

    Parameters
    ----------
    df : pd.DataFrame
        Featured daily data.
    cfg : dict
        Project config.
    target_var : str
        Target series.
    is_synthetic : bool
        Synthetic flag.

    Returns
    -------
    pd.DataFrame
        Regime statistics comparison table.
    """
    logger.info("=== Running Atmospheric Regime Analysis ===")
    regimes = segment_regimes(df)
    summary_df = evaluate_regime_statistics(regimes, target_var=target_var)
    plot_regime_boxplots(regimes, target_var=target_var, is_synthetic=is_synthetic)
    return summary_df
