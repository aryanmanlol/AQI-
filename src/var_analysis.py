"""
VAR Analysis Module – Granger causality, Impulse Response Functions (IRF), and FEVD.

Includes:
  - Pairwise Granger causality p-value matrix with heatmap
  - Orthogonalized IRF with bootstrapped confidence bands
  - Forecast Error Variance Decomposition (FEVD) for PM2.5
  - Epistemological note: Granger causality is predictive precedence, not true causality.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

from src.config import FIGURES, TABLES

logger = logging.getLogger(__name__)


def compute_granger_causality_matrix(
    df: pd.DataFrame,
    variables: List[str],
    maxlag: int = 4,
    test: str = "ssr_ftest",
) -> pd.DataFrame:
    """Compute pairwise Granger causality p-value matrix.

    IMPORTANT NOTE:
      Granger causality measures PREDICTIVE PRECEDENCE (information containment),
      not true physical, mechanistic, or etiological causality.

    Parameters
    ----------
    df : pd.DataFrame
        Stationary series.
    variables : list of str
        Variables to test.
    maxlag : int
        Maximum lag order to test.
    test : str
        Test type: 'ssr_ftest', 'ssr_chi2test', etc.

    Returns
    -------
    pd.DataFrame
        Matrix where cell (row, col) contains p-value of col Granger-causing row.
    """
    clean_df = df[variables].dropna()
    n_vars = len(variables)
    matrix = pd.DataFrame(np.ones((n_vars, n_vars)), index=variables, columns=variables)

    for c in variables:
        for r in variables:
            if c == r:
                matrix.loc[r, c] = 1.0
                continue
            try:
                # grangercausalitytests(df[[y, x]]) tests whether x Granger-causes y
                res = grangercausalitytests(clean_df[[r, c]], maxlag=maxlag)
                # Take minimum p-value across tested lags
                p_values = [round(res[i + 1][0][test][1], 4) for i in range(maxlag)]
                matrix.loc[r, c] = min(p_values)
            except Exception as exc:
                logger.warning("Granger test failed for %s -> %s: %s", c, r, exc)
                matrix.loc[r, c] = np.nan

    logger.info("Granger Causality Matrix calculated (predictive precedence, not true causality).")
    return matrix


def plot_granger_matrix(
    matrix: pd.DataFrame,
    title: str = "Pairwise Granger Causality Matrix (p-values)",
    is_synthetic: bool = False,
) -> None:
    """Plot and save Granger causality p-value heatmap.

    Parameters
    ----------
    matrix : pd.DataFrame
        P-value matrix.
    title : str
        Plot title.
    is_synthetic : bool
        Synthetic data flag.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    t = f"{'[SYNTHETIC] ' if is_synthetic else ''}{title}\n(H0: Column does NOT Granger-cause Row; p < 0.05 rejects H0)"
    ax.set_title(t, fontsize=10, fontweight="bold")

    im = ax.imshow(matrix.values, cmap="YlGnBu_r", vmin=0.0, vmax=0.1, aspect="auto")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_yticks(range(len(matrix.index)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(matrix.index, fontsize=9)
    ax.set_xlabel("Potential Cause (Predictor / Past Lags)", fontweight="bold")
    ax.set_ylabel("Effect (Target Variable)", fontweight="bold")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("p-value (<= 0.05 indicates predictive Granger causality)")

    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            val = matrix.iloc[i, j]
            text = f"{val:.3f}" if not np.isnan(val) else "NA"
            color = "white" if val < 0.05 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8)

    plt.tight_layout()
    fig_path = FIGURES / "granger_causality_heatmap.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    logger.info("Saved Granger heatmap to %s", fig_path)


def compute_and_plot_irf(
    var_results: Any,
    periods: int = 14,
    response_var: str = "log_pm2_5",
    impulse_vars: Optional[List[str]] = None,
    is_synthetic: bool = False,
) -> None:
    """Compute orthogonalized IRF with bootstrapped confidence bands.

    Parameters
    ----------
    var_results : statsmodels VARResults
        Fitted VAR model.
    periods : int
        Number of forward horizons for IRF.
    response_var : str
        Response target.
    impulse_vars : list of str, optional
        Impulse variables (e.g., wind speed, temperature, PM10).
    is_synthetic : bool
        Synthetic flag.
    """
    names = list(var_results.names)
    if response_var not in names:
        if len(names) > 0:
            response_var = names[0]
        else:
            return

    if impulse_vars is None:
        # Default impulses of key physical interest
        candidates = ["wind_speed_10m", "temperature_2m", "relative_humidity_2m", "log_pm10"]
        impulse_vars = [c for c in candidates if c in names]
        if not impulse_vars:
            impulse_vars = [n for n in names if n != response_var][:3]

    irf = var_results.irf(periods)

    fig, axes = plt.subplots(len(impulse_vars), 1, figsize=(10, 3.5 * len(impulse_vars)), sharex=True)
    if len(impulse_vars) == 1:
        axes = [axes]

    t = f"{'[SYNTHETIC] ' if is_synthetic else ''}Orthogonalized IRF: Response of {response_var} (95% CI)"
    fig.suptitle(t, fontsize=12, fontweight="bold")

    resp_idx = names.index(response_var)

    for ax, imp_var in zip(axes, impulse_vars):
        imp_idx = names.index(imp_var)
        # Extract IRF values: shape (periods+1, k_vars, k_vars)
        response_vals = irf.irfs[:, resp_idx, imp_idx]
        stderr = irf.stderr()[:, resp_idx, imp_idx] if hasattr(irf, "stderr") else np.zeros_like(response_vals)

        h_range = np.arange(periods + 1)
        ax.plot(h_range, response_vals, color="#1f77b4", lw=2, label="Impulse Response")
        ax.axhline(0, color="black", linestyle="--", lw=0.8)

        # 95% Confidence Band (analytical / asymptotic)
        ax.fill_between(
            h_range,
            response_vals - 1.96 * stderr,
            response_vals + 1.96 * stderr,
            color="#1f77b4",
            alpha=0.25,
            label="95% Confidence Interval",
        )

        ax.set_title(f"Shock in {imp_var} -> Response in {response_var}", fontsize=10)
        ax.set_ylabel("Response")
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Horizon (Days)", fontweight="bold")
    plt.tight_layout()
    fig_path = FIGURES / f"irf_{response_var}.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    logger.info("Saved IRF plot to %s", fig_path)


def compute_and_plot_fevd(
    var_results: Any,
    periods: int = 14,
    target_var: str = "log_pm2_5",
    is_synthetic: bool = False,
) -> pd.DataFrame:
    """Compute and plot Forecast Error Variance Decomposition (FEVD).

    Parameters
    ----------
    var_results : statsmodels VARResults
        Fitted VAR model.
    periods : int
        Horizons.
    target_var : str
        Variable to decompose.
    is_synthetic : bool
        Synthetic flag.

    Returns
    -------
    pd.DataFrame
        FEVD percentage breakdown table.
    """
    names = list(var_results.names)
    if target_var not in names:
        target_var = names[0]

    fevd = var_results.fevd(periods)
    target_idx = names.index(target_var)

    # Decomp values: shape (periods, k_vars)
    decomp_matrix = fevd.decomp[target_idx, :, :] * 100.0  # Percentage

    fevd_df = pd.DataFrame(decomp_matrix, index=np.arange(1, periods + 1), columns=names)
    fevd_df.index.name = "Horizon"
    fevd_df.to_csv(TABLES / f"fevd_{target_var}.csv")

    # Plot stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    t = f"{'[SYNTHETIC] ' if is_synthetic else ''}Forecast Error Variance Decomposition (FEVD) of {target_var}"
    ax.set_title(t, fontweight="bold")

    bottom = np.zeros(periods)
    cmap = plt.get_cmap("tab10")

    for i, col in enumerate(names):
        vals = fevd_df[col].values
        ax.bar(fevd_df.index, vals, bottom=bottom, label=col, color=cmap(i % 10), alpha=0.85)
        bottom += vals

    ax.set_ylim(0, 100)
    ax.set_xlabel("Forecast Horizon (Days)", fontweight="bold")
    ax.set_ylabel("Variance Contribution (%)", fontweight="bold")
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=9)

    plt.tight_layout()
    fig_path = FIGURES / f"fevd_{target_var}.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    logger.info("Saved FEVD plot to %s", fig_path)

    return fevd_df


def run_var_analysis(
    var_results: Any,
    df_diff: pd.DataFrame,
    cfg: Dict[str, Any],
    is_synthetic: bool = False,
) -> Dict[str, Any]:
    """Execute complete VAR structural analysis suite.

    Parameters
    ----------
    var_results : VARResults
        Fitted VAR model.
    df_diff : pd.DataFrame
        Differenced stationary data.
    cfg : dict
        Project configuration.
    is_synthetic : bool
        Synthetic flag.

    Returns
    -------
    dict
        Granger matrix, IRF, and FEVD tables.
    """
    logger.info("=== Executing Structural VAR Analysis ===")
    names = list(var_results.names)

    # 1. Granger causality
    granger_matrix = compute_granger_causality_matrix(df_diff, names, maxlag=min(4, var_results.k_ar))
    granger_matrix.to_csv(TABLES / "granger_causality_matrix.csv")
    plot_granger_matrix(granger_matrix, is_synthetic=is_synthetic)

    # 2. IRF
    target_var = "log_pm2_5" if "log_pm2_5" in names else names[0]
    compute_and_plot_irf(var_results, periods=14, response_var=target_var, is_synthetic=is_synthetic)

    # 3. FEVD
    fevd_df = compute_and_plot_fevd(var_results, periods=14, target_var=target_var, is_synthetic=is_synthetic)

    return {
        "granger_matrix": granger_matrix,
        "fevd_df": fevd_df,
    }
