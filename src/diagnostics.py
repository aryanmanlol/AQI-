"""
Model Residual Diagnostics Module.

For EVERY fitted model, computes:
  - Residual ACF and Ljung-Box test (lags 10 and 20) for serial correlation
  - Jarque-Bera test for residual normality
  - ARCH-LM test for autoregressive conditional heteroskedasticity
  - Quantile-Quantile (QQ) normal probability plot
  - Aggregates results into outputs/tables/diagnostics_summary.csv with pass/fail flags.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

from src.config import FIGURES, TABLES

logger = logging.getLogger(__name__)


def compute_model_diagnostics(
    residuals: pd.Series | np.ndarray,
    model_name: str,
    target_var: str = "pm2_5",
    save_plot: bool = True,
    is_synthetic: bool = False,
) -> Dict[str, Any]:
    """Perform rigorous residual diagnostics on a single model's residuals.

    Parameters
    ----------
    residuals : Series or 1D array
        Model residuals.
    model_name : str
        Name/identifier of the model.
    target_var : str
        Target series name.
    save_plot : bool
        Whether to generate and save the QQ plot.
    is_synthetic : bool
        Synthetic data flag.

    Returns
    -------
    dict
        Diagnostic statistics, p-values, and pass/fail indicators.
    """
    clean_resid = pd.Series(residuals).dropna().values
    n = len(clean_resid)

    if n < 25:
        logger.warning("Too few residuals (%d) for comprehensive diagnostics on %s", n, model_name)
        return {"model": model_name, "target": target_var, "status": "insufficient_data"}

    # 1. Ljung-Box Test (lags 10 and 20)
    max_lag = min(20, n // 5)
    lb_df = acorr_ljungbox(clean_resid, lags=[min(10, max_lag), max_lag], return_df=True)
    p_lb_10 = float(lb_df["lb_pvalue"].iloc[0])
    p_lb_20 = float(lb_df["lb_pvalue"].iloc[-1])
    # Pass if p > 0.05 (no significant autocorrelation)
    pass_lb = bool(p_lb_20 > 0.05)

    # 2. Jarque-Bera Test (Normality)
    jb_res = stats.jarque_bera(clean_resid)
    jb_stat, jb_pvalue = float(jb_res.statistic), float(jb_res.pvalue)
    # Pass if p > 0.05 (normally distributed)
    pass_jb = bool(jb_pvalue > 0.05)

    # 3. ARCH-LM Test (Heteroskedasticity)
    try:
        arch_res = het_arch(clean_resid, nlags=min(5, n // 10))
        arch_lm_pvalue = float(arch_res[1])
        # Pass if p > 0.05 (homoskedastic residuals)
        pass_arch = bool(arch_lm_pvalue > 0.05)
    except Exception as exc:
        logger.warning("ARCH-LM test error for %s: %s", model_name, exc)
        arch_lm_pvalue = np.nan
        pass_arch = False

    # 4. Generate QQ-plot
    if save_plot:
        fig, ax = plt.subplots(figsize=(6, 5))
        stats.probplot(clean_resid, dist="norm", plot=ax)
        t = f"{'[SYNTHETIC] ' if is_synthetic else ''}Normal Q-Q Plot: {model_name} ({target_var})"
        ax.set_title(t, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        safe_name = f"qq_{model_name.replace(' ', '_').replace('/', '_')}_{target_var}"
        fig_path = FIGURES / f"{safe_name}.png"
        fig.savefig(fig_path, dpi=120)
        plt.close(fig)
        logger.info("Saved QQ plot to %s", fig_path)

    res_dict = {
        "model": model_name,
        "target": target_var,
        "n_obs": n,
        "mean_resid": round(float(np.mean(clean_resid)), 4),
        "std_resid": round(float(np.std(clean_resid)), 4),
        "ljung_box_p10": round(p_lb_10, 4),
        "ljung_box_p20": round(p_lb_20, 4),
        "no_autocorr_pass": pass_lb,
        "jarque_bera_pvalue": round(float(jb_pvalue), 4),
        "normality_pass": pass_jb,
        "arch_lm_pvalue": round(arch_lm_pvalue, 4) if not np.isnan(arch_lm_pvalue) else np.nan,
        "homoskedastic_pass": pass_arch,
        "overall_pass": bool(pass_lb and pass_jb and pass_arch),
    }

    return res_dict


def aggregate_diagnostics(diagnostics_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """Consolidate multiple model diagnostics into a single table.

    Parameters
    ----------
    diagnostics_list : list of dict
        List of single model diagnostic dictionaries.

    Returns
    -------
    pd.DataFrame
        Diagnostics summary table saved to outputs/tables/diagnostics_summary.csv.
    """
    df_diag = pd.DataFrame(diagnostics_list)
    out_path = TABLES / "diagnostics_summary.csv"
    df_diag.to_csv(out_path, index=False)
    logger.info("Saved consolidated diagnostics summary to %s", out_path)
    return df_diag
