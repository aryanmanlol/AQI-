"""
VAR and VECM Modeling Module.

Handles:
  - Lag order selection via AIC, BIC, HQIC, FPE
  - VAR fitting on stationary system
  - Stability verification (eigenvalues/roots inside unit circle) and root plotting
  - Johansen cointegration test on I(1) system
  - VECM fitting when cointegration is detected
  - Multi-step forecasting with analytical/asymptotic intervals
"""
from __future__ import annotations

import logging
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen

from src.config import FIGURES, MODELS, TABLES

logger = logging.getLogger(__name__)


def select_var_lag(
    df_endo: pd.DataFrame,
    maxlags: int = 14,
    exog: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Select optimal lag order for VAR using AIC, BIC, HQIC, FPE.

    Parameters
    ----------
    df_endo : pd.DataFrame
        Endogenous variables (stationary).
    maxlags : int
        Maximum lag to test.
    exog : pd.DataFrame, optional
        Exogenous regressors.

    Returns
    -------
    dict
        Lag selection results and order recommendation table.
    """
    clean_df = df_endo.dropna()
    clean_exog = exog.loc[clean_df.index] if exog is not None else None

    # Ensure maxlags does not exceed degrees of freedom
    n_obs, n_vars = clean_df.shape
    safe_maxlags = min(maxlags, max(1, (n_obs - 10) // (n_vars + 1)))

    model = VAR(clean_df, exog=clean_exog)
    lag_order = model.select_order(maxlags=safe_maxlags)

    summary_dict = {
        "aic": lag_order.aic,
        "bic": lag_order.bic,
        "hqic": lag_order.hqic,
        "fpe": lag_order.fpe,
    }

    # Best lag chosen primarily by BIC (parsimonious) or AIC (predictive)
    chosen_lag = lag_order.bic if lag_order.bic > 0 else (lag_order.aic if lag_order.aic > 0 else 1)
    logger.info("VAR Lag Selection Orders: AIC=%s, BIC=%s, HQIC=%s, FPE=%s. Selected lag=%d",
                lag_order.aic, lag_order.bic, lag_order.hqic, lag_order.fpe, chosen_lag)

    # Save summary table
    summary_df = pd.DataFrame([summary_dict])
    summary_df["selected_lag"] = chosen_lag
    summary_df.to_csv(TABLES / "var_lag_selection.csv", index=False)

    return {"lag_order": lag_order, "selected_lag": chosen_lag, "summary_df": summary_df}


def check_var_stability(var_results: Any, save_plot: bool = True) -> bool:
    """Verify stability of fitted VAR (all roots must lie inside unit circle).

    Parameters
    ----------
    var_results : statsmodels VARResults
        Fitted VAR model.
    save_plot : bool
        Whether to generate and save the roots plot.

    Returns
    -------
    bool
        True if all roots are strictly inside the complex unit circle (< 1).
    """
    is_stable = var_results.is_stable(verbose=False)
    roots = var_results.roots
    moduli = np.abs(roots)
    max_modulus = np.max(moduli) if len(moduli) > 0 else 0.0

    logger.info("VAR Stability: is_stable=%s, max root modulus=%.4f (must be < 1.0)",
                is_stable, max_modulus)

    if save_plot:
        fig, ax = plt.subplots(figsize=(6, 6))
        # Unit circle
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.7, label="Unit Circle (|z|=1)")
        ax.scatter(np.real(roots), np.imag(roots), color="crimson", s=40, zorder=5, label=f"Roots (Max={max_modulus:.3f})")
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.axvline(0, color="gray", linewidth=0.8)
        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-1.25, 1.25)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"VAR Polynomial Inverse Roots\nStable: {is_stable} (Max |z| = {max_modulus:.3f})", fontweight="bold")
        ax.set_xlabel("Real Part")
        ax.set_ylabel("Imaginary Part")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig_path = FIGURES / "var_roots.png"
        fig.savefig(fig_path, dpi=120)
        plt.close(fig)
        logger.info("Saved VAR roots plot to %s", fig_path)

    return bool(is_stable)


def johansen_cointegration_test(
    df_level: pd.DataFrame,
    det_order: int = 0,
    k_ar_diff: int = 1,
) -> Dict[str, Any]:
    """Run Johansen cointegration test on I(1) system.

    Parameters
    ----------
    df_level : pd.DataFrame
        Level time series for I(1) variables.
    det_order : int
        -1: no constant; 0: constant in cointegrating eq; 1: linear trend.
    k_ar_diff : int
        Number of lagged differences in VECM.

    Returns
    -------
    dict
        Trace and Eigenvalue test statistics, critical values, and cointegration rank.
    """
    clean_df = df_level.dropna()
    coint_res = coint_johansen(clean_df, det_order, k_ar_diff)

    trace_stats = coint_res.lr1
    trace_crit = coint_res.cvt[:, 1]  # 5% critical values
    eigen_stats = coint_res.lr2
    eigen_crit = coint_res.cvm[:, 1]  # 5% critical values

    # Determine cointegration rank r where stat > crit
    r_trace = int(np.sum(trace_stats > trace_crit))
    r_eigen = int(np.sum(eigen_stats > eigen_crit))
    coint_rank = min(r_trace, r_eigen)

    logger.info("Johansen Test: trace rank=%d, eigen rank=%d -> Chosen rank=%d",
                r_trace, r_eigen, coint_rank)

    records = []
    cols = clean_df.columns
    for i in range(len(cols)):
        records.append({
            "null_hypo": f"r <= {i}",
            "trace_stat": round(trace_stats[i], 3),
            "trace_crit_5pct": round(trace_crit[i], 3),
            "trace_reject": bool(trace_stats[i] > trace_crit[i]),
            "eigen_stat": round(eigen_stats[i], 3),
            "eigen_crit_5pct": round(eigen_crit[i], 3),
            "eigen_reject": bool(eigen_stats[i] > eigen_crit[i]),
        })
    df_res = pd.DataFrame(records)
    df_res.to_csv(TABLES / "johansen_cointegration.csv", index=False)

    return {
        "coint_res": coint_res,
        "coint_rank": coint_rank,
        "summary_table": df_res,
    }


def fit_var_model(
    df_stationary: pd.DataFrame,
    lags: int = 1,
    exog: Optional[pd.DataFrame] = None,
) -> Any:
    """Fit a Vector Autoregression (VAR) model.

    Parameters
    ----------
    df_stationary : pd.DataFrame
        Stationary series.
    lags : int
        Lag order.
    exog : pd.DataFrame, optional
        Exogenous regressors.

    Returns
    -------
    statsmodels VARResults
    """
    clean_df = df_stationary.dropna()
    clean_exog = exog.loc[clean_df.index] if exog is not None else None

    var = VAR(clean_df, exog=clean_exog)
    fitted = var.fit(lags)
    logger.info("Fitted VAR(p=%d) with AIC=%.2f, BIC=%.2f", lags, fitted.aic, fitted.bic)
    return fitted


def fit_vecm_model(
    df_level: pd.DataFrame,
    coint_rank: int = 1,
    k_ar_diff: int = 1,
    deterministic: str = "co",
    exog: Optional[pd.DataFrame] = None,
) -> Any:
    """Fit a Vector Error Correction Model (VECM).

    Parameters
    ----------
    df_level : pd.DataFrame
        Level time series.
    coint_rank : int
        Cointegration rank (r).
    k_ar_diff : int
        Lags of differences.
    deterministic : str
        Deterministic terms: "co" = constant inside coint relation.
    exog : pd.DataFrame, optional
        Exogenous regressors.

    Returns
    -------
    statsmodels VECMResults
    """
    clean_df = df_level.dropna()
    clean_exog = exog.loc[clean_df.index] if exog is not None else None

    vecm = VECM(
        clean_df,
        exog=clean_exog,
        k_ar_diff=k_ar_diff,
        coint_rank=coint_rank,
        deterministic=deterministic,
    )
    fitted = vecm.fit()
    logger.info("Fitted VECM(rank=%d, k_ar_diff=%d)", coint_rank, k_ar_diff)
    return fitted


def forecast_var(
    var_results: Any,
    steps: int,
    exog_future: Optional[np.ndarray] = None,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate multi-step forecasts with analytical confidence intervals for VAR.

    Parameters
    ----------
    var_results : VARResults
        Fitted VAR model.
    steps : int
        Forecast horizon.
    exog_future : np.ndarray, optional
        Future exogenous values.
    alpha : float
        Significance level (0.05 for 95% CI).

    Returns
    -------
    (point_forecast, lower_ci, upper_ci) : tuple of np.ndarray
    """
    k_ar = var_results.k_ar
    y_hist = var_results.endog[-k_ar:]

    point = var_results.forecast(y=y_hist, steps=steps, exog_future=exog_future)
    forecast_interval = var_results.forecast_interval(y=y_hist, steps=steps, alpha=alpha, exog_future=exog_future)
    lower = forecast_interval[1]
    upper = forecast_interval[2]

    return point, lower, upper


def forecast_vecm(
    vecm_results: Any,
    steps: int,
    exog_future: Optional[np.ndarray] = None,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate multi-step forecasts with intervals for VECM.

    Parameters
    ----------
    vecm_results : VECMResults
        Fitted VECM model.
    steps : int
        Horizon.
    exog_future : np.ndarray, optional
        Exog values.
    alpha : float
        Significance level.

    Returns
    -------
    (point_forecast, lower_ci, upper_ci) : tuple of np.ndarray
    """
    point = vecm_results.predict(steps=steps, exog_fc=exog_future)
    interval = vecm_results.predict(steps=steps, alpha=alpha, exog_fc=exog_future)
    lower = interval[1]
    upper = interval[2]
    return point, lower, upper


def run_var_pipeline(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    train_end: Optional[pd.Timestamp] = None,
    horizon: int = 7,
) -> Dict[str, Any]:
    """Execute complete VAR/VECM modeling pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Featured daily DataFrame.
    cfg : dict
        Project configuration.
    train_end : pd.Timestamp, optional
        Train window cutoff.
    horizon : int
        Prediction horizon.

    Returns
    -------
    dict
        Fitted models, forecasts, diagnostics, and cointegration decision.
    """
    logger.info("=== Running VAR / VECM Modeling Pipeline ===")
    endo_vars = [c for c in cfg.get("var_endogenous", []) if c in df.columns]
    exog_vars = [c for c in cfg.get("var_exogenous", []) if c in df.columns]

    if len(endo_vars) < 2:
        raise ValueError(f"At least 2 endogenous variables required for VAR. Found: {endo_vars}")
    if len(endo_vars) > 7:
        logger.warning("Parameter explosion risk: %d endogenous variables selected", len(endo_vars))

    # Split train/test
    if train_end is None:
        n = len(df)
        train_end = df.index[int(n * cfg.get("eval_initial_train_frac", 0.7))]

    train_df = df.loc[:train_end]
    test_df = df.loc[train_end + pd.Timedelta(days=1):]

    # Prepare stationary differenced data for VAR
    train_diff = train_df[endo_vars].diff().dropna()
    train_exog = train_df[exog_vars].loc[train_diff.index] if exog_vars else None

    # Lag selection
    max_lags = cfg.get("var_max_lags", 14)
    lag_sel = select_var_lag(train_diff, maxlags=max_lags, exog=train_exog)
    chosen_lag = max(1, lag_sel["selected_lag"])

    # Fit VAR
    var_model = fit_var_model(train_diff, lags=chosen_lag, exog=train_exog)
    is_stable = check_var_stability(var_model, save_plot=True)

    # Johansen Cointegration test on levels
    train_levels = train_df[endo_vars].dropna()
    johansen_res = johansen_cointegration_test(train_levels, det_order=0, k_ar_diff=chosen_lag)
    coint_rank = johansen_res["coint_rank"]

    vecm_model = None
    model_choice = "VAR_differenced"
    if coint_rank > 0 and coint_rank < len(endo_vars):
        logger.info("Cointegration detected (r=%d). Fitting VECM in levels...", coint_rank)
        try:
            vecm_model = fit_vecm_model(
                train_levels,
                coint_rank=coint_rank,
                k_ar_diff=chosen_lag,
                exog=train_df[exog_vars].loc[train_levels.index] if exog_vars else None,
            )
            model_choice = "VECM"
            logger.info("VECM successfully fitted and chosen over differenced VAR.")
        except Exception as exc:
            logger.warning("VECM fit failed (%s). Falling back to VAR on differences.", exc)
            model_choice = "VAR_differenced"
    else:
        logger.info("No cointegration (r=%d). Retaining differenced VAR.", coint_rank)

    # Future exog
    future_exog_mat = None
    if exog_vars:
        if len(test_df) >= horizon:
            future_exog_mat = test_df[exog_vars].iloc[:horizon].values
        else:
            last_exog = train_df[exog_vars].iloc[-1:].values
            future_exog_mat = np.repeat(last_exog, horizon, axis=0)

    # Produce forecast
    if model_choice == "VECM" and vecm_model is not None:
        point, lower, upper = forecast_vecm(vecm_model, steps=horizon, exog_future=future_exog_mat)
        forecast_levels = point
        lower_levels = lower
        upper_levels = upper
    else:
        # Forecast differenced VAR and reconstruct levels
        point_diff, lower_diff, upper_diff = forecast_var(var_model, steps=horizon, exog_future=future_exog_mat)
        last_levels = train_df[endo_vars].iloc[-1].values
        forecast_levels = last_levels + np.cumsum(point_diff, axis=0)
        lower_levels = last_levels + np.cumsum(lower_diff, axis=0)
        upper_levels = last_levels + np.cumsum(upper_diff, axis=0)

    future_index = pd.date_range(train_end + pd.Timedelta(days=1), periods=horizon, freq="D")
    df_forecast = pd.DataFrame(forecast_levels, index=future_index, columns=endo_vars)
    df_lower = pd.DataFrame(lower_levels, index=future_index, columns=endo_vars)
    df_upper = pd.DataFrame(upper_levels, index=future_index, columns=endo_vars)

    # Save fitted model
    with open(MODELS / "var_model.pkl", "wb") as f:
        pickle.dump({
            "var_model": var_model,
            "vecm_model": vecm_model,
            "model_choice": model_choice,
            "chosen_lag": chosen_lag,
            "is_stable": is_stable,
            "coint_rank": coint_rank,
            "endo_vars": endo_vars,
            "exog_vars": exog_vars,
        }, f)

    return {
        "var_model": var_model,
        "vecm_model": vecm_model,
        "model_choice": model_choice,
        "chosen_lag": chosen_lag,
        "is_stable": is_stable,
        "coint_rank": coint_rank,
        "forecast": df_forecast,
        "lower_ci": df_lower,
        "upper_ci": df_upper,
        "endo_vars": endo_vars,
    }
