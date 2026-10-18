"""
Stationarity Testing – ADF and KPSS for each variable with decision logic.

Outputs a stationarity table with I(0)/I(1) decisions, handling the case
where ADF and KPSS may disagree.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

from src.config import TABLES

logger = logging.getLogger(__name__)


def adf_test(series: pd.Series, significance: float = 0.05) -> Dict[str, Any]:
    """Augmented Dickey-Fuller test.

    Parameters
    ----------
    series : pd.Series
        Time series.
    significance : float
        Significance level.

    Returns
    -------
    dict
        Test results: statistic, p-value, decision.
    """
    clean = series.dropna()
    if len(clean) < 20:
        return {"adf_stat": np.nan, "adf_pvalue": np.nan, "adf_reject": False}

    try:
        result = adfuller(clean, autolag="AIC")
        return {
            "adf_stat": round(result[0], 4),
            "adf_pvalue": round(result[1], 4),
            "adf_reject": result[1] < significance,  # True = stationary
        }
    except Exception as exc:
        logger.warning("ADF test failed for %s: %s", series.name, exc)
        return {"adf_stat": np.nan, "adf_pvalue": np.nan, "adf_reject": False}


def kpss_test(
    series: pd.Series, regression: str = "c", significance: float = 0.05
) -> Dict[str, Any]:
    """KPSS stationarity test.

    Parameters
    ----------
    series : pd.Series
        Time series.
    regression : str
        'c' for level, 'ct' for trend.
    significance : float
        Significance level.

    Returns
    -------
    dict
        Test results: statistic, p-value, decision.
    """
    clean = series.dropna()
    if len(clean) < 20:
        return {"kpss_stat": np.nan, "kpss_pvalue": np.nan, "kpss_reject": False}

    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, p, nlags, crit = kpss(clean, regression=regression, nlags="auto")
        return {
            "kpss_stat": round(stat, 4),
            "kpss_pvalue": round(p, 4),
            "kpss_reject": p < significance,  # True = non-stationary
        }
    except Exception as exc:
        logger.warning("KPSS test failed for %s: %s", series.name, exc)
        return {"kpss_stat": np.nan, "kpss_pvalue": np.nan, "kpss_reject": False}


def stationarity_decision(adf_reject: bool, kpss_reject: bool) -> Tuple[str, str]:
    """Determine integration order from ADF and KPSS results.

    The two tests can disagree:
      - ADF rejects & KPSS does not reject → Stationary I(0)
      - ADF does not reject & KPSS rejects → Non-stationary I(1)
      - Both reject → Trend-stationary (difference to be safe)
      - Neither rejects → Inconclusive (treat as I(1))

    Parameters
    ----------
    adf_reject : bool
        True if ADF null (unit root) is rejected.
    kpss_reject : bool
        True if KPSS null (stationarity) is rejected.

    Returns
    -------
    (order, reasoning) : tuple of str
    """
    if adf_reject and not kpss_reject:
        return "I(0)", "ADF rejects unit root, KPSS does not reject stationarity → Stationary"
    elif not adf_reject and kpss_reject:
        return "I(1)", "ADF fails to reject unit root, KPSS rejects stationarity → Non-stationary"
    elif adf_reject and kpss_reject:
        return "I(1)*", "Both reject: trend-stationary likely, difference to be safe"
    else:
        return "I(1)*", "Neither rejects conclusively → Inconclusive, treat as non-stationary"


def run_stationarity_tests(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    include_diff: bool = True,
) -> pd.DataFrame:
    """Run ADF and KPSS tests for all specified columns.

    Tests at level and after first differencing.

    Parameters
    ----------
    df : pd.DataFrame
        Data.
    cols : list of str, optional
        Columns to test.
    include_diff : bool
        Also test first-differenced series.

    Returns
    -------
    pd.DataFrame
        Stationarity table saved to outputs/tables/stationarity.csv.
    """
    logger.info("=== Stationarity Tests ===")

    if cols is None:
        cols = [c for c in df.columns if df[c].dtype in [np.float64, np.float32, np.int64]]

    records = []
    for col in cols:
        if col not in df.columns:
            continue

        series = df[col].dropna()
        if len(series) < 30:
            logger.warning("Skipping %s: too few observations (%d)", col, len(series))
            continue

        # Level test
        adf_res = adf_test(series)
        kpss_res = kpss_test(series)
        order, reasoning = stationarity_decision(adf_res["adf_reject"], kpss_res["kpss_reject"])
        records.append({
            "variable": col,
            "transform": "level",
            **adf_res,
            **kpss_res,
            "decision": order,
            "reasoning": reasoning,
        })

        # First difference
        if include_diff:
            diff_series = series.diff().dropna()
            diff_series.name = col
            adf_d = adf_test(diff_series)
            kpss_d = kpss_test(diff_series)
            order_d, reasoning_d = stationarity_decision(adf_d["adf_reject"], kpss_d["kpss_reject"])
            records.append({
                "variable": col,
                "transform": "1st_diff",
                **adf_d,
                **kpss_d,
                "decision": order_d,
                "reasoning": reasoning_d,
            })

    result = pd.DataFrame(records)
    result.to_csv(TABLES / "stationarity.csv", index=False)
    logger.info("Stationarity table:\n%s", result[["variable", "transform", "adf_pvalue", "kpss_pvalue", "decision"]].to_string(index=False))
    return result
