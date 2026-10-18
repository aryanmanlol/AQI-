"""
AQI Utilities – CPCB AQI sub-index breakpoints and category mapping.

The breakpoint table is based on the Indian National Air Quality Index (NAQI)
published by CPCB. **VERIFY these breakpoints against the latest CPCB
notification before using in any official capacity.**

Reference: CPCB National Air Quality Index, 2014.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── CPCB AQI Breakpoint Table ───────────────────────────────
# Format: (Cp_lo, Cp_hi, I_lo, I_hi)
# Cp = pollutant concentration, I = sub-index value
# VERIFY against current CPCB notification!
BREAKPOINTS: Dict[str, List[Tuple[float, float, int, int]]] = {
    "pm2_5": [
        # (Cp_lo, Cp_hi, I_lo, I_hi) – 24-hr avg, µg/m³
        (0, 30, 0, 50),          # Good
        (31, 60, 51, 100),       # Satisfactory
        (61, 90, 101, 200),      # Moderate
        (91, 120, 201, 300),     # Poor
        (121, 250, 301, 400),    # Very Poor
        (251, 500, 401, 500),    # Severe
    ],
    "pm10": [
        (0, 50, 0, 50),
        (51, 100, 51, 100),
        (101, 250, 101, 200),
        (251, 350, 201, 300),
        (351, 430, 301, 400),
        (431, 600, 401, 500),
    ],
    "nitrogen_dioxide": [
        # NO2 – 24-hr avg, µg/m³
        (0, 40, 0, 50),
        (41, 80, 51, 100),
        (81, 180, 101, 200),
        (181, 280, 201, 300),
        (281, 400, 301, 400),
        (401, 800, 401, 500),
    ],
    "sulphur_dioxide": [
        # SO2 – 24-hr avg, µg/m³
        (0, 40, 0, 50),
        (41, 80, 51, 100),
        (81, 380, 101, 200),
        (381, 800, 201, 300),
        (801, 1600, 301, 400),
        (1601, 2400, 401, 500),
    ],
    "carbon_monoxide": [
        # CO – 8-hr avg, mg/m³ (Open-Meteo gives µg/m³; convert!)
        (0, 1000, 0, 50),
        (1001, 2000, 51, 100),
        (2001, 10000, 101, 200),
        (10001, 17000, 201, 300),
        (17001, 34000, 301, 400),
        (34001, 50000, 401, 500),
    ],
    "ozone": [
        # O3 – 8-hr avg, µg/m³
        (0, 50, 0, 50),
        (51, 100, 51, 100),
        (101, 168, 101, 200),
        (169, 208, 201, 300),
        (209, 748, 301, 400),
        (749, 1000, 401, 500),
    ],
}

# AQI Category definitions
AQI_CATEGORIES = [
    (0, 50, "Good"),
    (51, 100, "Satisfactory"),
    (101, 200, "Moderate"),
    (201, 300, "Poor"),
    (301, 400, "Very Poor"),
    (401, 500, "Severe"),
]


def compute_sub_index(concentration: float, pollutant: str) -> float:
    """Compute the AQI sub-index for a single pollutant concentration.

    Uses linear interpolation within the CPCB breakpoint table.

    Parameters
    ----------
    concentration : float
        Pollutant concentration value.
    pollutant : str
        Pollutant name (must be a key in BREAKPOINTS).

    Returns
    -------
    float
        Sub-index value, or NaN if concentration is out of range.
    """
    if np.isnan(concentration) or concentration < 0:
        return np.nan

    bp = BREAKPOINTS.get(pollutant)
    if bp is None:
        logger.warning("No breakpoints defined for %s", pollutant)
        return np.nan

    for cp_lo, cp_hi, i_lo, i_hi in bp:
        if cp_lo <= concentration <= cp_hi:
            # Linear interpolation
            sub_idx = i_lo + (concentration - cp_lo) * (i_hi - i_lo) / (cp_hi - cp_lo)
            return round(sub_idx, 1)

    # Exceeds table
    if concentration > bp[-1][1]:
        return 500.0
    return np.nan


def compute_overall_aqi(sub_indices: Dict[str, float]) -> float:
    """Compute the overall AQI as the maximum of all sub-indices.

    Parameters
    ----------
    sub_indices : dict
        Mapping of pollutant name to sub-index value.

    Returns
    -------
    float
        Overall AQI value.
    """
    valid = [v for v in sub_indices.values() if not np.isnan(v)]
    if not valid:
        return np.nan
    return max(valid)


def aqi_category(aqi_value: float) -> str:
    """Map an AQI value to its CPCB category.

    Parameters
    ----------
    aqi_value : float
        AQI value (0-500+).

    Returns
    -------
    str
        Category name.
    """
    if np.isnan(aqi_value):
        return "Unknown"
    for lo, hi, cat in AQI_CATEGORIES:
        if lo <= aqi_value <= hi:
            return cat
    if aqi_value > 500:
        return "Severe"
    return "Unknown"


def compute_aqi_series(
    df: pd.DataFrame,
    pollutants: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute AQI sub-indices and overall AQI for a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Daily data with pollutant columns.
    pollutants : list of str, optional
        Pollutant column names. Defaults to all available in BREAKPOINTS.

    Returns
    -------
    pd.DataFrame
        DataFrame with sub-index columns, overall AQI, and category.
    """
    if pollutants is None:
        pollutants = [p for p in BREAKPOINTS if p in df.columns]

    result = df.copy()
    sub_idx_cols = []
    for poll in pollutants:
        col_name = f"aqi_sub_{poll}"
        result[col_name] = df[poll].apply(lambda x: compute_sub_index(x, poll))
        sub_idx_cols.append(col_name)

    # Overall AQI
    result["cpcb_aqi"] = result[sub_idx_cols].max(axis=1)
    result["aqi_category"] = result["cpcb_aqi"].apply(aqi_category)

    logger.info(
        "AQI computed: %d rows, categories: %s",
        len(result),
        result["aqi_category"].value_counts().to_dict(),
    )
    return result
