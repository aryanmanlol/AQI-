"""
Configuration loader for the AQI-VAR Forecasting project.

Reads config.yaml, seeds RNGs, and provides path helpers.
"""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)

# ── Paths ───────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUTS = PROJECT_ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"
MODELS = OUTPUTS / "models"
REPORTS = PROJECT_ROOT / "reports"

ALL_DIRS = [DATA_RAW, DATA_INTERIM, DATA_PROCESSED, FIGURES, TABLES, MODELS, REPORTS]


def ensure_dirs() -> None:
    """Create all project directories if they don't exist."""
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load and return the YAML configuration dictionary.

    Parameters
    ----------
    path : Path, optional
        Override path to config.yaml.  Defaults to PROJECT_ROOT / config.yaml.

    Returns
    -------
    dict
        Parsed configuration.
    """
    path = path or CONFIG_PATH
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility.

    Parameters
    ----------
    seed : int
        The random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    logger.info("Random seed set to %d", seed)


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with a standard format.

    Parameters
    ----------
    level : str
        Logging level name.
    """
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s │ %(name)-28s │ %(levelname)-7s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_city_coords(cfg: Dict[str, Any], city: str) -> Dict[str, float]:
    """Return latitude/longitude for the requested city.

    Parameters
    ----------
    cfg : dict
        Loaded config dictionary.
    city : str
        City name (case-sensitive, must match config key).

    Returns
    -------
    dict
        ``{"latitude": float, "longitude": float}``

    Raises
    ------
    KeyError
        If the city is not in the config.
    """
    cities = cfg["cities"]
    if city not in cities:
        raise KeyError(
            f"City '{city}' not in config.yaml. Available: {list(cities.keys())}"
        )
    return cities[city]


def get_seasonal_period(cfg: Dict[str, Any]) -> int:
    """Return the seasonal period based on the chosen frequency.

    Parameters
    ----------
    cfg : dict
        Loaded config dictionary.

    Returns
    -------
    int
        7 for daily, 24 for hourly.
    """
    freq = cfg.get("frequency", "daily")
    if freq == "hourly":
        return cfg.get("seasonal_period_hourly", 24)
    return cfg.get("seasonal_period_daily", 7)


def get_var_endogenous(cfg: Dict[str, Any]) -> List[str]:
    """Return the list of VAR endogenous variable names.

    Warns if more than 7 endogenous variables are selected (parameter explosion).

    Parameters
    ----------
    cfg : dict
        Loaded config dictionary.

    Returns
    -------
    list of str
    """
    endo = cfg.get("var_endogenous", [])
    if len(endo) > 7:
        logger.warning(
            "More than 7 endogenous variables (%d) selected – "
            "this will cause parameter explosion in VAR models.",
            len(endo),
        )
    return endo


def get_var_exogenous(cfg: Dict[str, Any]) -> List[str]:
    """Return the list of VAR exogenous variable names.

    Parameters
    ----------
    cfg : dict
        Loaded config dictionary.

    Returns
    -------
    list of str
    """
    return cfg.get("var_exogenous", [])


def init_project(city: Optional[str] = None) -> Dict[str, Any]:
    """One-call project initialisation.

    Loads config, sets up logging, seeds RNGs, creates directories.

    Parameters
    ----------
    city : str, optional
        Override default city.

    Returns
    -------
    dict
        The loaded (and potentially patched) configuration.
    """
    cfg = load_config()
    setup_logging(cfg["project"].get("log_level", "INFO"))
    seed_everything(cfg["project"]["random_seed"])
    ensure_dirs()
    if city:
        cfg["default_city"] = city
    return cfg
