"""
Data Fetch Module – downloads pollutant and weather data from Open-Meteo APIs.

Handles:
  - CPCB CSV import (if data/raw/cpcb_<city>.csv exists)
  - Open-Meteo Air Quality API for pollutants
  - Open-Meteo Archive API for weather
  - Synthetic data generation (--allow-synthetic fallback)
  - Caching to disk (parquet) to avoid redundant network calls
  - Yearly chunking, exponential-backoff retries
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from src.config import (
    DATA_RAW,
    DATA_INTERIM,
    get_city_coords,
    get_seasonal_period,
)

logger = logging.getLogger(__name__)

# ─── API endpoints ──────────────────────────────────────────
AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"


# ─── Retry wrapper ──────────────────────────────────────────
def _api_get(
    url: str,
    params: Dict[str, Any],
    max_retries: int = 5,
    backoff_base: float = 2.0,
) -> Dict[str, Any]:
    """GET with exponential backoff.

    Parameters
    ----------
    url : str
        API endpoint URL.
    params : dict
        Query parameters.
    max_retries : int
        Maximum number of retries.
    backoff_base : float
        Base for exponential backoff.

    Returns
    -------
    dict
        Parsed JSON response.

    Raises
    ------
    requests.exceptions.RequestException
        If all retries are exhausted.
    """
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            if "error" in data and data.get("error"):
                raise ValueError(f"API error: {data.get('reason', data)}")
            return data
        except (requests.RequestException, ValueError) as exc:
            wait = backoff_base ** attempt
            logger.warning(
                "Attempt %d/%d failed (%s). Retrying in %.1fs …",
                attempt,
                max_retries,
                exc,
                wait,
            )
            if attempt == max_retries:
                raise
            time.sleep(wait)
    raise RuntimeError("Unreachable")  # pragma: no cover


# ─── Year-chunked fetcher ───────────────────────────────────
def _year_chunks(
    start: str, end: str, chunk_years: int = 1
) -> List[Tuple[str, str]]:
    """Split a date range into yearly chunks.

    Parameters
    ----------
    start, end : str
        ISO date strings.
    chunk_years : int
        Number of years per chunk.

    Returns
    -------
    list of (str, str) tuples
    """
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    chunks: List[Tuple[str, str]] = []
    while s < e:
        ce = min(s.replace(year=s.year + chunk_years) - timedelta(days=1), e)
        chunks.append((s.strftime("%Y-%m-%d"), ce.strftime("%Y-%m-%d")))
        s = ce + timedelta(days=1)
    return chunks


def _detect_earliest_date(
    url: str,
    lat: float,
    lon: float,
    variables: List[str],
    configured_start: str,
    api_cfg: Dict[str, Any],
) -> str:
    """Try the configured start date; if the API returns an error or empty data,
    probe forward until we find a valid start (global AQ data ~mid-2022).

    Parameters
    ----------
    url : str
        API base URL.
    lat, lon : float
        Coordinates.
    variables : list of str
        Variables to request.
    configured_start : str
        Start date from config.
    api_cfg : dict
        API retry settings.

    Returns
    -------
    str
        Earliest usable date in ISO format.
    """
    probe_start = datetime.strptime(configured_start, "%Y-%m-%d")
    today = datetime.utcnow() - timedelta(days=5)  # safety margin
    while probe_start < today:
        probe_end = probe_start + timedelta(days=10)
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join(variables[:1]),  # only one var for speed
            "start_date": probe_start.strftime("%Y-%m-%d"),
            "end_date": probe_end.strftime("%Y-%m-%d"),
        }
        try:
            data = _api_get(
                url,
                params,
                max_retries=api_cfg.get("max_retries", 3),
                backoff_base=api_cfg.get("backoff_base", 2.0),
            )
            hourly = data.get("hourly", {})
            vals = hourly.get(variables[0], [])
            non_null = [v for v in vals if v is not None]
            if non_null:
                logger.info("Earliest available date: %s", probe_start.strftime("%Y-%m-%d"))
                return probe_start.strftime("%Y-%m-%d")
        except Exception:
            pass
        probe_start += timedelta(days=60)
    # fallback
    fallback = "2022-07-01"
    logger.warning("Could not detect earliest date; falling back to %s", fallback)
    return fallback


# ─── Pollutant fetch ────────────────────────────────────────
def fetch_pollutants(
    cfg: Dict[str, Any],
    city: str,
    cache_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Download hourly pollutant data from Open-Meteo Air Quality API.

    Parameters
    ----------
    cfg : dict
        Project config.
    city : str
        City name.
    cache_path : Path, optional
        Override cache file path.

    Returns
    -------
    pd.DataFrame
        Hourly pollutant data with DatetimeIndex.
    """
    cache = cache_path or DATA_RAW / f"openmeteo_aq_{city}.parquet"
    if cache.exists():
        logger.info("Loading cached pollutant data from %s", cache)
        return pd.read_parquet(cache)

    coords = get_city_coords(cfg, city)
    lat, lon = coords["latitude"], coords["longitude"]
    variables = cfg["pollutant_vars"] + ["us_aqi"]
    api_cfg = cfg.get("api", {})

    start = _detect_earliest_date(
        AQ_URL, lat, lon, cfg["pollutant_vars"], cfg["start_date"], api_cfg
    )
    if start != cfg["start_date"]:
        logger.warning(
            "Configured start_date %s unavailable; shifted to %s",
            cfg["start_date"],
            start,
        )
    end = cfg["end_date"]

    chunks = _year_chunks(start, end, api_cfg.get("chunk_years", 1))
    frames: List[pd.DataFrame] = []
    for cs, ce in tqdm(chunks, desc="Fetching AQ data"):
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join(variables),
            "start_date": cs,
            "end_date": ce,
        }
        data = _api_get(
            AQ_URL,
            params,
            max_retries=api_cfg.get("max_retries", 5),
            backoff_base=api_cfg.get("backoff_base", 2.0),
        )
        hourly = data.get("hourly", {})
        if "time" not in hourly:
            logger.warning("No hourly data for chunk %s – %s", cs, ce)
            continue
        df = pd.DataFrame(hourly)
        df["time"] = pd.to_datetime(df["time"])
        frames.append(df)

    if not frames:
        raise RuntimeError(
            f"No pollutant data retrieved for {city}. "
            "Check network or use --allow-synthetic."
        )

    result = pd.concat(frames, ignore_index=True).drop_duplicates(subset="time")
    result = result.sort_values("time").set_index("time")
    result.index.name = "datetime"
    result.to_parquet(cache)
    logger.info("Pollutant data cached to %s (%d rows)", cache, len(result))
    return result


# ─── Weather fetch ──────────────────────────────────────────
def fetch_weather(
    cfg: Dict[str, Any],
    city: str,
    cache_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Download hourly weather data from Open-Meteo Archive API.

    Parameters
    ----------
    cfg : dict
        Project config.
    city : str
        City name.
    cache_path : Path, optional
        Override cache file path.

    Returns
    -------
    pd.DataFrame
        Hourly weather data with DatetimeIndex.
    """
    cache = cache_path or DATA_RAW / f"openmeteo_weather_{city}.parquet"
    if cache.exists():
        logger.info("Loading cached weather data from %s", cache)
        return pd.read_parquet(cache)

    coords = get_city_coords(cfg, city)
    lat, lon = coords["latitude"], coords["longitude"]
    variables = cfg["weather_vars"]
    api_cfg = cfg.get("api", {})
    start = cfg["start_date"]
    end = cfg["end_date"]

    chunks = _year_chunks(start, end, api_cfg.get("chunk_years", 1))
    frames: List[pd.DataFrame] = []
    for cs, ce in tqdm(chunks, desc="Fetching weather data"):
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join(variables),
            "start_date": cs,
            "end_date": ce,
        }
        data = _api_get(
            WEATHER_URL,
            params,
            max_retries=api_cfg.get("max_retries", 5),
            backoff_base=api_cfg.get("backoff_base", 2.0),
        )
        hourly = data.get("hourly", {})
        if "time" not in hourly:
            logger.warning("No weather data for chunk %s – %s", cs, ce)
            continue
        df = pd.DataFrame(hourly)
        df["time"] = pd.to_datetime(df["time"])
        frames.append(df)

    if not frames:
        raise RuntimeError(
            f"No weather data retrieved for {city}. "
            "Check network or use --allow-synthetic."
        )

    result = pd.concat(frames, ignore_index=True).drop_duplicates(subset="time")
    result = result.sort_values("time").set_index("time")
    result.index.name = "datetime"
    result.to_parquet(cache)
    logger.info("Weather data cached to %s (%d rows)", cache, len(result))
    return result


# ─── CPCB CSV loader ────────────────────────────────────────
def load_cpcb_csv(city: str, cfg: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Load a user-provided CPCB/Kaggle CSV if it exists.

    Uses the flexible column mapper from config to normalise column names.

    Parameters
    ----------
    city : str
        City name.
    cfg : dict
        Project config (contains cpcb_column_map).

    Returns
    -------
    pd.DataFrame or None
        Normalised DataFrame, or None if file not found.
    """
    path = DATA_RAW / f"cpcb_{city}.csv"
    if not path.exists():
        return None

    logger.info("Found CPCB CSV at %s – using as pollutant source", path)
    df = pd.read_csv(path)
    col_map = cfg.get("cpcb_column_map", {})

    rename = {}
    for orig_col in df.columns:
        stripped = orig_col.strip()
        if stripped in col_map:
            rename[orig_col] = col_map[stripped]
    df = df.rename(columns=rename)

    # Try to parse a datetime column
    date_col = None
    for candidate in ["date", "datetime", "Date", "Datetime", "Timestamp"]:
        if candidate in df.columns:
            date_col = candidate
            break
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
        df = df.set_index(date_col)
        df.index.name = "datetime"

    # Convert numeric columns
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ─── Synthetic data generator ───────────────────────────────
def generate_synthetic(
    cfg: Dict[str, Any], city: str, seed: int = 42
) -> pd.DataFrame:
    """Generate synthetic AQI data for offline testing.

    Produces ~3 years of daily data with:
      - Seasonality (annual + weekly)
      - Trend
      - AR(1) noise
      - Weather coupling

    Parameters
    ----------
    cfg : dict
        Project config.
    city : str
        City name (for labelling).
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Synthetic hourly data with "SYNTHETIC" stamp.
    """
    logger.warning("⚠ Generating SYNTHETIC data for %s – NOT real measurements!", city)
    rng = np.random.default_rng(seed)

    start = pd.Timestamp(cfg["start_date"])
    end = pd.Timestamp(cfg["end_date"])
    idx = pd.date_range(start, end, freq="h", name="datetime")
    n = len(idx)

    # Day-of-year for seasonality
    doy = idx.dayofyear.values.astype(float)
    hour = idx.hour.values.astype(float)

    # Temperature (seasonal)
    temp = 25 + 10 * np.sin(2 * np.pi * (doy - 120) / 365) + rng.normal(0, 2, n)

    # Humidity (inverse to temp roughly)
    rh = 60 - 15 * np.sin(2 * np.pi * (doy - 120) / 365) + rng.normal(0, 5, n)
    rh = np.clip(rh, 10, 100)

    # Wind speed
    ws = 3 + 1.5 * np.sin(2 * np.pi * doy / 365) + rng.exponential(1, n)

    # Wind direction
    wd = rng.uniform(0, 360, n)

    # Precipitation (monsoon-heavy)
    precip = np.zeros(n)
    monsoon_mask = np.isin(idx.month, [6, 7, 8, 9])
    precip[monsoon_mask] = rng.exponential(2, monsoon_mask.sum())
    precip += rng.exponential(0.1, n)

    # Surface pressure
    sp = 1013 - 5 * np.sin(2 * np.pi * doy / 365) + rng.normal(0, 2, n)

    # PM2.5 with seasonality, diurnal, AR(1), weather coupling
    pm25 = np.zeros(n)
    pm25[0] = 80
    for i in range(1, n):
        seasonal = 30 * np.sin(2 * np.pi * (doy[i] - 300) / 365)  # winter peak
        diurnal = 10 * np.sin(2 * np.pi * (hour[i] - 8) / 24)
        weather_effect = -0.5 * ws[i] + 0.1 * rh[i] - 0.5 * precip[i]
        pm25[i] = 0.85 * pm25[i - 1] + 0.15 * (80 + seasonal + diurnal + weather_effect) + rng.normal(0, 8)
    pm25 = np.clip(pm25, 5, 500)

    # PM10
    pm10 = pm25 * (1.5 + rng.normal(0, 0.1, n))
    pm10 = np.clip(pm10, 8, 800)

    # NO2
    no2 = 0.3 * pm25 + 10 + rng.normal(0, 5, n)
    no2 = np.clip(no2, 2, 200)

    # SO2
    so2 = 8 + rng.exponential(3, n)
    so2 = np.clip(so2, 1, 100)

    # CO
    co = 300 + 0.5 * pm25 + rng.normal(0, 50, n)
    co = np.clip(co, 100, 2000)

    # O3 (inverse to PM in winter)
    o3 = 40 - 0.2 * pm25 + 20 * np.sin(2 * np.pi * (doy - 80) / 365) + rng.normal(0, 8, n)
    o3 = np.clip(o3, 5, 200)

    # Simple AQI proxy
    us_aqi = np.clip(pm25 * 1.2 + rng.normal(0, 5, n), 10, 500)

    df = pd.DataFrame(
        {
            "pm2_5": pm25,
            "pm10": pm10,
            "carbon_monoxide": co,
            "nitrogen_dioxide": no2,
            "sulphur_dioxide": so2,
            "ozone": o3,
            "us_aqi": us_aqi,
            "temperature_2m": temp,
            "relative_humidity_2m": rh,
            "wind_speed_10m": ws,
            "wind_direction_10m": wd,
            "precipitation": precip,
            "surface_pressure": sp,
        },
        index=idx,
    )
    df.attrs["synthetic"] = True
    cache = DATA_RAW / f"synthetic_{city}.parquet"
    df.to_parquet(cache)
    logger.info("Synthetic data saved to %s (%d rows)", cache, len(df))
    return df


# ─── Merge & aggregate ──────────────────────────────────────
def aggregate_hourly_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly data to daily.

    Pollutants and weather use daily mean; precipitation uses daily sum.

    Parameters
    ----------
    df : pd.DataFrame
        Hourly DataFrame.

    Returns
    -------
    pd.DataFrame
        Daily DataFrame.
    """
    agg_rules: Dict[str, str] = {}
    for col in df.columns:
        if col == "precipitation":
            agg_rules[col] = "sum"
        else:
            agg_rules[col] = "mean"
    daily = df.resample("D").agg(agg_rules)
    daily.index.name = "datetime"
    return daily


def data_quality_report(df: pd.DataFrame, label: str = "") -> pd.DataFrame:
    """Generate a data quality report.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    label : str
        Label for logging.

    Returns
    -------
    pd.DataFrame
        Quality metrics per variable.
    """
    records = []
    for col in df.columns:
        series = df[col]
        total = len(series)
        missing = series.isna().sum()
        pct = missing / total * 100 if total > 0 else 0

        # Longest continuous gap
        is_na = series.isna()
        if is_na.any():
            groups = (is_na != is_na.shift()).cumsum()
            gap_lengths = is_na.groupby(groups).sum()
            longest_gap = int(gap_lengths.max())
        else:
            longest_gap = 0

        records.append(
            {
                "variable": col,
                "n_obs": total,
                "missing": int(missing),
                "missing_pct": round(pct, 2),
                "longest_gap": longest_gap,
                "start": str(series.first_valid_index()),
                "end": str(series.last_valid_index()),
            }
        )

    report = pd.DataFrame(records)
    if label:
        logger.info("Data quality report (%s):", label)
        for _, row in report.iterrows():
            level = logging.WARNING if row["missing_pct"] > 15 else logging.INFO
            logger.log(
                level,
                "  %-25s  %5d obs  %5.1f%% missing  longest gap %d",
                row["variable"],
                row["n_obs"],
                row["missing_pct"],
                row["longest_gap"],
            )
    return report


# ─── Main orchestrator ──────────────────────────────────────
def fetch_all_data(
    cfg: Dict[str, Any],
    city: str,
    allow_synthetic: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, bool]:
    """Fetch / load all data sources and return hourly + daily DataFrames.

    Priority:
      1. CPCB CSV for pollutants (still fetches weather from API)
      2. Open-Meteo for both pollutants and weather
      3. Synthetic fallback (only if allow_synthetic=True)

    Parameters
    ----------
    cfg : dict
        Project config.
    city : str
        City name.
    allow_synthetic : bool
        Allow synthetic data fallback.

    Returns
    -------
    (hourly_df, daily_df, is_synthetic) : tuple
        Merged DataFrames and a boolean flag.
    """
    is_synthetic = False

    # Priority 1: CPCB CSV
    cpcb_df = load_cpcb_csv(city, cfg)

    try:
        if cpcb_df is not None:
            logger.info("Using CPCB CSV for pollutants, Open-Meteo for weather")
            weather_df = fetch_weather(cfg, city)
            # Merge on datetime
            pollutant_df = cpcb_df
            # Align indices
            common_start = max(pollutant_df.index.min(), weather_df.index.min())
            common_end = min(pollutant_df.index.max(), weather_df.index.max())
            pollutant_df = pollutant_df.loc[common_start:common_end]
            weather_df = weather_df.loc[common_start:common_end]
            hourly = pollutant_df.join(weather_df, how="outer")
        else:
            # Priority 2: Open-Meteo for everything
            logger.info("Fetching all data from Open-Meteo APIs")
            pollutant_df = fetch_pollutants(cfg, city)
            weather_df = fetch_weather(cfg, city)
            # Align
            common_start = max(pollutant_df.index.min(), weather_df.index.min())
            common_end = min(pollutant_df.index.max(), weather_df.index.max())
            pollutant_df = pollutant_df.loc[common_start:common_end]
            weather_df = weather_df.loc[common_start:common_end]
            hourly = pollutant_df.join(weather_df, how="outer")

    except (requests.RequestException, RuntimeError) as exc:
        if allow_synthetic:
            logger.warning(
                "Network unavailable (%s). Falling back to synthetic data.", exc
            )
            hourly = generate_synthetic(cfg, city, cfg["project"]["random_seed"])
            is_synthetic = True
        else:
            raise RuntimeError(
                f"Data fetch failed: {exc}. Use --allow-synthetic for offline testing."
            ) from exc

    # Save hourly
    hourly_path = DATA_RAW / f"hourly_{city}.parquet"
    hourly.to_parquet(hourly_path)
    logger.info("Hourly data saved: %s (%d rows, %d cols)", hourly_path, *hourly.shape)

    # Save hourly CSV too
    hourly_csv = DATA_RAW / f"hourly_{city}.csv"
    hourly.to_csv(hourly_csv)

    # Aggregate to daily
    daily = aggregate_hourly_to_daily(hourly)
    daily_path = DATA_RAW / f"daily_{city}.parquet"
    daily.to_parquet(daily_path)
    logger.info("Daily data saved: %s (%d rows, %d cols)", daily_path, *daily.shape)

    # Quality report
    from src.config import TABLES
    report = data_quality_report(daily, label=f"{city} daily")
    report.to_csv(TABLES / "data_quality.csv", index=False)

    return hourly, daily, is_synthetic
