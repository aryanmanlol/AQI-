"""
Tests for Data Fetch, Cleaning, and Aggregation Pipeline (Offline / Synthetic).
"""
import numpy as np
import pandas as pd
import pytest
from src.config import load_config
from src.data_clean import clean_data, enforce_regular_index
from src.data_fetch import aggregate_hourly_to_daily, generate_synthetic


@pytest.fixture
def test_config():
    """Load configuration dictionary."""
    return load_config()


def test_synthetic_data_generation(test_config):
    """Verify synthetic generation outputs valid hourly DataFrame with expected columns."""
    df_synth = generate_synthetic(test_config, city="Delhi", seed=42)
    assert isinstance(df_synth.index, pd.DatetimeIndex)
    assert "pm2_5" in df_synth.columns
    assert "precipitation" in df_synth.columns
    assert len(df_synth) > 1000
    assert df_synth.attrs.get("synthetic") is True


def test_hourly_to_daily_aggregation():
    """Verify daily aggregation sums precipitation while averaging other variables."""
    idx = pd.date_range("2024-01-01 00:00:00", periods=24, freq="h")
    df_hour = pd.DataFrame({
        "pm2_5": [50.0] * 24,
        "precipitation": [2.0] * 24,
    }, index=idx)

    daily = aggregate_hourly_to_daily(df_hour)
    assert len(daily) == 1
    # Mean of 50 is 50
    assert daily["pm2_5"].iloc[0] == pytest.approx(50.0)
    # Sum of 24 * 2.0 is 48.0
    assert daily["precipitation"].iloc[0] == pytest.approx(48.0)


def test_cleaning_pipeline_no_nans(test_config):
    """Verify cleaning enforces regular index and interpolates gaps without trailing NaNs."""
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    df_raw = pd.DataFrame({
        "pm2_5": [40.0, np.nan, 42.0, 45.0, np.nan, np.nan, 50.0, 52.0, 55.0, 58.0],
    }, index=idx)

    cleaned = clean_data(df_raw, test_config, city="Delhi", freq="D")
    assert isinstance(cleaned.index, pd.DatetimeIndex)
    assert cleaned["pm2_5"].isna().sum() == 0
