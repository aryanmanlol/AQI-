"""
Tests for CPCB AQI Breakpoint Calculations and Category Mapping.
"""
import numpy as np
import pytest
from src.aqi_utils import aqi_category, compute_overall_aqi, compute_sub_index


def test_pm25_sub_indices_known_values():
    """Verify PM2.5 concentrations match official CPCB breakpoints."""
    # Good: 0 to 30 -> 0 to 50
    assert compute_sub_index(0.0, "pm2_5") == 0.0
    assert compute_sub_index(30.0, "pm2_5") == 50.0
    assert compute_sub_index(15.0, "pm2_5") == 25.0

    # Satisfactory: 31 to 60 -> 51 to 100
    assert compute_sub_index(60.0, "pm2_5") == 100.0

    # Severe upper boundary
    assert compute_sub_index(500.0, "pm2_5") == 500.0
    assert compute_sub_index(600.0, "pm2_5") == 500.0  # Cap at 500


def test_aqi_category_mapping():
    """Verify categorical string mappings across NAQI brackets."""
    assert aqi_category(35.0) == "Good"
    assert aqi_category(75.0) == "Satisfactory"
    assert aqi_category(150.0) == "Moderate"
    assert aqi_category(250.0) == "Poor"
    assert aqi_category(350.0) == "Very Poor"
    assert aqi_category(450.0) == "Severe"
    assert aqi_category(550.0) == "Severe"
    assert aqi_category(np.nan) == "Unknown"


def test_overall_aqi_maximum_rule():
    """Verify overall AQI is the max of valid sub-indices."""
    sub_indices = {"pm2_5": 180.0, "pm10": 110.0, "no2": 65.0}
    assert compute_overall_aqi(sub_indices) == 180.0
