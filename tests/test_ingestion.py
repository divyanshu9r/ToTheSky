import math
from datetime import date
import numpy as np
import pandas as pd
import pytest
from ingestion.ingest import (
    clean_record_for_db,
    fetch_or_generate_weather_data,
    fetch_or_generate_flight_data,
    ingest_daily_data,
)


def test_clean_record_for_db_handles_all_missing_types():
    """
    CRITICAL REQUIREMENT TEST:
    Replace all NaN, 'nan', and None values with Python None before inserting into MySQL.
    """
    dirty_record = {
        "flight_number": "AA123",
        "raw_nan_float": float("nan"),
        "raw_nan_numpy": np.nan,
        "string_nan_lower": "nan",
        "string_nan_upper": "NAN",
        "string_none": "None",
        "string_null": "null",
        "pandas_na": pd.NA,
        "explicit_none": None,
        "valid_string": "JFK",
        "valid_int": 42,
        "valid_float": 18.5,
    }

    cleaned = clean_record_for_db(dirty_record)

    assert cleaned["flight_number"] == "AA123"
    assert cleaned["raw_nan_float"] is None
    assert cleaned["raw_nan_numpy"] is None
    assert cleaned["string_nan_lower"] is None
    assert cleaned["string_nan_upper"] is None
    assert cleaned["string_none"] is None
    assert cleaned["string_null"] is None
    assert cleaned["pandas_na"] is None
    assert cleaned["explicit_none"] is None
    assert cleaned["valid_string"] == "JFK"
    assert cleaned["valid_int"] == 42
    assert cleaned["valid_float"] == 18.5


def test_weather_generation():
    test_date = date(2026, 10, 15)
    weather = fetch_or_generate_weather_data(test_date)
    assert weather["date"] == test_date
    assert weather["temperature_c"] is not None
    assert weather["wind_speed_kmh"] is not None
    assert weather["weather_condition"] in ["Clear", "Light Rain", "Thunderstorm", "Heavy Wind / Gusts", "Dense Fog"]


def test_flight_generation():
    test_date = date(2026, 10, 15)
    weather = fetch_or_generate_weather_data(test_date)
    flights = fetch_or_generate_flight_data(test_date, weather, count=25)
    assert len(flights) == 25
    for f in flights:
        assert f["flight_date"] == test_date
        assert "flight_number" in f
        assert "destination" in f
        # Verify no uncleaned NaN values
        for val in f.values():
            if isinstance(val, float):
                assert not math.isnan(val)


def test_ingest_daily_data_e2e():
    test_date = date(2026, 10, 20)
    result = ingest_daily_data(test_date, flight_count=20)
    assert result["success"] is True
    assert result["flights_ingested"] == 20
    assert result["date"] == "2026-10-20"
