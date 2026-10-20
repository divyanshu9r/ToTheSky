from datetime import datetime, date
import pandas as pd
import pytest
from recommender.llm_controller import (
    optimize_schedule_with_llm,
    save_optimized_schedule_to_db,
    get_optimized_schedule_for_date,
    _build_llm_prompt,
)


def test_build_llm_prompt():
    sample_flights = [
        {
            "flight_number": "DL102",
            "airline": "Delta Air Lines",
            "destination": "LAX",
            "scheduled_departure": "2026-10-07T08:30:00",
            "predicted_delay_min": 35.0,
            "risk_level": "Moderate",
        }
    ]
    weather = {
        "weather_condition": "Thunderstorm",
        "temperature_c": 19.0,
        "wind_speed_kmh": 42.0,
        "precipitation_mm": 20.0,
        "visibility_km": 3.0,
    }

    prompt = _build_llm_prompt(sample_flights, weather)
    assert "DL102" in prompt
    assert "Thunderstorm" in prompt
    assert "42.0" in prompt or "42" in prompt
    assert "change_reason" in prompt


def test_llm_controller_optimization_and_explanations():
    # Use None for id so the FK lookup falls back to querying the DB
    # (or inserts with NULL flight_id, which is allowed — nullable FK)
    flights_df = pd.DataFrame([
        {
            "id": None,
            "flight_number": "UA404",
            "airline": "United Airlines",
            "destination": "SFO",
            "scheduled_departure": datetime(2026, 10, 7, 9, 0),
            "predicted_delay_min": 45.0,
            "risk_level": "High",
        },
        {
            "id": None,
            "flight_number": "AA505",
            "airline": "American Airlines",
            "destination": "ORD",
            "scheduled_departure": datetime(2026, 10, 7, 9, 10),
            "predicted_delay_min": 5.0,
            "risk_level": "Low",
        },
    ])

    weather = {
        "weather_condition": "Heavy Wind / Gusts",
        "wind_speed_kmh": 45.0,
        "precipitation_mm": 2.0,
    }

    result = optimize_schedule_with_llm(flights_df, weather)
    assert result["success"] is True
    records = result["records"]
    assert len(records) == 2

    # Check high-risk flight adjustment & human-readable reasoning
    ua_flight = next(r for r in records if r["flight_number"] == "UA404")
    assert ua_flight["delay_adjustment_min"] > 0
    assert len(ua_flight["change_reason"]) > 10
    # Must contain human-readable explanation
    assert any(term in ua_flight["change_reason"].lower() for term in ["wind", "delay", "runway", "mitigate", "capacity"])

    # Test database persistence
    test_date = date(2026, 10, 7)
    saved = save_optimized_schedule_to_db(records, test_date)
    assert saved is True

    fetched = get_optimized_schedule_for_date(test_date)
    assert len(fetched) == 2
    assert fetched[0]["flight_number"] in ["UA404", "AA505"]
