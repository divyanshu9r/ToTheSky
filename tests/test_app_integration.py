from datetime import date
import pandas as pd
import pytest
from app import load_day_data
from ml.predict import predict_flight_delays
from recommender.llm_controller import optimize_schedule_with_llm


def test_dashboard_end_to_end_flow():
    test_date = date(2026, 10, 7)
    weather, raw_flights_df, opt_df = load_day_data(test_date)

    # Weather and flights exist
    assert weather is not None
    assert not raw_flights_df.empty
    assert len(raw_flights_df) > 0

    # Predictions run seamlessly
    enriched_df = predict_flight_delays(raw_flights_df, weather)
    assert "predicted_delay_min" in enriched_df.columns
    assert "risk_level" in enriched_df.columns

    # Optimization run seamlessly
    opt_res = optimize_schedule_with_llm(enriched_df, weather)
    assert opt_res["success"] is True
    assert len(opt_res["records"]) > 0
