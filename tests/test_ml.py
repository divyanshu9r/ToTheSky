from datetime import datetime, date
import pandas as pd
import pytest
from ml.predict import (
    generate_training_data,
    train_delay_model,
    load_delay_model,
    predict_flight_delays,
    get_feature_importances,
    get_model_metrics,
)


def test_generate_training_data():
    df = generate_training_data(n_samples=200)
    assert len(df) == 200
    assert "delay_min" in df.columns
    assert "wind_speed_kmh" in df.columns
    assert "precipitation_mm" in df.columns
    assert "runway_congestion_factor" in df.columns
    assert (df["delay_min"] >= 0).all()


def test_train_delay_model_accuracy():
    # Train with adequate sample size to verify target accuracy
    metrics = train_delay_model(n_samples=2500)
    assert metrics["r2_score"] >= 0.90, f"R2 score was {metrics['r2_score']}, expected >= 0.90"
    assert metrics["target_accuracy_pct"] >= 90.0
    assert metrics["mae_minutes"] < 5.0
    assert "feature_importances" in metrics


def test_model_artifact_loading():
    artifact = load_delay_model()
    assert "model" in artifact
    assert "encoder" in artifact
    assert "feature_columns" in artifact


def test_predict_flight_delays():
    sample_flights = pd.DataFrame([
        {
            "flight_number": "DL101",
            "airline": "Delta Air Lines",
            "destination": "LAX",
            "scheduled_departure": datetime(2026, 10, 7, 8, 30),
            "scheduled_arrival": datetime(2026, 10, 7, 14, 0),
        },
        {
            "flight_number": "AA202",
            "airline": "American Airlines",
            "destination": "ORD",
            "scheduled_departure": datetime(2026, 10, 7, 12, 15),
            "scheduled_arrival": datetime(2026, 10, 7, 14, 45),
        },
    ])

    weather = {
        "temperature_c": 18.0,
        "wind_speed_kmh": 35.0,
        "precipitation_mm": 12.0,
        "visibility_km": 4.5,
        "humidity_pct": 80.0,
    }

    pred_df = predict_flight_delays(sample_flights, weather)
    assert "predicted_delay_min" in pred_df.columns
    assert "risk_level" in pred_df.columns
    assert len(pred_df) == 2
    assert pred_df["risk_level"].iloc[0] in ["Low", "Moderate", "High"]
    assert pred_df["predicted_delay_min"].iloc[0] >= 0.0


def test_feature_importances():
    importances = get_feature_importances()
    assert len(importances) > 0
    assert "precipitation_mm" in importances or "wind_speed_kmh" in importances
