import json
from pathlib import Path
from datetime import datetime, date
import pandas as pd
import pytest

from config import RAW_DATA_DIR, SFT_DATASET_PATH, LORA_ADAPTER_DIR
from scripts.fetch_kaggle_data import download_and_extract_kaggle_data, generate_synthetic_raw_dataset
from scripts.prepare_sft_dataset import prepare_sft_dataset, generate_sft_conversation
from scripts.train_llm import train_sft_model, save_mock_lora_adapter
from recommender.llm_controller import optimize_schedule_with_custom_llama, optimize_schedule_with_llm


def test_fetch_kaggle_data_generates_valid_csv(tmp_path):
    test_raw_dir = tmp_path / "raw"
    res = download_and_extract_kaggle_data(raw_dir=test_raw_dir, fallback_synthetic=True, synthetic_rows=100)
    assert res["success"] is True
    flight_file = test_raw_dir / "flights.csv"
    assert flight_file.exists()

    df = pd.read_csv(flight_file)
    assert len(df) == 100
    assert "AIRLINE" in df.columns
    assert "DEPARTURE_DELAY" in df.columns
    assert "SCHEDULED_DEPARTURE" in df.columns


def test_prepare_sft_dataset_chatml_structure(tmp_path):
    test_raw_dir = tmp_path / "raw"
    generate_synthetic_raw_dataset(test_raw_dir, num_rows=100)

    test_output_jsonl = tmp_path / "sft_flight_dataset.jsonl"
    res = prepare_sft_dataset(
        raw_dir=test_raw_dir,
        output_path=test_output_jsonl,
        num_samples=10,
        window_size=3,
    )
    assert res["success"] is True
    assert res["total_conversations"] == 10
    assert test_output_jsonl.exists()

    # Validate JSONL content and ChatML schema
    with open(test_output_jsonl, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) == 10
        first_conv = json.loads(lines[0])

        assert "messages" in first_conv
        messages = first_conv["messages"]
        assert len(messages) == 3

        assert messages[0]["role"] == "system"
        assert "Air Traffic Control Dispatcher" in messages[0]["content"]

        assert messages[1]["role"] == "user"
        assert "Operational Dispatch Request" in messages[1]["content"]

        assert messages[2]["role"] == "assistant"
        assert "De-conflicted Departure" in messages[2]["content"]


def test_train_sft_model_dry_run(tmp_path):
    # Setup test dataset
    test_dataset = tmp_path / "test_sft.jsonl"
    with open(test_dataset, "w", encoding="utf-8") as f:
        sample = {
            "messages": [
                {"role": "system", "content": "You are an AI Air Traffic Control Dispatcher."},
                {"role": "user", "content": "Test input schedule"},
                {"role": "assistant", "content": "Test output schedule"},
            ]
        }
        for _ in range(5):
            f.write(json.dumps(sample) + "\n")

    test_adapter_dir = tmp_path / "test_adapter"
    res = train_sft_model(
        dataset_path=test_dataset,
        output_dir=test_adapter_dir,
        max_samples=5,
        dry_run=True,
    )
    assert res["success"] is True
    assert (test_adapter_dir / "adapter_config.json").exists()

    with open(test_adapter_dir / "adapter_config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
        assert cfg["peft_type"] == "LORA"
        assert cfg["r"] == 16


def test_dual_model_inference_routing():
    flights_df = pd.DataFrame([
        {
            "id": None,
            "flight_number": "DL999",
            "airline": "Delta Air Lines",
            "destination": "LAX",
            "scheduled_departure": datetime(2026, 10, 7, 8, 30),
            "predicted_delay_min": 35.0,
            "risk_level": "Moderate",
        }
    ])
    weather = {
        "weather_condition": "Thunderstorm",
        "temperature_c": 21.0,
        "wind_speed_kmh": 40.0,
        "precipitation_mm": 15.0,
        "visibility_km": 4.0,
    }

    # Test Local LoRA model routing
    res_lora = optimize_schedule_with_llm(
        flights_df=flights_df,
        weather_dict=weather,
        model_engine="Custom Fine-Tuned Llama-3.2 (Local LoRA)",
    )
    assert res_lora["success"] is True
    assert "Llama" in res_lora["source"] or "LoRA" in res_lora["source"]
    assert len(res_lora["records"]) == 1
    assert res_lora["records"][0]["flight_number"] == "DL999"

    # Test direct custom llama function
    res_direct = optimize_schedule_with_custom_llama(flights_df, weather)
    assert res_direct["success"] is True
    assert len(res_direct["records"]) == 1


def test_train_online_gemini_dataframe_format(tmp_path):
    from scripts.train_online_gemini import load_or_generate_tuning_data, start_gemini_tuning

    # Prepare small sample dataset
    test_jsonl = tmp_path / "test_sft.jsonl"
    with open(test_jsonl, "w", encoding="utf-8") as f:
        for i in range(15):
            sample = {
                "messages": [
                    {"role": "system", "content": "You are an AI Air Traffic Control Dispatcher."},
                    {"role": "user", "content": f"Schedule request {i}"},
                    {"role": "assistant", "content": f"Deconfliction plan {i}"},
                ]
            }
            f.write(json.dumps(sample) + "\n")

    df = load_or_generate_tuning_data(dataset_path=test_jsonl, target_count=10)
    assert len(df) == 10
    assert "text_input" in df.columns
    assert "output" in df.columns
    assert "Schedule request" in df.iloc[0]["text_input"]
    assert "Deconfliction plan" in df.iloc[0]["output"]

    # Test dry run
    res = start_gemini_tuning(
        api_key="test_key",
        dataset_path=test_jsonl,
        num_samples=10,
        dry_run=True,
        auto_update_env=False,
    )
    assert res["success"] is True
    assert "tunedModels/" in res["tuned_model_id"]
    assert res["samples_count"] == 10


def test_compare_rag_vs_finetuned():
    from recommender.llm_controller import compare_rag_vs_finetuned

    sample_flights = [
        {
            "flight_number": "AA102",
            "airline": "American Airlines",
            "destination": "ORD",
            "scheduled_departure": datetime(2026, 10, 7, 8, 30),
            "predicted_delay_min": 35.0,
            "risk_level": "High",
        },
        {
            "flight_number": "DL404",
            "airline": "Delta Air Lines",
            "destination": "LAX",
            "scheduled_departure": datetime(2026, 10, 7, 8, 45),
            "predicted_delay_min": 10.0,
            "risk_level": "Low",
        },
    ]

    weather = {
        "weather_condition": "Thunderstorm",
        "temperature_c": 22.0,
        "wind_speed_kmh": 45.0,
        "precipitation_mm": 18.0,
        "visibility_km": 3.0,
    }

    res = compare_rag_vs_finetuned(
        flights_batch=sample_flights,
        weather_dict=weather,
        tuned_model_id="tunedModels/test-atc-tuned",
    )

    assert "rag" in res
    assert "finetuned" in res
    assert "token_savings_pct" in res
    assert res["token_savings_pct"] > 50  # Fine-tuned prompt is dramatically smaller
    assert len(res["rag"]["records"]) == 2
    assert len(res["finetuned"]["records"]) == 2

