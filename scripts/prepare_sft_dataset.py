"""
Dataset Preparation & Supervised Fine-Tuning (SFT) Formatting Script for ToTheSky Phase 2.

Loads and cleans raw flight and delay datasets from ./data/raw/, simulates airport meteorological
conditions, computes XGBoost delay risk features, and compiles instruction-tuning pairs in
standard OpenAI / ChatML JSONL format (data/sft_flight_dataset.jsonl) for SFT training.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

# Ensure parent directory is on sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (
    RAW_DATA_DIR,
    SFT_DATASET_PATH,
    AIRPORT_RUNWAYS,
    DEFAULT_AIRPORT_CODE,
)
from scripts.fetch_kaggle_data import download_and_extract_kaggle_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_sft_dataset")

SYSTEM_PROMPT = "You are an AI Air Traffic Control Dispatcher."


def find_raw_flight_file(raw_dir: Path) -> Optional[Path]:
    """Finds flights.csv or any matching CSV file in raw_dir."""
    candidates = list(raw_dir.glob("flights*.csv")) + list(raw_dir.glob("*.csv"))
    if candidates:
        return candidates[0]
    return None


def generate_sft_conversation(
    flights_batch: List[Dict[str, Any]],
    weather: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Creates a single ChatML / OpenAI JSONL instruction conversation:
    - system: AI Air Traffic Control Dispatcher
    - user: raw flight schedules, weather, XGBoost delay risk predictions
    - assistant: deconflicted schedule adjustment & step-by-step operational rationale
    """
    hub = weather.get("airport_code", DEFAULT_AIRPORT_CODE)
    cond = weather.get("weather_condition", "Clear")
    temp = weather.get("temperature_c", 20.0)
    wind = weather.get("wind_speed_kmh", 15.0)
    precip = weather.get("precipitation_mm", 0.0)
    vis = weather.get("visibility_km", 10.0)

    # Format user prompt
    flight_lines = []
    for f in flights_batch:
        f_num = f["flight_number"]
        carrier = f.get("airline_name", f.get("airline", "Carrier"))
        dest = f["destination"]
        dep_str = f["scheduled_departure"].strftime("%H:%M")
        pred_delay = f["predicted_delay_min"]
        risk = f["risk_level"]
        runway = f.get("assigned_runway", AIRPORT_RUNWAYS[0])
        flight_lines.append(
            f"- Flight {f_num} ({carrier}) to {dest} at {dep_str} | Predicted Delay: {pred_delay:.1f}m ({risk} Risk) | Assigned: {runway}"
        )

    flights_text = "\n".join(flight_lines)

    user_content = (
        f"Operational Dispatch Request:\n"
        f"Hub Airport: {hub}\n"
        f"Current Meteorological Conditions:\n"
        f"- Weather: {cond}\n"
        f"- Temperature: {temp:.1f}°C\n"
        f"- Wind Speed: {wind:.1f} km/h (Crosswind component active)\n"
        f"- Precipitation: {precip:.1f} mm\n"
        f"- Visibility: {vis:.1f} km\n"
        f"- Available Runways: {', '.join(AIRPORT_RUNWAYS)}\n\n"
        f"Flights Scheduled in Operational Window:\n"
        f"{flights_text}\n\n"
        f"Task:\n"
        f"Provide a de-conflicted flight timetable with step-by-step dispatch rationales to avoid cascading delay bottlenecks and maintain safe runway wake separation."
    )

    # Build assistant deconflicted response
    assistant_lines = ["AI Dispatch Optimization & Deconfliction Plan:\n"]
    runway_clock = {}

    for i, f in enumerate(flights_batch):
        f_num = f["flight_number"]
        dest = f["destination"]
        orig_dep = f["scheduled_departure"]
        orig_time_str = orig_dep.strftime("%H:%M")
        pred_delay = f["predicted_delay_min"]
        risk = f["risk_level"]

        # Run deconfliction logic
        runway = AIRPORT_RUNWAYS[i % len(AIRPORT_RUNWAYS)]

        if risk == "High" or pred_delay >= 35.0:
            shift_min = 25 + (int(pred_delay) % 15)
            new_dep = orig_dep + timedelta(minutes=shift_min)
            rationale = (
                f"Pushed back by {shift_min} mins to absorb severe {pred_delay:.0f}m predicted delay under {cond.lower()} "
                f"({wind:.1f} km/h winds, {precip:.1f} mm rain). Mitigates runway queue saturation on {runway}."
            )
        elif risk == "Moderate" or pred_delay >= 15.0:
            shift_min = 15
            new_dep = orig_dep + timedelta(minutes=shift_min)
            rationale = (
                f"Buffered by 15 mins to maintain minimum 4-minute wake turbulence separation on {runway} and prevent apron choke points."
            )
        else:
            shift_min = 0
            new_dep = orig_dep
            rationale = (
                f"Cleared on-schedule ({orig_time_str}) via {runway}. Weather parameters and aircraft turnaround within nominal tolerances."
            )

        # Enforce runway separation against previous flight on same runway
        last_time = runway_clock.get(runway)
        if last_time and (new_dep - last_time).total_seconds() < 240:
            extra_spacing = 4
            new_dep = last_time + timedelta(minutes=extra_spacing)
            shift_min += extra_spacing
            rationale += f" Staggered by an additional {extra_spacing} mins for safe departure corridor separation."

        runway_clock[runway] = new_dep
        new_time_str = new_dep.strftime("%H:%M")

        assistant_lines.append(
            f"{i+1}. Flight {f_num} ({hub} -> {dest}):\n"
            f"   - Original Departure: {orig_time_str}\n"
            f"   - De-conflicted Departure: {new_time_str} ({'+' if shift_min > 0 else ''}{shift_min} mins)\n"
            f"   - Assigned Runway: {runway}\n"
            f"   - Dispatch Rationale: {rationale}\n"
        )

    assistant_content = "\n".join(assistant_lines)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def prepare_sft_dataset(
    raw_dir: Path = RAW_DATA_DIR,
    output_path: Path = SFT_DATASET_PATH,
    num_samples: int = 1500,
    window_size: int = 4,
) -> Dict[str, Any]:
    """
    Cleans raw flight data and creates an instruction-tuning JSONL dataset.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    csv_path = find_raw_flight_file(raw_dir)
    if csv_path is None or not csv_path.exists():
        logger.info("Raw flight data not found in raw_dir. Initiating automatic fetch/synthesis...")
        download_and_extract_kaggle_data(raw_dir=raw_dir)
        csv_path = find_raw_flight_file(raw_dir)

    if csv_path is None:
        raise FileNotFoundError(f"Could not locate or generate raw flight CSV in {raw_dir}")

    logger.info(f"Loading raw flight data from {csv_path}...")
    df = pd.read_csv(csv_path, nrows=num_samples * window_size * 2)

    # Normalize column names
    col_map = {c.upper(): c for c in df.columns}
    airline_col = col_map.get("AIRLINE") or col_map.get("CARRIER") or "AIRLINE"
    flight_col = col_map.get("FLIGHT_NUMBER") or col_map.get("FLIGHT_NUM") or "FLIGHT_NUMBER"
    dest_col = col_map.get("DESTINATION_AIRPORT") or col_map.get("DEST") or "DESTINATION_AIRPORT"
    delay_col = col_map.get("DEPARTURE_DELAY") or col_map.get("DEP_DELAY") or "DEPARTURE_DELAY"
    sched_col = col_map.get("SCHEDULED_DEPARTURE") or col_map.get("CRS_DEP_TIME") or "SCHEDULED_DEPARTURE"

    logger.info(f"Using columns: carrier={airline_col}, flight={flight_col}, dest={dest_col}, delay={delay_col}")

    # Clean records
    df = df.dropna(subset=[airline_col, flight_col, dest_col])
    carrier_names = {
        "DL": "Delta Air Lines",
        "AA": "American Airlines",
        "UA": "United Airlines",
        "B6": "JetBlue Airways",
        "WN": "Southwest Airlines",
        "AS": "Alaska Airlines",
        "NK": "Spirit Airlines",
    }

    # Generate meteorological condition presets
    weather_scenarios = [
        {"weather_condition": "Clear", "temperature_c": 22.0, "wind_speed_kmh": 12.0, "precipitation_mm": 0.0, "visibility_km": 10.0},
        {"weather_condition": "Thunderstorm", "temperature_c": 24.5, "wind_speed_kmh": 46.0, "precipitation_mm": 22.0, "visibility_km": 3.0},
        {"weather_condition": "Dense Fog", "temperature_c": 11.0, "wind_speed_kmh": 6.0, "precipitation_mm": 0.5, "visibility_km": 1.2},
        {"weather_condition": "Heavy Wind / Gusts", "temperature_c": 16.0, "wind_speed_kmh": 58.0, "precipitation_mm": 4.0, "visibility_km": 8.0},
        {"weather_condition": "Moderate Rain", "temperature_c": 14.0, "wind_speed_kmh": 26.0, "precipitation_mm": 9.0, "visibility_km": 6.0},
    ]

    conversations = []
    total_rows = len(df)
    base_date = datetime(2026, 10, 7)

    idx = 0
    while len(conversations) < num_samples and (idx + window_size) <= total_rows:
        batch_df = df.iloc[idx : idx + window_size]
        idx += window_size

        weather = weather_scenarios[len(conversations) % len(weather_scenarios)].copy()
        weather["airport_code"] = DEFAULT_AIRPORT_CODE

        # Build flight records for this window
        flights_batch = []
        base_hour = 6 + (len(conversations) % 16)
        base_time = base_date.replace(hour=base_hour, minute=0, second=0)

        for offset, (_, row) in enumerate(batch_df.iterrows()):
            airline_code = str(row.get(airline_col, "DL")).strip()
            f_num = f"{airline_code}{int(row.get(flight_col, 100)) % 9000}"
            dest = str(row.get(dest_col, "LAX")).strip()[:3]
            raw_delay = float(row.get(delay_col, 0.0)) if pd.notna(row.get(delay_col)) else 0.0

            # Compute weather-adjusted delay
            w_impact = (weather["wind_speed_kmh"] / 20.0) * 10.0 + (weather["precipitation_mm"] * 1.5)
            pred_delay = max(0.0, round(raw_delay * 0.6 + w_impact * 0.7, 1))

            risk = "High" if pred_delay >= 35.0 else ("Moderate" if pred_delay >= 15.0 else "Low")
            sched_dep = base_time + timedelta(minutes=offset * 8)

            flights_batch.append({
                "flight_number": f_num,
                "airline": airline_code,
                "airline_name": carrier_names.get(airline_code, f"{airline_code} Airlines"),
                "destination": dest,
                "scheduled_departure": sched_dep,
                "predicted_delay_min": pred_delay,
                "risk_level": risk,
                "assigned_runway": AIRPORT_RUNWAYS[offset % len(AIRPORT_RUNWAYS)],
            })

        conv = generate_sft_conversation(flights_batch, weather)
        conversations.append(conv)

    # Write JSONL file
    with open(output_path, "w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    logger.info(f"Successfully generated {len(conversations):,} SFT conversation pairs in {output_path}")
    logger.info(f"Output file size: {output_path.stat().st_size / (1024*1024):.2f} MB")

    return {
        "success": True,
        "total_conversations": len(conversations),
        "output_path": str(output_path),
        "file_size_mb": round(output_path.stat().st_size / (1024 * 1024), 2),
    }


def main():
    parser = argparse.ArgumentParser(description="ToTheSky: SFT Dataset Preparation")
    parser.add_argument("--raw_dir", type=str, default=str(RAW_DATA_DIR), help="Directory containing raw flight CSVs")
    parser.add_argument("--output_file", type=str, default=str(SFT_DATASET_PATH), help="Output JSONL file path")
    parser.add_argument("--num_samples", type=int, default=1500, help="Number of instruction-tuning conversation pairs")
    parser.add_argument("--window_size", type=int, default=4, help="Flights per dispatch window")

    args = parser.parse_args()

    result = prepare_sft_dataset(
        raw_dir=Path(args.raw_dir),
        output_path=Path(args.output_file),
        num_samples=args.num_samples,
        window_size=args.window_size,
    )

    print(f"\n[SUCCESS] SFT Dataset Created.")
    print(f"Total Conversations: {result['total_conversations']}")
    print(f"Destination: {result['output_path']}")
    print(f"Size: {result['file_size_mb']} MB\n")


if __name__ == "__main__":
    main()
