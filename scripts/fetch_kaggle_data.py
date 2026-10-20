"""
Kaggle Big Data Ingestion Script for ToTheSky Phase 2.

Downloads large flight and delay datasets from Kaggle using the Kaggle Python API,
extracts raw CSV files into ./data/raw/, and includes robust error handling for missing
credentials, with an optional realistic synthetic generator for offline/local pipeline testing.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

# Ensure parent directory is on sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (
    KAGGLE_USERNAME,
    KAGGLE_KEY,
    RAW_DATA_DIR,
    DEFAULT_AIRPORT_CODE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fetch_kaggle_data")


def verify_kaggle_credentials() -> bool:
    """Checks environment variables and config for Kaggle credentials."""
    username = os.environ.get("KAGGLE_USERNAME") or KAGGLE_USERNAME
    key = os.environ.get("KAGGLE_KEY") or KAGGLE_KEY

    if username and key:
        os.environ["KAGGLE_USERNAME"] = username.strip()
        os.environ["KAGGLE_KEY"] = key.strip()
        return True

    # Check ~/.kaggle/kaggle.json
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        logger.info(f"Found Kaggle credentials in {kaggle_json}")
        return True

    return False


def generate_synthetic_raw_dataset(output_dir: Path, num_rows: int = 15000) -> Path:
    """
    Generates a realistic large flight operations dataset modeled on the US DOT / Kaggle
    flight-delays schema for local training, validation, and offline demonstrations.
    """
    logger.info(f"Synthesizing high-volume raw flight operations dataset ({num_rows:,} rows)...")
    output_dir.mkdir(parents=True, exist_ok=True)
    flights_file = output_dir / "flights.csv"

    np.random.seed(42)

    airlines = ["DL", "AA", "UA", "B6", "WN", "AS", "NK"]
    airports = ["JFK", "LAX", "ORD", "ATL", "DFW", "DEN", "SFO", "SEA", "MIA", "BOS"]

    years = np.full(num_rows, 2026)
    months = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], size=num_rows)
    days = np.random.randint(1, 29, size=num_rows)
    day_of_week = np.random.randint(1, 8, size=num_rows)

    carrier = np.random.choice(airlines, size=num_rows, p=[0.22, 0.20, 0.18, 0.12, 0.15, 0.08, 0.05])
    flight_nums = np.random.randint(100, 4999, size=num_rows)

    origin = np.random.choice(airports, size=num_rows)
    dest = np.random.choice(airports, size=num_rows)
    # Ensure origin != dest
    same_mask = origin == dest
    dest[same_mask] = np.where(origin[same_mask] == "JFK", "LAX", "JFK")

    # Scheduled departure in hhmm format
    hours = np.random.randint(6, 23, size=num_rows)
    minutes = np.random.choice([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55], size=num_rows)
    sched_dep = hours * 100 + minutes

    scheduled_duration = np.random.choice([75, 130, 180, 240, 310, 360], size=num_rows)

    # Physics-informed realistic delay distribution
    is_peak = ((hours >= 8) & (hours <= 10)) | ((hours >= 17) & (hours <= 20))
    weather_event_prob = np.random.uniform(0.0, 1.0, size=num_rows)

    weather_delay = np.zeros(num_rows)
    severe_weather = weather_event_prob > 0.85
    moderate_weather = (weather_event_prob > 0.70) & (~severe_weather)

    weather_delay[severe_weather] = np.random.exponential(scale=45.0, size=np.sum(severe_weather)) + 20
    weather_delay[moderate_weather] = np.random.uniform(10.0, 35.0, size=np.sum(moderate_weather))
    weather_delay = np.round(weather_delay, 1)

    nas_delay = np.where(is_peak, np.random.exponential(scale=18.0, size=num_rows), 0.0)
    carrier_delay = np.random.choice([0.0, 5.0, 15.0, 30.0], size=num_rows, p=[0.70, 0.15, 0.10, 0.05])

    total_dep_delay = np.maximum(0.0, weather_delay + nas_delay + carrier_delay + np.random.normal(0, 4, num_rows))
    total_dep_delay = np.round(total_dep_delay, 1)

    df = pd.DataFrame({
        "YEAR": years,
        "MONTH": months,
        "DAY": days,
        "DAY_OF_WEEK": day_of_week,
        "AIRLINE": carrier,
        "FLIGHT_NUMBER": flight_nums,
        "ORIGIN_AIRPORT": origin,
        "DESTINATION_AIRPORT": dest,
        "SCHEDULED_DEPARTURE": sched_dep,
        "DEPARTURE_DELAY": total_dep_delay,
        "SCHEDULED_TIME": scheduled_duration,
        "WEATHER_DELAY": weather_delay,
        "AIR_SYSTEM_DELAY": np.round(nas_delay, 1),
        "AIRLINE_DELAY": carrier_delay,
        "CANCELLED": np.where(weather_delay > 90, 1, 0),
    })

    df.to_csv(flights_file, index=False)
    logger.info(f"Synthetic raw dataset saved successfully to {flights_file} ({flights_file.stat().st_size / (1024*1024):.2f} MB)")
    return flights_file


def download_and_extract_kaggle_data(
    dataset_identifier: str = "us-dot/flight-delays",
    raw_dir: Path = RAW_DATA_DIR,
    fallback_synthetic: bool = True,
    synthetic_rows: int = 15000,
) -> Dict[str, Any]:
    """
    Downloads dataset from Kaggle into raw_dir.
    Falls back to generating synthetic big-data operations if Kaggle credentials are missing.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)

    has_credentials = verify_kaggle_credentials()

    if not has_credentials:
        warning_msg = (
            "Kaggle API credentials not found. To download real datasets from Kaggle:\n"
            "  1. Set KAGGLE_USERNAME and KAGGLE_KEY in your .env file, or\n"
            "  2. Place kaggle.json inside ~/.kaggle/kaggle.json\n"
        )
        logger.warning(warning_msg)

        if fallback_synthetic:
            logger.info("Initializing fallback: Generating high-volume Big Data dataset locally...")
            file_path = generate_synthetic_raw_dataset(raw_dir, num_rows=synthetic_rows)
            return {
                "success": True,
                "source": "Synthetic Big Data Generator",
                "raw_dir": str(raw_dir),
                "dataset": "synthetic/flight-delays",
                "files": [file_path.name],
            }
        else:
            return {
                "success": False,
                "error": "Missing Kaggle credentials and fallback is disabled.",
                "raw_dir": str(raw_dir),
            }

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        logger.info(f"Authenticated with Kaggle. Downloading dataset '{dataset_identifier}' to {raw_dir}...")
        api.dataset_download_files(dataset_identifier, path=str(raw_dir), unzip=True)
        files = [f.name for f in raw_dir.iterdir() if f.is_file()]
        logger.info(f"Kaggle download & extraction complete. Extracted files: {files}")
        return {
            "success": True,
            "source": f"Kaggle API ({dataset_identifier})",
            "raw_dir": str(raw_dir),
            "dataset": dataset_identifier,
            "files": files,
        }
    except Exception as e:
        logger.error(f"Kaggle download failed: {e}")
        if fallback_synthetic:
            logger.info("Falling back to local high-volume synthetic flight data generator...")
            file_path = generate_synthetic_raw_dataset(raw_dir, num_rows=synthetic_rows)
            return {
                "success": True,
                "source": "Synthetic Fallback Generator",
                "raw_dir": str(raw_dir),
                "dataset": "synthetic/flight-delays",
                "files": [file_path.name],
                "warning": str(e),
            }
        return {
            "success": False,
            "error": str(e),
            "raw_dir": str(raw_dir),
        }


def main():
    parser = argparse.ArgumentParser(description="ToTheSky: Big Data Ingestion from Kaggle")
    parser.add_argument("--dataset", type=str, default="us-dot/flight-delays", help="Kaggle dataset identifier")
    parser.add_argument("--output_dir", type=str, default=str(RAW_DATA_DIR), help="Output raw data directory")
    parser.add_argument("--no_fallback", action="store_true", help="Disable synthetic fallback if Kaggle auth fails")
    parser.add_argument("--synthetic_rows", type=int, default=15000, help="Row count for synthetic dataset")

    args = parser.parse_args()

    result = download_and_extract_kaggle_data(
        dataset_identifier=args.dataset,
        raw_dir=Path(args.output_dir),
        fallback_synthetic=not args.no_fallback,
        synthetic_rows=args.synthetic_rows,
    )

    if result.get("success"):
        print(f"\n[SUCCESS] Ingestion complete.")
        print(f"Source: {result.get('source')}")
        print(f"Location: {result.get('raw_dir')}")
        print(f"Extracted files: {result.get('files')}\n")
    else:
        print(f"\n[FAILED] {result.get('error')}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
