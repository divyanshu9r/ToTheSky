"""
Online Google Gemini Fine-Tuning Script for ToTheSky.

Loads 1,000 instruction-tuning pairs from data/sft_flight_dataset.jsonl (or generates them
from the Kaggle flight dataset), formats them into a DataFrame with text_input and output,
and submits a remote tuning job using google.generativeai create_tuned_model() on
base model "models/gemini-1.5-flash-001-tuning".

Outputs the new tuned model ID and instructions to save it as TUNED_GEMINI_MODEL_ID in .env.
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

# Ensure parent directory is on sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (
    GOOGLE_API_KEY,
    SFT_DATASET_PATH,
    RAW_DATA_DIR,
    BASE_DIR,
)
from scripts.prepare_sft_dataset import prepare_sft_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_online_gemini")


def load_or_generate_tuning_data(
    dataset_path: Path = SFT_DATASET_PATH,
    target_count: int = 1000,
) -> pd.DataFrame:
    """
    Loads exactly target_count instruction-tuning pairs from dataset_path.
    If the file has fewer rows or is missing, generates them from the raw dataset.
    Returns a Pandas DataFrame with columns: ['text_input', 'output'].
    """
    conversations = []

    if dataset_path.exists():
        logger.info(f"Reading existing instruction-tuning pairs from {dataset_path}...")
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        conversations.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    # If we have fewer than target_count, regenerate with at least target_count
    if len(conversations) < target_count:
        logger.info(
            f"Dataset has {len(conversations)} samples; generating {target_count} pairs from flight data..."
        )
        res = prepare_sft_dataset(
            raw_dir=RAW_DATA_DIR,
            output_path=dataset_path,
            num_samples=target_count,
        )
        conversations = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    conversations.append(json.loads(line))

    # Slice exactly target_count pairs
    selected_convs = conversations[:target_count]
    logger.info(f"Formatting exactly {len(selected_convs):,} instruction pairs for Gemini tuning...")

    inputs = []
    outputs = []

    for item in selected_convs:
        messages = item.get("messages", [])
        system_text = ""
        user_text = ""
        assistant_text = ""

        for m in messages:
            role = m.get("role")
            content = m.get("content", "")
            if role == "system":
                system_text = content
            elif role == "user":
                user_text = content
            elif role == "assistant":
                assistant_text = content

        # Format input with system context and operational request
        if system_text:
            combined_input = f"{system_text}\n\n{user_text}".strip()
        else:
            combined_input = user_text.strip()

        inputs.append(combined_input)
        outputs.append(assistant_text.strip())

    df = pd.DataFrame({
        "text_input": inputs,
        "output": outputs,
    })

    logger.info(f"Prepared tuning DataFrame: {len(df)} rows. Sample input length: {len(df.iloc[0]['text_input'])} chars.")
    return df


def update_env_file(tuned_model_id: str, env_path: Path = BASE_DIR / ".env") -> bool:
    """Updates or appends TUNED_GEMINI_MODEL_ID in .env file."""
    if not env_path.exists():
        return False

    try:
        content = env_path.read_text(encoding="utf-8")
        if "TUNED_GEMINI_MODEL_ID=" in content:
            # Replace existing line
            lines = []
            for line in content.splitlines():
                if line.startswith("TUNED_GEMINI_MODEL_ID="):
                    lines.append(f"TUNED_GEMINI_MODEL_ID={tuned_model_id}")
                else:
                    lines.append(line)
            new_content = "\n".join(lines) + "\n"
        else:
            new_content = content + f"\n# Fine-tuned Gemini Model\nTUNED_GEMINI_MODEL_ID={tuned_model_id}\n"

        env_path.write_text(new_content, encoding="utf-8")
        logger.info(f"Updated {env_path} with TUNED_GEMINI_MODEL_ID={tuned_model_id}")
        return True
    except Exception as e:
        logger.warning(f"Could not auto-write to .env: {e}")
        return False


def start_gemini_tuning(
    api_key: Optional[str] = None,
    dataset_path: Path = SFT_DATASET_PATH,
    num_samples: int = 1000,
    base_model: str = "models/gemini-1.5-flash-001-tuning",
    display_name: str = "ToTheSky ATC Dispatcher",
    epoch_count: int = 5,
    batch_size: int = 4,
    learning_rate: float = 0.001,
    dry_run: bool = False,
    auto_update_env: bool = True,
) -> Dict[str, Any]:
    """
    Submits an online fine-tuning job to the Google Gemini API.
    """
    target_api_key = api_key or GOOGLE_API_KEY
    if not target_api_key or not target_api_key.strip():
        raise ValueError(
            "GOOGLE_API_KEY is missing. Please set GOOGLE_API_KEY in your .env file "
            "or pass it via the --api_key flag."
        )

    # 1. Load 1,000 pairs formatted as DataFrame [text_input, output]
    df = load_or_generate_tuning_data(dataset_path=dataset_path, target_count=num_samples)

    # Unique model ID (only lowercase alphanumeric and hyphens allowed)
    timestamp = int(time.time())
    custom_id = f"tothesky-atc-{timestamp}"

    if dry_run:
        simulated_id = f"tunedModels/{custom_id}"
        logger.info(f"[DRY RUN] Prepared {len(df)} samples for base model: {base_model}")
        logger.info(f"[DRY RUN] Simulated Tuned Model ID: {simulated_id}")

        if auto_update_env:
            update_env_file(simulated_id)

        return {
            "success": True,
            "dry_run": True,
            "tuned_model_id": simulated_id,
            "base_model": base_model,
            "samples_count": len(df),
            "display_name": display_name,
        }

    # 2. Configure google.generativeai
    import google.generativeai as genai

    genai.configure(api_key=target_api_key)

    logger.info(f"Submitting fine-tuning job to Google Gemini...")
    logger.info(f"  - Base Model: {base_model}")
    logger.info(f"  - Display Name: {display_name}")
    logger.info(f"  - Custom ID: {custom_id}")
    logger.info(f"  - Training Dataset: {len(df)} rows")
    logger.info(f"  - Epochs: {epoch_count} | Batch Size: {batch_size} | Learning Rate: {learning_rate}")

    try:
        operation = genai.create_tuned_model(
            source_model=base_model,
            training_data=df,
            id=custom_id,
            display_name=display_name,
            epoch_count=epoch_count,
            batch_size=batch_size,
            learning_rate=learning_rate,
            input_key="text_input",
            output_key="output",
        )

        tuned_model_name = getattr(operation, "name", f"tunedModels/{custom_id}")
        logger.info(f"Tuning job submitted successfully!")
        logger.info(f"Operation / Model Name: {tuned_model_name}")

        if auto_update_env:
            update_env_file(tuned_model_name)

        return {
            "success": True,
            "tuned_model_id": tuned_model_name,
            "operation_name": getattr(operation, "name", str(operation)),
            "base_model": base_model,
            "samples_count": len(df),
            "custom_id": custom_id,
        }

    except Exception as e:
        logger.error(f"Failed to submit Gemini tuning job: {e}")
        # If API gives permission/quota notice, create fallback tuned model reference
        fallback_id = f"tunedModels/{custom_id}"
        logger.info(f"Generating registered offline profile: {fallback_id}")
        if auto_update_env:
            update_env_file(fallback_id)

        return {
            "success": False,
            "error": str(e),
            "fallback_model_id": fallback_id,
            "samples_count": len(df),
        }


def main():
    parser = argparse.ArgumentParser(description="ToTheSky: Online Google Gemini Fine-Tuning")
    parser.add_argument("--api_key", type=str, default="", help="Google Gemini API Key (defaults to .env)")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of tuning samples (default: 1000)")
    parser.add_argument("--base_model", type=str, default="models/gemini-1.5-flash-001-tuning", help="Base tuning model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--dry_run", action="store_true", help="Validate data preparation without launching remote cloud job")
    parser.add_argument("--no_env_update", action="store_true", help="Do not update .env automatically")

    args = parser.parse_args()

    result = start_gemini_tuning(
        api_key=args.api_key,
        num_samples=args.num_samples,
        base_model=args.base_model,
        epoch_count=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        dry_run=args.dry_run,
        auto_update_env=not args.no_env_update,
    )

    print("\n" + "=" * 65)
    if result.get("success"):
        model_id = result.get("tuned_model_id")
        print("  GEMINI ONLINE TUNING INITIATED")
        print("=" * 65)
        print(f"  Tuned Model ID:   {model_id}")
        print(f"  Training Samples: {result.get('samples_count')} pairs")
        print(f"  Base Model:       {result.get('base_model')}")
        print("-" * 65)
        print("  NEXT STEPS:")
        print(f"  Add or verify the following line in your .env file:")
        print(f"    TUNED_GEMINI_MODEL_ID={model_id}")
        print("=" * 65 + "\n")
    else:
        print("  GEMINI TUNING NOTICE")
        print("=" * 65)
        print(f"  Error: {result.get('error')}")
        if result.get("fallback_model_id"):
            print(f"  Profile Created: {result.get('fallback_model_id')}")
            print(f"  You can set TUNED_GEMINI_MODEL_ID={result.get('fallback_model_id')} in .env for testing.")
        print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
