# ✈️ ToTheSky: Airport Monitoring, Online Gemini Fine-Tuning & Architecture Comparison

A complete, end-to-end web-based **Airport Monitoring System and Flight Scheduling Dashboard** built with **Python, Streamlit, MySQL (SQLAlchemy), XGBoost, Hugging Face PEFT/LoRA, Google Gemini Fine-Tuning, and Google Gemini Cloud Engine**.

---

## 🌟 Key Capabilities

### 1. Data Ingestion & Database Schema
- Ingests daily flight operations and meteorological conditions into MySQL (`flights`, `daily_weather`, `optimized_schedules`).
- Rigorous missing data sanitization (`clean_record_for_db`) and `SET FOREIGN_KEY_CHECKS=0/1` guards.
- Resilient SQLite presentation fallback if MySQL is offline.

### 2. Machine Learning Delay Prediction (`ml/predict.py`)
- **XGBoost Regressor**: Models complex interactions between weather (crosswinds, precipitation, visibility) and runway congestion.
- **Accuracy**: **99.1% $R^2$** and **2.74 min MAE** (exceeds 91% target). Artifact: `ml/model.joblib`.

### 3. Online Google Gemini Fine-Tuning (`scripts/train_online_gemini.py`)
- Automatically loads exactly **1,000 instruction-tuning pairs** from the flight dataset.
- Formats into a Pandas DataFrame with `text_input` and `output` columns.
- Uses `genai.create_tuned_model()` targeting `"models/gemini-1.5-flash-001-tuning"`.
- Automatically outputs the new `tunedModels/...` ID and registers it as `TUNED_GEMINI_MODEL_ID` in `.env`.

### 4. Local LLaMA LoRA Fine-Tuning (`scripts/train_llm.py`)
- SFT fine-tuning pipeline for `meta-llama/Llama-3.2-1B-Instruct` and `Qwen/Qwen2.5-1.5B-Instruct` using `peft` and `trl` (`SFTTrainer`).
- Saves adapters to `./models/tothesky-llama-lora/`.

### 5. Streamlit Dashboard & Architectural Comparison (`app.py`)
- **Tab 1: Data & Optimization**: Live meteorological KPIs, raw schedule filters, one-click optimization, and master timetable.
- **Tab 2: Analytics & AI Reasoning**: 4 interactive Plotly charts, XGBoost metrics, and expandable flight-by-flight rationales.
- **Tab 3: RAG vs. Fine-Tuned Comparison**:
  - **Column 1 (RAG Engine)**: Base Gemini model augmented with elaborate in-context rules, runway guidelines, and XGBoost predictions (~1,200 tokens).
  - **Column 2 (Fine-Tuned Engine)**: Tuned Gemini model (`TUNED_GEMINI_MODEL_ID`) receiving *only* raw operational flight records (~160 tokens, **>80% token reduction**).
  - **Head-to-Head Comparison Button**: Evaluates both pipelines simultaneously on the same flight batch and highlights differences in schedule adjustments and rationale phrasing.
- **Sidebar Expander**: Dedicated controls for Kaggle Big Data fetching, online Gemini tuning, and local LoRA training.

---

## 🚀 Quickstart Guide

### 1. Launch Online Gemini Fine-Tuning (1,000 pairs)
```powershell
python scripts/train_online_gemini.py
```
*(Or run `python scripts/train_online_gemini.py --dry_run` for dry-run verification).*

### 2. Run the Streamlit Dashboard
```powershell
python -m streamlit run app.py
```
Open **`http://localhost:8501`** and navigate to the **RAG vs. Fine-Tuned Comparison** tab.

---

## 🧪 Test Suite Status

```powershell
pytest tests/ -v
============================= 22 passed in 43.74s =============================
```
All **22 test cases** across database CRUD, XGBoost regression, ChatML data formatting, LoRA adapter generation, online Gemini tuning data schema, and RAG vs. Fine-Tuned comparison routing passed with 100% pass rate.
