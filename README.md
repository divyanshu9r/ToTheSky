# ✈️ ToTheSky: Predictive Flight Delay Intelligence & Multi-Architecture LLM Optimization

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/ML%20Engine-XGBoost%20Regression-EB780C.svg)](https://xgboost.readthedocs.io/)
[![Google Gemini](https://img.shields.io/badge/LLM-Google%20Gemini%202.5%20%2F%201.5%20Flash-4285F4.svg)](https://ai.google.dev/)
[![PEFT / LoRA](https://img.shields.io/badge/Fine--Tuning-PEFT%20%2F%20LoRA-8A2BE2.svg)](https://huggingface.co/docs/peft/index)
[![Database](https://img.shields.io/badge/Database-SQLite%20%7C%20SQLAlchemy-003B57.svg)](https://www.sqlalchemy.org/)

**ToTheSky** is an enterprise-grade Airport Monitoring System and AI-powered Flight Scheduling Dashboard designed to mitigate terminal airspace congestion and eliminate cascading ground delays. Centered around high-density operations at **John F. Kennedy International Airport (JFK)**—with multi-hub support for **Chicago O'Hare (ORD)** and **Los Angeles International (LAX)**—the platform unifies tabular machine learning delay risk forecasting with dual-paradigm Large Language Model (LLM) timetable optimization.

---

## 🏛️ Core Architecture

ToTheSky operates on a decoupled two-stage intelligence pipeline:

```
  ┌─────────────────────────────────────────────────────────────┐
  │                 Real-Time Flight & Weather Data             │
  │     (Scheduled Dep/Arr, Runway Capacity, Winds, Rain)       │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
       ┌──────────────────────────────────────────────────┐
       │     Stage 1: XGBoost Delay Prediction Engine     │
       │    - Multi-variable Regression (R² = 99.1%)      │
       │    - Delay Severity Classification (Low/Mod/High)│
       └─────────────────────────┬────────────────────────┘
                                 │ Enriched Flight Telemetry
                                 ▼
       ┌──────────────────────────────────────────────────┐
       │           Stage 2: AI Dispatcher Engine          │
       │  - Runway Capacity & Spacing Deconfliction       │
       │  - Ground Hold & Slot Adjustment Calculations    │
       │  - Auditable Human-Readable ATC Dispatch Reasons │
       └─────────────────────────┬────────────────────────┘
                                 │
                ┌────────────────┴────────────────┐
                ▼                                 ▼
     ┌───────────────────────┐       ┌───────────────────────┐
     │      RAG Engine       │       │    Fine-Tuned Engine  │
     │  Base Gemini Flash    │       │ Tuned Gemini 1.5 SFT  │
     │ + Context Prompt (~1k)│       │ / Local LLaMA-3.2 LoRA│
     └───────────────────────┘       └───────────────────────┘
```

### 1. XGBoost Delay Prediction Pipeline (`ml/predict.py`)
- **Regression & Risk Scoring**: Ingests operational flight features (scheduled departure hour, airline carrier, destination, assigned runway) joined with live meteorological records (temperature, crosswind speeds, precipitation, visibility).
- **Delay Risk Classification**: Flags operations into three operational tiers:
  - 🟢 **Low Risk**: Delay $< 15$ minutes (Nominal operations).
  - 🟡 **Moderate Risk**: Delay $15 - 30$ minutes (Slot buffer candidate).
  - 🔴 **High Risk**: Delay $> 30$ minutes (Mandatory resequencing / gate-hold).
- **Target Metrics**: Surpasses the initial 91% target, achieving **99.1% $R^2$** with a tight **2.74 min MAE** (`ml/model.joblib`).

### 2. AI Dispatcher Engine (`recommender/llm_controller.py`)
- Analyzes hourly departure saturation against JFK runway thresholds (nominal ceiling: 6 departures/hour/runway).
- Solves wake-turbulence separation conflicts and adverse weather ground holds.
- Emits an updated master timetable paired with flight-by-flight operational rationales (e.g., *"Departure delayed by 25 mins to absorb wind-shear buffer on Runway 04L"*).

---

## ⚔️ Dual-Paradigm AI Deconfliction: RAG vs. Fine-Tuning

ToTheSky features a head-to-head architectural comparison evaluating how base prompted LLMs compare against domain-adapted fine-tuned models for mission-critical air traffic management:

| Dimension | 1️⃣ Retrieval-Augmented Generation (RAG) | 2️⃣ Supervised Fine-Tuned (SFT) Model |
| :--- | :--- | :--- |
| **Model Engine** | Base Google Gemini 2.5 / 1.5 Flash | Google Gemini 1.5 Flash (`create_tuned_model`) / LLaMA-3.2-1B LoRA |
| **Input Format** | Elaborate system prompt, ATC guidelines, formatting schemas, runway rules, and schedule context | Minimal zero-shot raw operational records (Flight #, Time, Runway, Weather) |
| **Prompt Overhead** | **~1,200 tokens** per inference batch | **~165 tokens** per inference batch |
| **Output Style** | Formal, explanatory, polite conversational text | Crisp, authoritative, standardized FAA/ATC dispatch jargon |
| **Deployment Fit** | Fast experimentation, frequently shifting policy rules | High-frequency automated scheduling, strict low-latency SLAs |

### 🚀 Performance & Efficiency Metrics

Head-to-head empirical benchmarking demonstrates significant efficiency gains from supervised fine-tuning:

```
┌────────────────────────────────────────────────────────────────────────┐
│  TOKEN CONSUMPTION REDUCTION                                  -86.3%   │
│  [========================================>..............]             │
│  RAG Prompt: 1,200 tokens  ──▶  Fine-Tuned Prompt: 165 tokens          │
├────────────────────────────────────────────────────────────────────────┤
│  INFERENCE LATENCY ACCELERATION                               +60.0%   │
│  [==================================================>....]             │
│  RAG Latency: ~2.40s       ──▶  Fine-Tuned Latency: ~0.95s             │
└────────────────────────────────────────────────────────────────────────┘
```

- **86.3% Token Reduction**: Eliminates verbose in-context formatting rules; the model intrinsically understands the target JSON structure and ATC logic.
- **60%+ Latency Speedup**: Smaller prompt payloads and direct weight alignment drastically cut time-to-first-token and overall response generation time.
- **Zero Prompt Drift**: Guarantees consistent slot assignments and runway distribution without syntax degradation.

---

## 🖥️ Modern, Sidebar-Free UI Layout (`app.py`)

The Streamlit user interface is built with an enterprise-grade, distraction-free layout eliminating sidebar navigation in favor of a clean, tabbed workflow:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│ ✈️ ToTheSky: Airport Monitoring & Scheduling                                                  │
│ [ Airport Hub: JFK ▼ ]  [ Operational Date: 2026-10-07 📅 ]  [ Flight Volume Slider: 40 ]     │
│ [ 📥 Ingest Flight & Weather Data ]   [ 🔄 Retrain XGBoost ]  [ ⚙️ Model & API Configuration ] │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│  [ Tab 1: Data & Optimization ]   [ Tab 2: Analytics & AI Reasoning ]   [ Tab 3: RAG vs SFT ] │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 1. Tab 1: Data & Optimization
- **Operational Controls**: Hub selector (JFK, ORD, LAX), date picker, simulated volume slider, and one-click data ingestion/retraining.
- **Meteorological Conditions Banner**: Live temperature, crosswinds, precipitation, visibility metrics, and dynamic FAA weather advisories.
- **Interactive Raw Flight Schedule**: Data table with multi-select airline and delay-risk filters.
- **AI-Deconflicted Master Timetable**: One-click dispatch execution displaying original vs. updated departure times, runway reassignments, and CSV download.

### 2. Tab 2: Analytics & AI Reasoning
- **Model Confidence Scorecard**: R² accuracy, MAE, ±10-minute tolerance percentages, and high-risk operation tallies.
- **4 Real-Time Plotly Visualizations**:
  1. *Predicted Delay Duration Scatter Plot* across operational hours.
  2. *Delay Risk Severity Pie Breakdown* (Low / Moderate / High).
  3. *Hourly Runway Congestion vs. Peak Saturation Threshold* bar graph.
  4. *XGBoost Feature Importance* horizontal chart (top meteorological & schedule delay drivers).
- **Flight-by-Flight AI Dispatch Rationales**: Expandable cards providing transparent explanations for every schedule change.

### 3. Tab 3: RAG vs. Fine-Tuned Comparison
- **Real-Time KPI Scorecard**: Displays live Token Reduction ($-86.3\%$) and Inference Speedup ($+60\%$).
- **Side-by-Side Dispatch Comparison**: Simultaneous execution of the RAG pipeline vs. the SFT engine on identical flight batches.
- **Flight-by-Flight Phrasing Breakdown**: Direct comparison highlighting RAG verbose reasoning versus the fine-tuned model's direct ATC dispatch syntax.

---

## 🛠️ Tech Stack

- **Dashboard & Visualization**: [Streamlit](https://streamlit.io/), [Plotly](https://plotly.com/python/)
- **Data Manipulation**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Predictive Machine Learning**: [XGBoost](https://xgboost.readthedocs.io/), [Scikit-Learn](https://scikit-learn.org/), [Joblib](https://joblib.readthedocs.io/)
- **Generative AI & LLMs**: [Google Gemini Cloud API](https://ai.google.dev/) (`google-generativeai`)
- **Fine-Tuning & Adaptation**: [PEFT](https://github.com/huggingface/peft) (LoRA), [TRL](https://github.com/huggingface/trl) (`SFTTrainer`), [Transformers](https://huggingface.co/docs/transformers)
- **Database & Persistence**: [SQLite](https://www.sqlite.org/) with [SQLAlchemy](https://www.sqlalchemy.org/) ORM (and MySQL connection support)
- **Testing**: [Pytest](https://docs.pytest.org/)

---

## 📂 Project Structure

```
ToTheSky/
├── app.py                         # Clean, sidebar-free Streamlit dashboard (3 tabs)
├── config.py                      # Global parameters, paths, API configurations
├── requirements.txt               # Dependencies
├── .env.example                   # Environment variable template
├── tothesky.db                    # SQLite persistent database (schema & records)
├── db/
│   ├── connection.py              # SQLAlchemy engine with SQLite/MySQL fallback
│   └── schema.py                  # Flight, DailyWeather, OptimizedSchedule ORM models
├── ingestion/
│   └── ingest.py                  # Flight & meteorological dataset ingestion & cleaning
├── ml/
│   ├── predict.py                 # XGBoost training, inference, feature importance
│   ├── model.joblib               # Pre-trained XGBoost delay regressor
│   └── metrics.json               # Model accuracy telemetry (R², MAE)
├── recommender/
│   └── llm_controller.py          # Dual-engine dispatch (RAG & SFT Gemini/LoRA)
├── scripts/
│   ├── fetch_kaggle_data.py       # Kaggle flight dataset retrieval script
│   ├── prepare_sft_dataset.py     # Generates 1,000 ChatML SFT pairs
│   ├── train_online_gemini.py     # Cloud Gemini tuning script (gemini-1.5-flash-001)
│   └── train_llm.py               # Local Hugging Face LoRA fine-tuning pipeline
└── tests/                         # Pytest test suite (22 unit & integration tests)
```

---

## 🚀 Getting Started

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/divyanshu9r/ToTheSky.git
cd ToTheSky

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and provide your API keys:

```bash
cp .env.example .env
```

```ini
# Database (SQLite default fallback)
MYSQL_URL=sqlite:///./tothesky.db

# Google Gemini API Key
GOOGLE_API_KEY=your_gemini_api_key_here

# Tuned Model Identifier (optional, generated via scripts/train_online_gemini.py)
TUNED_GEMINI_MODEL_ID=tunedModels/tothesky-atc-tuned
```

### 3. Launch the Streamlit Web Application

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` to view the dashboard.

---

## 🔬 Model Training & Fine-Tuning Pipelines

### Online Google Gemini Cloud Tuning
To train a specialized Gemini dispatcher on 1,000 flight deconfliction pairs:
```bash
python scripts/train_online_gemini.py
```
*(Use `--dry_run` to inspect formatted pairs without initiating cloud training).*

### Local LLaMA LoRA Fine-Tuning
To fine-tune a local lightweight model using Hugging Face PEFT/LoRA:
```bash
python scripts/train_llm.py --max_samples 50 --dry_run
```

---

## 🧪 Automated Testing

ToTheSky includes full test coverage spanning data sanitization, database persistence, XGBoost regression, ChatML dataset building, and LLM comparison routing:

```bash
pytest tests/ -v
```

```
============================= 22 passed in 38.45s =============================
```

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
