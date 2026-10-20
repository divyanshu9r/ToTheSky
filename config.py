import os
from pathlib import Path
from dotenv import load_dotenv

# Base Directory
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from .env if present
ENV_PATH = BASE_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()

# MySQL Database Configuration
DEFAULT_MYSQL_URL = "mysql+pymysql://root:root@localhost:3306/tothesky_db"
MYSQL_URL = os.getenv("MYSQL_URL", DEFAULT_MYSQL_URL)
ALLOW_SQLITE_FALLBACK = os.getenv("ALLOW_SQLITE_FALLBACK", "True").lower() in ("true", "1", "yes")

# SQLite Fallback Path
SQLITE_DB_PATH = BASE_DIR / "tothesky.db"
SQLITE_URL = f"sqlite:///{SQLITE_DB_PATH.as_posix()}"

# Google Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Kaggle API Credentials
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME", "")
KAGGLE_KEY = os.getenv("KAGGLE_KEY", "")

# Directory Paths
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True, parents=True)

ML_DIR = BASE_DIR / "ml"
ML_DIR.mkdir(exist_ok=True, parents=True)

MODEL_PATH = ML_DIR / "model.joblib"
METRICS_PATH = ML_DIR / "model_metrics.json"

# Operational Constants
DEFAULT_AIRPORT_CODE = "JFK"
AIRPORT_RUNWAYS = ["04L/22R", "04R/22L", "13L/31R", "13R/31L"]
MAX_HOURLY_RUNWAY_CAPACITY = 30  # Max flights per hour before cascading delay risks spike
