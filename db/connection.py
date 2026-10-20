import logging
from urllib.parse import urlparse
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
from config import MYSQL_URL, SQLITE_URL, ALLOW_SQLITE_FALLBACK
from db.schema import Base

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_engine = None
_SessionLocal = None
_active_backend = "unknown"
_connection_error = None


def _ensure_mysql_db_exists(url_str: str) -> None:
    """Connect to MySQL server without a database to create the database if it doesn't exist."""
    try:
        parsed = urlparse(url_str)
        db_name = parsed.path.lstrip("/")
        if not db_name:
            return

        # Server-level URL without database name
        server_url = f"{parsed.scheme}://{parsed.netloc}/"
        server_engine = create_engine(server_url, isolation_level="AUTOCOMMIT")
        with server_engine.connect() as conn:
            conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{db_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"))
        server_engine.dispose()
        logger.info(f"Database `{db_name}` verified/created successfully.")
    except Exception as e:
        logger.warning(f"Could not automatically create MySQL database: {e}")


def get_engine():
    """Initializes and returns the database engine with automatic fallback if configured."""
    global _engine, _SessionLocal, _active_backend, _connection_error
    if _engine is not None:
        return _engine

    # Try MySQL first
    mysql_failed = False
    try:
        logger.info(f"Connecting to MySQL database at {MYSQL_URL.split('@')[-1] if '@' in MYSQL_URL else MYSQL_URL}...")
        _ensure_mysql_db_exists(MYSQL_URL)
        engine = create_engine(
            MYSQL_URL,
            pool_recycle=3600,
            pool_pre_ping=True,
            connect_args={"connect_timeout": 5},
        )
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        _engine = engine
        _active_backend = "mysql"
        _connection_error = None
        logger.info("Successfully connected to MySQL database.")
    except Exception as e:
        mysql_failed = True
        _connection_error = str(e)
        logger.warning(f"MySQL connection failed: {e}")

    # Fallback to SQLite if MySQL fails and fallback allowed
    if mysql_failed:
        if ALLOW_SQLITE_FALLBACK:
            logger.info("Falling back to local SQLite database...")
            _engine = create_engine(SQLITE_URL, connect_args={"check_same_thread": False})
            _active_backend = "sqlite"
        else:
            raise RuntimeError(f"Failed to connect to MySQL database: {_connection_error}")

    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
    return _engine


def get_session():
    """Yields a database session."""
    global _SessionLocal
    if _SessionLocal is None:
        get_engine()
    session = _SessionLocal()
    try:
        return session
    except Exception:
        session.close()
        raise


def init_db():
    """Initializes all database tables defined in Base."""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    logger.info(f"Database schema initialized using {_active_backend} backend.")
    return _active_backend


def check_db_connection() -> dict:
    """Returns database backend status and connection info for dashboard telemetry."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {
            "connected": True,
            "backend": _active_backend,
            "url": SQLITE_URL if _active_backend == "sqlite" else MYSQL_URL.split("@")[-1],
            "error": _connection_error if _active_backend == "sqlite" else None,
        }
    except Exception as e:
        return {
            "connected": False,
            "backend": "none",
            "url": None,
            "error": str(e),
        }
