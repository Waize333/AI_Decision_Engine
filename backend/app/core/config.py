"""
backend/app/core/config.py
==========================
PURPOSE:
  Central configuration management using Pydantic Settings.
  Reads from environment variables (or .env file) and validates them.

WHY THIS PATTERN?
  - Single source of truth for all settings
  - Type-validated at startup (crash early, not at runtime)
  - Easy to override in tests by setting env vars
  - Auto-documented by Pydantic

USAGE:
  from app.core.config import settings
  print(settings.APP_NAME)
"""

from functools import lru_cache
from typing import List, Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    All application settings.
    Values are loaded from environment variables or .env file.
    Field names match exactly the variable names in .env.example
    """

    model_config = SettingsConfigDict(
        env_file=".env",          # load from .env if it exists
        env_file_encoding="utf-8",
        case_sensitive=True,      # APP_NAME ≠ app_name
        extra="ignore",           # silently ignore unknown env vars
    )

    # ─── APPLICATION ─────────────────────────────────────────
    APP_NAME: str = "AI Decision Engine"
    APP_ENV: str = "development"          # development | staging | production
    APP_DEBUG: bool = True
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000

    # ─── SECURITY ────────────────────────────────────────────
    SECRET_KEY: str = "change-this-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # ─── POSTGRESQL ──────────────────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/ai_decision_engine"

    # ─── MONGODB ─────────────────────────────────────────────
    MONGO_URL: str = "mongodb://mongo:mongo@localhost:27017/ai_decision_engine"
    MONGO_DB: str = "ai_decision_engine"

    # ─── REDIS ───────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"

    # ─── RATE LIMITING ───────────────────────────────────────
    RATE_LIMIT_REQUESTS: int = 100        # max requests
    RATE_LIMIT_WINDOW_SECONDS: int = 60   # per this many seconds

    # ─── CORS ────────────────────────────────────────────────
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8080"]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Allow CORS_ORIGINS to be a comma-separated string in .env"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    # ─── INFERENCE ───────────────────────────────────────────
    MODEL_REGISTRY_PATH: str = "./models"
    DEFAULT_MODEL_VERSION: str = "v1"
    INFERENCE_TIMEOUT_SECONDS: int = 30

    # ─── DRIFT ───────────────────────────────────────────────
    DRIFT_CHECK_INTERVAL_HOURS: int = 6
    DRIFT_ALERT_THRESHOLD: float = 0.15   # PSI score threshold

    # ─── LOGGING ─────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"           # json | text

    # ─── METRICS ─────────────────────────────────────────────
    METRICS_ENABLED: bool = True
    METRICS_PORT: int = 9090

    # ─── COMPUTED PROPERTIES ─────────────────────────────────
    @property
    def is_production(self) -> bool:
        """True when running in production — disables debug features"""
        return self.APP_ENV == "production"

    @property
    def is_development(self) -> bool:
        return self.APP_ENV == "development"


@lru_cache()
def get_settings() -> Settings:
    """
    Returns a cached Settings instance.

    WHY lru_cache?
      Settings() reads from disk (the .env file) on every call.
      lru_cache makes it so the file is only read ONCE per process,
      and all callers share the same Settings object.

    In tests, call get_settings.cache_clear() to reset the cache
    when you need to change settings between tests.
    """
    return Settings()


# Convenience alias — import this everywhere
settings = get_settings()
