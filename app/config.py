"""
Application configuration.

Loads environment variables (via python-dotenv) into a single typed
Settings object. Nothing in this file talks to FastAPI or the database
directly — per §7.1 of the spec, this is pure configuration, importable
from anywhere (services, repositories, jobs, tests) without pulling in
the HTTP layer.

Guard thresholds (SIMILARITY_THRESHOLD, CONFIDENCE_ACCEPT_THRESHOLD,
CONFIDENCE_FLAG_THRESHOLD, MAX_RETRIES) live here as configuration,
never hardcoded in business logic — per FR-3.5.
"""

from functools import lru_cache

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env into the process environment before Settings reads it.
# Safe to call even if .env is missing (e.g. in CI where real env vars
# are injected directly) — python-dotenv just no-ops in that case.
load_dotenv()


class Settings(BaseSettings):
    """
    Typed application settings, sourced from environment variables.

    Field names match the keys in .env.example (§22) exactly, so no
    manual mapping is needed — pydantic-settings reads them by name.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- AI Provider ---
    GEMINI_API_KEY: str

    # --- Database (Neon Postgres) ---
    DATABASE_URL: str

    # --- Guard Thresholds (§13.3) ---
    SIMILARITY_THRESHOLD: float = Field(default=0.75, ge=0.0, le=1.0)
    CONFIDENCE_ACCEPT_THRESHOLD: float = Field(default=0.60, ge=0.0, le=1.0)
    CONFIDENCE_FLAG_THRESHOLD: float = Field(default=0.60, ge=0.0, le=1.0)
    MAX_RETRIES: int = Field(default=3, ge=0)

    # --- App ---
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"

    @property
    def async_database_url(self) -> str:
        """
        Neon (and most Postgres hosts) hand out a plain `postgresql://`
        connection string. SQLAlchemy's async engine needs the driver
        named explicitly: `postgresql+asyncpg://`. We convert here in
        code rather than asking the user to hand-edit .env, so the
        .env value can always be pasted directly from the Neon
        dashboard without modification.
        """
        url = self.DATABASE_URL
        if url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+asyncpg://", 1)
        if url.startswith("postgres://"):
            # Some providers (e.g. Heroku-style) use the short form.
            return url.replace("postgres://", "postgresql+asyncpg://", 1)
        return url


@lru_cache
def get_settings() -> Settings:
    """
    Cached settings accessor. Import and call this everywhere config
    is needed (`from app.config import get_settings`) instead of
    constructing Settings() directly — keeps env parsing to once per
    process and gives tests a single point to monkeypatch/override.
    """
    return Settings()