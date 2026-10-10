"""Application settings, read from environment variables and backend/.env."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    """All config. Every field has a safe default so the app starts without a .env."""

    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    database_url: str = ""
    test_database_url: str = ""

    google_api_key: str = ""
    llm_model_main: str = ""
    llm_model_fast: str = ""
    embedding_model: str = ""
    embedding_dim: int = 768

    groq_api_key: str = ""
    fallback_model: str = ""

    sales_api_key: str = ""
    cors_origins: str = "http://localhost:5173"

    kb_path: str = "data/kb/Vantis_Energy_Solutions_Knowledge_Base.pdf"
    log_level: str = "INFO"

    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "vantis-sales-agent"

    @property
    def sqlalchemy_database_url(self) -> str:
        """DATABASE_URL with the psycopg 3 driver prefix SQLAlchemy needs."""
        return to_sqlalchemy_url(self.database_url)

    @property
    def cors_origin_list(self) -> list[str]:
        """CORS_ORIGINS split into a list."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def kb_file(self) -> Path:
        """KB PDF path; relative values resolve from the repo root."""
        path = Path(self.kb_path)
        return path if path.is_absolute() else REPO_ROOT / path


def to_sqlalchemy_url(url: str) -> str:
    """Rewrite postgresql:// or postgres:// to the psycopg 3 driver form."""
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance."""
    return Settings()
