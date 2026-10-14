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
    # Gemini's service tier (T13d): standard (default, nothing sent), priority (Tier 2/3 paid projects, costs more) or
    # flex
    gemini_service_tier: str = "standard"
    embedding_model: str = ""
    embedding_dim: int = 768
    # Relevance gate, checked in T4 (docs/EVAL.md): best similarity was 0.68-0.84 for the 24 KB
    # questions and 0.49-0.58 for the 8 off-topic ones, so 0.63 sits in the gap.
    retrieval_min_similarity: float = 0.63

    groq_api_key: str = ""
    fallback_model: str = ""
    # T13d: a second Groq model for answers, tried after the first; its tokens a minute are its own (Groq limits each
    # model separately). Empty: none
    fallback_model_2: str = ""

    sales_api_key: str = ""
    cors_origins: str = "http://localhost:5173"
    # Header a trusted proxy writes with the real client IP (Render: CF-Connecting-IP, which Cloudflare
    # overwrites on every request). Empty: use the connection's address (local runs, tests).
    client_ip_header: str = ""
    # /api/chat messages per client IP (PLAN.md part 3.11 explains the numbers)
    chat_limit_per_minute: int = 10
    chat_limit_per_day: int = 100

    kb_path: str = "data/kb/Vantis_Energy_Solutions_Knowledge_Base.pdf"
    log_level: str = "INFO"

    # Lead score and status in /api/chat's `lead` field: on for the demo panel, off for production
    chat_show_lead_score: bool = False

    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "vantis-sales-agent"
    # Hide parts of every trace (LangSmith only hides on exactly "true"; see app/core/tracing.py)
    langsmith_hide_inputs: bool = False
    langsmith_hide_outputs: bool = False
    langsmith_hide_metadata: bool = False

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
