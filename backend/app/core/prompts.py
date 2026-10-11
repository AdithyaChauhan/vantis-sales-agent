"""Load LLM prompts from the /prompts folder (no prompt text lives in code)."""
from functools import lru_cache

from app.core.config import REPO_ROOT

PROMPTS_DIR = REPO_ROOT / "prompts"


@lru_cache
def load_prompt(name: str) -> str:
    """Text of prompts/<name>.md, stripped. Raises FileNotFoundError if the file is missing."""
    return (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8").strip()
