"""List the Gemini models your key can use, split into chat and embedding models.

Run: python -m scripts.list_models
"""
import sys

from google import genai

from app.core.config import get_settings


def main() -> int:
    key = get_settings().google_api_key
    if not key:
        print("GOOGLE_API_KEY is empty. Set it in backend/.env first.")
        return 1
    chat, embed = [], []
    client = genai.Client(api_key=key)  # keep a reference: a temporary client gets closed
    for model in client.models.list():
        actions = model.supported_actions or []
        row = (model.name or "", model.display_name or "", model.input_token_limit or 0)
        if "generateContent" in actions:
            chat.append(row)
        if "embedContent" in actions:
            embed.append(row)
    for title, rows in (("CHAT MODELS", chat), ("EMBEDDING MODELS", embed)):
        print(f"\n{title} ({len(rows)})")
        for name, display, limit in sorted(rows):
            short = name.removeprefix("models/")
            print(f"  {short:<45} {display:<35} in={limit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
