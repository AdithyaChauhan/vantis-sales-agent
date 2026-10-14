"""Key words of a text, for matching a customer's message or a model's reply to KB chunks without a model (T13d).

Used by the safety net (app/agent/fallback.py: which KB sentence answers the message) and by the missing-source guard
(app/rag/answer.py: which chunk a reply without a KB tag used). The synonym map only helps matching; it never adds a
fact.
"""
import re

# Words a customer may use for the KB's words (Hinglish and synonyms), for matching only: they never add a fact
MATCH_WORDS = {
    "kharcha": "cost", "kharch": "cost", "keemat": "price", "kimat": "price", "daam": "price", "bijli": "electricity",
    "chhat": "rooftop", "chhath": "rooftop", "roof": "rooftop", "terrace": "rooftop", "sarkar": "government",
    "sarkari": "government", "kist": "financing", "instalment": "financing", "installment": "financing",
    "emi": "financing", "loan": "financing", "finance": "financing", "guarantee": "warranty",
    "lagwana": "install", "lagwane": "install", "lagana": "install", "kharab": "issue", "problem": "issue",
    "panel": "panel", "maintain": "maintenance", "servicing": "maintenance",
    "clean": "maintenance", "cleaning": "maintenance", "safai": "maintenance", "saaf": "maintenance",  # T13e
}
# Words that say nothing about the topic (English, Hinglish, and words in almost every KB line)
STOP_WORDS = {
    "the", "and", "for", "you", "your", "are", "can", "what", "how", "much", "many", "does", "with", "this", "that",
    "have", "has", "will", "would", "like", "want", "need", "please", "tell", "know", "about", "any", "there", "from",
    "into", "also", "get", "our", "they", "their", "them", "who", "which", "when", "where", "why", "may", "not", "all",
    "but", "was", "were", "been", "being", "than", "then", "its", "it's", "yes", "kya", "hai", "hain", "hota", "hoti",
    "hoga", "hogi", "kar", "karte", "karna", "padega", "sakte", "sakta", "aap", "log", "mujhe", "mera", "meri", "mere",
    "kitna", "kitne", "kitni", "kaise", "kab", "kaunsa", "konsa", "chahiye", "milti", "milta", "aata", "wala", "wali",
    "thoda", "bhi", "agar", "toh", "pura", "vantis", "solar", "customer", "customers", "lena", "jaye", "saal",
}
_WORD_RE = re.compile(r"[a-z0-9']+")


def key_words(text: str) -> set[str]:
    """The words of a text that carry its topic: lowercase, no stop words, a plural's "s" dropped, synonyms mapped."""
    words = set()
    for word in _WORD_RE.findall(text.lower()):
        if len(word) < 3 or word in STOP_WORDS:
            continue
        if word.startswith("install"):
            word = "install"
        elif word.endswith("ies") and len(word) > 4:
            word = word[:-3] + "y"
        elif word.endswith("s") and not word.endswith("ss") and len(word) > 3:
            word = word[:-1]
        word = MATCH_WORDS.get(word, word)
        if word not in STOP_WORDS:
            words.add(word)
    return words
