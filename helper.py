import re

def clean_response(text: str) -> str:
    """Remove <think>...</think> tags and contents."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()