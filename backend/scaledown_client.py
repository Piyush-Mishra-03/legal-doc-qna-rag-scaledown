import os
import requests


# These must come from environment variables
SCALEDOWN_API_URL = os.getenv("https://scaledown.ai/getapikey")
SCALEDOWN_API_KEY = os.getenv("UpYGg7q97W9CbWfctJ1XA3y6e06QwyZ18WhtjVLx")


def compress_text(text: str) -> str:
    """
    Calls ScaleDown API.
    Falls back to original text if API fails.
    """

    if not text or not text.strip():
        return text

    # Safe fallback when API is not configured
    if not SCALEDOWN_API_URL or not SCALEDOWN_API_KEY:
        print("⚠️ ScaleDown not configured. Skipping compression.")
        return text

    try:
        response = requests.post(
            SCALEDOWN_API_URL,
            headers={
                "Authorization": f"Bearer {SCALEDOWN_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "text": text
            },
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        # keep this flexible for unknown API response
        return data.get("compressed_text", text)

    except Exception as e:
        print("⚠️ ScaleDown error:", e)
        return text


def estimate_tokens(text: str) -> int:
    """
    Light-weight token estimation.
    """
    if not text:
        return 0

    return max(1, int(len(text.split()) * 1.3))
