"""
The core agent loop.

Brick 1 (today): a bare single-turn call to Gemini — no tools, no memory.
This exists purely to prove the environment and API key work end to end.

Brick 2 (next): add the web_search tool and let the model decide when to call it.
Brick 3: multi-turn conversation history.
Brick 4: local caching + retrieval.
"""

import time

from google import genai
from google.genai import types
from google.genai import errors

import config


def ask(user_message: str, max_retries: int = 3) -> str:
    """Send a single message to Gemini and return the text response.

    Retries with backoff on transient server-side errors (e.g. 503
    UNAVAILABLE when Google's servers are under heavy load) — this is
    common on free-tier traffic and isn't something retrying-by-hand
    should be necessary for.
    """
    client = genai.Client(api_key=config.GEMINI_API_KEY)

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=config.MODEL_NAME,
                contents=user_message,
                config=types.GenerateContentConfig(
                    max_output_tokens=config.MAX_TOKENS,
                ),
            )
            return response.text
        except errors.ServerError:
            if attempt == max_retries - 1:
                return (
                    "Gemini's servers are overloaded right now and retries "
                    "didn't succeed. This is temporary — try again in a "
                    "minute."
                )
            wait_seconds = 2 ** attempt  # 1s, 2s, 4s
            time.sleep(wait_seconds)