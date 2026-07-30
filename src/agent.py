"""
The core agent loop.

Brick 1: a bare single-turn call to Gemini — no tools, no memory. Done.
Brick 2 (today): give the model a real web_search tool. Gemini decides
for itself whether a question needs a search, and google-genai's
automatic function calling handles running the tool and feeding results
back in — we don't manage that loop by hand.
Brick 3 (next): multi-turn conversation history.
Brick 4: local caching + retrieval.
"""

import time

from google import genai
from google.genai import types
from google.genai import errors

import config
from src.tools.search import web_search


def ask(user_message: str, max_retries: int = 3) -> str:
    """Send a single message to Gemini and return the text response.

    Gemini has access to the web_search tool and will call it on its
    own if it decides the question needs current information.

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
                    tools=[web_search],
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        maximum_remote_calls=5,
                    ),
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