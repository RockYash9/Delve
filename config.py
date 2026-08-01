"""
Central configuration for Delve.

Every other module should pull settings from here instead of
calling os.getenv() directly — keeps config in one place as
the project grows.
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()  # reads .env into the process environment

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Model used for the agent loop. gemini-3.5-flash-lite is the current
# free-tier model — fast, more free-tier headroom than full Flash, and
# supports function/tool calling.
# Note: Google retires model IDs fairly often — if this model ever 404s,
# check https://ai.google.dev/gemini-api/docs/changelog for the current name.
MODEL_NAME = "gemini-3.5-flash-lite"

MAX_TOKENS = 1024


def validate_config() -> None:
    """Fail fast and clearly if required keys are missing."""
    missing = []
    if not GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY")
    if not TAVILY_API_KEY:
        missing.append("TAVILY_API_KEY")

    if missing:
        print(
            f"Missing required environment variable(s): {', '.join(missing)}\n"
            f"Copy .env.example to .env and fill in your key(s)."
        )
        sys.exit(1)
