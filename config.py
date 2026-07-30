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

# Model used for the agent loop. Gemini 2.5 Flash is the free-tier model:
# generous daily request limit, supports function/tool calling, no card needed.
MODEL_NAME = "gemini-3.5-flash-lite"

MAX_TOKENS = 1024


def validate_config() -> None:
    """Fail fast and clearly if required keys are missing."""
    missing = []
    if not GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY")

    if missing:
        print(
            f"Missing required environment variable(s): {', '.join(missing)}\n"
            f"Copy .env.example to .env and fill in your key(s)."
        )
        sys.exit(1)