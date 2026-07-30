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

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Model used for the agent loop. Kept as a named constant so it's
# easy to swap later (e.g. if you want to test a cheaper/faster model).
MODEL_NAME = "claude-sonnet-4-6"

MAX_TOKENS = 1024


def validate_config() -> None:
    """Fail fast and clearly if required keys are missing."""
    missing = []
    if not ANTHROPIC_API_KEY:
        missing.append("ANTHROPIC_API_KEY")

    if missing:
        print(
            f"Missing required environment variable(s): {', '.join(missing)}\n"
            f"Copy .env.example to .env and fill in your key(s)."
        )
        sys.exit(1)
