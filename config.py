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

MAX_TOKENS = int(os.getenv("MAX_TOKENS", "8192"))

# How many results Tavily returns per search, and how deep it looks.
# "advanced" costs ~2x the API credits of "basic" per Tavily's pricing,
# but returns meaningfully more complete per-source content — worth it
# for a research tool, where a handful of thin snippets isn't enough
# material to actually work with. Tune SEARCH_MAX_RESULTS down (or
# SEARCH_DEPTH back to "basic") if the free-tier search quota runs out
# faster than expected.
SEARCH_MAX_RESULTS = int(os.getenv("SEARCH_MAX_RESULTS", "8"))
SEARCH_DEPTH = os.getenv("SEARCH_DEPTH", "advanced")

# ---------- Production-readiness settings (brick 10) ----------

# Rate limit applied to the expensive endpoints (/chat, /chat/stream) —
# each one triggers real Gemini/Tavily API calls, so this protects the
# free-tier quotas from being burned by accidental loops or abuse.
# Format: slowapi/limits syntax, e.g. "20/minute", "5/second".
RATE_LIMIT = os.getenv("RATE_LIMIT", "20/minute")

# How long a cached search result stays eligible for reuse before it's
# treated as stale and a fresh search is done instead. Prevents serving
# outdated info on fast-changing topics (news, prices) indefinitely.
CACHE_TTL_HOURS = int(os.getenv("CACHE_TTL_HOURS", "24"))

# How long an API session can sit idle before it's eligible for cleanup,
# to cap memory growth on a long-running server (sessions currently
# live in memory only — see api.py).
SESSION_IDLE_TTL_MINUTES = int(os.getenv("SESSION_IDLE_TTL_MINUTES", "120"))

# CORS: comma-separated list of allowed origins, or "*" for all (fine
# for local development, NOT recommended once a real frontend domain
# exists — brick 11 should set this to that exact domain).
ALLOWED_ORIGINS = [
    origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
]


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
