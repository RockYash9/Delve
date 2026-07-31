"""
Shared pytest configuration.

Sets dummy environment variables before any test imports config, so
config.validate_config() never fails in CI (which has no real keys)
and so no test can accidentally make a real network call using a
real key that happens to be sitting in a local .env file.
"""

import os

os.environ.setdefault("GEMINI_API_KEY", "test-dummy-key")
os.environ.setdefault("TAVILY_API_KEY", "test-dummy-key")
