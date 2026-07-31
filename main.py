"""
Entry point. Run with: python main.py
"""

from logging_config import setup_logging

setup_logging()

from src.cli import run  # noqa: E402 (must come after logging setup)

if __name__ == "__main__":
    run()