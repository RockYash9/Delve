"""
Application-wide logging configuration.

Deliberately separate from the Rich panels/spinners in src/cli.py —
those are curated UX for the person using the app. This is structured,
persistent logging for debugging and observability: what did the agent
actually do during a session, independent of what happened to be shown
on screen at the time.

Logs go to logs/delve.log (rotated so it doesn't grow forever) and,
at WARNING level and above, also to the console — so real problems
surface immediately without cluttering normal operation with routine
INFO-level noise.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent / "logs"


def setup_logging(level: int = logging.INFO) -> None:
    LOG_DIR.mkdir(exist_ok=True)

    file_handler = RotatingFileHandler(
        LOG_DIR / "delve.log",
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    logging.basicConfig(level=level, handlers=[file_handler, console_handler], force=True)

    # Third-party libraries are noisy at INFO — quiet them down so our
    # own log file stays readable and actually useful.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
