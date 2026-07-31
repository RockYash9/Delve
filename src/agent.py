"""
The core agent.

Brick 1: a bare single-turn call to Gemini. Done.
Brick 2: web_search tool, model decides when to call it. Done.
Brick 3 (today): wrap a persistent chat session so multi-turn context
works — follow-ups correctly refer back to earlier turns instead of
every message being answered in isolation.
Brick 4 (next): local caching + retrieval.
"""

import logging
import time

from google.genai import errors

from src.memory.conversation import Conversation

logger = logging.getLogger(__name__)


class Agent:
    """Owns one ongoing conversation and handles transient API errors."""

    def __init__(self):
        self._conversation = Conversation()

    def ask(self, user_message: str, max_retries: int = 3) -> str:
        """Send a message within the ongoing conversation and return the reply.

        Retries with backoff on transient server-side errors (e.g. 503
        UNAVAILABLE when Google's servers are under heavy load).
        """
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1")

        for attempt in range(max_retries):
            try:
                return self._conversation.send(user_message)
            except errors.ServerError:
                if attempt == max_retries - 1:
                    logger.warning(
                        "gemini_overloaded_giving_up attempts=%d", max_retries
                    )
                    return (
                        "Gemini's servers are overloaded right now and "
                        "retries didn't succeed. This is temporary — try "
                        "again in a minute."
                    )
                wait_seconds = 2**attempt  # 1s, 2s, 4s
                logger.info(
                    "gemini_overloaded_retrying attempt=%d wait_seconds=%d",
                    attempt + 1,
                    wait_seconds,
                )
                time.sleep(wait_seconds)

        # Unreachable in practice (the loop above always returns or raises),
        # but keeps the function's return type honest for the type checker
        # and any future refactor that changes the loop.
        raise RuntimeError("ask() exited its retry loop without returning")

    def reset(self) -> None:
        """Start a brand new conversation, discarding prior context."""
        self._conversation = Conversation()

    def get_transcript(self) -> list[tuple[str, str]]:
        """The current conversation's (role, text) history, for export."""
        return self._conversation.get_transcript()