"""
The core agent.

Brick 1: a bare single-turn call to Gemini. Done.
Brick 2: web_search tool, model decides when to call it. Done.
Brick 3 (today): wrap a persistent chat session so multi-turn context
works — follow-ups correctly refer back to earlier turns instead of
every message being answered in isolation.
Brick 4 (next): local caching + retrieval.
"""

import time

from google.genai import errors

from src.memory.conversation import Conversation


class Agent:
    """Owns one ongoing conversation and handles transient API errors."""

    def __init__(self):
        self._conversation = Conversation()

    def ask(self, user_message: str, max_retries: int = 3) -> str:
        """Send a message within the ongoing conversation and return the reply.

        Retries with backoff on transient server-side errors (e.g. 503
        UNAVAILABLE when Google's servers are under heavy load).
        """
        for attempt in range(max_retries):
            try:
                return self._conversation.send(user_message)
            except errors.ServerError:
                if attempt == max_retries - 1:
                    return (
                        "Gemini's servers are overloaded right now and "
                        "retries didn't succeed. This is temporary — try "
                        "again in a minute."
                    )
                wait_seconds = 2 ** attempt  # 1s, 2s, 4s
                time.sleep(wait_seconds)

    def reset(self) -> None:
        """Start a brand new conversation, discarding prior context."""
        self._conversation = Conversation()