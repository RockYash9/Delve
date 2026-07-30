"""
Conversation manager.

Wraps a single Gemini "chat session" — the SDK's built-in abstraction
for multi-turn conversations. It automatically keeps track of every
prior message and reply and resends that context with each new turn,
so follow-up questions ("what about the one before that?") correctly
refer back to earlier parts of the conversation instead of every
message being treated in isolation.
"""

from datetime import date

from google import genai
from google.genai import types

import config
from src.tools.search import web_search


def _system_instruction() -> str:
    return (
        f"Today's date is {date.today():%B %d, %Y}. When a question implies "
        f"recent or current information, use that as your reference point "
        f"rather than guessing a year."
    )


class Conversation:
    """A single multi-turn conversation with memory across turns."""

    def __init__(self):
        self._client = genai.Client(api_key=config.GEMINI_API_KEY)
        self._chat = self._client.chats.create(
            model=config.MODEL_NAME,
            config=types.GenerateContentConfig(
                system_instruction=_system_instruction(),
                max_output_tokens=config.MAX_TOKENS,
                tools=[web_search],
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    maximum_remote_calls=5,
                ),
            ),
        )

    def send(self, message: str) -> str:
        """Send a message within this conversation and return the reply."""
        response = self._chat.send_message(message)
        return response.text

    def turn_count(self) -> int:
        """How many messages (user + model) are in this conversation so far."""
        return len(self._chat.get_history())