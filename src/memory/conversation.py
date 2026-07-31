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
from src.tools.search import make_web_search_tool


def _system_instruction() -> str:
    return (
        f"Today's date is {date.today():%B %d, %Y}. When a question implies "
        f"recent or current information, use that as your reference point "
        f"rather than guessing a year."
    )


class Conversation:
    """A single multi-turn conversation with memory across turns."""

    def __init__(self):
        # Every conversation tracks its own search sources — never
        # shared with other conversations, so concurrent sessions
        # (brick 7's API) can't leak citations between each other.
        self.sources: list[dict] = []

        self._client = genai.Client(api_key=config.GEMINI_API_KEY)
        self._chat = self._client.chats.create(
            model=config.MODEL_NAME,
            config=types.GenerateContentConfig(
                system_instruction=_system_instruction(),
                max_output_tokens=config.MAX_TOKENS,
                tools=[make_web_search_tool(self.sources)],
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    maximum_remote_calls=5,
                ),
            ),
        )

    def send(self, message: str) -> str:
        """Send a message within this conversation and return the reply."""
        response = self._chat.send_message(message)
        if response.text is None:
            # Can happen if the response was safety-filtered or contained
            # only a function call with no accompanying text. Rare, but a
            # silent None would be a confusing crash three layers up.
            return (
                "I wasn't able to generate a text response to that — try "
                "rephrasing your question."
            )
        return response.text

    def turn_count(self) -> int:
        """How many messages (user + model) are in this conversation so far."""
        return len(self._chat.get_history())

    def get_transcript(self) -> list[tuple[str, str]]:
        """Return (role, text) pairs for the human-readable parts of this
        conversation — skips the internal tool-call/tool-result traffic,
        keeping just what the user asked and what the model answered.
        """
        transcript: list[tuple[str, str]] = []
        for content in self._chat.get_history():
            if content.role is None:
                continue
            text_parts = [
                part.text
                for part in (content.parts or [])
                if part.text
            ]
            if text_parts:
                transcript.append((content.role, "\n".join(text_parts)))
        return transcript