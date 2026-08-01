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

        # Short status messages recorded whenever the search tool runs
        # (e.g. "🔍 searching: ..."). Brick 8's streaming reads this to
        # surface search activity to the client in real time.
        self.status_events: list[str] = []

        self._client = genai.Client(api_key=config.GEMINI_API_KEY)
        self._chat = self._client.chats.create(
            model=config.MODEL_NAME,
            config=types.GenerateContentConfig(
                system_instruction=_system_instruction(),
                max_output_tokens=config.MAX_TOKENS,
                tools=[make_web_search_tool(self.sources, self.status_events)],
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

    def send_stream(self, message: str):
        """Send a message and yield structured events as the reply streams in.

        Yields dicts of the form {"type": "status"|"token", "text": ...}.
        Status events surface search activity (recorded in status_events
        by the search tool while the SDK runs it internally, mid-stream)
        interleaved with the actual answer text as it's generated, so a
        client can show "searching..." before the text that resulted
        from it rather than only seeing a long pause.
        """
        flushed_count = 0
        any_token_yielded = False

        def flush_pending_status():
            nonlocal flushed_count
            while flushed_count < len(self.status_events):
                yield {"type": "status", "text": self.status_events[flushed_count]}
                flushed_count += 1

        for chunk in self._chat.send_message_stream(message):
            yield from flush_pending_status()
            if chunk.text:
                any_token_yielded = True
                yield {"type": "token", "text": chunk.text}

        # Any status events recorded after the last chunk was produced
        # (e.g. a search that ran right before the stream closed) still
        # need to reach the client.
        yield from flush_pending_status()

        if not any_token_yielded:
            # Known SDK/API quirk (as of Gemini 3.5 Flash, mid-2026): when
            # automatic function calling runs a tool during a streamed
            # response, the final answer text sometimes never arrives as
            # a streamed chunk at all — even though the SDK *does* still
            # correctly record it in the chat's history once the stream
            # finishes. Rather than leave the client with an empty reply,
            # recover the text from history instead.
            history = self._chat.get_history()
            if history and history[-1].role == "model":
                fallback_text = "".join(
                    part.text for part in (history[-1].parts or []) if part.text
                )
                if fallback_text:
                    yield {"type": "token", "text": fallback_text}

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