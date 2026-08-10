"""
Conversation manager.

Wraps a single Gemini "chat session" — the SDK's built-in abstraction
for multi-turn conversations. It automatically keeps track of every
prior message and reply and resends that context with each new turn,
so follow-up questions ("what about the one before that?") correctly
refer back to earlier parts of the conversation instead of every
message being treated in isolation.
"""

import logging
from datetime import date

from google import genai
from google.genai import types

import config
from src import verification
from src.tools.search import make_web_search_tool

logger = logging.getLogger(__name__)


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
        if response.text:
            return response.text

        # `response.text` can be an empty string, not just None — a
        # falsy-but-not-None value that a naive `is None` check misses
        # entirely (this was a real bug caught during testing). It shows
        # up specifically after a tool call: the model occasionally
        # completes generation with genuinely empty text. One retry with
        # a short nudge — same session, so already-fetched search
        # results stay available — reliably resolves it in practice
        # rather than leaving the client with silence.
        logger.warning("empty_response_retrying_with_nudge")
        retry_response = self._chat.send_message(
            "Please provide your complete answer now, based on the "
            "information you already have."
        )
        if retry_response.text:
            return retry_response.text

        return (
            "I wasn't able to generate a text response to that — try "
            "rephrasing your question."
        )

    def send_stream(self, message: str, deep_research: bool = False):
        """Send a message and yield structured events for a live-feeling reply.

        Yields dicts of the form {"type": "status"|"token"|"sources"|
        "contradictions"|"verification", ...}.

        IMPORTANT DESIGN NOTE: this does NOT use Gemini's raw
        send_message_stream(). Gemini has a confirmed bug (gemini-3.5-flash,
        mid-2026) where combining streaming with automatic function
        calling can cause the model's final answer to come back
        genuinely empty whenever a tool runs mid-stream — not just a
        display glitch, the generation itself stops with no text. Since
        this agent's search tool is attached to every conversation, that
        failure mode isn't a rare edge case here, it's close to
        guaranteed. A history-based recovery attempt was tried first and
        confirmed insufficient (the answer isn't there to recover).

        Instead: get the complete, correct answer via send() — the
        non-streaming path already proven reliable throughout this
        project — then deliver it to the client in small word-sized
        pieces for a live-typing feel. Search status still surfaces in
        true real time, since status_events is populated by our own
        tool wrapper independent of Gemini's stream; only the final
        answer text is "simulated" streaming rather than raw
        network-level token streaming.

        deep_research=True additionally runs two extra, independent
        Gemini calls after the answer is complete: a contradiction
        check across this turn's sources, and a self-verification pass
        checking the answer's own claims against those sources. Off by
        default — each is a full extra API call, so this roughly
        doubles or triples request volume for this turn.
        """
        flushed_count = 0

        def flush_pending_status():
            nonlocal flushed_count
            while flushed_count < len(self.status_events):
                yield {"type": "status", "text": self.status_events[flushed_count]}
                flushed_count += 1

        sources_before = len(self.sources)
        reply = self.send(message)

        # By the time send() returns, any searches it triggered have
        # already recorded their status — flush all of it now, before
        # the answer text, matching the "searching... then answer" order
        # a client should show.
        yield from flush_pending_status()

        for i, word in enumerate(reply.split(" ")):
            piece = word if i == 0 else " " + word
            yield {"type": "token", "text": piece}

        # Only this turn's new sources, deduplicated by URL (a cache hit
        # or multiple searches can surface the same page more than
        # once) — self.sources accumulates across the whole
        # conversation, but a client rendering "sources for this
        # answer" needs just what was found just now.
        seen_urls: set[str] = set()
        new_sources = []
        for source in self.sources[sources_before:]:
            if source["url"] in seen_urls:
                continue
            seen_urls.add(source["url"])
            new_sources.append(source)

        if new_sources:
            yield {"type": "sources", "sources": new_sources}

        if deep_research and new_sources:
            yield {"type": "status", "text": "🔬 checking sources for contradictions..."}
            contradiction_report = verification.check_contradictions(new_sources)
            if contradiction_report.has_contradictions:
                yield {
                    "type": "contradictions",
                    "contradictions": [
                        c.model_dump() for c in contradiction_report.contradictions
                    ],
                }

            yield {"type": "status", "text": "🔬 verifying claims against sources..."}
            verification_report = verification.verify_claims(reply, new_sources)
            yield {
                "type": "verification",
                "total_claims_checked": verification_report.total_claims_checked,
                "supported_count": verification_report.supported_count,
                "unsupported_claims": [
                    c.model_dump() for c in verification_report.unsupported_claims
                ],
            }

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
