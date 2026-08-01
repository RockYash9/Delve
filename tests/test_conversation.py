"""
Tests for src/memory/conversation.py — specifically send_stream()'s
event interleaving logic.

Conversation.__init__ makes a real network connection (genai.Client),
so these tests build an instance via __new__ and manually wire in a
fake chat session instead of calling __init__ — a deliberate, narrow
bypass to unit-test the streaming logic in isolation.
"""

from src.memory.conversation import Conversation


class _FakeChunk:
    def __init__(self, text):
        self.text = text


class _FakePart:
    def __init__(self, text):
        self.text = text


class _FakeContent:
    def __init__(self, role, text):
        self.role = role
        self.parts = [_FakePart(text)]


class _FakeChatSession:
    """Stand-in for the real Gemini chat session's streaming method."""

    def __init__(self, chunks, history=None):
        self._chunks = chunks
        self._history = history or []

    def send_message_stream(self, message):
        yield from self._chunks

    def get_history(self):
        return self._history


def _make_conversation(chunks, status_events=None, history=None):
    conv = Conversation.__new__(Conversation)  # skip __init__, no network
    conv.sources = []
    conv.status_events = status_events if status_events is not None else []
    conv._chat = _FakeChatSession(chunks, history)  # type: ignore[assignment]
    return conv


def test_send_stream_yields_a_token_event_per_chunk():
    conv = _make_conversation([_FakeChunk("Hello"), _FakeChunk(" world")])

    events = list(conv.send_stream("hi"))

    assert events == [
        {"type": "token", "text": "Hello"},
        {"type": "token", "text": " world"},
    ]


def test_send_stream_flushes_status_events_before_the_next_token():
    # Status events get appended by the search tool DURING the SDK's
    # internal handling, before it yields the next chunk — this test
    # simulates that ordering directly.
    status_events = ["🔍 searching: example"]
    conv = _make_conversation([_FakeChunk("The answer is 4")], status_events)

    events = list(conv.send_stream("what is 2+2"))

    assert events[0] == {"type": "status", "text": "🔍 searching: example"}
    assert events[1] == {"type": "token", "text": "The answer is 4"}


def test_send_stream_flushes_trailing_status_after_last_chunk():
    status_events = ["💾 using cached results for: x"]
    conv = _make_conversation([_FakeChunk("partial")], status_events)

    events = list(conv.send_stream("hello"))
    types_seen = [e["type"] for e in events]

    assert "status" in types_seen
    assert "token" in types_seen


def test_send_stream_skips_empty_chunks():
    # Some chunks in a real stream carry no text (e.g. metadata-only
    # chunks) — these shouldn't produce empty token events.
    conv = _make_conversation([_FakeChunk(""), _FakeChunk(None), _FakeChunk("real text")])

    events = list(conv.send_stream("hi"))

    assert events == [{"type": "token", "text": "real text"}]


def test_send_stream_with_no_status_events_yields_only_tokens():
    conv = _make_conversation([_FakeChunk("just text")])

    events = list(conv.send_stream("hi"))

    assert events == [{"type": "token", "text": "just text"}]


def test_send_stream_falls_back_to_history_when_no_text_streamed():
    # Reproduces a known Gemini SDK/API quirk (gemini-3.5-flash, 2026):
    # when a tool call runs mid-stream, the final answer text sometimes
    # never arrives as a streamed chunk at all — but IS still correctly
    # recorded in chat history once the stream ends. send_stream should
    # recover it from there rather than yielding nothing.
    empty_chunks = [_FakeChunk(None), _FakeChunk("")]  # simulates the bug
    history = [
        _FakeContent("user", "what are EV tax credits?"),
        _FakeContent("model", "The federal EV tax credit changed in 2026."),
    ]
    conv = _make_conversation(empty_chunks, history=history)

    events = list(conv.send_stream("what are EV tax credits?"))

    assert events == [
        {"type": "token", "text": "The federal EV tax credit changed in 2026."}
    ]


def test_send_stream_does_not_use_fallback_when_streaming_worked_normally():
    # If real token text WAS streamed, the fallback must not also fire
    # and duplicate the answer.
    history = [_FakeContent("model", "duplicate text that should NOT appear")]
    conv = _make_conversation([_FakeChunk("streamed text")], history=history)

    events = list(conv.send_stream("hi"))

    assert events == [{"type": "token", "text": "streamed text"}]


def test_send_stream_fallback_with_empty_history_yields_nothing_extra():
    # If the fallback has nothing to recover either, don't fabricate output.
    conv = _make_conversation([_FakeChunk(None)], history=[])

    events = list(conv.send_stream("hi"))

    assert events == []