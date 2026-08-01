"""
Tests for src/agent.py — specifically the retry-with-backoff behavior.

We never construct a real Conversation (that would try to hit the
Gemini API over the network). Instead, src.agent.Conversation is
monkeypatched with a fake that fails a configurable number of times
before succeeding, so we can verify the retry logic itself: does it
retry, does it eventually give up, does it recover mid-way through.
"""

import pytest
from google.genai import errors

from src.agent import Agent


class _FakeConversation:
    """Stand-in for Conversation that fails `fail_times` calls, then succeeds."""

    def __init__(self, fail_times: int = 0, fail_immediately: bool = False):
        self.fail_times = fail_times
        self.fail_immediately = fail_immediately
        self.calls = 0

    def send(self, message: str) -> str:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise errors.ServerError(503, {"error": {"message": "overloaded"}})
        return "final answer"

    def send_stream(self, message: str):
        if self.fail_immediately:
            raise errors.ServerError(503, {"error": {"message": "overloaded"}})
        yield {"type": "token", "text": "streamed answer"}

    def get_transcript(self) -> list[tuple[str, str]]:
        return []  # not exercised by these tests


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch):
    """Don't actually wait during backoff — tests should run instantly."""
    monkeypatch.setattr("src.agent.time.sleep", lambda seconds: None)


def test_ask_returns_answer_on_first_try(monkeypatch):
    monkeypatch.setattr("src.agent.Conversation", lambda: _FakeConversation())
    agent = Agent()

    assert agent.ask("hello") == "final answer"


def test_ask_retries_and_recovers(monkeypatch):
    monkeypatch.setattr("src.agent.Conversation", lambda: _FakeConversation(fail_times=2))
    agent = Agent()

    result = agent.ask("hello", max_retries=3)

    assert result == "final answer"


def test_ask_gives_up_gracefully_after_max_retries(monkeypatch):
    monkeypatch.setattr("src.agent.Conversation", lambda: _FakeConversation(fail_times=99))
    agent = Agent()

    result = agent.ask("hello", max_retries=3)

    # Should return a friendly message, not raise an unhandled exception.
    assert "try again" in result.lower() or "overloaded" in result.lower()


def test_reset_creates_a_new_conversation(monkeypatch):
    created = []

    def factory():
        conv = _FakeConversation()
        created.append(conv)
        return conv

    monkeypatch.setattr("src.agent.Conversation", factory)
    agent = Agent()
    agent.reset()

    assert len(created) == 2  # one at Agent() construction, one at reset()


def test_ask_stream_yields_events_from_conversation(monkeypatch):
    monkeypatch.setattr("src.agent.Conversation", lambda: _FakeConversation())
    agent = Agent()

    events = list(agent.ask_stream("hello"))

    assert events == [{"type": "token", "text": "streamed answer"}]


def test_ask_stream_yields_error_event_on_overload(monkeypatch):
    monkeypatch.setattr(
        "src.agent.Conversation", lambda: _FakeConversation(fail_immediately=True)
    )
    agent = Agent()

    events = list(agent.ask_stream("hello"))

    assert len(events) == 1
    assert events[0]["type"] == "error"
    assert "overload" in events[0]["text"].lower()