"""
Tests for api.py.

Agent is entirely replaced with a fake — these test the HTTP layer
(session routing, status codes, request/response shapes), not the
agent's own logic, which already has its own tests in test_agent.py.
"""

import json
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

import api


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """The rate limiter (api.limiter) is a module-level singleton shared
    across every test in this file — without resetting it, requests
    made in one test would count toward the limit in unrelated later
    tests, causing spurious 429s. Reset before every test instead.
    """
    api.limiter.reset()
    yield


class _FakeAgent:
    """A stand-in Agent that just echoes messages, tracking calls."""

    def __init__(self):
        self.reset_called = False
        self._transcript: list[tuple[str, str]] = []

    def ask(self, message: str, max_retries: int = 3) -> str:
        self._transcript.append(("user", message))
        reply = f"echo: {message}"
        self._transcript.append(("model", reply))
        return reply

    def ask_stream(self, message: str, deep_research: bool = False):
        yield {"type": "token", "text": f"echo: {message}"}

    def reset(self) -> None:
        self.reset_called = True
        self._transcript = []

    def get_transcript(self) -> list[tuple[str, str]]:
        return self._transcript

    def get_sources(self) -> list[dict]:
        return []


def _client(monkeypatch) -> TestClient:
    """A fresh test client with a clean session store and fake Agent."""
    monkeypatch.setattr(api, "Agent", _FakeAgent)
    api._sessions.clear()
    return TestClient(api.app)


def test_health_check():
    client = TestClient(api.app)
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "active_sessions" in body


def test_root_serves_the_frontend_not_a_404():
    client = TestClient(api.app)
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Delve" in response.text


def test_chat_creates_new_session(monkeypatch):
    client = _client(monkeypatch)

    response = client.post("/chat", json={"message": "hello"})
    data = response.json()

    assert response.status_code == 200
    assert data["reply"] == "echo: hello"
    assert "session_id" in data
    assert len(api._sessions) == 1


def test_chat_reuses_existing_session(monkeypatch):
    client = _client(monkeypatch)

    first = client.post("/chat", json={"message": "hello"}).json()
    session_id = first["session_id"]

    second = client.post(
        "/chat", json={"message": "again", "session_id": session_id}
    ).json()

    assert second["session_id"] == session_id
    assert len(api._sessions) == 1  # same agent reused, not a new one


def test_different_sessions_stay_isolated(monkeypatch):
    client = _client(monkeypatch)

    first = client.post("/chat", json={"message": "hello"}).json()
    second = client.post("/chat", json={"message": "hi there"}).json()

    assert first["session_id"] != second["session_id"]
    assert len(api._sessions) == 2


def test_reset_unknown_session_returns_404(monkeypatch):
    client = _client(monkeypatch)

    response = client.post("/reset", json={"session_id": "does-not-exist"})

    assert response.status_code == 404


def test_reset_known_session(monkeypatch):
    client = _client(monkeypatch)

    session_id = client.post("/chat", json={"message": "hello"}).json()["session_id"]
    response = client.post("/reset", json={"session_id": session_id})

    assert response.status_code == 200
    assert api._sessions[session_id].reset_called is True  # type: ignore[attr-defined]


def test_export_unknown_session_returns_404(monkeypatch):
    client = _client(monkeypatch)

    response = client.get("/export/does-not-exist")

    assert response.status_code == 404


def test_export_with_no_messages_returns_400(monkeypatch):
    client = _client(monkeypatch)
    # Create a session with a chat call, then reset it so it's empty again
    session_id = client.post("/chat", json={"message": "hello"}).json()["session_id"]
    client.post("/reset", json={"session_id": session_id})

    response = client.get(f"/export/{session_id}")

    assert response.status_code == 400


def test_export_returns_markdown_with_transcript(monkeypatch):
    client = _client(monkeypatch)

    session_id = client.post("/chat", json={"message": "hello there"}).json()[
        "session_id"
    ]
    response = client.get(f"/export/{session_id}")

    assert response.status_code == 200
    assert "hello there" in response.text


def test_chat_stream_returns_sse_events(monkeypatch):
    client = _client(monkeypatch)

    response = client.post("/chat/stream", json={"message": "hello"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    body = response.text
    assert '"type": "session"' in body
    assert '"type": "token"' in body
    assert '"type": "done"' in body
    assert "echo: hello" in body


def test_chat_stream_new_session_gets_a_session_id(monkeypatch):
    client = _client(monkeypatch)

    response = client.post("/chat/stream", json={"message": "hi"})
    lines = [line for line in response.text.split("\n\n") if line.strip()]
    first_event = json.loads(lines[0].removeprefix("data: "))

    assert first_event["type"] == "session"
    assert "session_id" in first_event
    assert len(api._sessions) == 1


def test_rate_limit_blocks_requests_beyond_the_configured_limit(monkeypatch):
    monkeypatch.setattr(api, "Agent", _FakeAgent)
    api._sessions.clear()
    monkeypatch.setattr(api.config, "RATE_LIMIT", "3/minute")
    api.limiter.reset()
    client = TestClient(api.app)

    responses = [client.post("/chat", json={"message": "hi"}) for _ in range(4)]

    # First 3 succeed (the configured limit), the 4th is rejected.
    assert [r.status_code for r in responses[:3]] == [200, 200, 200]
    assert responses[3].status_code == 429


def test_rate_limit_does_not_apply_to_health_check(monkeypatch):
    monkeypatch.setattr(api, "Agent", _FakeAgent)
    monkeypatch.setattr(api.config, "RATE_LIMIT", "1/minute")
    api.limiter.reset()
    client = TestClient(api.app)

    responses = [client.get("/health") for _ in range(5)]

    assert all(r.status_code == 200 for r in responses)


def test_stale_sessions_are_purged_on_next_lookup(monkeypatch):
    client = _client(monkeypatch)
    monkeypatch.setattr(api.config, "SESSION_IDLE_TTL_MINUTES", 30)

    session_id = client.post("/chat", json={"message": "hello"}).json()["session_id"]
    assert session_id in api._sessions

    # Simulate this session having gone idle well past the TTL.
    api._session_last_used[session_id] = datetime.now(UTC) - timedelta(hours=1)

    # Any new lookup triggers the lazy purge — a fresh session request
    # is enough to exercise it.
    client.post("/chat", json={"message": "a different question"})

    assert session_id not in api._sessions


def test_recently_active_sessions_are_not_purged(monkeypatch):
    client = _client(monkeypatch)
    monkeypatch.setattr(api.config, "SESSION_IDLE_TTL_MINUTES", 30)

    session_id = client.post("/chat", json={"message": "hello"}).json()["session_id"]

    # Well within the TTL — should survive a purge pass.
    api._session_last_used[session_id] = datetime.now(UTC) - timedelta(minutes=5)
    client.post("/chat", json={"message": "another question"})

    assert session_id in api._sessions
