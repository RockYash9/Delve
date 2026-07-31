"""
Tests for api.py.

Agent is entirely replaced with a fake — these test the HTTP layer
(session routing, status codes, request/response shapes), not the
agent's own logic, which already has its own tests in test_agent.py.
"""

from fastapi.testclient import TestClient

import api


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
    assert response.json() == {"status": "ok"}


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