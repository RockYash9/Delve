"""
FastAPI backend for Delve.

This is a thin transport layer on top of the existing Agent class —
src/agent.py is completely untouched. The same agent that powers the
CLI (src/cli.py) is now reachable over HTTP, so any client (a web
frontend, curl, another program) can use it.

Run with: uvicorn api:app --reload
Docs at:  http://127.0.0.1:8000/docs  (FastAPI generates this automatically)
"""

import json
import logging
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from logging_config import setup_logging
from src import reports
from src.agent import Agent

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="Delve API", version="0.1.0")

# Wide open for local development. Brick 11 (deployment) will restrict
# this to the actual frontend's domain instead of allowing everything.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store: session_id -> Agent instance.
# Scope of this brick: sessions live only as long as this process does
# — restarting the server loses all conversations. A later brick moves
# this to persistent storage; that's an intentional, sequenced tradeoff,
# not an oversight.
_sessions: dict[str, Agent] = {}


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str


class SessionRequest(BaseModel):
    session_id: str


def _get_or_create_agent(session_id: str | None) -> tuple[str, Agent]:
    """Look up an existing session, or start a new one."""
    if session_id and session_id in _sessions:
        return session_id, _sessions[session_id]

    new_id = session_id or str(uuid.uuid4())
    _sessions[new_id] = Agent()
    logger.info("session_created session_id=%s", new_id)
    return new_id, _sessions[new_id]


@app.get("/health")
def health() -> dict:
    """Basic liveness check — used by hosting platforms and monitoring."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Send a message within a session (new or existing) and get a reply."""
    session_id, agent = _get_or_create_agent(request.session_id)
    reply = agent.ask(request.message)
    return ChatResponse(session_id=session_id, reply=reply)


@app.post("/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Like /chat, but streams the reply as Server-Sent Events instead of
    waiting for the full answer.

    Event shapes sent as `data: <json>\\n\\n` lines:
      {"type": "session", "session_id": "..."}   — sent first
      {"type": "status", "text": "..."}          — search activity
      {"type": "token", "text": "..."}           — a chunk of answer text
      {"type": "error", "text": "..."}           — on overload
      {"type": "done"}                            — stream finished

    Deliberately a POST endpoint rather than the browser-native
    EventSource (which only supports GET) — a session_id and message
    body are needed per request, so the frontend will consume this with
    fetch() + a stream reader instead. That's a standard, well-supported
    pattern for SSE-over-POST.
    """
    session_id, agent = _get_or_create_agent(request.session_id)

    def event_stream():
        yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"
        for event in agent.ask_stream(request.message):
            yield f"data: {json.dumps(event)}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/reset")
def reset(request: SessionRequest) -> dict:
    """Clear a session's conversation memory."""
    if request.session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Unknown session_id")
    _sessions[request.session_id].reset()
    logger.info("session_reset session_id=%s", request.session_id)
    return {"status": "reset"}


@app.get("/export/{session_id}", response_class=PlainTextResponse)
def export(session_id: str) -> str:
    """Return the session's conversation as a markdown report with sources."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Unknown session_id")

    transcript = _sessions[session_id].get_transcript()
    if not transcript:
        raise HTTPException(status_code=400, detail="Nothing to export yet")

    return reports.build_report(transcript, _sessions[session_id].get_sources())