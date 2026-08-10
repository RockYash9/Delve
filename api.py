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
from datetime import UTC, datetime, timedelta

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

import config
from logging_config import setup_logging
from src import reports
from src.agent import Agent

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="Delve API", version="0.1.0")

# CORS: configurable via ALLOWED_ORIGINS env var. "*" (the default) is
# fine for local development; brick 11's deployment should set this to
# the actual frontend's exact domain instead of allowing everything.
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting: protects the free-tier Gemini/Tavily quotas from being
# burned by accidental loops or abuse. Keyed by client IP. Applied only
# to the endpoints that actually trigger external API calls — /health
# and /export are local-only and stay unlimited.
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]
app.add_middleware(SlowAPIMiddleware)

# In-memory session store: session_id -> Agent instance.
# Sessions live only as long as this process does — restarting the
# server loses all conversations. A later brick may move this to
# persistent storage; that's an intentional, sequenced tradeoff, not
# an oversight. _session_last_used backs the idle-session cleanup
# below, which caps memory growth on a long-running server.
_sessions: dict[str, Agent] = {}
_session_last_used: dict[str, datetime] = {}


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    deep_research: bool = False


class ChatResponse(BaseModel):
    session_id: str
    reply: str


class SessionRequest(BaseModel):
    session_id: str


def _purge_stale_sessions() -> None:
    """Drop sessions idle longer than SESSION_IDLE_TTL_MINUTES.

    Called lazily on each new session lookup rather than via a
    background scheduler — simple, dependency-free, and sufficient
    for this scale: memory is bounded by "active sessions since the
    last cleanup pass" rather than growing forever.
    """
    cutoff = datetime.now(UTC) - timedelta(minutes=config.SESSION_IDLE_TTL_MINUTES)
    stale_ids = [
        session_id
        for session_id, last_used in _session_last_used.items()
        if last_used < cutoff
    ]
    for session_id in stale_ids:
        _sessions.pop(session_id, None)
        _session_last_used.pop(session_id, None)
    if stale_ids:
        logger.info("purged_stale_sessions count=%d", len(stale_ids))


def _get_or_create_agent(session_id: str | None) -> tuple[str, Agent]:
    """Look up an existing session, or start a new one."""
    _purge_stale_sessions()

    if session_id and session_id in _sessions:
        _session_last_used[session_id] = datetime.now(UTC)
        return session_id, _sessions[session_id]

    new_id = session_id or str(uuid.uuid4())
    _sessions[new_id] = Agent()
    _session_last_used[new_id] = datetime.now(UTC)
    logger.info("session_created session_id=%s", new_id)
    return new_id, _sessions[new_id]


@app.get("/health")
def health() -> dict:
    """Basic liveness check — used by hosting platforms and monitoring."""
    return {"status": "ok", "active_sessions": len(_sessions)}


@app.post("/chat", response_model=ChatResponse)
@limiter.limit(lambda: config.RATE_LIMIT)
def chat(request: Request, payload: ChatRequest) -> ChatResponse:
    """Send a message within a session (new or existing) and get a reply."""
    session_id, agent = _get_or_create_agent(payload.session_id)
    reply = agent.ask(payload.message)
    return ChatResponse(session_id=session_id, reply=reply)


@app.post("/chat/stream")
@limiter.limit(lambda: config.RATE_LIMIT)
def chat_stream(request: Request, payload: ChatRequest) -> StreamingResponse:
    """Like /chat, but streams the reply as Server-Sent Events instead of
    waiting for the full answer.

    Set payload.deep_research=True to also run two extra checks after
    the answer: cross-source contradiction detection and self-
    verification of the answer's claims against its sources. Off by
    default — each is a full extra Gemini API call, so this roughly
    doubles/triples request volume for the turn.

    Event shapes sent as `data: <json>\\n\\n` lines:
      {"type": "session", "session_id": "..."}   — sent first
      {"type": "status", "text": "..."}          — search/verification activity
      {"type": "token", "text": "..."}           — a chunk of answer text
      {"type": "sources", "sources": [...]}      — this turn's citations
      {"type": "contradictions", "contradictions": [...]}  — deep_research only
      {"type": "verification", "total_claims_checked": N,
       "supported_count": N, "unsupported_claims": [...]}  — deep_research only
      {"type": "error", "text": "..."}           — on overload
      {"type": "done"}                            — stream finished

    Deliberately a POST endpoint rather than the browser-native
    EventSource (which only supports GET) — a session_id and message
    body are needed per request, so the frontend consumes this with
    fetch() + a stream reader instead. That's a standard, well-supported
    pattern for SSE-over-POST.
    """
    session_id, agent = _get_or_create_agent(payload.session_id)

    def event_stream():
        yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"
        for event in agent.ask_stream(payload.message, deep_research=payload.deep_research):
            yield f"data: {json.dumps(event)}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/reset")
def reset(payload: SessionRequest) -> dict:
    """Clear a session's conversation memory."""
    if payload.session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Unknown session_id")
    _sessions[payload.session_id].reset()
    _session_last_used[payload.session_id] = datetime.now(UTC)
    logger.info("session_reset session_id=%s", payload.session_id)
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


# Serves the chat UI (static/index.html) at "/" and any other static
# assets under static/. Mounted LAST and at the root path so the
# explicit API routes above still take priority for their exact paths
# (Starlette checks routes in registration order) — everything else
# falls through to static files. This also means GET / now serves the
# frontend instead of a 404, and there's one service to run/deploy
# instead of two separate frontend/backend deployments.
app.mount("/", StaticFiles(directory="static", html=True), name="frontend")
