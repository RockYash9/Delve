"""
Deep Research Mode: contradiction detection and self-verification.

Two independent capabilities, both opt-in (never run by default):

1. check_contradictions() — cross-references multiple search-result
   sources on the same topic and flags genuine factual disagreements
   (different numbers, dates, statuses) instead of letting the main
   answer silently pick one and move on.

2. verify_claims() — a second, independent LLM call that checks the
   FINAL ANSWER's individual factual claims against the actual
   retrieved source content, flagging anything stated confidently but
   not actually grounded in what was retrieved. This is a direct,
   practical mitigation for the most common trust problem in RAG/agent
   systems: a fluent, confident-sounding answer that quietly contains
   an unsupported or fabricated detail.

Both are opt-in because each one is a full extra Gemini API call —
running them automatically would roughly double or triple request
volume per search-based answer, which matters directly against a
free-tier quota. Each makes its own narrowly-scoped call, separate
from the main conversational chat session, so a verification failure
can never corrupt or interfere with actual conversation history.
"""

import logging

from google import genai
from google.genai import types
from pydantic import BaseModel

import config

logger = logging.getLogger(__name__)


class Contradiction(BaseModel):
    topic: str
    source_a_title: str
    source_a_claim: str
    source_b_title: str
    source_b_claim: str
    explanation: str


class ContradictionReport(BaseModel):
    has_contradictions: bool
    contradictions: list[Contradiction]


class UnsupportedClaim(BaseModel):
    claim: str
    reason: str


class VerificationReport(BaseModel):
    total_claims_checked: int
    supported_count: int
    unsupported_claims: list[UnsupportedClaim]


def _get_client() -> genai.Client:
    return genai.Client(api_key=config.GEMINI_API_KEY)


def _format_sources(sources: list[dict]) -> str:
    return "\n\n---\n\n".join(
        f"SOURCE: {s.get('title', 'Untitled')}\nCONTENT: {s.get('content', '')}"
        for s in sources
    )


def check_contradictions(sources: list[dict]) -> ContradictionReport:
    """Cross-reference multiple sources for genuine factual disagreement.

    Needs at least 2 distinct sources to be meaningful — with fewer
    than that there's nothing to cross-reference, so this returns
    immediately without spending an API call.
    """
    if len(sources) < 2:
        return ContradictionReport(has_contradictions=False, contradictions=[])

    prompt = (
        "You are cross-referencing multiple search result sources on the "
        "same topic. Identify any factual claims where sources genuinely "
        "disagree — different numbers, dates, statuses, or conclusions. "
        "Do NOT flag differences in wording, emphasis, or level of "
        "detail — only real factual disagreement. If the sources agree "
        "or simply don't overlap on any checkable claim, report no "
        "contradictions rather than inventing one.\n\n"
        f"{_format_sources(sources)}"
    )

    try:
        client = _get_client()
        response = client.models.generate_content(
            model=config.MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ContradictionReport,
            ),
        )
        if not isinstance(response.parsed, ContradictionReport):
            logger.warning("contradiction_check_unparseable_response")
            return ContradictionReport(has_contradictions=False, contradictions=[])
        return response.parsed
    except Exception:
        # Deep Research is an enhancement, not a core guarantee — a
        # failure here should degrade gracefully, never break the
        # user's actual answer.
        logger.exception("contradiction_check_failed")
        return ContradictionReport(has_contradictions=False, contradictions=[])


def verify_claims(answer_text: str, sources: list[dict]) -> VerificationReport:
    """Fact-check the final answer's claims against the actual sources.

    This is deliberately a SEPARATE model call from the one that wrote
    the answer — the same pattern as having a second reviewer check a
    first author's work, rather than asking the original author to
    grade their own homework.
    """
    if not sources:
        return VerificationReport(
            total_claims_checked=0, supported_count=0, unsupported_claims=[]
        )

    prompt = (
        "You are fact-checking an AI-generated answer against the source "
        "material it was supposed to be based on. Break the answer into "
        "its individual factual claims. For each claim, determine whether "
        "it is genuinely supported by the source content below. Report "
        "the total number of claims checked, how many are clearly "
        "supported, and list any claims that are NOT clearly supported "
        "by the sources — these may be hallucinated, overgeneralized "
        "beyond what the sources say, or simply unverifiable from the "
        "given material — along with a short reason for each.\n\n"
        f"ANSWER TO CHECK:\n{answer_text}\n\n"
        f"SOURCE MATERIAL:\n{_format_sources(sources)}"
    )

    try:
        client = _get_client()
        response = client.models.generate_content(
            model=config.MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=VerificationReport,
            ),
        )
        if not isinstance(response.parsed, VerificationReport):
            logger.warning("verification_check_unparseable_response")
            return VerificationReport(
                total_claims_checked=0, supported_count=0, unsupported_claims=[]
            )
        return response.parsed
    except Exception:
        logger.exception("verification_check_failed")
        return VerificationReport(
            total_claims_checked=0, supported_count=0, unsupported_claims=[]
        )
