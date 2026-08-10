"""
Tests for src/memory/conversation.py — specifically send_stream()'s
behavior.

Conversation.__init__ makes a real network connection (genai.Client),
so these tests build an instance via __new__ and manually wire in
fakes instead of calling __init__ — a deliberate, narrow bypass to
unit-test the logic in isolation.

send_stream() deliberately does NOT use Gemini's raw
send_message_stream() — see the docstring in conversation.py for why
(a confirmed Gemini SDK bug where streaming + automatic function
calling can silently drop the final answer). Instead it calls send()
(the reliable non-streaming path) and chunks the result itself, so
these tests verify that chunking/interleaving behavior.
"""

from src.memory.conversation import Conversation


def _make_conversation(reply_text, status_events=None):
    conv = Conversation.__new__(Conversation)  # skip __init__, no network
    conv.sources = []
    conv.status_events = status_events if status_events is not None else []
    conv.send = lambda message: reply_text  # type: ignore[method-assign]
    return conv


def test_send_stream_yields_the_full_reply_as_word_chunks():
    conv = _make_conversation("Hello world")

    events = list(conv.send_stream("hi"))

    assert events == [
        {"type": "token", "text": "Hello"},
        {"type": "token", "text": " world"},
    ]


def test_send_stream_reassembles_to_the_original_text():
    original = "The federal EV tax credit changed significantly in 2026."
    conv = _make_conversation(original)

    events = list(conv.send_stream("what are EV tax credits?"))
    reassembled = "".join(e["text"] for e in events)

    assert reassembled == original


def test_send_stream_flushes_status_before_any_token():
    status_events = ["🔍 searching: EV tax credits 2026"]
    conv = _make_conversation("The answer is here.", status_events)

    events = list(conv.send_stream("what are EV tax credits?"))

    assert events[0] == {"type": "status", "text": "🔍 searching: EV tax credits 2026"}
    assert events[1]["type"] == "token"


def test_send_stream_with_no_status_events_yields_only_tokens():
    conv = _make_conversation("just text")

    events = list(conv.send_stream("hi"))

    assert all(e["type"] == "token" for e in events)


def test_send_stream_with_multiple_status_events_flushes_all_before_tokens():
    status_events = ["🔍 searching: query one", "🔍 searching: query two"]
    conv = _make_conversation("final answer", status_events)

    events = list(conv.send_stream("hi"))
    types_in_order = [e["type"] for e in events]

    assert types_in_order == ["status", "status", "token", "token"]


def _make_conversation_with_sources(reply_text, new_sources_added):
    """Like _make_conversation, but the fake send() also appends to
    conv.sources — simulating a real search tool call having run."""
    conv = Conversation.__new__(Conversation)
    conv.sources = []
    conv.status_events = []

    def fake_send(message):
        conv.sources.extend(new_sources_added)
        return reply_text

    conv.send = fake_send  # type: ignore[method-assign]
    return conv


def test_send_stream_yields_sources_event_after_tokens():
    new_sources = [{"title": "EV Guide", "url": "https://example.com/ev"}]
    conv = _make_conversation_with_sources("the answer", new_sources)

    events = list(conv.send_stream("hi"))

    assert events[-1] == {"type": "sources", "sources": new_sources}
    assert all(e["type"] == "token" for e in events[:-1])


def test_send_stream_deduplicates_sources_by_url():
    duplicated = [
        {"title": "EV Guide", "url": "https://example.com/ev"},
        {"title": "EV Guide (again)", "url": "https://example.com/ev"},
        {"title": "Other Source", "url": "https://example.com/other"},
    ]
    conv = _make_conversation_with_sources("the answer", duplicated)

    events = list(conv.send_stream("hi"))
    sources_event = next(e for e in events if e["type"] == "sources")

    assert len(sources_event["sources"]) == 2


def test_send_stream_omits_sources_event_when_no_search_happened():
    conv = _make_conversation("just an answer, no search needed")

    events = list(conv.send_stream("what is 2+2?"))

    assert all(e["type"] != "sources" for e in events)


def test_send_stream_only_includes_this_turns_new_sources():
    # Simulate a conversation that already has sources from an earlier
    # turn — send_stream must only report what THIS call added, not
    # everything accumulated in the conversation so far.
    conv = Conversation.__new__(Conversation)
    conv.sources = [{"title": "Old Source", "url": "https://example.com/old"}]
    conv.status_events = []

    new_source = {"title": "New Source", "url": "https://example.com/new"}

    def fake_send(message):
        conv.sources.append(new_source)
        return "the answer"

    conv.send = fake_send  # type: ignore[method-assign]

    events = list(conv.send_stream("hi"))
    sources_event = next(e for e in events if e["type"] == "sources")

    assert sources_event["sources"] == [new_source]


# ---------- Deep Research mode (deep_research=True/False) ----------


def test_deep_research_off_by_default_never_calls_verification(monkeypatch):
    """The core design guarantee: with deep_research left at its default
    (False), check_contradictions/verify_claims must never be called —
    zero extra API calls unless the caller explicitly opts in."""
    from src.memory import conversation as conversation_module

    check_calls = []
    verify_calls = []
    monkeypatch.setattr(
        conversation_module.verification,
        "check_contradictions",
        lambda sources: check_calls.append(sources),
    )
    monkeypatch.setattr(
        conversation_module.verification,
        "verify_claims",
        lambda answer, sources: verify_calls.append((answer, sources)),
    )

    new_sources = [
        {"title": "A", "url": "https://example.com/a", "content": "x"},
        {"title": "B", "url": "https://example.com/b", "content": "y"},
    ]
    conv = _make_conversation_with_sources("the answer", new_sources)

    events = list(conv.send_stream("hi"))  # deep_research defaults to False

    assert check_calls == []
    assert verify_calls == []
    assert all(e["type"] != "contradictions" for e in events)
    assert all(e["type"] != "verification" for e in events)


def test_deep_research_true_with_no_sources_skips_the_extra_calls(monkeypatch):
    """No search happened this turn, so there's nothing to cross-reference
    or verify against — deep_research=True shouldn't force calls that
    have nothing meaningful to check."""
    from src.memory import conversation as conversation_module

    check_calls = []
    monkeypatch.setattr(
        conversation_module.verification,
        "check_contradictions",
        lambda sources: check_calls.append(sources),
    )

    conv = _make_conversation("just an answer, no search needed")

    events = list(conv.send_stream("what is 2+2?", deep_research=True))

    assert check_calls == []
    assert all(e["type"] not in ("contradictions", "verification") for e in events)


def test_deep_research_true_yields_contradictions_when_found(monkeypatch):
    from src.memory import conversation as conversation_module

    fake_report = conversation_module.verification.ContradictionReport(
        has_contradictions=True,
        contradictions=[
            conversation_module.verification.Contradiction(
                topic="launch date",
                source_a_title="A",
                source_a_claim="March",
                source_b_title="B",
                source_b_claim="June",
                explanation="Sources disagree.",
            )
        ],
    )
    monkeypatch.setattr(
        conversation_module.verification, "check_contradictions", lambda sources: fake_report
    )
    monkeypatch.setattr(
        conversation_module.verification,
        "verify_claims",
        lambda answer, sources: conversation_module.verification.VerificationReport(
            total_claims_checked=0, supported_count=0, unsupported_claims=[]
        ),
    )

    new_sources = [
        {"title": "A", "url": "https://example.com/a", "content": "March"},
        {"title": "B", "url": "https://example.com/b", "content": "June"},
    ]
    conv = _make_conversation_with_sources("the answer", new_sources)

    events = list(conv.send_stream("hi", deep_research=True))
    contradiction_events = [e for e in events if e["type"] == "contradictions"]

    assert len(contradiction_events) == 1
    assert contradiction_events[0]["contradictions"][0]["topic"] == "launch date"


def test_deep_research_true_omits_contradictions_event_when_none_found(monkeypatch):
    from src.memory import conversation as conversation_module

    empty_report = conversation_module.verification.ContradictionReport(
        has_contradictions=False, contradictions=[]
    )
    monkeypatch.setattr(
        conversation_module.verification, "check_contradictions", lambda sources: empty_report
    )
    monkeypatch.setattr(
        conversation_module.verification,
        "verify_claims",
        lambda answer, sources: conversation_module.verification.VerificationReport(
            total_claims_checked=1, supported_count=1, unsupported_claims=[]
        ),
    )

    new_sources = [
        {"title": "A", "url": "https://example.com/a", "content": "x"},
        {"title": "B", "url": "https://example.com/b", "content": "y"},
    ]
    conv = _make_conversation_with_sources("the answer", new_sources)

    events = list(conv.send_stream("hi", deep_research=True))

    assert all(e["type"] != "contradictions" for e in events)


def test_deep_research_true_always_yields_verification_event(monkeypatch):
    """Unlike contradictions (only emitted when found), the verification
    summary is always sent when deep_research runs — the client should
    see "checked, all supported" as much as "checked, found issues"."""
    from src.memory import conversation as conversation_module

    monkeypatch.setattr(
        conversation_module.verification,
        "check_contradictions",
        lambda sources: conversation_module.verification.ContradictionReport(
            has_contradictions=False, contradictions=[]
        ),
    )
    fake_verification = conversation_module.verification.VerificationReport(
        total_claims_checked=3,
        supported_count=2,
        unsupported_claims=[
            conversation_module.verification.UnsupportedClaim(
                claim="It costs $50.", reason="No source mentions a price."
            )
        ],
    )
    monkeypatch.setattr(
        conversation_module.verification,
        "verify_claims",
        lambda answer, sources: fake_verification,
    )

    new_sources = [{"title": "A", "url": "https://example.com/a", "content": "x"}]
    conv = _make_conversation_with_sources("the answer", new_sources)

    events = list(conv.send_stream("hi", deep_research=True))
    verification_events = [e for e in events if e["type"] == "verification"]

    assert len(verification_events) == 1
    assert verification_events[0]["total_claims_checked"] == 3
    assert verification_events[0]["supported_count"] == 2
    assert len(verification_events[0]["unsupported_claims"]) == 1
