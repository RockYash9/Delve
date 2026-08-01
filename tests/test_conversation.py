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
