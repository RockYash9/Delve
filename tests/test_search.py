"""
Tests for src/tools/search.py.

search.py uses a factory (make_web_search_tool) rather than one shared
module-level web_search function — this is what keeps each conversation's
search citations isolated from every other conversation, which matters
once multiple sessions can run concurrently (brick 7's API). These tests
verify both the caching decision logic AND that isolation.
"""

from unittest.mock import MagicMock

from src.tools import search


def test_cache_hit_skips_tavily_call(monkeypatch):
    monkeypatch.setattr(
        search.cache,
        "find_similar",
        lambda query, top_k=5: [
            (0.9, "Cached Title", "https://example.com/cached", "cached content")
        ],
    )
    mock_tavily_instance = MagicMock()
    monkeypatch.setattr(search, "TavilyClient", lambda api_key: mock_tavily_instance)

    sources: list[dict] = []
    web_search = search.make_web_search_tool(sources)
    result = web_search("some query")

    assert "Cached Title" in result
    mock_tavily_instance.search.assert_not_called()
    assert sources == [
        {
            "title": "Cached Title",
            "url": "https://example.com/cached",
            "content": "cached content",
        }
    ]


def test_cache_miss_calls_tavily_and_populates_cache(monkeypatch):
    monkeypatch.setattr(search.cache, "find_similar", lambda query, top_k=5: [])

    stored_calls = []
    monkeypatch.setattr(
        search.cache,
        "store_result",
        lambda query, title, url, content: stored_calls.append(
            (query, title, url, content)
        ),
    )

    mock_tavily_instance = MagicMock()
    mock_tavily_instance.search.return_value = {
        "results": [
            {
                "title": "Live Result",
                "url": "https://example.com/live",
                "content": "live content",
            }
        ]
    }
    monkeypatch.setattr(search, "TavilyClient", lambda api_key: mock_tavily_instance)

    sources: list[dict] = []
    web_search = search.make_web_search_tool(sources)
    result = web_search("new query")

    assert "Live Result" in result
    assert len(stored_calls) == 1
    assert stored_calls[0][1] == "Live Result"
    assert sources == [
        {
            "title": "Live Result",
            "url": "https://example.com/live",
            "content": "live content",
        }
    ]


def test_no_results_returns_friendly_message_not_crash(monkeypatch):
    monkeypatch.setattr(search.cache, "find_similar", lambda query, top_k=5: [])

    mock_tavily_instance = MagicMock()
    mock_tavily_instance.search.return_value = {"results": []}
    monkeypatch.setattr(search, "TavilyClient", lambda api_key: mock_tavily_instance)

    web_search = search.make_web_search_tool([])
    result = web_search("obscure query with no results")

    assert "No search results found" in result


def test_two_tool_instances_have_isolated_sources(monkeypatch):
    """The whole point of the factory: two conversations' search tools
    must never share or leak into each other's sources list."""
    monkeypatch.setattr(
        search.cache,
        "find_similar",
        lambda query, top_k=5: [
            (0.9, "Some Title", "https://example.com/x", "content")
        ],
    )

    sources_a: list[dict] = []
    sources_b: list[dict] = []
    web_search_a = search.make_web_search_tool(sources_a)
    _web_search_b = search.make_web_search_tool(sources_b)  # exists but deliberately not called

    web_search_a("query for conversation A")

    assert len(sources_a) == 1
    assert len(sources_b) == 0  # untouched by conversation A's search
