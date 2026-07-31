"""
Tests for src/tools/search.py.

The key behavior under test: web_search() must check the cache first
and skip Tavily entirely on a hit, and must populate the cache on a
miss. Both TavilyClient and the cache module are mocked — these are
unit tests for the *decision logic*, not integration tests against
real APIs.
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

    result = search.web_search("some query")

    assert "Cached Title" in result
    mock_tavily_instance.search.assert_not_called()


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

    result = search.web_search("new query")

    assert "Live Result" in result
    assert len(stored_calls) == 1
    assert stored_calls[0][1] == "Live Result"


def test_no_results_returns_friendly_message_not_crash(monkeypatch):
    monkeypatch.setattr(search.cache, "find_similar", lambda query, top_k=5: [])

    mock_tavily_instance = MagicMock()
    mock_tavily_instance.search.return_value = {"results": []}
    monkeypatch.setattr(search, "TavilyClient", lambda api_key: mock_tavily_instance)

    result = search.web_search("obscure query with no results")

    assert "No search results found" in result
