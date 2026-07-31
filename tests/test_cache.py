"""
Tests for src/storage/cache.py.

Uses a small deterministic fake embedder instead of the real
sentence-transformers model — this is a unit test for the caching and
similarity-threshold LOGIC, not a test of embedding quality itself.
Each test gets its own throwaway SQLite file via the isolated_db fixture,
so tests never touch or depend on your real delve_cache.db.
"""

import numpy as np
import pytest

from src.storage import cache


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Point the cache at a throwaway database for every test in this file."""
    monkeypatch.setattr(cache, "DB_PATH", tmp_path / "test_cache.db")


def _fake_embed(text: str) -> np.ndarray:
    """Deterministic stand-in for embed_text: groups text by topic keyword
    rather than real semantic meaning, just enough to test matching logic.
    """
    text = text.lower()
    if "ev" in text or "electric vehicle" in text or "tax credit" in text:
        return np.array([1.0, 0.1, 0.0])
    return np.array([0.0, 0.1, 1.0])


def test_store_and_find_similar_query(monkeypatch):
    monkeypatch.setattr(cache, "embed_text", _fake_embed)

    cache.store_result(
        query="electric vehicle tax credits",
        title="EV Incentives 2026",
        url="https://example.com/ev",
        content="Some article content about EV tax credits.",
    )

    matches = cache.find_similar("EV tax incentives")

    assert len(matches) == 1
    score, title, url, content = matches[0]
    assert title == "EV Incentives 2026"
    assert score >= cache.SIMILARITY_THRESHOLD


def test_unrelated_query_does_not_match(monkeypatch):
    monkeypatch.setattr(cache, "embed_text", _fake_embed)

    cache.store_result(
        query="electric vehicle tax credits",
        title="EV Incentives 2026",
        url="https://example.com/ev",
        content="Some article content.",
    )

    assert cache.find_similar("best pizza toppings") == []


def test_empty_cache_returns_no_matches(monkeypatch):
    monkeypatch.setattr(cache, "embed_text", _fake_embed)
    assert cache.find_similar("anything at all") == []


def test_top_k_limits_number_of_results(monkeypatch):
    monkeypatch.setattr(cache, "embed_text", _fake_embed)

    for i in range(10):
        cache.store_result(
            query="electric vehicle tax credits",
            title=f"EV Article {i}",
            url=f"https://example.com/ev{i}",
            content="content",
        )

    matches = cache.find_similar("EV tax credits", top_k=3)
    assert len(matches) == 3
