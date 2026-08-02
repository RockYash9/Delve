"""
Tests for src/storage/cache.py.

Uses a small deterministic fake embedder instead of the real
sentence-transformers model — this is a unit test for the caching and
similarity-threshold LOGIC, not a test of embedding quality itself.
Each test gets its own throwaway SQLite file via the isolated_db fixture,
so tests never touch or depend on your real delve_cache.db.
"""

import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

import config
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


def _insert_with_fetched_at(fetched_at: str) -> None:
    """Insert a row bypassing store_result's automatic timestamp, so
    TTL expiration can be tested deterministically without waiting.
    """
    conn = cache._get_connection()
    conn.execute(
        "INSERT INTO chunks (query, title, url, content, embedding, fetched_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            "electric vehicle tax credits",
            "EV Incentives 2026",
            "https://example.com/ev",
            "content",
            json.dumps(_fake_embed("electric vehicle tax credits").tolist()),
            fetched_at,
        ),
    )
    conn.commit()
    conn.close()


def test_stale_entry_beyond_ttl_is_excluded(monkeypatch):
    monkeypatch.setattr(cache, "embed_text", _fake_embed)
    monkeypatch.setattr(config, "CACHE_TTL_HOURS", 24)

    too_old = (datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=25)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    _insert_with_fetched_at(too_old)

    assert cache.find_similar("EV tax incentives") == []


def test_fresh_entry_within_ttl_is_included(monkeypatch):
    monkeypatch.setattr(cache, "embed_text", _fake_embed)
    monkeypatch.setattr(config, "CACHE_TTL_HOURS", 24)

    recent = (datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=1)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    _insert_with_fetched_at(recent)

    matches = cache.find_similar("EV tax incentives")
    assert len(matches) == 1


def test_ttl_boundary_respects_configured_hours(monkeypatch):
    monkeypatch.setattr(cache, "embed_text", _fake_embed)
    monkeypatch.setattr(config, "CACHE_TTL_HOURS", 1)  # much stricter TTL

    two_hours_old = (
        datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=2)
    ).strftime("%Y-%m-%d %H:%M:%S")
    _insert_with_fetched_at(two_hours_old)

    # 2 hours old, but TTL is only 1 hour — should be excluded now even
    # though the same data would have passed with the default 24h TTL.
    assert cache.find_similar("EV tax incentives") == []
