"""
Local knowledge cache.

Every web page Delve fetches gets stored here, along with an embedding
of its content. Future searches check this cache first via semantic
similarity — if something close enough has already been fetched, we
reuse it instead of spending a Tavily API call. This is what makes the
tool accumulate its own knowledge base the more it's used, rather than
staying purely a live-search wrapper.
"""

import json
import sqlite3
from pathlib import Path

import numpy as np

from src.storage.embeddings import cosine_similarity, embed_text

# Lives at the project root, next to main.py. Already covered by
# .gitignore ("*.db") so it never gets committed.
DB_PATH = Path(__file__).resolve().parent.parent.parent / "delve_cache.db"

# How close a cached chunk's meaning must be to the query to reuse it
# instead of doing a fresh search. 1.0 = identical meaning. Tuned to be
# fairly strict so we don't serve stale or off-topic cached content.
SIMILARITY_THRESHOLD = 0.75


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT,
            title TEXT,
            url TEXT,
            content TEXT,
            embedding TEXT,
            fetched_at TEXT DEFAULT (datetime('now'))
        )
        """
    )
    return conn


def store_result(query: str, title: str, url: str, content: str) -> None:
    """Cache one search result chunk, keyed by an embedding of the query.

    We embed the *query* (not the article content) because future
    lookups compare a new query against this — query-to-query semantic
    similarity is a much stronger, more reliable signal than comparing
    a short query against a long article's embedding, which tends to
    score lower even when they're clearly about the same thing.
    """
    embedding = embed_text(query)
    conn = _get_connection()
    conn.execute(
        "INSERT INTO chunks (query, title, url, content, embedding) "
        "VALUES (?, ?, ?, ?, ?)",
        (query, title, url, content, json.dumps(embedding.tolist())),
    )
    conn.commit()
    conn.close()


def find_similar(query: str, top_k: int = 5) -> list[tuple[float, str, str, str]]:
    """Return cached (score, title, url, content) tuples similar to query.

    Only returns chunks at or above SIMILARITY_THRESHOLD, best matches
    first. Empty list means "nothing close enough — go search live."
    """
    conn = _get_connection()
    rows = conn.execute("SELECT title, url, content, embedding FROM chunks").fetchall()
    conn.close()

    if not rows:
        return []

    query_embedding = embed_text(query)
    scored = []
    for title, url, content, embedding_json in rows:
        embedding = np.array(json.loads(embedding_json))
        score = cosine_similarity(query_embedding, embedding)
        if score >= SIMILARITY_THRESHOLD:
            scored.append((score, title, url, content))

    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:top_k]
