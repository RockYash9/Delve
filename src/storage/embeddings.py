"""
Embedding model — via the Gemini API, not a local model.

This originally ran a local sentence-transformers model, but that
requires PyTorch, and PyTorch alone exceeds the 512MB memory limit on
Render's free tier — the app would get OOM-killed on deploy before it
could even serve a request. Switching to Gemini's hosted embedding API
(gemini-embedding-001) removes that dependency entirely: no local ML
model, no PyTorch, and it uses the same GEMINI_API_KEY already
required everywhere else in this project, at no extra cost within the
free tier (~1,500 embedding requests/day as of when this was written).

The tradeoff: embeddings now require a network call instead of running
fully offline. Given the search tool already requires network access
for every query anyway, this doesn't change the app's actual
dependency on connectivity.
"""

import numpy as np
from google import genai
from google.genai.types import EmbedContentConfig

import config

_client: genai.Client | None = None

# 768 (vs. the model's default 3072) via Matryoshka Representation
# Learning truncation — meaningfully smaller to store per cached
# chunk, with only a marginal quality tradeoff for this use case
# (matching short queries against each other, not high-precision
# retrieval over a huge corpus).
EMBEDDING_DIMENSIONS = 768


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


def embed_text(text: str) -> np.ndarray:
    """Convert text into a vector capturing its semantic meaning."""
    client = _get_client()
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=EmbedContentConfig(output_dimensionality=EMBEDDING_DIMENSIONS),
    )
    if not response.embeddings or response.embeddings[0].values is None:
        raise RuntimeError("Gemini's embedding API returned no embedding for this text.")
    return np.array(response.embeddings[0].values)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """1.0 = identical meaning, 0.0 = unrelated, negative = opposite."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
