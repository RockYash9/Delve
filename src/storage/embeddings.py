"""
Local embedding model.

Runs entirely on your machine — no API calls, no per-use cost. This is
what lets brick 4's cache do *semantic* matching ("EV tax credits" ~=
"electric vehicle incentives") rather than exact keyword matching.

The model loads lazily and only once per process, since loading it is
the slow part (a few seconds); encoding individual pieces of text
after that is fast.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        # all-MiniLM-L6-v2: small, fast, good enough quality for this use
        # case, and light enough to run comfortably on a CPU.
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed_text(text: str) -> np.ndarray:
    """Convert text into a vector capturing its semantic meaning."""
    model = _get_model()
    return model.encode(text, convert_to_numpy=True)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """1.0 = identical meaning, 0.0 = unrelated, negative = opposite."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))