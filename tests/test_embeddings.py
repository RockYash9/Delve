"""
Tests for src/storage/embeddings.py — only the pure-math parts.

Deliberately does NOT test embed_text() itself, since that requires a
real network call to the Gemini embedding API. That's fine to skip
here: cosine_similarity is where the actual logic lives, and it's
fully testable in isolation.
"""

import numpy as np

from src.storage.embeddings import cosine_similarity


def test_identical_vectors_have_similarity_one():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(v, v) == 1.0


def test_orthogonal_vectors_have_similarity_zero():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(cosine_similarity(a, b)) < 1e-9


def test_opposite_vectors_have_similarity_negative_one():
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    assert cosine_similarity(a, b) == -1.0
