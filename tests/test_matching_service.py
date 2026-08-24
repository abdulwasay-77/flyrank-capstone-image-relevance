"""
§18 "Matching accuracy": known embedding pairs produce expected
relative ranking. Uses fabricated (not real API) vectors — cosine
similarity is pure math, testable with zero network calls.
"""

import math

import pytest

from app.services.matching_service import cosine_similarity


class TestCosineSimilarity:
    def test_identical_vectors_have_similarity_one(self):
        v = [1.0, 2.0, 3.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_have_similarity_zero(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors_have_similarity_negative_one(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero_not_nan(self):
        """A zero-magnitude vector would divide by zero in a naive
        implementation — must return 0.0, never NaN or raise."""
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        result = cosine_similarity(a, b)
        assert result == 0.0
        assert not math.isnan(result)

    def test_scale_invariant(self):
        """Cosine similarity only cares about direction, not
        magnitude — scaling a vector must not change the result."""
        a = [1.0, 2.0, 3.0]
        b = [2.0, 4.0, 6.0]  # same direction, different magnitude
        assert cosine_similarity(a, b) == pytest.approx(1.0)


class TestRelativeRanking:
    """§18's example: fox caption should be closer to a fox post than
    a wolf caption is. Simulated here with hand-constructed vectors in
    a small semantic-ish space (not real embeddings — real embeddings
    are exercised only in manual smoke tests per §18's "mocked in
    automated tests" rule), verifying the ranking LOGIC is correct
    regardless of where real vectors come from.
    """

    def test_closer_vector_ranks_above_farther_vector(self):
        # A "fox post" vector, and two candidate vectors — one
        # deliberately closer (small angle), one deliberately farther
        # (larger angle) in a toy 2D space.
        post_vector = [1.0, 0.0]
        fox_caption_vector = [0.95, 0.05]  # small angle from post
        wolf_caption_vector = [0.3, 0.95]  # large angle from post

        fox_score = cosine_similarity(post_vector, fox_caption_vector)
        wolf_score = cosine_similarity(post_vector, wolf_caption_vector)

        assert fox_score > wolf_score

    def test_ranking_sorts_descending_by_similarity(self):
        """Mirrors MatchingService.get_ranked_candidates' sort step
        directly, without needing a live database."""
        post_vector = [1.0, 0.0]
        candidates = [
            ("far", [0.1, 0.99]),
            ("close", [0.98, 0.1]),
            ("medium", [0.7, 0.7]),
        ]
        scored = [(name, cosine_similarity(post_vector, vec)) for name, vec in candidates]
        scored.sort(key=lambda pair: pair[1], reverse=True)

        assert [name for name, _ in scored] == ["close", "medium", "far"]