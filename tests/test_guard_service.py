"""
§18 "Guard logic": category mismatch always rejects regardless of
similarity; threshold boundaries behave correctly at the edge.
Pure unit tests on GuardService.evaluate with fabricated inputs — no
network, no database.
"""

import uuid

import pytest

from app.schemas.guard_config import GuardConfig
from app.services.guard_service import evaluate
from app.services.matching_service import MatchCandidate


@pytest.fixture
def guard_config():
    return GuardConfig(
        similarity_threshold=0.75,
        confidence_accept_threshold=0.60,
        confidence_flag_threshold=0.60,
        max_retries=3,
    )


def make_candidate(category="animal", confidence=0.9, similarity=0.85, filename="test.jpg"):
    return MatchCandidate(
        image_id=uuid.uuid4(),
        filename=filename,
        category=category,
        confidence=confidence,
        similarity_score=similarity,
    )


class TestCategoryMismatch:
    def test_category_mismatch_rejects_regardless_of_high_similarity(self, guard_config):
        """Even a near-perfect similarity score must not override a
        category mismatch — category is checked first and is a hard
        gate, per §13.1's ordering."""
        candidate = make_candidate(category="food", confidence=0.99, similarity=0.99)
        decisions = evaluate([candidate], post_categories=["animal"], guard_config=guard_config)
        assert decisions[0].decision == "REJECTED"
        assert "Category mismatch" in decisions[0].reason

    def test_category_match_via_any_of_top_k_categories(self, guard_config):
        """A candidate matches if its category is anywhere in the
        inferred post_categories list, not only the single top entry —
        this is the fix for the animal/nature flip-flop bug."""
        candidate = make_candidate(category="nature", confidence=0.9, similarity=0.85)
        decisions = evaluate([candidate], post_categories=["animal", "nature"], guard_config=guard_config)
        assert decisions[0].decision == "ACCEPTED"

    def test_no_post_categories_produces_conservative_no_match(self, guard_config):
        candidate = make_candidate()
        decisions = evaluate([candidate], post_categories=[], guard_config=guard_config)
        assert len(decisions) == 1
        assert decisions[0].decision == "NO_MATCH"
        assert decisions[0].candidate is None


class TestSimilarityThresholdBoundary:
    def test_similarity_exactly_at_threshold_passes(self, guard_config):
        """>= threshold, not > threshold — an exact match at the
        boundary should be accepted, not rejected."""
        candidate = make_candidate(similarity=0.75, confidence=0.9)
        decisions = evaluate([candidate], post_categories=["animal"], guard_config=guard_config)
        assert decisions[0].decision == "ACCEPTED"

    def test_similarity_just_below_threshold_rejects(self, guard_config):
        candidate = make_candidate(similarity=0.749, confidence=0.9)
        decisions = evaluate([candidate], post_categories=["animal"], guard_config=guard_config)
        assert decisions[0].decision == "REJECTED"
        assert "Similarity" in decisions[0].reason

    def test_similarity_just_above_threshold_passes(self, guard_config):
        candidate = make_candidate(similarity=0.751, confidence=0.9)
        decisions = evaluate([candidate], post_categories=["animal"], guard_config=guard_config)
        assert decisions[0].decision == "ACCEPTED"


class TestConfidenceThresholdBoundary:
    def test_confidence_exactly_at_threshold_passes(self, guard_config):
        candidate = make_candidate(similarity=0.9, confidence=0.60)
        decisions = evaluate([candidate], post_categories=["animal"], guard_config=guard_config)
        assert decisions[0].decision == "ACCEPTED"

    def test_confidence_just_below_threshold_rejects(self, guard_config):
        candidate = make_candidate(similarity=0.9, confidence=0.599)
        decisions = evaluate([candidate], post_categories=["animal"], guard_config=guard_config)
        assert decisions[0].decision == "REJECTED"
        assert "confidence" in decisions[0].reason.lower()


class TestFallThroughBehavior:
    def test_first_acceptable_candidate_wins_and_stops(self, guard_config):
        """Per §13.1's break behavior — candidates ranked below an
        accepted one are never evaluated."""
        candidates = [
            make_candidate(category="animal", similarity=0.9, confidence=0.9, filename="a.jpg"),
            make_candidate(category="animal", similarity=0.85, confidence=0.9, filename="b.jpg"),
        ]
        decisions = evaluate(candidates, post_categories=["animal"], guard_config=guard_config)
        assert len(decisions) == 1
        assert decisions[0].decision == "ACCEPTED"
        assert decisions[0].candidate.filename == "a.jpg"

    def test_rejects_then_accepts_next_candidate(self, guard_config):
        candidates = [
            make_candidate(category="food", similarity=0.95, confidence=0.95, filename="wrong.jpg"),
            make_candidate(category="animal", similarity=0.8, confidence=0.9, filename="right.jpg"),
        ]
        decisions = evaluate(candidates, post_categories=["animal"], guard_config=guard_config)
        assert len(decisions) == 2
        assert decisions[0].decision == "REJECTED"
        assert decisions[1].decision == "ACCEPTED"
        assert decisions[1].candidate.filename == "right.jpg"

    def test_all_candidates_fail_produces_no_match(self, guard_config):
        candidates = [
            make_candidate(category="animal", similarity=0.5, confidence=0.9),
            make_candidate(category="animal", similarity=0.4, confidence=0.9),
        ]
        decisions = evaluate(candidates, post_categories=["animal"], guard_config=guard_config)
        assert decisions[-1].decision == "NO_MATCH"
        assert all(d.decision != "ACCEPTED" for d in decisions)

    def test_empty_candidate_list_produces_no_match(self, guard_config):
        decisions = evaluate([], post_categories=["animal"], guard_config=guard_config)
        assert len(decisions) == 1
        assert decisions[0].decision == "NO_MATCH"


class TestReasonStrings:
    """§13.4: every reason must name the specific check and the
    actual numbers involved, never a generic message."""

    def test_category_mismatch_reason_names_both_categories(self, guard_config):
        candidate = make_candidate(category="food")
        decisions = evaluate([candidate], post_categories=["animal"], guard_config=guard_config)
        assert "animal" in decisions[0].reason
        assert "food" in decisions[0].reason

    def test_acceptance_reason_includes_actual_numbers(self, guard_config):
        candidate = make_candidate(similarity=0.83, confidence=0.94)
        decisions = evaluate([candidate], post_categories=["animal"], guard_config=guard_config)
        assert "0.83" in decisions[0].reason
        assert "0.94" in decisions[0].reason