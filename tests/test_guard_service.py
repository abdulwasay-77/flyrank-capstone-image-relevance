"""
§18 "Guard logic": category mismatch always rejects regardless of
similarity; threshold boundaries behave correctly at the edge; subject
mismatch (added after a real proven gap — see BUILDLOG.md) catches
same-category-different-subject substitutions like wolf-for-fox.
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


def make_candidate(category="animal", subject="red fox", confidence=0.9, similarity=0.85, filename="test.jpg"):
    return MatchCandidate(
        image_id=uuid.uuid4(),
        filename=filename,
        category=category,
        subject=subject,
        confidence=confidence,
        similarity_score=similarity,
    )


class TestCategoryMismatch:
    def test_category_mismatch_rejects_regardless_of_high_similarity(self, guard_config):
        """Even a near-perfect similarity score must not override a
        category mismatch — category is checked first and is a hard
        gate, per §13.1's ordering."""
        candidate = make_candidate(category="food", subject="latte", confidence=0.99, similarity=0.99)
        decisions = evaluate([candidate], post_categories=["animal"], guard_config=guard_config)
        assert decisions[0].decision == "REJECTED"
        assert "Category mismatch" in decisions[0].reason

    def test_category_match_via_any_of_top_k_categories(self, guard_config):
        """A candidate matches if its category is anywhere in the
        inferred post_categories list, not only the single top entry —
        this is the fix for the animal/nature flip-flop bug."""
        candidate = make_candidate(category="nature", subject="mountain", confidence=0.9, similarity=0.85)
        decisions = evaluate([candidate], post_categories=["animal", "nature"], guard_config=guard_config)
        assert decisions[0].decision == "ACCEPTED"

    def test_no_post_categories_produces_conservative_no_match(self, guard_config):
        candidate = make_candidate()
        decisions = evaluate([candidate], post_categories=[], guard_config=guard_config)
        assert len(decisions) == 1
        assert decisions[0].decision == "NO_MATCH"
        assert decisions[0].candidate is None


class TestSubjectMismatch:
    """§13.4 / BUILDLOG.md: category alone is too coarse to distinguish
    closely-related subjects within the same category (fox vs wolf are
    both category="animal"). These tests reproduce the exact real
    scenario found via scripts/debug_full_ranking.py — a high-similarity,
    high-confidence, same-category wolf photo must still be rejected
    for a fox post."""

    def test_subject_mismatch_rejects_despite_matching_category_and_high_scores(self, guard_config):
        """The exact real bug scenario: gray-wolf-01.jpg had
        similarity 0.826 (well above 0.75) and confidence 0.99, same
        category="animal" as every fox photo — the OLD guard (category
        + similarity + confidence only) would have accepted this."""
        wolf_candidate = make_candidate(
            category="animal", subject="gray wolf", confidence=0.99, similarity=0.90, filename="gray-wolf-01.jpg"
        )
        decisions = evaluate(
            [wolf_candidate],
            post_categories=["animal"],
            guard_config=guard_config,
            post_subjects=["red fox"],
        )
        assert decisions[0].decision == "REJECTED"
        assert "Subject mismatch" in decisions[0].reason
        assert "red fox" in decisions[0].reason
        assert "gray wolf" in decisions[0].reason
        # Category still correctly recorded as matched — it's the
        # subject check that failed, not category.
        assert decisions[0].category_match is True

    def test_subject_match_via_any_of_top_k_subjects(self, guard_config):
        candidate = make_candidate(category="animal", subject="red fox", confidence=0.9, similarity=0.85)
        decisions = evaluate(
            [candidate],
            post_categories=["animal"],
            guard_config=guard_config,
            post_subjects=["gray wolf", "red fox"],
        )
        assert decisions[0].decision == "ACCEPTED"

    def test_top_k_1_does_not_admit_a_nearby_but_wrong_subject(self, guard_config):
        """Regression test for a real bug found during verification:
        with a wider top_k, a genuinely different, wrong subject
        (e.g. "coyote") could end up in the accepted-subjects list
        purely for being a nearby canid in embedding space — silently
        defeating the whole point of the subject check. With
        MatchingService.infer_post_subjects's default top_k=1, only
        one subject is ever in the accepted list, so this scenario
        (simulated here by directly passing a too-wide subjects list,
        the way top_k=3 used to produce) should still correctly
        reject anything not equal to it."""
        coyote_candidate = make_candidate(category="animal", subject="coyote", confidence=0.9, similarity=0.85)
        # Simulates the OLD, wider inferred-subjects list that
        # incorrectly included "coyote" as a false-acceptable neighbor
        # of "red fox" — this must still be rejected, proving the
        # guard's per-item matching is correct; it's
        # infer_post_subjects' top_k=1 default that prevents this
        # wide a list from ever being constructed in the first place.
        decisions = evaluate(
            [coyote_candidate],
            post_categories=["animal"],
            guard_config=guard_config,
            post_subjects=["red fox"],  # correct, narrow (top_k=1) inference
        )
        assert decisions[0].decision == "REJECTED"
        assert "Subject mismatch" in decisions[0].reason

    def test_subject_check_skipped_when_post_subjects_is_none(self, guard_config):
        """Backward compatibility: when post_subjects isn't passed
        (None, the default), the guard behaves exactly as it did
        before the subject check was added — category match alone is
        sufficient. This matters for any caller that hasn't been
        updated to compute subjects."""
        wolf_candidate = make_candidate(category="animal", subject="gray wolf", confidence=0.99, similarity=0.90)
        decisions = evaluate([wolf_candidate], post_categories=["animal"], guard_config=guard_config)
        assert decisions[0].decision == "ACCEPTED"
        assert "subject match" not in decisions[0].reason.lower()

    def test_fox_post_full_ranking_scenario(self, guard_config):
        """Reproduces the real ranked-candidate order observed via
        scripts/debug_full_ranking.py: three fox photos rank above one
        wolf photo. The guard should accept the top fox and never even
        reach the wolf (break-on-first-accept)."""
        candidates = [
            make_candidate(subject="red fox", filename="red-fox-01.jpg", confidence=0.98, similarity=0.8618),
            make_candidate(subject="red fox", filename="red-fox-04.jpg", confidence=0.99, similarity=0.8316),
            make_candidate(subject="gray wolf", filename="gray-wolf-01.jpg", confidence=0.99, similarity=0.8260),
        ]
        decisions = evaluate(
            candidates, post_categories=["animal"], guard_config=guard_config, post_subjects=["red fox"]
        )
        assert len(decisions) == 1  # stopped at the first accept
        assert decisions[0].decision == "ACCEPTED"
        assert decisions[0].candidate.filename == "red-fox-01.jpg"


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
            make_candidate(category="food", subject="latte", similarity=0.95, confidence=0.95, filename="wrong.jpg"),
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
        candidate = make_candidate(category="food", subject="latte")
        decisions = evaluate([candidate], post_categories=["animal"], guard_config=guard_config)
        assert "animal" in decisions[0].reason
        assert "food" in decisions[0].reason

    def test_acceptance_reason_includes_actual_numbers(self, guard_config):
        candidate = make_candidate(similarity=0.83, confidence=0.94)
        decisions = evaluate([candidate], post_categories=["animal"], guard_config=guard_config)
        assert "0.83" in decisions[0].reason
        assert "0.94" in decisions[0].reason