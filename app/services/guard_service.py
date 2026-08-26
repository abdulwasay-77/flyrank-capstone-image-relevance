"""
Guard service — the Mismatch Guard's decision flow (§13.1), applied
top-down over ranked candidates until one is accepted or all are
exhausted.

Check order: category -> subject -> similarity -> confidence. The
subject check was added after direct testing (scripts/debug_full_ranking.py)
proved category alone is too coarse: the vision model tags both red
foxes and gray wolves with category="animal", so a wolf photo could
clear the category check for a fox post, and in the real corpus data,
one wolf photo (gray-wolf-01.jpg) even outranked two real fox photos
by raw similarity while clearing both the old category-match and
similarity-threshold checks. Subject-level matching (fox specifically,
not just "some animal") is what the original capstone brief's own
example reason string implies ("expected fox, detected wolf") — this
was a real, proven gap, not a hypothetical one, and is documented in
BUILDLOG.md.

Per §13.4: every reason string names the specific check that failed
and the actual values involved — never a generic "not a good match".
"""

from dataclasses import dataclass

from app.schemas.guard_config import GuardConfig
from app.services.matching_service import MatchCandidate


@dataclass
class GuardDecision:
    """One evaluated step in the decision trail. candidate is None only
    for the "no category/subject could be determined" and "no
    candidate cleared all checks" NO_MATCH cases — every other
    decision refers to a real ranked candidate."""

    candidate: MatchCandidate | None
    decision: str  # "ACCEPTED" | "REJECTED" | "NO_MATCH"
    reason: str
    category_match: bool


def evaluate(
    ranked_candidates: list[MatchCandidate],
    post_categories: list[str],
    guard_config: GuardConfig,
    post_subjects: list[str] | None = None,
) -> list[GuardDecision]:
    """
    Returns the full decision trail: every candidate actually
    evaluated (in rank order), stopping at the first ACCEPT — matching
    §13.1's `break` behavior exactly, since candidates ranked below an
    accepted one are never evaluated by design, not omitted by
    accident.

    post_categories and post_subjects are each small ranked lists (see
    MatchingService.infer_post_category / infer_post_subjects) rather
    than single strings — a candidate matches if its value appears
    anywhere in the respective list, not only if it equals the single
    top-ranked one. This keeps the guard robust to posts that
    legitimately straddle two close categories/subjects, while still
    rejecting genuinely unrelated ones.

    post_subjects defaults to None (not just an empty list) so that
    older callers/tests exercising only category-level behavior keep
    working unchanged — when None, the subject check is skipped
    entirely, matching the guard's pre-subject-check behavior exactly.
    Real callers (PostService) always pass a real list.
    """
    if not post_categories:
        # Conservative fallback (documented in
        # MatchingService.infer_post_category): if the corpus has no
        # classified images yet, there is no vocabulary to infer a
        # category against, so every candidate would otherwise appear
        # to "fail" a check we can't actually evaluate. NO_MATCH is
        # the honest answer, never a guess.
        return [
            GuardDecision(
                candidate=None,
                decision="NO_MATCH",
                reason="Could not determine an expected category for this post — no classified images exist in the corpus yet.",
                category_match=False,
            )
        ]

    decisions: list[GuardDecision] = []
    expected_category_label = " or ".join(post_categories)
    expected_subject_label = " or ".join(post_subjects) if post_subjects else None

    for candidate in ranked_candidates:
        category_match = candidate.category in post_categories

        if not category_match:
            decisions.append(
                GuardDecision(
                    candidate=candidate,
                    decision="REJECTED",
                    reason=f"Category mismatch: expected {expected_category_label}, detected {candidate.category}",
                    category_match=False,
                )
            )
            continue

        if post_subjects is not None:
            subject_match = candidate.subject in post_subjects
            if not subject_match:
                decisions.append(
                    GuardDecision(
                        candidate=candidate,
                        decision="REJECTED",
                        reason=f"Subject mismatch: expected {expected_subject_label}, detected {candidate.subject}",
                        category_match=True,  # category passed; subject is what failed here
                    )
                )
                continue

        if candidate.similarity_score < guard_config.similarity_threshold:
            decisions.append(
                GuardDecision(
                    candidate=candidate,
                    decision="REJECTED",
                    reason=(
                        f"Similarity {candidate.similarity_score:.2f} below threshold "
                        f"{guard_config.similarity_threshold:.2f}"
                    ),
                    category_match=True,
                )
            )
            continue

        if candidate.confidence < guard_config.confidence_accept_threshold:
            decisions.append(
                GuardDecision(
                    candidate=candidate,
                    decision="REJECTED",
                    reason=(
                        f"Classification confidence {candidate.confidence:.2f} below threshold "
                        f"{guard_config.confidence_accept_threshold:.2f}"
                    ),
                    category_match=True,
                )
            )
            continue

        decisions.append(
            GuardDecision(
                candidate=candidate,
                decision="ACCEPTED",
                reason=(
                    f"Category match ({candidate.category}); subject match ({candidate.subject}); "
                    f"similarity {candidate.similarity_score:.2f} >= "
                    f"{guard_config.similarity_threshold:.2f}; confidence {candidate.confidence:.2f} >= "
                    f"{guard_config.confidence_accept_threshold:.2f}"
                )
                if post_subjects is not None
                else (
                    f"Category match ({candidate.category}); similarity {candidate.similarity_score:.2f} >= "
                    f"{guard_config.similarity_threshold:.2f}; confidence {candidate.confidence:.2f} >= "
                    f"{guard_config.confidence_accept_threshold:.2f}"
                ),
                category_match=True,
            )
        )
        break
    else:
        decisions.append(
            GuardDecision(
                candidate=None,
                decision="NO_MATCH",
                reason="No candidate cleared category, subject, similarity, and confidence checks"
                if post_subjects is not None
                else "No candidate cleared category, similarity, and confidence checks",
                category_match=False,
            )
        )

    return decisions