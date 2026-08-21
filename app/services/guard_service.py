"""
Guard service — the Mismatch Guard's decision flow (§13.1), applied
top-down over ranked candidates until one is accepted or all are
exhausted.

This is a direct, literal implementation of the pseudocode in §13.1 —
same three checks in the same order (category, then similarity, then
confidence), same "first acceptable candidate wins" behavior, same
NO_MATCH fallback via Python's own for/else (which mirrors the
spec's own pseudocode using for/else). Per §13.4: every reason string
names the specific check that failed and the actual numbers involved
— never a generic "not a good match".
"""

from dataclasses import dataclass

from app.schemas.guard_config import GuardConfig
from app.services.matching_service import MatchCandidate


@dataclass
class GuardDecision:
    """One evaluated step in the decision trail. candidate is None only
    for the "no category could be determined" and "no candidate
    cleared all checks" NO_MATCH cases — every other decision refers
    to a real ranked candidate."""

    candidate: MatchCandidate | None
    decision: str  # "ACCEPTED" | "REJECTED" | "NO_MATCH"
    reason: str
    category_match: bool


def evaluate(
    ranked_candidates: list[MatchCandidate],
    post_category: str | None,
    guard_config: GuardConfig,
) -> list[GuardDecision]:
    """
    Returns the full decision trail: every candidate actually
    evaluated (in rank order), stopping at the first ACCEPT — matching
    §13.1's `break` behavior exactly, since candidates ranked below an
    accepted one are never evaluated by design, not omitted by
    accident.
    """
    if post_category is None:
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

    for candidate in ranked_candidates:
        category_match = candidate.category == post_category

        if not category_match:
            decisions.append(
                GuardDecision(
                    candidate=candidate,
                    decision="REJECTED",
                    reason=f"Category mismatch: expected {post_category}, detected {candidate.category}",
                    category_match=False,
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
                    f"Category match ({post_category}); similarity {candidate.similarity_score:.2f} >= "
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
                reason="No candidate cleared category, similarity, and confidence checks",
                category_match=False,
            )
        )

    return decisions