"""
scripts/verify_subject_fix.py

Automated verification of the subject-mismatch guard fix, using real
data — no Swagger clicking required. Does three things:

1. Clears stale suggestions for the fox and wolf posts (so results are
   freshly computed, not reused from before the fix).
2. Runs the real pipeline for both posts via PostService, prints and
   checks the results.
3. THE CRITICAL TEST: takes the fox post's real ranked candidates,
   removes every "red fox" subject entry (simulating "no fox photos
   available"), and runs the REAL guard against what's left — proving
   gray-wolf-01.jpg (or whichever image now ranks first) is correctly
   REJECTED using real embeddings and real inferred subjects, not a
   fabricated unit-test scenario.

Usage:
    python scripts/verify_subject_fix.py
"""

import asyncio
import sys
from pathlib import Path

from sqlalchemy import delete, select

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.models import Post, Suggestion  # noqa: E402
from app.db.session import AsyncSessionLocal, engine  # noqa: E402
from app.repositories.post_repository import PostRepository  # noqa: E402
from app.services.guard_service import evaluate  # noqa: E402
from app.services.matching_service import MatchingService  # noqa: E402
from app.services.post_service import PostService  # noqa: E402

FOX_POST_TITLE = "Tracking the Elusive Red Fox in Winter"
WOLF_POST_TITLE = "The Return of the Gray Wolf to Northern Forests"


def result(label: str, passed: bool) -> None:
    mark = "PASS" if passed else "FAIL"
    print(f"  [{mark}] {label}")


async def clear_stale_suggestions(db, post_title: str) -> None:
    result_row = await db.execute(select(Post).where(Post.title == post_title))
    post = result_row.scalar_one_or_none()
    if post is None:
        print(f"  WARNING: post '{post_title}' not found, skipping clear.")
        return
    await db.execute(delete(Suggestion).where(Suggestion.post_id == post.id))
    await db.commit()


async def run_post_check(db, post_title: str, expect_subject_in_reason: str | None = None) -> None:
    print(f"\n--- {post_title} ---")
    result_row = await db.execute(select(Post).where(Post.title == post_title))
    post = result_row.scalar_one_or_none()
    if post is None:
        print(f"  WARNING: post not found.")
        return

    service = PostService(db)
    response = await service.get_or_compute_suggestions(post.id)

    print(f"  decision: {response.decision}")
    if response.top_suggestion:
        print(f"  top_suggestion: {response.top_suggestion.filename}")
        print(f"  reason: {response.top_suggestion.reason}")
        result("Decision is ACCEPTED", response.decision == "ACCEPTED")
        if expect_subject_in_reason:
            result(
                "Reason mentions subject match",
                "subject match" in response.top_suggestion.reason.lower(),
            )
    else:
        print(f"  reason: {response.reason}")
        result("Decision is ACCEPTED", False)


async def run_forced_wolf_test(db) -> None:
    """
    The critical live-data test: get the fox post's REAL ranked
    candidates and REAL inferred category/subjects, remove every
    "red fox" entry to simulate fox being unavailable, and run the
    REAL guard against what's left. This proves the fix works with
    actual embeddings and actual data, not a hand-fabricated scenario.
    """
    print("\n--- CRITICAL TEST: forcing a non-fox candidate as top pick for the fox post ---")
    result_row = await db.execute(select(Post).where(Post.title == FOX_POST_TITLE))
    post = result_row.scalar_one_or_none()
    if post is None:
        print("  WARNING: fox post not found.")
        return

    post_repo = PostRepository(db)
    vector_row = await post_repo.get_vector(post.id)
    if vector_row is None:
        print("  WARNING: fox post has no embedding yet — run it once via the normal flow first.")
        return

    matching_service = MatchingService(db)
    all_candidates = await matching_service.get_ranked_candidates(vector_row.embedding)
    no_fox_candidates = [c for c in all_candidates if c.subject != "red fox"]

    if not no_fox_candidates:
        print("  WARNING: no non-fox candidates found to test against.")
        return

    print(
        f"  Top candidate after removing all 'red fox' entries: "
        f"{no_fox_candidates[0].filename} (subject={no_fox_candidates[0].subject}, "
        f"similarity={no_fox_candidates[0].similarity_score:.4f})"
    )

    from app.schemas.guard_config import GuardConfig

    guard_config = GuardConfig.from_settings()
    post_categories = await matching_service.infer_post_category(vector_row.embedding)
    post_subjects = await matching_service.infer_post_subjects(vector_row.embedding)
    print(f"  Inferred categories: {post_categories}")
    print(f"  Inferred subjects: {post_subjects}")

    decisions = evaluate(no_fox_candidates, post_categories, guard_config, post_subjects)

    top_decision = decisions[0]
    print(f"  Result: {top_decision.decision} - {top_decision.reason}")

    is_non_fox = top_decision.candidate and top_decision.candidate.subject != "red fox"
    result(
        "A non-fox top candidate was correctly REJECTED (not wrongly ACCEPTED)",
        top_decision.decision == "REJECTED" and bool(is_non_fox),
    )


async def main() -> None:
    async with AsyncSessionLocal() as db:
        print("Clearing stale suggestions for test posts...")
        await clear_stale_suggestions(db, FOX_POST_TITLE)
        await clear_stale_suggestions(db, WOLF_POST_TITLE)

        await run_post_check(db, FOX_POST_TITLE, expect_subject_in_reason="red fox")
        await run_post_check(db, WOLF_POST_TITLE, expect_subject_in_reason="wolf")
        await run_forced_wolf_test(db)

    await engine.dispose()
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())