"""
eval/run_eval.py

Runs the full matching + guard pipeline against every post in
eval/eval_set.json and computes top-1 precision (§17.2, FR-6.2):

    precision = (# posts where accepted top suggestion belongs to the
                  correct topic) / (total posts in eval set)

Ground-truth entries are keyed by post_title, not post_id — post IDs
are randomly generated at seed time and differ on every fresh clone
(NFR-6), so post_title (stable, human-readable) is the only safe join
key across environments.

Methodology note, recorded per §17.3's "record the reasoning"
expectation: ground truth uses a filename PREFIX (e.g.
"golden-retriever-dog-"), not one single hardcoded filename. This is
a deliberate correction after an earlier version of this eval set
used one arbitrarily-picked filename per post and scored 20% (2/10) —
not because the system was actually choosing bad images, but because
every "wrong" case was still a same-topic, correctly-tagged photo
from a corpus where multiple images per topic are all equally valid
matches (e.g. any of the 5 domestic-cat-*.jpg photos is a legitimate
match for a cat post). Forcing a single "correct" file when several
are equally correct measures whether the system guessed one arbitrary
label, not whether it found a relevant image — the wrong thing to
optimize for. Prefix matching tests what the guard is actually
designed to verify: topic/category correctness, without crowning one
photo among equals as uniquely right.

Usage:
    python eval/run_eval.py
"""

import asyncio
import json
import sys
from pathlib import Path

from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.models import Post  # noqa: E402
from app.db.session import AsyncSessionLocal, engine  # noqa: E402
from app.services.post_service import PostService  # noqa: E402

EVAL_SET_PATH = Path(__file__).resolve().parent / "eval_set.json"


async def run_one(db, post_title: str, correct_prefix: str) -> dict:
    """
    Runs the suggestions pipeline for one post and checks whether the
    top suggestion's filename starts with the expected topic prefix.
    Returns a result dict rather than raising — one post's failure
    (not found, pipeline error) should not stop the whole eval run,
    matching the same fault-isolation philosophy used in
    classification_job.py.
    """
    result = await db.execute(select(Post).where(Post.title == post_title))
    post = result.scalar_one_or_none()

    if post is None:
        return {
            "post_title": post_title,
            "correct": False,
            "reason": "Post not found in database — check seed_db.py ran and title matches exactly.",
        }

    service = PostService(db)
    try:
        response = await service.get_or_compute_suggestions(post.id)
    except Exception as exc:
        return {"post_title": post_title, "correct": False, "reason": f"Pipeline error: {exc}"}

    if response.decision != "ACCEPTED" or response.top_suggestion is None:
        return {
            "post_title": post_title,
            "correct": False,
            "reason": f"Expected ACCEPTED with prefix '{correct_prefix}', got {response.decision}"
            + (f" ({response.reason})" if response.reason else ""),
        }

    predicted = response.top_suggestion.filename
    is_correct = predicted.startswith(correct_prefix)
    return {
        "post_title": post_title,
        "correct": is_correct,
        "predicted": predicted,
        "expected_prefix": correct_prefix,
        "similarity_score": response.top_suggestion.similarity_score,
    }


async def main() -> None:
    if not EVAL_SET_PATH.exists():
        print(f"ERROR: {EVAL_SET_PATH} not found.")
        sys.exit(1)

    eval_set = json.loads(EVAL_SET_PATH.read_text(encoding="utf-8-sig"))
    print(f"Loaded {len(eval_set)} eval entries from {EVAL_SET_PATH}\n")

    results = []
    async with AsyncSessionLocal() as db:
        for entry in eval_set:
            print(f"Evaluating: {entry['post_title']}...")
            result = await run_one(db, entry["post_title"], entry["correct_filename_prefix"])
            results.append(result)
            status = "CORRECT" if result["correct"] else "WRONG"
            detail = result.get("predicted", result.get("reason", ""))
            print(f"  [{status}] {detail}")

    await engine.dispose()

    correct_count = sum(1 for r in results if r["correct"])
    total = len(results)
    precision = correct_count / total if total > 0 else 0.0

    print("\n" + "=" * 60)
    print("EVAL SUMMARY")
    print("=" * 60)
    for r in results:
        mark = "✓" if r["correct"] else "✗"
        print(f"  {mark} {r['post_title']}")
        if not r["correct"]:
            print(f"      predicted={r.get('predicted')} expected_prefix={r.get('expected_prefix')}")
            if r.get("reason"):
                print(f"      {r['reason']}")
    print("-" * 60)
    print(f"Top-1 Precision: {correct_count}/{total} = {precision:.2%}")
    print("=" * 60)
    print("\nCopy the precision line above verbatim into README.md (§17.2, FR-6.3).")


if __name__ == "__main__":
    asyncio.run(main())