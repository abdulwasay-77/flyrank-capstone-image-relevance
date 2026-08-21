"""
scripts/debug_suggestions.py

Runs PostService.get_or_compute_suggestions directly via asyncio.run(),
completely outside uvicorn/FastAPI. Prints a timestamped line before
and after every major step, so if it hangs, we see EXACTLY which line
it's stuck on — something the API's opaque 504 timeout can't tell us.

Usage:
    python scripts/debug_suggestions.py <post_id>
"""

import asyncio
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.session import AsyncSessionLocal, engine  # noqa: E402
from app.repositories.post_repository import PostRepository  # noqa: E402
from app.repositories.suggestion_repository import SuggestionRepository  # noqa: E402
from app.services.embedding_service import EmbeddingService  # noqa: E402
from app.services.guard_service import evaluate  # noqa: E402
from app.services.matching_service import MatchingService  # noqa: E402
from app.schemas.guard_config import GuardConfig  # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


async def main(post_id_str: str) -> None:
    post_id = uuid.UUID(post_id_str)

    async with AsyncSessionLocal() as db:
        log("Session opened")

        repo = PostRepository(db)
        post = await repo.get_by_id(post_id)
        log(f"Post fetched: {post.title if post else 'NOT FOUND'}")
        if post is None:
            return

        suggestion_repo = SuggestionRepository(db)
        existing = await suggestion_repo.get_latest_for_post(post_id)
        log(f"Existing suggestion check: {'found' if existing else 'none'}")

        embedding_service = EmbeddingService(db)
        existing_vector = await repo.get_vector(post_id)
        log(f"Existing post_vector check: {'found' if existing_vector else 'none'}")

        if existing_vector is not None:
            post_vector = existing_vector.embedding
            log("Reusing existing post vector, no embed call needed")
        else:
            log("Calling embed_post_text (real Gemini call)...")
            post_vector = await embedding_service.embed_post_text(post_id, post.title, post.body)
            log("embed_post_text returned")
            await repo.upsert_vector(post_id, post_vector, "gemini-embedding-001")
            await db.commit()
            log("post_vector committed")

        matching_service = MatchingService(db)
        log("Calling get_ranked_candidates...")
        ranked = await matching_service.get_ranked_candidates(post_vector)
        log(f"get_ranked_candidates returned {len(ranked)} candidates")

        log("Calling infer_post_category...")
        category = await matching_service.infer_post_category(post_vector)
        log(f"infer_post_category returned: {category}")

        guard_config = GuardConfig.from_settings()
        log("Running guard.evaluate...")
        decisions = evaluate(ranked, category, guard_config)
        log(f"guard.evaluate returned {len(decisions)} decisions")
        for d in decisions:
            print(f"    {d.decision} - {d.reason}")

    await engine.dispose()
    log("Done, engine disposed")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/debug_suggestions.py <post_id>")
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))
    