"""
scripts/debug_full_ranking.py

Shows the COMPLETE ranked candidate list for a post — every image,
its category, confidence, and similarity score, sorted by rank —
not just the guard's single top pick. The guard's break-on-first-
accept behavior means normal API responses never show where
lower-ranked candidates (e.g. a wolf photo for a fox post) actually
land; this script bypasses the guard entirely to answer that directly.

Usage:
    python scripts/debug_full_ranking.py <post_id>
"""

import asyncio
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.session import AsyncSessionLocal, engine  # noqa: E402
from app.repositories.post_repository import PostRepository  # noqa: E402
from app.services.matching_service import MatchingService  # noqa: E402


async def main(post_id_str: str) -> None:
    post_id = uuid.UUID(post_id_str)

    async with AsyncSessionLocal() as db:
        post_repo = PostRepository(db)
        post = await post_repo.get_by_id(post_id)
        if post is None:
            print(f"Post {post_id} not found.")
            return

        vector_row = await post_repo.get_vector(post_id)
        if vector_row is None:
            print("Post has no embedding yet — run GET /posts/{id}/suggestions once first.")
            return

        matching_service = MatchingService(db)
        candidates = await matching_service.get_ranked_candidates(vector_row.embedding)

        print(f"Post: {post.title}\n")
        print(f"{'Rank':<5} {'Filename':<32} {'Category':<15} {'Confidence':<11} {'Similarity':<10}")
        print("-" * 78)
        for i, c in enumerate(candidates, start=1):
            print(f"{i:<5} {c.filename:<32} {c.category:<15} {c.confidence:<11.2f} {c.similarity_score:<10.4f}")

    await engine.dispose()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/debug_full_ranking.py <post_id>")
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))