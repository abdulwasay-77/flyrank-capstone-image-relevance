"""
Matching service — semantic similarity ranking (§12) plus post
category inference (§13.2), the two inputs the Mismatch Guard needs.

Category inference approach: embedding-similarity against the corpus's
own category vocabulary (§13.2's first documented option, chosen over
the "lightweight prompt" alternative because it reuses EmbeddingService
directly with zero new prompt design, and is fully deterministic given
stable embeddings — no extra schema, no extra AI-response validation
surface). This choice is recorded here per §13.2's "document whichever
approach is implemented."
"""

import uuid
from dataclasses import dataclass

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Image, ImageMetadata, ImageVector
from app.services.embedding_service import EmbeddingService

# In-process cache of category-label embeddings, keyed by the exact
# category string. Categories come from a small, fairly static corpus
# vocabulary (~10 distinct values at this project's scale per §12.2),
# so re-embedding the same handful of labels on every single
# suggestions request would be wasted API calls against an already
# tight free-tier quota (see BUILDLOG.md for the earlier rate-limit
# incident). Cache lives for the process lifetime; a restart clears it,
# which is fine — labels are cheap to re-embed once.
_category_embedding_cache: dict[str, list[float]] = {}


@dataclass
class MatchCandidate:
    image_id: uuid.UUID
    filename: str
    category: str
    confidence: float
    similarity_score: float


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Per §12.2: at ~50 images, computed in-process — a simple
    numpy dot-product over normalized vectors, no vector DB needed."""
    a_arr = np.array(a, dtype=np.float64)
    b_arr = np.array(b, dtype=np.float64)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if denom == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)


class MatchingService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.embedding_service = EmbeddingService(db)

    async def get_ranked_candidates(self, post_vector: list[float]) -> list[MatchCandidate]:
        """
        §12.2-12.3: cosine similarity between the post vector and
        every classified image's vector, sorted descending. Only
        images with BOTH a completed classification AND an embedding
        are candidates — an image that failed classification or
        hasn't been embedded yet simply can't be scored.
        """
        result = await self.db.execute(
            select(Image, ImageMetadata, ImageVector)
            .join(ImageMetadata, ImageMetadata.image_id == Image.id)
            .join(ImageVector, ImageVector.image_id == Image.id)
            .where(ImageMetadata.status == "completed")
        )
        rows = result.all()

        candidates = [
            MatchCandidate(
                image_id=image.id,
                filename=image.filename,
                category=metadata.category,
                confidence=metadata.confidence,
                similarity_score=cosine_similarity(post_vector, vector.embedding),
            )
            for image, metadata, vector in rows
        ]

        candidates.sort(key=lambda c: c.similarity_score, reverse=True)
        return candidates

    async def infer_post_category(self, post_vector: list[float], top_k: int = 2) -> list[str]:
        """
        §13.2: derives the post's expected category (or categories)
        via embedding similarity against the corpus's own category
        vocabulary. Returns the top `top_k` closest categories, not
        just the single best match.

        Why top-k, not top-1: a post can legitimately straddle two
        close categories (e.g. a wolf-in-the-forest post sits close
        to both "animal" and "nature" in embedding space). A strict
        top-1 winner is fragile — small embedding variance between
        runs (e.g. the in-process category-embedding cache in
        matching_service.py being rebuilt after a server restart) can
        flip which single category "wins", producing a different
        guard outcome for the exact same post text on different runs.
        That's a real reproducibility problem, not just a theoretical
        one — observed directly during testing (see BUILDLOG.md).
        Returning the top few candidates and letting the guard accept
        a category-match against ANY of them keeps the guard's
        rejection power (still refuses "food", "electronics", etc.
        for an animal post) while not being fragile to near-ties
        between genuinely related categories.

        Returns an empty list if the corpus has no classified images
        yet — the caller treats that as "cannot determine a
        category", which should conservatively fail every candidate's
        category check rather than guess.
        """
        result = await self.db.execute(
            select(ImageMetadata.category).where(ImageMetadata.status == "completed").distinct()
        )
        categories = [row[0] for row in result.all()]
        if not categories:
            return []

        scored: list[tuple[str, float]] = []
        for category in categories:
            if category not in _category_embedding_cache:
                _category_embedding_cache[category] = await self.embedding_service.embed_label_text(category)
            score = cosine_similarity(post_vector, _category_embedding_cache[category])
            scored.append((category, score))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [category for category, _ in scored[:top_k]]