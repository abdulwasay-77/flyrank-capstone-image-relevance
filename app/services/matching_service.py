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

# In-process cache of category/subject-label embeddings, keyed by the
# exact label string. Both vocabularies are small and fairly static at
# this project's scale (§12.2), so re-embedding the same handful of
# labels on every single suggestions request would be wasted API calls
# against an already tight free-tier quota (see BUILDLOG.md for the
# earlier rate-limit incident). Cache lives for the process lifetime;
# a restart clears it, which is fine — labels are cheap to re-embed once.
# Shared between category and subject labels since they never collide
# in practice (e.g. "animal" vs "red fox").
_label_embedding_cache: dict[str, list[float]] = {}


@dataclass
class MatchCandidate:
    image_id: uuid.UUID
    filename: str
    category: str
    subject: str
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
                subject=metadata.subject,
                confidence=metadata.confidence,
                similarity_score=cosine_similarity(post_vector, vector.embedding),
            )
            for image, metadata, vector in rows
        ]

        candidates.sort(key=lambda c: c.similarity_score, reverse=True)
        return candidates

    async def _infer_top_k_labels(self, post_vector: list[float], label_column, top_k: int) -> list[str]:
        """
        Shared implementation for infer_post_category and
        infer_post_subjects — both do the same thing (embedding
        similarity against a distinct label vocabulary from
        image_metadata), differing only in which column supplies the
        vocabulary. Factored out to avoid duplicating the caching and
        scoring logic.
        """
        result = await self.db.execute(
            select(label_column).where(ImageMetadata.status == "completed").distinct()
        )
        labels = [row[0] for row in result.all()]
        if not labels:
            return []

        scored: list[tuple[str, float]] = []
        for label in labels:
            if label not in _label_embedding_cache:
                _label_embedding_cache[label] = await self.embedding_service.embed_label_text(label)
            score = cosine_similarity(post_vector, _label_embedding_cache[label])
            scored.append((label, score))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [label for label, _ in scored[:top_k]]

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
        runs (e.g. the in-process label-embedding cache in
        matching_service.py being rebuilt after a server restart) can
        flip which single category "wins", producing a different
        guard outcome for the exact same post text on different runs.
        That's a real reproducibility problem, not just a theoretical
        one — observed directly during testing (see BUILDLOG.md).

        Returns an empty list if the corpus has no classified images
        yet — the caller treats that as "cannot determine a
        category", which should conservatively fail every candidate's
        category check rather than guess.

        NOTE: category alone is too coarse to distinguish closely
        related subjects within the same category (e.g. "red fox" vs
        "gray wolf" both tag category="animal") — see
        infer_post_subjects below, added specifically to close that
        gap after it was caught by direct testing.
        """
        return await self._infer_top_k_labels(post_vector, ImageMetadata.category, top_k)

    async def infer_post_subjects(self, post_vector: list[float], top_k: int = 1) -> list[str]:
        """
        Derives the post's expected SUBJECT(s) — the fine-grained
        field (e.g. "red fox", "gray wolf"), not the coarse category
        (e.g. "animal") — via the same embedding-similarity technique
        as infer_post_category.

        Why this exists: category alone cannot catch a wolf photo
        being suggested for a fox post, since the vision model tags
        both with category="animal". This was proven directly, not
        theorized — scripts/debug_full_ranking.py showed
        gray-wolf-01.jpg outranking two real red-fox-*.jpg photos by
        raw similarity, while clearing both the category-match and
        similarity-threshold checks. Subject-level matching (fox
        specifically, not just "some animal") is what the original
        capstone brief's own example reason string implies
        ("expected fox, detected wolf") and what GuardService now
        checks as an additional, stricter gate alongside category.

        top_k=1, NOT a wider value like category's top_k=2: subjects
        are already fine-grained, so widening the accepted set risks
        pulling in a genuinely different, wrong subject as
        "acceptable" — caught directly during verification
        (scripts/verify_subject_fix.py): with top_k=3, "coyote" ended
        up in the accepted-subjects list for a fox post purely because
        it's a nearby canid in embedding space, which would have
        silently defeated the entire point of this check for a
        coyote-tagged candidate. top_k=1 keeps this check strict: the
        subject must be the single closest match, not merely "in the
        neighborhood". See BUILDLOG.md.
        """
        return await self._infer_top_k_labels(post_vector, ImageMetadata.subject, top_k)