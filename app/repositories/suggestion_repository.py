"""
Suggestion repository — pure data access for the suggestions table
(§9.7). Handles both guard-originated rows and manual-override rows
(source distinguishes them, per §15.2).
"""

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Image, Suggestion


class SuggestionRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def list_all(
        self, decision: str | None = None, review_status: str | None = None
    ) -> list[Suggestion]:
        """Supports the filters §10.3 calls for: GET /suggestions
        (filterable by decision, review_status)."""
        query = select(Suggestion).order_by(Suggestion.created_at.desc())
        if decision is not None:
            query = query.where(Suggestion.decision == decision)
        if review_status is not None:
            query = query.where(Suggestion.review_status == review_status)
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def get_by_id(self, suggestion_id: uuid.UUID) -> Suggestion | None:
        return await self.db.get(Suggestion, suggestion_id)

    async def get_latest_for_post(self, post_id: uuid.UUID) -> Suggestion | None:
        """
        Most recent suggestion for a post. Used by GET
        /posts/{id}/suggestions to decide whether to reuse an
        existing decision trail or compute a fresh one — re-running
        matching on every request would waste API calls (post
        embedding is stable once computed) and would otherwise create
        duplicate rows for the same decision run on every page view.
        """
        result = await self.db.execute(
            select(Suggestion)
            .where(Suggestion.post_id == post_id)
            .order_by(Suggestion.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def get_candidates_considered(self, post_id: uuid.UUID) -> list[tuple[Suggestion, str | None]]:
        """
        All suggestion rows for the same post_id, paired with each
        image's filename via left join (image_id is nullable for
        NO_MATCH rows). This is the "full decision trail" FR-5.5
        requires — every candidate the guard actually evaluated for
        this post, not just the winner.
        """
        result = await self.db.execute(
            select(Suggestion, Image.filename)
            .outerjoin(Image, Image.id == Suggestion.image_id)
            .where(Suggestion.post_id == post_id)
            .order_by(Suggestion.created_at)
        )
        return [(row[0], row[1]) for row in result.all()]

    async def create(
        self,
        *,
        post_id: uuid.UUID,
        image_id: uuid.UUID | None,
        similarity_score: float | None,
        category_match: bool,
        confidence: float | None,
        decision: str,
        reason: str,
        source: str,
        review_status: str = "PENDING",
    ) -> Suggestion:
        row = Suggestion(
            post_id=post_id,
            image_id=image_id,
            similarity_score=similarity_score,
            category_match=category_match,
            confidence=confidence,
            decision=decision,
            reason=reason,
            source=source,
            review_status=review_status,
        )
        self.db.add(row)
        await self.db.flush()
        return row