"""
Review service — human-in-the-loop actions on suggestions (§15).

Per §15.1: suggestion.decision (the guard's automated verdict) and
suggestion.review_status (the human reviewer's verdict) are
intentionally separate fields — a human can approve a guard REJECT,
or reject a guard ACCEPT, and that disagreement is itself visible,
auditable data, never silently overwritten.

Reviewer identity note: FR-5.2 asks for reviewer identity to be
persisted ("a static value is acceptable"), but the suggestions table
(§9.7) has no dedicated reviewer-identity column — only
reviewer_note, reviewed_at. Rather than silently add an unplanned
column, this project uses a fixed REVIEWER_IDENTITY constant folded
into reviewer_note (documented here and in BUILDLOG.md) — satisfying
FR-5.2's intent without an undocumented schema deviation.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Suggestion
from app.repositories.suggestion_repository import SuggestionRepository

REVIEWER_IDENTITY = "capstone_reviewer"  # static, per FR-5.2 — no auth system in scope


class SuggestionNotFoundError(Exception):
    def __init__(self, suggestion_id: uuid.UUID):
        self.suggestion_id = suggestion_id
        super().__init__(f"Suggestion {suggestion_id} not found")


def _prefix_note(note: str | None) -> str:
    """Folds the static reviewer identity into reviewer_note — see
    module docstring for why there's no dedicated identity column."""
    base = f"[{REVIEWER_IDENTITY}]"
    return f"{base} {note}" if note else base


class ReviewService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.repo = SuggestionRepository(db)

    async def approve(self, suggestion_id: uuid.UUID, note: str | None = None) -> Suggestion:
        """FR-5.2: approve a suggested pairing."""
        suggestion = await self.repo.get_by_id(suggestion_id)
        if suggestion is None:
            raise SuggestionNotFoundError(suggestion_id)

        suggestion.review_status = "APPROVED"
        suggestion.reviewer_note = _prefix_note(note)
        suggestion.reviewed_at = datetime.now(timezone.utc)
        await self.db.commit()
        return suggestion

    async def reject(self, suggestion_id: uuid.UUID, note: str | None = None) -> Suggestion:
        """FR-5.3: reject a suggested pairing, with an optional note."""
        suggestion = await self.repo.get_by_id(suggestion_id)
        if suggestion is None:
            raise SuggestionNotFoundError(suggestion_id)

        suggestion.review_status = "REJECTED"
        suggestion.reviewer_note = _prefix_note(note)
        suggestion.reviewed_at = datetime.now(timezone.utc)
        await self.db.commit()
        return suggestion

    async def create_override(
        self, post_id: uuid.UUID, image_id: uuid.UUID, note: str | None = None
    ) -> Suggestion:
        """
        FR-5.4 / §15.2: a reviewer manually assigns a different image
        than the guard suggested. Creates a NEW suggestions row —
        never edits an existing guard-originated row — with
        source=MANUAL_OVERRIDE, decision=ACCEPTED,
        review_status=APPROVED, bypassing the guard by design but
        always distinguishable from a guard acceptance via `source`.
        """
        row = await self.repo.create(
            post_id=post_id,
            image_id=image_id,
            similarity_score=None,
            category_match=False,  # not evaluated by the guard — bypassed by design
            confidence=None,
            decision="ACCEPTED",
            reason="Manually overridden by reviewer, bypassing the guard.",
            source="MANUAL_OVERRIDE",
            review_status="APPROVED",
        )
        if note:
            row.reviewer_note = _prefix_note(note)
            row.reviewed_at = datetime.now(timezone.utc)
        await self.db.commit()
        return row