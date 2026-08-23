"""
Suggestions router — the Review API (§10.3, FR-5.1/5.2/5.3/5.5).
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.repositories.suggestion_repository import SuggestionRepository
from app.schemas.api_models import (
    ReviewActionRequest,
    SuggestionDetailOut,
    SuggestionOut,
    RejectedCandidateOut,
)
from app.services.review_service import ReviewService, SuggestionNotFoundError

router = APIRouter(prefix="/suggestions", tags=["suggestions"])


@router.get("", response_model=list[SuggestionOut])
async def list_suggestions(
    decision: str | None = Query(default=None, description="Filter by ACCEPTED | REJECTED | NO_MATCH"),
    review_status: str | None = Query(default=None, description="Filter by PENDING | APPROVED | REJECTED"),
    db: AsyncSession = Depends(get_db),
) -> list[SuggestionOut]:
    """FR-5.1: list all suggested (post, image) pairings with their
    guard decision and reason, optionally filtered."""
    repo = SuggestionRepository(db)
    rows = await repo.list_all(decision=decision, review_status=review_status)
    return [SuggestionOut.model_validate(r) for r in rows]


@router.get("/{suggestion_id}", response_model=SuggestionDetailOut)
async def get_suggestion(suggestion_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> SuggestionDetailOut:
    """FR-5.5: the full decision trail for this suggestion's post —
    every candidate the guard evaluated, not just this one row."""
    repo = SuggestionRepository(db)
    suggestion = await repo.get_by_id(suggestion_id)
    if suggestion is None:
        raise HTTPException(status_code=404, detail=f"Suggestion {suggestion_id} not found")

    candidate_rows = await repo.get_candidates_considered(suggestion.post_id)
    candidates_considered = [
        RejectedCandidateOut(
            image_id=r.image_id,
            filename=fn or "",
            similarity_score=r.similarity_score,
            category_match=r.category_match,
            reason=r.reason,
        )
        for r, fn in candidate_rows
    ]

    base = SuggestionOut.model_validate(suggestion)
    return SuggestionDetailOut(**base.model_dump(), candidates_considered=candidates_considered)


@router.post("/{suggestion_id}/approve", response_model=SuggestionOut)
async def approve_suggestion(
    suggestion_id: uuid.UUID, body: ReviewActionRequest | None = None, db: AsyncSession = Depends(get_db)
) -> SuggestionOut:
    """FR-5.2: approve a suggested pairing."""
    service = ReviewService(db)
    note = body.note if body else None
    try:
        suggestion = await service.approve(suggestion_id, note)
    except SuggestionNotFoundError:
        raise HTTPException(status_code=404, detail=f"Suggestion {suggestion_id} not found")
    return SuggestionOut.model_validate(suggestion)


@router.post("/{suggestion_id}/reject", response_model=SuggestionOut)
async def reject_suggestion(
    suggestion_id: uuid.UUID, body: ReviewActionRequest | None = None, db: AsyncSession = Depends(get_db)
) -> SuggestionOut:
    """FR-5.3: reject a suggested pairing, with an optional note."""
    service = ReviewService(db)
    note = body.note if body else None
    try:
        suggestion = await service.reject(suggestion_id, note)
    except SuggestionNotFoundError:
        raise HTTPException(status_code=404, detail=f"Suggestion {suggestion_id} not found")
    return SuggestionOut.model_validate(suggestion)