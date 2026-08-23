"""
Posts router — HTTP layer for posts and the suggestions pipeline
trigger (§10.2, §10.6, §10.7).
"""

import asyncio
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.api_models import OverrideRequest, PostOut, PostSuggestionsOut, SuggestionOut
from app.services.post_service import PostNotFoundError, PostService
from app.services.review_service import ReviewService

router = APIRouter(prefix="/posts", tags=["posts"])

# Hard ceiling on the whole suggestions pipeline (embed + rank + infer
# category + guard). Prevents a worst-case stack of rate-limit retries
# (see embedding_service.py / vision_service.py's RATE_LIMIT_MAX_RETRIES
# comments) from hanging a request indefinitely — a real incident
# during testing, documented in BUILDLOG.md. Well above the realistic
# happy-path time (a few seconds) but well below "leave it stuck for
# 40 minutes".
SUGGESTIONS_TIMEOUT_SECONDS = 240  # raised to comfortably exceed the
# measured ~30s single-query latency (see app/db/session.py's
# command_timeout comment) with headroom for a real Gemini embedding
# call plus multiple queries in the same request.


@router.get("", response_model=list[PostOut])
async def list_posts(db: AsyncSession = Depends(get_db)) -> list[PostOut]:
    service = PostService(db)
    return await service.list_posts()


@router.get("/{post_id}", response_model=PostOut)
async def get_post(post_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> PostOut:
    service = PostService(db)
    try:
        return await service.get_post(post_id)
    except PostNotFoundError:
        raise HTTPException(status_code=404, detail=f"Post {post_id} not found")


@router.get("/{post_id}/suggestions", response_model=PostSuggestionsOut)
async def get_post_suggestions(post_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> PostSuggestionsOut:
    """
    Runs (or reuses) the full pipeline for this post: embed -> rank ->
    infer category -> guard -> persist decision trail -> return
    §10.6/§10.7 shaped response. This is a live network call on first
    request (post embedding via Gemini) — subsequent calls reuse the
    persisted result.

    Wrapped in a hard timeout: if the whole pipeline (including any
    429 retry backoff) exceeds SUGGESTIONS_TIMEOUT_SECONDS, this
    returns a clean 504 instead of hanging the request — and,
    critically, the `finally` in get_db() still runs on timeout,
    properly closing the database session rather than leaving a
    connection in a stuck/corrupted state for a future request to
    collide with.
    """
    service = PostService(db)
    try:
        return await asyncio.wait_for(
            service.get_or_compute_suggestions(post_id), timeout=SUGGESTIONS_TIMEOUT_SECONDS
        )
    except PostNotFoundError:
        raise HTTPException(status_code=404, detail=f"Post {post_id} not found")
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=(
                "Suggestion computation timed out, likely due to AI API rate limiting. "
                "Please wait a minute and try again."
            ),
        )


@router.post("/{post_id}/override", response_model=SuggestionOut)
async def override_suggestion(
    post_id: uuid.UUID, body: OverrideRequest, db: AsyncSession = Depends(get_db)
) -> SuggestionOut:
    """
    FR-5.4 / §15.2: a reviewer manually assigns a different image than
    the guard suggested. Does not touch or delete any existing
    guard-originated suggestion row — creates a new one with
    source=MANUAL_OVERRIDE, always distinguishable from an automated
    acceptance.
    """
    post_service = PostService(db)
    post = await post_service.repo.get_by_id(post_id)
    if post is None:
        raise HTTPException(status_code=404, detail=f"Post {post_id} not found")

    review_service = ReviewService(db)
    suggestion = await review_service.create_override(post_id, body.image_id, body.note)
    return SuggestionOut.model_validate(suggestion)