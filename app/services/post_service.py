"""
Post service — orchestrates the full suggestion pipeline for a post:
ensure embedding exists -> rank candidates -> infer category -> run
the guard -> persist every evaluated decision -> build the API
response shape (§10.6 / §10.7).

Per FR-3.3: every guard decision (accept or reject) is persisted, not
just the winning one — this is what makes GET /suggestions/{id}'s
"full decision trail" (FR-5.5) queryable directly from stored data
rather than needing to be recomputed on every read. A post is only
ever run through the guard once (results are then reused on repeat
requests) — see _build_response_from_existing_rows.
"""

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Suggestion
from app.repositories.post_repository import PostRepository
from app.repositories.suggestion_repository import SuggestionRepository
from app.schemas.api_models import PostOut, PostSuggestionsOut, RejectedCandidateOut, TopSuggestionOut
from app.schemas.guard_config import GuardConfig
from app.services.embedding_service import MODEL_NAME as EMBED_MODEL_NAME
from app.services.embedding_service import EmbeddingService
from app.services.guard_service import GuardDecision, evaluate
from app.services.matching_service import MatchCandidate, MatchingService


class PostNotFoundError(Exception):
    def __init__(self, post_id: uuid.UUID):
        self.post_id = post_id
        super().__init__(f"Post {post_id} not found")


class PostService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.repo = PostRepository(db)
        self.suggestion_repo = SuggestionRepository(db)
        self.embedding_service = EmbeddingService(db)
        self.matching_service = MatchingService(db)
        self.guard_config = GuardConfig.from_settings()

    async def list_posts(self) -> list[PostOut]:
        posts = await self.repo.list_all()
        return [PostOut.model_validate(p) for p in posts]

    async def get_post(self, post_id: uuid.UUID) -> PostOut:
        post = await self.repo.get_by_id(post_id)
        if post is None:
            raise PostNotFoundError(post_id)
        return PostOut.model_validate(post)

    async def _ensure_post_embedding(self, post_id: uuid.UUID, title: str, body: str) -> list[float]:
        """Embeds on first request, reuses on subsequent ones — a
        post's title/body don't change after seeding, so re-embedding
        on every suggestions request would be wasted API calls."""
        existing = await self.repo.get_vector(post_id)
        if existing is not None:
            return existing.embedding

        embedding = await self.embedding_service.embed_post_text(post_id, title, body)
        await self.repo.upsert_vector(post_id, embedding, EMBED_MODEL_NAME)
        await self.db.commit()
        return embedding

    async def get_or_compute_suggestions(self, post_id: uuid.UUID) -> PostSuggestionsOut:
        post = await self.repo.get_by_id(post_id)
        if post is None:
            raise PostNotFoundError(post_id)

        existing = await self.suggestion_repo.get_latest_for_post(post_id)
        if existing is not None:
            rows = await self.suggestion_repo.list_for_post_with_filenames(post_id)
            return self._build_response_from_existing_rows(post_id, post.title, rows)

        post_vector = await self._ensure_post_embedding(post_id, post.title, post.body)
        ranked_candidates = await self.matching_service.get_ranked_candidates(post_vector)
        post_category = await self.matching_service.infer_post_category(post_vector)

        decisions = evaluate(ranked_candidates, post_category, self.guard_config)

        for d in decisions:
            await self.suggestion_repo.create(
                post_id=post_id,
                image_id=d.candidate.image_id if d.candidate else None,
                similarity_score=d.candidate.similarity_score if d.candidate else None,
                category_match=d.category_match,
                confidence=d.candidate.confidence if d.candidate else None,
                decision=d.decision,
                reason=d.reason,
                source="GUARD",
            )
        await self.db.commit()

        return self._build_response_from_decisions(post_id, post.title, decisions)

    def _build_response_from_decisions(
        self, post_id: uuid.UUID, post_title: str, decisions: list[GuardDecision]
    ) -> PostSuggestionsOut:
        """Builds the response directly from freshly-computed
        GuardDecision objects (which already carry the full
        MatchCandidate, including filename) — used right after a
        fresh guard run, no DB round-trip needed."""
        accepted = next((d for d in decisions if d.decision == "ACCEPTED"), None)
        rejected = [d for d in decisions if d.decision == "REJECTED"]
        no_match = next((d for d in decisions if d.decision == "NO_MATCH"), None)

        rejected_candidates = [
            RejectedCandidateOut(
                image_id=d.candidate.image_id,
                filename=d.candidate.filename,
                similarity_score=d.candidate.similarity_score,
                category_match=d.category_match,
                reason=d.reason,
            )
            for d in rejected
        ]

        if accepted is not None:
            c = accepted.candidate
            return PostSuggestionsOut(
                post_id=post_id,
                post_title=post_title,
                decision="ACCEPTED",
                top_suggestion=TopSuggestionOut(
                    image_id=c.image_id,
                    filename=c.filename,
                    similarity_score=c.similarity_score,
                    confidence=c.confidence,
                    category_match=True,
                    reason=accepted.reason,
                ),
                rejected_candidates=rejected_candidates,
                reason=None,
            )

        return PostSuggestionsOut(
            post_id=post_id,
            post_title=post_title,
            decision="NO_MATCH",
            top_suggestion=None,
            rejected_candidates=rejected_candidates,
            reason=no_match.reason,
        )

    def _build_response_from_existing_rows(
        self, post_id: uuid.UUID, post_title: str, rows: list[tuple[Suggestion, str | None]]
    ) -> PostSuggestionsOut:
        """Builds the response from already-persisted Suggestion rows
        (paired with filenames via the repository's join) — used on
        repeat requests for a post that's already been through the
        guard once."""
        accepted = next(((r, fn) for r, fn in rows if r.decision == "ACCEPTED"), None)
        rejected = [(r, fn) for r, fn in rows if r.decision == "REJECTED"]
        no_match = next((r for r, _ in rows if r.decision == "NO_MATCH"), None)

        rejected_candidates = [
            RejectedCandidateOut(
                image_id=r.image_id,
                filename=fn or "",
                similarity_score=r.similarity_score,
                category_match=r.category_match,
                reason=r.reason,
            )
            for r, fn in rejected
        ]

        if accepted is not None:
            r, fn = accepted
            return PostSuggestionsOut(
                post_id=post_id,
                post_title=post_title,
                decision="ACCEPTED",
                top_suggestion=TopSuggestionOut(
                    image_id=r.image_id,
                    filename=fn or "",
                    similarity_score=r.similarity_score,
                    confidence=r.confidence,
                    category_match=r.category_match,
                    reason=r.reason,
                ),
                rejected_candidates=rejected_candidates,
                reason=None,
            )

        return PostSuggestionsOut(
            post_id=post_id,
            post_title=post_title,
            decision="NO_MATCH",
            top_suggestion=None,
            rejected_candidates=rejected_candidates,
            reason=no_match.reason if no_match else "No decision recorded",
        )