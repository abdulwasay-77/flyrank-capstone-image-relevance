"""
Embedding service — generates semantic embeddings for images and
posts, per §12.1.

  - Image side: embeds the caption (not the raw tags array) — captions
    carry the richest natural-language meaning.
  - Post side: embeds title + body concatenated (this project's
    implementation choice, documented here per §12.1's requirement to
    document whichever concatenation strategy is used — title and
    body are joined with a newline, no extra weighting applied to
    either).
  - Both sides use the same model and the same task_type
    (SEMANTIC_SIMILARITY) so the resulting vector spaces are directly
    comparable via cosine similarity (§12.2).

Model note: the spec (§8) targeted text-embedding-004. That model was
shut down by Google on January 14, 2026 (confirmed via Google's own
API changelog). gemini-embedding-001 is the current stable
text-embedding replacement and explicitly supports task_type,
including SEMANTIC_SIMILARITY — verified against the actually
installed google-genai SDK before writing this file, not assumed.
"""

import asyncio
import uuid

from google import genai
from google.genai import errors, types
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.repositories.image_repository import ImageRepository
from app.services.cost_tracker_service import CostTrackerService

MODEL_NAME = "gemini-embedding-001"
TASK_TYPE = "SEMANTIC_SIMILARITY"

RATE_LIMIT_MAX_RETRIES = 3  # reduced from 5 after a real incident: 5 retries
# at this backoff schedule (8s/16s/32s = worst case ~56s per call) times up
# to ~10 sequential category-label embeds in infer_post_category could
# previously hang a single HTTP request for 40+ minutes with no ceiling.
# See BUILDLOG.md.
RATE_LIMIT_BASE_DELAY_SECONDS = 8.0


class EmbeddingService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.image_repo = ImageRepository(db)
        self.cost_tracker = CostTrackerService(db)
        self.settings = get_settings()
        self._client = genai.Client(api_key=self.settings.GEMINI_API_KEY)

    async def _embed_text(self, text: str) -> tuple[list[float], int]:
        """
        One raw embedding call, with the same 429 retry-with-backoff
        pattern as VisionService._call_gemini (see that method's
        docstring for the full rationale — same free-tier rate-limit
        behavior applies here).

        Returns (embedding_vector, token_count). Embedding responses
        don't expose a separate input/output split the way
        generate_content does — embeddings only ever "consume" tokens,
        there's no generated-output token cost — so we log the single
        token count as input_tokens and 0 output_tokens when writing
        to ai_call_log.
        """
        last_exc: Exception | None = None
        for attempt in range(RATE_LIMIT_MAX_RETRIES):
            try:
                response = await self._client.aio.models.embed_content(
                    model=MODEL_NAME,
                    contents=text,
                    config=types.EmbedContentConfig(task_type=TASK_TYPE),
                )
                embedding = response.embeddings[0].values
                token_count = 0
                if response.metadata and response.metadata.billable_character_count:
                    # Embedding responses report billable characters,
                    # not tokens directly; used only for the cost
                    # estimate below, and since text-embedding pricing
                    # on the free tier is $0 anyway (see
                    # cost_tracker_service.py's PRICING table), this
                    # is a best-effort figure, not load-bearing.
                    token_count = response.metadata.billable_character_count
                return list(embedding), token_count
            except errors.APIError as exc:
                last_exc = exc
                if getattr(exc, "code", None) != 429:
                    raise
                delay = RATE_LIMIT_BASE_DELAY_SECONDS * (2**attempt)
                await asyncio.sleep(delay)

        raise last_exc

    async def embed_image(self, image_id: uuid.UUID, caption: str) -> list[float]:
        """
        Embeds one image's caption and persists it to image_vectors
        (§9.4, unique on image_id — upsert_vector handles both first
        embedding and re-embedding).
        """
        async with self.cost_tracker.track_call(
            call_type="EMBEDDING", model_name=MODEL_NAME, reference_id=image_id
        ) as ctx:
            embedding, token_count = await self._embed_text(caption)
            ctx["input_tokens"] = token_count

        await self.image_repo.upsert_vector(image_id, embedding, MODEL_NAME)
        await self.db.commit()
        return embedding

    async def embed_post_text(self, post_id: uuid.UUID, title: str, body: str) -> list[float]:
        """
        Embeds a post's title + body (concatenated with a newline —
        this project's documented choice per §12.1) and returns the
        vector. Does NOT persist — the caller (PostService, built
        alongside PostRepository) is responsible for the actual
        post_vectors upsert, since PostRepository doesn't exist yet
        at the time this file was written (same situation as
        seed_db.py — noted in BUILDLOG.md).
        """
        combined_text = f"{title}\n{body}"
        async with self.cost_tracker.track_call(
            call_type="EMBEDDING", model_name=MODEL_NAME, reference_id=post_id
        ) as ctx:
            embedding, token_count = await self._embed_text(combined_text)
            ctx["input_tokens"] = token_count

        return embedding

    async def embed_label_text(self, text: str) -> list[float]:
        """
        Embeds an arbitrary short text (currently: a category label,
        used by MatchingService.infer_post_category for §13.2's
        category-inference approach) with the same cost-tracking
        discipline as every other embedding call — §16 requires
        every AI API call to produce exactly one ai_call_log row,
        with no carve-out for "small" calls. reference_id is left
        None since a category label isn't tied to one specific
        image or post row.
        """
        async with self.cost_tracker.track_call(
            call_type="EMBEDDING", model_name=MODEL_NAME, reference_id=None
        ) as ctx:
            embedding, token_count = await self._embed_text(text)
            ctx["input_tokens"] = token_count

        return embedding