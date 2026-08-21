"""
Vision service — classifies a single image via the Gemini vision
model and validates its output against ImageTags.

Per §11.1-11.3:
  - Fixed, versioned system prompt (PROMPT_VERSION below) instructing
    JSON-only output, no prose, no markdown fences.
  - Defensive fence-stripping before parsing (models sometimes wrap
    JSON in ```json ... ``` despite instructions).
  - ImageTags.model_validate_json(raw_text) is the trust boundary —
    nothing downstream ever sees an unvalidated shape.
  - On ValidationError: retry up to MAX_RETRIES with a stricter
    re-prompt.
  - On repeated failure: image_metadata.status = "failed",
    raw_model_response preserved for debugging, no invented fallback
    tag.
  - On success with confidence < CONFIDENCE_FLAG_THRESHOLD:
    needs_review = True.

Every call (success or failure, first attempt or retry) goes through
CostTrackerService and writes one ai_call_log row (§16) — a retry is
a second real API call with its own real cost, not a free do-over.
"""

import asyncio
import json
import uuid

from google import genai
from google.genai import errors, types
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.repositories.image_repository import ImageRepository
from app.schemas.guard_config import GuardConfig
from app.schemas.image_tags import ImageTags
from app.services.cost_tracker_service import CostTrackerService

# gemini-3.6-flash (the flagship model) has an extremely tight free
# tier (observed: 5 requests/minute, 20/day) — far too low for a
# ~50-image batch. gemini-3.5-flash-lite is Google's current
# high-throughput/low-cost multimodal model, explicitly designed for
# exactly this kind of bulk classification workload, with a
# substantially higher free-tier quota. Confirmed via Google's own
# docs (ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite)
# as of this project's build date.
MODEL_NAME = "gemini-3.5-flash-lite"

# Separate from MAX_RETRIES (which governs schema-validation retries,
# §11.3) — rate-limit (429) errors are retried with backoff at the
# network-call level, since they're a transient infrastructure
# condition, not a "the model produced bad output" condition. Retrying
# a 429 does not consume one of the schema-validation retry attempts.
RATE_LIMIT_MAX_RETRIES = 3  # reduced from 5 — see embedding_service.py's
# identical constant for the incident that motivated this (BUILDLOG.md).
RATE_LIMIT_BASE_DELAY_SECONDS = 8.0

PROMPT_VERSION = "v1"
SYSTEM_PROMPT = """You are an image tagging system. Analyze the provided image and respond with ONLY a single JSON object — no prose, no explanation, no markdown code fences.

The JSON object must have exactly these fields:
{
  "subject": "the concrete thing depicted, e.g. red fox",
  "category": "a broader class it belongs to, e.g. animal",
  "attributes": ["3-6 short descriptive tags, e.g. orange fur, wild, forest"],
  "caption": "one natural-language sentence describing the image",
  "confidence": 0.0 to 1.0, your own certainty in this classification
}

Respond with the JSON object and nothing else."""

STRICT_RETRY_SUFFIX = """

IMPORTANT: Your previous response could not be parsed as valid JSON matching the required schema. Respond with ONLY the raw JSON object. Do not include markdown fences (no ```), no leading/trailing text, no explanation — just the JSON object starting with { and ending with }."""


def _strip_markdown_fences(text: str) -> str:
    """Defensive cleanup per §11.3 — models sometimes wrap JSON in
    ```json ... ``` fences despite explicit instructions not to."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        # Drop the opening fence line (```` ``` `` or ```` ```json ````)
        # and the closing ``` line, if present.
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines)
    return stripped.strip()


class VisionClassificationFailed(Exception):
    """Raised when classification exhausts MAX_RETRIES without producing
    a valid ImageTags — the caller (classification_job.py) catches this
    to mark the image as failed and move on to the next one (§11.4:
    one failure does not stop the batch)."""

    def __init__(self, image_id: uuid.UUID, raw_response: str | None):
        self.image_id = image_id
        self.raw_response = raw_response
        super().__init__(f"Vision classification failed for image {image_id} after retries")


class VisionService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.repo = ImageRepository(db)
        self.cost_tracker = CostTrackerService(db)
        self.settings = get_settings()
        self.guard_config = GuardConfig.from_settings()
        self._client = genai.Client(api_key=self.settings.GEMINI_API_KEY)

    async def _call_gemini(self, image_bytes: bytes, mime_type: str, prompt: str) -> tuple[str, int, int]:
        """
        One raw call to the Gemini vision model, with automatic
        retry-with-backoff specifically for 429 (rate limit) errors.
        Returns (response_text, input_tokens, output_tokens). Token
        counts come straight from the API's own usage_metadata — never
        estimated client-side — so cost logging is exact, not guessed.

        Uses self._client.aio.models.generate_content (the SDK's
        async client), NOT the sync self._client.models.generate_content.
        The sync version is a blocking network call; calling it
        directly inside an `async def` freezes the entire asyncio
        event loop for its full duration, which — with 5 "concurrent"
        images per §11.4 — would serialize everything anyway AND
        block the whole FastAPI server (including simple GET /jobs
        polling requests) for the entire batch's runtime.
        """
        last_exc: Exception | None = None
        for attempt in range(RATE_LIMIT_MAX_RETRIES):
            try:
                response = await self._client.aio.models.generate_content(
                    model=MODEL_NAME,
                    contents=[
                        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                        prompt,
                    ],
                )
                text = response.text or ""
                usage = response.usage_metadata
                input_tokens = usage.prompt_token_count if usage else 0
                output_tokens = usage.candidates_token_count if usage else 0
                return text, input_tokens, output_tokens
            except errors.APIError as exc:
                last_exc = exc
                if getattr(exc, "code", None) != 429:
                    raise  # not a rate limit — don't retry, surface immediately
                # Exponential backoff: 8s, 16s, 32s, 64s, 128s. Gemini's
                # 429 response usually includes a suggested retryDelay
                # (seen as low as 5-8s in testing); this backoff is
                # deliberately more conservative than the minimum
                # suggested delay to avoid hammering straight back into
                # the same per-minute window.
                delay = RATE_LIMIT_BASE_DELAY_SECONDS * (2**attempt)
                await asyncio.sleep(delay)

        raise last_exc  # exhausted rate-limit retries

    async def classify_image(self, image_id: uuid.UUID, image_bytes: bytes, mime_type: str = "image/jpeg") -> ImageTags:
        """
        Classifies one image, persists the result, and returns the
        validated ImageTags. Raises VisionClassificationFailed if all
        MAX_RETRIES attempts produce unparseable output — the caller
        is responsible for catching this and continuing the batch.
        """
        prompt = SYSTEM_PROMPT
        last_raw_response: str | None = None
        last_error: str | None = None

        for attempt in range(self.guard_config.max_retries + 1):
            async with self.cost_tracker.track_call(
                call_type="VISION_CLASSIFY", model_name=MODEL_NAME, reference_id=image_id
            ) as ctx:
                raw_text, input_tokens, output_tokens = await self._call_gemini(
                    image_bytes, mime_type, prompt
                )
                ctx["input_tokens"] = input_tokens
                ctx["output_tokens"] = output_tokens

            last_raw_response = raw_text
            cleaned = _strip_markdown_fences(raw_text)

            try:
                tags = ImageTags.model_validate_json(cleaned)
            except Exception as exc:
                last_error = str(exc)
                prompt = SYSTEM_PROMPT + STRICT_RETRY_SUFFIX
                continue

            # Success — persist per §11.3's needs_review rule.
            needs_review = tags.confidence < self.guard_config.confidence_flag_threshold
            await self.repo.upsert_metadata(
                image_id=image_id,
                subject=tags.subject,
                category=tags.category,
                attributes=tags.attributes,
                caption=tags.caption,
                confidence=tags.confidence,
                needs_review=needs_review,
                status="completed",
                raw_model_response={"raw_text": raw_text, "prompt_version": PROMPT_VERSION},
            )
            await self.db.commit()
            return tags

        # Exhausted all retries — persist failure, never invent a
        # fallback tag (§11.3).
        await self.repo.upsert_metadata(
            image_id=image_id,
            subject="",
            category="",
            attributes=[],
            caption="",
            confidence=0.0,
            needs_review=True,
            status="failed",
            raw_model_response={
                "raw_text": last_raw_response,
                "error": last_error,
                "prompt_version": PROMPT_VERSION,
            },
        )
        await self.db.commit()
        raise VisionClassificationFailed(image_id, last_raw_response)