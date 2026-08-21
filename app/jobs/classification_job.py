"""
Classification batch job.

Runs as a FastAPI BackgroundTask (§14.1 MVP choice, arq+Redis is the
documented stretch goal) — started by POST /images/classify-batch,
tracked via the batch_jobs row it updates as it progresses.

Per §11.4: bounded concurrency (5 concurrent Gemini calls) to respect
free-tier rate limits, and each image is processed independently —
one image's failure must not stop or crash the rest of the batch.

Each concurrent worker opens its OWN database session. AsyncSession
is not safe to share across concurrently-running coroutines (SQLAlchemy
raises on interleaved use of one session from multiple tasks), so
sharing a single session across the whole batch would silently
serialize everything or crash outright — this is why VisionService
takes a session per call rather than the job holding one long-lived
session for all 50 images.
"""

import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select

from app.db.models import BatchJob, Image
from app.db.session import AsyncSessionLocal
from app.repositories.image_repository import ImageRepository
from app.services.embedding_service import EmbeddingService
from app.services.vision_service import VisionClassificationFailed, VisionService

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
IMAGES_DIR = PROJECT_ROOT / "data" / "images"

CONCURRENCY_LIMIT = 3  # spec §11.4 suggests "e.g. 5"; reduced to 3 after
# hitting real 429 rate-limit errors in testing with gemini-3.5-flash-lite's
# free-tier RPM cap — 3 concurrent workers plus VisionService's own
# per-call backoff (see vision_service.py's RATE_LIMIT_MAX_RETRIES)
# comfortably stays under the per-minute quota. Documented in BUILDLOG.md.

# Basic extension -> MIME type mapping. All corpus images are .jpg
# (seed_corpus.py always writes this extension), but this keeps the
# job correct if a .png or .webp is added to the corpus later.
MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


async def _classify_one(image_id: uuid.UUID, filename: str, semaphore: asyncio.Semaphore) -> str:
    """
    Classifies a single image in its own database session. Returns
    one of "succeeded", "flagged", or "failed" — never raises, so one
    bad image can never take down the asyncio.gather() for the rest
    of the batch (§11.4).
    """
    async with semaphore:
        file_path = IMAGES_DIR / filename
        if not file_path.exists():
            return "failed"

        mime_type = MIME_TYPES.get(file_path.suffix.lower(), "image/jpeg")
        image_bytes = file_path.read_bytes()

        async with AsyncSessionLocal() as db:
            vision = VisionService(db)
            try:
                tags = await vision.classify_image(image_id, image_bytes, mime_type)
            except VisionClassificationFailed:
                return "failed"
            except Exception:
                # Defensive catch-all: anything unexpected (network
                # blip, SDK error not already handled inside
                # VisionService) still counts as a per-image failure,
                # not a batch-ending crash.
                return "failed"

            # Embed the caption immediately after a successful
            # classification — matching (§12) requires image_vectors
            # to exist for every candidate, and the caption (the text
            # we embed, per §12.1) only exists once classification has
            # produced one. Embedding failure does NOT downgrade a
            # successful classification to "failed": the image_metadata
            # row is already correctly persisted; it just won't be
            # matchable until re-embedded (e.g. via a future retry
            # mechanism, or by re-running this job with force=true).
            try:
                await EmbeddingService(db).embed_image(image_id, tags.caption)
            except Exception:
                pass

            return "flagged" if tags.confidence < vision.guard_config.confidence_flag_threshold else "succeeded"


async def run_classification_job(job_id: uuid.UUID, force: bool = False) -> None:
    """
    Entry point invoked via FastAPI BackgroundTasks from
    POST /images/classify-batch. Updates the batch_jobs row
    (§9.9) as it progresses: pending -> running -> completed.
    """
    async with AsyncSessionLocal() as db:
        job = await db.get(BatchJob, job_id)
        if job is None:
            return  # shouldn't happen; job row is created before this is scheduled

        job.status = "running"
        job.started_at = datetime.now(timezone.utc)
        await db.commit()

        repo = ImageRepository(db)
        images: list[Image] = await repo.list_unclassified(force=force)

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    results = await asyncio.gather(
        *[_classify_one(img.id, img.filename, semaphore) for img in images]
    )

    succeeded = sum(1 for r in results if r == "succeeded")
    flagged = sum(1 for r in results if r == "flagged")
    failed = sum(1 for r in results if r == "failed")

    async with AsyncSessionLocal() as db:
        job = await db.get(BatchJob, job_id)
        if job is not None:
            job.status = "completed"
            job.succeeded_items = succeeded + flagged  # flagged images still succeeded classification
            job.failed_items = failed
            job.flagged_items = flagged
            job.completed_at = datetime.now(timezone.utc)
            await db.commit()