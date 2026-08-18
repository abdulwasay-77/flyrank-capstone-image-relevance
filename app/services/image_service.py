"""
Image service — business logic layer for images.

Per §7.1: routers call services, services call repositories. This is
where ORM rows get translated into the Pydantic response shapes from
app/schemas/api_models.py, and where any business rule beyond plain
CRUD lives (e.g. deciding what counts as "already classified").

NOTE on scope: this file handles listing/reading images and kicking
off the classification batch job's bookkeeping (creating the
batch_jobs row, per §14.1). The actual vision-model call loop lives
in app/services/vision_service.py (built separately, next phase per
§25 Phase 2) and app/jobs/classification_job.py orchestrates the two
together. This keeps ImageService focused and testable without a live
AI dependency (NFR-7: schema validation and business logic tested
without live network calls).
"""

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import BatchJob
from app.repositories.image_repository import ImageRepository
from app.schemas.api_models import ClassifyBatchResponse, ImageDetailOut, ImageOut


class ImageNotFoundError(Exception):
    """Raised when a requested image_id doesn't exist. Routers translate
    this to a 404 — services never construct HTTP responses directly."""

    def __init__(self, image_id: uuid.UUID):
        self.image_id = image_id
        super().__init__(f"Image {image_id} not found")


class ImageService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.repo = ImageRepository(db)

    async def list_images(self) -> list[ImageOut]:
        images = await self.repo.list_all()
        return [ImageOut.model_validate(img) for img in images]

    async def get_image(self, image_id: uuid.UUID) -> ImageDetailOut:
        image = await self.repo.get_by_id(image_id)
        if image is None:
            raise ImageNotFoundError(image_id)
        return ImageDetailOut.model_validate(image)

    async def start_classification_batch(self, force: bool = False) -> ClassifyBatchResponse:
        """
        Creates the batch_jobs bookkeeping row and returns immediately
        with status="pending" (§14.1) — the router wires the actual
        background execution via FastAPI's BackgroundTasks, calling
        into app/jobs/classification_job.py, which is what will
        eventually update this row's progress counters as it runs.

        `force` is passed through unchanged — the job itself decides
        (via ImageRepository.list_unclassified(force=...)) whether to
        reprocess already-completed images, per §14.4.
        """
        candidates = await self.repo.list_unclassified(force=force)

        job = BatchJob(
            job_type="CLASSIFICATION",
            status="pending",
            total_items=len(candidates),
        )
        self.db.add(job)
        await self.db.flush()
        await self.db.commit()

        return ClassifyBatchResponse(job_id=job.id, status="pending")