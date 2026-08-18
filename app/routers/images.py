"""
Images router — HTTP layer only.

Per §7.1: "the HTTP layer never talks to the database directly."
Every function here does exactly three things: pull dependencies,
call the service, translate service outcomes to HTTP status codes.
No business logic, no SQL — that all lives in ImageService /
ImageRepository.

Endpoints implemented (§10.1):
  GET  /images                    - list all images with metadata + status
  GET  /images/{id}               - single image detail
  POST /images/classify-batch     - trigger the classification batch job
"""

import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.jobs.classification_job import run_classification_job
from app.schemas.api_models import ClassifyBatchResponse, ImageDetailOut, ImageOut
from app.services.image_service import ImageNotFoundError, ImageService

router = APIRouter(prefix="/images", tags=["images"])


@router.get("", response_model=list[ImageOut])
async def list_images(db: AsyncSession = Depends(get_db)) -> list[ImageOut]:
    """List all images with metadata + classification status (§10.1)."""
    service = ImageService(db)
    return await service.list_images()


@router.get("/{image_id}", response_model=ImageDetailOut)
async def get_image(image_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> ImageDetailOut:
    """Get single image detail including full tag metadata (§10.1)."""
    service = ImageService(db)
    try:
        return await service.get_image(image_id)
    except ImageNotFoundError:
        raise HTTPException(status_code=404, detail=f"Image {image_id} not found")


@router.post("/classify-batch", response_model=ClassifyBatchResponse)
async def classify_batch(
    background_tasks: BackgroundTasks,
    force: bool = Query(
        default=False,
        description="Reprocess images already classified as completed (§14.4).",
    ),
    db: AsyncSession = Depends(get_db),
) -> ClassifyBatchResponse:
    """
    Trigger the classification batch job (§10.1, §14.1).

    Returns immediately with a job_id and status="pending". The actual
    vision-model processing runs in a FastAPI BackgroundTask
    (app.jobs.classification_job.run_classification_job), which opens
    its own database sessions rather than reusing this request's —
    a BackgroundTask executes after the response is sent, by which
    point this request's `db` session is closed.
    """
    service = ImageService(db)
    response = await service.start_classification_batch(force=force)
    background_tasks.add_task(run_classification_job, response.job_id, force)
    return response