"""
Image repository — the only place in the codebase that writes raw
SQLAlchemy queries against images/image_metadata/image_vectors.

Per §7.1's layering: routers -> services -> repositories -> DB.
Everything here is pure data access — no business rules, no guard
logic, no HTTP concerns. A repository method takes/returns ORM model
instances or primitives; it never sees a Pydantic API-response model
(that translation happens in the service layer).
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import Image, ImageMetadata, ImageVector


class ImageRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    # --- Reads ---

    async def list_all(self) -> list[Image]:
        """All images with their metadata eager-loaded (avoids N+1 when
        the service layer builds ImageOut.metadata for each row)."""
        result = await self.db.execute(
            select(Image).options(selectinload(Image.metadata_row)).order_by(Image.created_at)
        )
        return list(result.scalars().all())

    async def get_by_id(self, image_id: uuid.UUID) -> Image | None:
        result = await self.db.execute(
            select(Image)
            .where(Image.id == image_id)
            .options(selectinload(Image.metadata_row))
        )
        return result.scalar_one_or_none()

    async def get_by_file_hash(self, file_hash: str) -> Image | None:
        """Used by the classification batch job for idempotency (§14.4):
        images.file_hash + image_metadata.status together determine
        whether an image needs (re)processing."""
        result = await self.db.execute(
            select(Image)
            .where(Image.file_hash == file_hash)
            .options(selectinload(Image.metadata_row))
        )
        return result.scalar_one_or_none()

    async def list_unclassified(self, force: bool = False) -> list[Image]:
        """
        Images that still need a vision-model pass.

        Without force: images with no metadata row yet, OR metadata
        with status in ("pending", "failed") — i.e. never successfully
        classified.
        With force=True: every image, regardless of current status —
        deliberate reprocessing per §14.4's `force=true` query param.
        """
        result = await self.db.execute(
            select(Image).options(selectinload(Image.metadata_row))
        )
        images = list(result.scalars().all())
        if force:
            return images
        return [
            img
            for img in images
            if img.metadata_row is None or img.metadata_row.status in ("pending", "failed")
        ]

    async def get_vector(self, image_id: uuid.UUID) -> ImageVector | None:
        result = await self.db.execute(
            select(ImageVector).where(ImageVector.image_id == image_id)
        )
        return result.scalar_one_or_none()

    async def list_all_vectors(self) -> list[ImageVector]:
        """Used by the matching engine (§12) to compute similarity
        against every classified image's embedding."""
        result = await self.db.execute(select(ImageVector))
        return list(result.scalars().all())

    # --- Writes ---

    async def create(
        self,
        filename: str,
        file_hash: str,
        license: str,
        source_url: str | None = None,
    ) -> Image:
        image = Image(
            filename=filename,
            file_hash=file_hash,
            license=license,
            source_url=source_url,
        )
        self.db.add(image)
        await self.db.flush()  # populates image.id without committing
        return image

    async def upsert_metadata(
        self,
        image_id: uuid.UUID,
        *,
        subject: str,
        category: str,
        attributes: list[str],
        caption: str,
        confidence: float,
        needs_review: bool,
        status: str,
        raw_model_response: dict | None = None,
    ) -> ImageMetadata:
        """
        Create or replace this image's metadata row. Called by
        VisionService after a successful (or exhausted-retry, failed)
        classification attempt — never by anything in the router layer
        directly (§7.1).
        """
        existing = await self.db.execute(
            select(ImageMetadata).where(ImageMetadata.image_id == image_id)
        )
        row = existing.scalar_one_or_none()

        if row is None:
            row = ImageMetadata(image_id=image_id)
            self.db.add(row)

        row.subject = subject
        row.category = category
        row.attributes = attributes
        row.caption = caption
        row.confidence = confidence
        row.needs_review = needs_review
        row.status = status
        row.raw_model_response = raw_model_response
        row.classified_at = datetime.now(timezone.utc) if status == "completed" else row.classified_at

        await self.db.flush()
        return row

    async def upsert_vector(
        self, image_id: uuid.UUID, embedding: list[float], model_name: str
    ) -> ImageVector:
        """Create or replace this image's embedding row (§9.4, unique on image_id)."""
        existing = await self.db.execute(
            select(ImageVector).where(ImageVector.image_id == image_id)
        )
        row = existing.scalar_one_or_none()

        if row is None:
            row = ImageVector(image_id=image_id)
            self.db.add(row)

        row.embedding = embedding
        row.model_name = model_name

        await self.db.flush()
        return row