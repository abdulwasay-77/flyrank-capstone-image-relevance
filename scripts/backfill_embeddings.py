"""
scripts/backfill_embeddings.py

One-off backfill: embeds every image with image_metadata.status =
"completed" that doesn't yet have an image_vectors row.

Needed because classification_job.py originally only classified
images and never embedded them — fixed for all *future* batch runs,
but the 50 images already classified before that fix need a targeted
backfill rather than a full, quota-burning re-classification via
force=true.

Usage:
    python scripts/backfill_embeddings.py
"""

import asyncio
import sys
from pathlib import Path

from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.models import Image, ImageMetadata, ImageVector  # noqa: E402
from app.db.session import AsyncSessionLocal  # noqa: E402
from app.services.embedding_service import EmbeddingService  # noqa: E402


async def main() -> None:
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Image, ImageMetadata)
            .join(ImageMetadata, ImageMetadata.image_id == Image.id)
            .outerjoin(ImageVector, ImageVector.image_id == Image.id)
            .where(ImageMetadata.status == "completed")
            .where(ImageVector.id.is_(None))
        )
        rows = result.all()

    print(f"Found {len(rows)} classified images without an embedding.")

    succeeded, failed = 0, 0
    for image, metadata in rows:
        async with AsyncSessionLocal() as db:
            service = EmbeddingService(db)
            try:
                await service.embed_image(image.id, metadata.caption)
                succeeded += 1
                print(f"  Embedded {image.filename}")
            except Exception as exc:
                failed += 1
                print(f"  FAILED {image.filename}: {exc}")

    print(f"\nDone. {succeeded} embedded, {failed} failed.")

    from app.db.session import engine

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())