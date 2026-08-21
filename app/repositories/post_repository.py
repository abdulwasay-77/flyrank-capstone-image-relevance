"""
Post repository — pure data access for posts and post_vectors.
Mirrors ImageRepository's shape and conventions (§7.1 layering).
"""

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Post, PostVector


class PostRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def list_all(self) -> list[Post]:
        result = await self.db.execute(select(Post).order_by(Post.created_at))
        return list(result.scalars().all())

    async def get_by_id(self, post_id: uuid.UUID) -> Post | None:
        return await self.db.get(Post, post_id)

    async def get_vector(self, post_id: uuid.UUID) -> PostVector | None:
        result = await self.db.execute(select(PostVector).where(PostVector.post_id == post_id))
        return result.scalar_one_or_none()

    async def upsert_vector(self, post_id: uuid.UUID, embedding: list[float], model_name: str) -> PostVector:
        existing = await self.get_vector(post_id)
        if existing is None:
            existing = PostVector(post_id=post_id)
            self.db.add(existing)

        existing.embedding = embedding
        existing.model_name = model_name

        await self.db.flush()
        return existing