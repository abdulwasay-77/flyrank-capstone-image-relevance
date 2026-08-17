"""
SQLAlchemy ORM models.

One class per table in spec §9. Column names, types, and constraints
match the spec tables exactly (§9.2–§9.9). Indexes follow the summary
in §9.10. Foreign keys are enforced per NFR-8.

Enum-like TEXT columns (status, decision, category, etc.) are modeled
as plain `String` with a comment noting the allowed values, rather
than native Postgres ENUM types — this keeps Alembic migrations
simpler (no ALTER TYPE dance when a new status value is added later)
while the application layer (Pydantic schemas) is what actually
enforces the allowed set on the way in.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    ARRAY,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


class Image(Base):
    """§9.2 — images"""

    __tablename__ = "images"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    filename: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    license: Mapped[str] = mapped_column(Text, nullable=False)
    file_hash: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    metadata_row: Mapped["ImageMetadata | None"] = relationship(
        back_populates="image", uselist=False, cascade="all, delete-orphan"
    )
    vector: Mapped["ImageVector | None"] = relationship(
        back_populates="image", uselist=False, cascade="all, delete-orphan"
    )


class ImageMetadata(Base):
    """§9.3 — image_metadata. status: pending | completed | failed"""

    __tablename__ = "image_metadata"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    image_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("images.id"), nullable=False, index=True
    )
    subject: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    attributes: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    caption: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    needs_review: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    raw_model_response: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    classified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    image: Mapped["Image"] = relationship(back_populates="metadata_row")

    __table_args__ = (
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="ck_image_metadata_confidence_range"),
    )


class ImageVector(Base):
    """§9.4 — image_vectors"""

    __tablename__ = "image_vectors"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    image_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("images.id"), nullable=False, unique=True, index=True
    )
    embedding: Mapped[list[float]] = mapped_column(ARRAY(Float), nullable=False)
    model_name: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    image: Mapped["Image"] = relationship(back_populates="vector")


class Post(Base):
    """§9.5 — posts"""

    __tablename__ = "posts"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    title: Mapped[str] = mapped_column(Text, nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    vector: Mapped["PostVector | None"] = relationship(
        back_populates="post", uselist=False, cascade="all, delete-orphan"
    )


class PostVector(Base):
    """§9.6 — post_vectors"""

    __tablename__ = "post_vectors"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    post_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("posts.id"), nullable=False, unique=True, index=True
    )
    embedding: Mapped[list[float]] = mapped_column(ARRAY(Float), nullable=False)
    model_name: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    post: Mapped["Post"] = relationship(back_populates="vector")


class Suggestion(Base):
    """
    §9.7 — suggestions.
    decision: ACCEPTED | REJECTED | NO_MATCH  (guard's automated verdict, §15.1)
    source: GUARD | MANUAL_OVERRIDE
    review_status: PENDING | APPROVED | REJECTED  (human verdict, independent of decision)
    """

    __tablename__ = "suggestions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    post_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("posts.id"), nullable=False, index=True
    )
    # nullable: NO_MATCH decisions have no image (§9.7)
    image_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("images.id"), nullable=True
    )
    similarity_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    category_match: Mapped[bool] = mapped_column(Boolean, nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    decision: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(Text, nullable=False)
    review_status: Mapped[str] = mapped_column(
        Text, nullable=False, default="PENDING", index=True
    )
    reviewer_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    post: Mapped["Post"] = relationship()
    image: Mapped["Image | None"] = relationship()


class AICallLog(Base):
    """§9.8 — ai_call_log. call_type: VISION_CLASSIFY | EMBEDDING"""

    __tablename__ = "ai_call_log"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    call_type: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    reference_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    model_name: Mapped[str] = mapped_column(Text, nullable=False)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    estimated_cost_usd: Mapped[float] = mapped_column(
        Numeric(10, 6), nullable=False, default=0
    )
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), index=True
    )


class BatchJob(Base):
    """§9.9 — batch_jobs. job_type: CLASSIFICATION | EMBEDDING. status: pending | running | completed | failed"""

    __tablename__ = "batch_jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    job_type: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    total_items: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    succeeded_items: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed_items: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    flagged_items: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )