"""
API request/response models.

One class per shape that crosses the HTTP boundary, covering every
endpoint in spec §10. Per NFR-2: "Input validation at every API
boundary via Pydantic — malformed input returns 422, never a raw 500."
These models are what make that true; FastAPI auto-validates request
bodies against them and auto-generates response schemas in OpenAPI docs.

Naming convention: `<Thing>Out` for API responses, `<Thing>In` /
`<Thing>Request` for request bodies. Response models are built from
ORM rows in the service layer, never returned as raw SQLAlchemy
objects (keeps the DB schema and the API contract independently
changeable — §7.1's "swap the DB without touching business logic"
cuts both ways: swap the API shape without touching the DB either).
"""

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# --- Shared enums-as-literals (kept as Literal, not native DB enums —
# see the comment at the top of app/db/models.py for why) ---

CallType = Literal["VISION_CLASSIFY", "EMBEDDING"]
JobType = Literal["CLASSIFICATION", "EMBEDDING"]
JobStatus = Literal["pending", "running", "completed", "failed", "indexed"]
ImageStatus = Literal["pending", "completed", "failed", "indexed"]
Decision = Literal["ACCEPTED", "REJECTED", "NO_MATCH"]
SuggestionSource = Literal["GUARD", "MANUAL_OVERRIDE"]
ReviewStatus = Literal["PENDING", "APPROVED", "REJECTED"]


# =====================================================================
# 10.1 Images
# =====================================================================


class ImageMetadataOut(BaseModel):
    """Nested inside ImageOut — the classification result, if any (§9.3)."""

    model_config = ConfigDict(from_attributes=True)

    subject: str
    category: str
    attributes: list[str]
    caption: str
    confidence: float
    needs_review: bool
    status: ImageStatus
    classified_at: datetime | None = None


class ImageOut(BaseModel):
    """GET /images list item, and the base of GET /images/{id}."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    source_url: str | None = None
    license: str
    created_at: datetime
    metadata: ImageMetadataOut | None = None


class ImageDetailOut(ImageOut):
    """GET /images/{id} — same as ImageOut; kept as a distinct name so
    the detail endpoint can grow extra fields later (e.g. embedding
    presence) without widening the list endpoint's payload."""

    pass


class ClassifyBatchResponse(BaseModel):
    """POST /images/classify-batch — the immediate response (§14.1);
    actual processing happens in the background, tracked via job_id."""

    job_id: uuid.UUID
    status: JobStatus = "pending"


# =====================================================================
# 10.2 Posts
# =====================================================================


class PostOut(BaseModel):
    """GET /posts list item and GET /posts/{id}."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    body: str
    created_at: datetime


# --- Suggestions payload shared by GET /posts/{id}/suggestions and
# GET /suggestions/{id} (§10.6, §10.7) ---


class TopSuggestionOut(BaseModel):
    """The accepted candidate, if any — null when decision = NO_MATCH."""

    image_id: uuid.UUID
    filename: str
    similarity_score: float
    confidence: float
    category_match: bool
    reason: str


class RejectedCandidateOut(BaseModel):
    """One entry in the rejected_candidates list (§10.6 example)."""

    image_id: uuid.UUID
    filename: str
    similarity_score: float | None = None
    category_match: bool
    reason: str


class PostSuggestionsOut(BaseModel):
    """
    GET /posts/{id}/suggestions response.

    Mirrors §10.6 (accepted case) and §10.7 (NO_MATCH case) exactly:
    top_suggestion is null and reason is top-level when decision is
    NO_MATCH; when ACCEPTED, reason lives inside top_suggestion instead
    and this top-level reason is omitted.
    """

    post_id: uuid.UUID
    post_title: str
    decision: Decision
    top_suggestion: TopSuggestionOut | None = None
    rejected_candidates: list[RejectedCandidateOut] = Field(default_factory=list)
    reason: str | None = None  # populated only for NO_MATCH (§10.7)


# =====================================================================
# 10.3 Suggestions / Review
# =====================================================================


class SuggestionOut(BaseModel):
    """GET /suggestions and GET /suggestions/{id} list/detail row."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    post_id: uuid.UUID
    image_id: uuid.UUID | None = None
    similarity_score: float | None = None
    category_match: bool
    confidence: float | None = None
    decision: Decision
    reason: str
    source: SuggestionSource
    review_status: ReviewStatus
    reviewer_note: str | None = None
    reviewed_at: datetime | None = None
    created_at: datetime


class SuggestionDetailOut(SuggestionOut):
    """
    GET /suggestions/{id} — "full decision trail" per §10.3. Extends
    the base row with the full ranked candidate list considered during
    matching, so a reviewer can see not just what was accepted/rejected
    but everything that was compared.
    """

    candidates_considered: list[RejectedCandidateOut] = Field(default_factory=list)


class ReviewActionRequest(BaseModel):
    """POST /suggestions/{id}/reject body. approve uses no body (or this,
    with note optional) since approval needs no justification text."""

    note: str | None = None


class OverrideRequest(BaseModel):
    """
    POST /posts/{id}/override body.

    Per §15.2: creates a new suggestions row with source=MANUAL_OVERRIDE,
    decision=ACCEPTED, review_status=APPROVED — bypassing the guard by
    design, always distinguishable from a guard-originated acceptance
    via `source`.
    """

    image_id: uuid.UUID
    note: str | None = None


# =====================================================================
# 10.4 Jobs
# =====================================================================


class BatchJobOut(BaseModel):
    """GET /jobs/{id} and GET /jobs list item."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    job_type: JobType
    status: JobStatus
    total_items: int
    succeeded_items: int
    failed_items: int
    flagged_items: int
    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_at: datetime


# =====================================================================
# 10.5 Cost & Ops
# =====================================================================


class CostByCallType(BaseModel):
    """One row inside CostSummaryOut.by_call_type."""

    call_type: CallType
    call_count: int
    total_cost_usd: float
    success_count: int
    failure_count: int


class CostSummaryOut(BaseModel):
    """
    GET /costs/summary response.

    This is the artifact pasted into EVIDENCE.md for the cost-tracking
    checklist item (§16) — keep it self-explanatory, since it may be
    read out of context in that document.
    """

    window_start: datetime | None = None
    window_end: datetime | None = None
    total_cost_usd: float
    by_call_type: list[CostByCallType]


class HealthOut(BaseModel):
    """GET /health — liveness/readiness."""

    status: Literal["ok"] = "ok"
    database: Literal["connected", "unreachable"]