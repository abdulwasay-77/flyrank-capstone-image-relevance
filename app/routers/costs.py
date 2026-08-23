"""
Costs & ops router (§10.5).
"""

from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.api_models import CostSummaryOut
from app.services.cost_tracker_service import CostTrackerService

router = APIRouter(prefix="/costs", tags=["costs"])


@router.get("/summary", response_model=CostSummaryOut)
async def get_cost_summary(
    window_start: datetime | None = Query(default=None, description="ISO 8601 start of window, inclusive"),
    window_end: datetime | None = Query(default=None, description="ISO 8601 end of window, inclusive"),
    db: AsyncSession = Depends(get_db),
) -> CostSummaryOut:
    """
    §16: aggregate AI cost by call_type, optionally bounded by a time
    window. With no window params, aggregates all-time — this is the
    artifact pasted into EVIDENCE.md for the cost-tracking checklist
    item.
    """
    service = CostTrackerService(db)
    summary = await service.get_summary(window_start, window_end)
    return CostSummaryOut(**summary)