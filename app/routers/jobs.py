"""
Jobs router — polling endpoint for batch job progress (§10.4).

Minimal by design: this router only reads batch_jobs rows. It never
starts a job itself (that's POST /images/classify-batch) — this is
purely "how is my job doing".
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import BatchJob
from app.db.session import get_db
from app.schemas.api_models import BatchJobOut

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("", response_model=list[BatchJobOut])
async def list_jobs(db: AsyncSession = Depends(get_db)) -> list[BatchJobOut]:
    result = await db.execute(select(BatchJob).order_by(BatchJob.created_at.desc()))
    jobs = result.scalars().all()
    return [BatchJobOut.model_validate(j) for j in jobs]


@router.get("/{job_id}", response_model=BatchJobOut)
async def get_job(job_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> BatchJobOut:
    job = await db.get(BatchJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return BatchJobOut.model_validate(job)