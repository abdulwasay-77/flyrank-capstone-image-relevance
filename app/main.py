"""
FastAPI application entrypoint.

Registers routers as they're built. Per §25's phased plan, routers
get added here incrementally — images first, then posts, suggestions,
jobs, costs — rather than stubbing all five up front with empty
bodies, since an empty router only adds import surface without
letting us test anything real.
"""

from fastapi import Depends, FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.routers import costs, images, jobs, posts, suggestions
from app.schemas.api_models import HealthOut

app = FastAPI(
    title="FlyRank Capstone — Image Relevance Suggestion Engine",
    description="Suggests corpus images for blog posts via AI vision tagging + semantic matching + a mismatch guard.",
    version="0.1.0",
)

app.include_router(images.router)
app.include_router(jobs.router)
app.include_router(posts.router)
app.include_router(suggestions.router)
app.include_router(costs.router)


@app.get("/health", tags=["ops"], response_model=HealthOut)
async def health(db: AsyncSession = Depends(get_db)) -> HealthOut:
    """
    Liveness/readiness check (§10.5). Now checks real DB connectivity
    (a trivial SELECT 1) rather than just process liveness — a
    server that's "up" but can't reach Neon should report that
    honestly, not claim full health.
    """
    try:
        await db.execute(text("SELECT 1"))
        return HealthOut(status="ok", database="connected")
    except Exception:
        return HealthOut(status="ok", database="unreachable")