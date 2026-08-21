"""
FastAPI application entrypoint.

Registers routers as they're built. Per §25's phased plan, routers
get added here incrementally — images first, then posts, suggestions,
jobs, costs — rather than stubbing all five up front with empty
bodies, since an empty router only adds import surface without
letting us test anything real.
"""

from fastapi import FastAPI

from app.routers import images, jobs, posts

app = FastAPI(
    title="FlyRank Capstone — Image Relevance Suggestion Engine",
    description="Suggests corpus images for blog posts via AI vision tagging + semantic matching + a mismatch guard.",
    version="0.1.0",
)

app.include_router(images.router)
app.include_router(jobs.router)
app.include_router(posts.router)


@app.get("/health", tags=["ops"])
async def health() -> dict:
    """
    Liveness/readiness check (§10.5).

    Deliberately minimal for now — returns process-level health only.
    A DB-connectivity check (matching HealthOut.database in
    api_models.py) gets added once app/routers/costs.py is built and
    we wire a real get_db() call through this endpoint too.
    """
    return {"status": "ok"}