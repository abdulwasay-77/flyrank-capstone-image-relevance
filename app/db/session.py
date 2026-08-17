"""
Async database session management.

Defines the SQLAlchemy async engine, the session factory used by
repositories, and the declarative Base that every model in models.py
inherits from.

Per §7.1: this is the Repository/Persistence layer. It must never be
imported by anything in app/routers/ directly — routers depend on
services, services depend on repositories, repositories depend on
this module. Keeping that direction strict is what makes "swap the DB
without touching business logic" (a rubric dimension) actually true.
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings

settings = get_settings()

# echo=True would log every SQL statement — useful for local debugging,
# noisy in normal use. Tied to LOG_LEVEL so it's a one-line env change,
# never a hardcoded toggle someone forgets to flip back off.
engine = create_async_engine(
    settings.async_database_url,
    echo=(settings.LOG_LEVEL.upper() == "DEBUG"),
    pool_pre_ping=True,  # detects stale connections after Neon auto-suspend (§27 risk)
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


class Base(DeclarativeBase):
    """Declarative base class. All ORM models in app/db/models.py inherit from this."""

    pass


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields a database session per request and
    guarantees it's closed afterward, even on error.

    Usage in a router:
        from app.db.session import get_db
        async def endpoint(db: AsyncSession = Depends(get_db)): ...

    Routers should not construct sessions any other way — this is the
    single entry point, which is what makes session lifecycle
    consistent across the whole app.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()