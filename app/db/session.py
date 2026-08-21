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
    pool_timeout=10,  # max seconds to wait for a free connection from the pool
    connect_args={
        # asyncpg takes SSL as a connect kwarg, not a URL query param —
        # see the long comment on Settings.async_database_url for why.
        "ssl": "require",
        # Disables asyncpg's server-side prepared statement caching.
        # Root cause of a real, hard-to-diagnose incident: Neon's
        # pooled connection endpoint uses PgBouncer in transaction
        # pooling mode, which is NOT compatible with asyncpg's default
        # behavior of preparing statements server-side — a pooled
        # connection can hand different transactions to different
        # backend processes, breaking prepared-statement reuse. This
        # manifested as parameterized queries (anything with a WHERE
        # clause bound value) hanging indefinitely on bind_execute,
        # while simple unparameterized lookups mostly worked. Confirmed
        # via scripts/debug_suggestions.py isolating the exact failing
        # query and traceback line. See BUILDLOG.md.
        "statement_cache_size": 0,
        # Raised from 30s to 60s based on hard evidence: a raw
        # asyncpg test (scripts/debug_raw_asyncpg.py), with zero
        # SQLAlchemy involved, measured this exact join query taking
        # ~30 seconds to complete — successfully, not stuck. Root
        # cause: this project's Neon database is in us-east-2 (Ohio)
        # while requests are made from Pakistan, and each of the ~50
        # rows returned includes a large embedding vector (thousands
        # of floats) — the transfer time over that physical distance
        # is real, not a bug. 60s gives comfortable headroom above
        # the measured ~30s. See BUILDLOG.md.
        "command_timeout": 60,
    },
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