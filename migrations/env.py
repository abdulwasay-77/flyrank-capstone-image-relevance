"""
Alembic migration environment.

Wired to use the project's async SQLAlchemy engine and settings, and
to autogenerate against app.db.models.Base.metadata (all 8 tables
from spec §9). This is the piece that makes
`alembic revision --autogenerate` actually see the ORM models instead
of generating an empty migration.

Do not hand-edit the database schema outside of migrations generated
through this file — per §9.11, every schema change must be a
migration committed to migrations/versions/.
"""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import create_async_engine

# Make `app.*` importable when Alembic is invoked from the project root.
# (alembic.ini sets prepend_sys_path = . to reinforce this, but we set
# it explicitly here too so env.py works even if invoked oddly.)
import os
import sys
sys.path.insert(0, os.getcwd())

from app.config import get_settings  # noqa: E402
from app.db.models import Base  # noqa: E402  (importing models registers them on Base.metadata)

# Alembic Config object, provides access to values in alembic.ini
config = context.config

# Interpret the config file for Python logging (from alembic.ini's
# [loggers]/[handlers]/[formatters] sections).
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# This is what autogenerate compares the database against.
target_metadata = Base.metadata

# Inject the real database URL from .env at runtime, rather than
# storing it in alembic.ini (which is committed to git) — per NFR-5.
settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.async_database_url)


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode — generates SQL without a live
    DB connection. Not our normal path (we always have Neon
    reachable), but kept for completeness / CI scenarios that only
    want to inspect generated SQL.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode using the project's async engine,
    matching how the app itself connects (asyncpg driver) so behavior
    during migration matches behavior at runtime.
    """
    connectable = create_async_engine(
        config.get_main_option("sqlalchemy.url"),
        poolclass=pool.NullPool,
        # asyncpg takes SSL as a connect kwarg, not a URL query param —
        # see the long comment on Settings.async_database_url in
        # app/config.py for why this is needed here too.
        connect_args={"ssl": "require"},
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())