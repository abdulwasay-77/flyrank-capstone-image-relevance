"""
scripts/debug_raw_asyncpg.py

Runs the exact same join query directly via raw asyncpg — no
SQLAlchemy, no ORM, none of our own connect_args. If this hangs too,
the problem is in asyncpg/the network path itself. If this works
instantly (matching what Neon's SQL Editor showed), the problem is
specifically in how SQLAlchemy is using asyncpg in this project.

Usage:
    python scripts/debug_raw_asyncpg.py
"""

import asyncio
import os
import sys
import time
from pathlib import Path

import asyncpg
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


async def main() -> None:
    raw_url = os.getenv("DATABASE_URL")
    if not raw_url:
        print("DATABASE_URL not found in .env")
        return

    # asyncpg's own connect() wants postgres:// or postgresql://, no
    # +asyncpg suffix, and no query string — strip both like our
    # app's Settings.async_database_url does, just without the
    # SQLAlchemy driver prefix this time.
    url = raw_url.split("?", 1)[0].replace("postgresql://", "postgres://", 1)

    log(f"Connecting directly via asyncpg...")
    conn = await asyncpg.connect(url, ssl="require", timeout=15)
    log("Connected")

    log("Running simple query (SELECT 1)...")
    result = await conn.fetchval("SELECT 1")
    log(f"Simple query returned: {result}")

    log("Running the join query...")
    rows = await conn.fetch(
        """
        SELECT i.id, i.filename, m.category, m.confidence, v.embedding
        FROM images i
        JOIN image_metadata m ON m.image_id = i.id
        JOIN image_vectors v ON v.image_id = i.id
        WHERE m.status = $1
        """,
        "completed",
    )
    log(f"Join query returned {len(rows)} rows")

    await conn.close()
    log("Connection closed, done")


if __name__ == "__main__":
    asyncio.run(main())