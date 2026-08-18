"""
scripts/seed_db.py

Loads the downloaded image corpus (data/manifest.json + data/images/)
and seed blog posts (data/seed_posts.json) into Postgres, per §21.8.

Usage:
    python scripts/seed_db.py

Idempotent: images are matched by file_hash (§14.4's idempotency
mechanism, reused here) and posts are matched by exact title — running
this twice does not create duplicate rows, so it's safe to re-run
after adding new entries to either JSON file.

NOTE on layering: images go through ImageRepository (already built).
Posts are inserted directly via the ORM here rather than through a
PostRepository, because app/repositories/post_repository.py hasn't
been built yet (it's next, alongside PostService). This is a
deliberate, temporary exception to the "always go through a
repository" rule (§7.1) scoped to this one-off seeding script — noted
here and in BUILDLOG.md rather than silently done. Once
PostRepository exists, this script should be updated to use it
instead of touching Post the ORM model directly.
"""

import asyncio
import hashlib
import json
import sys
from pathlib import Path

from sqlalchemy import select

# Make `app.*` importable when run as `python scripts/seed_db.py`
# from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.models import Post  # noqa: E402
from app.db.session import AsyncSessionLocal  # noqa: E402
from app.repositories.image_repository import ImageRepository  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "data" / "images"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.json"
SEED_POSTS_PATH = PROJECT_ROOT / "data" / "seed_posts.json"


def read_json(path: Path) -> list[dict]:
    """utf-8-sig tolerates a BOM if the file was saved by a Windows
    tool — see scripts/seed_corpus.py for why this matters here."""
    if not path.exists():
        print(f"ERROR: {path} not found.")
        sys.exit(1)
    content = path.read_text(encoding="utf-8-sig").strip()
    if not content:
        return []
    return json.loads(content)


def sha256_of_file(path: Path) -> str:
    """Used for images.file_hash — the idempotency key §14.4 relies on
    to detect whether an image has already been processed/reprocessed."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


async def seed_images(db) -> tuple[int, int]:
    manifest = read_json(MANIFEST_PATH)
    repo = ImageRepository(db)

    created, skipped = 0, 0
    for entry in manifest:
        filename = entry["filename"]
        file_path = IMAGES_DIR / filename

        if not file_path.exists():
            print(f"  WARNING: {filename} listed in manifest but not found on disk, skipping.")
            continue

        file_hash = sha256_of_file(file_path)

        existing = await repo.get_by_file_hash(file_hash)
        if existing is not None:
            skipped += 1
            continue

        await repo.create(
            filename=filename,
            file_hash=file_hash,
            license=entry.get("license", "Unknown"),
            source_url=entry.get("source_url"),
        )
        created += 1
        print(f"  Inserted image: {filename}")

    return created, skipped


async def seed_posts(db) -> tuple[int, int]:
    posts = read_json(SEED_POSTS_PATH)

    created, skipped = 0, 0
    for entry in posts:
        title = entry["title"]

        # Match-by-title idempotency check (see module docstring for
        # why this is a direct ORM query rather than PostRepository).
        result = await db.execute(select(Post).where(Post.title == title))
        existing = result.scalar_one_or_none()
        if existing is not None:
            skipped += 1
            continue

        db.add(Post(title=title, body=entry["body"]))
        created += 1
        print(f"  Inserted post: {title}")

    return created, skipped


async def main() -> None:
    async with AsyncSessionLocal() as db:
        print("Seeding images...")
        img_created, img_skipped = await seed_images(db)
        await db.commit()

        print("\nSeeding posts...")
        post_created, post_skipped = await seed_posts(db)
        await db.commit()

    print("\nDone.")
    print(f"Images: {img_created} inserted, {img_skipped} already existed (skipped)")
    print(f"Posts:  {post_created} inserted, {post_skipped} already existed (skipped)")

    # Explicitly dispose the engine's connection pool before the event
    # loop closes. Without this, asyncpg's SSL sockets can still be
    # mid-teardown when Windows' ProactorEventLoop shuts down at
    # process exit, producing a harmless but alarming-looking
    # "Fatal error on SSL transport" / "Event loop is closed"
    # traceback after the script has already finished its real work.
    from app.db.session import engine

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())