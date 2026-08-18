"""
scripts/seed_corpus.py

The "documented seed script" required by FR-1.1: downloads ~50
licensed images from Unsplash into data/images/ and writes
data/manifest.json (filename -> source_url, license, photographer).

Requires UNSPLASH_ACCESS_KEY in .env (free Unsplash Developer app,
see README.md setup steps).

Usage:
    python scripts/seed_corpus.py

Idempotent (§FR-4.4 spirit applied to seeding too): re-running skips
any filename already present in the existing manifest, so it's safe
to re-run after adding new categories or if a previous run was
interrupted partway through.

Rate limit note: Unsplash's free/demo tier allows 50 requests/hour
against api.unsplash.com. Actual image downloads (from
images.unsplash.com, the CDN) do NOT count against this limit — only
the search call and the optional download-tracking ping do. With 10
categories this script makes 10 search calls plus up to 50
download-tracking pings (~60 total) — if you hit a 403 partway
through, wait an hour and re-run; already-downloaded images are
skipped automatically.
"""

import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")
if not ACCESS_KEY:
    print("ERROR: UNSPLASH_ACCESS_KEY not found in .env.")
    print("Get a free key at https://unsplash.com/oauth/applications and add it to .env.")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "data" / "images"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.json"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Query -> number of images to fetch for that category.
# 10 categories x 5 images = ~50 total, satisfying FR-1.1's "~50
# locally stored images". Categories are deliberately diverse in
# subject/category, with a couple of near-neighbors (fox/wolf,
# dog/cat) included on purpose — these are useful later for
# demonstrating the Mismatch Guard actually rejecting a
# similar-but-wrong category (§13.1) rather than just accepting
# everything.
CATEGORIES: list[tuple[str, int]] = [
    ("red fox", 5),
    ("gray wolf", 5),
    ("golden retriever dog", 5),
    ("domestic cat", 5),
    ("honeybee macro", 5),
    ("mountain landscape", 5),
    ("coffee latte art", 5),
    ("laptop workspace", 5),
    ("sushi food", 5),
    ("bicycle city street", 5),
]

SEARCH_URL = "https://api.unsplash.com/search/photos"
HEADERS = {"Authorization": f"Client-ID {ACCESS_KEY}"}


def slugify(text: str) -> str:
    return "-".join(text.lower().split())


def load_existing_manifest() -> list[dict]:
    """
    Returns the existing manifest, or an empty list if the file
    doesn't exist yet OR exists but is empty (e.g. the placeholder
    empty file created during initial project scaffolding). Either
    case just means "nothing seeded yet" — not an error.

    Reads with utf-8-sig specifically because Windows tools (VS Code,
    PowerShell's default Set-Content) commonly save "UTF-8" files with
    a leading BOM, which plain utf-8 + json.loads rejects outright.
    utf-8-sig transparently strips a BOM if present and behaves
    exactly like utf-8 if not — safe either way.
    """
    if MANIFEST_PATH.exists():
        content = MANIFEST_PATH.read_text(encoding="utf-8-sig").strip()
        if not content:
            return []
        return json.loads(content)
    return []


def save_manifest(entries: list[dict]) -> None:
    """Writes without a BOM (default 'utf-8', not 'utf-8-sig') so this
    file stays plain, portable JSON — readable by seed_db.py, git diff
    tooling, and any other JSON parser without special-casing."""
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)


def main() -> None:
    manifest = load_existing_manifest()
    existing_filenames = {entry["filename"] for entry in manifest}
    total_downloaded = 0

    for query, count in CATEGORIES:
        print(f"\nSearching Unsplash for '{query}' (requesting {count})...")
        resp = requests.get(
            SEARCH_URL,
            headers=HEADERS,
            params={"query": query, "per_page": count, "orientation": "landscape"},
            timeout=15,
        )

        if resp.status_code == 401:
            print("ERROR: 401 Unauthorized — check UNSPLASH_ACCESS_KEY in .env.")
            sys.exit(1)
        if resp.status_code == 403:
            print("ERROR: 403 — rate limit likely exceeded (50 req/hour on free tier).")
            print("Wait an hour and re-run; already-downloaded images will be skipped.")
            sys.exit(1)
        resp.raise_for_status()

        results = resp.json().get("results", [])
        if not results:
            print(f"  No results for '{query}', skipping.")
            continue

        slug = slugify(query)
        for i, photo in enumerate(results, start=1):
            filename = f"{slug}-{i:02d}.jpg"

            if filename in existing_filenames:
                print(f"  Skipping {filename} (already in manifest)")
                continue

            image_url = photo["urls"]["regular"]
            photo_page_url = photo["links"]["html"]
            photographer = photo["user"]["name"]

            img_resp = requests.get(image_url, timeout=30)
            img_resp.raise_for_status()

            with open(IMAGES_DIR / filename, "wb") as f:
                f.write(img_resp.content)

            # Best-effort download-tracking ping per Unsplash API
            # guidelines. Deliberately non-blocking: tracking is a
            # courtesy/compliance requirement, not something the
            # image or manifest entry should be lost over if it fails
            # (e.g. hourly quota hit right at this moment).
            try:
                requests.get(
                    photo["links"]["download_location"], headers=HEADERS, timeout=10
                )
            except requests.RequestException:
                pass

            manifest.append(
                {
                    "filename": filename,
                    "source_url": photo_page_url,
                    "license": "Unsplash License",
                    "photographer": photographer,
                }
            )
            existing_filenames.add(filename)
            total_downloaded += 1
            print(f"  Downloaded {filename} (by {photographer})")

            time.sleep(0.2)  # gentle pacing

        # Save after every category, not just once at the end. This is
        # the difference between "resume where you left off" and "lose
        # all progress" if a rate limit (or any other error) interrupts
        # the run partway through — which is exactly what happened
        # during initial testing of this script.
        save_manifest(manifest)

    print(f"\nDone. {total_downloaded} new images downloaded this run.")
    print(f"Manifest now has {len(manifest)} total entries -> {MANIFEST_PATH}")


if __name__ == "__main__":
    main()