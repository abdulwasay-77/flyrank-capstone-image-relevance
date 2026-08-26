# FlyRank Capstone — Image Relevance Suggestion Engine

An AI-powered backend that suggests relevant images from a licensed corpus
for blog posts, using vision-model tagging, semantic embedding search, and
a **Mismatch Guard** safety layer that refuses low-confidence or
category-wrong suggestions rather than forcing a bad match.

Built as a FlyRank AI Internship capstone project.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  HTTP Layer (FastAPI)                    │
│  Routers: /images  /posts  /suggestions  /jobs  /costs   │
└───────────────────────┬───────────────────────────────────┘
                         │  (Pydantic request/response models)
┌───────────────────────▼───────────────────────────────────┐
│                  Service / Logic Layer                    │
│  VisionService · EmbeddingService · MatchingService        │
│  GuardService · ReviewService · CostTrackerService         │
└───────────────────────┬───────────────────────────────────┘
                         │
┌───────────────────────▼───────────────────────────────────┐
│              Repository / Persistence Layer                │
│        SQLAlchemy models + repository classes              │
└───────────────────────┬───────────────────────────────────┘
                         │
┌───────────────────────▼───────────────────────────────────┐
│              PostgreSQL (hosted on Neon, cloud)             │
└─────────────────────────────────────────────────────────────┘

        (Background jobs run via FastAPI BackgroundTasks,
         calling the same Service layer — never bypassing it)
```

**Rule enforced throughout**: the HTTP layer never talks to the database
directly, and the service layer never imports FastAPI. This is what makes
the matching/guard business logic swappable independently of both the web
framework and the database (NFR-10).

### Pipeline flow

```
Images ──(batch job)──► VisionService ──► {tags, caption, confidence}
                              │
                              ▼
                      EmbeddingService ──► image_vectors

Posts  ──(on request)──► EmbeddingService ──► post_vectors
                              │
                              ▼
                      MatchingService (cosine similarity ranking)
                              │
                              ▼
                      GuardService (category + threshold checks)
                              │
                              ▼
                 suggestions table (full decision trail persisted)
```

---

## Setup — from a fresh clone (target: under 15 minutes)

### Prerequisites
- Python 3.10+ installed
- Git installed
- A Google account (for a free Gemini API key — no card required)
- A Neon account (for free Postgres — no card required)

### 1. Clone and enter the repo

```powershell
git clone https://github.com/abdulwasay-77/flyrank-capstone-image-relevance.git
cd flyrank-capstone-image-relevance
```

### 2. Get a Gemini API key

Go to [aistudio.google.com](https://aistudio.google.com) → sign in → **Get API key** → **Create API key**. Free, no card required.

### 3. Create a free Neon Postgres database

Go to [neon.tech](https://neon.tech) → sign up → create a new project. Copy the connection string it gives you (looks like `postgresql://user:pass@host.neon.tech/dbname?sslmode=require`).

### 4. Install dependencies

This project does **not** use a virtual environment — dependencies install globally.

```powershell
pip install -r requirements.txt
```

### 5. Configure environment variables

```powershell
copy .env.example .env
```

Edit `.env` and fill in:
```
GEMINI_API_KEY=your_real_key_here
DATABASE_URL=your_real_neon_connection_string_here
UNSPLASH_ACCESS_KEY=your_real_unsplash_key_here   # only needed to re-run scripts/seed_corpus.py
MAX_BUDGET_USD=1.00                               # project-wide AI spend circuit breaker
```

(An Unsplash key is only required if you want to re-download the image
corpus from scratch. If `data/images/` and `data/manifest.json` are
already present in the repo, you can skip straight to seeding the
database.)

### 6. Run database migrations

```powershell
alembic upgrade head
```

### 7. Seed the corpus and database

```powershell
python scripts\seed_corpus.py    # downloads ~50 licensed images (skip if data/images/ already populated)
python scripts\seed_db.py        # loads images + seed posts into Postgres
```

### 8. Run the application

```powershell
uvicorn app.main:app --reload
```

API docs: **http://localhost:8000/docs**

### 9. Run the classification + embedding pipeline

```powershell
curl -X POST http://localhost:8000/images/classify-batch
```

Poll `GET /jobs/{job_id}` until `status: "completed"`. This classifies
every image and embeds its caption automatically.

### 10. Run tests

```powershell
pytest
```

Expect `57 passed` in a few seconds — no network calls, no database
connection required (all AI calls and DB sessions are mocked, per §18).

### 11. Run the evaluation script

```powershell
python eval\run_eval.py
```

Prints: **`Top-1 Precision: 10/10 = 100.00%`** (see Evaluation Result below).

---

## A note on network latency

This project's database (Neon, `us-east-2`/Ohio) may be geographically
distant from wherever it's run. A single request that fetches many
embedding vectors (e.g. `GET /posts/{id}/suggestions`) can genuinely take
30–90 seconds on its first run — this is real network transfer time for
large vector payloads, not a bug or a hang. Confirmed directly via an
isolated raw-`asyncpg` test during development (see `BUILDLOG.md`).
Results are cached after the first computation per post, so repeat
requests are near-instant.

---

## Evaluation Result

```
Top-1 Precision: 10/10 = 100.00%
```

Computed by `eval/run_eval.py` against `eval/eval_set.json` (10 posts,
each mapped to a corpus topic). Ground truth is defined by filename
*prefix* rather than one single hardcoded file — the corpus contains
multiple valid images per topic (e.g. 5 different cat photos), so
requiring an exact single "correct" filename would measure the wrong
thing. See `BUILDLOG.md` for the full reasoning behind this design
choice, including an earlier, flawed version of this eval set that
scored 20% for a labeling reason, not a system-quality reason.

A qualitative near-miss test (not part of the 10-post precision set, but
demonstrated live during development) further confirms the guard
actively discriminates rather than always accepting: a photo the vision
model mistagged as `"coyote"` (`gray-wolf-02.jpg`) was correctly
**rejected** — `"Category mismatch: expected animal, detected coyote"`
— before the guard moved on to accept a correctly-tagged wolf photo
instead.

---

## Key Design Decisions

- **Category inference uses the top-2 closest categories**, not just the
  single closest one — a post can legitimately straddle two related
  topics (e.g. "wolf" sits close to both `animal` and `nature`), and a
  strict top-1 match was found to be non-deterministic between runs for
  such posts. See `BUILDLOG.md`.
- **Every guard decision is persisted**, not just the winning one —
  `GET /suggestions/{id}` can show the full trail of every candidate the
  guard actually evaluated for a post.
- **`source` distinguishes automated vs. human decisions** — a manual
  override (`POST /posts/{id}/override`) always creates a new row with
  `source: MANUAL_OVERRIDE`, never edits or hides the guard's original
  verdict.
- **Every AI API call is cost-logged**, success or failure, including
  retries — visible via `GET /costs/summary`. Before each paid call, the
  project-wide `MAX_BUDGET_USD` cap is checked against all logged spend. Once
  reached, no further AI call is made: classification jobs finish as
  `failed`, and a first-time `GET /posts/{id}/suggestions` returns `503` with
  the configured limit and current tracked spend.

## Known Deviations from the Original Spec

Several are due to the AI landscape changing mid-project (models retired,
new ones released) rather than implementation choices. Full details and
reasoning for every deviation are in **`BUILDLOG.md`** — summary:

- `gemini-2.0-flash` → `gemini-3.5-flash-lite` (two model swaps; the
  first replacement, `gemini-3.6-flash`, had a free-tier quota far too
  tight for a 50-image batch)
- `text-embedding-004` → `gemini-embedding-001` (the original model was
  shut down by Google mid-project)
- Category matching uses top-2 inferred categories, not top-1 (see above)
- `app/services/image_service.py` added — not in the original file list,
  needed as the service-layer home for basic image CRUD

## API Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/images` | List all images with classification status |
| GET | `/images/{id}` | Single image detail |
| POST | `/images/classify-batch` | Trigger classification + embedding batch job |
| GET | `/jobs/{id}` | Poll batch job progress |
| GET | `/posts` | List all posts |
| GET | `/posts/{id}` | Single post detail |
| GET | `/posts/{id}/suggestions` | Run/fetch the matching + guard pipeline for a post |
| POST | `/posts/{id}/override` | Manually assign an image, bypassing the guard |
| GET | `/suggestions` | List all suggestions (filterable) |
| GET | `/suggestions/{id}` | Full decision trail for one suggestion's post |
| POST | `/suggestions/{id}/approve` | Human-approve a suggestion |
| POST | `/suggestions/{id}/reject` | Human-reject a suggestion |
| GET | `/costs/summary` | Aggregated AI API cost by call type |
| GET | `/health` | Liveness + real database connectivity check |

## Tech Stack

FastAPI · SQLAlchemy (async) · Alembic · PostgreSQL (Neon) · Pydantic ·
Google Gemini API (`gemini-3.5-flash-lite`, `gemini-embedding-001`) ·
pytest · numpy

## Project Documentation

- **`BUILDLOG.md`** — full, honest history of every deviation, bug, and
  fix encountered during development
- **`EVIDENCE.md`** — one proof artifact per Definition-of-Done checklist
  item
