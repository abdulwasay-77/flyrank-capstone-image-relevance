# Build Log

This document records deviations from the original spec, bugs found during
implementation, and the reasoning behind each fix. Per the project's own
principle: an honest account of what actually happened is worth more than a
sanitized summary — several of the entries below describe real debugging
detours, including ones where the initial fix was wrong before the real root
cause was found.

---

## Phase 1 — Project Setup & Corpus

### PowerShell UTF-8 BOM issues
Several config files (`alembic.ini`, `data/manifest.json`) were initially
written using PowerShell's `Set-Content -Encoding UTF8`, which writes a
UTF-8 byte-order mark (BOM) by default. Python's `configparser` and `json`
modules both reject a leading BOM outright. Fixed by switching to
`[System.IO.File]::WriteAllText(..., New-Object System.Text.UTF8Encoding($false))`
for hand-written config files, and by making `scripts/seed_corpus.py` and
`scripts/seed_db.py` read JSON with `encoding="utf-8-sig"` (tolerates a BOM
if present, behaves identically to `utf-8` if not).

### `google-genai` version pinning
Initial `pip install` backtracked to `google-genai` 1.4.0 (very old) due to
a dependency conflict with the original `fastapi==0.104.1` pin (which
required `anyio<4.0.0`, while newer `google-genai` needs `anyio>=4.0`).
Resolved by upgrading FastAPI to a current version (0.141.1), which resolved
the `anyio` constraint conflict. `pip check` confirmed no broken
requirements afterward, and `requirements.txt` was regenerated via
`pip freeze` to pin exact versions (NFR-6 reproducibility).

### `seed_corpus.py` — manifest never saved on interruption
Original version only called `save_manifest()` once, at the very end of the
script. When Unsplash's free-tier rate limit (50 req/hour) was hit partway
through, the script exited before ever reaching that save — so all
already-downloaded images were on disk, but `manifest.json` stayed empty,
and a retry re-downloaded everything from scratch, burning more quota.
**Fix**: `save_manifest()` now runs after every category completes, so
partial progress is never lost on interruption.

### `seed_db.py` — Windows asyncio/SSL shutdown noise
Running the seed script printed a harmless-but-alarming
`Fatal error on SSL transport` / `RuntimeError: Event loop is closed`
traceback after all data had already committed successfully. Root cause:
Windows' `ProactorEventLoop` is noisier than Linux/Mac about asyncpg SSL
socket teardown timing at process exit. Fixed by explicitly calling
`await engine.dispose()` before the script exits, so connections are closed
before the event loop shuts down.

---

## Phase 2 — Database Schema & Migrations

### `postgresql://` → `postgresql+asyncpg://` and `sslmode`
Neon's connection string uses the `sslmode=require` query parameter
(psycopg-style), but `asyncpg` does not accept `sslmode` as a URL query
param at all — it raised
`TypeError: connect() got an unexpected keyword argument 'sslmode'`.
Fixed in `Settings.async_database_url`: strips the query string entirely
and configures SSL via `connect_args={"ssl": "require"}` on
`create_async_engine` instead.

### SQLAlchemy's reserved `metadata` attribute
The `Image` ORM model's classification relationship is named
`metadata_row`, not `metadata`, because SQLAlchemy reserves `metadata` as a
class-level attribute (the schema registry object) on every declarative
model. This was caught later (see Phase 3) when a Pydantic schema used the
field name `metadata` directly.

---

## Phase 3 — Images Vertical Slice

### Undocumented file: `app/services/image_service.py`
The original folder structure (spec §20) listed services mapped to
pipeline stages (`vision_service.py`, `embedding_service.py`, etc.) but no
plain CRUD/listing service for images themselves. Since routers must not
call repositories directly (§7.1), `image_service.py` was added to fill
this gap — a reasonable, documented deviation from the original file list.

### `ImageOut.metadata` silently returning SQLAlchemy's internal `MetaData`
`GET /images` returned a 500 with `input_value=MetaData()` for the
`metadata` field. Root cause: `ImageOut.metadata`'s `from_attributes=True`
validation does `getattr(image, "metadata")`, which — because SQLAlchemy
reserves that name — returned the ORM's internal schema registry object,
not the actual classification data (stored under `metadata_row`, per the
Phase 2 note above). **Fix**: `ImageOut.metadata` now uses
`Field(validation_alias="metadata_row")`, so Pydantic reads from the
correct ORM attribute while the public API field name stays `metadata`,
unchanged.

---

## Phase 4 — Vision Classification

### `gemini-2.0-flash` retired
The spec (§8) targeted `gemini-2.0-flash`. Mid-project, Google retired this
model; the API returned a live `404` directing callers to
`gemini-3.6-flash`. Updated `MODEL_NAME` in `vision_service.py` and the
pricing table in `cost_tracker_service.py` accordingly.

### Cost-log durability bug — hid the real rate-limit error for hours
`CostTrackerService.log_call()` originally only called `db.flush()`, never
`db.commit()`. When a Gemini call failed, `track_call`'s exception handler
logged the failure via `flush()` then re-raised — but since the session
this happened in was never committed, and a session's context manager
rolls back on exception exit, every failure log was silently discarded
before ever reaching the database. This meant `ai_call_log` looked
completely empty even after real, repeated failures, hiding the actual
error message for an extended debugging session. **Fix**: `log_call()` now
calls `db.commit()` immediately, making every log entry durable the
instant it's written, regardless of what the caller does afterward.

### `gemini-3.6-flash` free-tier quota exhaustion
Once cost logging was fixed and the real error became visible: 50 requests
failed with `429`, quota `5 requests/minute, 20/day` for
`gemini-3.6-flash` — Google's current flagship model, with a much tighter
free tier than older models. **Fix**: switched to `gemini-3.5-flash-lite`,
Google's purpose-built high-throughput/low-cost multimodal model,
confirmed via Google's own docs as the correct tool for this workload.
Also added retry-with-exponential-backoff specifically for `429` errors
(separate from the schema-validation retry loop), and reduced batch
concurrency in `classification_job.py` from 5 to 3 as an extra safety
margin (§11.4 says "e.g. 5", not a hard requirement).

### `VisionService` blocking the entire event loop
`_call_gemini` originally called `self._client.models.generate_content(...)`
— the **synchronous** SDK method — directly inside an `async def`
function. A blocking network call inside async code freezes the entire
asyncio event loop for its duration, meaning the whole FastAPI server
(including unrelated `GET /jobs` polling requests) was unresponsive for
the full batch's runtime. **Fix**: switched to
`self._client.aio.models.generate_content(...)`, the SDK's actual async
client, verified against the installed SDK version's real method
signature before applying the fix.

---

## Phase 5 — Embedding

### `text-embedding-004` shut down
Confirmed via Google's own API changelog: shut down January 14, 2026.
Current stable replacement, `gemini-embedding-001`, was verified to
support the `task_type` parameter (including `SEMANTIC_SIMILARITY`,
required by §12.1) against the actually-installed SDK before use.

### Images were classified but never embedded
`EmbeddingService` was built and unit-tested but never actually wired into
the classification pipeline — `classification_job.py` classified images
but never called `embed_image()`. This wasn't caught until testing
`GET /posts/{id}/suggestions`, which returned `NO_MATCH` with an empty
`rejected_candidates` list for every post — the tell that
`MatchingService.get_ranked_candidates()` found zero candidates, because
`image_vectors` was entirely empty. **Fix**: `classification_job.py` now
calls `EmbeddingService.embed_image()` immediately after a successful
classification. A one-off `scripts/backfill_embeddings.py` was written to
catch up the 50 already-classified images without re-spending vision-model
quota re-classifying them.

---

## Phase 6 — Matching & the Mismatch Guard

### `post_service.py` — caught and rewritten before shipping
The first draft of `PostService` had three real bugs, caught during review
before any of it was shown as final: a duplicated/incorrect
`upsert_vector` call, a placeholder `post_id=uuid.UUID(int=0)` that a
comment incorrectly claimed would be "overwritten by caller context" (it
never was), and empty `filename=""` values on the reused-suggestions code
path. The file was fully rewritten rather than patched piecemeal, and a
`list_for_post_with_filenames` join query was added to
`SuggestionRepository` so reused results could resolve real filenames.

### The Neon "hanging query" investigation
This was the longest single debugging session of the project. Summary of
what was tried, in order, and what was actually wrong:

1. **First hypothesis (wrong): stuck/corrupted connection.** An early
   `asyncpg.exceptions._base.InternalClientError: cannot switch to state 11`
   appeared after an abandoned 5+ minute request. Diagnosed as a leftover
   orphaned connection from an interrupted retry loop.
2. **Second hypothesis (wrong): missing timeouts.** Added
   `command_timeout` (asyncpg-level) and an `asyncio.wait_for` wrapper
   (app-level) so no request could hang indefinitely. This was a good
   defensive addition regardless, but raising the timeout value from 15s
   to 30s made no difference to the actual failure — a sign the query
   wasn't slow, something else was wrong.
3. **Third hypothesis (wrong): PgBouncer/prepared-statement
   incompatibility.** Added `statement_cache_size=0` to `connect_args`.
   Also did not fix it.
4. **Isolated with evidence, not guesses:** wrote
   `scripts/debug_suggestions.py` (timestamped step-by-step logging,
   outside uvicorn) and `scripts/debug_raw_asyncpg.py` (bypassing
   SQLAlchemy entirely). The raw asyncpg test proved the actual root
   cause: the exact join query — the one returning image candidates with
   their full embedding vectors — genuinely took **~30 seconds to
   complete successfully**, not indefinitely, every time. Neon's SQL
   Editor had looked "instant" because it runs on Neon's own
   infrastructure, right next to the database; the app's real client is
   in Pakistan, and the database is in `us-east-2` (Ohio) — the ~30s was
   real network transfer time for ~50 rows × large embedding vectors
   across that physical distance, not a bug.

   **Actual fix**: `command_timeout` raised to 60s,
   `SUGGESTIONS_TIMEOUT_SECONDS` raised to 240s — both now measured
   against real, confirmed latency rather than guessed.

   Lesson recorded here deliberately: the first three "fixes" were
   plausible-sounding but never verified against hard evidence before
   being applied, which wasted significant time. The breakthrough came
   only from isolating the exact failing operation outside the full
   application stack.

### Category inference producing different results for the same post
`GET /posts/{id}/suggestions` for the same wolf post returned `ACCEPTED`
on one run and `NO_MATCH` (all rejected for "category mismatch") on a
later run — a real reproducibility bug (violates NFR-9). Root cause:
`infer_post_category` picked a single top-1 category via embedding
similarity against the corpus vocabulary; for a post that legitimately
straddles two close categories ("The Return of the Gray Wolf to Northern
Forests" sits close to both `animal` and `nature`), small variance in the
in-process category-embedding cache (rebuilt after every server restart)
could flip which single category "won." **Fix**: `infer_post_category`
now returns the top 2 closest categories instead of 1, and the guard
accepts a category match against any of them. Verified with unit tests
covering both category orderings, confirming the fix resolves the
flip-flop without weakening rejection of genuinely unrelated categories.

---

## Phase 7 — Review API

### Reviewer identity has no dedicated database column
FR-5.2 asks for reviewer identity to be persisted ("a static value is
acceptable"), but the `suggestions` table (§9.7) was never given a
dedicated identity column — only `reviewer_note` and `reviewed_at`.
Rather than silently add an unplanned column, a fixed
`REVIEWER_IDENTITY` constant is folded into `reviewer_note` (e.g.
`[capstone_reviewer] looks correct`) in `review_service.py`. Documented
here and in the code rather than left as a silent gap.

---

## Phase 8 — Cost Tracking & Evaluation

### Eval set ground truth was initially wrong, not the system
The first version of `eval/eval_set.json` picked one arbitrary filename
per post (e.g. `domestic-cat-01.jpg`) as "the" correct answer. Running the
eval scored 2/10 (20%) — but manual inspection showed every "wrong" result
was still a same-topic, correctly-tagged photo (e.g. `domestic-cat-05.jpg`
for the cat post) from a corpus where multiple images per topic are
equally valid matches. The eval was measuring "did the system guess one
arbitrary label" rather than "did it find a relevant image" — the wrong
thing to optimize for, and not a reflection of actual system quality.
**Fix**: ground truth changed from one hardcoded filename to a filename
*prefix* (e.g. `"domestic-cat-"`), checking topic correctness rather than
forcing a single "more correct" photo among several equally valid ones.
Re-running against the corrected ground truth scored **10/10 (100%)
top-1 precision**. This correction and its reasoning are documented
directly in `eval/run_eval.py`'s own docstring, not just here.

---

## Phase 9 — Testing

Full automated suite added per §18's requirements: 57 tests across schema
validation, guard decision logic (including the category-list fix from
Phase 6), matching math, idempotency/retry call-counting, cost-log
durability, and API contract shapes. All AI calls and database sessions
are mocked (`unittest.mock`, FastAPI `dependency_overrides`) — the suite
runs in a few seconds with zero network access and zero cost, per §18's
explicit rule. One real bug the suite itself caught: `TestClient` executes
`BackgroundTasks` synchronously, so the `classify-batch` endpoint's tests
initially tried to make a genuine (failing) database connection via the
real background job — fixed by mocking that specific background task for
those contract tests.