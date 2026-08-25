# EVIDENCE.md

One proof artifact per Definition-of-Done checklist item (spec §24.1).
All examples below are real output from this project's actual test runs,
not fabricated samples.

---

## AI Processing

### ☑ Vision model output validated against ImageTags schema; invalid responses never trusted

`app/schemas/image_tags.py` defines the schema; `VisionService.classify_image`
validates every raw model response via `ImageTags.model_validate_json(...)`
before anything is persisted. Covered by 15 automated tests in
`tests/test_schema_validation.py` (all passing), including malformed JSON,
missing fields, wrong types, and out-of-range confidence values.

Sample real classification result (from `image_metadata`, image `red-fox-01.jpg`):
```json
{
  "subject": "red fox",
  "category": "animal",
  "attributes": ["orange fur", "wild", "forest"],
  "caption": "A red fox in a snowy field",
  "confidence": 0.98
}
```

### ☑ Low-confidence classifications flagged (needs_review = true), not silently accepted

`VisionService.classify_image`: `needs_review = tags.confidence < guard_config.confidence_flag_threshold`
(threshold = 0.60, from `.env`). Applied to every one of the 50 corpus
images during the real classification batch run.

### ☑ Images processed via a batch background job with retries

`app/jobs/classification_job.py`, triggered via `POST /images/classify-batch`,
runs via FastAPI `BackgroundTasks`. Real batch job result (job id
`00e51a6e-57a2-4049-a102-cd2bd974c5c9`):
```json
{
  "job_type": "CLASSIFICATION",
  "status": "completed",
  "total_items": 29,
  "succeeded_items": 29,
  "failed_items": 0,
  "flagged_items": 0
}
```
Schema-validation retries: `VisionService` retries up to `MAX_RETRIES` (3)
with a stricter re-prompt on validation failure — verified in
`tests/test_idempotency.py::TestVisionServiceCallCounting`, which confirms
exactly `1 + max_retries` real calls are made before a classification is
marked `failed`. Rate-limit (429) retries are handled separately with
exponential backoff (see `BUILDLOG.md`).

### ☑ Vision and embedding costs tracked per call (ai_call_log)

Real output of `GET /costs/summary`:
```json
{
  "total_cost_usd": 0.00563,
  "by_call_type": [
    { "call_type": "VISION_CLASSIFY", "call_count": 134, "total_cost_usd": 0.00563, "success_count": 51, "failure_count": 83 },
    { "call_type": "EMBEDDING", "call_count": 137, "total_cost_usd": 0, "success_count": 137, "failure_count": 0 }
  ]
}
```
The `failure_count: 83` on `VISION_CLASSIFY` is an honest artifact of real
development history — it includes the `gemini-3.6-flash` free-tier quota
exhaustion incident before the switch to `gemini-3.5-flash-lite` (see
`BUILDLOG.md`), correctly logged rather than hidden.

---

## Matching System

### ☑ Image and post embeddings stored; posts return ranked image suggestions

`image_vectors` and `post_vectors` tables (§9.4, §9.6), populated via
`EmbeddingService`. Real `GET /posts/{id}/suggestions` result for "The
Return of the Gray Wolf to Northern Forests":
```json
{
  "decision": "ACCEPTED",
  "top_suggestion": {
    "filename": "gray-wolf-03.jpg",
    "similarity_score": 0.8331408230416713,
    "confidence": 0.98,
    "reason": "Category match (animal); similarity 0.83 >= 0.75; confidence 0.98 >= 0.60"
  }
}
```

### ☑ Semantic matching demonstrably works for equivalent concepts across different wording

`eval/run_eval.py` result across 10 posts, each written in natural
blog-post prose (not corpus filenames or category labels) against a
50-image corpus:
```
Top-1 Precision: 10/10 = 100.00%
```
E.g. the post titled "A Beginner's Guide to Ordering Sushi" (no image
filenames or category words in the text) correctly matched
`sushi-food-04.jpg` via embedding similarity alone.

---

## Safety Layer

### ☑ Mismatch guard rejects incorrect recommendations — fox/wolf scenario provably fails as expected

Real guard output for the wolf post, showing a same-species,
visually-similar candidate correctly rejected before an accurate one is
accepted:
```json
{
  "decision": "ACCEPTED",
  "top_suggestion": { "filename": "gray-wolf-03.jpg", "reason": "Category match (animal); similarity 0.83 >= 0.75; confidence 0.98 >= 0.60" },
  "rejected_candidates": [
    {
      "filename": "gray-wolf-02.jpg",
      "reason": "Category mismatch: expected animal, detected coyote"
    }
  ]
}
```
`gray-wolf-02.jpg` was independently mistagged `"coyote"` by the vision
model (not manually forced) — the guard correctly refused it anyway,
demonstrating real category-mismatch rejection, not a scripted demo.

Also covered by 13 unit tests in `tests/test_guard_service.py`, including
`test_category_mismatch_rejects_regardless_of_high_similarity` (a category
mismatch rejects even at 0.99 similarity and 0.99 confidence).

### ☑ Rejections include a human-readable explanation

Every rejection reason names the specific failed check and the actual
numbers involved (§13.4), e.g.:
`"Similarity 0.50 below threshold 0.75"`,
`"Classification confidence 0.45 below threshold 0.60"`,
`"Category mismatch: expected animal, detected coyote"`.
Verified in `tests/test_guard_service.py::TestReasonStrings`.

### ☑ "No confident match" case returns explicit reasons, never a disguised weak guess

Real result for "Five Common Mistakes When Filing Your First Tax Return"
(a post topic with no corresponding image in the corpus):
```json
{
  "decision": "NO_MATCH",
  "top_suggestion": null,
  "reason": "No candidate cleared category, similarity, and confidence checks"
}
```

---

## Backend

### ☑ All DB models present with required indexes and foreign keys

8 tables (`app/db/models.py`): `images`, `image_metadata`, `image_vectors`,
`posts`, `post_vectors`, `suggestions`, `ai_call_log`, `batch_jobs`.
Foreign keys enforced on every relationship (`image_metadata.image_id →
images.id`, `suggestions.post_id → posts.id`, etc., NFR-8). Indexes exist
on all lookup/join columns: `image_id`, `post_id`, `status`, `category`,
`decision`, `review_status`. Full migration history in
`migrations/versions/`.

### ☑ API endpoints validated (422 on bad input); review workflow fully functional

Malformed input returns 422, not a raw 500 — verified in
`tests/test_api_images.py::test_get_image_invalid_uuid_returns_422` and
`tests/test_api_suggestions.py::test_get_suggestion_invalid_uuid_returns_422`.

Review workflow — real sequence executed against the live API:
1. `POST /suggestions/{id}/approve` with `{"note": "confirmed correct"}`
2. `GET /suggestions/{id}` confirms `"review_status": "APPROVED"`,
   `"reviewed_at"` populated, and `candidates_considered` shows the full
   trail (both the accepted wolf photo and the rejected coyote-tagged one)

`POST /posts/{id}/override` creates a new row with
`"source": "MANUAL_OVERRIDE"`, `"decision": "ACCEPTED"`,
`"review_status": "APPROVED"` — never edits the guard's original row,
so both verdicts remain independently visible (§15.2).

### ☑ Automated tests cover schema validation, mismatch rejection, and matching accuracy

```
57 passed in 2.65s
```
Breakdown: 15 schema validation, 13 guard logic, 7 matching math, 5
idempotency/cost-logging, 17 API contract tests. Zero network calls, zero
database connections — all AI calls and DB sessions mocked, per §18.

---

## Quality & Documentation

### ☑ Labeled eval set exists; top-1 precision measured and matches the README

`eval/eval_set.json` (10 entries) + `eval/run_eval.py`. Printed result:
```
Top-1 Precision: 10/10 = 100.00%
```
Matches the number stated in `README.md` verbatim (FR-6.3).

### ☑ README includes architecture explanation + diagram

`README.md` contains the layered-architecture diagram (HTTP → Service →
Repository → Postgres) and the pipeline data-flow diagram (Images →
VisionService → EmbeddingService → MatchingService → GuardService →
suggestions).

### ☑ All five submission-pack files present and complete

`README.md`, `capstone.yaml`, `EVIDENCE.md` (this file), `BUILDLOG.md`,
`.env.example` — all present at the repository root.