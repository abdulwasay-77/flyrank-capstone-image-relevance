"""
§18 "Idempotency": re-running the batch job on already-processed
images does not create duplicates or extra AI calls.
§18 "Cost logging": every mocked AI call results in exactly one
ai_call_log row.

Per §18's rule: AI API calls are mocked, tests are deterministic with
zero network access and zero cost. The real database is also mocked
here (via unittest.mock.AsyncMock) rather than requiring a live Neon
connection during automated test runs — this keeps `pytest` runnable
offline in CI, with real end-to-end database behavior instead
verified via the manually-run debug_*.py scripts (per §18's carve-out
for "a small number of manually-run live scripts").
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.db.models import Image, ImageMetadata
from app.repositories.image_repository import ImageRepository
from app.schemas.image_tags import ImageTags
from app.services.vision_service import VisionService


class TestListUnclassifiedFiltering:
    """Tests the actual filtering logic in
    ImageRepository.list_unclassified against a mocked session — this
    is the mechanism idempotent reprocessing depends on (§14.4)."""

    def _make_image(self, filename: str, status: str | None):
        img = MagicMock(spec=Image)
        img.filename = filename
        if status is None:
            img.metadata_row = None
        else:
            img.metadata_row = MagicMock(spec=ImageMetadata)
            img.metadata_row.status = status
        return img

    @pytest.mark.asyncio
    async def test_excludes_completed_images_without_force(self):
        images = [
            self._make_image("completed.jpg", "completed"),
            self._make_image("pending.jpg", "pending"),
            self._make_image("failed.jpg", "failed"),
            self._make_image("never_classified.jpg", None),
        ]
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = images
        mock_db.execute.return_value = mock_result

        repo = ImageRepository(mock_db)
        result = await repo.list_unclassified(force=False)

        filenames = {img.filename for img in result}
        assert filenames == {"pending.jpg", "failed.jpg", "never_classified.jpg"}
        assert "completed.jpg" not in filenames

    @pytest.mark.asyncio
    async def test_force_true_includes_everything(self):
        images = [
            self._make_image("completed.jpg", "completed"),
            self._make_image("pending.jpg", "pending"),
        ]
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = images
        mock_db.execute.return_value = mock_result

        repo = ImageRepository(mock_db)
        result = await repo.list_unclassified(force=True)

        assert len(result) == 2


class TestVisionServiceCallCounting:
    """Verifies retries make real, separate, individually-logged API
    calls rather than silently reusing a cached response."""

    def _make_vision_service_with_mocks(self):
        mock_db = AsyncMock()
        service = VisionService.__new__(VisionService)  # bypass __init__'s real genai.Client()
        service.db = mock_db
        service.repo = AsyncMock()
        service.cost_tracker = MagicMock()
        service.settings = MagicMock(GEMINI_API_KEY="test-key")
        from app.schemas.guard_config import GuardConfig

        service.guard_config = GuardConfig(
            similarity_threshold=0.75,
            confidence_accept_threshold=0.60,
            confidence_flag_threshold=0.60,
            max_retries=2,
        )
        service._client = MagicMock()
        return service

    @pytest.mark.asyncio
    async def test_successful_first_attempt_calls_gemini_exactly_once(self):
        service = self._make_vision_service_with_mocks()

        valid_response = '{"subject": "fox", "category": "animal", "attributes": ["orange"], "caption": "A fox", "confidence": 0.9}'
        service._call_gemini = AsyncMock(return_value=(valid_response, 100, 20))

        # track_call is an async context manager — patch it to a
        # simple pass-through so we can count _call_gemini invocations
        # without needing a real db-backed cost tracker.
        service.cost_tracker.track_call = MagicMock(return_value=_FakeTrackCallCtx())

        image_id = uuid.uuid4()
        tags = await service.classify_image(image_id, b"fake_bytes")

        assert isinstance(tags, ImageTags)
        assert service._call_gemini.call_count == 1

    @pytest.mark.asyncio
    async def test_invalid_responses_retry_up_to_max_retries_then_fail(self):
        service = self._make_vision_service_with_mocks()

        # Always returns unparseable garbage — should exhaust all
        # retries (max_retries=2 means 3 total attempts: 1 initial + 2 retries).
        service._call_gemini = AsyncMock(return_value=("not valid json at all", 50, 10))
        service.cost_tracker.track_call = MagicMock(return_value=_FakeTrackCallCtx())

        from app.services.vision_service import VisionClassificationFailed

        image_id = uuid.uuid4()
        with pytest.raises(VisionClassificationFailed):
            await service.classify_image(image_id, b"fake_bytes")

        assert service._call_gemini.call_count == 3  # 1 + max_retries(2)

    @pytest.mark.asyncio
    async def test_each_attempt_logs_exactly_one_cost_entry(self):
        """§18 cost-logging rule: every AI call (including every
        retry) results in exactly one ai_call_log row — never zero,
        never more than one per call."""
        service = self._make_vision_service_with_mocks()
        service._call_gemini = AsyncMock(return_value=("garbage", 50, 10))

        track_ctx = _FakeTrackCallCtx()
        service.cost_tracker.track_call = MagicMock(return_value=track_ctx)

        from app.services.vision_service import VisionClassificationFailed

        with pytest.raises(VisionClassificationFailed):
            await service.classify_image(uuid.uuid4(), b"fake_bytes")

        # track_call was entered once per attempt (3 total)
        assert service.cost_tracker.track_call.call_count == 3


class _FakeTrackCallCtx:
    """Minimal async-context-manager stand-in for
    CostTrackerService.track_call, so VisionService's
    `async with self.cost_tracker.track_call(...) as ctx:` works
    without touching a real database."""

    async def __aenter__(self):
        return {"input_tokens": 0, "output_tokens": 0}

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False  # never swallow exceptions