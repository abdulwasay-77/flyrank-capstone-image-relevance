"""Budget circuit-breaker tests; all database and AI boundaries are mocked."""

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.db.models import BatchJob
from app.services.cost_tracker_service import BudgetExceededError, CostTrackerService


class TestCostBudgetCheck:
    @pytest.mark.asyncio
    async def test_budget_check_allows_calls_under_cap(self):
        tracker = CostTrackerService.__new__(CostTrackerService)
        tracker.settings = SimpleNamespace(MAX_BUDGET_USD=1.00)
        tracker.get_summary = AsyncMock(return_value={"total_cost_usd": 0.999999})
        await tracker.check_budget_ok()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("spend", [1.00, 1.25])
    async def test_budget_check_rejects_calls_at_or_over_cap(self, spend):
        tracker = CostTrackerService.__new__(CostTrackerService)
        tracker.settings = SimpleNamespace(MAX_BUDGET_USD=1.00)
        tracker.get_summary = AsyncMock(return_value={"total_cost_usd": spend})

        with pytest.raises(BudgetExceededError) as caught:
            await tracker.check_budget_ok()

        assert caught.value.limit_usd == 1.00
        assert caught.value.current_spend_usd == spend


class _FakeSessionContext:
    def __init__(self, db):
        self.db = db

    async def __aenter__(self):
        return self.db

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.mark.asyncio
async def test_budget_exhaustion_marks_batch_job_failed(monkeypatch):
    """A budget stop is visible to GET /jobs rather than swallowed as a worker error."""
    from app.jobs import classification_job

    job = MagicMock(spec=BatchJob)
    job_id = uuid.uuid4()
    images = [SimpleNamespace(id=uuid.uuid4(), filename="one.jpg")]
    first_db = AsyncMock()
    first_db.get.return_value = job
    second_db = AsyncMock()
    second_db.get.return_value = job

    monkeypatch.setattr(
        classification_job,
        "AsyncSessionLocal",
        MagicMock(side_effect=[_FakeSessionContext(first_db), _FakeSessionContext(second_db)]),
    )
    monkeypatch.setattr(classification_job.ImageRepository, "list_unclassified", AsyncMock(return_value=images))

    async def fake_classify_one(image_id, filename, semaphore, budget_exhausted):
        budget_exhausted.set()
        return "budget_exceeded"

    monkeypatch.setattr(classification_job, "_classify_one", fake_classify_one)
    await classification_job.run_classification_job(job_id)

    assert job.status == "failed"
    assert job.succeeded_items == 0
    assert job.failed_items == 0
    second_db.commit.assert_awaited()


def test_suggestions_returns_503_when_budget_is_exhausted(monkeypatch):
    from app.db.session import get_db
    from app.main import app
    from app.services import post_service

    async def fake_suggestions(self, post_id):
        raise BudgetExceededError(limit_usd=1.00, current_spend_usd=1.00)

    monkeypatch.setattr(post_service.PostService, "get_or_compute_suggestions", fake_suggestions)
    app.dependency_overrides[get_db] = lambda: AsyncMock()
    try:
        with TestClient(app) as client:
            response = client.get(f"/posts/{uuid.uuid4()}/suggestions")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert "Configured limit: $1.00" in response.json()["detail"]
    assert "current spend: $1.000000" in response.json()["detail"]
