"""
§18 "API contracts": each endpoint returns correct status codes and
shapes for valid/invalid input. Uses FastAPI's dependency_overrides to
replace get_db with a fake, so these tests never touch a real
database or make any network call — pure HTTP-contract testing.
"""

import uuid
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

import os

os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("DATABASE_URL", "postgresql://user:pass@localhost/test?sslmode=require")

from app.db.session import get_db  # noqa: E402
from app.main import app  # noqa: E402
from app.services.image_service import ImageNotFoundError  # noqa: E402


@pytest.fixture
def client():
    app.dependency_overrides[get_db] = lambda: AsyncMock()
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


class TestListImages:
    def test_list_images_returns_200_and_array(self, client, monkeypatch):
        from app.services import image_service

        async def fake_list_images(self):
            return []

        monkeypatch.setattr(image_service.ImageService, "list_images", fake_list_images)

        response = client.get("/images")
        assert response.status_code == 200
        assert isinstance(response.json(), list)


class TestGetImage:
    def test_get_image_invalid_uuid_returns_422(self, client):
        response = client.get("/images/not-a-valid-uuid")
        assert response.status_code == 422

    def test_get_image_not_found_returns_404(self, client, monkeypatch):
        from app.services import image_service

        async def fake_get_image(self, image_id):
            raise ImageNotFoundError(image_id)

        monkeypatch.setattr(image_service.ImageService, "get_image", fake_get_image)

        response = client.get(f"/images/{uuid.uuid4()}")
        assert response.status_code == 404


class TestClassifyBatch:
    def test_classify_batch_returns_job_id_and_pending_status(self, client, monkeypatch):
        from app.schemas.api_models import ClassifyBatchResponse
        from app.services import image_service
        from app.routers import images as images_router

        fake_job_id = uuid.uuid4()

        async def fake_start_batch(self, force=False):
            return ClassifyBatchResponse(job_id=fake_job_id, status="pending")

        async def fake_run_job(job_id, force=False):
            # BackgroundTasks run synchronously under TestClient — the
            # real job would try a genuine database connection, which
            # this contract test isn't concerned with (that path is
            # covered by test_idempotency.py and the manual debug
            # scripts). No-op stand-in.
            return None

        monkeypatch.setattr(image_service.ImageService, "start_classification_batch", fake_start_batch)
        monkeypatch.setattr(images_router, "run_classification_job", fake_run_job)

        response = client.post("/images/classify-batch")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "pending"
        assert body["job_id"] == str(fake_job_id)

    def test_classify_batch_accepts_force_query_param(self, client, monkeypatch):
        from app.schemas.api_models import ClassifyBatchResponse
        from app.services import image_service
        from app.routers import images as images_router

        captured = {}

        async def fake_start_batch(self, force=False):
            captured["force"] = force
            return ClassifyBatchResponse(job_id=uuid.uuid4(), status="pending")

        async def fake_run_job(job_id, force=False):
            return None

        monkeypatch.setattr(image_service.ImageService, "start_classification_batch", fake_start_batch)
        monkeypatch.setattr(images_router, "run_classification_job", fake_run_job)

        response = client.post("/images/classify-batch?force=true")
        assert response.status_code == 200
        assert captured["force"] is True