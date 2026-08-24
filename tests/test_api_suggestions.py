"""
§18 "API contracts" for the suggestions/review endpoints.
Same dependency-override approach as test_api_images.py — no real
database, no network calls.
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

import os

os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("DATABASE_URL", "postgresql://user:pass@localhost/test?sslmode=require")

from app.db.session import get_db  # noqa: E402
from app.main import app  # noqa: E402
from app.services.review_service import SuggestionNotFoundError  # noqa: E402


@pytest.fixture
def client():
    app.dependency_overrides[get_db] = lambda: AsyncMock()
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def make_fake_suggestion(**overrides):
    """A lightweight stand-in for an ORM Suggestion row, with only
    the attributes SuggestionOut.model_validate actually reads."""
    from types import SimpleNamespace

    defaults = dict(
        id=uuid.uuid4(),
        post_id=uuid.uuid4(),
        image_id=uuid.uuid4(),
        similarity_score=0.85,
        category_match=True,
        confidence=0.9,
        decision="ACCEPTED",
        reason="Category match; similarity 0.85 >= 0.75; confidence 0.9 >= 0.60",
        source="GUARD",
        review_status="PENDING",
        reviewer_note=None,
        reviewed_at=None,
        created_at=datetime.now(timezone.utc),
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestListSuggestions:
    def test_list_suggestions_returns_200(self, client, monkeypatch):
        from app.repositories import suggestion_repository

        async def fake_list_all(self, decision=None, review_status=None):
            return [make_fake_suggestion()]

        monkeypatch.setattr(suggestion_repository.SuggestionRepository, "list_all", fake_list_all)

        response = client.get("/suggestions")
        assert response.status_code == 200
        body = response.json()
        assert len(body) == 1
        assert body[0]["decision"] == "ACCEPTED"

    def test_list_suggestions_accepts_filters(self, client, monkeypatch):
        from app.repositories import suggestion_repository

        captured = {}

        async def fake_list_all(self, decision=None, review_status=None):
            captured["decision"] = decision
            captured["review_status"] = review_status
            return []

        monkeypatch.setattr(suggestion_repository.SuggestionRepository, "list_all", fake_list_all)

        response = client.get("/suggestions?decision=REJECTED&review_status=PENDING")
        assert response.status_code == 200
        assert captured["decision"] == "REJECTED"
        assert captured["review_status"] == "PENDING"


class TestGetSuggestion:
    def test_get_suggestion_not_found_returns_404(self, client, monkeypatch):
        from app.repositories import suggestion_repository

        async def fake_get_by_id(self, suggestion_id):
            return None

        monkeypatch.setattr(suggestion_repository.SuggestionRepository, "get_by_id", fake_get_by_id)

        response = client.get(f"/suggestions/{uuid.uuid4()}")
        assert response.status_code == 404

    def test_get_suggestion_invalid_uuid_returns_422(self, client):
        response = client.get("/suggestions/not-a-uuid")
        assert response.status_code == 422


class TestApproveReject:
    def test_approve_not_found_returns_404(self, client, monkeypatch):
        from app.services import review_service

        async def fake_approve(self, suggestion_id, note=None):
            raise SuggestionNotFoundError(suggestion_id)

        monkeypatch.setattr(review_service.ReviewService, "approve", fake_approve)

        response = client.post(f"/suggestions/{uuid.uuid4()}/approve")
        assert response.status_code == 404

    def test_approve_success_returns_updated_suggestion(self, client, monkeypatch):
        from app.services import review_service

        fake = make_fake_suggestion(review_status="APPROVED")

        async def fake_approve(self, suggestion_id, note=None):
            return fake

        monkeypatch.setattr(review_service.ReviewService, "approve", fake_approve)

        response = client.post(f"/suggestions/{uuid.uuid4()}/approve", json={"note": "looks good"})
        assert response.status_code == 200
        assert response.json()["review_status"] == "APPROVED"

    def test_reject_success_returns_updated_suggestion(self, client, monkeypatch):
        from app.services import review_service

        fake = make_fake_suggestion(review_status="REJECTED", decision="ACCEPTED")

        async def fake_reject(self, suggestion_id, note=None):
            return fake

        monkeypatch.setattr(review_service.ReviewService, "reject", fake_reject)

        response = client.post(f"/suggestions/{uuid.uuid4()}/reject")
        assert response.status_code == 200
        assert response.json()["review_status"] == "REJECTED"
        # Confirms decision (guard verdict) and review_status (human
        # verdict) are independent — a human can reject a guard ACCEPT.
        assert response.json()["decision"] == "ACCEPTED"


class TestOverride:
    def test_override_post_not_found_returns_404(self, client, monkeypatch):
        from app.services import post_service

        async def fake_get_by_id(self, post_id):
            return None

        monkeypatch.setattr(post_service.PostService, "get_post", fake_get_by_id)
        # Patch the repo lookup used directly inside the override route
        from app.repositories import post_repository

        async def fake_repo_get(self, post_id):
            return None

        monkeypatch.setattr(post_repository.PostRepository, "get_by_id", fake_repo_get)

        response = client.post(
            f"/posts/{uuid.uuid4()}/override",
            json={"image_id": str(uuid.uuid4()), "note": "manual pick"},
        )
        assert response.status_code == 404

    def test_override_missing_image_id_returns_422(self, client):
        response = client.post(f"/posts/{uuid.uuid4()}/override", json={})
        assert response.status_code == 422