"""
Mismatch Guard threshold configuration.

Spec §13.3. These four numbers are the tunable knobs of the whole
matching pipeline's strictness — they get tuned against the eval set
in Phase 4 (§17) and must always be loaded from environment/config,
never hardcoded inside guard_service.py's decision logic (FR-3.5).

This class intentionally mirrors app.config.Settings' four threshold
fields exactly. Settings is the source of truth (it reads .env);
GuardConfig is the shape guard_service.py actually consumes, built
from Settings via `GuardConfig.from_settings()` below — this indirection
keeps the guard's decision logic decoupled from *how* config is loaded
(env vars today, could be a config file or DB row later without
touching guard_service.py).
"""

from pydantic import BaseModel, Field


class GuardConfig(BaseModel):
    """Threshold configuration consumed by GuardService's decision flow (§13.1)."""

    similarity_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    confidence_accept_threshold: float = Field(default=0.60, ge=0.0, le=1.0)
    confidence_flag_threshold: float = Field(default=0.60, ge=0.0, le=1.0)
    max_retries: int = Field(default=3, ge=0)

    @classmethod
    def from_settings(cls) -> "GuardConfig":
        """Build a GuardConfig from the app's central Settings (app/config.py)."""
        from app.config import get_settings

        s = get_settings()
        return cls(
            similarity_threshold=s.SIMILARITY_THRESHOLD,
            confidence_accept_threshold=s.CONFIDENCE_ACCEPT_THRESHOLD,
            confidence_flag_threshold=s.CONFIDENCE_FLAG_THRESHOLD,
            max_retries=s.MAX_RETRIES,
        )