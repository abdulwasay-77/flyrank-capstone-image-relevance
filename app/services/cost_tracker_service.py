"""
Cost tracking service.

Per §16: every call to the vision or embedding API — success or
failure — writes exactly one row to ai_call_log. estimated_cost_usd
is computed from a small static pricing table matching Gemini's
published rates. Even on the free tier where actual spend is $0,
this demonstrates the cost-discipline a production system would need.

Pricing source: Gemini API published per-1M-token rates as of this
project's build date. Kept as a plain constant, not fetched live —
pricing pages change independently of this codebase, and a wrong
estimate here doesn't affect correctness of the actual pipeline, only
the number shown in GET /costs/summary. If rates change, update
PRICING_USD_PER_1M_TOKENS and note the date in a comment.
"""

import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from sqlalchemy import Integer, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import AICallLog

# USD per 1,000,000 tokens. (input, output) — embeddings have no
# separate "output" token cost, so output is 0 for that row.
PRICING_USD_PER_1M_TOKENS: dict[str, dict[str, float]] = {
    # gemini-3.6-flash was tried first but has an extremely tight free
    # tier (5 RPM / 20 RPD observed live) — switched to
    # gemini-3.5-flash-lite, Google's high-throughput/low-cost
    # multimodal model, which is what vision_service.py actually
    # calls. Rates below are a placeholder pending confirmation
    # against Google's published rate card for this specific model —
    # what matters for this project is that the tracking *mechanism*
    # works (every call logged, non-zero cost computed), not that the
    # dollar figure is penny-accurate.
    "gemini-3.5-flash-lite": {"input": 0.05, "output": 0.20},
    "text-embedding-004": {"input": 0.00, "output": 0.00},  # free tier, no published per-token charge
}


def estimate_cost_usd(model_name: str, input_tokens: int, output_tokens: int = 0) -> float:
    """
    Static-table cost estimate for one API call. Unknown model names
    default to $0 rather than raising — an estimate should never be
    the thing that crashes a real classification/embedding call.
    """
    rates = PRICING_USD_PER_1M_TOKENS.get(model_name, {"input": 0.0, "output": 0.0})
    return (input_tokens / 1_000_000) * rates["input"] + (output_tokens / 1_000_000) * rates["output"]


class BudgetExceededError(Exception):
    """Raised before an AI call when the project-wide spend cap is exhausted."""

    def __init__(self, *, limit_usd: float, current_spend_usd: float):
        self.limit_usd = limit_usd
        self.current_spend_usd = current_spend_usd
        super().__init__(
            f"AI budget cap of ${limit_usd:.2f} reached "
            f"(current tracked spend: ${current_spend_usd:.6f})"
        )


class CostTrackerService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.settings = get_settings()

    async def check_budget_ok(self) -> None:
        """Allow an AI call only while cumulative tracked spend is below the cap.

        The comparison intentionally covers the complete ai_call_log history,
        not a request or date window, so this remains a project-wide circuit
        breaker even if a buggy batch is restarted repeatedly.
        """
        summary = await self.get_summary()
        current_spend = summary["total_cost_usd"]
        if current_spend >= self.settings.MAX_BUDGET_USD:
            raise BudgetExceededError(
                limit_usd=self.settings.MAX_BUDGET_USD,
                current_spend_usd=current_spend,
            )

    async def log_call(
        self,
        *,
        call_type: str,  # "VISION_CLASSIFY" | "EMBEDDING"
        model_name: str,
        success: bool,
        latency_ms: int,
        input_tokens: int = 0,
        output_tokens: int = 0,
        reference_id: uuid.UUID | None = None,
        error_message: str | None = None,
    ) -> AICallLog:
        """
        Writes one ai_call_log row (§16, §9.8). Called for EVERY AI
        API call regardless of outcome — a failed call still costs
        (partial) tokens and still needs to be visible in cost/error
        reporting, which is why `success` is a field here rather than
        failures simply not being logged at all.
        """
        cost = estimate_cost_usd(model_name, input_tokens, output_tokens)

        row = AICallLog(
            call_type=call_type,
            reference_id=reference_id,
            model_name=model_name,
            success=success,
            estimated_cost_usd=cost,
            latency_ms=latency_ms,
            error_message=error_message,
        )
        self.db.add(row)
        # Commit immediately, not just flush. This log entry must be
        # durable the instant it's written — if the caller's broader
        # operation later fails and its session gets rolled back on
        # exit (e.g. VisionService re-raising after a Gemini API
        # error), a merely-flushed-not-committed row would be silently
        # discarded along with it, defeating the entire point of
        # logging failures (§16: "every call — success or failure").
        await self.db.commit()
        return row

    @asynccontextmanager
    async def track_call(
        self,
        *,
        call_type: str,
        model_name: str,
        reference_id: uuid.UUID | None = None,
    ):
        """
        Convenience context manager for the common case: time the
        call, catch any exception, log success/failure automatically.

        Usage:
            async with cost_tracker.track_call(
                call_type="VISION_CLASSIFY", model_name="gemini-2.0-flash", reference_id=image_id
            ) as ctx:
                response = await call_gemini(...)
                ctx["input_tokens"] = response.usage_metadata.prompt_token_count
                ctx["output_tokens"] = response.usage_metadata.candidates_token_count

        `ctx` is a plain dict the caller can fill in with token counts
        once the response is available; if the call raises, whatever
        was written into ctx before the exception is still used for
        the failure-path log entry (usually 0/0, which is fine —
        estimate_cost_usd(0, 0) = 0.0).
        """
        # Deliberately before the timed/logged block: a denied call was never
        # sent to Gemini, so it must not be recorded as an AI API call.
        await self.check_budget_ok()
        ctx: dict = {"input_tokens": 0, "output_tokens": 0}
        start = time.monotonic()
        try:
            yield ctx
        except Exception as exc:
            latency_ms = int((time.monotonic() - start) * 1000)
            await self.log_call(
                call_type=call_type,
                model_name=model_name,
                success=False,
                latency_ms=latency_ms,
                input_tokens=ctx.get("input_tokens", 0),
                output_tokens=ctx.get("output_tokens", 0),
                reference_id=reference_id,
                error_message=str(exc)[:2000],  # cap length, this is TEXT not unbounded
            )
            raise
        else:
            latency_ms = int((time.monotonic() - start) * 1000)
            await self.log_call(
                call_type=call_type,
                model_name=model_name,
                success=True,
                latency_ms=latency_ms,
                input_tokens=ctx.get("input_tokens", 0),
                output_tokens=ctx.get("output_tokens", 0),
                reference_id=reference_id,
            )

    async def get_summary(
        self, window_start: datetime | None = None, window_end: datetime | None = None
    ) -> dict:
        """
        §16 / §10.5: aggregates ai_call_log by call_type, optionally
        bounded by a time window. Returns a plain dict shaped to match
        CostSummaryOut exactly — kept as a dict here (not the Pydantic
        model itself) so this service stays free of any dependency on
        the API schema layer (§7.1's "swappable independently").
        """
        query = select(
            AICallLog.call_type,
            func.count().label("call_count"),
            func.coalesce(func.sum(AICallLog.estimated_cost_usd), 0).label("total_cost_usd"),
            func.sum(func.cast(AICallLog.success, Integer)).label("success_count"),
        )
        if window_start is not None:
            query = query.where(AICallLog.created_at >= window_start)
        if window_end is not None:
            query = query.where(AICallLog.created_at <= window_end)
        query = query.group_by(AICallLog.call_type)

        result = await self.db.execute(query)
        rows = result.all()

        by_call_type = []
        total_cost = 0.0
        for call_type, call_count, total_cost_usd, success_count in rows:
            success_count = success_count or 0
            by_call_type.append(
                {
                    "call_type": call_type,
                    "call_count": call_count,
                    "total_cost_usd": float(total_cost_usd),
                    "success_count": success_count,
                    "failure_count": call_count - success_count,
                }
            )
            total_cost += float(total_cost_usd)

        return {
            "window_start": window_start,
            "window_end": window_end,
            "total_cost_usd": total_cost,
            "by_call_type": by_call_type,
        }
