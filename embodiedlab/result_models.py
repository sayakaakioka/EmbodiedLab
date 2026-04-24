"""Result status, progress, and document models shared by the API and trainer."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ResultStatus(StrEnum):
    """Lifecycle states of a training result."""

    QUEUED = "queued"
    STARTING = "starting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(UTC).isoformat()


class Progress(BaseModel):
    """Training progress snapshot stored in each result document."""

    phase: ResultStatus
    current_step: int = Field(ge=0)
    total_steps: int = Field(ge=0)
    message: str


class ResultDocument(BaseModel):
    """Full result document written to Firestore."""

    submission_id: str
    status: ResultStatus
    progress: Progress
    summary: dict[str, Any] | None = None
    error: str | None = None
    artifacts: dict[str, Any] | None = None
    updated_at: str = Field(default_factory=utc_now_iso)


class ResultMessage(BaseModel):
    """Pub/Sub message payload emitted after each status transition."""

    submission_id: str
    status: ResultStatus
    progress: Progress | None = None
    summary: dict[str, Any] | None = None
    error: str | None = None
    artifacts: dict[str, Any] | None = None
    updated_at: str = Field(default_factory=utc_now_iso)


class ResultUpdate(BaseModel):
    """Partial update applied to an existing result document."""

    status: ResultStatus
    progress: Progress
    summary: dict[str, Any] | None = None
    error: str | None = None
    artifacts: dict[str, Any] | None = None
    updated_at: str = Field(default_factory=utc_now_iso)


def build_progress(
    *,
    phase: ResultStatus,
    current_step: int,
    total_steps: int,
    message: str,
) -> Progress:
    """Build a Progress model from explicit values."""
    return Progress(
        phase=phase,
        current_step=current_step,
        total_steps=total_steps,
        message=message,
    )


def queued_progress() -> Progress:
    """Return the queued-phase progress payload."""
    return build_progress(
        phase=ResultStatus.QUEUED,
        current_step=0,
        total_steps=0,
        message="Queued",
    )


def starting_progress(total_steps: int) -> Progress:
    """Return the starting-phase progress payload."""
    return build_progress(
        phase=ResultStatus.STARTING,
        current_step=0,
        total_steps=total_steps,
        message="Trainer job started",
    )


def running_progress(total_steps: int) -> Progress:
    """Return the running-phase progress payload."""
    return build_progress(
        phase=ResultStatus.RUNNING,
        current_step=0,
        total_steps=total_steps,
        message="Training",
    )


def completed_progress(total_steps: int) -> Progress:
    """Return the completed-phase progress payload."""
    return build_progress(
        phase=ResultStatus.COMPLETED,
        current_step=total_steps,
        total_steps=total_steps,
        message="Training completed",
    )


def failed_progress(message: str, total_steps: int = 0) -> Progress:
    """Return the failed-phase progress payload."""
    return build_progress(
        phase=ResultStatus.FAILED,
        current_step=0,
        total_steps=total_steps,
        message=message,
    )


def build_queued_result_document(submission_id: str) -> dict:
    """Return a Firestore-ready dict for a newly queued result."""
    document = ResultDocument(
        submission_id=submission_id,
        status=ResultStatus.QUEUED,
        progress=queued_progress(),
    )
    return document.model_dump(mode="json")


def build_result_update(
    *,
    status: ResultStatus,
    progress: dict | Progress,
    summary: dict[str, Any] | None = None,
    error: str | None = None,
    artifacts: dict[str, Any] | None = None,
) -> dict:
    """Return a Firestore-ready dict for a partial result update."""
    update = ResultUpdate(
        status=status,
        progress=progress,
        summary=summary,
        error=error,
        artifacts=artifacts,
    )
    return update.model_dump(mode="json")


def build_result_message(  # noqa: PLR0913
    submission_id: str,
    status: ResultStatus,
    progress: Progress,
    summary: dict[str, Any] | None = None,
    error: str | None = None,
    artifacts: dict[str, Any] | None = None,
) -> dict:
    """Return a dict suitable for publishing as a Pub/Sub message."""
    message = ResultMessage(
        submission_id=submission_id,
        status=status,
        progress=progress,
        summary=summary,
        error=error,
        artifacts=artifacts,
    )
    return message.model_dump(mode="json")


def parse_result_message(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a result event payload and return its normalized JSON form."""
    message = ResultMessage.model_validate(payload)
    return message.model_dump(mode="json", exclude_unset=True)
