from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ResultStatus(StrEnum):
	QUEUED = "queued"
	STARTING = "starting"
	RUNNING = "running"
	COMPLETED = "completed"
	FAILED = "failed"


def utc_now_iso() -> str:
	return datetime.now(UTC).isoformat()


class Progress(BaseModel):
	phase: ResultStatus
	current_step: int = Field(ge=0)
	total_steps: int = Field(ge=0)
	message: str


class ResultDocument(BaseModel):
	submission_id: str
	status: ResultStatus
	progress: Progress
	summary: dict[str, Any] | None = None
	error: str | None = None
	artifacts: dict[str, Any] | None = None
	updated_at: str = Field(default_factory=utc_now_iso)


class ResultUpdate(BaseModel):
	status: ResultStatus
	progress: Progress
	summary: dict[str, Any] | None = None
	error: str | None = None
	artifacts: dict[str, Any] | None = None
	updated_at: str = Field(default_factory=utc_now_iso)


def build_queued_result_document(submission_id: str) -> dict:
	document = ResultDocument(
		submission_id=submission_id,
		status=ResultStatus.QUEUED,
		progress=Progress(
			phase=ResultStatus.QUEUED,
			current_step=0,
			total_steps=0,
			message="Queued",
		),
	)
	return document.model_dump(mode="json")


def build_result_update(
	*,
	status: ResultStatus,
	progress: dict | Progress,
	summary=None,
	error=None,
	artifacts=None,
) -> dict:
	update = ResultUpdate(
		status=status,
		progress=progress,
		summary=summary,
		error=error,
		artifacts=artifacts,
	)
	return update.model_dump(mode="json")
