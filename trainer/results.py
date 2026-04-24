"""Firestore result document update helper for the trainer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from google.cloud import firestore

from embodiedlab.result_models import Progress, ResultStatus, build_result_update


def update_result(  # noqa: PLR0913
    result_ref: firestore.DocumentReference,
    *,
    status: ResultStatus,
    progress: dict | Progress,
    summary: dict[str, Any] | None = None,
    error: str | None = None,
    artifacts: dict[str, Any] | None = None,
) -> None:
    """Merge a status/progress update into the result document."""
    payload = build_result_update(
        status=status,
        progress=progress,
        summary=summary,
        error=error,
        artifacts=artifacts,
    )
    result_ref.set(payload, merge=True)
