"""Firestore helpers for reading and writing result documents."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from google.cloud import firestore

from embodiedlab.result_models import (
    Progress,
    build_queued_result_document,
    build_result_update,
)


def create_queued_result(db: firestore.Client, submission_id: str) -> None:
    """Write a new result document in the queued state."""
    db.collection("results").document(submission_id).set(
        build_queued_result_document(submission_id),
    )


def mark_result_failed(
    db: firestore.Client,
    submission_id: str,
    progress: Progress,
    message: str,
) -> None:
    """Update an existing result document to the failed state."""
    payload = build_result_update(
        status=progress.phase,
        progress=progress,
        error=message,
    )
    db.collection("results").document(submission_id).set(payload, merge=True)


def fetch_result(db: firestore.Client, submission_id: str) -> dict[str, Any] | None:
    """Return the result document dict, or None if it does not exist."""
    result_snap = db.collection("results").document(submission_id).get()
    if not result_snap.exists:
        return None

    return result_snap.to_dict()
