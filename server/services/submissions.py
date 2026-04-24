"""Firestore helpers for writing and checking submission documents."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from google.cloud import firestore

from embodiedlab.schemas import SubmitRequest, build_submission_document


def save_submission(db: firestore.Client, req: SubmitRequest) -> str:
    """Persist a new submission document and return its generated ID."""
    submission_id = str(uuid.uuid4())

    db.collection("submissions").document(submission_id).set(
        build_submission_document(submission_id, req),
    )

    return submission_id


def submission_exists(db: firestore.Client, submission_id: str) -> bool:
    """Return True if a submission document with the given ID exists."""
    submission_snap = db.collection("submissions").document(submission_id).get()
    return submission_snap.exists
