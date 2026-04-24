"""Repository interfaces and Firestore adapters for the API service."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from embodiedlab.repositories import (
    ResultFailureWriter,
    ResultQueueWriter,
    ResultReader,
    SubmissionExistenceChecker,
    SubmissionWriter,
)
from embodiedlab.result_models import (
    Progress,
    build_queued_result_document,
    build_result_update,
)
from embodiedlab.schemas import SubmitRequest, build_submission_document

if TYPE_CHECKING:
    from google.cloud import firestore


class FirestoreSubmissionRepository(SubmissionWriter, SubmissionExistenceChecker):
    """Firestore-backed submission repository."""

    def __init__(self, db: firestore.Client) -> None:
        """Bind the repository to a Firestore client."""
        self._db = db

    def save(self, req: SubmitRequest) -> str:
        """Persist a new submission document and return its generated ID."""
        submission_id = str(uuid.uuid4())
        self._db.collection("submissions").document(submission_id).set(
            build_submission_document(submission_id, req),
        )
        return submission_id

    def exists(self, submission_id: str) -> bool:
        """Return whether a submission document exists."""
        submission_snap = (
            self._db.collection("submissions").document(submission_id).get()
        )
        return submission_snap.exists


class FirestoreResultRepository(ResultQueueWriter, ResultFailureWriter, ResultReader):
    """Firestore-backed result repository."""

    def __init__(self, db: firestore.Client) -> None:
        """Bind the repository to a Firestore client."""
        self._db = db

    def create_queued(self, submission_id: str) -> None:
        """Write a queued result document."""
        self._db.collection("results").document(submission_id).set(
            build_queued_result_document(submission_id),
        )

    def mark_failed(
        self,
        submission_id: str,
        progress: Progress,
        message: str,
    ) -> None:
        """Update a result document to the failed state."""
        payload = build_result_update(
            status=progress.phase,
            progress=progress,
            error=message,
        )
        self._db.collection("results").document(submission_id).set(payload, merge=True)

    def fetch(self, submission_id: str) -> dict[str, Any] | None:
        """Return a result document, or None if it does not exist."""
        result_snap = self._db.collection("results").document(submission_id).get()
        if not result_snap.exists:
            return None

        return result_snap.to_dict()
