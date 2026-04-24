"""Repository interfaces and Firestore adapters for the trainer job."""

from __future__ import annotations

from typing import Any

from google.cloud import firestore

from embodiedlab.repositories import ResultUpdateWriter, SubmissionReader
from embodiedlab.result_models import Progress, ResultStatus, build_result_update


def create_firestore_client(db_id: str) -> firestore.Client:
    """Create and return a Firestore client for the given database."""
    return firestore.Client(database=db_id)


class FirestoreSubmissionRepository(SubmissionReader):
    """Firestore-backed trainer submission repository."""

    def __init__(self, db: firestore.Client) -> None:
        """Bind the repository to a Firestore client."""
        self._db = db

    def fetch(self, submission_id: str) -> dict[str, Any] | None:
        """Return the submission document dict, or None if it does not exist."""
        submission_snap = (
            self._db.collection("submissions").document(submission_id).get()
        )
        if not submission_snap.exists:
            return None

        return submission_snap.to_dict()


class FirestoreResultRepository(ResultUpdateWriter):
    """Firestore-backed trainer result repository."""

    def __init__(self, db: firestore.Client) -> None:
        """Bind the repository to a Firestore client."""
        self._db = db

    def write_update(  # noqa: PLR0913
        self,
        submission_id: str,
        *,
        status: ResultStatus,
        progress: Progress,
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
        self._db.collection("results").document(submission_id).set(payload, merge=True)
