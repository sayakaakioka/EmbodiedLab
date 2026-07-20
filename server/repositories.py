"""Repository interfaces and Firestore adapters for the API service."""

from __future__ import annotations

import hashlib
import uuid
from typing import TYPE_CHECKING, Any

from google.api_core.exceptions import AlreadyExists

from embodiedlab.repositories import (
    ResultQueueWriter,
    ResultReader,
    ResultUpdateWriter,
    SubmissionConflictError,
    SubmissionControlReader,
    SubmissionExecutionWriter,
    SubmissionExistenceChecker,
    SubmissionWriter,
)
from embodiedlab.result_models import (
    Progress,
    ResultBundle,
    ResultStatus,
    build_queued_result_document,
    build_result_update,
)
from embodiedlab.schemas import (
    ScenarioBundle,
    SubmissionControl,
    build_submission_document,
)

if TYPE_CHECKING:
    from google.cloud import firestore


class FirestoreSubmissionRepository(
    SubmissionWriter,
    SubmissionExistenceChecker,
    SubmissionControlReader,
    SubmissionExecutionWriter,
):
    """Firestore-backed submission repository."""

    def __init__(self, db: firestore.Client) -> None:
        """Bind the repository to a Firestore client."""
        self._db = db

    def save(
        self,
        scenario: ScenarioBundle,
        *,
        cancel_token_hash: str,
        idempotency_key: str | None = None,
    ) -> str:
        """Persist a submission, replaying an identical idempotent request."""
        if idempotency_key is None:
            submission_id = str(uuid.uuid4())
            self._db.collection("submissions").document(submission_id).set(
                build_submission_document(
                    submission_id,
                    scenario,
                    cancel_token_hash=cancel_token_hash,
                ),
            )
            return submission_id

        submission_id = self._submission_id_for(idempotency_key)
        document = build_submission_document(
            submission_id,
            scenario,
            cancel_token_hash=cancel_token_hash,
        )
        document_ref = self._db.collection("submissions").document(submission_id)
        try:
            document_ref.create(document)
        except AlreadyExists:
            existing = document_ref.get().to_dict() or {}
            if (
                existing.get("scenario") == scenario.model_dump(mode="json")
                and existing.get("control", {}).get("cancel_token_hash")
                == cancel_token_hash
            ):
                return submission_id
            raise SubmissionConflictError from None
        else:
            return submission_id

    @staticmethod
    def _submission_id_for(idempotency_key: str) -> str:
        digest = hashlib.sha256(
            f"embodiedlab/submissions/v1/{idempotency_key}".encode(),
        ).digest()
        return str(uuid.UUID(bytes=digest[:16]))

    def exists(self, submission_id: str) -> bool:
        """Return whether a submission document exists."""
        submission_snap = (
            self._db.collection("submissions").document(submission_id).get()
        )
        return submission_snap.exists

    def fetch_control(self, submission_id: str) -> SubmissionControl | None:
        """Return private cancellation and execution data for a submission."""
        submission_snap = (
            self._db.collection("submissions").document(submission_id).get()
        )
        if not submission_snap.exists:
            return None

        payload = submission_snap.to_dict() or {}
        control = payload.get("control")
        if control is None:
            return None
        return SubmissionControl.model_validate(control)

    def set_execution_name(self, submission_id: str, execution_name: str) -> None:
        """Store the exact Cloud Run Execution resource name."""
        self._db.collection("submissions").document(submission_id).set(
            {"control": {"execution_name": execution_name}},
            merge=True,
        )


class FirestoreResultRepository(ResultQueueWriter, ResultReader, ResultUpdateWriter):
    """Firestore-backed result repository."""

    def __init__(self, db: firestore.Client) -> None:
        """Bind the repository to a Firestore client."""
        self._db = db

    def create_queued(self, submission_id: str) -> None:
        """Write a queued result document."""
        self._db.collection("results").document(submission_id).set(
            build_queued_result_document(submission_id),
        )

    def write_update(  # noqa: PLR0913
        self,
        submission_id: str,
        *,
        status: ResultStatus,
        progress: Progress,
        summary: dict[str, Any] | None = None,
        error: str | None = None,
        result_bundle: dict[str, Any] | ResultBundle | None = None,
    ) -> None:
        """Merge a lifecycle update into a result document."""
        payload = build_result_update(
            status=status,
            progress=progress,
            summary=summary,
            error=error,
            result_bundle=result_bundle,
        )
        self._db.collection("results").document(submission_id).set(payload, merge=True)

    def fetch(self, submission_id: str) -> dict[str, Any] | None:
        """Return a result document, or None if it does not exist."""
        result_snap = self._db.collection("results").document(submission_id).get()
        if not result_snap.exists:
            return None

        return result_snap.to_dict()
