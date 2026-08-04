"""Repository interfaces and Firestore adapters for the trainer job."""

from __future__ import annotations

from typing import Any

from google.cloud import firestore

from embodiedlab.repositories import ResultStore, SubmissionReader
from embodiedlab.result_models import (
    Progress,
    ResultBundle,
    ResultStatus,
    build_result_update,
)
from embodiedlab.schemas import DispatchState


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

    def mark_dispatched(self, submission_id: str, execution_name: str) -> bool:
        """Recover the exact execution while dispatch remains unresolved."""
        submission_ref = self._db.collection("submissions").document(submission_id)
        transaction = self._db.transaction()

        @firestore.transactional
        def update(transaction: firestore.Transaction) -> bool:
            snapshot = submission_ref.get(transaction=transaction)
            if not snapshot.exists:
                return False
            submission = snapshot.to_dict() or {}
            control = submission.get("control", {})
            if (
                control.get("dispatch_state") == DispatchState.DISPATCHED.value
                and control.get("execution_name") == execution_name
            ):
                return True
            if control.get("dispatch_state") not in {
                DispatchState.DISPATCHING.value,
                DispatchState.AMBIGUOUS.value,
            }:
                return False
            transaction.set(
                submission_ref,
                {
                    "control": {
                        "dispatch_state": DispatchState.DISPATCHED.value,
                        "dispatch_error": None,
                        "execution_name": execution_name,
                    },
                },
                merge=True,
            )
            return True

        return update(transaction)


class FirestoreResultRepository(ResultStore):
    """Firestore-backed trainer result repository."""

    def __init__(self, db: firestore.Client) -> None:
        """Bind the repository to a Firestore client."""
        self._db = db

    def fetch(self, submission_id: str) -> dict[str, Any] | None:
        """Return the current result document."""
        snapshot = self._db.collection("results").document(submission_id).get()
        if not snapshot.exists:
            return None
        return snapshot.to_dict()

    def transition_if_status(  # noqa: PLR0913
        self,
        submission_id: str,
        *,
        expected_statuses: set[ResultStatus],
        status: ResultStatus,
        progress: Progress,
        error: str | None = None,
        result_bundle: dict[str, Any] | ResultBundle | None = None,
    ) -> dict[str, Any] | None:
        """Apply one trainer transition without reviving terminal state."""
        result_ref = self._db.collection("results").document(submission_id)
        transaction = self._db.transaction()

        @firestore.transactional
        def update(
            transaction: firestore.Transaction,
        ) -> dict[str, Any] | None:
            snapshot = result_ref.get(transaction=transaction)
            if not snapshot.exists:
                return None
            result = snapshot.to_dict() or {}
            try:
                current_status = ResultStatus(result.get("status"))
            except (TypeError, ValueError):
                return None
            if current_status not in expected_statuses:
                return None
            payload = build_result_update(
                status=status,
                progress=progress,
                error=error,
                result_bundle=result_bundle,
            )
            transaction.set(result_ref, payload, merge=True)
            return {**result, **payload}

        return update(transaction)
