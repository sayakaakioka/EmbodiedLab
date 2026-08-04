"""Repository interfaces and Firestore adapters for the API service."""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from google.api_core.exceptions import Aborted, AlreadyExists

from embodiedlab.repositories import (
    ResultProgressTransitionWriter,
    ResultReader,
    ResultTransitionWriter,
    SubmissionAcceptanceWriter,
    SubmissionCancellationWriter,
    SubmissionConflictError,
    SubmissionControlReader,
    SubmissionDispatchWriter,
    SubmissionRecoveryError,
)
from embodiedlab.result_models import (
    Progress,
    ResultBundle,
    ResultStatus,
    build_queued_result_document,
    build_result_update,
)
from embodiedlab.schemas import (
    CancellationState,
    DispatchState,
    ScenarioBundle,
    SubmissionControl,
    build_submission_document,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from google.cloud import firestore

TRANSACTION_ATTEMPTS = 3
CANCELLABLE_RESULT_STATUSES = {
    ResultStatus.QUEUED,
    ResultStatus.STARTING,
    ResultStatus.RUNNING,
    ResultStatus.CANCELLING,
}


class FirestoreSubmissionRepository(
    SubmissionAcceptanceWriter,
    SubmissionControlReader,
    SubmissionDispatchWriter,
    SubmissionCancellationWriter,
):
    """Firestore-backed submission repository."""

    def __init__(self, db: firestore.Client) -> None:
        """Bind the repository to a Firestore client."""
        self._db = db

    def accept(
        self,
        scenario: ScenarioBundle,
        *,
        cancel_token_hash: str,
        total_steps: int,
        idempotency_key: str,
    ) -> str:
        """Atomically persist a submission and its queued result."""
        submission_id = self._submission_id_for(idempotency_key)
        document = build_submission_document(
            submission_id,
            scenario,
            cancel_token_hash=cancel_token_hash,
        )
        document_ref = self._db.collection("submissions").document(submission_id)
        result_ref = self._db.collection("results").document(submission_id)
        batch = self._db.batch()
        batch.create(document_ref, document)
        batch.create(
            result_ref,
            build_queued_result_document(
                submission_id,
                total_steps=total_steps,
            ),
        )
        try:
            batch.commit()
        except AlreadyExists:
            existing = document_ref.get().to_dict() or {}
            control = existing.get("control")
            if not isinstance(control, dict):
                raise SubmissionRecoveryError from None
            if (
                existing.get("scenario") != scenario.model_dump(mode="json")
                or control.get("cancel_token_hash") != cancel_token_hash
            ):
                raise SubmissionConflictError from None
            result_snapshot = result_ref.get()
            result = result_snapshot.to_dict() if result_snapshot.exists else None
            if (
                _parse_dispatch_state(control.get("dispatch_state")) is None
                or not isinstance(result, dict)
                or _parse_result_status(result.get("status")) is None
            ):
                raise SubmissionRecoveryError from None
            return submission_id
        else:
            return submission_id

    @staticmethod
    def _submission_id_for(idempotency_key: str) -> str:
        digest = hashlib.sha256(
            f"embodiedlab/submissions/v1/{idempotency_key}".encode(),
        ).digest()
        return str(uuid.UUID(bytes=digest[:16]))

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

    def claim_dispatch(self, submission_id: str) -> bool:
        """Atomically claim a submission for one dispatch attempt."""
        started_at = datetime.now(UTC)
        return self._transition_pending_dispatch(
            submission_id,
            {
                "dispatch_state": DispatchState.DISPATCHING.value,
                "dispatch_started_at": started_at.isoformat(),
                "dispatch_error": None,
            },
        )

    def cancel_pending_dispatch(
        self,
        submission_id: str,
        *,
        progress: Progress,
    ) -> dict[str, Any] | None:
        """Atomically cancel a pending dispatch and its queued result."""
        submission_ref = self._db.collection("submissions").document(submission_id)
        result_ref = self._db.collection("results").document(submission_id)

        def attempt() -> dict[str, Any] | None:
            transaction = self._db.transaction()
            submission_snapshot = submission_ref.get(transaction=transaction)
            result_snapshot = result_ref.get(transaction=transaction)
            if not submission_snapshot.exists or not result_snapshot.exists:
                return None
            submission = submission_snapshot.to_dict() or {}
            result = result_snapshot.to_dict() or {}
            control = submission.get("control", {})
            if (
                control.get("dispatch_state") != DispatchState.PENDING.value
                or result.get("status") != ResultStatus.QUEUED.value
            ):
                return None

            transaction.set(
                submission_ref,
                {"control": {"dispatch_state": DispatchState.CANCELLED.value}},
                merge=True,
            )
            update = build_result_update(
                status=ResultStatus.CANCELLED,
                progress=progress,
            )
            transaction.set(result_ref, update, merge=True)
            transaction.commit()
            return {**result, **update}

        return _retry_aborted(attempt)

    def _transition_pending_dispatch(
        self,
        submission_id: str,
        control_update: dict[str, Any],
    ) -> bool:
        submission_ref = self._db.collection("submissions").document(submission_id)
        result_ref = self._db.collection("results").document(submission_id)

        def attempt() -> bool:
            transaction = self._db.transaction()
            submission_snapshot = submission_ref.get(transaction=transaction)
            result_snapshot = result_ref.get(transaction=transaction)
            if not submission_snapshot.exists or not result_snapshot.exists:
                return False
            submission = submission_snapshot.to_dict() or {}
            result = result_snapshot.to_dict() or {}
            control = submission.get("control", {})
            if (
                control.get("dispatch_state") != DispatchState.PENDING.value
                or control.get("execution_name") is not None
                or result.get("status") != ResultStatus.QUEUED.value
            ):
                return False

            transaction.set(
                submission_ref,
                {"control": control_update},
                merge=True,
            )
            transaction.commit()
            return True

        return _retry_aborted(attempt)

    def mark_dispatched(self, submission_id: str, execution_name: str) -> bool:
        """Store the exact execution without reviving a closed dispatch."""
        return self._transition_dispatch_state(
            submission_id,
            expected_states={DispatchState.DISPATCHING, DispatchState.AMBIGUOUS},
            control_update={
                "dispatch_state": DispatchState.DISPATCHED.value,
                "dispatch_error": None,
                "execution_name": execution_name,
            },
        )

    def mark_dispatch_ambiguous(self, submission_id: str, message: str) -> bool:
        """Store an unresolved dispatch without reviving a closed dispatch."""
        return self._transition_dispatch_state(
            submission_id,
            expected_states={DispatchState.DISPATCHING},
            control_update={
                "dispatch_state": DispatchState.AMBIGUOUS.value,
                "dispatch_started_at": datetime.now(UTC).isoformat(),
                "dispatch_error": message,
            },
        )

    def _transition_dispatch_state(
        self,
        submission_id: str,
        *,
        expected_states: set[DispatchState],
        control_update: dict[str, Any],
    ) -> bool:
        submission_ref = self._db.collection("submissions").document(submission_id)

        def attempt() -> bool:
            transaction = self._db.transaction()
            snapshot = submission_ref.get(transaction=transaction)
            if not snapshot.exists:
                return False
            submission = snapshot.to_dict() or {}
            control = submission.get("control", {})
            current_state = _parse_dispatch_state(control.get("dispatch_state"))
            if all(control.get(key) == value for key, value in control_update.items()):
                return True
            if current_state not in expected_states:
                return False
            transaction.set(
                submission_ref,
                {"control": control_update},
                merge=True,
            )
            transaction.commit()
            return True

        return _retry_aborted(attempt)

    def fail_dispatch_if_queued(
        self,
        submission_id: str,
        *,
        progress: Progress,
        error: str,
    ) -> bool:
        """Atomically fail dispatch control while its result is still queued."""
        submission_ref = self._db.collection("submissions").document(submission_id)
        result_ref = self._db.collection("results").document(submission_id)

        def attempt() -> bool:
            transaction = self._db.transaction()
            submission_snapshot = submission_ref.get(transaction=transaction)
            result_snapshot = result_ref.get(transaction=transaction)
            if not submission_snapshot.exists or not result_snapshot.exists:
                return False
            submission = submission_snapshot.to_dict() or {}
            result = result_snapshot.to_dict() or {}
            control = submission.get("control", {})
            if (
                control.get("dispatch_state")
                not in {
                    DispatchState.DISPATCHING.value,
                    DispatchState.AMBIGUOUS.value,
                }
                or result.get("status") != ResultStatus.QUEUED.value
            ):
                return False

            transaction.set(
                submission_ref,
                {
                    "control": {
                        "dispatch_state": DispatchState.FAILED.value,
                        "dispatch_error": error,
                    },
                },
                merge=True,
            )
            transaction.set(
                result_ref,
                build_result_update(
                    status=ResultStatus.FAILED,
                    progress=progress,
                    error=error,
                ),
                merge=True,
            )
            transaction.commit()
            return True

        return _retry_aborted(attempt)

    def claim_cancellation(
        self,
        submission_id: str,
        *,
        claimed_at: datetime,
        stale_before: datetime,
    ) -> str | None:
        """Claim or reclaim the durable cancellation intent lease."""
        submission_ref = self._db.collection("submissions").document(submission_id)
        result_ref = self._db.collection("results").document(submission_id)
        lease_token = uuid.uuid4().hex

        def attempt() -> str | None:
            transaction = self._db.transaction()
            submission_snapshot = submission_ref.get(transaction=transaction)
            result_snapshot = result_ref.get(transaction=transaction)
            if not submission_snapshot.exists or not result_snapshot.exists:
                return None
            submission = submission_snapshot.to_dict() or {}
            result = result_snapshot.to_dict() or {}
            control = submission.get("control", {})
            state = _parse_cancellation_state(control.get("cancellation_state"))
            started_at = _parse_datetime(control.get("cancellation_started_at"))
            if (
                state is None
                or state is CancellationState.REQUESTED
                or control.get("execution_name") is None
                or _parse_result_status(result.get("status"))
                not in CANCELLABLE_RESULT_STATUSES
                or (
                    state is CancellationState.REQUESTING
                    and started_at is not None
                    and started_at > stale_before
                )
            ):
                return None
            transaction.set(
                submission_ref,
                {
                    "control": {
                        "cancellation_state": CancellationState.REQUESTING.value,
                        "cancellation_started_at": claimed_at.isoformat(),
                        "cancellation_lease_token": lease_token,
                        "cancellation_error": None,
                    },
                },
                merge=True,
            )
            transaction.commit()
            return lease_token

        return _retry_aborted(attempt)

    def mark_cancellation_requested(
        self,
        submission_id: str,
        lease_token: str,
    ) -> bool:
        """Record that the exact execution cancellation RPC was attempted."""
        return self._transition_cancellation_state(
            submission_id,
            lease_token=lease_token,
            expected_states={CancellationState.REQUESTING},
            control_update={
                "cancellation_state": CancellationState.REQUESTED.value,
                "cancellation_error": None,
            },
        )

    def release_cancellation(
        self,
        submission_id: str,
        lease_token: str,
        error: str,
    ) -> bool:
        """Release a cancellation lease after a definitive rejection."""
        return self._transition_cancellation_state(
            submission_id,
            lease_token=lease_token,
            expected_states={
                CancellationState.REQUESTING,
                CancellationState.REQUESTED,
            },
            control_update={
                "cancellation_state": CancellationState.IDLE.value,
                "cancellation_started_at": None,
                "cancellation_lease_token": None,
                "cancellation_error": error,
            },
        )

    def _transition_cancellation_state(
        self,
        submission_id: str,
        *,
        lease_token: str,
        expected_states: set[CancellationState],
        control_update: dict[str, Any],
    ) -> bool:
        submission_ref = self._db.collection("submissions").document(submission_id)

        def attempt() -> bool:
            transaction = self._db.transaction()
            snapshot = submission_ref.get(transaction=transaction)
            if not snapshot.exists:
                return False
            submission = snapshot.to_dict() or {}
            control = submission.get("control", {})
            state = _parse_cancellation_state(control.get("cancellation_state"))
            if (
                state not in expected_states
                or control.get("cancellation_lease_token") != lease_token
            ):
                return False
            transaction.set(
                submission_ref,
                {"control": control_update},
                merge=True,
            )
            transaction.commit()
            return True

        return _retry_aborted(attempt)


class FirestoreResultRepository(
    ResultReader,
    ResultTransitionWriter,
    ResultProgressTransitionWriter,
):
    """Firestore-backed result repository."""

    def __init__(self, db: firestore.Client) -> None:
        """Bind the repository to a Firestore client."""
        self._db = db

    def fetch(self, submission_id: str) -> dict[str, Any] | None:
        """Return a result document, or None if it does not exist."""
        result_snap = self._db.collection("results").document(submission_id).get()
        if not result_snap.exists:
            return None

        return result_snap.to_dict()

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
        """Apply an API-owned transition only while the result is active."""
        result_ref = self._db.collection("results").document(submission_id)

        def attempt() -> dict[str, Any] | None:
            transaction = self._db.transaction()
            snapshot = result_ref.get(transaction=transaction)
            if not snapshot.exists:
                return None
            result = snapshot.to_dict() or {}
            current_status = _parse_result_status(result.get("status"))
            if current_status not in expected_statuses:
                return None
            update = build_result_update(
                status=status,
                progress=progress,
                error=error,
                result_bundle=result_bundle,
            )
            transaction.set(result_ref, update, merge=True)
            transaction.commit()
            return {**result, **update}

        return _retry_aborted(attempt)

    def transition_status_preserving_progress(
        self,
        submission_id: str,
        *,
        expected_statuses: set[ResultStatus],
        status: ResultStatus,
        message: str,
        error: str | None = None,
    ) -> dict[str, Any] | None:
        """Change status atomically without rolling progress counters back."""
        result_ref = self._db.collection("results").document(submission_id)

        def attempt() -> dict[str, Any] | None:
            transaction = self._db.transaction()
            snapshot = result_ref.get(transaction=transaction)
            if not snapshot.exists:
                return None
            result = snapshot.to_dict() or {}
            current_status = _parse_result_status(result.get("status"))
            if current_status not in expected_statuses:
                return None
            current_progress = Progress.model_validate(result.get("progress"))
            progress = Progress(
                phase=status,
                current_step=current_progress.current_step,
                total_steps=current_progress.total_steps,
                message=message,
            )
            update = build_result_update(
                status=status,
                progress=progress,
                error=error,
            )
            transaction.set(result_ref, update, merge=True)
            transaction.commit()
            return {**result, **update}

        return _retry_aborted(attempt)


def _parse_result_status(value: object) -> ResultStatus | None:
    try:
        return ResultStatus(value)
    except (TypeError, ValueError):
        return None


def _parse_dispatch_state(value: object) -> DispatchState | None:
    try:
        return DispatchState(value)
    except (TypeError, ValueError):
        return None


def _parse_cancellation_state(value: object) -> CancellationState | None:
    if value is None:
        return CancellationState.IDLE
    try:
        return CancellationState(value)
    except (TypeError, ValueError):
        return None


def _parse_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def _retry_aborted[T](operation: Callable[[], T]) -> T:
    """Retry a Firestore transaction without hiding retry exhaustion."""
    for attempt in range(1, TRANSACTION_ATTEMPTS + 1):
        try:
            return operation()
        except Aborted:
            if attempt == TRANSACTION_ATTEMPTS:
                raise
    raise RuntimeError
