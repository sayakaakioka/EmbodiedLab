import uuid
from copy import deepcopy

from google.api_core.exceptions import AlreadyExists

from embodiedlab.repositories import SubmissionConflictError, SubmissionRecoveryError
from embodiedlab.result_models import ResultBundle, ResultStatus, build_result_update
from embodiedlab.schemas import (
    CancellationState,
    DispatchState,
    ScenarioBundle,
    SubmissionControl,
    build_submission_document,
)


def merge_dicts(existing: dict, update: dict) -> dict:
    """Recursively merge Firestore-style payloads into an existing document."""
    merged = deepcopy(existing)
    for key, value in update.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)

    return merged


class FakeSnapshot:
    def __init__(self, data):
        self._data = deepcopy(data)
        self.exists = data is not None

    def to_dict(self):
        return deepcopy(self._data)


class FakeDocument:
    def __init__(self, store: dict, document_id: str):
        self.store = store
        self.document_id = document_id
        self.payloads = []

    def get(self, transaction=None):
        return FakeSnapshot(self.store.get(self.document_id))

    def create(self, data):
        if self.document_id in self.store:
            raise AlreadyExists(self.document_id)

        self.payloads.append({"data": deepcopy(data), "merge": False})
        self.store[self.document_id] = deepcopy(data)

    def set(self, data, merge: bool = False):
        self.payloads.append({"data": deepcopy(data), "merge": merge})
        if merge and self.document_id in self.store:
            self.store[self.document_id] = merge_dicts(
                self.store[self.document_id],
                data,
            )
        else:
            self.store[self.document_id] = deepcopy(data)


class FakeCollection:
    def __init__(self, store: dict):
        self.store = store
        self.documents = {}

    def document(self, document_id: str):
        if document_id not in self.documents:
            self.documents[document_id] = FakeDocument(self.store, document_id)

        return self.documents[document_id]


class FakeBatch:
    def __init__(self):
        self.operations = []

    def create(self, document_ref, data):
        self.operations.append(("create", document_ref, deepcopy(data), False))

    def set(self, document_ref, data, merge: bool = False):
        self.operations.append(("set", document_ref, deepcopy(data), merge))

    def commit(self):
        for operation, document_ref, _data, _merge in self.operations:
            if operation == "create" and document_ref.document_id in document_ref.store:
                raise AlreadyExists(document_ref.document_id)
        for operation, document_ref, data, merge in self.operations:
            if operation == "create":
                document_ref.create(data)
            else:
                document_ref.set(data, merge=merge)


class FakeTransaction:
    def __init__(self):
        self.operations = []

    def set(self, document_ref, data, merge: bool = False):
        self.operations.append((document_ref, deepcopy(data), merge))

    def commit(self):
        for document_ref, data, merge in self.operations:
            document_ref.set(data, merge=merge)


class FakeDb:
    def __init__(self):
        self.collections = {
            "submissions": {},
            "results": {},
        }
        self.collection_refs = {}

    def collection(self, name: str):
        if name not in self.collection_refs:
            self.collection_refs[name] = FakeCollection(self.collections[name])

        return self.collection_refs[name]

    def result_document(self, submission_id: str):
        return self.collection("results").document(submission_id)

    def batch(self):
        return FakeBatch()

    def transaction(self):
        return FakeTransaction()


class FakeSubmissionRepository:
    """Repository-oriented fake for submission persistence and lookup."""

    def __init__(self, initial_submissions: dict[str, dict] | None = None):
        self.submissions = deepcopy(initial_submissions or {})
        self.idempotent_submissions: dict[str, str] = {}
        self._result_repository = None

    def bind_result_repository(self, result_repository) -> None:
        self._result_repository = result_repository

    def accept(
        self,
        scenario: ScenarioBundle,
        *,
        cancel_token_hash: str,
        total_steps: int,
        idempotency_key: str,
    ) -> str:
        if idempotency_key in self.idempotent_submissions:
            submission_id = self.idempotent_submissions[idempotency_key]
            existing = self.submissions[submission_id]
            if (
                existing["scenario"] == scenario.model_dump(mode="json")
                and existing["control"]["cancel_token_hash"] == cancel_token_hash
            ):
                control = existing.get("control", {})
                result = (
                    self._result_repository.results.get(submission_id)
                    if self._result_repository is not None
                    else None
                )
                if control.get("dispatch_state") is None or result is None:
                    raise SubmissionRecoveryError
                return submission_id
            raise SubmissionConflictError

        submission_id = f"submission-{len(self.submissions) + 1}"
        self.submissions[submission_id] = build_submission_document(
            submission_id,
            scenario,
            cancel_token_hash=cancel_token_hash,
        )
        self.idempotent_submissions[idempotency_key] = submission_id
        if self._result_repository is not None:
            self._result_repository.create_queued(
                submission_id,
                total_steps=total_steps,
            )
        return submission_id

    def fetch(self, submission_id: str) -> dict | None:
        payload = self.submissions.get(submission_id)
        if payload is None:
            return None

        return deepcopy(payload)

    def fetch_control(self, submission_id: str) -> SubmissionControl | None:
        payload = self.submissions.get(submission_id)
        if payload is None or payload.get("control") is None:
            return None
        return SubmissionControl.model_validate(payload["control"])

    def claim_dispatch(self, submission_id: str) -> bool:
        submission = self.submissions[submission_id]
        control = submission.setdefault("control", {})
        result = (
            self._result_repository.results.get(submission_id)
            if self._result_repository is not None
            else None
        )
        if (
            control.get("dispatch_state") != DispatchState.PENDING
            or control.get("execution_name") is not None
            or result is None
            or result.get("status") != ResultStatus.QUEUED
        ):
            return False
        control["dispatch_state"] = DispatchState.DISPATCHING
        from datetime import UTC, datetime

        control["dispatch_started_at"] = datetime.now(UTC).isoformat()
        control["dispatch_error"] = None
        return True

    def cancel_pending_dispatch(self, submission_id: str, *, progress) -> dict | None:
        submission = self.submissions[submission_id]
        control = submission.setdefault("control", {})
        result = (
            self._result_repository.results.get(submission_id)
            if self._result_repository is not None
            else None
        )
        if (
            control.get("dispatch_state") != DispatchState.PENDING
            or result is None
            or result.get("status") != ResultStatus.QUEUED
        ):
            return None
        update = build_result_update(
            status=ResultStatus.CANCELLED,
            progress=progress,
        )
        control["dispatch_state"] = DispatchState.CANCELLED
        self._result_repository.results[submission_id] = merge_dicts(result, update)
        return self._result_repository.fetch(submission_id)

    def mark_dispatched(self, submission_id: str, execution_name: str) -> bool:
        submission = self.submissions[submission_id]
        control = submission.setdefault("control", {})
        if (
            control.get("dispatch_state") == DispatchState.DISPATCHED
            and control.get("execution_name") == execution_name
        ):
            return True
        if control.get("dispatch_state") not in {
            DispatchState.DISPATCHING,
            DispatchState.AMBIGUOUS,
        }:
            return False
        control["dispatch_state"] = DispatchState.DISPATCHED
        control["dispatch_error"] = None
        control["execution_name"] = execution_name
        return True

    def mark_dispatch_ambiguous(self, submission_id: str, message: str) -> bool:
        submission = self.submissions[submission_id]
        control = submission.setdefault("control", {})
        if (
            control.get("dispatch_state") == DispatchState.AMBIGUOUS
            and control.get("dispatch_error") == message
        ):
            return True
        if control.get("dispatch_state") != DispatchState.DISPATCHING:
            return False
        control["dispatch_state"] = DispatchState.AMBIGUOUS
        control["dispatch_error"] = message
        from datetime import UTC, datetime

        control["dispatch_started_at"] = datetime.now(UTC).isoformat()
        return True

    def fail_dispatch_if_queued(
        self,
        submission_id: str,
        *,
        progress,
        error: str,
    ) -> str | None:
        submission = self.submissions[submission_id]
        control = submission.setdefault("control", {})
        result = (
            self._result_repository.results.get(submission_id)
            if self._result_repository is not None
            else None
        )
        if (
            control.get("dispatch_state")
            not in {DispatchState.DISPATCHING, DispatchState.AMBIGUOUS}
            or result is None
            or result.get("status") != ResultStatus.QUEUED
        ):
            return False
        update = build_result_update(
            status=ResultStatus.FAILED,
            progress=progress,
            error=error,
        )
        control["dispatch_state"] = DispatchState.FAILED
        control["dispatch_error"] = error
        self._result_repository.results[submission_id] = merge_dicts(result, update)
        return True

    def claim_cancellation(
        self,
        submission_id: str,
        *,
        claimed_at,
        stale_before,
    ) -> bool:
        submission = self.submissions[submission_id]
        control = submission.setdefault("control", {})
        result = (
            self._result_repository.results.get(submission_id)
            if self._result_repository is not None
            else None
        )
        state = CancellationState(
            control.get("cancellation_state", CancellationState.IDLE),
        )
        started_at = control.get("cancellation_started_at")
        if isinstance(started_at, str):
            from datetime import datetime

            started_at = datetime.fromisoformat(started_at)
        lease_active = (
            state is CancellationState.REQUESTING
            and started_at is not None
            and started_at > stale_before
        )
        if (
            state is CancellationState.REQUESTED
            or control.get("execution_name") is None
            or result is None
            or ResultStatus(result.get("status"))
            not in {
                ResultStatus.QUEUED,
                ResultStatus.STARTING,
                ResultStatus.RUNNING,
                ResultStatus.CANCELLING,
            }
            or lease_active
        ):
            return None
        lease_token = uuid.uuid4().hex
        control["cancellation_state"] = CancellationState.REQUESTING
        control["cancellation_started_at"] = claimed_at.isoformat()
        control["cancellation_lease_token"] = lease_token
        control["cancellation_error"] = None
        return lease_token

    def mark_cancellation_requested(self, submission_id: str, lease_token: str) -> bool:
        control = self.submissions[submission_id].setdefault("control", {})
        if (
            control.get("cancellation_state") != CancellationState.REQUESTING
            or control.get("cancellation_lease_token") != lease_token
        ):
            return False
        control["cancellation_state"] = CancellationState.REQUESTED
        control["cancellation_error"] = None
        return True

    def release_cancellation(
        self,
        submission_id: str,
        lease_token: str,
        error: str,
    ) -> bool:
        control = self.submissions[submission_id].setdefault("control", {})
        if (
            control.get("cancellation_state")
            not in {
                CancellationState.REQUESTING,
                CancellationState.REQUESTED,
            }
            or control.get("cancellation_lease_token") != lease_token
        ):
            return False
        control["cancellation_state"] = CancellationState.IDLE
        control["cancellation_started_at"] = None
        control["cancellation_lease_token"] = None
        control["cancellation_error"] = error
        return True


class FakeResultRepository:
    """Repository-oriented fake for result reads and writes."""

    def __init__(self, initial_results: dict[str, dict] | None = None):
        self.results = deepcopy(initial_results or {})
        self.payloads_by_submission: dict[str, list[dict]] = {}

    def create_queued(self, submission_id: str, *, total_steps: int = 0) -> None:
        from embodiedlab.result_models import build_queued_result_document

        payload = build_queued_result_document(
            submission_id,
            total_steps=total_steps,
        )
        self.results[submission_id] = payload
        self.payloads_by_submission.setdefault(submission_id, []).append(
            {"data": deepcopy(payload), "merge": False},
        )

    def fetch(self, submission_id: str) -> dict | None:
        payload = self.results.get(submission_id)
        if payload is None:
            return None

        return deepcopy(payload)

    def write_update(  # noqa: PLR0913
        self,
        submission_id: str,
        *,
        status,
        progress,
        summary: dict | None = None,
        error: str | None = None,
        result_bundle: dict | ResultBundle | None = None,
    ) -> None:
        payload = build_result_update(
            status=status,
            progress=progress,
            summary=summary,
            error=error,
            result_bundle=result_bundle,
        )
        existing = self.results.get(submission_id, {})
        self.results[submission_id] = merge_dicts(existing, payload)
        self.payloads_by_submission.setdefault(submission_id, []).append(
            {"data": deepcopy(payload), "merge": True},
        )

    def payloads_for(self, submission_id: str) -> list[dict]:
        return deepcopy(self.payloads_by_submission.get(submission_id, []))

    def transition_if_status(  # noqa: PLR0913
        self,
        submission_id: str,
        *,
        expected_statuses,
        status,
        progress,
        summary=None,
        error=None,
        result_bundle=None,
    ) -> dict | None:
        result = self.results.get(submission_id)
        if (
            result is None
            or ResultStatus(result.get("status")) not in expected_statuses
        ):
            return None
        update = build_result_update(
            status=status,
            progress=progress,
            summary=summary,
            error=error,
            result_bundle=result_bundle,
        )
        self.results[submission_id] = merge_dicts(result, update)
        self.payloads_by_submission.setdefault(submission_id, []).append(
            {"data": deepcopy(update), "merge": True},
        )
        return self.fetch(submission_id)

    def transition_status_preserving_progress(
        self,
        submission_id: str,
        *,
        expected_statuses,
        status,
        message,
        error=None,
    ) -> dict | None:
        result = self.results.get(submission_id)
        if (
            result is None
            or ResultStatus(result.get("status")) not in expected_statuses
        ):
            return None
        current_progress = result.get("progress") or {
            "phase": result["status"],
            "current_step": 0,
            "total_steps": 0,
            "message": result["status"],
        }
        progress = {
            "phase": status,
            "current_step": current_progress["current_step"],
            "total_steps": current_progress["total_steps"],
            "message": message,
        }
        return self.transition_if_status(
            submission_id,
            expected_statuses=expected_statuses,
            status=status,
            progress=progress,
            error=error,
        )
