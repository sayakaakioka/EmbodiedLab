from copy import deepcopy

from google.api_core.exceptions import AlreadyExists

from embodiedlab.repositories import SubmissionConflictError
from embodiedlab.result_models import ResultBundle, build_result_update
from embodiedlab.schemas import (
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

    def get(self):
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


class FakeSubmissionRepository:
    """Repository-oriented fake for submission persistence and lookup."""

    def __init__(self, initial_submissions: dict[str, dict] | None = None):
        self.submissions = deepcopy(initial_submissions or {})
        self.idempotent_submissions: dict[str, str] = {}

    def save(
        self,
        scenario: ScenarioBundle,
        *,
        cancel_token_hash: str,
        idempotency_key: str | None = None,
    ) -> str:
        if idempotency_key in self.idempotent_submissions:
            submission_id = self.idempotent_submissions[idempotency_key]
            existing = self.submissions[submission_id]
            if (
                existing["scenario"] == scenario.model_dump(mode="json")
                and existing["control"]["cancel_token_hash"] == cancel_token_hash
            ):
                return submission_id
            raise SubmissionConflictError

        submission_id = f"submission-{len(self.submissions) + 1}"
        self.submissions[submission_id] = build_submission_document(
            submission_id,
            scenario,
            cancel_token_hash=cancel_token_hash,
        )
        if idempotency_key is not None:
            self.idempotent_submissions[idempotency_key] = submission_id
        return submission_id

    def exists(self, submission_id: str) -> bool:
        return submission_id in self.submissions

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

    def set_execution_name(self, submission_id: str, execution_name: str) -> None:
        submission = self.submissions[submission_id]
        control = submission.setdefault("control", {})
        control["execution_name"] = execution_name


class FakeResultRepository:
    """Repository-oriented fake for result reads and writes."""

    def __init__(self, initial_results: dict[str, dict] | None = None):
        self.results = deepcopy(initial_results or {})
        self.payloads_by_submission: dict[str, list[dict]] = {}

    def create_queued(self, submission_id: str) -> None:
        from embodiedlab.result_models import build_queued_result_document

        payload = build_queued_result_document(submission_id)
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
