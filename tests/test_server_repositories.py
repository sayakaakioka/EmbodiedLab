from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest
from google.api_core.exceptions import Aborted

from embodiedlab.repositories import SubmissionConflictError, SubmissionRecoveryError
from embodiedlab.result_models import ResultStatus, cancelled_progress, failed_progress
from server.repositories import FirestoreResultRepository, FirestoreSubmissionRepository
from tests.fakes import FakeDb, FakeTransaction, scenario_bundle

if TYPE_CHECKING:
    from embodiedlab.schemas import ScenarioBundle

CONCURRENT_UPDATE = "concurrent update"
IDEMPOTENCY_KEY = "submission-recovery-key-0000000001"


class AbortingTransaction(FakeTransaction):
    def commit(self):
        raise Aborted(CONCURRENT_UPDATE)


class AbortingDb(FakeDb):
    def transaction(self):
        return AbortingTransaction()


class OneShotAbortingDb(FakeDb):
    def __init__(self):
        super().__init__()
        self.transaction_count = 0

    def transaction(self):
        self.transaction_count += 1
        if self.transaction_count == 1:
            return AbortingTransaction()
        return FakeTransaction()


def test_firestore_submission_accept_is_atomic_and_idempotent() -> None:
    db = FakeDb()
    repository = FirestoreSubmissionRepository(db)
    scenario = scenario_bundle()

    first = repository.accept(
        scenario,
        cancel_token_hash="a" * 64,
        total_steps=5000,
        idempotency_key="submission-recovery-key-0000000001",
    )
    replay = repository.accept(
        scenario,
        cancel_token_hash="a" * 64,
        total_steps=5000,
        idempotency_key="submission-recovery-key-0000000001",
    )

    assert replay == first
    assert len(db.collections["submissions"]) == 1
    assert db.collections["results"][first]["status"] == "queued"
    assert db.collections["results"][first]["progress"]["total_steps"] == 5000


@pytest.mark.parametrize(
    ("scenario", "cancel_token_hash"),
    [
        (scenario_bundle().model_copy(update={"scenario_id": "different"}), "a" * 64),
        (scenario_bundle(), "b" * 64),
    ],
)
def test_firestore_submission_accept_rejects_conflicting_replay(
    scenario: ScenarioBundle,
    cancel_token_hash: str,
) -> None:
    db = FakeDb()
    repository = FirestoreSubmissionRepository(db)
    idempotency_key = "submission-recovery-key-0000000001"
    repository.accept(
        scenario_bundle(),
        cancel_token_hash="a" * 64,
        total_steps=5000,
        idempotency_key=idempotency_key,
    )

    with pytest.raises(SubmissionConflictError):
        repository.accept(
            scenario,
            cancel_token_hash=cancel_token_hash,
            total_steps=5000,
            idempotency_key=idempotency_key,
        )


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_result",
        "missing_dispatch_state",
        "invalid_dispatch_state",
        "invalid_result_status",
        "malformed_control",
    ],
)
def test_firestore_submission_rejects_unrecoverable_existing_submission(
    corruption: str,
) -> None:
    db = FakeDb()
    repository = FirestoreSubmissionRepository(db)
    idempotency_key = "submission-recovery-key-0000000001"
    submission_id = repository.accept(
        scenario_bundle(),
        cancel_token_hash="a" * 64,
        total_steps=5000,
        idempotency_key=idempotency_key,
    )
    if corruption == "missing_result":
        del db.collections["results"][submission_id]
    elif corruption == "missing_dispatch_state":
        del db.collections["submissions"][submission_id]["control"]["dispatch_state"]
    elif corruption == "invalid_dispatch_state":
        db.collections["submissions"][submission_id]["control"]["dispatch_state"] = (
            "unknown"
        )
    elif corruption == "invalid_result_status":
        db.collections["results"][submission_id]["status"] = "unknown"
    else:
        db.collections["submissions"][submission_id]["control"] = "malformed"

    with pytest.raises(SubmissionRecoveryError):
        repository.accept(
            scenario_bundle(),
            cancel_token_hash="a" * 64,
            total_steps=5000,
            idempotency_key=idempotency_key,
        )


def test_firestore_submission_dispatch_can_only_be_claimed_once() -> None:
    db = FakeDb()
    repository = FirestoreSubmissionRepository(db)
    accepted = repository.accept(
        scenario_bundle(),
        cancel_token_hash="a" * 64,
        total_steps=5000,
        idempotency_key=IDEMPOTENCY_KEY,
    )

    assert repository.claim_dispatch(accepted) is True
    assert repository.claim_dispatch(accepted) is False
    control = repository.fetch_control(accepted)
    assert control.dispatch_state == "dispatching"
    assert control.dispatch_started_at is not None


@pytest.mark.parametrize("execution_name", [None, "legacy-execution"])
def test_firestore_submission_does_not_claim_legacy_control(
    execution_name: str | None,
) -> None:
    db = FakeDb()
    repository = FirestoreSubmissionRepository(db)
    accepted = repository.accept(
        scenario_bundle(),
        cancel_token_hash="a" * 64,
        total_steps=5000,
        idempotency_key=IDEMPOTENCY_KEY,
    )
    control = db.collections["submissions"][accepted]["control"]
    control.pop("dispatch_state")
    control["execution_name"] = execution_name

    assert repository.claim_dispatch(accepted) is False
    assert control.get("dispatch_state") is None


def test_firestore_submission_claim_retries_one_aborted_transaction() -> None:
    db = OneShotAbortingDb()
    repository = FirestoreSubmissionRepository(db)
    accepted = repository.accept(
        scenario_bundle(),
        cancel_token_hash="a" * 64,
        total_steps=5000,
        idempotency_key=IDEMPOTENCY_KEY,
    )

    assert repository.claim_dispatch(accepted) is True
    assert db.transaction_count == 2


def test_firestore_submission_pending_dispatch_can_be_cancelled_once() -> None:
    db = FakeDb()
    repository = FirestoreSubmissionRepository(db)
    accepted = repository.accept(
        scenario_bundle(),
        cancel_token_hash="a" * 64,
        total_steps=5000,
        idempotency_key=IDEMPOTENCY_KEY,
    )

    progress = cancelled_progress(0, 5000)
    assert (
        repository.cancel_pending_dispatch(
            accepted,
            progress=progress,
        )["status"]
        == "cancelled"
    )
    assert (
        repository.cancel_pending_dispatch(
            accepted,
            progress=progress,
        )
        is None
    )
    assert repository.claim_dispatch(accepted) is False
    control = repository.fetch_control(accepted)
    assert control.dispatch_state == "cancelled"


def test_firestore_pending_cancel_retry_exhaustion_changes_neither_document() -> None:
    db = AbortingDb()
    repository = FirestoreSubmissionRepository(db)
    accepted = repository.accept(
        scenario_bundle(),
        cancel_token_hash="a" * 64,
        total_steps=5000,
        idempotency_key=IDEMPOTENCY_KEY,
    )

    with pytest.raises(Aborted):
        repository.cancel_pending_dispatch(
            accepted,
            progress=cancelled_progress(0, 5000),
        )

    assert repository.fetch_control(accepted).dispatch_state == "pending"
    assert db.collections["results"][accepted]["status"] == "queued"


def test_firestore_dispatch_finalization_does_not_revive_failed_control() -> None:
    db = FakeDb()
    repository = FirestoreSubmissionRepository(db)
    accepted = repository.accept(
        scenario_bundle(),
        cancel_token_hash="a" * 64,
        total_steps=5000,
        idempotency_key=IDEMPOTENCY_KEY,
    )
    assert repository.claim_dispatch(accepted) is True
    db.collections["submissions"][accepted]["control"]["dispatch_state"] = "failed"

    assert repository.mark_dispatched(accepted, "late-execution") is False
    control = repository.fetch_control(accepted)
    assert control.dispatch_state == "failed"
    assert control.execution_name is None


def test_firestore_cancellation_lease_has_one_owner_and_can_be_reclaimed() -> None:
    db = FakeDb()
    repository = FirestoreSubmissionRepository(db)
    submission_id = repository.accept(
        scenario_bundle(),
        cancel_token_hash="a" * 64,
        total_steps=5000,
        idempotency_key=IDEMPOTENCY_KEY,
    )
    assert repository.claim_dispatch(submission_id) is True
    assert repository.mark_dispatched(submission_id, "execution-1") is True
    first_claim = datetime.now(UTC)

    first_lease = repository.claim_cancellation(
        submission_id,
        claimed_at=first_claim,
        stale_before=first_claim - timedelta(minutes=2),
    )
    assert first_lease is not None
    assert (
        repository.claim_cancellation(
            submission_id,
            claimed_at=first_claim + timedelta(minutes=1),
            stale_before=first_claim - timedelta(minutes=1),
        )
        is None
    )
    second_lease = repository.claim_cancellation(
        submission_id,
        claimed_at=first_claim + timedelta(minutes=3),
        stale_before=first_claim + timedelta(minutes=1),
    )
    assert second_lease is not None
    assert second_lease != first_lease

    assert repository.mark_cancellation_requested(submission_id, first_lease) is False
    assert (
        repository.release_cancellation(
            submission_id,
            first_lease,
            "stale owner",
        )
        is False
    )

    control = repository.fetch_control(submission_id)
    assert control.cancellation_state == "requesting"
    assert control.cancellation_lease_token == second_lease

    assert repository.mark_cancellation_requested(submission_id, second_lease) is True
    assert (
        repository.claim_cancellation(
            submission_id,
            claimed_at=first_claim + timedelta(minutes=6),
            stale_before=first_claim + timedelta(minutes=4),
        )
        is None
    )


def test_firestore_result_transition_refuses_terminal_result() -> None:
    db = FakeDb()
    db.collections["results"]["submission-1"] = {
        "submission_id": "submission-1",
        "status": "completed",
    }
    repository = FirestoreResultRepository(db)

    result = repository.transition_if_status(
        "submission-1",
        expected_statuses={ResultStatus.RUNNING},
        status=ResultStatus.FAILED,
        progress=failed_progress("late failure", total_steps=5000),
        error="late failure",
    )

    assert result is None
    assert db.collections["results"]["submission-1"]["status"] == "completed"


def test_firestore_status_transition_preserves_current_progress() -> None:
    db = FakeDb()
    db.collections["results"]["submission-1"] = {
        "submission_id": "submission-1",
        "status": "running",
        "progress": {
            "phase": "running",
            "current_step": 60,
            "total_steps": 100,
            "message": "Training",
        },
    }
    repository = FirestoreResultRepository(db)

    result = repository.transition_status_preserving_progress(
        "submission-1",
        expected_statuses={ResultStatus.RUNNING},
        status=ResultStatus.CANCELLING,
        message="Cancelling training",
    )

    assert result["status"] == "cancelling"
    assert result["progress"] == {
        "phase": "cancelling",
        "current_step": 60,
        "total_steps": 100,
        "message": "Cancelling training",
    }
