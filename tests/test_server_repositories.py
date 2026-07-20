from __future__ import annotations

import pytest

from embodiedlab.repositories import SubmissionConflictError
from embodiedlab.schemas import ScenarioBundle
from server.repositories import FirestoreSubmissionRepository
from tests.fakes import FakeDb


def test_firestore_submission_save_is_idempotent() -> None:
    db = FakeDb()
    repository = FirestoreSubmissionRepository(db)
    scenario = ScenarioBundle()

    first = repository.save(
        scenario,
        cancel_token_hash="a" * 64,
        idempotency_key="submission-recovery-key-0000000001",
    )
    replay = repository.save(
        scenario,
        cancel_token_hash="a" * 64,
        idempotency_key="submission-recovery-key-0000000001",
    )

    assert replay == first
    assert len(db.collections["submissions"]) == 1


@pytest.mark.parametrize(
    ("scenario", "cancel_token_hash"),
    [
        (ScenarioBundle(scenario_id="different"), "a" * 64),
        (ScenarioBundle(), "b" * 64),
    ],
)
def test_firestore_submission_save_rejects_conflicting_replay(
    scenario: ScenarioBundle,
    cancel_token_hash: str,
) -> None:
    db = FakeDb()
    repository = FirestoreSubmissionRepository(db)
    idempotency_key = "submission-recovery-key-0000000001"
    repository.save(
        ScenarioBundle(),
        cancel_token_hash="a" * 64,
        idempotency_key=idempotency_key,
    )

    with pytest.raises(SubmissionConflictError):
        repository.save(
            scenario,
            cancel_token_hash=cancel_token_hash,
            idempotency_key=idempotency_key,
        )
