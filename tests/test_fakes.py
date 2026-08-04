from embodiedlab.result_models import failed_progress
from tests.fakes import (
    FakeDb,
    FakeResultRepository,
    FakeSubmissionRepository,
    scenario_bundle,
)

IDEMPOTENCY_KEY = "submission-recovery-key-0000000001"


def test_fake_db_merge_recursively_updates_nested_documents():
    fake_db = FakeDb()
    result_ref = fake_db.result_document("submission-1")

    result_ref.set(
        {
            "status": "running",
            "progress": {
                "phase": "running",
                "current_step": 1,
                "total_steps": 10,
                "message": "Training",
            },
        },
    )
    result_ref.set(
        {
            "progress": {
                "current_step": 10,
                "message": "Training completed",
            },
        },
        merge=True,
    )

    assert fake_db.collections["results"]["submission-1"] == {
        "status": "running",
        "progress": {
            "phase": "running",
            "current_step": 10,
            "total_steps": 10,
            "message": "Training completed",
        },
    }


def test_fake_submission_repository_persists_and_fetches_submission():
    repository = FakeSubmissionRepository()
    result_repository = FakeResultRepository()
    repository.bind_result_repository(result_repository)

    submission_id = repository.accept(
        scenario_bundle(),
        cancel_token_hash="a" * 64,
        total_steps=5000,
        idempotency_key=IDEMPOTENCY_KEY,
    )
    assert repository.fetch(submission_id)["submission_id"] == submission_id
    assert repository.fetch_control(submission_id).cancel_token_hash == "a" * 64

    assert repository.claim_dispatch(submission_id) is True
    assert (
        repository.mark_dispatched(
            submission_id,
            "projects/test/locations/test/executions/test-trainer-abcde",
        )
        is True
    )

    assert repository.fetch_control(submission_id).execution_name.endswith(
        "/executions/test-trainer-abcde",
    )


def test_fake_result_repository_tracks_payload_history():
    repository = FakeResultRepository()

    repository.create_queued("submission-1")
    progress = failed_progress("boom")
    repository.write_update(
        "submission-1",
        status=progress.phase,
        progress=progress,
        error="boom",
    )

    payloads = repository.payloads_for("submission-1")
    assert [payload["data"]["status"] for payload in payloads] == [
        "queued",
        "failed",
    ]
    assert repository.fetch("submission-1")["error"] == "boom"
