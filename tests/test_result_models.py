from embodiedlab.result_models import (
    ResultStatus,
    build_queued_result_document,
    build_result_message,
    build_result_update,
    completed_progress,
    failed_progress,
    parse_result_message,
    queued_progress,
    running_progress,
    starting_progress,
)


def test_build_queued_result_document_returns_firestore_payload():
    payload = build_queued_result_document("submission-1")

    assert payload["submission_id"] == "submission-1"
    assert payload["status"] == "queued"
    assert payload["progress"] == {
        "phase": "queued",
        "current_step": 0,
        "total_steps": 0,
        "message": "Queued",
    }
    assert payload["summary"] is None
    assert payload["error"] is None
    assert payload["artifacts"] is None
    assert isinstance(payload["updated_at"], str)


def test_build_result_update_serializes_status_enum():
    payload = build_result_update(
        status=ResultStatus.RUNNING,
        progress={
            "phase": ResultStatus.RUNNING,
            "current_step": 0,
            "total_steps": 10,
            "message": "Training",
        },
    )

    assert payload["status"] == "running"
    assert payload["progress"]["phase"] == "running"
    assert payload["progress"]["total_steps"] == 10


def test_progress_factories_return_expected_payloads():
    queued = queued_progress()
    starting = starting_progress(10)
    running = running_progress(10)
    completed = completed_progress(10)
    failed = failed_progress("boom", total_steps=10)

    assert queued.phase is ResultStatus.QUEUED
    assert starting.message == "Trainer job started"
    assert running.message == "Training"
    assert completed.current_step == 10
    assert completed.phase is ResultStatus.COMPLETED
    assert failed.phase is ResultStatus.FAILED
    assert failed.message == "boom"


def test_parse_result_message_validates_and_normalizes_payload():
    payload = build_result_message(
        submission_id="submission-1",
        status=ResultStatus.RUNNING,
        progress=running_progress(10),
    )

    parsed = parse_result_message(payload)

    assert parsed["submission_id"] == "submission-1"
    assert parsed["status"] == "running"
    assert parsed["progress"]["phase"] == "running"
