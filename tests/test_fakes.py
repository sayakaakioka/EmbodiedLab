from embodiedlab.result_models import failed_progress
from embodiedlab.schemas import SubmitRequest
from tests.fakes import FakeDb, FakeResultRepository, FakeSubmissionRepository


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

    submission_id = repository.save(SubmitRequest())

    assert repository.exists(submission_id) is True
    assert repository.fetch(submission_id)["submission_id"] == submission_id


def test_fake_result_repository_tracks_payload_history():
    repository = FakeResultRepository()

    repository.create_queued("submission-1")
    repository.mark_failed(
        "submission-1",
        failed_progress("boom"),
        "boom",
    )

    payloads = repository.payloads_for("submission-1")
    assert [payload["data"]["status"] for payload in payloads] == [
        "queued",
        "failed",
    ]
    assert repository.fetch("submission-1")["error"] == "boom"
