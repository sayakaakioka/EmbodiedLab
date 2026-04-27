import pytest

from embodiedlab.schemas import SubmitRequest
from tests.fakes import FakeResultRepository, FakeSubmissionRepository
from trainer.config import TrainerConfig
from trainer.job import run_training_job

_CONFIG = TrainerConfig(
    db_id="test-db",
    model_bucket="model-bucket",
    submission_id="submission-1",
    pubsub_topic="test-topic",
    project_id="test-project",
)

_NO_PUBLISH = lambda **kwargs: None  # noqa: E731


def test_run_training_job_updates_result_to_completed():
    submission = SubmitRequest().model_dump(mode="json")
    submission_repository = FakeSubmissionRepository(
        initial_submissions={"submission-1": submission},
    )
    result_repository = FakeResultRepository()
    calls = []

    def train_model(*, spec, training, model_output_path):
        calls.append(("train", spec, training, model_output_path))
        return {"score": 1.0}

    def upload_model(*, local_model_base_path, bucket_name, submission_id):
        calls.append(("upload", local_model_base_path, bucket_name, submission_id))
        return {
            "model": {
                "bucket": bucket_name,
                "path": f"models/{submission_id}/policy.zip",
            },
            "onnx_model": {
                "bucket": bucket_name,
                "path": f"models/{submission_id}/policy.onnx",
            },
            "sentis_model": {
                "bucket": bucket_name,
                "path": f"models/{submission_id}/policy.sentis.onnx",
            },
        }

    run_training_job(
        _CONFIG,
        create_db=lambda db_id: object(),
        create_submission_repository=lambda db: submission_repository,
        create_result_repository=lambda db: result_repository,
        train_model=train_model,
        upload_model=upload_model,
        publish_event=_NO_PUBLISH,
    )

    payloads = result_repository.payloads_for("submission-1")
    statuses = [payload["data"]["status"] for payload in payloads]
    assert statuses == ["starting", "running", "completed"]
    assert payloads[-1]["data"]["summary"] == {"score": 1.0}
    assert payloads[-1]["data"]["artifacts"]["model"]["bucket"] == "model-bucket"
    assert (
        payloads[-1]["data"]["artifacts"]["onnx_model"]["path"]
        == "models/submission-1/policy.onnx"
    )
    assert (
        payloads[-1]["data"]["artifacts"]["sentis_model"]["path"]
        == "models/submission-1/policy.sentis.onnx"
    )
    assert calls[0][0] == "train"
    assert calls[1][0] == "upload"


def test_run_training_job_marks_missing_submission_failed():
    submission_repository = FakeSubmissionRepository()
    result_repository = FakeResultRepository()

    run_training_job(
        _CONFIG,
        create_db=lambda db_id: object(),
        create_submission_repository=lambda db: submission_repository,
        create_result_repository=lambda db: result_repository,
        publish_event=_NO_PUBLISH,
    )

    payload = result_repository.payloads_for("submission-1")[0]["data"]
    assert payload["status"] == "failed"
    assert payload["progress"]["message"] == "Submission not found"
    assert payload["error"] == "Submission not found"


def test_run_training_job_marks_invalid_submission_failed():
    submission = SubmitRequest().model_dump(mode="json")
    submission["training"]["timesteps"] = 0
    submission_repository = FakeSubmissionRepository(
        initial_submissions={"submission-1": submission},
    )
    result_repository = FakeResultRepository()

    with pytest.raises(Exception):
        run_training_job(
            _CONFIG,
            create_db=lambda db_id: object(),
            create_submission_repository=lambda db: submission_repository,
            create_result_repository=lambda db: result_repository,
            train_model=lambda **kwargs: {"score": 1.0},
            upload_model=lambda **kwargs: {"model": {}},
            publish_event=_NO_PUBLISH,
        )

    payload = result_repository.payloads_for("submission-1")[0]["data"]
    assert payload["status"] == "failed"
    assert payload["progress"]["total_steps"] == 0
    assert "timesteps" in payload["error"]
