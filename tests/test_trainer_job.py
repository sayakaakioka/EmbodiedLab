import pytest

from embodiedlab.schemas import ScenarioBundle
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
    submission = {"scenario": ScenarioBundle().model_dump(mode="json")}
    submission_repository = FakeSubmissionRepository(
        initial_submissions={"submission-1": submission},
    )
    result_repository = FakeResultRepository()
    calls = []

    def train_model(  # noqa: PLR0913
        *,
        spec,
        training,
        model_output_path,
        progress_callback=None,
        diagnostic_callback=None,
        scenario_id=None,
        job_id=None,
    ):
        assert progress_callback is not None
        calls.append(("train", spec, training, model_output_path, scenario_id, job_id))
        return {
            "score": 1.0,
            "replay_bundle_dir": str(model_output_path) + "_replay",
        }

    def upload_model(
        *,
        local_model_base_path,
        bucket_name,
        submission_id,
        replay_bundle_dir=None,
    ):
        calls.append(
            (
                "upload",
                local_model_base_path,
                bucket_name,
                submission_id,
                replay_bundle_dir,
            ),
        )
        return {
            "model": {
                "bucket": bucket_name,
                "path": f"results/{submission_id}/model/policy.zip",
            },
            "onnx_model": {
                "bucket": bucket_name,
                "path": f"results/{submission_id}/model/policy.onnx",
            },
            "sentis_model": {
                "bucket": bucket_name,
                "path": f"results/{submission_id}/model/policy.sentis.onnx",
            },
            "replay_bundle": {
                "bucket": bucket_name,
                "path": f"results/{submission_id}/replay/manifest.json",
                "format": "json",
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
    assert payloads[-1]["data"]["summary"] == {
        "score": 1.0,
        "training_timesteps": 5000,
        "training_seed": 10,
    }
    assert "artifacts" not in payloads[-1]["data"]
    assert payloads[-1]["data"]["result_bundle"]["schema_version"] == (
        "result-bundle.v0"
    )
    assert payloads[-1]["data"]["result_bundle"]["summary"] == {
        "training_timesteps": 5000,
        "training_seed": 10,
        "success_rate": None,
        "average_episode_reward": None,
        "average_episode_steps": None,
    }
    assert (
        payloads[-1]["data"]["result_bundle"]["artifacts"]["model"]["path"]
        == "results/submission-1/model/policy.onnx"
    )
    assert (
        payloads[-1]["data"]["result_bundle"]["artifacts"]["onnx_model"]["path"]
        == "results/submission-1/model/policy.onnx"
    )
    assert (
        payloads[-1]["data"]["result_bundle"]["artifacts"]["sentis_model"]["path"]
        == "results/submission-1/model/policy.sentis.onnx"
    )
    assert (
        payloads[-1]["data"]["result_bundle"]["artifacts"]["replay_bundle"]["path"]
        == "results/submission-1/replay/manifest.json"
    )
    assert calls[0][0] == "train"
    assert calls[0][4:] == ("scenario_demo_001", "submission-1")
    assert calls[1][0] == "upload"
    assert calls[1][4].endswith("_replay")


def test_run_training_job_writes_training_progress_updates():
    submission = {"scenario": ScenarioBundle().model_dump(mode="json")}
    submission_repository = FakeSubmissionRepository(
        initial_submissions={"submission-1": submission},
    )
    result_repository = FakeResultRepository()
    published_events = []

    def train_model(  # noqa: PLR0913
        *,
        spec,
        training,
        model_output_path,
        progress_callback,
        diagnostic_callback=None,
        scenario_id=None,
        job_id=None,
    ):
        progress_callback(10000, training.timesteps)
        progress_callback(20000, training.timesteps)
        return {"score": 1.0}

    run_training_job(
        _CONFIG,
        create_db=lambda db_id: object(),
        create_submission_repository=lambda db: submission_repository,
        create_result_repository=lambda db: result_repository,
        train_model=train_model,
        upload_model=lambda **kwargs: {},
        publish_event=lambda **kwargs: published_events.append(kwargs),
    )

    payloads = result_repository.payloads_for("submission-1")
    running_steps = [
        payload["data"]["progress"]["current_step"]
        for payload in payloads
        if payload["data"]["status"] == "running"
    ]
    assert running_steps == [0, 10000, 20000]
    assert [
        event["progress"].current_step
        for event in published_events
        if event["status"] == "running"
    ] == [0, 10000, 20000]


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
    submission = {"scenario": ScenarioBundle().model_dump(mode="json")}
    submission["scenario"]["training"]["timesteps"] = 0
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


def test_run_training_job_writes_failed_result_bundle_after_runtime_failure():
    submission = {"scenario": ScenarioBundle().model_dump(mode="json")}
    submission_repository = FakeSubmissionRepository(
        initial_submissions={"submission-1": submission},
    )
    result_repository = FakeResultRepository()

    def fail_training(**kwargs):
        msg = "runtime exploded"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError):
        run_training_job(
            _CONFIG,
            create_db=lambda db_id: object(),
            create_submission_repository=lambda db: submission_repository,
            create_result_repository=lambda db: result_repository,
            train_model=fail_training,
            upload_model=lambda **kwargs: {},
            publish_event=_NO_PUBLISH,
        )

    payload = result_repository.payloads_for("submission-1")[-1]["data"]
    assert payload["status"] == "failed"
    assert "runtime exploded" in payload["error"]
    assert payload["result_bundle"]["status"] == "failed"
    assert "runtime exploded" in payload["result_bundle"]["error"]["message"]
