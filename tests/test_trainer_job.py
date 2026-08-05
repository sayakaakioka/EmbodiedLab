from dataclasses import replace

import pytest

from embodiedlab.result_models import build_queued_result_document
from tests.fakes import (
    FakeResultRepository,
    FakeSubmissionRepository,
    completed_artifacts,
    resolved_training_configuration,
    resolved_training_summary,
    scenario_bundle,
)
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


def _training_output():
    return {
        "summary": resolved_training_summary(),
        "replay_bundle_dir": "replay_bundle",
    }


def _queued_result_repository() -> FakeResultRepository:
    return FakeResultRepository(
        initial_results={
            "submission-1": build_queued_result_document(
                "submission-1",
                total_steps=5000,
            ),
        },
    )


def test_run_training_job_updates_result_to_completed():
    submission = {"scenario": scenario_bundle().model_dump(mode="json")}
    submission_repository = FakeSubmissionRepository(
        initial_submissions={"submission-1": submission},
    )
    result_repository = _queued_result_repository()
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
            **_training_output(),
            "replay_bundle_dir": str(model_output_path) + "_replay",
        }

    def upload_model(
        *,
        local_model_base_path,
        bucket_name,
        submission_id,
        scenario,
        replay_bundle_dir=None,
    ):
        assert scenario == scenario_bundle()
        calls.append(
            (
                "upload",
                local_model_base_path,
                bucket_name,
                submission_id,
                replay_bundle_dir,
            ),
        )
        return completed_artifacts(bucket_name, submission_id)

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
    assert "summary" not in payloads[-1]["data"]
    assert "artifacts" not in payloads[-1]["data"]
    assert payloads[-1]["data"]["result_bundle"]["schema_version"] == (
        "result-bundle.v0"
    )
    assert payloads[-1]["data"]["result_bundle"]["summary"] == {
        "success_rate": None,
        "average_episode_reward": None,
        "average_episode_steps": None,
        "configuration": resolved_training_configuration(),
    }
    assert "model" not in payloads[-1]["data"]["result_bundle"]["artifacts"]
    assert (
        payloads[-1]["data"]["result_bundle"]["artifacts"]["onnx_model"]["path"]
        == "results/submission-1/model/policy.onnx"
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
    submission = {"scenario": scenario_bundle().model_dump(mode="json")}
    submission_repository = FakeSubmissionRepository(
        initial_submissions={"submission-1": submission},
    )
    result_repository = _queued_result_repository()
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
        return _training_output()

    run_training_job(
        _CONFIG,
        create_db=lambda db_id: object(),
        create_submission_repository=lambda db: submission_repository,
        create_result_repository=lambda db: result_repository,
        train_model=train_model,
        upload_model=lambda **kwargs: completed_artifacts(
            kwargs["bucket_name"],
            kwargs["submission_id"],
        ),
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
    result_repository = _queued_result_repository()

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
    submission = {"scenario": scenario_bundle().model_dump(mode="json")}
    submission["scenario"]["training"]["timesteps"] = 0
    submission_repository = FakeSubmissionRepository(
        initial_submissions={"submission-1": submission},
    )
    result_repository = _queued_result_repository()

    with pytest.raises(Exception):
        run_training_job(
            _CONFIG,
            create_db=lambda db_id: object(),
            create_submission_repository=lambda db: submission_repository,
            create_result_repository=lambda db: result_repository,
            train_model=lambda **kwargs: _training_output(),
            upload_model=lambda **kwargs: completed_artifacts(
                "model-bucket",
                "submission-1",
            ),
            publish_event=_NO_PUBLISH,
        )

    payload = result_repository.payloads_for("submission-1")[0]["data"]
    assert payload["status"] == "failed"
    assert payload["progress"]["total_steps"] == 0
    assert "timesteps" in payload["error"]


def test_run_training_job_writes_failed_result_bundle_after_runtime_failure():
    submission = {"scenario": scenario_bundle().model_dump(mode="json")}
    submission_repository = FakeSubmissionRepository(
        initial_submissions={"submission-1": submission},
    )
    result_repository = _queued_result_repository()

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


def test_run_training_job_does_not_revive_closed_dispatch():
    execution_name = (
        "projects/test-project/locations/asia-northeast1/jobs/test-trainer/"
        "executions/test-trainer-abcde"
    )
    submission = {
        "scenario": scenario_bundle().model_dump(mode="json"),
        "control": {
            "cancel_token_hash": "a" * 64,
            "dispatch_state": "failed",
            "dispatch_error": "Dispatch outcome could not be confirmed",
            "execution_name": None,
        },
    }
    submission_repository = FakeSubmissionRepository(
        initial_submissions={"submission-1": submission},
    )
    result_repository = FakeResultRepository(
        initial_results={
            "submission-1": {
                "submission_id": "submission-1",
                "status": "failed",
            },
        },
    )
    training_calls = []

    run_training_job(
        replace(_CONFIG, execution_name=execution_name),
        create_db=lambda db_id: object(),
        create_submission_repository=lambda db: submission_repository,
        create_result_repository=lambda db: result_repository,
        train_model=lambda **kwargs: training_calls.append(kwargs),
        publish_event=_NO_PUBLISH,
    )

    assert training_calls == []
    assert result_repository.fetch("submission-1")["status"] == "failed"
    assert submission_repository.fetch_control("submission-1").dispatch_state == (
        "failed"
    )


def test_run_training_job_recovers_ambiguous_execution_before_training():
    execution_name = (
        "projects/test-project/locations/asia-northeast1/jobs/test-trainer/"
        "executions/test-trainer-abcde"
    )
    submission = {
        "scenario": scenario_bundle().model_dump(mode="json"),
        "control": {
            "cancel_token_hash": "a" * 64,
            "dispatch_state": "ambiguous",
            "dispatch_error": "Dispatch response was lost",
            "execution_name": None,
        },
    }
    submission_repository = FakeSubmissionRepository(
        initial_submissions={"submission-1": submission},
    )
    result_repository = _queued_result_repository()

    run_training_job(
        replace(_CONFIG, execution_name=execution_name),
        create_db=lambda db_id: object(),
        create_submission_repository=lambda db: submission_repository,
        create_result_repository=lambda db: result_repository,
        train_model=lambda **kwargs: _training_output(),
        upload_model=lambda **kwargs: completed_artifacts(
            kwargs["bucket_name"],
            kwargs["submission_id"],
        ),
        publish_event=_NO_PUBLISH,
    )

    control = submission_repository.fetch_control("submission-1")
    assert control.dispatch_state == "dispatched"
    assert control.execution_name == execution_name
    assert result_repository.fetch("submission-1")["status"] == "completed"


def test_run_training_job_preserves_completion_while_cancellation_is_pending():
    submission = {"scenario": scenario_bundle().model_dump(mode="json")}
    submission_repository = FakeSubmissionRepository(
        initial_submissions={"submission-1": submission},
    )
    result_repository = FakeResultRepository(
        initial_results={
            "submission-1": {
                "submission_id": "submission-1",
                "status": "cancelling",
                "progress": {
                    "phase": "cancelling",
                    "current_step": 0,
                    "total_steps": 5000,
                    "message": "Cancelling training",
                },
            },
        },
    )

    run_training_job(
        _CONFIG,
        create_db=lambda db_id: object(),
        create_submission_repository=lambda db: submission_repository,
        create_result_repository=lambda db: result_repository,
        train_model=lambda **kwargs: _training_output(),
        upload_model=lambda **kwargs: completed_artifacts(
            kwargs["bucket_name"],
            kwargs["submission_id"],
        ),
        publish_event=_NO_PUBLISH,
    )

    assert result_repository.fetch("submission-1")["status"] == "completed"
