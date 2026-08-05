from pathlib import Path

import pytest
from pydantic import ValidationError

from embodiedlab.training.training_converter import describe_runtime_conversion
from tests.fakes import (
    completed_artifacts,
    resolved_training_summary,
    scenario_bundle,
)
from trainer.training_service import execute_training_run, parse_training_submission


def test_execute_training_run_uploads_replay_bundle():
    scenario = scenario_bundle()
    captured = {}

    def train_model(*, spec, training, model_output_path, scenario_id, job_id):
        replay_bundle_dir = Path(model_output_path).parent / "replay_bundle"
        replay_bundle_dir.mkdir()
        (replay_bundle_dir / "manifest.json").write_text(
            '{"schema_version":"replay-bundle.v0","chunks":[]}',
            encoding="utf-8",
        )
        captured["train"] = {
            "scenario_id": scenario_id,
            "job_id": job_id,
        }
        return {
            "summary": resolved_training_summary(),
            "replay_bundle_dir": str(replay_bundle_dir),
        }

    def upload_model(
        *,
        local_model_base_path,
        bucket_name,
        submission_id,
        scenario,
        replay_bundle_dir,
    ):
        assert scenario == inputs.scenario
        captured["upload"] = {
            "bucket_name": bucket_name,
            "submission_id": submission_id,
            "replay_bundle_dir": replay_bundle_dir,
        }
        return completed_artifacts(bucket_name, submission_id)

    inputs = parse_training_submission(
        {"scenario": scenario.model_dump(mode="json")},
    )

    execution = execute_training_run(
        inputs=inputs,
        model_bucket="model-bucket",
        submission_id="submission-1",
        train_model=train_model,
        upload_model=upload_model,
    )

    assert captured["train"] == {
        "scenario_id": "scenario_demo_001",
        "job_id": "submission-1",
    }
    assert captured["upload"]["replay_bundle_dir"].endswith("replay_bundle")
    assert execution.result_bundle.artifacts.replay_bundle is not None


def test_execute_training_run_validates_summary_before_upload():
    inputs = parse_training_submission(
        {"scenario": scenario_bundle().model_dump(mode="json")},
    )
    upload_calls = []

    def upload_model(**kwargs):
        upload_calls.append(kwargs)
        return completed_artifacts("model-bucket", "submission-1")

    with pytest.raises(ValidationError):
        execute_training_run(
            inputs=inputs,
            model_bucket="model-bucket",
            submission_id="submission-1",
            train_model=lambda **_kwargs: {
                "summary": {},
                "replay_bundle_dir": "replay_bundle",
            },
            upload_model=upload_model,
        )

    assert upload_calls == []


def test_parse_training_submission_uses_continuous_runtime_spec():
    scenario = scenario_bundle()
    scenario.training.n_envs = 4
    scenario.training.cpu_count = 4
    scenario.training.torch_num_threads = 1

    inputs = parse_training_submission({"scenario": scenario.model_dump(mode="json")})

    expected_conversion = describe_runtime_conversion(scenario)
    assert inputs.conversion == expected_conversion
    assert inputs.spec.goal.x == scenario.world.goal.position.x
    assert inputs.spec.goal.z == scenario.world.goal.position.z
    assert inputs.spec.robot_start.x == scenario.robot.start_pose.position.x
    assert inputs.spec.robot_start.z == scenario.robot.start_pose.position.z
    assert inputs.training.n_envs == 4
    assert inputs.training.cpu_count == 4
    assert inputs.training.torch_num_threads == 1
