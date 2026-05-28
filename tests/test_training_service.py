from embodiedlab.schemas import ScenarioBundle
from embodiedlab.training.training_converter import describe_runtime_conversion
from trainer.training_service import (
    execute_training_run_from_submission,
    parse_training_submission,
)


def test_execute_training_run_uploads_replay_steps():
    scenario = ScenarioBundle()
    captured_replay_steps = []

    def train_model(*, spec, training, model_output_path):
        return {
            "score": 1.0,
            "replay_steps": [
                {
                    "episode_id": "episode_0001",
                    "step_index": 0,
                    "time_seconds": 0.0,
                    "robot": {
                        "position": {"x": 1.0, "z": 1.0},
                        "rotation_y_degrees": 0.0,
                    },
                    "action": {
                        "values": [
                            {"name": "forward", "value": 1.0},
                            {"name": "turn", "value": 0.0},
                        ],
                    },
                    "reward": {
                        "total": -0.2,
                        "components": [
                            {"name": "step_penalty", "value": -0.2},
                        ],
                    },
                    "events": [],
                    "sensors": [
                        {
                            "id": "front_distance",
                            "type": "envforge_manhattan_distance_meters",
                            "value": 14.0,
                        },
                    ],
                    "terminated": False,
                    "termination_reason": None,
                },
            ],
        }

    def upload_model(  # noqa: PLR0913
        *,
        local_model_base_path,
        bucket_name,
        submission_id,
        replay_steps,
        export_onnx,
        model_export_layout,
    ):
        captured_replay_steps.extend(replay_steps)
        return {
            "replay_log": {
                "bucket": bucket_name,
                "path": f"results/{submission_id}/replay/replay.jsonl",
                "format": "jsonl",
            },
        }

    execution = execute_training_run_from_submission(
        submission={"scenario": scenario.model_dump(mode="json")},
        model_bucket="model-bucket",
        submission_id="submission-1",
        train_model=train_model,
        upload_model=upload_model,
    )

    assert "replay_steps" not in execution.summary
    assert captured_replay_steps[0].schema_version == "replay-log.v0"
    assert captured_replay_steps[0].scenario_id == "scenario_demo_001"
    assert captured_replay_steps[0].job_id == "submission-1"
    assert execution.result_bundle.artifacts.replay_log is not None


def test_parse_training_submission_uses_continuous_runtime_spec():
    scenario = ScenarioBundle()

    inputs = parse_training_submission({"scenario": scenario.model_dump(mode="json")})

    expected_conversion = describe_runtime_conversion(scenario)
    assert inputs.conversion == expected_conversion
    assert inputs.spec.goal.x == scenario.world.goal.position.x
    assert inputs.spec.goal.z == scenario.world.goal.position.z
    assert inputs.spec.robot_start.x == scenario.robot.start_pose.position.x
    assert inputs.spec.robot_start.z == scenario.robot.start_pose.position.z
