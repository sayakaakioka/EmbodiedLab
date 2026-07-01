import json
from pathlib import Path

from embodiedlab.result_models import (
    ArtifactLocation,
    ModelArtifactLocation,
    ReplayLogStep,
    ReplayNamedValue,
    ReplayReward,
    ReplaySensorSummary,
    ResultArtifacts,
    ResultBundle,
    ResultCompatibility,
    ResultStatus,
    TrainingSummary,
    build_queued_result_document,
    build_result_bundle,
    build_result_message,
    build_result_update,
    completed_progress,
    failed_progress,
    parse_result_message,
    queued_progress,
    running_progress,
    serialize_replay_log_jsonl,
    starting_progress,
)
from embodiedlab.schemas import ScenarioBundle

FIXTURE_DIR = Path(__file__).parent / "fixtures"


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


def test_result_bundle_serializes_envforge_artifacts():
    bundle = ResultBundle(
        scenario_id="scenario_demo_001",
        job_id="job_001",
        status=ResultStatus.COMPLETED,
        compatibility=ResultCompatibility(),
        summary=TrainingSummary(
            training_timesteps=5000,
            training_seed=10,
            success_rate=0.82,
            average_episode_reward=6.4,
            average_episode_steps=118.5,
        ),
        artifacts=ResultArtifacts(
            model=ArtifactLocation(
                bucket="embodiedlab-models",
                path="results/job_001/model/policy.onnx",
                format="onnx",
            ),
            onnx_model=ArtifactLocation(
                bucket="embodiedlab-models",
                path="results/job_001/model/policy.onnx",
                format="onnx",
            ),
            sentis_model=ModelArtifactLocation(
                bucket="embodiedlab-models",
                path="results/job_001/model/policy.sentis.onnx",
                format="onnx",
                target="unity-sentis",
                opset_version=15,
                input={
                    "name": "observation",
                    "shape": [1, 7],
                    "dtype": "float32",
                    "layout": ["robot_x", "robot_z", "front_distance"],
                },
                output={
                    "name": "action",
                    "layout": ["forward", "turn"],
                },
            ),
            replay_bundle=ArtifactLocation(
                bucket="embodiedlab-models",
                path="results/job_001/replay/manifest.json",
                format="json",
            ),
        ),
    )

    payload = bundle.model_dump(mode="json")

    assert payload["schema_version"] == "result-bundle.v0"
    assert payload["compatibility"]["action_layout"] == ["forward", "turn"]
    assert payload["artifacts"]["model"]["format"] == "onnx"
    assert payload["artifacts"]["onnx_model"]["path"].endswith("policy.onnx")
    assert payload["artifacts"]["sentis_model"]["target"] == "unity-sentis"
    assert payload["artifacts"]["sentis_model"]["input"]["shape"] == [1, 7]
    assert payload["artifacts"]["replay_bundle"]["format"] == "json"


def test_replay_log_step_serializes_jsonl_row():
    step = ReplayLogStep(
        scenario_id="scenario_demo_001",
        job_id="job_001",
        phase="eval",
        checkpoint_step=1000,
        env_index=0,
        policy_mode="deterministic",
        episode_id="eval_env_00_episode_000001",
        step_index=1,
        time_seconds=0.1,
        robot={
            "position": {
                "x": 1.02,
                "z": 1.0,
            },
            "rotation_y_degrees": 0.0,
        },
        action={
            "values": [
                {"name": "forward", "value": 0.2},
                {"name": "turn", "value": 0.0},
            ],
        },
        reward=ReplayReward(
            total=0.04,
            components=[
                ReplayNamedValue(name="goal_progress", value=0.05),
                ReplayNamedValue(name="step_penalty", value=-0.01),
            ],
        ),
        sensors=[
            ReplaySensorSummary(
                id="front_distance",
                type="distance_meters",
                value=5.0,
            ),
            ReplaySensorSummary(
                id="camera_mount_height",
                type="envforge_camera_mount_height_meters",
                value=0.42,
            ),
        ],
    )

    payload = step.model_dump(mode="json")

    assert payload["schema_version"] == "replay-log.v0"
    assert payload["robot"]["position"]["x"] == 1.02
    assert payload["action"]["values"][0] == {"name": "forward", "value": 0.2}
    assert payload["reward"]["components"][0] == {
        "name": "goal_progress",
        "value": 0.05,
    }
    assert payload["sensors"][0] == {
        "id": "front_distance",
        "type": "distance_meters",
        "value": 5.0,
    }
    assert payload["sensors"][1] == {
        "id": "camera_mount_height",
        "type": "envforge_camera_mount_height_meters",
        "value": 0.42,
    }
    assert payload["terminated"] is False


def test_serialize_replay_log_jsonl_returns_json_lines():
    step = ReplayLogStep(
        scenario_id="scenario_demo_001",
        job_id="job_001",
        phase="train",
        checkpoint_step=0,
        env_index=1,
        policy_mode="stochastic",
        episode_id="train_env_01_episode_000001",
        step_index=0,
        time_seconds=0.0,
        robot={
            "position": {"x": 1.0, "z": 1.0},
            "rotation_y_degrees": 0.0,
        },
        reward=ReplayReward(total=0.0),
    )

    payload = serialize_replay_log_jsonl([step])

    assert payload.endswith("\n")
    rows = payload.splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["schema_version"] == "replay-log.v0"


def test_envforge_navigation_replay_fixture_matches_contract():
    fixture_path = FIXTURE_DIR / "envforge" / "navigation_default_replay_log.jsonl"
    rows = fixture_path.read_text(encoding="utf-8").splitlines()

    steps = [ReplayLogStep.model_validate_json(row) for row in rows]

    assert len(steps) == 2
    assert steps[0].scenario_id == "navigation_default"
    assert steps[0].action.values[0].name == "forward"
    assert steps[1].reward.components[0].name == "goal_progress"
    assert steps[1].sensors[0].id == "front_distance"


def test_build_result_bundle_maps_replay_bundle_artifact_metadata():
    bundle = build_result_bundle(
        scenario=ScenarioBundle(),
        job_id="job_001",
        status=ResultStatus.COMPLETED,
        summary={
            "training_timesteps": 5000,
            "training_seed": 10,
        },
        artifacts={
            "onnx_model": {
                "storage": "gcs",
                "bucket": "embodiedlab-models",
                "path": "results/job_001/model/policy.onnx",
            },
            "replay_bundle": {
                "storage": "gcs",
                "bucket": "embodiedlab-models",
                "path": "results/job_001/replay/manifest.json",
                "format": "json",
            },
        },
    )

    payload = bundle.model_dump(mode="json")

    assert payload["artifacts"]["model"] == {
        "storage": "gcs",
        "bucket": "embodiedlab-models",
        "path": "results/job_001/model/policy.onnx",
        "format": "onnx",
    }
    assert payload["artifacts"]["replay_bundle"] == {
        "storage": "gcs",
        "bucket": "embodiedlab-models",
        "path": "results/job_001/replay/manifest.json",
        "format": "json",
    }
