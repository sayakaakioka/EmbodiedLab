import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from embodiedlab.result_models import (
    ArtifactLocation,
    OnnxModelArtifactLocation,
    Progress,
    ReplayBundleManifest,
    ReplayLogStep,
    ReplayNamedValue,
    ReplayPosition,
    ReplayReward,
    ReplaySensorSummary,
    ResultArtifacts,
    ResultBundle,
    ResultCompatibility,
    ResultDocument,
    ResultStatus,
    SentisModelArtifactLocation,
    TrainingSummary,
    build_queued_result_document,
    build_result_bundle,
    build_result_message,
    build_result_update,
    cancelled_progress,
    cancelling_progress,
    completed_progress,
    failed_progress,
    parse_result_message,
    queued_progress,
    running_progress,
    serialize_replay_log_jsonl,
    starting_progress,
)
from tests.fakes import (
    completed_artifacts,
    resolved_training_configuration,
    scenario_bundle,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures"
SHA256 = "0" * 64


def _training_configuration():
    return resolved_training_configuration()


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
    assert "summary" not in payload
    assert payload["error"] is None
    assert "artifacts" not in payload
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
    cancelling = cancelling_progress(current_step=4, total_steps=10)
    cancelled = cancelled_progress(current_step=4, total_steps=10)

    assert queued.phase is ResultStatus.QUEUED
    assert starting.message == "Trainer job started"
    assert running.message == "Training"
    assert completed.current_step == 10
    assert completed.phase is ResultStatus.COMPLETED
    assert failed.phase is ResultStatus.FAILED
    assert failed.message == "boom"
    assert cancelling.phase is ResultStatus.CANCELLING
    assert cancelling.current_step == 4
    assert cancelled.phase is ResultStatus.CANCELLED
    assert cancelled.message == "Training cancelled"


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
    assert "artifacts" not in parsed


def test_result_message_rejects_unknown_progress_fields():
    payload = build_result_message(
        submission_id="submission-1",
        status=ResultStatus.RUNNING,
        progress=running_progress(10),
    )
    payload["progress"]["unknown"] = True

    with pytest.raises(ValidationError):
        parse_result_message(payload)


def test_progress_rejects_numeric_string_coercion():
    with pytest.raises(ValidationError):
        Progress.model_validate(
            {
                "phase": "running",
                "current_step": "1",
                "total_steps": 10,
                "message": "Training",
            }
        )


def test_replay_position_rejects_non_finite_numbers():
    with pytest.raises(ValidationError):
        ReplayPosition(x=float("nan"), z=0.0)


def test_result_document_rejects_status_phase_mismatch():
    payload = build_queued_result_document("submission-1")
    payload["status"] = "running"

    with pytest.raises(ValidationError, match="progress phase"):
        ResultDocument.model_validate(payload)


def test_result_bundle_serializes_downloadable_artifacts():
    bundle = ResultBundle(
        schema_version="result-bundle.v0",
        scenario_id="scenario_demo_001",
        job_id="job_001",
        status=ResultStatus.COMPLETED,
        compatibility=ResultCompatibility(
            scenario_schema_version="scenario-bundle.v0",
            robot_version="simple_robot.v1",
            sensor_version="basic_sensors.v0",
            action_layout=["forward", "turn"],
            observation_layout=["obs_0", "obs_1"],
        ),
        summary=TrainingSummary(
            success_rate=0.82,
            average_episode_reward=6.4,
            average_episode_steps=118.5,
            configuration=_training_configuration(),
        ),
        artifacts=ResultArtifacts(
            onnx_model=OnnxModelArtifactLocation(
                storage="gcs",
                bucket="embodiedlab-models",
                path="results/job_001/model/policy.onnx",
                format="onnx",
                size_bytes=1,
                sha256=SHA256,
                target="onnx-runtime",
                opset_version=17,
                inputs=[
                    {
                        "name": "obs_0",
                        "shape": [-1, 3, 84, 112],
                        "dtype": "float32",
                        "layout": [
                            "channel_0_unused",
                            "channel_1_traversable",
                            "channel_2_blocked_or_background",
                        ],
                    },
                    {
                        "name": "obs_1",
                        "shape": [-1, 2],
                        "dtype": "float32",
                        "layout": [
                            "goal_angle_degrees",
                            "goal_distance_meters",
                        ],
                    },
                ],
                output={
                    "name": "action",
                    "layout": ["forward", "turn"],
                    "action_mapping": None,
                },
            ),
            sentis_model=SentisModelArtifactLocation(
                storage="gcs",
                bucket="embodiedlab-models",
                path="results/job_001/model/policy.sentis.onnx",
                format="onnx",
                size_bytes=1,
                sha256=SHA256,
                target="unity-sentis",
                opset_version=15,
                inputs=[
                    {
                        "name": "observation",
                        "shape": [1, 28226],
                        "dtype": "float32",
                        "layout": [
                            "obs_0_chw_3x84x112",
                            "obs_1_goal_angle_degrees",
                            "obs_1_goal_distance_meters",
                        ],
                    },
                ],
                output={
                    "name": "action",
                    "layout": ["forward", "turn"],
                    "action_mapping": None,
                },
            ),
            replay_bundle=ArtifactLocation(
                storage="gcs",
                bucket="embodiedlab-models",
                path="results/job_001/replay/manifest.json",
                format="json",
                size_bytes=1,
                sha256=SHA256,
            ),
        ),
        error=None,
    )

    payload = bundle.model_dump(mode="json")

    assert payload["schema_version"] == "result-bundle.v0"
    assert payload["compatibility"]["action_layout"] == ["forward", "turn"]
    assert "model" not in payload["artifacts"]
    assert payload["artifacts"]["onnx_model"]["path"].endswith("policy.onnx")
    assert payload["artifacts"]["sentis_model"]["target"] == "unity-sentis"
    assert payload["artifacts"]["sentis_model"]["inputs"][0]["shape"] == [
        1,
        28226,
    ]
    assert payload["artifacts"]["replay_bundle"]["format"] == "json"


def test_model_artifact_rejects_non_onnx_format():
    with pytest.raises(ValidationError):
        OnnxModelArtifactLocation.model_validate(
            {
                "storage": "gcs",
                "bucket": "embodiedlab-models",
                "path": "results/job_001/model/policy.onnx",
                "format": "json",
                "size_bytes": 1,
                "sha256": SHA256,
                "target": "onnx-runtime",
                "opset_version": 17,
                "inputs": [
                    {
                        "name": "obs_0",
                        "shape": [-1, 2],
                        "dtype": "float32",
                        "layout": ["goal_angle_degrees", "goal_distance_meters"],
                    }
                ],
                "output": {
                    "name": "action",
                    "layout": ["forward", "turn"],
                    "action_mapping": None,
                },
            }
        )


def test_replay_manifest_artifact_rejects_non_json_format():
    with pytest.raises(ValidationError):
        ArtifactLocation.model_validate(
            {
                "storage": "gcs",
                "bucket": "embodiedlab-models",
                "path": "results/job_001/replay/manifest.json",
                "format": "onnx",
                "size_bytes": 1,
                "sha256": SHA256,
            }
        )


def test_completed_result_document_fixture_matches_contract():
    fixture_path = (
        FIXTURE_DIR / "envforge" / "navigation_completed_result_document.json"
    )
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))

    document = ResultDocument.model_validate(payload)
    result_bundle = ResultBundle.model_validate(document.result_bundle)

    assert document.model_dump(mode="json") == payload
    assert result_bundle.model_dump(mode="json") == payload["result_bundle"]
    assert "artifacts" not in payload
    assert result_bundle.artifacts.onnx_model is not None
    assert result_bundle.artifacts.sentis_model is not None
    assert result_bundle.artifacts.replay_bundle is not None


def test_result_document_rejects_unknown_top_level_fields():
    with pytest.raises(ValidationError):
        ResultDocument.model_validate(
            {
                "submission_id": "submission-1",
                "status": "completed",
                "private_control": {"secret": "must-not-leak"},
            },
        )


def test_result_bundle_rejects_unknown_top_level_fields():
    with pytest.raises(ValidationError):
        ResultBundle.model_validate(
            {
                "scenario_id": "scenario_demo_001",
                "job_id": "job_001",
                "status": "completed",
                "legacy_artifacts": {},
            },
        )


def test_completed_result_bundle_requires_summary_and_every_artifact():
    with pytest.raises(ValidationError, match="training summary"):
        build_result_bundle(
            scenario=scenario_bundle(),
            job_id="job_001",
            status=ResultStatus.COMPLETED,
            artifacts=completed_artifacts("bucket", "job_001"),
        )


def test_failed_result_bundle_requires_error_report():
    with pytest.raises(ValidationError, match="error report"):
        build_result_bundle(
            scenario=scenario_bundle(),
            job_id="job_001",
            status=ResultStatus.FAILED,
        )


def test_result_observation_layout_is_camera_then_goal_vector():
    scenario = scenario_bundle()
    reordered = scenario.model_copy(
        update={"sensors": list(reversed(scenario.sensors))},
    )

    bundle = build_result_bundle(
        scenario=reordered,
        job_id="job_001",
        status=ResultStatus.FAILED,
        error="boom",
    )

    assert bundle.compatibility.observation_layout == ["obs_0", "obs_1"]


def test_replay_log_step_serializes_jsonl_row():
    step = ReplayLogStep(
        schema_version="replay-log.v0",
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
                type="camera_mount_height_meters",
                value=0.42,
            ),
        ],
        events=[],
        terminated=False,
        termination_reason=None,
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
        "type": "camera_mount_height_meters",
        "value": 0.42,
    }
    assert payload["terminated"] is False


@pytest.mark.parametrize(
    "paths",
    [
        ["eval/chunk.jsonl.gz", "eval/chunk.jsonl.gz"],
        ["../chunk.jsonl.gz"],
        ["eval/../chunk.jsonl.gz"],
        ["manifest.json"],
        ["eval\\chunk.jsonl.gz"],
        ["train/chunk.jsonl.gz"],
    ],
)
def test_replay_bundle_manifest_rejects_unsafe_chunk_paths(paths):
    chunks = [
        {
            "phase": "eval",
            "policy_mode": "deterministic",
            "checkpoint_step": index,
            "start_step": None,
            "end_step": None,
            "path": path,
            "format": "jsonl.gz",
            "size_bytes": 1,
            "sha256": SHA256,
            "step_count": 1,
            "episode_count": 1,
            "success_rate": 0.0,
            "avg_reward": 0.0,
            "avg_steps": 1.0,
        }
        for index, path in enumerate(paths)
    ]

    with pytest.raises(ValidationError):
        ReplayBundleManifest(
            schema_version="replay-bundle.v0",
            job_id="job_001",
            scenario_id="scenario_demo_001",
            total_timesteps=1,
            chunks=chunks,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("total_timesteps", 0),
        ("chunks", []),
    ],
)
def test_replay_bundle_manifest_requires_nonempty_training_output(field, value):
    fixture_path = FIXTURE_DIR / "envforge" / "navigation_replay_bundle_manifest.json"
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    payload[field] = value

    with pytest.raises(ValidationError):
        ReplayBundleManifest.model_validate(payload)


def test_replay_bundle_manifest_rejects_checkpoint_after_total_timesteps():
    fixture_path = FIXTURE_DIR / "envforge" / "navigation_replay_bundle_manifest.json"
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    payload["chunks"][-1]["checkpoint_step"] = payload["total_timesteps"] + 1

    with pytest.raises(ValidationError, match="checkpoint_step must not exceed"):
        ReplayBundleManifest.model_validate(payload)


def test_serialize_replay_log_jsonl_returns_json_lines():
    step = ReplayLogStep(
        schema_version="replay-log.v0",
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
        action={
            "values": [
                {"name": "forward", "value": 0.0},
                {"name": "turn", "value": 0.0},
            ],
        },
        reward=ReplayReward(total=0.0, components=[]),
        events=[],
        sensors=[],
        terminated=False,
        termination_reason=None,
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
    assert steps[1].sensors == []


def test_build_result_bundle_maps_replay_bundle_artifact_metadata():
    bundle = build_result_bundle(
        scenario=scenario_bundle(),
        job_id="job_001",
        status=ResultStatus.COMPLETED,
        summary={
            "training_configuration": _training_configuration(),
        },
        artifacts={
            "onnx_model": {
                "storage": "gcs",
                "bucket": "embodiedlab-models",
                "path": "results/job_001/model/policy.onnx",
                "format": "onnx",
                "size_bytes": 1,
                "sha256": SHA256,
                "target": "onnx-runtime",
                "opset_version": 17,
                "inputs": [
                    {
                        "name": "obs_0",
                        "shape": [-1, 3, 84, 112],
                        "dtype": "float32",
                        "layout": [
                            "channel_0_unused",
                            "channel_1_traversable",
                            "channel_2_blocked_or_background",
                        ],
                    },
                    {
                        "name": "obs_1",
                        "shape": [-1, 2],
                        "dtype": "float32",
                        "layout": [
                            "goal_angle_degrees",
                            "goal_distance_meters",
                        ],
                    },
                ],
                "output": {
                    "name": "action",
                    "layout": ["forward", "turn"],
                    "action_mapping": {
                        "forward": "sigmoid(policy_forward)",
                        "turn": "clip(policy_turn, -3, 3) / 3",
                    },
                },
            },
            "sentis_model": {
                "storage": "gcs",
                "bucket": "embodiedlab-models",
                "path": "results/job_001/model/policy.sentis.onnx",
                "format": "onnx",
                "size_bytes": 1,
                "sha256": SHA256,
                "target": "unity-sentis",
                "opset_version": 15,
                "inputs": [
                    {
                        "name": "observation",
                        "shape": [1, 28226],
                        "dtype": "float32",
                        "layout": [
                            "obs_0_chw_3x84x112",
                            "obs_1_goal_angle_degrees",
                            "obs_1_goal_distance_meters",
                        ],
                    },
                ],
                "output": {
                    "name": "action",
                    "layout": ["forward", "turn"],
                    "action_mapping": {
                        "forward": "sigmoid(policy_forward)",
                        "turn": "clip(policy_turn, -3, 3) / 3",
                    },
                },
            },
            "replay_bundle": {
                "storage": "gcs",
                "bucket": "embodiedlab-models",
                "path": "results/job_001/replay/manifest.json",
                "format": "json",
                "size_bytes": 1,
                "sha256": SHA256,
            },
        },
    )

    payload = bundle.model_dump(mode="json")

    assert "model" not in payload["artifacts"]
    assert payload["artifacts"]["replay_bundle"] == {
        "storage": "gcs",
        "bucket": "embodiedlab-models",
        "path": "results/job_001/replay/manifest.json",
        "format": "json",
        "size_bytes": 1,
        "sha256": SHA256,
    }
