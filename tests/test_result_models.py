from embodiedlab.result_models import (
    ArtifactLocation,
    ReplayLogStep,
    ReplayReward,
    ResultArtifacts,
    ResultBundle,
    ResultCompatibility,
    ResultStatus,
    TrainingSummary,
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
            replay_log=ArtifactLocation(
                bucket="embodiedlab-models",
                path="results/job_001/replay/replay.jsonl",
                format="jsonl",
            ),
        ),
    )

    payload = bundle.model_dump(mode="json")

    assert payload["schema_version"] == "result-bundle.v0"
    assert payload["compatibility"]["action_layout"] == ["forward", "turn"]
    assert payload["artifacts"]["model"]["format"] == "onnx"
    assert payload["artifacts"]["replay_log"]["format"] == "jsonl"


def test_replay_log_step_serializes_jsonl_row():
    step = ReplayLogStep(
        scenario_id="scenario_demo_001",
        job_id="job_001",
        episode_id="episode_0001",
        step_index=1,
        time_seconds=0.1,
        robot={
            "x": 1.02,
            "z": 1.0,
            "rotation_y_degrees": 0.0,
        },
        action={
            "forward": 0.2,
            "turn": 0.0,
        },
        reward=ReplayReward(
            total=0.04,
            components={
                "goal_progress": 0.05,
                "step_penalty": -0.01,
            },
        ),
        sensors={"front_distance": 5.0},
    )

    payload = step.model_dump(mode="json")

    assert payload["schema_version"] == "replay-log.v0"
    assert payload["robot"]["x"] == 1.02
    assert payload["reward"]["components"]["goal_progress"] == 0.05
    assert payload["terminated"] is False
