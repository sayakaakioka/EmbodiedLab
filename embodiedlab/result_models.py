"""Result status, progress, and document models shared by the API and trainer."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Iterable

    from embodiedlab.schemas import ScenarioBundle

RESULT_SCHEMA_VERSION = "result-bundle.v0"
REPLAY_LOG_SCHEMA_VERSION = "replay-log.v0"


class ResultStatus(StrEnum):
    """Lifecycle states of a training result."""

    QUEUED = "queued"
    STARTING = "starting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ArtifactStorage(StrEnum):
    """Supported artifact storage backends."""

    GCS = "gcs"


class ArtifactFormat(StrEnum):
    """Supported artifact formats."""

    ONNX = "onnx"
    JSONL = "jsonl"
    ZIP = "zip"


class ArtifactLocation(BaseModel):
    """Location and format of a result artifact."""

    storage: ArtifactStorage = ArtifactStorage.GCS
    bucket: str = Field(min_length=1)
    path: str = Field(min_length=1)
    format: ArtifactFormat


class ResultCompatibility(BaseModel):
    """Compatibility metadata needed by EnvForge when loading a result."""

    scenario_schema_version: str = Field(default="scenario-bundle.v0", min_length=1)
    envforge_min_version: str = Field(default="0.1.0", min_length=1)
    robot_version: str = Field(default="simple_robot.v0", min_length=1)
    sensor_version: str = Field(default="basic_sensors.v0", min_length=1)
    action_layout: list[str] = Field(default_factory=lambda: ["forward", "turn"])
    observation_layout: list[str] = Field(
        default_factory=lambda: [
            "front_camera_semantic",
            "front_distance",
        ],
    )


class TrainingSummary(BaseModel):
    """High-level metrics from a completed training run."""

    training_timesteps: int = Field(ge=0)
    training_seed: int
    success_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    average_episode_reward: float | None = None
    average_episode_steps: float | None = Field(default=None, ge=0.0)


class ResultArtifacts(BaseModel):
    """Artifacts produced by a training run."""

    model: ArtifactLocation | None = None
    replay_log: ArtifactLocation | None = None


class ErrorReport(BaseModel):
    """Structured failure details for failed result bundles."""

    message: str = Field(min_length=1)
    details: str | None = None


class ResultBundle(BaseModel):
    """EnvForge-facing training result bundle."""

    schema_version: str = RESULT_SCHEMA_VERSION
    scenario_id: str = Field(min_length=1)
    job_id: str = Field(min_length=1)
    status: ResultStatus
    compatibility: ResultCompatibility = Field(default_factory=ResultCompatibility)
    summary: TrainingSummary | None = None
    artifacts: ResultArtifacts = Field(default_factory=ResultArtifacts)
    error: ErrorReport | None = None


class ReplayPosition(BaseModel):
    """A continuous replay position on the x/z plane."""

    x: float
    z: float


class ReplayRobotState(BaseModel):
    """Robot state emitted in a replay step."""

    position: ReplayPosition
    rotation_y_degrees: float


class ReplayReward(BaseModel):
    """Reward values emitted for a replay step."""

    total: float
    components: dict[str, float] = Field(default_factory=dict)


class ReplayLogStep(BaseModel):
    """One JSON Lines row in an EnvForge replay log."""

    schema_version: str = REPLAY_LOG_SCHEMA_VERSION
    scenario_id: str = Field(min_length=1)
    job_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    step_index: int = Field(ge=0)
    time_seconds: float = Field(ge=0.0)
    robot: ReplayRobotState
    action: dict[str, float] = Field(default_factory=dict)
    reward: ReplayReward
    events: list[dict[str, Any]] = Field(default_factory=list)
    sensors: dict[str, Any] = Field(default_factory=dict)
    terminated: bool = False
    termination_reason: str | None = None


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(UTC).isoformat()


class Progress(BaseModel):
    """Training progress snapshot stored in each result document."""

    phase: ResultStatus
    current_step: int = Field(ge=0)
    total_steps: int = Field(ge=0)
    message: str


class ResultDocument(BaseModel):
    """Full result document written to Firestore."""

    submission_id: str
    status: ResultStatus
    progress: Progress
    summary: dict[str, Any] | None = None
    error: str | None = None
    artifacts: dict[str, Any] | None = None
    result_bundle: dict[str, Any] | ResultBundle | None = None
    updated_at: str = Field(default_factory=utc_now_iso)


class ResultMessage(BaseModel):
    """Pub/Sub message payload emitted after each status transition."""

    submission_id: str
    status: ResultStatus
    progress: Progress | None = None
    summary: dict[str, Any] | None = None
    error: str | None = None
    artifacts: dict[str, Any] | None = None
    result_bundle: dict[str, Any] | ResultBundle | None = None
    updated_at: str = Field(default_factory=utc_now_iso)


class ResultUpdate(BaseModel):
    """Partial update applied to an existing result document."""

    status: ResultStatus
    progress: Progress
    summary: dict[str, Any] | None = None
    error: str | None = None
    artifacts: dict[str, Any] | None = None
    result_bundle: dict[str, Any] | ResultBundle | None = None
    updated_at: str = Field(default_factory=utc_now_iso)


def _artifact_from_payload(
    payload: dict[str, Any] | None,
    *,
    default_format: ArtifactFormat,
) -> ArtifactLocation | None:
    """Convert an uploaded artifact dict into a ResultBundle location."""
    if payload is None:
        return None

    artifact_format = payload.get("format", default_format)
    return ArtifactLocation(
        storage=payload.get("storage", ArtifactStorage.GCS),
        bucket=payload["bucket"],
        path=payload["path"],
        format=artifact_format,
    )


def build_result_compatibility(scenario: ScenarioBundle) -> ResultCompatibility:
    """Build EnvForge compatibility metadata from the submitted scenario."""
    sensor_layout = [sensor.id for sensor in scenario.sensors]
    return ResultCompatibility(
        scenario_schema_version=scenario.schema_version,
        envforge_min_version=scenario.compatibility.envforge_min_version,
        robot_version=scenario.compatibility.robot_version,
        sensor_version=scenario.compatibility.sensor_version,
        action_layout=list(scenario.robot.action_space.layout),
        observation_layout=sensor_layout,
    )


def build_training_summary(summary: dict[str, Any]) -> TrainingSummary:
    """Normalize the current runner summary into the ResultBundle summary."""
    return TrainingSummary(
        training_timesteps=summary["training_timesteps"],
        training_seed=summary["training_seed"],
        success_rate=summary.get("success_rate"),
        average_episode_reward=summary.get(
            "average_episode_reward",
            summary.get("avg_reward"),
        ),
        average_episode_steps=summary.get(
            "average_episode_steps",
            summary.get("avg_steps"),
        ),
    )


def build_result_bundle(  # noqa: PLR0913
    *,
    scenario: ScenarioBundle,
    job_id: str,
    status: ResultStatus,
    summary: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
    error: str | None = None,
) -> ResultBundle:
    """Build the EnvForge-facing ResultBundle from trainer outputs."""
    artifacts = artifacts or {}
    result_error = ErrorReport(message=error) if error is not None else None
    model_payload = artifacts.get("onnx_model")
    model_format = ArtifactFormat.ONNX
    if model_payload is None:
        model_payload = artifacts.get("model")
        model_format = ArtifactFormat.ZIP

    return ResultBundle(
        scenario_id=scenario.scenario_id,
        job_id=job_id,
        status=status,
        compatibility=build_result_compatibility(scenario),
        summary=build_training_summary(summary) if summary is not None else None,
        artifacts=ResultArtifacts(
            model=_artifact_from_payload(
                model_payload,
                default_format=model_format,
            ),
            replay_log=_artifact_from_payload(
                artifacts.get("replay_log"),
                default_format=ArtifactFormat.JSONL,
            ),
        ),
        error=result_error,
    )


def serialize_replay_log_jsonl(steps: Iterable[ReplayLogStep]) -> str:
    """Serialize replay steps to the JSON Lines artifact format."""
    lines = [step.model_dump_json() for step in steps]
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def build_progress(
    *,
    phase: ResultStatus,
    current_step: int,
    total_steps: int,
    message: str,
) -> Progress:
    """Build a Progress model from explicit values."""
    return Progress(
        phase=phase,
        current_step=current_step,
        total_steps=total_steps,
        message=message,
    )


def queued_progress() -> Progress:
    """Return the queued-phase progress payload."""
    return build_progress(
        phase=ResultStatus.QUEUED,
        current_step=0,
        total_steps=0,
        message="Queued",
    )


def starting_progress(total_steps: int) -> Progress:
    """Return the starting-phase progress payload."""
    return build_progress(
        phase=ResultStatus.STARTING,
        current_step=0,
        total_steps=total_steps,
        message="Trainer job started",
    )


def running_progress(total_steps: int) -> Progress:
    """Return the running-phase progress payload."""
    return build_progress(
        phase=ResultStatus.RUNNING,
        current_step=0,
        total_steps=total_steps,
        message="Training",
    )


def completed_progress(total_steps: int) -> Progress:
    """Return the completed-phase progress payload."""
    return build_progress(
        phase=ResultStatus.COMPLETED,
        current_step=total_steps,
        total_steps=total_steps,
        message="Training completed",
    )


def failed_progress(message: str, total_steps: int = 0) -> Progress:
    """Return the failed-phase progress payload."""
    return build_progress(
        phase=ResultStatus.FAILED,
        current_step=0,
        total_steps=total_steps,
        message=message,
    )


def build_queued_result_document(submission_id: str) -> dict:
    """Return a Firestore-ready dict for a newly queued result."""
    document = ResultDocument(
        submission_id=submission_id,
        status=ResultStatus.QUEUED,
        progress=queued_progress(),
    )
    return document.model_dump(mode="json")


def build_result_update(  # noqa: PLR0913
    *,
    status: ResultStatus,
    progress: dict | Progress,
    summary: dict[str, Any] | None = None,
    error: str | None = None,
    artifacts: dict[str, Any] | None = None,
    result_bundle: dict[str, Any] | ResultBundle | None = None,
) -> dict:
    """Return a Firestore-ready dict for a partial result update."""
    update = ResultUpdate(
        status=status,
        progress=progress,
        summary=summary,
        error=error,
        artifacts=artifacts,
        result_bundle=result_bundle,
    )
    return update.model_dump(mode="json")


def build_result_message(  # noqa: PLR0913
    submission_id: str,
    status: ResultStatus,
    progress: Progress,
    summary: dict[str, Any] | None = None,
    error: str | None = None,
    artifacts: dict[str, Any] | None = None,
    result_bundle: dict[str, Any] | ResultBundle | None = None,
) -> dict:
    """Return a dict suitable for publishing as a Pub/Sub message."""
    message = ResultMessage(
        submission_id=submission_id,
        status=status,
        progress=progress,
        summary=summary,
        error=error,
        artifacts=artifacts,
        result_bundle=result_bundle,
    )
    return message.model_dump(mode="json")


def parse_result_message(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a result event payload and return its normalized JSON form."""
    message = ResultMessage.model_validate(payload)
    return message.model_dump(mode="json", exclude_unset=True)
