"""Result status, progress, and document models shared by the API and trainer."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

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


class Pose2D(BaseModel):
    """Robot pose on the replay x/z plane."""

    x: float
    z: float
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
    robot: Pose2D
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
    updated_at: str = Field(default_factory=utc_now_iso)


class ResultMessage(BaseModel):
    """Pub/Sub message payload emitted after each status transition."""

    submission_id: str
    status: ResultStatus
    progress: Progress | None = None
    summary: dict[str, Any] | None = None
    error: str | None = None
    artifacts: dict[str, Any] | None = None
    updated_at: str = Field(default_factory=utc_now_iso)


class ResultUpdate(BaseModel):
    """Partial update applied to an existing result document."""

    status: ResultStatus
    progress: Progress
    summary: dict[str, Any] | None = None
    error: str | None = None
    artifacts: dict[str, Any] | None = None
    updated_at: str = Field(default_factory=utc_now_iso)


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


def build_result_update(
    *,
    status: ResultStatus,
    progress: dict | Progress,
    summary: dict[str, Any] | None = None,
    error: str | None = None,
    artifacts: dict[str, Any] | None = None,
) -> dict:
    """Return a Firestore-ready dict for a partial result update."""
    update = ResultUpdate(
        status=status,
        progress=progress,
        summary=summary,
        error=error,
        artifacts=artifacts,
    )
    return update.model_dump(mode="json")


def build_result_message(  # noqa: PLR0913
    submission_id: str,
    status: ResultStatus,
    progress: Progress,
    summary: dict[str, Any] | None = None,
    error: str | None = None,
    artifacts: dict[str, Any] | None = None,
) -> dict:
    """Return a dict suitable for publishing as a Pub/Sub message."""
    message = ResultMessage(
        submission_id=submission_id,
        status=status,
        progress=progress,
        summary=summary,
        error=error,
        artifacts=artifacts,
    )
    return message.model_dump(mode="json")


def parse_result_message(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a result event payload and return its normalized JSON form."""
    message = ResultMessage.model_validate(payload)
    return message.model_dump(mode="json", exclude_unset=True)
