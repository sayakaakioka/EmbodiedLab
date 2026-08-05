"""Result status, progress, and document models shared by the API and trainer."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import ConfigDict, Field, model_validator

from embodiedlab.schemas import (
    MAX_IDENTIFIER_LENGTH,
    MAX_PARALLEL_ENVS,
    MAX_PPO_EPOCHS,
    MAX_PPO_ROLLOUT_STEPS,
    MAX_RANDOM_SEED,
    MAX_REPLAY_CHUNK_STEPS,
    MAX_STATS_WINDOW_SIZE,
    MAX_TRAINING_CPU_COUNT,
    MAX_TRAINING_TIMESTEPS,
    ContractModel,
    ForwardCameraSensor,
    GoalVectorSensor,
    ScenarioBundle,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


RESULT_SCHEMA_VERSION = "result-bundle.v0"
REPLAY_LOG_SCHEMA_VERSION = "replay-log.v0"
REPLAY_BUNDLE_SCHEMA_VERSION = "replay-bundle.v0"
MAX_REPLAY_BUNDLE_CHUNKS = 4_096
MAX_REPLAY_CHUNK_PATH_LENGTH = 1_024
MAX_ARTIFACT_PATH_LENGTH = 1_024
MAX_MODEL_IO_ENTRIES = 16
MAX_LAYOUT_ENTRIES = 256
MAX_REPLAY_EVENTS = 256
MAX_REPLAY_SENSORS = 32
MAX_REPLAY_REWARD_COMPONENTS = 64


class ResultStatus(StrEnum):
    """Lifecycle states of a training result."""

    QUEUED = "queued"
    STARTING = "starting"
    RUNNING = "running"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"


class ArtifactStorage(StrEnum):
    """Supported artifact storage backends."""

    GCS = "gcs"


class ArtifactLocation(ContractModel):
    """Location and format of a result artifact."""

    model_config = ConfigDict(extra="forbid")

    storage: ArtifactStorage
    bucket: str = Field(min_length=3, max_length=63)
    path: str = Field(min_length=1, max_length=MAX_ARTIFACT_PATH_LENGTH)
    format: Literal["json"]
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ModelInput(ContractModel):
    """Input metadata for a client-loadable model artifact."""

    name: str = Field(min_length=1, max_length=MAX_IDENTIFIER_LENGTH)
    model_config = ConfigDict(extra="forbid")

    shape: list[int] = Field(min_length=1, max_length=8)
    dtype: str = Field(min_length=1, max_length=32)
    layout: list[str] = Field(max_length=MAX_LAYOUT_ENTRIES)


class ModelOutput(ContractModel):
    """Output metadata for a client-loadable model artifact."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=MAX_IDENTIFIER_LENGTH)
    layout: list[str] = Field(min_length=1, max_length=MAX_LAYOUT_ENTRIES)
    action_mapping: dict[str, str] | None


class OnnxModelArtifactLocation(ArtifactLocation):
    """Canonical ONNX Runtime artifact metadata."""

    format: Literal["onnx"]
    target: Literal["onnx-runtime"]
    opset_version: Literal[18]
    inputs: list[ModelInput] = Field(
        min_length=1,
        max_length=MAX_MODEL_IO_ENTRIES,
    )
    output: ModelOutput


class ResultCompatibility(ContractModel):
    """Compatibility metadata needed by clients when loading a result."""

    model_config = ConfigDict(extra="forbid")

    scenario_schema_version: str = Field(
        min_length=1,
        max_length=MAX_IDENTIFIER_LENGTH,
    )
    robot_version: str = Field(min_length=1, max_length=MAX_IDENTIFIER_LENGTH)
    sensor_version: str = Field(min_length=1, max_length=MAX_IDENTIFIER_LENGTH)
    action_layout: list[str] = Field(min_length=1, max_length=MAX_LAYOUT_ENTRIES)
    observation_layout: list[str] = Field(
        min_length=1,
        max_length=MAX_LAYOUT_ENTRIES,
    )


class TrainingSummary(ContractModel):
    """High-level metrics from a completed training run."""

    model_config = ConfigDict(extra="forbid")

    success_rate: float | None = Field(ge=0.0, le=1.0)
    average_episode_reward: float | None
    average_episode_steps: float | None = Field(ge=0.0)
    configuration: ResolvedTrainingConfig


class ResolvedTrainingConfig(ContractModel):
    """Exact library, hyperparameters, and resources used by a training run."""

    model_config = ConfigDict(extra="forbid")

    library: Literal["stable-baselines3"]
    library_version: str = Field(min_length=1, max_length=MAX_IDENTIFIER_LENGTH)
    algorithm: Literal["ppo"]
    device: Literal["cpu"]
    timesteps: int = Field(ge=1, le=MAX_TRAINING_TIMESTEPS)
    seed: int = Field(ge=0, le=MAX_RANDOM_SEED)
    max_episode_steps: int = Field(ge=1, le=MAX_REPLAY_CHUNK_STEPS)
    n_envs: int = Field(ge=1, le=MAX_PARALLEL_ENVS)
    requested_cpu_count: int | None = Field(ge=1, le=MAX_TRAINING_CPU_COUNT)
    cpu_count: int = Field(ge=1, le=MAX_TRAINING_CPU_COUNT)
    requested_torch_num_threads: int | None = Field(
        ge=1,
        le=MAX_TRAINING_CPU_COUNT,
    )
    torch_num_threads: int = Field(ge=1, le=MAX_TRAINING_CPU_COUNT)
    n_steps: int = Field(ge=1, le=MAX_PPO_ROLLOUT_STEPS)
    batch_size: int = Field(ge=1, le=MAX_PPO_ROLLOUT_STEPS)
    n_epochs: int = Field(ge=1, le=MAX_PPO_EPOCHS)
    gamma: float = Field(gt=0.0, le=1.0)
    gae_lambda: float = Field(gt=0.0, le=1.0)
    learning_rate: float = Field(gt=0.0)
    clip_range: float = Field(gt=0.0)
    clip_range_vf: float | None = Field(gt=0.0)
    normalize_advantage: bool
    ent_coef: float = Field(ge=0.0)
    vf_coef: float = Field(ge=0.0)
    max_grad_norm: float = Field(ge=0.0)
    use_sde: bool
    sde_sample_freq: int = Field(ge=-1)
    target_kl: float | None = Field(gt=0.0)
    stats_window_size: int = Field(ge=1, le=MAX_STATS_WINDOW_SIZE)
    eval_episodes: int = Field(ge=1, le=MAX_REPLAY_CHUNK_STEPS)
    replay_eval_interval_steps: int = Field(ge=0)
    replay_train_chunk_steps: int = Field(ge=1, le=MAX_REPLAY_CHUNK_STEPS)
    randomize_start: bool


class ResultArtifacts(ContractModel):
    """Artifacts produced by a training run."""

    model_config = ConfigDict(extra="forbid")

    onnx_model: OnnxModelArtifactLocation | None
    replay_bundle: ArtifactLocation | None


class ErrorReport(ContractModel):
    """Structured failure details for failed result bundles."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=4_096)
    details: str | None


class ResultBundle(ContractModel):
    """Client-facing training result bundle."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "oneOf": [
                {
                    "properties": {
                        "status": {"const": "completed"},
                        "summary": {"not": {"type": "null"}},
                        "artifacts": {
                            "properties": {
                                "onnx_model": {"not": {"type": "null"}},
                                "replay_bundle": {"not": {"type": "null"}},
                            },
                        },
                        "error": {"type": "null"},
                    },
                },
                {
                    "properties": {
                        "status": {"const": "failed"},
                        "summary": {"type": "null"},
                        "artifacts": {
                            "properties": {
                                "onnx_model": {"type": "null"},
                                "replay_bundle": {"type": "null"},
                            },
                        },
                        "error": {"not": {"type": "null"}},
                    },
                },
            ],
        },
    )

    schema_version: Literal[RESULT_SCHEMA_VERSION]
    scenario_id: str = Field(min_length=1)
    job_id: str = Field(min_length=1)
    status: Literal[ResultStatus.COMPLETED, ResultStatus.FAILED]
    compatibility: ResultCompatibility
    summary: TrainingSummary | None
    artifacts: ResultArtifacts
    error: ErrorReport | None

    @model_validator(mode="after")
    def validate_completed_artifacts(self) -> ResultBundle:
        """Require every current downloadable artifact on completed results."""
        if self.status is ResultStatus.COMPLETED:
            if self.summary is None:
                msg = "completed result bundles require a training summary"
                raise ValueError(msg)
            if (
                self.artifacts.onnx_model is None
                or self.artifacts.replay_bundle is None
            ):
                msg = "completed result bundles require all downloadable artifacts"
                raise ValueError(msg)
            if self.error is not None:
                msg = "completed result bundles must not contain an error report"
                raise ValueError(msg)
        if self.status is ResultStatus.FAILED:
            if self.error is None:
                msg = "failed result bundles require an error report"
                raise ValueError(msg)
            if self.summary is not None or any(
                artifact is not None
                for artifact in (
                    self.artifacts.onnx_model,
                    self.artifacts.replay_bundle,
                )
            ):
                msg = "failed result bundles must not contain completed outputs"
                raise ValueError(msg)
        return self


class ReplayPosition(ContractModel):
    """A continuous replay position on the x/z plane."""

    model_config = ConfigDict(extra="forbid")

    x: float
    z: float


class ReplayRobotState(ContractModel):
    """Robot state emitted in a replay step."""

    model_config = ConfigDict(extra="forbid")

    position: ReplayPosition
    rotation_y_degrees: float


class ReplayNamedValue(ContractModel):
    """A named scalar value in a JsonUtility-friendly replay payload."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=MAX_IDENTIFIER_LENGTH)
    value: float


class ReplayForwardActionValue(ContractModel):
    """The forward component of the continuous action."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["forward"]
    value: float


class ReplayTurnActionValue(ContractModel):
    """The turn component of the continuous action."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["turn"]
    value: float


class ReplayAction(ContractModel):
    """Action values emitted for a replay step."""

    model_config = ConfigDict(extra="forbid")

    values: tuple[ReplayForwardActionValue, ReplayTurnActionValue]


class ReplayReward(ContractModel):
    """Reward values emitted for a replay step."""

    model_config = ConfigDict(extra="forbid")

    total: float
    components: list[ReplayNamedValue] = Field(
        max_length=MAX_REPLAY_REWARD_COMPONENTS,
    )


class ReplayEvent(ContractModel):
    """A compact event emitted during replay."""

    model_config = ConfigDict(extra="forbid")

    type: str = Field(min_length=1, max_length=MAX_IDENTIFIER_LENGTH)
    object_id: str | None
    message: str | None


class ReplaySensorSummary(ContractModel):
    """A compact sensor summary emitted during replay."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=MAX_IDENTIFIER_LENGTH)
    type: str = Field(min_length=1, max_length=MAX_IDENTIFIER_LENGTH)
    value: float


class ReplayLogStep(ContractModel):
    """One JSON Lines row in a Replay Log."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "oneOf": [
                {
                    "properties": {
                        "phase": {"const": "train"},
                        "policy_mode": {"const": "stochastic"},
                    },
                },
                {
                    "properties": {
                        "phase": {"const": "eval"},
                        "policy_mode": {"const": "deterministic"},
                    },
                },
            ],
        },
    )

    schema_version: Literal[REPLAY_LOG_SCHEMA_VERSION]
    scenario_id: str = Field(min_length=1, max_length=MAX_IDENTIFIER_LENGTH)
    job_id: str = Field(min_length=1, max_length=MAX_IDENTIFIER_LENGTH)
    phase: Literal["train", "eval"]
    checkpoint_step: int = Field(ge=0)
    env_index: int = Field(ge=0)
    policy_mode: Literal["stochastic", "deterministic"]
    episode_id: str = Field(min_length=1, max_length=MAX_IDENTIFIER_LENGTH)
    step_index: int = Field(ge=0)
    time_seconds: float = Field(ge=0.0)
    robot: ReplayRobotState
    action: ReplayAction
    reward: ReplayReward
    events: list[ReplayEvent] = Field(max_length=MAX_REPLAY_EVENTS)
    sensors: list[ReplaySensorSummary] = Field(max_length=MAX_REPLAY_SENSORS)
    terminated: bool
    termination_reason: str | None

    @model_validator(mode="after")
    def validate_phase_policy_mode(self) -> ReplayLogStep:
        """Require the policy mode defined for each Replay phase."""
        expected = "stochastic" if self.phase == "train" else "deterministic"
        if self.policy_mode != expected:
            msg = f"Replay {self.phase} rows require {expected} policy mode"
            raise ValueError(msg)
        return self


class ReplayBundleChunkBase(ContractModel):
    """Fields shared by every compressed Replay Bundle chunk."""

    model_config = ConfigDict(extra="forbid")

    checkpoint_step: int = Field(ge=0)
    path: str = Field(min_length=1, max_length=MAX_REPLAY_CHUNK_PATH_LENGTH)
    format: Literal["jsonl.gz"]
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class TrainReplayBundleChunk(ReplayBundleChunkBase):
    """One stochastic training Replay chunk."""

    phase: Literal["train"]
    policy_mode: Literal["stochastic"]
    start_step: int = Field(ge=0)
    end_step: int = Field(ge=0)
    path: str = Field(
        min_length=1,
        max_length=MAX_REPLAY_CHUNK_PATH_LENGTH,
        pattern=r"^train/[A-Za-z0-9._-]+\.jsonl\.gz$",
    )
    step_count: int = Field(ge=1, le=MAX_REPLAY_CHUNK_STEPS)
    episode_count: None
    success_rate: None
    avg_reward: None
    avg_steps: None

    @model_validator(mode="after")
    def validate_step_range(self) -> TrainReplayBundleChunk:
        """Require ordered train chunk bounds ending at its checkpoint."""
        if self.start_step > self.end_step:
            msg = "Replay train chunk start_step must not exceed end_step"
            raise ValueError(msg)
        if self.checkpoint_step != self.end_step:
            msg = "Replay train chunk checkpoint_step must equal end_step"
            raise ValueError(msg)
        return self


class EvalReplayBundleChunk(ReplayBundleChunkBase):
    """One deterministic evaluation Replay chunk."""

    phase: Literal["eval"]
    policy_mode: Literal["deterministic"]
    start_step: None
    end_step: None
    path: str = Field(
        min_length=1,
        max_length=MAX_REPLAY_CHUNK_PATH_LENGTH,
        pattern=r"^eval/[A-Za-z0-9._-]+\.jsonl\.gz$",
    )
    step_count: int = Field(ge=0, le=MAX_REPLAY_CHUNK_STEPS)
    episode_count: int = Field(ge=0)
    success_rate: float = Field(ge=0.0, le=1.0)
    avg_reward: float
    avg_steps: float = Field(ge=0.0)


ReplayBundleChunk = Annotated[
    TrainReplayBundleChunk | EvalReplayBundleChunk,
    Field(discriminator="phase"),
]


class ReplayBundleManifest(ContractModel):
    """Manifest describing the chunks in one Replay Bundle."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[REPLAY_BUNDLE_SCHEMA_VERSION]
    job_id: str = Field(min_length=1, max_length=MAX_IDENTIFIER_LENGTH)
    scenario_id: str = Field(min_length=1, max_length=MAX_IDENTIFIER_LENGTH)
    total_timesteps: int = Field(ge=1, le=MAX_TRAINING_TIMESTEPS)
    chunks: list[ReplayBundleChunk] = Field(
        min_length=1,
        max_length=MAX_REPLAY_BUNDLE_CHUNKS,
    )

    @model_validator(mode="after")
    def validate_chunk_paths(self) -> ReplayBundleManifest:
        """Require unique normalized relative paths for every Replay chunk."""
        paths = [chunk.path for chunk in self.chunks]
        if len(paths) != len(set(paths)):
            msg = "Replay Bundle chunk paths must be unique"
            raise ValueError(msg)
        for path in paths:
            parsed = PurePosixPath(path)
            if (
                "\\" in path
                or parsed.is_absolute()
                or ".." in parsed.parts
                or parsed.as_posix() != path
                or path == "manifest.json"
            ):
                msg = (
                    f"Replay Bundle chunk path must be normalized and relative: {path}"
                )
                raise ValueError(msg)
        for chunk in self.chunks:
            if not chunk.path.startswith(f"{chunk.phase}/"):
                msg = (
                    "Replay Bundle chunk path must match its phase directory: "
                    f"{chunk.path}"
                )
                raise ValueError(msg)
            if chunk.checkpoint_step > self.total_timesteps:
                msg = "Replay chunk checkpoint_step must not exceed total_timesteps"
                raise ValueError(msg)
        return self


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(UTC).isoformat()


def _result_state_schema_conditions() -> dict[str, Any]:
    """Return JSON Schema conditions shared by result wire messages."""
    conditions: list[dict[str, Any]] = [
        {
            "properties": {
                "status": {"const": status.value},
                "progress": {
                    "type": "object",
                    "properties": {"phase": {"const": status.value}},
                    "required": ["phase"],
                },
                "error": {"type": "null"},
                "result_bundle": {"type": "null"},
            },
        }
        for status in (
            ResultStatus.QUEUED,
            ResultStatus.STARTING,
            ResultStatus.RUNNING,
            ResultStatus.CANCELLING,
            ResultStatus.CANCELLED,
        )
    ]
    conditions.extend(
        [
            {
                "properties": {
                    "status": {"const": ResultStatus.COMPLETED.value},
                    "progress": {
                        "type": "object",
                        "properties": {
                            "phase": {"const": ResultStatus.COMPLETED.value},
                        },
                        "required": ["phase"],
                    },
                    "error": {"type": "null"},
                    "result_bundle": {
                        "type": "object",
                        "properties": {
                            "status": {"const": ResultStatus.COMPLETED.value},
                        },
                        "required": ["status"],
                    },
                },
            },
            {
                "properties": {
                    "status": {"const": ResultStatus.FAILED.value},
                    "progress": {
                        "type": "object",
                        "properties": {
                            "phase": {"const": ResultStatus.FAILED.value},
                        },
                        "required": ["phase"],
                    },
                    "error": {"type": "string", "minLength": 1},
                    "result_bundle": {
                        "anyOf": [
                            {"type": "null"},
                            {
                                "type": "object",
                                "properties": {
                                    "status": {"const": ResultStatus.FAILED.value},
                                },
                                "required": ["status"],
                            },
                        ],
                    },
                },
            },
        ],
    )
    return {"oneOf": conditions}


def _validate_completed_result_state(
    error: str | None,
    result_bundle: ResultBundle | None,
) -> None:
    """Require the outputs and absence of error for completed results."""
    if result_bundle is None:
        msg = "completed results require a completed result bundle"
        raise ValueError(msg)
    if error is not None:
        msg = "completed results must not contain an error"
        raise ValueError(msg)


def _validate_failed_result_state(error: str | None) -> None:
    """Require a nonempty error for failed results."""
    if error is None or not error:
        msg = "failed results require an error"
        raise ValueError(msg)


def _validate_result_state(
    *,
    status: ResultStatus,
    progress: Progress,
    error: str | None,
    result_bundle: ResultBundle | None,
    submission_id: str | None,
) -> None:
    """Validate one complete result-state snapshot or update."""
    if progress.phase is not status:
        msg = "result progress phase must match status"
        raise ValueError(msg)
    if result_bundle is not None:
        if result_bundle.status is not status:
            msg = "result bundle status must match status"
            raise ValueError(msg)
        if submission_id is not None and result_bundle.job_id != submission_id:
            msg = "result bundle job_id must match submission_id"
            raise ValueError(msg)
    if status is ResultStatus.COMPLETED:
        _validate_completed_result_state(error, result_bundle)
    elif status is ResultStatus.FAILED:
        _validate_failed_result_state(error)
    elif error is not None or result_bundle is not None:
        msg = "nonterminal and cancelled results must not contain terminal outputs"
        raise ValueError(msg)


class Progress(ContractModel):
    """Training progress snapshot stored in each result document."""

    model_config = ConfigDict(extra="forbid")

    phase: ResultStatus
    current_step: int = Field(ge=0)
    total_steps: int = Field(ge=0)
    message: str = Field(min_length=1, max_length=4_096)


class ResultDocument(ContractModel):
    """Full result document written to Firestore."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra=_result_state_schema_conditions(),
    )

    submission_id: str = Field(min_length=1, max_length=MAX_IDENTIFIER_LENGTH)
    status: ResultStatus
    progress: Progress
    error: str | None
    result_bundle: ResultBundle | None
    updated_at: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_state(self) -> ResultDocument:
        """Require a coherent, complete Firestore result snapshot."""
        _validate_result_state(
            status=self.status,
            progress=self.progress,
            error=self.error,
            result_bundle=self.result_bundle,
            submission_id=self.submission_id,
        )
        return self


class ResultMessage(ContractModel):
    """Pub/Sub message payload emitted after each status transition."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra=_result_state_schema_conditions(),
    )

    submission_id: str = Field(min_length=1, max_length=MAX_IDENTIFIER_LENGTH)
    status: ResultStatus
    progress: Progress
    error: str | None
    result_bundle: ResultBundle | None
    updated_at: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_state(self) -> ResultMessage:
        """Require a coherent, complete result transition message."""
        _validate_result_state(
            status=self.status,
            progress=self.progress,
            error=self.error,
            result_bundle=self.result_bundle,
            submission_id=self.submission_id,
        )
        return self


class ResultUpdate(ContractModel):
    """Partial update applied to an existing result document."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra=_result_state_schema_conditions(),
    )

    status: ResultStatus
    progress: Progress
    error: str | None
    result_bundle: ResultBundle | None
    updated_at: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_state(self) -> ResultUpdate:
        """Require a coherent, complete Firestore result update."""
        _validate_result_state(
            status=self.status,
            progress=self.progress,
            error=self.error,
            result_bundle=self.result_bundle,
            submission_id=None,
        )
        return self


def build_result_compatibility(scenario: ScenarioBundle) -> ResultCompatibility:
    """Build client compatibility metadata from the submitted scenario."""
    camera = next(
        sensor for sensor in scenario.sensors if isinstance(sensor, ForwardCameraSensor)
    )
    goal_vector = next(
        sensor for sensor in scenario.sensors if isinstance(sensor, GoalVectorSensor)
    )
    return ResultCompatibility(
        scenario_schema_version=scenario.schema_version,
        robot_version=scenario.compatibility.robot_version,
        sensor_version=scenario.compatibility.sensor_version,
        action_layout=list(scenario.robot.action_space.layout),
        observation_layout=[camera.observation_name, goal_vector.observation_name],
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
    """Build the client-facing ResultBundle from trainer outputs."""
    result_error = (
        ErrorReport(message=error, details=None) if error is not None else None
    )
    return ResultBundle(
        schema_version=RESULT_SCHEMA_VERSION,
        scenario_id=scenario.scenario_id,
        job_id=job_id,
        status=status,
        compatibility=build_result_compatibility(scenario),
        summary=(
            TrainingSummary.model_validate(summary) if summary is not None else None
        ),
        artifacts=ResultArtifacts.model_validate(
            artifacts
            if artifacts is not None
            else {
                "onnx_model": None,
                "replay_bundle": None,
            },
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


def queued_progress(total_steps: int = 0) -> Progress:
    """Return the queued-phase progress payload."""
    return build_progress(
        phase=ResultStatus.QUEUED,
        current_step=0,
        total_steps=total_steps,
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


def running_progress(total_steps: int, current_step: int = 0) -> Progress:
    """Return the running-phase progress payload."""
    return build_progress(
        phase=ResultStatus.RUNNING,
        current_step=current_step,
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


def cancelling_progress(current_step: int, total_steps: int) -> Progress:
    """Return the progress payload emitted after cancellation is accepted."""
    return build_progress(
        phase=ResultStatus.CANCELLING,
        current_step=current_step,
        total_steps=total_steps,
        message="Cancelling training",
    )


def cancelled_progress(current_step: int, total_steps: int) -> Progress:
    """Return the terminal progress payload for a cancelled execution."""
    return build_progress(
        phase=ResultStatus.CANCELLED,
        current_step=current_step,
        total_steps=total_steps,
        message="Training cancelled",
    )


def failed_progress(message: str, total_steps: int = 0) -> Progress:
    """Return the failed-phase progress payload."""
    return build_progress(
        phase=ResultStatus.FAILED,
        current_step=0,
        total_steps=total_steps,
        message=message,
    )


def build_queued_result_document(submission_id: str, *, total_steps: int = 0) -> dict:
    """Return a Firestore-ready dict for a newly queued result."""
    document = ResultDocument(
        submission_id=submission_id,
        status=ResultStatus.QUEUED,
        progress=queued_progress(total_steps),
        error=None,
        result_bundle=None,
        updated_at=utc_now_iso(),
    )
    return document.model_dump(mode="json")


def build_result_update(
    *,
    status: ResultStatus,
    progress: dict | Progress,
    error: str | None = None,
    result_bundle: dict[str, Any] | ResultBundle | None = None,
) -> dict:
    """Return a Firestore-ready dict for a partial result update."""
    update = ResultUpdate(
        status=status,
        progress=progress,
        error=error,
        result_bundle=result_bundle,
        updated_at=utc_now_iso(),
    )
    return update.model_dump(mode="json")


def build_result_message(
    submission_id: str,
    status: ResultStatus,
    progress: Progress,
    error: str | None = None,
    result_bundle: dict[str, Any] | ResultBundle | None = None,
) -> dict:
    """Return a dict suitable for publishing as a Pub/Sub message."""
    message = ResultMessage(
        submission_id=submission_id,
        status=status,
        progress=progress,
        error=error,
        result_bundle=result_bundle,
        updated_at=utc_now_iso(),
    )
    return message.model_dump(mode="json")


def parse_result_message(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a result event payload and return its normalized JSON form."""
    message = ResultMessage.model_validate(payload)
    return message.model_dump(mode="json", exclude_unset=True)
