"""Pydantic schemas for EmbodiedLab scenario submissions."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from types import UnionType
from typing import Annotated, Literal, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCENARIO_SCHEMA_VERSION = "scenario-bundle.v0"
MAX_REPLAY_CHUNK_STEPS = 100_000
MAX_CAMERA_DIMENSION_PIXELS = 512
MAX_FORWARD_STEP_METERS = 10.0
MAX_TRAINING_TIMESTEPS = 10_000_000
MAX_PARALLEL_ENVS = 32
MAX_TRAINING_CPU_COUNT = 32
MAX_PPO_ROLLOUT_STEPS = 65_536
MAX_PPO_EPOCHS = 100
MAX_STATS_WINDOW_SIZE = 100_000
MAX_RANDOM_SEED = (2**32) - 1
MAX_IDENTIFIER_LENGTH = 128
MAX_WORLD_GEOMETRY_ELEMENTS = 128
MAX_SCENARIO_SENSORS = 3


class ContractModel(BaseModel):
    """Strict base for every public wire-contract object."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    @model_validator(mode="before")
    @classmethod
    def reject_primitive_type_coercion(cls, value: object) -> object:
        """Keep server validation aligned with JSON Schema primitive types."""
        if not isinstance(value, dict):
            return value
        for name, raw_value in value.items():
            field = cls.model_fields.get(name)
            if field is None or raw_value is None:
                continue
            primitive_types = _primitive_types(field.annotation)
            if bool in primitive_types and type(raw_value) is not bool:
                msg = f"{name} must be a JSON boolean"
                raise ValueError(msg)
            if (
                int in primitive_types
                and float not in primitive_types
                and type(raw_value) is not int
            ):
                msg = f"{name} must be a JSON integer"
                raise ValueError(msg)
            if float in primitive_types and (
                not isinstance(raw_value, int | float) or isinstance(raw_value, bool)
            ):
                msg = f"{name} must be a JSON number"
                raise ValueError(msg)
        return value


def _primitive_types(annotation: object) -> set[type]:
    """Return primitive JSON number/bool types accepted by one annotation."""
    if annotation in {bool, int, float}:
        return {annotation}
    origin = get_origin(annotation)
    if origin is Annotated:
        return _primitive_types(get_args(annotation)[0])
    if origin in {UnionType, Union}:
        primitive_types: set[type] = set()
        for member in get_args(annotation):
            primitive_types.update(_primitive_types(member))
        return primitive_types
    return set()


class CoordinateSystem(StrEnum):
    """Supported world coordinate systems for scenario bundles."""

    LEFT_HANDED_Y_UP_METERS = "left_handed_y_up_meters"


class RobotType(StrEnum):
    """Robot archetypes supported by the current scenario contract."""

    SIMPLE_ROBOT = "simple_robot"


class ActionSpaceType(StrEnum):
    """Supported robot action space families."""

    CONTINUOUS = "continuous"


class SensorType(StrEnum):
    """Supported sensor kinds in scenario bundles."""

    FORWARD_CAMERA = "forward_camera"
    DISTANCE_SENSOR = "distance_sensor"
    GOAL_VECTOR = "goal_vector"


class SemanticMode(StrEnum):
    """Supported semantic camera encodings."""

    TRAVERSABLE_VS_BLOCKED = "traversable_vs_blocked"


def semantic_channel_layout(mode: SemanticMode) -> tuple[str, ...]:
    """Return the policy channel order defined by a semantic mode."""
    if mode is SemanticMode.TRAVERSABLE_VS_BLOCKED:
        return (
            "channel_0_unused",
            "channel_1_traversable",
            "channel_2_blocked_or_background",
        )
    msg = f"Unsupported semantic mode: {mode}"
    raise ValueError(msg)


class SensorDirection(StrEnum):
    """Supported distance sensor directions."""

    FORWARD = "forward"


class RewardComponentType(StrEnum):
    """Supported declarative reward component kinds."""

    TERMINAL_REWARD = "terminal_reward"
    DISTANCE_DELTA = "distance_delta"
    COLLISION = "collision"
    PER_STEP = "per_step"
    MINIMUM_ABSOLUTE_ANGLE = "minimum_absolute_angle"
    MAXIMUM_ABSOLUTE_FORWARD = "maximum_absolute_forward"


class TrainingAlgorithm(StrEnum):
    """Supported training algorithms for scenario bundles."""

    PPO = "ppo"


class TrainingDevice(StrEnum):
    """Training devices supported by the current Cloud Run runtime."""

    CPU = "cpu"


class DispatchState(StrEnum):
    """Private server-owned dispatch states for a submission."""

    PENDING = "pending"
    DISPATCHING = "dispatching"
    AMBIGUOUS = "ambiguous"
    DISPATCHED = "dispatched"
    CANCELLED = "cancelled"
    FAILED = "failed"


class CancellationState(StrEnum):
    """Private durable cancellation intent states for a submission."""

    IDLE = "idle"
    REQUESTING = "requesting"
    REQUESTED = "requested"


class CreatedBy(ContractModel):
    """Metadata about the tool that created a scenario bundle."""

    tool: str = Field(min_length=1)
    version: str = Field(min_length=1)


class Compatibility(ContractModel):
    """Compatibility metadata required by EmbodiedLab clients."""

    robot_version: str = Field(min_length=1)
    sensor_version: str = Field(min_length=1)


class Position2D(ContractModel):
    """A point on the horizontal x/z plane."""

    x: float
    z: float


class Size2D(ContractModel):
    """A positive x/z footprint size in meters."""

    x: float = Field(gt=0)
    z: float = Field(gt=0)


class Bounds2D(ContractModel):
    """Axis-aligned world bounds on the x/z plane."""

    min: Position2D
    max: Position2D

    @model_validator(mode="after")
    def validate_bounds(self) -> Bounds2D:
        """Ensure the maximum corner is greater than the minimum corner."""
        if self.max.x <= self.min.x or self.max.z <= self.min.z:
            msg = "bounds.max must be greater than bounds.min on both axes"
            raise ValueError(msg)
        return self

    def contains(self, position: Position2D) -> bool:
        """Return whether the position is inside the bounds."""
        return (
            self.min.x <= position.x <= self.max.x
            and self.min.z <= position.z <= self.max.z
        )


class StaticWall(ContractModel):
    """A fixed wall segment in the scenario."""

    id: str = Field(min_length=1, max_length=MAX_IDENTIFIER_LENGTH)
    center: Position2D
    size: Size2D
    height: float = Field(gt=0)
    rotation_y_degrees: float


class StaticObstacle(ContractModel):
    """A fixed obstacle in the scenario."""

    id: str = Field(min_length=1, max_length=MAX_IDENTIFIER_LENGTH)
    shape: Literal["box"]
    center: Position2D
    size: Size2D
    height: float = Field(gt=0)
    rotation_y_degrees: float


class GoalSpec(ContractModel):
    """Goal region used for navigation training."""

    id: str = Field(min_length=1, max_length=MAX_IDENTIFIER_LENGTH)
    position: Position2D
    radius: float = Field(gt=0)


class WorldSpec(ContractModel):
    """Static world geometry for the current scenario contract."""

    coordinate_system: CoordinateSystem
    bounds: Bounds2D
    static_walls: list[StaticWall] = Field(max_length=MAX_WORLD_GEOMETRY_ELEMENTS)
    static_obstacles: list[StaticObstacle] = Field(
        max_length=MAX_WORLD_GEOMETRY_ELEMENTS,
    )
    goal: GoalSpec

    @model_validator(mode="after")
    def validate_world_positions(self) -> WorldSpec:
        """Ensure point-based world objects are inside the declared bounds."""
        if len(self.static_walls) + len(self.static_obstacles) > (
            MAX_WORLD_GEOMETRY_ELEMENTS
        ):
            msg = (
                "static_walls and static_obstacles together cannot exceed "
                f"{MAX_WORLD_GEOMETRY_ELEMENTS}"
            )
            raise ValueError(msg)
        positions = [
            ("goal.position", self.goal.position),
            *(
                (f"static_walls[{index}].center", wall.center)
                for index, wall in enumerate(self.static_walls)
            ),
            *(
                (f"static_obstacles[{index}].center", obstacle.center)
                for index, obstacle in enumerate(self.static_obstacles)
            ),
        ]
        for field_name, position in positions:
            if not self.bounds.contains(position):
                msg = f"{field_name} must be inside world bounds"
                raise ValueError(msg)
        return self


class Pose2D(ContractModel):
    """Robot pose on the x/z plane."""

    position: Position2D
    rotation_y_degrees: float


class ActionSpace(ContractModel):
    """Robot action layout expected by the policy."""

    type: ActionSpaceType
    layout: list[Literal["forward", "turn"]] = Field(
        min_length=2,
        max_length=2,
    )
    forward_step_meters: float = Field(gt=0, le=MAX_FORWARD_STEP_METERS)
    turn_degrees_per_step: float = Field(gt=0)
    step_duration_seconds: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_layout(self) -> ActionSpace:
        """Require the initial simple robot action order."""
        if self.layout != ["forward", "turn"]:
            msg = "action layout must be ['forward', 'turn']"
            raise ValueError(msg)
        return self


class RobotSpec(ContractModel):
    """Robot descriptor for a scenario bundle."""

    type: RobotType
    radius: float = Field(gt=0)
    start_pose: Pose2D
    action_space: ActionSpace


class ForwardCameraSensor(ContractModel):
    """Forward semantic camera sensor configuration."""

    id: str = Field(min_length=1, max_length=MAX_IDENTIFIER_LENGTH)
    type: Literal[SensorType.FORWARD_CAMERA]
    width: int = Field(ge=20, le=MAX_CAMERA_DIMENSION_PIXELS)
    height: int = Field(ge=20, le=MAX_CAMERA_DIMENSION_PIXELS)
    semantic_mode: SemanticMode
    mount_height_meters: float = Field(gt=0)
    mount_height_min_meters: float | None = Field(gt=0)
    mount_height_max_meters: float | None = Field(gt=0)
    pitch_degrees: float
    vertical_fov_degrees: float = Field(gt=0, lt=180)
    near_clip_meters: float = Field(gt=0)
    far_clip_meters: float = Field(gt=0)
    observation_name: str = Field(min_length=1, max_length=MAX_IDENTIFIER_LENGTH)

    @model_validator(mode="after")
    def validate_mount_height_range(self) -> ForwardCameraSensor:
        """Require a complete, ordered optional camera height range."""
        has_min = self.mount_height_min_meters is not None
        has_max = self.mount_height_max_meters is not None
        if has_min != has_max:
            msg = "camera mount height range requires both min and max values"
            raise ValueError(msg)
        if (
            self.mount_height_min_meters is not None
            and self.mount_height_max_meters is not None
            and self.mount_height_min_meters > self.mount_height_max_meters
        ):
            msg = "camera mount height min must be less than or equal to max"
            raise ValueError(msg)
        return self


class DistanceSensor(ContractModel):
    """Forward distance sensor configuration."""

    id: str = Field(min_length=1, max_length=MAX_IDENTIFIER_LENGTH)
    type: Literal[SensorType.DISTANCE_SENSOR]
    range_meters: float = Field(gt=0)
    direction: SensorDirection


class GoalVectorSensor(ContractModel):
    """Goal-relative numeric observation consumed by the policy."""

    id: str = Field(min_length=1, max_length=MAX_IDENTIFIER_LENGTH)
    type: Literal[SensorType.GOAL_VECTOR]
    target: str = Field(min_length=1, max_length=MAX_IDENTIFIER_LENGTH)
    observation_name: str = Field(min_length=1, max_length=MAX_IDENTIFIER_LENGTH)
    values: list[Literal["goal_angle_degrees", "goal_distance_meters"]] = Field(
        min_length=2,
        max_length=2,
    )

    @model_validator(mode="after")
    def validate_values(self) -> GoalVectorSensor:
        """Require the current numeric observation order."""
        if self.values != ["goal_angle_degrees", "goal_distance_meters"]:
            msg = (
                "goal vector values must be "
                "['goal_angle_degrees', 'goal_distance_meters']"
            )
            raise ValueError(msg)
        return self


SensorSpec = Annotated[
    ForwardCameraSensor | DistanceSensor | GoalVectorSensor,
    Field(discriminator="type"),
]


class TerminalRewardComponent(ContractModel):
    """Terminal reward paid when the task succeeds."""

    name: str = Field(min_length=1)
    type: Literal[RewardComponentType.TERMINAL_REWARD]
    weight: float


class DistanceDeltaRewardComponent(ContractModel):
    """Reward based on distance progress toward a target object."""

    name: str = Field(min_length=1)
    type: Literal[RewardComponentType.DISTANCE_DELTA]
    target: str = Field(min_length=1)
    weight: float
    minimum_delta_meters: float = Field(gt=0)


class CollisionRewardComponent(ContractModel):
    """Reward component emitted on collisions."""

    name: str = Field(min_length=1)
    type: Literal[RewardComponentType.COLLISION]
    weight: float


class PerStepRewardComponent(ContractModel):
    """Reward component emitted at each simulation step."""

    name: str = Field(min_length=1)
    type: Literal[RewardComponentType.PER_STEP]
    weight: float


class MinimumAbsoluteAngleRewardComponent(ContractModel):
    """Penalty enabled above an absolute goal-angle threshold."""

    name: str = Field(min_length=1)
    type: Literal[RewardComponentType.MINIMUM_ABSOLUTE_ANGLE]
    weight: float
    minimum_absolute_angle_degrees: float = Field(gt=0, le=180)


class MaximumAbsoluteForwardRewardComponent(ContractModel):
    """Penalty enabled when forward action stays below a threshold."""

    name: str = Field(min_length=1)
    type: Literal[RewardComponentType.MAXIMUM_ABSOLUTE_FORWARD]
    weight: float
    maximum_absolute_forward: float = Field(ge=0, le=1)


RewardComponent = Annotated[
    TerminalRewardComponent
    | DistanceDeltaRewardComponent
    | CollisionRewardComponent
    | PerStepRewardComponent
    | MinimumAbsoluteAngleRewardComponent
    | MaximumAbsoluteForwardRewardComponent,
    Field(discriminator="type"),
]


class RewardSpec(ContractModel):
    """Declarative reward configuration for training."""

    components: list[RewardComponent] = Field(
        min_length=7,
        max_length=7,
    )


class TrainingSpec(ContractModel):
    """Training request parameters for scenario bundles."""

    algorithm: TrainingAlgorithm
    device: TrainingDevice
    timesteps: int = Field(ge=1, le=MAX_TRAINING_TIMESTEPS)
    seed: int = Field(ge=0, le=MAX_RANDOM_SEED)
    max_episode_steps: int = Field(ge=1, le=MAX_REPLAY_CHUNK_STEPS)
    n_envs: int = Field(ge=1, le=MAX_PARALLEL_ENVS)
    cpu_count: int | None = Field(ge=1, le=MAX_TRAINING_CPU_COUNT)
    torch_num_threads: int | None = Field(ge=1, le=MAX_TRAINING_CPU_COUNT)
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
    replay_train_chunk_steps: int = Field(
        ge=1,
        le=MAX_REPLAY_CHUNK_STEPS,
    )
    randomize_start: bool

    @model_validator(mode="after")
    def validate_eval_replay_size(self) -> TrainingSpec:
        """Keep one deterministic evaluation chunk within the SDK row budget."""
        if self.eval_episodes * (self.max_episode_steps + 1) > MAX_REPLAY_CHUNK_STEPS:
            msg = (
                "eval_episodes * (max_episode_steps + 1) must be less than or "
                f"equal to {MAX_REPLAY_CHUNK_STEPS}"
            )
            raise ValueError(msg)
        rollout_steps = self.n_steps * self.n_envs
        if rollout_steps > MAX_PPO_ROLLOUT_STEPS:
            msg = (
                "n_steps * n_envs must be less than or equal to "
                f"{MAX_PPO_ROLLOUT_STEPS}"
            )
            raise ValueError(msg)
        if rollout_steps % self.batch_size != 0:
            msg = "n_steps * n_envs must be divisible by batch_size"
            raise ValueError(msg)
        return self


class ScenarioBundle(ContractModel):
    """Top-level request body for POST /submissions."""

    schema_version: Literal[SCENARIO_SCHEMA_VERSION]
    scenario_id: str = Field(min_length=1, max_length=MAX_IDENTIFIER_LENGTH)
    created_by: CreatedBy
    compatibility: Compatibility
    world: WorldSpec
    robot: RobotSpec
    sensors: list[SensorSpec] = Field(
        min_length=1,
        max_length=MAX_SCENARIO_SENSORS,
    )
    reward: RewardSpec
    training: TrainingSpec

    @model_validator(mode="after")
    def validate_world_and_sensors(self) -> ScenarioBundle:
        """Validate world and sensor cross-field references."""
        if not self.world.bounds.contains(self.robot.start_pose.position):
            msg = "robot.start_pose.position must be inside world bounds"
            raise ValueError(msg)

        sensor_ids = [sensor.id for sensor in self.sensors]
        if len(sensor_ids) != len(set(sensor_ids)):
            msg = "sensor ids must be unique"
            raise ValueError(msg)

        camera_sensors = [
            sensor for sensor in self.sensors if isinstance(sensor, ForwardCameraSensor)
        ]
        goal_vector_sensors = [
            sensor for sensor in self.sensors if isinstance(sensor, GoalVectorSensor)
        ]
        distance_sensors = [
            sensor for sensor in self.sensors if isinstance(sensor, DistanceSensor)
        ]
        if len(camera_sensors) != 1 or len(goal_vector_sensors) != 1:
            msg = "scenario requires exactly one forward camera and one goal vector"
            raise ValueError(msg)
        if len(distance_sensors) > 1:
            msg = "scenario supports at most one forward distance sensor"
            raise ValueError(msg)
        observation_names = [
            camera_sensors[0].observation_name,
            goal_vector_sensors[0].observation_name,
        ]
        if len(observation_names) != len(set(observation_names)):
            msg = "policy observation names must be unique"
            raise ValueError(msg)
        if goal_vector_sensors[0].target != self.world.goal.id:
            msg = "goal vector target must match world.goal.id"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_reward_contract(self) -> ScenarioBundle:
        """Validate reward names, types, and target references."""
        expected_reward_types = {
            "goal_reached": RewardComponentType.TERMINAL_REWARD,
            "goal_progress": RewardComponentType.DISTANCE_DELTA,
            "collision_penalty": RewardComponentType.COLLISION,
            "step_penalty": RewardComponentType.PER_STEP,
            "wide_angle_penalty": RewardComponentType.MINIMUM_ABSOLUTE_ANGLE,
            "rear_angle_penalty": RewardComponentType.MINIMUM_ABSOLUTE_ANGLE,
            "inactive_penalty": RewardComponentType.MAXIMUM_ABSOLUTE_FORWARD,
        }
        reward_names = [component.name for component in self.reward.components]
        if len(reward_names) != len(set(reward_names)):
            msg = "reward component names must be unique"
            raise ValueError(msg)

        actual_reward_names = set(reward_names)
        expected_reward_names = set(expected_reward_types)
        if actual_reward_names != expected_reward_names:
            missing = sorted(expected_reward_names - actual_reward_names)
            unknown = sorted(actual_reward_names - expected_reward_names)
            msg = (
                "reward components must match the continuous runtime contract; "
                f"missing={missing}, unknown={unknown}"
            )
            raise ValueError(msg)

        for component in self.reward.components:
            expected_type = expected_reward_types[component.name]
            if component.type != expected_type:
                msg = (
                    f"reward component {component.name} must use type "
                    f"{expected_type.value}"
                )
                raise ValueError(msg)
            if (
                isinstance(component, DistanceDeltaRewardComponent)
                and component.target != self.world.goal.id
            ):
                msg = (
                    f"goal_progress target must match world.goal.id: {component.target}"
                )
                raise ValueError(msg)

        components_by_name = {
            component.name: component for component in self.reward.components
        }
        wide_angle = components_by_name["wide_angle_penalty"]
        rear_angle = components_by_name["rear_angle_penalty"]
        if (
            isinstance(wide_angle, MinimumAbsoluteAngleRewardComponent)
            and isinstance(rear_angle, MinimumAbsoluteAngleRewardComponent)
            and wide_angle.minimum_absolute_angle_degrees
            >= rear_angle.minimum_absolute_angle_degrees
        ):
            msg = (
                "wide_angle_penalty threshold must be less than "
                "rear_angle_penalty threshold"
            )
            raise ValueError(msg)

        return self


class SubmissionControl(BaseModel):
    """Private control data stored with a submitted scenario."""

    cancel_token_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    dispatch_state: DispatchState = DispatchState.PENDING
    dispatch_started_at: datetime | None = None
    dispatch_error: str | None = None
    execution_name: str | None = None
    cancellation_state: CancellationState = CancellationState.IDLE
    cancellation_started_at: datetime | None = None
    cancellation_lease_token: str | None = None
    cancellation_error: str | None = None


class SubmissionDocument(BaseModel):
    """Firestore document stored at submissions/{submission_id}."""

    submission_id: str
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    scenario: ScenarioBundle
    control: SubmissionControl


def build_submission_document(
    submission_id: str,
    scenario: ScenarioBundle,
    *,
    cancel_token_hash: str,
) -> dict:
    """Return a Firestore-ready dict for a new scenario submission."""
    document = SubmissionDocument(
        submission_id=submission_id,
        scenario=scenario,
        control=SubmissionControl(cancel_token_hash=cancel_token_hash),
    )
    return document.model_dump(mode="json")
