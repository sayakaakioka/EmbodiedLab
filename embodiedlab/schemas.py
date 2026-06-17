"""Pydantic schemas for EnvForge scenario submissions."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCENARIO_SCHEMA_VERSION = "scenario-bundle.v0"


class CoordinateSystem(StrEnum):
    """Supported world coordinate systems for scenario bundles."""

    ENVFORGE_XZ_METERS = "envforge_xz_meters"


class RobotType(StrEnum):
    """Robot archetypes supported by the first EnvForge integration."""

    SIMPLE_ROBOT = "simple_robot"


class ActionSpaceType(StrEnum):
    """Supported robot action space families."""

    CONTINUOUS = "continuous"


class SensorType(StrEnum):
    """Supported sensor kinds in scenario bundles."""

    FORWARD_CAMERA = "forward_camera"
    DISTANCE_SENSOR = "distance_sensor"


class SemanticMode(StrEnum):
    """Supported semantic camera encodings."""

    TRAVERSABLE_VS_BLOCKED = "traversable_vs_blocked"


class SensorDirection(StrEnum):
    """Supported distance sensor directions."""

    FORWARD = "forward"


class RewardComponentType(StrEnum):
    """Supported declarative reward component kinds."""

    TERMINAL_REWARD = "terminal_reward"
    DISTANCE_DELTA = "distance_delta"
    COLLISION = "collision"
    PER_STEP = "per_step"


class TrainingAlgorithm(StrEnum):
    """Supported training algorithms for EnvForge scenarios."""

    PPO = "ppo"


class CreatedBy(BaseModel):
    """Metadata about the tool that created a scenario bundle."""

    tool: str = Field(default="EnvForge", min_length=1)
    version: str = Field(default="0.1.0", min_length=1)


class Compatibility(BaseModel):
    """Compatibility metadata required by EnvForge and EmbodiedLab."""

    envforge_min_version: str = Field(default="0.1.0", min_length=1)
    robot_version: str = Field(default="simple_robot.v0", min_length=1)
    sensor_version: str = Field(default="basic_sensors.v0", min_length=1)


class Position2D(BaseModel):
    """A point on the EnvForge horizontal x/z plane."""

    x: float
    z: float


class Size2D(BaseModel):
    """A positive x/z footprint size in meters."""

    x: float = Field(gt=0)
    z: float = Field(gt=0)


class Bounds2D(BaseModel):
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


class StaticWall(BaseModel):
    """A fixed wall segment in the scenario."""

    id: str = Field(min_length=1)
    center: Position2D
    size: Size2D
    height: float = Field(default=2.0, gt=0)
    rotation_y_degrees: float = 0.0


class StaticObstacle(BaseModel):
    """A fixed obstacle in the scenario."""

    id: str = Field(min_length=1)
    shape: Literal["box"] = "box"
    center: Position2D
    size: Size2D
    height: float = Field(default=1.0, gt=0)
    rotation_y_degrees: float = 0.0


class GoalSpec(BaseModel):
    """Goal region used for navigation training."""

    id: str = Field(min_length=1)
    position: Position2D
    radius: float = Field(gt=0)


class WorldSpec(BaseModel):
    """Static world geometry for the first EnvForge contract."""

    coordinate_system: CoordinateSystem = CoordinateSystem.ENVFORGE_XZ_METERS
    bounds: Bounds2D = Field(
        default_factory=lambda: Bounds2D(
            min=Position2D(x=0.0, z=0.0),
            max=Position2D(x=10.0, z=10.0),
        ),
    )
    static_walls: list[StaticWall] = Field(default_factory=list)
    static_obstacles: list[StaticObstacle] = Field(default_factory=list)
    goal: GoalSpec = Field(
        default_factory=lambda: GoalSpec(
            id="goal_001",
            position=Position2D(x=8.5, z=8.5),
            radius=0.5,
        ),
    )

    @model_validator(mode="after")
    def validate_world_positions(self) -> WorldSpec:
        """Ensure point-based world objects are inside the declared bounds."""
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


class Pose2D(BaseModel):
    """Robot pose on the x/z plane."""

    position: Position2D
    rotation_y_degrees: float = 0.0


class ActionSpace(BaseModel):
    """Robot action layout expected by the policy."""

    type: ActionSpaceType = ActionSpaceType.CONTINUOUS
    layout: list[Literal["forward", "turn"]] = Field(
        default_factory=lambda: ["forward", "turn"],
        min_length=2,
        max_length=2,
    )

    @model_validator(mode="after")
    def validate_layout(self) -> ActionSpace:
        """Require the initial simple robot action order."""
        if self.layout != ["forward", "turn"]:
            msg = "action layout must be ['forward', 'turn']"
            raise ValueError(msg)
        return self


class RobotSpec(BaseModel):
    """Robot descriptor for an EnvForge scenario."""

    type: RobotType = RobotType.SIMPLE_ROBOT
    start_pose: Pose2D = Field(
        default_factory=lambda: Pose2D(
            position=Position2D(x=1.0, z=1.0),
            rotation_y_degrees=0.0,
        ),
    )
    action_space: ActionSpace = Field(default_factory=ActionSpace)


class ForwardCameraSensor(BaseModel):
    """Forward semantic camera sensor configuration."""

    id: str = Field(min_length=1)
    type: Literal[SensorType.FORWARD_CAMERA] = SensorType.FORWARD_CAMERA
    width: int = Field(default=112, ge=1)
    height: int = Field(default=84, ge=1)
    semantic_mode: SemanticMode = SemanticMode.TRAVERSABLE_VS_BLOCKED
    mount_height_meters: float = Field(default=0.6, gt=0)
    mount_height_min_meters: float | None = Field(default=None, gt=0)
    mount_height_max_meters: float | None = Field(default=None, gt=0)
    pitch_degrees: float = Field(default=0.0)
    vertical_fov_degrees: float = Field(default=70.0, gt=0, lt=180)
    near_clip_meters: float = Field(default=0.05, gt=0)
    far_clip_meters: float = Field(default=100.0, gt=0)

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


class DistanceSensor(BaseModel):
    """Forward distance sensor configuration."""

    id: str = Field(min_length=1)
    type: Literal[SensorType.DISTANCE_SENSOR] = SensorType.DISTANCE_SENSOR
    range_meters: float = Field(default=5.0, gt=0)
    direction: SensorDirection = SensorDirection.FORWARD


SensorSpec = Annotated[
    ForwardCameraSensor | DistanceSensor,
    Field(discriminator="type"),
]


class TerminalRewardComponent(BaseModel):
    """Terminal reward paid when the task succeeds."""

    name: str = Field(min_length=1)
    type: Literal[RewardComponentType.TERMINAL_REWARD] = (
        RewardComponentType.TERMINAL_REWARD
    )
    weight: float


class DistanceDeltaRewardComponent(BaseModel):
    """Reward based on distance progress toward a target object."""

    name: str = Field(min_length=1)
    type: Literal[RewardComponentType.DISTANCE_DELTA] = (
        RewardComponentType.DISTANCE_DELTA
    )
    target: str = Field(min_length=1)
    weight: float


class CollisionRewardComponent(BaseModel):
    """Reward component emitted on collisions."""

    name: str = Field(min_length=1)
    type: Literal[RewardComponentType.COLLISION] = RewardComponentType.COLLISION
    weight: float


class PerStepRewardComponent(BaseModel):
    """Reward component emitted at each simulation step."""

    name: str = Field(min_length=1)
    type: Literal[RewardComponentType.PER_STEP] = RewardComponentType.PER_STEP
    weight: float


RewardComponent = Annotated[
    TerminalRewardComponent
    | DistanceDeltaRewardComponent
    | CollisionRewardComponent
    | PerStepRewardComponent,
    Field(discriminator="type"),
]


class RewardSpec(BaseModel):
    """Declarative reward configuration for training."""

    components: list[RewardComponent] = Field(default_factory=list)


class TrainingSpec(BaseModel):
    """Training request parameters for EnvForge scenario bundles."""

    algorithm: TrainingAlgorithm = TrainingAlgorithm.PPO
    timesteps: int = Field(default=5_000, ge=1)
    seed: int = 10
    max_episode_steps: int = Field(default=512, ge=1)
    n_envs: int = Field(default=1, ge=1)
    cpu_count: int | None = Field(default=None, ge=1)
    torch_num_threads: int | None = Field(default=None, ge=1)
    n_steps: int = Field(default=32, ge=1)
    batch_size: int = Field(default=32, ge=1)
    n_epochs: int = Field(default=3, ge=1)
    gamma: float = Field(default=0.99, gt=0.0, le=1.0)
    learning_rate: float = Field(default=3e-4, gt=0.0)
    ent_coef: float = Field(default=0.0, ge=0.0)
    eval_episodes: int = Field(default=20, ge=1)


class ScenarioBundle(BaseModel):
    """Top-level request body for POST /submissions."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[SCENARIO_SCHEMA_VERSION] = SCENARIO_SCHEMA_VERSION
    scenario_id: str = Field(default="scenario_demo_001", min_length=1)
    created_by: CreatedBy = Field(default_factory=CreatedBy)
    compatibility: Compatibility = Field(default_factory=Compatibility)
    world: WorldSpec = Field(default_factory=WorldSpec)
    robot: RobotSpec = Field(default_factory=RobotSpec)
    sensors: list[SensorSpec] = Field(
        default_factory=lambda: [
            ForwardCameraSensor(id="front_camera"),
            DistanceSensor(id="front_distance"),
        ],
        min_length=1,
    )
    reward: RewardSpec = Field(
        default_factory=lambda: RewardSpec(
            components=[
                TerminalRewardComponent(
                    name="goal_reached",
                    weight=100.0,
                ),
                DistanceDeltaRewardComponent(
                    name="goal_progress",
                    target="goal_001",
                    weight=0.1,
                ),
                CollisionRewardComponent(
                    name="collision_penalty",
                    weight=-50.0,
                ),
                PerStepRewardComponent(
                    name="step_penalty",
                    weight=-0.01,
                ),
                PerStepRewardComponent(
                    name="wide_angle_penalty",
                    weight=-0.1,
                ),
                PerStepRewardComponent(
                    name="rear_angle_penalty",
                    weight=-5.0,
                ),
                PerStepRewardComponent(
                    name="inactive_penalty",
                    weight=-0.1,
                ),
                PerStepRewardComponent(
                    name="movement_threshold",
                    weight=0.001,
                ),
            ],
        ),
    )
    training: TrainingSpec = Field(default_factory=TrainingSpec)

    @model_validator(mode="after")
    def validate_scenario(self) -> ScenarioBundle:
        """Validate cross-field references in the scenario bundle."""
        if not self.world.bounds.contains(self.robot.start_pose.position):
            msg = "robot.start_pose.position must be inside world bounds"
            raise ValueError(msg)

        sensor_ids = [sensor.id for sensor in self.sensors]
        if len(sensor_ids) != len(set(sensor_ids)):
            msg = "sensor ids must be unique"
            raise ValueError(msg)

        object_ids = {
            self.world.goal.id,
            *(wall.id for wall in self.world.static_walls),
            *(obstacle.id for obstacle in self.world.static_obstacles),
        }
        for component in self.reward.components:
            if (
                isinstance(component, DistanceDeltaRewardComponent)
                and component.target not in object_ids
            ):
                msg = f"reward component target not found: {component.target}"
                raise ValueError(msg)

        return self


class SubmissionDocument(BaseModel):
    """Firestore document stored at submissions/{submission_id}."""

    submission_id: str
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    scenario: ScenarioBundle


def build_submission_document(submission_id: str, scenario: ScenarioBundle) -> dict:
    """Return a Firestore-ready dict for a new scenario submission."""
    document = SubmissionDocument(
        submission_id=submission_id,
        scenario=scenario,
    )
    return document.model_dump(mode="json")
