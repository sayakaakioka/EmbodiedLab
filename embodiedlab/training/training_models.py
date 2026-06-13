"""Data models representing the continuous navigation training runtime."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ContinuousBounds:
    """Axis-aligned x/z training bounds in EnvForge meters."""

    min_x: float
    min_z: float
    max_x: float
    max_z: float


@dataclass(frozen=True)
class ContinuousBoxObstacle:
    """Static box obstacle on the EnvForge x/z plane."""

    obstacle_id: str
    center_x: float
    center_z: float
    size_x: float
    size_z: float
    height: float = 2.0
    rotation_y_degrees: float = 0.0


@dataclass(frozen=True)
class ContinuousGoal:
    """Circular goal region on the EnvForge x/z plane."""

    goal_id: str
    x: float
    z: float
    radius: float


@dataclass(frozen=True)
class ContinuousRobotStart:
    """Initial robot pose on the EnvForge x/z plane."""

    x: float
    z: float
    rotation_y_degrees: float


@dataclass(frozen=True)
class ContinuousCameraSpec:
    """Fixed forward camera parameters for semantic 2.5D rendering."""

    width: int = 112
    height: int = 84
    mount_height_meters: float = 0.6
    pitch_degrees: float = 0.0
    vertical_fov_degrees: float = 60.0
    near_clip_meters: float = 0.05
    far_clip_meters: float = 5.0


@dataclass(frozen=True)
class ContinuousRewardWeights:
    """Reward weights used by the continuous EnvForge runtime."""

    goal_reached: float = 100.0
    goal_progress: float = 0.1
    collision_penalty: float = -50.0
    step_penalty: float = -0.01
    wide_angle_penalty: float = -0.1
    rear_angle_penalty: float = -5.0
    inactive_penalty: float = -0.1
    movement_threshold: float = 0.001


@dataclass(frozen=True)
class ContinuousNavigationSpec:
    """Continuous EnvForge-compatible navigation runtime specification."""

    bounds: ContinuousBounds
    obstacles: tuple[ContinuousBoxObstacle, ...]
    goal: ContinuousGoal
    robot_start: ContinuousRobotStart
    robot_type: str
    distance_sensor_range_meters: float = 5.0
    camera: ContinuousCameraSpec = field(default_factory=ContinuousCameraSpec)
    reward_weights: ContinuousRewardWeights = field(
        default_factory=ContinuousRewardWeights,
    )
    forward_step_meters: float = 0.2
    turn_degrees_per_step: float = 15.0
