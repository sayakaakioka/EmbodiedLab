"""Data models representing the continuous navigation training runtime."""

from __future__ import annotations

from dataclasses import dataclass


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
    height: float
    rotation_y_degrees: float


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
    """Forward camera parameters for semantic 2.5D rendering."""

    width: int
    height: int
    mount_height_meters: float
    mount_height_min_meters: float
    mount_height_max_meters: float
    pitch_degrees: float
    vertical_fov_degrees: float
    near_clip_meters: float
    far_clip_meters: float


@dataclass(frozen=True)
class ContinuousRewardWeights:
    """Reward weights used by the continuous EnvForge runtime."""

    goal_reached: float
    goal_progress: float
    collision_penalty: float
    step_penalty: float
    wide_angle_penalty: float
    rear_angle_penalty: float
    inactive_penalty: float
    movement_threshold: float


@dataclass(frozen=True)
class ContinuousNavigationSpec:
    """Continuous EnvForge-compatible navigation runtime specification."""

    bounds: ContinuousBounds
    obstacles: tuple[ContinuousBoxObstacle, ...]
    goal: ContinuousGoal
    robot_start: ContinuousRobotStart
    robot_type: str
    robot_radius: float
    distance_sensor_range_meters: float
    camera: ContinuousCameraSpec
    reward_weights: ContinuousRewardWeights
    forward_step_meters: float
    turn_degrees_per_step: float
