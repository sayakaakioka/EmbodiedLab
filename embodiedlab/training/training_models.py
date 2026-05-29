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
class ContinuousNavigationSpec:
    """Continuous EnvForge-compatible navigation runtime specification."""

    bounds: ContinuousBounds
    obstacles: tuple[ContinuousBoxObstacle, ...]
    goal: ContinuousGoal
    robot_start: ContinuousRobotStart
    robot_type: str
    distance_sensor_range_meters: float = 5.0
    forward_step_meters: float = 0.2
    turn_degrees_per_step: float = 15.0
