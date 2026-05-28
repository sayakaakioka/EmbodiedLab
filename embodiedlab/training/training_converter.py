"""Convert EnvForge scenario bundles into the current training runtime spec."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor

from embodiedlab.schemas import (
    DistanceSensor,
    Position2D,
    ScenarioBundle,
    StaticObstacle,
    StaticWall,
)
from embodiedlab.training.training_models import (
    ContinuousBounds,
    ContinuousBoxObstacle,
    ContinuousGoal,
    ContinuousNavigationSpec,
    ContinuousRobotStart,
    GridPosition,
    GridWorldSpec,
)


@dataclass(frozen=True)
class ScenarioRuntimeConversion:
    """Boundary metadata for the temporary ScenarioBundle-to-grid adapter."""

    source_coordinate_system: str
    runtime_coordinate_system: str
    coordinate_mapping: str
    omitted_contract_fields: tuple[str, ...]
    lossy: bool
    notes: tuple[str, ...]


def parse_scenario_bundle(
    submission: dict[str, object] | ScenarioBundle,
) -> ScenarioBundle:
    """Parse a Firestore submission document or scenario payload."""
    if isinstance(submission, ScenarioBundle):
        return submission

    payload = submission.get("scenario", submission)
    return ScenarioBundle.model_validate(payload)


def describe_grid_runtime_conversion(
    submission: dict[str, object] | ScenarioBundle,
) -> ScenarioRuntimeConversion:
    """Describe the legacy adapter from continuous EnvForge space to grid cells."""
    scenario = parse_scenario_bundle(submission)
    return ScenarioRuntimeConversion(
        source_coordinate_system=scenario.world.coordinate_system.value,
        runtime_coordinate_system="grid_world_cells",
        coordinate_mapping="subtract_bounds_min_then_floor_envforge_xz_meters_to_non_negative_xy_cells",
        omitted_contract_fields=(
            "world.static_walls",
            "world.static_obstacles[].size",
            "world.static_obstacles[].rotation_y_degrees",
            "world.goal.radius",
            "robot.start_pose.rotation_y_degrees",
            "robot.action_space",
            "sensors",
            "reward.components",
            "training.max_episode_steps",
        ),
        lossy=True,
        notes=(
            "Continuous x/z meter positions subtract bounds.min, then floor "
            "to grid cell indices.",
            "Object sizes and rotation_y_degrees are not represented by the "
            "current grid runtime.",
            "Reward components remain declarative at the contract boundary but "
            "are not fully mapped to runtime reward logic yet.",
        ),
    )


def describe_runtime_conversion(
    submission: dict[str, object] | ScenarioBundle,
) -> ScenarioRuntimeConversion:
    """Describe the preferred continuous EnvForge scenario runtime mapping."""
    scenario = parse_scenario_bundle(submission)
    return ScenarioRuntimeConversion(
        source_coordinate_system=scenario.world.coordinate_system.value,
        runtime_coordinate_system="envforge_xz_meters",
        coordinate_mapping="direct_envforge_xz_meters",
        omitted_contract_fields=(
            "sensors.forward_camera.image_observation",
            "reward.components",
        ),
        lossy=True,
        notes=(
            "Runtime positions, rotations, bounds, goal radius, static walls, "
            "and static obstacle footprints stay in EnvForge x/z meters.",
            "Forward camera output remains an abstraction at this runtime layer; "
            "distance sensor range is represented directly.",
            "Declarative reward components are not yet carried into the "
            "continuous runtime spec; reward weights remain runtime defaults.",
        ),
    )


def _position_to_grid(position: Position2D, origin: Position2D) -> GridPosition:
    return GridPosition(
        x=max(0, floor(position.x - origin.x)),
        y=max(0, floor(position.z - origin.z)),
    )


def convert_submission_to_spec(
    submission: dict[str, object] | ScenarioBundle,
) -> GridWorldSpec:
    """Convert a ScenarioBundle into the current internal training spec."""
    scenario = parse_scenario_bundle(submission)
    bounds = scenario.world.bounds
    width = max(2, floor(bounds.max.x - bounds.min.x))
    height = max(2, floor(bounds.max.z - bounds.min.z))

    obstacles = {
        _position_to_grid(obstacle.center, bounds.min)
        for obstacle in scenario.world.static_obstacles
    }

    return GridWorldSpec(
        width=width,
        height=height,
        obstacles=frozenset(obstacles),
        goal=_position_to_grid(scenario.world.goal.position, bounds.min),
        robot_start=_position_to_grid(scenario.robot.start_pose.position, bounds.min),
        robot_type=scenario.robot.type.value,
        envforge_origin_x=bounds.min.x,
        envforge_origin_z=bounds.min.z,
    )


def _distance_sensor_range(scenario: ScenarioBundle) -> float:
    for sensor in scenario.sensors:
        if isinstance(sensor, DistanceSensor):
            return sensor.range_meters
    return 5.0


def _wall_to_obstacle(wall_index: int, wall: StaticWall) -> ContinuousBoxObstacle:
    return ContinuousBoxObstacle(
        obstacle_id=wall.id or f"wall_{wall_index:03d}",
        center_x=wall.center.x,
        center_z=wall.center.z,
        size_x=wall.size.x,
        size_z=wall.size.z,
        rotation_y_degrees=wall.rotation_y_degrees,
    )


def _box_to_obstacle(obstacle: StaticObstacle) -> ContinuousBoxObstacle:
    return ContinuousBoxObstacle(
        obstacle_id=obstacle.id,
        center_x=obstacle.center.x,
        center_z=obstacle.center.z,
        size_x=obstacle.size.x,
        size_z=obstacle.size.z,
        rotation_y_degrees=obstacle.rotation_y_degrees,
    )


def convert_submission_to_continuous_spec(
    submission: dict[str, object] | ScenarioBundle,
) -> ContinuousNavigationSpec:
    """Convert a ScenarioBundle into the continuous navigation runtime spec."""
    scenario = parse_scenario_bundle(submission)
    bounds = scenario.world.bounds
    goal = scenario.world.goal
    start_pose = scenario.robot.start_pose
    obstacles = [
        *(
            _wall_to_obstacle(index, wall)
            for index, wall in enumerate(scenario.world.static_walls)
        ),
        *(_box_to_obstacle(obstacle) for obstacle in scenario.world.static_obstacles),
    ]

    return ContinuousNavigationSpec(
        bounds=ContinuousBounds(
            min_x=bounds.min.x,
            min_z=bounds.min.z,
            max_x=bounds.max.x,
            max_z=bounds.max.z,
        ),
        obstacles=tuple(obstacles),
        goal=ContinuousGoal(
            goal_id=goal.id,
            x=goal.position.x,
            z=goal.position.z,
            radius=goal.radius,
        ),
        robot_start=ContinuousRobotStart(
            x=start_pose.position.x,
            z=start_pose.position.z,
            rotation_y_degrees=start_pose.rotation_y_degrees,
        ),
        robot_type=scenario.robot.type.value,
        distance_sensor_range_meters=_distance_sensor_range(scenario),
    )
