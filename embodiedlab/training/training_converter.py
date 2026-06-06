"""Convert EnvForge scenario bundles into the training runtime spec."""

from __future__ import annotations

from dataclasses import dataclass

from embodiedlab.schemas import (
    CollisionRewardComponent,
    DistanceDeltaRewardComponent,
    DistanceSensor,
    PerStepRewardComponent,
    ScenarioBundle,
    StaticObstacle,
    StaticWall,
    TerminalRewardComponent,
)
from embodiedlab.training.training_models import (
    ContinuousBounds,
    ContinuousBoxObstacle,
    ContinuousGoal,
    ContinuousNavigationSpec,
    ContinuousRewardWeights,
    ContinuousRobotStart,
)


@dataclass(frozen=True)
class ScenarioRuntimeConversion:
    """Boundary metadata for the ScenarioBundle-to-runtime mapping."""

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


def describe_runtime_conversion(
    submission: dict[str, object] | ScenarioBundle,
) -> ScenarioRuntimeConversion:
    """Describe the EnvForge scenario runtime mapping."""
    scenario = parse_scenario_bundle(submission)
    return ScenarioRuntimeConversion(
        source_coordinate_system=scenario.world.coordinate_system.value,
        runtime_coordinate_system="envforge_xz_meters",
        coordinate_mapping="direct_envforge_xz_meters",
        omitted_contract_fields=("sensors.forward_camera.image_observation",),
        lossy=True,
        notes=(
            "Runtime positions, rotations, bounds, goal radius, static walls, "
            "and static obstacle footprints stay in EnvForge x/z meters.",
            "Forward camera output remains an abstraction at this runtime layer; "
            "distance sensor range is represented directly.",
            "Supported declarative reward component weights are carried into "
            "the continuous runtime spec.",
        ),
    )


def _distance_sensor_range(scenario: ScenarioBundle) -> float:
    for sensor in scenario.sensors:
        if isinstance(sensor, DistanceSensor):
            return sensor.range_meters
    return 5.0


def _reward_weights(scenario: ScenarioBundle) -> ContinuousRewardWeights:
    weights = ContinuousRewardWeights()
    values = {
        "goal_reached": weights.goal_reached,
        "goal_progress": weights.goal_progress,
        "collision_penalty": weights.collision_penalty,
        "step_penalty": weights.step_penalty,
        "wide_angle_penalty": weights.wide_angle_penalty,
        "rear_angle_penalty": weights.rear_angle_penalty,
        "inactive_penalty": weights.inactive_penalty,
        "movement_threshold": weights.movement_threshold,
    }
    for component in scenario.reward.components:
        if (
            isinstance(
                component,
                (
                    TerminalRewardComponent,
                    DistanceDeltaRewardComponent,
                    CollisionRewardComponent,
                    PerStepRewardComponent,
                ),
            )
            and component.name in values
        ):
            values[component.name] = component.weight
    return ContinuousRewardWeights(**values)


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


def convert_submission_to_spec(
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
        reward_weights=_reward_weights(scenario),
    )
