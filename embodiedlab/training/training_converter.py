"""Convert EnvForge scenario bundles into the training runtime spec."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from embodiedlab.schemas import (
    CollisionRewardComponent,
    DistanceDeltaRewardComponent,
    DistanceSensor,
    ForwardCameraSensor,
    PerStepRewardComponent,
    ScenarioBundle,
    StaticObstacle,
    StaticWall,
    TerminalRewardComponent,
)
from embodiedlab.training.training_models import (
    ContinuousBounds,
    ContinuousBoxObstacle,
    ContinuousCameraSpec,
    ContinuousGoal,
    ContinuousNavigationSpec,
    ContinuousRewardWeights,
    ContinuousRobotStart,
)

POLICY_CAMERA_WIDTH = 112
POLICY_CAMERA_HEIGHT = 84


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
        omitted_contract_fields=(),
        lossy=True,
        notes=(
            "Runtime positions, rotations, bounds, goal radius, static walls, "
            "static obstacle footprints, and object heights stay in EnvForge meters.",
            "Forward camera output is rendered as a semantic 2.5D projection; "
            "materials, lighting, shadows, and Unity post-processing remain lossy.",
            "Supported declarative reward component weights are carried into "
            "the continuous runtime spec.",
        ),
    )


def _distance_sensor_range(scenario: ScenarioBundle) -> float:
    for sensor in scenario.sensors:
        if isinstance(sensor, DistanceSensor):
            return sensor.range_meters
    return 5.0


def _forward_camera_sensor(scenario: ScenarioBundle) -> ForwardCameraSensor:
    for sensor in scenario.sensors:
        if isinstance(sensor, ForwardCameraSensor):
            return sensor
    return ForwardCameraSensor(id="front_camera")


def _default_camera_far_clip_meters(scenario: ScenarioBundle) -> float:
    bounds = scenario.world.bounds
    return hypot(bounds.max.x - bounds.min.x, bounds.max.z - bounds.min.z)


def _camera_spec(scenario: ScenarioBundle) -> ContinuousCameraSpec:
    camera = _forward_camera_sensor(scenario)
    far_clip_meters = camera.far_clip_meters
    if far_clip_meters is None:
        far_clip_meters = _default_camera_far_clip_meters(scenario)
    if camera.width != POLICY_CAMERA_WIDTH or camera.height != POLICY_CAMERA_HEIGHT:
        msg = "forward camera size must be 112x84 for the current policy network"
        raise ValueError(msg)
    return ContinuousCameraSpec(
        width=camera.width,
        height=camera.height,
        mount_height_meters=camera.mount_height_meters,
        pitch_degrees=camera.pitch_degrees,
        vertical_fov_degrees=camera.vertical_fov_degrees,
        near_clip_meters=camera.near_clip_meters,
        far_clip_meters=far_clip_meters,
    )


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
        height=wall.height,
        rotation_y_degrees=wall.rotation_y_degrees,
    )


def _box_to_obstacle(obstacle: StaticObstacle) -> ContinuousBoxObstacle:
    return ContinuousBoxObstacle(
        obstacle_id=obstacle.id,
        center_x=obstacle.center.x,
        center_z=obstacle.center.z,
        size_x=obstacle.size.x,
        size_z=obstacle.size.z,
        height=obstacle.height,
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
    distance_sensor_range_meters = _distance_sensor_range(scenario)

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
        distance_sensor_range_meters=distance_sensor_range_meters,
        camera=_camera_spec(scenario),
        reward_weights=_reward_weights(scenario),
    )
