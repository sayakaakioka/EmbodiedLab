"""Convert scenario bundles into the training runtime spec."""

from __future__ import annotations

from dataclasses import dataclass

from embodiedlab.schemas import (
    DistanceDeltaRewardComponent,
    DistanceSensor,
    ForwardCameraSensor,
    GoalVectorSensor,
    MaximumAbsoluteForwardRewardComponent,
    MinimumAbsoluteAngleRewardComponent,
    ScenarioBundle,
    StaticObstacle,
    StaticWall,
    semantic_channel_layout,
)
from embodiedlab.training.training_models import (
    ContinuousBounds,
    ContinuousBoxObstacle,
    ContinuousCameraSpec,
    ContinuousGoal,
    ContinuousGoalVectorSpec,
    ContinuousNavigationSpec,
    ContinuousRewardSettings,
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


def describe_runtime_conversion(
    scenario: ScenarioBundle,
) -> ScenarioRuntimeConversion:
    """Describe the scenario runtime mapping."""
    return ScenarioRuntimeConversion(
        source_coordinate_system=scenario.world.coordinate_system.value,
        runtime_coordinate_system=scenario.world.coordinate_system.value,
        coordinate_mapping="direct_left_handed_y_up_meters",
        omitted_contract_fields=(),
        lossy=True,
        notes=(
            "Runtime positions, rotations, bounds, goal radius, static walls, "
            "static obstacle footprints, and object heights stay in declared meters.",
            "Forward camera output is rendered as a semantic 2.5D projection; "
            "materials, lighting, shadows, and Unity post-processing remain lossy.",
            "Supported declarative reward component weights are carried into "
            "the continuous runtime spec.",
        ),
    )


def _distance_sensor(scenario: ScenarioBundle) -> DistanceSensor | None:
    for sensor in scenario.sensors:
        if isinstance(sensor, DistanceSensor):
            return sensor
    return None


def _forward_camera_sensor(scenario: ScenarioBundle) -> ForwardCameraSensor:
    for sensor in scenario.sensors:
        if isinstance(sensor, ForwardCameraSensor):
            return sensor
    msg = "scenario requires a forward camera sensor"
    raise ValueError(msg)


def _goal_vector_sensor(scenario: ScenarioBundle) -> GoalVectorSensor:
    for sensor in scenario.sensors:
        if isinstance(sensor, GoalVectorSensor):
            return sensor
    msg = "scenario requires a goal vector sensor"
    raise ValueError(msg)


def _camera_spec(scenario: ScenarioBundle) -> ContinuousCameraSpec:
    camera = _forward_camera_sensor(scenario)
    far_clip_meters = camera.far_clip_meters
    mount_height_min_meters = camera.mount_height_min_meters
    if mount_height_min_meters is None:
        mount_height_min_meters = camera.mount_height_meters
    mount_height_max_meters = camera.mount_height_max_meters
    if mount_height_max_meters is None:
        mount_height_max_meters = camera.mount_height_meters
    return ContinuousCameraSpec(
        width=camera.width,
        height=camera.height,
        mount_height_meters=camera.mount_height_meters,
        mount_height_min_meters=mount_height_min_meters,
        mount_height_max_meters=mount_height_max_meters,
        pitch_degrees=camera.pitch_degrees,
        vertical_fov_degrees=camera.vertical_fov_degrees,
        near_clip_meters=camera.near_clip_meters,
        far_clip_meters=far_clip_meters,
        semantic_mode=camera.semantic_mode.value,
        observation_name=camera.observation_name,
        channel_layout=semantic_channel_layout(camera.semantic_mode),
    )


def _reward_settings(scenario: ScenarioBundle) -> ContinuousRewardSettings:
    components = {component.name: component for component in scenario.reward.components}
    goal_progress = components["goal_progress"]
    wide_angle = components["wide_angle_penalty"]
    rear_angle = components["rear_angle_penalty"]
    inactive = components["inactive_penalty"]
    if not isinstance(goal_progress, DistanceDeltaRewardComponent):
        msg = "goal_progress reward type was not validated"
        raise TypeError(msg)
    if not isinstance(wide_angle, MinimumAbsoluteAngleRewardComponent):
        msg = "wide_angle_penalty reward type was not validated"
        raise TypeError(msg)
    if not isinstance(rear_angle, MinimumAbsoluteAngleRewardComponent):
        msg = "rear_angle_penalty reward type was not validated"
        raise TypeError(msg)
    if not isinstance(inactive, MaximumAbsoluteForwardRewardComponent):
        msg = "inactive_penalty reward type was not validated"
        raise TypeError(msg)
    return ContinuousRewardSettings(
        goal_reached=components["goal_reached"].weight,
        goal_progress=goal_progress.weight,
        goal_progress_minimum_delta_meters=goal_progress.minimum_delta_meters,
        collision_penalty=components["collision_penalty"].weight,
        step_penalty=components["step_penalty"].weight,
        wide_angle_penalty=wide_angle.weight,
        wide_angle_minimum_absolute_degrees=(wide_angle.minimum_absolute_angle_degrees),
        rear_angle_penalty=rear_angle.weight,
        rear_angle_minimum_absolute_degrees=(rear_angle.minimum_absolute_angle_degrees),
        inactive_penalty=inactive.weight,
        inactive_maximum_absolute_forward=inactive.maximum_absolute_forward,
    )


def _wall_to_obstacle(wall: StaticWall) -> ContinuousBoxObstacle:
    return ContinuousBoxObstacle(
        obstacle_id=wall.id,
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


def convert_scenario_to_spec(scenario: ScenarioBundle) -> ContinuousNavigationSpec:
    """Convert a ScenarioBundle into the continuous navigation runtime spec."""
    bounds = scenario.world.bounds
    goal = scenario.world.goal
    start_pose = scenario.robot.start_pose
    obstacles = [
        *(_wall_to_obstacle(wall) for wall in scenario.world.static_walls),
        *(_box_to_obstacle(obstacle) for obstacle in scenario.world.static_obstacles),
    ]
    distance_sensor = _distance_sensor(scenario)
    goal_vector = _goal_vector_sensor(scenario)

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
        robot_radius=scenario.robot.radius,
        distance_sensor_id=(
            distance_sensor.id if distance_sensor is not None else None
        ),
        distance_sensor_range_meters=(
            distance_sensor.range_meters if distance_sensor is not None else None
        ),
        camera=_camera_spec(scenario),
        goal_vector=ContinuousGoalVectorSpec(
            sensor_id=goal_vector.id,
            target=goal_vector.target,
            observation_name=goal_vector.observation_name,
            values=tuple(goal_vector.values),
        ),
        reward_settings=_reward_settings(scenario),
        forward_step_meters=scenario.robot.action_space.forward_step_meters,
        turn_degrees_per_step=scenario.robot.action_space.turn_degrees_per_step,
        step_duration_seconds=scenario.robot.action_space.step_duration_seconds,
    )
