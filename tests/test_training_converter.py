import pytest

from embodiedlab.training.training_converter import (
    convert_scenario_to_spec,
    describe_runtime_conversion,
)
from tests.fakes import scenario_bundle


def test_convert_scenario_to_continuous_runtime_spec():
    scenario = scenario_bundle(
        world={
            "bounds": {
                "min": {"x": -5.0, "z": -2.0},
                "max": {"x": 20.0, "z": 15.0},
            },
            "static_walls": [
                {
                    "id": "wall_001",
                    "center": {"x": 0.0, "z": 4.0},
                    "size": {"x": 8.0, "z": 0.2},
                    "height": 2.5,
                    "rotation_y_degrees": 90.0,
                },
            ],
            "static_obstacles": [
                {
                    "id": "box_001",
                    "shape": "box",
                    "center": {"x": 4.75, "z": 5.25},
                    "size": {"x": 1.2, "z": 0.8},
                    "height": 0.75,
                    "rotation_y_degrees": 45.0,
                },
            ],
            "goal": {
                "id": "goal_001",
                "position": {"x": 8.75, "z": 8.25},
                "radius": 0.75,
            },
        },
        robot={
            "radius": 0.3,
            "start_pose": {
                "position": {"x": 1.9, "z": 2.1},
                "rotation_y_degrees": 90.0,
            },
        },
        sensors=[
            {
                "id": "front_camera",
                "type": "forward_camera",
                "mount_height_meters": 0.7,
                "mount_height_min_meters": 0.1,
                "mount_height_max_meters": 1.0,
                "pitch_degrees": 5.0,
                "vertical_fov_degrees": 55.0,
                "near_clip_meters": 0.1,
                "far_clip_meters": 6.5,
            },
            {
                "id": "goal_vector",
                "type": "goal_vector",
                "target": "goal_001",
            },
            {
                "id": "front_distance",
                "type": "distance_sensor",
                "range_meters": 7.5,
                "direction": "forward",
            },
        ],
        reward={
            "components": [
                {"name": "goal_reached", "type": "terminal_reward", "weight": 101.0},
                {
                    "name": "goal_progress",
                    "type": "distance_delta",
                    "target": "goal_001",
                    "weight": 0.25,
                    "minimum_delta_meters": 0.01,
                },
                {"name": "collision_penalty", "type": "collision", "weight": -12.0},
                {"name": "step_penalty", "type": "per_step", "weight": -0.2},
                {
                    "name": "wide_angle_penalty",
                    "type": "minimum_absolute_angle",
                    "weight": -0.4,
                    "minimum_absolute_angle_degrees": 80.0,
                },
                {
                    "name": "rear_angle_penalty",
                    "type": "minimum_absolute_angle",
                    "weight": -7.0,
                    "minimum_absolute_angle_degrees": 140.0,
                },
                {
                    "name": "inactive_penalty",
                    "type": "maximum_absolute_forward",
                    "weight": -0.5,
                    "maximum_absolute_forward": 0.02,
                },
            ],
        },
    )

    conversion = describe_runtime_conversion(scenario)
    spec = convert_scenario_to_spec(scenario)

    assert conversion.runtime_coordinate_system == "left_handed_y_up_meters"
    assert conversion.coordinate_mapping == "direct_left_handed_y_up_meters"
    assert conversion.lossy is True
    assert "reward.components" not in conversion.omitted_contract_fields
    assert spec.bounds.min_x == -5.0
    assert spec.bounds.max_z == 15.0
    assert spec.goal.goal_id == "goal_001"
    assert spec.goal.radius == 0.75
    assert spec.robot_start.x == 1.9
    assert spec.robot_radius == 0.3
    assert spec.robot_start.rotation_y_degrees == 90.0
    assert spec.distance_sensor_id == "front_distance"
    assert spec.distance_sensor_range_meters == 7.5
    assert spec.camera.mount_height_meters == 0.7
    assert spec.camera.mount_height_min_meters == 0.1
    assert spec.camera.mount_height_max_meters == 1.0
    assert spec.camera.pitch_degrees == 5.0
    assert spec.camera.vertical_fov_degrees == 55.0
    assert spec.camera.near_clip_meters == 0.1
    assert spec.camera.far_clip_meters == 6.5
    assert spec.reward_settings.goal_reached == 101.0
    assert spec.reward_settings.goal_progress == 0.25
    assert spec.reward_settings.goal_progress_minimum_delta_meters == 0.01
    assert spec.reward_settings.collision_penalty == -12.0
    assert spec.reward_settings.step_penalty == -0.2
    assert spec.reward_settings.wide_angle_penalty == -0.4
    assert spec.reward_settings.wide_angle_minimum_absolute_degrees == 80.0
    assert spec.reward_settings.rear_angle_penalty == -7.0
    assert spec.reward_settings.rear_angle_minimum_absolute_degrees == 140.0
    assert spec.reward_settings.inactive_penalty == -0.5
    assert spec.reward_settings.inactive_maximum_absolute_forward == 0.02
    assert [obstacle.obstacle_id for obstacle in spec.obstacles] == [
        "wall_001",
        "box_001",
    ]
    assert spec.obstacles[0].height == 2.5
    assert spec.obstacles[1].height == 0.75
    assert spec.obstacles[1].rotation_y_degrees == 45.0


def test_camera_far_clip_uses_declared_value():
    scenario = scenario_bundle(
        world={
            "bounds": {
                "min": {"x": -5.0, "z": -2.0},
                "max": {"x": 20.0, "z": 15.0},
            },
        },
        sensors=[
            {
                "id": "front_camera",
                "type": "forward_camera",
                "far_clip_meters": 100.0,
            },
            {
                "id": "goal_vector",
                "type": "goal_vector",
                "target": "goal_001",
            },
            {
                "id": "front_distance",
                "type": "distance_sensor",
                "range_meters": 7.5,
                "direction": "forward",
            },
        ],
    )

    spec = convert_scenario_to_spec(scenario)

    assert spec.distance_sensor_range_meters == 7.5
    assert spec.camera.far_clip_meters == pytest.approx(100.0)
