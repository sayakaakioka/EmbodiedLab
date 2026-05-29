import numpy as np

from embodiedlab.continuous_navigation_env import ContinuousNavigationEnv
from embodiedlab.schemas import ScenarioBundle
from embodiedlab.training.training_converter import (
    convert_submission_to_spec,
    describe_runtime_conversion,
)


def test_continuous_runtime_conversion_preserves_envforge_coordinates():
    scenario = ScenarioBundle(
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
                    "rotation_y_degrees": 90.0,
                },
            ],
            "static_obstacles": [
                {
                    "id": "box_001",
                    "shape": "box",
                    "center": {"x": 4.75, "z": 5.25},
                    "size": {"x": 1.2, "z": 0.8},
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
            "start_pose": {
                "position": {"x": 1.9, "z": 2.1},
                "rotation_y_degrees": 90.0,
            },
        },
        sensors=[
            {"id": "front_camera", "type": "forward_camera"},
            {"id": "front_distance", "type": "distance_sensor", "range_meters": 7.5},
        ],
    )

    conversion = describe_runtime_conversion(scenario)
    spec = convert_submission_to_spec(scenario)

    assert conversion.runtime_coordinate_system == "envforge_xz_meters"
    assert conversion.coordinate_mapping == "direct_envforge_xz_meters"
    assert conversion.lossy is True
    assert "reward.components" in conversion.omitted_contract_fields
    assert spec.bounds.min_x == -5.0
    assert spec.bounds.max_z == 15.0
    assert spec.goal.goal_id == "goal_001"
    assert spec.goal.radius == 0.75
    assert spec.robot_start.x == 1.9
    assert spec.robot_start.rotation_y_degrees == 90.0
    assert spec.distance_sensor_range_meters == 7.5
    assert [obstacle.obstacle_id for obstacle in spec.obstacles] == [
        "wall_001",
        "box_001",
    ]
    assert spec.obstacles[1].rotation_y_degrees == 45.0


def test_continuous_env_moves_forward_in_envforge_xz_space():
    spec = convert_submission_to_spec(ScenarioBundle())
    env = ContinuousNavigationEnv(spec=spec, max_steps=10)

    obs, info = env.reset()
    next_obs, reward, terminated, truncated, next_info = env.step(
        np.array([1.0, 0.0], dtype=np.float32),
    )

    assert obs["robot"].tolist() == [1.0, 1.0, 0.0]
    assert next_obs["robot"][0] == 1.0
    assert next_obs["robot"][1] > 1.0
    assert next_info["distance"] < info["distance"]
    assert reward > -0.01
    assert terminated is False
    assert truncated is False


def test_continuous_env_blocks_rotated_obstacle_collision():
    scenario = ScenarioBundle(
        world={
            "static_obstacles": [
                {
                    "id": "box_001",
                    "shape": "box",
                    "center": {"x": 1.0, "z": 1.2},
                    "size": {"x": 1.0, "z": 0.2},
                    "rotation_y_degrees": 0.0,
                },
            ],
        },
    )
    env = ContinuousNavigationEnv(
        spec=convert_submission_to_spec(scenario),
        max_steps=10,
    )
    obs, _info = env.reset()

    next_obs, reward, _terminated, _truncated, next_info = env.step(
        np.array([1.0, 0.0], dtype=np.float32),
    )

    assert next_obs["robot"].tolist() == obs["robot"].tolist()
    assert next_info["collision"] is True
    assert next_info["collision_id"] == "box_001"
    assert next_info["front_distance"] <= 0.2
    assert reward < -1.0


def test_continuous_env_blocks_thin_obstacle_between_movement_endpoints():
    scenario = ScenarioBundle(
        world={
            "static_obstacles": [
                {
                    "id": "thin_wall",
                    "shape": "box",
                    "center": {"x": 1.0, "z": 1.1},
                    "size": {"x": 1.0, "z": 0.02},
                    "rotation_y_degrees": 0.0,
                },
            ],
        },
    )
    env = ContinuousNavigationEnv(
        spec=convert_submission_to_spec(scenario),
        max_steps=10,
    )
    obs, _info = env.reset()

    next_obs, _reward, _terminated, _truncated, next_info = env.step(
        np.array([1.0, 0.0], dtype=np.float32),
    )

    assert next_obs["robot"].tolist() == obs["robot"].tolist()
    assert next_info["collision"] is True
