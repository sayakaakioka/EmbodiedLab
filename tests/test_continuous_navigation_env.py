import numpy as np
import pytest

from embodiedlab.continuous_navigation_env import (
    CAMERA_FOV_DEGREES,
    CAMERA_NEAR_METERS,
    IMAGE_OBSERVATION_CHANNELS,
    IMAGE_OBSERVATION_HEIGHT,
    IMAGE_OBSERVATION_WIDTH,
    ContinuousNavigationEnv,
)
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
    assert "reward.components" not in conversion.omitted_contract_fields
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
    _next_obs, reward, terminated, truncated, next_info = env.step(
        np.array([1.0, 0.0], dtype=np.float32),
    )

    assert obs["obs_0"].shape == (3, 84, 112)
    assert obs["obs_1"].tolist() == [45.0, pytest.approx(10.606601717798213)]
    assert next_info["robot_x"] == 1.0
    assert next_info["robot_z"] > 1.0
    assert next_info["distance"] < info["distance"]
    assert info["distance"] - next_info["distance"] > 0.0
    assert reward == pytest.approx(-0.01 + 0.1)
    assert terminated is False
    assert truncated is False


def test_continuous_env_rewards_goal_progress_fixed_when_distance_decreases():
    scenario = ScenarioBundle(
        reward={
            "components": [
                {
                    "name": "goal_progress",
                    "type": "distance_delta",
                    "target": "goal_001",
                    "weight": 0.5,
                },
                {"name": "step_penalty", "type": "per_step", "weight": 0.0},
            ],
        },
    )
    env = ContinuousNavigationEnv(
        spec=convert_submission_to_spec(scenario),
        max_steps=10,
    )
    _obs, info = env.reset()

    _next_obs, reward, _terminated, _truncated, next_info = env.step(
        np.array([1.0, 1.0], dtype=np.float32),
    )

    assert info["distance"] - next_info["distance"] > 0.0
    assert reward == pytest.approx(0.5)


def test_continuous_env_uses_declared_reward_weights():
    scenario = ScenarioBundle(
        reward={
            "components": [
                {"name": "goal_reached", "type": "terminal_reward", "weight": 100.0},
                {
                    "name": "goal_progress",
                    "type": "distance_delta",
                    "target": "goal_001",
                    "weight": 0.0,
                },
                {"name": "collision_penalty", "type": "collision", "weight": -50.0},
                {"name": "step_penalty", "type": "per_step", "weight": -0.2},
                {"name": "inactive_penalty", "type": "per_step", "weight": -0.4},
                {"name": "movement_threshold", "type": "per_step", "weight": 0.001},
            ],
        },
    )
    env = ContinuousNavigationEnv(
        spec=convert_submission_to_spec(scenario),
        max_steps=10,
    )
    _obs, _info = env.reset()

    _next_obs, reward, _terminated, _truncated, _next_info = env.step(
        np.array([1.0, 1.0], dtype=np.float32),
    )

    assert reward == pytest.approx(-0.2)


def test_continuous_env_penalizes_min_forward_even_when_turning_fast():
    scenario = ScenarioBundle(
        reward={
            "components": [
                {
                    "name": "goal_progress",
                    "type": "distance_delta",
                    "target": "goal_001",
                    "weight": 0.0,
                },
                {"name": "step_penalty", "type": "per_step", "weight": 0.0},
                {"name": "inactive_penalty", "type": "per_step", "weight": -0.4},
                {"name": "movement_threshold", "type": "per_step", "weight": 0.001},
            ],
        },
    )
    env = ContinuousNavigationEnv(
        spec=convert_submission_to_spec(scenario),
        max_steps=10,
    )
    _obs, _info = env.reset()

    _next_obs, reward, _terminated, _truncated, _next_info = env.step(
        np.array([-1.0, 1.0], dtype=np.float32),
    )

    assert _next_info["applied_forward"] <= 0.001
    assert _next_info["applied_turn"] == pytest.approx(1.0)
    assert reward == pytest.approx(-0.4)


def test_continuous_env_penalizes_zero_forward_as_inactive():
    scenario = ScenarioBundle(
        reward={
            "components": [
                {
                    "name": "goal_progress",
                    "type": "distance_delta",
                    "target": "goal_001",
                    "weight": 0.0,
                },
                {"name": "step_penalty", "type": "per_step", "weight": 0.0},
                {"name": "inactive_penalty", "type": "per_step", "weight": -0.4},
                {"name": "movement_threshold", "type": "per_step", "weight": 0.001},
            ],
        },
    )
    env = ContinuousNavigationEnv(
        spec=convert_submission_to_spec(scenario),
        max_steps=10,
    )
    _obs, _info = env.reset()

    _next_obs, reward, _terminated, _truncated, _next_info = env.step(
        np.array([-1.0, 0.2], dtype=np.float32),
    )

    assert _next_info["applied_forward"] <= 0.001
    assert reward == pytest.approx(-0.4)


def test_continuous_env_does_not_penalize_forward_without_turning_as_inactive():
    scenario = ScenarioBundle(
        reward={
            "components": [
                {
                    "name": "goal_progress",
                    "type": "distance_delta",
                    "target": "goal_001",
                    "weight": 0.0,
                },
                {"name": "step_penalty", "type": "per_step", "weight": 0.0},
                {"name": "inactive_penalty", "type": "per_step", "weight": -0.4},
                {"name": "movement_threshold", "type": "per_step", "weight": 0.001},
            ],
        },
    )
    env = ContinuousNavigationEnv(
        spec=convert_submission_to_spec(scenario),
        max_steps=10,
    )
    _obs, _info = env.reset()

    _next_obs, reward, _terminated, _truncated, _next_info = env.step(
        np.array([1.0, 0.0], dtype=np.float32),
    )

    assert _next_info["applied_forward"] == pytest.approx(1.0)
    assert reward == pytest.approx(0.0)


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
    _obs, _info = env.reset()

    _next_obs, reward, terminated, truncated, next_info = env.step(
        np.array([1.0, 0.0], dtype=np.float32),
    )

    assert terminated is True
    assert truncated is False
    assert next_info["robot_x"] == pytest.approx(_info["robot_x"])
    assert next_info["robot_z"] == pytest.approx(_info["robot_z"])
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
    _obs, _info = env.reset()

    _next_obs, _reward, _terminated, _truncated, next_info = env.step(
        np.array([1.0, 0.0], dtype=np.float32),
    )

    assert next_info["robot_x"] == pytest.approx(_info["robot_x"])
    assert next_info["robot_z"] == pytest.approx(_info["robot_z"])
    assert next_info["collision"] is True


def test_segmentation_observation_matches_scalar_collision_probe():
    scenario = ScenarioBundle(
        world={
            "bounds": {
                "min": {"x": -2.0, "z": -2.0},
                "max": {"x": 2.0, "z": 2.0},
            },
            "static_obstacles": [
                {
                    "id": "rotated_box",
                    "shape": "box",
                    "center": {"x": 0.4, "z": 0.9},
                    "size": {"x": 1.0, "z": 0.4},
                    "rotation_y_degrees": 35.0,
                },
            ],
            "goal": {
                "id": "goal_001",
                "position": {"x": 1.5, "z": 1.5},
                "radius": 0.5,
            },
        },
        robot={
            "start_pose": {
                "position": {"x": 0.0, "z": 0.0},
                "rotation_y_degrees": 10.0,
            },
        },
        sensors=[
            {"id": "front_camera", "type": "forward_camera"},
            {"id": "front_distance", "type": "distance_sensor", "range_meters": 3.0},
        ],
    )
    env = ContinuousNavigationEnv(
        spec=convert_submission_to_spec(scenario),
        max_steps=10,
    )
    obs, _info = env.reset()

    expected = np.zeros(
        (
            IMAGE_OBSERVATION_CHANNELS,
            IMAGE_OBSERVATION_HEIGHT,
            IMAGE_OBSERVATION_WIDTH,
        ),
        dtype=np.float32,
    )
    max_range = env.spec.distance_sensor_range_meters
    for row in range(IMAGE_OBSERVATION_HEIGHT):
        row_ratio = 1.0 - row / max(1, IMAGE_OBSERVATION_HEIGHT - 1)
        distance = CAMERA_NEAR_METERS + row_ratio * (max_range - CAMERA_NEAR_METERS)
        for column in range(IMAGE_OBSERVATION_WIDTH):
            column_ratio = column / max(1, IMAGE_OBSERVATION_WIDTH - 1) - 0.5
            ray_degrees = (
                env.robot_rotation_y_degrees + column_ratio * CAMERA_FOV_DEGREES
            )
            ray = np.array(
                [
                    np.sin(np.deg2rad(ray_degrees)),
                    np.cos(np.deg2rad(ray_degrees)),
                ],
                dtype=np.float32,
            )
            probe = env.robot_pos + ray * distance
            if env._collision_id(probe) is None:  # noqa: SLF001
                expected[1, row, column] = 1.0
            else:
                expected[2, row, column] = 1.0

    np.testing.assert_array_equal(obs["obs_0"], expected)


def test_continuous_env_maps_raw_action_to_navigation_final_contract():
    spec = convert_submission_to_spec(ScenarioBundle())
    env = ContinuousNavigationEnv(spec=spec, max_steps=10)

    _obs, _info = env.reset()
    _next_obs, _reward, _terminated, _truncated, _next_info = env.step(
        np.array([0.0, 1.0], dtype=np.float32),
    )

    assert env.action_space.low.tolist() == [-1.0, -1.0]
    assert env.action_space.high.tolist() == [1.0, 1.0]
    assert _next_info["raw_forward"] == pytest.approx(0.0)
    assert _next_info["raw_turn"] == pytest.approx(1.0)
    assert _next_info["applied_forward"] == pytest.approx(0.5)
    assert _next_info["applied_turn"] == pytest.approx(1.0)
    assert _next_info["robot_z"] > _info["robot_z"]


def test_continuous_env_randomizes_start_pose_when_enabled():
    scenario = ScenarioBundle(
        world={
            "bounds": {
                "min": {"x": -4.0, "z": -4.0},
                "max": {"x": 4.0, "z": 4.0},
            },
            "static_obstacles": [
                {
                    "id": "center_box",
                    "shape": "box",
                    "center": {"x": 0.0, "z": 0.0},
                    "size": {"x": 1.0, "z": 1.0},
                },
            ],
            "goal": {
                "id": "goal_001",
                "position": {"x": 3.0, "z": 3.0},
                "radius": 0.5,
            },
        },
        robot={
            "start_pose": {
                "position": {"x": -3.0, "z": -3.0},
                "rotation_y_degrees": 0.0,
            },
        },
    )
    spec = convert_submission_to_spec(scenario)
    env = ContinuousNavigationEnv(spec=spec, max_steps=10, randomize_start=True)

    _obs, info = env.reset(seed=123)

    assert spec.bounds.min_x <= info["robot_x"] <= spec.bounds.max_x
    assert spec.bounds.min_z <= info["robot_z"] <= spec.bounds.max_z
    assert env._collision_id(env.robot_pos) is None  # noqa: SLF001
    assert env._clearance_collision_id(env.robot_pos, 0.65) is None  # noqa: SLF001
    assert info["distance"] > spec.goal.radius + 0.65
    assert -180.0 <= info["robot_rotation_y_degrees"] <= 180.0
    assert (info["robot_x"], info["robot_z"]) != (-3.0, -3.0)

    second_env = ContinuousNavigationEnv(
        spec=spec,
        max_steps=10,
        randomize_start=True,
    )
    _second_obs, second_info = second_env.reset(seed=123)

    assert second_info["robot_x"] == pytest.approx(info["robot_x"])
    assert second_info["robot_z"] == pytest.approx(info["robot_z"])
    assert second_info["robot_rotation_y_degrees"] == pytest.approx(
        info["robot_rotation_y_degrees"],
    )
