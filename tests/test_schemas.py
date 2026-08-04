import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from embodiedlab.schemas import (
    DistanceDeltaRewardComponent,
    Position2D,
    ScenarioBundle,
    WorldSpec,
    build_submission_document,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _reward_components(**overrides):
    components = ScenarioBundle().model_dump(mode="json")["reward"]["components"]
    for component in components:
        component.update(overrides.get(component["name"], {}))
    return components


def test_scenario_bundle_defaults_are_valid():
    scenario = ScenarioBundle()

    assert scenario.schema_version == "scenario-bundle.v0"
    assert scenario.scenario_id == "scenario_demo_001"
    assert scenario.world.coordinate_system == "envforge_xz_meters"
    assert scenario.world.goal.position.x == 8.5
    assert scenario.compatibility.robot_version == "simple_robot.v1"
    assert scenario.robot.type == "simple_robot"
    assert scenario.robot.radius == 0.45
    assert scenario.world.goal.radius == scenario.robot.radius
    assert scenario.robot.action_space.layout == ["forward", "turn"]
    assert [sensor.id for sensor in scenario.sensors] == [
        "front_camera",
        "front_distance",
    ]
    assert scenario.training.algorithm == "ppo"
    assert scenario.training.max_episode_steps == 512


def test_build_submission_document_returns_firestore_payload():
    scenario = ScenarioBundle()

    payload = build_submission_document(
        "submission-1",
        scenario,
        cancel_token_hash="a" * 64,
    )

    assert payload["submission_id"] == "submission-1"
    assert isinstance(payload["created_at"], str)
    assert payload["scenario"]["schema_version"] == "scenario-bundle.v0"
    assert payload["scenario"]["scenario_id"] == "scenario_demo_001"
    assert payload["scenario"]["compatibility"]["robot_version"] == "simple_robot.v1"
    assert payload["scenario"]["robot"]["type"] == "simple_robot"
    assert payload["scenario"]["robot"]["radius"] == 0.45
    assert payload["scenario"]["training"]["algorithm"] == "ppo"
    assert payload["control"] == {
        "cancel_token_hash": "a" * 64,
        "execution_name": None,
    }


def test_scenario_bundle_accepts_documented_shape():
    scenario = ScenarioBundle.model_validate(
        {
            "schema_version": "scenario-bundle.v0",
            "scenario_id": "scenario_custom",
            "world": {
                "bounds": {
                    "min": {"x": 0.0, "z": 0.0},
                    "max": {"x": 4.0, "z": 4.0},
                },
                "static_obstacles": [
                    {
                        "id": "box_001",
                        "shape": "box",
                        "center": {"x": 2.0, "z": 2.0},
                        "size": {"x": 1.0, "z": 1.0},
                    },
                ],
                "goal": {
                    "id": "goal_001",
                    "position": {"x": 3.5, "z": 3.5},
                    "radius": 0.5,
                },
            },
            "robot": {
                "radius": 0.3,
                "start_pose": {
                    "position": {"x": 1.0, "z": 1.0},
                    "rotation_y_degrees": 0.0,
                },
                "action_space": {
                    "type": "continuous",
                    "layout": ["forward", "turn"],
                },
            },
            "sensors": [
                {
                    "id": "front_camera",
                    "type": "forward_camera",
                    "width": 112,
                    "height": 84,
                    "semantic_mode": "traversable_vs_blocked",
                    "mount_height_meters": 0.6,
                    "pitch_degrees": 0.0,
                    "vertical_fov_degrees": 60.0,
                    "near_clip_meters": 0.05,
                    "far_clip_meters": 5.0,
                },
                {
                    "id": "front_distance",
                    "type": "distance_sensor",
                    "range_meters": 5.0,
                    "direction": "forward",
                },
            ],
            "reward": {
                "components": _reward_components(
                    goal_progress={"weight": 0.5},
                ),
            },
            "training": {
                "algorithm": "ppo",
                "timesteps": 5000,
                "seed": 10,
                "max_episode_steps": 512,
            },
        },
    )

    assert scenario.scenario_id == "scenario_custom"
    assert scenario.world.static_obstacles[0].id == "box_001"
    assert scenario.robot.radius == 0.3
    assert scenario.sensors[0].width == 112
    assert scenario.sensors[0].height == 84
    assert scenario.sensors[0].mount_height_meters == 0.6
    assert scenario.sensors[0].mount_height_min_meters is None
    assert scenario.sensors[0].mount_height_max_meters is None
    assert isinstance(scenario.reward.components[1], DistanceDeltaRewardComponent)


def test_forward_camera_mount_height_range_must_be_complete_and_ordered():
    with pytest.raises(ValidationError, match="requires both min and max"):
        ScenarioBundle(
            sensors=[
                {
                    "id": "front_camera",
                    "type": "forward_camera",
                    "mount_height_min_meters": 0.1,
                },
            ],
        )

    with pytest.raises(ValidationError, match="min must be less than or equal"):
        ScenarioBundle(
            sensors=[
                {
                    "id": "front_camera",
                    "type": "forward_camera",
                    "mount_height_min_meters": 1.0,
                    "mount_height_max_meters": 0.1,
                },
            ],
        )


def test_envforge_navigation_fixture_matches_scenario_bundle_contract():
    fixture_path = FIXTURE_DIR / "envforge" / "navigation_default_scenario_bundle.json"
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))

    scenario = ScenarioBundle.model_validate(payload)

    assert scenario.scenario_id == "navigation_default"
    assert scenario.world.bounds.min.x == -8.0
    assert scenario.world.bounds.max.z == 6.0
    assert scenario.robot.start_pose.position.x == -6.0
    assert scenario.world.goal.radius == 0.45
    assert scenario.world.goal.radius == scenario.robot.radius
    assert scenario.robot.action_space.layout == ["forward", "turn"]
    assert all("height" in wall for wall in payload["world"]["static_walls"])
    assert all(
        "height" in obstacle for obstacle in payload["world"]["static_obstacles"]
    )
    assert [sensor.id for sensor in scenario.sensors] == [
        "front_camera",
        "front_distance",
    ]
    assert set(payload["sensors"][0]) >= {
        "width",
        "height",
        "semantic_mode",
        "mount_height_meters",
        "mount_height_min_meters",
        "mount_height_max_meters",
        "pitch_degrees",
        "vertical_fov_degrees",
        "near_clip_meters",
        "far_clip_meters",
    }
    assert scenario.sensors[0].mount_height_min_meters == 0.6
    assert scenario.sensors[0].mount_height_max_meters == 0.6
    assert scenario.training.max_episode_steps == 1000
    assert set(payload["training"]) >= {
        "algorithm",
        "timesteps",
        "seed",
        "max_episode_steps",
        "n_envs",
        "cpu_count",
        "torch_num_threads",
        "n_steps",
        "batch_size",
        "n_epochs",
        "gamma",
        "learning_rate",
        "ent_coef",
        "eval_episodes",
    }
    assert scenario.training.n_envs == 1
    assert scenario.training.cpu_count is None
    assert scenario.training.torch_num_threads is None
    assert scenario.training.n_epochs == 3


@pytest.mark.parametrize(
    "world",
    [
        {"bounds": {"min": {"x": 1.0, "z": 0.0}, "max": {"x": 1.0, "z": 2.0}}},
        {"goal": {"id": "goal_001", "position": {"x": 99.0, "z": 1.0}, "radius": 1}},
        {
            "static_obstacles": [
                {
                    "id": "box_001",
                    "shape": "box",
                    "center": {"x": 99.0, "z": 1.0},
                    "size": {"x": 1.0, "z": 1.0},
                }
            ],
        },
    ],
)
def test_world_rejects_invalid_geometry(world):
    with pytest.raises(ValidationError):
        WorldSpec(**world)


def test_scenario_rejects_robot_outside_bounds():
    with pytest.raises(ValidationError):
        ScenarioBundle(
            robot={
                "start_pose": {
                    "position": {"x": 99.0, "z": 1.0},
                    "rotation_y_degrees": 0.0,
                }
            }
        )


def test_scenario_rejects_duplicate_sensor_ids():
    with pytest.raises(ValidationError):
        ScenarioBundle(
            sensors=[
                {"id": "front_distance", "type": "distance_sensor"},
                {"id": "front_distance", "type": "distance_sensor"},
            ]
        )


def test_scenario_rejects_missing_reward_target():
    with pytest.raises(ValidationError):
        ScenarioBundle(
            reward={
                "components": _reward_components(
                    goal_progress={"target": "missing_goal"},
                ),
            }
        )


def test_scenario_rejects_missing_reward_component():
    with pytest.raises(ValidationError, match="at least 8"):
        ScenarioBundle(
            reward={"components": _reward_components()[:-1]},
        )


def test_scenario_rejects_duplicate_reward_component():
    components = _reward_components()
    components[-1] = components[0]

    with pytest.raises(ValidationError, match="must be unique"):
        ScenarioBundle(reward={"components": components})


def test_scenario_rejects_unknown_reward_component():
    with pytest.raises(ValidationError, match="unknown="):
        ScenarioBundle(
            reward={
                "components": _reward_components(
                    movement_threshold={"name": "custom_reward"},
                ),
            },
        )


def test_scenario_rejects_wrong_reward_component_type():
    with pytest.raises(ValidationError, match="must use type terminal_reward"):
        ScenarioBundle(
            reward={
                "components": _reward_components(
                    goal_reached={"type": "per_step"},
                ),
            },
        )


def test_training_rejects_eval_replay_over_sdk_row_limit():
    with pytest.raises(
        ValidationError,
        match=r"eval_episodes \* max_episode_steps",
    ):
        ScenarioBundle(
            training={
                "max_episode_steps": 1001,
                "eval_episodes": 100,
            },
        )


def test_action_layout_is_fixed_for_initial_robot():
    with pytest.raises(ValidationError):
        ScenarioBundle(
            robot={
                "action_space": {
                    "type": "continuous",
                    "layout": ["turn", "forward"],
                }
            }
        )


def test_position_model_uses_xz_plane():
    position = Position2D(x=1.0, z=2.0)

    assert position.model_dump() == {"x": 1.0, "z": 2.0}
