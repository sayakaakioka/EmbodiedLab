import pytest
from pydantic import ValidationError

from embodiedlab.schemas import (
    DistanceDeltaRewardComponent,
    Position2D,
    ScenarioBundle,
    WorldSpec,
    build_submission_document,
)


def test_scenario_bundle_defaults_are_valid():
    scenario = ScenarioBundle()

    assert scenario.schema_version == "scenario-bundle.v0"
    assert scenario.scenario_id == "scenario_demo_001"
    assert scenario.world.coordinate_system == "envforge_xz_meters"
    assert scenario.world.goal.position.x == 8.5
    assert scenario.robot.type == "simple_robot"
    assert scenario.robot.action_space.layout == ["forward", "turn"]
    assert [sensor.id for sensor in scenario.sensors] == [
        "front_camera",
        "front_distance",
    ]
    assert scenario.training.algorithm == "ppo"
    assert scenario.training.max_episode_steps == 512


def test_build_submission_document_returns_firestore_payload():
    scenario = ScenarioBundle()

    payload = build_submission_document("submission-1", scenario)

    assert payload["submission_id"] == "submission-1"
    assert isinstance(payload["created_at"], str)
    assert payload["scenario"]["schema_version"] == "scenario-bundle.v0"
    assert payload["scenario"]["scenario_id"] == "scenario_demo_001"
    assert payload["scenario"]["robot"]["type"] == "simple_robot"
    assert payload["scenario"]["training"]["algorithm"] == "ppo"


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
                    "width": 84,
                    "height": 84,
                    "semantic_mode": "traversable_vs_blocked",
                },
                {
                    "id": "front_distance",
                    "type": "distance_sensor",
                    "range_meters": 5.0,
                    "direction": "forward",
                },
            ],
            "reward": {
                "components": [
                    {
                        "name": "goal_progress",
                        "type": "distance_delta",
                        "target": "goal_001",
                        "weight": 0.5,
                    },
                ],
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
    assert isinstance(scenario.reward.components[0], DistanceDeltaRewardComponent)


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
                "components": [
                    {
                        "name": "goal_progress",
                        "type": "distance_delta",
                        "target": "missing_goal",
                        "weight": 1.0,
                    }
                ]
            }
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
