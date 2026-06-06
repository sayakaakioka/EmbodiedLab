import pytest
from pydantic import ValidationError

from embodiedlab.schemas import ScenarioBundle
from embodiedlab.training.training_converter import (
    convert_submission_to_spec,
    describe_runtime_conversion,
    parse_scenario_bundle,
)


def test_parse_scenario_bundle_from_model():
    scenario = ScenarioBundle()

    parsed = parse_scenario_bundle(scenario)

    assert parsed is scenario


def test_parse_scenario_bundle_from_firestore_document():
    scenario = ScenarioBundle().model_dump(mode="json")
    submission = {
        "submission_id": "submission-1",
        "created_at": "2026-04-17T00:00:00+00:00",
        "scenario": scenario,
    }

    parsed = parse_scenario_bundle(submission)

    assert parsed.scenario_id == "scenario_demo_001"


def test_convert_scenario_to_continuous_runtime_spec():
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
        reward={
            "components": [
                {"name": "goal_reached", "type": "terminal_reward", "weight": 101.0},
                {
                    "name": "goal_progress",
                    "type": "distance_delta",
                    "target": "goal_001",
                    "weight": 0.25,
                },
                {"name": "collision_penalty", "type": "collision", "weight": -12.0},
                {"name": "step_penalty", "type": "per_step", "weight": -0.2},
                {"name": "wide_angle_penalty", "type": "per_step", "weight": -0.4},
                {"name": "rear_angle_penalty", "type": "per_step", "weight": -7.0},
                {"name": "inactive_penalty", "type": "per_step", "weight": -0.5},
                {"name": "movement_threshold", "type": "per_step", "weight": 0.02},
            ],
        },
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
    assert spec.reward_weights.goal_reached == 101.0
    assert spec.reward_weights.goal_progress == 0.25
    assert spec.reward_weights.collision_penalty == -12.0
    assert spec.reward_weights.step_penalty == -0.2
    assert spec.reward_weights.wide_angle_penalty == -0.4
    assert spec.reward_weights.rear_angle_penalty == -7.0
    assert spec.reward_weights.inactive_penalty == -0.5
    assert spec.reward_weights.movement_threshold == 0.02
    assert [obstacle.obstacle_id for obstacle in spec.obstacles] == [
        "wall_001",
        "box_001",
    ]
    assert spec.obstacles[1].rotation_y_degrees == 45.0


def test_parse_scenario_bundle_rejects_invalid_dict():
    with pytest.raises(ValidationError):
        parse_scenario_bundle(
            {
                "scenario": {
                    "schema_version": "scenario-bundle.v0",
                    "robot": {
                        "start_pose": {
                            "position": {"x": 99.0, "z": 1.0},
                            "rotation_y_degrees": 0.0,
                        }
                    },
                }
            }
        )
