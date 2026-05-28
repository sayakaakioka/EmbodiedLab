import pytest
from pydantic import ValidationError

from embodiedlab.schemas import ScenarioBundle
from embodiedlab.training.training_converter import (
    convert_submission_to_spec,
    describe_runtime_conversion,
    parse_scenario_bundle,
)
from embodiedlab.training.training_models import GridPosition


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


def test_convert_scenario_model_to_current_runtime_spec():
    scenario = ScenarioBundle()

    spec = convert_submission_to_spec(scenario)

    assert spec.width == 10
    assert spec.height == 10
    assert spec.goal == GridPosition(x=8, y=8)
    assert spec.robot_start == GridPosition(x=1, y=1)
    assert spec.obstacles == frozenset()
    assert spec.robot_type == "simple_robot"
    assert spec.envforge_origin_x == 0.0
    assert spec.envforge_origin_z == 0.0


def test_convert_scenario_dict_to_spec_includes_obstacles():
    scenario = ScenarioBundle(
        world={
            "static_obstacles": [
                {
                    "id": "box_001",
                    "shape": "box",
                    "center": {"x": 4.5, "z": 5.0},
                    "size": {"x": 1.0, "z": 1.0},
                }
            ],
        },
    )
    submission = {"scenario": scenario.model_dump(mode="json")}

    spec = convert_submission_to_spec(submission)

    assert spec.obstacles == frozenset({GridPosition(x=4, y=5)})


def test_describe_runtime_conversion_marks_grid_adapter_as_lossy():
    scenario = ScenarioBundle(
        world={
            "static_obstacles": [
                {
                    "id": "rotated_box",
                    "shape": "box",
                    "center": {"x": 4.75, "z": 5.25},
                    "size": {"x": 1.2, "z": 0.8},
                    "rotation_y_degrees": 45.0,
                }
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
    )

    conversion = describe_runtime_conversion(scenario)

    assert conversion.source_coordinate_system == "envforge_xz_meters"
    assert conversion.runtime_coordinate_system == "grid_world_cells"
    assert conversion.coordinate_mapping == (
        "subtract_bounds_min_then_floor_envforge_xz_meters_to_non_negative_xy_cells"
    )
    assert conversion.lossy is True
    assert "world.static_obstacles[].size" in conversion.omitted_contract_fields
    assert (
        "world.static_obstacles[].rotation_y_degrees"
        in conversion.omitted_contract_fields
    )
    assert "world.goal.radius" in conversion.omitted_contract_fields
    assert "reward.components" in conversion.omitted_contract_fields
    assert any("floor" in note for note in conversion.notes)


def test_runtime_conversion_description_matches_flooring_behavior():
    scenario = ScenarioBundle(
        world={
            "static_obstacles": [
                {
                    "id": "box_001",
                    "shape": "box",
                    "center": {"x": 4.75, "z": 5.25},
                    "size": {"x": 1.0, "z": 1.0},
                }
            ],
            "goal": {
                "id": "goal_001",
                "position": {"x": 8.9, "z": 8.1},
                "radius": 0.5,
            },
        },
        robot={
            "start_pose": {
                "position": {"x": 1.9, "z": 2.1},
                "rotation_y_degrees": 30.0,
            },
        },
    )

    conversion = describe_runtime_conversion(scenario)
    spec = convert_submission_to_spec(scenario)

    assert conversion.lossy is True
    assert spec.robot_start == GridPosition(x=1, y=2)
    assert spec.goal == GridPosition(x=8, y=8)
    assert spec.obstacles == frozenset({GridPosition(x=4, y=5)})


def test_convert_scenario_with_translated_bounds_uses_bounds_relative_cells():
    scenario = ScenarioBundle(
        world={
            "bounds": {
                "min": {"x": 100.0, "z": 200.0},
                "max": {"x": 110.0, "z": 210.0},
            },
            "static_obstacles": [
                {
                    "id": "box_001",
                    "shape": "box",
                    "center": {"x": 104.75, "z": 205.25},
                    "size": {"x": 1.0, "z": 1.0},
                }
            ],
            "goal": {
                "id": "goal_001",
                "position": {"x": 108.9, "z": 208.1},
                "radius": 0.5,
            },
        },
        robot={
            "start_pose": {
                "position": {"x": 101.9, "z": 202.1},
                "rotation_y_degrees": 30.0,
            },
        },
    )

    spec = convert_submission_to_spec(scenario)

    assert spec.width == 10
    assert spec.height == 10
    assert spec.robot_start == GridPosition(x=1, y=2)
    assert spec.goal == GridPosition(x=8, y=8)
    assert spec.obstacles == frozenset({GridPosition(x=4, y=5)})
    assert spec.envforge_origin_x == 100.0
    assert spec.envforge_origin_z == 200.0


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
