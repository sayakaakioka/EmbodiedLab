import pytest
from pydantic import ValidationError

from embodiedlab.schemas import ScenarioBundle
from embodiedlab.training.training_converter import (
    convert_submission_to_spec,
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
