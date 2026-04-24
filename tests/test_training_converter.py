import pytest
from pydantic import ValidationError

from embodiedlab.schemas import SubmitRequest
from embodiedlab.training.training_converter import convert_submission_to_spec
from embodiedlab.training.training_models import GridPosition


def test_convert_submission_model_to_spec():
    req = SubmitRequest()

    spec = convert_submission_to_spec(req)

    assert spec.width == 2
    assert spec.height == 2
    assert spec.goal == GridPosition(x=1, y=1)
    assert spec.robot_start == GridPosition(x=0, y=0)
    assert spec.obstacles == frozenset()
    assert spec.robot_type == "simple"


def test_convert_submission_dict_to_spec_ignores_firestore_metadata():
    submission = SubmitRequest().model_dump(mode="json")
    submission["submission_id"] = "submission-1"
    submission["created_at"] = "2026-04-17T00:00:00+00:00"

    spec = convert_submission_to_spec(submission)

    assert spec.width == 2
    assert spec.height == 2
    assert spec.goal == GridPosition(x=1, y=1)


def test_convert_submission_rejects_invalid_dict():
    with pytest.raises(ValidationError):
        convert_submission_to_spec(
            {
                "environment": {
                    "size": [2, 2],
                    "goal": {"x": 2, "y": 0},
                },
                "robot": {"type": "simple"},
            }
        )
