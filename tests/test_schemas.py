import pytest
from pydantic import ValidationError

from embodiedlab.schemas import Environment, SubmitRequest, build_submission_document


def test_submit_request_defaults_are_valid():
    req = SubmitRequest()

    assert req.environment.size == [2, 2]
    assert req.environment.goal.x == 1
    assert req.environment.goal.y == 1
    assert req.environment.robot_start.x == 0
    assert req.environment.robot_start.y == 0
    assert req.robot.type == "simple"


def test_build_submission_document_returns_firestore_payload():
    req = SubmitRequest()

    payload = build_submission_document("submission-1", req)

    assert payload["submission_id"] == "submission-1"
    assert isinstance(payload["created_at"], str)
    assert payload["environment"] == {
        "size": [2, 2],
        "obstacles": [],
        "goal": {"x": 1, "y": 1},
        "robot_start": {"x": 0, "y": 0},
    }
    assert payload["robot"] == {"type": "simple"}
    assert payload["training"]["algorithm"] == "ppo"


@pytest.mark.parametrize(
    "environment",
    [
        {"goal": {"x": 2, "y": 0}},
        {"obstacles": [{"x": 1, "y": 1}]},
        {"goal": {"x": 0, "y": 0}, "robot_start": {"x": 0, "y": 0}},
        {"robot_start": {"x": 1, "y": 1}, "obstacles": [{"x": 1, "y": 1}]},
    ],
)
def test_environment_rejects_invalid_layouts(environment):
    with pytest.raises(ValidationError):
        Environment(**environment)


@pytest.mark.parametrize("size", [[1, 2], [2], [2, 2, 2]])
def test_environment_rejects_invalid_size(size):
    with pytest.raises(ValidationError):
        Environment(size=size)
