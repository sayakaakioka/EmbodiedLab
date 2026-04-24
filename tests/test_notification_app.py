import base64
import json

from fastapi.testclient import TestClient

from embodiedlab.result_models import (
    ResultStatus,
    build_result_message,
    running_progress,
)
from notification.main import create_app


def build_push_payload(event: dict) -> dict:
    return {
        "message": {
            "data": base64.b64encode(json.dumps(event).encode("utf-8")).decode("utf-8"),
        },
    }


def test_pubsub_push_rejects_invalid_payload():
    client = TestClient(create_app())

    response = client.post("/internal/pubsub/push", json={})

    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid Pub/Sub message"}


def test_pubsub_push_rejects_invalid_encoded_event():
    client = TestClient(create_app())

    response = client.post(
        "/internal/pubsub/push",
        json={"message": {"data": "not-base64"}},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid encoded event"}


def test_pubsub_push_rejects_invalid_result_event():
    client = TestClient(create_app())

    response = client.post(
        "/internal/pubsub/push",
        json=build_push_payload({"submission_id": "submission-1"}),
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid result event"}


def test_pubsub_push_fans_out_to_matching_websocket():
    client = TestClient(create_app())
    event = build_result_message(
        submission_id="submission-1",
        status=ResultStatus.RUNNING,
        progress=running_progress(100).model_copy(update={"current_step": 10}),
    )

    with client.websocket_connect("/ws/results/submission-1") as websocket:
        connected_message = websocket.receive_json()

        response = client.post("/internal/pubsub/push", json=build_push_payload(event))
        pushed_message = websocket.receive_json()

    assert connected_message == {
        "type": "connected",
        "submission_id": "submission-1",
    }
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert pushed_message == event
