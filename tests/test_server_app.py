import json
from pathlib import Path

from google.api_core.exceptions import RetryError

import server.routes
import trainer.job
from server.config import ServerConfig
from server.dependencies import (
    get_cancellation_requester,
    get_config,
    get_execution_outcome_reader,
    get_result_event_publisher,
    get_result_repository,
    get_submission_repository,
)
from server.main import create_app
from server.services.execution_reconciliation import ExecutionOutcome
from tests.fakes import FakeResultRepository, FakeSubmissionRepository

FIXTURE_DIR = Path(__file__).parent / "fixtures"
EXECUTION_NAME = (
    "projects/test/locations/asia-northeast1/jobs/test-trainer/"
    "executions/test-trainer-abcde"
)
PENDING_MESSAGE = "cancellation still pending"


class CompletedCancellationOperation:
    def __init__(self):
        self.timeouts = []

    def result(self, *, timeout):
        self.timeouts.append(timeout)
        return object()


class PendingCancellationOperation:
    def result(self, *, timeout):
        raise RetryError(PENDING_MESSAGE, TimeoutError())


def build_test_app(
    submission_repository: FakeSubmissionRepository,
    result_repository: FakeResultRepository,
    *,
    read_execution_outcome=lambda _config, _execution_name: None,
    request_cancellation=lambda _config, _execution_name: None,
    publish_result_event=lambda **_kwargs: None,
):
    app = create_app()
    app.dependency_overrides[get_config] = lambda: ServerConfig(
        db_id="test-db",
        region="asia-northeast1",
        job_path="projects/test/locations/asia-northeast1/jobs/test-trainer",
        project_id="test-project",
        pubsub_topic="test-topic",
    )
    app.dependency_overrides[get_submission_repository] = lambda: submission_repository
    app.dependency_overrides[get_result_repository] = lambda: result_repository
    app.dependency_overrides[get_execution_outcome_reader] = lambda: (
        read_execution_outcome
    )
    app.dependency_overrides[get_cancellation_requester] = lambda: request_cancellation
    app.dependency_overrides[get_result_event_publisher] = lambda: publish_result_event
    return app


def test_create_app_registers_routes():
    app = create_app()

    paths = {route.path for route in app.routes}

    assert "/submissions" in paths
    assert "/submissions/{submission_id}/train" in paths
    assert "/submissions/{submission_id}/cancel" in paths
    assert "/results/{submission_id}" in paths


def test_create_submission_persists_default_payload():
    from fastapi.testclient import TestClient

    submission_repository = FakeSubmissionRepository()
    result_repository = FakeResultRepository()
    client = TestClient(build_test_app(submission_repository, result_repository))

    response = client.post("/submissions", json={})

    assert response.status_code == 200
    submission_id = response.json()["submission_id"]
    cancel_token = response.json()["cancel_token"]
    submission = submission_repository.fetch(submission_id)
    assert submission["submission_id"] == submission_id
    assert len(cancel_token) >= 32
    assert submission["control"]["cancel_token_hash"] != cancel_token
    assert len(submission["control"]["cancel_token_hash"]) == 64
    assert cancel_token not in json.dumps(submission)
    scenario = submission["scenario"]
    assert scenario["schema_version"] == "scenario-bundle.v0"
    assert scenario["world"]["goal"]["position"] == {"x": 8.5, "z": 8.5}
    assert scenario["robot"]["type"] == "simple_robot"
    assert scenario["robot"]["action_space"]["layout"] == ["forward", "turn"]
    assert scenario["training"]["algorithm"] == "ppo"


def test_create_submission_accepts_envforge_navigation_fixture():
    from fastapi.testclient import TestClient

    submission_repository = FakeSubmissionRepository()
    result_repository = FakeResultRepository()
    client = TestClient(build_test_app(submission_repository, result_repository))
    fixture_path = FIXTURE_DIR / "envforge" / "navigation_default_scenario_bundle.json"
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))

    response = client.post("/submissions", json=payload)

    assert response.status_code == 200
    submission_id = response.json()["submission_id"]
    submission = submission_repository.fetch(submission_id)
    scenario = submission["scenario"]
    assert scenario["scenario_id"] == "navigation_default"
    assert scenario["world"]["bounds"]["min"] == {"x": -8.0, "z": -6.0}
    assert scenario["robot"]["start_pose"]["position"] == {"x": -6.0, "z": -4.0}
    assert scenario["training"]["max_episode_steps"] == 1000


def test_train_queues_result_and_runs_job(monkeypatch):
    from fastapi.testclient import TestClient

    submission_repository = FakeSubmissionRepository(
        initial_submissions={
            "submission-1": {
                "submission_id": "submission-1",
                "control": {"cancel_token_hash": "a" * 64},
            },
        },
    )
    result_repository = FakeResultRepository()
    calls = []

    def run_job(config, submission_id):
        calls.append((config, submission_id))
        return EXECUTION_NAME

    monkeypatch.setattr(server.routes, "run_training_job", run_job)
    client = TestClient(build_test_app(submission_repository, result_repository))

    response = client.post("/submissions/submission-1/train")

    assert response.status_code == 200
    assert response.json() == {"status": "accepted", "submission_id": "submission-1"}
    assert calls[0][1] == "submission-1"
    assert submission_repository.fetch_control("submission-1").execution_name == (
        EXECUTION_NAME
    )
    result = result_repository.fetch("submission-1")
    assert result["status"] == "queued"
    assert result["progress"]["phase"] == "queued"


def test_train_marks_result_failed_when_job_start_fails(monkeypatch):
    from fastapi.testclient import TestClient

    submission_repository = FakeSubmissionRepository(
        initial_submissions={
            "submission-1": {
                "submission_id": "submission-1",
                "control": {"cancel_token_hash": "a" * 64},
            },
        },
    )
    result_repository = FakeResultRepository()

    def raise_job_error(config, submission_id):
        raise RuntimeError("boom")

    monkeypatch.setattr(server.routes, "run_training_job", raise_job_error)
    client = TestClient(build_test_app(submission_repository, result_repository))

    response = client.post("/submissions/submission-1/train")

    assert response.status_code == 500
    assert response.json() == {"detail": "Failed to start trainer job"}
    result = result_repository.fetch("submission-1")
    assert result["status"] == "failed"
    assert result["progress"]["phase"] == "failed"
    assert result["error"] == "Failed to start trainer job"


def test_cancel_running_job_persists_and_publishes_transitions():
    from fastapi.testclient import TestClient

    submission_repository = FakeSubmissionRepository()
    result_repository = FakeResultRepository()
    cancellation_calls = []
    published_events = []
    operation = CompletedCancellationOperation()

    def request_cancellation(config, execution_name):
        cancellation_calls.append((config, execution_name))
        return operation

    client = TestClient(
        build_test_app(
            submission_repository,
            result_repository,
            request_cancellation=request_cancellation,
            publish_result_event=lambda **kwargs: published_events.append(kwargs),
        ),
    )
    create_response = client.post("/submissions", json={})
    submission_id = create_response.json()["submission_id"]
    cancel_token = create_response.json()["cancel_token"]
    submission_repository.set_execution_name(submission_id, EXECUTION_NAME)
    result_repository.create_queued(submission_id)
    result_repository.write_update(
        submission_id,
        status="running",
        progress={
            "phase": "running",
            "current_step": 12,
            "total_steps": 100,
            "message": "Training",
        },
    )

    response = client.post(
        f"/submissions/{submission_id}/cancel",
        headers={"Authorization": f"Bearer {cancel_token}"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"
    assert response.json()["progress"] == {
        "phase": "cancelled",
        "current_step": 12,
        "total_steps": 100,
        "message": "Training cancelled",
    }
    assert cancellation_calls[0][1] == EXECUTION_NAME
    assert len(operation.timeouts) == 1
    assert [event["status"].value for event in published_events] == [
        "cancelling",
        "cancelled",
    ]


def test_cancel_rejects_missing_or_invalid_capability_token():
    from fastapi.testclient import TestClient

    submission_repository = FakeSubmissionRepository()
    result_repository = FakeResultRepository()
    cancellation_calls = []
    client = TestClient(
        build_test_app(
            submission_repository,
            result_repository,
            request_cancellation=lambda *args: cancellation_calls.append(args),
        ),
    )
    create_response = client.post("/submissions", json={})
    submission_id = create_response.json()["submission_id"]
    submission_repository.set_execution_name(submission_id, EXECUTION_NAME)
    result_repository.create_queued(submission_id)

    missing_response = client.post(f"/submissions/{submission_id}/cancel")
    invalid_response = client.post(
        f"/submissions/{submission_id}/cancel",
        headers={"Authorization": "Bearer wrong-token"},
    )

    assert missing_response.status_code == 403
    assert invalid_response.status_code == 403
    assert cancellation_calls == []


def test_cancel_returns_accepted_while_cloud_run_cancellation_is_pending():
    from fastapi.testclient import TestClient

    submission_repository = FakeSubmissionRepository()
    result_repository = FakeResultRepository()
    client = TestClient(
        build_test_app(
            submission_repository,
            result_repository,
            request_cancellation=lambda _config, _execution_name: (
                PendingCancellationOperation()
            ),
        ),
    )
    create_response = client.post("/submissions", json={})
    submission_id = create_response.json()["submission_id"]
    cancel_token = create_response.json()["cancel_token"]
    submission_repository.set_execution_name(submission_id, EXECUTION_NAME)
    result_repository.create_queued(submission_id)

    response = client.post(
        f"/submissions/{submission_id}/cancel",
        headers={"Authorization": f"Bearer {cancel_token}"},
    )

    assert response.status_code == 202
    assert response.json()["status"] == "cancelling"


def test_cancel_rejects_completed_job():
    from fastapi.testclient import TestClient

    submission_repository = FakeSubmissionRepository()
    result_repository = FakeResultRepository()
    client = TestClient(build_test_app(submission_repository, result_repository))
    create_response = client.post("/submissions", json={})
    submission_id = create_response.json()["submission_id"]
    cancel_token = create_response.json()["cancel_token"]
    submission_repository.set_execution_name(submission_id, EXECUTION_NAME)
    result_repository.results[submission_id] = {
        "submission_id": submission_id,
        "status": "completed",
    }

    response = client.post(
        f"/submissions/{submission_id}/cancel",
        headers={"Authorization": f"Bearer {cancel_token}"},
    )

    assert response.status_code == 409


def test_cancel_is_idempotent_after_job_is_cancelled():
    from fastapi.testclient import TestClient

    submission_repository = FakeSubmissionRepository()
    result_repository = FakeResultRepository()
    cancellation_calls = []
    client = TestClient(
        build_test_app(
            submission_repository,
            result_repository,
            request_cancellation=lambda *args: cancellation_calls.append(args),
        ),
    )
    create_response = client.post("/submissions", json={})
    submission_id = create_response.json()["submission_id"]
    cancel_token = create_response.json()["cancel_token"]
    submission_repository.set_execution_name(submission_id, EXECUTION_NAME)
    result_repository.results[submission_id] = {
        "submission_id": submission_id,
        "status": "cancelled",
        "progress": {
            "phase": "cancelled",
            "current_step": 12,
            "total_steps": 100,
            "message": "Training cancelled",
        },
    }

    response = client.post(
        f"/submissions/{submission_id}/cancel",
        headers={"Authorization": f"Bearer {cancel_token}"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"
    assert cancellation_calls == []


def test_get_result_returns_existing_result():
    from fastapi.testclient import TestClient

    submission_repository = FakeSubmissionRepository()
    result_repository = FakeResultRepository(
        initial_results={
            "submission-1": {
                "submission_id": "submission-1",
                "status": "completed",
            },
        },
    )
    client = TestClient(build_test_app(submission_repository, result_repository))

    response = client.get("/results/submission-1")

    assert response.status_code == 200
    assert response.json() == {
        "submission_id": "submission-1",
        "status": "completed",
    }


def test_get_result_marks_active_result_failed_after_exact_cloud_run_failure():
    from fastapi.testclient import TestClient

    submission_repository = FakeSubmissionRepository()
    submission_repository.submissions["submission-1"] = {
        "submission_id": "submission-1",
        "control": {
            "cancel_token_hash": "a" * 64,
            "execution_name": (
                "projects/test/locations/asia-northeast1/jobs/test-trainer/"
                "executions/test-trainer-abcde"
            ),
        },
    }
    result_repository = FakeResultRepository(
        initial_results={
            "submission-1": {
                "submission_id": "submission-1",
                "status": "running",
                "progress": {
                    "phase": "running",
                    "current_step": 0,
                    "total_steps": 1500000,
                    "message": "Training",
                },
            },
        },
    )

    def read_execution_outcome(config, execution_name):
        assert execution_name.endswith("/executions/test-trainer-abcde")
        assert config.job_path.endswith("/jobs/test-trainer")
        return ExecutionOutcome(
            status="failed",
            message="The configured timeout was reached.",
        )

    client = TestClient(
        build_test_app(
            submission_repository,
            result_repository,
            read_execution_outcome=read_execution_outcome,
        ),
    )

    response = client.get("/results/submission-1")

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "failed"
    assert result["progress"]["phase"] == "failed"
    assert result["progress"]["total_steps"] == 1500000
    assert "test-trainer-abcde" in result["error"]
    assert "configured timeout" in result["error"]


def test_get_result_marks_cancelling_result_cancelled_after_exact_execution():
    from fastapi.testclient import TestClient

    submission_repository = FakeSubmissionRepository(
        initial_submissions={
            "submission-1": {
                "submission_id": "submission-1",
                "control": {
                    "cancel_token_hash": "a" * 64,
                    "execution_name": (
                        "projects/test/locations/asia-northeast1/jobs/test-trainer/"
                        "executions/test-trainer-abcde"
                    ),
                },
            },
        },
    )
    result_repository = FakeResultRepository(
        initial_results={
            "submission-1": {
                "submission_id": "submission-1",
                "status": "cancelling",
                "progress": {
                    "phase": "cancelling",
                    "current_step": 12,
                    "total_steps": 100,
                    "message": "Cancelling training",
                },
            },
        },
    )
    published_events = []
    client = TestClient(
        build_test_app(
            submission_repository,
            result_repository,
            read_execution_outcome=lambda _config, _execution_name: ExecutionOutcome(
                status="cancelled",
                message="Cloud Run execution cancelled",
            ),
            publish_result_event=lambda **kwargs: published_events.append(kwargs),
        ),
    )

    response = client.get("/results/submission-1")

    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"
    assert response.json()["progress"] == {
        "phase": "cancelled",
        "current_step": 12,
        "total_steps": 100,
        "message": "Training cancelled",
    }
    assert [event["status"].value for event in published_events] == ["cancelled"]


def test_get_result_returns_404_for_missing_result():
    from fastapi.testclient import TestClient

    submission_repository = FakeSubmissionRepository()
    result_repository = FakeResultRepository()
    client = TestClient(build_test_app(submission_repository, result_repository))

    response = client.get("/results/missing")

    assert response.status_code == 404


def test_submission_train_and_result_flow_integrates_with_trainer(monkeypatch):
    from fastapi.testclient import TestClient

    submission_repository = FakeSubmissionRepository()
    result_repository = FakeResultRepository()
    published_events = []

    def run_trainer(config, submission_id):
        trainer.job.run_training_job(
            trainer.job.TrainerConfig(
                db_id=config.db_id,
                model_bucket="model-bucket",
                submission_id=submission_id,
                pubsub_topic="test-topic",
                project_id="test-project",
            ),
            create_db=lambda db_id: object(),
            create_submission_repository=lambda db: submission_repository,
            create_result_repository=lambda db: result_repository,
            train_model=lambda **kwargs: {"score": 1.0},
            upload_model=lambda **kwargs: {
                "model": {
                    "bucket": "model-bucket",
                    "path": f"results/{submission_id}/model/policy.zip",
                },
            },
            publish_event=lambda **kwargs: published_events.append(kwargs),
        )
        return EXECUTION_NAME

    monkeypatch.setattr(server.routes, "run_training_job", run_trainer)
    client = TestClient(build_test_app(submission_repository, result_repository))

    create_response = client.post("/submissions", json={})
    submission_id = create_response.json()["submission_id"]

    train_response = client.post(f"/submissions/{submission_id}/train")
    result_response = client.get(f"/results/{submission_id}")

    assert create_response.status_code == 200
    assert train_response.status_code == 200
    assert result_response.status_code == 200
    assert result_response.json()["status"] == "completed"
    summary = result_response.json()["summary"]
    assert summary["score"] == 1.0
    assert summary["training_timesteps"] == 5000
    assert summary["training_seed"] == 10
    assert result_response.json()["result_bundle"]["summary"] == {
        "training_timesteps": 5000,
        "training_seed": 10,
        "success_rate": None,
        "average_episode_reward": None,
        "average_episode_steps": None,
    }
    assert result_response.json()["artifacts"]["model"]["bucket"] == "model-bucket"
    assert [event["status"].value for event in published_events] == [
        "starting",
        "running",
        "completed",
    ]
