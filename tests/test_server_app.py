import server.routes
import trainer.job
from server.config import ServerConfig
from server.dependencies import (
    get_config,
    get_result_repository,
    get_submission_repository,
)
from server.main import create_app
from tests.fakes import FakeResultRepository, FakeSubmissionRepository


def build_test_app(
    submission_repository: FakeSubmissionRepository,
    result_repository: FakeResultRepository,
):
    app = create_app()
    app.dependency_overrides[get_config] = lambda: ServerConfig(
        db_id="test-db",
        region="asia-northeast1",
        job_path="projects/test/locations/asia-northeast1/jobs/test-trainer",
    )
    app.dependency_overrides[get_submission_repository] = lambda: submission_repository
    app.dependency_overrides[get_result_repository] = lambda: result_repository
    return app


def test_create_app_registers_routes():
    app = create_app()

    paths = {route.path for route in app.routes}

    assert "/submissions" in paths
    assert "/submissions/{submission_id}/train" in paths
    assert "/results/{submission_id}" in paths


def test_create_submission_persists_default_payload():
    from fastapi.testclient import TestClient

    submission_repository = FakeSubmissionRepository()
    result_repository = FakeResultRepository()
    client = TestClient(build_test_app(submission_repository, result_repository))

    response = client.post("/submissions", json={})

    assert response.status_code == 200
    submission_id = response.json()["submission_id"]
    submission = submission_repository.fetch(submission_id)
    assert submission["submission_id"] == submission_id
    assert submission["environment"]["size"] == [2, 2]
    assert submission["environment"]["goal"] == {"x": 1, "y": 1}
    assert submission["environment"]["robot_start"] == {"x": 0, "y": 0}
    assert submission["robot"] == {"type": "simple"}
    assert submission["training"]["algorithm"] == "ppo"


def test_train_queues_result_and_runs_job(monkeypatch):
    from fastapi.testclient import TestClient

    submission_repository = FakeSubmissionRepository(
        initial_submissions={"submission-1": {"submission_id": "submission-1"}},
    )
    result_repository = FakeResultRepository()
    calls = []
    monkeypatch.setattr(
        server.routes,
        "run_training_job",
        lambda config, submission_id: calls.append((config, submission_id)),
    )
    client = TestClient(build_test_app(submission_repository, result_repository))

    response = client.post("/submissions/submission-1/train")

    assert response.status_code == 200
    assert response.json() == {"status": "accepted", "submission_id": "submission-1"}
    assert calls[0][1] == "submission-1"
    result = result_repository.fetch("submission-1")
    assert result["status"] == "queued"
    assert result["progress"]["phase"] == "queued"


def test_train_marks_result_failed_when_job_start_fails(monkeypatch):
    from fastapi.testclient import TestClient

    submission_repository = FakeSubmissionRepository(
        initial_submissions={"submission-1": {"submission_id": "submission-1"}},
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
                    "path": f"models/{submission_id}/policy.zip",
                },
            },
            publish_event=lambda **kwargs: published_events.append(kwargs),
        )

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
    assert result_response.json()["summary"] == {"score": 1.0}
    assert result_response.json()["artifacts"]["model"]["bucket"] == "model-bucket"
    assert [event["status"].value for event in published_events] == [
        "starting",
        "running",
        "completed",
    ]
