from server.config import ServerConfig
from server.dependencies import get_config, get_db
from server.main import create_app
from tests.fakes import FakeDb


def build_test_app(fake_db: FakeDb):
	app = create_app()
	app.dependency_overrides[get_db] = lambda: fake_db
	app.dependency_overrides[get_config] = lambda: ServerConfig(
		db_id="test-db",
		region="asia-northeast1",
		job_path="projects/test/locations/asia-northeast1/jobs/test-trainer",
	)
	return app


def test_create_app_registers_routes():
	app = create_app()

	paths = {route.path for route in app.routes}

	assert "/submissions" in paths
	assert "/submissions/{submission_id}/train" in paths
	assert "/results/{submission_id}" in paths


def test_create_submission_persists_default_payload():
	from fastapi.testclient import TestClient

	fake_db = FakeDb()
	client = TestClient(build_test_app(fake_db))

	response = client.post("/submissions", json={})

	assert response.status_code == 200
	submission_id = response.json()["submission_id"]
	submission = fake_db.collections["submissions"][submission_id]
	assert submission["submission_id"] == submission_id
	assert submission["environment"]["size"] == [2, 2]
	assert submission["environment"]["goal"] == {"x": 1, "y": 1}
	assert submission["environment"]["robot_start"] == {"x": 0, "y": 0}
	assert submission["robot"] == {"type": "simple"}
	assert submission["training"]["algorithm"] == "ppo"


def test_train_queues_result_and_runs_job(monkeypatch):
	from fastapi.testclient import TestClient
	import server.routes

	fake_db = FakeDb()
	fake_db.collections["submissions"]["submission-1"] = {"submission_id": "submission-1"}
	calls = []
	monkeypatch.setattr(
		server.routes,
		"run_training_job",
		lambda config, submission_id: calls.append((config, submission_id)),
	)
	client = TestClient(build_test_app(fake_db))

	response = client.post("/submissions/submission-1/train")

	assert response.status_code == 200
	assert response.json() == {"status": "accepted", "submission_id": "submission-1"}
	assert calls[0][1] == "submission-1"
	result = fake_db.collections["results"]["submission-1"]
	assert result["status"] == "queued"
	assert result["progress"]["phase"] == "queued"


def test_train_marks_result_failed_when_job_start_fails(monkeypatch):
	from fastapi.testclient import TestClient
	import server.routes

	fake_db = FakeDb()
	fake_db.collections["submissions"]["submission-1"] = {"submission_id": "submission-1"}

	def raise_job_error(config, submission_id):
		raise RuntimeError("boom")

	monkeypatch.setattr(server.routes, "run_training_job", raise_job_error)
	client = TestClient(build_test_app(fake_db))

	response = client.post("/submissions/submission-1/train")

	assert response.status_code == 500
	assert response.json() == {"detail": "Failed to start trainer job"}
	result = fake_db.collections["results"]["submission-1"]
	assert result["status"] == "failed"
	assert result["progress"]["phase"] == "failed"
	assert result["error"] == "Failed to start trainer job"


def test_get_result_returns_existing_result():
	from fastapi.testclient import TestClient

	fake_db = FakeDb()
	fake_db.collections["results"]["submission-1"] = {
		"submission_id": "submission-1",
		"status": "completed",
	}
	client = TestClient(build_test_app(fake_db))

	response = client.get("/results/submission-1")

	assert response.status_code == 200
	assert response.json() == {
		"submission_id": "submission-1",
		"status": "completed",
	}


def test_get_result_returns_404_for_missing_result():
	from fastapi.testclient import TestClient

	fake_db = FakeDb()
	client = TestClient(build_test_app(fake_db))

	response = client.get("/results/missing")

	assert response.status_code == 404
