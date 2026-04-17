import pytest

from embodiedlab.schemas import SubmitRequest
from tests.fakes import FakeDb
from trainer.config import TrainerConfig
from trainer.job import run_training_job


def test_run_training_job_updates_result_to_completed():
	config = TrainerConfig(
		db_id="test-db",
		model_bucket="model-bucket",
		submission_id="submission-1",
	)
	fake_db = FakeDb()
	submission = SubmitRequest().model_dump(mode="json")
	calls = []

	def train_model(*, spec, training, model_output_path):
		calls.append(("train", spec, training, model_output_path))
		return {"score": 1.0}

	def upload_model(*, local_model_base_path, bucket_name, submission_id):
		calls.append(("upload", local_model_base_path, bucket_name, submission_id))
		return {"model": {"bucket": bucket_name, "path": f"models/{submission_id}/policy.zip"}}

	run_training_job(
		config,
		create_db=lambda db_id: fake_db,
		fetch_submission_by_id=lambda db, submission_id: submission,
		train_model=train_model,
		upload_model=upload_model,
	)

	result_ref = fake_db.result_document("submission-1")
	statuses = [payload["data"]["status"] for payload in result_ref.payloads]
	assert statuses == ["starting", "running", "completed"]
	assert result_ref.payloads[-1]["data"]["summary"] == {"score": 1.0}
	assert result_ref.payloads[-1]["data"]["artifacts"]["model"]["bucket"] == "model-bucket"
	assert calls[0][0] == "train"
	assert calls[1][0] == "upload"


def test_run_training_job_marks_missing_submission_failed():
	config = TrainerConfig(
		db_id="test-db",
		model_bucket="model-bucket",
		submission_id="submission-1",
	)
	fake_db = FakeDb()

	run_training_job(
		config,
		create_db=lambda db_id: fake_db,
		fetch_submission_by_id=lambda db, submission_id: None,
	)

	payload = fake_db.result_document("submission-1").payloads[0]["data"]
	assert payload["status"] == "failed"
	assert payload["progress"]["message"] == "Submission not found"
	assert payload["error"] == "Submission not found"


def test_run_training_job_marks_invalid_submission_failed():
	config = TrainerConfig(
		db_id="test-db",
		model_bucket="model-bucket",
		submission_id="submission-1",
	)
	fake_db = FakeDb()
	submission = SubmitRequest().model_dump(mode="json")
	submission["training"]["timesteps"] = 0

	with pytest.raises(Exception):
		run_training_job(
			config,
			create_db=lambda db_id: fake_db,
			fetch_submission_by_id=lambda db, submission_id: submission,
			train_model=lambda **kwargs: {"score": 1.0},
			upload_model=lambda **kwargs: {"model": {}},
		)

	payload = fake_db.result_document("submission-1").payloads[0]["data"]
	assert payload["status"] == "failed"
	assert payload["progress"]["total_steps"] == 0
	assert "timesteps" in payload["error"]
