from __future__ import annotations

import os
import tempfile
import traceback

from embodiedlab.result_models import ResultStatus
from embodiedlab.training.training_config import TrainingConfig
from embodiedlab.training.training_converter import convert_submission_to_spec
from trainer.config import TrainerConfig
from trainer.progress import (
	completed_progress,
	failed_progress,
	running_progress,
	starting_progress,
)
from trainer.repositories import create_firestore_client, fetch_submission, get_result_ref
from trainer.results import update_result


def get_default_train_model():
	from embodiedlab.training.runner import run_gridworld_training

	return run_gridworld_training


def get_default_upload_model():
	from trainer.artifacts import upload_model_to_gcs

	return upload_model_to_gcs


def run_training_job(
	config: TrainerConfig,
	*,
	create_db=create_firestore_client,
	fetch_submission_by_id=fetch_submission,
	train_model=None,
	upload_model=None,
) -> None:
	submission_id = config.submission_id

	db = create_db(config.db_id)
	result_ref = get_result_ref(db, submission_id)

	print(f"trainer_job started: submission_id={submission_id}")

	submission = fetch_submission_by_id(db, submission_id)
	if submission is None:
		update_result(
			result_ref,
			status=ResultStatus.FAILED,
			progress=failed_progress("Submission not found"),
			error="Submission not found",
		)
		print(f"submission not found: submission_id={submission_id}")
		return

	total_steps = 0

	try:
		training = TrainingConfig.model_validate(submission["training"])
		total_steps = training.timesteps
		update_result(
			result_ref,
			status=ResultStatus.STARTING,
			progress=starting_progress(total_steps),
		)

		spec = convert_submission_to_spec(submission)
		train_model = train_model or get_default_train_model()
		upload_model = upload_model or get_default_upload_model()

		update_result(
			result_ref,
			status=ResultStatus.RUNNING,
			progress=running_progress(total_steps),
		)

		with tempfile.TemporaryDirectory() as tmpdir:
			model_base_path = os.path.join(tmpdir, "policy")

			summary = train_model(
				spec=spec,
				training=training,
				model_output_path=model_base_path,
			)

			artifacts = upload_model(
				local_model_base_path=model_base_path,
				bucket_name=config.model_bucket,
				submission_id=submission_id,
			)

		update_result(
			result_ref,
			status=ResultStatus.COMPLETED,
			progress=completed_progress(total_steps),
			summary=summary,
			error=None,
			artifacts=artifacts,
		)

		print(f"trainer_job completed: submission_id={submission_id}")

	except Exception:
		tb = traceback.format_exc()

		update_result(
			result_ref,
			status=ResultStatus.FAILED,
			progress=failed_progress("Training failed", total_steps=total_steps),
			error=tb,
		)

		print(tb)
		raise
