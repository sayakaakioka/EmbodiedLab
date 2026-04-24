"""Trainer orchestration for Firestore updates, model training, and event publishing."""

from __future__ import annotations

import traceback
from collections.abc import Callable
from typing import Any

from embodiedlab.repositories import ResultUpdateWriter, SubmissionReader
from embodiedlab.result_models import (
    ResultStatus,
    completed_progress,
    failed_progress,
    running_progress,
    starting_progress,
)
from embodiedlab.training.runner import run_gridworld_training
from trainer.artifacts import upload_model_to_gcs
from trainer.config import TrainerConfig
from trainer.logging_utils import log_trainer_event
from trainer.pubsub import publish_training_event
from trainer.repositories import (
    FirestoreResultRepository,
    FirestoreSubmissionRepository,
    create_firestore_client,
)
from trainer.training_service import (
    TrainModel,
    UploadModel,
    execute_training_run,
    parse_training_submission,
)
from trainer.transitions import TrainerResultTransitions

CreateDb = Callable[[str], Any]
CreateSubmissionRepository = Callable[[Any], SubmissionReader]
CreateResultRepository = Callable[[Any], ResultUpdateWriter]
PublishEvent = Callable[..., None]


def run_training_job(  # noqa: PLR0913
    config: TrainerConfig,
    *,
    create_db: CreateDb = create_firestore_client,
    create_submission_repository: CreateSubmissionRepository = (
        FirestoreSubmissionRepository
    ),
    create_result_repository: CreateResultRepository = FirestoreResultRepository,
    train_model: TrainModel = run_gridworld_training,
    upload_model: UploadModel = upload_model_to_gcs,
    publish_event: PublishEvent = publish_training_event,
) -> None:
    """Execute the trainer job for a single submission."""
    submission_id = config.submission_id
    db = create_db(config.db_id)
    submission_repository = create_submission_repository(db)
    result_repository = create_result_repository(db)
    transitions = TrainerResultTransitions(
        config=config,
        submission_id=submission_id,
        result_repository=result_repository,
        publish_event=publish_event,
    )

    log_trainer_event("trainer_job_started", submission_id=submission_id)

    submission = submission_repository.fetch(submission_id)
    if submission is None:
        transitions.write(
            status=ResultStatus.FAILED,
            progress=failed_progress("Submission not found"),
            error="Submission not found",
        )
        log_trainer_event(
            "submission_not_found",
            submission_id=submission_id,
        )
        return

    total_steps = 0

    try:
        inputs = parse_training_submission(submission)
        total_steps = inputs.training.timesteps

        transitions.write(
            status=ResultStatus.STARTING,
            progress=starting_progress(total_steps),
        )

        transitions.write(
            status=ResultStatus.RUNNING,
            progress=running_progress(total_steps),
        )

        execution = execute_training_run(
            inputs=inputs,
            model_bucket=config.model_bucket,
            submission_id=submission_id,
            train_model=train_model,
            upload_model=upload_model,
        )

        transitions.write(
            status=ResultStatus.COMPLETED,
            progress=completed_progress(total_steps),
            summary=execution.summary,
            artifacts=execution.artifacts,
        )

        log_trainer_event(
            "trainer_job_completed",
            submission_id=submission_id,
            total_steps=total_steps,
        )

    except Exception:
        error_message = traceback.format_exc()
        transitions.write(
            status=ResultStatus.FAILED,
            progress=failed_progress("Training failed", total_steps=total_steps),
            error=error_message,
        )
        log_trainer_event(
            "trainer_job_failed",
            submission_id=submission_id,
            total_steps=total_steps,
            error=error_message,
        )
        raise
