"""Trainer orchestration for Firestore updates, model training, and event publishing."""

from __future__ import annotations

import traceback
from collections.abc import Callable
from typing import Any, Protocol

from embodiedlab.repositories import ResultStore
from embodiedlab.result_models import (
    ResultStatus,
    build_result_bundle,
    completed_progress,
    failed_progress,
    running_progress,
    starting_progress,
)
from embodiedlab.training.runner import run_continuous_navigation_training
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


class TrainerSubmissionRepository(Protocol):
    """Submission reads plus exact Cloud Run execution recovery."""

    def fetch(self, submission_id: str) -> dict[str, Any] | None:
        """Return the submitted scenario."""

    def mark_dispatched(self, submission_id: str, execution_name: str) -> bool:
        """Recover an unresolved exact execution name."""


CreateSubmissionRepository = Callable[[Any], TrainerSubmissionRepository]
CreateResultRepository = Callable[[Any], ResultStore]
PublishEvent = Callable[..., None]
TRAINER_ACTIVE_STATUSES = {
    ResultStatus.QUEUED,
    ResultStatus.STARTING,
    ResultStatus.RUNNING,
    ResultStatus.CANCELLING,
}


def run_training_job(  # noqa: PLR0913
    config: TrainerConfig,
    *,
    create_db: CreateDb = create_firestore_client,
    create_submission_repository: CreateSubmissionRepository = (
        FirestoreSubmissionRepository
    ),
    create_result_repository: CreateResultRepository = FirestoreResultRepository,
    train_model: TrainModel = run_continuous_navigation_training,
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

    if config.execution_name is not None and not submission_repository.mark_dispatched(
        submission_id,
        config.execution_name,
    ):
        log_trainer_event(
            "trainer_dispatch_already_closed",
            submission_id=submission_id,
            execution_name=config.execution_name,
        )
        return

    submission = submission_repository.fetch(submission_id)
    if submission is None:
        transitions.write(
            expected_statuses={ResultStatus.QUEUED},
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
    inputs = None

    try:
        inputs = parse_training_submission(submission)
        total_steps = inputs.training.timesteps
        log_trainer_event(
            "training_inputs_prepared",
            submission_id=submission_id,
            total_steps=total_steps,
            n_envs=inputs.training.n_envs,
            env_kind="single" if inputs.training.n_envs == 1 else "subproc_vec",
            cpu_count=inputs.training.cpu_count,
            torch_num_threads=inputs.training.torch_num_threads,
            n_steps=inputs.training.n_steps,
            batch_size=inputs.training.batch_size,
            max_steps=inputs.training.max_steps,
        )

        if (
            not transitions.write(
                expected_statuses={ResultStatus.QUEUED},
                status=ResultStatus.STARTING,
                progress=starting_progress(total_steps),
            )
            and transitions.current_status() is not ResultStatus.CANCELLING
        ):
            return

        if (
            not transitions.write(
                expected_statuses={ResultStatus.STARTING},
                status=ResultStatus.RUNNING,
                progress=running_progress(total_steps),
            )
            and transitions.current_status() is not ResultStatus.CANCELLING
        ):
            return

        def report_training_progress(current_step: int, total_steps: int) -> None:
            log_trainer_event(
                "training_progress",
                submission_id=submission_id,
                current_step=current_step,
                total_steps=total_steps,
            )
            transitions.write(
                expected_statuses={ResultStatus.RUNNING},
                status=ResultStatus.RUNNING,
                progress=running_progress(
                    total_steps,
                    current_step=current_step,
                ),
            )

        def report_training_diagnostic(
            event: str,
            fields: dict[str, Any],
        ) -> None:
            log_trainer_event(
                event,
                submission_id=submission_id,
                **fields,
            )

        execution = execute_training_run(
            inputs=inputs,
            model_bucket=config.model_bucket,
            submission_id=submission_id,
            train_model=train_model,
            upload_model=upload_model,
            progress_callback=report_training_progress,
            diagnostic_callback=report_training_diagnostic,
        )

        completed = transitions.write(
            expected_statuses={
                ResultStatus.STARTING,
                ResultStatus.RUNNING,
                ResultStatus.CANCELLING,
            },
            status=ResultStatus.COMPLETED,
            progress=completed_progress(total_steps),
            summary=execution.summary,
            result_bundle=execution.result_bundle,
        )

        if not completed:
            log_trainer_event(
                "trainer_result_already_closed",
                submission_id=submission_id,
            )
            return

        log_trainer_event(
            "trainer_job_completed",
            submission_id=submission_id,
            total_steps=total_steps,
        )

    except Exception:
        error_message = traceback.format_exc()
        failed_bundle = None
        if inputs is not None:
            failed_bundle = build_result_bundle(
                scenario=inputs.scenario,
                job_id=submission_id,
                status=ResultStatus.FAILED,
                error=error_message,
            )
        transitions.write(
            expected_statuses=TRAINER_ACTIVE_STATUSES,
            status=ResultStatus.FAILED,
            progress=failed_progress("Training failed", total_steps=total_steps),
            error=error_message,
            result_bundle=failed_bundle,
        )
        log_trainer_event(
            "trainer_job_failed",
            submission_id=submission_id,
            total_steps=total_steps,
            error=error_message,
        )
        raise
