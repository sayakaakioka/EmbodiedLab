"""Application service for starting training from the API layer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from embodiedlab.result_models import failed_progress
from server.config import ServerConfig

if TYPE_CHECKING:
    from embodiedlab.repositories import (
        ResultQueueWriter,
        ResultUpdateWriter,
        SubmissionExecutionWriter,
        SubmissionExistenceChecker,
    )

TriggerTrainingJob = Callable[[ServerConfig, str], str]


class SubmissionNotFoundError(Exception):
    """Raised when an API request references a missing submission."""


@dataclass(frozen=True)
class TrainingStartError(Exception):
    """Raised when the API cannot start the background training job."""

    DEFAULT_MESSAGE = "Failed to start trainer job"

    message: str

    def __init__(self, message: str = DEFAULT_MESSAGE) -> None:
        """Initialize the error with the shared user-facing message."""
        super().__init__(message)
        object.__setattr__(self, "message", message)


def start_training_for_submission(
    *,
    submission_repository: SubmissionExistenceChecker | SubmissionExecutionWriter,
    result_repository: ResultQueueWriter | ResultUpdateWriter,
    config: ServerConfig,
    submission_id: str,
    trigger_job: TriggerTrainingJob,
) -> str:
    """Queue result tracking and trigger the configured trainer job."""
    if not submission_repository.exists(submission_id):
        raise SubmissionNotFoundError(submission_id)

    result_repository.create_queued(submission_id)

    try:
        execution_name = trigger_job(config, submission_id)
        submission_repository.set_execution_name(submission_id, execution_name)
    except Exception as exc:
        error_message = TrainingStartError.DEFAULT_MESSAGE
        progress = failed_progress(error_message)
        result_repository.write_update(
            submission_id,
            status=progress.phase,
            progress=progress,
            error=error_message,
        )
        raise TrainingStartError from exc

    return execution_name
