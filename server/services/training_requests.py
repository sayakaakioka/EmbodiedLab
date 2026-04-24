"""Application service for starting training from the API layer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from embodiedlab.result_models import failed_progress
from server.config import ServerConfig

if TYPE_CHECKING:
    from embodiedlab.repositories import (
        ResultFailureWriter,
        ResultQueueWriter,
        SubmissionExistenceChecker,
    )

TriggerTrainingJob = Callable[[ServerConfig, str], None]


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
    submission_repository: SubmissionExistenceChecker,
    result_repository: ResultQueueWriter | ResultFailureWriter,
    config: ServerConfig,
    submission_id: str,
    trigger_job: TriggerTrainingJob,
) -> None:
    """Queue result tracking and trigger the configured trainer job."""
    if not submission_repository.exists(submission_id):
        raise SubmissionNotFoundError(submission_id)

    result_repository.create_queued(submission_id)

    try:
        trigger_job(config, submission_id)
    except Exception as exc:
        error_message = TrainingStartError.DEFAULT_MESSAGE
        result_repository.mark_failed(
            submission_id,
            failed_progress(error_message),
            error_message,
        )
        raise TrainingStartError from exc
