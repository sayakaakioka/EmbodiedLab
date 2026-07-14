"""Application service for cancelling active Cloud Run training executions."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from google.api_core.exceptions import RetryError

from embodiedlab.result_models import (
    Progress,
    ResultStatus,
    cancelled_progress,
    cancelling_progress,
)
from server.services.cancellation_tokens import verify_cancel_token

if TYPE_CHECKING:
    from embodiedlab.repositories import (
        ResultReader,
        ResultUpdateWriter,
        SubmissionControlReader,
    )
    from server.config import ServerConfig

ACTIVE_RESULT_STATUSES = {
    ResultStatus.QUEUED,
    ResultStatus.STARTING,
    ResultStatus.RUNNING,
}
CANCELLATION_WAIT_SECONDS = 60
LOGGER = logging.getLogger(__name__)


class CancellationOperation(Protocol):
    """Long-running Cloud Run cancellation operation used by the service."""

    def result(self, *, timeout: float) -> object:
        """Wait for cancellation completion."""


CancellationRequester = Callable[["ServerConfig", str], CancellationOperation]
ResultEventPublisher = Callable[..., None]


class InvalidCancelTokenError(Exception):
    """Raised when the caller does not present the submission capability."""


class CancellationResultNotFoundError(Exception):
    """Raised when no job result exists for the requested submission."""


class CancellationNotAllowedError(Exception):
    """Raised when a terminal or unstarted job cannot be cancelled."""


class CancellationRequestError(Exception):
    """Raised when Cloud Run rejects a cancellation request."""


@dataclass(frozen=True)
class CancellationResult:
    """Latest result payload and whether Cloud Run is still cancelling it."""

    result: dict[str, Any]
    pending: bool


def cancel_training_for_submission(  # noqa: PLR0913
    *,
    config: ServerConfig,
    submission_id: str,
    cancel_token: str,
    submission_repository: SubmissionControlReader,
    result_repository: ResultReader | ResultUpdateWriter,
    request_cancellation: CancellationRequester,
    publish_event: ResultEventPublisher,
) -> CancellationResult:
    """Cancel an active submission after validating its capability token."""
    control = submission_repository.fetch_control(submission_id)
    if control is None or not verify_cancel_token(
        cancel_token,
        control.cancel_token_hash,
    ):
        raise InvalidCancelTokenError

    result = result_repository.fetch(submission_id)
    if result is None:
        raise CancellationResultNotFoundError

    status = _parse_status(result.get("status"))
    if status is ResultStatus.CANCELLED:
        return CancellationResult(result=result, pending=False)
    if status is ResultStatus.CANCELLING:
        return CancellationResult(result=result, pending=True)
    if status not in ACTIVE_RESULT_STATUSES or not control.execution_name:
        raise CancellationNotAllowedError

    progress = _parse_progress(result)
    try:
        operation = request_cancellation(config, control.execution_name)
    except Exception as exc:
        raise CancellationRequestError from exc

    cancelling = cancelling_progress(
        current_step=progress.current_step,
        total_steps=progress.total_steps,
    )
    _write_and_publish(
        config=config,
        submission_id=submission_id,
        result_repository=result_repository,
        publish_event=publish_event,
        status=ResultStatus.CANCELLING,
        progress=cancelling,
    )

    try:
        operation.result(timeout=CANCELLATION_WAIT_SECONDS)
    except RetryError:
        return CancellationResult(
            result=_fetch_required_result(result_repository, submission_id),
            pending=True,
        )
    except Exception as exc:
        _write_and_publish(
            config=config,
            submission_id=submission_id,
            result_repository=result_repository,
            publish_event=publish_event,
            status=status,
            progress=progress,
        )
        raise CancellationRequestError from exc

    cancelled = cancelled_progress(
        current_step=progress.current_step,
        total_steps=progress.total_steps,
    )
    _write_and_publish(
        config=config,
        submission_id=submission_id,
        result_repository=result_repository,
        publish_event=publish_event,
        status=ResultStatus.CANCELLED,
        progress=cancelled,
    )
    return CancellationResult(
        result=_fetch_required_result(result_repository, submission_id),
        pending=False,
    )


def _parse_status(value: object) -> ResultStatus | None:
    if isinstance(value, ResultStatus):
        return value
    if isinstance(value, str):
        try:
            return ResultStatus(value)
        except ValueError:
            return None
    return None


def _parse_progress(result: dict[str, Any]) -> Progress:
    progress = result.get("progress")
    if isinstance(progress, dict):
        return Progress.model_validate(progress)
    return Progress(
        phase=ResultStatus.QUEUED,
        current_step=0,
        total_steps=0,
        message="Queued",
    )


def _fetch_required_result(
    result_repository: ResultReader,
    submission_id: str,
) -> dict[str, Any]:
    result = result_repository.fetch(submission_id)
    if result is None:
        raise CancellationResultNotFoundError
    return result


def _write_and_publish(  # noqa: PLR0913
    *,
    config: ServerConfig,
    submission_id: str,
    result_repository: ResultUpdateWriter,
    publish_event: ResultEventPublisher,
    status: ResultStatus,
    progress: Progress,
) -> None:
    result_repository.write_update(
        submission_id,
        status=status,
        progress=progress,
    )
    try:
        publish_event(
            project_id=config.project_id,
            pubsub_topic=config.pubsub_topic,
            submission_id=submission_id,
            status=status,
            progress=progress,
        )
    except Exception:
        LOGGER.exception(
            "Failed to publish result transition submission_id=%s status=%s",
            submission_id,
            status,
        )
