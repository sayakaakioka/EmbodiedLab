"""Application service for cancelling active Cloud Run training executions."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Protocol

from google.api_core.exceptions import RetryError

from embodiedlab.result_models import (
    Progress,
    ResultStatus,
    cancelled_progress,
)
from embodiedlab.schemas import DispatchState
from server.services.cancellation_tokens import verify_cancel_token
from server.services.jobs import CancellationRequestRejectedError

if TYPE_CHECKING:
    from embodiedlab.repositories import (
        ApiResultStore,
        SubmissionControlStore,
    )
    from server.config import ServerConfig

ACTIVE_RESULT_STATUSES = {
    ResultStatus.QUEUED,
    ResultStatus.STARTING,
    ResultStatus.RUNNING,
    ResultStatus.CANCELLING,
}
CANCELLATION_WAIT_SECONDS = 60
CANCELLATION_LEASE = timedelta(minutes=2)
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
    submission_repository: SubmissionControlStore,
    result_repository: ApiResultStore,
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
    if (
        status is ResultStatus.QUEUED
        and control.dispatch_state is DispatchState.PENDING
    ):
        return _cancel_pending_submission(
            config=config,
            submission_id=submission_id,
            submission_repository=submission_repository,
            publish_event=publish_event,
            result=result,
        )
    if status not in ACTIVE_RESULT_STATUSES or not control.execution_name:
        raise CancellationNotAllowedError

    return _cancel_dispatched_submission(
        config=config,
        submission_id=submission_id,
        execution_name=control.execution_name,
        submission_repository=submission_repository,
        result_repository=result_repository,
        request_cancellation=request_cancellation,
        publish_event=publish_event,
    )


def _cancel_dispatched_submission(  # noqa: PLR0911, PLR0913
    *,
    config: ServerConfig,
    submission_id: str,
    execution_name: str,
    submission_repository: SubmissionControlStore,
    result_repository: ApiResultStore,
    request_cancellation: CancellationRequester,
    publish_event: ResultEventPublisher,
) -> CancellationResult:
    """Cancel an exact execution through a durable, reclaimable intent."""
    claimed_at = datetime.now(UTC)
    lease_token = submission_repository.claim_cancellation(
        submission_id,
        claimed_at=claimed_at,
        stale_before=claimed_at - CANCELLATION_LEASE,
    )
    if lease_token is None:
        return _current_cancellation_result(result_repository, submission_id)

    try:
        operation = request_cancellation(config, execution_name)
    except CancellationRequestRejectedError as exc:
        submission_repository.release_cancellation(
            submission_id,
            lease_token,
            "Cloud Run rejected the cancellation request",
        )
        raise CancellationRequestError from exc
    except Exception:
        LOGGER.exception(
            "Cancellation request outcome is ambiguous for submission %s",
            submission_id,
        )
        return _record_pending_cancellation(
            config=config,
            submission_id=submission_id,
            result_repository=result_repository,
            publish_event=publish_event,
        )

    if not submission_repository.mark_cancellation_requested(
        submission_id,
        lease_token,
    ):
        return _current_cancellation_result(result_repository, submission_id)

    try:
        operation.result(timeout=CANCELLATION_WAIT_SECONDS)
    except RetryError:
        return _record_pending_cancellation(
            config=config,
            submission_id=submission_id,
            result_repository=result_repository,
            publish_event=publish_event,
        )
    except Exception:
        LOGGER.exception(
            "Cancellation operation outcome is ambiguous for submission %s",
            submission_id,
        )
        return _record_pending_cancellation(
            config=config,
            submission_id=submission_id,
            result_repository=result_repository,
            publish_event=publish_event,
        )

    pending = _record_pending_cancellation(
        config=config,
        submission_id=submission_id,
        result_repository=result_repository,
        publish_event=publish_event,
    )
    if not pending.pending:
        return pending
    transitioned = _transition_preserving_progress_and_publish(
        config=config,
        submission_id=submission_id,
        result_repository=result_repository,
        publish_event=publish_event,
        expected_statuses={ResultStatus.CANCELLING},
        status=ResultStatus.CANCELLED,
        message="Training cancelled",
    )
    if transitioned is not None:
        return CancellationResult(result=transitioned, pending=False)
    return _current_cancellation_result(result_repository, submission_id)


def _record_pending_cancellation(
    *,
    config: ServerConfig,
    submission_id: str,
    result_repository: ApiResultStore,
    publish_event: ResultEventPublisher,
) -> CancellationResult:
    latest = _fetch_required_result(result_repository, submission_id)
    status = _parse_status(latest.get("status"))
    if status is ResultStatus.CANCELLING:
        return CancellationResult(result=latest, pending=True)
    if status not in ACTIVE_RESULT_STATUSES:
        return CancellationResult(result=latest, pending=False)
    transitioned = _transition_preserving_progress_and_publish(
        config=config,
        submission_id=submission_id,
        result_repository=result_repository,
        publish_event=publish_event,
        expected_statuses=ACTIVE_RESULT_STATUSES - {ResultStatus.CANCELLING},
        status=ResultStatus.CANCELLING,
        message="Cancelling training",
    )
    if transitioned is not None:
        return CancellationResult(result=transitioned, pending=True)
    return _current_cancellation_result(result_repository, submission_id)


def _cancel_pending_submission(
    *,
    config: ServerConfig,
    submission_id: str,
    submission_repository: SubmissionControlStore,
    publish_event: ResultEventPublisher,
    result: dict[str, Any],
) -> CancellationResult:
    progress = _parse_progress(result)
    cancelled = cancelled_progress(
        current_step=progress.current_step,
        total_steps=progress.total_steps,
    )
    transitioned = submission_repository.cancel_pending_dispatch(
        submission_id,
        progress=cancelled,
    )
    if transitioned is None:
        raise CancellationNotAllowedError
    _publish_transition(
        config=config,
        submission_id=submission_id,
        publish_event=publish_event,
        status=ResultStatus.CANCELLED,
        progress=cancelled,
    )
    return CancellationResult(result=transitioned, pending=False)


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
    result_repository: ApiResultStore,
    submission_id: str,
) -> dict[str, Any]:
    result = result_repository.fetch(submission_id)
    if result is None:
        raise CancellationResultNotFoundError
    return result


def _transition_preserving_progress_and_publish(  # noqa: PLR0913
    *,
    config: ServerConfig,
    submission_id: str,
    result_repository: ApiResultStore,
    publish_event: ResultEventPublisher,
    expected_statuses: set[ResultStatus],
    status: ResultStatus,
    message: str,
) -> dict[str, Any] | None:
    result = result_repository.transition_status_preserving_progress(
        submission_id,
        expected_statuses=expected_statuses,
        status=status,
        message=message,
    )
    if result is None:
        return None
    progress = _parse_progress(result)
    _publish_transition(
        config=config,
        submission_id=submission_id,
        publish_event=publish_event,
        status=status,
        progress=progress,
    )
    return result


def _publish_transition(
    *,
    config: ServerConfig,
    submission_id: str,
    publish_event: ResultEventPublisher,
    status: ResultStatus,
    progress: Progress,
) -> None:
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


def _current_cancellation_result(
    result_repository: ApiResultStore,
    submission_id: str,
) -> CancellationResult:
    result = _fetch_required_result(result_repository, submission_id)
    return CancellationResult(
        result=result,
        pending=_parse_status(result.get("status")) in ACTIVE_RESULT_STATUSES,
    )
