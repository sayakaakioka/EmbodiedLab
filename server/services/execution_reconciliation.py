"""Exact Cloud Run execution reconciliation for active result documents."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from google.api_core.client_options import ClientOptions
from google.cloud import run_v2
from google.cloud.run_v2.types import Condition

from embodiedlab.result_models import (
    Progress,
    ResultStatus,
)

if TYPE_CHECKING:
    from embodiedlab.repositories import (
        ApiResultStore,
        SubmissionControlReader,
    )
    from server.config import ServerConfig
    from server.services.cancellations import ResultEventPublisher

ACTIVE_RESULT_STATUSES = {
    ResultStatus.QUEUED,
    ResultStatus.STARTING,
    ResultStatus.RUNNING,
    ResultStatus.CANCELLING,
}
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExecutionOutcome:
    """Terminal outcome discovered from an exact Cloud Run Execution."""

    status: ResultStatus
    message: str


ExecutionOutcomeReader = Callable[["ServerConfig", str], ExecutionOutcome | None]
CreateExecutionsClient = Callable[..., run_v2.ExecutionsClient]


def read_execution_outcome(
    config: ServerConfig,
    execution_name: str,
    *,
    create_executions_client: CreateExecutionsClient = run_v2.ExecutionsClient,
) -> ExecutionOutcome | None:
    """Return the terminal outcome of one exact Cloud Run Execution."""
    executions_client = create_executions_client(
        client_options=ClientOptions(
            api_endpoint=f"{config.region}-run.googleapis.com",
        ),
    )
    execution = executions_client.get_execution(
        request=run_v2.GetExecutionRequest(name=execution_name),
    )
    if execution.cancelled_count > 0:
        return ExecutionOutcome(
            status=ResultStatus.CANCELLED,
            message="Cloud Run execution cancelled",
        )
    if execution.failed_count > 0:
        failure_condition = _failed_condition(execution)
        return ExecutionOutcome(
            status=ResultStatus.FAILED,
            message=(
                failure_condition.message
                if failure_condition is not None and failure_condition.message
                else "Cloud Run execution failed"
            ),
        )
    if execution.completion_time:
        return ExecutionOutcome(
            status=ResultStatus.FAILED,
            message="Cloud Run execution completed without a terminal result",
        )
    return None


def reconcile_result_with_execution(  # noqa: PLR0913
    *,
    config: ServerConfig,
    submission_id: str,
    submission_repository: SubmissionControlReader,
    result_repository: ApiResultStore,
    result: dict[str, Any],
    read_outcome: ExecutionOutcomeReader,
    publish_event: ResultEventPublisher,
) -> dict[str, Any]:
    """Apply a failed or cancelled execution outcome to an active result."""
    status = _parse_result_status(result.get("status"))
    if status not in ACTIVE_RESULT_STATUSES:
        return result

    control = submission_repository.fetch_control(submission_id)
    if control is None or not control.execution_name:
        return result

    try:
        outcome = read_outcome(config, control.execution_name)
    except Exception:
        LOGGER.exception(
            "Failed to reconcile Cloud Run execution for submission %s",
            submission_id,
        )
        return result
    if outcome is None:
        return result

    outcome_status = ResultStatus(outcome.status)
    if outcome_status is ResultStatus.CANCELLED:
        message = "Training cancelled"
        error = None
    else:
        execution_id = control.execution_name.rsplit("/", maxsplit=1)[-1]
        message = f"Cloud Run execution {execution_id} failed: {outcome.message}"
        error = message

    transitioned = result_repository.transition_status_preserving_progress(
        submission_id,
        expected_statuses=ACTIVE_RESULT_STATUSES,
        status=outcome_status,
        message=message,
        error=error,
    )
    if transitioned is None:
        return result_repository.fetch(submission_id) or result
    progress = Progress.model_validate(transitioned.get("progress"))
    _publish_reconciled_transition(
        config=config,
        submission_id=submission_id,
        status=outcome_status,
        progress=progress,
        error=error,
        publish_event=publish_event,
    )
    return transitioned


def _parse_result_status(value: object) -> ResultStatus | None:
    if isinstance(value, ResultStatus):
        return value
    if isinstance(value, str):
        try:
            return ResultStatus(value)
        except ValueError:
            return None
    return None


def _failed_condition(execution: run_v2.Execution) -> Condition | None:
    for condition in execution.conditions:
        if condition.state == Condition.State.CONDITION_FAILED:
            return condition
    return None


def _publish_reconciled_transition(  # noqa: PLR0913
    *,
    config: ServerConfig,
    submission_id: str,
    status: ResultStatus,
    progress: Progress,
    error: str | None,
    publish_event: ResultEventPublisher,
) -> None:
    try:
        publish_event(
            project_id=config.project_id,
            pubsub_topic=config.pubsub_topic,
            submission_id=submission_id,
            status=status,
            progress=progress,
            error=error,
        )
    except Exception:
        LOGGER.exception(
            "Failed to publish reconciled result submission_id=%s status=%s",
            submission_id,
            status,
        )
