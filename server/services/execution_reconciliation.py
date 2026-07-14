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
    cancelled_progress,
    failed_progress,
)

if TYPE_CHECKING:
    from embodiedlab.repositories import (
        ResultReader,
        ResultUpdateWriter,
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
    if execution.failed_count <= 0:
        return None

    failure_condition = _failed_condition(execution)
    return ExecutionOutcome(
        status=ResultStatus.FAILED,
        message=(
            failure_condition.message
            if failure_condition is not None and failure_condition.message
            else "Cloud Run execution failed"
        ),
    )


def reconcile_result_with_execution(  # noqa: PLR0913
    *,
    config: ServerConfig,
    submission_id: str,
    submission_repository: SubmissionControlReader,
    result_repository: ResultReader | ResultUpdateWriter,
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
    current_step, total_steps = _result_steps(result)
    if outcome_status is ResultStatus.CANCELLED:
        progress = cancelled_progress(current_step, total_steps)
        error = None
    else:
        execution_id = control.execution_name.rsplit("/", maxsplit=1)[-1]
        message = f"Cloud Run execution {execution_id} failed: {outcome.message}"
        progress = failed_progress(message, total_steps=total_steps)
        error = message

    result_repository.write_update(
        submission_id,
        status=outcome_status,
        progress=progress,
        error=error,
    )
    _publish_reconciled_transition(
        config=config,
        submission_id=submission_id,
        status=outcome_status,
        progress=progress,
        error=error,
        publish_event=publish_event,
    )
    refreshed = result_repository.fetch(submission_id)
    return refreshed if refreshed is not None else result


def _parse_result_status(value: object) -> ResultStatus | None:
    if isinstance(value, ResultStatus):
        return value
    if isinstance(value, str):
        try:
            return ResultStatus(value)
        except ValueError:
            return None
    return None


def _result_steps(result: dict[str, Any]) -> tuple[int, int]:
    progress = result.get("progress")
    if not isinstance(progress, dict):
        return 0, 0

    current_step = progress.get("current_step", 0)
    total_steps = progress.get("total_steps", 0)
    if not isinstance(current_step, int) or current_step < 0:
        current_step = 0
    if not isinstance(total_steps, int) or total_steps < 0:
        total_steps = 0
    return current_step, total_steps


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
