"""Cloud Run execution failure reconciliation for result documents."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from google.api_core.client_options import ClientOptions
from google.cloud import run_v2
from google.cloud.run_v2.types import Condition

from embodiedlab.result_models import ResultStatus, failed_progress

if TYPE_CHECKING:
    from embodiedlab.repositories import ResultFailureWriter, ResultReader
    from server.config import ServerConfig

ACTIVE_RESULT_STATUSES = {
    ResultStatus.QUEUED,
    ResultStatus.STARTING,
    ResultStatus.RUNNING,
}
SUBMISSION_ID_ENV = "SUBMISSION_ID"
MAX_EXECUTIONS_TO_SCAN = 50
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExecutionFailure:
    """Failure details discovered from a completed Cloud Run execution."""

    execution_name: str
    message: str


ExecutionFailureFinder = Callable[["ServerConfig", str], ExecutionFailure | None]


def find_failed_execution_for_submission(
    config: ServerConfig,
    submission_id: str,
) -> ExecutionFailure | None:
    """Return a failed Cloud Run execution matching the submission ID."""
    executions_client = run_v2.ExecutionsClient(
        client_options=ClientOptions(
            api_endpoint=f"{config.region}-run.googleapis.com",
        ),
    )
    request = run_v2.ListExecutionsRequest(
        parent=config.job_path,
        page_size=MAX_EXECUTIONS_TO_SCAN,
    )

    scanned_count = 0
    for execution in executions_client.list_executions(request=request):
        if scanned_count >= MAX_EXECUTIONS_TO_SCAN:
            break
        scanned_count += 1

        if not _execution_matches_submission(execution, submission_id):
            continue

        return _failure_from_execution(execution)

    return None


def reconcile_result_with_execution_failure(
    *,
    config: ServerConfig,
    submission_id: str,
    result_repository: ResultReader | ResultFailureWriter,
    result: dict[str, Any],
    find_failed_execution: ExecutionFailureFinder,
) -> dict[str, Any]:
    """Mark active result documents failed when their Cloud Run execution failed."""
    status = _parse_result_status(result.get("status"))
    if status not in ACTIVE_RESULT_STATUSES:
        return result

    try:
        failure = find_failed_execution(config, submission_id)
    except Exception:
        LOGGER.exception(
            "Failed to reconcile Cloud Run execution for submission %s",
            submission_id,
        )
        return result

    if failure is None:
        return result

    total_steps = _result_total_steps(result)
    message = f"Cloud Run execution {failure.execution_name} failed: {failure.message}"
    result_repository.mark_failed(
        submission_id,
        failed_progress(message, total_steps=total_steps),
        message,
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


def _result_total_steps(result: dict[str, Any]) -> int:
    progress = result.get("progress")
    if not isinstance(progress, dict):
        return 0

    value = progress.get("total_steps", 0)
    return value if isinstance(value, int) and value >= 0 else 0


def _execution_matches_submission(
    execution: run_v2.Execution,
    submission_id: str,
) -> bool:
    for container in execution.template.containers:
        for env in container.env:
            if env.name == SUBMISSION_ID_ENV and env.value == submission_id:
                return True
    return False


def _failure_from_execution(execution: run_v2.Execution) -> ExecutionFailure | None:
    if execution.failed_count <= 0 and execution.cancelled_count <= 0:
        return None

    failure_condition = _failed_condition(execution)
    execution_name = execution.name.rsplit("/", maxsplit=1)[-1]
    if failure_condition is None:
        return ExecutionFailure(
            execution_name=execution_name,
            message="Cloud Run execution failed",
        )

    return ExecutionFailure(
        execution_name=execution_name,
        message=failure_condition.message or "Cloud Run execution failed",
    )


def _failed_condition(execution: run_v2.Execution) -> Condition | None:
    for condition in execution.conditions:
        if condition.state == Condition.State.CONDITION_FAILED:
            return condition
    return None
