"""Server-owned workflow for accepting and dispatching training submissions."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from embodiedlab.random_start import build_random_start_area
from embodiedlab.result_models import failed_progress
from embodiedlab.schemas import DispatchState
from embodiedlab.training.training_converter import convert_scenario_to_spec
from server.config import ServerConfig
from server.services.jobs import TrainingDispatchRejectedError

if TYPE_CHECKING:
    from embodiedlab.repositories import SubmissionWorkflowRepository
    from embodiedlab.schemas import ScenarioBundle
    from server.services.cancellations import ResultEventPublisher

TrainingJobRunner = Callable[[ServerConfig, str], str]
DISPATCH_FAILURE_MESSAGE = "Failed to start trainer job"
DISPATCH_AMBIGUOUS_MESSAGE = "Training dispatch outcome could not be confirmed"
EXECUTION_METADATA_WRITE_ATTEMPTS = 3
DISPATCH_CONFIRMATION_TIMEOUT = timedelta(minutes=5)
LOGGER = logging.getLogger(__name__)


class SubmissionDispatchUnavailableError(Exception):
    """Raised when the server cannot safely claim a pending dispatch."""


@dataclass(frozen=True)
class SubmissionWorkflow:
    """Accept, dispatch, and reconcile one server-owned submission lifecycle."""

    submission_repository: SubmissionWorkflowRepository
    config: ServerConfig
    run_job: TrainingJobRunner
    publish_event: ResultEventPublisher

    def submit(
        self,
        scenario: ScenarioBundle,
        *,
        cancel_token_hash: str,
        idempotency_key: str,
    ) -> str:
        """Atomically accept a submission and claim its only dispatch attempt."""
        if scenario.training.randomize_start:
            build_random_start_area(convert_scenario_to_spec(scenario))
        submission_id = self.submission_repository.accept(
            scenario,
            cancel_token_hash=cancel_token_hash,
            total_steps=scenario.training.timesteps,
            idempotency_key=idempotency_key,
        )
        self.dispatch(submission_id, total_steps=scenario.training.timesteps)
        return submission_id

    def dispatch(self, submission_id: str, *, total_steps: int) -> None:
        """Claim and dispatch a submission, recording server-owned failures."""
        try:
            claimed = self.submission_repository.claim_dispatch(submission_id)
        except Exception as exc:
            raise SubmissionDispatchUnavailableError from exc
        if not claimed:
            return

        try:
            execution_name = self.run_job(self.config, submission_id)
        except TrainingDispatchRejectedError:
            LOGGER.exception("Failed to dispatch submission %s", submission_id)
            self._record_failure(
                submission_id,
                total_steps=total_steps,
                message=DISPATCH_FAILURE_MESSAGE,
            )
        except Exception:
            LOGGER.exception(
                "Dispatch outcome is ambiguous for submission %s",
                submission_id,
            )
            self._mark_ambiguous(submission_id)
        else:
            self._persist_execution_name(submission_id, execution_name)

    def reconcile(
        self,
        submission_id: str,
        *,
        total_steps: int,
        now: datetime | None = None,
    ) -> None:
        """Resume a pending dispatch or fail a stale ambiguous outcome."""
        control = self.submission_repository.fetch_control(submission_id)
        if control is None or control.execution_name:
            return
        if control.dispatch_state is DispatchState.PENDING:
            self.dispatch(submission_id, total_steps=total_steps)
            return
        if control.dispatch_state not in {
            DispatchState.DISPATCHING,
            DispatchState.AMBIGUOUS,
        }:
            return

        observed_at = now or datetime.now(UTC)
        started_at = control.dispatch_started_at
        if (
            started_at is not None
            and observed_at - started_at < DISPATCH_CONFIRMATION_TIMEOUT
        ):
            return
        if control.dispatch_state is DispatchState.DISPATCHING:
            self._mark_ambiguous(submission_id)
            return
        self._record_failure(
            submission_id,
            total_steps=total_steps,
            message=DISPATCH_AMBIGUOUS_MESSAGE,
        )

    def _persist_execution_name(
        self,
        submission_id: str,
        execution_name: str,
    ) -> None:
        for attempt in range(1, EXECUTION_METADATA_WRITE_ATTEMPTS + 1):
            try:
                applied = self.submission_repository.mark_dispatched(
                    submission_id,
                    execution_name,
                )
            except Exception:
                LOGGER.exception(
                    "Failed to persist execution metadata for submission %s attempt %s",
                    submission_id,
                    attempt,
                )
            else:
                if not applied:
                    LOGGER.warning(
                        "Ignored late execution metadata for closed submission %s",
                        submission_id,
                    )
                return
        self._mark_ambiguous(submission_id)

    def _mark_ambiguous(self, submission_id: str) -> None:
        try:
            applied = self.submission_repository.mark_dispatch_ambiguous(
                submission_id,
                DISPATCH_AMBIGUOUS_MESSAGE,
            )
        except Exception:
            LOGGER.exception(
                "Failed to persist ambiguous dispatch for submission %s",
                submission_id,
            )
        else:
            if not applied:
                LOGGER.warning(
                    "Ignored ambiguous transition for closed submission %s",
                    submission_id,
                )

    def _record_failure(
        self,
        submission_id: str,
        *,
        total_steps: int,
        message: str,
    ) -> None:
        progress = failed_progress(message, total_steps=total_steps)
        try:
            applied = self.submission_repository.fail_dispatch_if_queued(
                submission_id,
                progress=progress,
                error=message,
            )
        except Exception:
            LOGGER.exception(
                "Failed to persist atomic dispatch failure for submission %s",
                submission_id,
            )
            return
        if not applied:
            return
        try:
            self.publish_event(
                project_id=self.config.project_id,
                pubsub_topic=self.config.pubsub_topic,
                submission_id=submission_id,
                status=progress.phase,
                progress=progress,
                error=message,
            )
        except Exception:
            LOGGER.exception(
                "Failed to publish dispatch failure for submission %s",
                submission_id,
            )
