"""Helpers for persisting and publishing trainer result state transitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from embodiedlab.result_models import ResultStatus

if TYPE_CHECKING:
    from embodiedlab.repositories import ResultStore
    from embodiedlab.result_models import Progress, ResultBundle
    from trainer.config import TrainerConfig
    from trainer.job import PublishEvent


@dataclass(frozen=True)
class TrainerResultTransitions:
    """Coordinate Firestore result updates with matching Pub/Sub events."""

    config: TrainerConfig
    submission_id: str
    result_repository: ResultStore
    publish_event: PublishEvent

    def write(  # noqa: PLR0913
        self,
        *,
        expected_statuses: set[ResultStatus],
        status: ResultStatus,
        progress: Progress,
        summary: dict[str, Any] | None = None,
        error: str | None = None,
        result_bundle: dict[str, Any] | ResultBundle | None = None,
    ) -> bool:
        """Persist and publish a transition only from an expected state."""
        result = self.result_repository.transition_if_status(
            self.submission_id,
            expected_statuses=expected_statuses,
            status=status,
            progress=progress,
            summary=summary,
            error=error,
            result_bundle=result_bundle,
        )
        if result is None:
            return False
        self.publish_event(
            config=self.config,
            submission_id=self.submission_id,
            status=status,
            progress=progress,
            summary=summary,
            error=error,
            result_bundle=result_bundle,
        )
        return True

    def current_status(self) -> ResultStatus | None:
        """Return the canonical current result status."""
        result = self.result_repository.fetch(self.submission_id)
        if result is None:
            return None
        try:
            return ResultStatus(result.get("status"))
        except (TypeError, ValueError):
            return None
