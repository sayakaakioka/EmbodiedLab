"""Helpers for persisting and publishing trainer result state transitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from embodiedlab.repositories import ResultUpdateWriter
    from embodiedlab.result_models import Progress, ResultStatus
    from trainer.config import TrainerConfig
    from trainer.job import PublishEvent


@dataclass(frozen=True)
class TrainerResultTransitions:
    """Coordinate Firestore result updates with matching Pub/Sub events."""

    config: TrainerConfig
    submission_id: str
    result_repository: ResultUpdateWriter
    publish_event: PublishEvent

    def write(
        self,
        *,
        status: ResultStatus,
        progress: Progress,
        summary: dict[str, Any] | None = None,
        error: str | None = None,
        artifacts: dict[str, Any] | None = None,
    ) -> None:
        """Persist a result transition and publish the corresponding event."""
        self.result_repository.write_update(
            self.submission_id,
            status=status,
            progress=progress,
            summary=summary,
            error=error,
            artifacts=artifacts,
        )
        self.publish_event(
            config=self.config,
            submission_id=self.submission_id,
            status=status,
            progress=progress,
            summary=summary,
            error=error,
            artifacts=artifacts,
        )
