"""Pub/Sub publisher that emits ordered training status events."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from embodiedlab.result_events import publish_result_event

if TYPE_CHECKING:
    from embodiedlab.result_models import Progress, ResultBundle, ResultStatus
    from trainer.config import TrainerConfig


def publish_training_event(  # noqa: PLR0913
    *,
    config: TrainerConfig,
    submission_id: str,
    status: ResultStatus,
    progress: Progress,
    summary: dict[str, Any] | None = None,
    error: str | None = None,
    artifacts: dict[str, Any] | None = None,
    result_bundle: dict[str, Any] | ResultBundle | None = None,
) -> None:
    """Publish a training status event to the configured Pub/Sub topic."""
    publish_result_event(
        project_id=config.project_id,
        pubsub_topic=config.pubsub_topic,
        submission_id=submission_id,
        status=status,
        progress=progress,
        summary=summary,
        error=error,
        artifacts=artifacts,
        result_bundle=result_bundle,
    )
