"""Pub/Sub publisher that emits ordered training status events."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from google.cloud import pubsub_v1

from embodiedlab.result_models import (
    Progress,
    ResultStatus,
    build_result_message,
)

if TYPE_CHECKING:
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
) -> None:
    """Publish a training status event to the configured Pub/Sub topic."""
    message = build_result_message(
        submission_id=submission_id,
        status=status,
        progress=progress,
        summary=summary,
        error=error,
        artifacts=artifacts,
    )

    publisher = pubsub_v1.PublisherClient(
        publisher_options=pubsub_v1.types.PublisherOptions(
            enable_message_ordering=True,
        ),
    )
    topic_path = publisher.topic_path(config.project_id, config.pubsub_topic)
    future = publisher.publish(
        topic_path,
        data=json.dumps(message).encode("utf-8"),
        ordering_key=submission_id,
    )
    future.result()
