"""Pub/Sub publisher shared by result-producing services."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from google.cloud import pubsub_v1

from embodiedlab.result_models import build_result_message

if TYPE_CHECKING:
    from embodiedlab.result_models import Progress, ResultBundle, ResultStatus


def publish_result_event(  # noqa: PLR0913
    *,
    project_id: str,
    pubsub_topic: str,
    submission_id: str,
    status: ResultStatus,
    progress: Progress,
    summary: dict[str, Any] | None = None,
    error: str | None = None,
    artifacts: dict[str, Any] | None = None,
    result_bundle: dict[str, Any] | ResultBundle | None = None,
) -> None:
    """Publish one ordered result lifecycle event."""
    message = build_result_message(
        submission_id=submission_id,
        status=status,
        progress=progress,
        summary=summary,
        error=error,
        artifacts=artifacts,
        result_bundle=result_bundle,
    )
    publisher = pubsub_v1.PublisherClient(
        publisher_options=pubsub_v1.types.PublisherOptions(
            enable_message_ordering=True,
        ),
    )
    future = publisher.publish(
        publisher.topic_path(project_id, pubsub_topic),
        data=json.dumps(message).encode("utf-8"),
        ordering_key=submission_id,
    )
    future.result()
