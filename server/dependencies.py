"""FastAPI dependency providers for config and Firestore client."""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from google.cloud import firestore

from embodiedlab.result_events import publish_result_event
from server.config import ServerConfig, load_server_config
from server.repositories import (
    FirestoreResultRepository,
    FirestoreSubmissionRepository,
)
from server.services.cancellations import (
    CancellationRequester,  # noqa: TC001 - FastAPI resolves annotations
    ResultEventPublisher,  # noqa: TC001 - FastAPI resolves annotations
)
from server.services.execution_reconciliation import (
    ExecutionOutcomeReader,
    read_execution_outcome,
)
from server.services.jobs import request_training_cancellation, run_training_job
from server.services.submission_workflow import (
    SubmissionWorkflow,
    TrainingJobRunner,
)


@lru_cache(maxsize=1)
def get_config() -> ServerConfig:
    """Return the server configuration singleton."""
    return load_server_config()


@lru_cache(maxsize=1)
def _create_db() -> firestore.Client:
    return firestore.Client(database=get_config().db_id)


def get_db() -> firestore.Client:
    """Return the Firestore client singleton."""
    return _create_db()


def get_submission_repository(
    db: Annotated[firestore.Client, Depends(get_db)],
) -> FirestoreSubmissionRepository:
    """Return the submission repository bound to the shared Firestore client."""
    return FirestoreSubmissionRepository(db)


def get_result_repository(
    db: Annotated[firestore.Client, Depends(get_db)],
) -> FirestoreResultRepository:
    """Return the result repository bound to the shared Firestore client."""
    return FirestoreResultRepository(db)


def get_execution_outcome_reader() -> ExecutionOutcomeReader:
    """Return the exact Cloud Run execution outcome reader."""
    return read_execution_outcome


def get_cancellation_requester() -> CancellationRequester:
    """Return the Cloud Run execution cancellation requester."""
    return request_training_cancellation


def get_result_event_publisher() -> ResultEventPublisher:
    """Return the shared ordered result event publisher."""
    return publish_result_event


def get_training_job_runner() -> TrainingJobRunner:
    """Return the Cloud Run training job dispatcher."""
    return run_training_job


def get_submission_workflow(
    config: Annotated[ServerConfig, Depends(get_config)],
    submission_repository: Annotated[
        FirestoreSubmissionRepository,
        Depends(get_submission_repository),
    ],
    run_job: Annotated[TrainingJobRunner, Depends(get_training_job_runner)],
    publish_event: Annotated[
        ResultEventPublisher,
        Depends(get_result_event_publisher),
    ],
) -> SubmissionWorkflow:
    """Return the server-owned submission workflow."""
    return SubmissionWorkflow(
        submission_repository=submission_repository,
        config=config,
        run_job=run_job,
        publish_event=publish_event,
    )
