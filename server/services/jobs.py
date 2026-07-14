"""Cloud Run job trigger helpers for the API service."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from google.api_core.client_options import ClientOptions
from google.cloud import run_v2

from server.config import ServerConfig

if TYPE_CHECKING:
    from google.api_core.operation import Operation


CreateJobsClient = Callable[..., run_v2.JobsClient]
CreateExecutionsClient = Callable[..., run_v2.ExecutionsClient]


def run_training_job(
    config: ServerConfig,
    submission_id: str,
    *,
    create_jobs_client: CreateJobsClient = run_v2.JobsClient,
) -> str:
    """Trigger the configured Cloud Run Job with the submission ID override."""
    jobs_client = create_jobs_client(
        client_options=ClientOptions(
            api_endpoint=f"{config.region}-run.googleapis.com",
        ),
    )

    request = run_v2.RunJobRequest(
        name=config.job_path,
        overrides=run_v2.RunJobRequest.Overrides(
            container_overrides=[
                run_v2.RunJobRequest.Overrides.ContainerOverride(
                    env=[
                        run_v2.EnvVar(
                            name="SUBMISSION_ID",
                            value=submission_id,
                        ),
                    ],
                ),
            ],
        ),
    )
    operation = jobs_client.run_job(request=request)
    metadata = operation.metadata
    execution_name = getattr(metadata, "name", "") if metadata is not None else ""
    if not execution_name:
        msg = "Cloud Run did not return an execution name"
        raise RuntimeError(msg)
    return execution_name


def request_training_cancellation(
    config: ServerConfig,
    execution_name: str,
    *,
    create_executions_client: CreateExecutionsClient = run_v2.ExecutionsClient,
) -> Operation:
    """Request cancellation of one exact Cloud Run Execution."""
    executions_client = create_executions_client(
        client_options=ClientOptions(
            api_endpoint=f"{config.region}-run.googleapis.com",
        ),
    )
    return executions_client.cancel_execution(
        request=run_v2.CancelExecutionRequest(name=execution_name),
    )
