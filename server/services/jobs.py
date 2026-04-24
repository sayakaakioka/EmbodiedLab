"""Cloud Run job trigger helpers for the API service."""

from __future__ import annotations

from google.api_core.client_options import ClientOptions
from google.cloud import run_v2

from server.config import ServerConfig


def run_training_job(config: ServerConfig, submission_id: str) -> None:
    """Trigger the configured Cloud Run Job with the submission ID override."""
    jobs_client = run_v2.JobsClient(
        client_options=ClientOptions(api_endpoint=f"{config.region}-run.googleapis.com"),
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
    jobs_client.run_job(request=request)
