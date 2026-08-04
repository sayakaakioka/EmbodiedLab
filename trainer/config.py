"""Trainer configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

from embodiedlab.config_utils import get_required_env


@dataclass(frozen=True)
class TrainerConfig:
    """Immutable runtime configuration for the Cloud Run Job."""

    db_id: str
    model_bucket: str
    submission_id: str
    pubsub_topic: str
    project_id: str
    execution_name: str | None = None


def load_trainer_config() -> TrainerConfig:
    """Build TrainerConfig from environment variables."""
    project_id = get_required_env("PROJECT_ID")
    return TrainerConfig(
        db_id=get_required_env("DB_ID"),
        model_bucket=get_required_env("MODEL_BUCKET"),
        submission_id=get_required_env("SUBMISSION_ID"),
        pubsub_topic=get_required_env("PUBSUB_TOPIC"),
        project_id=project_id,
        execution_name=_cloud_run_execution_name(project_id),
    )


def _cloud_run_execution_name(project_id: str) -> str | None:
    execution_id = os.environ.get("CLOUD_RUN_EXECUTION")
    job_name = os.environ.get("CLOUD_RUN_JOB")
    if execution_id is None and job_name is None:
        return None
    if not execution_id or not job_name:
        msg = "CLOUD_RUN_EXECUTION and CLOUD_RUN_JOB must be set together"
        raise RuntimeError(msg)
    region = get_required_env("REGION")
    return (
        f"projects/{project_id}/locations/{region}/jobs/{job_name}/"
        f"executions/{execution_id}"
    )
