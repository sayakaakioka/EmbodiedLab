"""Server configuration loaded from environment variables."""

from __future__ import annotations

from dataclasses import dataclass

from embodiedlab.config_utils import get_required_env


@dataclass(frozen=True)
class ServerConfig:
    """Immutable runtime configuration for the API server."""

    db_id: str
    region: str
    job_path: str


def load_server_config() -> ServerConfig:
    """Build ServerConfig from environment variables."""
    project_id = get_required_env("PROJECT_ID")
    region = get_required_env("REGION")
    trainer_job_name = get_required_env("TRAINER_JOB_NAME")

    return ServerConfig(
        db_id=get_required_env("DB_ID"),
        region=region,
        job_path=f"projects/{project_id}/locations/{region}/jobs/{trainer_job_name}",
    )
