"""Trainer configuration loaded from environment variables."""

from __future__ import annotations

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

def load_trainer_config() -> TrainerConfig:
    """Build TrainerConfig from environment variables."""
    return TrainerConfig(
        db_id=get_required_env("DB_ID"),
        model_bucket=get_required_env("MODEL_BUCKET"),
        submission_id=get_required_env("SUBMISSION_ID"),
        pubsub_topic=get_required_env("PUBSUB_TOPIC"),
        project_id=get_required_env("PROJECT_ID"),
    )
