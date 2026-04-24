"""Training execution helpers for the Cloud Run trainer job."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from embodiedlab.training.training_config import TrainingConfig
from embodiedlab.training.training_converter import convert_submission_to_spec

TrainModel = Callable[..., dict[str, Any]]
UploadModel = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class TrainingInputs:
    """Validated runtime inputs required to execute training."""

    training: TrainingConfig
    spec: object


@dataclass(frozen=True)
class TrainingExecution:
    """Outcome of a completed training run."""

    summary: dict[str, Any]
    artifacts: dict[str, Any]


def parse_training_submission(
    submission: dict[str, Any],
) -> TrainingInputs:
    """Validate a submission payload and convert it into runtime training inputs."""
    training = TrainingConfig.model_validate(submission["training"])
    spec = convert_submission_to_spec(submission)
    return TrainingInputs(training=training, spec=spec)


def execute_training_run(
    *,
    inputs: TrainingInputs,
    model_bucket: str,
    submission_id: str,
    train_model: TrainModel,
    upload_model: UploadModel,
) -> TrainingExecution:
    """Train and upload using already-validated runtime inputs."""
    with TemporaryDirectory() as tmpdir:
        model_base_path = str(Path(tmpdir) / "policy")
        summary = train_model(
            spec=inputs.spec,
            training=inputs.training,
            model_output_path=model_base_path,
        )
        artifacts = upload_model(
            local_model_base_path=model_base_path,
            bucket_name=model_bucket,
            submission_id=submission_id,
        )

    return TrainingExecution(summary=summary, artifacts=artifacts)


def execute_training_run_from_submission(
    *,
    submission: dict[str, Any],
    model_bucket: str,
    submission_id: str,
    train_model: TrainModel,
    upload_model: UploadModel,
) -> TrainingExecution:
    """Train a policy, upload the saved model, and return the resulting payloads."""
    inputs = parse_training_submission(submission)
    return execute_training_run(
        inputs=inputs,
        model_bucket=model_bucket,
        submission_id=submission_id,
        train_model=train_model,
        upload_model=upload_model,
    )
