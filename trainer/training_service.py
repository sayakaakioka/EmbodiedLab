"""Training execution helpers for the Cloud Run trainer job."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

from embodiedlab.result_models import ResultBundle, ResultStatus, build_result_bundle
from embodiedlab.training.training_config import TrainingConfig
from embodiedlab.training.training_converter import (
    ScenarioRuntimeConversion,
    convert_submission_to_spec,
    describe_runtime_conversion,
    parse_scenario_bundle,
)

TrainModel = Callable[..., dict[str, Any]]
if TYPE_CHECKING:
    from embodiedlab.schemas import ScenarioBundle


UploadModel = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class TrainingInputs:
    """Validated runtime inputs required to execute training."""

    scenario: ScenarioBundle
    training: TrainingConfig
    spec: object
    conversion: ScenarioRuntimeConversion


@dataclass(frozen=True)
class TrainingExecution:
    """Outcome of a completed training run."""

    summary: dict[str, Any]
    artifacts: dict[str, Any]
    result_bundle: ResultBundle


def parse_training_submission(
    submission: dict[str, Any],
) -> TrainingInputs:
    """Validate a submission payload and convert it into runtime training inputs."""
    scenario = parse_scenario_bundle(submission)
    training = TrainingConfig.model_validate(
        {
            **scenario.training.model_dump(mode="json"),
            "max_steps": scenario.training.max_episode_steps,
        },
    )
    spec = convert_submission_to_spec(scenario)
    conversion = describe_runtime_conversion(scenario)
    return TrainingInputs(
        scenario=scenario,
        training=training,
        spec=spec,
        conversion=conversion,
    )


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

    summary = {
        **summary,
        "training_timesteps": summary.get(
            "training_timesteps",
            inputs.training.timesteps,
        ),
        "training_seed": summary.get("training_seed", inputs.training.seed),
    }
    result_bundle = build_result_bundle(
        scenario=inputs.scenario,
        job_id=submission_id,
        status=ResultStatus.COMPLETED,
        summary=summary,
        artifacts=artifacts,
    )

    return TrainingExecution(
        summary=summary,
        artifacts=artifacts,
        result_bundle=result_bundle,
    )


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
