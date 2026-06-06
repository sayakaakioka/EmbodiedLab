"""Training execution helpers for the Cloud Run trainer job."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

from embodiedlab.result_models import (
    ReplayLogStep,
    ResultBundle,
    ResultStatus,
    build_result_bundle,
)
from embodiedlab.training.training_config import TrainingConfig
from embodiedlab.training.training_converter import (
    ScenarioRuntimeConversion,
    convert_submission_to_spec,
    describe_runtime_conversion,
    parse_scenario_bundle,
)

TrainModel = Callable[..., dict[str, Any]]
TrainingProgressCallback = Callable[[int, int], None]
TrainingDiagnosticCallback = Callable[[str, dict[str, object]], None]
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


def execute_training_run(  # noqa: PLR0913
    *,
    inputs: TrainingInputs,
    model_bucket: str,
    submission_id: str,
    train_model: TrainModel,
    upload_model: UploadModel,
    progress_callback: TrainingProgressCallback | None = None,
    diagnostic_callback: TrainingDiagnosticCallback | None = None,
) -> TrainingExecution:
    """Train and upload using already-validated runtime inputs."""
    with TemporaryDirectory() as tmpdir:
        model_base_path = str(Path(tmpdir) / "policy")
        train_kwargs = {
            "spec": inputs.spec,
            "training": inputs.training,
            "model_output_path": model_base_path,
        }
        if progress_callback is not None:
            train_kwargs["progress_callback"] = progress_callback
        if diagnostic_callback is not None:
            train_kwargs["diagnostic_callback"] = diagnostic_callback

        summary = train_model(**train_kwargs)
        replay_payloads = summary.pop("replay_steps", [])
        replay_steps = [
            ReplayLogStep.model_validate(
                {
                    **payload,
                    "schema_version": "replay-log.v0",
                    "scenario_id": inputs.scenario.scenario_id,
                    "job_id": submission_id,
                },
            )
            for payload in replay_payloads
        ]
        artifacts = upload_model(
            local_model_base_path=model_base_path,
            bucket_name=model_bucket,
            submission_id=submission_id,
            replay_steps=replay_steps,
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

