"""Training execution helpers for the Cloud Run trainer job."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

from embodiedlab.result_models import (
    ResultBundle,
    ResultStatus,
    TrainingSummary,
    build_result_bundle,
)
from embodiedlab.schemas import ScenarioBundle, TrainingSpec
from embodiedlab.training.training_converter import (
    ScenarioRuntimeConversion,
    convert_scenario_to_spec,
    describe_runtime_conversion,
)

if TYPE_CHECKING:
    from embodiedlab.training.training_models import ContinuousNavigationSpec

TrainModel = Callable[..., dict[str, Any]]
TrainingProgressCallback = Callable[[int, int], None]
TrainingDiagnosticCallback = Callable[[str, dict[str, object]], None]


UploadModel = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class TrainingInputs:
    """Validated runtime inputs required to execute training."""

    scenario: ScenarioBundle
    training: TrainingSpec
    spec: ContinuousNavigationSpec
    conversion: ScenarioRuntimeConversion


@dataclass(frozen=True)
class TrainingExecution:
    """Outcome of a completed training run."""

    result_bundle: ResultBundle


def parse_training_submission(
    submission: dict[str, Any],
) -> TrainingInputs:
    """Validate a submission payload and convert it into runtime training inputs."""
    scenario = ScenarioBundle.model_validate(submission["scenario"])
    spec = convert_scenario_to_spec(scenario)
    conversion = describe_runtime_conversion(scenario)
    return TrainingInputs(
        scenario=scenario,
        training=scenario.training,
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
            "scenario_id": inputs.scenario.scenario_id,
            "job_id": submission_id,
        }
        if progress_callback is not None:
            train_kwargs["progress_callback"] = progress_callback
        if diagnostic_callback is not None:
            train_kwargs["diagnostic_callback"] = diagnostic_callback

        training_output = train_model(**train_kwargs)
        summary = TrainingSummary.model_validate(training_output["summary"])
        artifacts = upload_model(
            local_model_base_path=model_base_path,
            bucket_name=model_bucket,
            submission_id=submission_id,
            scenario=inputs.scenario,
            replay_bundle_dir=training_output["replay_bundle_dir"],
        )

    result_bundle = build_result_bundle(
        scenario=inputs.scenario,
        job_id=submission_id,
        status=ResultStatus.COMPLETED,
        summary=summary,
        artifacts=artifacts,
    )

    return TrainingExecution(result_bundle=result_bundle)
