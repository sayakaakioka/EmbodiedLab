"""GCS upload helper for trained model artifacts."""

from __future__ import annotations

import gzip
import logging
import re
import warnings
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

import onnx
import torch
from google.api_core.exceptions import PreconditionFailed
from google.cloud import storage
from stable_baselines3 import PPO

from embodiedlab.artifact_integrity import (
    FileIntegrity,
    compute_file_integrity,
)
from embodiedlab.result_models import (
    EvalReplayBundleChunk,
    ReplayBundleManifest,
    ReplayLogStep,
    TrainReplayBundleChunk,
)
from embodiedlab.schemas import (
    ForwardCameraSensor,
    GoalVectorSensor,
    ScenarioBundle,
    semantic_channel_layout,
)
from embodiedlab.training.navigation_final_policy import (
    navigation_final_deterministic_action,
)

if TYPE_CHECKING:
    from stable_baselines3.common.policies import BasePolicy


_PYTORCH_LEAFSPEC_WARNING = (
    "`isinstance(treespec, LeafSpec)` is deprecated, use "
    "`isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead."
)
_PYTORCH_LEAFSPEC_WARNING_PATTERN = rf"^{re.escape(_PYTORCH_LEAFSPEC_WARNING)}$"


class _OptionalTorchvisionRegistrationFilter(logging.Filter):
    """Hide only unavailable optional torchvision operator registrations."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Keep every log except the expected unused torchvision registrations."""
        return not (
            record.name == "torch.onnx._internal.exporter._registration"
            and record.getMessage().startswith(
                "torchvision is not installed. Skipping torchvision::",
            )
        )


class OnnxableContinuousNavigationPolicy(torch.nn.Module):
    """Wrapper exposing the continuous policy as ONNX-friendly inputs."""

    def __init__(
        self,
        policy: BasePolicy,
        *,
        image_observation_name: str,
        goal_vector_observation_name: str,
    ) -> None:
        """Store the trained Stable-Baselines3 policy."""
        super().__init__()
        self.policy = policy
        self.image_observation_name = image_observation_name
        self.goal_vector_observation_name = goal_vector_observation_name

    def forward(
        self,
        image_observation: torch.Tensor,
        goal_vector_observation: torch.Tensor,
    ) -> torch.Tensor:
        """Return deterministic continuous action values for a batch."""
        return navigation_final_deterministic_action(
            self.policy,
            {
                self.image_observation_name: image_observation,
                self.goal_vector_observation_name: goal_vector_observation,
            },
        )


def _observation_contract(
    scenario: ScenarioBundle,
) -> tuple[ForwardCameraSensor, GoalVectorSensor]:
    camera = next(
        sensor for sensor in scenario.sensors if isinstance(sensor, ForwardCameraSensor)
    )
    goal_vector = next(
        sensor for sensor in scenario.sensors if isinstance(sensor, GoalVectorSensor)
    )
    return camera, goal_vector


def _validate_policy_observation_contract(
    policy: BasePolicy,
    scenario: ScenarioBundle,
) -> None:
    """Reject a saved policy whose observation tensors differ from the scenario."""
    camera, goal_vector = _observation_contract(scenario)
    expected = {
        camera.observation_name: (
            len(semantic_channel_layout(camera.semantic_mode)),
            camera.height,
            camera.width,
        ),
        goal_vector.observation_name: (len(goal_vector.values),),
    }
    observation_space = getattr(policy, "observation_space", None)
    spaces = getattr(observation_space, "spaces", None)
    if spaces is None:
        msg = "Saved policy must expose a dictionary observation space"
        raise ValueError(msg)
    actual = {name: tuple(space.shape) for name, space in spaces.items()}
    if actual != expected:
        msg = (
            "Saved policy observation contract does not match the scenario: "
            f"expected={expected}, actual={actual}"
        )
        raise ValueError(msg)


def _onnx_shape(value_info: onnx.ValueInfoProto) -> list[int]:
    """Return a normalized ONNX tensor shape, using -1 for symbolic dimensions."""
    return [
        dimension.dim_value if dimension.HasField("dim_value") else -1
        for dimension in value_info.type.tensor_type.shape.dim
    ]


def _validate_exported_onnx(
    path: str,
    *,
    expected_opset: int,
    expected_inputs: dict[str, list[int]],
    expected_output_shape: list[int],
) -> None:
    """Validate the actual exported graph against the published contract."""
    model = onnx.load(path)
    onnx.checker.check_model(model)
    opsets = {item.domain: item.version for item in model.opset_import}
    if opsets.get("") != expected_opset:
        msg = f"ONNX opset does not match contract: {opsets.get('')}"
        raise ValueError(msg)
    graph_inputs = list(model.graph.input)
    actual_inputs = {
        value.name: {
            "dtype": value.type.tensor_type.elem_type,
            "shape": _onnx_shape(value),
        }
        for value in graph_inputs
    }
    actual_input_shapes = {value.name: _onnx_shape(value) for value in graph_inputs}
    if actual_input_shapes != expected_inputs or any(
        value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT
        for value in graph_inputs
    ):
        msg = (
            "ONNX inputs do not match contract: "
            f"expected={expected_inputs}, actual={actual_inputs}"
        )
        raise ValueError(msg)
    outputs = list(model.graph.output)
    actual_output_shape = _onnx_shape(outputs[0]) if len(outputs) == 1 else []
    if (
        len(outputs) != 1
        or outputs[0].name != "action"
        or outputs[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT
        or len(actual_output_shape) != len(expected_output_shape)
        or actual_output_shape[0] != expected_output_shape[0]
        or actual_output_shape[1] not in {-1, expected_output_shape[1]}
    ):
        actual_outputs = [
            {
                "name": output.name,
                "dtype": output.type.tensor_type.elem_type,
                "shape": _onnx_shape(output),
            }
            for output in outputs
        ]
        msg = f"ONNX action output does not match contract: {actual_outputs}"
        raise ValueError(msg)


def export_model_to_onnx(
    local_model_base_path: str,
    scenario: ScenarioBundle,
) -> str:
    """Convert the saved Stable-Baselines3 continuous policy zip to ONNX."""
    model = PPO.load(local_model_base_path)
    _validate_policy_observation_contract(model.policy, scenario)
    camera, goal_vector = _observation_contract(scenario)
    image_shape = (
        len(semantic_channel_layout(camera.semantic_mode)),
        camera.height,
        camera.width,
    )
    onnx_path = f"{local_model_base_path}.onnx"
    onnxable_policy = OnnxableContinuousNavigationPolicy(
        model.policy,
        image_observation_name=camera.observation_name,
        goal_vector_observation_name=goal_vector.observation_name,
    )
    onnxable_policy.eval()
    dummy_obs_0 = torch.zeros(
        (1, *image_shape),
        dtype=torch.float32,
    )
    dummy_obs_1 = torch.zeros((1, len(goal_vector.values)), dtype=torch.float32)
    registration_logger = logging.getLogger(
        "torch.onnx._internal.exporter._registration",
    )
    registration_filter = _OptionalTorchvisionRegistrationFilter()
    registration_logger.addFilter(registration_filter)
    try:
        with warnings.catch_warnings():
            # PyTorch 2.13.0 deep-copies LeafSpec during Dynamo export. Remove
            # this workaround after https://github.com/pytorch/pytorch/pull/191416.
            warnings.filterwarnings(
                "ignore",
                message=_PYTORCH_LEAFSPEC_WARNING_PATTERN,
                category=FutureWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=(
                    r"Anomaly Detection has been enabled\. This mode will increase "
                    r"the runtime and should only be enabled for debugging\."
                ),
                category=UserWarning,
            )
            torch.onnx.export(
                onnxable_policy,
                (dummy_obs_0, dummy_obs_1),
                onnx_path,
                input_names=[camera.observation_name, goal_vector.observation_name],
                output_names=["action"],
                dynamic_shapes=(
                    {0: torch.export.Dim.AUTO},
                    {0: torch.export.Dim.AUTO},
                ),
                opset_version=18,
                dynamo=True,
                external_data=False,
                verbose=False,
            )
    finally:
        registration_logger.removeFilter(registration_filter)
    _validate_exported_onnx(
        onnx_path,
        expected_opset=18,
        expected_inputs={
            camera.observation_name: [-1, *image_shape],
            goal_vector.observation_name: [-1, len(goal_vector.values)],
        },
        expected_output_shape=[-1, 2],
    )
    return onnx_path


def upload_file(
    *,
    bucket: storage.Bucket,
    local_path: str,
    blob_path: str,
    content_type: str,
) -> None:
    """Upload a local file to the configured GCS bucket."""
    blob = bucket.blob(blob_path)
    integrity = compute_file_integrity(local_path)
    blob.metadata = {
        "sha256": integrity.sha256,
        "size_bytes": str(integrity.size_bytes),
    }
    try:
        blob.upload_from_filename(
            local_path,
            content_type=content_type,
            if_generation_match=0,
        )
    except PreconditionFailed as exc:
        blob.reload()
        remote_metadata = blob.metadata or {}
        if (
            blob.size == integrity.size_bytes
            and remote_metadata.get("sha256") == integrity.sha256
            and remote_metadata.get("size_bytes") == str(integrity.size_bytes)
        ):
            return
        msg = f"GCS object already exists with different content: {blob_path}"
        raise ValueError(msg) from exc


def _load_replay_manifest(
    bundle_dir: Path,
) -> tuple[ReplayBundleManifest, Path]:
    """Load a Replay manifest after rejecting links and undeclared files."""
    manifest_path = bundle_dir / "manifest.json"
    if manifest_path.is_symlink():
        msg = f"Replay Bundle must not contain symbolic links: {manifest_path}"
        raise ValueError(msg)
    if not manifest_path.is_file():
        msg = f"Replay Bundle manifest not found: {manifest_path}"
        raise FileNotFoundError(msg)

    actual_paths: set[str] = set()
    for local_path in bundle_dir.rglob("*"):
        if local_path.is_symlink():
            msg = f"Replay Bundle must not contain symbolic links: {local_path}"
            raise ValueError(msg)
        if local_path.is_file():
            actual_paths.add(local_path.relative_to(bundle_dir).as_posix())

    manifest = ReplayBundleManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8"),
    )
    expected_paths = {"manifest.json", *(chunk.path for chunk in manifest.chunks)}
    if actual_paths != expected_paths:
        msg = (
            "Replay Bundle files must match the manifest; "
            f"missing={sorted(expected_paths - actual_paths)}, "
            f"unexpected={sorted(actual_paths - expected_paths)}"
        )
        raise ValueError(msg)
    return manifest, manifest_path


def _validated_replay_files(
    bundle_dir: Path,
    manifest: ReplayBundleManifest,
) -> list[tuple[str, Path, str]]:
    """Return all Replay files after validating every declared chunk."""
    chunk_by_path = {chunk.path: chunk for chunk in manifest.chunks}
    relative_paths = sorted(
        {"manifest.json", *chunk_by_path},
        key=lambda value: (value == "manifest.json", value),
    )
    files: list[tuple[str, Path, str]] = []
    for relative_path in relative_paths:
        parsed_path = PurePosixPath(relative_path)
        local_path = bundle_dir / Path(*parsed_path.parts)
        if relative_path != "manifest.json":
            _validate_replay_chunk(
                local_path=local_path,
                relative_path=relative_path,
                declared=chunk_by_path[relative_path],
                manifest=manifest,
            )
        content_type = (
            "application/gzip" if local_path.suffix == ".gz" else "application/json"
        )
        files.append((relative_path, local_path, content_type))
    return files


def _validate_replay_chunk(
    *,
    local_path: Path,
    relative_path: str,
    declared: TrainReplayBundleChunk | EvalReplayBundleChunk,
    manifest: ReplayBundleManifest,
) -> None:
    """Validate one compressed Replay chunk against its manifest entry."""
    actual = compute_file_integrity(local_path)
    if actual.size_bytes != declared.size_bytes or actual.sha256 != declared.sha256:
        msg = f"Replay chunk integrity does not match manifest: {relative_path}"
        raise ValueError(msg)
    checkpoint_steps: list[int] = []
    episode_ids: set[str] = set()
    with gzip.open(local_path, mode="rt", encoding="utf-8") as replay_file:
        for line_number, line in enumerate(replay_file, start=1):
            if not line.strip():
                msg = (
                    f"Replay chunk contains an empty row: {relative_path}:{line_number}"
                )
                raise ValueError(msg)
            step = ReplayLogStep.model_validate_json(line)
            if (
                step.job_id != manifest.job_id
                or step.scenario_id != manifest.scenario_id
                or step.phase != declared.phase
                or step.policy_mode != declared.policy_mode
            ):
                msg = (
                    "Replay row identity does not match its manifest chunk: "
                    f"{relative_path}:{line_number}"
                )
                raise ValueError(msg)
            if (
                declared.phase == "eval"
                and step.checkpoint_step != declared.checkpoint_step
            ):
                msg = (
                    "Replay eval row checkpoint does not match its chunk: "
                    f"{relative_path}:{line_number}"
                )
                raise ValueError(msg)
            checkpoint_steps.append(step.checkpoint_step)
            episode_ids.add(step.episode_id)
    if len(checkpoint_steps) != declared.step_count:
        msg = f"Replay chunk step_count does not match its rows: {relative_path}"
        raise ValueError(msg)
    if isinstance(declared, TrainReplayBundleChunk) and (
        min(checkpoint_steps) != declared.start_step
        or max(checkpoint_steps) != declared.end_step
    ):
        msg = (
            "Replay train row checkpoint range does not match its chunk: "
            f"{relative_path}"
        )
        raise ValueError(msg)
    if isinstance(declared, EvalReplayBundleChunk) and (
        len(episode_ids) != declared.episode_count
    ):
        msg = f"Replay eval episode_count does not match its rows: {relative_path}"
        raise ValueError(msg)


def upload_replay_bundle_to_gcs(
    *,
    bucket_name: str,
    submission_id: str,
    scenario_id: str,
    replay_bundle_dir: str,
) -> dict:
    """Upload a Replay Bundle directory and return manifest artifact metadata."""
    manifest_path, replay_files = _prepare_replay_bundle(
        submission_id=submission_id,
        scenario_id=scenario_id,
        replay_bundle_dir=replay_bundle_dir,
    )
    bucket = storage.Client().bucket(bucket_name)
    return _upload_prepared_replay_bundle(
        bucket=bucket,
        bucket_name=bucket_name,
        submission_id=submission_id,
        manifest_path=manifest_path,
        replay_files=replay_files,
    )


def _prepare_replay_bundle(
    *,
    submission_id: str,
    scenario_id: str,
    replay_bundle_dir: str,
) -> tuple[Path, list[tuple[str, Path, str]]]:
    """Validate every local Replay artifact before any remote write."""
    bundle_dir = Path(replay_bundle_dir)
    manifest, manifest_path = _load_replay_manifest(bundle_dir)
    if manifest.job_id != submission_id or manifest.scenario_id != scenario_id:
        msg = "Replay Bundle identity does not match the submitted training run"
        raise ValueError(msg)
    return manifest_path, _validated_replay_files(bundle_dir, manifest)


def _upload_prepared_replay_bundle(
    *,
    bucket: storage.Bucket,
    bucket_name: str,
    submission_id: str,
    manifest_path: Path,
    replay_files: list[tuple[str, Path, str]],
) -> dict:
    """Upload one already-validated Replay Bundle, manifest last."""
    for relative_path, local_path, content_type in replay_files:
        blob_path = f"results/{submission_id}/replay/{relative_path}"
        upload_file(
            bucket=bucket,
            local_path=str(local_path),
            blob_path=blob_path,
            content_type=content_type,
        )

    manifest_integrity = compute_file_integrity(manifest_path)
    return {
        "replay_bundle": {
            "storage": "gcs",
            "bucket": bucket_name,
            "path": f"results/{submission_id}/replay/manifest.json",
            "format": "json",
            **manifest_integrity.as_dict(),
        },
    }


def _action_output_metadata() -> dict:
    return {
        "name": "action",
        "layout": ["forward", "turn"],
        "action_mapping": {
            "forward": "sigmoid(policy_forward)",
            "turn": "clip(policy_turn, -3, 3) / 3",
        },
    }


def _onnx_metadata(
    *,
    bucket_name: str,
    path: str,
    scenario: ScenarioBundle,
    integrity: FileIntegrity,
) -> dict:
    camera, goal_vector = _observation_contract(scenario)
    channel_layout = semantic_channel_layout(camera.semantic_mode)
    return {
        "storage": "gcs",
        "bucket": bucket_name,
        "path": path,
        "format": "onnx",
        "target": "onnx-runtime",
        "opset_version": 18,
        **integrity.as_dict(),
        "inputs": [
            {
                "name": camera.observation_name,
                "shape": [
                    -1,
                    len(channel_layout),
                    camera.height,
                    camera.width,
                ],
                "dtype": "float32",
                "layout": list(channel_layout),
            },
            {
                "name": goal_vector.observation_name,
                "shape": [-1, len(goal_vector.values)],
                "dtype": "float32",
                "layout": list(goal_vector.values),
            },
        ],
        "output": _action_output_metadata(),
    }


def upload_model_to_gcs(
    *,
    local_model_base_path: str,
    bucket_name: str,
    submission_id: str,
    scenario: ScenarioBundle,
    replay_bundle_dir: str,
) -> dict:
    """Upload the ONNX policy and a Replay Bundle to GCS."""
    local_onnx_path = export_model_to_onnx(local_model_base_path, scenario)
    onnx_blob_path = f"results/{submission_id}/model/policy.onnx"
    onnx_integrity = compute_file_integrity(local_onnx_path)
    manifest_path, replay_files = _prepare_replay_bundle(
        submission_id=submission_id,
        scenario_id=scenario.scenario_id,
        replay_bundle_dir=replay_bundle_dir,
    )

    bucket = storage.Client().bucket(bucket_name)

    upload_file(
        bucket=bucket,
        local_path=local_onnx_path,
        blob_path=onnx_blob_path,
        content_type="application/octet-stream",
    )
    replay_artifact = _upload_prepared_replay_bundle(
        bucket=bucket,
        bucket_name=bucket_name,
        submission_id=submission_id,
        manifest_path=manifest_path,
        replay_files=replay_files,
    )
    return {
        "onnx_model": _onnx_metadata(
            bucket_name=bucket_name,
            path=onnx_blob_path,
            scenario=scenario,
            integrity=onnx_integrity,
        ),
        **replay_artifact,
    }
