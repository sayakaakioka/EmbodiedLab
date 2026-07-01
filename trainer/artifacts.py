"""GCS upload helper for trained model artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from google.cloud import storage
from stable_baselines3 import PPO

from embodiedlab.continuous_navigation_env import (
    IMAGE_OBSERVATION_CHANNELS,
    IMAGE_OBSERVATION_HEIGHT,
    IMAGE_OBSERVATION_WIDTH,
    NUMERIC_OBSERVATION_SIZE,
)
from embodiedlab.training.navigation_final_policy import (
    navigation_final_deterministic_action,
)

if TYPE_CHECKING:
    from stable_baselines3.common.policies import BasePolicy

IMAGE_OBSERVATION_SIZE = (
    IMAGE_OBSERVATION_CHANNELS * IMAGE_OBSERVATION_HEIGHT * IMAGE_OBSERVATION_WIDTH
)
SENTIS_OBSERVATION_SIZE = IMAGE_OBSERVATION_SIZE + NUMERIC_OBSERVATION_SIZE


class OnnxableContinuousNavigationPolicy(torch.nn.Module):
    """Wrapper exposing the continuous policy as ONNX-friendly inputs."""

    def __init__(self, policy: BasePolicy) -> None:
        """Store the trained Stable-Baselines3 policy."""
        super().__init__()
        self.policy = policy

    def forward(
        self,
        obs_0: torch.Tensor,
        obs_1: torch.Tensor,
    ) -> torch.Tensor:
        """Return deterministic continuous action values for a batch."""
        return navigation_final_deterministic_action(
            self.policy,
            {"obs_0": obs_0, "obs_1": obs_1},
        )


class SentisContinuousNavigationPolicy(torch.nn.Module):
    """Wrapper exposing a fixed continuous observation tensor for Sentis."""

    def __init__(self, policy: BasePolicy) -> None:
        """Store the trained Stable-Baselines3 policy."""
        super().__init__()
        self.policy = OnnxableContinuousNavigationPolicy(policy)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Return [forward, turn] actions for a compact observation tensor."""
        obs_0 = observation[:, 0:IMAGE_OBSERVATION_SIZE].reshape(
            -1,
            IMAGE_OBSERVATION_CHANNELS,
            IMAGE_OBSERVATION_HEIGHT,
            IMAGE_OBSERVATION_WIDTH,
        )
        obs_1 = observation[
            :,
            IMAGE_OBSERVATION_SIZE:SENTIS_OBSERVATION_SIZE,
        ]
        return self.policy(obs_0, obs_1)


def export_model_to_onnx(local_model_base_path: str) -> str:
    """Convert the saved Stable-Baselines3 continuous policy zip to ONNX."""
    model = PPO.load(local_model_base_path)
    onnx_path = f"{local_model_base_path}.onnx"
    onnxable_policy = OnnxableContinuousNavigationPolicy(model.policy)
    dummy_obs_0 = torch.zeros(
        (
            1,
            IMAGE_OBSERVATION_CHANNELS,
            IMAGE_OBSERVATION_HEIGHT,
            IMAGE_OBSERVATION_WIDTH,
        ),
        dtype=torch.float32,
    )
    dummy_obs_1 = torch.zeros((1, NUMERIC_OBSERVATION_SIZE), dtype=torch.float32)
    torch.onnx.export(
        onnxable_policy,
        (dummy_obs_0, dummy_obs_1),
        onnx_path,
        input_names=["obs_0", "obs_1"],
        output_names=["action"],
        dynamic_axes={
            "obs_0": {0: "batch"},
            "obs_1": {0: "batch"},
            "action": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    return onnx_path


def export_model_to_sentis_onnx(local_model_base_path: str) -> str:
    """Convert the saved continuous policy zip to a Sentis-compatible ONNX file."""
    model = PPO.load(local_model_base_path)
    onnx_path = f"{local_model_base_path}.sentis.onnx"
    sentis_policy = SentisContinuousNavigationPolicy(model.policy)
    dummy_observation = torch.zeros((1, SENTIS_OBSERVATION_SIZE), dtype=torch.float32)

    torch.onnx.export(
        sentis_policy,
        dummy_observation,
        onnx_path,
        input_names=["observation"],
        output_names=["action"],
        opset_version=15,
        dynamo=False,
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
    blob.upload_from_filename(local_path, content_type=content_type)


def upload_replay_bundle_to_gcs(
    *,
    bucket_name: str,
    submission_id: str,
    replay_bundle_dir: str,
) -> dict:
    """Upload a Replay Bundle directory and return manifest artifact metadata."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    bundle_dir = Path(replay_bundle_dir)
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        msg = f"Replay Bundle manifest not found: {manifest_path}"
        raise FileNotFoundError(msg)

    for local_path in bundle_dir.rglob("*"):
        if not local_path.is_file():
            continue
        relative_path = local_path.relative_to(bundle_dir).as_posix()
        blob_path = f"results/{submission_id}/replay/{relative_path}"
        content_type = (
            "application/gzip" if local_path.suffix == ".gz" else "application/json"
        )
        upload_file(
            bucket=bucket,
            local_path=str(local_path),
            blob_path=blob_path,
            content_type=content_type,
        )

    return {
        "replay_bundle": {
            "storage": "gcs",
            "bucket": bucket_name,
            "path": f"results/{submission_id}/replay/manifest.json",
            "format": "json",
            "schema_version": "replay-bundle.v0",
        },
    }


def _sentis_metadata(*, bucket_name: str, path: str) -> dict:
    return {
        "storage": "gcs",
        "bucket": bucket_name,
        "path": path,
        "format": "onnx",
        "target": "unity-sentis",
        "opset_version": 15,
        "input": {
            "name": "observation",
            "shape": [1, SENTIS_OBSERVATION_SIZE],
            "dtype": "float32",
            "layout": [
                (
                    "obs_0_chw_"
                    f"{IMAGE_OBSERVATION_CHANNELS}x"
                    f"{IMAGE_OBSERVATION_HEIGHT}x"
                    f"{IMAGE_OBSERVATION_WIDTH}"
                ),
                "obs_1_angle_degrees",
                "obs_1_distance_meters",
            ],
        },
        "output": {
            "name": "action",
            "layout": ["forward", "turn"],
            "action_mapping": {
                "forward": "sigmoid(policy_forward)",
                "turn": "clip(policy_turn, -3, 3) / 3",
            },
        },
    }


def upload_model_to_gcs(
    *,
    local_model_base_path: str,
    bucket_name: str,
    submission_id: str,
    replay_bundle_dir: str | None = None,
) -> dict:
    """Upload the saved model zip, ONNX exports, and replay log to GCS."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    local_zip_path = f"{local_model_base_path}.zip"
    local_onnx_path = export_model_to_onnx(local_model_base_path)
    local_sentis_path = export_model_to_sentis_onnx(local_model_base_path)
    zip_blob_path = f"results/{submission_id}/model/policy.zip"
    onnx_blob_path = f"results/{submission_id}/model/policy.onnx"
    sentis_blob_path = f"results/{submission_id}/model/policy.sentis.onnx"

    upload_file(
        bucket=bucket,
        local_path=local_zip_path,
        blob_path=zip_blob_path,
        content_type="application/zip",
    )
    upload_file(
        bucket=bucket,
        local_path=local_onnx_path,
        blob_path=onnx_blob_path,
        content_type="application/octet-stream",
    )
    upload_file(
        bucket=bucket,
        local_path=local_sentis_path,
        blob_path=sentis_blob_path,
        content_type="application/octet-stream",
    )

    replay_artifact = (
        upload_replay_bundle_to_gcs(
            bucket_name=bucket_name,
            submission_id=submission_id,
            replay_bundle_dir=replay_bundle_dir,
        )
        if replay_bundle_dir is not None
        else {}
    )
    return {
        "model": {
            "storage": "gcs",
            "bucket": bucket_name,
            "path": zip_blob_path,
        },
        "onnx_model": {
            "storage": "gcs",
            "bucket": bucket_name,
            "path": onnx_blob_path,
        },
        "sentis_model": _sentis_metadata(
            bucket_name=bucket_name,
            path=sentis_blob_path,
        ),
        **replay_artifact,
    }
