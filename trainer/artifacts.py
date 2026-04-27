"""GCS upload helper for trained model artifacts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from google.cloud import storage
from stable_baselines3 import PPO

if TYPE_CHECKING:
    from stable_baselines3.common.policies import BasePolicy


class OnnxableGridWorldPolicy(torch.nn.Module):
    """Small wrapper that exposes the GridWorld policy as ONNX-friendly inputs."""

    def __init__(self, policy: BasePolicy) -> None:
        """Store the trained Stable-Baselines3 policy."""
        super().__init__()
        self.policy = policy

    def forward(self, agent: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Return the deterministic action for an agent/goal observation batch."""
        actions, _values, _log_prob = self.policy(
            {"agent": agent, "goal": goal},
            deterministic=True,
        )
        return actions


def export_model_to_onnx(local_model_base_path: str) -> str:
    """Convert the saved Stable-Baselines3 policy zip to ONNX."""
    model = PPO.load(local_model_base_path)
    onnx_path = f"{local_model_base_path}.onnx"
    onnxable_policy = OnnxableGridWorldPolicy(model.policy)
    dummy_agent = torch.zeros((1, 2), dtype=torch.float32)
    dummy_goal = torch.zeros((1, 2), dtype=torch.float32)

    torch.onnx.export(
        onnxable_policy,
        (dummy_agent, dummy_goal),
        onnx_path,
        input_names=["agent", "goal"],
        output_names=["action"],
        dynamic_axes={
            "agent": {0: "batch"},
            "goal": {0: "batch"},
            "action": {0: "batch"},
        },
        opset_version=17,
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


def upload_model_to_gcs(
    local_model_base_path: str,
    bucket_name: str,
    submission_id: str,
) -> dict:
    """Upload the saved model zip and ONNX export to GCS."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    local_zip_path = f"{local_model_base_path}.zip"
    local_onnx_path = export_model_to_onnx(local_model_base_path)
    zip_blob_path = f"models/{submission_id}/policy.zip"
    onnx_blob_path = f"models/{submission_id}/policy.onnx"

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
    }
