"""GCS upload helper for trained model artifacts."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import torch
from google.cloud import storage
from stable_baselines3 import PPO

from embodiedlab.result_models import ReplayLogStep, serialize_replay_log_jsonl

if TYPE_CHECKING:
    from collections.abc import Iterable

    from stable_baselines3.common.policies import BasePolicy


class OnnxableContinuousNavigationPolicy(torch.nn.Module):
    """Wrapper exposing the continuous policy as ONNX-friendly inputs."""

    def __init__(self, policy: BasePolicy) -> None:
        """Store the trained Stable-Baselines3 policy."""
        super().__init__()
        self.policy = policy

    def forward(
        self,
        robot: torch.Tensor,
        goal: torch.Tensor,
        front_distance: torch.Tensor,
    ) -> torch.Tensor:
        """Return deterministic continuous action values for a batch."""
        actions, _values, _log_prob = self.policy(
            {
                "robot": robot,
                "goal": goal,
                "front_distance": front_distance,
            },
            deterministic=True,
        )
        return actions


class SentisContinuousNavigationPolicy(torch.nn.Module):
    """Wrapper exposing a fixed continuous observation tensor for Sentis."""

    def __init__(self, policy: BasePolicy) -> None:
        """Store the trained Stable-Baselines3 policy."""
        super().__init__()
        self.policy = policy

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Return [forward, turn] actions for a compact observation tensor."""
        robot = observation[:, 0:3]
        goal = observation[:, 3:6]
        front_distance = observation[:, 6:7]
        actions, _values, _log_prob = self.policy(
            {
                "robot": robot,
                "goal": goal,
                "front_distance": front_distance,
            },
            deterministic=True,
        )
        return actions


def export_model_to_onnx(local_model_base_path: str) -> str:
    """Convert the saved Stable-Baselines3 continuous policy zip to ONNX."""
    model = PPO.load(local_model_base_path)
    onnx_path = f"{local_model_base_path}.onnx"
    onnxable_policy = OnnxableContinuousNavigationPolicy(model.policy)
    dummy_robot = torch.zeros((1, 3), dtype=torch.float32)
    dummy_goal = torch.zeros((1, 3), dtype=torch.float32)
    dummy_front_distance = torch.zeros((1, 1), dtype=torch.float32)
    torch.onnx.export(
        onnxable_policy,
        (dummy_robot, dummy_goal, dummy_front_distance),
        onnx_path,
        input_names=["robot", "goal", "front_distance"],
        output_names=["action"],
        dynamic_axes={
            "robot": {0: "batch"},
            "goal": {0: "batch"},
            "front_distance": {0: "batch"},
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
    dummy_observation = torch.zeros((1, 7), dtype=torch.float32)

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


def upload_replay_log_to_gcs(
    *,
    bucket_name: str,
    submission_id: str,
    replay_steps: Iterable[ReplayLogStep],
) -> dict:
    """Serialize replay steps as JSONL, upload them, and return artifact metadata."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob_path = f"results/{submission_id}/replay/replay.jsonl"

    with TemporaryDirectory() as tmpdir:
        local_path = Path(tmpdir) / "replay.jsonl"
        with local_path.open("w", encoding="utf-8") as replay_file:
            replay_file.write(serialize_replay_log_jsonl(replay_steps))
        upload_file(
            bucket=bucket,
            local_path=str(local_path),
            blob_path=blob_path,
            content_type="application/jsonl",
        )

    return {
        "replay_log": {
            "storage": "gcs",
            "bucket": bucket_name,
            "path": blob_path,
            "format": "jsonl",
            "schema_version": "replay-log.v0",
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
            "shape": [1, 7],
            "dtype": "float32",
            "layout": [
                "robot_x",
                "robot_z",
                "robot_rotation_y_degrees",
                "goal_x",
                "goal_z",
                "goal_radius",
                "front_distance",
            ],
        },
        "output": {
            "name": "action",
            "layout": ["forward", "turn"],
        },
    }


def upload_model_to_gcs(
    *,
    local_model_base_path: str,
    bucket_name: str,
    submission_id: str,
    replay_steps: Iterable[ReplayLogStep] = (),
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

    replay_artifact = upload_replay_log_to_gcs(
        bucket_name=bucket_name,
        submission_id=submission_id,
        replay_steps=replay_steps,
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
