from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces

from embodiedlab.continuous_navigation_env import (
    IMAGE_OBSERVATION_CHANNELS,
    IMAGE_OBSERVATION_HEIGHT,
    IMAGE_OBSERVATION_WIDTH,
    NUMERIC_OBSERVATION_SIZE,
)
from embodiedlab.result_models import ReplayAction, ReplayLogStep, ReplayReward
from trainer import artifacts

IMAGE_OBSERVATION_SIZE = (
    IMAGE_OBSERVATION_CHANNELS * IMAGE_OBSERVATION_HEIGHT * IMAGE_OBSERVATION_WIDTH
)
SENTIS_OBSERVATION_SIZE = IMAGE_OBSERVATION_SIZE + NUMERIC_OBSERVATION_SIZE


def fake_onnx_export(local_model_base_path):
    return f"{local_model_base_path}.onnx"


def fake_sentis_export(local_model_base_path):
    return f"{local_model_base_path}.sentis.onnx"


class FakeBlob:
    def __init__(self, path):
        self.path = path
        self.uploads = []

    def upload_from_filename(self, local_path, content_type=None):
        upload = {
            "local_path": local_path,
            "content_type": content_type,
        }
        if content_type == "application/jsonl":
            with Path(local_path).open(encoding="utf-8") as uploaded_file:
                upload["contents"] = uploaded_file.read()
        self.uploads.append(upload)


class FakeBucket:
    def __init__(self):
        self.blobs = {}

    def blob(self, path):
        blob = FakeBlob(path)
        self.blobs[path] = blob
        return blob


class FakeStorageClient:
    def __init__(self, bucket):
        self._bucket = bucket

    def bucket(self, bucket_name):
        return self._bucket


class ConstantActionNet(torch.nn.Module):
    def __init__(self, action):
        super().__init__()
        self.action = torch.as_tensor(action, dtype=torch.float32)

    def forward(self, latent):
        return self.action.expand(latent.shape[0], -1)


class FakeMlpExtractor(torch.nn.Module):
    def forward(self, features):
        return features, features


class FakeDistribution:
    def __init__(self, action):
        self.action = action

    def get_actions(self, *, deterministic):
        assert deterministic is True
        return self.action


class FakePolicy(torch.nn.Module):
    def __init__(self, action):
        super().__init__()
        self.squash_output = False
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.share_features_extractor = True
        self.mlp_extractor = FakeMlpExtractor()
        self.action_net = ConstantActionNet(action)
        self.action = torch.as_tensor(action, dtype=torch.float32)

    def extract_features(self, obs):
        return torch.zeros((obs["obs_1"].shape[0], 256), dtype=torch.float32)

    def get_distribution(self, obs):
        return FakeDistribution(self.action.expand(obs["obs_1"].shape[0], -1))


def test_onnxable_policy_applies_navigation_final_action_mapping():
    policy = FakePolicy([[-100.0, 4.5]])
    onnxable = artifacts.OnnxableContinuousNavigationPolicy(policy)

    action = onnxable(
        torch.zeros((1, 3, 84, 112), dtype=torch.float32),
        torch.zeros((1, 2), dtype=torch.float32),
    )

    assert torch.allclose(
        action,
        torch.tensor([[0.0, 1.0]], dtype=torch.float32),
    )


def test_onnxable_policy_maps_zero_raw_action_to_half_forward():
    policy = FakePolicy([[0.0, 0.0]])
    onnxable = artifacts.OnnxableContinuousNavigationPolicy(policy)

    action = onnxable(
        torch.zeros((1, 3, 84, 112), dtype=torch.float32),
        torch.zeros((1, 2), dtype=torch.float32),
    )

    assert torch.allclose(action, torch.tensor([[0.5, 0.0]], dtype=torch.float32))


def test_sentis_policy_accepts_flattened_environment_observation():
    policy = FakePolicy([[0.0, 0.0]])
    sentis_policy = artifacts.SentisContinuousNavigationPolicy(policy)

    action = sentis_policy(
        torch.zeros((1, SENTIS_OBSERVATION_SIZE), dtype=torch.float32),
    )

    assert torch.allclose(action, torch.tensor([[0.5, 0.0]], dtype=torch.float32))


def test_upload_model_to_gcs_uploads_zip_onnx_and_sentis(monkeypatch):
    bucket = FakeBucket()
    model_base_path = "policy"
    monkeypatch.setattr(
        artifacts.storage,
        "Client",
        lambda: FakeStorageClient(bucket),
    )
    monkeypatch.setattr(
        artifacts,
        "export_model_to_onnx",
        fake_onnx_export,
    )
    monkeypatch.setattr(
        artifacts,
        "export_model_to_sentis_onnx",
        fake_sentis_export,
    )

    result = artifacts.upload_model_to_gcs(
        local_model_base_path=model_base_path,
        bucket_name="model-bucket",
        submission_id="submission-1",
    )

    assert result == {
        "model": {
            "storage": "gcs",
            "bucket": "model-bucket",
            "path": "results/submission-1/model/policy.zip",
        },
        "onnx_model": {
            "storage": "gcs",
            "bucket": "model-bucket",
            "path": "results/submission-1/model/policy.onnx",
        },
        "replay_log": {
            "storage": "gcs",
            "bucket": "model-bucket",
            "path": "results/submission-1/replay/replay.jsonl",
            "format": "jsonl",
            "schema_version": "replay-log.v0",
        },
        "sentis_model": {
            "storage": "gcs",
            "bucket": "model-bucket",
            "path": "results/submission-1/model/policy.sentis.onnx",
            "format": "onnx",
            "target": "unity-sentis",
            "opset_version": 15,
            "input": {
                "name": "observation",
                "shape": [1, SENTIS_OBSERVATION_SIZE],
                "dtype": "float32",
                "layout": [
                    "obs_0_chw_3x84x112",
                    "obs_1_angle_degrees",
                    "obs_1_distance_meters",
                ],
            },
            "output": {
                "name": "action",
                "layout": ["forward", "turn"],
                "action_mapping": {
                    "forward": "(policy_forward + 1) / 2",
                    "turn": "policy_turn",
                },
            },
        },
    }
    assert bucket.blobs["results/submission-1/model/policy.zip"].uploads == [
        {
            "local_path": "policy.zip",
            "content_type": "application/zip",
        },
    ]
    assert bucket.blobs["results/submission-1/model/policy.onnx"].uploads == [
        {
            "local_path": "policy.onnx",
            "content_type": "application/octet-stream",
        },
    ]
    assert bucket.blobs["results/submission-1/model/policy.sentis.onnx"].uploads == [
        {
            "local_path": "policy.sentis.onnx",
            "content_type": "application/octet-stream",
        },
    ]
    replay_upload = bucket.blobs["results/submission-1/replay/replay.jsonl"].uploads[0]
    assert replay_upload["content_type"] == "application/jsonl"
    assert replay_upload["contents"] == ""


def test_upload_replay_log_to_gcs_uploads_jsonl_metadata(monkeypatch):
    bucket = FakeBucket()
    monkeypatch.setattr(
        artifacts.storage,
        "Client",
        lambda: FakeStorageClient(bucket),
    )
    replay_steps = [
        ReplayLogStep(
            scenario_id="scenario_demo_001",
            job_id="submission-1",
            episode_id="episode_0001",
            step_index=0,
            time_seconds=0.0,
            robot={
                "position": {"x": 1.0, "z": 1.0},
                "rotation_y_degrees": 0.0,
            },
            action=ReplayAction(
                values=[
                    {"name": "forward", "value": 0.0},
                    {"name": "turn", "value": 0.0},
                ],
            ),
            reward=ReplayReward(total=0.0),
        ),
    ]

    result = artifacts.upload_replay_log_to_gcs(
        bucket_name="model-bucket",
        submission_id="submission-1",
        replay_steps=replay_steps,
    )

    assert result == {
        "replay_log": {
            "storage": "gcs",
            "bucket": "model-bucket",
            "path": "results/submission-1/replay/replay.jsonl",
            "format": "jsonl",
            "schema_version": "replay-log.v0",
        },
    }
    upload = bucket.blobs["results/submission-1/replay/replay.jsonl"].uploads[0]
    assert upload["content_type"] == "application/jsonl"
    assert '"schema_version":"replay-log.v0"' in upload["contents"]
    assert upload["contents"].endswith("\n")
