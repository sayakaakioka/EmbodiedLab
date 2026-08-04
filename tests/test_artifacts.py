from pathlib import Path

import numpy as np
import onnx
import torch
from gymnasium import spaces

from embodiedlab.continuous_navigation_env import (
    IMAGE_OBSERVATION_CHANNELS,
    IMAGE_OBSERVATION_HEIGHT,
    IMAGE_OBSERVATION_WIDTH,
    NUMERIC_OBSERVATION_SIZE,
)
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
            low=np.array([-8.0, -3.0], dtype=np.float32),
            high=np.array([8.0, 3.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.share_features_extractor = True
        self.mlp_extractor = FakeMlpExtractor()
        self.action_net = ConstantActionNet(action)
        self.action = torch.as_tensor(action, dtype=torch.float32)

    def extract_features(self, obs):
        return torch.zeros((obs["obs_1"].shape[0], 256), dtype=torch.float32)

    def get_distribution(self, obs):
        dependency = obs["obs_0"].mean(dim=(1, 2, 3)) + obs["obs_1"].mean(dim=1)
        action = self.action.expand(obs["obs_1"].shape[0], -1)
        return FakeDistribution(action + dependency.unsqueeze(1) * 1e-6)


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


def _onnx_shape(value_info):
    return [
        dimension.dim_value if dimension.HasField("dim_value") else -1
        for dimension in value_info.type.tensor_type.shape.dim
    ]


def test_exported_onnx_graphs_match_published_metadata(monkeypatch, tmp_path):
    model = type("Model", (), {"policy": FakePolicy([[0.0, 0.0]])})()
    monkeypatch.setattr(artifacts.PPO, "load", lambda _: model)
    bucket = FakeBucket()
    monkeypatch.setattr(
        artifacts.storage,
        "Client",
        lambda: FakeStorageClient(bucket),
    )
    model_path = str(tmp_path / "policy")
    metadata = artifacts.upload_model_to_gcs(
        local_model_base_path=model_path,
        bucket_name="bucket",
        submission_id="submission",
    )

    onnx_graph = onnx.load(f"{model_path}.onnx")
    onnx_metadata = metadata["onnx_model"]
    assert onnx_graph.opset_import[0].version == onnx_metadata["opset_version"]
    assert [value.name for value in onnx_graph.graph.input] == [
        item["name"] for item in onnx_metadata["inputs"]
    ]
    assert [_onnx_shape(value) for value in onnx_graph.graph.input] == [
        item["shape"] for item in onnx_metadata["inputs"]
    ]
    assert [value.name for value in onnx_graph.graph.output] == [
        onnx_metadata["output"]["name"]
    ]

    sentis_graph = onnx.load(f"{model_path}.sentis.onnx")
    sentis_metadata = metadata["sentis_model"]
    assert sentis_graph.opset_import[0].version == sentis_metadata["opset_version"]
    assert [value.name for value in sentis_graph.graph.input] == [
        item["name"] for item in sentis_metadata["inputs"]
    ]
    assert [_onnx_shape(value) for value in sentis_graph.graph.input] == [
        item["shape"] for item in sentis_metadata["inputs"]
    ]
    assert [value.name for value in sentis_graph.graph.output] == [
        sentis_metadata["output"]["name"]
    ]


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
            "format": "onnx",
            "target": "onnx-runtime",
            "opset_version": 17,
            "inputs": [
                {
                    "name": "obs_0",
                    "shape": [-1, 3, 84, 112],
                    "dtype": "float32",
                    "layout": [
                        "channel_0_unused",
                        "channel_1_traversable",
                        "channel_2_blocked_or_background",
                    ],
                },
                {
                    "name": "obs_1",
                    "shape": [-1, 2],
                    "dtype": "float32",
                    "layout": [
                        "goal_angle_degrees",
                        "goal_distance_meters",
                    ],
                },
            ],
            "output": {
                "name": "action",
                "layout": ["forward", "turn"],
                "action_mapping": {
                    "forward": "sigmoid(policy_forward)",
                    "turn": "clip(policy_turn, -3, 3) / 3",
                },
            },
        },
        "sentis_model": {
            "storage": "gcs",
            "bucket": "model-bucket",
            "path": "results/submission-1/model/policy.sentis.onnx",
            "format": "onnx",
            "target": "unity-sentis",
            "opset_version": 15,
            "inputs": [
                {
                    "name": "observation",
                    "shape": [1, SENTIS_OBSERVATION_SIZE],
                    "dtype": "float32",
                    "layout": [
                        "obs_0_chw_3x84x112",
                        "obs_1_angle_degrees",
                        "obs_1_distance_meters",
                    ],
                },
            ],
            "output": {
                "name": "action",
                "layout": ["forward", "turn"],
                "action_mapping": {
                    "forward": "sigmoid(policy_forward)",
                    "turn": "clip(policy_turn, -3, 3) / 3",
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


def test_upload_replay_bundle_to_gcs_uploads_manifest_and_chunks(
    monkeypatch,
    tmp_path,
):
    bucket = FakeBucket()
    monkeypatch.setattr(
        artifacts.storage,
        "Client",
        lambda: FakeStorageClient(bucket),
    )
    replay_dir = tmp_path / "replay_bundle"
    replay_dir.mkdir()
    (replay_dir / "manifest.json").write_text("{}", encoding="utf-8")
    chunk_dir = replay_dir / "eval"
    chunk_dir.mkdir()
    (chunk_dir / "checkpoint_000001.jsonl.gz").write_bytes(b"gzip-bytes")

    result = artifacts.upload_replay_bundle_to_gcs(
        bucket_name="model-bucket",
        submission_id="submission-1",
        replay_bundle_dir=str(replay_dir),
    )

    assert result == {
        "replay_bundle": {
            "storage": "gcs",
            "bucket": "model-bucket",
            "path": "results/submission-1/replay/manifest.json",
            "format": "json",
            "schema_version": "replay-bundle.v0",
        },
    }
    manifest_upload = bucket.blobs["results/submission-1/replay/manifest.json"].uploads[
        0
    ]
    chunk_upload = bucket.blobs[
        "results/submission-1/replay/eval/checkpoint_000001.jsonl.gz"
    ].uploads[0]
    assert manifest_upload["content_type"] == "application/json"
    assert chunk_upload["content_type"] == "application/gzip"
