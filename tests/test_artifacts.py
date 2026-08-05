import copy
import gzip
import hashlib
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from google.api_core.exceptions import PreconditionFailed
from gymnasium import spaces

from embodiedlab.artifact_integrity import compute_file_integrity
from embodiedlab.result_models import (
    EvalReplayBundleChunk,
    ReplayBundleManifest,
    ReplayLogStep,
)
from embodiedlab.schemas import semantic_channel_layout
from tests.fakes import scenario_bundle
from trainer import artifacts

SCENARIO = scenario_bundle()
CAMERA = SCENARIO.sensors[0]
GOAL_VECTOR = SCENARIO.sensors[1]
IMAGE_SHAPE = (
    len(semantic_channel_layout(CAMERA.semantic_mode)),
    CAMERA.height,
    CAMERA.width,
)


def fake_onnx_export(local_model_base_path, scenario):
    assert scenario == SCENARIO
    path = Path(f"{local_model_base_path}.onnx")
    path.write_bytes(b"onnx")
    return str(path)


def _write_replay_fixture(
    replay_dir: Path,
    *,
    submission_id: str = "submission-1",
) -> tuple[Path, Path]:
    replay_dir.mkdir()
    chunk_dir = replay_dir / "eval"
    chunk_dir.mkdir()
    chunk_path = chunk_dir / "checkpoint_000001.jsonl.gz"
    replay_row = {
        "schema_version": "replay-log.v0",
        "scenario_id": SCENARIO.scenario_id,
        "job_id": submission_id,
        "phase": "eval",
        "checkpoint_step": 1,
        "env_index": 0,
        "policy_mode": "deterministic",
        "episode_id": "eval_env_00_episode_000001",
        "step_index": 0,
        "time_seconds": 0.0,
        "robot": {
            "position": {"x": 1.0, "z": 1.0},
            "rotation_y_degrees": 0.0,
        },
        "action": {
            "values": [
                {"name": "forward", "value": 0.0},
                {"name": "turn", "value": 0.0},
            ],
        },
        "reward": {"total": 0.0, "components": []},
        "events": [],
        "sensors": [],
        "terminated": False,
        "termination_reason": None,
    }
    serialized_row = ReplayLogStep.model_validate(replay_row).model_dump_json() + "\n"
    chunk_path.write_bytes(gzip.compress(serialized_row.encode("utf-8"), mtime=0))
    chunk_integrity = compute_file_integrity(chunk_path)
    manifest_path = replay_dir / "manifest.json"
    manifest_path.write_text(
        ReplayBundleManifest(
            schema_version="replay-bundle.v0",
            job_id=submission_id,
            scenario_id=SCENARIO.scenario_id,
            total_timesteps=1,
            chunks=[
                EvalReplayBundleChunk(
                    phase="eval",
                    policy_mode="deterministic",
                    checkpoint_step=1,
                    start_step=None,
                    end_step=None,
                    path="eval/checkpoint_000001.jsonl.gz",
                    format="jsonl.gz",
                    size_bytes=chunk_integrity.size_bytes,
                    sha256=chunk_integrity.sha256,
                    step_count=1,
                    episode_count=1,
                    success_rate=0.0,
                    avg_reward=0.0,
                    avg_steps=1.0,
                ),
            ],
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    return manifest_path, chunk_path


class FakeBlob:
    def __init__(self, path):
        self.path = path
        self.uploads = []

    def upload_from_filename(
        self,
        local_path,
        content_type=None,
        if_generation_match=None,
    ):
        upload = {
            "local_path": local_path,
            "content_type": content_type,
            "if_generation_match": if_generation_match,
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


class ExistingBlob(FakeBlob):
    def __init__(self, path, *, remote_size, remote_metadata):
        super().__init__(path)
        self.remote_size = remote_size
        self.remote_metadata = remote_metadata
        self.size = None
        self.metadata = None

    def upload_from_filename(self, *args, **kwargs):
        raise PreconditionFailed("already exists")  # noqa: TRY003

    def reload(self):
        self.size = self.remote_size
        self.metadata = self.remote_metadata


class ExistingBucket:
    def __init__(self, blob):
        self._blob = blob

    def blob(self, path):
        assert path == self._blob.path
        return self._blob


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
        self.observation_space = spaces.Dict(
            {
                CAMERA.observation_name: spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=IMAGE_SHAPE,
                    dtype=np.float32,
                ),
                GOAL_VECTOR.observation_name: spaces.Box(
                    low=np.array([-180.0, 0.0], dtype=np.float32),
                    high=np.array([180.0, 20.0], dtype=np.float32),
                    dtype=np.float32,
                ),
            },
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
    onnxable = artifacts.OnnxableContinuousNavigationPolicy(
        policy,
        image_observation_name=CAMERA.observation_name,
        goal_vector_observation_name=GOAL_VECTOR.observation_name,
    )

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
    onnxable = artifacts.OnnxableContinuousNavigationPolicy(
        policy,
        image_observation_name=CAMERA.observation_name,
        goal_vector_observation_name=GOAL_VECTOR.observation_name,
    )

    action = onnxable(
        torch.zeros((1, 3, 84, 112), dtype=torch.float32),
        torch.zeros((1, 2), dtype=torch.float32),
    )

    assert torch.allclose(action, torch.tensor([[0.5, 0.0]], dtype=torch.float32))


def test_pytorch_leafspec_export_workaround_is_still_needed():
    # PyTorch's Dynamo ONNX exporter deep-copies the same LeafSpec structure.
    _leaves, leaf_spec = torch.utils._pytree.tree_flatten(0)  # noqa: SLF001

    with pytest.warns(
        FutureWarning,
        match=artifacts._PYTORCH_LEAFSPEC_WARNING_PATTERN,  # noqa: SLF001
    ):
        copy.deepcopy(leaf_spec)


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
    replay_dir = tmp_path / "replay_bundle"
    _write_replay_fixture(replay_dir, submission_id="submission")
    metadata = artifacts.upload_model_to_gcs(
        local_model_base_path=model_path,
        bucket_name="bucket",
        submission_id="submission",
        scenario=SCENARIO,
        replay_bundle_dir=str(replay_dir),
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


def test_export_rejects_policy_observation_shape_mismatch(monkeypatch, tmp_path):
    policy = FakePolicy([[0.0, 0.0]])
    policy.observation_space = spaces.Dict(
        {
            CAMERA.observation_name: spaces.Box(
                low=0.0,
                high=1.0,
                shape=(3, 64, 64),
                dtype=np.float32,
            ),
            GOAL_VECTOR.observation_name: policy.observation_space.spaces[
                GOAL_VECTOR.observation_name
            ],
        },
    )
    model = type("Model", (), {"policy": policy})()
    monkeypatch.setattr(artifacts.PPO, "load", lambda _: model)

    with pytest.raises(ValueError, match="does not match the scenario"):
        artifacts.export_model_to_onnx(str(tmp_path / "policy"), SCENARIO)


def test_onnx_validation_rejects_an_undeclared_non_float_input(tmp_path):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Identity", ["obs_0"], ["action"])],
        "unexpected-input",
        [
            onnx.helper.make_tensor_value_info(
                "obs_0",
                onnx.TensorProto.FLOAT,
                [None, 2],
            ),
            onnx.helper.make_tensor_value_info(
                "undeclared",
                onnx.TensorProto.INT64,
                [None, 1],
            ),
        ],
        [
            onnx.helper.make_tensor_value_info(
                "action",
                onnx.TensorProto.FLOAT,
                [None, 2],
            ),
        ],
    )
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_opsetid("", 18)],
    )
    model_path = tmp_path / "unexpected-input.onnx"
    onnx.save(model, model_path)

    with pytest.raises(ValueError, match="inputs do not match contract"):
        artifacts._validate_exported_onnx(  # noqa: SLF001
            str(model_path),
            expected_opset=18,
            expected_inputs={"obs_0": [-1, 2]},
            expected_output_shape=[-1, 2],
        )


def test_upload_model_to_gcs_uploads_onnx_with_integrity(
    monkeypatch,
    tmp_path,
):
    bucket = FakeBucket()
    model_base_path = str(tmp_path / "policy")
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
    replay_dir = tmp_path / "replay_bundle"
    manifest_path, _chunk_path = _write_replay_fixture(replay_dir)
    manifest_integrity = compute_file_integrity(manifest_path)

    result = artifacts.upload_model_to_gcs(
        local_model_base_path=model_base_path,
        bucket_name="model-bucket",
        submission_id="submission-1",
        scenario=SCENARIO,
        replay_bundle_dir=str(replay_dir),
    )

    assert result == {
        "onnx_model": {
            "storage": "gcs",
            "bucket": "model-bucket",
            "path": "results/submission-1/model/policy.onnx",
            "format": "onnx",
            "target": "onnx-runtime",
            "opset_version": 18,
            "size_bytes": 4,
            "sha256": hashlib.sha256(b"onnx").hexdigest(),
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
        "replay_bundle": {
            "storage": "gcs",
            "bucket": "model-bucket",
            "path": "results/submission-1/replay/manifest.json",
            "format": "json",
            "size_bytes": manifest_integrity.size_bytes,
            "sha256": manifest_integrity.sha256,
        },
    }
    assert bucket.blobs["results/submission-1/model/policy.onnx"].uploads == [
        {
            "local_path": f"{model_base_path}.onnx",
            "content_type": "application/octet-stream",
            "if_generation_match": 0,
        },
    ]


def test_upload_file_accepts_an_identical_existing_object(tmp_path):
    local_path = tmp_path / "artifact.bin"
    local_path.write_bytes(b"same content")
    integrity = compute_file_integrity(local_path)
    blob = ExistingBlob(
        "results/job/artifact.bin",
        remote_size=integrity.size_bytes,
        remote_metadata={
            "sha256": integrity.sha256,
            "size_bytes": str(integrity.size_bytes),
        },
    )

    artifacts.upload_file(
        bucket=ExistingBucket(blob),
        local_path=str(local_path),
        blob_path=blob.path,
        content_type="application/octet-stream",
    )


def test_upload_file_rejects_a_different_existing_object(tmp_path):
    local_path = tmp_path / "artifact.bin"
    local_path.write_bytes(b"new content")
    blob = ExistingBlob(
        "results/job/artifact.bin",
        remote_size=99,
        remote_metadata={"sha256": "f" * 64, "size_bytes": "99"},
    )

    with pytest.raises(ValueError, match="different content"):
        artifacts.upload_file(
            bucket=ExistingBucket(blob),
            local_path=str(local_path),
            blob_path=blob.path,
            content_type="application/octet-stream",
        )


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
    manifest_path, _chunk_path = _write_replay_fixture(replay_dir)
    manifest_integrity = compute_file_integrity(manifest_path)

    result = artifacts.upload_replay_bundle_to_gcs(
        bucket_name="model-bucket",
        submission_id="submission-1",
        scenario_id=SCENARIO.scenario_id,
        replay_bundle_dir=str(replay_dir),
    )

    assert result == {
        "replay_bundle": {
            "storage": "gcs",
            "bucket": "model-bucket",
            "path": "results/submission-1/replay/manifest.json",
            "format": "json",
            "size_bytes": manifest_integrity.size_bytes,
            "sha256": manifest_integrity.sha256,
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
    assert manifest_upload["if_generation_match"] == 0
    assert chunk_upload["if_generation_match"] == 0


def test_upload_replay_bundle_rejects_chunk_integrity_mismatch(
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
    _manifest_path, chunk_path = _write_replay_fixture(replay_dir)
    chunk_path.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="integrity does not match"):
        artifacts.upload_replay_bundle_to_gcs(
            bucket_name="model-bucket",
            submission_id="submission-1",
            scenario_id=SCENARIO.scenario_id,
            replay_bundle_dir=str(replay_dir),
        )

    assert bucket.blobs == {}


def test_upload_replay_bundle_rejects_unexpected_file(monkeypatch, tmp_path):
    bucket = FakeBucket()
    monkeypatch.setattr(
        artifacts.storage,
        "Client",
        lambda: FakeStorageClient(bucket),
    )
    replay_dir = tmp_path / "replay_bundle"
    _write_replay_fixture(replay_dir)
    (replay_dir / "unexpected.txt").write_text("unexpected", encoding="utf-8")

    with pytest.raises(ValueError, match="must match the manifest"):
        artifacts.upload_replay_bundle_to_gcs(
            bucket_name="model-bucket",
            submission_id="submission-1",
            scenario_id=SCENARIO.scenario_id,
            replay_bundle_dir=str(replay_dir),
        )

    assert bucket.blobs == {}


def test_upload_replay_bundle_rejects_manifest_identity_mismatch(
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
    manifest_path, _chunk_path = _write_replay_fixture(replay_dir)
    manifest = ReplayBundleManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8"),
    )
    manifest_path.write_text(
        manifest.model_copy(update={"job_id": "another-job"}).model_dump_json(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="identity does not match"):
        artifacts.upload_replay_bundle_to_gcs(
            bucket_name="model-bucket",
            submission_id="submission-1",
            scenario_id=SCENARIO.scenario_id,
            replay_bundle_dir=str(replay_dir),
        )

    assert bucket.blobs == {}


def test_upload_replay_bundle_rejects_manifest_step_count_mismatch(
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
    manifest_path, _chunk_path = _write_replay_fixture(replay_dir)
    manifest = ReplayBundleManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8"),
    )
    changed_chunk = manifest.chunks[0].model_copy(update={"step_count": 2})
    manifest_path.write_text(
        manifest.model_copy(update={"chunks": [changed_chunk]}).model_dump_json(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="step_count does not match"):
        artifacts.upload_replay_bundle_to_gcs(
            bucket_name="model-bucket",
            submission_id="submission-1",
            scenario_id=SCENARIO.scenario_id,
            replay_bundle_dir=str(replay_dir),
        )

    assert bucket.blobs == {}


def test_upload_replay_bundle_rejects_manifest_episode_count_mismatch(
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
    manifest_path, _chunk_path = _write_replay_fixture(replay_dir)
    manifest = ReplayBundleManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8"),
    )
    changed_chunk = manifest.chunks[0].model_copy(update={"episode_count": 2})
    manifest_path.write_text(
        manifest.model_copy(update={"chunks": [changed_chunk]}).model_dump_json(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="episode_count does not match"):
        artifacts.upload_replay_bundle_to_gcs(
            bucket_name="model-bucket",
            submission_id="submission-1",
            scenario_id=SCENARIO.scenario_id,
            replay_bundle_dir=str(replay_dir),
        )

    assert bucket.blobs == {}


def test_upload_model_preflights_replay_before_any_gcs_write(monkeypatch, tmp_path):
    bucket = FakeBucket()
    monkeypatch.setattr(
        artifacts.storage,
        "Client",
        lambda: FakeStorageClient(bucket),
    )
    monkeypatch.setattr(artifacts, "export_model_to_onnx", fake_onnx_export)
    replay_dir = tmp_path / "replay_bundle"
    _manifest_path, chunk_path = _write_replay_fixture(replay_dir)
    chunk_path.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="integrity does not match"):
        artifacts.upload_model_to_gcs(
            local_model_base_path=str(tmp_path / "policy"),
            bucket_name="model-bucket",
            submission_id="submission-1",
            scenario=SCENARIO,
            replay_bundle_dir=str(replay_dir),
        )

    assert bucket.blobs == {}
