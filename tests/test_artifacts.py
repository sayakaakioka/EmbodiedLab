from pathlib import Path

from embodiedlab.result_models import ReplayAction, ReplayLogStep, ReplayReward
from trainer import artifacts


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
