from trainer import artifacts


class FakeBlob:
    def __init__(self, path):
        self.path = path
        self.uploads = []

    def upload_from_filename(self, local_path, content_type=None):
        self.uploads.append(
            {
                "local_path": local_path,
                "content_type": content_type,
            },
        )


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
        lambda local_model_base_path: f"{local_model_base_path}.onnx",
    )
    monkeypatch.setattr(
        artifacts,
        "export_model_to_sentis_onnx",
        lambda local_model_base_path: f"{local_model_base_path}.sentis.onnx",
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
            "path": "models/submission-1/policy.zip",
        },
        "onnx_model": {
            "storage": "gcs",
            "bucket": "model-bucket",
            "path": "models/submission-1/policy.onnx",
        },
        "sentis_model": {
            "storage": "gcs",
            "bucket": "model-bucket",
            "path": "models/submission-1/policy.sentis.onnx",
            "format": "onnx",
            "target": "unity-sentis",
            "opset_version": 15,
            "input": {
                "name": "observation",
                "shape": [1, 4],
                "dtype": "float32",
                "layout": ["robot_x", "robot_y", "goal_x", "goal_y"],
            },
            "output": {
                "name": "action_logits",
                "action_mapping": {
                    "0": "up",
                    "1": "right",
                    "2": "down",
                    "3": "left",
                },
            },
        },
    }
    assert bucket.blobs["models/submission-1/policy.zip"].uploads == [
        {
            "local_path": "policy.zip",
            "content_type": "application/zip",
        },
    ]
    assert bucket.blobs["models/submission-1/policy.onnx"].uploads == [
        {
            "local_path": "policy.onnx",
            "content_type": "application/octet-stream",
        },
    ]
    assert bucket.blobs["models/submission-1/policy.sentis.onnx"].uploads == [
        {
            "local_path": "policy.sentis.onnx",
            "content_type": "application/octet-stream",
        },
    ]
