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


def test_upload_model_to_gcs_uploads_zip_and_onnx(monkeypatch):
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
