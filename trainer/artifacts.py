from __future__ import annotations

from google.cloud import storage


def upload_model_to_gcs(local_model_base_path: str, bucket_name: str, submission_id: str) -> dict:
	storage_client = storage.Client()
	bucket = storage_client.bucket(bucket_name)

	local_zip_path = f"{local_model_base_path}.zip"
	blob_path = f"models/{submission_id}/policy.zip"
	blob = bucket.blob(blob_path)
	blob.upload_from_filename(local_zip_path)

	return {
		"model": {
			"storage": "gcs",
			"bucket": bucket_name,
			"path": blob_path,
		}
	}
