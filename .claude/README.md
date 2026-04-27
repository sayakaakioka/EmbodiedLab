# Claude Code Notes

This directory stores Claude Code settings for the repository.

## Current Project Assumptions

- Use `uv` for Python commands and dependency management.
- `Makefile` includes `.env` and exports those variables before running local
  helper commands.
- `tools/ws_client.py` builds the notification WebSocket URL from
  `NOTIFICATION_SERVICE_NAME`, `HASH`, `REGION`, and `SUBMISSION_ID`.
- Trainer artifacts are written to `MODEL_BUCKET` under
  `models/<submission_id>/` as both `policy.zip` and `policy.onnx`.
- `make create_model_bucket` grants public object read on `MODEL_BUCKET`.

## Safety Notes

- Do not commit `.env` or credential files.
- Be careful with recursive GCS deletion commands. To clear generated model
  artifacts while preserving bucket configuration, target `models/**` rather
  than deleting the bucket.
