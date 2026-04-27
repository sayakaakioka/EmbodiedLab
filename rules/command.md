# Commands

## Development

```bash
# Install dependencies (API + trainer + notification)
make local_setup

# Run tests
make local_test           # uv sync --frozen --all-groups, then pytest
uv run pytest tests/test_schemas.py   # single test file
uv run pytest tests/ -k test_name     # single test by name

# Lint / format (ruff)
uv run ruff check .
uv run ruff format .

# Lint / format (markdown)
npx markdownlint-cli2 --fix "**/*.md"

# Run API locally
make server_local         # uvicorn on port 8000
```

## Manual end-to-end flow

Requires deployed infra and `.env`.

```bash
make submit               # POST payload.json → /submissions, saves submission_id
make train                # POST /submissions/<id>/train
make get_result           # GET /results/<id>
make get_result_ws        # WebSocket stream via tools/ws_client.py
```

`make get_result_ws` exports `SUBMISSION_ID` from `.last_submission_id`.
`tools/ws_client.py` builds the WebSocket URL from `NOTIFICATION_SERVICE_NAME`,
`HASH`, `REGION`, and `SUBMISSION_ID`.

## Deploy

```bash
make deploy_all           # builds and pushes all three Docker images, deploys Cloud Run
make deploy_api
make deploy_trainer
make deploy_notification
```

## GCP Storage

```bash
make create_model_bucket  # creates MODEL_BUCKET and grants public object read
gcloud storage rm --recursive gs://$MODEL_BUCKET/models/**
```

The model bucket stores completed artifacts under `models/<submission_id>/`:
`policy.zip`, `policy.onnx`, and `policy.sentis.onnx`. Removing `models/**`
clears generated artifacts while keeping the bucket, IAM policy, and
public-read configuration.
