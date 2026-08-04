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
make submit               # POST the canonical Scenario fixture, saves submission_id
make train                # POST /submissions/<id>/train
make get_result           # GET /results/<id>
make get_result_ws        # WebSocket stream via tools/ws_client.py
```

`make get_result_ws` exports `SUBMISSION_ID` from `.last_submission_id`.
`tools/ws_client.py` builds the WebSocket URL from `NOTIFICATION_SERVICE_NAME`,
`HASH`, `REGION`, and `SUBMISSION_ID`.

## Deploy

```bash
make deploy_all           # builds, pushes, and deploys all services
make deploy_api
make deploy_trainer
make deploy_notification
```

## GCP Storage

`make create_model_bucket` creates `MODEL_BUCKET` and grants public object
read for the current prototype. Completed artifacts are stored under
`results/<submission_id>/`.

Before deleting any cloud result, follow EnvForge's
[`cloud-result-retention.md`](https://github.com/sayakaakioka/EnvForge/blob/main/docs/implementation/cloud-result-retention.md)
and verify the submission, Firestore documents, GCS prefix, and Cloud Run
execution together.
