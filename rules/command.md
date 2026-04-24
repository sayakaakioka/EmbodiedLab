# Commands

## Development

```bash
# Install dependencies (API + trainer + notification)
make local_setup

# Run tests
make local_test           # installs requirements-dev.txt then runs pytest
.venv/bin/python -m pytest tests/test_schemas.py   # single test file
.venv/bin/python -m pytest tests/ -k test_name     # single test by name

# Lint / format (ruff)
.venv/bin/ruff check .
.venv/bin/ruff format .

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

## Deploy

```bash
make deploy_all           # builds and pushes all three Docker images, deploys Cloud Run
make deploy_api
make deploy_trainer
make deploy_notification
```
