# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Python environment

Use `uv` for all project and package management. Never use `pip` or the system
Python directly.

- Run Python via `uv run python` or through the `.venv` managed by uv
- Install packages with `uv add` / `uv sync`, not `pip install`
- The virtual environment is always `.venv/` in the project root
- Runtime dependencies are split into `embodiedlab`, `server`, `trainer`, and
  `notification` groups; developer tooling lives in `dev`
- For full local work, use `uv sync --frozen --all-groups`
- For service-scoped work, sync `embodiedlab` plus the relevant service group

## Expectations

- Before reporting a task as complete, run the relevant command and show the
  actual output.
- If a piece of information has not been verified by running a command or
  reading a file, explicitly state that it is unverified.

## Ignore list

These directories contain generated or tool-specific files and should not be read or modified:

- `.venv/` — Python virtual environment
- `.git/` — Git internals
- `.github/` — CI workflow definitions (read-only reference)
- `node_modules/` — npm-installed tooling dependencies
- `.vscode/` — Editor settings
- `.pytest_cache/` — pytest cache
- `.ruff_cache/` — ruff lint cache
- `.claude/` — Claude Code settings

## Commands

See `rules/command.md`.

For a human-oriented development entry point, see
`docs/development.md`.

## Architecture

EmbodiedLab is a three-service platform for embodied AI research: clients submit
grid-world training configurations, a Cloud Run Job trains a PPO policy, and
results are streamed back in real time.

### Services

**`server/`** — FastAPI Cloud Run Service (API)

- Accepts `POST /submissions` (saves to Firestore `submissions/{id}`)
- `POST /submissions/{id}/train` creates a `results/{id}` doc at `queued`,
  then starts the Cloud Run Job with `SUBMISSION_ID` injected as an env var
- `GET /results/{id}` reads from Firestore `results/{id}`
- Config is loaded lazily from env vars and fails fast if any required value is
  missing: `DB_ID`, `REGION`, `PROJECT_ID`, `TRAINER_JOB_NAME`
- `dependencies.py` exposes cached providers for `ServerConfig` and the
  Firestore client

**`trainer/`** — Cloud Run Job

- Entry point: `trainer/main.py` → `trainer/job.py:run_training_job()`
- Fetches submission from Firestore, calls `embodiedlab/training/runner.py`,
  uploads the saved model artifacts to GCS, writes the result back to
  Firestore, and publishes the matching Pub/Sub event for each state transition
- Model artifacts include the Stable-Baselines3 zip (`policy.zip`), the general
  ONNX export (`policy.onnx`), and a Unity Sentis-oriented ONNX export
  (`policy.sentis.onnx`) under `models/{submission_id}/`
- `trainer/job.py` keeps Firestore updates and Pub/Sub publishing aligned via
  `write_result_update()`
- Required env vars: `DB_ID`, `MODEL_BUCKET`, `SUBMISSION_ID`, `PUBSUB_TOPIC`,
  `PROJECT_ID`

**`notification/`** — FastAPI Cloud Run Service (WebSocket relay)

- `GET /ws/results/{submission_id}` — WebSocket endpoint; clients subscribe here
- On connect, the WebSocket sends `{ "type": "connected", "submission_id": ... }`
- `POST /internal/pubsub/push` — Pub/Sub push endpoint; fans events out to subscribed WebSocket clients
- Invalid push payloads return `400` (`Invalid Pub/Sub message`, `Missing data`,
  `Invalid encoded event`, or `Missing submission_id`)
- Deployed as a single instance with `--min-instances 1`; connection state is in-process memory

### Shared library (`embodiedlab/`)

| Module | Purpose |
| --- | --- |
| `schemas.py` | Pydantic request/response models (`SubmitRequest`, `SubmissionDocument`, etc.) |
| `result_models.py` | `ResultStatus` enum, `Progress`/`ResultDocument`/`ResultMessage` models, builder helpers |
| `gridworld_env.py` | Gymnasium-compatible `GridWorldTrainingEnv` |
| `training/training_models.py` | `GridWorldSpec`, `GridPosition` |
| `training/training_config.py` | `TrainingConfig` (PPO hyperparameters, defaults) |
| `training/training_converter.py` | Converts `SubmitRequest`/Firestore dict → `GridWorldSpec` |
| `training/runner.py` | `run_gridworld_training()` — trains via Stable-Baselines3 PPO, evaluates, saves model |

### Data flow

```text
Client
  → POST /submissions           (API writes Firestore submissions/{id})
  → POST /submissions/{id}/train
      → API writes Firestore results/{id} = queued
      → API triggers Cloud Run Job (SUBMISSION_ID env var)
          → Job reads Firestore submissions/{id}
          → Job trains PPO policy (embodiedlab/training/runner.py)
          → Job saves model artifacts to GCS:
              gs://{MODEL_BUCKET}/models/{id}/policy.zip
              gs://{MODEL_BUCKET}/models/{id}/policy.onnx
              gs://{MODEL_BUCKET}/models/{id}/policy.sentis.onnx
          → Job writes Firestore results/{id} = completed
          → Job publishes Pub/Sub event (ordered, key = submission_id)
              → Pub/Sub push → notification /internal/pubsub/push
                  → Fan-out to WebSocket subscribers /ws/results/{id}
```

### Testing approach

Tests live in `tests/`. Integration with GCP is avoided — `tests/fakes.py`
provides Firestore-style fakes and repository-oriented fakes. The server app
overrides FastAPI dependencies (`app.dependency_overrides`) to inject fake
repositories and a hardcoded `ServerConfig`. Trainer job tests inject fake
repository factories plus `train_model` / `upload_model` callables.
Notification tests use `fastapi.testclient.TestClient` to validate Pub/Sub push
parsing and WebSocket fan-out.

Verified locally after the current refactor:

- `uv run pytest` → `36 passed`
- `uv run ruff check embodiedlab server trainer tests notification` → `All checks passed!`

## Configuration

All config is loaded from environment variables. In development, `Makefile`
reads `.env` and exports everything. Required vars per service:

| Service | Required env vars |
| --- | --- |
| API | `DB_ID`, `REGION`, `PROJECT_ID`, `TRAINER_JOB_NAME` |
| Trainer | `DB_ID`, `MODEL_BUCKET`, `SUBMISSION_ID`, `PUBSUB_TOPIC`, `PROJECT_ID` |
| Notification deployment (Makefile) | `NOTIFICATION_SERVICE_NAME`, `NOTIFICATION_PUSH_PATH` |
| `tools/ws_client.py` via Makefile | `NOTIFICATION_SERVICE_NAME`, `HASH`, `REGION`, `SUBMISSION_ID` |

`Makefile` creates `MODEL_BUCKET` with uniform bucket-level access and grants
`allUsers` `roles/storage.objectViewer`, so completed model artifacts are public
read by default.

Recommended `uv` sync patterns:

- full local environment: `uv sync --frozen --all-groups`
- API only: `uv sync --frozen --group embodiedlab --group server`
- trainer only: `uv sync --frozen --group embodiedlab --group trainer`
- notification only: `uv sync --frozen --group embodiedlab --group notification`

## Code style

Read `rules/coding-style.md`.
