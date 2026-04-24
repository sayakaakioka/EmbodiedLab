# EmbodiedLab

EmbodiedLab is an experimental platform for embodied AI research. The current
prototype accepts grid-world training requests through a Cloud Run API, starts a
Cloud Run Job to train a reinforcement learning policy, stores model artifacts
in GCS, and streams status updates to clients over WebSockets.

The project is intentionally small right now: it focuses on a minimal
end-to-end loop from environment definition to training, artifact storage, and
result streaming.

## Architecture

```text
Client
  -> POST /submissions
      -> Firestore submissions/{submission_id}
  -> POST /submissions/{submission_id}/train
      -> Firestore results/{submission_id} = queued
      -> Cloud Run Job with SUBMISSION_ID
          -> Firestore submission lookup
          -> GridWorld PPO training
          -> GCS model upload
          -> Firestore result update
          -> Pub/Sub event
              -> notification service push endpoint
                  -> WebSocket subscribers
  -> GET /results/{submission_id}
  -> WebSocket /ws/results/{submission_id}
```

## Services

```text
embodiedlab/
  Shared domain models, request schemas, result models, GridWorld environment,
  and training logic.

server/
  FastAPI API service for accepting submissions, starting training jobs, and
  serving result documents.

trainer/
  Cloud Run Job for loading a submission, running PPO training, uploading
  artifacts, updating Firestore, and publishing Pub/Sub events.

notification/
  FastAPI WebSocket relay service for fan-out of Pub/Sub push events to clients.

tests/
  Pytest coverage for schemas, conversion, API routes, result models, progress
  helpers, trainer job flow, and notification delivery.
```

## Verified Status

The repository currently has automated coverage for:

- shared schemas and result models
- API submission, training trigger, and result lookup routes
- trainer job state transitions, failure handling, and artifact flow
- notification Pub/Sub push validation and WebSocket fan-out

Latest locally verified commands:

```bash
uv run pytest
uv run ruff check embodiedlab server trainer tests notification
```

Detailed payload and config reference:

- [docs/data-models.md](/workspaces/EmbodiedLab/docs/data-models.md)
- [docs/development.md](/workspaces/EmbodiedLab/docs/development.md)

## Dependency Groups

Python dependencies are managed with `uv` dependency groups instead of one
flat runtime set.

- `embodiedlab`: shared models and utilities used across services
- `server`: FastAPI API and Cloud Run trigger dependencies
- `trainer`: PPO training and GCP artifact/event dependencies
- `notification`: WebSocket relay and Pub/Sub push relay dependencies
- `dev`: test and lint tooling

Common install patterns:

```bash
uv sync --frozen --all-groups
uv sync --frozen --group embodiedlab --group server
uv sync --frozen --group embodiedlab --group trainer
uv sync --frozen --group embodiedlab --group notification
```

## API

### Create Submission

```http
POST /submissions
Content-Type: application/json
```

Example payload:

```json
{
  "environment": {
    "size": [4, 4],
    "obstacles": [{"x": 1, "y": 1}],
    "goal": {"x": 3, "y": 3},
    "robot_start": {"x": 0, "y": 0}
  },
  "robot": {
    "type": "simple"
  },
  "training": {
    "algorithm": "ppo",
    "timesteps": 5000
  }
}
```

Response:

```json
{
  "status": "accepted",
  "submission_id": "..."
}
```

### Start Training

```http
POST /submissions/{submission_id}/train
```

This creates or replaces `results/{submission_id}` with `queued` status, then
starts the configured Cloud Run Job with `SUBMISSION_ID`.

### Get Result

```http
GET /results/{submission_id}
```

Result documents include:

- `status`: `queued`, `starting`, `running`, `completed`, or `failed`
- `progress`: phase, current step, total steps, and message
- `summary`: training and evaluation summary when completed
- `artifacts`: GCS model location when completed
- `error`: failure detail when failed

### Stream Result Updates

```http
GET /ws/results/{submission_id}
```

Clients can subscribe to live status updates through the notification service.
The notification service also sends an initial connection message:

```json
{
  "type": "connected",
  "submission_id": "..."
}
```

## Configuration

The `Makefile` includes `.env` and exports its variables.

Required API variables:

- `DB_ID`
- `REGION`
- `PROJECT_ID`
- `TRAINER_JOB_NAME`

Required trainer variables:

- `DB_ID`
- `MODEL_BUCKET`
- `SUBMISSION_ID`
- `PUBSUB_TOPIC`
- `PROJECT_ID`

Required notification deployment variables used by the Makefile:

- `NOTIFICATION_SERVICE_NAME`
- `NOTIFICATION_PUSH_PATH`

Common deployment and utility variables used by the Make targets include:

- `PROJECT_ID`
- `REGION`
- `ARTIFACT_REPO`
- `MODEL_BUCKET`
- `PUBSUB_TOPIC`
- `PUBSUB_SUBSCRIPTION`
- `RUNTIME_SA_NAME`
- `API_SERVICE_NAME`
- `TRAINER_JOB_NAME`
- `NOTIFICATION_SERVICE_NAME`
- `NOTIFICATION_PUSH_PATH`
- `API_URL`

## Development

Install the virtual environment and all dependency groups:

```bash
make local_setup
```

If you only need one service locally, you can also sync a narrower set:

```bash
uv sync --frozen --group embodiedlab --group server
uv sync --frozen --group embodiedlab --group trainer
uv sync --frozen --group embodiedlab --group notification
```

Run the API locally:

```bash
make server_local
```

Run tests:

```bash
make local_test
```

Run lint checks:

```bash
uv run ruff check embodiedlab server trainer tests notification
```

You can also run individual pytest commands with `uv run pytest ...`.

## Deployment

Bootstrap the required GCP resources:

```bash
make gcp_bootstrap
```

Build and deploy all services:

```bash
make deploy_all
```

You can also deploy services individually:

```bash
make deploy_api
make deploy_trainer
make deploy_notification
```

Useful operational commands:

```bash
make show_env
make show_notification_url
make recreate_pubsub_push
make logs_api
make logs_trainer
make logs_notification
```

## Manual Flow

Submit the sample payload:

```bash
make submit
```

Start training for the last submission:

```bash
make train
```

Fetch the last result:

```bash
make get_result
```

Watch status updates over WebSocket:

```bash
make get_result_ws
```

## Current Scope

The current implementation supports:

- Grid-world environment definitions
- A simple robot descriptor
- PPO training through Stable-Baselines3
- Firestore-backed submissions and results
- GCS model artifact upload
- Pub/Sub-backed result notifications
- Cloud Run API, Cloud Run Job, and WebSocket relay deployment

## Non-Goals For Now

- High-fidelity simulation
- User-imported robot models
- Multiple training algorithms in production
- A browser UI
- Production-grade authentication and quota handling

## License

To be decided.
