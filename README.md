# EmbodiedLab

EmbodiedLab is an experimental platform for embodied AI research. The current
prototype accepts EnvForge Scenario Bundle submissions through a Cloud Run API,
starts a Cloud Run Job to train a reinforcement learning policy, stores ONNX,
Sentis, and Replay Bundle artifacts in GCS, and streams status updates to
clients over WebSockets.

The project is intentionally small right now: it focuses on a minimal
end-to-end loop from environment definition to training, artifact storage, and
result streaming.

## Documentation Notes

Markdown files under `docs/` are primarily agent-facing project notes. They
describe the current product direction, implementation, and operating rules.
Completed phase notes are removed when they no longer explain the active
system; Git history and pull requests retain the development history.

## Architecture

```text
Client
  -> POST /submissions with recovery headers
      -> Firestore submissions/{submission_id} with a hashed cancel capability
      -> identical retries resolve to the same submission
  -> POST /submissions/{submission_id}/train
      -> Firestore results/{submission_id} = queued
      -> Cloud Run Job with SUBMISSION_ID
          -> Firestore submission lookup
          -> Continuous navigation PPO training
          -> GCS model and Replay Bundle upload
          -> Firestore result update
          -> Pub/Sub event
              -> notification service push endpoint
                  -> WebSocket subscribers
  -> POST /submissions/{submission_id}/cancel
      -> cancel the exact stored Cloud Run Execution
      -> Pub/Sub cancelling/cancelled events
  -> GET /results/{submission_id}
  -> WebSocket /ws/results/{submission_id}
```

## Services

```text
embodiedlab/
  Shared domain models, request schemas, result models, continuous navigation
  environment, and training logic.

server/
  FastAPI API service for accepting submissions, starting or cancelling
  training jobs, and serving result documents.

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
- API submission, training trigger, cancellation, and result lookup routes
- trainer job state transitions, failure handling, and artifact flow
- notification Pub/Sub push validation and WebSocket fan-out

Latest locally verified commands:

```bash
uv run pytest
uv run ruff check embodiedlab server trainer tests notification
```

Detailed payload and config reference:

- [docs/implementation/data-models.md](docs/implementation/data-models.md)
- [docs/implementation/development.md](docs/implementation/development.md)

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
Idempotency-Key: {idempotency_key}
X-EmbodiedLab-Cancel-Token: {cancel_token}
```

Use the complete canonical payload at
[`tests/fixtures/envforge/navigation_default_scenario_bundle.json`](tests/fixtures/envforge/navigation_default_scenario_bundle.json).
The canonical fixture makes the world, robot, sensor, reward, and training
configuration explicit instead of relying on schema defaults.

The recovery headers are optional, but clients must send both or neither.
`make submit` generates and persists both values so an identical retry can
recover the same submission.

Response:

```json
{
  "status": "accepted",
  "submission_id": "...",
  "cancel_token": "example_cancel_token_0123456789abcdef"
}
```

The cancellation capability is returned only in this response. Persist it if
the client must be able to cancel the job after restarting. The server stores
only its SHA-256 digest.

### Start Training

```http
POST /submissions/{submission_id}/train
```

This creates or replaces `results/{submission_id}` with `queued` status, then
starts the configured Cloud Run Job with `SUBMISSION_ID`.
Creating a submission does not start training. This endpoint is currently a
separate operation and is not idempotent; retrying it may start another Cloud
Run execution.

### Cancel Training

```http
POST /submissions/{submission_id}/cancel
Authorization: Bearer {cancel_token}
```

Cancellation targets the exact Cloud Run Execution recorded when training was
started. The response is the latest result document and is idempotent after the
job reaches `cancelled`.

### Get Result

```http
GET /results/{submission_id}
```

Result documents include:

- `status`: `queued`, `starting`, `running`, `cancelling`, `cancelled`,
  `completed`, or `failed`
- `progress`: phase, current step, total steps, and message
- `summary`: training and evaluation summary when completed
- `result_bundle`: typed summary, artifact locations, compatibility metadata,
  and structured failure details
- `error`: failure detail when failed

Canonical artifacts exist only under `result_bundle.artifacts`:

- `onnx_model`: opset 17 model with `obs_0` and `obs_1` inputs
- `sentis_model`: opset 15 model with one fixed-length observation input
- `model`: compatibility alias for `policy.onnx`
- `replay_bundle`: manifest location for gzip JSONL train and evaluation chunks

See the complete canonical result at
[`tests/fixtures/envforge/navigation_completed_result_document.json`](tests/fixtures/envforge/navigation_completed_result_document.json).

### Stream Result Updates

```http
GET /ws/results/{submission_id}
```

Clients can subscribe to live status updates through the notification service.
The notification service sends an initial connection message, followed by the
latest Firestore Result Document when one exists:

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
- `PUBSUB_TOPIC`
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

To submit another complete Scenario Bundle, override the fixture explicitly:

```bash
make submit SUBMISSION_PAYLOAD=path/to/scenario.json
```

Start training for the last submission:

```bash
make train
```

This is currently a separate, non-idempotent operation. Do not retry it unless
starting another Cloud Run execution is intended.

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

- Continuous navigation Scenario Bundle definitions
- A simple robot descriptor
- PPO training through Stable-Baselines3
- Firestore-backed submissions and results
- GCS model and Replay Bundle artifact upload
- Pub/Sub-backed result notifications
- capability-protected Cloud Run job cancellation
- Cloud Run API, Cloud Run Job, and WebSocket relay deployment

## Non-Goals For Now

- High-fidelity simulation
- User-imported robot models
- Multiple training algorithms in production
- A browser UI
- Production-grade authentication and quota handling

## License

To be decided.
