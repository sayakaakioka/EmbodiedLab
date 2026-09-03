# EmbodiedLab

EmbodiedLab is the cloud training backend and wire-contract source of truth for
the EmbodiedLab ecosystem. It owns the versioned Scenario Bundle, Result Bundle,
and Replay Bundle schemas. The separate
[`EmbodiedLab.Unity`](https://github.com/sayakaakioka/EmbodiedLab.Unity) package
provides Unity DTOs, transport, artifact validation, replay, and inference APIs
over those contracts. Its importable Quickstart is the starting point for Unity
clients. [`EnvForge`](https://github.com/sayakaakioka/EnvForge) remains responsible
for authoring scenarios and presenting results through that SDK.

The current prototype accepts complete Scenario Bundle submissions through a
Cloud Run API, starts a Cloud Run Job to train a reinforcement learning policy,
stores ONNX and Replay Bundle artifacts in GCS, and streams
status updates to clients over WebSockets.

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
  -> POST /submissions with required recovery headers
      -> atomically create the Firestore submission and queued result
      -> identical retries resolve to the same submission
      -> claim one server-owned dispatch attempt
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

Both recovery headers are required. `make submit` generates and persists both
values so an identical retry can recover the same submission without starting
a second paid training execution.

Response:

```json
{
  "status": "accepted",
  "submission_id": "...",
  "cancel_token": "example_cancel_token_0123456789abcdef"
}
```

The client generates the cancellation capability before the request, sends it
in `X-EmbodiedLab-Cancel-Token`, and receives the same value in this response.
Persist it if the client must recover or cancel the job after restarting. The
server stores only its SHA-256 digest.

Creating the submission also starts the server-owned training workflow. A
definitive dispatch rejection becomes a terminal failed Result Document.
Timeouts and lost responses are never retried as a second Cloud Run execution;
the server retains the ambiguous dispatch for reconciliation. Before training,
the trainer recovers the exact execution name from the Cloud Run runtime and
persists it with a compare-and-set transition.

The current deployment target is unauthenticated and has no per-caller quota.
Do not expose it as a production paid-training API until authentication, quota,
and bounded training-resource validation are in place.

### Cancel Training

```http
POST /submissions/{submission_id}/cancel
Authorization: Bearer {cancel_token}
```

Cancellation targets the exact Cloud Run Execution recorded when training was
started. A private, expiring intent lease and fencing token give one request
ownership of the RPC without allowing a stale owner to modify a newer lease.
If the initial RPC returns no definitive acceptance response, an identical
retry may resend cancellation for the same exact execution after the lease
expires. Once Cloud Run returns an Operation, the request is reconciled without
redispatch. The response is the latest result document and is idempotent after
the job reaches `cancelled`.

### Get Result

```http
GET /results/{submission_id}
```

Result documents include:

- `status`: `queued`, `starting`, `running`, `cancelling`, `cancelled`,
  `completed`, or `failed`
- `progress`: phase, current step, total steps, and message
- `result_bundle`: the sole typed training summary, artifact locations,
  compatibility metadata,
  and structured failure details
- `error`: failure detail when failed

A `completed` result always contains a non-null `result_bundle`. A `failed`
result always contains a non-empty top-level `error`; it also contains a failed
`result_bundle` when the trainer had a validated Scenario Bundle from which it
could build compatibility metadata. Earlier submission or scenario failures
leave `result_bundle` as `null`.

Canonical artifacts exist only under `result_bundle.artifacts`:

- `onnx_model`: opset 18 model whose input names and shapes come from the
  submitted Scenario Bundle
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

This starts the API process locally, but it still uses the configured Firestore,
Cloud Run Job, and Pub/Sub resources. It is not an offline end-to-end backend.

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

The following commands target the deployed `API_URL`. `make submit` creates a
real server-owned training submission and may start billable Cloud Run work.
Confirm the target project, API URL, resource limits, and expected cost before
running it. The command also writes local recovery files for the idempotency key,
cancel capability, response, and submission ID; these files are gitignored.

Submit the sample payload:

```bash
make submit
```

To submit another complete Scenario Bundle, override the fixture explicitly:

```bash
make submit SUBMISSION_PAYLOAD=path/to/scenario.json
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
