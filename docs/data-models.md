# Data Models

This document summarizes the main startup configuration and runtime payloads
used by the `server`, `trainer`, and `notification` services.

## Service Startup Config

### `server`

Environment variables loaded by `server/config.py`:

| Name | Type | Purpose |
| --- | --- | --- |
| `DB_ID` | string | Firestore database ID |
| `REGION` | string | Cloud Run region |
| `PROJECT_ID` | string | GCP project ID |
| `TRAINER_JOB_NAME` | string | Cloud Run Job name used to construct `job_path` |

Resolved runtime shape:

```json
{
  "db_id": "my-firestore-db",
  "region": "asia-northeast1",
  "job_path": "projects/my-project/locations/asia-northeast1/jobs/my-trainer-job"
}
```

### `trainer`

Environment variables loaded by `trainer/config.py`:

| Name | Type | Purpose |
| --- | --- | --- |
| `DB_ID` | string | Firestore database ID |
| `MODEL_BUCKET` | string | GCS bucket for model artifacts |
| `SUBMISSION_ID` | string | Submission to train |
| `PUBSUB_TOPIC` | string | Topic used for ordered result events |
| `PROJECT_ID` | string | GCP project ID |

Resolved runtime shape:

```json
{
  "db_id": "my-firestore-db",
  "model_bucket": "my-model-bucket",
  "submission_id": "submission-123",
  "pubsub_topic": "trainer-results",
  "project_id": "my-project"
}
```

### `notification`

The notification service does not currently load a dedicated Python config
object at startup. Deployment still depends on these Makefile-level variables:

| Name | Type | Purpose |
| --- | --- | --- |
| `NOTIFICATION_SERVICE_NAME` | string | Cloud Run service name |
| `NOTIFICATION_PUSH_PATH` | string | Pub/Sub push path, typically `/internal/pubsub/push` |

### `tools/ws_client.py`

The local WebSocket helper is usually run via `make get_result_ws`. It builds
the Cloud Run WebSocket URL from environment variables exported by `Makefile`:

| Name | Type | Purpose |
| --- | --- | --- |
| `NOTIFICATION_SERVICE_NAME` | string | Notification Cloud Run service name |
| `HASH` | string | Cloud Run service URL hash suffix |
| `REGION` | string | Cloud Run region |
| `SUBMISSION_ID` | string | Submission to subscribe to |

## External API Payloads

### `POST /submissions`

Request model: `embodiedlab.schemas.SubmitRequest`

```json
{
  "environment": {
    "size": [4, 4],
    "obstacles": [
      { "x": 1, "y": 1 }
    ],
    "goal": { "x": 3, "y": 3 },
    "robot_start": { "x": 0, "y": 0 }
  },
  "robot": {
    "type": "simple"
  },
  "training": {
    "algorithm": "ppo",
    "timesteps": 5000,
    "seed": 10,
    "max_steps": 50,
    "n_steps": 32,
    "batch_size": 32,
    "gamma": 0.99,
    "learning_rate": 0.0003,
    "ent_coef": 0.0,
    "eval_episodes": 20
  }
}
```

Response:

```json
{
  "status": "accepted",
  "submission_id": "submission-123"
}
```

### `POST /submissions/{submission_id}/train`

Request body: none

Successful response:

```json
{
  "status": "accepted",
  "submission_id": "submission-123"
}
```

Failure responses:

```json
{
  "detail": "Submission not found"
}
```

```json
{
  "detail": "Failed to start trainer job"
}
```

### `GET /results/{submission_id}`

Response model shape: `embodiedlab.result_models.ResultDocument`

```json
{
  "submission_id": "submission-123",
  "status": "completed",
  "progress": {
    "phase": "completed",
    "current_step": 5000,
    "total_steps": 5000,
    "message": "Training completed"
  },
  "summary": {
    "policy": "ppo",
    "score": 0.95,
    "grid_width": 4,
    "grid_height": 4,
    "episodes": 20,
    "obstacle_count": 1,
    "goal": { "x": 3, "y": 3 },
    "robot_start": { "x": 0, "y": 0 },
    "robot_type": "simple",
    "success_rate": 1.0,
    "avg_reward": 0.95,
    "avg_steps": 6.1,
    "training_timesteps": 5000,
    "training_seed": 10
  },
  "error": null,
  "artifacts": {
    "model": {
      "storage": "gcs",
      "bucket": "my-model-bucket",
      "path": "models/submission-123/policy.zip"
    },
    "onnx_model": {
      "storage": "gcs",
      "bucket": "my-model-bucket",
      "path": "models/submission-123/policy.onnx"
    }
  },
  "updated_at": "2026-04-24T12:34:56.000000+00:00"
}
```

Artifact paths are GCS object paths under `models/{submission_id}/`. The
Makefile-created model bucket is configured for public object read, so clients
can download both `policy.zip` and `policy.onnx` directly when the project
allows public bucket IAM.

## Firestore Document Shapes

### `submissions/{submission_id}`

Stored shape: `embodiedlab.schemas.SubmissionDocument`

```json
{
  "submission_id": "submission-123",
  "created_at": "2026-04-24T12:34:56.000000+00:00",
  "environment": {
    "size": [4, 4],
    "obstacles": [
      { "x": 1, "y": 1 }
    ],
    "goal": { "x": 3, "y": 3 },
    "robot_start": { "x": 0, "y": 0 }
  },
  "robot": {
    "type": "simple"
  },
  "training": {
    "algorithm": "ppo",
    "timesteps": 5000,
    "seed": 10,
    "max_steps": 50,
    "n_steps": 32,
    "batch_size": 32,
    "gamma": 0.99,
    "learning_rate": 0.0003,
    "ent_coef": 0.0,
    "eval_episodes": 20
  }
}
```

### `results/{submission_id}`

Stored shape: `embodiedlab.result_models.ResultDocument`

Common status values:

```json
["queued", "starting", "running", "completed", "failed"]
```

Progress shape:

```json
{
  "phase": "running",
  "current_step": 0,
  "total_steps": 5000,
  "message": "Training"
}
```

## Service-To-Service Payloads

### API -> Cloud Run Job Override

The API does not send JSON to the trainer service directly. It starts a Cloud
Run Job and overrides one environment variable:

```json
{
  "name": "SUBMISSION_ID",
  "value": "submission-123"
}
```

### Trainer -> Pub/Sub Result Event

Published shape: `embodiedlab.result_models.ResultMessage`

```json
{
  "submission_id": "submission-123",
  "status": "running",
  "progress": {
    "phase": "running",
    "current_step": 0,
    "total_steps": 5000,
    "message": "Training"
  },
  "summary": null,
  "error": null,
  "artifacts": null,
  "updated_at": "2026-04-24T12:35:12.000000+00:00"
}
```

Completed result events include the same `artifacts` block returned by
`GET /results/{submission_id}`, including both `model` and `onnx_model`.

### Pub/Sub Push -> Notification Service

The notification service receives the standard Pub/Sub push envelope. The
result event above is JSON-encoded and then base64-encoded into `message.data`.

```json
{
  "message": {
    "data": "eyJzdWJtaXNzaW9uX2lkIjogInN1Ym1pc3Npb24tMTIzIiwgInN0YXR1cyI6ICJydW5uaW5nIiwgLi4ufQ=="
  }
}
```

The notification service validates the decoded payload against
`embodiedlab.result_models.ResultMessage`.

### Notification -> WebSocket Client

Initial handshake message:

```json
{
  "type": "connected",
  "submission_id": "submission-123"
}
```

Subsequent pushed message:

```json
{
  "submission_id": "submission-123",
  "status": "running",
  "progress": {
    "phase": "running",
    "current_step": 0,
    "total_steps": 5000,
    "message": "Training"
  },
  "summary": null,
  "error": null,
  "artifacts": null,
  "updated_at": "2026-04-24T12:35:12.000000+00:00"
}
```

## Shared Nested Models

### `GridPosition`

```json
{
  "x": 0,
  "y": 0
}
```

### `TrainingConfig`

```json
{
  "algorithm": "ppo",
  "timesteps": 5000,
  "seed": 10,
  "max_steps": 50,
  "n_steps": 32,
  "batch_size": 32,
  "gamma": 0.99,
  "learning_rate": 0.0003,
  "ent_coef": 0.0,
  "eval_episodes": 20
}
```

### `GridWorldSpec`

`GridWorldSpec` is an internal Python dataclass passed to the training runner.
It is not stored or transmitted directly as JSON, but it resolves to:

```json
{
  "width": 4,
  "height": 4,
  "obstacles": [
    { "x": 1, "y": 1 }
  ],
  "goal": { "x": 3, "y": 3 },
  "robot_start": { "x": 0, "y": 0 },
  "robot_type": "simple"
}
```
