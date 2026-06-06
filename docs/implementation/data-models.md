# データモデル

この文書は、`server`、`trainer`、`notification` service が使う
主な startup config と runtime payload をまとめる。

## Service Startup Config

### `server`

`server/config.py` が読む環境変数:

| Name | Type | Purpose |
| --- | --- | --- |
| `DB_ID` | string | Firestore database ID |
| `REGION` | string | Cloud Run region |
| `PROJECT_ID` | string | GCP project ID |
| `TRAINER_JOB_NAME` | string | Cloud Run Job name |

Makefile の `deploy_trainer` は `TRAINER_TASK_TIMEOUT` を読み、Cloud Run Job の
task timeout として適用する。現在の既定値は `24h` である。

resolved runtime shape:

```json
{
  "db_id": "my-firestore-db",
  "region": "asia-northeast1",
  "job_path": "projects/my-project/locations/asia-northeast1/jobs/my-trainer-job"
}
```

### `trainer`

`trainer/config.py` が読む環境変数:

| Name | Type | Purpose |
| --- | --- | --- |
| `DB_ID` | string | Firestore database ID |
| `MODEL_BUCKET` | string | GCS bucket for model artifacts |
| `SUBMISSION_ID` | string | Submission to train |
| `PUBSUB_TOPIC` | string | Topic for ordered result events |
| `PROJECT_ID` | string | GCP project ID |

resolved runtime shape:

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

notification service は、現在 dedicated Python config object を
startup 時に読み込んでいない。
deploy は Makefile-level variables に依存している。

| Name | Type | Purpose |
| --- | --- | --- |
| `NOTIFICATION_SERVICE_NAME` | string | Cloud Run service name |
| `NOTIFICATION_PUSH_PATH` | string | Pub/Sub push endpoint path |

### `tools/ws_client.py`

local WebSocket helper は通常 `make get_result_ws` 経由で実行する。
Makefile が export する環境変数から Cloud Run WebSocket URL を作る。

| Name | Type | Purpose |
| --- | --- | --- |
| `NOTIFICATION_SERVICE_NAME` | string | Notification service name |
| `HASH` | string | Cloud Run service URL hash suffix |
| `REGION` | string | Cloud Run region |
| `SUBMISSION_ID` | string | Submission to subscribe to |

## External API Payloads

### `POST /submissions`

request model: `embodiedlab.schemas.ScenarioBundle`

```json
{
  "schema_version": "scenario-bundle.v0",
  "scenario_id": "scenario_demo_001",
  "created_by": {
    "tool": "EnvForge",
    "version": "0.1.0"
  },
  "compatibility": {
    "envforge_min_version": "0.1.0",
    "robot_version": "simple_robot.v0",
    "sensor_version": "basic_sensors.v0"
  },
  "world": {
    "coordinate_system": "envforge_xz_meters",
    "bounds": {
      "min": { "x": 0.0, "z": 0.0 },
      "max": { "x": 10.0, "z": 10.0 }
    },
    "static_walls": [],
    "static_obstacles": [],
    "goal": {
      "id": "goal_001",
      "position": { "x": 8.5, "z": 8.5 },
      "radius": 0.5
    }
  },
  "robot": {
    "type": "simple_robot",
    "start_pose": {
      "position": { "x": 1.0, "z": 1.0 },
      "rotation_y_degrees": 0.0
    },
    "action_space": {
      "type": "continuous",
      "layout": ["forward", "turn"]
    }
  },
  "sensors": [
    {
      "id": "front_camera",
      "type": "forward_camera",
      "width": 84,
      "height": 84,
      "semantic_mode": "traversable_vs_blocked"
    },
    {
      "id": "front_distance",
      "type": "distance_sensor",
      "range_meters": 5.0,
      "direction": "forward"
    }
  ],
  "reward": {
    "components": []
  },
  "training": {
    "algorithm": "ppo",
    "timesteps": 5000,
    "seed": 10,
    "max_episode_steps": 512,
    "n_steps": 32,
    "batch_size": 32,
    "gamma": 0.99,
    "learning_rate": 0.0003,
    "ent_coef": 0.0,
    "eval_episodes": 20
  }
}
```

response:

```json
{
  "status": "accepted",
  "submission_id": "submission-123"
}
```

### `POST /submissions/{submission_id}/train`

request body はない。

successful response:

```json
{
  "status": "accepted",
  "submission_id": "submission-123"
}
```

failure responses:

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

response model shape: `embodiedlab.result_models.ResultDocument`

active status（`queued`、`starting`、`running`）の result を返す場合、API は
Cloud Run execution を `SUBMISSION_ID` override で照合する。対応する execution が
timeout などで失敗済みなら、Firestore result を `failed` に更新してから返す。
これは trainer process が Cloud Run に強制終了され、trainer 自身の失敗更新が
実行されない場合の補正である。
この照合には runtime service account の `run.executions.list` 権限が必要であり、
bootstrap では `roles/run.viewer` を付与する。

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
    }
  },
  "updated_at": "2026-04-24T12:34:56.000000+00:00"
}
```

artifact path は `models/{submission_id}/` 配下の GCS object path である。
現在の Makefile-created model bucket は public object read を許可する。
これは prototype 用であり、今後 access control を見直す。

## Firestore Document Shapes

### `submissions/{submission_id}`

stored shape: `embodiedlab.schemas.SubmissionDocument`

```json
{
  "submission_id": "submission-123",
  "created_at": "2026-04-24T12:34:56.000000+00:00",
  "scenario": {
    "schema_version": "scenario-bundle.v0",
    "scenario_id": "scenario_demo_001",
    "world": {
      "coordinate_system": "envforge_xz_meters"
    },
    "robot": {
      "type": "simple_robot"
    },
    "training": {
      "algorithm": "ppo",
      "timesteps": 5000,
      "max_episode_steps": 512
    }
  }
}
```

### `results/{submission_id}`

stored shape: `embodiedlab.result_models.ResultDocument`

common status values:

```json
["queued", "starting", "running", "completed", "failed"]
```

progress shape:

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

API は trainer service に JSON を直接送らない。
Cloud Run Job を起動し、環境変数 `SUBMISSION_ID` を override する。
trainer job の task timeout は Makefile の `TRAINER_TASK_TIMEOUT` で指定し、
現在の既定値は `24h` である。

```json
{
  "name": "SUBMISSION_ID",
  "value": "submission-123"
}
```

### Trainer -> Pub/Sub Result Event

published shape: `embodiedlab.result_models.ResultMessage`

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

### Pub/Sub Push -> Notification Service

notification service は standard Pub/Sub push envelope を受け取る。
result event は JSON encode 後、`message.data` に base64 encode される。

```json
{
  "message": {
    "data": "eyJzdWJtaXNzaW9uX2lkIjogInN1Ym1pc3Npb24tMTIzIn0="
  }
}
```

### Notification -> WebSocket Client

initial handshake message:

```json
{
  "type": "connected",
  "submission_id": "submission-123"
}
```

subsequent pushed message は `ResultMessage` と同じ形である。

## Shared Nested Models

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
