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
| `PUBSUB_TOPIC` | string | Topic for cancellation and reconciled result events |
| `TRAINER_JOB_NAME` | string | Cloud Run Job name |

Makefile の `deploy_trainer` は `TRAINER_TASK_TIMEOUT` を読み、Cloud Run Job の
task timeout として適用する。現在の既定値は `24h` である。

resolved runtime shape:

```json
{
  "db_id": "my-firestore-db",
  "region": "asia-northeast1",
  "job_path": "projects/my-project/locations/asia-northeast1/jobs/my-trainer-job",
  "project_id": "my-project",
  "pubsub_topic": "trainer-results"
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
    "robot_version": "simple_robot.v1",
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
    "radius": 0.45,
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
  "submission_id": "submission-123",
  "cancel_token": "one-time-plaintext-capability"
}
```

`cancel_token` はこの response で一度だけ返す。server は平文を保存せず、
`submissions/{submission_id}.control.cancel_token_hash` に SHA-256 digest だけを保存する。
Unity client が再起動後もキャンセルする必要がある場合、client 側が token を保持する。

### `POST /submissions/{submission_id}/train`

request body はない。

successful response:

```json
{
  "status": "accepted",
  "submission_id": "submission-123"
}
```

学習開始時に Cloud Run `jobs.run` の Operation metadata から正確な Execution resource
name を取得し、submission の private control data に保存してから response を返す。

training failure responses:

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

### `POST /submissions/{submission_id}/cancel`

request body はない。submission 作成時に返された capability を bearer token として渡す。

```http
Authorization: Bearer <cancel_token>
```

API は token hash を検証し、保存済みの正確な Execution resource に対してキャンセルを
要求する。status は `cancelling`、完了後は `cancelled` となり、両 transition を
Pub/Sub / WebSocket へ publish する。Cloud Run の完了待ちが timeout した場合は
`202` と `cancelling` の Result Document を返し、後続の Result Document 再同期で
`cancelled` を確定する。

token がない、または一致しない場合は `403`、`completed` / `failed` の job は `409`
とする。すでに `cancelled` の job に対する再実行は idempotent に現在値を返す。

### `GET /results/{submission_id}`

response model shape: `embodiedlab.result_models.ResultDocument`

active status（`queued`、`starting`、`running`、`cancelling`）の result を返す場合、
API は submission に保存された正確な Cloud Run Execution resource を取得する。
対応する execution が timeout などで失敗済みなら `failed`、キャンセル済みなら
`cancelled` に更新してから返し、更新を Pub/Sub へ publish する。
これは trainer process が Cloud Run に強制終了され、trainer 自身の失敗更新が
実行されない場合の補正である。
この取得には runtime service account の `run.executions.get` 権限が必要であり、
bootstrap では `roles/run.viewer` を付与する。キャンセルは project custom role の
`run.executions.cancel` だけを追加し、広い Cloud Run Developer role は付与しない。

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
  },
  "control": {
    "cancel_token_hash": "sha256-hex-digest",
    "execution_name": "projects/my-project/locations/asia-northeast1/jobs/my-trainer-job/executions/my-trainer-job-abcde"
  }
}
```

`control` は外部 API response に含めない private server data である。機能追加前に作成した
submission にはこの field がないため、監視と成果物取得はできるがキャンセルはできない。

### `results/{submission_id}`

stored shape: `embodiedlab.result_models.ResultDocument`

common status values:

```json
["queued", "starting", "running", "cancelling", "cancelled", "completed", "failed"]
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
