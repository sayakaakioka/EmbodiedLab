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
| `REGION` | string | Cloud Run region for exact execution names |

resolved runtime shape:

```json
{
  "db_id": "my-firestore-db",
  "model_bucket": "my-model-bucket",
  "submission_id": "submission-123",
  "pubsub_topic": "trainer-results",
  "project_id": "my-project",
  "execution_name": "projects/my-project/locations/asia-northeast1/jobs/my-trainer-job/executions/my-trainer-job-abcde"
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

`EmbodiedLab.Unity` は submission response 消失後も同じ request を安全に再試行できるよう、
次の二つの header を生成して送る。

```http
Idempotency-Key: <32文字以上のURL-safe random value>
X-EmbodiedLab-Cancel-Token: <32文字以上のURL-safe cancellation capability>
```

二つのheaderは必須であり、同時に指定する。同じ idempotency key、正規化後の
Scenario Bundle、cancel token での再試行は同じ submission response を返す。
scenario または token が異なる key 再利用は `409 Conflict` とする。

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
      "radius": 0.45
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
      "width": 112,
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
    "components": [
      {"name": "goal_reached", "type": "terminal_reward", "weight": 100.0},
      {
        "name": "goal_progress",
        "type": "distance_delta",
        "target": "goal_001",
        "weight": 0.1
      },
      {"name": "collision_penalty", "type": "collision", "weight": -50.0},
      {"name": "step_penalty", "type": "per_step", "weight": -0.01},
      {"name": "wide_angle_penalty", "type": "per_step", "weight": -0.1},
      {"name": "rear_angle_penalty", "type": "per_step", "weight": -5.0},
      {"name": "inactive_penalty", "type": "per_step", "weight": -0.1},
      {"name": "movement_threshold", "type": "per_step", "weight": 0.001}
    ]
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

continuous runtime の reward contract は上記 8 component を必須とし、欠落、重複、
未知の name、name と type の不一致を拒否する。`goal_progress.target` は
`world.goal.id` と一致しなければならない。また deterministic evaluation を
1 chunk に収めるため、`eval_episodes * max_episode_steps <= 100000` を要求する。

response:

```json
{
  "status": "accepted",
  "submission_id": "submission-123",
  "cancel_token": "one-time-plaintext-capability"
}
```

client は両方の回収headerを必ず送り、`cancel_token`をrequest前から保持して
responseでも同じ値を受け取る。
server は平文を保存せず、
`submissions/{submission_id}.control.cancel_token_hash` に SHA-256 digest だけを保存する。
Unity client が再起動後もキャンセルする必要がある場合、client 側が token を保持する。

`POST /submissions` は submission と queued Result を原子的に保存し、同じ server-owned
workflow で Cloud Run dispatch を開始する。dispatch の確定拒否は response を
失敗させず、Result Document を terminal `failed` にする。RPC timeout、response loss、
execution name の
保存失敗は同じ submission を自動再dispatchせず、private control の `ambiguous` state として
調停する。queued progress は Scenario の `training.timesteps` を total steps として持つ。

### `POST /submissions/{submission_id}/cancel`

request body はない。submission 作成時に返された capability を bearer token として渡す。

```http
Authorization: Bearer <cancel_token>
```

API は token hash を検証し、private control に期限付きの cancellation intent lease を
CAS保存してから、正確な Execution resource にキャンセルを要求する。各leaseは一意な
fencing tokenを持ち、markとreleaseも同じtokenのownerだけが行う。同時requestでは
leaseを得たownerだけがRPCを送る。最初のRPCが受付結果を返さず不明になった場合は、
stale leaseを同じrequestの再試行が回収し、同じexact executionへのcancelを再送し得る。
Cloud RunがOperationを返した時点で`requested`を保存し、受付済みrequestは再送せず
exact executionを再調停する。確定拒否だけleaseを解放する。

status はRPC試行後に、受理済みまたは結果不明なら `cancelling`、完了後は `cancelled`
となり、両 transition を Pub/Sub / WebSocket へ publish する。Cloud Run の完了待ちが
timeoutした場合は
`202`を返す。trainerが先に成果物生成まで完了した場合はterminal CASで`completed`が
確定し、キャンセル処理はこれを上書きしない。stale lease後の再試行または後続の
Result Document再同期で最終状態を確定する。

token がない、または一致しない場合は `403`、`completed` / `failed` の job は `409`
とする。すでに `cancelled` の job に対する再実行は idempotent に現在値を返す。

### `GET /results/{submission_id}`

response model shape: `embodiedlab.result_models.ResultDocument`

active status（`queued`、`starting`、`running`、`cancelling`）の result を返す場合、
API は submission に保存された正確な Cloud Run Execution resource を取得する。
APIがdispatch responseを失った場合も、trainerは`starting`へ遷移する前にruntimeの
execution IDを同じcontrolへCAS保存する。既にdispatchがterminalならtrainerは学習を
開始せず、terminal Resultを復活させない。
対応する execution が timeout などで失敗済み、または正常終了したのに terminal Result が
なければ `failed`、キャンセル済みなら `cancelled` に更新してから返し、更新を Pub/Sub へ
publish する。dispatch が一定時間 `ambiguous` のままなら、別 execution を開始せず
terminal `failed` として確定する。
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
    "runtime": "continuous_navigation",
    "score": 6.4,
    "episodes": 20,
    "obstacle_count": 1,
    "goal": { "x": 8.5, "z": 8.5, "radius": 0.45 },
    "robot_start": {
      "x": 1.0,
      "z": 1.0,
      "rotation_y_degrees": 0.0
    },
    "robot_type": "simple_robot",
    "robot_radius": 0.45,
    "success_rate": 0.95,
    "avg_reward": 6.4,
    "avg_steps": 118.5,
    "training_timesteps": 5000,
    "training_seed": 10
  },
  "error": null,
  "result_bundle": {
    "schema_version": "result-bundle.v0",
    "scenario_id": "scenario_demo_001",
    "job_id": "submission-123",
    "status": "completed",
    "compatibility": {
      "scenario_schema_version": "scenario-bundle.v0",
      "envforge_min_version": "0.1.0",
      "robot_version": "simple_robot.v1",
      "sensor_version": "basic_sensors.v0",
      "action_layout": ["forward", "turn"],
      "observation_layout": ["obs_0", "obs_1"]
    },
    "summary": {
      "training_timesteps": 5000,
      "training_seed": 10,
      "success_rate": 0.95,
      "average_episode_reward": 6.4,
      "average_episode_steps": 118.5
    },
    "artifacts": {
      "model": {
        "storage": "gcs",
        "bucket": "my-model-bucket",
        "path": "results/submission-123/model/policy.onnx",
        "format": "onnx"
      },
      "onnx_model": {
        "storage": "gcs",
        "bucket": "my-model-bucket",
        "path": "results/submission-123/model/policy.onnx",
        "format": "onnx",
        "target": "onnx-runtime",
        "opset_version": 17,
        "inputs": [
          {
            "name": "obs_0",
            "shape": [-1, 3, 84, 112],
            "dtype": "float32",
            "layout": [
              "channel_0_unused",
              "channel_1_traversable",
              "channel_2_blocked_or_background"
            ]
          },
          {
            "name": "obs_1",
            "shape": [-1, 2],
            "dtype": "float32",
            "layout": ["goal_angle_degrees", "goal_distance_meters"]
          }
        ],
        "output": {
          "name": "action",
          "layout": ["forward", "turn"],
          "action_mapping": {
            "forward": "sigmoid(policy_forward)",
            "turn": "clip(policy_turn, -3, 3) / 3"
          }
        }
      },
      "sentis_model": {
        "storage": "gcs",
        "bucket": "my-model-bucket",
        "path": "results/submission-123/model/policy.sentis.onnx",
        "format": "onnx",
        "target": "unity-sentis",
        "opset_version": 15,
        "inputs": [
          {
            "name": "observation",
            "shape": [1, 28226],
            "dtype": "float32",
            "layout": [
              "obs_0_chw_3x84x112",
              "obs_1_angle_degrees",
              "obs_1_distance_meters"
            ]
          }
        ],
        "output": {
          "name": "action",
          "layout": ["forward", "turn"],
          "action_mapping": {
            "forward": "sigmoid(policy_forward)",
            "turn": "clip(policy_turn, -3, 3) / 3"
          }
        }
      },
      "replay_bundle": {
        "storage": "gcs",
        "bucket": "my-model-bucket",
        "path": "results/submission-123/replay/manifest.json",
        "format": "json"
      }
    }
  },
  "updated_at": "2026-04-24T12:34:56.000000+00:00"
}
```

artifact metadata の正規の格納先は `result_bundle.artifacts` だけであり、
Result Document の top-level には複製しない。旧 top-level artifact だけを持つ
result は現行 Unity client の対象外とする。

artifact path は `results/{submission_id}/model/` または
`results/{submission_id}/replay/` 配下の GCS object path である。
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
    "dispatch_state": "dispatched",
    "dispatch_started_at": "2026-04-24T12:34:57.000000+00:00",
    "dispatch_error": null,
    "execution_name": "projects/my-project/locations/asia-northeast1/jobs/my-trainer-job/executions/my-trainer-job-abcde",
    "cancellation_state": "idle",
    "cancellation_started_at": null,
    "cancellation_lease_token": null,
    "cancellation_error": null
  }
}
```

`control` は外部 API response に含めない private server data である。現行dispatch fieldまたは
Resultが欠けた旧submissionは、同じidempotency keyによる再受理を明示的に拒否する。
既存Resultの直接監視と成果物取得は継続できる。
idempotency key自体も保存せず、そのSHA-256 digestから安定したsubmission IDを導出する。

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
trainer は Cloud Run runtime が提供する `CLOUD_RUN_JOB` と `CLOUD_RUN_EXECUTION`、および
deployment の `REGION` から exact execution resource name を再構成し、API側のresponseが
失われた場合もdispatch controlへCAS保存してから学習を開始する。
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
  "result_bundle": null,
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
