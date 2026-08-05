# データモデル

この文書は、`server`、`trainer`、`notification` が使う外部 payload と startup
config の境界をまとめる。field の完全な型定義は次を正本とする。

- Pydantic model: `embodiedlab/schemas.py`、`embodiedlab/result_models.py`
- versioned JSON Schema: `contracts/v0/`
- canonical payload: `tests/fixtures/envforge/`

説明用 JSON をここへ複製して陳腐化させず、変更時は上記3か所と contract test を同時に
更新する。

## Service Startup Config

### `server`

`server/config.py` が読む環境変数:

| Name | Type | Purpose |
| --- | --- | --- |
| `DB_ID` | string | Firestore database ID |
| `REGION` | string | Cloud Run region |
| `PROJECT_ID` | string | GCP project ID |
| `PUBSUB_TOPIC` | string | result event topic |
| `TRAINER_JOB_NAME` | string | Cloud Run Job name |

Makefile の `deploy_trainer` は `TRAINER_TASK_TIMEOUT` を Cloud Run Job の task timeout
として適用する。現在の既定値は `24h` である。

### `trainer`

`trainer/config.py` が読む環境変数:

| Name | Type | Purpose |
| --- | --- | --- |
| `DB_ID` | string | Firestore database ID |
| `MODEL_BUCKET` | string | GCS bucket for result artifacts |
| `SUBMISSION_ID` | string | submission to train |
| `PUBSUB_TOPIC` | string | ordered result event topic |
| `PROJECT_ID` | string | GCP project ID |
| `REGION` | string | region used to reconstruct the exact execution name |

Cloud Run runtime の `CLOUD_RUN_JOB` と `CLOUD_RUN_EXECUTION` も読み、trainer 自身の
exact execution resource を private submission control に保存する。

### `notification`

notification service は Pub/Sub push を受け取り、submission ごとの WebSocket client へ
`ResultMessage` を配信する。deploy 時の service 名と push path は Makefile の設定である。

## External API Payloads

### `POST /submissions`

request model は `ScenarioBundle` である。現在の canonical request は
`tests/fixtures/envforge/navigation_default_scenario_bundle.json` を参照する。

重要な契約は次のとおり。

- 座標は `left_handed_y_up_meters`。
- wall と obstacle は `height` を明示する。
- robot action は layout に加えて `forward_step_meters`、
  `turn_degrees_per_step`、Replay 時刻にも使う `step_duration_seconds` を持つ。
- policy input は forward camera と goal vector を明示する。camera の input 名、解像度、
  semantic mode、画角、clip、mount height は Scenario の値を使う。
- 固定 tutorial は policy input に使わない distance sensor を持たない。汎用
  `DistanceSensor` 型は contract に残す。
- reward は7 component で、goal progress、wide/rear angle、inactive の発火条件も JSON に
  明示する。
- PPO hyperparameter、environment 数、CPU、PyTorch thread、Replay interval、start pose
  randomization を training object に明示する。
- 1 submission が task timeout または rollout memory を無制限に占有しないよう、
  `timesteps <= 10,000,000`、`n_envs <= 32`、CPU/thread は32以下、
  `n_steps * n_envs <= 65,536`、`n_epochs <= 100` を公開契約で検証する。
- `seed` は unsigned 32-bit の `0..4294967295` とする。wall と obstacle は合計128件、
  sensor は forward camera、goal vector、任意の distance sensor の最大3件に制限する。
  identifier は128文字以下とする。
- Scenario と全 nested object の未知 field は拒否する。全 wire field は JSON で明示する。
  非有限数と、number/bool field への文字列・整数 coercion も拒否する。
  `cpu_count` と `torch_num_threads` の `null` は runtime
  による自動選択を表す。

次の二つの header は必須である。

```http
Idempotency-Key: <32文字以上のURL-safe random value>
X-EmbodiedLab-Cancel-Token: <32文字以上のURL-safe cancellation capability>
```

同じ idempotency key、正規化済み Scenario、cancel token の再試行は同じ submission を
返す。scenario または token が異なる key 再利用は `409 Conflict` とする。server は
submission 保存と trainer dispatch を一つの受理操作として所有し、client に二段階操作や
部分失敗の後処理を要求しない。

### `POST /submissions/{submission_id}/cancel`

submission 作成時に返された capability を bearer token として渡す。

```http
Authorization: Bearer <cancel_token>
```

server は保存済み SHA-256 hash と照合し、正確な Cloud Run Execution を cancel する。
token がない、または一致しない場合は `403`、terminal job は `409`、完了待ちが timeout
した場合は `202` を返す。

### `GET /results/{submission_id}`

response model は `ResultDocument` である。公開状態は `queued`、`starting`、`running`、
`cancelling`、`cancelled`、`completed`、`failed`。

全 Result Document/Event は `progress`、`error`、`result_bundle`、`updated_at` を省略せず
持つ。`status` と `progress.phase` は一致し、completed は completed Result Bundle を必須、
failed は非空の error を必須とする。Result Bundle の `job_id` は外側の
`submission_id` と一致する。

completed `ResultBundle` は次を必須とする。

- submitted Scenario から導出した compatibility metadata
- metrics と、実行時に解決した Stable-Baselines3 version、全 PPO 値、CPU/thread 数、
  Replay 設定を含む training configuration
- `policy.onnx`
- Replay Bundle の `manifest.json`

公開する2成果物は `storage`、`bucket`、`path`、`format`、`size_bytes`、`sha256` を持つ。
model artifact は target、opset、input/output metadata も持つ。`policy.zip` と重複した
`model` field は公開しない。artifact metadata の格納先は
`result_bundle.artifacts` だけであり、Result Document top-level へ複製しない。

## Replay Bundle

Replay Bundle は次の構造を持つ。

```text
replay/
  manifest.json
  train/chunk_<index>.jsonl.gz
  eval/checkpoint_<step>.jsonl.gz
```

manifest の各 chunk entry は phase、policy mode、step 範囲、path、件数、圧縮後
`size_bytes`、圧縮後 bytes の `sha256` を持つ。trainer は gzip timestamp を固定し、同じ
入力から同じ bytes を生成する。upload 前に manifest と実ファイルの集合、size、digest を
照合し、symlink、path traversal、未宣言ファイルを拒否する。train chunk は stochastic と
step 範囲、eval chunk は deterministic と episode metrics を必須とする。各 Replay row の
phase、policy mode、checkpoint 範囲、episode 数、action の `[forward, turn]` 順まで照合し、
全 model/Replay artifact の preflight 完了後に初めて upload する。manifest は chunk upload
後に upload する。

Replay Bundle は1件以上の chunk と1以上の `total_timesteps` を必須とする。1 chunk は
最大100,000行、manifest は最大4,096 chunk である。Scenario は
`eval_episodes * max_episode_steps <= 100000` を満たす必要がある。Result と Replay の
number/bool field でも文字列や整数への coercion と非有限数を拒否する。

## Firestore Document Shapes

### `submissions/{submission_id}`

`SubmissionDocument` は `submission_id`、`created_at`、正規化済み `scenario`、private
`control` を持つ。control には cancel token hash、dispatch state、exact execution name、
cancellation lease を保存する。control と平文 capability は公開 response に含めない。

### `results/{submission_id}`

`ResultDocument` は現在状態、progress、error、Result Bundle、更新時刻を持つ。
公開 training summary は `result_bundle.summary` だけに置く。terminal state は
compare-and-set で確定し、遅い dispatch/cancel 処理が completed
結果を上書きしない。

## Service-To-Service Payloads

### API -> Cloud Run Job

API は trainer へ Scenario JSON を直接送らない。Firestore に保存後、Cloud Run Job の
`SUBMISSION_ID` override を設定する。trainer は submission を読み、同じ Scenario contract
から runtime、policy、Replay を構成する。

### Trainer -> Pub/Sub -> WebSocket

trainer と API の調停処理は `ResultMessage` を Pub/Sub へ publish する。notification は
standard Pub/Sub push envelope を検証し、同じ typed message を該当 WebSocket client へ
送る。接続時は最新 Result snapshot も返す。

## Training Runtime

`TrainingConfig` は `ScenarioBundle.training` の全 field を1対1で受け取る。runtime 内の
`max_steps` property は保存値を重複させず `max_episode_steps` を参照する。library の
暗黙既定値へ依存せず、Scenario の PPO 値を `PPO` constructor へ明示的に渡す。

`cpu_count` が指定された Linux job では process affinity を設定する。利用可能数を超える
CPU、effective CPU を超える `n_envs` または `torch_num_threads` は補正せず失敗させる。
解決した値は Result Bundle に記録する。
