# 現在のアーキテクチャ

## 概要

EmbodiedLab は現在、EnvForge Scenario Bundle を continuous navigation
runtime へ変換し、クラウド上で PPO 学習する最小限の学習ループを実装している。

FastAPI service が submission を受け取り、Firestore に保存し、
Cloud Run Job を起動する。trainer は Stable-Baselines3 PPO で
方策を学習し、model artifact を Google Cloud Storage にアップロードし、
結果状態を Firestore に書き戻し、Pub/Sub と WebSocket relay を通して
更新を通知する。

## Runtime Flow

    Client
      -> POST /submissions
          -> SDK は idempotency key と cancellation capability を生成して送る
          -> 同じ key、scenario、capability の再試行は同じ submission を返す
          -> client が保持する cancellation capability を response に返す
          -> Firestore submissions/{submission_id} に token hash を保存
      -> POST /submissions/{submission_id}/train
          -> Firestore results/{submission_id} = queued
          -> Cloud Run Job with SUBMISSION_ID override
              -> Operation metadata の正確な Execution name を submission に保存
              -> Firestore submission lookup
              -> Continuous navigation PPO training
              -> GCS artifact upload
              -> Firestore result update
              -> Pub/Sub event
                  -> notification service push endpoint
                      -> WebSocket subscribers
      -> POST /submissions/{submission_id}/cancel
          -> cancellation capability を検証
          -> 保存済みの正確な Cloud Run Execution を cancel
          -> cancelling / cancelled event を Pub/Sub へ publish
      -> GET /results/{submission_id}
      -> WebSocket /ws/results/{submission_id}

## Services

### `server/`

FastAPI API service である。
submission の受理、result document の queued 化、Cloud Run Job の起動とキャンセル、
result document の返却を担当する。キャンセルは submission ごとの capability token で
保護し、Firestore には SHA-256 hash だけを保存する。
submission response が失われた場合、client は同じ `Idempotency-Key` と
`X-EmbodiedLab-Cancel-Token` で再試行する。server は同一 request を同じ submission へ
解決し、異なる scenario または capability での key 再利用を拒否する。

主な endpoint は以下である。

- `POST /submissions`
- `POST /submissions/{submission_id}/train`
- `POST /submissions/{submission_id}/cancel`
- `GET /results/{submission_id}`

Cloud Run Job が timeout などで Python trainer の cleanup 前に終了した場合、
trainer 自身は Firestore result を更新できない。このため API は
`GET /results/{submission_id}` で active status
（`queued`、`starting`、`running`、`cancelling`）の result を返す前に、学習開始時に
保存した正確な Cloud Run Execution を取得する。対応 execution が失敗済みなら
`failed`、キャンセル済みなら `cancelled` に更新して Pub/Sub へ publish する。
recent execution の走査や `SUBMISSION_ID` override による推測は行わない。

### `trainer/`

Cloud Run Job service である。
submission を読み、training spec に変換し、PPO training を実行し、
artifact をアップロードし、Firestore を更新し、status event を publish する。
trainer job の task timeout は約 1 日を想定し、Makefile の
`TRAINER_TASK_TIMEOUT` で `24h` を既定値にしている。

### `notification/`

FastAPI WebSocket relay service である。
Pub/Sub push event を受け取り、submission id に対応する WebSocket client へ
broadcast する。

### `embodiedlab/`

schemas、result models、repository protocols、continuous navigation environment、
training converter、training runner を含む shared library である。

## 現在のデータモデル

現在の API は EnvForge Scenario Bundle を受け取る。主経路の環境は、
Gymnasium-compatible continuous navigation runtime である。action は PPO 内部では
raw `forward` と raw `turn` を分けて扱い、runtime 適用時は
`forward=sigmoid(raw_forward)`、`turn=clip(raw_turn,-3,3)/3` に写像される。
observation は `obs_0` の semantic camera
`3 x 84 x 112` と、`obs_1` の `[goal_angle_degrees, goal_distance_meters]` である。
旧 grid-world runtime は削除済みで、必要な履歴は Git history を参照する。

## 現在の成果物

trainer job が完了すると、以下の成果物をアップロードする。

    results/<submission_id>/
      policy.zip
      policy.onnx
      policy.sentis.onnx
      replay/replay.jsonl

`policy.zip` は Stable-Baselines3 model である。`policy.onnx` は
continuous navigation の dict observation を `robot`、`goal`、
`front_distance` input として公開する一般 ONNX artifact である。
`policy.sentis.onnx` は Unity Sentis 向けに固定長 `float32[1,28226]` input
へまとめた ONNX artifact であり、output は `[forward, turn]` の
continuous action である。Replay Log は EnvForge がローカル再生するための
JSON Lines artifact である。Result Bundle には通常 ONNX と Sentis ONNX の
artifact location と input/output layout metadata を含める。

## 現在の強み

- API、trainer、notification、shared models が分離されている。
- Firestore-backed submissions と results がある。
- Cloud Run Job execution path が存在する。
- Artifact upload path が存在する。
- Pub/Sub と WebSocket の status update path が存在する。
- Repository protocols と fake repositories により core flow が testable である。

## 現在の連携状態と次の不足

現在の production training path は `ContinuousNavigationEnv` と
`run_continuous_navigation_training` を使う。この runtime は EnvForge の x/z meter
座標、Y 回転、連続 action forward/turn、goal radius、static walls、
static obstacles、回転付き box collision、距離センサ range を表現する。
Replay Bundle は continuous runtime の実座標と実 action から生成する。

Scenario Bundle、Result Bundle、Replay Bundle の契約と、EnvForge からのジョブ投入、
進捗監視、artifact download、Replay 再生、ONNX Runtime 推論の主導線は実装済みである。

次に不足しているものは以下である。

- Unity 向け API client、状態監視、artifact 取得、bundle DTO が EnvForge 内にあり、
  ほかの Unity フロントエンドから再利用できない。
- `EmbodiedLab.Unity` と EmbodiedLab API の version compatibility を検証する仕組みがない。
- 現在の Scenario Bundle は固定マップを表し、episode ごとの宣言的な環境生成規則を
  表現できない。
- ONNX export は continuous 主経路と Result Bundle metadata に接続済みだが、
  複数の Unity フロントエンドで同じ互換性検査を再利用する層がない。
- reward component の主要 weight は Scenario Bundle から continuous runtime へ
  反映する。現時点では `goal_reached`、`goal_progress`、
  `collision_penalty`、`step_penalty`、
  `wide_angle_penalty`、`rear_angle_penalty`、`inactive_penalty`、
  `movement_threshold` を扱う。
- robot と sensor descriptor が最小限である。
- forward camera observation はまだ抽象化されたままである。
- artifact access が public-read 前提である。

固定マップは今後も既定動作として維持する。episode ごとの環境生成は明示的な
`generated` mode として追加し、ユーザ提供の任意コードではなく、検証可能な宣言的
schema と seed に基づいて EmbodiedLab runtime が実行する。
