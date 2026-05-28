# 現在のアーキテクチャ

## 概要

EmbodiedLab は現在、grid-world 強化学習のための最小限の
クラウド学習ループを実装している。

FastAPI service が submission を受け取り、Firestore に保存し、
Cloud Run Job を起動する。trainer は Stable-Baselines3 PPO で
方策を学習し、model artifact を Google Cloud Storage にアップロードし、
結果状態を Firestore に書き戻し、Pub/Sub と WebSocket relay を通して
更新を通知する。

## Runtime Flow

    Client
      -> POST /submissions
          -> Firestore submissions/{submission_id}
      -> POST /submissions/{submission_id}/train
          -> Firestore results/{submission_id} = queued
          -> Cloud Run Job with SUBMISSION_ID override
              -> Firestore submission lookup
              -> GridWorld PPO training
              -> GCS artifact upload
              -> Firestore result update
              -> Pub/Sub event
                  -> notification service push endpoint
                      -> WebSocket subscribers
      -> GET /results/{submission_id}
      -> WebSocket /ws/results/{submission_id}

## Services

### `server/`

FastAPI API service である。
submission の受理、result document の queued 化、Cloud Run Job の起動、
result document の返却を担当する。

主な endpoint は以下である。

- `POST /submissions`
- `POST /submissions/{submission_id}/train`
- `GET /results/{submission_id}`

### `trainer/`

Cloud Run Job service である。
submission を読み、training spec に変換し、PPO training を実行し、
artifact をアップロードし、Firestore を更新し、status event を publish する。

### `notification/`

FastAPI WebSocket relay service である。
Pub/Sub push event を受け取り、submission id に対応する WebSocket client へ
broadcast する。

### `embodiedlab/`

schemas、result models、repository protocols、grid-world environment、
training converter、training runner を含む shared library である。

## 現在のデータモデル

現在の API は EnvForge Scenario Bundle を受け取る方向へ移行中である。
旧 grid-world submission は公開契約として残さない。

- grid size
- obstacle cells
- goal cell
- robot start cell
- robot type
- PPO training configuration

現在の環境は、Gymnasium-compatible grid world である。
action は discrete four-action policy である。

- up
- right
- down
- left

observation は agent 座標と goal 座標を含む dictionary である。

## 現在の成果物

trainer job が完了すると、以下の成果物をアップロードする。

    models/<submission_id>/
      policy.zip
      policy.onnx
      policy.sentis.onnx

`policy.zip` は Stable-Baselines3 model である。
`policy.onnx` は一般的な ONNX export である。
`policy.sentis.onnx` は Unity Sentis 向けの ONNX export であり、
固定の `float32[1,4]` observation input を持つ。

## 現在の強み

- API、trainer、notification、shared models が分離されている。
- Firestore-backed submissions と results がある。
- Cloud Run Job execution path が存在する。
- Artifact upload path が存在する。
- Pub/Sub と WebSocket の status update path が存在する。
- Repository protocols と fake repositories により core flow が testable である。

## EnvForge 連携に向けた不足

- Scenario Bundle contract がない。
- Result Bundle contract がない。
- Replay Log artifact がない。
- reward logic が `GridWorldTrainingEnv` に固定されている。
- robot と sensor descriptor が最小限である。
- environment が grid-based で、EnvForge の壁、障害物、センサ、
  continuous robot motion をまだ表現していない。
- artifact access が public-read 前提である。
