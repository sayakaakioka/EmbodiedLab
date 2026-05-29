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
          -> Firestore submissions/{submission_id}
      -> POST /submissions/{submission_id}/train
          -> Firestore results/{submission_id} = queued
          -> Cloud Run Job with SUBMISSION_ID override
              -> Firestore submission lookup
              -> Continuous navigation PPO training
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

schemas、result models、repository protocols、continuous navigation environment、
training converter、training runner を含む shared library である。

## 現在のデータモデル

現在の API は EnvForge Scenario Bundle を受け取る。主経路の環境は、
Gymnasium-compatible continuous navigation runtime である。action は continuous
forward/turn policy であり、observation は robot、goal、front_distance を含む
dictionary である。旧 grid-world runtime は削除済みで、必要な履歴は Git history を
参照する。

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
`policy.sentis.onnx` は Unity Sentis 向けに固定長 `float32[1,7]` input
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

## EnvForge 連携に向けた不足

Phase 4 の準備として、ContinuousNavigationEnv を追加した。
この runtime は EnvForge の x/z meter 座標、Y 回転、連続 action
forward/turn、goal radius、static walls、static obstacles、
回転付き box collision、距離センサ range を表現する。

現在の production training path は ContinuousNavigationEnv と
run_continuous_navigation_training を使う。Replay Log は continuous runtime の
実座標と実 action から生成する。

- Scenario Bundle contract がない。
- Result Bundle contract がない。
- ONNX/Sentis export は continuous 主経路と Result Bundle metadata に
  接続済みだが、EnvForge 側の Sentis runtime inference 検証はまだである。
- reward component の主要 weight は Scenario Bundle から continuous runtime へ
  反映する。現時点では `goal_reached`、`goal_progress`、
  `collision_penalty`、`step_penalty`、`movement_reward`、
  `wide_angle_penalty`、`rear_angle_penalty`、`inactive_penalty`、
  `movement_threshold`、`turn_activity_threshold` を扱う。
- robot と sensor descriptor が最小限である。
- forward camera observation はまだ抽象化されたままである。
- artifact access が public-read 前提である。
