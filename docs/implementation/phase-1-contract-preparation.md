# EnvForge 連携 Phase 1 契約準備

## 目的

このフェーズの目的は、EnvForge と EmbodiedLab の接続を
明示的なデータ契約として定義することである。

Phase 0 の基盤整備は完了扱いとする。EmbodiedLab 側では、
Python 3.13、`uv`、Ruff、PyMarkdown、pytest を使った
`make check` が通る状態を前提に、契約設計と実装準備へ進む。

## 対象契約

最初に定義する契約は以下である。

- Scenario Bundle
- Result Bundle
- Replay Log

Scenario Bundle は、EnvForge で作成された学習シナリオを表す。
Result Bundle は、EmbodiedLab が返す学習結果を表す。
Replay Log は、EnvForge がローカル再生するための構造化ログである。

## EmbodiedLab 側の責務

EmbodiedLab は以下を担当する。

- Scenario Bundle の受信
- schema version と互換性情報の検証
- 学習環境への変換
- 学習ジョブの実行
- モデル成果物の保存
- 学習結果とエラー情報の保存
- Replay Log の出力
- Result Bundle の生成

EmbodiedLab は Unity や ML-Agents を必ずしも実行しない。
クラウド向きの学習環境で、EnvForge のシナリオ条件を
意味的に再現することを優先する。

## Scenario Bundle 初期入力

初期版では、以下を受け取れるようにする。

- schema version
- scenario id
- EnvForge binary compatibility
- world size または座標系情報
- static walls
- static obstacles
- goal
- robot start pose
- robot type
- sensor spec
- reward spec
- training spec
- seed

現在の grid-world submission は、公開契約としては残さない。
Phase 1 では既存の `/submissions` 入力を Scenario Bundle で
上書きし、旧契約との互換 API は用意しない。

## Result Bundle 初期出力

初期版では、以下を返す。

- result schema version
- scenario id
- job id
- status
- training summary
- evaluation summary
- model artifact location
- replay log location
- error report
- compatibility metadata

compatibility metadata には、EnvForge が結果を読めるか判断するための
情報を含める。

## Replay Log 初期出力

Replay Log は動画ではなく、構造化ログとして保存する。

初期版では以下を含める。

- episode id
- step index
- simulation time
- robot position
- robot rotation
- action
- total reward
- reward component breakdown
- termination reason
- collision events
- compact sensor summaries

全ステップの画像観測は、初期版では必須にしない。

## Scenario Bundle v0 サンプル

最初の Scenario Bundle は、EnvForge で作った静的な
ナビゲーションシナリオを表す。座標系は、EnvForge と EmbodiedLab の
間で共有する右手系の 2D 平面として扱う。`x` と `z` を水平面、
`y` を高さとする。単位は meter とする。

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
    "static_walls": [
      {
        "id": "wall_north",
        "center": { "x": 5.0, "z": 10.0 },
        "size": { "x": 10.0, "z": 0.2 },
        "rotation_y_degrees": 0.0
      }
    ],
    "static_obstacles": [
      {
        "id": "box_001",
        "shape": "box",
        "center": { "x": 4.5, "z": 5.0 },
        "size": { "x": 1.0, "z": 1.0 },
        "rotation_y_degrees": 0.0
      }
    ],
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
    "components": [
      {
        "name": "goal_reached",
        "type": "terminal_reward",
        "weight": 10.0
      },
      {
        "name": "goal_progress",
        "type": "distance_delta",
        "target": "goal_001",
        "weight": 0.5
      },
      {
        "name": "collision_penalty",
        "type": "collision",
        "weight": -5.0
      },
      {
        "name": "step_penalty",
        "type": "per_step",
        "weight": -0.01
      }
    ]
  },
  "training": {
    "algorithm": "ppo",
    "timesteps": 5000,
    "seed": 10,
    "max_episode_steps": 512
  }
}
```

## Result Bundle v0 サンプル

Result Bundle は、EmbodiedLab が学習完了または失敗時に返す結果である。
成功時は model artifact と Replay Log の場所を含む。失敗時は
`status` を `failed` にし、`error` に診断情報を入れる。

```json
{
  "schema_version": "result-bundle.v0",
  "scenario_id": "scenario_demo_001",
  "job_id": "job_20260528_001",
  "status": "completed",
  "compatibility": {
    "scenario_schema_version": "scenario-bundle.v0",
    "envforge_min_version": "0.1.0",
    "robot_version": "simple_robot.v0",
    "sensor_version": "basic_sensors.v0",
    "action_layout": ["forward", "turn"],
    "observation_layout": [
      "front_camera_semantic",
      "front_distance"
    ]
  },
  "summary": {
    "training_timesteps": 5000,
    "training_seed": 10,
    "success_rate": 0.82,
    "average_episode_reward": 6.4,
    "average_episode_steps": 118.5
  },
  "artifacts": {
    "model": {
      "storage": "gcs",
      "bucket": "embodiedlab-models",
      "path": "results/job_20260528_001/model/policy.onnx",
      "format": "onnx"
    },
    "replay_log": {
      "storage": "gcs",
      "bucket": "embodiedlab-models",
      "path": "results/job_20260528_001/replay/replay.jsonl",
      "format": "jsonl"
    }
  },
  "error": null
}
```

## Replay Log v0 サンプル

Replay Log は、episode と step の列として扱う。初期版では JSON Lines を
候補にする。ここでは読みやすさのため、2 step 分を配列で示す。

```json
[
  {
    "schema_version": "replay-log.v0",
    "scenario_id": "scenario_demo_001",
    "job_id": "job_20260528_001",
    "episode_id": "episode_0001",
    "step_index": 0,
    "time_seconds": 0.0,
    "robot": {
      "position": { "x": 1.0, "z": 1.0 },
      "rotation_y_degrees": 0.0
    },
    "action": {
      "forward": 0.2,
      "turn": 0.0
    },
    "reward": {
      "total": -0.01,
      "components": {
        "step_penalty": -0.01
      }
    },
    "events": [],
    "sensors": {
      "front_distance": 5.0
    },
    "terminated": false,
    "termination_reason": null
  },
  {
    "schema_version": "replay-log.v0",
    "scenario_id": "scenario_demo_001",
    "job_id": "job_20260528_001",
    "episode_id": "episode_0001",
    "step_index": 1,
    "time_seconds": 0.1,
    "robot": {
      "position": { "x": 1.02, "z": 1.0 },
      "rotation_y_degrees": 0.0
    },
    "action": {
      "forward": 0.2,
      "turn": 0.0
    },
    "reward": {
      "total": 0.04,
      "components": {
        "goal_progress": 0.05,
        "step_penalty": -0.01
      }
    },
    "events": [],
    "sensors": {
      "front_distance": 5.0
    },
    "terminated": false,
    "termination_reason": null
  }
]
```

## 実装ステップ

1. Scenario Bundle の最小 Pydantic model を追加する。
2. Result Bundle の最小 Pydantic model を追加する。
3. Replay Log の最小 Pydantic model を追加する。
4. 既存の `/submissions` 入力を Scenario Bundle で置き換える。
5. EnvForge scenario を training spec に変換する層を追加する。
6. reward component を現在の固定報酬から分離する。
7. replay artifact を GCS に保存する流れを追加する。
8. fake repository と tests を追加する。

## 現 runtime adapter の境界

現在の Scenario Bundle から grid-world runtime への変換は、
Phase 1 契約の最終形ではなく一時 adapter として扱う。
EnvForge/EmbodiedLab 契約は `envforge_xz_meters` の連続 x/z meter 座標を
保持する一方、現 runtime は非負の grid cell 座標へ floor して扱うため
lossy である。

この adapter では、障害物サイズ、回転、goal radius、センサ定義、
宣言的 reward component など、契約上の情報の一部は runtime へ完全には
反映されない。この制約は `describe_runtime_conversion` とそのテストで
明示し、grid-world runtime を置き換えるまでの暫定境界として管理する。

## 保留事項

- JSON Schema を出力するか。
- 契約定義を EmbodiedLab に置くか、共有パッケージにするか。
- Replay Log の圧縮と分割。
- GCS 上の成果物アクセス制御。
- model format を ONNX に固定するか。
