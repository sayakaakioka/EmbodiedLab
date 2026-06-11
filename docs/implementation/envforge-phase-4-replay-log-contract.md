# EnvForge 連携 Phase 4 Replay Log 契約

## 目的

Phase 4 では、学習中のロボット挙動を EnvForge 側で再生するための Replay Log
契約を整える。

初期版では動画を生成しない。EmbodiedLab は JSON Lines の Replay Log artifact を
保存し、EnvForge はそれを読み込んでローカル再生する。

## 今回の範囲

今回の最小範囲は以下である。

- `ReplayLogStep` を `JsonUtility` 互換の shape にする。
- EnvForge と同じ canonical Replay Log fixture を置く。
- fixture の各 JSONL 行を `ReplayLogStep` として検証する。
- GCS artifact metadata は `result_bundle.artifacts.replay_log` で返す。
- trainer evaluation の先頭 episode から、空ではない Replay Log steps を生成し、
  upload 境界へ渡す。

fixture は以下に置く。

    tests/fixtures/envforge/navigation_default_replay_log.jsonl

## JsonUtility 互換の方針

Unity `JsonUtility` は arbitrary dictionary を直接扱いにくい。そのため Replay Log
v0 では、action、reward components、sensor summaries を key/value map ではなく
配列で表す。

action は以下の形にする。

    "action": {
      "values": [
        { "name": "forward", "value": 0.2 },
        { "name": "turn", "value": 0.0 }
      ]
    }

reward components も `{ "name": ..., "value": ... }` の配列にする。
これにより、ユーザ定義 reward component 名が増えても、EnvForge 側は固定 DTO で
読み込める。

## 到達点

現在の runtime はまだ EnvForge の連続空間そのものではなく、一時的な grid-world
adapter である。Replay Log の robot position は Scenario Bundle の `bounds.min`
を使って EnvForge の x/z meter 座標へ戻すが、行動は grid action から
`forward` / `turn` へ近似している。

それでも、空の Replay Log artifact ではなく、評価 episode の EnvForge 座標、
行動、報酬成分、イベント、距離 sensor summary を JSONL として保存できる入口が
できた。

## 保留事項

- EnvForge-compatible runtime から Replay Log step を生成する処理。
- continuous `forward` / `turn` action semantics への移行。
- collision/contact event の詳細 schema。
- 数値以外の sensor summary を扱う場合の型別 schema。
- 長い Replay Log の圧縮、分割、部分読み込み。
