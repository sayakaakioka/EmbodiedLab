# EnvForge 連携ロードマップ

## 背景

EnvForge と EmbodiedLab は別リポジトリとして維持する。

EnvForge は、ユーザがシナリオを作成し、学習中の挙動を再生する
Unity アプリケーションである。

EmbodiedLab は、EnvForge から受け取ったシナリオをもとに
クラウド側で学習を実行し、成果物を保存し、状態更新を通知する
バックエンドである。

現在の EmbodiedLab は、FastAPI submission endpoint、Firestore persistence、
Cloud Run Job execution、Stable-Baselines3 PPO training、GCS artifact storage、
Pub/Sub event、WebSocket fan-out を持つ。
これは EnvForge 連携に使える学習実行基盤である。

## 現在の EmbodiedLab baseline

- 入力モデルは EnvForge Scenario Bundle へ置き換える。
- training backend は Gymnasium-compatible grid world と
  Stable-Baselines3 PPO である。
- 成果物は `models/<submission_id>/` に保存される。
- 保存される model は `policy.zip`、`policy.onnx`、
  `policy.sentis.onnx` である。
- 結果通知は Firestore result document、Pub/Sub、WebSocket で行う。
- tests は API routes、trainer transitions、artifact flow、schemas、
  notification fan-out を cover している。

## Target Integration Shape

EnvForge は Unity build ではなく Scenario Bundle を送信する。
EmbodiedLab はそれを検証し、クラウド向きの学習環境へ変換し、
学習を実行し、Result Bundle を返す。

    EnvForge
      -> Scenario Bundle
      -> EmbodiedLab API
      -> Training Job
      -> Model artifacts + Replay Log
      -> Result Bundle
      -> EnvForge local replay and model use

## Phase 0: 基盤整備

Phase 0 は完了扱いとする。

完了済みの内容は以下である。

- docs 構成を EnvForge の運用に近づけた。
- `CLAUDE.md` を廃止し、`AGENTS.md` を作成した。
- Python 3.13 と `uv` を前提に依存関係を同期できることを確認した。
- Ruff の target version を Python 3.13 に合わせた。
- PyMarkdown を dev dependency に追加した。
- `make check` で Python lint、Markdown lint、pytest を実行できるようにした。
- `.env` がなくても local check は動くようにした。
- cloud/API 系 target では不足環境変数を明示して止めるようにした。

## Phase 1: Contract Definition

次の中心課題は、以下の契約を定義することである。

- Scenario Bundle
- Result Bundle
- Replay Log

各契約には schema version と compatibility metadata を含める。

推奨する source-of-truth の分担は以下である。

- EnvForge は user-facing scenario と replay semantics を主に持つ。
- EmbodiedLab は backend runtime representation と job result document を持つ。
- 両リポジトリは shared contract を検証する tests を持つ。

## Phase 2: Reward Components

現在の固定 grid-world reward を宣言的 reward component に置き換える。

初期セットは小さく保つ。

- goal reached reward
- goal progress reward
- collision penalty
- obstacle proximity penalty
- step penalty
- stuck penalty
- zone entry reward or penalty

任意のユーザ提供 reward code は初期連携では扱わない。

## Phase 3: Replay Logs

EmbodiedLab は動画ではなく構造化 Replay Log を生成する。
EnvForge はこのログをローカルで再生し、ユーザが視点変更、
報酬イベント確認、失敗ケース分析をできるようにする。

初期 Replay Log は以下を含める。

- episode id
- step index
- robot position and rotation
- action
- total reward
- reward components
- termination reason
- collision or contact events
- compact sensor summaries

全観測画像は初期版では必須にしない。

Replay Log v0 は Unity `JsonUtility` で読み込める形を優先する。
action、reward components、sensor summaries は arbitrary dictionary ではなく
`name` / `value` を持つ配列として表す。

現在の grid-world adapter では、trainer evaluation の先頭 episode から
Replay Log steps を生成し、GCS replay artifact へ渡す。これは EnvForge-compatible
runtime へ移行するまでの暫定ログである。robot position は EnvForge の x/z meter
座標へ戻すが、行動 semantics はまだ grid action からの近似である。

## Phase 4: Environment Upgrade

現在の grid-world から、EnvForge-compatible scenario model へ移行する。

Unity をクラウドで実行することは必須ではない。
重要なのは、学習に必要な EnvForge のシナリオ条件を再現することである。

初期版では以下を support する。

- static walls
- static obstacles
- fixed robot type
- one forward-facing camera abstraction
- one distance sensor abstraction
- goal and episode termination conditions
- declarative reward components

dynamic obstacles、humans、curriculum learning、multiple robot types は
後続フェーズに送る。

現在、ContinuousNavigationEnv と ContinuousNavigationSpec を追加し、
Scenario Bundle から EnvForge x/z meter 座標を保った runtime spec へ
変換できるようにした。初期実装では、連続 action forward/turn、Y 回転、
goal radius、static walls、static obstacles、回転付き box collision、
距離センサ range を扱う。

まだ既定の trainer/export 経路は grid-world runner のままである。
次の作業では、この連続 runtime を trainer の主経路へ接続する。

## Phase 5: Model Compatibility

EmbodiedLab は内部で ML-Agents を使う必要はない。
ただし EnvForge に返す結果には、モデル互換性情報を含める。

- model format
- observation input layout
- action output layout
- robot version
- sensor version
- scenario schema version
- EnvForge binary compatibility

ONNX は引き続き有力候補である。
ただし最終判断は、EnvForge が安定して load / run できる形式に従う。

## Open Issues

- Scenario Bundle schema をどちらのリポジトリで管理するか。
- 共有 package を作るか。
- 旧 grid-world API は公開契約として残さず、既存 `/submissions` を上書きする。
- Replay Log をどう圧縮・分割するか。
- user-specific result を導入した後の GCS access をどうするか。
- public-read model artifact をいつまで許容するか。
- WSL2 `aarch64` 開発と Cloud Run `linux/amd64` deploy の差をどう扱うか。
